"""Loudness-aware choice of ONE contiguous excerpt of real solo dialogue as a cue's voice reference.

Why one excerpt: the reference gate (beta_stable_gate._pcm_matches / _match_channel) requires the reference WAV to be
bit-identical to an exact slice of the dialogue stem, so a reference can neither be gain-scaled nor spliced
from loudness-matched pieces; selection must pick a real excerpt. The owner's approved demo cloned each line from the
line's OWN audio first (a shouted line from shouting, a quiet line from quiet audio) and only then added same-speaker
lines nearest in loudness. Within the gate's constraint the equivalent is: own span when it is clean and long enough,
else the single same-speaker excerpt whose active-frame loudness is closest to the cue's.

choose_reference is pure: it reads dicts and returns a span plus a receipt. It never touches audio, so the caller cuts
``speech[a:b]`` itself (the receipt's ``sample_span`` when ``rate`` is given) and the usual validate/admit path
decides. ``measure_span_db`` (speech_activity.active_rms_db over the exact slice that would be cut) produces the
``source_db`` values; ``active_rms_db`` is re-exported from relay.speech_activity for the same purpose. Eligibility reuses the rules already in actor_pipeline.line_reference_plan / actor_reference.make_reference:
an ``overlapped`` or ``turn_overlap`` span, a ``singing`` span, a span crossing any ``overlap_intervals``, or (when
``raw_turns`` are supplied) a span touching another raw speaker's turn or an unlabelled turn is never used. ``unclear``
is NOT a reason to refuse (the existing code uses such excerpts reference-only, with an empty transcript); it is
reported in the receipt so the caller can set the transcript basis.

Own-span rules copy line_reference_plan: begin/stop are the cue's speech span clipped to its start/end; a span of
at least ``min_seconds`` is used, truncated from its start to ``max_seconds``; a shorter span may be extended, centred and
to exactly ``min_seconds``, only inside the single continuous solo turn the caller names in ``cue['solo_turn']``
({'start','end'}, already cleared of every other speaker, as the existing code computes it).

Gate compatibility, stated plainly: an own / own_extended result is the same kind of span the line-reference gate
replays for a ``line-`` actor. A ``neighbour`` result is another cue's excerpt; it can only pass the gate through
the bank-style make_reference route (spans equal to that cue's padded excerpt, transcript of that cue). Give such
candidates their padded span as ``start``/``end`` if that is the route; the span is used verbatim.
"""
from __future__ import annotations

import math

from .speech_activity import active_rms_db

VERSION = 'loudness-reference-v1'
PREFER = ('own', 'loudness')
_RANK_REPORT = 5
_SNAP = 1e-7           # the tolerance line_reference_plan uses when it snaps seconds to samples


def choose_reference(cue, candidates, *, min_seconds=1.0, max_seconds, prefer='own', rate=None, origin=0.0,
                     raw_turns=None, allow_extension=True):
    """Return ``(span, receipt)``: ``span`` is ``(start, end)`` in absolute seconds, or None when nothing is eligible
    (then ``receipt['reason']`` says why and the caller keeps original audio honestly).

    ``cue``: dict with ``id``, ``start``, ``end`` (optional ``speech_start``/``speech_end``), ``speaker``,
    ``source_db`` and the flags above (optionally ``solo_turn``, ``raw_speaker``). ``candidates``: dicts with
    ``id``, ``start``, ``end`` (absolute seconds), ``speaker``, ``source_db`` (active-frame RMS dB of the dialogue
    stem over that span) and the same flags; the cue itself, if listed, is skipped.

    ``prefer='own'`` (default): own span first (direct, then extended inside its solo turn); only when that is
    unusable choose the neighbour with minimal ``|source_db - cue.source_db|`` (ties: longer, then earlier, then id).
    ``prefer='loudness'``: the cue's own usable span competes as a delta-0 candidate, so an equally matched but longer
    neighbour wins ties on length.

    ``max_seconds`` bounds a neighbour; an own span is truncated from its start to it. ``rate``/``origin``: when
    given, every span is snapped inward to whole samples exactly as line_reference_plan does, lengths are checked in
    samples, and the receipt carries ``sample_span`` = [a, b] (the slice ``speech[a:b]``). Spans already on the sample
    grid (the usual case: stored make_reference intervals) are preserved exactly. Without ``rate`` spans are
    returned verbatim in float seconds."""
    min_s, max_s = _number(min_seconds, 'min_seconds'), _number(max_seconds, 'max_seconds')
    if min_s <= 0 or max_s < min_s:
        raise ValueError('Need 0 < min_seconds <= max_seconds')
    if prefer not in PREFER:
        raise ValueError('prefer must be one of %s' % (PREFER,))
    if rate is not None:
        rate = _number(rate, 'rate')
        if rate <= 0:
            raise ValueError('rate must be positive')
        origin = _number(origin, 'origin')
    grid = _Grid(rate, origin, min_s, max_s)
    if grid.cap < grid.minimum:
        raise ValueError('No whole-sample excerpt fits between min_seconds and max_seconds')
    if not isinstance(cue, dict):
        raise ValueError('cue must be a dict')
    candidates = list(candidates or ())
    speaker = cue.get('speaker')
    speaker = speaker if isinstance(speaker, str) and speaker.strip() else None
    cue_db = _db(cue.get('source_db'))
    receipt = {'version': VERSION, 'cue_id': cue.get('id'), 'speaker': speaker, 'prefer': prefer, 'selected': False,
               'basis': None, 'reason': None, 'span': None, 'sample_span': None, 'seconds': None,
               'truncated': False, 'cue_source_db': cue_db, 'chosen': None, 'own': None,
               'considered': len(candidates), 'eligible': 0, 'excluded': {}, 'ranking': [],
               'min_seconds': min_s, 'max_seconds': max_s, 'rate': rate, 'origin': origin if rate else None}

    own = _own_span(cue, grid, raw_turns, allow_extension)
    receipt['own'] = {'usable': own['span'] is not None, 'reason': own['reason'], 'basis': own['basis']}
    own_chosen = {'id': cue.get('id'), 'speaker': speaker, 'source_db': cue_db, 'delta_db': 0.0,
                  'unclear': bool(cue.get('unclear')), 'text_cue_id': cue.get('id')}
    if own['span'] is not None and prefer == 'own':
        return _finish(receipt, grid, own['span'], own['basis'], own['truncated'], own_chosen)
    if own['span'] is not None and cue_db is None:
        receipt['own']['note'] = 'cue loudness unavailable; own span used without ranking'
        return _finish(receipt, grid, own['span'], own['basis'], own['truncated'], own_chosen)

    ranked = []
    if own['span'] is not None:               # prefer == 'loudness': the cue's own span competes at delta 0
        a, b = own['span']
        ranked.append((0.0, -round(grid.length(a, b), 6), a, str(cue.get('id')), b, 'own', own, None))
    if cue_db is None:
        return _refuse(receipt, 'cue_source_db_unavailable')
    if speaker is None:
        return _refuse(receipt, 'cue_speaker_missing')
    for item in candidates:
        why = _candidate_problem(item, cue, speaker, grid, raw_turns, own['bounds'])
        if isinstance(why, str):
            receipt['excluded'][why] = receipt['excluded'].get(why, 0) + 1
            continue
        a, b, db = why
        ranked.append((round(abs(db - cue_db), 6), -round(grid.length(a, b), 6), a, str(item.get('id')), b,
                       'neighbour', None, item))
    receipt['excluded'] = dict(sorted(receipt['excluded'].items()))
    receipt['eligible'] = len(ranked)
    if not ranked:
        return _refuse(receipt, 'no_eligible_reference')
    ranked.sort(key=lambda row: row[:5])
    receipt['ranking'] = [{'id': cue.get('id') if row[7] is None else row[7].get('id'), 'basis': row[5],
                           'delta_db': row[0], 'seconds': grid.length(row[2], row[4]),
                           'span': [grid.seconds(row[2]), grid.seconds(row[4])]}
                          for row in ranked[:_RANK_REPORT]]
    best = ranked[0]
    if best[5] == 'own':
        return _finish(receipt, grid, own['span'], own['basis'], own['truncated'], own_chosen)
    item = best[7]
    chosen = {'id': item.get('id'), 'speaker': item.get('speaker'), 'source_db': _db(item.get('source_db')),
              'delta_db': best[0], 'unclear': bool(item.get('unclear')), 'text_cue_id': item.get('id')}
    return _finish(receipt, grid, (best[2], best[4]), 'neighbour', False, chosen)


def measure_span_db(speech, rate, start, end, *, origin=0.0, channel=None):
    """``source_db`` for a span: active-frame RMS dB (speech_activity.active_rms_db) of the dialogue-stem slice that
    would be cut, i.e. the excerpt ``speech[a:b]`` with the same inward sample snapping as choose_reference(rate=...).

    ``speech`` is the stem, mono (n,) or sample-major (n, channels). For several channels the measured channel is
    ``channel`` if given, else the one with the most energy in the span (the channel make_reference / make_line_reference
    would write). An empty slice measures as silence (-120 dB), so such a candidate never wins a loudness match."""
    import numpy as np
    rate, origin = _number(rate, 'rate'), _number(origin, 'origin')
    if rate <= 0:
        raise ValueError('rate must be positive')
    a = max(0, math.ceil((_number(start, 'start') - origin) * rate - _SNAP))
    b = min(len(speech), math.floor((_number(end, 'end') - origin) * rate + _SNAP))
    excerpt = np.asarray(speech[a:max(a, b)])
    if excerpt.ndim == 2:
        if channel is None:
            channel = int(np.argmax(np.sum(np.square(excerpt, dtype=np.float64), axis=0))) if len(excerpt) else 0
        excerpt = excerpt[:, int(channel)]
    return active_rms_db(excerpt, rate)


class _Grid:
    """Spans in whole samples when a rate is given (the unit the stem is cut in), else in float seconds."""

    def __init__(self, rate, origin, min_s, max_s):
        self.rate, self.origin = rate, origin
        if rate is None:
            self.minimum, self.cap, self.eps = min_s, max_s, 1e-9
        else:
            self.minimum, self.cap, self.eps = math.ceil(min_s * rate - _SNAP), math.floor(max_s * rate + _SNAP), 0

    def start(self, t):
        return float(t) if self.rate is None else math.ceil((t - self.origin) * self.rate - _SNAP)

    def end(self, t):
        return float(t) if self.rate is None else math.floor((t - self.origin) * self.rate + _SNAP)

    def seconds(self, x):
        return float(x) if self.rate is None else self.origin + x / self.rate

    def length(self, a, b):
        return (b - a) if self.rate is None else (b - a) / self.rate


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError('%s must be a number' % name)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('%s must be a finite number' % name) from error
    if not math.isfinite(number):
        raise ValueError('%s must be a finite number' % name)
    return number


def _db(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _span_problem(item, x, y, raw_turns, raw_label):
    """Why [x, y] (absolute seconds) may not be a reference, else None. Same rules as the existing reference code."""
    if item.get('overlapped') or item.get('turn_overlap') is True:
        return 'overlapped'
    if item.get('singing'):
        return 'singing'
    for pair in item.get('overlap_intervals') or ():
        try:
            left, right = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError, KeyError, OverflowError):
            return 'overlap_interval_invalid'
        if not math.isfinite(left) or not math.isfinite(right):
            return 'overlap_interval_invalid'
        if min(right, y) > max(left, x):
            return 'overlap_interval'
    if raw_turns is not None:
        touched = False
        for turn in raw_turns:
            try:
                left, right, label = float(turn['start']), float(turn['end']), turn.get('speaker')
            except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
                return 'raw_turn_invalid'
            if not math.isfinite(left) or not math.isfinite(right) or left >= right:
                return 'raw_turn_invalid'
            if min(right, y) > max(left, x):
                touched = True
                if label is None or (isinstance(label, str) and not label.strip()):
                    return 'unlabelled_turn'
                if label != raw_label:
                    return 'other_speaker_turn'
        if not touched:
            return 'no_speech_turn'
    return None


def _own_span(cue, grid, raw_turns, allow_extension):
    """The cue's own usable span in grid units, mirroring line_reference_plan (span, truncation, solo-turn extension).

    Returns {'span': (a, b) | None, 'basis', 'reason', 'truncated', 'bounds': (a, b) of the plain own span}."""
    out = {'span': None, 'basis': None, 'reason': None, 'truncated': False, 'bounds': (None, None)}
    try:
        start, end = float(cue['start']), float(cue['end'])
        begin = max(start, float(cue.get('speech_start') or start))
        stop = min(end, float(cue.get('speech_end') or end))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('cue needs numeric start and end') from error
    if not all(math.isfinite(x) for x in (start, end, begin, stop)) or not begin < stop:
        out['reason'] = 'own_span_invalid'
        return out
    a, b = grid.start(begin), grid.end(stop)
    out['bounds'] = (a, b)
    raw_label = cue.get('raw_speaker', cue.get('speaker'))
    why = _span_problem(cue, begin, stop, raw_turns, raw_label)
    if why:
        out['reason'] = 'own_' + why
        return out
    if b - a >= grid.minimum - grid.eps:
        out.update(span=(a, min(b, a + grid.cap)), basis='own', truncated=b - a > grid.cap + grid.eps)
        return out
    turn = cue.get('solo_turn')
    if not allow_extension or not isinstance(turn, dict):
        out['reason'] = 'own_too_short'
        return out
    try:
        low, high = grid.start(float(turn['start'])), grid.end(float(turn['end']))
    except (KeyError, TypeError, ValueError, OverflowError):
        out['reason'] = 'own_too_short'
        return out
    if not (low <= a and high >= b) or high - low < grid.minimum - grid.eps:
        out['reason'] = 'own_too_short'
        return out
    need = grid.minimum - (b - a)
    half = need / 2 if grid.rate is None else need // 2
    x = max(low, a - half)
    y = min(high, x + grid.minimum)
    x = max(low, y - grid.minimum)
    if not (x <= a and y >= b and y - x >= grid.minimum - grid.eps):
        out['reason'] = 'own_too_short'
        return out
    why = _span_problem(cue, grid.seconds(x), grid.seconds(y), raw_turns, raw_label)
    if why:
        out['reason'] = 'own_extension_' + why
        return out
    out.update(span=(x, y), basis='own_extended')
    return out


def _candidate_problem(item, cue, speaker, grid, raw_turns, own_bounds):
    """Exclusion reason (str) or (a, b, source_db) in grid units for an eligible same-speaker neighbour."""
    if not isinstance(item, dict):
        return 'invalid_candidate'
    if item.get('id') is not None and item.get('id') == cue.get('id'):
        return 'own_cue'
    if item.get('speaker') != speaker:
        return 'other_speaker'
    try:
        x, y = float(item['start']), float(item['end'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return 'invalid_span'
    if not math.isfinite(x) or not math.isfinite(y) or not x < y:
        return 'invalid_span'
    # Same actor id, so same raw diarization label unless the candidate says otherwise.
    why = _span_problem(item, x, y, raw_turns, item.get('raw_speaker', cue.get('raw_speaker', item.get('speaker'))))
    if why:
        return why
    a, b = grid.start(x), grid.end(y)
    if (a, b) == own_bounds:
        return 'own_span'
    if b - a < grid.minimum - grid.eps:
        return 'too_short'
    if b - a > grid.cap + grid.eps:
        return 'too_long'
    db = _db(item.get('source_db'))
    if db is None:
        return 'source_db_unavailable'
    return a, b, db


def _finish(receipt, grid, span, basis, truncated, chosen):
    a, b = span
    start, end = grid.seconds(a), grid.seconds(b)
    receipt.update(selected=True, basis=basis, span=[start, end], seconds=grid.length(a, b), truncated=truncated,
                   chosen=chosen)
    if grid.rate is not None:
        receipt['sample_span'] = [int(a), int(b)]
    return (start, end), receipt


def _refuse(receipt, reason):
    receipt['reason'] = reason
    return None, receipt
