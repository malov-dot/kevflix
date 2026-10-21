"""Two people talking at the same time: one pass per voice (pure policy; no model is loaded or started here).

Owner decision 2026-10-20 (owner): "If there are multiple people speaking, then you run that segment over 2 times and include
both speaking voices". Reading: where two DIFFERENT diarized speakers really talk together, that stretch is separated into two
isolated voices (relay.overlap.OverlapSeparator: ClearerVoice MossFormer2_SS_16K, Apache-2.0, two 16 kHz stems), every stem is
transcribed on its own, every transcript becomes its own cue, every cue is cloned from ITS OWN stem (never a bank voice) and both
takes may play at the same time. The standing rules stay: dub only where somebody really talks, one unique clone per dubbed line.

Anything the evidence does not carry is a FALLBACK, never a guess: the caller keeps today's behaviour for that stretch (the
overlapped cue stays flagged, relax_solo_overlap / kept original decide as before). Fallback reasons: more than two speakers in the
stretch, a stretch longer than max_stretch_seconds, ambiguous stem-to-speaker assignment, stems that leak into each other, a second
voice that is absent or too weak, a voice too short to clone from, a high residual, a voice with no dubbable transcript.

Owner rules of the same day that shape the design (owner): one-word interjections ARE the common case ("someone is only saying 1
word"), so a stretch starts at a 0.1 s overlap and a one-word voice is a voice; "the dubbed audio must be the same volume as the
original ... you must not cut that word off": every take is level-matched to ITS OWN stem (source_scale + upsample_voice feed the
existing actor_mix_levels, nothing is boosted beyond matching), is never put against its partner as a wall (schedule_partnered,
next_other_start), and a stem that is still speaking at the edge of its window is extended, never cut (run_stretch / extend_stretch).
A voice that cannot be dubbed is not silenced by this lane: only the dubbed voice's own stem is to be removed (phase-2 note).

What this module does NOT do: it loads no model, runs nothing on a GPU, writes nothing except the stem WAVs asked for through
write_voice_stems, and changes no existing cue. The caller (relay.actor_pipeline, phase 2) wires it in. The functions:

  detect_two_voice_stretches   raw diarization turns + cues -> stretches (padded, merged, cue-closed, split into <= 6 s windows)
  extend_stretch               a stretch grown at its edges (a voice still speaking there is never cut)
  separate_stretch             one separator call per window (a fake separator works in tests)
  assign_stems                 which stem is which diarized speaker, consistent across windows; ambiguous -> no split
  quality_gate                 leakage / weak voice / residual / hand-over / cut at the edge -> reason strings
  decide_stretch / run_stretch assign + stitch + gate (+ edge extension and re-separation) -> the two isolated voices or a fallback
  probe_candidates / probe_hidden_voice   a second voice inside a single-speaker cue that the diarizer did not report (flag only)
  reference_plan / own_level / source_scale / next_other_start   one-word voices, per-voice level, partner-aware slot
  write_voice_stems            the isolated voices as WAV files (the reference source of each cue)
  build_voice_cues             transcripts of the stems -> pipeline-shaped cues (voice_source 'line-own', two_voice receipt)
  voice_turns / upsample_voice per-voice raw turns and a 48 kHz sample-major stem, so make_line_reference works on the OWN stem
  schedule_partnered           schedule_lines in which the two takes of one stretch do not wall each other (they may overlap)
  new_receipt / record_stretch / close_receipt   the audit record
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from . import speech_activity as _activity
from .overlap import MODEL_SHA256

VERSION = 'two-voice-segments-v1'
RATE = 16000
OWNER_RULE = ('2026-10-20 owner: "If there are multiple people speaking, then you run that segment over 2 times and '
              'include both speaking voices"')
ENGINE = {'name': 'ClearerVoice MossFormer2_SS_16K', 'upstream': 'modelscope/ClearerVoice-Studio', 'license': 'Apache-2.0',
          'model_sha256': MODEL_SHA256, 'device': 'cpu', 'rate': RATE}
MAX_SEPARATOR_SAMPLES = 96000          # relay.overlap.OverlapSeparator.separate: .5 s .. 6 s at 16 kHz
MIN_SEPARATOR_SAMPLES = 8000

# (default, low, high). Every default is a measured or conservative value; the report says which.
_KNOBS = {
    'min_overlap_seconds': (.1, .02, 5.),        # two different speakers together for at least this long = a stretch (one-word interjections)
    'min_window_seconds': (.5, .5, 2.),          # the separator's minimum; a shorter stretch is widened around its centre
    'include_uncued': (0., 0., 1.),              # 1 = also split stretches in which no cue exists (otherwise left to the original-audio protection)
    'pad_seconds': (.35, 0., 2.),                # context before and after the overlap, so the model hears each voice alone too
    'window_seconds': (6., 2., 6.),              # separator limit
    'window_overlap_seconds': (.5, .2, 1.5),     # consecutive windows of one long stretch share this much (permutation alignment)
    'max_stretch_seconds': (12., 6., 30.),       # longer stretches are not split (same ceiling as an ordinary multi-cue unit)
    'cue_tolerance_seconds': (.12, 0., 1.),      # a cue belongs to the stretch when it shares more than this with it
    'third_speaker_seconds': (.2, 0., 2.),       # a third diarized voice this long inside the stretch forbids the split
    'solo_erode_seconds': (.05, 0., .5),         # solo regions shrink by this at every diarizer boundary (turn edges are imprecise)
    'solo_min_seconds': (.12, .05, 2.),          # least solo audio (one contiguous region) that counts as evidence
    'solo_min_db': (6., 1., 30.),                # stem-energy difference in a solo region that decides an assignment
    'embedding_margin': (.12, .02, .8),          # cosine margin that decides an assignment
    'embedding_min_seconds': (.4, .1, 3.),       # least active audio of a stem that may be embedded
    'chain_margin': (.4, .1, 1.8),               # correlation margin that aligns the stems of consecutive windows
    'max_residual': (.6, .05, 1.5),              # |mixture - sum(stems)| / |mixture| above this = the model did not explain the audio
    'min_voice_db': (-52., -80., -20.),          # each voice must reach this (active RMS, dBFS) inside the overlap
    'max_core_gap_db': (24., 6., 60.),           # louder minus quieter voice inside the overlap
    'max_cross_corr': (.45, .05, .95),           # |zero-lag correlation| of the two stems inside the overlap
    'min_solo_rejection_db': (9., 3., 40.),      # in a speaker's solo region the other stem must be this much quieter
    'max_voice_similarity': (.6, .2, .95),       # embedding cosine of the two isolated voices above this = one voice split in two (needs embed)
    'min_voice_seconds': (.25, .1, 5.),          # below this a stem holds no voice at all (the engine's own reference floor is .25 s)
    'own_reference_seconds': (1., .5, 5.),       # a voice with less than this of its own audio needs reference help (reference_plan)
    'min_simultaneous_seconds': (.1, .02, 2.),   # both stems must really be active together this long (else it was a hand-over)
    'edge_seconds': (.04, .01, .2),              # a voice still speaking in the first/last edge_seconds of its stem was cut off
    'edge_extend_seconds': (.5, .1, 1.5),        # ... and the stretch grows by this at that side, then is separated again
    'max_edge_extensions': (2., 0., 4.),
    'hidden_max_gap_db': (18., 6., 40.),         # probe: the weaker stem is at most this far under the louder one inside the cue
    'hidden_min_seconds': (.25, .1, 2.),         # probe: the weaker stem is active this long, together with the louder one
    'hidden_max_cross_corr': (.3, .05, .9),      # probe: a copy / echo of the same voice correlates above this
    'probe_max_seconds': (60., 0., 600.),        # probe: audio seconds of cues that may be separated per window (CPU budget)
}


def policy(settings=None):
    """Validated knobs from an actor_beta settings dict (keys 'two_voice_<knob>'; 'two_voice_segments': true enables the lane).
    Default OFF. A malformed or out-of-range value falls back to its default, like pipeline.overlap_policy."""
    settings = settings if isinstance(settings, dict) else {}
    result = {'enabled': settings.get('two_voice_segments') is True}
    for key, (default, low, high) in _KNOBS.items():
        try:
            value = float(settings.get('two_voice_' + key, default))
        except (TypeError, ValueError):
            value = default
        result[key] = value if math.isfinite(value) and low <= value <= high else default
    return result


def _knobs(knobs):
    return policy() if knobs is None else knobs


def fingerprint(settings=None):
    """What config.engine_fingerprint() must fold in (like slot_fill.fingerprint): None while the lane is off, else the version, the
    model pin and the EFFECTIVE knobs (defaults filled in), so a changed default or setting never reuses a chunk dubbed without it."""
    k = policy(settings)
    return None if not k['enabled'] else {'version': VERSION, 'model_sha256': MODEL_SHA256, 'knobs': {x: k[x] for x in sorted(k)}}


# ---------------------------------------------------------------------------------------------------- interval helpers

def _span(item, what):
    a, b = float(item['start']), float(item['end'])
    if not math.isfinite(a) or not math.isfinite(b) or b <= a:
        raise ValueError('%s has invalid geometry' % what)
    return a, b


def _turns(items):
    """[(start, end, label)] of the raw diarization turns; a turn without a label can never be 'a different speaker'."""
    result = []
    for item in items or ():
        a, b = _span(item, 'diarization turn')
        label = item.get('speaker')
        result.append((a, b, label if isinstance(label, str) and label else None))
    return result


def _union(spans):
    result = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def _subtract(spans, cut):
    result = []
    for a, b in spans:
        pieces = [(a, b)]
        for c, d in cut:
            kept = []
            for x, y in pieces:
                if d <= x or c >= y:
                    kept.append((x, y))
                    continue
                if c > x:
                    kept.append((x, c))
                if d < y:
                    kept.append((d, y))
            pieces = kept
        result.extend(pieces)
    return result


def _length(spans):
    return sum(b - a for a, b in spans)


def _shared(a, b, c, d):
    return max(0., min(b, d) - max(a, c))


def _round(value, digits=6):
    return round(float(value), digits)


def _plain(value):
    """JSON-ready copy: tuples to lists, numpy scalars to numbers, floats rounded (the receipt is deterministic)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return _round(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------------------------------- (a) detection

def _dominant_label(cue, turns):
    """Raw diarization label with the greatest total word overlap (None on no overlap or an exact tie)."""
    scores = {}
    words = cue.get('words') or [{'start': cue['start'], 'end': cue['end']}]
    for word in words:
        try:
            a, b = float(word['start']), float(word['end'])
        except (KeyError, TypeError, ValueError):
            continue
        for c, d, label in turns:
            amount = _shared(a, b, c, d)
            if amount and label:
                scores[label] = scores.get(label, 0.) + amount
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked or (len(ranked) > 1 and math.isclose(ranked[0][1], ranked[1][1], rel_tol=0., abs_tol=1e-9)):
        return None
    return ranked[0][0]


def plan_windows(start, end, *, max_seconds=6., overlap=.5, rate=RATE, minimum=.5):
    """Separator windows covering [start, end]: one window up to max_seconds, else equal windows sharing ``overlap`` seconds.
    Every window is at most MAX_SEPARATOR_SAMPLES after rounding to the sample grid; [] when the span is shorter than ``minimum``."""
    length = end - start
    if length < minimum - 1e-9:
        return []
    limit = min(float(max_seconds), MAX_SEPARATOR_SAMPLES / rate) - 2. / rate     # two samples of rounding headroom
    if length <= limit:
        spans = [(start, end)]
    else:
        count = math.ceil((length - overlap) / (limit - overlap) - 1e-9)
        step = (length - overlap) / count
        spans = [(start + i * step, start + i * step + step + overlap) for i in range(count)]
        spans[-1] = (spans[-1][0], end)
    return [{'index': i, 'start': round(a * rate) / rate, 'end': round(b * rate) / rate} for i, (a, b) in enumerate(spans)]


def _stretch_id(start, end, pair):
    return 'tv-' + hashlib.sha256(json.dumps([VERSION, 'stretch', _round(start, 3), _round(end, 3), list(pair)],
                                             separators=(',', ':')).encode()).hexdigest()[:16]


def detect_two_voice_stretches(raw_turns, cues, *, knobs=None, bounds=None, within=None):
    """Stretches where two DIFFERENT raw diarization speakers talk together for >= min_overlap_seconds.

    raw_turns: [{'start','end','speaker'}] absolute seconds (the report's raw_turns). cues: the pipeline's cues (absolute
    seconds, 'words', 'text'); only their spans and word times are read. bounds: (first, last) second of the analysed audio
    (stretches are clamped to it). within: (first, last) render range; stretches outside it are dropped (no CPU spent on them).

    Per speaker pair the overlaps are padded by pad_seconds and merged when the padded spans touch; the stretch then grows to
    contain every cue that shares more than cue_tolerance_seconds with it (a cue is replaced as a whole, otherwise the part of a
    line outside the overlap would lose its dub), repeating until stable. The result is cut into separator windows of at most 6 s.
    Returns, sorted by start, dicts:
      id, start, end, speakers (a, b) sorted, core [[s, e], ...] (the real overlaps), overlap_seconds, windows [{index,start,end}],
      cue_ids / transcribed_lines (cues replaced by the split), voices_transcribed (labels with a cue), cue_state
      ('both_voices_transcribed' | 'one_voice_transcribed' | 'none_transcribed'), eligible (False = leave as today) and reasons.
    Raises ValueError for a turn or cue with impossible geometry (observed timestamps are never repaired)."""
    k = _knobs(knobs)
    turns = _turns(raw_turns)
    cue_spans = [(_span(c, 'cue'), c) for c in cues or ()]
    lo, hi = (float('-inf'), float('inf')) if bounds is None else (float(bounds[0]), float(bounds[1]))
    found = []
    for i, (a0, a1, la) in enumerate(turns):
        for b0, b1, lb in turns[i + 1:]:
            if la and lb and la != lb and min(a1, b1) - max(a0, b0) >= k['min_overlap_seconds'] - 1e-6:
                found.append((max(a0, b0), min(a1, b1), tuple(sorted((la, lb)))))
    groups = []
    for pair in sorted({p for _, _, p in found}):
        current = None
        for s, e, _ in sorted(x for x in found if x[2] == pair):
            if current is not None and s - k['pad_seconds'] <= max(y for _, y in current['core']) + k['pad_seconds']:
                current['core'].append((s, e))
            else:
                current = {'pair': pair, 'core': [(s, e)]}
                groups.append(current)
    for g in groups:
        g['start'] = max(lo, min(s for s, _ in g['core']) - k['pad_seconds'])
        g['end'] = min(hi, max(e for _, e in g['core']) + k['pad_seconds'])
        g['conflict'] = False
    stable = False
    while not stable:                                 # cue closure and same-pair merging until nothing changes
        stable = True
        for g in groups:
            for (cs, ce), _ in cue_spans:
                if _shared(g['start'], g['end'], cs, ce) > k['cue_tolerance_seconds'] and (cs < g['start'] or ce > g['end']):
                    g['start'], g['end'] = min(g['start'], cs), max(g['end'], ce)
                    stable = False
        merged = None
        for i, g in enumerate(groups):
            for h in groups[i + 1:]:
                if _shared(g['start'], g['end'], h['start'], h['end']) > 0:
                    if g['pair'] == h['pair']:
                        merged = (g, h)
                        break
                    g['conflict'] = h['conflict'] = True      # two different pairs in one stretch: more than two voices
            if merged:
                break
        if merged:
            g, h = merged
            g['core'] += h['core']
            g['start'], g['end'] = min(g['start'], h['start']), max(g['end'], h['end'])
            groups.remove(h)
            stable = False
    result = []
    for g in sorted(groups, key=lambda x: (x['start'], x['pair'])):
        g['start'], g['end'] = _widen(g['start'], g['end'], k['min_window_seconds'], lo, hi)
        if within is not None and _shared(g['start'], g['end'], float(within[0]), float(within[1])) <= 0:
            continue
        a, b = g['pair']
        reasons = []
        members = [(c, _dominant_label(c, turns)) for (cs, ce), c in cue_spans
                   if _shared(g['start'], g['end'], cs, ce) > k['cue_tolerance_seconds']]
        voices = sorted({lab for _, lab in members if lab in g['pair']})
        spans = {lab: [_span(c, 'cue') for c, l in members if l == lab] for lab in voices}
        joint = len(voices) == 2 and any(_shared(*x, *y) > k['cue_tolerance_seconds'] for x in spans[a] for y in spans[b])
        flagged = any(c.get('overlapped') or c.get('turn_overlap') is True for c, _ in members)
        if g['conflict']:
            reasons.append('more_than_two_speakers')
        elif any(l and l not in g['pair'] and _shared(g['start'], g['end'], s, e) > k['third_speaker_seconds']
                 for s, e, l in turns) or any(lab and lab not in g['pair'] for _, lab in members):
            reasons.append('third_speaker_present')
        if g['end'] - g['start'] > k['max_stretch_seconds']:
            reasons.append('stretch_longer_than_%gs' % k['max_stretch_seconds'])
        windows = plan_windows(g['start'], g['end'], max_seconds=k['window_seconds'], overlap=k['window_overlap_seconds'],
                               minimum=k['min_window_seconds'])
        if not windows:
            reasons.append('window_under_%gs' % k['min_window_seconds'])
        if not members and k['include_uncued'] < .5:
            reasons.append('no_transcribed_cue')              # such a region is the original-audio protection's, not this lane's
        elif members and len(voices) == 2 and not joint and not flagged:
            reasons.append('served_by_ordinary_cues')         # both voices already have their own, separate, unflagged cue
        result.append({'id': _stretch_id(g['start'], g['end'], g['pair']), 'version': VERSION, 'start': g['start'], 'end': g['end'],
                       'speakers': (a, b), 'core': [list(c) for c in sorted(g['core'])],
                       'overlap_seconds': sum(e - s for s, e in _union(g['core'])), 'windows': windows,
                       'bounds': None if bounds is None else [lo, hi],
                       'cue_ids': [c.get('id') for c, _ in members], 'transcribed_lines': len(members),
                       'cue_spans': [[lab, *_span(c, 'cue')] for c, lab in members],
                       'voices_transcribed': voices,
                       'cue_state': ('both_voices_transcribed' if len(voices) == 2 else
                                     'one_voice_transcribed' if members else 'none_transcribed'),
                       'eligible': not reasons, 'reasons': reasons})
    return result


def _widen(start, end, minimum, lo, hi):
    """A stretch shorter than the separator's minimum grows around its centre (shifted back inside the bounds)."""
    if end - start >= minimum - 1e-9:
        return start, end
    mid = (start + end) / 2.
    a, b = mid - minimum / 2., mid + minimum / 2.
    if a < lo:
        a, b = lo, lo + minimum
    if b > hi:
        a, b = hi - minimum, hi
    return max(a, lo), min(b, hi)


def extend_stretch(stretch, *, start_by=0., end_by=0., knobs=None):
    """The stretch grown at one or both ends (never past its bounds), with new windows and id. Used when a voice was still
    speaking at the edge of its stem: the word is not cut, the stretch is separated again with more context."""
    k = _knobs(knobs)
    lo, hi = stretch.get('bounds') or (float('-inf'), float('inf'))
    start, end = max(lo, stretch['start'] - start_by), min(hi, stretch['end'] + end_by)
    reasons = [r for r in stretch['reasons'] if not r.startswith('stretch_longer') and not r.startswith('window_under')]
    if end - start > k['max_stretch_seconds']:
        reasons.append('stretch_longer_than_%gs' % k['max_stretch_seconds'])
    windows = plan_windows(start, end, max_seconds=k['window_seconds'], overlap=k['window_overlap_seconds'], minimum=k['min_window_seconds'])
    if not windows:
        reasons.append('window_under_%gs' % k['min_window_seconds'])
    return dict(stretch, start=start, end=end, windows=windows, id=_stretch_id(start, end, stretch['speakers']),
                extensions=stretch.get('extensions', 0) + 1, eligible=not reasons, reasons=reasons)


# ---------------------------------------------------------------------------------------------------- separation

def window_slice(window, origin, length, rate=RATE):
    """(first, last) sample of ``window`` in an array that starts at ``origin``; never longer than the separator accepts."""
    first = max(0, round((window['start'] - origin) * rate))
    last = min(length, round((window['end'] - origin) * rate), first + MAX_SEPARATOR_SAMPLES)
    return first, last


def separate_stretch(separator, mixture, origin, stretch, *, rate=RATE, cancelled=lambda: False):
    """One separator call per window of the stretch. ``separator.separate(samples, rate, cancelled=...)`` returns
    (stems (2, n), metrics) exactly like relay.overlap.OverlapSeparator. Returns the parts the later steps read."""
    mixture = np.asarray(mixture, dtype=np.float32)
    parts = []
    for window in stretch['windows']:
        first, last = window_slice(window, origin, len(mixture), rate)
        if last - first < MIN_SEPARATOR_SAMPLES:
            raise ValueError('The analysed audio does not cover this window')
        samples = mixture[first:last]
        stems, metrics = separator.separate(samples, rate, cancelled=cancelled)
        stems = np.asarray(stems, dtype=np.float32)
        if stems.shape != (2, len(samples)):
            raise ValueError('The separator returned the wrong geometry')
        parts.append({'index': window['index'], 'start': origin + first / rate, 'end': origin + last / rate,
                      'mixture': samples, 'stems': stems, 'metrics': dict(metrics)})
    return parts


# ---------------------------------------------------------------------------------------------------- (b) assignment

def _ms(stem, rate, base, spans):
    """(mean square, seconds) of the stem inside the absolute ``spans``."""
    chunks = []
    for a, b in spans:
        i, j = max(0, round((a - base) * rate)), min(len(stem), round((b - base) * rate))
        if j > i:
            chunks.append(stem[i:j].astype(np.float64))
    if not chunks:
        return 0., 0.
    x = np.concatenate(chunks)
    return float(np.mean(x * x)), len(x) / rate


def _db(power):
    return 10. * math.log10(max(power, 1e-12))


def _solo(turns, label, start, end, erode):
    """Absolute spans in [start, end] where only ``label`` is diarized as speaking, shrunk by ``erode`` at every edge."""
    mine = _union([(max(s, start), min(e, end)) for s, e, l in turns if l == label and min(e, end) > max(s, start)])
    others = _union([(s, e) for s, e, l in turns if l != label])
    return [(a + erode, b - erode) for a, b in _subtract(mine, others) if b - a > 2 * erode + 1e-9]


def _active(stem, rate, rel_db=-30., floor_db=-62.):
    """Concatenated 20 ms frames of the stem within rel_db of its loudest frame and above floor_db."""
    hop = int(.02 * rate)
    frames = len(stem) // hop
    if frames < 1:
        return stem[:0]
    body = np.asarray(stem[:frames * hop], dtype=np.float64).reshape(frames, hop)
    db = 20 * np.log10(np.sqrt(np.mean(body * body, axis=1)) + 1e-9)
    return body[db > max(float(db.max()) + rel_db, floor_db)].reshape(-1).astype(np.float32)


def _cos(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or not x.size:
        return None
    n = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / n) if n > 1e-12 else None


def _ncc(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / n) if n > 1e-12 else 0.


def _window_evidence(part, turns, a, b, vectors, embed, k, rate):
    """Evidence for ONE window that stem 0 is speaker a and stem 1 is speaker b (positive) or the reverse (negative)."""
    base, stems = part['start'], part['stems']
    solo_seconds, diffs, weights = {}, [], []
    for index, label in enumerate((a, b)):
        own, other = (index, 1 - index)
        solo_seconds[label] = 0.
        for span in _solo(turns, label, part['start'], part['end'], k['solo_erode_seconds']):
            p_own, seconds = _ms(stems[own], rate, base, [span])
            p_other, _ = _ms(stems[other], rate, base, [span])
            if seconds >= k['solo_min_seconds'] and max(p_own, p_other) > 1e-10:
                diffs.append(max(-30., min(30., _db(p_own) - _db(p_other))))     # > 0: the stem of this label is the louder one
                weights.append(seconds)
                solo_seconds[label] = _round(solo_seconds[label] + seconds, 3)
    solo = float(np.average(diffs, weights=weights)) if diffs else None
    # one solo region saying 'direct' and another 'swapped' (two speakers claimed alone where one stem holds both, or the model swapped
    # its channels inside the window) is a contradiction, not an average
    solo_conflict = bool(diffs) and min(diffs) <= -k['solo_min_db'] and max(diffs) >= k['solo_min_db']
    embedding = None
    if embed is not None and vectors and a in vectors and b in vectors:
        voiced = [_active(stems[i], rate) for i in (0, 1)]
        if all(len(v) / rate >= k['embedding_min_seconds'] for v in voiced):
            e = [embed(v) for v in voiced]
            grid = [[_cos(e[i], vectors[label]) if e[i] is not None else None for label in (a, b)] for i in (0, 1)]
            if all(c is not None for row in grid for c in row):
                embedding = ((grid[0][0] + grid[1][1]) - (grid[0][1] + grid[1][0])) / 2.
    solo_vote = 0 if solo is None or solo_conflict else (1 if solo >= k['solo_min_db'] else -1 if solo <= -k['solo_min_db'] else 0)
    emb_vote = 0 if embedding is None else (1 if embedding >= k['embedding_margin'] else -1 if embedding <= -k['embedding_margin'] else 0)
    conflict = solo_conflict or (solo_vote and emb_vote and solo_vote != emb_vote)
    return {'window': part['index'], 'solo_seconds': solo_seconds, 'solo_db': None if solo is None else _round(solo, 3),
            'solo_conflict': bool(solo_conflict), 'embedding': None if embedding is None else _round(embedding, 4),
            'vote': 0 if conflict else (solo_vote or emb_vote),
            'basis': 'conflict' if conflict else ('solo+embedding' if solo_vote and emb_vote else
                                                   'solo' if solo_vote else 'embedding' if emb_vote else 'none')}


def _chain_link(previous, part, k, rate):
    """+1 when stem i of the previous window continues as stem i, -1 when the channels swapped, 0 when it cannot be told.
    The two windows share audio, so the stems are compared there (permutation alignment of continuous separation)."""
    first, last = part['start'], previous['end']
    if last - first < .15:
        return 0, None
    x = previous['stems'][:, round((first - previous['start']) * rate):round((last - previous['start']) * rate)]
    y = part['stems'][:, :round((last - first) * rate)]
    n = min(x.shape[1], y.shape[1])
    c = [[_ncc(x[i, :n], y[j, :n]) for j in (0, 1)] for i in (0, 1)]
    relation = ((c[0][0] + c[1][1]) - (c[0][1] + c[1][0])) / 2.
    return (1 if relation >= k['chain_margin'] else -1 if relation <= -k['chain_margin'] else 0), _round(relation, 4)


def assign_stems(parts, turns, speakers, *, vectors=None, embed=None, knobs=None, rate=RATE):
    """Which stem is which diarized speaker, per window, consistent over the whole stretch.

    Evidence per window (all optional; a source that is missing simply does not vote):
      solo      in the spans where ONLY speaker a (or b) is diarized, the stem carrying that voice is the louder one; the
                energy difference in dB of the better stem decides when it reaches solo_min_db (spans shrink by solo_erode_seconds);
      embedding embed(samples) of each stem's active audio against vectors[a] / vectors[b] (cosine margin >= embedding_margin).
    Consecutive windows share audio: their stems are aligned by correlation in the shared part (swapped channels are common).
    A ONE-WORD voice (a reply of 0.2-0.6 s) is never embedded (under embedding_min_seconds of active audio an embedding is not
    trustworthy) and usually has no solo audio of its own (its turn lies inside the other speaker's). It is then assigned by
    ELIMINATION: there are two stems and two speakers, so the long speaker's solo audio, which is nearly always there, decides for
    both. If neither speaker is ever alone and the short stem cannot be embedded, the answer is 'ambiguous' and nothing is split.
    Rules: a window decides by its solo vote, else its embedding vote; when both vote and disagree, or the solo evidence of a and b
    contradicts itself, the window has NO vote. Windows linked by a known chain relation form one group, whose orientation is the
    sign of the sum of its votes; a group with no vote, or with votes of both signs, is AMBIGUOUS. Any ambiguity means no split:
    the caller keeps today's behaviour.
    Returns {'decision': 'assigned' | 'ambiguous', 'reasons': [...], 'order': [(stem of a, stem of b) per window] | None,
             'windows': [per-window evidence], 'links': [...], 'basis': 'solo' | 'embedding' | 'solo+embedding' | 'chain'}."""
    k = _knobs(knobs)
    a, b = speakers
    spans = _turns(turns)
    evidence = [_window_evidence(p, spans, a, b, vectors, embed, k, rate) for p in parts]
    links = [0] * len(parts)
    link_values = [None] * len(parts)
    for i in range(1, len(parts)):
        links[i], link_values[i] = _chain_link(parts[i - 1], parts[i], k, rate)
    group, sign = [0] * len(parts), [1] * len(parts)
    for i in range(1, len(parts)):
        group[i] = group[i - 1] if links[i] else group[i - 1] + 1
        sign[i] = sign[i - 1] * links[i] if links[i] else 1
    reasons, orient = [], {}
    for g in sorted(set(group)):
        votes = [evidence[i]['vote'] * sign[i] for i in range(len(parts)) if group[i] == g and evidence[i]['vote']]
        if not votes:
            reasons.append('no_assignment_evidence' if not any(e['basis'] == 'conflict' for i, e in enumerate(evidence) if group[i] == g)
                           else 'assignment_evidence_conflict')
        elif len(set(votes)) > 1:
            reasons.append('window_votes_disagree')
        else:
            orient[g] = votes[0]
    order = None if reasons else [((0, 1) if orient[group[i]] * sign[i] > 0 else (1, 0)) for i in range(len(parts))]
    basis = {e['basis'] for e in evidence} - {'none', 'conflict'}
    return {'decision': 'ambiguous' if reasons else 'assigned', 'reasons': sorted(set(reasons)), 'order': order, 'windows': evidence,
            'links': [{'window': i, 'relation': links[i], 'value': link_values[i]} for i in range(1, len(parts))],
            'basis': '+'.join(sorted(basis)) if basis else None}


# ---------------------------------------------------------------------------------------------------- stitching

def stitch_voices(parts, order, speakers, *, rate=RATE):
    """One continuous 16 kHz stem per speaker: windows are placed on a common grid and cross-faded (sin^2 / cos^2, weights sum to 1)
    where they share audio. -> {'start': first sample's second, 'rate', 'voices': {label: float32 array}}."""
    base = parts[0]['start']
    total = max(round((p['end'] - base) * rate) for p in parts)
    out = [np.zeros(total), np.zeros(total)]
    weight = np.zeros(total)
    for index, part in enumerate(parts):
        offset, length = round((part['start'] - base) * rate), part['stems'].shape[1]
        w = np.ones(length)
        if index:
            shared = min(length, max(0, round((parts[index - 1]['end'] - part['start']) * rate)))
            if shared:
                w[:shared] = np.sin(np.linspace(0, np.pi / 2, shared, endpoint=False)) ** 2
        if index + 1 < len(parts):
            shared = min(length, max(0, round((part['end'] - parts[index + 1]['start']) * rate)))
            if shared:
                w[length - shared:] = np.minimum(w[length - shared:], np.cos(np.linspace(0, np.pi / 2, shared, endpoint=False)) ** 2)
        for v in (0, 1):
            out[v][offset:offset + length] += part['stems'][order[index][v]].astype(np.float64) * w
        weight[offset:offset + length] += w
    covered = weight > 1e-9
    for v in (0, 1):
        out[v][covered] /= weight[covered]
    return {'start': base, 'rate': rate, 'voices': {speakers[0]: out[0].astype(np.float32), speakers[1]: out[1].astype(np.float32)}}


# ---------------------------------------------------------------------------------------------------- (c) gate

def _core_samples(voice, base, rate, spans):
    chunks = [voice[max(0, round((a - base) * rate)):min(len(voice), round((b - base) * rate))] for a, b in spans]
    chunks = [c for c in chunks if len(c)]
    return np.concatenate(chunks) if chunks else voice[:0]


def _active_spans(voice, rate, base):
    """Absolute spans in which the isolated voice speaks (10 ms frames, within 28 dB of its own loudest frame, gaps <= 50 ms bridged)."""
    return _activity.merge_intervals(_activity.activity_intervals(voice, rate, base, base + len(voice) / rate, base), .05)


def _intersection(xs, ys):
    result = []
    for a, b in xs:
        for c, d in ys:
            if min(b, d) > max(a, c):
                result.append((max(a, c), min(b, d)))
    return _union(result)


def _edge_active(voice, rate, edge_seconds):
    """{'start': bool, 'end': bool}: the voice is still speaking in its first / last edge_seconds (it was cut by the window)."""
    n = max(1, int(edge_seconds * rate))
    level = _activity.active_rms_db(voice, rate)
    threshold = max(level - 20., -60.)

    def loud(x):
        return len(x) > 0 and 20 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-9) > threshold
    return {'start': loud(voice[:n]), 'end': loud(voice[-n:])}


def _cue_boundary(stretch, label, side, tolerance=.25):
    """The stretch edge coincides with the start / end of a cue of this voice (cue closure put it there): the whole line is inside the
    window, so the voice's energy at that edge is the line's own onset or tail, not a cut word."""
    return any(lab in (label, None) and abs((s if side == 'start' else e) - stretch[side]) <= tolerance
               for lab, s, e in stretch.get('cue_spans') or ())


def quality_gate(parts, stitched, stretch, turns, *, knobs=None, embed=None):
    """Reject a split whose isolated voices cannot be trusted to clone from. -> {'accepted', 'reasons', 'metrics'}.

    The measurements are taken where BOTH voices really speak (the intersection of the two stems' activity; the diarizer's overlap
    is used only when the stems barely overlap), so a one-word interjection is judged on its own word, not diluted by the long
    voice around it. Reasons (stable codes): non_finite_stems, mixture_silent, residual_too_high, voices_too_similar (embedding cosine of the two voices; only when embed is given and both voices are long
    enough), no_simultaneous_speech (the
    stems only take turns: a hand-over, not an overlap), second_voice_too_weak, stems_too_alike, leakage_in_solo:<speaker>,
    voice_under_<s>s:<speaker> (the stem holds no voice at all), cut_at_edge:<speaker>:start|end (the voice is still speaking where
    its window ends: run_stretch extends the stretch and separates again). Metrics are reported whether or not the split is accepted:
      residual_ratio (worst window; relay.overlap.source_level), simultaneous_seconds, overlap_db {speaker: dBFS of the voice where
      both speak}, overlap_gap_db, cross_corr (zero-lag correlation of the two voices where both speak), solo_rejection_db {speaker:
      level of the OTHER voice relative to this speaker's own solo audio; negative = clean; None = no solo audio}, voice_seconds,
      edges {speaker: {'start','end'}}."""
    k = _knobs(knobs)
    rate, base = stitched['rate'], stitched['start']
    a, b = stretch['speakers']
    voices = stitched['voices']
    reasons, metrics = [], {}
    if not all(np.isfinite(p['stems']).all() for p in parts):
        return {'accepted': False, 'reasons': ['non_finite_stems'], 'metrics': metrics}
    residual = [float(p['metrics'].get('residual_ratio', 0.)) for p in parts]
    gains = [float(p['metrics'].get('gain', 1.)) for p in parts]
    metrics['residual_ratio'] = _round(max(residual), 4)
    if not any(g > 0 for g in gains):
        reasons.append('mixture_silent')
    elif max(residual) > k['max_residual']:
        reasons.append('residual_too_high')
    active = {label: _active_spans(voices[label], rate, base) for label in (a, b)}
    together = _intersection(active[a], active[b])
    metrics['simultaneous_seconds'] = _round(_length(together), 3)
    if _length(together) < k['min_simultaneous_seconds']:
        reasons.append('no_simultaneous_speech')
        together = [tuple(c) for c in stretch['core']]
    region = {label: _core_samples(voices[label], base, rate, together) for label in (a, b)}
    db = {label: _activity.active_rms_db(region[label], rate) for label in (a, b)}
    metrics['overlap_db'] = {label: _round(v, 2) for label, v in db.items()}
    metrics['overlap_gap_db'] = _round(abs(db[a] - db[b]), 2)
    if min(db.values()) < k['min_voice_db'] or abs(db[a] - db[b]) > k['max_core_gap_db']:
        reasons.append('second_voice_too_weak')
    n = min(len(region[a]), len(region[b]))
    metrics['cross_corr'] = _round(_ncc(region[a][:n], region[b][:n]), 4)
    if abs(metrics['cross_corr']) > k['max_cross_corr']:
        reasons.append('stems_too_alike')
    spans_turns = _turns(turns)
    rejection = {}
    for label, other in ((a, b), (b, a)):
        solo = _solo(spans_turns, label, stretch['start'], stretch['end'], k['solo_erode_seconds'])
        p_own, seconds = _ms(voices[label], rate, base, solo)
        p_other, _ = _ms(voices[other], rate, base, solo)
        if seconds >= k['solo_min_seconds'] and p_own > 1e-10:
            rejection[label] = _round(_db(p_other) - _db(p_own), 2)
            if rejection[label] > -k['min_solo_rejection_db']:
                reasons.append('leakage_in_solo:' + label)
        else:
            rejection[label] = None
    metrics['solo_rejection_db'] = rejection
    metrics['voice_seconds'] = {label: _round(_length(active[label]), 3) for label in (a, b)}
    for label in (a, b):
        if metrics['voice_seconds'][label] < k['min_voice_seconds']:
            reasons.append('voice_under_%gs:%s' % (k['min_voice_seconds'], label))
    if embed is not None:
        voiced = {label: _active(voices[label], rate) for label in (a, b)}
        if all(len(v) / rate >= k['embedding_min_seconds'] for v in voiced.values()):       # a one-word voice is never embedded
            e = [embed(voiced[label]) for label in (a, b)]
            similarity = _cos(e[0], e[1]) if all(x is not None for x in e) else None
            metrics['voice_similarity'] = None if similarity is None else _round(similarity, 4)
            if similarity is not None and similarity > k['max_voice_similarity']:
                reasons.append('voices_too_similar')
    lo, hi = stretch.get('bounds') or (float('-inf'), float('inf'))
    edges = {label: _edge_active(voices[label], rate, k['edge_seconds']) for label in (a, b)}
    metrics['edges'] = edges
    for label in (a, b):
        if edges[label]['start'] and stretch['start'] - lo > .02 and not _cue_boundary(stretch, label, 'start'):
            reasons.append('cut_at_edge:%s:start' % label)
        if edges[label]['end'] and hi - stretch['end'] > .02 and not _cue_boundary(stretch, label, 'end'):
            reasons.append('cut_at_edge:%s:end' % label)
    return {'accepted': not reasons, 'reasons': sorted(set(reasons)), 'metrics': metrics}


# ---------------------------------------------------------------------------------------------------- decision

def fallback_outcome(stretch, reasons, **extra):
    """The caller keeps today's behaviour for this stretch."""
    return {'decision': 'fallback', 'stretch_id': stretch['id'], 'reasons': sorted(set(reasons)), 'voices': [], **extra}


def decide_stretch(stretch, parts, turns, *, vectors=None, embed=None, knobs=None):
    """assign_stems + stitch_voices + quality_gate. -> outcome: {'decision': 'split' | 'fallback', 'reasons', 'assignment', 'gate',
    'voices': [{'label', 'slot', 'start', 'end', 'rate', 'samples'}]} (slot 0 = the sorted-first speaker)."""
    k = _knobs(knobs)
    if not stretch.get('eligible', True):
        return fallback_outcome(stretch, stretch['reasons'])
    seconds = sum(float(p['metrics'].get('seconds', 0.)) for p in parts)
    assignment = assign_stems(parts, turns, stretch['speakers'], vectors=vectors, embed=embed, knobs=k)
    summary = {'separator_seconds': _round(seconds, 3),
               'assignment': {x: assignment[x] for x in ('decision', 'reasons', 'order', 'windows', 'links', 'basis')},
               'windows': [{'index': p['index'], 'start': p['start'], 'end': p['end'], 'metrics': p['metrics']} for p in parts]}
    if assignment['decision'] != 'assigned':
        return fallback_outcome(stretch, ['speaker_assignment_ambiguous'] + assignment['reasons'], **summary)
    stitched = stitch_voices(parts, assignment['order'], stretch['speakers'])
    gate = quality_gate(parts, stitched, stretch, turns, knobs=k, embed=embed)
    if not gate['accepted']:
        return fallback_outcome(stretch, gate['reasons'], gate=gate, **summary)
    a, b = stretch['speakers']
    end = stitched['start'] + len(stitched['voices'][a]) / stitched['rate']
    voices = [{'label': label, 'slot': slot, 'start': stitched['start'], 'end': end, 'rate': stitched['rate'],
               'samples': stitched['voices'][label]} for slot, label in enumerate((a, b))]
    return {'decision': 'split', 'stretch_id': stretch['id'], 'reasons': [], 'voices': voices, 'gate': gate, **summary}


def run_stretch(separator, mixture, origin, stretch, turns, *, vectors=None, embed=None, knobs=None, cancelled=lambda: False):
    """separate_stretch + decide_stretch, extending the stretch when a voice is still speaking at the edge of its stem.

    A word is never cut: when the only reasons are cut_at_edge:* the stretch grows by edge_extend_seconds at that side (not past the
    analysed audio or its bounds) and is separated again, at most max_edge_extensions times. A separator ValueError (invalid geometry,
    silence for audible input) is a fallback; a cancellation or any other error propagates. -> outcome; outcome['stretch'] is the
    stretch that was finally used (its id, span and windows may differ from the one passed in)."""
    k = _knobs(knobs)
    reach = (origin, origin + len(mixture) / RATE)
    lo, hi = stretch.get('bounds') or (float('-inf'), float('inf'))
    current = dict(stretch, bounds=[max(lo, reach[0]), min(hi, reach[1])])
    limit = int(k['max_edge_extensions'])
    for attempt in range(limit + 1):
        if not current.get('eligible', True):
            return dict(fallback_outcome(current, current['reasons']), stretch=current)
        try:
            parts = separate_stretch(separator, mixture, origin, current, cancelled=cancelled)
        except ValueError as error:
            return dict(fallback_outcome(current, ['separator_refused'], separator_error=str(error)[:160]), stretch=current)
        outcome = decide_stretch(current, parts, turns, vectors=vectors, embed=embed, knobs=k)
        edges = [r for r in outcome['reasons'] if r.startswith('cut_at_edge:')]
        if outcome['decision'] == 'split' or not edges or len(edges) != len(outcome['reasons']) or attempt >= limit:
            outcome['stretch'] = current
            outcome['extensions'] = attempt
            return outcome
        grow = k['edge_extend_seconds']
        current = extend_stretch(current, start_by=grow if any(r.endswith(':start') for r in edges) else 0.,
                                 end_by=grow if any(r.endswith(':end') for r in edges) else 0., knobs=k)
    raise AssertionError('unreachable')


# ---------------------------------------------------------------------------------------------------- stems on disk, per-voice views

def write_float_wav(path, samples, rate):
    """Mono 32-bit float WAV with a fixed header (no PEAK chunk, no timestamp): the same samples always give the same bytes and hash."""
    import struct
    data = np.ascontiguousarray(samples, dtype='<f4').tobytes()
    header = (b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' + struct.pack('<IHHIIHH', 16, 3, 1, int(rate), int(rate) * 4, 4, 32)
              + b'data' + struct.pack('<I', len(data)))
    with open(path, 'wb') as handle:
        handle.write(header + data)


def write_voice_stems(outcome, folder, *, prefix=None):
    """Write each isolated voice as a mono float32 16 kHz WAV (the reference source of its cues). Adds 'file' and 'sha256' to the
    outcome's voices (in place) and returns them. The files are named from the stretch id, so a rerun overwrites the same names, and
    the bytes depend only on the samples (write_float_wav), so the hash in the receipt is reproducible."""
    from pathlib import Path
    from .media import file_hash
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for voice in outcome['voices']:
        path = folder / ('%s-voice%d.wav' % (prefix or outcome['stretch_id'], voice['slot']))
        write_float_wav(path, voice['samples'], voice['rate'])
        voice.update(file=str(path), sha256=file_hash(path))
    return outcome['voices']


def voice_turns(samples, label, start, *, rate=RATE, join_seconds=.25):
    """Raw-turn dicts of ONE isolated voice (absolute seconds), from the stem's own speech activity. They stand in for the
    diarizer's turns when make_line_reference / dominant_reference_label look at the voice's own stem: inside it the speaker is solo."""
    intervals = _activity.activity_intervals(samples, rate, start, start + len(samples) / rate, start)
    return [{'start': a, 'end': b, 'speaker': label} for a, b in _activity.merge_intervals(intervals, join_seconds)]


def upsample_voice(samples, start, origin, frames, *, rate=48000, source_rate=RATE, channels=2, pan=None):
    """The isolated voice as a sample-major (frames, channels) float32 array on the pipeline's 48 kHz grid (zero outside the
    stretch): a drop-in ``speech`` for make_line_reference / actor_mix_levels, so a cue is cut from, and levelled by, its OWN
    voice. ``pan`` (per-channel weights) defaults to equal weights."""
    from scipy.signal import resample_poly
    if rate % source_rate:
        raise ValueError('The target rate must be a multiple of the stem rate')
    wide = resample_poly(np.asarray(samples, dtype=np.float64), rate // source_rate, 1)
    weights = np.ones(channels) if pan is None else np.asarray(pan, dtype=np.float64)
    out = np.zeros((frames, channels), dtype=np.float32)
    first = round((start - origin) * rate)
    lo, hi = max(0, first), min(frames, first + len(wide))
    if hi > lo:
        out[lo:hi] = (wide[lo - first:hi - first, None] * weights[None, :]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------------------------- (d) cues

def _cue_id(slot, cue):
    key = [VERSION, 'cue', slot, _round(cue['start'], 3), _round(cue['end'], 3), hashlib.sha256(str(cue['text']).encode()).hexdigest()[:12]]
    return hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()[:24]


def reference_plan(samples, start, cue, *, knobs=None, rate=RATE):
    """What the cue's OWN stem can offer as its clone reference. own_seconds = speech of this voice within 0.1 s around the cue;
    mode 'own' (>= own_reference_seconds: the ordinary line-own reference works), 'own_short' (a one-word interjection: shorter than
    that but a real voice, >= min_voice_seconds; the reference must be this audio, optionally helped by more audio of the SAME
    diarized speaker, never a stranger's bank voice) or 'too_short' (no usable voice: the cue is not built). span = the stem speech
    around the cue, the part of the stem to cut the reference from."""
    k = _knobs(knobs)
    around = _intersection(_active_spans(samples, rate, start), [(float(cue['start']) - .1, float(cue['end']) + .1)])
    own = _length(around)
    mode = 'own' if own >= k['own_reference_seconds'] else 'own_short' if own >= k['min_voice_seconds'] else 'too_short'
    return {'own_seconds': _round(own, 3), 'mode': mode, 'needs_reference_help': mode != 'own',
            'span': [around[0][0], around[-1][1]] if around else None}


def own_level(samples, start, cue, *, rate=RATE):
    """Loudness of this voice in its own stem inside the cue: active-frame RMS in dBFS (speech_activity.active_rms_db, the measure
    actor_mix_levels uses), in the SEPARATOR INPUT scale. Multiply the stem by source_scale() to get speech-stem units."""
    i, j = max(0, round((float(cue['start']) - start) * rate)), min(len(samples), round((float(cue['end']) - start) * rate))
    return {'own_active_db': _round(_activity.active_rms_db(np.asarray(samples[i:j], dtype=np.float32), rate), 2), 'scale': 'separator-input'}


def source_scale(mixture, speech, rate=RATE):
    """Linear factor that maps the separator's input (the 16 kHz recognition track) to the speech stem's scale: ratio of the active
    RMS of ``speech`` (the strongest non-LFE channel of the speech/removal stem, resampled to 16 kHz, over the same seconds) to that
    of ``mixture``. Needed because the recognition track carries the centre-gain, mid mix, local gain and limiter that the stem
    does not. Voice stems times this factor are the original voices' loudness: feed upsample_voice(stem * scale) to
    actor_mix_levels and every take is matched to ITS OWN voice (a one-word reply as loud as the original word, nothing boosted).
    A time-varying local gain makes this an average over the stretch (documented approximation)."""
    low, high = _activity.active_rms_db(mixture, rate), _activity.active_rms_db(speech, rate)
    if low <= _activity.SILENCE_DB or high <= _activity.SILENCE_DB:
        raise ValueError('The level scale cannot be established from silence')
    return float(10 ** ((high - low) / 20))


def next_other_start(cue, cues, default):
    """Start of the next cue that is NOT the partner take of the same stretch (render_window_seconds / silence_slack must not wall a
    take at its partner: the partner may overlap, so the take keeps its full length)."""
    group = (cue.get('two_voice') or {}).get('stretch_id')
    return min([c['start'] for c in cues if c['start'] > cue['start'] and not (group and (c.get('two_voice') or {}).get('stretch_id') == group)]
               + [default])


def build_voice_cues(stretch, outcome, transcripts, *, language=None, plan_options=None, allow_partial=False, knobs=None):
    """Pipeline-shaped cues for the two isolated voices of an accepted stretch.

    outcome: decide_stretch's 'split' outcome after write_voice_stems (every voice has 'file' and 'sha256'). transcripts: {label:
    Whisper-style segments of THAT voice's stem, times relative to the start of the stem file}. Each voice is planned with the
    pipeline's own dialogue.plan_dialogue (one synthetic turn covering the whole stem, so the speaker label is the diarized one),
    which gives the same cue keys, word times and 'unclear' flags as a normal cue, and then marked:
      id          stable: sha256(version, voice slot, start, end, text) - the same line gets the same id on a rerun;
      speaker     the diarized raw label (the pipeline's binding step then makes it 'line-<id>');
      voice_source {'mode': 'line-own', 'line_reason': 'two_voice_own_stem', 'reference_source': 'two-voice-stem', ...}: the reference is
                  the voice's OWN stem, never a bank voice (the caller must keep mode 'line-own' for these cues);
      two_voice   the receipt (stretch, slot, lane = slot, partner, core, stem file/sha256, gate metrics, replaced cue ids, model sha);
      overlapped / turn_overlap False and no overlap_intervals: the cue is built from an isolated voice, not from a mixture.
    Per voice: no transcript, an unclear / non-verbal cue (low ASR confidence on the stem), or a reference under min_voice_seconds make
    the voice not dubbable. By default that is a FALLBACK for the whole stretch (no cues). With ``allow_partial`` the dubbable voice
    is still dubbed and the other is returned in 'kept_voices' (its stem + reasons): it gets NO cue and stays audible as the original
    voice, because its stem is taken out of the removal stem (without_voice); it is never silenced. A voice's dub is level-matched to
    its own stem (two_voice.level, source_scale). -> {'cues', 'fallback', 'reasons', 'kept_voices', 'discarded', 'issues'}."""
    from .dialogue import plan_dialogue
    from .nonverbal import nonverbal_reason, sparse_speech_reason
    if outcome.get('decision') != 'split':
        return {'cues': [], 'fallback': True, 'reasons': list(outcome.get('reasons') or ['not_split']), 'kept_voices': [], 'discarded': [],
                'issues': []}
    stretch = outcome.get('stretch') or stretch                     # the stretch as finally used (edge extensions)
    voices = sorted(outcome['voices'], key=lambda v: v['slot'])
    cues, reasons, discarded, issues, kept = [], [], [], [], []
    for voice in voices:
        partner = next(v['label'] for v in voices if v is not voice)
        turn = [{'start': voice['start'], 'end': voice['end'], 'speaker': voice['label']}]
        planned, problems, dropped = plan_dialogue(list(transcripts.get(voice['label']) or ()), turn, (), origin=voice['start'],
                                                   **(plan_options or {}))
        discarded += dropped
        issues += problems
        planned.sort(key=lambda c: (c['start'], c['end']))
        for cue in planned:
            reason = nonverbal_reason(cue.get('text')) or sparse_speech_reason(float(cue['end']) - float(cue['start']), len(cue.get('words') or ()))
            if reason:
                cue['unclear'], cue['nonverbal'] = True, reason
        mine = []
        if not planned:
            mine.append('no_transcript:' + voice['label'])
        elif any(c.get('unclear') for c in planned):
            mine.append('voice_not_dubbable:' + voice['label'])
        made = []
        for cue in planned:
            ident = _cue_id(voice['slot'], cue)
            plan = reference_plan(voice['samples'], voice['start'], cue, knobs=knobs, rate=voice['rate'])
            if plan['mode'] == 'too_short' and not cue.get('unclear'):
                mine.append('reference_too_short:' + voice['label'])
            cue.update(id=ident, speaker=voice['label'], local_speaker=voice['label'], basis='audio', overlapped=False, turn_overlap=False,
                       overlap_intervals=[], overlap_seconds=0.0, cue_seconds=float(cue['end']) - float(cue['start']),
                       speech_start=cue['start'], speech_end=cue['end'], language=language, language_from_audio=False,
                       voice_source={'version': 'voice-source-v1', 'mode': 'line-own', 'local_speaker': voice['label'], 'bank_ident': None,
                                     'bind_reason': None, 'line_reason': 'two_voice_own_stem', 'reference_source': 'two-voice-stem',
                                     'reference_file': voice.get('file'), 'reference_sha256': voice.get('sha256')},
                       two_voice={'version': VERSION, 'stretch_id': stretch['id'], 'voice': voice['label'], 'slot': voice['slot'],
                                  'lane': voice['slot'], 'partner': partner, 'stretch': [stretch['start'], stretch['end']],
                                  'core': [list(c) for c in stretch['core']], 'replaces_cue_ids': list(stretch['cue_ids']),
                                  'stem': {'file': voice.get('file'), 'sha256': voice.get('sha256'), 'rate': voice['rate'],
                                           'start': voice['start'], 'samples': int(len(voice['samples']))},
                                  'gate': (outcome.get('gate') or {}).get('metrics'), 'model_sha256': MODEL_SHA256,
                                  'assignment_basis': (outcome.get('assignment') or {}).get('basis'),
                                  'reference_plan': plan, 'level': own_level(voice['samples'], voice['start'], cue, rate=voice['rate'])})
            made.append(cue)
        if mine and allow_partial:
            kept.append({'label': voice['label'], 'slot': voice['slot'], 'reasons': sorted(set(mine)), 'stem': voice.get('file'),
                         'sha256': voice.get('sha256'), 'start': voice['start'], 'end': voice['end']})
        else:
            cues += made
        reasons += mine
    if len({c['id'] for c in cues}) != len(cues):
        raise ValueError('Two-voice cue ids collide')
    if (reasons and not allow_partial) or not cues:
        return {'cues': [], 'fallback': True, 'reasons': sorted(set(reasons)), 'kept_voices': [], 'discarded': discarded, 'issues': issues}
    return {'cues': cues, 'fallback': False, 'reasons': sorted(set(reasons)), 'kept_voices': kept, 'discarded': discarded, 'issues': issues}


def without_voice(removal, voice_view):
    """The removal stem with one isolated voice taken out (a copy): where this voice is NOT dubbed it must stay audible. The mix
    subtracts ``removal`` from the original wherever a dub plays; subtracting ``removal - voice`` leaves that voice's own sound in
    the soundtrack at its original loudness and removes only the dubbed voice. ``voice_view`` is upsample_voice(stem * source_scale())
    on the same grid as ``removal`` (a sample-major array of the same shape)."""
    removal = np.asarray(removal)
    if removal.shape != np.asarray(voice_view).shape:
        raise ValueError('The voice view and the removal stem must share one grid')
    return (removal.astype(np.float32) - np.asarray(voice_view, dtype=np.float32)).astype(removal.dtype if removal.dtype.kind == 'f' else np.float32)


# ---------------------------------------------------------------------------------------------------- hidden second voice

def probe_candidates(cues, turns, stretches=(), *, knobs=None, within=None):
    """Transcribed cues that MAY hide a second voice the diarizer did not report (a reply or an exclamation inside a longer line).

    The diarizer cannot be the only witness: a cue whose span has one diarized speaker can still hold a second voice. Separating
    every cue would cost the whole dialogue, so only cues with a cheap sign of trouble are proposed, strongest first, until
    probe_max_seconds of audio is used: flagged_overlap (annotate_overlaps), other_speaker_touches (a different speaker's turn
    shorter than the overlap threshold), low_word_confidence (>= 25 % of the words under 0.5 probability), unclear. A cue already
    inside an eligible stretch is skipped. -> [{'cue_id', 'label', 'start', 'end', 'signals', 'windows'}] sorted by start."""
    k = _knobs(knobs)
    spans = _turns(turns)
    taken = [(s['start'], s['end']) for s in stretches if s.get('eligible')]
    rows = []
    for cue in cues or ():
        a, b = _span(cue, 'cue')
        if (within is not None and _shared(a, b, float(within[0]), float(within[1])) <= 0) or b - a > k['max_stretch_seconds']:
            continue
        if any(_shared(a, b, x, y) > k['cue_tolerance_seconds'] for x, y in taken):
            continue
        label = _dominant_label(cue, spans)
        words = cue.get('words') or []
        signals = [name for name, hit in (
            ('flagged_overlap', bool(cue.get('overlapped') or cue.get('turn_overlap') is True)),
            ('other_speaker_touches', any(l and l != label and _shared(a, b, s, e) > 0 for s, e, l in spans)),
            ('low_word_confidence', bool(words) and sum(1 for w in words if float(w.get('probability', 1.)) < .5) / len(words) >= .25),
            ('unclear', bool(cue.get('unclear')))) if hit]
        if signals:
            rows.append({'cue_id': cue.get('id'), 'label': label, 'start': a, 'end': b, 'signals': signals})
    rows.sort(key=lambda r: (-len(r['signals']), r['start']))
    budget, chosen = k['probe_max_seconds'], []
    for row in rows:
        if row['end'] - row['start'] <= budget:
            budget -= row['end'] - row['start']
            row['windows'] = plan_windows(row['start'] - .2, row['end'] + .2, max_seconds=k['window_seconds'],
                                          overlap=k['window_overlap_seconds'], minimum=k['min_window_seconds'])
            chosen.append(row)
    return sorted(chosen, key=lambda r: r['start'])


def probe_hidden_voice(parts, cue, turns, *, knobs=None, rate=RATE):
    """Is there a second voice inside this single-speaker cue? (flag only: nothing is dubbed from a probe.)

    ``parts`` = separate_stretch output over the cue's windows. Per window, inside the cue: the louder stem is the primary voice, the
    quieter one the candidate. Flagged when the diarizer reported no other speaker there, the candidate reaches min_voice_db and is at
    most hidden_max_gap_db under the primary, is active >= hidden_min_seconds, speaks together with the primary for >=
    min_simultaneous_seconds, and does NOT correlate with it (> hidden_max_cross_corr is an echo or a copy of the same voice, the
    usual false positive), and the residual is acceptable. The flag must still be confirmed by transcribing the candidate stem: real
    words only, never a repeated stock phrase (Whisper invents those on noise). -> {'flagged', 'reasons', 'windows': [...],
    'secondary_intervals': [[s, e]], 'primary_stem': {window: stem index}}."""
    k = _knobs(knobs)
    spans = _turns(turns)
    label = _dominant_label(cue, spans)
    a, b = _span(cue, 'cue')
    rows, reasons, intervals, primary = [], set(), [], {}
    for part in parts:
        base, stems = part['start'], part['stems']
        lo, hi = max(a, base), min(b, part['end'])
        if hi - lo < .2:
            continue
        db = [_activity.active_rms_db(_core_samples(stems[i], base, rate, [(lo, hi)]), rate) for i in (0, 1)]
        major = int(db[1] > db[0])
        minor = 1 - major
        act = [_intersection(_active_spans(stems[i], rate, base), [(lo, hi)]) for i in (0, 1)]
        together = _intersection(act[0], act[1])
        x = _core_samples(stems[major], base, rate, together)
        y = _core_samples(stems[minor], base, rate, together)
        n = min(len(x), len(y))
        row = {'window': part['index'], 'primary_db': _round(db[major], 2), 'candidate_db': _round(db[minor], 2),
               'gap_db': _round(db[major] - db[minor], 2), 'candidate_seconds': _round(_length(act[minor]), 3),
               'simultaneous_seconds': _round(_length(together), 3), 'cross_corr': _round(_ncc(x[:n], y[:n]), 4) if n else None,
               'residual_ratio': _round(float(part['metrics'].get('residual_ratio', 0.)), 4)}
        why = []
        if any(l and l != label and _shared(lo, hi, s, e) > 0 for s, e, l in spans):
            why.append('reported_by_diarizer')
        if row['residual_ratio'] > k['max_residual']:
            why.append('residual_too_high')
        if db[minor] < k['min_voice_db'] or row['gap_db'] > k['hidden_max_gap_db']:
            why.append('second_stem_too_weak')
        if _length(act[minor]) < k['hidden_min_seconds']:
            why.append('second_stem_too_short')
        if _length(together) < k['min_simultaneous_seconds']:
            why.append('no_simultaneous_speech')
        if row['cross_corr'] is not None and abs(row['cross_corr']) > k['hidden_max_cross_corr']:
            why.append('stems_too_alike')
        row['reasons'] = why
        rows.append(row)
        if not why:
            intervals += [list(i) for i in act[minor]]
            primary[part['index']] = major
        reasons |= set(why)
    flagged = bool(intervals)
    return {'flagged': flagged, 'reasons': [] if flagged else sorted(reasons or {'no_window_inside_cue'}), 'windows': rows,
            'secondary_intervals': intervals, 'primary_stem': primary}


# ---------------------------------------------------------------------------------------------------- scheduling

def schedule_partnered(cues, durations, *, origin, end, max_early_seconds=.75, gap_seconds=.04, fits=None):
    """relay.timing.schedule_lines in which the two takes of ONE stretch do not wall each other (they may overlap in time).

    The rule of schedule_lines is unchanged for every other pair: a take starts no earlier than the end of every earlier take (plus
    the gap) and ends before the start of the next cue (minus the gap). The only difference: a cue whose two_voice.stretch_id equals
    another cue's ignores that cue, both as 'previous take' and as 'next start'. Without two_voice cues the output is
    byte-identical to schedule_lines (tests pin that). ``cues`` are sorted by start like schedule_lines expects."""
    from . import timing
    if len(cues) != len(durations):
        raise ValueError('Each line needs its generated duration')
    if not 0 <= max_early_seconds <= 2 or gap_seconds < 0:
        raise ValueError('Invalid dialogue scheduling limits')
    scheduled = durations if fits is None else timing.fitted_durations(durations, fits)
    group = [(c.get('two_voice') or {}).get('stretch_id') for c in cues]
    result, issues = [], []
    for index, (cue, duration) in enumerate(zip(cues, scheduled)):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Generated duration must be positive')
        others = [j for j in range(len(cues)) if j != index and not (group[index] and group[j] == group[index])]
        following = [cues[j]['start'] for j in others if j > index]
        before = [result[j]['end'] for j in others if j < index]
        limit = min(end, (following[0] if following else end) - gap_seconds)
        earliest = max(origin, (max(before) + gap_seconds) if before else origin, cue['start'] - max_early_seconds)
        target = max(earliest, min(cue['start'], limit - duration))
        stop = target + duration
        overrun = stop > limit + .0001
        if overrun:
            issues.append({'kind': 'speech_timing_overrun', 'cue': cue['id'], 'seconds': duration, 'available_seconds': max(0, limit - earliest)})
        result.append({'start': target, 'end': stop, 'source_start': cue['start'], 'source_end': cue['end'],
                       'early_seconds': max(0, cue['start'] - target), 'speed': 1.0, 'timing_overrun': overrun})
    if fits is not None:
        timing.annotate_fits(result, issues, cues, fits)
    return result, issues


# ---------------------------------------------------------------------------------------------------- (e) receipt

def new_receipt(knobs=None, *, threads=None):
    """The audit record of one window. Nothing here is a claim of quality: stems are 'unverified vocal candidates'."""
    return {'version': VERSION, 'owner_rule': OWNER_RULE, 'state': 'unverified_vocal_candidates',
            'engine': dict(ENGINE, threads=threads), 'policy': _plain(_knobs(knobs)), 'stretches': [], 'totals': {}}


def record_stretch(receipt, stretch, outcome, built=None):
    """Append one stretch with everything a human needs: windows, evidence, gate metrics, decision, fallback reasons, cue ids."""
    stretch = outcome.get('stretch') or stretch                      # as finally used (edge extensions change span and windows)
    row = {'id': stretch['id'], 'start': stretch['start'], 'end': stretch['end'], 'speakers': list(stretch['speakers']),
           'extensions': outcome.get('extensions', 0),
           'core': stretch['core'], 'overlap_seconds': stretch['overlap_seconds'], 'windows': stretch['windows'],
           'cue_state': stretch['cue_state'], 'voices_transcribed': stretch['voices_transcribed'],
           'replaced_cue_ids': stretch['cue_ids'], 'eligible': stretch['eligible'], 'detect_reasons': stretch['reasons'],
           'decision': outcome['decision'], 'reasons': outcome['reasons'], 'separator_seconds': outcome.get('separator_seconds', 0.),
           'assignment': outcome.get('assignment'), 'gate': outcome.get('gate'), 'separator_windows': outcome.get('windows'),
           'separator_error': outcome.get('separator_error'),
           'voices': [{k: v.get(k) for k in ('label', 'slot', 'file', 'sha256', 'start', 'end', 'rate')} for v in outcome['voices']]}
    if built is not None:
        row['cue_fallback'] = built['fallback']
        row['cue_reasons'] = built['reasons']
        row['kept_voices'] = built.get('kept_voices') or []
        row['cues'] = [{'id': c['id'], 'voice': c['two_voice']['voice'], 'start': c['start'], 'end': c['end'], 'text': c['text']}
                       for c in built['cues']]
        if built['fallback'] and outcome['decision'] == 'split':
            row['decision'] = 'fallback'
            row['reasons'] = sorted(set(outcome['reasons']) | set(built['reasons']))
    receipt['stretches'].append(_plain(row))
    return receipt


def close_receipt(receipt):
    """Totals: stretches seen / eligible / split / fallback, fallback reasons counted, separator CPU seconds."""
    rows = receipt['stretches']
    reasons = {}
    for row in rows:
        if row['decision'] != 'split':
            for reason in row['reasons']:
                reasons[reason] = reasons.get(reason, 0) + 1
    receipt['totals'] = {'stretches': len(rows), 'eligible': sum(1 for r in rows if r['eligible']),
                         'split': sum(1 for r in rows if r['decision'] == 'split'),
                         'fallback': sum(1 for r in rows if r['decision'] != 'split'), 'fallback_reasons': dict(sorted(reasons.items())),
                         'separator_seconds': _round(sum(r['separator_seconds'] or 0. for r in rows), 3),
                         'audio_seconds': _round(sum(r['end'] - r['start'] for r in rows if r['eligible']), 3)}
    return receipt
