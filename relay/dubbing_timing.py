"""Conservative, model-free source timing plans; never rewrite or stretch speech.

All timestamps are absolute movie seconds. Plans are JSON-compatible and retain
the complete translated text. A plan is evidence for a later renderer, not an
approval of translation, speaker identity, separation, or audible quality.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

# Optional cross-window carry (see docs/MOVIE-TIMING-CARRY-B4.md). Without a
# ``carry`` argument nothing below changes a plan.
CARRY_VERSION = 'timing-carry-v1'
# Same 10 ms edit transition protect_dialogue_mix places outside a kept span.
PROTECTION_TRANSITION_SECONDS = .01
# Window-edge identity tolerance; far below one 48 kHz sample (as same_span).
CARRY_EDGE_TOLERANCE = 1e-6
CARRY_SCOPE_FIELDS = ('source_fingerprint', 'audio_index', 'channel_layout',
                      'revision', 'engine_fingerprint', 'policy_version')
_CARRY_LAYOUTS = ('mono', 'stereo', '5.1', '5.1(side)')
_CARRY_OUT_KEYS = frozenset({'version', 'scope', 'scope_key', 'policy', 'origin', 'edge',
                             'horizon', 'cues', 'protected', 'next_speech_start', 'digest'})


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a finite number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} must be a finite number')
    return result


def _interval(row, label):
    start, end = _number(row['start'], label), _number(row['end'], label)
    if end <= start:
        raise ValueError(f'{label} must have positive duration')
    return start, end


def _intersects(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def _slots(cue, start, end, minimum_pause):
    """Keep explicit source windows, or expose word-gap candidates honestly."""
    explicit = cue.get('speech_windows')
    rows = explicit if explicit is not None else cue.get('words', ())
    basis = 'explicit-source-speech-windows' if explicit is not None else 'asr-word-gaps'
    intervals = sorted(_interval(row, 'source speech window') for row in rows)
    if not intervals:
        return [{'start': start, 'end': end}], [], 'cue-boundaries'
    if any(a < start or b > end for a, b in intervals):
        raise ValueError('Source speech windows must stay within their cue')
    merged = []
    for a, b in intervals:
        # Explicit windows preserve every positive pause; word timing exposes
        # only larger gaps, which still require acoustic/phrase confirmation.
        threshold = 0.0 if explicit is not None else minimum_pause
        if merged and a - merged[-1]['end'] <= threshold:
            merged[-1]['end'] = max(merged[-1]['end'], b)
        else:
            merged.append({'start': a, 'end': b})
    pauses = [{'start': left['end'], 'end': right['start'], 'basis': basis}
              for left, right in zip(merged, merged[1:])]
    return merged, pauses, basis


def _validate_scope(scope):
    if not isinstance(scope, Mapping):
        raise ValueError('Timing carry needs a scope mapping')
    missing = [field for field in CARRY_SCOPE_FIELDS if field not in scope]
    if missing:
        raise ValueError('Timing carry scope is missing ' + ', '.join(missing))
    if not isinstance(scope['source_fingerprint'], str) or not scope['source_fingerprint']:
        raise ValueError('Timing carry scope needs the source fingerprint')
    for field in ('audio_index', 'revision'):
        if isinstance(scope[field], bool) or not isinstance(scope[field], int):
            raise ValueError(f'Timing carry scope needs an integer {field}')
    if scope['revision'] < 1:
        raise ValueError('Timing carry revision must be positive')
    if scope['channel_layout'] not in _CARRY_LAYOUTS:
        raise ValueError('Timing carry scope needs an explicit channel layout')
    if not scope['engine_fingerprint'] or not scope['policy_version']:
        raise ValueError('Timing carry scope needs the engine fingerprint and policy version')
    result = dict(scope)
    json.dumps(result, sort_keys=True)  # must be JSON-compatible to be keyed
    return result


def timing_carry_scope(*, source_fingerprint, audio_index, channel_layout, revision,
                       engine_fingerprint, policy_version, **extra):
    """Everything a carried timing decision depends on (cf. protection_scope).

    Extra JSON-compatible fields (for example a job ID or render settings)
    are part of the key. Any change invalidates a previous window's carry.
    """
    return _validate_scope({'source_fingerprint': source_fingerprint, 'audio_index': audio_index,
                            'channel_layout': channel_layout, 'revision': revision,
                            'engine_fingerprint': engine_fingerprint,
                            'policy_version': policy_version, **extra})


def _sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def timing_carry_key(scope, policy):
    """Same hashing as dialogue_ledger.scope_key, over scope + timing policy."""
    return _sha256({**_validate_scope(scope), 'timing_policy': dict(policy),
                    'carry_version': CARRY_VERSION})


def _cue_digest(cue):
    return _sha256([cue['id'], _number(cue['start'], 'cue'), _number(cue['end'], 'cue'),
                    cue.get('speaker'), cue.get('english')])


def _carry_body(carry):
    return {key: value for key, value in carry.items() if key != 'digest'}


def _verify_previous(previous):
    """A restored carry must be exactly what a plan emitted; never repaired."""
    if not isinstance(previous, Mapping) or set(previous) != _CARRY_OUT_KEYS:
        raise ValueError('Previous timing carry is malformed')
    if previous['version'] != CARRY_VERSION:
        raise ValueError('Previous timing carry has an unsupported version')
    if previous['digest'] != _sha256(_carry_body(previous)):
        raise ValueError('Previous timing carry digest does not match its content')
    _validate_scope(previous['scope'])
    _number(previous['edge'], 'carry edge')
    for row in previous['cues']:
        _interval({'start': row['source_start'], 'end': row['source_end']}, 'carried cue')
        if _number(row['latest_end'], 'carried latest end') < row['source_start']:
            raise ValueError('Carried cue ends before it starts')
    for row in previous['protected']:
        _interval(row, 'carried protected span')
    return previous


def check_timing_carry(scope, previous=None):
    """Validate a carry scope and restored previous carry before any work.

    Raises ValueError like plan_dubbing_timing would; returns the scope.
    """
    scope = _validate_scope(scope)
    if previous is not None:
        _verify_previous(previous)
    return scope


def _carry_state(carry, origin, end, policy):
    if not isinstance(carry, Mapping):
        raise ValueError('Timing carry must be a mapping')
    unknown = set(carry) - {'scope', 'previous', 'lookahead', 'protected'}
    if unknown:
        raise ValueError('Unknown timing carry fields: ' + ', '.join(sorted(unknown)))
    scope = _validate_scope(carry.get('scope'))
    key = timing_carry_key(scope, policy)
    horizon, ahead = end, []
    lookahead = carry.get('lookahead')
    if lookahead is not None:
        if not isinstance(lookahead, Mapping) or set(lookahead) - {'end', 'speech'}:
            raise ValueError('Timing lookahead needs only end and speech')
        horizon = _number(lookahead['end'], 'lookahead end')
        if horizon < end:
            raise ValueError('Timing lookahead must not end before the plan window')
        ahead = [(row, _interval(row, 'lookahead speech')) for row in lookahead.get('speech', ())]
    protected = [(row.get('cue'), _interval(row, 'protected original span'))
                 for row in carry.get('protected', ())]
    state = {'scope': scope, 'key': key, 'horizon': horizon, 'ahead': ahead,
             'status': 'fresh', 'reason': None, 'changed': [], 'cues': [], 'protected': protected}
    previous = carry.get('previous')
    if previous is None:
        return state
    previous = _verify_previous(previous)
    if previous['scope_key'] != key:
        old = previous['scope']
        changed = sorted(k for k in set(old) | set(scope) if old.get(k) != scope.get(k))
        if previous['policy'] != policy:
            changed.append('timing_policy')
        state.update(status='invalidated', reason='scope_changed', changed=changed or ['scope_key'])
    elif abs(previous['edge'] - origin) > CARRY_EDGE_TOLERANCE:
        # Seek forward/backward, a gap or a partially overlapping window: the
        # previous output timeline is not continued here.
        state.update(status='invalidated', reason='window_not_adjacent')
    else:
        state.update(status='applied', cues=[dict(row) for row in previous['cues']])
        state['protected'] = protected + [(row.get('cue'), (row['start'], row['end']))
                                          for row in previous['protected']]
    return state


def plan_dubbing_timing(cues, *, origin, end, speech_regions=(), pause_windows=(),
                        max_tail_seconds=.25, min_pause_seconds=.25,
                        guard_seconds=.04, carry=None):
    """Build independent actor windows before synthesis without changing text.

    ``cues`` must include every potentially competing cue in this window, not
    only the cues selected for rendering. ``speech_regions`` are additional
    absolute source speech intervals (including uncovered/untranscribed speech).
    ``pause_windows`` are acoustically confirmed global silence intervals, never
    gaps inferred solely from missing captions/transcripts. Neither is required;
    absent evidence never grants additional silence.

    A cue may have absolute ``speech_windows`` or ``words`` and a resolved
    ``speaker``. ``speech_end`` is accepted only as same-speaker source support
    for a bounded tail. Internal pauses produce ``pause_alignment_required``:
    this primitive cannot infer target-language phrase boundaries. The caller
    must resolve those cases before synthesis/playback, retaining full meaning.

    ``carry`` (optional) continues timing across adjacent windows; ``None``
    leaves every result exactly as before. It is a mapping with ``scope``
    (see ``timing_carry_scope``), ``previous`` (the prior window's
    ``plan['carry']['carry_out']`` or None), ``lookahead`` (``end`` and the
    complete observed source ``speech`` beyond this window up to that end)
    and ``protected`` (absolute kept-original spans, e.g. active
    ProtectionLedger rows). Contract: docs/MOVIE-TIMING-CARRY-B4.md.
    """
    origin, end = _interval({'start': origin, 'end': end}, 'plan window')
    tail = _number(max_tail_seconds, 'tail')
    minimum_pause = _number(min_pause_seconds, 'minimum pause')
    guard = _number(guard_seconds, 'guard')
    if not 0 <= tail <= 1 or minimum_pause <= 0 or not 0 <= guard <= 1:
        raise ValueError('Invalid timing policy limits')
    cues = list(cues)
    identities = [cue['id'] for cue in cues]
    if any(not isinstance(ident, str) or not ident for ident in identities):
        raise ValueError('Cue IDs must be nonempty strings')
    if len(set(identities)) != len(identities):
        raise ValueError('Cue IDs must be unique')
    bounds = [_interval(cue, 'cue') for cue in cues]
    regions = [(row, _interval(row, 'speech region')) for row in speech_regions]
    silence = [_interval(row, 'confirmed pause') for row in pause_windows]
    policy = {'max_tail_seconds': tail, 'min_pause_seconds': minimum_pause, 'guard_seconds': guard}
    state = None if carry is None else _carry_state(carry, origin, end, policy)
    # Without carry: horizon is the window end, nothing is carried, and the
    # competing cues/regions are exactly the inputs (previous behaviour).
    horizon = end if state is None else state['horizon']
    carried = {} if state is None else {row['id']: row for row in state['cues']}
    others = list(zip(cues, bounds))
    occupied = regions
    if state is not None:
        # A carried cue's source stays occupied here even if the caller omits it.
        others += [({'id': row['id'], 'speaker': row['speaker']}, (row['source_start'], row['source_end']))
                   for row in carried.values() if row['id'] not in identities]
        occupied = regions + state['ahead']
    issues, result, skipped = [], [], []
    for cue, (start, stop) in zip(cues, bounds):
        ident, speaker = cue['id'], cue.get('speaker')
        if ident in carried:
            # Emitted exactly once, by the window that owned its start.
            skipped.append(ident)
            if _cue_digest(cue) != carried[ident]['digest']:
                issues.append({'kind': 'carried_cue_changed', 'cue': ident})
            continue
        local_issues = []
        def issue(kind, **detail):
            value = {'kind': kind, 'cue': ident, **detail}
            local_issues.append(value)
            issues.append(value)
        if not speaker:
            issue('speaker_unresolved')
        if start < origin or stop > end:
            # With carry, a cue starting here may cross the end edge only when
            # its whole source interval was observed (lookahead horizon).
            if state is None or not (origin <= start < end and stop <= horizon):
                issue('source_cue_crosses_window')
        if cue.get('overlapped'):
            issue('overlapping_dialogue_unresolved')
        if cue.get('singing'):
            issue('singing_requires_pitch_tempo_path')
        slots, pauses, basis = _slots(cue, start, stop, minimum_pause)
        if cue.get('speech_windows') == []:
            issue('explicit_source_speech_windows_empty')
        if slots[0]['start'] > start or slots[-1]['end'] < stop:
            issue('source_window_edges_require_alignment')
        if pauses:
            issue('pause_alignment_required', pause_count=len(pauses), basis=basis)
        text = cue.get('english')
        if not isinstance(text, str) or not text.strip():
            issue('translation_unresolved')

        # A later word in this same actor's turn can support a short natural
        # tail. It never grants permission to consume another actor's turn.
        supported_end, tail_basis = stop, 'none'
        speech_end = _number(cue.get('speech_end', stop), 'speech end')
        if speaker and speech_end > stop:
            supported_end, tail_basis = speech_end, 'same-speaker-source-tail'
        for a, b in silence:
            if a <= stop < b and b > supported_end:
                supported_end, tail_basis = b, 'confirmed-global-pause'
        latest = min(horizon, stop + tail, supported_end)
        overlaps, blockers = [], []
        for other, interval in others:
            if other['id'] == ident:
                continue
            a, b = interval
            if _intersects((start, stop), interval):
                overlaps.append({'cue': other['id'], 'speaker': other.get('speaker'),
                                 'start': max(start, a), 'end': min(stop, b)})
                if speaker and other.get('speaker') == speaker:
                    issue('same_speaker_source_overlap', other_cue=other['id'])
            # Preserve original overlap; prohibit extending it beyond this
            # cue's own source end, or creating a new overlap in its tail.
            if b > stop and a < latest + guard:
                blockers.append({'kind': 'cue', 'id': other['id'],
                                 'speaker': other.get('speaker')})
                latest = min(latest, max(stop, a - guard))
        for region, (a, b) in occupied:
            # Only this containing source turn supports a same-actor tail.
            # A later untranscribed return by that actor is still occupied
            # speech and must not be consumed as if it were a silent gap.
            if region.get('speaker') == speaker and speaker and a <= start and b >= stop:
                continue
            if b > stop and a < latest + guard:
                blockers.append({'kind': 'source-speech', 'speaker': region.get('speaker')})
                latest = min(latest, max(stop, a - guard))
        if state is not None:
            pad = guard + PROTECTION_TRANSITION_SECONDS
            for protected_cue, (a, b) in state['protected']:
                if _intersects((start, stop), (a, b)):
                    issue('cue_overlaps_protected_original', protected_cue=protected_cue,
                          start=a, end=b)
                # Kept original audio and its edit transition are hard walls.
                if b > stop and a < latest + pad:
                    blockers.append({'kind': 'protected-original', 'cue': protected_cue})
                    latest = min(latest, max(stop, a - pad))
            for row in carried.values():
                a, b = max(origin, row['source_end']), row['latest_end']
                if b > a and b > stop and a < latest + guard:
                    blockers.append({'kind': 'carried-generated-tail', 'id': row['id'],
                                     'speaker': row['speaker']})
                    latest = min(latest, max(stop, a - guard))
        latest = max(start, latest)
        result.append({'id': ident, 'speaker': speaker, 'text': text,
                       'source_start': start, 'source_end': stop,
                       'earliest_start': start, 'latest_end': latest,
                       'budget_seconds': max(0., latest - start),
                       'speech_seconds': sum(s['end'] - s['start'] for s in slots),
                       'speech_slots': slots, 'pauses': pauses, 'slot_basis': basis,
                       'tail_seconds': max(0., latest - stop),
                       'tail_basis': tail_basis if latest > stop else 'none',
                       'tail_blockers': blockers, 'overlaps': overlaps,
                       'issues': local_issues})
    plan = {'version': 'source-pause-timing-v1', 'origin': origin, 'end': end,
            'policy': {'max_tail_seconds': tail, 'min_pause_seconds': minimum_pause,
                       'guard_seconds': guard, 'early_shift_seconds': 0.,
                       'speed': 1.0, 'translation_rewriting': False},
            'cues': result, 'issues': issues}
    if state is not None:
        plan['carry'] = _finish_carry(state, plan, cues, bounds, carried, others, occupied,
                                      skipped, policy, origin, end, guard)
    return plan


def _finish_carry(state, plan, cues, bounds, carried, others, occupied, skipped, policy,
                  origin, end, guard):
    """Check carried tails against this window, then emit the next carry."""
    issues = plan['issues']
    planned = [(cue, interval) for cue, interval in zip(cues, bounds) if cue['id'] not in carried]
    pad = guard + PROTECTION_TRANSITION_SECONDS
    for row in carried.values():
        a, b = max(origin, row['source_end']), row['latest_end']
        if b <= a:
            continue
        def conflict(kind, **detail):
            value = {'kind': kind, 'cue': row['id'], **detail}
            if value not in issues:
                issues.append(value)
        # The previous window may already have released this generated tail.
        # Speech it did not observe here means it was wrong: block, never trim.
        for cue, (c, d) in planned:
            if c < b + guard and d > a:
                conflict('carried_tail_crosses_source_speech', other_cue=cue['id'], start=c, end=d)
        for region, (c, d) in occupied:
            # The same actor's continuing turn is the tail's own source support.
            if row['speaker'] and region.get('speaker') == row['speaker'] and c <= a + CARRY_EDGE_TOLERANCE:
                continue
            if c < b + guard and d > a:
                conflict('carried_tail_crosses_source_speech', speaker=region.get('speaker'),
                         start=c, end=d)
        for protected_cue, (c, d) in state['protected']:
            if c < b + pad and d + PROTECTION_TRANSITION_SECONDS > a:
                conflict('carried_tail_crosses_protected_original', protected_cue=protected_cue,
                         start=c, end=d)
    by_id = {cue['id']: cue for cue in cues}
    out_cues = []
    for row in plan['cues']:
        owned = row['source_start'] >= origin and not any(
            i['kind'] == 'source_cue_crosses_window' for i in row['issues'])
        if owned and (row['source_end'] > end or row['latest_end'] > end):
            out_cues.append({'id': row['id'], 'speaker': row['speaker'],
                             'digest': _cue_digest(by_id[row['id']]),
                             'source_start': row['source_start'], 'source_end': row['source_end'],
                             'latest_end': row['latest_end']})
    out_cues.extend(dict(row) for row in carried.values()
                    if row['source_end'] > end or row['latest_end'] > end)
    out_protected = []
    for cue, (a, b) in sorted(state['protected'], key=lambda item: (item[1], str(item[0]))):
        value = {'cue': cue, 'start': a, 'end': b}
        if b > end and value not in out_protected:
            out_protected.append(value)
    horizon = state['horizon']
    spans = ([interval for _, interval in others] + [interval for _, interval in occupied]
             + [interval for _, interval in state['protected']])
    following = [max(a, end) for a, b in spans if b > end and a <= horizon]
    carry_out = {'version': CARRY_VERSION, 'scope': state['scope'], 'scope_key': state['key'],
                 'policy': policy, 'origin': origin, 'edge': end, 'horizon': horizon,
                 'cues': out_cues, 'protected': out_protected,
                 'next_speech_start': min(following) if following else None}
    carry_out['digest'] = _sha256(carry_out)
    return {'version': CARRY_VERSION, 'status': state['status'], 'reason': state['reason'],
            'changed_scope_fields': state['changed'], 'scope_key': state['key'],
            'horizon': horizon, 'carried_cue_ids': skipped, 'carry_out': carry_out}


def place_timed_lines(plan, durations):
    """Return complete-duration placements and issues; never trim an overrun.

    ``durations`` maps cue ID to measured complete generated audio seconds.
    Placements retain plan order and original simultaneous starts. The caller
    must block release if ANY returned issue exists, including plan issues.
    Missing measurements are review cases. Unknown IDs are a caller error.
    Plan text must still match the exact text used for synthesis/readback.
    """
    if not isinstance(durations, Mapping):
        raise ValueError('Durations must map cue IDs to measured seconds')
    if set(durations) - {cue['id'] for cue in plan['cues']}:
        raise ValueError('Duration refers to a cue outside this timing plan')
    issues = [dict(item) for item in plan['issues']]
    placements = []
    for cue in plan['cues']:
        ident = cue['id']
        if ident not in durations:
            issues.append({'kind': 'generated_duration_unresolved', 'cue': ident})
            placements.append(None)
            continue
        duration = _number(durations[ident], 'generated duration')
        if duration <= 0:
            raise ValueError('Generated duration must be positive')
        start = cue['earliest_start']
        stop = start + duration
        overrun_seconds = max(0., stop - cue['latest_end'])
        # A tiny float tolerance does not grant an audible timing extension.
        overrun = overrun_seconds > 1e-9
        if overrun:
            issues.append({'kind': 'speech_timing_overrun', 'cue': ident,
                           'seconds': duration, 'available_seconds': cue['budget_seconds'],
                           'overrun_seconds': overrun_seconds})
        placements.append({'cue': ident, 'speaker': cue['speaker'], 'start': start,
                           'end': stop, 'source_start': cue['source_start'],
                           'source_end': cue['source_end'], 'early_seconds': 0.,
                           'speed': 1.0, 'timing_overrun': overrun,
                           'budget_seconds': cue['budget_seconds'],
                           'overrun_seconds': overrun_seconds})
    return placements, issues
