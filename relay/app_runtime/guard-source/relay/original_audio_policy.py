"""Explicit natural-short exceptions and channel-preserving mix protection.

No model calls or file writes. Keeping original speech is a reported exception,
not a resolved actor identity. Mixing evidence concerns decoded PCM only; it
does not establish separator/music/effects fidelity or listening acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .short_utterance_policy import plan_short_original


# Only failures that retaining the original short utterance actually avoids.
# Overlap, singing, timing, stale identity, clipping and unknown failures must
# not be removed just because another issue calls a cue's speaker unresolved.
SHORT_ORIGINAL_ISSUE_KINDS = frozenset({
    'speaker_unresolved', 'actor_reference_unavailable', 'uncertain_transcript',
    'unclear_dialogue_without_aligned_caption',
})
LAYOUT_CHANNELS = {'mono': 1, 'stereo': 2, '5.1': 6, '5.1(side)': 6}


def plan_original_audio(cues, speech_regions, issues, *, origin, end,
                        allowed_issue_kinds=SHORT_ORIGINAL_ISSUE_KINDS,
                        max_seconds=1.0, max_words=2,
                        continuation_gap_seconds=.35, boundary_margin_seconds=.15):
    """Select natural short unresolved cues; retain all other issue blockers.

    ``issues`` is the complete issue list for these source cues, including
    global issues. Only explicit ``cue`` IDs are eligible for an exception;
    unscoped/stale-ID issues block selection until the caller resolves scope.
    Embedded cue issues are also considered and never silently discarded.
    Full acoustic source regions and observation-window boundaries are required.
    Returned intervals remain absolute and must travel across chunk boundaries.
    """
    cues, issues = list(cues), deepcopy(list(issues))
    allowed = frozenset(allowed_issue_kinds)
    if not allowed <= SHORT_ORIGINAL_ISSUE_KINDS:
        raise ValueError('Only supported short-original issue kinds may be allowed')
    def normalize_flags(row):
        result = deepcopy(row)
        if result.get('overlap'):
            result['overlapped'] = True
        if re.search('[\u266a\u266b]', str(result.get('text') or '')):
            result['singing'] = True
        return result
    natural = plan_short_original(
        [normalize_flags(c) for c in cues], [normalize_flags(r) for r in speech_regions],
        origin=origin, end=end, max_seconds=max_seconds,
        max_words=max_words, continuation_gap_seconds=continuation_gap_seconds,
        boundary_margin_seconds=boundary_margin_seconds)
    ids = {cue['id'] for cue in cues}
    for cue in cues:
        embedded = cue.get('issues', [])
        if not isinstance(embedded, (list, tuple)):
            raise ValueError('Embedded cue issues must be a list')
        for issue in embedded:
            if not isinstance(issue, dict):
                raise ValueError('Issues must be dictionaries with a kind')
            item = deepcopy(issue)
            item.setdefault('cue', cue['id'])
            if item['cue'] != cue['id']:
                raise ValueError('Embedded issue names a different cue')
            if item not in issues:
                issues.append(item)
    if any(not isinstance(issue, dict) or not isinstance(issue.get('kind'), str)
           or not issue['kind'] for issue in issues):
        raise ValueError('Issues must be dictionaries with a nonempty kind')
    if any('cue' in i and i['cue'] is not None and not isinstance(i['cue'], str) for i in issues):
        raise ValueError('Issue cue must be an ID string or absent')
    global_issues = [i for i in issues if i.get('cue') not in ids]
    preserved, intervals, decisions = [], [], []
    for item in natural['decisions']:
        cue_id = item['id']
        relevant = [issue for issue in issues if issue.get('cue') == cue_id]
        blockers = [issue for issue in relevant if issue['kind'] not in allowed]
        reasons = list(item['reasons'])
        if not relevant:
            reasons.append('no_explicit_unresolved_issue')
        if blockers:
            reasons.append('issue_not_covered_by_short_original_policy')
        if global_issues:
            reasons.append('unscoped_or_unknown_cue_issue_requires_review')
        keep = item['eligible'] and bool(relevant) and not blockers and not global_issues
        decision = {**deepcopy(item), 'action': 'keep_original' if keep else 'normal_resolution',
                    'policy_reasons': reasons, 'issues': deepcopy(relevant),
                    'identity_resolved': False, 'human_listening_acceptance': False}
        decisions.append(decision)
        if keep:
            preserved.append(cue_id)
            intervals.append({'cue': cue_id, **item['original_keep_interval'],
                              'basis': 'standalone_short_source_utterance'})
    covered = [deepcopy(i) for i in issues if i.get('cue') in preserved and i['kind'] in allowed]
    remaining = [deepcopy(i) for i in issues if not (i.get('cue') in preserved and i['kind'] in allowed)]
    return {'version': 'original-audio-policy-v1', 'natural_policy': natural['version'],
            'policy': {**natural['policy'], 'allowed_issue_kinds': sorted(allowed)},
            'observed_window': {'start': float(origin), 'end': float(end)},
            'preserved_cue_ids': preserved,
            'render_cue_ids': [cue['id'] for cue in cues if cue['id'] not in preserved],
            'protected_intervals': intervals, 'decisions': decisions,
            'covered_issues': covered, 'remaining_issues': remaining,
            'all_original_issues': issues, 'human_listening_acceptance': False,
            'scope': 'Explicit original-audio exceptions; other blockers remain; no identity acceptance'}


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be finite')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def _intervals(intervals):
    result = []
    for interval in intervals:
        try:
            begin, end = ((interval['start'], interval['end']) if isinstance(interval, dict)
                          else interval)
            begin, end = _finite(begin, 'protection start'), _finite(end, 'protection end')
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('Invalid original protection interval') from error
        if begin < 0 or end <= begin:
            raise ValueError('Original protection intervals must be positive absolute spans')
        result.append((begin, end))
    return result


def original_protection_masks(frames, rate, origin, protected_intervals, *, transition_seconds=.01):
    """Return (exact-source mask, allowed-edit gain) for this absolute PCM block.

    The entire sample cell intersecting a protected span is protected. Fades
    occur outside that span only. Absolute sample coordinates make partitioned
    and full-window mixing agree; pass every relevant interval to every block.
    """
    import numpy as np
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
        raise ValueError('frames must be a nonnegative integer')
    if isinstance(rate, bool) or not isinstance(rate, int) or not 1 <= rate <= 384000:
        raise ValueError('rate must be a supported positive integer')
    origin, transition = _finite(origin, 'origin'), _finite(transition_seconds, 'transition')
    if origin < 0 or not 0 <= transition <= 1:
        raise ValueError('Invalid origin or transition duration')
    intervals = _intervals(protected_intervals)
    first = round(origin*rate)
    positions = np.arange(frames, dtype=np.int64)+first
    protected = np.zeros(frames, dtype=bool)
    allowed = np.ones(frames, dtype=np.float32)
    fade = round(transition*rate)
    for begin, end in intervals:
        # Tiny tolerance avoids an extra cell caused solely by binary rounding.
        left, right = math.floor(begin*rate+1e-9), math.ceil(end*rate-1e-9)
        core = (positions >= left) & (positions < right)
        protected |= core
        allowed[core] = 0
        if fade:
            before = (positions >= left-fade) & (positions < left)
            after = (positions >= right) & (positions < right+fade)
            allowed[before] = np.minimum(allowed[before],
                np.sin((left-positions[before])/fade*(np.pi/2))**2)
            allowed[after] = np.minimum(allowed[after],
                np.sin((positions[after]-right+1)/fade*(np.pi/2))**2)
    return protected, allowed


def protect_dialogue_mix(original, speech, generated, envelope, *, origin, rate,
                         channel_layout, protected_intervals, transition_seconds=.01):
    """Mix with exact original PCM on protected natural-short source intervals.

    A generated tail overlapping protection is suppressed and explicitly marked
    for rescheduling/review. Callers must not publish such a take as complete.
    Input arrays are never modified; there is no downmix or channel reordering.
    This low-level helper accepts approved intervals from plan_original_audio;
    it does not independently authorize arbitrary original-dialogue passthrough.
    """
    import numpy as np
    from .cinematic import replace_dialogue

    tracks = [np.asarray(value) for value in (original, speech, generated)]
    original, speech, generated = tracks
    envelope = np.asarray(envelope)
    channels = LAYOUT_CHANNELS.get(channel_layout)
    if (channels is None or original.ndim != 2 or original.shape[1] != channels
            or any(track.shape != original.shape for track in tracks)
            or envelope.ndim != 1 or len(envelope) != len(original)):
        raise ValueError('Mix geometry must match the explicit mono/stereo/5.1 layout')
    if (original.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or any(track.dtype != original.dtype for track in tracks)):
        raise ValueError('Audio tracks must share float32 or float64 PCM dtype')
    if any(not np.issubdtype(track.dtype, np.floating) or not np.isfinite(track).all()
           for track in tracks):
        raise ValueError('Mix audio must be finite floating-point PCM')
    if (not np.issubdtype(envelope.dtype, np.number) or np.iscomplexobj(envelope)
            or not np.isfinite(envelope).all() or np.any(envelope < 0) or np.any(envelope > 1)):
        raise ValueError('Removal envelope must be finite and between zero and one')
    intervals = _intervals(protected_intervals)
    protected, allowed = original_protection_masks(len(original), rate, origin, intervals,
                                                    transition_seconds=transition_seconds)
    effective_envelope = (envelope*allowed).astype(original.dtype)
    effective_generated = (generated*allowed[:, None]).astype(original.dtype)
    if len(original):
        mixed, receipt = replace_dialogue(original, speech, effective_generated, effective_envelope)
    else:
        mixed = original.copy()
        receipt = {'unchanged_frames': 0, 'original_preserved_outside_replacements': True,
                   'voice_limited_frames': 0, 'background_peak': 0., 'overrange_samples': 0}
    # Even signed zero and original overrange values survive exactly. An
    # overrange source stays a reported output blocker; it is never normalized.
    mixed[protected] = original[protected]
    voice_present = np.any(generated != 0, axis=1)
    suppressed_core = protected & voice_present
    modified_voice = (allowed < 1) & voice_present
    original_unchanged = mixed.dtype == original.dtype and mixed[protected].tobytes() == original[protected].tobytes()
    receipt.update(
        protection_policy='exact-original-core-outside-fade-v1', channel_layout=channel_layout,
        channels=channels, sample_rate=rate, origin_sample=round(float(origin)*rate),
        protected_frames=int(protected.sum()), protected_intervals=[list(i) for i in intervals],
        original_preserved_in_protected_spans=original_unchanged,
        removal_suppressed_core_frames=int(np.count_nonzero(protected & (envelope != 0))),
        generated_suppressed_core_frames=int(suppressed_core.sum()),
        generated_modified_transition_frames=int(np.count_nonzero(modified_voice & ~protected)),
        generated_protection_conflict=bool(modified_voice.any()),
        requires_rescheduling=bool(modified_voice.any()),
        overrange_samples=int(np.count_nonzero(np.abs(mixed) > 1)),
        human_listening_acceptance=False,
        scope='Decoded PCM protection and geometry only; not separator or music/effects fidelity')
    if not original_unchanged:
        raise ValueError('Protected original PCM invariant failed')
    return mixed, receipt
