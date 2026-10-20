"""Validate a prepared candidate before any voice model or playback starts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .media import file_hash
from .original_audio_policy import LAYOUT_CHANNELS, SHORT_ORIGINAL_ISSUE_KINDS, plan_original_audio

ORIGINAL_POLICY_VERSION = 'original-audio-policy-v1'
NATURAL_POLICY_VERSION = 'standalone-short-original-v1'


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Prepared original-audio protection has a non-finite time')
    return float(value)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Original-audio policy evidence is not finite JSON data') from error


def validate_original_policy(policy, *, origin, end, synthesis_cues, channel_layout, channels,
                             all_cues, speech_regions, render_start=None, render_end=None):
    """Validate kept-original decisions separately from the synthesis list.

    Returns absolute (start, end) protected intervals. Unresolved blockers are
    never accepted here: remaining issues must be empty and covered issues must
    be cue-scoped short-original kinds for the preserved cues only.

    Saved eligibility is never trusted. The complete policy is recomputed with
    the current planner and default limits from the plan's own source evidence
    (``all_cues``, complete raw ``speech_regions`` and the policy's
    ``all_original_issues``) and must match exactly. Every source cue inside
    the render range must be either synthesized or kept.
    """
    if not isinstance(policy, dict):
        raise ValueError('Prepared original-audio policy is not a policy receipt')
    if not isinstance(all_cues, list) or not isinstance(speech_regions, list):
        raise ValueError('Original-audio policy validation needs all source cues and raw speech regions')
    render_start = origin if render_start is None else render_start
    render_end = end if render_end is None else render_end
    if policy.get('version') != ORIGINAL_POLICY_VERSION or policy.get('natural_policy') != NATURAL_POLICY_VERSION:
        raise ValueError('Prepared original-audio policy version is stale')
    if LAYOUT_CHANNELS.get(channel_layout) != channels:
        raise ValueError('Protected original audio requires the explicit matching channel layout')
    window = policy.get('observed_window') or {}
    if (_seconds(window.get('start')), _seconds(window.get('end'))) != (float(origin), float(end)):
        raise ValueError('Original-audio policy was planned for a different observation window')
    allowed = (policy.get('policy') or {}).get('allowed_issue_kinds')
    if not isinstance(allowed, list) or not set(allowed) <= SHORT_ORIGINAL_ISSUE_KINDS:
        raise ValueError('Original-audio policy allows unsupported issue kinds')
    kept = policy.get('preserved_cue_ids')
    if (not isinstance(kept, list) or any(not isinstance(i, str) or not i for i in kept)
            or len(set(kept)) != len(kept)):
        raise ValueError('Preserved original cue IDs must be unique strings')
    synthesis = {cue['id'] for cue in synthesis_cues}
    if synthesis & set(kept):
        raise ValueError('A kept-original cue cannot also be synthesized')
    if not synthesis <= set(policy.get('render_cue_ids') or ()):
        raise ValueError('Original-audio policy did not classify every synthesis cue')
    if policy.get('remaining_issues'):
        raise ValueError('Original-audio policy retains unresolved blockers')
    for issue in policy.get('covered_issues') or ():
        if issue.get('cue') not in kept or issue.get('kind') not in allowed:
            raise ValueError('Covered issue is outside the kept short-original cues')
    decisions = {d.get('id'): d for d in policy.get('decisions') or () if isinstance(d, dict)}
    rows = policy.get('protected_intervals')
    if not isinstance(rows, list) or sorted(r.get('cue') for r in rows if isinstance(r, dict)) != sorted(kept)             or len(rows) != len(kept):
        raise ValueError('Each kept cue needs exactly one protected interval')
    intervals = []
    for row in rows:
        begin, stop = _seconds(row.get('start')), _seconds(row.get('end'))
        if not origin <= begin < stop <= end:
            raise ValueError('Protected original interval crosses the prepared window')
        decision = decisions.get(row['cue']) or {}
        if (decision.get('action') != 'keep_original' or decision.get('eligible') is not True
                or decision.get('original_keep_interval') != {'start': row['start'], 'end': row['end']}):
            raise ValueError('Protected interval lacks its matching keep-original decision')
        source = decision.get('cue') or {}
        if decision.get('reasons') or decision.get('policy_reasons') or any(source.get(flag) for flag in (
                'singing', 'overlapped', 'overlap', 'clipped', 'truncated', 'partial', 'source_clipped',
                'left_clipped', 'right_clipped', 'benchmark_cut')):
            raise ValueError('Kept-original decision carries a singing, overlap, clipping or review reason')
        for cue in synthesis_cues:
            if min(stop, cue['end']) > max(begin, cue['start']):
                raise ValueError('Protected original audio overlaps a synthesis cue')
        intervals.append((begin, stop))
    issues = policy.get('all_original_issues')
    if not isinstance(issues, list):
        raise ValueError('Original-audio policy lacks its complete original issue list')
    try:
        expected = plan_original_audio(all_cues, speech_regions, issues, origin=origin, end=end,
                                       allowed_issue_kinds=allowed)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Original-audio policy source evidence is malformed') from error
    if _canonical(expected) != _canonical(policy):
        raise ValueError('Original-audio policy does not match a recomputation from its source evidence')
    source = {cue.get('id'): cue for cue in all_cues if isinstance(cue, dict)}
    for cue in synthesis_cues:
        row = source.get(cue['id'])
        if row is None or (row.get('start'), row.get('end')) != (cue['start'], cue['end']):
            raise ValueError('A synthesis cue is not one of the plan source cues')
    selected = {ident for ident, cue in source.items()
                if cue['end'] > render_start and cue['start'] < render_end}
    if selected != synthesis | set(kept):
        raise ValueError('Every source cue in the render range must be synthesized or explicitly kept')
    return sorted(intervals)


def validate_persisted_protection(rows, scope):
    """Shape check for the prepare-time snapshot of other windows' keeps.

    The snapshot only adds constraints; it never proves source coverage. A
    live ProtectionLedger supplied to the renderer replaces it with fresh rows.
    """
    if scope is not None and not isinstance(scope, dict):
        raise ValueError('Prepared protection scope is malformed')
    if not isinstance(rows, list):
        raise ValueError('Persisted original protection must be a list')
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('cue'), str) or not row['cue']:
            raise ValueError('Persisted original protection row lacks its cue')
        if _seconds(row.get('start')) >= _seconds(row.get('end')) or row['start'] < 0:
            raise ValueError('Persisted original protection row has an invalid interval')
    if rows and scope is None:
        raise ValueError('Persisted original protection lacks its scope')


def load_prepared_actor(config, plan_path, source_wav):
    import numpy as np
    import soundfile as sf

    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('engine_fingerprint') != config.engine_fingerprint():
        raise ValueError('Prepared actor plan is stale; prepare it with the current engine')
    report = plan['report']
    origin, end = float(report['origin']), float(report['end'])
    if not all(math.isfinite(v) for v in (origin, end)) or not 0 <= origin < end or end-origin > 120:
        raise ValueError('Prepared actor timeline is invalid or exceeds its bounded window')
    if report['issues']:
        raise ValueError('Unresolved dialogue cannot enter the playback buffer')
    if report['render_start'] != origin or report['render_end'] != end:
        raise ValueError('Progressive rendering requires the complete prepared window')
    if plan.get('first_pass') is not None:
        raise ValueError('The progressive renderer does not support first-pass plans yet (needs the removal stem)')
    target_plan = report.get('caption_target_plan') or {}
    if target_plan.get('superseded_cue_ids') or target_plan.get('carry_source_cue_ids'):
        raise ValueError('The progressive renderer does not support caption-target plans yet')
    source, rate = sf.read(source_wav, dtype='float32', always_2d=True)
    if not np.isfinite(source).all() or source.shape[1] not in (1, 2, 6):
        raise ValueError('Prepared source samples or channel count are invalid')
    if hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest() != plan['pcm_sha256']:
        raise ValueError('Prepared source PCM changed')
    stem_path = plan_path.parent / 'speech-estimate.wav'
    if file_hash(stem_path) != plan['speech_sha256']:
        raise ValueError('Prepared speech estimate changed')
    speech, stem_rate = sf.read(stem_path, dtype='float32', always_2d=True)
    if rate != 48000 or stem_rate != rate or source.shape != speech.shape or not np.isfinite(speech).all():
        raise ValueError('Prepared PCM geometry or samples changed')
    if abs(len(source) - (end-origin)*rate) > 1:
        raise ValueError('Prepared source does not cover its declared timeline')
    cues, voices = plan['cues'], plan['voices']
    ids = set()
    for cue in cues:
        begin, stop = float(cue['start']), float(cue['end'])
        if not all(math.isfinite(v) for v in (begin, stop)) or not origin <= begin < stop <= end:
            raise ValueError('Progressive cue crosses the prepared window')
        if cue['id'] in ids:
            raise ValueError('Prepared plan repeats an utterance')
        ids.add(cue['id'])
        if not cue.get('speaker') or cue['speaker'] not in voices or not str(cue.get('english', '')).strip():
            raise ValueError('Prepared cue lacks a resolved actor or English translation')
        if cue.get('overlapped') or cue.get('singing'):
            raise ValueError('Overlapping speech and singing require their dedicated validated paths')
    if cues != sorted(cues, key=lambda c: (c['start'], c['end'], c['id'])):
        raise ValueError('Prepared utterances must be in timeline order')
    previous_end = origin
    for cue in cues:
        # Detect the source geometry independently of a diarizer's overlap
        # threshold/flag. Tiny overlaps still need a resolved source path.
        if cue['start'] < previous_end:
            raise ValueError('Simultaneous source utterances require validated overlap separation')
        previous_end = max(previous_end, cue['end'])
    if 'channel_layout' in plan and LAYOUT_CHANNELS.get(plan['channel_layout']) != source.shape[1]:
        raise ValueError('Prepared channel layout does not match the source PCM')
    if 'original_audio_policy' in plan:
        validate_original_policy(plan['original_audio_policy'], origin=origin, end=end,
                                 synthesis_cues=cues, channel_layout=plan.get('channel_layout'),
                                 channels=source.shape[1], all_cues=plan.get('all_cues'),
                                 speech_regions=report.get('raw_turns'))
    validate_persisted_protection(plan.get('persisted_original_protection', []), plan.get('protection_scope'))
    for ident in {cue['speaker'] for cue in cues}:
        voice = voices[ident]
        if file_hash(voice['reference']) != voice['reference_sha256']:
            raise ValueError('Prepared actor reference changed')
    return plan, source, speech, rate
