"""Pure timing and PCM level policies for generic English dubbing.

Source spans are absolute movie times. No model, identity, or subtitle decisions
are made here; source-time coverage remains distinct from translation accuracy.
"""
from __future__ import annotations

import math


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('generic_policy_nonfinite')
    return value


def _intervals(spans):
    result = []
    for span in spans:
        a, b = (span['start'], span['end']) if isinstance(span, dict) else span
        a, b = _number(a), _number(b)
        if b <= a:
            raise ValueError('generic_policy_interval_invalid')
        result.append((a, b))
    merged = []
    for a, b in sorted(result):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def level_gain(raw_vocals, spans, take_pcm, rate, chunk_regions, *, origin):
    """Match active English to active unboosted source speech; never boost the bed."""
    import numpy as np

    rate, origin = _number(rate), _number(origin)
    source = np.asarray(raw_vocals)
    take = np.asarray(take_pcm)
    if (rate <= 0 or source.ndim != 2 or source.shape[1] < 1 or not len(source)
            or take.ndim != 1 or not len(take)):
        raise ValueError('generic_level_geometry_invalid')
    if not np.isfinite(source).all() or not np.isfinite(take).all():
        raise ValueError('generic_level_nonfinite_pcm')

    frame_samples = max(1, round(rate * .04))

    def active_rms(parts):
        powers = []
        for part in parts:
            for offset in range(0, len(part), frame_samples):
                powers.append(np.square(part[offset:offset + frame_samples], dtype=np.float64).mean(axis=0))
        if not powers:
            return 0.0, 0
        powers = np.asarray(powers)
        count = max(1, math.ceil(.2 * len(powers)))
        # Each source channel has its own active estimate; use the louder one.
        loudest = np.sort(powers, axis=0)[-count:]
        return float(np.sqrt(loudest.mean(axis=0)).max()), len(powers)

    def source_active(intervals):
        parts = []
        for a, b in _intervals(intervals):
            left = max(0, min(len(source), round((a - origin) * rate)))
            right = max(0, min(len(source), round((b - origin) * rate)))
            if right > left:
                parts.append(source[left:right])
        return active_rms(parts)

    source_spans = _intervals(spans)
    target, source_frames = source_active(source_spans)
    basis = 'source-active'
    if source_frames < 5 or target < 1e-5:
        target, source_frames = source_active(chunk_regions)
        if source_frames < 5 or target < 1e-5:
            raise ValueError('level_reference_unavailable')
        basis = 'chunk-active'
    take_rms, take_frames = active_rms([take[:, None]])
    if not math.isfinite(target) or not math.isfinite(take_rms) or take_rms <= 1e-8:
        raise ValueError('generic_level_take_silent_or_invalid')
    requested = target / take_rms
    gain = min(2.0, requested)
    return gain, {'policy': 'source-active-rms-v2', 'estimator': 'active20-40ms-v1',
                  'target_active_rms': target, 'take_active_rms': take_rms,
                  'source_frames': source_frames, 'take_frames': take_frames,
                  'frame_ms': 40, 'active_fraction': .2,
                  'target_rms': target,
                  'take_rms': take_rms, 'gain': gain, 'gain_db': 20 * math.log10(gain),
                  'basis': basis, 'capped': requested > 2.0, 'source_spans': source_spans}


def line_onset(segment, region, ordinal):
    start, end = _intervals([segment])[0]
    region_start, _ = _intervals([region])[0]
    if ordinal == 0 and 0 <= start - region_start <= 1.0:
        return region_start
    return start


def owns(onset, begin, end, prior_sources, predecessor_takes, *, source_spans,
         predecessor_ready):
    onset, begin, end = map(_number, (onset, begin, end))
    if begin >= end:
        raise ValueError('generic_policy_interval_invalid')
    if begin <= onset < end:
        return {'owned': True, 'reason': 'onset_in_chunk', 'placement_floor': None,
                'seam_rescue': False}
    refused = {'owned': False, 'reason': 'seam_unowned', 'placement_floor': None,
               'seam_rescue': False}
    if not (begin - .3 <= onset < begin) or not predecessor_ready:
        return refused
    spans = _intervals(source_spans)
    if not spans:
        return refused
    # Caller supplies only takes from a verified, same-revision ready receipt.
    # Missing source provenance must never authorize a duplicate seam take.
    if any(not take.get('source_spans') for take in predecessor_takes):
        return refused
    existing = _intervals(prior_sources) + _intervals(
        [span for take in predecessor_takes for span in take['source_spans']])
    if any(min(b, y) > max(a, x) for a, b in spans for x, y in existing):
        return refused
    return {'owned': True, 'reason': 'unpublished_predecessor_seam',
            'placement_floor': begin, 'seam_rescue': True}


def fit_speed(raw_seconds, target):
    raw_seconds, target = _number(raw_seconds), _number(target)
    if raw_seconds <= 0 or target <= 0:
        raise ValueError('generic_duration_invalid')
    return max(.90, min(1.35, raw_seconds / target))
