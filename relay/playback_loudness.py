"""Playback loudness: ONE constant gain per job, measured on the original source film.

Ported from the textless candidate's standalone module and changed in three ways: the K-weighting
filters run through ``scipy.signal.lfilter`` (the earlier hand-rolled blocked biquad failed numerical
parity), the per-excerpt block energies are exposed (so an interrupted measurement can resume), and
the limiter also exists as an FFmpeg filter expression (``mux_filter``) so the HLS mux applies exactly
the same stateless curve as the NumPy reference (``apply_gain_and_limit``).

Design
------
1. One constant gain per job and source film. It is applied identically to every 30 s segment of the
   dub mix, so the voice/music balance and phrase dynamics are kept and nothing pumps between
   segments. There is no per-chunk or per-line gain.
2. The gain is measured on the ORIGINAL SOURCE audio, never on dubbed output, with ITU-R BS.1770-4
   integrated loudness: K-weighting (two biquads derived for any rate; at 48 kHz they reproduce the
   standard table), 400 ms blocks with 75 % overlap, absolute gate -70 LUFS, relative gate -10 LU.
   Channel weights: mono 1.0; stereo 1.0/1.0; 5.1 (L R C LFE Ls Rs) 1.0 1.0 1.0 0 1.41 1.41.
3. No quiet-first-scene bias: ``sample_plan`` is a deterministic film-wide set of evenly spaced
   excerpts (first/last 2 % excluded). Gating is pooled over all excerpts; no block spans an excerpt
   join. Fewer than ``min_gated_seconds`` of gated audio gives 0 dB ('insufficient_gated_audio').
4. Peak safety: a stateless sample-wise soft-knee limiter. At or below the knee the output is
   exactly ``x * g``; above it a tanh curve (continuous, slope 1 at the knee) approaches but never
   reaches the ceiling. No look-ahead and no state, so the output is identical however the audio is
   split into segments. Trade-off: rare peaks above the knee are softly saturated (a waveshaper, not
   a look-ahead limiter) in exchange for exact segment invariance.

Pure functions only (NumPy; SciPy is imported lazily inside ``k_weight``). No file or process I/O:
``loudness_job`` owns decoding and caching. Every receipt is JSON-serialisable.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

VERSION = 'playback-loudness-v2'  # v2: downmix measured like the engine (16-bit, normalized); v1 records are void
STANDARD = 'ITU-R BS.1770-4'
LIMITER_ID = 'stateless-tanh-soft-knee-v1'

ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1  # 75 % overlap
LOUDNESS_OFFSET = -0.691

DEFAULT_TARGET_LUFS = -23.0
DEFAULT_MAX_GAIN_DB = 30.0
DEFAULT_MIN_GAIN_DB = -10.0
DEFAULT_MIN_GATED_SECONDS = 60.0
DEFAULT_KNEE_DBFS = -3.0
DEFAULT_CEILING_DBFS = -1.0

MONO_WEIGHTS = (1.0,)
STEREO_WEIGHTS = (1.0, 1.0)
SURROUND_51_WEIGHTS = (1.0, 1.0, 1.0, 0.0, 1.41, 1.41)  # L R C LFE Ls Rs
SURROUND_51_CENTER_WEIGHTS = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # the dialogue channel alone

_SHELF_F0 = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_SHELF_VB_EXP = 0.4996667741545416
_HIGHPASS_F0 = 38.13547087602444
_HIGHPASS_Q = 0.5003270373238773
_MIN_RATE = 8000


def _receipt(**fields):
    return {'version': VERSION, **fields}


def _finite_float(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number') from None
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite')
    return number


def _check_rate(rate):
    if isinstance(rate, bool) or int(rate) != rate or int(rate) < _MIN_RATE:
        raise ValueError(f'rate must be an integer >= {_MIN_RATE} Hz')
    return int(rate)


# --- K-weighting -----------------------------------------------------------------------------

def k_weighting_coefficients(rate):
    """Return ((b_shelf, a_shelf), (b_highpass, a_highpass)), a[0] == 1, derived for any rate."""
    rate = _check_rate(rate)
    k = math.tan(math.pi * _SHELF_F0 / rate)
    vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    vb = vh ** _SHELF_VB_EXP
    a0 = 1.0 + k / _SHELF_Q + k * k
    shelf_b = ((vh + vb * k / _SHELF_Q + k * k) / a0,
               2.0 * (k * k - vh) / a0,
               (vh - vb * k / _SHELF_Q + k * k) / a0)
    shelf_a = (1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _SHELF_Q + k * k) / a0)
    k = math.tan(math.pi * _HIGHPASS_F0 / rate)
    a0 = 1.0 + k / _HIGHPASS_Q + k * k
    highpass_b = (1.0, -2.0, 1.0)
    highpass_a = (1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _HIGHPASS_Q + k * k) / a0)
    return (shelf_b, shelf_a), (highpass_b, highpass_a)


def k_weight(signal, rate):
    """Apply both K-weighting stages (zero initial state) to one channel with scipy.signal.lfilter."""
    from scipy.signal import lfilter
    y = np.asarray(signal, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return y.copy()
    for b, a in k_weighting_coefficients(rate):
        y = lfilter(np.asarray(b, dtype=np.float64), np.asarray(a, dtype=np.float64), y)
    return y


def default_weights(channels):
    table = {1: MONO_WEIGHTS, 2: STEREO_WEIGHTS, 6: SURROUND_51_WEIGHTS}
    if channels not in table:
        raise ValueError(f'no default channel weights for {channels} channels; pass channel_weights')
    return table[channels]


# --- integrated loudness ---------------------------------------------------------------------

def _frames(excerpt, index=0):
    data = np.asarray(excerpt, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError(f'excerpt {index} must be (frames,) or (frames, channels)')
    if not np.isfinite(data).all():
        raise ValueError(f'excerpt {index} contains nonfinite samples')
    return data


def block_energies(excerpt, rate, channel_weights=None):
    """Channel-weighted mean-square energy of every 400 ms block (100 ms hop) of ONE excerpt.

    ``excerpt`` is (frames,) or (frames, channels). K-weighting starts from zero state for the excerpt
    and blocks never extend past it. Returns a float64 array (possibly empty).
    """
    rate = _check_rate(rate)
    data = _frames(excerpt)
    weights = default_weights(data.shape[1]) if channel_weights is None else tuple(float(w) for w in channel_weights)
    if len(weights) != data.shape[1]:
        raise ValueError(f'{len(weights)} channel weights for {data.shape[1]} channels')
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('channel_weights must be non-empty, finite and non-negative')
    hop = int(round(HOP_SECONDS * rate))
    block = 4 * hop
    subs = data.shape[0] // hop
    count = max(subs - 3, 0)
    energy = np.zeros(count)
    if count == 0:
        return energy
    for channel, weight in enumerate(weights):
        if weight == 0.0:
            continue
        y = k_weight(data[:, channel], rate)
        sub = np.square(y[:subs * hop]).reshape(subs, hop).sum(axis=1)
        energy += weight * ((sub[:-3] + sub[1:-2] + sub[2:-1] + sub[3:]) / block)
    return energy


def integrate(energy_blocks, rate=48000):
    """Pooled BS.1770-4 gating over block energies. Returns ``(lufs_or_None, gated_seconds, receipt)``.

    ``energy_blocks``: iterable of per-excerpt energy arrays (from ``block_energies``). Gating is applied
    once over the pooled set. ``gated_seconds`` = gated blocks x the 100 ms hop.
    """
    rate = _check_rate(rate)
    hop = int(round(HOP_SECONDS * rate))
    parts = [np.asarray(e, dtype=np.float64).reshape(-1) for e in energy_blocks]
    energy = np.concatenate(parts) if parts else np.zeros(0)
    with np.errstate(divide='ignore'):
        loudness = LOUDNESS_OFFSET + 10.0 * np.log10(energy)
    absolute = loudness > ABSOLUTE_GATE_LUFS
    relative_gate = None
    gated = np.zeros_like(absolute)
    if absolute.any():
        relative_gate = LOUDNESS_OFFSET + 10.0 * math.log10(float(energy[absolute].mean())) + RELATIVE_GATE_LU
        gated = absolute & (loudness > relative_gate)
    lufs = None
    if gated.any():
        lufs = LOUDNESS_OFFSET + 10.0 * math.log10(float(energy[gated].mean()))
    gated_seconds = int(gated.sum()) * hop / rate
    receipt = _receipt(
        function='integrate', standard=STANDARD, block_seconds=4 * hop / rate, hop_seconds=hop / rate,
        absolute_gate_lufs=ABSOLUTE_GATE_LUFS, relative_gate_lu=RELATIVE_GATE_LU,
        excerpts=[{'blocks': int(p.size)} for p in parts], blocks_total=int(energy.size),
        blocks_above_absolute_gate=int(absolute.sum()), blocks_gated=int(gated.sum()),
        relative_gate_lufs=relative_gate, integrated_lufs=lufs, gated_seconds=gated_seconds,
        gated_seconds_definition='gated 400 ms blocks x 100 ms hop')
    return lufs, gated_seconds, receipt


def measure_integrated(blocks_iter_or_arrays, rate, channel_weights=None):
    """BS.1770-4 integrated loudness pooled over one or more SOURCE excerpts.

    One array or an iterable of arrays, each (frames,) or (frames, channels). Returns
    ``(lufs_or_None, gated_seconds, receipt)``. Feed ORIGINAL source audio only, never dubbed output.
    """
    rate = _check_rate(rate)
    arrays = [blocks_iter_or_arrays] if isinstance(blocks_iter_or_arrays, np.ndarray) else list(blocks_iter_or_arrays)
    weights, channels, parts, frames = None if channel_weights is None else tuple(channel_weights), None, [], []
    for index, excerpt in enumerate(arrays):
        data = _frames(excerpt, index)
        if channels is None:
            channels = data.shape[1]
            if weights is None:
                weights = default_weights(channels)
        elif data.shape[1] != channels:
            raise ValueError(f'excerpt {index} has {data.shape[1]} channels, expected {channels}')
        frames.append(int(data.shape[0]))
        parts.append(block_energies(data, rate, weights))
    lufs, seconds, receipt = integrate(parts, rate)
    receipt.update(function='measure_integrated', rate=rate,
                   channel_weights=list(weights) if weights is not None else None,
                   k_weighting=[{'b': list(b), 'a': list(a)} for b, a in k_weighting_coefficients(rate)],
                   excerpts=[{'frames': f, 'blocks': int(p.size)} for f, p in zip(frames, parts)],
                   measured_on='caller-supplied excerpts; must be original source audio, never dubbed output')
    return lufs, seconds, receipt


# --- sampling plan ---------------------------------------------------------------------------

def sample_plan(duration_seconds, count=16, excerpt_seconds=20.0, margin_fraction=0.02):
    """Deterministic film-wide excerpt plan. Returns ``(excerpts, receipt)``.

    ``excerpts`` is a list of ``[start, end]`` seconds (millisecond rounded), evenly spaced inside
    ``[margin, duration - margin]``, deduplicated and clipped. If the excerpts would cover or overlap
    the whole window, the window itself is the single excerpt (no audio is counted twice).
    """
    duration = _finite_float(duration_seconds, 'duration_seconds')
    excerpt = _finite_float(excerpt_seconds, 'excerpt_seconds')
    margin = _finite_float(margin_fraction, 'margin_fraction')
    if isinstance(count, bool) or int(count) != count or int(count) < 1:
        raise ValueError('count must be a positive integer')
    count = int(count)
    if excerpt <= 0 or not 0 <= margin < 0.5:
        raise ValueError('excerpt_seconds must be > 0 and margin_fraction in [0, 0.5)')
    lo, hi = round(duration * margin, 3), round(duration * (1.0 - margin), 3)
    excerpts, mode = [], 'empty'
    if duration > 0 and hi > lo:
        usable = hi - lo
        if count * excerpt >= usable:
            excerpts, mode = [[lo, hi]], 'whole-window'
        else:
            spacing = (usable - excerpt) / (count - 1) if count > 1 else 0.0
            first = lo if count > 1 else lo + (usable - excerpt) / 2.0
            seen = set()
            for i in range(count):
                start = min(max(round(first + i * spacing, 3), lo), hi)
                end = min(max(round(start + excerpt, 3), lo), hi)
                if end > start and (start, end) not in seen:
                    seen.add((start, end))
                    excerpts.append([start, end])
            mode = 'evenly-spaced'
    receipt = _receipt(
        function='sample_plan', duration_seconds=duration, count=count,
        excerpt_seconds=excerpt, margin_fraction=margin, window=[lo, hi], mode=mode,
        excerpts=[list(e) for e in excerpts],
        total_seconds=round(sum(e - s for s, e in excerpts), 3))
    return excerpts, receipt


def visit_order(count):
    """Bit-reversal visiting order of ``range(count)`` (0, 8, 4, 12, 2, ... for 16).

    Decoding excerpts in this order means a measurement stopped early by its time budget still covers
    the whole film evenly instead of only its first part.
    """
    count = int(count)
    if count <= 1:
        return list(range(max(count, 0)))
    bits = (count - 1).bit_length()
    order = []
    for i in range(1 << bits):
        reversed_bits = int(format(i, '0%db' % bits)[::-1], 2)
        if reversed_bits < count:
            order.append(reversed_bits)
    return order


# --- gain ------------------------------------------------------------------------------------

def choose_job_gain(integrated_lufs, gated_seconds, *, target_lufs=DEFAULT_TARGET_LUFS,
                    max_gain_db=DEFAULT_MAX_GAIN_DB, min_gain_db=DEFAULT_MIN_GAIN_DB,
                    min_gated_seconds=DEFAULT_MIN_GATED_SECONDS):
    """gain = clamp(target - measured). Returns ``(gain_db, receipt)``; 0 dB when the evidence is too thin."""
    target = _finite_float(target_lufs, 'target_lufs')
    upper = _finite_float(max_gain_db, 'max_gain_db')
    lower = _finite_float(min_gain_db, 'min_gain_db')
    required = _finite_float(min_gated_seconds, 'min_gated_seconds')
    if lower > upper:
        raise ValueError('min_gain_db must not exceed max_gain_db')
    measured = None if integrated_lufs is None else float(integrated_lufs)
    seconds = 0.0 if gated_seconds is None else float(gated_seconds)
    raw, clamp = None, None
    if measured is None or not math.isfinite(measured) or not math.isfinite(seconds) or seconds < required:
        gain, reason = 0.0, 'insufficient_gated_audio'
        measured = measured if measured is not None and math.isfinite(measured) else None
    else:
        raw = target - measured
        gain = min(max(raw, lower), upper)
        if raw > upper:
            clamp, reason = 'max', 'clamped_max'
        elif raw < lower:
            clamp, reason = 'min', 'clamped_min'
        else:
            reason = 'measured'
    receipt = _receipt(
        function='choose_job_gain', integrated_lufs=measured,
        gated_seconds=seconds if math.isfinite(seconds) else None,
        target_lufs=target, max_gain_db=upper, min_gain_db=lower, min_gated_seconds=required,
        raw_gain_db=raw, gain_db=gain, linear_gain=10.0 ** (gain / 20.0),
        clamped=clamp is not None, clamp=clamp, reason=reason,
        scope='one constant gain for the whole final mix of this job and source')
    return gain, receipt


# --- limiter ---------------------------------------------------------------------------------

def _limiter_levels(knee_dbfs, ceiling_dbfs):
    knee_dbfs = _finite_float(knee_dbfs, 'knee_dbfs')
    ceiling_dbfs = _finite_float(ceiling_dbfs, 'ceiling_dbfs')
    if knee_dbfs >= ceiling_dbfs:
        raise ValueError('knee_dbfs must be below ceiling_dbfs')
    return knee_dbfs, ceiling_dbfs, 10.0 ** (knee_dbfs / 20.0), 10.0 ** (ceiling_dbfs / 20.0)


def apply_gain_and_limit(pcm, gain_db, *, knee_dbfs=DEFAULT_KNEE_DBFS, ceiling_dbfs=DEFAULT_CEILING_DBFS):
    """Constant gain plus the stateless soft-knee limiter. Returns ``(pcm, receipt)``.

    float32/float64 input keeps its dtype (other inputs become float64). At or below the knee the output
    is exactly ``pcm * dtype(10 ** (gain_db / 20))``. Above it:
    ``knee + (ceiling - knee) * tanh((|y| - knee) / (ceiling - knee))`` with the sign restored, capped at
    the largest value below the ceiling. Sample-wise and stateless, so chunking never changes the output.
    This is the reference for ``mux_filter``.
    """
    gain_db = _finite_float(gain_db, 'gain_db')
    knee_dbfs, ceiling_dbfs, _, _ = _limiter_levels(knee_dbfs, ceiling_dbfs)
    x = np.asarray(pcm)
    if x.dtype not in (np.float32, np.float64):
        x = x.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('pcm contains nonfinite samples')
    kind = x.dtype.type
    gain = kind(10.0 ** (gain_db / 20.0))
    knee, ceiling = kind(10.0 ** (knee_dbfs / 20.0)), kind(10.0 ** (ceiling_dbfs / 20.0))
    y = np.array(x * gain, dtype=x.dtype, copy=True)
    magnitude = np.abs(y)
    over = magnitude > knee
    above = int(over.sum())
    max_gained = float(magnitude.max()) if magnitude.size else 0.0
    if above:
        k64, span = float(knee), float(ceiling) - float(knee)
        mapped = k64 + span * np.tanh((magnitude[over].astype(np.float64) - k64) / span)
        mapped = np.minimum(mapped.astype(x.dtype), np.nextafter(ceiling, kind(0)))
        y[over] = np.copysign(mapped, y[over])
    receipt = _receipt(
        function='apply_gain_and_limit', gain_db=gain_db, linear_gain=float(gain),
        knee_dbfs=knee_dbfs, ceiling_dbfs=ceiling_dbfs, knee_linear=float(knee),
        ceiling_linear=float(ceiling), limiter=LIMITER_ID,
        lookahead_samples=0, chunk_invariant=True, dtype=str(x.dtype),
        samples=int(y.size), samples_above_knee=above,
        max_abs_input=float(np.abs(x).max()) if x.size else 0.0,
        max_abs_after_gain=max_gained,
        max_abs_output=float(np.abs(y).max()) if y.size else 0.0)
    return y, receipt


def limiter_expression(knee_dbfs=DEFAULT_KNEE_DBFS, ceiling_dbfs=DEFAULT_CEILING_DBFS):
    """The limiter curve as an FFmpeg ``aeval`` expression (per channel, ``val(ch)``)."""
    _, _, knee, ceiling = _limiter_levels(knee_dbfs, ceiling_dbfs)
    span = ceiling - knee
    x, k, s = 'val(ch)', '%.15f' % knee, '%.15f' % span
    return 'sgn(%s)*if(lte(abs(%s),%s),abs(%s),%s+%s*tanh((abs(%s)-%s)/%s))' % (x, x, k, x, k, s, x, k, s)


def mux_filter(gain_db, *, knee_dbfs=DEFAULT_KNEE_DBFS, ceiling_dbfs=DEFAULT_CEILING_DBFS):
    """FFmpeg audio filter chain ``volume=<gain>dB,aeval=<limiter>`` for the HLS mux (no ``apad``).

    ``volume`` is one constant gain in double precision (``eval=once``); ``aeval`` is the stateless
    curve of ``apply_gain_and_limit``. Neither keeps state between segments, so a segment rendered
    alone equals the same span of a whole-film render. The caller appends ``apad``. The expression is
    quoted for FFmpeg's filtergraph syntax; pass it as one argv element (no shell).
    """
    gain_db = _finite_float(gain_db, 'gain_db')
    return "volume=%.3fdB:precision=double,aeval=exprs='%s':c=same" % (
        gain_db, limiter_expression(knee_dbfs, ceiling_dbfs))


def mux_identity(gain_db, *, knee_dbfs=DEFAULT_KNEE_DBFS, ceiling_dbfs=DEFAULT_CEILING_DBFS):
    """JSON-able element for the mux cache key: it changes when the gain, curve or filter version changes."""
    return {'loudness': VERSION, 'limiter': LIMITER_ID, 'gain_db': round(_finite_float(gain_db, 'gain_db'), 3),
            'knee_dbfs': float(knee_dbfs), 'ceiling_dbfs': float(ceiling_dbfs)}


# --- cache key -------------------------------------------------------------------------------

def persist_key(source_fingerprint, plan, target, *, max_gain_db=DEFAULT_MAX_GAIN_DB,
                min_gain_db=DEFAULT_MIN_GAIN_DB, min_gated_seconds=DEFAULT_MIN_GATED_SECONDS,
                audio_index=None, layout='stereo'):
    """Cache key for one gain per job/source. Returns ``(key, receipt)``.

    ``plan`` is the excerpt list or the receipt from ``sample_plan``. The key covers everything that
    changes the measured number: source, excerpts, target, clamps, gate minimum, audio track and the
    decode layout.
    """
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError('source_fingerprint must be a non-empty string')
    excerpts = plan.get('excerpts') if isinstance(plan, dict) else plan
    if excerpts is None:
        raise ValueError('plan has no excerpts')
    canonical_plan = [[round(float(s), 3), round(float(e), 3)] for s, e in excerpts]
    material = {
        'version': VERSION, 'standard': STANDARD, 'source_fingerprint': source_fingerprint,
        'plan': canonical_plan, 'target_lufs': _finite_float(target, 'target'),
        'max_gain_db': _finite_float(max_gain_db, 'max_gain_db'),
        'min_gain_db': _finite_float(min_gain_db, 'min_gain_db'),
        'min_gated_seconds': _finite_float(min_gated_seconds, 'min_gated_seconds'),
        'audio_index': None if audio_index is None else int(audio_index), 'layout': str(layout)}
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    key = f'{VERSION}:{digest.hexdigest()}'
    return key, _receipt(function='persist_key', key=key, material=material)
