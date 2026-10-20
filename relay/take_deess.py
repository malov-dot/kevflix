"""Take de-esser: tame the hot fricative bursts ("s", "sh", "f", "ch") that Vox puts into a synthesized take.

Measured on the 973 dubbed lines of the Russian test film: in a Vox take 9% of the active 50 ms frames are
high-frequency bursts (more than 70% of the frame's energy above 5 kHz) against 0% in the same lines spoken by the
actors, and the loudest burst sits 4.8 dB below the vowels (median; +2.9 dB at the 90th percentile, i.e. louder than the
vowels) against 9.7 dB below in the originals. Played loudly they sound like bursts of static. This module lowers the
high band ONLY inside frames whose high-band level rises above a ceiling set relative to the take's own voiced level.
Everything else, including every take with no hot burst, is returned unchanged (same array, applied False).

Pure numpy, deterministic, same length in and out, mono take at 48 kHz (anything else is left alone)."""
import numpy as np

VERSION = 'take-deess-v1'
RATE = 48000
FRAME = 1024                 # 21.3 ms at 48 kHz
HOP = FRAME // 4             # 5.3 ms: Hann analysis + Hann synthesis sum to a constant (1.5) at this hop
SPLIT_HZ = 5000.0            # bins above this are the "high band"
RAMP_HZ = 1500.0             # raised-cosine ramp below SPLIT_HZ (no sharp spectral edge)
ACTIVE_RANGE_DB = 30.0       # a frame is speech when within this many dB of the loudest frame
BURST_SHARE = 0.55           # share of frame energy above SPLIT_HZ from which a frame is a burst frame
CEILING_DB = -5.0            # the high band of a frame may sit at most this far (dB) from the take's voiced level (the actors' own p90 is -4)
MAX_REDUCTION_DB = 8.0
MIN_APPLIED_DB = 0.5         # a take whose strongest reduction is below this is returned untouched
MIN_VOICED_FRAMES = 3


def _frames(x):
    padded = np.concatenate([np.zeros(FRAME, dtype=np.float64), x.astype(np.float64), np.zeros(2 * FRAME, dtype=np.float64)])
    count = 1 + (len(padded) - FRAME) // HOP
    index = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]
    return padded[index], count


def deess(samples, rate=RATE, *, ceiling_db=CEILING_DB, max_reduction_db=MAX_REDUCTION_DB):
    """(samples, info). info['applied'] is False (and `samples` is the very same array) when nothing was lowered."""
    info = {'version': VERSION, 'applied': False, 'reason': None, 'ceiling_db': float(ceiling_db), 'split_hz': SPLIT_HZ}
    x = np.asarray(samples)
    if rate != RATE or x.ndim != 1 or len(x) < FRAME or not np.isfinite(x).all():
        info['reason'] = 'unsupported_geometry'
        return samples, info
    window = np.hanning(FRAME + 1)[:-1]
    frames, count = _frames(x)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / rate)
    high = freqs >= SPLIT_HZ
    total = power.sum(axis=1) + 1e-12
    high_power = power[:, high].sum(axis=1) + 1e-12
    level = 10.0 * np.log10(total)
    high_level = 10.0 * np.log10(high_power)
    active = level > level.max() - ACTIVE_RANGE_DB
    burst = active & (high_power / total > BURST_SHARE)
    voiced = active & ~burst
    if int(voiced.sum()) < MIN_VOICED_FRAMES:
        info['reason'] = 'no_voiced_reference'
        return samples, info
    reference = float(np.percentile(level[voiced], 90))
    ceiling = reference + float(ceiling_db)
    reduction = np.where(active, np.clip(high_level - ceiling, 0.0, float(max_reduction_db)), 0.0)
    info.update(voiced_reference_db=round(reference, 2), burst_frames=int(burst.sum()), active_frames=int(active.sum()))
    if float(reduction.max()) < MIN_APPLIED_DB:
        info['reason'] = 'no_hot_burst'
        return samples, info
    # smooth over time: widest of three frames, then a three-tap mean, so a reduction ramps in and out instead of stepping
    wide = np.maximum(np.maximum(reduction, np.roll(reduction, 1)), np.roll(reduction, -1))
    smooth = np.convolve(wide, np.array([0.25, 0.5, 0.25]), mode='same')
    ramp = np.clip((freqs - (SPLIT_HZ - RAMP_HZ)) / RAMP_HZ, 0.0, 1.0)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)                       # 0 below the ramp, 1 above SPLIT_HZ
    gain_high = 10.0 ** (-smooth / 20.0)                          # per frame
    gains = 1.0 - ramp[None, :] * (1.0 - gain_high[:, None])
    shaped = np.fft.irfft(spectrum * gains, n=FRAME, axis=1) * window[None, :]
    out = np.zeros(len(x) + 3 * FRAME, dtype=np.float64)
    for i in range(count):
        out[i * HOP:i * HOP + FRAME] += shaped[i]
    out = (out / 1.5)[FRAME:FRAME + len(x)]
    reduced = smooth > 0.25
    info.update(applied=True, frames_reduced=int(reduced.sum()), max_reduction_db=round(float(smooth.max()), 2),
                mean_reduction_db=round(float(smooth[reduced].mean()), 2))
    return out.astype(x.dtype if x.dtype in (np.float32, np.float64) else np.float32), info
