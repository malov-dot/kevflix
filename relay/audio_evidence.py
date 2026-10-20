"""Acoustic evidence of one recognized line, measured on the ORIGINAL separated vocals (numpy only, no model).

The measurements are the ones the nonverbal rules were calibrated on (relay/nonverbal.py nonverbal_audio_reason): active-frame
level, share of voiced frames, the longest steadily held pitch, the pitch range, and spectral flatness. The pitch tracker is
the YIN difference function, computed for every frame at once. `cue_features` is deterministic and never raises on odd input
(it returns {} when the excerpt is too short to measure).
"""
from __future__ import annotations

import math

VERSION = 'audio-evidence-v1'
YIN_FRAME = 1024
YIN_HOP = 320
YIN_THRESHOLD = 0.15
F0_MIN, F0_MAX = 70.0, 500.0
RATE = 16000
MIN_SECONDS = 0.15


def to_mono_16k(samples, rate):
    """float32 mono at 16 kHz from a (frames,) or (frames, channels) array: the loudest channel, resampled."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, int(np.argmax((x.astype(np.float64) ** 2).sum(0)))]
    if rate != RATE and len(x):
        from scipy.signal import resample_poly
        g = math.gcd(int(rate), RATE)
        x = resample_poly(x, RATE // g, int(rate) // g).astype(np.float32)
    return x


def yin_f0(x, sr=RATE):
    """(f0 per frame in Hz, 0 where unvoiced; seconds per frame)."""
    import numpy as np
    tau_min, tau_max = int(sr / F0_MAX), int(sr / F0_MIN)
    wl = YIN_FRAME + tau_max
    starts = np.arange(0, len(x) - YIN_FRAME - tau_max, YIN_HOP)
    if not len(starts):
        return np.zeros(0), YIN_HOP / sr
    x64 = np.asarray(x, dtype=np.float64)
    frames = np.stack([x64[s:s + wl] for s in starts])
    quiet = np.sqrt(np.mean(frames[:, :YIN_FRAME] ** 2, axis=1)) < 1e-4
    spectrum = np.fft.rfft(frames, 2 * wl, axis=1)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :tau_max + 1]
    cum = np.concatenate([np.zeros((len(starts), 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    taus = np.arange(1, tau_max + 1)
    d = np.zeros((len(starts), tau_max + 1))
    d[:, 1:] = cum[:, YIN_FRAME:YIN_FRAME + 1] + (cum[:, YIN_FRAME + taus] - cum[:, taus]) - 2 * acf[:, taus]
    running = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[None, :] / np.maximum(running, 1e-12)
    out = np.zeros(len(starts))
    below = cmnd[:, tau_min:tau_max] < YIN_THRESHOLD
    first = below.argmax(axis=1) + tau_min
    for i in range(len(starts)):
        if quiet[i] or not below[i].any():
            continue
        t = int(first[i])
        while t + 1 < tau_max and cmnd[i, t + 1] < cmnd[i, t]:
            t += 1
        out[i] = sr / t
    return out, YIN_HOP / sr


def cue_features(seg, sr=RATE):
    """{'rms_active_db', 'flatness_median', 'frames', 'voiced_fraction', 'f0_median_hz', 'pitch_range_semitones',
    'longest_held_note_s'} for a mono excerpt (the keys the rules read), or {} when it is shorter than MIN_SECONDS."""
    import numpy as np
    try:
        seg = np.asarray(seg, dtype=np.float64)
        if len(seg) < sr * MIN_SECONDS:
            return {}
        hop = sr // 100
        k = len(seg) // hop
        if k < 2:
            return {}
        rms = np.sqrt((seg[:k * hop].reshape(k, hop) ** 2).mean(1)) + 1e-9
        db = 20 * np.log10(rms)
        active = db > db.max() - 28
        spec = np.abs(np.fft.rfft(seg[:k * hop].reshape(k, hop) * np.hanning(hop), axis=1)) + 1e-12
        flat = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
        f0, step = yin_f0(seg.astype(np.float32), sr)
        voiced = f0 > 0
        result = {'rms_active_db': round(float(20 * np.log10(np.sqrt((rms[active] ** 2).mean()))), 1),
                  'flatness_median': round(float(np.median(flat[active])), 3) if active.any() else None,
                  'frames': int(len(f0)), 'voiced_fraction': round(float(voiced.mean()), 3) if len(f0) else None}
        if voiced.sum() >= 5:
            semis = 12 * np.log2(f0[voiced] / 100.0)
            result['f0_median_hz'] = round(float(np.median(f0[voiced])), 1)
            result['pitch_range_semitones'] = round(float(np.percentile(semis, 95) - np.percentile(semis, 5)), 2)
            best = run = 0
            base = None
            for value, is_voiced in zip(12 * np.log2(np.maximum(f0, 1) / 100.0), voiced):
                if not is_voiced:
                    best, run, base = max(best, run), 0, None
                    continue
                if base is None or abs(value - base) > 0.8:
                    best, run, base = max(best, run), 1, value
                else:
                    run += 1
            result['longest_held_note_s'] = round(max(best, run) * step, 2)
        return result
    except Exception:       # a measurement must never break dubbing
        return {}
