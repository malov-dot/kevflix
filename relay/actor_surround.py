"""5.1 output for dubbed films: the film's own 5.1 soundtrack with the original speech removed from EVERY channel it is in and the
English voice placed where that speech was.

The dub itself is built in stereo (the first pass works on the ITU downmix of the 5.1 original, relay/demucs_bed.py ``itu_stereo``:
.414 L + .293 C + .293 SL, no LFE) and the film's own 5.1 is kept next to it (``original-layout.wav``). For a surround dub:

1. The speech inside each of the five full-range channels (FL FR FC SL SR) is estimated by projecting that channel on the speech
   estimate the stereo mix already subtracts (the removal stem): per replaced line, per frequency bin, a small ridge regression of the
   channel on the stem's left and right. Only the part of a channel that is coherent with the speech is taken, so music, effects and
   ambience in the same channel are left alone, and the estimate can never be louder than the channel itself.
2. The English voice goes back into the channels in proportion to the speech energy found in each of them, at the level that makes the
   ITU downmix of the 5.1 result carry the same voice as the stereo mix.
3. Both steps go through the same ``protect_dialogue_mix`` as the stereo mix (same envelope, same protected original spans, same
   headroom limiting), so everything outside the replaced lines is the original 5.1 bit for bit and the LFE is never touched.

The result is stored per chunk as ``actor-surround.flac`` (six channels) and published beside ``english.wav`` as
``english-surround.flac`` with a receipt that binds it to that ``english.wav`` by hash. The TV mux (relay/tv.py) uses it only when
that binding holds. The stereo mix is unchanged and stays the fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

VERSION = 'surround-mix-v2'
SIDECAR = 'actor-surround.flac'
PUBLISHED_SIDECAR = 'english-surround.flac'
RECEIPT = 'english-surround.json'
FULL_RANGE = (0, 1, 2, 4, 5)                  # 5.1 and 5.1(side) order: FL FR FC LFE BL|SL BR|SR; the LFE (3) is never touched
# ITU downmix weights of the stereo the voice level was matched in (relay/demucs_bed.py itu_stereo), per full-range channel:
# L = .414 FL + .293 FC + .293 SL ; R = .414 FR + .293 FC + .293 SR (the LFE is not in it).
_SIDE = 0.7071067811865476
_NORM = 1 + 2 * _SIDE
DOWNMIX_L = (1 / _NORM, 0.0, _SIDE / _NORM, _SIDE / _NORM, 0.0)       # FL FR FC SL SR
DOWNMIX_R = (0.0, 1 / _NORM, _SIDE / _NORM, 0.0, _SIDE / _NORM)
NPERSEG, NOVERLAP = 2048, 1536
RIDGE = 1e-3                                  # regression regularisation, relative to the stem's energy in that bin
SMOOTH_BINS = 3                               # the normal equations are averaged over +-3 neighbouring bins
GROUP_GAP_SECONDS = 0.1                       # replaced runs closer than this share one set of weights
MAX_GROUP_SECONDS = 4.0                       # a longer run is split where the replacement envelope is lowest (one speaker's position per set)
MIN_GROUP_SECONDS = 0.02
CHANNEL_GATE = 0.5                            # a channel takes part (speech removed, English voice added) only if its speech is at least 50 %
                                              # (-6 dB) of the loudest channel's; weaker coherence with the stem is its music leaking into the
                                              # stem, and a channel that has no speech keeps its audio exactly


def enabled(config):
    """actor_beta.surround_mix (default on): write the surround mix of every dubbed 5.1 chunk."""
    try:
        beta = getattr(config, 'data', config)['actor_beta']
        return bool(beta.get('surround_mix', True))
    except (KeyError, TypeError, AttributeError):
        return True


def replaced_groups(envelope, rate):
    """[(first, last_exclusive)] sample runs where the stereo mix replaced speech (envelope > 0), nearby runs joined."""
    import numpy as np
    envelope = np.asarray(envelope)
    active = np.flatnonzero(envelope > 0)
    if not active.size:
        return []
    gap = max(1, int(GROUP_GAP_SECONDS * rate))
    breaks = np.flatnonzero(np.diff(active) > gap)
    firsts = np.concatenate(([active[0]], active[breaks + 1]))
    lasts = np.concatenate((active[breaks], [active[-1]])) + 1
    longest = int(MAX_GROUP_SECONDS * rate)
    groups = []
    for first, last in zip(firsts, lasts):
        first, last = int(first), int(last)
        while last - first > longest:
            low, high = first + longest // 2, first + longest
            cut = low + int(np.argmin(envelope[low:high]))
            groups.append((first, cut))
            first = cut
        groups.append((first, last))
    return [(a, b) for a, b in groups if (b - a) >= MIN_GROUP_SECONDS * rate]


def _stft(x, rate):
    from scipy.signal import stft
    return stft(x, fs=rate, window='hann', nperseg=NPERSEG, noverlap=NOVERLAP, boundary='zeros', padded=True)[2]


def _istft(z, rate, length):
    from scipy.signal import istft
    out = istft(z, fs=rate, window='hann', nperseg=NPERSEG, noverlap=NOVERLAP, boundary=True)[1]
    out = out[:length]
    return out if len(out) == length else __import__('numpy').pad(out, (0, length - len(out)))


def _smooth(a, bins):
    """Moving average along the frequency axis (axis 0) of a (freq, ...) array."""
    import numpy as np
    if bins <= 0:
        return a
    kernel = np.ones(2 * bins + 1) / (2 * bins + 1)
    padded = np.pad(a, [(bins, bins)] + [(0, 0)] * (a.ndim - 1), mode='edge')
    out = np.zeros_like(a)
    for i, k in enumerate(kernel):
        out += k * padded[i:i + a.shape[0]]
    return out


def speech_in_channel(channel, stem, rate):
    """The part of ``channel`` (n,) that is coherent with the speech stem ``stem`` (n, 2), same length. Per frequency bin, ridge
    regression of the channel on the stem's left and right over every time frame of the segment; the estimate is capped at the
    channel's own magnitude per time-frequency cell."""
    import numpy as np
    length = len(channel)
    x = _stft(channel.astype(np.float64), rate)
    sl, sr = _stft(stem[:, 0].astype(np.float64), rate), _stft(stem[:, 1].astype(np.float64), rate)
    # Normal equations per bin: [[<sl,sl> <sl,sr>] [<sr,sl> <sr,sr>]] [a b]^T = [<sl,x> <sr,x>]^T (sums over time frames).
    ll = np.sum(np.abs(sl) ** 2, axis=1)
    rr = np.sum(np.abs(sr) ** 2, axis=1)
    lr = np.sum(np.conj(sl) * sr, axis=1)
    lx = np.sum(np.conj(sl) * x, axis=1)
    rx = np.sum(np.conj(sr) * x, axis=1)
    ll, rr, lr, lx, rx = (_smooth(v, SMOOTH_BINS) for v in (ll, rr, lr, lx, rx))
    ridge = RIDGE * (ll + rr) + 1e-18
    det = (ll + ridge) * (rr + ridge) - np.abs(lr) ** 2
    det = np.where(np.abs(det) < 1e-30, 1e-30, det)
    a = ((rr + ridge) * lx - lr * rx) / det
    b = ((ll + ridge) * rx - np.conj(lr) * lx) / det
    estimate = a[:, None] * sl + b[:, None] * sr
    magnitude, limit = np.abs(estimate), np.abs(x)
    scale = np.where(magnitude > limit, limit / np.maximum(magnitude, 1e-30), 1.0)
    return _istft(estimate * scale, rate, length)


def _downmix_matrix():
    import numpy as np
    return np.array([DOWNMIX_L, DOWNMIX_R], dtype=np.float64)                 # (2, 5) over FL FR FC SL SR


def reconcile(speech5, stem, weights):
    """Make the ITU downmix of the five speech estimates equal the stereo removal stem exactly, so the surround dub removes what the
    stereo dub removes. The difference is handed back to the channels in proportion to ``weights`` (their speech energy), through
    the weighted pseudo-inverse of the downmix: a channel without speech receives nothing."""
    import numpy as np
    d = _downmix_matrix()
    w = np.asarray(weights, dtype=np.float64)
    gram = (d * w) @ d.T                                                      # D W D^T (2, 2)
    gram = gram + np.eye(2) * (1e-3 * max(np.trace(gram), 1e-12))
    residual = stem.astype(np.float64) - speech5.astype(np.float64) @ d.T     # (n, 2)
    share = residual @ np.linalg.inv(gram)                                    # (n, 2)
    return speech5 + ((share @ d) * w).astype(np.float32)


def speech_components(source51, stem, envelope, rate):
    """(n, 6) speech estimate of the original 5.1 (zero in the LFE and outside the replaced runs).
    source51: (n, 6) float32, stem: (n, 2) float32 removal stem of the stereo mix, envelope: (n,) replacement envelope.
    Per replaced run: the speech coherent with the stem in each channel; channels whose speech is far below the loudest one's are
    treated as having none (their music and effects stay untouched); then the estimates are reconciled to the stem."""
    import numpy as np
    n = len(source51)
    out = np.zeros((n, 6), dtype=np.float32)
    margin = NPERSEG * 2
    for first, last in replaced_groups(envelope, rate):
        a, b = max(0, first - margin), min(n, last + margin)
        parts = np.stack([speech_in_channel(source51[a:b, channel], stem[a:b], rate) for channel in FULL_RANGE], axis=1)
        inside = slice(first - a, last - a)
        levels = np.array([_rms(parts[inside, k]) for k in range(5)])
        top = levels.max()
        if top <= 0:
            levels = np.array([0.0, 0.0, 1.0, 0.0, 0.0])                      # nothing coherent found: the center, where film dialogue lives
            parts[:] = 0
        else:
            levels = np.where(levels >= CHANNEL_GATE * top, levels, 0.0)
            parts = parts * (levels > 0)
        keep = np.zeros(b - a, dtype=np.float32)
        keep[inside] = 1.0
        parts = reconcile(parts * keep[:, None], stem[a:b] * keep[:, None], levels / max(levels.max(), 1e-12))
        for k, channel in enumerate(FULL_RANGE):
            out[a:b, channel] += parts[:, k]
    return out


def _rms(x):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def voice_channels(speech, generated, envelope, rate):
    """(n, 6) English voice for the surround mix: the stereo mix's voice spread over the channels in proportion to the speech
    energy found in each (per replaced run), at the level that makes the ITU downmix of the result equal the stereo voice."""
    import numpy as np
    n = len(generated)
    voice = np.zeros((n, 6), dtype=np.float32)
    mono = (generated[:, 0].astype(np.float64) + generated[:, 1].astype(np.float64)) / 2
    groups = replaced_groups(envelope, rate)
    extension = int(0.5 * rate)
    for index, (first, last) in enumerate(groups):
        # a take may start before / end after its run; the windows of neighbouring runs meet at their midpoint, so no sample of the voice
        # is added twice
        a, b = max(0, first - extension), min(n, last + extension)
        if index > 0:
            a = max(a, (groups[index - 1][1] + first) // 2)
        if index + 1 < len(groups):
            b = min(b, (last + groups[index + 1][0]) // 2)
        active = np.asarray(envelope[first:last]) > 0.5
        weights = np.array([_rms(speech[first:last, c][active]) for c in FULL_RANGE])
        if weights.max() <= 0:
            weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])              # nothing found: the center, where film dialogue lives
        weights = np.where(weights >= CHANNEL_GATE * weights.max(), weights, 0.0)
        have = np.abs(mono[a:b]) > 0
        u = _rms(mono[a:b][have])
        if u <= 0:
            continue
        q_left, q_right = _rms(generated[a:b, 0][have]) / u, _rms(generated[a:b, 1][have]) / u
        down_l, down_r = float(np.dot(DOWNMIX_L, weights)), float(np.dot(DOWNMIX_R, weights))
        denominator = down_l * down_l + down_r * down_r
        lam = (q_left * down_l + q_right * down_r) / denominator if denominator > 0 else 0.0
        for k, channel in enumerate(FULL_RANGE):
            if weights[k] > 0:
                voice[a:b, channel] += (mono[a:b] * (lam * weights[k])).astype(np.float32)
    return voice


def compose_surround(source51, stem, generated, envelope, *, origin, rate, protected_intervals, layout):
    """The 5.1 mix of one render window (same samples as the stereo mix) and its receipt.

    source51: (n, 6) float32 original 5.1; stem: (n, 2) removal stem; generated: (n, 2) English voices as placed in the stereo mix;
    envelope: (n,) the stereo mix's replacement envelope; protected_intervals: original spans kept exactly; layout: '5.1' or
    '5.1(side)'. The mix goes through the stereo mix's own protect_dialogue_mix, so outside the replaced lines it is the original
    5.1 sample for sample, protected spans stay the original and the voice only gives way where it would clip the bed.
    """
    import numpy as np
    from .original_audio_policy import protect_dialogue_mix
    source51 = np.ascontiguousarray(source51, dtype=np.float32)
    stem = np.asarray(stem, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    envelope = np.asarray(envelope, dtype=np.float32)
    if (source51.ndim != 2 or source51.shape[1] != 6 or stem.shape != (len(source51), 2) or generated.shape != stem.shape
            or envelope.shape != (len(source51),)):
        raise ValueError('Surround mix needs the 5.1 original, the removal stem, the voice and the envelope on one timeline')
    speech = speech_components(source51, stem, envelope, rate)
    voice = voice_channels(speech, generated, envelope, rate)
    mixed, receipt = protect_dialogue_mix(source51, speech, voice, envelope, origin=origin, rate=rate, channel_layout=layout,
                                          protected_intervals=protected_intervals)
    energy = {name: round(_rms(speech[:, c]), 6) for name, c in zip(('FL', 'FR', 'FC', 'SL', 'SR'), FULL_RANGE)}
    return np.ascontiguousarray(mixed, dtype=np.float32), {
        'version': VERSION, 'frames': int(len(mixed)), 'groups': len(replaced_groups(envelope, rate)),
        'speech_rms_by_channel': energy, 'voice_limited_frames': receipt['voice_limited_frames'],
        'overrange_samples': receipt['overrange_samples'], 'unchanged_frames': receipt['unchanged_frames']}


def write_sidecar(path, mixed, rate):
    """24-bit FLAC: lossless for this purpose, and the untouched stretches (most of a film) compress well."""
    import soundfile as sf
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    try:
        sf.write(str(temporary), mixed, rate, format='FLAC', subtype='PCM_24')
        os.replace(temporary, path)                 # a cancelled or superseded run never leaves half a file under the final name
    finally:
        temporary.unlink(missing_ok=True)


def publish(source, directory, surround_report, english_sha256, slice_with):
    """Publish the chunk's surround mix beside english.wav (english-surround.flac + a receipt binding it to that english.wav by hash).

    Only the file THIS render's report names (by sha256) is published: a mix left in the folder by an earlier render, a window that
    ended as original audio, or a file another run replaced never is. ``slice_with(source, target)`` cuts the published 30 s out of the
    render window (ffmpeg); a failure there only means this chunk has no surround mix. Any earlier published pair is removed first.
    Returns True when a mix was published."""
    from .media import atomic_json, file_hash
    directory = Path(directory)
    published, receipt = directory / PUBLISHED_SIDECAR, directory / RECEIPT
    for stale in (published, receipt):
        stale.unlink(missing_ok=True)
    source = Path(source)
    wanted = (surround_report or {}).get('sha256')
    try:
        if not (wanted and source.is_file() and file_hash(source) == wanted):
            return False
        slice_with(source, published)
        atomic_json(receipt, {'version': 1, 'english_sha256': english_sha256, 'surround_sha256': file_hash(published),
                              'file': PUBLISHED_SIDECAR})
        return True
    except (OSError, RuntimeError, ValueError):
        for stale in (published, receipt):
            stale.unlink(missing_ok=True)
        return False
