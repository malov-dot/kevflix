"""5.1 output for dubbed films: the film's own 5.1 soundtrack with the original speech removed from EVERY channel it is in and the
English voice placed where that speech was.

The dub itself is built in stereo (the first pass works on the ITU downmix of the 5.1 original, relay/demucs_bed.py ``itu_stereo``:
.414 L + .293 C + .293 SL, no LFE) and the film's own 5.1 is kept next to it (``original-layout.wav``). For a surround dub:

1. The speech inside each of the five full-range channels (FL FR FC SL SR) is estimated with the SAME vocal mask the stereo mix uses:
   the removal stem relative to the ITU downmix, per time-frequency cell. Applying it to every channel removes the original-language
   speech from the front, the center and the rear channels alike (a regression on the stem missed it in FL, FR and both rears, where
   the dialogue's reverb and echo live, and the Russian stayed audible).
2. The English voice goes into the CENTER channel only (film dialogue lives there; copies in FL/FR/surrounds sounded like an echo), at the
   level that makes the ITU downmix of the 5.1 result carry the same voice as the stereo mix.
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
GROUP_GAP_SECONDS = 0.1                       # replaced runs closer than this share one set of weights
MAX_GROUP_SECONDS = 4.0                       # a longer run is split where the replacement envelope is lowest (one speaker's position per set)
MIN_GROUP_SECONDS = 0.02


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


def processing_stereo(source51):
    """The ITU stereo downmix the first pass works on (no LFE): L = .414 FL + .293 FC + .293 SL, R = .414 FR + .293 FC + .293 SR. The removal
    stem of the stereo mix was made from exactly this signal."""
    import numpy as np
    x = np.asarray(source51, dtype=np.float64)[:, list(FULL_RANGE)]
    return np.stack([x @ np.asarray(DOWNMIX_L), x @ np.asarray(DOWNMIX_R)], axis=1)


def vocal_mask(processing, stem, rate):
    """(freq, frames) soft mask in [0, 1]: how much of each time-frequency cell of the stereo downmix is the speech the stereo mix removes
    (the removal stem relative to the downmix). The stereo dub is clean with this stem; the same mask is applied to every channel of the
    5.1, so the original-language speech goes from the front, the center AND the rear channels (its reverb and echo included), not only from
    the channels where a regression happened to find it. (Measured on a real dubbed chunk: before, speech recognition heard the Russian
    prayer clearly in FL, FR, BL and BR of the published mix; with the mask it hears nothing in any channel.)"""
    import numpy as np
    xl, xr = _stft(processing[:, 0], rate), _stft(processing[:, 1], rate)
    sl, sr = _stft(stem[:, 0].astype(np.float64), rate), _stft(stem[:, 1].astype(np.float64), rate)
    return np.clip((np.abs(sl) + np.abs(sr)) / (np.abs(xl) + np.abs(xr) + 1e-9), 0.0, 1.0)


def speech_in_channel(channel, mask, rate):
    """The original speech in one channel (n,): the channel's own spectrum times the vocal mask."""
    import numpy as np
    return _istft(_stft(np.asarray(channel, dtype=np.float64), rate) * mask, rate, len(channel))


def speech_components(source51, stem, envelope, rate):
    """(n, 6) speech estimate of the original 5.1 (zero in the LFE and outside the replaced runs).
    source51: (n, 6) float32, stem: (n, 2) float32 removal stem of the stereo mix, envelope: (n,) replacement envelope.
    Per replaced run, every full-range channel gets the same vocal mask (see vocal_mask); what the mask does not touch (music, effects,
    ambience that is not speech-like) stays, and the LFE is never touched."""
    import numpy as np
    n = len(source51)
    out = np.zeros((n, 6), dtype=np.float32)
    margin = NPERSEG * 2
    for first, last in replaced_groups(envelope, rate):
        a, b = max(0, first - margin), min(n, last + margin)
        mask = vocal_mask(processing_stereo(source51[a:b]), stem[a:b], rate)
        keep = np.zeros(b - a, dtype=np.float32)
        keep[first - a:last - a] = 1.0
        for channel in FULL_RANGE:
            out[a:b, channel] += (speech_in_channel(source51[a:b, channel], mask, rate) * keep).astype(np.float32)
    return out


def _rms(x):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def voice_channels(speech, generated, envelope, rate):
    """(n, 6) English voice for the surround mix: the stereo mix's voice in the CENTER channel only, at the level that makes the ITU
    downmix of the result carry the same voice as the stereo mix. (Not spread over FL/FR/surrounds: that played the same voice from
    several speakers at once and sounded like a heavy echo.)"""
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
        # The English voice goes to the CENTER channel only, where film dialogue lives. It used to be copied into every channel that held
        # at least half as much speech as the loudest one (FL, FR, the surrounds), so one dry voice played from several speakers at the
        # same instant; on a soundbar or receiver whose channels do not arrive together that is heard as a strong echo. `speech` still
        # decides which channels have the ORIGINAL speech removed (speech_components), not where the English voice is put.
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
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
