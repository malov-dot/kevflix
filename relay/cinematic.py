"""KevFLIX Dialogue Bed: bounded CPU inference using BandIt v2 speech estimates.

BandIt v2 is by Karn Watcharasupat and collaborators (Apache-2.0 code,
CC-BY-SA-4.0 Multi weights). The bandit-infer packaging preserves that graph.
KevFLIX omits unused decoder heads and optionally quantizes GRU/Linear layers.
It subtracts the estimated speech only where replacement is scheduled, retaining
the original samples elsewhere. Separation quality still requires real material.
"""
from __future__ import annotations
import sys
import threading
from pathlib import Path
import time

CHECKPOINT_SHA256 = 'abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc'


class CinematicSeparator:
    def __init__(self, checkpoint, code_root, *, threads=4, quantized=True,
                 window_seconds=16, hop_seconds=12, batch_channels=2):
        import torch
        if not 1 <= threads <= 8 or not 4 <= window_seconds <= 32 or not 0 < hop_seconds < window_seconds:
            raise ValueError('Invalid bounded CPU separator settings')
        sys.path.insert(0, str(Path(code_root).resolve() / 'src'))
        from bandit_infer import BanditSession
        torch.set_num_threads(threads)
        self.lock = threading.Lock()
        self.window = int(window_seconds * 48000)
        self.hop = int(hop_seconds * 48000)
        self.batch_channels = max(1, min(6, int(batch_channels)))
        self.quantized, self.threads = quantized, threads
        start = time.perf_counter()
        session = BanditSession('v2-multi', device='cpu', checkpoint_path=Path(checkpoint), checkpoint_sha256=CHECKPOINT_SHA256).load()
        model = session._runtime.model
        model.mask_estim = torch.nn.ModuleDict({'speech': model.mask_estim['speech']})
        model.stems = ['speech']
        if quantized:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8, inplace=False)
        self.model = model.eval()
        session.close()
        self.load_seconds = time.perf_counter()-start

    def separate(self, audio, *, sample_rate=48000, lfe_index=None, cancelled=lambda: False):
        import numpy as np
        import torch
        from .process import Cancelled
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if sample_rate != 48000 or samples.ndim != 2 or not len(samples) or not np.isfinite(samples).all():
            raise ValueError('Separator requires finite sample-major 48 kHz audio')
        count, channels = samples.shape
        if channels > 8 or count > 120*sample_rate:
            raise ValueError('Process a bounded movie window, not a complete soundtrack at once')
        overlap = self.window-self.hop
        ramp = np.sin(np.linspace(0, np.pi/2, overlap, endpoint=False, dtype=np.float32))**2
        weights = np.ones(self.window, dtype=np.float32)
        weights[:overlap], weights[-overlap:] = ramp, ramp[::-1]
        speech = np.zeros_like(samples)
        divisor = np.zeros(count, dtype=np.float32)
        selected = [c for c in range(channels) if c != lfe_index]
        start_time = time.perf_counter()
        with self.lock, torch.inference_mode():
            for start in range(0, count, self.hop):
                if cancelled():
                    raise Cancelled('Cinematic processing yielded')
                left, right = max(0, start), min(count, start+self.window)
                if right <= left:
                    continue
                window = samples[left:right]
                front = left-start
                model_frames = min(self.window, max(4*sample_rate, front+right-left))
                back = model_frames-front-(right-left)
                padded = np.pad(window, ((front, back),(0,0)), mode='reflect' if len(window)>1 else 'edge')
                blend = weights[front:front+right-left].copy()
                # No duplicate leading/trailing inference window is needed at
                # a bounded clip edge. Only internal joins crossfade.
                if start == 0:
                    blend[:min(overlap,len(blend))] = 1
                if right == count:
                    blend[max(0,len(blend)-overlap):] = 1
                for at in range(0, len(selected), self.batch_channels):
                    if cancelled():
                        raise Cancelled('Cinematic processing yielded')
                    group = selected[at:at+self.batch_channels]
                    tensor = torch.from_numpy(np.ascontiguousarray(padded[:,group].T))[:,None,:]
                    result = self.model({'mixture': {'audio': tensor}})['estimates']['speech']['audio'][:,0].numpy().T
                    if result.shape != (model_frames,len(group)) or not np.isfinite(result).all():
                        raise ValueError('Cinematic separator returned invalid samples')
                    speech[left:right,group] += result[front:front+right-left] * blend[:,None]
                divisor[left:right] += blend
                if right == count:
                    break
        if np.any(divisor <= 0):
            raise ValueError('Separator overlap-add left uncovered samples')
        speech /= divisor[:,None]
        return speech, {'engine':'BandIt v2 Multi / KevFLIX Dialogue Bed', 'device':'cpu',
                        'threads':self.threads, 'quantized':self.quantized, 'channels':channels,
                        'lfe_preserved':lfe_index, 'seconds':time.perf_counter()-start_time,
                        'input_seconds':count/sample_rate}


LEVEL_POLICY_V1 = 'source-cue-rms-v1'
LEVEL_POLICY_V2 = 'source-active-frame-rms-v2'
LEVEL_GAIN_CAP_V1 = 2.
LEVEL_GAIN_CAP_V2 = 8.


def actor_mix_levels(speech, voice, cue, origin, rate, *, lfe_index=None, policy=LEVEL_POLICY_V2):
    """Pan and gain that make a generated take follow the original line's loudness (both actor renderers).

    ``policy='source-active-frame-rms-v2'`` (default, approved on the owner's demo film): the source level is the RMS (dB)
    of the speech-bearing 10 ms frames of the separated source stem over the cue (frames within 28 dB of the loudest
    frame; pauses do not count), taken for the strongest non-LFE channel; the generated level is measured the same
    way on the voice take; ``gain = min(8, 10**((source_db-generated_db)/20))`` so a quiet line is dubbed quiet and
    a shouted line loud. ``policy='source-cue-rms-v1'`` keeps the previous whole-cue RMS, 2x-capped behaviour.
    Pan is the per-channel whole-cue RMS ratio of the source stem in both policies.

    This is static level/pan matching, not spatial reverb reconstruction or
    proof that the separated speech contains only the intended actor.
    """
    import numpy as np
    if policy not in (LEVEL_POLICY_V1, LEVEL_POLICY_V2):
        raise ValueError('Unknown actor level policy %r' % (policy,))
    left=max(0,round((cue['start']-origin)*rate))
    right=min(len(speech),round((cue['end']-origin)*rate))
    if right<=left or speech.ndim!=2 or voice.ndim!=1 or not len(voice):
        raise ValueError('Actor gain window or voice geometry is invalid')
    clip=speech[left:right]
    if not np.isfinite(clip).all() or not np.isfinite(voice).all():
        raise ValueError('Actor gain measurement contains nonfinite samples')
    power=np.sqrt(np.mean(np.square(clip,dtype=np.float64),axis=0))
    if lfe_index is not None:power[lfe_index]=0
    target=float(power.max())
    reference=float(np.sqrt(np.mean(np.square(voice,dtype=np.float64))))
    if target<=1e-8 or reference<=1e-8:
        raise ValueError('Actor gain cannot be established from silent source or generated speech')
    pan=power/target
    if policy==LEVEL_POLICY_V1:
        gain=min(LEVEL_GAIN_CAP_V1,target/reference)
        return pan,gain,{'policy':LEVEL_POLICY_V1,'source_start':origin+left/rate,
                        'source_end':origin+right/rate,'source_rms_by_channel':power.tolist(),
                        'generated_rms':reference,'gain':gain,'gain_capped':target/reference>LEVEL_GAIN_CAP_V1}
    from .speech_activity import active_rms_db,SILENCE_DB
    channels=[c for c in range(clip.shape[1]) if c!=lfe_index]
    source_db=max(active_rms_db(clip[:,c],rate) for c in channels)
    generated_db=active_rms_db(voice,rate)
    if source_db<=SILENCE_DB or generated_db<=SILENCE_DB:
        raise ValueError('Actor gain cannot be established from silent source or generated speech')
    ratio=10**((source_db-generated_db)/20)
    gain=float(min(LEVEL_GAIN_CAP_V2,ratio))
    return pan,gain,{'policy':LEVEL_POLICY_V2,'source_start':origin+left/rate,
                    'source_end':origin+right/rate,'source_rms_by_channel':power.tolist(),
                    'generated_rms':reference,'source_db':float(source_db),'generated_db':float(generated_db),
                    'gain':gain,'gain_capped':bool(ratio>LEVEL_GAIN_CAP_V2),'gain_cap':LEVEL_GAIN_CAP_V2,
                    'uncapped_gain':float(ratio)}


def activity_removal_intervals(speech_stem, cue, origin, rate, lfe_index=None):
    """Absolute (start_s, end_s) spans where the separated source stem is actually speaking inside ``cue``.

    Remove the original dialogue only inside these spans (pauses, footsteps and whispers between the words stay
    bit-exact original). The speech stem is mono or sample-major ``(n, channels)`` starting at ``origin``; the level is
    the strongest non-LFE channel per 10 ms frame, the same channels ``actor_mix_levels`` measures. Hand the result to
    ``replacement_envelope(..., fade_seconds=ACTIVITY_FADE_SECONDS)`` or ``activity_replacement_envelope``.
    Silent, empty or too-short stems return []. See ``speech_activity`` for the method and constants.

    Do NOT hand this a PADDED replacement span: its reference is the loudest frame of the window, so a whisper next to a
    shout (or the quiet half of a cue that turns loud) is not found. The mix uses ``speech_only_spans`` /
    ``speech_only_removal`` on the UNPADDED source spans instead; this function stays for whole-cue analysis.
    """
    from .speech_activity import activity_intervals
    return activity_intervals(speech_stem, rate, cue['start'], cue['end'], origin, lfe_index=lfe_index)


def merge_intervals(intervals, join_gap=.002):
    """Sorted non-overlapping intervals; touching and near-touching (< join_gap s apart) ones are joined.

    Merge before ``replacement_envelope``: two touching intervals notch to zero at the shared sample, because each
    fades out at its own edge. See ``speech_activity.merge_intervals``."""
    from .speech_activity import merge_intervals as merge
    return merge(intervals, join_gap)


def speech_only_spans(stem, cue_spans, origin, rate, lfe_index=None, *, pad_before=.12, pad_after=.25):
    """Absolute ``(start_s, end_s)`` spans where the separated stem is speaking, for the UNPADDED source spans of one cue/take.

    ``cue_spans`` are the source speech spans (absolute seconds, NOT padded: a cue's ``start``/``end`` or its
    ``authorized_audio_spans``). Per span, activity is searched inside ``[start - pad_before, end + pad_after]`` clamped to
    the stem, but the level reference comes only from the frames inside the unpadded span (90th percentile of the
    above-noise 10 ms frame levels), so a neighbouring loud cue never decides what counts as speech here. A frame is
    speech above ``max(-65, min(reference_dB - 28, -55))`` dBFS; gaps <= 40 ms are bridged, the mask is dilated 20 ms and
    clamped to the padded span (``speech_activity.span_activity_intervals`` has the full definition and the reasons).
    The strongest non-LFE channel per frame is used; spans partly outside the stem are clamped.

    Returns the sorted, merged intervals of all spans (``speech_activity.merge_intervals``): hand them to
    ``replacement_envelope(..., fade_seconds=ACTIVITY_FADE_SECONDS)``. A span with nothing to report (silent or
    unusable stem, nothing above the noise floor) contributes NOTHING: the CALLER must then keep that span's padded
    interval ``[start - pad_before, end + pad_after]``. ``speech_only_removal`` does exactly that per span; use it unless
    you need the raw analysis. Non-finite samples or invalid geometry raise ValueError.
    """
    from .speech_activity import merge_intervals, span_activity_intervals
    found = []
    for begin, end in cue_spans:
        found.extend(span_activity_intervals(stem, rate, begin, end, origin, pad_before=pad_before,
                                             pad_after=pad_after, lfe_index=lfe_index))
    return merge_intervals(found)


def speech_only_removal(stem, cue_spans, origin, rate, lfe_index=None, *, pad_before=.12, pad_after=.25):
    """``speech_only_spans`` with the fail-safe applied span by span: the removal intervals the mix should use.

    Every span whose analysis reports nothing (silent, noise-only or too-short stem, nothing above the noise floor,
    or a ValueError from an unusable stem) keeps its full padded interval ``[start - pad_before, end + pad_after]``, so
    original speech is never left under a dub for want of evidence. (A stem with nothing above -65 dBFS there holds at
    most noise, so removing the padded span subtracts almost nothing.) Spans with non-finite bounds raise ValueError.
    Result: sorted, merged, ready for ``replacement_envelope(..., fade_seconds=.01)``.
    """
    from .speech_activity import merge_intervals, span_activity_intervals
    import math
    found = []
    for begin, end in cue_spans:
        begin, end = float(begin), float(end)
        if not (math.isfinite(begin) and math.isfinite(end)):
            raise ValueError('Cue span must be finite')
        try:
            part = span_activity_intervals(stem, rate, begin, end, origin, pad_before=pad_before, pad_after=pad_after,
                                           lfe_index=lfe_index)
        except ValueError:
            part = []
        found.extend(part if part else [(begin - pad_before, end + pad_after)])
    return merge_intervals(found)


def take_removal_intervals(stem, record, origin, rate, lfe_index=None):
    """Speech-only removal of one cue, line or canonical ledger take, chosen from what the record knows.

    The spans analysed are, in order: ``authorized_audio_spans`` (target takes: counted source speech only); else
    ``source_start``/``source_end`` (a ledger take or placement: the source cue); else ``replacement_start``/
    ``replacement_end`` (an old take that recorded only its padded removal span: used as is, with no extra padding);
    else ``start``/``end`` of a cue. The same record gives the same intervals whether the owner window mixes it or a later
    window carries it, because both call this function with the same stem. See ``speech_only_removal``.
    """
    if record.get('authorized_audio_spans'):
        return speech_only_removal(stem, [(a, b) for a, b in record['authorized_audio_spans']], origin, rate, lfe_index)
    if record.get('source_start') is not None and record.get('source_end') is not None:
        return speech_only_removal(stem, [(record['source_start'], record['source_end'])], origin, rate, lfe_index)
    if record.get('replacement_start') is not None and record.get('replacement_end') is not None:
        return speech_only_removal(stem, [(record['replacement_start'], record['replacement_end'])], origin, rate,
                                   lfe_index, pad_before=0., pad_after=0.)
    if record.get('start') is not None and record.get('end') is not None:
        return speech_only_removal(stem, [(record['start'], record['end'])], origin, rate, lfe_index)
    raise ValueError('The take records no source or replacement span')


def activity_replacement_envelope(frames, rate, origin, intervals):
    """``replacement_envelope`` with the 10 ms fades used for activity intervals."""
    from .speech_activity import ACTIVITY_FADE_SECONDS
    return replacement_envelope(frames, rate, origin, intervals, fade_seconds=ACTIVITY_FADE_SECONDS)


def replacement_envelope(frames, rate, origin, intervals, fade_seconds=.04):
    import numpy as np
    envelope = np.zeros(frames, dtype=np.float32)
    for begin, end in intervals:
        absolute_left=round(begin*rate)-round(origin*rate)
        absolute_right=round(end*rate)-round(origin*rate)
        left,right=max(0,absolute_left),min(frames,absolute_right)
        if right <= left:
            continue
        current = np.ones(right-left, dtype=np.float32)
        fade = min(round(fade_seconds*rate), (absolute_right-absolute_left)//2)
        if fade:
            ramp = np.sin(np.linspace(0,np.pi/2,fade,endpoint=False,dtype=np.float32))**2
            positions=np.arange(left,right)
            rising=positions-absolute_left
            falling=absolute_right-1-positions
            rising_mask=rising<fade;falling_mask=falling<fade
            current[rising_mask]=ramp[rising[rising_mask]]
            current[falling_mask]=np.minimum(current[falling_mask],ramp[falling[falling_mask]])
        envelope[left:right] = np.maximum(envelope[left:right], current)
    return envelope


ROOM_TONE_FRAME_SECONDS = .02
ROOM_TONE_CONTEXT_SECONDS = .8        # the room tone is measured this far before and after a replaced run
ROOM_TONE_MIN_RUN_SECONDS = .04
ROOM_TONE_MARGIN_DB = -3.0            # the fill reaches 3 dB under the surrounding room tone
ROOM_TONE_TRIGGER = .5                # ... and only where the bed falls under half (-6 dB) of that level


def _room_tone_noise(quiet_blocks, frame, length, seed):
    """Unit-RMS noise with the average spectrum of the quiet blocks (comfort noise), reproducible from ``seed``."""
    import numpy as np
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    window = np.hanning(frame)
    power = np.mean(np.abs(np.fft.rfft(quiet_blocks * window[None, :], axis=1)) ** 2, axis=0)
    amplitude = np.sqrt(power)
    white = np.fft.rfft(rng.standard_normal(length))
    shape = np.interp(np.linspace(0, 1, len(white)), np.linspace(0, 1, len(amplitude)), amplitude)
    noise = np.fft.irfft(white * shape, n=length)
    level = float(np.sqrt(np.mean(noise ** 2)))
    return noise / level if level > 0 else noise


def fill_room_tone(bed, original, envelope, rate):
    """(bed, runs_filled). Removing the original speech can take the whole sound with it (a loud line, a quiet scene): the bed inside a
    replaced run falls to digital silence while the surroundings have a room tone, and the listener hears the sound drop out completely
    for the length of the original line. Where the bed of a run falls clearly under the room tone measured in the 0.8 s before and after
    it (outside every replaced run), noise with that room tone's spectrum is added, only as much as is missing. Nothing outside a
    replaced run changes, and a bed that already holds its room tone is returned unchanged."""
    import numpy as np
    out = np.array(bed, copy=True)
    frames, channels = out.shape
    active = np.asarray(envelope) > 0
    index = np.flatnonzero(active)
    if not index.size:
        return out, 0
    frame = max(1, int(ROOM_TONE_FRAME_SECONDS * rate))
    context = int(ROOM_TONE_CONTEXT_SECONDS * rate)
    breaks = np.flatnonzero(np.diff(index) > int(.05 * rate))
    firsts = np.concatenate(([index[0]], index[breaks + 1]))
    lasts = np.concatenate((index[breaks], [index[-1]])) + 1
    filled = 0
    for first, last in zip(firsts, lasts):
        first, last = int(first), int(last)
        length = last - first
        if length < ROOM_TONE_MIN_RUN_SECONDS * rate:
            continue
        low, high = max(0, first - context), min(frames, last + context)
        touched = False
        for channel in range(channels):
            sides = [original[a:b, channel][~active[a:b]] for a, b in ((low, first), (last, high)) if (~active[a:b]).any()]
            if not sides:
                continue
            reference = np.concatenate(sides).astype(np.float64)
            count = len(reference) // frame
            if count < 3:
                continue
            blocks = reference[:count * frame].reshape(count, frame)
            rms = np.sqrt(np.mean(blocks ** 2, axis=1))
            floor = float(np.percentile(rms, 25))
            if floor < 1e-5:                                   # the surroundings are digital silence as well: nothing to keep
                continue
            target = floor * 10 ** (ROOM_TONE_MARGIN_DB / 20)
            segment = out[first:last, channel].astype(np.float64)
            n = len(segment) // frame
            if n < 1:
                continue
            level = np.sqrt(np.mean(segment[:n * frame].reshape(n, frame) ** 2, axis=1))
            deficit = np.sqrt(np.maximum(0.0, target ** 2 - level ** 2))
            if deficit.max() < ROOM_TONE_TRIGGER * target:
                continue
            deficit = np.convolve(np.pad(deficit, 2, mode='edge'), np.ones(5) / 5, mode='valid')
            gain = np.interp(np.arange(length), (np.arange(n) + .5) * frame, deficit)
            quiet = blocks[rms <= np.percentile(rms, 50)]
            noise = _room_tone_noise(quiet if len(quiet) >= 2 else blocks, frame, length, seed=first * 8 + channel)
            out[first:last, channel] = (segment + noise * gain).astype(out.dtype)
            touched = True
        filled += int(touched)
    return out, filled


def replace_dialogue(original, speech, generated, envelope, rate=None):
    """Preserve unscheduled samples; reduce only the added voice if peaks collide. With ``rate`` the room tone is kept where the
    removal of the original speech would leave the bed silent (see fill_room_tone)."""
    import numpy as np
    if original.shape != speech.shape or generated.shape != original.shape or len(envelope) != len(original):
        raise ValueError('Dialogue tracks must share geometry and timeline')
    bed = original - speech*envelope[:,None]
    room_tone_runs = 0
    if rate:
        bed, room_tone_runs = fill_room_tone(bed, original, envelope, rate)
    # Headroom limiting never turns down the soundtrack to accommodate a dub.
    headroom = np.maximum(0, .999-np.abs(bed))
    gain = np.minimum(1, headroom/np.maximum(1e-8,np.abs(generated))).min(axis=1)
    result = bed + generated*gain[:,None]
    untouched = (envelope == 0) & ~np.any(generated != 0,axis=1)
    result[untouched] = original[untouched]
    if not np.isfinite(result).all():
        raise ValueError('Dialogue mix is not finite')
    return result, {'unchanged_frames':int(untouched.sum()),
                    'original_preserved_outside_replacements':bool(np.array_equal(result[untouched],original[untouched])),
                    'voice_limited_frames':int(np.count_nonzero(gain<.999)),
                    'background_peak':float(np.max(np.abs(bed))) if bed.size else 0,
                    'overrange_samples':int(np.count_nonzero(np.abs(result)>1)),
                    'room_tone_runs':room_tone_runs}
