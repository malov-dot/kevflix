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


def actor_mix_levels(speech, voice, cue, origin, rate, *, lfe_index=None):
    """Use the same bounded source-cue RMS policy in both actor renderers.

    This is static level/pan matching, not spatial reverb reconstruction or
    proof that the separated speech contains only the intended actor.
    """
    import numpy as np
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
    gain=min(2.,target/reference)
    return pan,gain,{'policy':'source-cue-rms-v1','source_start':origin+left/rate,
                    'source_end':origin+right/rate,'source_rms_by_channel':power.tolist(),
                    'generated_rms':reference,'gain':gain,'gain_capped':target/reference>2.}


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


def replace_dialogue(original, speech, generated, envelope):
    """Preserve unscheduled samples; reduce only the added voice if peaks collide."""
    import numpy as np
    if original.shape != speech.shape or generated.shape != original.shape or len(envelope) != len(original):
        raise ValueError('Dialogue tracks must share geometry and timeline')
    bed = original - speech*envelope[:,None]
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
                    'overrange_samples':int(np.count_nonzero(np.abs(result)>1))}
