"""PROTOTYPE, default off: one-pass htdemucs giving raw vocals AND background from explicit stereo.

Never imported by the PC server. Runs in an owned model child (first_pass_child.py), or
in model-free fixtures with an injected `infer`. Not authorized for any model run yet.

One separation input -> ONE infer call -> all stems. Outputs per window (48 kHz, 2 ch):
  processing   S(original_source) with the ffmpeg -ac 2 convention the worker already uses
               for english.wav (mono [m/sqrt2, m/sqrt2], stereo identity, 5.1 ITU without LFE).
  raw_vocals   Demucs 'vocals' stem: for ASR, actor references and pan/level measurement.
  background   sum of the non-vocal stems.
  removal      processing - background: the stem the scheduled mix removes, so
               processing - envelope*removal == background wherever envelope == 1.
raw_vocals != removal in general (stems do not sum exactly to the input); each serves
its own purpose and both are hashed. The original source WAV and its layout are kept as
separate lineage; a stereo stem is never labelled as the 6-channel source.
Cache trust lives in first_pass_key (stdlib, shared with the server parent).
Model reference quality is UNVERIFIED until a root-authorized comparison.
"""
import hashlib
import json
import math
from pathlib import Path

from .first_pass_key import (LAYOUT_CHANNELS, MODEL_RATE, PARAMS, RATE, TRANSFORMS, cache_key, canonical_scope,
                             code_hashes, key_inputs, library_versions, sha256_file)

VERSION = 'demucs-first-pass-v0'
UP, DOWN = 147, 160  # 48000 * 147/160 = 44100


class Refused(Exception):
    """Geometry or provenance cannot be established."""


def pcm_sha256(array):
    import numpy as np
    data = np.ascontiguousarray(array)
    return hashlib.sha256(str((data.dtype.str, data.shape)).encode() + data.tobytes()).hexdigest()


def processing_stereo(original, layout):
    """Explicit stereo processing input, matching the worker's ffmpeg -ac 2 (proof:
    evidence/demucs-bed-option/ffmpeg-channel-convention.json and the 0435 ITU fixture)."""
    import numpy as np
    x = np.asarray(original, dtype=np.float64)
    if x.ndim != 2 or LAYOUT_CHANNELS.get(layout) != x.shape[1]:
        raise Refused('layout %r does not match %s' % (layout, getattr(x, 'shape', None)))
    if layout == 'mono':
        return np.repeat(x * 2**-.5, 2, axis=1)
    if layout == 'stereo':
        return x.copy()
    return np.stack((x[:, 0] + 2**-.5 * (x[:, 2] + x[:, 4]), x[:, 1] + 2**-.5 * (x[:, 2] + x[:, 5])), 1) / (1 + 2**.5)


def separate_pair(stereo48, infer, sources):
    """ONE infer call on one stereo span -> (raw_vocals48, background48, metrics), exact input frames.

    infer(normalized (2, n44) float32 array) -> (len(sources), 2, n44) stems. Normalization and
    denormalization follow the accepted demucs_bed child.
    """
    import numpy as np
    from scipy.signal import resample_poly
    x = np.asarray(stereo48, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not np.isfinite(x).all():
        raise Refused('first pass needs finite explicit 48 kHz stereo processing audio')
    if 'vocals' not in sources:
        raise Refused('model sources lack vocals')
    wav = np.ascontiguousarray(resample_poly(x, UP, DOWN).T).astype(np.float32)
    reference = wav.mean(0)
    # torch.Tensor.std in the accepted child is unbiased (n-1); float32 reductions may differ from torch in the last ulp.
    mean, std = reference.mean(), (reference.std(ddof=1) if reference.size > 1 else np.float32(0)) + 1e-8
    stems = np.asarray(infer((wav - mean) / std))
    if stems.shape != (len(sources), 2, wav.shape[1]) or not np.isfinite(stems).all():
        raise Refused('infer returned stems of shape %s, need %s' % (stems.shape, (len(sources), 2, wav.shape[1])))
    stems = stems.astype(np.float64) * std + mean
    index = list(sources).index('vocals')
    vocals = stems[index].T
    background = stems[[i for i in range(len(sources)) if i != index]].sum(0).T
    back_v, back_b = resample_poly(vocals, DOWN, UP), resample_poly(background, DOWN, UP)
    if len(back_v) < len(x) or len(back_b) < len(x):
        raise Refused('resampled stems shorter than input')
    return back_v[:len(x)], back_b[:len(x)], {'infer_calls': 1, 'input_frames': len(x), 'model_frames': wav.shape[1]}


def scoped_pair(processing, scope, infer, sources, cancelled=None):
    """Run a canonical scope (first_pass_key.canonical_scope) over the window's processing audio.

    'full': one input for the whole window. 'scoped': cores grouped with their context exactly like
    subtitle_runtime.separate (no input sample processed twice); EMPTY frames -> zero calls.
    Outside the cores raw_vocals is 0 and background = processing, i.e. removal 0 (no removal),
    the current 'speech stem 0' contract; timing/mix walls remain B's responsibility.
    """
    import numpy as np
    processing = np.asarray(processing, dtype=np.float64)
    if processing.ndim != 2 or processing.shape[1] != 2:
        raise Refused('scoped first pass needs stereo processing audio')
    if scope['mode'] == 'full':
        if scope['frames'] != [[0, len(processing)]]:
            raise Refused('full scope does not cover this window')
        vocals, background, metrics = separate_pair(processing, infer, sources)
        return vocals, background, {**metrics, 'calls': 1, 'mode': 'full', 'groups': [[0, len(processing)]]}
    if scope['mode'] != 'scoped':
        raise Refused('unknown scope mode %r' % scope.get('mode'))
    vocals, background = np.zeros_like(processing), processing.copy()
    context, groups = scope['context_frames'], []
    for a, b in scope['frames']:
        left, right = max(0, a - context), min(len(processing), b + context)
        if groups and left <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], right)
            groups[-1][2].append((a, b))
        else:
            groups.append([left, right, [(a, b)]])
    for left, right, cores in groups:
        if cancelled is not None and cancelled():
            raise InterruptedError('Separation cancelled')
        part_v, part_b, _ = separate_pair(processing[left:right], infer, sources)
        for a, b in cores:
            vocals[a:b], background[a:b] = part_v[a - left:b - left], part_b[a - left:b - left]
    return vocals, background, {'calls': len(groups), 'infer_calls': len(groups), 'mode': 'scoped',
                                'groups': [[l, r] for l, r, _ in groups]}


def write_bundle(folder, *, original_source_wav, source_layout_evidence, original, origin, scope, infer, sources,
                 model_pins, threads, model_info=None, code_directory=None, cancelled=None, device='cpu'):
    """Write the producer bundle (float32 WAVs) and first-pass-bundle.json; returns the bundle dict.

    `scope` is the raw subtitle_runtime.plan scope (or None); it is canonicalized and keyed here.
    """
    import time
    import numpy as np
    import soundfile as sf
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    layout = source_layout_evidence['layout']
    original = np.asarray(original, dtype=np.float32)
    if LAYOUT_CHANNELS.get(layout) != original.shape[1]:
        raise Refused('source layout evidence does not match the original source WAV')
    canonical = canonical_scope(scope, len(original))
    inputs = key_inputs(source_sha256=sha256_file(original_source_wav), original_pcm_sha256=pcm_sha256(original),
                        source_layout=layout, frames=len(original), model_pins=model_pins, threads=threads,
                        code=code_hashes(code_directory or Path(__file__).resolve().parent), scope=canonical,
                        libraries=library_versions(), device=device)
    processing = processing_stereo(original, layout).astype(np.float32).astype(np.float64)
    started = time.perf_counter()
    vocals, background, metrics = scoped_pair(processing, canonical, infer, sources, cancelled)
    metrics['separation_seconds'] = time.perf_counter() - started
    arrays = {k: v.astype(np.float32) for k, v in (('processing', processing), ('raw_vocals', vocals),
                                                     ('background', background))}
    arrays['removal_stem'] = (arrays['processing'].astype(np.float64) - arrays['background']).astype(np.float32)
    files = {}
    for name, data in arrays.items():
        path = folder / ('first-pass-%s.wav' % name.replace('_', '-'))
        temporary = path.with_name(path.stem + '.partial.wav')
        sf.write(str(temporary), data, RATE, subtype='FLOAT')
        temporary.replace(path)
        files[name] = {'wav': str(path), 'sha256': sha256_file(path), 'channels': 2, 'layout': 'stereo',
                       'frames': len(data), 'rate': RATE, 'pcm_sha256': pcm_sha256(data)}
    residual = arrays['raw_vocals'].astype(np.float64) - arrays['removal_stem']
    bundle = {
        'version': VERSION, 'key': cache_key(inputs), 'key_inputs': inputs, 'origin': float(origin),
        'frames': len(original), 'scope_raw': scope,
        'original_source': {'wav': str(original_source_wav), 'sha256': inputs['source_sha256'],
                            'pcm_sha256': inputs['original_pcm_sha256'], 'layout': layout,
                            'channels': int(original.shape[1]), 'layout_evidence': source_layout_evidence},
        'processing': {**files['processing'], 'transform': TRANSFORMS[layout], 'derived_from': 'original_source'},
        'raw_vocals': {**files['raw_vocals'], 'use': 'ASR, actor reference, pan/level measurement'},
        'background': {**files['background'], 'use': 'bed where the removal envelope is 1'},
        'removal_stem': {**files['removal_stem'], 'definition': 'processing - background',
                         'use': 'protect_dialogue_mix speech argument; protected spans restore processing exactly'},
        'raw_vocals_minus_removal_rms': float(math.sqrt(float(np.mean(residual**2)))) if residual.size else 0.0,
        'model': {'pins': dict(model_pins), 'sources': list(sources), **inputs['params'], **(model_info or {})},
        'metrics': metrics,
        'quality': 'UNVERIFIED: Demucs vocals as ASR/reference input not compared with BandIt; root comparison required',
    }
    path = folder / 'first-pass-bundle.json'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
    return bundle


class DemucsInfer:
    """Real infer for the model child ONLY (not constructed by fixtures, not authorized to run yet)."""

    def __init__(self, model_dir, expected_pins, threads=4, device='cpu'):
        # movie-trial-v1 overlay: device 'cuda' (resident trial worker only) runs the SAME pinned htdemucs bag on the
        # GPU; the device is keyed (first_pass_key.key_inputs) so a cuda bundle is never mistaken for a cpu one.
        if device not in ('cpu', 'cuda'):
            raise Refused('unknown Demucs device %r' % (device,))
        self.model_dir, self.expected_pins, self.threads = Path(model_dir), dict(expected_pins), int(threads)
        self.device = device
        self.model, self.info = None, {}

    def load(self):
        import time
        import yaml
        if self.model is not None:
            self.info['cached'] = True
            return self
        manifest = self.model_dir / 'htdemucs.yaml'
        bag = yaml.safe_load(manifest.read_text())
        paths = [self.model_dir / (str(n) + '.safetensors') for n in bag.get('models') or []]
        pins = {p.name: sha256_file(p) for p in [manifest, *paths]}
        if len(paths) != 1 or pins != self.expected_pins:
            raise Refused('Demucs model files do not match the configured pins')
        import torch
        from demucs.apply import BagOfModels
        from demucs.hf import load_safetensors_model
        torch.set_num_threads(self.threads)
        started = time.perf_counter()
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise Refused('demucs_device cuda requested but CUDA is not available in this process')
        self.model = BagOfModels([load_safetensors_model(p) for p in paths], bag.get('weights'), bag.get('segment')).to(self.device).eval()
        if self.model.samplerate != MODEL_RATE or self.model.audio_channels != 2:
            raise Refused('unexpected htdemucs geometry')
        if self.device == 'cpu' and torch.cuda.is_initialized():
            raise Refused('CUDA initialized in a cpu Demucs child')
        if torch.get_num_threads() != self.threads:
            raise Refused('torch threads differ from the keyed value')
        self.info = {'cached': False, 'load_seconds': time.perf_counter() - started, 'torch': torch.__version__,
                     'actual_threads': torch.get_num_threads(), 'device': self.device,
                     'cuda_device_name': torch.cuda.get_device_name(0) if self.device == 'cuda' else None}
        return self

    @property
    def sources(self):
        return list(self.model.sources)

    def __call__(self, normalized):
        import torch
        from demucs.apply import apply_model
        with torch.inference_mode():
            stems = apply_model(self.model, torch.from_numpy(normalized)[None], device=self.device, shifts=PARAMS['shifts'],
                                split=PARAMS['split'], overlap=PARAMS['overlap'], progress=False, num_workers=0)[0]
        return stems.detach().to('cpu').numpy()
