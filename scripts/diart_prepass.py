#!/usr/bin/env python3
"""Whole-film Diart pre-pass: film-global speaker turns (who speaks when, stable ids, overlap) as ONE JSON artifact.

Runs under the held WSL environment (torch + pyannote.audio 3.4 + diart 0.9.2), never in the Windows venv. Offline: no
download, no network. Models are the app's own pyannote community-1 checkpoints, loaded with the strict manual loader
(zero missing/unexpected keys; Diart's stock loader fails on these checkpoints under pyannote.audio 3.4).

  python3 scripts/diart_prepass.py --audio film-16k-mono.wav --out <source_fingerprint>.json \
      --source-fingerprint <fingerprint> [--origin 0.0] [--models DIR] [--delta-new 0.7] [--sweep-delta 0.6,0.7,0.8]

Input is a 16 kHz mono 16-bit WAV of the film's dialogue channel; ``--origin`` is its start in film seconds (0 for a
whole film). Output times are film seconds. Speaker ids D000.. are Diart's global centre indices for this one stream.
Model passes (GPU) run once; ``--sweep-delta`` only repeats the cheap online clustering on the cached model outputs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ''   # pass --models <community-1 folder as seen from WSL>; the live sidecar always receives it from the app config
EXPECTED_SHA256 = {'segmentation': '7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e',
                   'embedding': '6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929'}
RATE = 16000
VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def read_wav(path):
    import numpy as np
    with wave.open(str(path), 'rb') as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (RATE, 1, 2):
            raise SystemExit('input must be 16 kHz mono 16-bit PCM WAV')
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype('float32') / 32768.0


def manual_loader(path, torch, importlib, PowersetAdapter):
    """Strict load of a community-1 checkpoint (pyannote.audio 4.0-format file under 3.4 classes), CPU, no network."""
    allow = []
    for name in sorted(torch.serialization.get_unsafe_globals_in_checkpoint(str(path))):
        module, _, attribute = name.rpartition('.')
        allow.append(getattr(importlib.import_module(module), attribute))

    def load():
        with torch.serialization.safe_globals(allow):
            checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)
        architecture = checkpoint['pyannote.audio']['architecture']
        klass = getattr(importlib.import_module(architecture['module']), architecture['class'])
        model = klass(**(checkpoint.get('hyper_parameters') or {}))
        model.specifications = checkpoint['pyannote.audio']['specifications']
        model.setup()
        result = model.load_state_dict(checkpoint['state_dict'], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise SystemExit('checkpoint keys do not match: %s' % path)
        model.eval()
        return PowersetAdapter(model) if model.specifications.powerset else model
    return load


def build_pipeline(models, params, device):
    import importlib
    import torch
    from diart.blocks.diarization import SpeakerDiarization, SpeakerDiarizationConfig
    from diart.models import EmbeddingModel, PowersetAdapter, SegmentationModel
    paths = {'segmentation': Path(models) / 'segmentation' / 'pytorch_model.bin',
             'embedding': Path(models) / 'embedding' / 'pytorch_model.bin'}
    identities = {}
    for name, path in paths.items():
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256[name]:
            raise SystemExit('%s checkpoint sha256 differs from the pinned community-1 file' % name)
        identities[name] = {'file': 'pyannote-community-1/%s/pytorch_model.bin' % name, 'sha256': digest,
                            'bytes': path.stat().st_size}
    segmentation = SegmentationModel(manual_loader(paths['segmentation'], torch, importlib, PowersetAdapter))
    embedding = EmbeddingModel(manual_loader(paths['embedding'], torch, importlib, PowersetAdapter))
    config = SpeakerDiarizationConfig(
        segmentation=segmentation, embedding=embedding, duration=params['duration'], step=params['step'],
        latency=params['latency'], tau_active=params['tau_active'], rho_update=params['rho_update'],
        delta_new=params['delta_new'], gamma=params['gamma'], beta=params['beta'], max_speakers=params['max_speakers'],
        device=torch.device(device))
    return SpeakerDiarization(config), identities


def model_pass(pipeline, samples, params, batch_size):
    """Segmentation + overlap-aware embeddings for every 5 s window at every 0.5 s step (the only GPU work)."""
    import numpy as np
    import torch
    size, step = round(params['duration'] * RATE), round(params['step'] * RATE)
    padded = np.concatenate([samples, np.zeros(size, dtype='float32')])      # zeros flush the last real seconds
    starts = list(range(0, len(padded) - size + 1, step))
    segmentations, embeddings = [], []
    for first in range(0, len(starts), batch_size):
        batch = torch.from_numpy(np.stack([padded[s:s + size] for s in starts[first:first + batch_size]])[:, :, None])
        segmentation = pipeline.segmentation(batch)
        embedding = pipeline.embedding(batch, segmentation)
        segmentations.extend(segmentation.cpu().numpy())
        embeddings.extend(embedding.detach().cpu())
    return [s / RATE for s in starts], segmentations, embeddings


def cluster_pass(starts, segmentations, embeddings, params):
    """Diart's online clustering + delayed aggregation + binarization over cached model outputs -> Annotation."""
    from diart.blocks.aggregation import DelayedAggregation
    from diart.blocks.clustering import OnlineSpeakerClustering
    from diart.blocks.utils import Binarize
    from diart.sinks import PredictionAccumulator
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    clustering = OnlineSpeakerClustering(params['tau_active'], params['rho_update'], params['delta_new'], 'cosine',
                                         params['max_speakers'])
    aggregation = DelayedAggregation(params['step'], params['latency'], strategy='hamming', cropping_mode='loose')
    binarize = Binarize(params['tau_active'])
    accumulator = PredictionAccumulator()
    buffer = []
    for start, segmentation, embedding in zip(starts, segmentations, embeddings):
        resolution = params['duration'] / segmentation.shape[0]
        feature = SlidingWindowFeature(segmentation, SlidingWindow(start=start, duration=resolution, step=resolution))
        buffer.append(clustering(feature, embedding))
        accumulator.on_next(binarize(aggregation(buffer)))
        if len(buffer) == aggregation.num_overlapping_windows:
            buffer = buffer[1:]
    accumulator.on_completed()
    return accumulator.get_prediction()


def annotation_turns(annotation, seconds, origin):
    turns = []
    if annotation is not None:
        for segment, _, label in annotation.itertracks(yield_label=True):
            start, end = max(0.0, segment.start), min(seconds, segment.end)
            if end - start >= 0.02:
                turns.append({'start': round(origin + start, 3), 'end': round(origin + end, 3),
                              'speaker': 'D%03d' % int(str(label).replace('speaker', ''))})
    turns.sort(key=lambda row: (row['start'], row['end'], row['speaker']))
    return turns


def summary(turns):
    sys.path.insert(0, str(ROOT))
    from relay.diart_turns import split_exclusive
    aligned, overlaps = split_exclusive(turns)
    return aligned, overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--audio', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-fingerprint', required=True)
    parser.add_argument('--origin', type=float, default=0.0)
    parser.add_argument('--models', default=DEFAULT_MODELS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--step', type=float, default=0.5)
    parser.add_argument('--latency', type=float, default=5.0)
    parser.add_argument('--tau-active', type=float, default=0.5)
    parser.add_argument('--rho-update', type=float, default=0.3)
    parser.add_argument('--delta-new', type=float, default=0.7)     # 0.6 split one real speaker in two in the test excerpt
    parser.add_argument('--gamma', type=float, default=3.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--max-speakers', type=int, default=40)
    parser.add_argument('--sweep-delta', default='')
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    import torch
    torch.set_num_threads(4)
    params = {'duration': args.duration, 'step': args.step, 'latency': args.latency, 'tau_active': args.tau_active,
              'rho_update': args.rho_update, 'delta_new': args.delta_new, 'gamma': args.gamma, 'beta': args.beta,
              'max_speakers': args.max_speakers, 'metric': 'cosine', 'aggregation': 'hamming/loose',
              'tail_padding_seconds': args.duration, 'batch_size': args.batch_size}
    samples = read_wav(args.audio)
    seconds = len(samples) / RATE
    started = time.perf_counter()
    pipeline, identities = build_pipeline(args.models, params, args.device)
    timing = {'load_seconds': round(time.perf_counter() - started, 2)}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    stamp = time.perf_counter()
    starts, segmentations, embeddings = model_pass(pipeline, samples, params, args.batch_size)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    timing['model_pass_seconds'] = round(time.perf_counter() - stamp, 2)
    sweep = {}
    for value in sorted({float(x) for x in args.sweep_delta.split(',') if x.strip()} | {args.delta_new}):
        trial = {**params, 'delta_new': value}
        stamp = time.perf_counter()
        trial_turns = annotation_turns(cluster_pass(starts, segmentations, embeddings, trial), seconds, args.origin)
        aligned_, overlaps_ = summary(trial_turns)
        sweep['%.3f' % value] = {'speaker_count': len({t['speaker'] for t in trial_turns}), 'turns': len(trial_turns),
                                 'speech_seconds': round(sum(r['end'] - r['start'] for r in aligned_)
                                                         + sum(r['end'] - r['start'] for r in overlaps_), 1),
                                 'overlap_seconds': round(sum(r['end'] - r['start'] for r in overlaps_), 1),
                                 'cluster_seconds': round(time.perf_counter() - stamp, 2)}
        if value == args.delta_new:
            turns = trial_turns
    timing['cluster_seconds'] = sweep['%.3f' % args.delta_new]['cluster_seconds']
    aligned, overlaps = summary(turns)
    peak = ({'torch_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20),
             'torch_peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20)}
            if args.device == 'cuda' else {})
    import diart
    import numpy
    import pyannote.audio
    artifact = {
        'schema': 'kevflix-diart-turns', 'version': VERSION,
        'source_fingerprint': args.source_fingerprint,
        'audio_sha256': sha256_file(args.audio),
        'coverage': {'start': round(args.origin, 3), 'end': round(args.origin + seconds, 3)},
        'params': params,
        'models': {**identities, 'diart': getattr(diart, '__version__', None),
                   'pyannote.audio': pyannote.audio.__version__, 'torch': torch.__version__, 'numpy': numpy.__version__,
                   'loader': 'manual strict state_dict load (zero missing/unexpected keys), CPU load then device move'},
        'turns': turns, 'aligned': aligned, 'overlaps': overlaps,
        'speaker_count': len({t['speaker'] for t in turns}),
        'diagnostics': {'audio_seconds': round(seconds, 2), 'windows': len(starts), 'timing': timing, **peak,
                        'wall_seconds_per_audio_minute': round((timing['model_pass_seconds'] + timing['cluster_seconds'])
                                                               / max(seconds / 60.0, 1e-9), 3),
                        'delta_sweep': sweep}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + '.part')
    temporary.write_text(json.dumps(artifact, separators=(',', ':')), encoding='utf-8')
    os.replace(temporary, out)
    print(json.dumps({'out': str(out), 'speaker_count': artifact['speaker_count'], 'turns': len(turns),
                      'diagnostics': artifact['diagnostics']}, indent=1))


if __name__ == '__main__':
    main()
