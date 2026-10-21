#!/usr/bin/env python3
"""Streaming Diart sidecar: who speaks when, live, one persistent clustering state per stream.

Runs under the held WSL environment (torch + pyannote.audio 3.4 + diart 0.9.2), never in the Windows venv, started on
demand by relay/diart_stream.py. Offline: no download, no network. It loads the app's own pyannote community-1
checkpoints once (the strict manual loader of scripts/diart_prepass.py: zero missing/unexpected keys, pinned sha256) and
then serves framed requests on stdin/stdout (format in relay/diart_stream.py). Only frames ever reach stdout; all logging
goes to stderr. It exits when stdin closes, so it never outlives the process that started it.

Per stream the model passes run once per 0.5 s step over 5 s windows (causal: frames in order), the online clustering
and the delayed aggregation advance exactly as in the whole-file pre-pass, and the answer for the fed audio is
  * committed turns   - final, up to (fed - latency), and
  * a provisional tail - computed on a COPY of the clustering state with the missing windows zero-padded, replaced by the
                         next feed (the real state is never touched by the tail).
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RATE = 16000
MAX_STREAMS = 4


class Stream:
    """The state of one source: audio not yet consumed by a complete window, the clustering state, the aggregation buffer."""

    def __init__(self, params, origin):
        import numpy as np
        from diart.blocks.clustering import OnlineSpeakerClustering
        self.clustering = OnlineSpeakerClustering(params['tau_active'], params['rho_update'], params['delta_new'],
                                                  params['metric'], params['max_speakers'])
        self.buffer = []
        self.audio = np.zeros(0, dtype=np.float32)
        self.base = 0                 # stream sample index of audio[0]
        self.fed = 0                  # samples received
        self.k = 0                    # next window index (window k starts at k * step)
        self.origin = origin          # absolute seconds of the first sample
        self.returned = 0.0           # stream seconds already returned as committed


class Engine:
    def __init__(self, models, device, params):
        import torch
        from diart.blocks.aggregation import DelayedAggregation
        from diart.blocks.utils import Binarize
        from diart_prepass import build_pipeline
        self.torch, self.params, self.device = torch, params, device
        self.pipeline, self.identities = build_pipeline(models, params, device)
        self.aggregation = DelayedAggregation(params['step'], params['latency'], strategy='hamming', cropping_mode='loose')
        self.binarize = Binarize(params['tau_active'])
        self.step_n = round(params['step'] * RATE)
        self.size_n = round(params['duration'] * RATE)

    def memory(self):
        if self.device != 'cuda':
            return {}
        torch = self.torch
        return {'torch_allocated_mib': round(torch.cuda.memory_allocated() / 2**20),
                'torch_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20),
                'torch_reserved_mib': round(torch.cuda.memory_reserved() / 2**20)}

    def model_pass(self, windows):
        """Segmentation + overlap-aware embeddings for every row of ``windows`` (n, samples): the only GPU work."""
        torch = self.torch
        segmentations, embeddings = [], []
        batch_size = self.params['batch_size']
        for first in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[first:first + batch_size][:, :, None])
            segmentation = self.pipeline.segmentation(batch)
            embedding = self.pipeline.embedding(batch, segmentation)
            segmentations.extend(segmentation.cpu().numpy())
            embeddings.extend(embedding.detach().cpu())
        return segmentations, embeddings

    def step(self, clustering, buffer, k, segmentation, embedding):
        """One 0.5 s step of the online pipeline (same calls as scripts/diart_prepass.py cluster_pass) -> chunk Annotation."""
        from pyannote.core import SlidingWindow, SlidingWindowFeature
        resolution = self.params['duration'] / segmentation.shape[0]
        feature = SlidingWindowFeature(segmentation, SlidingWindow(start=k * self.step_n / RATE, duration=resolution,
                                                                   step=resolution))
        buffer.append(clustering(feature, embedding))
        annotation = self.binarize(self.aggregation(buffer))
        if len(buffer) == self.aggregation.num_overlapping_windows:
            del buffer[0]
        return annotation

    def rows(self, chunks, lower, upper, origin):
        from pyannote.core import Annotation
        merged = Annotation()
        for chunk in chunks:
            merged.update(chunk)
        merged = merged.support(0.05)
        rows = []
        for segment, _, label in merged.itertracks(yield_label=True):
            a, b = max(lower, segment.start), min(upper, segment.end)
            if b - a >= 0.02:
                rows.append({'start': round(origin + a, 3), 'end': round(origin + b, 3),
                             'speaker': 'D%03d' % int(str(label).replace('speaker', ''))})
        rows.sort(key=lambda row: (row['start'], row['end'], row['speaker']))
        return rows

    def feed(self, st, pcm):
        import numpy as np
        started = time.perf_counter()
        step_n, size_n = self.step_n, self.size_n
        st.audio = np.concatenate([st.audio, pcm])
        st.fed += len(pcm)
        last = (st.fed - size_n) // step_n if st.fed >= size_n else -1
        committed_chunks = []
        ks = list(range(st.k, last + 1))
        if ks:
            windows = np.stack([st.audio[k * step_n - st.base:k * step_n - st.base + size_n] for k in ks])
            segmentations, embeddings = self.model_pass(windows)
            for k, segmentation, embedding in zip(ks, segmentations, embeddings):
                committed_chunks.append(self.step(st.clustering, st.buffer, k, segmentation, embedding))
            st.k = last + 1
        committed = st.k * step_n / RATE if st.k else 0.0
        # Provisional tail: the windows that still miss audio, zero padded, on a COPY of the state.
        tail_chunks, flush = [], 0
        fks = list(range(st.k, -(-st.fed // step_n)))
        if fks:
            padded = np.zeros((len(fks), size_n), dtype=np.float32)
            for row, k in enumerate(fks):
                piece = st.audio[k * step_n - st.base:k * step_n - st.base + size_n]
                padded[row, :len(piece)] = piece
            clustering, buffer = copy.deepcopy(st.clustering), list(st.buffer)
            segmentations, embeddings = self.model_pass(padded)
            for k, segmentation, embedding in zip(fks, segmentations, embeddings):
                tail_chunks.append(self.step(clustering, buffer, k, segmentation, embedding))
            flush = len(fks)
        rows = self.rows(committed_chunks + tail_chunks, st.returned, st.fed / RATE, st.origin)
        reply = {'from': round(st.origin + st.returned, 3), 'committed_until': round(st.origin + committed, 3),
                 'fed_until': round(st.origin + st.fed / RATE, 3), 'turns': rows,
                 'timing': {'seconds': round(time.perf_counter() - started, 3), 'windows': len(ks), 'flush_windows': flush}}
        st.returned = committed
        drop = st.k * step_n - st.base
        if drop > 0:
            st.audio, st.base = st.audio[drop:], st.base + drop
        return reply


def handle(engine, streams, header, payload):
    import numpy as np
    op, name = header.get('op'), str(header.get('stream') or '')
    if op == 'ping':
        return {'pong': True, 'streams': list(streams), **engine.memory()}
    if op in ('open', 'reset', 'close'):
        streams.pop(name, None)                       # the next feed (create=true) starts the stream at its own time
        return {'stream': name}
    if op != 'feed':
        raise ValueError('unknown op %r' % (op,))
    start, count = float(header['start']), int(header['n'])
    pcm = np.frombuffer(payload, dtype='<f4')
    if len(pcm) != count or not np.isfinite(pcm).all():
        raise ValueError('pcm payload does not match n or is not finite')
    stream = streams.get(name)
    if stream is None:
        if not header.get('create'):
            raise ValueError('unknown_stream')
        while len(streams) >= MAX_STREAMS:
            streams.popitem(last=False)
        stream = streams[name] = Stream(engine.params, start)
    else:
        streams.move_to_end(name)
        expected = stream.origin + stream.fed / RATE
        if abs(start - expected) > 0.05:
            raise ValueError('discontinuity: expected %.3f got %.3f' % (expected, start))
    try:
        reply = engine.feed(stream, pcm.astype(np.float32))
    except BaseException:
        streams.pop(name, None)                       # a half-advanced stream is never kept
        raise
    reply['stream'] = name
    return reply


def main():
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                           # nothing but frames may reach the real stdout
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    from relay import diart_stream as protocol
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--models', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--params', default='')
    args = parser.parse_args()
    params = {**protocol.PARAMS, **(json.loads(args.params) if args.params else {})}
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        engine = Engine(args.models, args.device, params)
        warm = (np.random.RandomState(0).standard_normal((1, engine.size_n)) * 0.05).astype(np.float32)
        engine.model_pass(warm)                       # first-call kernel cost is paid here, not by the first window
        if args.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except BaseException as error:                    # noqa: BLE001 - reported as one frame, then the process ends
        protocol.write_frame(out, {'event': 'fatal', 'error': '%s: %s' % (type(error).__name__, error)})
        sys.stderr.write('fatal: %r\n' % (error,))
        return 3
    protocol.write_frame(out, {'event': 'ready', 'load_seconds': round(time.perf_counter() - started, 2),
                               'device': args.device, 'models': engine.identities, 'params': params,
                               'torch': torch.__version__, **engine.memory()})
    streams = collections.OrderedDict()
    while True:
        try:
            frame = protocol.read_frame(sys.stdin.buffer)
        except (EOFError, protocol.ProtocolError, OSError) as error:
            sys.stderr.write('protocol end: %r\n' % (error,))
            return 0
        if frame is None:
            return 0                                  # stdin closed: the parent is gone or finished
        header, payload = frame
        try:
            reply = {'ok': True, **handle(engine, streams, header, payload)}
        except Exception as error:                    # noqa: BLE001 - one failed request never ends the sidecar
            sys.stderr.write('request %r failed: %r\n' % (header.get('op'), error))
            reply = {'ok': False, 'error': '%s: %s' % (type(error).__name__, error)}
        protocol.write_frame(out, {'id': header.get('id'), **reply})


if __name__ == '__main__':
    sys.exit(main())
