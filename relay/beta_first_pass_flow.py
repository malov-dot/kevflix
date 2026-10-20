"""Demucs first-pass caller flow for ActorPipeline (actor_beta.demucs_first_pass; default off). B glue only.

C owns relay.first_pass_parent (run/validate/cache) and the demucs guard; this module only turns C's validated
bundle into the pipeline's arrays and records, exactly as MINIMAL-CALLER-PROPOSAL section 3 (root-approved design):
  original := processing (float32 STEREO; mix, protection, ledger), speech := raw_vocals (ASR, speakers,
  references, pan/levels), removal := removal_stem (= processing - background; mix removal ONLY),
  source original (1/2/6-channel original-layout.wav) kept separately as lineage; lfe None; layout 'stereo'.
No fake 6-channel stem, no speech-estimate.wav, no BandIt identity in the reported scope.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

VERSION = 'first-pass-caller-flow-v1'
STEMS = ('processing', 'raw_vocals', 'background', 'removal_stem')


def enabled(config):
    data = getattr(config, 'data', config)
    beta = data.get('actor_beta') if isinstance(data, dict) else None
    return isinstance(beta, dict) and beta.get('enabled') is True and beta.get('demucs_first_pass') is True


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(bundle):
    """Identity stored in the prepared plan and checked on reuse (bundle key + every stem file hash)."""
    return {'key': bundle['key'], **{n + '_sha256': bundle[n]['sha256'] for n in STEMS},
            **{n + '_wav': bundle[n]['wav'] for n in STEMS}, 'original_source_sha256': bundle['original_source']['sha256']}


def arrays(bundle, frames, rate=48000):
    """Read the bundle WAVs, re-verify sha256/geometry; returns dict name -> float32 [frames, 2]."""
    import numpy as np
    import soundfile as sf
    out = {}
    for name in STEMS:
        entry = bundle.get(name) or {}
        path = Path(entry.get('wav', ''))
        if not path.is_file() or _sha(path) != entry.get('sha256') or entry.get('layout') != 'stereo':
            raise ValueError('first-pass %s missing, changed or not stereo' % name)
        data, file_rate = sf.read(str(path), dtype='float32', always_2d=True)
        if file_rate != rate or data.shape != (frames, 2) or not np.isfinite(data).all():
            raise ValueError('first-pass %s is not finite 48 kHz stereo of the window' % name)
        out[name] = data
    return out


def separation(bundle, scope, caller_wall_seconds=None):
    """report['separation'] for the first pass; the subtitle scope keeps its frames/unprocessed walls but names the
    actual producer (the BandIt-keyed scope identity is replaced by the bundle key).

    Timing semantics (C clarification): bundle.metrics.separation_seconds is model inference + resampling ONLY;
    `caller_wall_seconds` is the ActorPipeline-measured wall around first_pass_parent.run (child start, guard, I/O,
    validation). Neither is a stage, request-to-ready or mux time; root's request-to-ready/mux timing is authoritative."""
    record_scope = None
    if scope is not None:
        record_scope = {k: v for k, v in scope.items() if k not in ('key', 'separator')}
        record_scope.update(key=bundle['key'], separator={'engine': 'htdemucs', 'first_pass_key': bundle['key'],
                                                          'params': (bundle.get('key_inputs') or {}).get('params')},
                            scoped=True, first_pass=True)
    metrics = dict(bundle.get('metrics') or {})
    inference = metrics.get('separation_seconds', 0) if not bundle.get('reused') else 0
    return {**metrics, 'seconds': inference,
            'seconds_scope': 'model inference + resampling only (bundle.metrics.separation_seconds); not stage or end-to-end',
            'inference_and_resampling_seconds': inference,
            'caller_wall_seconds': None if caller_wall_seconds is None else round(float(caller_wall_seconds), 6),
            'caller_wall_scope': 'ActorPipeline wall around first_pass_parent.run (child + guard + I/O + validation); '
                                 'request-to-ready and mux are measured by root',
            'engine': 'htdemucs-first-pass', 'first_pass': record(bundle), 'reused': bool(bundle.get('reused')),
            'stale_cache_reasons': bundle.get('stale_cache_reasons') or [], 'subtitle_scope': record_scope}


def layout_evidence(bundle):
    return {'layout': 'stereo', 'basis': 'first-pass processing transform',
            'transform': (bundle.get('processing') or {}).get('transform'),
            'source_layout': bundle['original_source']['layout'],
            'source_layout_evidence': bundle['original_source'].get('layout_evidence')}


def prepared_matches(prepared, bundle):
    """Prepared reuse requires the stored first-pass identity to equal the CURRENT validated bundle exactly."""
    stored = prepared.get('first_pass')
    return isinstance(stored, dict) and stored == record(bundle)
