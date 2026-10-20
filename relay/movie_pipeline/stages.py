"""Stage adapters around the REAL relay code (relay.movie_pipeline, W10; moved from movie-trial-v1/pipeline). Each worker returns an immutable result.

No stage here re-implements relay policy; they split worker.process_actor_chunk into its existing steps:

  separate_window  = process_actor_chunk's preamble (source/engine/subtitle fingerprint checks, original-layout.wav
                     extraction with the selected stream's own layout, window captions), then the first pass exactly as
                     ActorPipeline._process calls it (resolve_channel_layout, subtitle_runtime.plan scope,
                     first_pass_parent.run -> the resident GPU Demucs). The bundle lands in the chunk folder, so every
                     later _process call of this window reuses it (first_pass_parent.reusable, key-checked).
  prepare_window   = ActorPipeline.process(..., prepare_only=True) with process_actor_chunk's own arguments, on
                     coordinator._ClaimedStore(<private snapshot>, job, idx, token): ASR + rescue + speaker analysis +
                     SpeakerBank binding + references + translation + the prepared plan (actor-prepared-plan.json).
                     Speaker writes stay in the snapshot (actors.capture records them). Every model call goes through
                     the injected AnalysisService clients (service.language(owner) / service.speakers(owner)) for this
                     window's owner: GPU models in the lease-owned analysis worker, the raw-inference cache in that
                     worker (outside the disposable snapshot). Captions are optional: without subtitles the window
                     still runs automatic ASR -> translation through the same service.
  render_window    = worker.process_actor_chunk on coordinator._ClaimedStore(<real store>, ...) with the render
                     pipeline: _process reuses the plan only if key and every voice equal the real bank, then Vox
                     cloning, the v4 line ledger (claims, artifacts, await, commit), timing/mix, rendered-audibility
                     gate, english.wav, receipt and the fenced chunk settle.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Claimed:
    job_id: str
    index: int
    token: int
    revision: int
    priority: int
    claimed_at: float


@dataclass(frozen=True)
class Separated:
    claim: Claimed
    ok: bool
    original_wav: str = ''
    original_sha256: str = ''
    bundle_sha256: str = ''
    # producer: the bundle's OWN producer record, written by the worker that produced it (observed version/pid/
    # device of that process). requested: the parent's request settings (first_pass_parent 'parent' record), never
    # presented as observed evidence. reused: the bundle came from the window's cache, not from this run.
    producer: dict = field(default_factory=dict)
    requested: dict = field(default_factory=dict)
    reused: bool = False
    seconds: float = 0.0
    error: str = ''
    error_kind: str = ''           # outcomes.classify kind: 'cancelled' | 'resource' | 'recoverable' | 'fatal'
    fault: dict = field(default_factory=dict)   # relay.movie_faults record {kind, code, detail} of the failure


@dataclass(frozen=True)
class Prepared:
    claim: Claimed
    state: str                      # 'prepared' or the report state preparation stopped at, or 'failed'
    plan_path: str = ''
    plan_sha256: str = ''
    plan_reused: bool = False
    plan_voices: dict = field(default_factory=dict)
    snapshot_speakers: dict = field(default_factory=dict)
    capture: object = None
    timing: dict = field(default_factory=dict)
    seconds: float = 0.0
    error: str = ''
    error_kind: str = ''
    fault: dict = field(default_factory=dict)


def _chunk_inputs(config, store, job_id, index):
    """process_actor_chunk's own checks and window geometry (same order, same messages)."""
    from ..process import Cancelled
    from ..media import fingerprint
    from ..worker import context_seconds
    job = store.job(job_id)
    if not job or job['state'] == 'cancelled':
        raise Cancelled('Job is cancelled')
    spec = job['payload']
    chunk = store.chunk(job_id, index)['payload']
    if spec.get('speech_engine', 'actor') != 'actor':
        raise ValueError('This job was created for another speech engine; create a new job')
    context = context_seconds(config)
    if fingerprint(spec['source']) != spec['source_fingerprint'] or config.engine_fingerprint() != spec['engine_fingerprint']:
        raise ValueError('Source or engine changed; create a new job')
    if spec.get('subtitles') and fingerprint(spec['subtitles']) != spec.get('subtitle_fingerprint'):
        raise ValueError('Subtitles changed; create a new job')
    origin = max(0, chunk['start'] - context)
    end = min(spec['duration'], chunk['end'] + context)
    directory = config.cache / job_id / f'chunk-{index:06d}'
    directory.mkdir(parents=True, exist_ok=True)
    audio_stream = next((s for s in spec['media']['streams'] if s.get('index') == spec['audio_index']), None)
    if audio_stream is None or audio_stream.get('codec_type', 'audio') != 'audio':
        raise ValueError('The selected audio stream is missing from the probed media')
    channels = int(audio_stream.get('channels', 0))
    if channels not in (1, 2, 6) or (channels == 6 and audio_stream.get('channel_layout') not in ('5.1', '5.1(side)')):
        raise ValueError('The actor path requires a known mono, stereo or 5.1 layout')
    layout = audio_stream['channel_layout'] if channels == 6 else {1: 'mono', 2: 'stereo'}[channels]
    return {'spec': spec, 'chunk': chunk, 'origin': origin, 'end': end, 'directory': directory,
            'audio_stream': audio_stream, 'channels': channels, 'layout': layout,
            'original': directory / 'original-layout.wav'}


def _caption_kwargs(config, spec):
    from ..embedded_subtitles import spec_choice
    caption = (config.data if hasattr(config, 'data') else config).get('actor_beta')
    return ({'subtitle_choice': spec_choice(spec)} if isinstance(caption, dict) and caption.get('enabled') is True
            and caption.get('caption_targets') is True else {})


def _error_kind(error):
    """outcomes.classify kind ('cancelled' | 'resource' | 'recoverable' | 'fatal'), decided where the error happened."""
    from .outcomes import classify
    return classify(error)[0]


def fault_of(error):
    """The relay.movie_faults record {kind, code, detail} of error: its explicit metadata when present, otherwise
    outcomes.classify's kind with the exception type as the code (legacy relay exceptions)."""
    from ..movie_faults import fault_record
    record = fault_record(error)
    if record is not None:
        return {'kind': record['kind'], 'code': record['code'][:120], 'detail': record['detail'][:4000]}
    return {'kind': _error_kind(error), 'code': 'unclassified_%s' % type(error).__name__,
            'detail': str(error)[:4000]}


def separate_window(config, store, claim, owner):
    """Separation worker: original extraction + captions + the first pass (resident GPU Demucs). Immutable result."""
    started = time.perf_counter()
    try:
        import soundfile as sf
        from .. import first_pass_parent, subtitle_runtime as _subs
        from ..embedded_subtitles import chunk_captions
        from ..media import extract_audio, file_hash, resolve_channel_layout, wav_format
        w = _chunk_inputs(config, store, claim.job_id, claim.index)
        spec = w['spec']
        store.event(claim.job_id, 'stage', chunk=claim.index, stage='trial-separation')
        extract_audio(config, owner, spec['source'], spec['audio_index'], w['origin'], w['end'] - w['origin'],
                      w['original'], channels=w['channels'], preserve_layout=w['layout'])
        captions, _ = chunk_captions(config, owner, spec, w['origin'], w['end'], w['directory'])
        original, rate = sf.read(str(w['original']), dtype='float32', always_2d=True)
        layout_evidence = resolve_channel_layout(original.shape[1], source=dict(w['audio_stream']),
                                                 wav=wav_format(w['original']))
        lfe = 3 if original.shape[1] == 6 else None
        scope = _subs.plan(config, original, rate, w['origin'], captions, lfe, None)
        owner.check()
        bundle = first_pass_parent.run(config, owner, folder=w['directory'], original_source_wav=w['original'],
                                       source_layout_evidence=layout_evidence, origin=w['origin'],
                                       frames=len(original), scope=scope)
        if bundle is None:
            raise ValueError('demucs_first_pass is off; the trial requires the first pass')
        bundle_path = Path(w['directory']) / 'first-pass' / 'first-pass-bundle.json'
        return Separated(claim=claim, ok=True, original_wav=str(w['original']), original_sha256=file_hash(w['original']),
                         bundle_sha256=file_hash(bundle_path),
                         producer=dict(bundle.get('producer') or {}) if isinstance(bundle.get('producer'), dict)
                         else {},
                         requested=dict(bundle.get('parent') or {}) if isinstance(bundle.get('parent'), dict) else {},
                         reused=bool(bundle.get('reused')),
                         seconds=round(time.perf_counter() - started, 3))
    except BaseException as error:   # noqa: BLE001 - every outcome is returned to the coordinator as data
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        fault = fault_of(error)
        return Separated(claim=claim, ok=False, seconds=round(time.perf_counter() - started, 3),
                         error='%s: %s' % (type(error).__name__, str(error)[:800]), error_kind=fault['kind'],
                         fault=fault)


def prepare_window(config, snapshot_claimed, claim, owner, pipeline, separated, base_speakers, actors_module):
    """Analysis worker: ActorPipeline.process(prepare_only=True) on the private snapshot. Immutable result."""
    started = time.perf_counter()
    cap = None
    try:
        from ..embedded_subtitles import chunk_captions
        from ..media import file_hash
        from ..worker import final_chunk
        w = _chunk_inputs(config, snapshot_claimed, claim.job_id, claim.index)
        spec = w['spec']
        if Path(separated.original_wav) != Path(w['original']) or file_hash(w['original']) != separated.original_sha256:
            raise ValueError('original-layout.wav changed after separation')
        captions, _ = chunk_captions(config, owner, spec, w['origin'], w['end'], w['directory'])
        plan_path = Path(w['directory']) / 'actor-prepared-plan.json'
        with actors_module.capture(base_speakers) as cap:
            report = pipeline.process(w['original'], w['origin'], captions, w['directory'], snapshot_claimed,
                                      claim.job_id, owner, render_start=w['chunk']['start'], render_end=w['chunk']['end'],
                                      revision=spec.get('revision', 1), source_channel_layout=dict(w['audio_stream']),
                                      source_identity={'source_fingerprint': spec['source_fingerprint'],
                                                       'audio_index': spec['audio_index']},
                                      prepare_only=True, final_window=final_chunk(spec, claim.index),
                                      **_caption_kwargs(config, spec))
        state = report.get('state') or 'unknown'
        voices, plan_sha = {}, ''
        if state == 'prepared' and plan_path.is_file():
            plan_sha = file_hash(plan_path)
            voices = json.loads(plan_path.read_text(encoding='utf-8')).get('voices') or {}
        snapshot_speakers = snapshot_claimed.speakers(claim.job_id)
        return Prepared(claim=claim, state=state, plan_path=str(plan_path) if plan_sha else '', plan_sha256=plan_sha,
                        plan_reused=bool(report.get('preparation_reused')), plan_voices=voices,
                        snapshot_speakers=snapshot_speakers, capture=cap, timing=dict(report.get('timing') or {}),
                        seconds=round(time.perf_counter() - started, 3))
    except BaseException as error:   # noqa: BLE001
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        fault = fault_of(error)
        return Prepared(claim=claim, state='failed', capture=cap, seconds=round(time.perf_counter() - started, 3),
                        error='%s: %s' % (type(error).__name__, str(error)[:800]), error_kind=fault['kind'],
                        fault=fault)


def render_window(config, real_claimed, claim, owner, pipeline):
    """Coordinator thread: the relay's own process_actor_chunk under the real chunk claim (raises like the relay)."""
    from ..worker import process_actor_chunk
    result = process_actor_chunk(config, real_claimed, claim.job_id, claim.index, owner, pipeline)
    after = real_claimed.chunk(claim.job_id, claim.index)
    if after['state'] == 'working' and after['payload'].get('claim_token') == claim.token:
        raise RuntimeError('Worker exited without a chunk receipt')
    return result
