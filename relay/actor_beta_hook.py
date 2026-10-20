"""Opt-in actor-beta hooks for the existing KevFLIX PC ActorPipeline path (installed as relay/actor_beta_hook.py).

Everything here is inert unless config['actor_beta']['enabled'] is literally True. Works with
relay.config.Config (__getitem__/.data/.resolve, no .get) and plain dicts. B's gate and A's automatic
factory are loaded only on the beta path, so the default relay import graph is unchanged.
"""
import hashlib
import importlib.util
import sys
import threading
import time
from pathlib import Path

# The app tree: <app>/relay (this file) and <app>/actor-tts (Vox runtime adapter, automatic renderer, reference cache).
# Nothing is loaded from outside the app tree.
APP_ROOT = Path(__file__).resolve().parents[1]
ACTOR_TTS_ROOT = APP_ROOT / 'actor-tts'
# Cue-scoped preparation uncertainty emitted by ActorPipeline.process (actor_pipeline.py:114,377,399).
CUE_FALLBACK_KINDS = frozenset({'speaker_unresolved', 'actor_reference_unavailable', 'uncertain_transcript',
                                'unclear_dialogue_without_aligned_caption', 'overlapping_dialogue',
                                'possible_merged_speakers'})
# Cue-scoped take problems in _render_prepared: the take is dropped and that cue keeps original audio.
TAKE_FALLBACK_KINDS = frozenset({'speech_readback_mismatch', 'critical_readback_mismatch', 'actor_gain_unresolved',
                                 'speech_crosses_window', 'generated_speech_touches_protected_original',
                                 'speech_timing_overrun'})
# A's ActorUnavailable reasons (actor-tts/runtime-adapter/errors.py REASONS).
CUE_REASONS = frozenset({'reference_unverified', 'reference_invalid', 'untrusted_actor', 'text_invalid', 'output_invalid'})
# Legacy reason-only wait list, consulted ONLY for errors without fault metadata. Explicit fault_kind/fault_code
# (relay.movie_faults.fault_record) is read first and always wins. server_unavailable is not a wait: an unavailable or
# misconfigured runtime is fatal, never an indefinite resource hold.
WAIT_REASONS = frozenset({'gpu_admission_hold'})
# Remaining cue-scoped uncertainty that may play a whole window as original (worker.py); never global kinds.
# Only a mapped merged-speaker issue is cue-scoped. An unmappable label still blocks the window.
WINDOW_ORIGINAL_KINDS = (CUE_FALLBACK_KINDS - {'possible_merged_speakers'}) | TAKE_FALLBACK_KINDS | frozenset({
    'overlapping_dialogue', 'singing_requires_pitch_tempo_path'})
# The speaker clustering/embedding policy is NOT a constant here: the backend reports its effective policy
# (SpeakerAnalyzer.policy_metadata(), via the AnalysisService) and ActorPipeline records it per window
# (report['speaker_policy'], admission window 'speaker_policy'). model_provenance() refuses when it is missing.
LANE_WAIT_SECONDS = 30.0


def fault_of(error):
    """Explicit {kind, code, detail} fault metadata of a local or separately loaded adapter error, else None."""
    from .movie_faults import fault_record
    return fault_record(error)


class ActorResourceWait(InterruptedError):
    """A real lease/VRAM/preemption/lane hold (fault_kind 'resource'). An InterruptedError, so the existing
    waiting_resources path applies, and it carries its fault record for outcomes.classify."""
    fault_kind = 'resource'

    def __init__(self, code, detail):
        self.fault_code, self.detail = str(code), str(detail)
        super().__init__('%s: %s' % (self.fault_code, self.detail))


def is_resource_wait(error):
    """True only for a resource hold: explicit fault_kind 'resource', or (no fault metadata) the legacy
    ResourceYield/InterruptedError. Fatal, recoverable and cancelled metadata are never a wait."""
    fault = fault_of(error)
    if fault is not None:
        return fault['kind'] == 'resource'
    from .resource_budget import ResourceYield
    return isinstance(error, (ResourceYield, InterruptedError))


def _lookup(config, key):
    try:
        return config[key]
    except (KeyError, TypeError, IndexError):
        data = getattr(config, 'data', None)
        return data.get(key) if isinstance(data, dict) else None


def beta_settings(config):
    value = _lookup(config, 'actor_beta')
    return value if isinstance(value, dict) and value.get('enabled') is True else None


def _root(config):
    """<app>/actor-tts (actor_beta.actor_tts_root may point at another install of the same tree)."""
    return Path((beta_settings(config) or {}).get('actor_tts_root') or ACTOR_TTS_ROOT)


def _load(name, path):
    """Load a frozen A/B module once and register it so its own relative lookups resolve."""
    module = sys.modules.get(name)
    if module is not None and Path(getattr(module, '__file__', '')).resolve() == Path(path).resolve():
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


STABLE_GATE_SHA256 = 'e9a5be1d7b7090ea3bdac5cbc600b7aa0d748039760076fa3116c25755898bdf'

# ---- actor engine (actor-tts-vox-v2) ---------------------------------------------------------------------
# Movie dubbing is Vox only (owner decision 2026-10-17: "I think switching to vox is best"). actor_beta.actor_engine
# may be absent or "vox"; anything else (including the former "breeze") is refused. The Breeze path that live beta.5
# loads (A's actor-tts/automatic-v1) is no longer reachable from this file; A's frozen files stay on disk unchanged, and
# rollback is reinstalling the previous actor_beta_hook.py (sha256 90aaf6f9...), not a flag.
# The Vox candidate is imported only after every file in actor_beta.vox.sources_sha256 matches (the pins are part of
# the actor_beta options, so Config.engine_fingerprint() changes whenever the engine or its code changes).
ACTOR_ENGINES = ('vox',)
VOX_ENGINE = 'VoxCPM2/audio.cpp'
# Code only: the runtime manifest is a machine-specific file located by config (actor_runtime_manifest) and checked by
# content (model + native artifact hashes, approved.py); it is not part of the app tree.
VOX_REQUIRED_SOURCES = ('automatic-v1/automatic_renderer.py', 'reference-cache/reference_cache.py')


def actor_engine(config):
    beta = beta_settings(config)
    if beta is None:
        return None
    engine = beta.get('actor_engine', 'vox')
    if engine not in ACTOR_ENGINES:
        raise ValueError('actor_beta.actor_engine must be "vox" (movie dubbing is Vox only), not %r' % (engine,))
    return engine


def vox_root(config):
    vox = (beta_settings(config) or {}).get('vox') or {}
    return Path(vox.get('root') or _root(config))


def verify_vox_sources(config):
    """Refuse (fatal ValueError) unless actor_beta.vox.sources_sha256 pins exactly the Vox candidate's code."""
    root = vox_root(config).resolve()
    pins = ((beta_settings(config) or {}).get('vox') or {}).get('sources_sha256')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('actor_beta.vox.sources_sha256 is required for actor_engine "vox"')
    expected = set(VOX_REQUIRED_SOURCES) | {'runtime-adapter/' + p.name for p in (root / 'runtime-adapter').glob('*.py')}
    if set(pins) != expected:
        raise ValueError('actor_beta.vox.sources_sha256 must pin exactly: %s' % sorted(expected))
    for relative, digest in sorted(pins.items()):
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _file_hash(path) != digest:
            raise ValueError('Vox candidate file changed or missing: %s' % relative)
    return root


def _load_verified(name, path, digest, extra=None):
    """Exec exactly the bytes whose sha256 equals `digest` (no .pyc, no re-read after the check; review P3-4)."""
    import types
    path = Path(path)
    module = sys.modules.get(name)
    if (module is not None and getattr(module, '__vox_loaded_sha256__', None) == digest
            and Path(getattr(module, '__file__', '')).resolve() == path.resolve()):
        return module
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Vox candidate file changed or missing: %s' % path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(extra or {})
    module.__vox_loaded_sha256__ = digest
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def automatic_module(config):
    """The pinned Vox automatic renderer module (the only movie actor engine), exec'd from its verified bytes. Its
    adapter package is then loaded from bytes checked against the same pins (__vox_pins__)."""
    actor_engine(config)
    root = verify_vox_sources(config)
    pins = dict((beta_settings(config) or {})['vox']['sources_sha256'])
    return _load_verified('_movie_vox_automatic_renderer', root / 'automatic-v1' / 'automatic_renderer.py',
                          pins['automatic-v1/automatic_renderer.py'], {'__vox_pins__': pins})


# ---- movie GPU lease v2 (the lease v2 contract, section 2): Sem priority + process ownership ------------------------
# actor_beta.vox.gpu_lease = {
#   "client": <movie-gpu-lease-v2/lease_client.py>, "client_sha256": <pin>,     (exec'd from the verified bytes)
#   "sem_mode": "absent" | "running",          REQUIRED, explicit, never inferred (contract section 1)
#   "reserve_mib": 3072, "grace_s": 0.15,       (contract defaults)
#   "budget_mib": optional (default vox.instances x actor_required_gpu_mib), "ttl_s": 30, "wait_s": 0,
#   "state_dir": optional}
# Absent -> None: the Vox guard then fails closed (every take waits; nothing starts without a lease). Every engine is
# spawned into the lease's kill-on-close Job (host process_owner='lease'). This file never opens Sem's status.json;
# only W5's pinned client does (running mode) or checks process identity (absent mode).
_LEASE_CLIENTS = {}
SEM_MODES = ('absent', 'running')

# ---- the app's movie lease hub (INTERFACES.md, LEASE VIEW API) -------------------------------------------------------
# In the app, the ONE movie GPU lease of the process belongs to W10's LeaseHub (fed by W7's holder). Vox takes VIEWS of
# it: the hub is REGISTERED here explicitly (register_lease_hub), never patched into this module. APP_RUNTIME_LEASE tells
# MoviePipeline that this hook resolves the hub's source itself.
APP_RUNTIME_LEASE = True
_HUB = {'hub': None}
_HUB_LOCK = threading.Lock()


def register_lease_hub(hub):
    """Called by the movie pipeline once its LeaseHub exists (hub.source() -> callable(name) -> LeaseView, .key)."""
    with _HUB_LOCK:
        _HUB['hub'] = hub
    return hub


def unregister_lease_hub(hub=None):
    """Called by the movie pipeline at close (after its engines stopped)."""
    with _HUB_LOCK:
        if hub is None or _HUB['hub'] is hub:
            _HUB['hub'] = None


def registered_lease_source():
    with _HUB_LOCK:
        hub = _HUB['hub']
    return hub.source() if hub is not None else None


def _app_runtime_present():
    """True inside the app tree (relay/app_runtime): then only the registered hub may provide the lease."""
    return (Path(__file__).with_name('app_runtime') / '__init__.py').is_file()


def movie_lease_source(config):
    """The Vox lease source: the registered app hub's views (no second lease in the process); inside the app with no
    hub registered -> None (fails closed: every take waits); outside the app -> the configured lease-v2 client."""
    hub_source = registered_lease_source()
    if hub_source is not None:
        return hub_source
    if _app_runtime_present():
        return None
    beta = beta_settings(config) or {}
    vox = beta.get('vox') or {}
    cfg = vox.get('gpu_lease')
    if not isinstance(cfg, dict):
        return None
    sem_mode = cfg.get('sem_mode')
    if sem_mode not in SEM_MODES:
        raise ValueError('actor_beta.vox.gpu_lease.sem_mode must be "absent" or "running" (explicit), not %r' % (sem_mode,))
    path = Path(cfg['client'])
    if not path.is_file():
        raise ValueError('movie GPU lease client missing: %s' % path)
    reserve, grace = int(cfg.get('reserve_mib', 3072)), float(cfg.get('grace_s', 0.15))
    key = (str(path.resolve()), cfg.get('client_sha256'), str(cfg.get('state_dir') or ''), sem_mode, reserve, grace)
    client = _LEASE_CLIENTS.get(key)
    if client is None:
        module = _load_verified('_movie_gpu_lease_client', path, cfg.get('client_sha256'))   # ValueError if changed
        client = _LEASE_CLIENTS[key] = module.LeaseClient(sem_mode=sem_mode, reserve_mib=reserve, grace_s=grace,
                                                          state_dir=cfg.get('state_dir') or None)
    # FIX 6: the lease asks for room for the configured concurrency bound; the GRANTED budget (lease v2: free VRAM above
    # the 3072 MiB floor) and the measured per-instance peak then decide how many actually run (admitted_concurrency).
    value = vox.get('max_instances', vox.get('instances', 'auto'))
    pool = int(vox.get('max_instances_cap') or 2) if str(value).strip().lower() == 'auto' else int(value or 1)
    budget = int(cfg.get('budget_mib') or pool * int(_lookup(config, 'actor_required_gpu_mib') or 2700))
    ttl, wait = float(cfg.get('ttl_s', 30.0)), float(cfg.get('wait_s', 0.0))
    source = lambda holder: client.acquire(budget, ttl_s=ttl, wait_s=wait, holder=holder)   # noqa: E731
    source.key = key + (budget, ttl, wait)
    return source


def release_vox_engines(config, reason):
    """Stop the Vox engines and give their GPU lease back now (the job lease keeper otherwise holds them job_lease_idle_seconds after
    the last window). Returns how many keepers were released; 0 when Vox was never started in this process. May block while the
    engines drain: call it from a worker thread."""
    module = sys.modules.get('_movie_vox_automatic_renderer')
    keepers = list(getattr(module, '_KEEPERS', {}).values()) if module is not None else []
    for keeper in keepers:
        keeper.release(reason)
    return len(keepers)


def movie_lease_guard_factory(config, module):
    """The Vox guard: one process-level JobLeaseKeeper per lease configuration, so ONE GPU lease and warm engines
    span the windows, chunks and verification retries of a job (review P2-1). No gpu_lease -> fails closed."""
    import functools
    source = movie_lease_source(config)
    if source is None:
        return functools.partial(module.MovieLeaseGuard, lease_source=None)
    idle = float(((beta_settings(config) or {}).get('vox') or {}).get('job_lease_idle_seconds', 150))
    keeper = module.shared_keeper(source.key + (idle,), source, idle_seconds=idle)
    return functools.partial(module.MovieLeaseGuard, keeper=keeper)


# ---- program-owned reference provenance (written at E3; verified by actor-tts-vox-v2 reference_provenance.py) ----
PROVENANCE_VERSION = 'vox-actor-reference-provenance/v1'
SOURCE_TEXT_VERSION = 'vox-actor-reference-source-text/v1'
SELECTION_RULE = ('cues of this diarized speaker in this window whose actual padded excerpt span intersects raw '
                  'diarization turns of exactly one non-empty label with speech in the cue core (verified for every '
                  'candidate cue, selected or context-only), each 0.25 s '
                  'to actor_reference_seconds; clear transcripts first, then longest; no overlapping spans; total at '
                  'most actor_reference_seconds, stopping at 5 s; joined in time order with 0.12 s gaps; the speech-stem '
                  'channel with the most energy')


def selection_module_sha256():
    """The two installed reference cutters; no voice metadata can add a function or hash."""
    return {'relay.actor_reference.make_reference': _file_hash(Path(__file__).with_name('actor_reference.py')),
            'relay.actor_pipeline.make_line_reference': _file_hash(Path(__file__).with_name('actor_pipeline.py'))}


def _atomic_write(path, payload):
    import json
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    descriptor, temporary_name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError('existing reference record differs: %s' % path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def write_reference_provenance(config, context, receipt, receipt_path, ident, voice):
    """Two separate program records for a reference make_reference just cut and B just receipted:
    <job>/reference-provenance/<key>.json (source file hashes, time spans, speaker id, selection rule, B receipt; NO
    text) and <job>/reference-source-text/<key>.json (the exact source-language transcript; no target text).
    Line-own keys include the admitted job/revision/actor identity; legacy keys remain the reference sha.
    Returns the provenance path. Raises (the caller records the issue; that cue later keeps its original audio)."""
    job_dir = Path(context['receipt_dir']).parent
    sha = receipt['reference_sha256']
    if sha != str(voice.get('reference_sha256') or '').lower():
        raise ValueError('reference changed between attach and provenance')
    spoken = receipt.get('reference_text', '')
    line_reference = voice.get('line_reference')
    function = ('relay.actor_pipeline.make_line_reference' if 'line_reference' in voice
                else 'relay.actor_reference.make_reference')
    selection_rule = {'function': function, 'module_sha256': selection_module_sha256()[function],
                      'limit_seconds': float(_lookup(config, 'actor_reference_seconds')), 'rule': SELECTION_RULE}
    record_key = sha
    if 'line_reference' in voice:
        if not isinstance(line_reference, dict) or line_reference != receipt.get('line_reference'):
            raise ValueError('line reference metadata differs from admitted receipt')
        selection_rule['rule'] = ('one cue own speech span, a clean trimmed subset, or a short-span extension '
                                  'inside one actual continuous solo raw turn; at least 1 second and at most '
                                  'actor_reference_seconds; exact samples from the same admitted speech stem; '
                                  'the channel with the most energy')
        import json
        actor_id, revision, job_id = receipt['actor_id'], receipt['revision'], receipt['job_id']
        if not isinstance(revision, int) or isinstance(revision, bool) or actor_id != ident \
                or actor_id != 'line-' + str(line_reference.get('cue_id')):
            raise ValueError('line reference identity differs from admitted receipt')
        identity = json.dumps([job_id, revision, actor_id], separators=(',', ':'), ensure_ascii=True).encode('ascii')
        record_key += '-' + hashlib.sha256(identity).hexdigest()[:16]
        selection_rule['line_reference'] = {key: line_reference[key] for key in (
            'version', 'cue_id', 'cue_speech_span', 'source_interval', 'raw_turn', 'sample_span')}
    text_path = job_dir / 'reference-source-text' / (record_key + '.json')
    text_sha = _atomic_write(text_path, {
        'version': SOURCE_TEXT_VERSION, 'reference_sha256': sha, 'speaker_id': ident, 'language': 'source',
        'reference_text': spoken, 'reference_text_sha256': hashlib.sha256(spoken.encode('utf-8')).hexdigest()})
    path = job_dir / 'reference-provenance' / (record_key + '.json')
    _atomic_write(path, {
        'version': PROVENANCE_VERSION, 'created_by': 'program',
        'creator': 'relay.actor_pipeline E3 -> actor_beta_hook.persist_reference_receipt',
        'job_id': receipt['job_id'], 'revision': receipt['revision'], 'speaker_id': ident,
        'reference': {'path': str(Path(voice['reference']).resolve()), 'sha256': sha,
                      'pcm_sha256': receipt['reference_pcm_sha256'], 'rate': receipt['rate'],
                      'channel': receipt['channel']},
        'source': {'source_wav': dict(receipt['source_wav']), 'speech_stem': dict(receipt['speech_stem']),
                   'source_fingerprint': receipt['source_fingerprint'], 'audio_index': receipt['audio_index'],
                   'window_origin_seconds': receipt['origin']},
        'spans_movie_seconds': [list(map(float, span)) for span in receipt['intervals']],
        'selection_rule': selection_rule,
        'admission_receipt': {'path': str(receipt_path), 'sha256': _file_hash(receipt_path),
                              'version': receipt['version']},
        'source_text': {'path': str(text_path), 'sha256': text_sha}})
    return path


def adopt_fitted_text(config, report, cues, lines):
    """actor_engine "vox" only. When the Vox duration fit chose an EXISTING shorter alternative translation for a
    take (before synthesis; never tempo or truncation), that text becomes the cue's English, so the unchanged readback
    and ledger compare against the words actually spoken. Returns the adopted rows (also in report['vox_duration_fit'])."""
    if actor_engine(config) != 'vox':
        return []
    by_id = {c['id']: c for c in cues}
    adopted = []
    for line in lines:
        fit = line.get('duration_fit')
        if line.get('canonical_id') or line.get('engine') != VOX_ENGINE or not isinstance(fit, dict):
            continue
        if fit.get('text_changed') is not True:
            continue
        cue = by_id.get(line.get('cue'))
        allowed = {a.get('text'): a.get('basis') for a in fit.get('alternatives') or [] if isinstance(a, dict)}
        if cue is None or cue.get('english') != fit.get('requested_text') or line.get('text') not in allowed:
            continue
        cue['english_requested'] = cue['english']
        cue['english'] = line['text']
        cue['english_basis'] = 'vox-duration-fit:' + str(allowed[line['text']])
        adopted.append({'cue': cue['id'], 'requested': cue['english_requested'], 'spoken': line['text'],
                        'basis': allowed[line['text']], 'seed': line.get('seed')})
    if adopted:
        report.setdefault('vox_duration_fit', []).extend(adopted)
    return adopted


def pipeline_gate(config):
    packaged = Path(__file__).with_name('beta_stable_gate.py')   # byte-identical reviewed copy, no research path
    if packaged.is_file() and hashlib.sha256(packaged.read_bytes()).hexdigest() == STABLE_GATE_SHA256:
        from . import beta_stable_gate
        return beta_stable_gate
    # The app ships only the packaged reviewed gate; there is no research-path fallback.
    raise ValueError('relay/beta_stable_gate.py is missing or not the reviewed gate (sha256 %s)' % STABLE_GATE_SHA256)


_HASHES = {}


def _file_hash(path):
    """sha256 memoized by (path, size, mtime_ns); large model files are not re-read every window."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        _HASHES[key] = digest.hexdigest()
    return _HASHES[key]


# ---- B E1/E2/E3 -----------------------------------------------------------------------------------

def admission_window(config, *, job_id, revision, origin, rate, source_identity, folder, source_wav):
    """E1: bound right after the decoded source WAV load, before the prepared-plan lookup."""
    if beta_settings(config) is None:
        return None
    return {'job_id': job_id, 'revision': revision, 'origin': origin, 'rate': rate,
            'source_identity': source_identity, 'folder': str(folder),
            'source_wav': {'path': str(Path(source_wav).resolve()), 'sha256': _file_hash(source_wav)}}


def _resolved(config, key):
    value = config[key]
    return Path(config.resolve(value) if hasattr(config, 'resolve') else value)


def model_provenance(config, policy):
    """E2: the speaker embedding model the backend ACTUALLY uses, plus its effective policy.

    `policy` is SpeakerAnalyzer.policy_metadata() as reported for this window (fresh or reused plan). The normal
    pipeline uses the native Community-1 PyTorch embedding for centroids, refinement and references; an external
    ONNX encoder is named only when the backend reports external_onnx_used. Raises when the policy is unreported or
    the model file is absent (never fabricated)."""
    if not isinstance(policy, dict) or not policy:
        raise ValueError('speaker policy metadata (SpeakerAnalyzer.policy_metadata()) was not reported for this window')
    if policy.get('external_onnx_used') is True:
        model, family = _resolved(config, 'speaker_onnx_model'), 'speaker_onnx_model'
    else:
        # models/pyannote-community-1/config.yaml: embedding: $model/embedding (diarization_model may name the
        # folder or its config.yaml).
        root = _resolved(config, 'diarization_model')
        root = root if root.is_dir() else root.parent
        model, family = root / 'embedding' / 'pytorch_model.bin', 'pyannote-community-1/embedding'
    if not model.is_file():
        raise ValueError('Speaker model file is missing: %s' % model)
    return {'family': family, 'model_path': str(model), 'model_sha256': _file_hash(model),
            'policy': dict(policy), 'policy_source': 'SpeakerAnalyzer.policy_metadata'}


def admission_context(config, window, original, speech, cues, all_cues, voices, cache, report=None):
    """E2: complete B context for the CURRENT window. Missing window/stem/model/policy raise (global, explicit).
    The speaker policy is the one ActorPipeline recorded for this window (window['speaker_policy']), else the
    report's own report['speaker_policy']."""
    if not window:
        raise ValueError('actor_beta admission window was not bound for this job window')
    policy = window.get('speaker_policy') or (report or {}).get('speaker_policy')
    if window.get('speech_stem'):   # first pass: the raw-vocals file the pipeline actually loaded (bundle-validated)
        speech_stem = dict(window['speech_stem'])
    else:
        stem = Path(window['folder']) / 'speech-estimate.wav'
        speech_stem = {'path': str(stem.resolve()), 'sha256': _file_hash(stem)}
    context = {**{k: window[k] for k in ('job_id', 'revision', 'origin', 'rate', 'source_identity', 'source_wav')},
               'original': original, 'speech': speech, 'cues': cues, 'all_cues': all_cues, 'voices': voices,
               'speech_stem': speech_stem, 'model_provenance': model_provenance(config, policy),
               'receipt_dir': str(Path(cache) / window['job_id'] / 'reference-receipts'),
               # The configured movie cache root: every source, stem and reference lives under it (B's gate admits only
               # its DEFAULT_ROOTS plus these; nothing broader, no Sem root). All PCM/hash/model checks are unchanged.
               'allowed_roots': [str(Path(cache).resolve())]}
    context['actor_reference_seconds'] = float(_lookup(config, 'actor_reference_seconds'))
    if report is not None and 'raw_turns' in report:
        raw_turns = report['raw_turns']
        context['raw_turns'] = ([{k: turn.get(k) for k in ('start', 'end', 'speaker')}
                                 if isinstance(turn, dict) else turn for turn in raw_turns]
                                if isinstance(raw_turns, (list, tuple)) else raw_turns)
    if report is not None:
        from copy import deepcopy
        for key in ('dialogue_raw_cues', 'speaker_binding_evidence'):
            if key in report:
                context[key] = deepcopy(report[key])
        if all(key in report for key in ('origin', 'end', 'render_start', 'render_end')):
            context['window'] = {key: report[key] for key in ('origin', 'end', 'render_start', 'render_end')}
    if window.get('caption_inputs') is not None and report is not None:
        context.update(caption_inputs=window['caption_inputs'], speaker_evidence=report.get('speaker_evidence'),
                       window_issues=report.get('issues'),
                       window={'origin': report['origin'], 'end': report['end'],
                               'render_start': report['render_start'], 'render_end': report['render_end']},
                       # The flow's kept-original set (settle's kept_ids): policy-preserved cues | beta fallback cues.
                       kept_original_cue_ids=sorted({str(c) for c in report.get('kept_original_cue_ids') or ()} |
                                                    {str(c) for c in report.get('beta_original_fallback') or {}}))
    return context


# Sem/personal roots a movie reference may never come from (component-wise, as the adapter's approved.py). The movie
# cache and actor_beta.vox.movie_roots are allowed even when they lie under one of these.
SEM_ROOTS = ('C:/Sem-Voice', 'C:/Sem-Trials', 'C:/Sem-Data')


def _parts(path):
    return [p.lower() for p in Path(str(path)).resolve().parts]


def _under(path, root):
    child, parent = _parts(path), _parts(root)
    return len(child) > len(parent) and child[:len(parent)] == parent


def reference_root_problem(config, path, job_dir):
    """None when a cut reference may be admitted by location: inside THIS job folder and not under a Sem root
    (unless under the movie cache or a configured movie root). Otherwise the refusal reason."""
    vox = (beta_settings(config) or {}).get('vox') or {}
    if not _under(path, job_dir):
        return 'reference is not inside this job folder (%s)' % job_dir
    allowed = [r for r in [_lookup(config, 'cache_dir'), getattr(config, 'cache', None)] + list(vox.get('movie_roots') or [])
               if r]
    sem = list(SEM_ROOTS) + [r for r in vox.get('sem_roots') or () if r]
    if any(_under(path, r) for r in sem) and not any(_under(path, r) for r in allowed):
        return 'reference path is under a protected Sem/personal root'
    return None


def admit_reference(config, window, original, speech, active, all_cues, ident, voice, path, choice, cache, report):
    """E3: clean anonymous reference admission for a reference make_reference just cut, BEFORE attach_reference.

    Returns None when it may be attached; otherwise the refusal reason (the cut file is removed, nothing is attached,
    and the caller scopes actor_reference_unavailable to that speaker's cues, which keep original audio). Guards: the
    candidate voice is exactly what attach_reference would store; location (job folder, no Sem root); B's
    source_bound_receipt (exact program-cut excerpt of THIS window's source speech stem, transcript equal to the
    contributing source cues or '' for reference-only); vox: the program provenance + source-text records. The source
    transcript is optional; an empty one selects reference-only cloning. Not actor_beta: None (legacy attach)."""
    if beta_settings(config) is None:
        return None
    text = choice.get('text') or ''
    candidate = {**voice, 'reference': str(Path(path).resolve()), 'reference_sha256': _file_hash(path),
                 'reference_text': text, 'source_intervals': list(choice['intervals'])}
    record = {'speaker': ident, 'reference_sha256': candidate['reference_sha256'],
              'cloning_mode': 'reference+transcript' if text else 'reference-only',
              'intervals': candidate['source_intervals'], 'seconds': choice.get('seconds'),
              'channel': choice.get('channel'), 'admitted': False}
    from .movie_faults import MovieConfigurationError
    try:
        # Missing reviewed gate, admission window, speech stem, speaker model or speaker policy: configuration,
        # terminal for the window (never a per-cue fallback, never a wait).
        gate = pipeline_gate(config)
        context = admission_context(config, window, original, speech, active, all_cues, {ident: candidate}, cache,
                                    report=report)
    except ValueError as error:
        raise MovieConfigurationError('reference_admission_unconfigured', str(error)) from error
    try:
        problem = reference_root_problem(config, path, Path(cache) / window['job_id'])
        if problem:
            raise ValueError(problem)
        receipt = gate.source_bound_receipt(context, ident)
        receipt_path = str(gate.persist_receipt(receipt, context['receipt_dir']))
        record['receipt'] = receipt_path
        if actor_engine(config) == 'vox':
            record['provenance'] = str(write_reference_provenance(config, context, receipt, receipt_path, ident,
                                                                  candidate))
    except (OSError, ValueError, KeyError, TypeError) as error:
        record['reason'] = str(error)
        report.setdefault('actor_references', {})[ident] = record
        try:
            Path(path).unlink()
        except OSError:
            pass
        return 'reference admission refused: %s' % error
    record['admitted'] = True
    report.setdefault('actor_references', {})[ident] = record
    return None


# ---- per-cue fallback policy ------------------------------------------------------------------------

def split_cue_fallbacks(config, issues, *, cue_speakers=None):
    """actor_beta only: (global-policy issues, {cue_id: [kinds]} kept original).

    cue_speakers is the selected source-cue/local-label map captured before binding changes speaker IDs.
    A merged-speaker issue is scoped only when that map identifies affected cues; otherwise it stays global.
    """
    if beta_settings(config) is None:
        return list(issues), {}
    remaining, fallback = [], {}
    for issue in issues:
        if issue.get('kind') == 'possible_merged_speakers':
            label = issue.get('speaker')
            affected = [cue_id for cue_id, speaker in (cue_speakers or {}).items()
                        if label is not None and speaker == label]
            if not affected:
                remaining.append(issue)
                continue
            for cue_id in affected:
                fallback.setdefault(cue_id, []).append('possible_merged_speakers')
        elif issue.get('kind') in CUE_FALLBACK_KINDS and issue.get('cue') is not None:
            fallback.setdefault(issue['cue'], []).append(issue['kind'])
        else:
            remaining.append(issue)
    return remaining, fallback


def take_fallbacks(config, issues, fallback):
    """actor_beta only: move cue-scoped take issues out of `issues` (in place) into `fallback`; return moved ids."""
    if beta_settings(config) is None:
        return set()
    moved, keep = set(), []
    for issue in issues:
        if issue.get('kind') in TAKE_FALLBACK_KINDS and issue.get('cue') is not None:
            fallback.setdefault(issue['cue'], []).append(issue['kind'])
            moved.add(issue['cue'])
        else:
            keep.append(issue)
    issues[:] = keep
    return moved


def fallback_spans(fallback, all_cues):
    """Source intervals of every beta fallback cue; joins the runtime mix protection wall only."""
    ids = set(fallback)
    return [(float(c['start']), float(c['end'])) for c in all_cues if c['id'] in ids]


def _overlapping(cues, spans, exclude):
    return [c for c in cues if c['id'] not in exclude
            and any(min(float(c['end']), b) > max(float(c['start']), a) for a, b in spans)]


def window_fallback(config, report, cues, all_cues):
    """Start of _render_prepared (fresh AND reused plans): recompute this window's fallback wall.

    report['beta_original_fallback'] is the persisted preparation decision; report['beta_runtime_fallback']
    is reset for this render. A synthesis cue overlapping a fallback span also keeps original (it could not be
    scheduled without crossing that wall). Returns (synthesis cues, fallback spans).
    """
    if beta_settings(config) is None:
        return cues, []
    prepared = dict(report.get('beta_original_fallback') or {})
    runtime = report['beta_runtime_fallback'] = {}
    while True:
        spans = fallback_spans({**prepared, **runtime}, all_cues)
        hit = _overlapping(cues, spans, runtime)
        if not hit:
            break
        for cue in hit:
            runtime.setdefault(cue['id'], []).append('overlaps_original_fallback')
    report['beta_fallback_intervals'] = [list(s) for s in spans]
    return [c for c in cues if c['id'] not in runtime], spans


def _cascade(report, cues, lines, placements, all_cues):
    runtime = report['beta_runtime_fallback']
    while True:
        spans = fallback_spans({**(report.get('beta_original_fallback') or {}), **runtime}, all_cues)
        hit = _overlapping(cues, spans, runtime)
        if not hit:
            break
        for cue in hit:
            runtime.setdefault(cue['id'], []).append('overlaps_original_fallback')
    keep = [i for i, c in enumerate(cues) if c['id'] not in runtime]
    report['beta_fallback_intervals'] = [list(s) for s in spans]
    return ([cues[i] for i in keep], [lines[i] for i in keep],
            None if placements is None else [placements[i] for i in keep], spans)


def after_takes(config, report, cues, lines, all_cues):
    """After the render loop: lines exist only for rendered cues; align cues by id and extend the wall."""
    if beta_settings(config) is None:
        return cues, lines, []
    adopt_fitted_text(config, report, cues, lines)
    rendered = {line['cue'] for line in lines}
    for cue in cues:
        if cue['id'] not in rendered:
            report['beta_runtime_fallback'].setdefault(cue['id'], []).append('no_take')
    by_id = {line['cue']: line for line in lines}
    cues = [c for c in cues if c['id'] in rendered]
    cues, lines, _, spans = _cascade(report, cues, [by_id[c['id']] for c in cues], None, all_cues)
    return cues, lines, spans


def after_schedule(config, issues, report, cues, lines, placements, all_cues):
    """Before the pre-mix needs_review: readback/timing take issues keep only their cue original."""
    if beta_settings(config) is None:
        return cues, lines, placements, []
    take_fallbacks(config, issues, report['beta_runtime_fallback'])
    cues, lines, placements, spans = _cascade(report, cues, lines, placements, all_cues)
    report['placements'] = placements
    return cues, lines, placements, spans


def after_mix_takes(config, issues, report, pending, all_cues):
    """After per-take mixing: gain/window/protected-touch take issues keep only their cue original."""
    if beta_settings(config) is None:
        return pending, []
    moved = take_fallbacks(config, issues, report['beta_runtime_fallback'])
    spans = fallback_spans({**(report.get('beta_original_fallback') or {}), **report['beta_runtime_fallback']}, all_cues)
    report['beta_fallback_intervals'] = [list(s) for s in spans]
    return [p for p in pending if p[0]['id'] not in moved], spans


def unavailable_reason(error):
    return getattr(error, 'reason', None) if type(error).__name__ == 'ActorUnavailable' else None


def is_cue_unavailable(config, error):
    """Runtime actor error that keeps ONLY that cue original (actor_beta only).

    Explicit fault metadata (fault_kind/fault_code/detail, relay.movie_faults.fault_record) is read FIRST and wins
    over every reason list: 'recoverable' -> True (the cue keeps original); 'resource' (a real lease/VRAM/preemption
    hold) -> ActorResourceWait, the existing waiting_resources path, carrying its record; 'fatal' and 'cancelled' ->
    False, so the caller re-raises the ORIGINAL error with its metadata (path/asset/policy/configuration/model/CUDA
    faults are terminal, never a soft wait). Only an error without metadata uses the legacy reason lists.
    """
    if beta_settings(config) is None:
        return False
    if type(error).__name__ == 'DeadlineExpired' or getattr(error, 'reason', None) == 'deadline_missed':
        return True        # too late to start: that cue keeps its original audio (review P2-3), never a window error
    fault = fault_of(error)
    if fault is not None:
        if fault['kind'] == 'resource':
            raise ActorResourceWait(fault['code'], fault['detail']) from error
        return fault['kind'] == 'recoverable'
    reason = unavailable_reason(error)
    if reason in WAIT_REASONS:
        raise ActorResourceWait('vox_' + reason, str(error)) from error
    return reason in CUE_REASONS


def render_retry_or_keep(config, render, line):
    """Verification retry: an unavailable cue keeps the first take's receipt (then falls back on its WER)."""
    try:
        return render()
    except Exception as error:
        if not is_cue_unavailable(config, error):
            raise
        return Path(line['audio']), {k: v for k, v in line.items() if k not in ('readback', 'retry_evidence')}


# ---- unattended chunk publication (worker.py) ------------------------------------------------------

def window_original_eligible(result):
    """True only if EVERY remaining issue is cue-scoped uncertainty (each names its cue).

    Global geometry/source/hash/layout/model/policy/ledger/mix issues (soundtrack_peak_overrange,
    possible_merged_speakers, kept_original_*, canonical_speech_collision, persisted-protection, cue-less
    issues) keep the window blocked. Fatal exceptions never reach here (chunk 'error').
    """
    issues = result.get('issues') or []
    return bool(issues) and all(isinstance(i, dict) and i.get('kind') in WINDOW_ORIGINAL_KINDS
                                and i.get('cue') is not None for i in issues)


def unattended_window(config, result, original_wav):
    """worker.process_actor_chunk, before the english.wav extraction (actor_beta only).

    A window blocked ONLY by cue-scoped uncertainty plays the ORIGINAL source audio for that window with its
    explicit reasons; any other needs_review stays blocked. A dubbed candidate keeps its audio.
    Nothing is relabelled as human-accepted.
    """
    if beta_settings(config) is None:
        return result
    state = result.get('state')
    if state == 'needs_review' and not window_original_eligible(result):
        result['beta_blocked_reason'] = sorted({str(i.get('kind', 'unknown')) for i in result.get('issues') or []
                                                if isinstance(i, dict)}) or ['unknown']
    elif state == 'needs_review':
        result['beta_original_reason'] = sorted({i.get('kind', 'unknown') for i in result.get('issues') or []}) or ['unknown']
        result['audio'] = str(original_wav)
        result['mode'] = 'original-unverified'
    elif state == 'candidate':
        result['beta_unattended'] = True
    return result


def unattended_state(config, result, state):
    """Chunk state after mapping (actor_beta only): candidate and original-fallback windows are playable."""
    if beta_settings(config) is None or state != 'needs_review':
        return state
    if result.get('pipeline_state') == 'candidate' and result.get('audio') and result.get('beta_unattended'):
        result.setdefault('quality', {})['beta_unattended_machine_only'] = True
        return 'ready'
    if (result.get('pipeline_state') == 'needs_review' and result.get('mode') == 'original-unverified'
            and result.get('audio') and result.get('beta_original_reason')):
        result.setdefault('quality', {})['beta_unattended_machine_only'] = True
        return 'ready'
    return state


# ---- program GPU lane + per-window renderer --------------------------------------------------------

class GpuLane:
    """Program-owned single actor GPU lane. Separation stays on CPU and never takes it."""

    def __init__(self):
        self._lock, self.holder = threading.Lock(), None

    def acquire(self, holder, timeout=0.0, check=None):
        deadline = time.monotonic() + timeout
        while not self._lock.acquire(timeout=min(.2, max(0.0, deadline - time.monotonic()))):
            if check:
                check()
            if time.monotonic() >= deadline:
                # A real in-process hold: another window's renderer holds the single actor GPU lane.
                raise ActorResourceWait('actor_gpu_lane_busy', 'Movie actor GPU lane is busy')
        self.holder = holder

    def release(self, holder):
        if self.holder is holder:
            self.holder = None
            self._lock.release()

    def held_by(self, holder):
        return self.holder is holder

    def available_for(self, holder):
        return lambda: self.holder is holder


GPU_LANE = GpuLane()


class BetaWindowRenderer:
    """One renderer per prepared window, bound to that window's B context.

    close() is idempotent and releases the lane; a later render() in the SAME window (verification retry)
    re-acquires it within LANE_WAIT_SECONDS. The next window builds a fresh renderer and context.
    """

    def __init__(self, inner, holder, lane=None):
        self.inner, self.holder, self.lane, self.closed_count = inner, holder, lane or GPU_LANE, 0

    @property
    def owner(self):
        return self.inner.owner

    @owner.setter
    def owner(self, value):
        self.inner.owner = value

    @property
    def model(self):
        return self.inner.model

    @property
    def fallbacks(self):
        return getattr(self.inner, 'fallbacks', {})

    def _ensure_lane(self):
        if not self.lane.held_by(self.holder):
            owner = self.inner.owner
            self.lane.acquire(self.holder, LANE_WAIT_SECONDS, getattr(owner, 'check', None))

    def prepare_references(self, voices):
        self._ensure_lane()
        return self.inner.prepare_references(voices)

    def render(self, *args, **options):
        self._ensure_lane()
        return self.inner.render(*args, **options)

    # Concurrency API (actor_engine "vox"): only when the inner renderer provides it.
    def submit(self, *args, **options):
        if not hasattr(self.inner, 'submit'):
            raise AttributeError('the configured actor engine has no concurrent submit()')
        self._ensure_lane()
        return self.inner.submit(*args, **options)

    def cancel_job(self, job_id, reason='scheduler_cancel'):
        return self.inner.cancel_job(job_id, reason) if hasattr(self.inner, 'cancel_job') else 0

    def concurrency(self):
        return self.inner.concurrency() if hasattr(self.inner, 'concurrency') else None

    def admitted_concurrency(self):
        """FIX 6: lines this window may have in flight (1 when the engine has no concurrent submit)."""
        if not hasattr(self.inner, 'admitted_concurrency') or not hasattr(self.inner, 'submit'):
            return 1
        self._ensure_lane()
        return self.inner.admitted_concurrency()

    def note_refusal(self, reason=''):
        if hasattr(self.inner, 'note_refusal'):
            self.inner.note_refusal(reason)

    def close(self):
        self.closed_count += 1
        try:
            return self.inner.close()
        finally:
            self.lane.release(self.holder)


class ConfigView(dict):
    """A's factory does dict(config); relay.config.Config has no keys(). Same values, same attributes."""

    def __init__(self, config):
        data = getattr(config, 'data', config)
        super().__init__(data)
        self._config = config

    def __getattr__(self, name):
        return getattr(self.__dict__['_config'], name)


def _reference_cache(config):
    """A's reference_cache from <app>/actor-tts/reference-cache (pinned in vox.sources_sha256), imported as A's runtime
    does (one module identity for TrustAdmission); passed explicitly to B's packaged gates."""
    directory = str(_root(config) / 'reference-cache')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import reference_cache
    return reference_cache


def caption_gate_kwargs(config):
    """actor_beta.caption_targets: A's factory gets B's packaged caption-target gate (program recomputation)."""
    beta = beta_settings(config)
    if beta is None:
        return {}
    import functools
    if beta.get('caption_targets') is not True:
        # B's packaged stable gate with the app's reference_cache (the automatic renderer never loads a gate itself).
        # Borrowed line references (relay/borrowed_reference.py) are admitted around the reviewed gate, never inside it.
        from .borrowed_reference import BorrowAwareGate
        return {'gate_builder': functools.partial(BorrowAwareGate(pipeline_gate(config)).make_pipeline_gate,
                                                  reference_cache_module=_reference_cache(config))}
    from . import beta_caption_gate
    from .borrowed_reference import BorrowAwareGate
    from .dialogue import caption_text
    return {'gate_builder': functools.partial(beta_caption_gate.make_caption_target_gate, caption_text=caption_text,
                                              stable_gate_module=BorrowAwareGate(beta_caption_gate.load_stable()),
                                              reference_cache_module=_reference_cache(config))}


def make_actor_renderer(config, owner, *, pipeline_context=None, default_factory=None,
                        automatic_factory=None, guard_factory=None, gpu_free=None, lane=None):
    beta = beta_settings(config)
    if beta is None:
        if default_factory is None:
            from .actor_render import ActorRenderer as default_factory
        return default_factory(config, owner)
    lane = lane or GPU_LANE
    if automatic_factory is None or guard_factory is None:
        module = automatic_module(config)
        automatic_factory = automatic_factory or module.make_automatic_renderer
        if guard_factory is None:
            # Sem priority through the movie GPU lease (never by reading Sem's status.json from the relay).
            guard_factory = movie_lease_guard_factory(config, module)
    if gpu_free is None:
        from .resource_budget import ResourceBudget
        gpu_free = ResourceBudget.gpu_free
    holder = object()
    lane.acquire(holder, 0.0)
    try:
        guard = guard_factory(lane.available_for(holder), gpu_free=gpu_free)
        # Constructor/gate errors (source, stem, geometry, hash, layout, model provenance) propagate: fatal.
        context = dict(pipeline_context or {})
        if actor_engine(config) == 'vox':
            context['selection_module_sha256'] = selection_module_sha256()
        inner = automatic_factory(ConfigView(config), owner, pipeline_context=context,
                                  resource_guard=guard, **caption_gate_kwargs(config))
    except BaseException:
        lane.release(holder)
        raise
    return BetaWindowRenderer(inner, holder, lane)
