"""Owned warm GPU analysis worker (relay.movie_pipeline, V2-GPU). Source only; nothing here was executed.

    python -I -B <app>/relay/movie_pipeline/analysis_child.py

Started ONLY by analysis_service.AnalysisService through lease_source(...).spawn(), i.e. inside the movie GPU lease's
kill-on-close Job, beside the resident Demucs and the Vox engines. The HTTP server never loads these models.

Models (Codex gpu_models, constructed exactly once per child):
    LanguageTools(config, device='cuda')
    SpeakerAnalyzer(config, cinematic=False, device='cuda', cache=<AnalysisCache>)
SpeakerAnalyzer caches raw diarization and embeddings INTERNALLY through the AnalysisCache it is given. Raw
LanguageTools.speech()/english() results are cached here through an external AnalysisCache whose namespace carries
the model bytes, precision, device, source policy and the shared CT2 policy identity (relay.ct2_policy POLICY_ID, the
chosen compute type and this interpreter's installed ctranslate2 version). refine_utterances is never cached as a whole (it binds known
actors); its embeddings hit the analyzer's own cache. text_language is cheap CPU text detection and is not cached.

Translation: settings translation_engine 'm2m100' (default) or 'qwen3'. 'english' translates one text and 'english_batch'
translates {"rows": [{"text", "language"}, ...]} (up to MAX_BATCH_ROWS) and answers one {"english","basis"} or
{"failure": <fault record>} per row; a failing row never fails its neighbours. Rows are cached one by one under the
engine's own operation name (m2m100-english-v1 / qwen3-english-v1); the Qwen3 namespace carries the content identity of
the Qwen3 folder and the whole prompt/decoding policy. Qwen3-8B loads on its first translation request (not at READY),
only after the allocator cap of this process was checked to be able to hold it. Settings translation_quantization 'none'
(default, bf16 weights) or 'int8' (weight-only INT8 decoder linears, ~9.4 GB resident instead of 16.4 GB) selects the
weights; it is part of the Qwen3 cache namespace and of the VRAM need checked above.

Protocol: one JSON object per line, stdin -> stdout. Nothing else is printed to stdout (library prints go to stderr).
No pickle, no sockets, no listener (relay.demucs_bed._block_network audits socket use).
  in : {"id": 0, "op": "init", "protocol", "config_file", "io_dir", "cache_root", "load_deadline_s",
        "lease": {"cancel_event_name", "sem_mode"}, "budget": {"budget_mib", "other_residents_mib"}}
  out: {"id": 0, "result": {"ready": true, ...}}     or {"id": 0, "failure": <fault record>} then exit 2 / 75
  in : {"id": n, "op": <op>, "deadline_s", "args": {...}}   or ... "args_file": <ref>
  out: {"id": n, "started": true}  then  {"id": n, "result": ...} | {"id": n, "result_file": <ref>}
                                         | {"id": n, "failure": {"kind", "code", "detail"}}
       every final reply also carries "stats" (cache counters) and "devices" (current device_report()s)
  in : {"op": "cancel", "target": n}   a queued request n is skipped (failure kind 'cancelled'); a running one finishes
                                       and its late reply is discarded by the parent
  in : {"op": "shutdown"} or EOF on stdin -> exit 0
A <ref> is {"path", "sha256", "bytes"}. Arguments may contain {"__file__": <ref>} (an immutable per-request audio
copy the parent staged inside io_dir, verified by location, sha256 and size before use) and {"__array__": <ref> +
{"dtype": "float32", "shape": [n]}} (an .npy inside io_dir whose declared shape and NPY header are validated BEFORE
any allocation; never np.load/pickle). Lines, argument/result files, arrays and audio are bounded.

Guard: relay.first_pass_resident.LeaseGuard (lease cancel event every 0.2 s, verified Sem absence in sem_mode
'absent', RAM bounds; exit 75 = resource). Its wall deadline is armed for the model load (load_deadline_s) and for
every running request (deadline_s), so hung or abandoned work cannot hold the GPU even with no parent watching
(exit 76 = fatal deadline; 77 = guard failure). The lease watcher also ends this process on preemption.
Budget: the view's budget_mib minus the other residents' measured MiB caps the PyTorch allocator of this process
(torch.cuda.set_per_process_memory_fraction). Unknown residents are never treated as zero: the cap then uses the
budget alone and the lease's 3072 MiB free floor stays the hard guard.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import stat
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent              # <app>/relay/movie_pipeline
APP = HERE.parent.parent                            # <app>
PROTOCOL = 'kevflix-analysis-rpc-v1'
EXIT_OK, EXIT_REFUSED, EXIT_RESOURCE = 0, 2, 75
MAX_LINE_CHARS = 1 << 20                            # one protocol line (either direction)
INLINE_MAX = 512 * 1024                             # larger args/results travel as owned JSON files
MAX_ARGS_BYTES = 256 << 20
MAX_RESULT_BYTES = 256 << 20
MAX_AUDIO_BYTES = 1 << 30
MAX_ARRAY_BYTES = 512 << 20
NPY_HEADER_SLACK = 4096
OPS = frozenset({'text_language', 'speech', 'english', 'english_batch', 'analyze', 'embed', 'refine_utterances',
                 'device_report', 'generic_vad', 'generic_synthesize', 'language_id', 'audio_events'})
MAX_BATCH_ROWS = 256                                # rows of one english_batch request (the client sends fewer)
# Cache operation names of translations. They differ per engine on top of the namespace (which already carries the
# translator's content identity and policy), so an M2M100 entry can never answer a Qwen3 request or the reverse.
M2M100_OPERATION = 'm2m100-english-v1'
QWEN3_OPERATION = 'qwen3-english-v1'
_MISS = object()


class _CacheMiss(Exception):
    """Raised by a probe compute() so AnalysisCache.get_or_compute reports 'not cached' without storing anything."""


MIB = 1024 * 1024


# ---- helpers shared with the parent (stdlib only) ------------------------------------------------------------------
def sha256_file(path, limit):
    """(sha256 hex, size) of one regular file; ValueError when it is not regular or exceeds limit bytes."""
    digest, size = hashlib.sha256(), 0
    with open(path, 'rb') as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError('not a regular file: %s' % path)
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            size += len(block)
            if size > limit:
                raise ValueError('file exceeds its %d byte bound: %s' % (limit, path))
            digest.update(block)
    return digest.hexdigest(), size


def directly_inside(path, root):
    """True when path is a direct child of root (no traversal, no nested folders)."""
    try:
        p, r = Path(path).resolve(), Path(root).resolve()
    except (OSError, RuntimeError):
        return False
    return os.path.normcase(str(p.parent)) == os.path.normcase(str(r))


def verify_ref(ref, limit, *, root=None):
    """Validate a {"path","sha256","bytes"} reference against the file's actual bytes. Returns the resolved Path."""
    if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not isinstance(ref.get('sha256'), str):
        raise ValueError('malformed file reference')
    path = Path(ref['path'])
    if not path.is_absolute():
        raise ValueError('file reference must be absolute')
    if root is not None and not directly_inside(path, root):
        raise ValueError('file reference outside the owned exchange folder')
    digest, size = sha256_file(path, limit)
    if digest != ref['sha256'] or size != int(ref.get('bytes', -1)):
        raise ValueError('file reference changed after it was written: %s' % path)
    return path.resolve()


def write_owned_json(directory, prefix, value, limit):
    """Atomic owned JSON file in directory; returns its reference."""
    text = json.dumps(value, allow_nan=True, separators=(',', ':'))
    data = text.encode('utf-8')
    if len(data) > limit:
        raise ValueError('payload exceeds its %d byte bound' % limit)
    path = Path(directory) / ('%s-%s.json' % (prefix, uuid.uuid4().hex))
    temporary = path.with_suffix('.tmp')
    with open(temporary, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def stage_copy(source, directory, limit):
    """Immutable owned copy of one regular file inside directory (hashed while copied, bounded, atomic); returns its
    reference. The caller's original may change afterwards without affecting the request."""
    source = Path(source)
    suffix = ''.join(ch for ch in source.suffix[:8] if ch.isalnum() or ch == '.') or '.bin'
    path = Path(directory) / ('f-%s%s' % (uuid.uuid4().hex, suffix))
    temporary = path.with_name(path.name + '.tmp')
    digest, size = hashlib.sha256(), 0
    try:
        with open(source, 'rb') as reader:
            if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                raise ValueError('not a regular file: %s' % source)
            with open(temporary, 'wb') as writer:
                while True:
                    block = reader.read(1 << 20)
                    if not block:
                        break
                    size += len(block)
                    if size > limit:
                        raise ValueError('file exceeds its %d byte bound: %s' % (limit, source))
                    digest.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
        os.replace(temporary, path)
    except BaseException:
        unlink_quiet(temporary)
        raise
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': size}


def read_owned_json(ref, directory, limit):
    path = verify_ref(ref, limit, root=directory)
    return json.loads(path.read_text(encoding='utf-8'))


def unlink_quiet(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


# ---- child side ----------------------------------------------------------------------------------------------------
class _Out:
    def __init__(self, stream):
        self.stream, self.lock = stream, threading.Lock()

    def send(self, value):
        text = json.dumps(value, allow_nan=True, separators=(',', ':'), default=str)
        with self.lock:
            self.stream.write(text + '\n')
            self.stream.flush()


def _plain(value):
    """numpy/tuple-free JSON value (NaN/inf kept: json round-trips them; callers check finiteness themselves)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    module = type(value).__module__
    if module == 'numpy' or module.startswith('numpy.'):
        convert = getattr(value, 'tolist', None) or getattr(value, 'item', None)
        return _plain(convert())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError('analysis result is not JSON representable: %s' % type(value).__name__)


def _record(error):
    """Fault record of any failure in this child: explicit metadata first, then CUDA/host memory as a real resource
    hold, everything else fatal (configuration/path/model/CUDA/runtime). Never silently retried."""
    from relay.movie_faults import fault_record
    record = fault_record(error)
    if record is not None:
        return {'kind': record['kind'], 'code': record['code'][:120], 'detail': record['detail'][:4000]}
    name, text = type(error).__name__, str(error)
    lower = text.lower()
    if isinstance(error, MemoryError):
        return {'kind': 'resource', 'code': 'host_memory_exhausted', 'detail': ('%s: %s' % (name, text))[:4000]}
    if 'out of memory' in lower and ('cuda' in lower or name == 'OutOfMemoryError'):
        return {'kind': 'resource', 'code': 'gpu_out_of_memory', 'detail': ('%s: %s' % (name, text))[:4000]}
    return {'kind': 'fatal', 'code': ('analysis_%s' % name)[:120], 'detail': ('%s: %s' % (name, text))[:4000]}


def _identity_memo(cache_root):
    path = Path(cache_root) / 'model-identity-memo.json'
    try:
        memo = json.loads(path.read_text(encoding='utf-8'))
        return path, memo if isinstance(memo, dict) else {}
    except (OSError, ValueError):
        return path, {}


def _signature(path):
    info = os.stat(path)
    return [os.path.normcase(str(Path(path).resolve())), int(info.st_dev), int(info.st_ino), int(info.st_size),
            int(info.st_mtime_ns), int(info.st_ctime_ns)]


def _valid_identity(identity, size):
    if not isinstance(identity, dict):
        return False                                  # fresh or malformed memo entry: a miss, never an error
    digest = identity.get('sha256')
    return (identity.get('kind') == 'file' and isinstance(digest, str) and len(digest) == 64
            and all(c in '0123456789abcdef' for c in digest) and identity.get('size_bytes') == size)


def _tree_identity(root, memo, file_identity, used, skip_dirs=()):
    """Content identity of a model folder: the sha256 of every regular file's bytes (relay.analysis_cache.file_identity)
    under sorted relative names. The identity is content, never stat data. skip_dirs names folders (e.g. '.cache', the
    huggingface-hub download bookkeeping) whose files are not part of the model and are neither hashed nor listed.

    The memo only avoids re-hashing an unchanged multi-GB file on every child start. An entry is reused only while the
    full changed-file signature (path, device, file id, size, mtime, ctime) is identical and its digest/size are
    well-formed for that size. A miss is hashed with a before/after signature stability check: a file that changes
    while it is hashed refuses (fatal), so a replaced model can never key a stale raw cache. The memo is pruned to the
    files this start actually identified (`used`)."""
    from relay.movie_faults import MovieConfigurationError
    root = Path(root)
    if root.is_file():
        files = [root]
        base = root.parent
    else:
        files = sorted(p for p in root.rglob('*') if p.is_file()
                       and not any(part in skip_dirs for part in p.relative_to(root).parts[:-1]))
        base = root
    entries = {}
    for path in files:
        before = _signature(path)
        key = '|'.join(str(x) for x in before)
        entry = memo.get(key)
        identity = entry.get('identity') if isinstance(entry, dict) and entry.get('signature') == before else None
        if not _valid_identity(identity, before[3]):
            identity = file_identity(path)
            after = _signature(path)
            if after != before or not _valid_identity(identity, before[3]):
                raise MovieConfigurationError('model_asset_changed',
                                              'model file changed while it was identified: %s' % path)
        used[key] = {'signature': before, 'identity': identity}
        entries[path.relative_to(base).as_posix()] = identity['sha256']
    if not entries:
        raise FileNotFoundError('model folder has no files: %s' % root)
    encoded = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _library_versions():
    """Installed versions of the inference libraries the language cache depends on (None when not installed)."""
    from importlib import metadata
    versions = {}
    for name in ('faster-whisper', 'ctranslate2', 'transformers', 'tokenizers', 'sentencepiece', 'torch', 'numpy',
                 'piper-tts', 'onnxruntime-gpu', 'onnxruntime', 'soundfile'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _ct2_policy_identity(language, chosen, libraries):
    """Shared CT2 policy identity of THIS child (relay.ct2_policy, Codex): policy id + identity hash, the compute type
    this worker actually chose, and the ctranslate2 version from this interpreter's own installed metadata
    (LanguageTools.policy_metadata()['runtime'], else importlib.metadata). Part of the raw language-cache namespace and
    of every ready/device report. Refuses when the chosen type is not the policy type or no version is known."""
    from relay.ct2_policy import CUDA_COMPUTE_TYPE, POLICY_ID
    from relay.movie_faults import MovieConfigurationError
    metadata = language.policy_metadata()
    policy, runtime = metadata.get('policy') or {}, metadata.get('runtime') or {}
    version, source = runtime.get('version'), 'selected-child-distribution-metadata'
    if not version:
        version, source = libraries.get('ctranslate2'), 'importlib.metadata'
    if policy.get('id') != POLICY_ID or chosen != CUDA_COMPUTE_TYPE or not version:
        raise MovieConfigurationError('ct2_policy_identity_invalid',
                                      'analysis worker CT2 policy %r chose %r (required %s %s); ctranslate2 %r'
                                      % (policy.get('id'), chosen, POLICY_ID, CUDA_COMPUTE_TYPE, version))
    return {'id': POLICY_ID, 'identity': policy.get('identity'), 'compute_type': chosen,
            'ctranslate2_version': str(version), 'version_source': source, 'legacy': bool(runtime.get('legacy')),
            'compatibility': runtime.get('compatibility')}


def _save_memo(path, memo):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.%s.tmp' % uuid.uuid4().hex[:8])
        temporary.write_text(json.dumps(memo, sort_keys=True), encoding='utf-8')
        os.replace(temporary, path)
    except OSError:
        pass                                          # memo only; identities are recomputed next time


def translation_identity(settings, generic, qwen3, memo, file_identity, used, ct2_precision):
    """The 'translation' part of the language-cache namespace (what a cached translation depends on).

    m2m100 (default): the M2M100 folder's content identity and precision, exactly as before. qwen3: the content identity
    of the Qwen3 folder (hashed once per file, then memoised; huggingface-hub's .cache bookkeeping excluded) plus the whole
    prompt/decoding policy, so a different model or prompt can never be served M2M100 or older Qwen3 translations.
    generic: Whisper translates itself. ct2_precision() is only called for the CT2 M2M100 variant.

    qwen3 + settings translation_quantization 'int8' (weight-only INT8 decoder linears): the precision, the policy (its
    basis and quantization scheme) and an explicit 'quantization' key all differ from bf16, so int8 and bf16 translations
    can never share a namespace. 'none' (default) yields exactly the identity that existed before the key."""
    if qwen3:
        from relay.language import qwen3_policy, translation_quantization_of
        quantization = translation_quantization_of(settings)
        identity = {'engine': 'qwen3', 'precision': 'torch.bfloat16', 'policy': qwen3_policy(quantization),
                    'model': _tree_identity(settings['translation_llm_model'], memo, file_identity, used,
                                            skip_dirs=('.cache',))}
        if quantization != 'none':
            identity['precision'] = 'torch.bfloat16 activations, int8 weight-only decoder linears'
            identity['quantization'] = quantization
        return identity
    translation = {'mode': 'Whisper direct to English'} if generic else {
        'tokenizer_and_model': _tree_identity(settings['translation_model'], memo, file_identity, used),
        'ct2': None, 'precision': None}
    if not generic and settings.get('translation_ct2_model'):
        translation.update(ct2=_tree_identity(settings['translation_ct2_model'], memo, file_identity, used),
                           precision=ct2_precision())
    elif not generic:
        translation['precision'] = 'torch.float16'
    return translation


def qwen3_admission(report, model_dir, quantization, memory, budget_mib, others):
    """The Qwen3 VRAM admission check of Worker.__init__ (a function so it can be tested without CUDA).

    The torch allocator cap is fixed for the process. Qwen3-8B bf16 needs ~16 GB of it (17,159 MiB: the weights plus a fixed
    headroom); the int8 weight-only mode needs the exact int8 layout (~9.0 GB) plus the headroom computed from the model's
    shapes (relay.language.qwen3_int8_budget: KV cache, prefill activations, one dequantized matrix; 11,162 MiB for
    Qwen3-8B). A budget that cannot hold it is refused now (a resource hold, seconds) instead of as an OOM after a 9-16 GB
    read. `report` is qwen3_model_report(model_dir, quantization); `memory` is the worker's memory record, updated in place."""
    from relay.movie_faults import MovieConfigurationError, MovieResourceWait
    if not report['ok']:
        raise MovieConfigurationError('translation_llm_assets_invalid', 'Qwen3 model folder %s: %s' % (
            model_dir, '; '.join(report['problems'])))
    memory['qwen3_required_mib'] = report['footprint_mib']
    if quantization != 'none':                          # bf16 memory records keep their original keys
        memory['qwen3_quantization'] = quantization
        memory['qwen3_resident_weights_mib'] = report['budget']['resident_mib']
    if memory['allowed_mib'] is not None and memory['allowed_mib'] < report['footprint_mib']:
        raise MovieResourceWait('analysis_vram_budget',
                                'lease budget %s MiB minus %s MiB of residents leaves %s MiB for this worker; '
                                'Qwen3-8B %s needs about %s MiB' % (
                                    budget_mib, others, memory['allowed_mib'],
                                    'int8 weight-only' if quantization == 'int8' else 'bf16', report['footprint_mib']))
    return report['footprint_mib']


class Worker:
    def __init__(self, init):
        from relay import gpu_environment
        # Before torch/ctranslate2/pyannote are imported: Windows DLL search for this interpreter's CUDA wheels. The
        # handles stay referenced by gpu_environment for the life of this process; the list is kept for the report.
        self.dll_dirs = gpu_environment.configure_cuda_dll_search()
        from relay.analysis_cache import AnalysisCache, file_identity
        from relay.config import Config
        from relay.movie_faults import MovieConfigurationError, MovieResourceWait
        self.io_dir = Path(init['io_dir'])
        if not self.io_dir.is_dir():
            raise MovieConfigurationError('analysis_io_missing', 'owned exchange folder missing: %s' % self.io_dir)
        self.config = Config(init['config_file'])
        if self.config['speech_engine'] == 'xtranslator':
            raise MovieConfigurationError('xt_legacy_analysis_forbidden',
                                          'X-Translator uses native Qwen recognition; legacy analysis is forbidden')
        cache_root = Path(init['cache_root'])
        import torch
        from relay.language import (LanguageTools, ct2_compute_type, iso, qwen3_model_report, qwen3_policy_text,
                                    translation_quantization_of)
        self.iso, self.file_identity = iso, file_identity
        # translation_engine 'qwen3' replaces M2M100 for the Whisper-transcribe path only; generic mode translates in
        # Whisper itself and never reaches the translator. Absent key = 'm2m100' (every existing settings file).
        self.qwen3 = (self.config.data.get('speech_engine') != 'generic'
                      and self.config.data.get('translation_engine') == 'qwen3')
        # translation_quantization ('none' bf16 | 'int8' weight-only): validated by Config; only a qwen3 run reads it.
        self.quantization = translation_quantization_of(self.config.data) if self.qwen3 else 'none'
        self.english_operation = QWEN3_OPERATION if self.qwen3 else M2M100_OPERATION
        if not torch.cuda.is_available():
            raise MovieConfigurationError('cuda_unavailable',
                                          'CUDA is unavailable in the analysis worker; CPU inference is not substituted')
        index = 0
        total_mib = int(torch.cuda.get_device_properties(index).total_memory // MIB)
        budget = init.get('budget') or {}
        budget_mib, others = budget.get('budget_mib'), budget.get('other_residents_mib')
        self.memory = {'total_mib': total_mib, 'budget_mib': budget_mib, 'other_residents_mib': others,
                       'allowed_mib': None, 'torch_memory_fraction': None,
                       'scope': 'PyTorch allocator only; CTranslate2 allocations are bounded by the lease floor'}
        if budget_mib is not None:
            allowed = int(budget_mib) - (int(others) if others is not None else 0)
            if allowed <= 0:
                raise MovieResourceWait('analysis_vram_budget',
                                        'movie lease budget %s MiB leaves nothing beside %s MiB of residents'
                                        % (budget_mib, others))
            fraction = min(1.0, float(allowed) / float(max(1, total_mib)))
            torch.cuda.set_per_process_memory_fraction(fraction, index)
            self.memory.update(allowed_mib=allowed, torch_memory_fraction=round(fraction, 4))
        if self.qwen3:
            qwen3_admission(qwen3_model_report(self.config.data['translation_llm_model'], self.quantization),
                            self.config.data['translation_llm_model'], self.quantization, self.memory, budget_mib, others)
        self.language = LanguageTools(self.config, device='cuda')
        self.audio_events = None
        if self.config.data.get('audio_event_model'):
            from relay.audio_events import AudioEventClassifier
            self.audio_events = AudioEventClassifier(self.config.data['audio_event_model'], device='cuda:%d' % index)
        self.speaker_cache = AnalysisCache(cache_root, {'kevflix': PROTOCOL, 'role': 'community1-speakers'})
        settings = self.config.data
        self.generic = settings.get('speech_engine') == 'generic'
        self.speakers = None
        self.generic_speech = None
        self.generic_vad_identity = None
        if self.generic:
            from relay.speech import GenericSpeech
            from faster_whisper import vad as vad_module
            from faster_whisper.utils import get_assets_path
            self.generic_speech = GenericSpeech(self.config)
            self.generic_vad_identity = {
                'device': 'cpu', 'backend': 'faster-whisper-bundled-silero',
                'scope': 'independent speech-boundary helper on isolated vocals; no Whisper output used',
                'model': file_identity(Path(get_assets_path()) / 'silero_vad_v6.onnx'),
                'code_sha256': sha256_file(vad_module.__file__, 64 << 20)[0],
                'sample_rate': 16000, 'options': {'min_silence_duration_ms': 500, 'speech_pad_ms': 100},
            }
        else:
            from relay.speakers import SpeakerAnalyzer
            self.speakers = SpeakerAnalyzer(self.config, cinematic=False, device='cuda', cache=self.speaker_cache)
        memo_path, memo = _identity_memo(cache_root)
        used = {}
        try:
            translation = translation_identity(settings, self.generic, self.qwen3, memo, file_identity, used,
                                               lambda: ct2_compute_type(self.language.device_name))
            asr = {'model': _tree_identity(settings['asr_model'], memo, file_identity, used),
                   'precision': ct2_compute_type(self.language.device_name)}
        except OSError as error:
            raise MovieConfigurationError('language_assets_missing', str(error)) from error
        _save_memo(memo_path, used)                   # bounded: only the files this start identified
        # Exact code and library identity (computed once per worker start): a code or dependency update never reuses
        # raw outputs produced by older code. Fixed policy strings alone are not identity.
        code = {name: sha256_file(path, 64 << 20)[0] for name, path in (
            ('relay/language.py', APP / 'relay' / 'language.py'),
            ('relay/ct2_policy.py', APP / 'relay' / 'ct2_policy.py'),
            ('relay/speech.py', APP / 'relay' / 'speech.py'),
            ('relay/movie_pipeline/analysis_child.py', Path(__file__).resolve()))}
        libraries = _library_versions()
        self.ct2_policy = _ct2_policy_identity(self.language, asr['precision'], libraries)
        self.language_namespace = {
            'kevflix': PROTOCOL, 'role': 'language', 'device': self.language.device_name,
            'gpu': torch.cuda.get_device_name(index), 'code': code, 'libraries': libraries,
            'ct2_policy': self.ct2_policy, 'asr': asr, 'translation': translation,
            'generic_vad': self.generic_vad_identity,
            'generic_voice': self.generic_speech.identity() if self.generic_speech else None,
            'policy': {'asr': 'faster-whisper task explicit; word_timestamps only for transcribe; condition_on_previous_text=False',
                       'translation': qwen3_policy_text(self.quantization) if self.qwen3 else 'm2m100 beam 5 to English'}}
        self.language_cache = AnalysisCache(cache_root, self.language_namespace)
        self.torch = torch

    def ready(self, started):
        info = {'ready': True, 'protocol': PROTOCOL, 'pid': os.getpid(), 'load_seconds': round(time.monotonic() - started, 3),
                'devices': self.devices(), 'speaker_policy': self.speakers.policy_metadata() if self.speakers else None, 'memory': self.memory,
                'ct2_policy': dict(self.ct2_policy), 'cuda_dll_dirs': list(self.dll_dirs or ())}
        if self.qwen3:                                 # m2m100 ready replies keep their existing shape
            info['translation'] = {'engine': 'qwen3', 'operation': self.english_operation, 'loaded': False,
                                   'quantization': self.quantization,
                                   'note': 'Qwen3-8B loads on the first translation request, not at READY'}
        return info

    def devices(self):
        report = {'language': self.language.device_report(), 'speakers': self.speakers.device_report() if self.speakers else None,
                  'generic_vad': self.generic_vad_identity,
                  'generic_tts': self.generic_speech.identity() if self.generic_speech else None,
                  'ct2_policy': dict(self.ct2_policy), 'cpu_fallback': False}
        try:
            report['torch_allocated_mib'] = int(self.torch.cuda.max_memory_allocated() // MIB)
            report['torch_reserved_mib'] = int(self.torch.cuda.memory_reserved() // MIB)
        except Exception:  # noqa: BLE001 - telemetry only
            pass
        return report

    def stats(self):
        return {'speakers': self.speaker_cache.stats(), 'language': self.language_cache.stats()}

    # -- argument decoding ----------------------------------------------------------------------------------------
    def decode(self, value):
        if isinstance(value, dict):
            if set(value) == {'__file__'}:
                # Audio is an immutable per-request copy the parent staged inside the owned exchange folder.
                return str(verify_ref(value['__file__'], MAX_AUDIO_BYTES, root=self.io_dir))
            if set(value) == {'__array__'}:
                return self._array(value['__array__'])
            return {k: self.decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        return value

    def _array(self, ref):
        """Validate the declared shape and the NPY header BEFORE allocating; then read exactly that many samples."""
        import numpy as np
        from numpy.lib import format as npy
        if not isinstance(ref, dict):
            raise ValueError('malformed analysis array reference')
        shape = ref.get('shape')
        if (ref.get('dtype') != 'float32' or not isinstance(shape, list) or len(shape) != 1
                or not isinstance(shape[0], int) or isinstance(shape[0], bool) or shape[0] < 0
                or shape[0] * 4 > MAX_ARRAY_BYTES):
            raise ValueError('analysis array reference has an invalid declared float32 shape')
        count = shape[0]
        path = verify_ref(ref, MAX_ARRAY_BYTES + NPY_HEADER_SLACK, root=self.io_dir)
        size = path.stat().st_size
        with open(path, 'rb') as handle:
            version = npy.read_magic(handle)
            if version == (1, 0):
                header = npy.read_array_header_1_0(handle)
            elif version == (2, 0):
                header = npy.read_array_header_2_0(handle)
            else:
                raise ValueError('unsupported NPY version %s' % (version,))
            found_shape, fortran, dtype = header
            offset = handle.tell()
            if (tuple(found_shape) != (count,) or fortran or dtype != np.dtype('<f4') or dtype.hasobject
                    or offset > NPY_HEADER_SLACK or offset + count * 4 != size):
                raise ValueError('analysis array header does not match its declared float32 shape')
            array = np.fromfile(handle, dtype='<f4', count=count)
        if array.shape != (count,):
            raise ValueError('analysis array is shorter than its header')
        return array

    def _cached(self, operation, inputs, compute):
        """External cache for raw LanguageTools results. A cache-only conversion/storage refusal after a successful
        inference returns that inference uncached; inference failures propagate and are never cached."""
        box = []

        def run():
            value = _plain(compute())
            box.append(value)
            return value
        try:
            return self.language_cache.get_or_compute(operation, inputs, run)
        except (ValueError, TypeError):
            if box:
                return box[0]
            raise

    def _probe(self, operation, inputs):
        """The cached value of one translation row, or _MISS. Uses only AnalysisCache.get_or_compute: the probe's
        compute() raises, which stores nothing (its counters count that as a miss plus a compute failure)."""
        def absent():
            raise _CacheMiss()
        try:
            return self.language_cache.get_or_compute(operation, inputs, absent)
        except _CacheMiss:
            return _MISS
        except (ValueError, TypeError):
            return _MISS                              # cache refused the row: translate it, uncached

    def _store(self, operation, inputs, value):
        """Publish one finished translation row (a miss that returns the already computed value)."""
        try:
            self.language_cache.get_or_compute(operation, inputs, lambda: value)
        except (ValueError, TypeError):
            pass                                      # storage refusal never fails a computed translation

    def english_batch(self, args):
        """One result per row, in order: {'english', 'basis'} or {'failure': <fault record>}. Cached rows are answered
        from the cache and only the rest are translated, together, by LanguageTools.english_batch. A failing row is
        reported alone (never cached); a whole-call failure (model cannot load) raises."""
        rows = args.get('rows')
        if not isinstance(rows, list) or len(rows) > MAX_BATCH_ROWS:
            raise ValueError('english_batch needs a list of at most %d rows' % MAX_BATCH_ROWS)
        operation, results, misses = self.english_operation, [None] * len(rows), []
        keys, words = [None] * len(rows), {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get('text'), str):
                results[index] = {'failure': {'kind': 'recoverable', 'code': 'translation_row_invalid',
                                              'detail': 'A translation row needs string text and a language'}}
                continue
            code = self.iso(row.get('language'))
            budget = row.get('max_words')
            budget = budget if isinstance(budget, int) and not isinstance(budget, bool) and 1 <= budget <= 60 else None
            words[index] = budget
            keys[index] = {'text': row['text'], 'language': code, **({'max_words': budget} if budget else {})}
            if code == 'en' or not code or code in ('und', 'unknown'):
                misses.append(index)                  # passthrough / unresolved: decided without a model or the cache
                continue
            hit = self._probe(operation, keys[index])
            if hit is _MISS:
                misses.append(index)
            else:
                results[index] = {'english': str(hit[0]), 'basis': str(hit[1])}
        if misses:
            outcomes = self.language.english_batch([(rows[i]['text'], rows[i].get('language')) + ((words[i],) if words.get(i) else ())
                                                    for i in misses])
            for index, outcome in zip(misses, outcomes):
                if isinstance(outcome, BaseException):
                    results[index] = {'failure': _record(outcome)}
                    continue
                english, basis = outcome
                results[index] = {'english': english, 'basis': basis}
                if basis != 'already-English':
                    self._store(operation, keys[index], [english, basis])
        return results

    # -- operations -----------------------------------------------------------------------------------------------
    def run(self, op, args):
        if op == 'text_language':
            return list(self.language.text_language(args['text'], args.get('declared')))
        if op == 'speech':
            wav, language = args['wav'], args.get('language')
            vad, beam = bool(args.get('vad', True)), int(args.get('beam_size', 5))
            task = args.get('task', 'transcribe')
            return self._cached('faster-whisper-speech-v2',
                                {'audio': self.file_identity(wav), 'language': language, 'vad': vad,
                                 'beam_size': beam, 'task': task},
                                lambda: list(self.language.speech(wav, language, vad=vad, beam_size=beam, task=task)))
        if op in {'generic_vad', 'generic_synthesize'} and not self.generic:
            from relay.movie_faults import MovieConfigurationError
            raise MovieConfigurationError('generic_mode_required', 'Generic requests require the configured generic child')
        if op == 'generic_vad':
            from faster_whisper.audio import decode_audio
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            def compute_vad():
                wave = decode_audio(str(args['wav']), sampling_rate=16000)
                stamps = get_speech_timestamps(wave, VadOptions(**self.generic_vad_identity['options']), sampling_rate=16000)
                return [{'start': float(row['start']) / 16000, 'end': float(row['end']) / 16000} for row in stamps]
            regions = self._cached('generic-independent-silero-v1',
                                   {'audio': self.file_identity(args['wav']), 'vad': self.generic_vad_identity}, compute_vad)
            return {'regions': regions, 'receipt': {**self.generic_vad_identity, 'regions': len(regions),
                                                   'audio': self.file_identity(args['wav'])}}
        if op == 'generic_synthesize':
            if self.generic_speech is None:
                from relay.movie_faults import MovieConfigurationError
                raise MovieConfigurationError('generic_synthesis_disabled', 'X-Translator never falls back to Piper')
            output = self.io_dir / ('g-%s.wav' % uuid.uuid4().hex)
            try:
                receipt = self.generic_speech.synthesize(args['text'], args['voice'], output)
                digest, size = sha256_file(output, MAX_AUDIO_BYTES)
                receipt.pop('audio', None)
                return {**receipt, 'audio_ref': {'path': str(output), 'sha256': digest, 'bytes': size}}
            except BaseException:
                unlink_quiet(output)
                raise
        if op == 'language_id':
            return self.language.language_id(args['wav'], [[float(a), float(b)] for a, b in args['spans']])
        if op == 'audio_events':
            if self.audio_events is None or not self.audio_events.available:
                return [None] * len(args['spans'])        # no classifier folder: the nonverbal rules use text and pitch only
            return self.audio_events.classify_spans(args['wav'], [[float(a), float(b)] for a, b in args['spans']])
        if op == 'english':
            text, language = str(args['text']), args.get('language')
            return self._cached(self.english_operation, {'text': text, 'language': self.iso(language)},
                                lambda: list(self.language.english(text, language)))
        if op == 'english_batch':
            return self.english_batch(args)
        if op == 'analyze':
            if self.speakers is None:raise ValueError('speaker analysis is disabled in generic mode')
            return list(self.speakers.analyze(args['wav'], float(args.get('offset', 0.0))))
        if op == 'embed':
            if self.speakers is None:raise ValueError('speaker embeddings are disabled in generic mode')
            return self.speakers.embed(args['waveform'])
        if op == 'refine_utterances':
            if self.speakers is None:raise ValueError('speaker refinement is disabled in generic mode')
            return list(self.speakers.refine_utterances(
                args['waveform'], int(args['rate']), args['segments'], args['vectors'], args.get('offset', 0),
                speech_regions=args.get('speech_regions') or (), known_speakers=args.get('known_speakers'),
                whisper_no_speech=bool(args.get('whisper_no_speech', False))))
        if op == 'device_report':
            return {'devices': self.devices(), 'speaker_policy': self.speakers.policy_metadata() if self.speakers else None,
                    'memory': self.memory, 'ct2_policy': dict(self.ct2_policy)}
        raise ValueError('unknown analysis op %r' % (op,))


def _reader(stdin, requests, cancelled):
    """stdin -> request queue; cancel messages are applied at once (they never wait behind a running request)."""
    try:
        while True:
            raw = stdin.readline(MAX_LINE_CHARS + 1)
            if not raw:
                break
            if len(raw) > MAX_LINE_CHARS:
                requests.put(('violation', 'request line exceeds %d characters' % MAX_LINE_CHARS))
                return
            raw = raw.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except ValueError:
                requests.put(('violation', 'request line is not JSON'))
                return
            if not isinstance(message, dict):
                requests.put(('violation', 'request is not an object'))
                return
            if message.get('op') == 'cancel':
                cancelled.add(message.get('target'))
                continue
            if message.get('op') == 'shutdown':
                break
            requests.put(('request', message))
    except (OSError, ValueError):
        pass
    requests.put(None)


def serve(stdin, stdout):
    out = _Out(stdout)
    if HERE.name != 'movie_pipeline' or HERE.parent.name != 'relay':
        out.send({'id': 0, 'failure': {'kind': 'fatal', 'code': 'analysis_child_location',
                                       'detail': 'the analysis worker must run from relay/movie_pipeline'}})
        return EXIT_REFUSED
    sys.path.insert(0, str(APP))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
                      PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    raw = stdin.readline(MAX_LINE_CHARS + 1)
    try:
        init = json.loads(raw)
        if not isinstance(init, dict) or init.get('op') != 'init' or init.get('id') != 0 \
                or init.get('protocol') != PROTOCOL:
            raise ValueError('first message must be the init request of protocol %s' % PROTOCOL)
    except (ValueError, TypeError) as error:
        out.send({'id': 0, 'failure': {'kind': 'fatal', 'code': 'analysis_protocol_violation', 'detail': str(error)}})
        return EXIT_REFUSED
    try:
        from relay import demucs_bed
        from relay.app_runtime import lease_client
        from relay.first_pass_resident import LeaseGuard
        guard = LeaseGuard(init.get('lease') or {}, demucs_bed, lease_client)
        demucs_bed._block_network()
    except Exception as error:  # noqa: BLE001 - setup of the guard itself is configuration
        out.send({'id': 0, 'failure': {'kind': 'fatal', 'code': 'analysis_guard_setup',
                                       'detail': '%s: %s' % (type(error).__name__, error)}})
        return EXIT_REFUSED
    ready_sent = False
    try:
        with guard:                                   # admission (lease event, Sem absence, RAM) before model work
            # The guard's wall deadline bounds the load even if no parent is watching (exit 76, fatal).
            guard.arm(None, float(init.get('load_deadline_s') or 900.0))
            try:
                worker = Worker(init)
            except BaseException as error:  # noqa: BLE001
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                record = _record(error)
                out.send({'id': 0, 'failure': record})
                return EXIT_RESOURCE if record['kind'] == 'resource' else EXIT_REFUSED
            finally:
                guard.disarm()
            out.send({'id': 0, 'result': worker.ready(started)})
            ready_sent = True
            requests, cancelled = queue.Queue(), set()
            threading.Thread(target=_reader, args=(stdin, requests, cancelled), name='analysis-child-reader',
                             daemon=True).start()
            while True:
                item = requests.get()
                if item is None:
                    return EXIT_OK
                kind, message = item
                if kind == 'violation':
                    sys.stderr.write('analysis worker protocol violation: %s\n' % message)
                    return EXIT_REFUSED
                rid, op = message.get('id'), message.get('op')
                if not isinstance(rid, int) or rid <= 0:
                    sys.stderr.write('analysis worker request without a valid id\n')
                    return EXIT_REFUSED
                if rid in cancelled:
                    cancelled.discard(rid)
                    out.send({'id': rid, 'failure': {'kind': 'cancelled', 'code': 'analysis_request_cancelled',
                                                     'detail': 'cancelled before it started'}})
                    continue
                deadline = message.get('deadline_s')
                deadline = float(deadline) if isinstance(deadline, (int, float)) and deadline > 0 else 900.0
                out.send({'id': rid, 'started': True})
                # A hung or abandoned request cannot hold the GPU forever: the guard exits 76 (fatal) past it.
                guard.arm(None, deadline)
                try:
                    if op not in OPS:
                        raise ValueError('unknown analysis op %r' % (op,))
                    if 'args_file' in message:
                        args = read_owned_json(message['args_file'], worker.io_dir, MAX_ARGS_BYTES)
                    else:
                        args = message.get('args') or {}
                    if not isinstance(args, dict):
                        raise ValueError('analysis args must be an object')
                    value = _plain(worker.run(op, worker.decode(args)))
                    text = json.dumps(value, allow_nan=True, separators=(',', ':'))
                    reply = {'id': rid}
                    if len(text) > INLINE_MAX:
                        reply['result_file'] = write_owned_json(worker.io_dir, 'r-%d' % rid, value, MAX_RESULT_BYTES)
                    else:
                        reply['result'] = value
                except BaseException as error:  # noqa: BLE001 - every failure is a structured record
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    reply = {'id': rid, 'failure': _record(error)}
                finally:
                    guard.disarm()
                try:                                  # actual, current device state (lazily loaded ASR/MT included)
                    reply.update(stats=worker.stats(), devices=worker.devices())
                except Exception:  # noqa: BLE001 - telemetry only
                    pass
                cancelled.discard(rid)
                out.send(reply)
    except (demucs_bed.ResourceAbort, MemoryError) as error:  # admission: lease cancel, Sem present, RAM bound
        text = '%s: %s' % (type(error).__name__, error)
        sys.stderr.write('analysis worker yielded: %s\n' % text)
        if not ready_sent:
            out.send({'id': 0, 'failure': {'kind': 'resource', 'code': 'analysis_guard_yield', 'detail': text[:4000]}})
        return EXIT_RESOURCE
    except demucs_bed.Refused as error:                         # the guard itself could not start
        if not ready_sent:
            out.send({'id': 0, 'failure': {'kind': 'fatal', 'code': 'analysis_guard_failure',
                                           'detail': str(error)[:4000]}})
        return EXIT_REFUSED


def main():
    stdout = sys.stdout
    sys.stdout = sys.stderr                           # library prints never reach the protocol pipe
    return serve(sys.stdin, stdout)


if __name__ == '__main__':
    sys.exit(main())
