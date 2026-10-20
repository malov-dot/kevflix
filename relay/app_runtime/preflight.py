"""Startup preflight and GPU-first settings migration (v2 corrections 1 and 6). Stdlib + relay.config/movie_faults.

Run by the app guard (guard.py) after the instance is registered and BEFORE the server (and therefore any job)
starts. Static checks only: paths are resolved and stat()ed and the runtime manifest JSON is read. No model, native
library, CUDA context or GPU probe is loaded here; GPU model availability is checked by the child at actual load.

migrate_gpu_first(profile)   persists relay.config.gpu_first_migration on the user's settings file: ONLY the
                             app-owned CPU-only movie keys (device, actor_backend, actor_beta.demucs_device and the old
                             6144 gpu_lease.budget_mib default) change. Atomic read-modify-write, retried when the
                             file changes meanwhile, private ACL kept. Never creates settings, never resets anything
                             from config.example.json, never touches Plex/account/media/UI/startup/personal data.
preflight(...)               raises MovieConfigurationError (fault_kind 'fatal': retrying unchanged configuration
                             cannot repair it) when
  * protected_sem_path   a runtime/app/profile/output/cache root or a configured model/manifest path (including the
                         model/library the manifest names) lies under a protected Sem root: the DEFAULT_SEM_ROOTS of
                         actor-tts/runtime-adapter/approved.py (C:/Sem-Voice, C:/Sem-Trials, C:/Sem-Data; component-
                         wise, never a substring) plus actor_beta.vox.sem_roots. Checked in every mode, because the
                         Vox host refuses every child command line that names such a path.
  * missing_model / missing_runtime_manifest / invalid_runtime_manifest / missing_native_runtime /
    cuda_backend_not_configured   only when the server will run movie jobs (not setup-only).
                         With translation_engine "qwen3" the M2M100 folder is not required; the user-supplied
                         translation_llm_model folder must instead hold config.json, the tokenizer, generation_config.json
                         and safetensors shards of a plausible Qwen3-8B size (missing_model otherwise; static, no load).
  * ct2_runtime_missing / ct2_runtime_version_unsupported / ct2_runtime_metadata_invalid / analysis_python_unresolved
                         (jobs only) the CTranslate2 distribution metadata of the SELECTED ANALYSIS CHILD interpreter
                         (the Python AnalysisService launches: an existing configured `python`, else the server's own
                         interpreter; the bootstrap interpreter running this preflight may differ) is outside
                         relay/ct2_policy.py. Static dist-info reading only: that interpreter is never launched and no
                         CTranslate2/model is imported. 4.7.1 is the pinned route; an existing 4.6.0 is allowed only
                         as the explicit 'legacy FP16 route' and is never reported as upgraded. The inspection is
                         recorded under 'ct2_runtime' in startup-preflight.json and is NOT an inference-ready proof.
"""
from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path

from .. import ct2_policy
from ..config import Config, gpu_first_migration
from ..movie_faults import MovieConfigurationError

APPROVED = Path('actor-tts') / 'runtime-adapter' / 'approved.py'
# Same policy as approved.DEFAULT_SEM_ROOTS; the bundled approved.py value is read and united with this copy, so the
# preflight never protects fewer roots than the Vox host refuses.
DEFAULT_SEM_ROOTS = ('C:/Sem-Voice', 'C:/Sem-Trials', 'C:/Sem-Data')
OUTPUT_KEYS = ('state_dir', 'cache_dir', 'publish_dir', 'actor_bank_dir')
RUNTIME_KEYS = ('python', 'ffmpeg', 'ffprobe')
MODEL_KEYS = ('asr_model', 'translation_model', 'translation_ct2_model', 'translation_llm_model', 'audio_event_model', 'diarization_model',
              'speaker_onnx_model', 'separation_model_dir', 'actor_runtime_manifest', 'cinematic_checkpoint',
              'cinematic_code_root', 'overlap_checkpoint', 'overlap_code_root', 'overlap_dependency_root')
REQUIRED_MODELS = ('asr_model', 'translation_model', 'diarization_model')
OPTIONAL_MODELS = ('translation_ct2_model', 'audio_event_model', 'speaker_onnx_model', 'cinematic_checkpoint', 'overlap_checkpoint')
MIGRATION_RECORD = 'gpu-first-migration.json'
PREFLIGHT_RECORD = 'startup-preflight.json'


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


# ---- GPU-first migration ------------------------------------------------------------------------------------------
def migrate_gpu_first(profile, *, acl_from=None, attempts=5):
    """Persist the GPU-first conversion of the app-owned CPU-only movie keys. Returns a record (changes may be [])."""
    path = Path(profile.config)
    if not path.is_file():
        return {'changes': [], 'note': 'no settings file'}
    for _ in range(attempts):
        before = path.stat()
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
        changes = gpu_first_migration(raw)
        if not changes:
            return {'changes': []}
        tmp = path.with_name('.kevflix-gpu-first-%d-%d.tmp' % (os.getpid(), time.monotonic_ns()))
        try:
            with tmp.open('x', encoding='utf-8') as stream:
                if acl_from is not None:
                    acl_from(path, tmp)              # private ACL before any secret is written
                json.dump(raw, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            now = path.stat()
            if (now.st_mtime_ns, now.st_size) != (before.st_mtime_ns, before.st_size):
                continue                             # a concurrent settings save: recompute from the new file
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        record = {'migrated_utc': utc(), 'file': str(path), 'changes': changes,
                  'note': 'only app-owned CPU-only movie keys were changed'}
        note = Path(profile.runtime) / MIGRATION_RECORD
        try:
            history = json.loads(note.read_text(encoding='utf-8')).get('history', []) if note.is_file() else []
        except (OSError, ValueError, AttributeError):
            history = []
        _atomic_json(note, {'latest': record, 'history': (history + [record])[-20:]})
        return record
    # Not fatal: Config applies the same conversion in memory, so the server still runs GPU-first.
    return {'changes': [], 'error': 'the settings file kept changing; the GPU-first migration is applied in memory '
                                    'and retried at the next start'}


# ---- protected Sem roots ------------------------------------------------------------------------------------------
def _approved_literals(install_root, names):
    path = Path(install_root) / APPROVED
    if not path.is_file():
        return {}
    out = {}
    for node in ast.parse(path.read_text(encoding='utf-8'), str(path)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def protected_roots(install_root, data):
    roots = list(DEFAULT_SEM_ROOTS)
    roots += [str(r) for r in _approved_literals(install_root, {'DEFAULT_SEM_ROOTS'}).get('DEFAULT_SEM_ROOTS', ())]
    vox = ((data.get('actor_beta') or {}).get('vox') or {}) if isinstance(data.get('actor_beta'), dict) else {}
    roots += [str(r) for r in (vox.get('sem_roots') or ()) if r]
    unique = []
    for root in roots:
        if _parts(root) not in [_parts(r) for r in unique]:
            unique.append(root)
    return unique


def _parts(path):
    """Resolved (junctions/links followed, as approved._parts does) lower-case components; an unresolvable path
    falls back to its absolute form so it is still checked."""
    p = Path(os.path.expandvars(str(path))).expanduser()
    try:
        p = p.resolve()
    except (OSError, RuntimeError, ValueError):
        p = p.absolute()
    return [part.lower() for part in p.parts]


def protected_root_of(path, roots):
    """The protected root `path` equals or lies under (component-wise, like approved._under_root), else None."""
    child = _parts(path)
    for root in roots:
        parent = _parts(root)
        if len(child) >= len(parent) and child[:len(parent)] == parent:
            return root
    return None


def _path_like(value):
    text = str(value or '').strip()
    return bool(text) and any(sep in text for sep in ('/', '\\'))


def _read_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        return None, '%s: %s' % (type(error).__name__, error)
    if not isinstance(manifest, dict):
        return None, 'not a JSON object'
    return manifest, None


def checked_paths(install_root, profile, data, *, child_python=None):
    """[(label, path)] of every root and configured model/manifest path the preflight protects."""
    out = [('install_root', install_root), ('profile_root', profile.root), ('settings_file', profile.config)]
    if child_python:
        out.append(('server_python', child_python))
    out += [(key, data.get(key)) for key in OUTPUT_KEYS + RUNTIME_KEYS + MODEL_KEYS]
    beta = data.get('actor_beta') if isinstance(data.get('actor_beta'), dict) else {}
    vox = beta.get('vox') if isinstance(beta.get('vox'), dict) else {}
    out.append(('actor_beta.vox.python', vox.get('python')))
    out.append(('actor_beta.vox.gpu_lease.client', (vox.get('gpu_lease') or {}).get('client')))
    manifest_path = data.get('actor_runtime_manifest')
    if manifest_path and Path(manifest_path).is_file():
        manifest, _ = _read_manifest(manifest_path)
        if manifest:
            out.append(('actor_runtime_manifest.model', manifest.get('model')))
            out.append(('actor_runtime_manifest.library', manifest.get('library')))
            out += [('actor_runtime_manifest.dependency_directories', d)
                    for d in manifest.get('dependency_directories') or ()]
    return [(label, str(path)) for label, path in out if _path_like(path) or label in ('install_root', 'profile_root')]


# ---- preflight ----------------------------------------------------------------------------------------------------
def _exists(path, *, folder=None):
    p = Path(str(path))
    return p.is_dir() if folder else (p.is_file() if folder is False else p.exists())


def preflight(install_root, profile, *, setup_only=False, child_python=None):
    """Validate the settings (Config) and the startup paths. Returns a record; raises MovieConfigurationError."""
    try:
        config = Config(profile.config)
    except (ValueError, KeyError, TypeError) as error:
        raise MovieConfigurationError('invalid_settings', 'KevFLIX settings %s are invalid: %s'
                                      % (profile.config, error)) from error
    data = config.data
    roots = protected_roots(install_root, data)
    record = {'at_utc': utc(), 'protected_roots': roots, 'setup_only': bool(setup_only or data.get('setup_only')),
              'checked': [], 'assets': 'deferred (setup-only)', 'ct2_runtime': 'deferred (setup-only)'}
    violations = []
    for label, path in checked_paths(install_root, profile, data, child_python=child_python):
        root = protected_root_of(path, roots)
        record['checked'].append(label)
        if root:
            violations.append('%s = %s (under %s)' % (label, path, root))
    if violations:
        raise MovieConfigurationError(
            'protected_sem_path',
            'KevFLIX did not start: %s. The app, profile, outputs, caches, runtime and models must not lie under a '
            'protected Sem folder (%s); Vox refuses every child command line that names one. Install/provision them '
            'in a normal location (for example %%LOCALAPPDATA%%\\KevFLIX).' % ('; '.join(violations), ', '.join(roots)))
    if record['setup_only']:
        return record                                # pairing/settings only: no job can start in this run

    missing = []
    # translation_engine 'qwen3' replaces M2M100: its own folder is checked instead (never both). The default
    # 'm2m100' (also when the key is absent from the settings file) is checked exactly as before.
    qwen3 = data.get('translation_engine') == 'qwen3'
    for key in REQUIRED_MODELS:
        if qwen3 and key == 'translation_model':
            continue
        if not data.get(key) or not _exists(data[key]):
            missing.append('%s = %s' % (key, data.get(key) or '(not set)'))
    for key in OPTIONAL_MODELS:
        if qwen3 and key == 'translation_ct2_model':
            continue
        if data.get(key) and not _exists(data[key]):
            missing.append('%s = %s' % (key, data[key]))
    if qwen3:
        from ..language import qwen3_model_report     # light: no torch/transformers import, no weights read
        quantization = data.get('translation_quantization') or 'none'      # validated by Config: 'none' | 'int8'
        report = qwen3_model_report(data.get('translation_llm_model'), quantization)
        if quantization == 'int8':
            weights_note = ('about %.1f GB of resident weights on the GPU (int8 weight-only decoder linears, bf16 '
                            'embeddings/norms/lm_head; the bf16 checkpoint is quantized while it is loaded)'
                            % (report.get('resident_weights_bytes', 0) / 1e9))
        else:
            weights_note = 'about 16.4 GB of weights on the GPU'
        record['translation'] = {
            'engine': 'qwen3', 'quantization': quantization, 'model': data.get('translation_llm_model'),
            'files_ok': report['ok'],
            'problems': report['problems'], 'weights_bytes': report['weights_bytes'],
            'required_vram_mib': report['footprint_mib'],
            'vram_note': weights_note + '; the movie lease free-VRAM floor (reserve_mib >= 3072 MiB) '
                         'still applies on top of the worker budget',
            'inference_ready': False,
            'evidence': 'static file checks of the user-supplied folder only; nothing was downloaded or loaded'}
        if not report['ok']:
            missing.append('translation_llm_model = %s (%s; the Qwen3-8B folder is supplied by you and is never '
                           'downloaded by KevFLIX: see docs/INSTALL.md)'
                           % (data.get('translation_llm_model') or '(not set)', '; '.join(report['problems'])))
    beta = data.get('actor_beta') if isinstance(data.get('actor_beta'), dict) else {}
    beta_on = beta.get('enabled') is True
    if beta_on and (not data.get('separation_model_dir') or not _exists(data['separation_model_dir'], folder=True)):
        missing.append('separation_model_dir = %s' % (data.get('separation_model_dir') or '(not set)'))
    if missing:
        raise MovieConfigurationError('missing_model', 'Configured model files are missing: %s. Provision the models '
                                      '(Setup-Worker.ps1); this is not retried.' % '; '.join(missing))

    cuda = []
    if data.get('device') != 'cuda':
        cuda.append('device = %r (needs "cuda")' % data.get('device'))
    if data.get('actor_backend') != 'cuda':
        cuda.append('actor_backend = %r (needs "cuda")' % data.get('actor_backend'))
    if beta_on and beta.get('demucs_device') != 'cuda':
        cuda.append('actor_beta.demucs_device = %r (needs "cuda")' % beta.get('demucs_device'))
    if beta_on and beta.get('first_pass_admission') != 'movie-lease':
        cuda.append('actor_beta.first_pass_admission = %r (needs "movie-lease")' % beta.get('first_pass_admission'))
    if cuda:
        raise MovieConfigurationError('cuda_backend_not_configured', 'The GPU-first movie path is not configured: %s.'
                                      % '; '.join(cuda))

    record['ct2_runtime'] = _check_ct2_runtime(install_root, config, child_python)
    if data.get('speech_engine') == 'actor':
        _check_actor_runtime(install_root, data, beta if beta_on else {})
    record['assets'] = 'present (existence only; identities and GPU load are checked by the children)'
    return record


def analysis_child_python(install_root, config, server_python):
    """(interpreter, source) of the Python that will run relay/movie_pipeline/analysis_child.py.

    AnalysisService launches Config['python'] as resolved INSIDE the server process (cwd = install root): an existing
    configured interpreter, else the server's own interpreter (server_python, the guard's child_python). Config in
    this preflight process falls back to the bootstrap interpreter instead, so that fallback is never inspected."""
    value = str(config.python_setting or '').strip()
    if value:
        path = Path(os.path.expandvars(value)).expanduser()
        if not path.is_absolute():
            path = Path(install_root) / path
        if path.is_file():
            return str(path), 'configured python'
    if server_python:
        source = 'server interpreter (configured python %s not found)' % value if value else \
                 'server interpreter (python not configured)'
        return str(server_python), source
    raise MovieConfigurationError('analysis_python_unresolved', 'KevFLIX did not start: the analysis child interpreter '
                                  'cannot be determined (python is not configured and no server interpreter was given)')


def _check_ct2_runtime(install_root, config, server_python):
    """Static CT2 dependency-policy check of the selected analysis child interpreter (dist-info metadata only)."""
    python, source = analysis_child_python(install_root, config, server_python)
    try:
        inspected = ct2_policy.require_runtime(python, allow_legacy_46=True)
    except MovieConfigurationError as error:
        raise MovieConfigurationError(
            error.fault_code, 'KevFLIX did not start: the analysis child interpreter %s (%s) fails CTranslate2 policy '
            '%s: %s. Re-run Setup-Worker.ps1 (requirements pin ctranslate2==%s); this is not retried.'
            % (python, source, ct2_policy.POLICY_ID, error.detail, ct2_policy.RECOMMENDED_CT2_VERSION)) from error
    legacy = bool(inspected.get('legacy'))
    out = {**inspected, 'child': 'relay/movie_pipeline/analysis_child.py', 'interpreter_source': source,
           'policy_id': ct2_policy.POLICY_ID, 'compute_type': ct2_policy.CUDA_COMPUTE_TYPE, 'legacy': legacy,
           'route': 'legacy FP16 route' if legacy else 'pinned %s FP16 route' % ct2_policy.RECOMMENDED_CT2_VERSION,
           'inference_ready': False,
           'evidence': 'static distribution metadata of the selected child interpreter only; the interpreter was '
                       'not launched and no model was loaded. Not an inference-ready proof.'}
    if legacy:
        out.update(upgraded=False, note='CTranslate2 %s retained under explicit CUDA float16; it was not upgraded'
                                        % inspected.get('version'))
    return out


def _check_actor_runtime(install_root, data, beta):
    manifest_path = data.get('actor_runtime_manifest')
    if not manifest_path:
        raise MovieConfigurationError('missing_runtime_manifest', 'actor_runtime_manifest is not configured; run '
                                      'scripts/provision_vox.py (Setup-Worker.ps1 -VoxModelSource/-VoxRuntimeSource)')
    if not Path(manifest_path).is_file():
        raise MovieConfigurationError('missing_runtime_manifest', 'actor_runtime_manifest is missing: %s' % manifest_path)
    manifest, error = _read_manifest(manifest_path)
    if manifest is None or not manifest.get('model') or not manifest.get('library'):
        raise MovieConfigurationError('invalid_runtime_manifest', 'actor_runtime_manifest %s is unreadable or names no '
                                      'model/library: %s' % (manifest_path, error or 'model/library missing'))
    if not Path(manifest['model']).is_file():
        raise MovieConfigurationError('missing_model', 'the Vox model named by %s is missing: %s'
                                      % (manifest_path, manifest['model']))
    library = Path(manifest['library'])
    if not library.is_file():
        raise MovieConfigurationError('missing_native_runtime', 'the Vox native library named by %s is missing: %s'
                                      % (manifest_path, library))
    if beta.get('actor_engine', 'vox') == 'vox' and beta:
        artifacts = _approved_literals(install_root, {'RUNTIME_ARTIFACTS_SHA256'}).get('RUNTIME_ARTIFACTS_SHA256')
        if not isinstance(artifacts, dict) or not artifacts:
            raise MovieConfigurationError('vox_adapter_missing', '%s is missing or defines no runtime artifacts'
                                          % (Path(install_root) / APPROVED))
        absent = sorted(name for name in artifacts if not (library.parent / name).is_file())
        if absent:
            raise MovieConfigurationError('missing_native_runtime', 'the Vox CUDA/native runtime in %s lacks %s'
                                          % (library.parent, ', '.join(absent)))
        client = ((beta.get('vox') or {}).get('gpu_lease') or {}).get('client')
        if not client or not Path(client).is_file():
            raise MovieConfigurationError('cuda_backend_not_configured', 'actor_beta.vox.gpu_lease.client is missing: %s'
                                          % (client or '(not set)'))


def write_record(profile, record):
    try:
        _atomic_json(Path(profile.runtime) / PREFLIGHT_RECORD, record)
    except OSError:
        pass
