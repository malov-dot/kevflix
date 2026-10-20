"""Vox actor-engine provisioning (fix 1). Stdlib only. Run explicitly by Setup-Worker.ps1 / the installer
(scripts/provision_vox.py); never at app startup and never implicitly.

What the app's Vox path needs (W3's runtime requirements):
  * model  voxcpm2-actor-q4_k.gguf, sha256/size/quantization from actor-tts/runtime-adapter/approved.py
           (MODEL_SHA256, MODEL_BYTES, QUANTIZATION)
  * native runtime  every file in approved.py RUNTIME_ARTIFACTS_SHA256 (audiocpp.dll v0.9.1 + PCM callback, ggml*,
           CUDA 13 cublas/cublasLt/cudart, cufft 12, msvcp/vcruntime, LICENSE), each sha256-verified
  * engine python  actor_beta.vox.python, else top-level `python`, else the relay interpreter; Python 3.11 with
           numpy + soundfile (installed into .venv by Setup-Worker from requirements.txt)
  * config  top-level actor_runtime_manifest (a per-machine JSON written here), actor_beta.vox.sources_sha256
           (automatic-v1/automatic_renderer.py, reference-cache/reference_cache.py, every runtime-adapter/*.py),
           actor_beta.vox.gpu_lease {client, client_sha256, sem_mode}
DECISION (stated in docs/INSTALL.md and desktop-release.json): the 1.7 GB model and the CUDA/native DLLs are NOT
embedded in the installer. They are provisioned from a SOURCE the user/installer names (--model-source,
--runtime-source; an existing install/folder), verified, then copied into <profile>/models and <profile>/tools/
vox-native (or, with --reference-in-place, used where they are). An already configured, fully verified
actor_runtime_manifest is kept as is. A missing or mismatching artifact is an explicit error; there is no silent
fallback to piper. Sem's private folders are never used as a source and never copied.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from .paths import INSTALL_ROOT, Profile, publish_if_absent

MODEL_NAME = 'voxcpm2-actor-q4_k.gguf'
UPSTREAM = 'VoxCPM2/audio.cpp v0.9.1 with PCM callback ABI'
LIBRARY = 'audiocpp.dll'
APPROVED = Path('actor-tts') / 'runtime-adapter' / 'approved.py'
VOX_SOURCE_FILES = ('automatic-v1/automatic_renderer.py', 'reference-cache/reference_cache.py')
SEM_PRIVATE_ROOTS = ('c:\\sem-voice', 'c:\\sem-trials', 'c:\\sem-data\\profile', 'sem-5.0', 'sem-4.0',
                     'c:\\sem-migration-staging\\sem-')


class ProvisionError(Exception):
    pass


def _sha256(path, block=8 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def approved_constants(install_root=INSTALL_ROOT):
    """Literal constants of approved.py, read with ast (the file is not executed here)."""
    path = Path(install_root) / APPROVED
    if not path.is_file():
        raise ProvisionError('%s is missing from this install; the Vox adapter is not installed' % APPROVED)
    tree = ast.parse(path.read_text(encoding='utf-8'), str(path))
    wanted = {'MODEL_SHA256', 'MODEL_BYTES', 'QUANTIZATION', 'RUNTIME_ARTIFACTS_SHA256'}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in wanted:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = sorted(wanted - set(out))
    if missing:
        raise ProvisionError('approved.py does not define %s' % missing)
    return out


def _refuse_private(path):
    text = str(Path(path).resolve()).replace('/', '\\').lower()
    if any(root in text for root in SEM_PRIVATE_ROOTS):
        raise ProvisionError('refused: %s is inside a private Sem folder; KevFLIX never uses or copies Sem data' % path)


def _model_in(source):
    source = Path(source)
    return source / MODEL_NAME if source.is_dir() else source


def verify_model(path, approved):
    path = Path(path)
    if not path.is_file():
        raise ProvisionError('Vox model not found: %s (expected %s)' % (path, MODEL_NAME))
    if path.stat().st_size != int(approved['MODEL_BYTES']):
        raise ProvisionError('Vox model %s has %d bytes, expected %d' % (path, path.stat().st_size, approved['MODEL_BYTES']))
    have = _sha256(path)
    if have != approved['MODEL_SHA256']:
        raise ProvisionError('Vox model %s sha256 %s != approved %s' % (path, have, approved['MODEL_SHA256']))
    return have


def verify_runtime(folder, approved):
    folder = Path(folder)
    problems = []
    for name, digest in sorted(approved['RUNTIME_ARTIFACTS_SHA256'].items()):
        path = folder / name
        if not path.is_file():
            problems.append('missing %s' % name)
        elif _sha256(path) != digest:
            problems.append('%s sha256 mismatch' % name)
    if problems:
        raise ProvisionError('Vox native runtime in %s is not the approved one: %s' % (folder, '; '.join(problems)))
    return {name: digest for name, digest in approved['RUNTIME_ARTIFACTS_SHA256'].items()}


def _copy_verified(source, target, digest):
    """Copy to target via a temp file; an existing target is reused only if it already has the digest, never
    overwritten if it differs."""
    target = Path(target)
    if target.exists():
        if _sha256(target) == digest:
            return 'reused'
        raise ProvisionError('existing %s differs from the approved file; it is not overwritten' % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name('.%s.%d.part' % (target.name, os.getpid()))
    try:
        shutil.copyfile(source, tmp)
        if _sha256(tmp) != digest:
            raise ProvisionError('copy of %s changed in transit' % source)
        if not publish_if_absent(tmp, target):
            if _sha256(target) != digest:
                raise ProvisionError('a different %s appeared during the copy; not overwritten' % target)
        return 'copied'
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def configured_manifest_ok(config_data, approved):
    """(ok, reason) for an already configured actor_runtime_manifest: every recorded file must verify."""
    value = (config_data or {}).get('actor_runtime_manifest')
    if not value:
        return False, 'actor_runtime_manifest is not configured'
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_file():
        return False, 'configured actor_runtime_manifest is missing: %s' % path
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    try:
        if manifest.get('modelSha256') != approved['MODEL_SHA256']:
            return False, 'configured manifest names another model'
        verify_model(manifest['model'], approved)
        library = Path(manifest['library'])
        verify_runtime(library.parent, approved)
    except (KeyError, ProvisionError) as error:
        return False, str(error)
    return True, 'configured Vox install verified: %s' % path


def vox_sources_sha256(install_root=INSTALL_ROOT):
    root = Path(install_root) / 'actor-tts'
    names = list(VOX_SOURCE_FILES) + ['runtime-adapter/' + p.name for p in sorted((root / 'runtime-adapter').glob('*.py'))]
    out = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise ProvisionError('Vox adapter source missing from this install: actor-tts/%s' % name)
        out[name] = _sha256(path)
    return out


def engine_python(config_data, fallback):
    vox = ((config_data or {}).get('actor_beta') or {}).get('vox') or {}
    value = vox.get('python') or (config_data or {}).get('python') or fallback
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_file():
        raise ProvisionError('Vox engine python not found: %s (set actor_beta.vox.python or run Setup-Worker.ps1)' % path)
    return path


def write_runtime_manifest(path, *, model, library, artifacts, approved, dependency_directories=()):
    manifest = {'upstream': UPSTREAM, 'quantization': approved['QUANTIZATION'], 'model': str(model),
                'modelSha256': approved['MODEL_SHA256'], 'library': str(library),
                'artifacts': [{'file': name, 'sha256': digest} for name, digest in sorted(artifacts.items())],
                'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'written_by': 'KevFLIX provisioning'}
    if dependency_directories:
        manifest['dependency_directories'] = [str(d) for d in dependency_directories]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(tmp, path)                       # this per-machine file is owned by provisioning
    return manifest


def merge_config(config_path, updates_fn, *, acl_from=None, attempts=5):
    """Read-modify-write of the user config that changes ONLY the Vox keys; retried if the file changed while we
    computed the update (compare mtime/size), so a concurrent settings save is never lost."""
    config_path = Path(config_path)
    for _ in range(attempts):
        before = config_path.stat()
        raw = json.loads(config_path.read_text(encoding='utf-8-sig'))
        updated = updates_fn(json.loads(json.dumps(raw)))
        tmp = config_path.with_name('.kevflix-vox-%d-%d.tmp' % (os.getpid(), time.monotonic_ns()))
        with tmp.open('x', encoding='utf-8') as stream:
            if acl_from is not None:
                acl_from(config_path, tmp)
            json.dump(updated, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        now = config_path.stat()
        if (now.st_mtime_ns, now.st_size) != (before.st_mtime_ns, before.st_size):
            tmp.unlink()
            continue
        os.replace(tmp, config_path)
        return updated
    raise ProvisionError('the settings file kept changing; Vox provisioning did not modify it')


def provision(profile, *, model_source=None, runtime_source=None, reference_in_place=False, relay_python=None,
              acl_from=None, install_root=INSTALL_ROOT):
    """Returns a record. Raises ProvisionError with an explicit message on anything missing or mismatching."""
    profile = profile if isinstance(profile, Profile) else Profile(profile, install_root)
    if not profile.config.is_file():
        raise ProvisionError('no KevFLIX settings file yet (%s); install first' % profile.config)
    approved = approved_constants(install_root)
    config = json.loads(profile.config.read_text(encoding='utf-8-sig'))
    record = {'at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'embedded_in_installer': False}
    ok, reason = configured_manifest_ok(config, approved)
    if ok and not model_source and not runtime_source:
        manifest_path = Path(os.path.expandvars(str(config['actor_runtime_manifest'])))
        record.update(mode='kept existing configured install', manifest=str(manifest_path), reason=reason)
    else:
        if not model_source or not runtime_source:
            raise ProvisionError('Vox is not provisioned (%s). Give --model-source <folder or %s> and '
                                 '--runtime-source <folder with %s and the CUDA/ggml DLLs>.' % (reason, MODEL_NAME, LIBRARY))
        _refuse_private(model_source)
        _refuse_private(runtime_source)
        model = _model_in(model_source)
        verify_model(model, approved)
        artifacts = verify_runtime(runtime_source, approved)
        if reference_in_place:
            model_path, library_dir, mode = model, Path(runtime_source), 'referenced in place'
        else:
            model_path = profile.root / 'models' / 'actor-voxcpm2' / MODEL_NAME
            library_dir = profile.root / 'tools' / 'vox-native'
            record['model_copy'] = _copy_verified(model, model_path, approved['MODEL_SHA256'])
            record['runtime_copy'] = {n: _copy_verified(Path(runtime_source) / n, library_dir / n, d)
                                      for n, d in sorted(artifacts.items())}
            mode = 'copied into the profile'
        manifest_path = profile.root / 'models' / 'actor-q4-runtime.json'
        write_runtime_manifest(manifest_path, model=model_path, library=library_dir / LIBRARY, artifacts=artifacts,
                               approved=approved)
        record.update(mode=mode, manifest=str(manifest_path), model=str(model_path), runtime=str(library_dir))
    sources = vox_sources_sha256(install_root)
    from .leasehold import LEASE_PATH, LEASE_SHA256
    if not LEASE_PATH.is_file() or _sha256(LEASE_PATH) != LEASE_SHA256:
        raise ProvisionError('bundled lease client is missing or changed: %s' % LEASE_PATH)
    sem_mode = ((config.get('gpu_lease') or {}).get('sem_mode')) or 'absent'

    example = {}
    example_path = Path(install_root) / 'config.example.json'
    if example_path.is_file():
        example = (json.loads(example_path.read_text(encoding='utf-8-sig')).get('actor_beta') or {})

    def fill_missing(target, defaults):
        for key, value in defaults.items():
            if key not in target:
                target[key] = json.loads(json.dumps(value))
            elif isinstance(target[key], dict) and isinstance(value, dict):
                fill_missing(target[key], value)

    def update(raw):
        raw['actor_runtime_manifest'] = str(manifest_path)
        if raw.get('speech_engine') in (None, 'actor'):
            raw['speech_engine'] = 'actor'                   # preserve an explicit generic/legacy selection
        beta = raw.setdefault('actor_beta', {})
        # the approved Vox movie pipeline keys the user does not have yet (never overrides an existing value)
        fill_missing(beta, {k: v for k, v in example.items() if k != 'vox'})
        beta['enabled'] = True
        beta['actor_engine'] = 'vox'
        vox = beta.setdefault('vox', {})
        vox['sources_sha256'] = sources
        lease = vox.setdefault('gpu_lease', {})
        lease.update(client=str(LEASE_PATH), client_sha256=LEASE_SHA256, sem_mode=lease.get('sem_mode') or sem_mode)
        return raw
    merge_config(profile.config, update, acl_from=acl_from)
    record.update(sources_sha256=sources, lease_client=str(LEASE_PATH),
                  note='engine python needs numpy + soundfile (Setup-Worker installs requirements.txt into .venv)')
    out = profile.runtime / 'vox-provisioning.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        python = engine_python(json.loads(profile.config.read_text(encoding='utf-8-sig')),
                               relay_python or (Path(install_root) / '.venv' / 'Scripts' / 'python.exe'))
    except ProvisionError as error:
        record.update(usable=False, reason=str(error))
        out.write_text(json.dumps(record, indent=2), encoding='utf-8')
        raise ProvisionError('Vox model and runtime are verified and configured, but the engine is not usable yet: '
                             '%s' % error)
    record.update(usable=True, engine_python=str(python))
    out.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record
