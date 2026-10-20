"""Shared CUDA FP16 policy and selected-interpreter metadata inspection.

Stdlib only: importing this module never imports CTranslate2, torch, or a model.
Static distribution metadata is compatibility evidence, not an inference check.
The live child still checks its loaded package and the model's actual dtype.

Upstream provenance:
https://github.com/OpenNMT/CTranslate2/issues/1865
https://github.com/OpenNMT/CTranslate2/pull/1982
https://github.com/OpenNMT/CTranslate2/releases/tag/v4.7.1
"""
from __future__ import annotations

from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re

from .movie_faults import MovieConfigurationError

POLICY_ID = 'kevflix-ct2-cuda-fp16-v1'
CUDA_COMPUTE_TYPE = 'float16'
RECOMMENDED_CT2_VERSION = '4.7.1'
LEGACY_CT2_VERSION = '4.6.0'
MAX_METADATA_BYTES = 256 << 10


def policy_metadata():
    """Return a fresh JSON-safe policy value, suitable for raw-cache identity."""
    policy = {'id': POLICY_ID, 'device': 'cuda', 'compute_type': CUDA_COMPUTE_TYPE,
              'auto_selection': False, 'cpu_fallback': False,
              'fresh_setup_version': RECOMMENDED_CT2_VERSION,
              'legacy_fp16_version': LEGACY_CT2_VERSION,
              'legacy_policy': '4.6.0 is allowed only with explicit float16; it is not upgraded',
              'reason': 'Avoid unsupported INT8 GEMM on RTX 50-series/Blackwell'}
    payload = json.dumps(policy, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {**policy, 'identity': hashlib.sha256(payload).hexdigest()}


def _read_bounded(path, limit=MAX_METADATA_BYTES):
    with Path(path).open('rb') as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError('Dependency metadata exceeds its bounded size: ' + str(path))
    return value


def _environment_roots(executable, site_packages=None):
    """Windows venv/base layouts, resolved through directory junctions.

    An explicit site_packages override is for an installer's known custom child
    layout. It is never inferred from the bootstrap's sys.path/site module.
    """
    if site_packages is not None:
        values = [site_packages] if isinstance(site_packages, (str, Path)) else list(site_packages)
        return [Path(value).resolve() for value in values]
    prefix = executable.parent.parent if executable.parent.name.lower() in ('scripts', 'bin') else executable.parent
    roots = [prefix / 'Lib' / 'site-packages']
    venv = prefix / 'pyvenv.cfg'
    if venv.is_file():
        settings = {}
        for line in _read_bounded(venv, 16 << 10).decode('utf-8-sig').splitlines():
            key, sep, value = line.partition('=')
            if sep:
                settings[key.strip().lower()] = value.strip()
        if settings.get('include-system-site-packages', '').lower() == 'true':
            home = settings.get('home')
            if home:
                roots.append(Path(home) / 'Lib' / 'site-packages')
    return [root.resolve() for root in roots]


def inspect_runtime(python_executable, *, allow_legacy_46=True, site_packages=None):
    """Inspect ONLY the selected child interpreter's installed CT2 metadata.

    Returns an allowed/reason record without launching that interpreter or
    importing its code. Unknown layouts/versions fail closed. A local package
    without metadata cannot be misrepresented as an inherited base package.
    """
    record = {'selected_interpreter': str(python_executable), 'resolved_interpreter': None,
              'inspection': 'selected-child-distribution-metadata', 'inference_ready': False,
              'policy': policy_metadata(), 'allowed': False, 'legacy': False,
              'allow_legacy_46': bool(allow_legacy_46), 'version': None,
              'metadata_path': None, 'package_root': None, 'site_packages': [],
              'fault_code': None, 'reason': None}
    try:
        executable = Path(python_executable).expanduser().resolve(strict=True)
        record['resolved_interpreter'] = str(executable)
        if not executable.is_file():
            raise ValueError('Selected child interpreter is not a file')
        roots = _environment_roots(executable, site_packages)
        record['site_packages'] = [str(root) for root in roots]
        found = None
        for root in roots:
            candidates = []
            for directory in sorted(root.glob('ctranslate2-*.dist-info')):
                metadata_path = directory / 'METADATA'
                metadata_bytes = _read_bounded(metadata_path)
                metadata = BytesParser().parsebytes(metadata_bytes, headersonly=True)
                name = re.sub(r'[-_.]+', '-', str(metadata.get('Name', '')).lower())
                if name == 'ctranslate2':
                    candidates.append((str(metadata.get('Version', '')).strip(), metadata_path,
                                       hashlib.sha256(metadata_bytes).hexdigest()))
            if len(candidates) > 1:
                raise ValueError('Multiple CTranslate2 distributions exist in ' + str(root))
            package = root / 'ctranslate2'
            if candidates:
                if not (package / '__init__.py').is_file():
                    raise ValueError('CTranslate2 metadata exists without its installed package in ' + str(root))
                found = (candidates[0], package)
                break
            if package.exists():
                raise ValueError('Local CTranslate2 package has no matching distribution metadata in ' + str(root))
        if found is None:
            record.update(fault_code='ct2_runtime_missing', reason='No CTranslate2 distribution in the selected child environment')
            return record
        (version, metadata_path, digest), package = found
        record.update(version=version, metadata_path=str(metadata_path), metadata_sha256=digest,
                      package_root=str(package.resolve()))
        if version == RECOMMENDED_CT2_VERSION:
            record.update(allowed=True, compatibility='reviewed-windows-fp16', reason='Pinned Windows runtime with explicit CUDA float16')
        elif version == LEGACY_CT2_VERSION and allow_legacy_46:
            record.update(allowed=True, legacy=True, compatibility='legacy-4.6.0-explicit-fp16',
                          reason='Existing 4.6.0 retained with explicit CUDA float16; no dependency upgrade was performed')
        else:
            record.update(fault_code='ct2_runtime_version_unsupported',
                          reason='CTranslate2 %s is outside this reviewed Windows FP16 policy (fresh setup %s; legacy %s)' %
                                 (version, RECOMMENDED_CT2_VERSION, LEGACY_CT2_VERSION if allow_legacy_46 else 'disabled'))
    except (OSError, ValueError, TypeError) as error:
        record.update(fault_code='ct2_runtime_metadata_invalid', reason=str(error))
    return record


def require_runtime(python_executable, *, allow_legacy_46=True, site_packages=None):
    record = inspect_runtime(python_executable, allow_legacy_46=allow_legacy_46, site_packages=site_packages)
    if not record['allowed']:
        raise MovieConfigurationError(record['fault_code'] or 'ct2_runtime_metadata_invalid', record['reason'])
    return record


def require_loaded_runtime(module, record):
    """Reject a shadowed/different CT2 after import; metadata alone is not proof."""
    version = str(getattr(module, '__version__', ''))
    origin = getattr(module, '__file__', None)
    if version != record.get('version') or not origin:
        raise MovieConfigurationError('ct2_loaded_runtime_mismatch',
                                      'Loaded CT2 version does not match selected interpreter metadata')
    if Path(origin).resolve().parent != Path(record['package_root']).resolve():
        raise MovieConfigurationError('ct2_loaded_runtime_shadowed',
                                      'Loaded CT2 package differs from selected interpreter distribution')
    return {'version': version, 'package_file': str(Path(origin).resolve()), 'metadata_matched': True}
