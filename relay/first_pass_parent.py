"""PROTOTYPE PC-server side of the one-pass Demucs producer (actor_beta.demucs_first_pass, default off).

Stdlib only, like the accepted demucs_bed_parent: never imports torch/first_pass/Guard, never
exits the process. Starts the owned child with the explicit project python (-I), validates the
bundle (hashes, geometry, key, layout provenance) and returns it. Owner cancel -> Cancelled
(child killed only); a real resource abort (Sem/lease/RAM guard) -> ResourceYield carrying a
'resource' fault record; a wall-time bound -> fatal fault (a deadline is not resource evidence);
refusal or invalid bundle -> ValueError. Not authorized for a model run; the caller anchor is B's (see proposal).
"""
from __future__ import annotations

import hashlib
import json
import os
import wave
from pathlib import Path

from .first_pass_key import bundle_problems, code_hashes, expected_request
from .process import Cancelled
from .resource_budget import ResourceYield

VERSION = 'demucs-first-pass-parent-v0'
HERE = Path(__file__).resolve().parent
CHILD = HERE / 'first_pass_child.py'
STEMS = ('processing', 'raw_vocals', 'background', 'removal_stem')
LAUNCH_ALLOWANCE_SECONDS = 60
RATE = 48000


def faulted(error, kind, code, detail=None):
    """Attach explicit relay.movie_faults metadata (fault_kind/fault_code/detail) to an existing exception type."""
    error.fault_kind, error.fault_code = kind, str(code)[:120]
    error.detail = str(error if detail is None else detail)[:4000]
    return error


def abort_error(abort):
    """Exception for a child's abort.json: its explicit fault record when present (lease guard v2), else the legacy
    kinds. A legacy wall-time bound ('... second bound') is fatal; other legacy guard yields are resource waits."""
    fault = abort.get('fault') if isinstance(abort, dict) else None
    text = str((abort or {}).get('error'))
    if isinstance(fault, dict) and fault.get('kind') in ('resource', 'fatal') and isinstance(fault.get('code'), str):
        if fault['kind'] == 'resource':
            return faulted(ResourceYield('Demucs first pass yielded: %s' % text), 'resource', fault['code'], text)
        return faulted(RuntimeError('Demucs first pass stopped: %s' % text), 'fatal', fault['code'], text)
    if (abort or {}).get('kind') == 'ResourceYield':
        if 'second bound' in text:
            return faulted(RuntimeError('Demucs first pass exceeded its wall bound: %s' % text), 'fatal',
                           'demucs_window_deadline', text)
        return faulted(ResourceYield('Demucs first pass yielded: %s' % text), 'resource', 'demucs_guard_yield', text)
    return None


def options(config):
    data = getattr(config, 'data', config)
    beta = data.get('actor_beta') if hasattr(data, 'get') else None
    if not isinstance(beta, dict) or beta.get('enabled') is not True or beta.get('demucs_first_pass') is not True:
        return None
    if beta.get('demucs_mix_bed') is True:
        raise faulted(ValueError('demucs_first_pass replaces the post-worker demucs_mix_bed; enable only one'),
                      'fatal', 'demucs_config_invalid')
    return beta


def config_path(config, key):
    """Absolute path for a configured file/dir: Config.resolve (relative to the config file) when present,
    otherwise it must already be absolute. The child runs with cwd = output folder, so relative paths never pass."""
    value = config[key]
    if not value:
        raise faulted(ValueError('demucs_first_pass requires %s' % key), 'fatal', 'demucs_config_invalid')
    resolve = getattr(config, 'resolve', None)
    path = Path(resolve(value)) if callable(resolve) else Path(str(value))
    if not path.is_absolute():
        raise faulted(ValueError('%s must resolve to an absolute path, got %r' % (key, value)), 'fatal',
                      'demucs_config_invalid')
    return path


def _parent_record(expected, paths=None):
    record = {'version': VERSION, 'sources': dict(expected['inputs']['code'])}
    if paths:
        record.update(request_sha256=sha256(paths['request.json']), receipt_sha256=sha256(paths['receipt.json']))
    return record


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _float_wav_geometry(path):
    """(channels, rate, frames) of a WAVE_FORMAT_IEEE_FLOAT / EXTENSIBLE file via RIFF headers (stdlib)."""
    with open(path, 'rb') as handle:
        if handle.read(4) != b'RIFF' or (handle.read(4), handle.read(4))[1] != b'WAVE':
            raise ValueError('not a RIFF/WAVE file')
        channels = rate = block = data = None
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            name, size = header[:4], int.from_bytes(header[4:], 'little')
            if name == b'fmt ':
                fmt = handle.read(size)
                channels, rate = int.from_bytes(fmt[2:4], 'little'), int.from_bytes(fmt[4:8], 'little')
                block = int.from_bytes(fmt[12:14], 'little')
            elif name == b'data':
                data = size
                break
            else:
                handle.seek(size + (size & 1), 1)
    if None in (channels, rate, block, data) or not block:
        raise ValueError('incomplete WAV header')
    return channels, rate, data // block


def problems(bundle, expected, *, folder, frames):
    """All reasons a bundle cannot be trusted for THIS caller request (empty list: trusted)."""
    if bundle.get('state') == 'refused':
        return ['child refused: %s' % bundle.get('reason')]
    found = bundle_problems(bundle, expected)
    for name in STEMS:
        entry = bundle.get(name) or {}
        path = Path(entry.get('wav', ''))
        if path.parent != Path(folder) or not path.is_file() or sha256(path) != entry.get('sha256'):
            found.append('%s missing or hash changed' % name)
        elif entry.get('layout') != 'stereo' or _float_wav_geometry(path) != (2, RATE, frames):
            found.append('%s is not 48 kHz stereo of the window' % name)
    return found


def reusable(folder, expected, *, frames):
    """Cached bundle for this exact request, or (None, reasons): a stale-but-valid bundle is refused, never reused."""
    path = Path(folder) / 'first-pass' / 'first-pass-bundle.json'
    if not path.is_file():
        return None, ['no cached bundle']
    try:
        bundle = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        return None, ['unreadable cached bundle: %s' % error]
    found = problems(bundle, expected, folder=path.parent, frames=frames)
    return (None, found) if found else (bundle, [])


# ---- movie lease admission + resident GPU producer (W10) ---------------------------------------------
# actor_beta.first_pass_admission == 'movie-lease' (the app's movie pipeline): the first pass is admitted by the movie
# GPU lease (relay.app_runtime, lease v2) instead of demucs_bed.Guard's Sem status.json rule, which can never admit
# Sem OFF and opens Sem's status file. The producer is the RESIDENT worker (relay/first_pass_resident.py, one model
# load), started INSIDE the lease's Job by relay.movie_pipeline.resident_demucs (lease.spawn). Its in-process guard
# keeps the RAM / working-set / wall bounds and watches the lease cancel event, plus verified Sem process absence in
# sem_mode 'absent'. Without a registered resident producer this mode refuses: an unowned per-window GPU child is never
# started. Any other admission value keeps the original per-window child and status guard unchanged.
# actor_beta.demucs_device 'cuda' is accepted only with movie-lease admission and is keyed into the bundle.
import threading as _threading

RESIDENT = None
ADMISSIONS = ('movie-lease',)
_FOLDER_LOCKS = {}
_FOLDER_LOCKS_GUARD = _threading.Lock()


def _folder_lock(folder):
    key = os.path.normcase(str(Path(folder).resolve()))
    with _FOLDER_LOCKS_GUARD:
        return _FOLDER_LOCKS.setdefault(key, _threading.Lock())


def admission(beta):
    value = beta.get('first_pass_admission')
    if value is None:
        return None
    if value not in ADMISSIONS:
        raise faulted(ValueError('actor_beta.first_pass_admission must be "movie-lease" or not set, not %r'
                                 % (value,)), 'fatal', 'demucs_config_invalid')
    return value


def device_for(beta, mode):
    device = beta.get('demucs_device', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise faulted(ValueError('actor_beta.demucs_device must be cpu or cuda, not %r' % (device,)), 'fatal',
                      'demucs_config_invalid')
    if device == 'cuda' and mode != 'movie-lease':
        raise faulted(ValueError('demucs_device cuda requires actor_beta.first_pass_admission "movie-lease"'),
                      'fatal', 'demucs_config_invalid')
    return device


def ensure_fault(error, code='demucs_first_pass_failed'):
    """The same exception with explicit fault metadata: kept when present (abort_error, the resident producer);
    owner Cancelled -> cancelled; a raw ResourceYield -> resource; a TimeoutError -> fatal demucs_window_deadline;
    every other error (invalid config/path/permission, refusal, invalid bundle; ValueError keeps its type) -> fatal."""
    if getattr(error, 'fault_kind', None) and getattr(error, 'fault_code', None):
        return error
    if isinstance(error, Cancelled):
        return faulted(error, 'cancelled', 'cancelled')
    if isinstance(error, ResourceYield):
        return faulted(error, 'resource', 'demucs_resource_yield')
    if isinstance(error, TimeoutError):
        return faulted(error, 'fatal', 'demucs_window_deadline', '%s: %s' % (type(error).__name__, error))
    return faulted(error, 'fatal', code, '%s: %s' % (type(error).__name__, error))


def run(config, owner, *, folder, original_source_wav, source_layout_evidence, origin, frames, scope):
    """Owned producer -> validated bundle dict, or None when the flag is off (caller keeps BandIt). Every raised
    error carries relay.movie_faults metadata (ensure_fault)."""
    try:
        return _run(config, owner, folder=folder, original_source_wav=original_source_wav,
                    source_layout_evidence=source_layout_evidence, origin=origin, frames=frames, scope=scope)
    except Exception as error:  # noqa: BLE001 - same object re-raised, metadata attached
        ensure_fault(error)
        raise


def _run(config, owner, *, folder, original_source_wav, source_layout_evidence, origin, frames, scope):
    beta = options(config)
    if beta is None:
        return None
    python = config['python']
    if not python or not Path(python).is_absolute() or not Path(python).is_file():
        raise faulted(ValueError('demucs_first_pass requires an explicit absolute project python'), 'fatal',
                      'demucs_runtime_missing')
    mode = admission(beta)
    device = device_for(beta, mode)
    model_dir = config_path(config, 'separation_model_dir')
    status_file = None if mode == 'movie-lease' else config_path(config, 'sem_status_file')
    timeout = float(beta.get('demucs_timeout_seconds', 120))
    threads = int(beta.get('demucs_threads', 4))
    expected = expected_request(original_source_wav=original_source_wav, source_layout=source_layout_evidence['layout'],
                                frames=frames, model_pins=beta['demucs_model_pins'], threads=threads,
                                code_directory=HERE, scope=scope, origin=origin, device=device)
    with _folder_lock(folder):         # a concurrent caller of the same window waits, then reuses the bundle
        cached, stale = reusable(folder, expected, frames=frames)
        if cached is not None:
            return {**cached, 'reused': True, 'stale_cache_reasons': [], 'parent': _parent_record(expected)}
        producer = RESIDENT if mode == 'movie-lease' else None
        if mode == 'movie-lease' and producer is None:
            raise faulted(ValueError('first_pass_admission movie-lease requires the resident movie producer '
                                     '(relay.movie_pipeline)'), 'fatal', 'demucs_producer_missing')
        folder = Path(folder) / 'first-pass'
        folder.mkdir(parents=True, exist_ok=True)
        paths = {n: folder / ('first-pass-' + n) for n in ('request.json', 'receipt.json', 'abort.json')}
        for leftover in (paths['receipt.json'], paths['abort.json']):
            leftover.unlink(missing_ok=True)
        request = {'version': VERSION, 'folder': str(folder), 'original_source_wav': str(original_source_wav),
                   'source_layout_evidence': source_layout_evidence, 'origin': float(origin), 'scope': scope,
                   'model_dir': str(model_dir), 'model_pins': beta['demucs_model_pins'],
                   'sem_status_file': None if status_file is None else str(status_file), 'threads': threads,
                   'timeout_seconds': timeout, 'receipt': str(paths['receipt.json']), 'abort': str(paths['abort.json']),
                   'device': device, 'admission': mode or 'sem-status'}
        temporary = Path(str(paths['request.json']) + '.tmp')
        temporary.write_text(json.dumps(request, indent=2, default=str), encoding='utf-8')
        os.replace(temporary, paths['request.json'])
        if producer is not None:
            # produce() raises Cancelled (owner), ResourceYield with a 'resource' fault (lease/guard/preemption), a
            # fatal fault (deadline, protocol, configuration) or ValueError (refused).
            producer.produce(request, owner, timeout=timeout + LAUNCH_ALLOWANCE_SECONDS)
            via = 'resident'
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
            try:
                owner.run([python, '-I', CHILD, paths['request.json']], timeout=timeout + LAUNCH_ALLOWANCE_SECONDS,
                          env=env, cwd=folder)
            except Cancelled:
                raise
            except TimeoutError as error:
                raise faulted(RuntimeError('Demucs first pass exceeded its bound: %s' % error), 'fatal',
                              'demucs_window_deadline') from error
            except RuntimeError:
                if paths['abort.json'].is_file():
                    try:
                        abort = json.loads(paths['abort.json'].read_text(encoding='utf-8'))
                    except (OSError, ValueError):
                        abort = None
                    error = abort_error(abort)
                    if error is not None:
                        raise error
                if paths['receipt.json'].is_file():
                    refused = json.loads(paths['receipt.json'].read_text(encoding='utf-8'))
                    if refused.get('state') == 'refused':
                        raise faulted(ValueError('Demucs first pass refused: %s' % refused.get('reason')), 'fatal',
                                      'demucs_first_pass_refused')
                raise
            via = 'child'
        bundle = json.loads(paths['receipt.json'].read_text(encoding='utf-8'))
        if bundle.get('state') == 'refused':
            raise faulted(ValueError('Demucs first pass refused: %s' % bundle.get('reason')), 'fatal',
                          'demucs_first_pass_refused')
        found = problems(bundle, expected, folder=folder, frames=frames)
        if code_hashes(HERE) != expected['inputs']['code']:
            found.append('producer code changed during the child run')
        if found:
            raise faulted(ValueError('first-pass bundle does not match the caller request: ' + '; '.join(found)),
                          'fatal', 'demucs_bundle_invalid')
        bundle['stale_cache_reasons'] = stale
        bundle['parent'] = {**(bundle.get('parent') or {}), **_parent_record(expected, paths), 'producer': via, 'device': device,
                            'admission': mode or 'sem-status'}
        return bundle
