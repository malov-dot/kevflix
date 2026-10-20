"""Movie-side GPU lease client v2 (candidate movie-gpu-lease-v2; STAGED, not wired into the live relay). Windows only.

Contract: CONTRACT.md. Stdlib only. Sem is READ ONLY: this module reads Sem's status.json and queries VRAM (NVML
nvmlDeviceGetMemoryInfo, read-only; nvidia-smi --query-gpu fallback). It never writes in a Sem folder, never asks Sem
to unload, and never opens, signals or kills any process that is not its own child.

v2 architecture (fixes for the W6 review of v1, LEASE-ADVERSARIAL-REVIEW-20261017.md):
* acquire() starts a small WATCHER PROCESS (this same file, executed from sha-verified bytes, ABOVE_NORMAL priority).
  The watcher, not the holder's Python threads, proves Sem idle, so busy holder threads / GIL-holding calls cannot
  delay detection (review P1-3). The watcher owns:
    - the machine-wide named mutex Global\\KevFLIX-Movie-GPU-Lease (one lease per machine for every copy/version;
      freed by the OS if the watcher dies)                                                         (review P2-1)
    - a Job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE holding the holder's GPU children (lease.spawn() /
      lease.adopt()). The watcher closes it at preempt + grace_s, at release, and when the holder dies; if the
      watcher itself dies the OS closes it. So no GPU child outlives its lease                        (review P1-1)
    - a named manual-reset cancel event (set at preempt; holder threads and native children can wait on it)
    - a 5 Hz VRAM governor that tolerates Sem's ~0.5 s Infernix dips (sustained-below rule)          (review P2-2)
* Status reads are notify-only everywhere, settle and wait included (ReadDirectoryChangesW on the directory; no handle
  on status.json except one short read per new Sem version). The frozen sem_idle_pid verdict is computed on a private
  copy of exactly those bytes, so each version costs one open of Sem's file                        (review P3-1)
* Unrelated directory events do not wake the loop                                                    (review P3-2)
* Every piece of guard code that runs is exec'd from bytes whose sha256 was checked: the frozen relay package's
  import-time closure (5 files), tolerant_status.py, and this file in the watcher                    (review P3-3)
* A process holds at most one lease object until release(); a second acquire() raises previous_lease_unreleased
                                                                                                     (review P1-2)
"""
from __future__ import annotations

import collections
import ctypes
import hashlib
import json
import os
import queue
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import uuid
from ctypes import wintypes
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

if os.name != 'nt':  # pragma: no cover
    raise ImportError('movie-gpu-lease-v2 is Windows only (named mutex, Job object, ReadDirectoryChangesW)')

SOURCE_PATH = Path(__file__).resolve()
SOURCE_SHA256 = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
HERE = SOURCE_PATH.parent
HOUR = HERE.parent
DEFAULT_STATUS = Path('C:/Sem-Data/profile-v1/local-coding-inbox/status.json')

MUTEX_NAME = 'Global\\KevFLIX-Movie-GPU-Lease'               # fixed: no env var, no state_dir dependence
TEST_MUTEX_PREFIX = 'Local\\KevFLIX-Movie-GPU-Lease-TEST-'   # the only accepted override (tests / review repros)
OBJECT_PREFIX = 'Local\\KevFLIX-Movie-GPU-Lease-'

TOLERANT_SHA256 = 'd58b4ec2fcb37d3ad0f45dd24c18a9fbd3d6c1b5dae5aca8b7e9910c6aaf587f'
TOLERANT_PATH = HOUR / 'guard-status-read-v1' / 'tolerant_status.py'
# Import-time closure of the frozen guard relay/demucs_bed.py (sem_idle_pid), in exec order. Lazy imports inside
# functions (numpy, torch, .process, ...) are never reached by sem_idle_pid and would fail closed (empty __path__).
GUARD_CLOSURE = (
    ('__init__.py', '04c1165f92486791c29e7f5a7fb19046396d52c090e943ae98e271bcb9a1ec24'),
    ('short_utterance_policy.py', '4d8b55e27c649aa560bf1593558de85eb936b870ab56aa8845943aa559150204'),
    ('original_audio_policy.py', '16c379652457953a7c42dea5546041037aed0d8891daedb369735bdcade441a2'),
    ('cinematic.py', 'ef39c3954c39aba9c08b5bdf4c1dd4cc99148285c0434d9e0b6cb9ece9a6c206'),
    ('demucs_bed.py', '49ec437aae77f56db41fad5ec913eff61b9baa3d44e787bd5e4dcb3bff1ba32e'),
)
GUARD_SHA256 = GUARD_CLOSURE[-1][1]
GUARD_PACKAGES = tuple(HOUR / name / 'scratch' / 'base-live'
                       for name in ('beta5-asr-gain-fix-v3', 'beta5-asr-gain-fix-v2', 'beta5-asr-gain-fix-v1'))

MODES = ('no-release', 'sem-release')
SEM_MODES = ('running', 'absent')   # 'absent' must be selected explicitly; it is never inferred from a missing file
SEM_SCAN_S = 0.025           # absent mode: process scan period (incremental scan ~0.1 ms; full reconcile every 2 s)
DRAIN_TIMEOUT_S = 10.0       # release/preempt: wait this long for every Job process to be gone (then keep waiting,
                             # holding the machine-wide mutex: the lock is never freed while a GPU child may run)
# Sem-owned process identity (absent mode). Paths/markers are compared lower-case with backslashes. Source: the
# 16:18Z inventory and evidence/owner-shutdown-sem-for-movies-20261017/RECEIPT.json (descendantsBefore).
SEM_PROCESS_RULES = {
    # any process with one of these image names is Sem (any install), even if its path cannot be read
    'names_always': ['desktop companion model 3.0.exe', 'sembackgroundhost.exe'],
    # these names are Sem only if the image path or the command line matches below
    'names_checked': ['infernix-serve.exe', 'livekit-server.exe', 'python.exe', 'pythonw.exe', 'node.exe'],
    'exe_roots': ['c:\\sem-trials\\', 'c:\\sem-migration-staging\\sem-5.0\\', 'c:\\sem-migration-staging\\sem-4.0\\',
                  'c:\\sem-developertools\\sem-decision\\', 'c:\\sem-developertools\\sem-image-native\\'],
    'cmd_markers': ['\\sem-trials\\', '\\sem-migration-staging\\sem-5.0\\', '\\sem-migration-staging\\sem-4.0\\',
                    '\\sem-data\\profile-v1\\', 'sem_image_native', 'nano-voxcpm2\\q8_voice_worker.py',
                    'native-audio-worker\\worker.py', 'sem-decision-worker.py'],
}
LATENCY_BOUND_S = 0.2        # watcher detection: not-idle status visible -> cancel event set
POLL_S = 0.025               # watcher loop period
PROOF_DEADLINE_S = 0.15      # idle must be re-proved this often, else status_unreadable
REVERIFY_S = 1.0             # re-read when no change event arrived for this long
MAX_STATUS_AGE_S = 2.0       # stricter than the frozen 15 s: Sem writes every 500 ms (review P2-3)
FROZEN_INTERVAL_S = 0.2      # frozen check cadence in poll fallback only
SETTLE_S = 2.0
GRACE_S = 0.15               # preempt -> Job closed (GPU children killed); W4 hard-stops at 0.75 x 0.2 s
TTL_S = 30.0
RESERVE_MIB = 3072
GPU_SAMPLE_S = 0.2           # 5 Hz
GPU_BASELINE_S = 2.0         # acquire budget = median free over this window (a <= 0.6 s dip is <= 3 of 10 samples)
GPU_SUSTAIN_S = 1.5          # gpu_reserve only if every sample for this long is below the reserve
GPU_MAX_AGE_S = 5.0
GPU_INTERVAL_S = 2.0         # holder-fed probe period (gpu_probe callable; tests / compatibility)
FIRST_GPU_TIMEOUT_S = 12.0
WATCHER_TIMEOUT_S = 15.0     # watcher start + settle margin
STALE_AFTER_S, SKEW_BEFORE_S = 15.0, -2.0   # frozen freshness rule

TRANSIENT_WINERRORS = frozenset({5, 32, 33})


# ---- errors and events --------------------------------------------------------------------------------------
class LeaseError(Exception):
    pass


class LeaseUnavailable(LeaseError):
    def __init__(self, reason, detail='', last=None):
        super().__init__('%s: %s' % (reason, detail) if detail else reason)
        self.reason, self.detail, self.last = reason, detail, last


class LeaseModeUnavailable(LeaseUnavailable):
    pass


class GpuProbeError(LeaseError):
    pass


class NoSampleYet(GpuProbeError):
    """The holder-fed probe has not delivered its first sample yet (not an error; bounded by first_gpu_timeout_s)."""


class LeasePreempted(LeaseError):
    def __init__(self, event):
        super().__init__('lease %s ended: %s (%s)' % (event.lease_id, event.reason, event.detail))
        self.event = event


class NotIdle(Exception):
    def __init__(self, reason, detail=''):
        super().__init__('%s: %s' % (reason, detail))
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class PreemptEvent:
    reason: str
    detail: str
    lease_id: str
    detected_at: float            # time.perf_counter() (QPC, comparable across processes on Windows), in the watcher
    last_idle_proof_at: float     # perf_counter of the last successful idle proof, in the watcher
    latency_bound_s: float = LATENCY_BOUND_S
    received_at: float = 0.0      # perf_counter when the holder process processed it (GIL-dependent)


@dataclass(frozen=True)
class VramSample:
    free_mib: int
    total_mib: int
    used_mib: int
    at: float                     # perf_counter
    gpus: int = 1
    baseline_mib: int | None = None   # median free over the baseline window (watcher)
    min_mib: int | None = None        # minimum free over the observed window
    window_s: float = 0.0

    def age_s(self, clock=time.perf_counter):
        return clock() - self.at


# ---- Win32 ----------------------------------------------------------------------------------------------------
_k32 = ctypes.WinDLL('kernel32', use_last_error=True)
_H = wintypes.HANDLE
_INVALID = _H(-1).value


def _proto(name, restype, *argtypes):
    function = getattr(_k32, name)
    function.restype, function.argtypes = restype, list(argtypes)
    return function


_CreateFileW = _proto('CreateFileW', _H, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                      wintypes.DWORD, wintypes.DWORD, _H)
_ReadFile = _proto('ReadFile', wintypes.BOOL, _H, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                   wintypes.LPVOID)
_GetFileSizeEx = _proto('GetFileSizeEx', wintypes.BOOL, _H, ctypes.POINTER(ctypes.c_longlong))
_CloseHandle = _proto('CloseHandle', wintypes.BOOL, _H)
_CreateEventW = _proto('CreateEventW', _H, wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
_OpenEventW = _proto('OpenEventW', _H, wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
_SetEvent = _proto('SetEvent', wintypes.BOOL, _H)
_ResetEvent = _proto('ResetEvent', wintypes.BOOL, _H)
_CreateMutexW = _proto('CreateMutexW', _H, wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
_ReleaseMutex = _proto('ReleaseMutex', wintypes.BOOL, _H)
_WaitForSingleObject = _proto('WaitForSingleObject', wintypes.DWORD, _H, wintypes.DWORD)
_WaitForMultipleObjects = _proto('WaitForMultipleObjects', wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_H),
                                 wintypes.BOOL, wintypes.DWORD)
_CreateJobObjectW = _proto('CreateJobObjectW', _H, wintypes.LPVOID, wintypes.LPCWSTR)
_OpenJobObjectW = _proto('OpenJobObjectW', _H, wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
_SetInformationJobObject = _proto('SetInformationJobObject', wintypes.BOOL, _H, ctypes.c_int, wintypes.LPVOID,
                                  wintypes.DWORD)
_QueryInformationJobObject = _proto('QueryInformationJobObject', wintypes.BOOL, _H, ctypes.c_int, wintypes.LPVOID,
                                    wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
_AssignProcessToJobObject = _proto('AssignProcessToJobObject', wintypes.BOOL, _H, _H)
_TerminateJobObject = _proto('TerminateJobObject', wintypes.BOOL, _H, wintypes.UINT)
_OpenProcess = _proto('OpenProcess', _H, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_GetCurrentProcess = _proto('GetCurrentProcess', _H)
_SetPriorityClass = _proto('SetPriorityClass', wintypes.BOOL, _H, wintypes.DWORD)
_CreateToolhelp32Snapshot = _proto('CreateToolhelp32Snapshot', _H, wintypes.DWORD, wintypes.DWORD)
_OpenThread = _proto('OpenThread', _H, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_ResumeThread = _proto('ResumeThread', wintypes.DWORD, _H)
_ReadDirectoryChangesW = None   # prototyped below with _OVERLAPPED
_TerminateProcess = _proto('TerminateProcess', wintypes.BOOL, _H, wintypes.UINT)

_GENERIC_READ, _SHARE_ALL, _OPEN_EXISTING = 0x80000000, 0x7, 3
_SYNCHRONIZE, _EVENT_MODIFY_STATE = 0x00100000, 0x0002
_JOB_ASSIGN, _JOB_QUERY = 0x0001, 0x0004
_WAIT_OBJECT_0, _WAIT_ABANDONED, _WAIT_TIMEOUT, _WAIT_FAILED = 0, 0x80, 0x102, 0xFFFFFFFF
_ERROR_ALREADY_EXISTS = 183
_ABOVE_NORMAL = 0x8000
_CREATE_SUSPENDED = 0x4
_KILL_ON_JOB_CLOSE = 0x2000


class _THREADENTRY32(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ThreadID', wintypes.DWORD),
                ('th32OwnerProcessID', wintypes.DWORD), ('tpBasePri', wintypes.LONG), ('tpDeltaPri', wintypes.LONG),
                ('dwFlags', wintypes.DWORD)]


_Thread32First = _proto('Thread32First', wintypes.BOOL, _H, ctypes.POINTER(_THREADENTRY32))
_Thread32Next = _proto('Thread32Next', wintypes.BOOL, _H, ctypes.POINTER(_THREADENTRY32))


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                       'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _JOBOBJECT_BASIC_LIMIT_INFORMATION), ('IoInfo', _IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [('TotalUserTime', ctypes.c_int64), ('TotalKernelTime', ctypes.c_int64),
                ('ThisPeriodTotalUserTime', ctypes.c_int64), ('ThisPeriodTotalKernelTime', ctypes.c_int64),
                ('TotalPageFaultCount', wintypes.DWORD), ('TotalProcesses', wintypes.DWORD),
                ('ActiveProcesses', wintypes.DWORD), ('TotalTerminatedProcesses', wintypes.DWORD)]


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [('Internal', ctypes.c_void_p), ('InternalHigh', ctypes.c_void_p), ('Offset', wintypes.DWORD),
                ('OffsetHigh', wintypes.DWORD), ('hEvent', _H)]


_ReadDirectoryChangesW = _proto('ReadDirectoryChangesW', wintypes.BOOL, _H, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.BOOL, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                ctypes.POINTER(_OVERLAPPED), wintypes.LPVOID)
_GetOverlappedResult = _proto('GetOverlappedResult', wintypes.BOOL, _H, ctypes.POINTER(_OVERLAPPED),
                              ctypes.POINTER(wintypes.DWORD), wintypes.BOOL)
_CancelIoEx = _proto('CancelIoEx', wintypes.BOOL, _H, ctypes.POINTER(_OVERLAPPED))


def _ok_handle(handle):
    return handle not in (None, 0, _INVALID)


def _winerror():
    return ctypes.WinError(ctypes.get_last_error())


# ---- reading status.json ------------------------------------------------------------------------------------
def is_transient_lock(error):
    """Same classification as guard-status-read-v1 is_transient_lock (a missing file is never a lock)."""
    if isinstance(error, (FileNotFoundError, NotADirectoryError)):
        return False
    if isinstance(error, PermissionError):
        return True
    return isinstance(error, OSError) and getattr(error, 'winerror', None) in TRANSIENT_WINERRORS


def read_shared(path, limit=1 << 20):
    """One short read with full sharing (the handle is open for well under a millisecond)."""
    handle = _CreateFileW(str(path), _GENERIC_READ, _SHARE_ALL, None, _OPEN_EXISTING, 0, None)
    if not _ok_handle(handle):
        raise _winerror()
    try:
        size = ctypes.c_longlong(0)
        if not _GetFileSizeEx(handle, ctypes.byref(size)):
            raise _winerror()
        if size.value > limit:
            raise ValueError('status file too large: %d bytes' % size.value)
        buffer = ctypes.create_string_buffer(max(1, size.value))
        done = wintypes.DWORD(0)
        if not _ReadFile(handle, buffer, size.value, ctypes.byref(done), None):
            raise _winerror()
        return buffer.raw[:done.value]
    finally:
        _CloseHandle(handle)


def read_state(path, reader=read_shared):
    return json.loads(reader(path).decode('utf-8-sig'))


def classify_state(state, now_wall=None, max_age_s=MAX_STATUS_AGE_S):
    """None when idle, else (reason, detail). Every frozen sem_idle_pid condition plus stricter extras."""
    if not isinstance(state, dict):
        return 'status_unreadable', 'status is not a JSON object'
    pid = state.get('pid')
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return 'status_unreadable', 'pid missing or invalid'
    try:
        at = datetime.fromisoformat(str(state['at']).replace('Z', '+00:00'))
        now = datetime.now(timezone.utc) if now_wall is None else datetime.fromtimestamp(now_wall, timezone.utc)
        age = (now - at).total_seconds()
    except (KeyError, ValueError, TypeError) as error:
        return 'status_unreadable', 'bad "at": %s' % error
    voice = state.get('voice') or {}
    if not isinstance(voice, dict):
        return 'sem_not_idle', 'voice is not an object'
    if voice.get('active'):
        return 'voice_active', 'voice.active'
    if voice.get('closing'):
        return 'voice_closing', 'voice.closing'
    limit = STALE_AFTER_S if max_age_s is None else min(STALE_AFTER_S, float(max_age_s))
    if age > limit or age < SKEW_BEFORE_S:
        return 'status_stale', 'status age %.2f s (limit %.1f s)' % (age, limit)
    if state.get('codexVoiceActive'):
        return 'sem_not_idle', 'codexVoiceActive'
    if not state.get('serverReady'):
        return 'sem_not_idle', 'serverReady false'
    if state.get('busy'):
        return 'sem_not_idle', 'busy'
    # ---- extras (stricter than the frozen rule; a missing required field is not idle)
    if voice.get('quarantined'):
        return 'sem_not_idle', 'voice.quarantined'
    for key in ('otherBusy', 'codingBusy'):
        if state.get(key, True) is not False:
            return 'sem_not_idle', '%s not false' % key
    resources = state.get('resources')
    if not isinstance(resources, dict):
        return 'sem_not_idle', 'resources missing'
    if resources.get('preparing', True) is not False:
        return 'sem_not_idle', 'resources.preparing'
    if resources.get('activeJobKind', 'missing') is not None:
        return 'sem_not_idle', 'resources.activeJobKind %r' % resources.get('activeJobKind', 'missing')
    if resources.get('queuedKinds', None) != []:
        return 'sem_not_idle', 'resources.queuedKinds %r' % (resources.get('queuedKinds'),)
    if resources.get('workerMayBeRunning', True) is not False:
        return 'sem_not_idle', 'resources.workerMayBeRunning'
    if (state.get('nativeBrain') or {}).get('starting'):
        return 'sem_not_idle', 'nativeBrain.starting'
    if (state.get('decisionAddon') or {}).get('loading'):
        return 'sem_not_idle', 'decisionAddon.loading'
    return None


# ---- verified code loading (review P3-3) ------------------------------------------------------------------------
def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_verified(name, path, expect_sha256, *, package='', is_package=False):
    """Exec exactly the bytes whose sha256 was checked (no bytecode cache, no second read)."""
    path = Path(path)
    data = path.read_bytes()
    digest = _sha256_bytes(data)
    if digest != expect_sha256:
        raise RuntimeError('%s drifted: sha256 %s, expected %s' % (path, digest, expect_sha256))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = package
    if is_package:
        module.__path__ = []          # nothing else can be imported from this package: lazy imports fail closed
    module.W5_VERIFIED_SHA256 = digest
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), 'exec', dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_tolerant(path=TOLERANT_PATH):
    module = sys.modules.get('w5_tolerant_status')
    if module is not None and getattr(module, 'W5_VERIFIED_SHA256', None) == TOLERANT_SHA256:
        return module
    return load_verified('w5_tolerant_status', path, TOLERANT_SHA256)


def guard_package_root(packages=GUARD_PACKAGES):
    for root in packages:
        relay = Path(root) / 'relay'
        if all((relay / f).is_file() and _sha256_bytes((relay / f).read_bytes()) == s for f, s in GUARD_CLOSURE):
            return relay
    raise RuntimeError('no staged frozen relay package whose 5-file import closure matches the pins')


def load_frozen_guard(packages=GUARD_PACKAGES):
    """relay.demucs_bed under the private package name w5_frozen_relay, from verified bytes only."""
    module = sys.modules.get('w5_frozen_relay.demucs_bed')
    if module is not None and getattr(module, 'W5_VERIFIED_SHA256', None) == GUARD_SHA256:
        return module
    relay = guard_package_root(packages)
    for name in [n for n in list(sys.modules) if n == 'w5_frozen_relay' or n.startswith('w5_frozen_relay.')]:
        sys.modules.pop(name)
    for filename, digest in GUARD_CLOSURE:
        if filename == '__init__.py':
            load_verified('w5_frozen_relay', relay / filename, digest, package='w5_frozen_relay', is_package=True)
        else:
            stem = filename[:-3]
            module = load_verified('w5_frozen_relay.' + stem, relay / filename, digest, package='w5_frozen_relay')
            setattr(sys.modules['w5_frozen_relay'], stem, module)
    return sys.modules['w5_frozen_relay.demucs_bed']


def frozen_tolerant_check():
    """TolerantSemCheck(frozen sem_idle_pid) for callers that want the v1-style direct check (W4 compatibility)."""
    return load_tolerant().wrap_frozen_guard(load_frozen_guard())


# ---- GPU memory (read only) -------------------------------------------------------------------------------------
class _NvmlMemory(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class Nvml:
    """nvmlDeviceGetMemoryInfo on GPU 0 (read only; what nvidia-smi uses; ~2 us per call; no CUDA context)."""

    def __init__(self, index=0):
        self.dll = ctypes.WinDLL(os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'System32', 'nvml.dll'))
        if self.dll.nvmlInit_v2() != 0:
            raise GpuProbeError('nvmlInit_v2 failed')
        self.handle = ctypes.c_void_p()
        if self.dll.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
            raise GpuProbeError('nvmlDeviceGetHandleByIndex_v2 failed')
        count = ctypes.c_uint(0)
        self.gpus = count.value if self.dll.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0 else 1

    def __call__(self):
        memory = _NvmlMemory()
        code = self.dll.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory))
        if code != 0:
            raise GpuProbeError('nvmlDeviceGetMemoryInfo returned %d' % code)
        return VramSample(memory.free >> 20, memory.total >> 20, memory.used >> 20, time.perf_counter(), self.gpus)

    def close(self):
        try:
            self.dll.nvmlShutdown()
        except Exception:
            pass


def nvidia_vram(clock=time.perf_counter, timeout=10):
    """Fallback: nvidia-smi --query-gpu (read only; no compute). GPU 0; reports the GPU count."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total,memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=timeout,
                             check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
        rows = [line for line in out.strip().splitlines() if line.strip()]
        free, total, used = (int(float(x)) for x in rows[0].split(','))
    except Exception as error:
        raise GpuProbeError('nvidia-smi: %s: %s' % (type(error).__name__, error)) from error
    return VramSample(free, total, used, clock(), len(rows))


def default_vram_probe():
    try:
        return Nvml()
    except Exception:
        return nvidia_vram


class GpuGovernor(threading.Thread):
    """5 Hz sampler with a dip-aware view (review P2-2). Sem's Infernix expert cache dips free VRAM for ~0.4-0.6 s
    about every 30.5 s (to as low as 790 MiB). The budget uses the median over GPU_BASELINE_S; gpu_reserve fires only
    when every sample for GPU_SUSTAIN_S is below the reserve; every dip is recorded."""

    def __init__(self, probe, *, interval=GPU_SAMPLE_S, keep_s=120.0, reserve_mib=RESERVE_MIB, clock=time.perf_counter):
        super().__init__(daemon=True, name='w5-gpu-governor')
        self.probe, self.interval, self.keep_s, self.reserve, self.clock = probe, interval, keep_s, reserve_mib, clock
        self.samples = collections.deque()
        self.error, self.error_at, self.last = None, None, None
        self.first, self.stop_flag, self.mutex = threading.Event(), threading.Event(), threading.Lock()
        self.dips, self._dip = [], None

    def run(self):
        while not self.stop_flag.is_set():
            try:
                sample = self.probe()
                if not isinstance(sample, VramSample):
                    raise GpuProbeError('probe returned %r' % (sample,))
                error = None
            except NoSampleYet:
                self.stop_flag.wait(min(self.interval, 0.02))
                continue
            except Exception as exc:
                sample, error = None, '%s: %s' % (type(exc).__name__, exc)
            now = self.clock()
            with self.mutex:
                if sample is not None:
                    if self.last is None or sample.at != self.last[0]:     # a holder feed may repeat a sample
                        self.samples.append((sample.at, sample.free_mib, sample.total_mib, sample.used_mib))
                        while self.samples and now - self.samples[0][0] > self.keep_s:
                            self.samples.popleft()
                        self._track_dip(sample.at, sample.free_mib)
                    self.last, self.error = (sample.at, sample), None
                else:
                    self.error, self.error_at = error, now
            self.first.set()
            self.stop_flag.wait(self.interval)

    def _track_dip(self, now, free):
        if free < self.reserve:
            if self._dip is None:
                self._dip = {'start': now, 'min_free_mib': free}
            self._dip['min_free_mib'] = min(self._dip['min_free_mib'], free)
        elif self._dip is not None:
            self._dip['duration_s'] = round(now - self._dip['start'], 3)
            self.dips.append(self._dip)
            self.dips = self.dips[-50:]
            self._dip = None

    def snapshot(self, baseline_s=GPU_BASELINE_S):
        now = self.clock()
        with self.mutex:
            last, error = self.last, self.error
            recent = [s for s in self.samples if now - s[0] <= baseline_s]
            window = list(self.samples)
        if error:
            return None, error, None
        if last is None:
            return None, None, None
        at, sample = last
        baseline = int(statistics.median(s[1] for s in recent)) if recent else sample.free_mib
        sample = VramSample(sample.free_mib, sample.total_mib, sample.used_mib, at, sample.gpus, baseline,
                            min(s[1] for s in window), round(now - window[0][0], 3) if window else 0.0)
        return sample, None, now - at

    def sustained_below(self, reserve, sustain_s):
        now = self.clock()
        with self.mutex:
            recent = [s for s in self.samples if now - s[0] <= sustain_s]
            covered = bool(self.samples) and now - self.samples[0][0] >= sustain_s - self.interval
        return covered and len(recent) >= 2 and all(s[1] < reserve for s in recent) and \
            recent[-1][0] - recent[0][0] >= sustain_s - 2 * self.interval

    def stop(self):
        self.stop_flag.set()


class HolderFeed:
    """Watcher-side probe for samples forwarded by the holder (gpu_probe callable; tests / W4 compatibility)."""

    def __init__(self):
        self.sample, self.error, self.mutex = None, None, threading.Lock()

    def put(self, sample=None, error=None):
        with self.mutex:
            self.sample, self.error = sample, error

    def __call__(self):
        with self.mutex:
            sample, error = self.sample, self.error
        if error:
            raise GpuProbeError(error)
        if sample is None:
            raise NoSampleYet('no sample from the holder yet')
        return sample


# ---- directory change watch (notify-only reads) ------------------------------------------------------------------
class DirWatch:
    """Overlapped ReadDirectoryChangesW on a DIRECTORY (never a handle on the watched files). collect() returns the
    set of watched names that changed (all of them on buffer overflow)."""
    mode = 'notify'
    FILTER = 0x1 | 0x8 | 0x10        # FILE_NAME | SIZE | LAST_WRITE

    def __init__(self, directory, names):
        self.directory, self.names = Path(directory), {n.lower() for n in names}
        self.handle = _CreateFileW(str(self.directory), 0x1, _SHARE_ALL, None, _OPEN_EXISTING,
                                   0x02000000 | 0x40000000, None)   # LIST_DIRECTORY; BACKUP_SEMANTICS|OVERLAPPED
        if not _ok_handle(self.handle):
            raise _winerror()
        self.event = _CreateEventW(None, True, False, None)
        if not _ok_handle(self.event):
            error = _winerror()
            _CloseHandle(self.handle)
            raise error
        self.buffer = ctypes.create_string_buffer(64 * 1024)
        self.overlapped = _OVERLAPPED()
        self.overlapped.hEvent = self.event
        self.issued = False
        self.unrelated = self.overflows = 0
        self._issue()

    def _issue(self):
        _ResetEvent(self.event)
        if not _ReadDirectoryChangesW(self.handle, self.buffer, len(self.buffer), False, self.FILTER, None,
                                      ctypes.byref(self.overlapped), None):
            raise _winerror()
        self.issued = True

    def collect(self):
        size = wintypes.DWORD(0)
        self.issued = False
        if not _GetOverlappedResult(self.handle, ctypes.byref(self.overlapped), ctypes.byref(size), False):
            raise _winerror()
        if size.value == 0:
            self.overflows += 1
            matched = set(self.names)
        else:
            matched = self._names(size.value) & self.names
            if not matched:
                self.unrelated += 1
        self._issue()
        return matched

    def _names(self, size):
        raw, offset, found = self.buffer.raw[:size], 0, set()
        while True:
            next_offset, _action, length = (int.from_bytes(raw[offset + i:offset + i + 4], 'little') for i in (0, 4, 8))
            found.add(raw[offset + 12:offset + 12 + length].decode('utf-16-le', 'replace').lower())
            if not next_offset:
                return found
            offset += next_offset

    def close(self):
        if self.handle:
            if self.issued:
                _CancelIoEx(self.handle, ctypes.byref(self.overlapped))
                size = wintypes.DWORD(0)
                _GetOverlappedResult(self.handle, ctypes.byref(self.overlapped), ctypes.byref(size), True)
            _CloseHandle(self.handle)
            _CloseHandle(self.event)
            self.handle = self.event = None


class DeafWatch(DirWatch):
    """TEST ONLY (review E5): every change is 'missed'; only the periodic re-read can see it."""
    mode = 'deaf'

    def collect(self):
        super().collect()
        return set()


def wait_changes(watches, timeout):
    """Wait up to `timeout` for a watched name to change; returns {(watch_index, name)}. Unrelated events do NOT end
    the wait (review P3-2). Matching events already queued when the first one is collected are drained without
    waiting (0 ms), so one Sem rename usually costs one read (at most two, both right after the rename)."""
    if not watches:
        time.sleep(max(0.0, timeout))
        return set()
    events = (_H * len(watches))(*[w.event for w in watches])
    deadline = time.perf_counter() + max(0.0, timeout)
    matched, coalesce_until = set(), None
    while True:
        now = time.perf_counter()
        limit = coalesce_until if matched else deadline
        if matched:
            ms = 0
        elif now >= limit:
            return matched
        else:
            ms = max(1, int((limit - now) * 1000 + 0.999))
        result = _WaitForMultipleObjects(len(watches), events, False, ms)
        if result == _WAIT_TIMEOUT:
            if matched:
                return matched
            continue
        index = result - _WAIT_OBJECT_0
        if result == _WAIT_FAILED or not 0 <= index < len(watches):
            raise _winerror()
        changed = watches[index].collect()
        if changed and not matched:
            coalesce_until = time.perf_counter()             # drain only what is already signalled
        matched |= {(index, name) for name in changed}


# ---- Job object for the holder's GPU children (review P1-1) -------------------------------------------------------
def _create_kill_on_close_job(name):
    job = _CreateJobObjectW(None, name)
    if not _ok_handle(job):
        raise _winerror()
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        _CloseHandle(job)
        raise OSError('job object %s already exists' % name)
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
    if not _SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = _winerror()
        _CloseHandle(job)
        raise error
    return job


def job_active_processes(job):
    info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
    if not _QueryInformationJobObject(job, 1, ctypes.byref(info), ctypes.sizeof(info), None):
        raise _winerror()
    return {'active': info.ActiveProcesses, 'total': info.TotalProcesses}


_ntdll = ctypes.WinDLL('ntdll')
_NtResumeProcess = _ntdll.NtResumeProcess
_NtResumeProcess.restype, _NtResumeProcess.argtypes = ctypes.c_long, [_H]


def _resume_child(child):
    """Resume an own child created with CREATE_SUSPENDED. NtResumeProcess on the Popen's own process handle; the
    thread-snapshot path is the fallback (retried: a fresh snapshot once missed the new thread under load)."""
    if _NtResumeProcess(int(child._handle)) == 0:
        return 1
    for _ in range(5):
        resumed = _resume_main_thread(child.pid)
        if resumed:
            return resumed
        time.sleep(0.01)
    return 0


def _resume_main_thread(pid):
    snap = _CreateToolhelp32Snapshot(0x4, 0)       # TH32CS_SNAPTHREAD
    if not _ok_handle(snap):
        raise _winerror()
    try:
        entry = _THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        resumed = 0
        ok = _Thread32First(snap, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == pid:
                thread = _OpenThread(0x2, False, entry.th32ThreadID)   # THREAD_SUSPEND_RESUME
                if _ok_handle(thread):
                    if _ResumeThread(thread) != 0xFFFFFFFF:
                        resumed += 1
                    _CloseHandle(thread)
            ok = _Thread32Next(snap, ctypes.byref(entry))
        return resumed
    finally:
        _CloseHandle(snap)


# ---- Sem process presence (absent mode; read only: names, image paths, command lines) -----------------------------
class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ProcessID', wintypes.DWORD),
                ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', wintypes.DWORD),
                ('cntThreads', wintypes.DWORD), ('th32ParentProcessID', wintypes.DWORD),
                ('pcPriClassBase', wintypes.LONG), ('dwFlags', wintypes.DWORD), ('szExeFile', ctypes.c_wchar * 260)]


_Process32FirstW = _proto('Process32FirstW', wintypes.BOOL, _H, ctypes.POINTER(_PROCESSENTRY32W))
_Process32NextW = _proto('Process32NextW', wintypes.BOOL, _H, ctypes.POINTER(_PROCESSENTRY32W))
_QueryFullProcessImageNameW = _proto('QueryFullProcessImageNameW', wintypes.BOOL, _H, wintypes.DWORD,
                                     wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))
_NtQueryInformationProcess = _ntdll.NtQueryInformationProcess
_NtQueryInformationProcess.restype = ctypes.c_long
_NtQueryInformationProcess.argtypes = [_H, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
_QUERY_LIMITED = 0x1000      # PROCESS_QUERY_LIMITED_INFORMATION: read-only (image path, command line)


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [('Length', wintypes.USHORT), ('MaximumLength', wintypes.USHORT), ('Buffer', ctypes.c_void_p)]


def _norm(text):
    return (text or '').replace('/', '\\').lower()


_NtQuerySystemInformation = _ntdll.NtQuerySystemInformation
_NtQuerySystemInformation.restype = ctypes.c_long
_NtQuerySystemInformation.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
_SPI_BUFFER = [None]
_SPI_ROW = struct.Struct('<I28xQ16xH6xQ8xQQ')     # Next, CreateTime, ImageName.Length, ImageName.Buffer, Pid, PPid


def _process_snapshot_nt():
    """One NtQuerySystemInformation(SystemProcessInformation) call; x64 SYSTEM_PROCESS_INFORMATION layout:
    +0 NextEntryOffset, +32 CreateTime, +56 ImageName (Length, MaximumLength, Buffer at +64), +80 UniqueProcessId,
    +88 InheritedFromUniqueProcessId. Returns [(pid, ppid, image_name, create_time)]."""
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise OSError('x64 layout only')
    size = 1 << 20 if _SPI_BUFFER[0] is None else len(_SPI_BUFFER[0])
    for _ in range(8):
        buffer = _SPI_BUFFER[0] if _SPI_BUFFER[0] is not None and len(_SPI_BUFFER[0]) >= size else \
            ctypes.create_string_buffer(size)
        needed = wintypes.ULONG(0)
        status = _NtQuerySystemInformation(5, buffer, len(buffer), ctypes.byref(needed)) & 0xFFFFFFFF
        if status == 0xC0000004:                 # STATUS_INFO_LENGTH_MISMATCH
            size = max(needed.value + (256 << 10), len(buffer) * 2)
            _SPI_BUFFER[0] = None
            continue
        if status != 0:
            raise OSError('NtQuerySystemInformation status %#x' % status)
        _SPI_BUFFER[0] = buffer
        base, limit, view = ctypes.addressof(buffer), needed.value or len(buffer), memoryview(buffer)
        rows, offset, unpack = [], 0, _SPI_ROW.unpack_from
        while True:
            nxt, create, length, pointer, pid, ppid = unpack(buffer, offset)
            at = pointer - base
            if pointer and length and 0 <= at and at + length <= limit:
                name = view[at:at + length].tobytes().decode('utf-16-le', 'replace')
            elif pointer and length:
                name = ctypes.wstring_at(pointer, length // 2)
            else:
                name = ''
            rows.append((pid, ppid, name, create))
            if not nxt:
                break
            offset += nxt
        if len(rows) < 2:
            raise OSError('implausible process list')
        return rows
    raise OSError('process list did not fit')


def _process_snapshot_toolhelp():
    snap = _CreateToolhelp32Snapshot(0x2, 0)       # TH32CS_SNAPPROCESS (fallback; no create time)
    if not _ok_handle(snap):
        raise _winerror()
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        rows = []
        ok = _Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            rows.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile, 0))
            ok = _Process32NextW(snap, ctypes.byref(entry))
        if not rows:
            raise OSError('empty process snapshot')
        return rows
    finally:
        _CloseHandle(snap)


def _process_snapshot():
    try:
        return _process_snapshot_nt()
    except OSError:
        return _process_snapshot_toolhelp()


_GetExitCodeProcess = _proto('GetExitCodeProcess', wintypes.BOOL, _H, ctypes.POINTER(wintypes.DWORD))
EXITED = ('exited', 'exited')


def _process_identity(pid):
    """(image path, command line) read with PROCESS_QUERY_LIMITED_INFORMATION; None where unreadable.
    Returns EXITED for a process that already ended (still listed while handles to it are open) or whose PID is
    gone (ERROR_INVALID_PARAMETER)."""
    handle = _OpenProcess(_QUERY_LIMITED, False, pid)
    if not _ok_handle(handle):
        return EXITED if ctypes.get_last_error() == 87 else (None, None)
    try:
        code = wintypes.DWORD(0)
        if _GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:     # not STILL_ACTIVE
            return EXITED
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        image = buffer.value if _QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)) else None
        command, needed = None, wintypes.ULONG(0)
        _NtQueryInformationProcess(handle, 60, None, 0, ctypes.byref(needed))   # ProcessCommandLineInformation
        if needed.value:
            raw = ctypes.create_string_buffer(needed.value)
            if _NtQueryInformationProcess(handle, 60, raw, needed.value, ctypes.byref(needed)) == 0:
                us = _UNICODE_STRING.from_buffer(raw)
                if us.Buffer and us.Length:
                    command = ctypes.wstring_at(us.Buffer, us.Length // 2)
        return image, command
    finally:
        _CloseHandle(handle)


_K32EnumProcesses = _proto('K32EnumProcesses', wintypes.BOOL, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
                           ctypes.POINTER(wintypes.DWORD))
SEM_FULL_SCAN_S = 2.0        # absent mode: full (pid, create time) reconcile period; in between only new PIDs are read


def _enum_pids():
    size = 4096
    while True:
        array = (wintypes.DWORD * size)()
        used = wintypes.DWORD(0)
        if not _K32EnumProcesses(array, ctypes.sizeof(array), ctypes.byref(used)):
            raise _winerror()
        count = used.value // 4
        if count < size:
            return set(array[:count])
        size *= 2


class SemPresence:
    """Read-only scan for Sem-owned processes (absent mode). Like the frozen relay's
    ResourceBudget.watched_process_present it uses process names, and adds the image path / command line:
    * every scan: K32EnumProcesses (PIDs only, ~0.06 ms); each NEW pid is identified once (image path + command
      line with PROCESS_QUERY_LIMITED_INFORMATION). If it cannot be opened, its name comes from a system snapshot
      (a Sem-relevant name that cannot be verified is UNVERIFIED: fail closed);
    * every SEM_FULL_SCAN_S: a full NtQuerySystemInformation snapshot reconciles (pid, create time), so a reused
      PID is re-identified."""

    def __init__(self, rules=None, exclude_pids=(), full_scan_s=SEM_FULL_SCAN_S):
        rules = rules or SEM_PROCESS_RULES
        self.always = {n.lower() for n in rules['names_always']}
        self.checked = {n.lower() for n in rules['names_checked']}
        self.roots = [_norm(r) for r in rules['exe_roots']]
        self.markers = [_norm(m) for m in rules['cmd_markers']]
        self.exclude = set(int(p) for p in exclude_pids)
        self.full_scan_s, self.last_full = float(full_scan_s), None
        self.known, self.created = {}, {}         # pid -> (verdict, detail); pid -> create time (full scans)
        self.scans = self.full_scans = self.identified = 0

    def _verdict(self, pid, name, image, command):
        name = (name or '').lower()
        if name in self.always:
            return 'sem', '%s pid %d (image name)' % (name, pid)
        if name not in self.checked:
            return 'other', None
        if image is None and command is None:
            return 'unverified', '%s pid %d: identity unreadable' % (name, pid)
        ni, nc = _norm(image), _norm(command)
        if any(ni.startswith(r) for r in self.roots):
            return 'sem', '%s pid %d: image %s' % (name, pid, image)
        for marker in self.markers:
            if marker in nc:
                return 'sem', '%s pid %d: command line has %s' % (name, pid, marker)
        return 'other', None

    def _identify(self, pid, snapshot_name=None):
        identity = _process_identity(pid)
        self.identified += 1
        if identity is EXITED:
            return 'other', None
        image, command = identity
        if image is None and command is None:
            if snapshot_name is None:
                return None                         # caller resolves the name from a snapshot
            return self._verdict(pid, snapshot_name, None, None)
        name = snapshot_name or (image.replace('/', '\\').rsplit('\\', 1)[-1] if image else '')
        return self._verdict(pid, name, image, command)

    def _full(self):
        rows = _process_snapshot()
        self.full_scans += 1
        known, created = {}, {}
        for pid, ppid, name, create in rows:
            if pid in self.exclude or pid == 0:
                known[pid] = ('other', None)
                continue
            if pid in self.known and self.created.get(pid) == create:
                known[pid] = self.known[pid]
            elif name.lower() in self.always or name.lower() in self.checked:
                known[pid] = self._identify(pid, name)
            else:
                known[pid] = ('other', None)
            created[pid] = create
        self.known, self.created = known, created

    def scan(self):
        """[(verdict, detail)] for every Sem or unverified process now. Raises OSError if the scan failed."""
        now = time.perf_counter()
        self.scans += 1
        if self.last_full is None or now - self.last_full >= self.full_scan_s:
            self._full()
            self.last_full = now
        else:
            pids = _enum_pids()
            for pid in [p for p in self.known if p not in pids]:
                del self.known[pid]
                self.created.pop(pid, None)
            unresolved = []
            for pid in pids:
                if pid in self.known:
                    continue
                if pid in self.exclude:
                    self.known[pid] = ('other', None)
                    continue
                verdict = self._identify(pid)
                if verdict is None:
                    unresolved.append(pid)
                else:
                    self.known[pid] = verdict
            if unresolved:
                names = {pid: name for pid, _ppid, name, _create in _process_snapshot()}
                for pid in unresolved:
                    if pid in names:
                        self.known[pid] = self._verdict(pid, names[pid], None, None)
        return [v for v in self.known.values() if v[0] != 'other']


def sem_processes(rules=None):
    """Module-level, read-only: the Sem-owned (or unverifiable) processes running now (for operators/tests)."""
    return SemPresence(rules, exclude_pids=(os.getpid(),)).scan()


# ---- out-of-holder API (any same-user process; needs only the published names; no GIL hop via the holder) -------
def assign_to_job(job_name, child):
    """Put an already running OWN child (subprocess.Popen) into the lease's Job. Prefer spawn_in_job()."""
    if not isinstance(child, subprocess.Popen):
        raise TypeError('assign_to_job() takes a subprocess.Popen created by this process')
    job = _OpenJobObjectW(_JOB_ASSIGN | _JOB_QUERY, False, str(job_name))
    if not _ok_handle(job):
        raise LeaseError('lease Job %s is gone (lease ended): %s' % (job_name, _winerror()))
    try:
        if not _AssignProcessToJobObject(job, int(child._handle)):
            raise _winerror()
    finally:
        _CloseHandle(job)
    return child


def spawn_in_job(job_name, args, **popen_kwargs):
    """Start a GPU child INSIDE the lease's Job: created suspended, assigned, then resumed (no escape window).
    Its own subprocesses are Job members too (no breakaway). Raises LeaseError if the lease already ended."""
    flags = popen_kwargs.pop('creationflags', 0) | _CREATE_SUSPENDED
    child = subprocess.Popen(args, creationflags=flags, **popen_kwargs)
    try:
        assign_to_job(job_name, child)
        if _resume_child(child) < 1:
            raise OSError('could not resume the suspended child')
    except BaseException:
        _TerminateProcess(int(child._handle), 0xC0DE)    # our own child, created suspended, never ran
        child.wait(5)
        raise
    return child


def _event_state(name, timeout_s):
    handle = _OpenEventW(_SYNCHRONIZE, False, str(name))
    if not _ok_handle(handle):
        return True        # the event is gone: the watcher ended (lease over) -> treat as cancelled (fail closed)
    try:
        ms = 0xFFFFFFFF if timeout_s is None else max(0, int(timeout_s * 1000))
        return _WaitForSingleObject(handle, ms) == _WAIT_OBJECT_0
    finally:
        _CloseHandle(handle)


def cancel_is_set(cancel_event_name):
    """True once the watcher set the lease's cancel event (preempt or release) or the lease is gone."""
    return _event_state(cancel_event_name, 0)


def wait_cancel(cancel_event_name, timeout=None):
    """Kernel wait on the lease's cancel event (for GPU children). True when cancelled / lease gone."""
    return _event_state(cancel_event_name, timeout)


def request_kill(kill_event_name):
    """Ask the WATCHER process to end every process in the lease's Job now (no grace) and preempt with
    'killed_on_request'. Returns False if the lease is already gone."""
    handle = _OpenEventW(_EVENT_MODIFY_STATE, False, str(kill_event_name))
    if not _ok_handle(handle):
        return False
    try:
        return bool(_SetEvent(handle))
    finally:
        _CloseHandle(handle)


# =============================================================================================================
# WATCHER PROCESS
# =============================================================================================================
class Watcher:
    def __init__(self, cfg):
        self.cfg, self.clock = cfg, time.perf_counter
        self.lease_id = cfg['lease_id']
        self.status = Path(cfg['status_file'])
        self.out_lock = threading.Lock()
        self.commands = queue.Queue()
        self.feed = HolderFeed() if cfg.get('gpu_source') == 'holder' else None
        self.telemetry = {'reads': 0, 'frozen_checks': 0, 'locked_reads': 0, 'change_events': 0, 'polls': 0,
                          'proofs': 0, 'max_proof_gap_s': 0.0, 'reverify_reads': 0, 'settle_reads': 0,
                          'watch_mode': None, 'unrelated_events': 0, 'renewals': 0}
        self.test = str(cfg.get('mutex_name', '')).startswith(TEST_MUTEX_PREFIX)
        self.copy_dir = Path(tempfile.mkdtemp(prefix='w5-lease-copy-'))
        self.copy_path = self.copy_dir / 'status.json'
        self.mutex = self.job = self.cancel = self.holder = self.kill = None
        self.mutex_owned = False
        self.preempt_event = None
        self.job_closed = None
        self.state, self.last_read, self.last_proof, self.last_frozen = None, 0.0, 0.0, 0.0
        self.sem_pid = None

    # -- output
    def emit(self, kind, **fields):
        line = json.dumps(dict(type=kind, t=self.clock(), **fields))
        with self.out_lock:
            try:
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
            except (OSError, ValueError):
                pass

    def record(self, state, **extra):
        state_dir = self.cfg.get('state_dir')
        if not state_dir:
            return
        try:
            target = Path(state_dir) / 'lease.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name('lease.json.%s.tmp' % uuid.uuid4().hex[:8])
            body = {'lease_id': self.lease_id, 'state': state, 'mode': 'no-release', 'holder': self.cfg.get('holder'),
                    'holder_pid': self.cfg.get('holder_pid'), 'watcher_pid': os.getpid(),
                    'budget_mib': self.cfg.get('budget_mib'), 'sem_pid': self.sem_pid,
                    'mutex': self.cfg.get('mutex_name'), 'written_at': datetime.now(timezone.utc).isoformat()}
            body.update(extra)
            tmp.write_text(json.dumps(body, indent=2, default=str), encoding='utf-8')
            for attempt in range(4):
                try:
                    os.replace(tmp, target)
                    return
                except PermissionError:
                    time.sleep(0.005 * (attempt + 1))
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    # -- stdin commands (from the holder)
    def _stdin(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line.startswith('GPU ') and self.feed is not None:
                try:
                    d = json.loads(line[4:])
                    if 'error' in d:
                        self.feed.put(error=d['error'])
                    else:
                        self.feed.put(VramSample(int(d['free']), int(d['total']), int(d['used']), self.clock()))
                except Exception as error:
                    self.feed.put(error='bad GPU line: %s' % error)
                continue
            self.commands.put(line)
        self.commands.put('EOF')

    # -- status evaluation (one read of Sem's file per version; frozen verdict on a copy of the same bytes)
    def evaluate_version(self, tolerant=False):
        started = self.clock()
        try:
            if self.test and self.cfg.get('test_locked_reads_flag') and os.path.exists(self.cfg['test_locked_reads_flag']):
                raise PermissionError(13, 'Permission denied (test flag)', str(self.status))
            data = self.tolerant_read(self.status) if tolerant else read_shared(self.status)
        except Exception as error:
            if is_transient_lock(error) or isinstance(error, FileNotFoundError) \
                    or type(error).__name__ == 'StatusUnreadable':
                self.telemetry['locked_reads'] += 1
                return None, ('status_unreadable', '%s: %s' % (type(error).__name__, error))
            return False, ('status_unreadable', '%s: %s' % (type(error).__name__, error))
        self.telemetry['reads'] += 1
        try:
            state = json.loads(data.decode('utf-8-sig'))
        except ValueError as error:
            return False, ('status_unreadable', 'malformed JSON: %s' % error)
        bad = classify_state(state, None, self.cfg['max_status_age_s'])
        if bad:
            return False, bad
        if self.sem_pid is not None and state['pid'] != self.sem_pid:
            return False, ('sem_pid_changed', '%s -> %s' % (self.sem_pid, state['pid']))
        if self.watch.mode != 'poll' or started - self.last_frozen >= self.cfg['frozen_interval_s']:
            try:
                self.copy_path.write_bytes(data)
                pid = self.frozen.sem_idle_pid(self.copy_path)
                if self.test and self.cfg.get('test_frozen_disagree_flag') and \
                        os.path.exists(self.cfg['test_frozen_disagree_flag']):
                    raise self.frozen.ResourceAbort('Sem state is not fresh idle (test flag)')
            except Exception as error:
                return False, ('sem_not_idle', 'frozen guard: %s: %s' % (type(error).__name__, error))
            self.telemetry['frozen_checks'] += 1
            self.last_frozen = started
            if pid != state['pid']:
                return False, ('sem_pid_changed', 'frozen guard %s vs status %s' % (pid, state['pid']))
        self.state, self.last_read = state, started
        return True, state['pid']

    def time_rule(self):
        return classify_state(self.state, None, self.cfg['max_status_age_s']) if self.state is not None else \
            ('status_unreadable', 'no status read yet')

    def wait_for_change(self, timeout):
        changed = wait_changes(self.watches, timeout)
        if changed:
            self.telemetry['change_events'] += 1
        return changed

    def _revoked(self):
        path = self.cfg.get('revoke_file')
        if not path:
            return None
        try:
            data = json.loads(read_shared(path).decode('utf-8-sig'))
        except FileNotFoundError:
            return None
        except Exception as error:
            if is_transient_lock(error):
                return None
            return 'revoke file unreadable: %s' % error
        if isinstance(data, dict) and (data.get('revoked') or data.get('state') in ('revoked', 'refused')):
            return str(data.get('reason') or data.get('revoked') or data.get('state'))
        return None

    # -- phases
    def setup(self):
        try:
            _SetPriorityClass(_GetCurrentProcess(), _ABOVE_NORMAL)
        except Exception:
            pass
        try:   # 1 ms timer resolution for THIS process (Windows 10 2004+ scopes it per process); default is 15.6 ms
            self.timer_period = ctypes.WinDLL('winmm').timeBeginPeriod(1) == 0
        except Exception:
            self.timer_period = False
        self.holder = _OpenProcess(_SYNCHRONIZE, False, int(self.cfg['holder_pid']))
        if not _ok_handle(self.holder):
            raise LeaseUnavailable('holder_gone', 'cannot open holder process %s' % self.cfg['holder_pid'])
        self.mutex = _CreateMutexW(None, False, self.cfg['mutex_name'])
        if not _ok_handle(self.mutex):
            raise LeaseUnavailable('lock_unavailable', str(_winerror()))
        result = _WaitForSingleObject(self.mutex, 0)
        if result == _WAIT_TIMEOUT:
            raise LeaseUnavailable('lease_held_elsewhere', self.cfg['mutex_name'])
        if result not in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
            raise LeaseUnavailable('lock_unavailable', 'wait returned %#x' % result)
        self.mutex_owned = True
        self.telemetry['previous_holder_abandoned'] = result == _WAIT_ABANDONED
        self.job = _create_kill_on_close_job(self.cfg['job_name'])
        self.cancel = _CreateEventW(None, True, False, self.cfg['cancel_name'])
        if not _ok_handle(self.cancel) or ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            raise LeaseUnavailable('lock_unavailable', 'cancel event name collision')
        self.kill = _CreateEventW(None, True, False, self.cfg['kill_name'])
        if not _ok_handle(self.kill) or ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
            raise LeaseUnavailable('lock_unavailable', 'kill event name collision')
        self.frozen = load_frozen_guard(self.cfg.get('guard_packages') or GUARD_PACKAGES)
        self.presence = SemPresence(self.cfg.get('sem_rules'),
                                    exclude_pids=(os.getpid(), os.getppid(), int(self.cfg['holder_pid'])))
        self.last_scan = 0.0
        tolerant = load_tolerant(self.cfg.get('tolerant_path') or TOLERANT_PATH)
        self.tolerant_read = tolerant.TolerantSemCheck(read_shared)
        threading.Thread(target=self._stdin, daemon=True, name='w5-watcher-stdin').start()
        probe = self.feed if self.feed is not None else default_vram_probe()
        self.governor = GpuGovernor(probe, interval=self.cfg['gpu_sample_s'], reserve_mib=self.cfg['reserve_mib'])
        self.governor.start()
        mode = self.cfg.get('watch_mode', 'notify')
        if mode == 'deaf' and not self.test:
            raise LeaseUnavailable('bad_config', 'deaf watch is test-only')
        self.watch, self.watches = None, []
        try:
            if mode == 'poll':
                raise OSError('poll mode requested')
            cls = DeafWatch if mode == 'deaf' else DirWatch
            self.watch = cls(self.status.parent, [self.status.name])
            self.watches = [self.watch]
            revoke = self.cfg.get('revoke_file')
            if revoke:
                if Path(revoke).parent.resolve() == self.status.parent.resolve():
                    self.watch.names.add(Path(revoke).name.lower())
                else:
                    self.watches.append(DirWatch(Path(revoke).parent, [Path(revoke).name]))
        except OSError as error:
            if mode != 'poll':
                self.telemetry['watch_setup_error'] = str(error)
            self.watch = types.SimpleNamespace(mode='poll')
            self.watches = []
        self.telemetry['watch_mode'] = self.watch.mode

    def settle(self):
        """Idle continuously for settle_s with one PID. Notify-only: a read happens at start, after each change of
        status.json, and after reverify_s without changes; nothing is polled (review P3-1)."""
        settle_s, wait_s = self.cfg['settle_s'], self.cfg['wait_s']
        deadline = self.clock() + wait_s
        idle_since, last_bad, need_read, next_read = None, None, True, 0.0
        while True:
            now = self.clock()
            if now >= next_read and (need_read or self.watch.mode == 'poll'
                                     or now - self.last_read >= self.cfg['reverify_s']):
                next_read = now + 0.05                       # a failed (locked / missing) read is retried >= 50 ms later
                self.telemetry['settle_reads'] += 1
                ok, value = self.evaluate_version(tolerant=True)
                need_read = False
                if ok is True:
                    if self.sem_pid is None:
                        self.sem_pid = value
                elif ok is None:
                    need_read = True
                    last_bad = value
                else:
                    last_bad = value
                    if value[0] == 'sem_pid_changed' and wait_s > 0:
                        self.sem_pid = None
                    idle_since = None
                    self.state = None
            bad = self.time_rule() if self.state is not None else (last_bad or ('status_unreadable', 'unread'))
            if bad is None:
                if idle_since is None:
                    idle_since = self.last_read
                if self.clock() - idle_since >= settle_s:
                    return self.sem_pid
            else:
                idle_since = None
                last_bad = bad
                if self.clock() >= deadline:
                    if wait_s > 0:
                        raise LeaseUnavailable('timeout', 'waited %.1f s; last %s: %s' % (wait_s, bad[0], bad[1]),
                                               last=LeaseUnavailable(*bad))
                    raise LeaseUnavailable(*bad)
            if self.cmd_eof():
                raise LeaseUnavailable('holder_gone', 'holder closed the pipe during acquire')
            remaining = [max(next_read - self.clock(), self.cfg['reverify_s'] - (self.clock() - self.last_read))]
            if idle_since is not None:
                remaining.append(settle_s - (self.clock() - idle_since))
            if bad is not None:
                remaining.append(deadline - self.clock())
            timeout = max(0.001, min(self.cfg['poll_s'] if self.watch.mode == 'poll' else 0.25, *remaining))
            if self.wait_for_change(timeout):
                need_read = True

    def status_fresh(self):
        """Absent mode: a status.json written in the last 15 s means Sem may be running (no synthesized status)."""
        try:
            state = json.loads(read_shared(self.status).decode('utf-8-sig'))
        except FileNotFoundError:
            return None
        except Exception as error:
            return 'status.json present but unreadable: %s: %s' % (type(error).__name__, error)
        try:
            at = datetime.fromisoformat(str(state['at']).replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - at).total_seconds()
        except Exception as error:
            return 'status.json present with bad "at": %s' % error
        if age <= STALE_AFTER_S:
            return 'status.json written %.1f s ago (pid %s)' % (age, state.get('pid'))
        return None

    def absent_check(self):
        """None when Sem is verified absent now, else (reason, detail)."""
        try:
            found = self.presence.scan()
        except OSError as error:
            return 'sem_presence_unverified', 'process scan failed: %s' % error
        self.telemetry['sem_scans'] = self.presence.scans
        sem = [d for v, d in found if v == 'sem']
        if sem:
            return 'sem_started', '; '.join(sem[:3])
        unverified = [d for v, d in found if v == 'unverified']
        if unverified:
            return 'sem_presence_unverified', '; '.join(unverified[:3])
        return None

    def settle_absent(self):
        """Absent mode admission: settle_s of consecutive scans with no Sem-owned process and no fresh status."""
        settle_s, wait_s = self.cfg['settle_s'], self.cfg['wait_s']
        deadline = self.clock() + wait_s
        clear_since, status_checked, status_bad = None, -1e9, None
        while True:
            now = self.clock()
            bad = self.absent_check()
            if bad is None:
                if now - status_checked >= self.cfg['reverify_s']:
                    status_bad, status_checked = self.status_fresh(), now
                if status_bad:
                    bad = ('sem_started', status_bad)
            if bad is None:
                clear_since = now if clear_since is None else clear_since
                self.last_proof = now
                if now - clear_since >= settle_s:
                    return None
            else:
                clear_since = None
                if now >= deadline:
                    if wait_s > 0:
                        raise LeaseUnavailable('timeout', 'waited %.1f s; last %s: %s' % (wait_s, bad[0], bad[1]),
                                               last=LeaseUnavailable(*bad))
                    raise LeaseUnavailable(*bad)
            if self.cmd_eof():
                raise LeaseUnavailable('holder_gone', 'holder closed the pipe during acquire')
            if self.wait_for_change(self.cfg['sem_scan_s']):
                status_checked = -1e9          # status.json changed: re-check it now

    def probe_absent(self):
        t = self.telemetry
        started = self.clock()
        t['polls'] += 1
        if self.need_read:                      # status.json was written: Sem (or something) is writing it
            self.need_read = False
            fresh = self.status_fresh()
            if fresh:
                return 'sem_started', fresh
        if started - self.last_scan >= self.cfg['sem_scan_s']:
            self.last_scan = started
            bad = self.absent_check()
            if bad:
                return bad
            t['max_proof_gap_s'] = max(t['max_proof_gap_s'], round(started - self.last_proof, 6))
            self.last_proof = started
            t['proofs'] += 1
        return self.common_checks(self.clock())

    def cmd_eof(self):
        try:
            item = self.commands.get_nowait()
        except queue.Empty:
            return False
        if item == 'EOF':
            self.commands.put('EOF')
            return True
        self.commands.put(item)
        return False

    def budget_check(self):
        budget, reserve = self.cfg['budget_mib'], self.cfg['reserve_mib']
        if not self.governor.first.wait(self.cfg['first_gpu_timeout_s']):
            raise LeaseUnavailable('gpu_probe_failed', 'no GPU sample within %.1f s' % self.cfg['first_gpu_timeout_s'])
        need = self.cfg['gpu_baseline_s']
        deadline = self.clock() + need + 1.0
        while True:
            sample, error, age = self.governor.snapshot(need)
            if error:
                raise LeaseUnavailable('gpu_probe_failed', error)
            if sample is not None and sample.window_s >= need - self.cfg['gpu_sample_s'] or self.clock() >= deadline:
                break
            time.sleep(self.cfg['gpu_sample_s'])
        if sample is None:
            raise LeaseUnavailable('gpu_probe_failed', 'no sample')
        if sample.baseline_mib - reserve < budget:
            raise LeaseUnavailable('insufficient_vram', 'baseline free %d MiB (median %.1f s; latest %d, min %d) - '
                                   'reserve %d MiB < budget %d MiB' % (sample.baseline_mib, need, sample.free_mib,
                                                                       sample.min_mib, reserve, budget))
        return sample

    def preempt(self, reason, detail):
        if self.preempt_event is not None:
            return
        _SetEvent(self.cancel)                              # first: native children / holder waits see it now
        now = self.clock()
        self.preempt_event = {'reason': reason, 'detail': detail, 'lease_id': self.lease_id, 'detected_at': now,
                              'last_idle_proof_at': self.last_proof, 'latency_bound_s': LATENCY_BOUND_S}
        self.emit('preempt', event=self.preempt_event)
        self.record('preempted', event=self.preempt_event)
        if self.cfg['grace_s'] <= 0:
            self.close_job('preempt')

    def close_job(self, why):
        if self.job_closed is not None or not self.job:
            return
        started = self.clock()
        before = job_active_processes(self.job)
        _TerminateJobObject(self.job, 0xC0DE)               # only processes the holder spawned/adopted into this job
        while job_active_processes(self.job)['active'] and self.clock() - started < self.cfg['drain_timeout_s']:
            time.sleep(0.002)
        after = job_active_processes(self.job)
        while after['active']:                               # never free the job/mutex while a GPU child may run
            self.emit('drain_pending', active=after['active'], waited_s=round(self.clock() - started, 1))
            time.sleep(1.0)
            after = job_active_processes(self.job)
        _CloseHandle(self.job)
        self.job = None
        self.job_closed = {'why': why, 'at': started, 'active_before': before['active'],
                           'total_assigned': before['total'], 'active_after': after['active'],
                           'stop_seconds': round(self.clock() - started, 4)}
        self.emit('job_closed', **self.job_closed)

    def probe_once(self):
        t = self.telemetry
        started = self.clock()
        t['polls'] += 1
        if self.need_read or self.watch.mode == 'poll' or started - self.last_read >= self.cfg['reverify_s']:
            if not self.need_read and self.watch.mode != 'poll':
                t['reverify_reads'] += 1
            ok, value = self.evaluate_version()
            if ok is False:
                return value
            self.need_read = ok is None
        if not self.need_read:
            bad = self.time_rule()
            if bad:
                return bad
            t['max_proof_gap_s'] = max(t['max_proof_gap_s'], round(started - self.last_proof, 6))
            self.last_proof = started
            t['proofs'] += 1
        now = self.clock()
        if now - self.last_proof > self.cfg['proof_deadline_s']:
            return 'status_unreadable', 'no idle proof for %.0f ms' % ((now - self.last_proof) * 1000)
        return self.common_checks(now)

    def common_checks(self, now):
        if self.revoke_due:
            self.revoke_due = False
            revoked = self._revoked()
            if revoked:
                return 'revoked', revoked
        if now > self.expires_at:
            return 'lease_expired', 'no renew within %.1f s' % self.cfg['ttl_s']
        sample, error, age = self.governor.snapshot(self.cfg['gpu_baseline_s'])
        if error:
            return 'gpu_probe_failed', error
        if age is None or age > self.cfg['gpu_max_age_s']:
            return 'gpu_probe_stale', 'GPU sample age %s s' % (None if age is None else round(age, 2))
        if self.governor.sustained_below(self.cfg['reserve_mib'], self.cfg['gpu_sustain_s']):
            return 'gpu_reserve', 'free < reserve %d MiB for %.1f s (latest %d MiB)' % (
                self.cfg['reserve_mib'], self.cfg['gpu_sustain_s'], sample.free_mib)
        return None

    def handle_commands(self):
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                return None
            if cmd == 'EOF':
                return 'holder_gone'
            if cmd == 'RENEW':
                gap = self.clock() - self.last_proof
                if self.preempt_event is None and gap <= self.cfg['proof_deadline_s']:
                    self.expires_at = self.clock() + self.cfg['ttl_s']
                    self.telemetry['renewals'] += 1
                    self.emit('renewed', expires_in=self.cfg['ttl_s'])
                else:
                    self.emit('renew_refused', event=self.preempt_event, gap_s=round(gap, 4))
            elif cmd.startswith('REVOKE'):
                if self.preempt_event is None:
                    self.preempt('revoked', cmd[6:].strip() or 'holder revoke_check')
            elif cmd.startswith('RELEASE'):
                return 'release:' + cmd[7:].strip()

    def kill_requested(self):
        return bool(self.kill) and _WaitForSingleObject(self.kill, 0) == _WAIT_OBJECT_0

    def run(self):
        code = 0
        try:
            try:
                self.setup()
                self.sem_pid = None
                if self.cfg['sem_mode'] == 'absent':
                    self.settle_absent()
                else:
                    self.sem_pid = self.settle()
                sample = self.budget_check()
                if self.cmd_eof():
                    raise LeaseUnavailable('holder_gone', 'holder closed the pipe during acquire')
                # re-prove right before the grant
                bad = self.absent_check() if self.cfg['sem_mode'] == 'absent' else self.time_rule()
                if bad:
                    raise LeaseUnavailable(*bad)
            except LeaseUnavailable as refusal:
                self.emit('refused', reason=refusal.reason, detail=refusal.detail,
                          last=None if refusal.last is None else {'reason': refusal.last.reason,
                                                                  'detail': refusal.last.detail},
                          telemetry=self.telemetry)
                return 0
            self.last_proof = self.clock()
            self.expires_at = self.clock() + self.cfg['ttl_s']
            self.need_read, self.revoke_due = True, bool(self.cfg.get('revoke_file'))
            self.record('held', vram=asdict(sample))         # before the grant message: the holder may read it at once
            self.emit('granted', sem_pid=self.sem_pid, sem_mode=self.cfg['sem_mode'], job_name=self.cfg['job_name'],
                      cancel_name=self.cfg['cancel_name'], kill_name=self.cfg['kill_name'],
                      watcher_pid=os.getpid(), vram=asdict(sample), telemetry=self.telemetry)
            if self.test and self.cfg.get('test_crash_after_grant'):
                time.sleep(float(self.cfg['test_crash_after_grant']))
                os._exit(3)                                  # TEST ONLY: watcher death -> kill-on-close, abandoned mutex
            end = None
            while end is None:
                cmd = self.handle_commands()
                if cmd is not None:
                    end = cmd
                    break
                if _WaitForSingleObject(self.holder, 0) == _WAIT_OBJECT_0:
                    end = 'holder_gone'
                    break
                if self.job_closed is None and self.kill_requested():
                    self.preempt('killed_on_request', 'request_kill() on the lease kill event')
                    self.close_job('kill_request')
                if self.preempt_event is None:
                    outcome = self.probe_absent() if self.cfg['sem_mode'] == 'absent' else self.probe_once()
                    if outcome:
                        self.preempt(*outcome)
                if self.preempt_event is not None and self.job_closed is None and \
                        self.clock() >= self.preempt_event['detected_at'] + self.cfg['grace_s']:
                    self.close_job('preempt+grace')
                if self.preempt_event is None:
                    remaining = self.last_proof + self.cfg['proof_deadline_s'] - self.clock()
                    if self.cfg['sem_mode'] == 'absent':
                        remaining = self.last_scan + self.cfg['sem_scan_s'] - self.clock()
                    timeout = max(0.001, min(self.cfg['poll_s'], remaining + 0.001))
                    if self.watch.mode == 'poll':
                        time.sleep(timeout)
                    else:
                        changed = self.wait_for_change(timeout)
                        names = {n for _i, n in changed}
                        if self.status.name.lower() in names:
                            self.need_read = True
                        if self.cfg.get('revoke_file') and Path(self.cfg['revoke_file']).name.lower() in names:
                            self.revoke_due = True
                else:
                    time.sleep(0.005 if self.job_closed is None else self.cfg['poll_s'])
            if end == 'holder_gone':
                self.preempt('holder_gone', 'holder process exited or closed its pipe')
                self.close_job('holder_gone')
                self.record('holder_gone', event=self.preempt_event, job=self.job_closed)
            else:
                _SetEvent(self.cancel)                       # the lease ended: native children waiting on it stop
                self.close_job('release')
                self.telemetry['unrelated_events'] = sum(getattr(w, 'unrelated', 0) for w in self.watches)
                self.telemetry['overflows'] = sum(getattr(w, 'overflows', 0) for w in self.watches)
                self.record('released', reason=end[8:], event=self.preempt_event, job=self.job_closed)
                self.emit('released', job=self.job_closed, telemetry=self.telemetry,
                          dips=self.governor.dips[-20:])
        except BaseException as error:   # watcher failure: fail closed (cancel set; job closed below)
            code = 1
            try:
                if self.cancel and self.preempt_event is None:
                    self.preempt('watcher_failed', '%s: %s' % (type(error).__name__, error))
            except Exception:
                pass
        finally:
            try:
                if self.job:
                    self.close_job('watcher_exit')
            except Exception:
                pass
            for watch in getattr(self, 'watches', []):
                try:
                    watch.close()
                except Exception:
                    pass
            if getattr(self, 'governor', None):
                self.governor.stop()
            if self.mutex:
                if self.mutex_owned:
                    _ReleaseMutex(self.mutex)
                _CloseHandle(self.mutex)
            for handle in (self.cancel, self.holder, self.kill):
                if handle:
                    _CloseHandle(handle)
            shutil.rmtree(self.copy_dir, ignore_errors=True)
            if getattr(self, 'timer_period', False):
                try:
                    ctypes.WinDLL('winmm').timeEndPeriod(1)
                except Exception:
                    pass
        return code


def watcher_main(cfg_json):
    return Watcher(json.loads(cfg_json)).run()


_BOOTSTRAP = (
    "import hashlib,sys,types\n"
    "p,want,cfg=sys.argv[1],sys.argv[2],sys.argv[3]\n"
    "cfg=sys.stdin.readline() if cfg=='-' else cfg\n"
    "data=open(p,'rb').read()\n"
    "if hashlib.sha256(data).hexdigest()!=want:\n"
    "    sys.stdout.write('{\"type\":\"refused\",\"reason\":\"source_drift\",\"detail\":\"lease_client.py changed\"}\\n');sys.exit(70)\n"
    "m=types.ModuleType('w5_lease_watcher');m.__file__=p;sys.modules[m.__name__]=m\n"
    "exec(compile(data,p,'exec',dont_inherit=True),m.__dict__)\n"
    "sys.exit(m.watcher_main(cfg))\n")


# =============================================================================================================
# HOLDER SIDE
# =============================================================================================================
_PROCESS_LEASES = set()
_PROCESS_LOCK = threading.Lock()


class CancelToken:
    """threading.Event-like. is_set() is also true as soon as the watcher sets the named cancel event, even if this
    process's reader thread has not run yet (GIL)."""

    def __init__(self):
        self._event, self._handle = threading.Event(), None

    def _attach(self, name):
        handle = _OpenEventW(_SYNCHRONIZE, False, name)
        if _ok_handle(handle):
            self._handle = handle

    def is_set(self):
        if self._event.is_set():
            return True
        handle = self._handle
        return bool(handle) and _WaitForSingleObject(handle, 0) == _WAIT_OBJECT_0

    def wait(self, timeout=None):
        if self._event.is_set():
            return True
        handle = self._handle
        if handle:
            ms = 0xFFFFFFFF if timeout is None else max(0, int(timeout * 1000))
            if _WaitForSingleObject(handle, ms) == _WAIT_OBJECT_0:
                return True
            return self._event.is_set()
        return self._event.wait(timeout)

    def set(self):
        self._event.set()

    def _close(self):
        handle, self._handle = self._handle, None
        if handle:
            _CloseHandle(handle)


class _Pipe:
    """The holder's end of one watcher: serialized stdin writes, the optional GPU feeder (gpu_probe callable)."""

    def __init__(self, proc, client, first_line=None):
        self.proc, self.client, self.lock, self.done = proc, client, threading.Lock(), threading.Event()
        if first_line is not None:
            self.send(first_line)                    # the watcher's config (read before anything else)
        if client.gpu_probe is not None:
            threading.Thread(target=self._feed, daemon=True, name='w5-lease-gpu-feed').start()

    def send(self, line):
        with self.lock:
            try:
                self.proc.stdin.write(line + '\n')
                self.proc.stdin.flush()
            except (OSError, ValueError):
                pass

    def _feed(self):
        probe, interval = self.client.gpu_probe, self.client.gpu_interval_s
        while not self.done.is_set():
            try:
                s = probe()
                self.send('GPU ' + json.dumps({'free': s.free_mib, 'total': s.total_mib, 'used': s.used_mib}))
            except Exception as error:
                self.send('GPU ' + json.dumps({'error': '%s: %s' % (type(error).__name__, error)}))
            self.done.wait(interval)


class Lease:
    def __init__(self, client, pipe, lease_id, budget_mib, ttl_s, holder, granted):
        proc = pipe.proc
        self.client, self._pipe, self._proc, self.lease_id = client, pipe, proc, lease_id
        self.budget_mib, self.ttl_s, self.holder, self.mode = budget_mib, float(ttl_s), holder, client.mode
        self.sem_pid, self.sem_mode = granted['sem_pid'], granted.get('sem_mode', 'running')
        self.job_name, self.cancel_event_name = granted['job_name'], granted['cancel_name']
        self.kill_event_name, self.watcher_pid = granted['kill_name'], granted['watcher_pid']
        self.vram = VramSample(**granted['vram'])
        self.cancel_token = CancelToken()
        self.cancel_token._attach(self.cancel_event_name)
        self._mutex = threading.Lock()
        self._callbacks, self._event, self._released = [], None, False
        self._replies, self._done = queue.Queue(), threading.Event()
        self.acquired_at = time.perf_counter()
        self.expires_at = self.acquired_at + self.ttl_s
        self.children = []
        self.telemetry = {'watcher_pid': proc.pid, 'watcher_grant': granted.get('telemetry'), 'job': None,
                          'callback_errors': [], 'callback_seconds': [], 'watcher_final': None, 'dips': None,
                          'adopted': 0}
        self._reader = threading.Thread(target=self._read, daemon=True, name='w5-lease-reader-' + lease_id[:8])
        self._reader.start()
        self._revoker = None
        if client.revoke_check is not None:
            self._revoker = threading.Thread(target=self._revoke_loop, daemon=True, name='w5-lease-revoke')
            self._revoker.start()

    # -- holder API
    @property
    def preempted(self):
        return self._event is not None and self._event.reason != 'released'

    @property
    def event(self):
        return self._event

    def on_preempt(self, callback):
        with self._mutex:
            if self._event is None:
                self._callbacks.append(callback)
                return
            event = self._event
        if event.reason != 'released':
            self._call(callback, event)

    def wait_preempted(self, timeout=None):
        """True once preempted (or released). Waits on the named event, so it wakes without the reader thread."""
        if self.cancel_token.wait(timeout):
            deadline = time.perf_counter() + 1.0
            while self._event is None and time.perf_counter() < deadline:   # let the reader attach the details
                time.sleep(0.001)
            return True
        return False

    def check(self):
        event = self._event
        if event is not None:
            raise LeasePreempted(event)
        if self.cancel_token.is_set():
            raise LeasePreempted(PreemptEvent('preempted', 'cancel event set by the watcher', self.lease_id, 0.0, 0.0))
        if time.perf_counter() > self.expires_at:
            raise LeasePreempted(PreemptEvent('lease_expired', 'no renew within %.1f s' % self.ttl_s,
                                              self.lease_id, time.perf_counter(), 0.0))

    def renew(self):
        """Heartbeat to the watcher. Returns seconds until the new expiry. LeasePreempted if the lease ended."""
        self.check()
        while not self._replies.empty():
            self._replies.get_nowait()
        self._send('RENEW')
        try:
            reply = self._replies.get(timeout=2.0)
        except queue.Empty:
            return max(0.0, self.expires_at - time.perf_counter())   # reply delayed (GIL); the watcher still renews
        if reply['type'] == 'renewed':
            self.expires_at = time.perf_counter() + float(reply['expires_in'])
            return float(reply['expires_in'])
        deadline = time.perf_counter() + 1.0
        while self._event is None and time.perf_counter() < deadline:
            time.sleep(0.001)
        raise LeasePreempted(self._event or PreemptEvent('preempted', 'renew refused', self.lease_id, 0.0, 0.0))

    def spawn(self, args, **popen_kwargs):
        """Start a GPU child INSIDE the lease's Job (spawn_in_job with this lease's Job)."""
        self.check()
        try:
            child = spawn_in_job(self.job_name, args, **popen_kwargs)
        except LeaseError:
            raise LeasePreempted(self._event or PreemptEvent('preempted', 'lease Job gone', self.lease_id, 0.0, 0.0))
        self.telemetry['adopted'] += 1
        self.children.append(child)
        return child

    def adopt(self, child):
        """Put an already running OWN child (subprocess.Popen) into the lease's Job. Prefer spawn(): a child that
        allocated GPU memory or started grandchildren before adopt() is covered from adopt() on only."""
        if not isinstance(child, subprocess.Popen):
            raise TypeError('adopt() takes a subprocess.Popen created by this process')
        self.check()
        try:
            assign_to_job(self.job_name, child)
        except LeaseError:
            raise LeasePreempted(self._event or PreemptEvent('preempted', 'lease Job gone', self.lease_id, 0.0, 0.0))
        self.telemetry['adopted'] += 1
        self.children.append(child)
        return child

    def request_kill(self):
        """End the Job's processes now via the watcher (same as lc.request_kill(lease.kill_event_name))."""
        return request_kill(self.kill_event_name)

    def release(self, reason='done'):
        with self._mutex:
            if self._released:
                return
            self._released = True
            first = self._event is None
            if first:
                self._event = PreemptEvent('released', str(reason), self.lease_id, time.perf_counter(), 0.0)
            self.cancel_token.set()
        self._send('RELEASE %s' % str(reason).replace('\n', ' '))
        self._pipe.done.set()
        try:
            with self._pipe.lock:
                self._proc.stdin.close()
        except OSError:
            pass
        # Synchronous: the watcher ends and drains every Job process, THEN frees the machine-wide mutex and exits.
        # It is never killed here: while a child may still run it keeps the Job and the mutex (fail closed).
        try:
            self._proc.wait(self.client.cfg['drain_timeout_s'] + 10.0)
        except subprocess.TimeoutExpired:
            self.telemetry['release_drain_pending'] = True
        self._done.wait(2.0)
        self.cancel_token._close()
        with _PROCESS_LOCK:
            _PROCESS_LEASES.discard(self.lease_id)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release('context-exit')
        return False

    # -- internals
    def _send(self, line):
        self._pipe.send(line)

    def _call(self, callback, event):
        began = time.perf_counter()
        try:
            callback(event)
        except Exception as error:   # never let a holder callback stop the other callbacks
            self.telemetry['callback_errors'].append('%s: %s' % (type(error).__name__, error))
        self.telemetry['callback_seconds'].append(round(time.perf_counter() - began, 6))

    def _preempted(self, data):
        with self._mutex:
            if self._event is not None:
                return
            self._event = PreemptEvent(received_at=time.perf_counter(), **data)
            self.cancel_token.set()          # before any callback
            callbacks = list(self._callbacks)
        for callback in callbacks:
            self._call(callback, self._event)

    def _read(self):
        try:
            for line in self._proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                kind = msg.get('type')
                if kind == 'preempt':
                    self._preempted(msg['event'])
                elif kind in ('renewed', 'renew_refused'):
                    self._replies.put(msg)
                elif kind == 'job_closed':
                    self.telemetry['job'] = msg
                elif kind == 'released':
                    self.telemetry['watcher_final'] = msg.get('telemetry')
                    self.telemetry['dips'] = msg.get('dips')
                    if msg.get('job'):
                        self.telemetry['job'] = msg['job']
        finally:
            self._pipe.done.set()
            if not self._released:
                self._preempted({'reason': 'watcher_died', 'detail': 'watcher process ended unexpectedly',
                                 'lease_id': self.lease_id, 'detected_at': time.perf_counter(),
                                 'last_idle_proof_at': 0.0})
            self._done.set()

    def _revoke_loop(self):
        while not self._done.is_set() and not self._released and self._event is None:
            try:
                revoked = self.client.revoke_check()
            except Exception as error:
                revoked = 'revoke check failed: %s' % error
            if revoked:
                self._send('REVOKE %s' % revoked)
                return
            self._done.wait(self.client.poll_s)


class LeaseClient:
    """One per holder process. v1-compatible surface; see CONTRACT.md for the parameters that changed meaning."""

    def __init__(self, *, status_file=DEFAULT_STATUS, state_dir=None, mode='no-release', reserve_mib=RESERVE_MIB,
                 gpu_probe=None, gpu_interval_s=GPU_INTERVAL_S, revoke_file=None, revoke_check=None,
                 poll_s=POLL_S, proof_deadline_s=PROOF_DEADLINE_S, reverify_s=REVERIFY_S,
                 max_status_age_s=MAX_STATUS_AGE_S, frozen_interval_s=FROZEN_INTERVAL_S, settle_s=SETTLE_S,
                 grace_s=GRACE_S, gpu_sample_s=GPU_SAMPLE_S, gpu_baseline_s=GPU_BASELINE_S,
                 gpu_sustain_s=GPU_SUSTAIN_S, gpu_max_age_s=GPU_MAX_AGE_S, first_gpu_timeout_s=FIRST_GPU_TIMEOUT_S,
                 watch_mode='notify', mutex_name=None, python=None, guard_packages=None, test_hooks=None,
                 watcher_stderr=None, sem_mode='running', sem_rules=None, sem_scan_s=SEM_SCAN_S,
                 drain_timeout_s=DRAIN_TIMEOUT_S,
                 # v1 parameters accepted for compatibility; no effect in v2 (the watcher process does this work):
                 sem_check=None, raw_check=None, reader=None, settle_poll_s=None, acquire_retry_s=None,
                 watch_factory=None, clock=None, wall=None, sleep=None):
        if mode not in MODES:
            raise ValueError('mode must be one of %s' % (MODES,))
        if sem_mode not in SEM_MODES:
            raise ValueError('sem_mode must be one of %s (explicit; never inferred)' % (SEM_MODES,))
        if sem_rules is not None and mutex_name is None:
            raise ValueError('custom sem_rules need a TEST mutex namespace (production uses SEM_PROCESS_RULES)')
        if not 0 < poll_s < proof_deadline_s or proof_deadline_s + poll_s > LATENCY_BOUND_S:
            raise ValueError('need 0 < poll_s < proof_deadline_s and proof_deadline_s + poll_s <= %.1f s' % LATENCY_BOUND_S)
        if watch_mode not in ('notify', 'poll', 'deaf'):
            raise ValueError('watch_mode must be notify, poll or deaf (test only)')
        if mutex_name is not None and not str(mutex_name).startswith(TEST_MUTEX_PREFIX):
            raise ValueError('the lease mutex is fixed (%s); only %s* names are accepted (tests)' % (MUTEX_NAME, TEST_MUTEX_PREFIX))
        hooks = dict(test_hooks or {})
        if (hooks or watch_mode == 'deaf') and mutex_name is None:
            raise ValueError('test hooks / deaf watch need a TEST mutex namespace')
        if not 0 <= grace_s <= 1.0:
            raise ValueError('grace_s must be in [0, 1] s')
        self.mode, self.status_file = mode, Path(status_file)
        self.state_dir = None if state_dir is None else Path(state_dir)
        self.reserve_mib, self.gpu_probe, self.gpu_interval_s = int(reserve_mib), gpu_probe, float(gpu_interval_s)
        self.revoke_file = None if revoke_file is None else str(Path(revoke_file))
        self.revoke_check, self.poll_s = revoke_check, float(poll_s)
        self.mutex_name = mutex_name or MUTEX_NAME
        self.python = python or sys.executable
        self.watcher_stderr = watcher_stderr
        self.cfg = {'status_file': str(self.status_file), 'state_dir': None if state_dir is None else str(state_dir),
                    'mutex_name': self.mutex_name, 'reserve_mib': self.reserve_mib, 'poll_s': float(poll_s),
                    'proof_deadline_s': float(proof_deadline_s), 'reverify_s': float(reverify_s),
                    'max_status_age_s': float(max_status_age_s), 'frozen_interval_s': float(frozen_interval_s),
                    'settle_s': float(settle_s), 'grace_s': float(grace_s), 'gpu_sample_s': float(gpu_sample_s),
                    'gpu_baseline_s': float(gpu_baseline_s), 'gpu_sustain_s': float(gpu_sustain_s),
                    'gpu_max_age_s': float(gpu_max_age_s), 'first_gpu_timeout_s': float(first_gpu_timeout_s),
                    'watch_mode': watch_mode, 'revoke_file': self.revoke_file,
                    'gpu_source': 'holder' if gpu_probe is not None else 'nvml',
                    'guard_packages': [str(p) for p in (guard_packages or GUARD_PACKAGES)],
                    'tolerant_path': str(TOLERANT_PATH), 'sem_mode': sem_mode, 'sem_rules': sem_rules,
                    'sem_scan_s': float(sem_scan_s), 'drain_timeout_s': float(drain_timeout_s)}
        self.sem_mode = sem_mode
        for key, value in hooks.items():
            self.cfg['test_' + key] = value
        self.ignored_v1_parameters = sorted(k for k, v in dict(
            sem_check=sem_check, raw_check=raw_check, reader=reader, settle_poll_s=settle_poll_s,
            acquire_retry_s=acquire_retry_s, watch_factory=watch_factory, clock=clock, wall=wall, sleep=sleep).items()
            if v is not None)

    # -- public
    def available_vram(self):
        """One read-only sample now (NVML, or the gpu_probe callable). For the dip-aware view use a lease's vram."""
        if self.gpu_probe is not None:
            sample = self.gpu_probe()
        else:
            probe = default_vram_probe()
            try:
                sample = probe()
            finally:
                getattr(probe, 'close', lambda: None)()
        if not isinstance(sample, VramSample):
            raise GpuProbeError('probe returned %r' % (sample,))
        return sample

    def acquire(self, budget_mib, *, ttl_s=TTL_S, wait_s=0.0, holder=''):
        if self.mode != 'no-release':
            raise LeaseModeUnavailable('mode_unavailable', 'sem-release needs the Sem-side lease (SEM-SIDE-DELTA.md) '
                                       'and Kevin\'s acceptance; not implemented')
        if not isinstance(budget_mib, int) or isinstance(budget_mib, bool) or budget_mib <= 0:
            raise ValueError('budget_mib must be a positive int')
        if not ttl_s > 0:
            raise ValueError('ttl_s must be > 0')
        lease_id = uuid.uuid4().hex
        with _PROCESS_LOCK:
            if _PROCESS_LEASES:
                raise LeaseUnavailable('previous_lease_unreleased', 'this process holds lease %s; stop its GPU work '
                                       'and call release() before acquire()' % next(iter(_PROCESS_LEASES)))
            _PROCESS_LEASES.add(lease_id)
        try:
            return self._acquire(lease_id, budget_mib, float(ttl_s), max(0.0, float(wait_s)), holder)
        except BaseException:
            with _PROCESS_LOCK:
                _PROCESS_LEASES.discard(lease_id)
            raise

    def _acquire(self, lease_id, budget_mib, ttl_s, wait_s, holder):
        cfg = dict(self.cfg, lease_id=lease_id, budget_mib=budget_mib, ttl_s=ttl_s, wait_s=wait_s, holder=holder,
                   holder_pid=os.getpid(), job_name=OBJECT_PREFIX + 'Job-' + lease_id,
                   cancel_name=OBJECT_PREFIX + 'Cancel-' + lease_id, kill_name=OBJECT_PREFIX + 'Kill-' + lease_id)
        source = SOURCE_PATH.read_bytes()
        if _sha256_bytes(source) != SOURCE_SHA256:
            raise LeaseUnavailable('source_drift', 'lease_client.py changed since import')
        # The config goes over stdin, never on a command line: Sem identity markers in it would otherwise make this
        # watcher (or its venv launcher) look like a Sem process to the absent-mode scan.
        proc = subprocess.Popen([self.python, '-I', '-B', '-u', '-c', _BOOTSTRAP, str(SOURCE_PATH), SOURCE_SHA256,
                                 '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=open(self.watcher_stderr, 'a') if self.watcher_stderr else subprocess.DEVNULL,
                                text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        pipe = _Pipe(proc, self, first_line=json.dumps(cfg))
        replies = queue.Queue()

        def first_line():
            replies.put(proc.stdout.readline())
        reader = threading.Thread(target=first_line, daemon=True)
        reader.start()
        timeout = WATCHER_TIMEOUT_S + cfg['settle_s'] + wait_s + cfg['first_gpu_timeout_s'] + cfg['gpu_baseline_s']
        try:
            line = replies.get(timeout=timeout)
        except queue.Empty:
            pipe.done.set()
            proc.kill()                                      # our own watcher child, stuck before granting
            proc.wait(5)
            raise LeaseUnavailable('watcher_timeout', 'no answer within %.0f s' % timeout)
        try:
            msg = json.loads(line) if line else {'type': 'refused', 'reason': 'watcher_failed',
                                                 'detail': 'watcher exited (code %s)' % proc.poll()}
        except ValueError:
            msg = {'type': 'refused', 'reason': 'watcher_failed', 'detail': line[:200]}
        if msg.get('type') != 'granted':
            pipe.done.set()
            try:
                with pipe.lock:
                    proc.stdin.close()
            except OSError:
                pass
            proc.wait(10)
            last = msg.get('last')
            raise LeaseUnavailable(msg.get('reason', 'watcher_failed'), msg.get('detail', ''),
                                   last=None if not last else LeaseUnavailable(last['reason'], last['detail']))
        return Lease(self, pipe, lease_id, budget_mib, ttl_s, holder, msg)


__all__ = ['LeaseClient', 'Lease', 'PreemptEvent', 'VramSample', 'CancelToken', 'LeaseError', 'LeaseUnavailable',
           'LeaseModeUnavailable', 'LeasePreempted', 'GpuProbeError', 'classify_state', 'read_state', 'read_shared',
           'frozen_tolerant_check', 'load_frozen_guard', 'load_tolerant', 'nvidia_vram', 'Nvml', 'GpuGovernor',
           'DirWatch', 'MUTEX_NAME', 'LATENCY_BOUND_S', 'SOURCE_SHA256', 'spawn_in_job', 'assign_to_job',
           'cancel_is_set', 'wait_cancel', 'request_kill']
