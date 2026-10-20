"""Who owns a movie-Vox engine process (review P1-1: a crashed holder must never leave an engine running).

Two owners, same Protocol (`spawn(args, **popen_kwargs) -> subprocess.Popen`, `describe() -> dict`, `close()`):

* LeaseProcessOwner: the movie GPU lease v2 (lease v2 contract section 2). `lease.spawn()` creates the child
  suspended INSIDE the lease's kill-on-close Job, held by W5's separate watcher process. The watcher ends the Job at
  preemption + grace_s, at release(), and when the holder dies; if the watcher dies the OS closes the Job. None of that
  needs the holder's GIL (review P1-2 for preemption).
* OwnJobProcessOwner: fallback when the bound lease has no spawn() (lease v1, test fakes). This holder process owns an
  anonymous Job with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; the child is created CREATE_SUSPENDED, assigned, then resumed
  (no escape window). The only handle is this process's, so ANY death of the holder (crash, os._exit, TerminateProcess)
  closes it and the OS ends every child. A preemption still needs the holder (GIL-dependent; production uses v2).

Defence in depth (vox_movie_child.py): the child also watches its parent process handle and the lease's named cancel
event on kernel waits, and exits by itself (os._exit) on either. Windows only; stdlib only.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import threading
from ctypes import wintypes

CREATE_SUSPENDED = 0x4
KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1


class ProcessOwnerError(OSError):
    pass


if os.name == 'nt':
    _k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _ntdll = ctypes.WinDLL('ntdll')
    _H = wintypes.HANDLE

    class _BASIC(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                     'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

    class _EXTENDED(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', _BASIC), ('IoInfo', _IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t)]

    class _ACCOUNTING(ctypes.Structure):
        _fields_ = [('TotalUserTime', ctypes.c_int64), ('TotalKernelTime', ctypes.c_int64),
                    ('ThisPeriodTotalUserTime', ctypes.c_int64), ('ThisPeriodTotalKernelTime', ctypes.c_int64),
                    ('TotalPageFaultCount', wintypes.DWORD), ('TotalProcesses', wintypes.DWORD),
                    ('ActiveProcesses', wintypes.DWORD), ('TotalTerminatedProcesses', wintypes.DWORD)]

    _k32.CreateJobObjectW.restype, _k32.CreateJobObjectW.argtypes = _H, [ctypes.c_void_p, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.SetInformationJobObject.argtypes = [_H, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [_H, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    _k32.AssignProcessToJobObject.restype, _k32.AssignProcessToJobObject.argtypes = wintypes.BOOL, [_H, _H]
    _k32.TerminateJobObject.restype, _k32.TerminateJobObject.argtypes = wintypes.BOOL, [_H, wintypes.UINT]
    _k32.TerminateProcess.restype, _k32.TerminateProcess.argtypes = wintypes.BOOL, [_H, wintypes.UINT]
    _k32.CloseHandle.restype, _k32.CloseHandle.argtypes = wintypes.BOOL, [_H]
    _ntdll.NtResumeProcess.restype, _ntdll.NtResumeProcess.argtypes = ctypes.c_long, [_H]


def _error(what):
    return ProcessOwnerError('%s failed: WinError %d' % (what, ctypes.get_last_error()))


class OwnJobProcessOwner:
    """A kill-on-close Job held only by this process (see module docstring)."""

    kind = 'own-kill-on-close-job'

    def __init__(self):
        if os.name != 'nt':
            raise ProcessOwnerError('kill-on-close Job objects need Windows')
        self._lock = threading.Lock()
        self._job = _k32.CreateJobObjectW(None, None)
        if not self._job:
            raise _error('CreateJobObjectW')
        info = _EXTENDED()
        info.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(self._job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                            ctypes.sizeof(info)):
            error = _error('SetInformationJobObject')
            _k32.CloseHandle(self._job)
            self._job = None
            raise error
        self.spawned = 0

    def spawn(self, args, **popen_kwargs):
        flags = popen_kwargs.pop('creationflags', 0) | CREATE_SUSPENDED
        with self._lock:
            if not self._job:
                raise ProcessOwnerError('process owner closed')
            child = subprocess.Popen(args, creationflags=flags, **popen_kwargs)
            try:
                if not _k32.AssignProcessToJobObject(self._job, int(child._handle)):
                    raise _error('AssignProcessToJobObject')
                if _ntdll.NtResumeProcess(int(child._handle)) != 0:
                    raise ProcessOwnerError('NtResumeProcess failed')
            except BaseException:
                _k32.TerminateProcess(int(child._handle), 0xC0DE)   # our own child, created suspended, never ran
                child.wait(5)
                raise
            self.spawned += 1
            return child

    def active(self):
        info = _ACCOUNTING()
        with self._lock:
            if not self._job or not _k32.QueryInformationJobObject(self._job, JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                                                                   ctypes.byref(info), ctypes.sizeof(info), None):
                return None
        return int(info.ActiveProcesses)

    def describe(self):
        return {'kind': self.kind, 'kill_on_close': True, 'spawned': self.spawned, 'active': self.active()}

    def close(self):
        """Ends every child in the Job (closing the only handle triggers KILL_ON_JOB_CLOSE)."""
        with self._lock:
            job, self._job = self._job, None
        if job:
            _k32.TerminateJobObject(job, 0xC0DE)
            _k32.CloseHandle(job)


class LeaseProcessOwner:
    """Children go into the movie GPU lease v2 Job via lease.spawn() (contract §2 rule 1)."""

    kind = 'lease-v2-job'

    def __init__(self, lease):
        if not callable(getattr(lease, 'spawn', None)):
            raise TypeError('lease has no spawn(); use OwnJobProcessOwner')
        self.lease = lease
        self.spawned = 0

    def spawn(self, args, **popen_kwargs):
        child = self.lease.spawn(args, **popen_kwargs)
        self.spawned += 1
        return child

    def describe(self):
        return {'kind': self.kind, 'lease_id': getattr(self.lease, 'lease_id', None), 'spawned': self.spawned,
                'cancel_event_name': getattr(self.lease, 'cancel_event_name', None)}

    def close(self):
        """Nothing to close here: the lease's watcher owns the Job (it ends at release / preemption)."""
