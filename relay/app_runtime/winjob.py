"""Windows helpers for the KevFLIX app guard (process A). Stdlib + ctypes only. Ported from movie-trial-v1/launch.

* KillOnCloseJob: the guard Job (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE): when A dies, the OS ends S and P.
* launch_suspended(): CREATE_SUSPENDED -> assign hooks (guard Job, then lease.adopt) -> NtResumeProcess, so no
  descendant of S or P can start outside its Job(s).
* tcp_listeners(): the OS TCP listener table (contacts no process).
* ram_available_mib(), process_identity(), wait_identity_exit(), the named stop event, the single-instance mutex,
  and the Sem-path command-line refusal.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from ctypes import wintypes

# A child command line must never contain one of these (lease v2 absent-mode command-line rule, plus Sem roots).
SEM_PATH_MARKERS = ('\\sem-trials\\', '\\sem-5.0\\', '\\sem-4.0\\', '\\sem-data\\profile-v1\\', 'sem_image_native',
                    'sem-image-native', 'nano-voxcpm2', 'native-audio-worker', 'sem-decision', '\\sem-voice\\',
                    '\\sem-developertools\\', 'breeze')
STOP_EVENT_PREFIX = 'Local\\KevFLIX-App-Stop-'
INSTANCE_MUTEX = 'Local\\KevFLIX-App-Instance'

_k32 = ctypes.WinDLL('kernel32', use_last_error=True)
_ntdll = ctypes.WinDLL('ntdll')
_iphlp = ctypes.WinDLL('iphlpapi')

_k32.CreateJobObjectW.restype = wintypes.HANDLE
_k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
_k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
_k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID]
_k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
_k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
_k32.CreateEventW.restype = wintypes.HANDLE
_k32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_k32.OpenEventW.restype = wintypes.HANDLE
_k32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_k32.SetEvent.argtypes = [wintypes.HANDLE]
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.OpenJobObjectW.restype = wintypes.HANDLE
_k32.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
_ntdll.NtResumeProcess.restype = ctypes.c_long

CREATE_SUSPENDED, CREATE_NO_WINDOW = 0x4, 0x08000000
KILL_ON_JOB_CLOSE = 0x2000
EVENT_MODIFY_STATE, SYNCHRONIZE = 0x0002, 0x00100000
QUERY_LIMITED = 0x1000


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]


class _EXT_LIMIT(ctypes.Structure):
    _fields_ = [('Basic', _BASIC_LIMIT), ('Io', _IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t)]


class _ACCOUNTING(ctypes.Structure):
    _fields_ = [('TotalUserTime', ctypes.c_int64), ('TotalKernelTime', ctypes.c_int64),
                ('ThisPeriodTotalUserTime', ctypes.c_int64), ('ThisPeriodTotalKernelTime', ctypes.c_int64),
                ('TotalPageFaultCount', wintypes.DWORD), ('TotalProcesses', wintypes.DWORD),
                ('ActiveProcesses', wintypes.DWORD), ('TotalTerminatedProcesses', wintypes.DWORD)]


class KillOnCloseJob:
    """The guard Job. Holds only processes A created (S, P and their descendants)."""

    def __init__(self):
        self.handle = _k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _EXT_LIMIT()
        info.Basic.LimitFlags = KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            _k32.CloseHandle(self.handle)
            self.handle = None
            raise error

    def assign(self, popen):
        if not _k32.AssignProcessToJobObject(self.handle, int(popen._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        return popen

    def active_processes(self):
        info = _ACCOUNTING()
        if not self.handle or not _k32.QueryInformationJobObject(self.handle, 1, ctypes.byref(info),
                                                                 ctypes.sizeof(info), None):
            return None
        return int(info.ActiveProcesses)

    def terminate(self):
        return bool(self.handle) and bool(_k32.TerminateJobObject(self.handle, 1))

    def close(self):
        if self.handle:
            _k32.CloseHandle(self.handle)
            self.handle = None


JOB_OBJECT_QUERY, ERROR_FILE_NOT_FOUND = 0x0004, 2


def named_job_active(name):
    """Read-only drain evidence for a named Job (the movie lease Job): its active process count; 0 when the Job no
    longer exists by name (the lease Job is kill-on-close: when the watcher's last handle closes the OS ends its
    members); None when it cannot be read (unknown is never zero)."""
    handle = _k32.OpenJobObjectW(JOB_OBJECT_QUERY, False, str(name))
    if not handle:
        return 0 if ctypes.get_last_error() == ERROR_FILE_NOT_FOUND else None
    try:
        info = _ACCOUNTING()
        if not _k32.QueryInformationJobObject(handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            return None
        return int(info.ActiveProcesses)
    finally:
        _k32.CloseHandle(handle)


def assert_no_sem_path(argv):
    """Hard rule: no Sem path on any child command line (the lease's absent mode would also count it as Sem)."""
    text = ' '.join(str(a) for a in argv).replace('/', '\\').lower()
    found = [m for m in SEM_PATH_MARKERS if m in text]
    if found:
        raise PermissionError('child command line names a Sem/Breeze path %s; refused' % found)


def launch_suspended(argv, *, cwd, env, stdin, stdout, stderr, assign=()):
    """Popen created suspended; each assign(popen) runs before the first instruction; then resume.
    On any failure the suspended child is killed (it never ran)."""
    argv = [str(a) for a in argv]
    assert_no_sem_path(argv)
    child = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=stdin, stdout=stdout, stderr=stderr,
                             creationflags=CREATE_SUSPENDED | CREATE_NO_WINDOW)
    try:
        for hook in assign:
            hook(child)
        status = _ntdll.NtResumeProcess(int(child._handle))
        if status != 0:
            raise OSError('NtResumeProcess failed: 0x%08x' % (status & 0xFFFFFFFF))
    except BaseException:
        try:
            child.kill()
        finally:
            child.wait(10)
        raise
    return child


# ---- TCP listeners (read-only OS table) -------------------------------------------------------------------------------
class _TCPROW(ctypes.Structure):
    _fields_ = [('state', wintypes.DWORD), ('laddr', wintypes.DWORD), ('lport', wintypes.DWORD),
                ('raddr', wintypes.DWORD), ('rport', wintypes.DWORD), ('pid', wintypes.DWORD)]


class _TCP6ROW(ctypes.Structure):
    _fields_ = [('laddr', ctypes.c_ubyte * 16), ('lscope', wintypes.DWORD), ('lport', wintypes.DWORD),
                ('raddr', ctypes.c_ubyte * 16), ('rscope', wintypes.DWORD), ('rport', wintypes.DWORD),
                ('state', wintypes.DWORD), ('pid', wintypes.DWORD)]


def tcp_listeners():
    """{port: sorted pids} of LISTENING TCP sockets (IPv4 + IPv6)."""
    found = {}
    for family, row_type in ((2, _TCPROW), (23, _TCP6ROW)):
        size = wintypes.DWORD(0)
        _iphlp.GetExtendedTcpTable(None, ctypes.byref(size), False, family, 3, 0)
        buffer = ctypes.create_string_buffer(size.value + 4096)
        size = wintypes.DWORD(len(buffer))
        if _iphlp.GetExtendedTcpTable(buffer, ctypes.byref(size), False, family, 3, 0) != 0:
            raise OSError('GetExtendedTcpTable failed (family %d)' % family)
        count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
        rows = ctypes.cast(ctypes.addressof(buffer) + ctypes.sizeof(wintypes.DWORD),
                           ctypes.POINTER(row_type * count))[0]
        for row in rows:
            port = ((row.lport & 0xFF) << 8) | ((row.lport >> 8) & 0xFF)
            found.setdefault(port, set()).add(int(row.pid))
    return {port: sorted(pids) for port, pids in found.items()}


def process_identity(pid):
    """{pid, created_filetime, image} with query-limited access only, or None."""
    handle = _k32.OpenProcess(QUERY_LIMITED, False, int(pid))
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not _k32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return None
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not _k32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        return {'pid': int(pid), 'created_filetime': created, 'image': os.path.normcase(buffer.value)}
    finally:
        _k32.CloseHandle(handle)


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)] + \
               [(n, ctypes.c_ulonglong) for n in ('total', 'avail', 'tpage', 'apage', 'tvirt', 'avirt', 'aext')]


def ram_available_mib():
    status = _MEMSTATUS()
    status.dwLength = ctypes.sizeof(status)
    if not _k32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return status.avail // (1024 * 1024)


# ---- named stop event ------------------------------------------------------------------------------------------------------
def stop_event_name(run_id):
    return STOP_EVENT_PREFIX + str(run_id)


class StopEvent:
    """Manual-reset named event created by A; --stop opens it by name and sets it."""

    def __init__(self, stamp):
        self.name = stop_event_name(stamp)
        self.handle = _k32.CreateEventW(None, True, False, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def is_set(self):
        return bool(self.handle) and _k32.WaitForSingleObject(self.handle, 0) == 0

    def close(self):
        if self.handle:
            _k32.CloseHandle(self.handle)
            self.handle = None


def signal_stop_name(name):
    """True when the named stop event existed and was set. Only KevFLIX stop-event names are accepted."""
    if not str(name).startswith(STOP_EVENT_PREFIX):
        return False
    handle = _k32.OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, False, str(name))
    if not handle:
        return False
    try:
        return bool(_k32.SetEvent(handle))
    finally:
        _k32.CloseHandle(handle)


# ---- single instance -----------------------------------------------------------------------------------------------------
_k32.CreateMutexW.restype = wintypes.HANDLE
_k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
_k32.ReleaseMutex.argtypes = [wintypes.HANDLE]
ERROR_ALREADY_EXISTS = 183


class InstanceMutex:
    """Named mutex held by A for its lifetime; the OS frees it when A exits (also on a crash)."""

    def __init__(self, name=INSTANCE_MUTEX):
        self.name = name
        self.handle = _k32.CreateMutexW(None, True, name)
        error = ctypes.get_last_error()
        if not self.handle:
            raise ctypes.WinError(error)
        if error == ERROR_ALREADY_EXISTS:
            _k32.CloseHandle(self.handle)
            self.handle = None
            raise FileExistsError('another KevFLIX instance is running')

    def close(self):
        if self.handle:
            _k32.ReleaseMutex(self.handle)
            _k32.CloseHandle(self.handle)
            self.handle = None


def wait_identity_exit(identity, timeout_s):
    """Wait for the process with exactly this identity (pid + creation time + image) to exit. Returns True when it
    is gone, False on timeout. A reused PID with another identity counts as gone (never waited on, never touched)."""
    pid = int(identity['pid'])
    handle = _k32.OpenProcess(SYNCHRONIZE | QUERY_LIMITED, False, pid)
    if not handle:
        return True
    try:
        if process_identity(pid) != identity:
            return True
        return _k32.WaitForSingleObject(handle, max(0, int(timeout_s * 1000))) == 0
    finally:
        _k32.CloseHandle(handle)
