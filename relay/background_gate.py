"""Background finish (tv.background_finish): when may the dub of the film just watched continue with nobody watching,
and at which process priority. Standard library only.

The background runs only while ALL of these hold (BackgroundGate.check):
  * the setting is on;
  * no TV playback session exists (dubbed or not);
  * nobody has touched keyboard or mouse for tv.background_idle_minutes (user32.GetLastInputInfo);
  * the graphics card is below tv.background_max_gpu_celsius (nvidia-smi), and after a pause for heat only once it is
    RESUME_BELOW_CELSIUS cooler again.
Every reading that fails counts as "do not run": idle time 0, temperature None.
The other AI app is not handled here: the movie GPU lease stops the dub within a second when it starts, as always."""
from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time

IDLE_MINUTES = 5.0                 # tv.background_idle_minutes (1..120)
MAX_GPU_CELSIUS = 78.0             # tv.background_max_gpu_celsius (50..90)
RESUME_BELOW_CELSIUS = 5.0         # after a pause for heat: resume at or below the limit minus this
TEMPERATURE_EVERY_S = 10.0         # nvidia-smi is asked at most this often
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
NORMAL_PRIORITY_CLASS = 0x20


def user_idle_seconds():
    """Seconds since the last keyboard or mouse input in this Windows session (user32.GetLastInputInfo).
    0.0, meaning "in use", on any failure and on every other system: the background never runs on a guess."""
    if os.name != "nt":
        return 0.0
    try:
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        tick = ctypes.windll.kernel32.GetTickCount()
        return max(0.0, ((int(tick) - int(info.dwTime)) & 0xFFFFFFFF) / 1000.0)      # the tick counter wraps every 49.7 days
    except Exception:  # noqa: BLE001 - any failure means "in use"
        return 0.0


# Temperature in C of the hottest NVIDIA GPU, or None when it cannot be read (then the background does not run).
# F28: one copy of the reader, shared with the live dub's heat guard (relay/gpu_heat.py).
from .gpu_heat import gpu_temperature_c  # noqa: E402


def _number(value, default, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return default
    return min(high, max(low, float(value)))


class BackgroundGate:
    """check() -> (allowed, reason). Thread-safe; the temperature is read at most every TEMPERATURE_EVERY_S."""

    def __init__(self, tv_cfg=None, *, playback=None, idle=user_idle_seconds, temperature=gpu_temperature_c,
                 clock=time.monotonic):
        cfg = tv_cfg if isinstance(tv_cfg, dict) else {}
        self.enabled = cfg.get("background_finish") is True
        self.idle_seconds = 60.0 * _number(cfg.get("background_idle_minutes"), IDLE_MINUTES, 1.0, 120.0)
        self.max_celsius = _number(cfg.get("background_max_gpu_celsius"), MAX_GPU_CELSIUS, 50.0, 90.0)
        self.playback, self.idle, self.temperature, self.clock = playback, idle, temperature, clock
        self._lock = threading.Lock()
        self._hot = False                  # paused for heat: resume only RESUME_BELOW_CELSIUS under the limit
        self._temp, self._temp_at = None, None

    def check(self):
        with self._lock:
            if not self.enabled:
                return False, "switched_off"
            if self.playback is not None and self.playback():
                return False, "playback"
            if self.idle() < self.idle_seconds:
                return False, "pc_in_use"
            now = self.clock()
            if self._temp_at is None or now - self._temp_at >= TEMPERATURE_EVERY_S:
                self._temp, self._temp_at = self.temperature(), now
            if self._temp is None:
                return False, "gpu_temperature_unreadable"
            if self._temp >= self.max_celsius or (self._hot and self._temp > self.max_celsius - RESUME_BELOW_CELSIUS):
                self._hot = True
                return False, "gpu_hot"
            self._hot = False
            return True, "open"


def child_processes(root_pid):
    """Process ids of every descendant of root_pid (Toolhelp snapshot). [] on any failure and outside Windows."""
    if os.name != "nt":
        return []
    try:
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                        ("szExeFile", ctypes.c_wchar * 260)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)          # TH32CS_SNAPPROCESS
        if not snapshot or snapshot == wintypes.HANDLE(-1).value:
            return []
        parents = {}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            more = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
            while more:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                more = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        finally:
            kernel32.CloseHandle(snapshot)
        root, found, frontier = int(root_pid), [], {int(root_pid)}
        while frontier:
            frontier = {pid for pid, parent in parents.items() if parent in frontier and pid != root and pid not in found}
            found.extend(sorted(frontier))
        return found
    except Exception:  # noqa: BLE001
        return []


def set_priority_class(pid, wanted, only_from):
    """Give process `pid` the priority class `wanted` if (and only if) its class is `only_from` now. True when changed.
    A process at another class (the lease watcher runs above normal) is never touched."""
    if os.name != "nt":
        return False
    try:
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.GetPriorityClass.restype = wintypes.DWORD
        kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x0200 | 0x1000, False, int(pid))     # SET_INFORMATION | QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            if kernel32.GetPriorityClass(handle) != only_from:
                return False
            return bool(kernel32.SetPriorityClass(handle, wanted))
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001
        return False


class BackgroundPriority:
    """Keeps the server's child processes (analysis worker, Demucs, Vox engines, ffmpeg) at below-normal priority while
    only background windows are in production, and puts back exactly the ones it lowered."""

    def __init__(self, root_pid=None, *, children=child_processes, change=set_priority_class):
        self.root = int(root_pid or os.getpid())
        self.children, self.change, self.lowered = children, change, set()
        self._lock = threading.Lock()          # lower() is called from the coordinator thread and from the gate watcher

    def lower(self):
        with self._lock:
            for pid in self.children(self.root):
                if pid not in self.lowered and self.change(pid, BELOW_NORMAL_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS):
                    self.lowered.add(pid)
            return len(self.lowered)

    def restore(self):
        with self._lock:
            for pid in list(self.lowered):
                self.change(pid, NORMAL_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS)
            count, self.lowered = len(self.lowered), set()
            return count
