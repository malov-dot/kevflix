"""In-child watchdogs for a movie-Vox engine process (stdlib, ctypes; Windows kernel waits). No relative imports: it is
loaded from verified bytes by vox_movie_child.py and by tests/fake_vox_child.py.

All three run on daemon threads of the CHILD, so none of them needs the holder's GIL (review P1-2). The engine's
native generate call (ActorVoice -> audio.cpp via ctypes) releases the child's GIL, so they also fire mid-synthesis.

* watch_parent(pid): kernel wait on the parent (holder) process handle; on its exit -> os._exit(EXIT_PARENT). Started
  before the model load, so an orphan cannot survive a crashed holder in the load phase either (review P1-1 b).
* watch_event(name): kernel wait on the movie GPU lease v2 cancel event (`lease.cancel_event_name`); set at preemption
  or release -> os._exit(EXIT_LEASE). An event that cannot be opened means the lease is over -> exit (fail closed).
* HardCancel: while a request is in flight, the cancel marker `<output>.cancel` is polled every `poll_s`. The
  cooperative path (marker check in every PCM callback) acknowledges and keeps the engine warm; if the request is still
  running `hard_s` after the marker first appeared (prefill, a stuck native call), -> os._exit(EXIT_CANCEL).
"""
import ctypes
import os
import threading
import time
from ctypes import wintypes
from pathlib import Path

EXIT_PARENT, EXIT_LEASE, EXIT_CANCEL = 70, 71, 72
SYNCHRONIZE = 0x00100000
INFINITE = 0xFFFFFFFF

if os.name == 'nt':
    _k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _k32.OpenProcess.restype, _k32.OpenProcess.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenEventW.restype, _k32.OpenEventW.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.CloseHandle.restype, _k32.CloseHandle.argtypes = wintypes.BOOL, [wintypes.HANDLE]


def _exit(code, note=None):
    try:
        if note is not None:
            note(code)
    finally:
        os._exit(code)


def _wait_handle(handle):
    _k32.WaitForSingleObject(handle, INFINITE)


def watch_parent(pid, *, on_exit=None, note=None):
    """Exit when process `pid` exits (or cannot be opened: already gone)."""
    pid = int(pid)
    handle = _k32.OpenProcess(SYNCHRONIZE, False, pid)
    action = on_exit or (lambda: _exit(EXIT_PARENT, note))
    if not handle:
        action()
        return None

    def run():
        _wait_handle(handle)
        action()
    thread = threading.Thread(target=run, name='vox-parent-watchdog', daemon=True)
    thread.start()
    return thread


def watch_event(name, *, on_exit=None, note=None):
    """Exit when the named (lease cancel) event is set, or at once if it does not exist."""
    handle = _k32.OpenEventW(SYNCHRONIZE, False, str(name))
    action = on_exit or (lambda: _exit(EXIT_LEASE, note))
    if not handle:
        action()
        return None

    def run():
        _wait_handle(handle)
        action()
    thread = threading.Thread(target=run, name='vox-lease-event-watchdog', daemon=True)
    thread.start()
    return thread


class HardCancel:
    def __init__(self, hard_s=0.1, poll_s=0.005, *, on_exit=None, note=None, clock=time.perf_counter):
        self.hard_s, self.poll_s, self.clock = float(hard_s), float(poll_s), clock
        self._on_exit = on_exit or (lambda: _exit(EXIT_CANCEL, note))
        self._lock = threading.Lock()
        self._marker, self._seen = None, None
        self._thread = threading.Thread(target=self._run, name='vox-hard-cancel', daemon=True)
        self._thread.start()

    def arm(self, output):
        with self._lock:
            self._marker, self._seen = Path(str(output) + '.cancel'), None

    def disarm(self):
        with self._lock:
            self._marker, self._seen = None, None

    def _run(self):
        while True:
            time.sleep(self.poll_s)
            with self._lock:
                marker = self._marker
                if marker is None:
                    continue
                try:
                    present = marker.exists()
                except OSError:
                    present = False
                if not present:
                    continue
                now = self.clock()
                if self._seen is None:
                    self._seen = now
                fire = now - self._seen >= self.hard_s
            if fire:
                self._on_exit()
                return
