"""Admission/yield checks for movie processing alongside an active voice task."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time


class ResourceYield(RuntimeError):
    pass


class ResourceBudget:
    @classmethod
    def for_actor(cls,config):
        return cls(stage='voice-cloning',status_files=config['resource_status_files'],
            coordination_file=config['resource_coordination_file'],hold_file=config['resource_hold_file'],
            reserve_gpu_mib=config['actor_reserve_gpu_mib'],required_gpu_mib=config['actor_required_gpu_mib'],
            process_names=config['resource_process_names'],require_process_absence=config['resource_require_process_absence'])

    def __init__(self, *, status_files=(), coordination_file=None, hold_file=None,
                 reserve_gpu_mib=3072, required_gpu_mib=1024, reserve_ram_mib=12288, stage=None, process_names=(),require_process_absence=False):
        self.stage = stage
        self.status_files = [Path(p) for p in status_files]
        self.coordination_file = Path(coordination_file) if coordination_file else None
        self.hold_file = Path(hold_file) if hold_file else None
        self.reserve_gpu_mib, self.required_gpu_mib = reserve_gpu_mib, required_gpu_mib
        self.reserve_ram_mib = reserve_ram_mib
        self.reason = None
        self._gpu_checked = 0
        self.process_names={str(n).casefold() for n in process_names}
        self.require_process_absence=require_process_absence
        if require_process_absence and not self.process_names:raise ValueError('Exclusive companion guard needs process names')
        self._process_checked=0
        self._process_present=True

    def watched_process_present(self):
        """Read process names only; never send window input or stop a process."""
        if time.monotonic()-self._process_checked<2:return self._process_present
        if os.name!='nt':raise OSError('Process absence detection is only available on Windows')
        from ctypes import wintypes as W
        class Entry(ctypes.Structure):
            _fields_=[('size',W.DWORD),('usage',W.DWORD),('pid',W.DWORD),('heap',ctypes.c_size_t),
                      ('module',W.DWORD),('threads',W.DWORD),('parent',W.DWORD),('priority',W.LONG),
                      ('flags',W.DWORD),('name',W.WCHAR*260)]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateToolhelp32Snapshot.argtypes=[W.DWORD,W.DWORD];kernel.CreateToolhelp32Snapshot.restype=W.HANDLE
        kernel.Process32FirstW.argtypes=[W.HANDLE,ctypes.POINTER(Entry)];kernel.Process32FirstW.restype=W.BOOL
        kernel.Process32NextW.argtypes=[W.HANDLE,ctypes.POINTER(Entry)];kernel.Process32NextW.restype=W.BOOL
        kernel.CloseHandle.argtypes=[W.HANDLE]
        handle=kernel.CreateToolhelp32Snapshot(2,0)
        if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
        present=False;entry=Entry();entry.size=ctypes.sizeof(entry)
        try:
            if not kernel.Process32FirstW(handle,ctypes.byref(entry)):raise ctypes.WinError(ctypes.get_last_error())
            while True:
                if entry.name.casefold() in self.process_names:present=True;break
                if not kernel.Process32NextW(handle,ctypes.byref(entry)):
                    if ctypes.get_last_error()!=18:raise ctypes.WinError(ctypes.get_last_error())
                    break
        finally:kernel.CloseHandle(handle)
        self._process_present=present;self._process_checked=time.monotonic()
        return present

    def voice_busy(self):
        if self.hold_file and self.hold_file.exists():
            return 'GPU hold requested'
        if self.coordination_file and self.coordination_file.is_file():
            try:
                record = json.loads(self.coordination_file.read_text(encoding='utf-8-sig'))
                if record.get('voice_work', {}).get('gpu_benchmark_hold'):
                    return 'Voice measurement holds the GPU'
            except (OSError, ValueError):
                return 'Resource coordination is temporarily unreadable'
        if self.require_process_absence:
            try:
                if self.watched_process_present():return 'Sem restarted; yielding its reserved resources'
            except OSError:return 'Sem process state could not be verified'
        fresh = False
        for path in self.status_files:
            try:
                if not path.is_file() or time.time()-path.stat().st_mtime > 20:
                    continue
                state = json.loads(path.read_text(encoding='utf-8-sig'))
                fresh = True
                voice, resources = state.get('voice', {}), state.get('resources', {})
                if voice.get('active') or voice.get('closing') or state.get('busy') or state.get('codingBusy') or resources.get('preparing') or resources.get('activeJobKind'):
                    return 'Sem is handling active work or a voice call'
            except (OSError, ValueError):
                return 'Active voice state is temporarily unreadable'
        if self.status_files and not fresh:
            if self.process_names:
                try:
                    if not self.watched_process_present():return None
                except OSError:return 'Sem process state could not be verified'
            return 'No fresh voice-state record is available'
        return None

    @staticmethod
    def gpu_free():
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        result = subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                                capture_output=True, timeout=5, check=True, **flags)
        return int(result.stdout.decode().splitlines()[0].strip())

    @staticmethod
    def ram_free():
        if os.name != 'nt':
            return None
        class Memory(ctypes.Structure):
            _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [(n,ctypes.c_ulonglong) for n in
                ('totalPhysical','availablePhysical','totalPage','availablePage','totalVirtual','availableVirtual','extended')]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError('Cannot read available memory')
        return int(memory.availablePhysical/1024**2)

    def admit_gpu(self):
        if self.stage != 'voice-cloning':
            raise ResourceYield('GPU use is restricted to actor voice cloning')
        reason = self.voice_busy()
        free = self.gpu_free()
        ram = self.ram_free()
        if not reason and free < self.reserve_gpu_mib + self.required_gpu_mib:
            reason = 'Insufficient free GPU memory for the stage and voice reserve'
        if not reason and ram is not None and ram < self.reserve_ram_mib:
            reason = 'System memory reserve is in use'
        if reason:
            self.reason = reason
            raise ResourceYield(reason)
        return {'gpu_free_mib':free,'ram_free_mib':ram,'reserve_gpu_mib':self.reserve_gpu_mib}

    def should_yield(self):
        reason = self.voice_busy()
        if not reason and time.monotonic()-self._gpu_checked >= 2:
            self._gpu_checked = time.monotonic()
            try:
                if self.gpu_free() < self.reserve_gpu_mib:
                    reason = 'GPU memory fell below the voice reserve'
                ram = self.ram_free()
                if ram is not None and ram < self.reserve_ram_mib:
                    reason = 'System memory fell below the voice reserve'
            except (OSError, ValueError, subprocess.SubprocessError):
                reason = 'Resource availability could not be read'
        if reason:
            self.reason = reason
        return bool(reason)
