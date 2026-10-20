"""Process-level movie-Vox engine host: an elastic pool of N owned movie-Vox child processes (vox_movie_child.py),
kept warm between windows, preempted at once by any Sem activity.

Why a pool and not a batch API: the approved engine (audio.cpp v0.9.1 + relay/actor_voice.py, VoxCPM2 Q4_K) is
batch-one. Its C ABI renders one request per audiocpp_sem_stream_generate/session_run call, its graph modules assert
batch size 1 ("expected single-batch audio feature tensor"), prompt_cache_slots is 1, and ActorVoice owns one session
per model load. Multi-sequence batching exists only in the separate nano-vLLM VoxCPM2 runtime (BF16 PyTorch,
max_num_seqs), which is not the approved engine and has no parity validation. Concurrency is therefore N processes,
each with its own model copy, CUDA session, cancel markers and measured VRAM.

* Separate processes. Not Sem's live Vox worker (Sem-5.0 q8_voice_worker.py is never touched or shared).
* Jobs. Every request is a job with a job_id, an optional playback deadline (UNIX time) and a priority. The queue is
  earliest-deadline-first (deadline None runs after every deadlined job, FIFO); ties go to the higher priority. A
  queued job whose deadline (+ deadline_grace_seconds) has passed is never started: it fails with
  jobs.DeadlineExpired and the caller keeps that cue's original audio. A running job is never cut short by its
  deadline (no truncation); its receipt records deadline_missed.
* Concurrency knob. `instances` (1..4) is the pool size; set_limit(n) lowers/raises the live limit; pool_capacity()
  turns a VRAM budget into an instance count from the measured per-instance VRAM (else the configured estimate).
  Cold loads are sequential, so each headroom reading reflects every already-loaded instance.
* Elastic. Instance k+1 is started only while every loaded instance is busy and VRAM headroom holds:
  before load  free - required >= reserve   (required = the largest measured instance, else actor_required_gpu_mib)
  after load   free >= reserve              (actor_reserve_gpu_mib, 3072), else that instance is closed again and the
                                            pool is capped at k instances until an instance unloads.
* Kept loaded between windows. Lease-counted per window renderer; after the last lease, instances stay warm for
  `idle_grace_seconds` (150 s), then close.
* Sem always wins, through the movie GPU lease (movie GPU lease contract). This package NEVER opens or polls
  Sem's status.json (an open handle makes Sem's own status rename fail). Nothing runs without a bound lease
  (bind_lease). Lease preemption (voice active/closing, Sem not idle, stale status, PID change, GPU reserve, expiry)
  or release kills every instance at once (on_preempt callback, plus the 0.05 s watcher on lease.cancel_token;
  TerminateProcess frees the VRAM with the process) and fails every queued job with HostHold -> waiting_resources.
  Synthesis stops within 0.2 s of the lease preemption. The child keeps ResourceBudget.should_yield() in every PCM
  callback for the GPU reserve only (no Sem status files are given to it).
* Process ownership (review P1-1). Every engine process is started inside a kill-on-close Job: the bound lease v2's
  Job via lease.spawn() (W5's watcher process ends it at preemption + grace_s, at release, and when this holder dies),
  else this host's own kill-on-close Job (process_owner.py; any death of this process ends the children). Each child
  also watches this process's handle and the lease's cancel event itself (child_watchdog.py), started before its
  model load.
* Bounded cancel without this process's GIL (review P1-2). A cancel writes the marker in the caller's thread; the
  child acknowledges in its next PCM callback or its hard-cancel watchdog exits it hard_cancel_ms (60 ms) after the
  marker appeared. The host's ack-wait-then-kill (cancel_ack_seconds) is only a backup on a background thread.
  cancel_marker(job_id) gives the marker path, so ANOTHER process (a scheduler supervisor) can cancel a line by
  creating that file, without this process's GIL.
* Cancel marker. Each request has `output + ".cancel"`. A scheduler/renderer cancel of a running job creates it, waits
  `cancel_ack_seconds` (0.08 s) for the child's acknowledgement (the instance stays warm) and otherwise kills that
  instance, so synthesis stops within 0.2 s. A queued job is removed immediately.
* VRAM measurement per instance: the child reports device free memory around its own load; the host adds the child's
  per-process WDDM dedicated usage (Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory, the method of
  Sem-Cadence/q8-integration-20261014), after load and every `vram_interval_seconds`. Rows go to
  <log_dir>/vox-movie-engine.jsonl tagged with the instance number.
"""
from __future__ import annotations

import atexit
import collections
import datetime as dt
import heapq
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import approved
from . import process_owner as _owners
from .errors import ActorCancelled
from .jobs import DeadlineExpired

HERE = Path(__file__).resolve().parent
CHILD = HERE / 'vox_movie_child.py'
LOG_NAME = 'vox-movie-engine.jsonl'
MAX_INSTANCES = 4
# PLACEHOLDER, not a footprint: the configured budget actor_required_gpu_mib (2700). The per-instance VRAM of the
# movie-Vox engine is UNMEASURED and is not inferred from the model file size (Codex note 3). Only an in-take WDDM
# reading (peak under load) replaces it; until then pool_capacity() never returns more than 1 instance.
ESTIMATED_INSTANCE_MIB = 2700


def pool_capacity(vram_budget_mib, *, per_instance_mib=None, measured=False, max_instances=MAX_INSTANCES,
                  limit=None, allow_estimate=False):
    """How many movie-Vox instances fit in `vram_budget_mib` (MiB the GPU governor grants to movie TTS AFTER Sem's
    reserve has been taken out). Pure function; the governor (W4) calls it, or VoxEngineHost.pool_capacity().

    Returns {'instances', 'per_instance_mib', 'basis', 'vram_budget_mib', 'max_instances', 'limit', 'headroom_mib'}.
    basis 'measured' = the largest WDDM per-process dedicated reading of a loaded instance; 'estimate' = the config/
    plan estimate (ESTIMATED_INSTANCE_MIB), which a real run must replace before a default > 1 is adopted. On an
    estimate the result is capped at 1 instance unless allow_estimate=True (review P3-3)."""
    per = float(per_instance_mib) if per_instance_mib else float(ESTIMATED_INSTANCE_MIB)
    if per <= 0:
        raise ValueError('per_instance_mib must be positive')
    budget = max(0.0, float(vram_budget_mib))
    cap = int(max_instances if limit is None else min(int(limit), max_instances))
    if not measured and not allow_estimate:
        cap = min(cap, 1)
    count = max(0, min(cap, int(budget // per)))
    return {'instances': count, 'per_instance_mib': round(per, 1), 'basis': 'measured' if measured else 'estimate',
            'vram_budget_mib': round(budget, 1), 'max_instances': int(max_instances), 'limit': cap,
            'headroom_mib': round(budget - count * per, 1)}


class HostHold(Exception):
    """The engine may not run now (Sem activity, VRAM headroom, child budget yield). Caller: waiting_resources."""

    def __init__(self, reason, detail=None, *, fault_code=None):
        self.reason, self.detail = reason, detail
        missing_lease = reason == 'sem_active' and detail == 'no_movie_gpu_lease'
        closed = reason == 'host_closed' or (reason == 'sem_active' and detail == 'host_closed')
        self.fault_kind = 'fatal' if missing_lease else ('cancelled' if closed else 'resource')
        self.fault_code = fault_code or ('vox_lease_not_bound' if missing_lease else 'vox_' + reason)
        super().__init__(reason if detail is None else '%s: %s' % (reason, detail))


class EngineError(Exception):
    """The engine failed. kind: start_failed, ready_timeout, model_unverified, child_exited, timeout, protocol,
    child_error. first_pcm_seconds is set when the failure happened after audio started."""

    def __init__(self, kind, detail=None, *, first_pcm_seconds=None, child_kind=None,
                 fault_kind='fatal', fault_code=None):
        self.kind, self.detail, self.first_pcm_seconds, self.child_kind = kind, detail, first_pcm_seconds, child_kind
        self.fault_kind = fault_kind if fault_kind in ('fatal', 'resource', 'recoverable', 'cancelled') else 'fatal'
        self.fault_code = fault_code or 'vox_' + kind
        super().__init__(kind if detail is None else '%s: %s' % (kind, detail))


def _sem_path_args(command):
    """Arguments of a child command line that name a path under a Sem root (approved.SEM_ROOTS, component-wise)."""
    found = []
    for arg in command:
        text = str(arg)
        if not any(sep in text for sep in ('/', '\\')):
            continue
        try:
            if approved.refused_reference_root(text) == 'sem':
                found.append(text)
        except (OSError, ValueError):
            continue
    return found


def _sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def _stamp():
    return utc_now().isoformat().replace('+00:00', 'Z')


class LeaseActivity:
    """Sem priority from the movie GPU lease (movie GPU lease contract). NO file I/O.

    This package never opens or polls Sem's status.json: on Windows any open handle on it makes Sem's own status rename
    fail. The lease watcher (W5) proves Sem idle without holding the file and sets lease.cancel_token on preemption
    (voice_active, sem_not_idle, status_stale, gpu_reserve, lease_expired, ...) and at release.

    __call__ returns None while the bound lease is live, else a reason: 'no_movie_gpu_lease' (nothing bound: the engine
    never starts without a lease) or 'lease_<reason>' (preempted, expired or released)."""

    def __init__(self):
        self._lease = None
        self._lock = threading.Lock()

    @property
    def lease(self):
        return self._lease

    def bind(self, lease, on_preempt):
        """Bind `lease` (idempotent). on_preempt(event) is registered once per lease (called at once if the lease is
        already preempted, per the contract)."""
        with self._lock:
            if lease is self._lease:
                return False
            self._lease = lease
        lease.on_preempt(on_preempt)
        return True

    def __call__(self):
        lease = self._lease
        if lease is None:
            return 'no_movie_gpu_lease'
        if lease.cancel_token.is_set():
            event = getattr(lease, 'event', None)
            return 'lease_%s' % (getattr(event, 'reason', None) or 'cancelled')
        return None


def nvidia_free_mib():
    """Device free memory (relay ResourceBudget.gpu_free)."""
    flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, timeout=5, check=True, **flags)
    return int(result.stdout.decode().splitlines()[0].strip())


def wddm_process_vram(pid):
    """Per-process WDDM GPU memory of `pid` (same counter as Sem-Cadence/q8-integration-20261014/live-test.cjs)."""
    script = ("$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory"
              " | Where-Object { $_.Name -match '^pid_%d_' } | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)"
              " | ConvertTo-Json -Depth 3" % int(pid))
    flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    done = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script], capture_output=True,
                          timeout=30, check=True, **flags)
    text = done.stdout.decode('utf-8', 'replace').strip()
    rows = json.loads(text) if text else []
    rows = rows if isinstance(rows, list) else [rows]
    dedicated = sum(int(r.get('DedicatedUsage') or 0) for r in rows)
    shared = sum(int(r.get('SharedUsage') or 0) for r in rows)
    return {'pid': int(pid), 'process_dedicated_bytes': dedicated, 'process_dedicated_mib': round(dedicated / 2 ** 20, 1),
            'process_shared_mib': round(shared / 2 ** 20, 1), 'rows': rows,
            'method': 'WDDM Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory (accounting, not CUDA peak)'}


class _Log:
    def __init__(self, log_dir):
        self.dir = Path(log_dir) if log_dir else None
        self.events = collections.deque(maxlen=600)
        self._lock = threading.Lock()

    def __call__(self, event, **fields):
        row = {'at': _stamp(), 'event': event, **fields}
        with self._lock:
            self.events.append(row)
            if self.dir is not None:
                try:
                    self.dir.mkdir(parents=True, exist_ok=True)
                    with (self.dir / LOG_NAME).open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, default=str) + '\n')
                except OSError:
                    pass
        return row


class _Child:
    def __init__(self, process, generation, log):
        self.process, self.generation, self.log = process, generation, log
        self.ready = threading.Event()
        self.payload = None
        self.failure = None
        self.exit_event = None
        self.reader = None

    def alive(self):
        return self.process.poll() is None


class VoxEngineInstance:
    """One movie-Vox child process (one model copy, one CUDA session, one request at a time)."""

    def __init__(self, index, *, manifest, relay_root, python, budget, capacity_seconds, reserve_mib, required_mib,
                 sem_activity, gpu_free, vram_probe, log, log_dir, child_command, cancel_ack_seconds, ready_timeout,
                 request_timeout, idle_grace_seconds, expected_model_sha256, clock, process_owner=None,
                 lease_event=lambda: None, host_closed=lambda: False, hard_cancel_ms=60.0, reserve_ram_mib=None,
                 gpu_sustain_seconds=1.5, sibling_pins=None, vram_busy_interval_seconds=5.0):
        self.index = index
        self.manifest, self.relay_root, self.python = str(manifest), str(relay_root), str(python)
        self.budget, self.capacity_seconds = dict(budget), float(capacity_seconds)
        self.reserve_mib, self.required_mib = int(reserve_mib), int(required_mib)
        self.sem_activity, self.gpu_free, self.vram_probe = sem_activity, gpu_free, vram_probe
        self.log, self.log_dir, self.child_command = log, (Path(log_dir) if log_dir else None), child_command
        self.cancel_ack_seconds, self.ready_timeout = float(cancel_ack_seconds), float(ready_timeout)
        self.request_timeout, self.idle_grace_seconds = float(request_timeout), float(idle_grace_seconds)
        self.expected_model_sha256, self.clock = expected_model_sha256, clock
        self.process_owner, self.lease_event, self.host_closed = process_owner, lease_event, host_closed
        self.hard_cancel_ms, self.reserve_ram_mib = float(hard_cancel_ms), reserve_ram_mib
        self.gpu_sustain_seconds, self.sibling_pins = float(gpu_sustain_seconds), dict(sibling_pins or {})
        self.vram_busy_interval_seconds = float(vram_busy_interval_seconds)
        self._last_take_probe = None
        self.idle_footprint_mib = None      # largest WDDM reading while no take ran (NOT a capacity basis)
        self.peak_mib = None                # largest WDDM reading during a take (the capacity basis, review P2-2)
        self.take_probes = 0
        self.owner_kind = None
        self._lock = threading.RLock()
        self._request_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._generations = itertools.count(1)
        self._child = None
        self._exiting = []          # Popens killed/finished but not yet proven exited (R-INT-1)
        self._loading = False
        self._inflight = None
        self._last_probe = None
        self._probing = False
        self._preempted = None
        self.busy_since = None
        self.last_used = clock()
        self.last_vram = {}
        self.measured_dedicated_mib = None
        self.stats = {'loads': 0, 'releases': 0, 'cancels': 0, 'kills': 0, 'requests': 0, 'takes': 0,
                      'audio_seconds': 0.0, 'busy_seconds': 0.0,
                      'cancel_stop_seconds': collections.deque(maxlen=50)}

    # -- state ------------------------------------------------------------------
    def loaded(self):
        child = self._child
        return child is not None and child.ready.is_set() and child.alive()

    def busy(self):
        return self._request_lock.locked()

    def health(self):
        child = self._child
        payload = dict(child.payload or {}) if child is not None else {}
        return {'instance': self.index, 'status': 'loaded' if self.loaded() else ('loading' if self._loading else 'idle'),
                'busy': self.busy(), 'pid': payload.get('pid'), 'generation': child.generation if child else None,
                'model_sha256': payload.get('model_sha256'), 'quantization': payload.get('quantization'),
                'backend': payload.get('backend'), 'device': payload.get('device'),
                'measured_dedicated_mib': self.measured_dedicated_mib, 'idle_footprint_mib': self.idle_footprint_mib,
                'peak_mib': self.peak_mib, 'take_probes': self.take_probes, 'process_owner': self.owner_kind,
                'watchdogs': (child.payload or {}).get('watchdogs') if child is not None else None,
                'last_vram': dict(self.last_vram),
                'stats': {k: (list(v) if isinstance(v, collections.deque) else v) for k, v in self.stats.items()}}

    # -- child lifecycle ----------------------------------------------------------
    def watchdog_args(self):
        """In-child watchdog arguments (child_watchdog.py): parent = this (holder) process, the bound lease's cancel
        event, the hard-cancel window. Fake children in tests take the same arguments."""
        args = ['--parent-pid', str(os.getpid()), '--hard-cancel-ms', repr(self.hard_cancel_ms)]
        event = self.lease_event()
        if event:
            args += ['--cancel-event', str(event)]
        return args

    def _command(self, budget_file):
        if self.child_command is not None:
            return list(self.child_command(self, budget_file))
        pins = {name: self.sibling_pins.get(name) or _sha256_file(HERE / name) for name in ('approved.py', 'child_watchdog.py')}
        command = [self.python, '-I', '-B', '-X', 'utf8', str(CHILD), '--relay-root', self.relay_root,
                   '--manifest', self.manifest, '--budget', str(budget_file), '--capacity', repr(self.capacity_seconds),
                   '--idle-seconds', repr(self.idle_grace_seconds + 60.0), '--actor-voice-sha256',
                   approved.ACTOR_VOICE_SHA256, '--sibling-pins', json.dumps(pins, sort_keys=True),
                   '--gpu-sustain-seconds', repr(self.gpu_sustain_seconds)] + self.watchdog_args()
        if self.reserve_ram_mib is not None:
            command += ['--reserve-ram-mib', str(int(self.reserve_ram_mib))]
        return command

    def _budget_file(self):
        folder = self.log_dir or Path(os.environ.get('TEMP') or '.')
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ('vox-movie-resource-budget-%d.json' % self.index)
        payload = {'stage': 'voice-cloning', 'reserve_gpu_mib': self.reserve_mib, 'required_gpu_mib': self.required_mib,
                   **self.budget}
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=1), encoding='utf-8')
        os.replace(temporary, path)
        return path

    def _spawn(self):
        log = None
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name == 'nt' \
            else {'start_new_session': True}
        try:
            budget_file = self._budget_file()
            env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            env.pop('CUDA_VISIBLE_DEVICES', None)
            log = (self.log_dir / ('vox-movie-child-%d.log' % self.index)).open('ab') if self.log_dir is not None else None
            owner = self.process_owner() if self.process_owner is not None else None
            command = self._command(budget_file)
            sem_args = _sem_path_args(command)
            if sem_args:
                # lease v2 absent mode treats a Sem path on a command line as a Sem start and preempts: never spawn one.
                raise EngineError('start_failed', 'child command line names a Sem path: %s' % sem_args,
                                  fault_code='vox_protected_path')
        except (HostHold, EngineError):
            if log is not None:
                log.close()
            raise
        except Exception as error:   # noqa: BLE001 - configuration/I/O preparation cannot be solved by more VRAM
            if log is not None:
                log.close()
            raise EngineError('start_failed', str(error), fault_code='vox_startup_configuration') from error
        spawn = owner.spawn if owner is not None else subprocess.Popen
        try:
            process = spawn(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=log if log is not None else subprocess.DEVNULL, text=True, encoding='utf-8',
                            bufsize=1, env=env, cwd=str(HERE), **flags)
        except OSError as error:
            if log is not None:
                log.close()
            code = 'vox_runtime_missing' if isinstance(error, FileNotFoundError) else 'vox_spawn_failed'
            if isinstance(error, PermissionError):
                code = 'vox_spawn_permission'
            raise EngineError('start_failed', str(error), fault_code=code) from error
        except Exception as error:   # noqa: BLE001 - LeasePreempted from lease.spawn(): the lease Job is gone
            if log is not None:
                log.close()
            if type(error).__name__ in ('LeasePreempted', 'ViewEnded'):   # actual preemption, not malformed lease setup
                raise HostHold('sem_active', 'lease ended before spawn: %s' % (error,)) from error
            raise EngineError('start_failed', str(error), fault_code='vox_spawn_policy') from error
        self.owner_kind = owner.kind if owner is not None else 'unowned-popen'
        self.log('spawned', instance=self.index, pid=process.pid, process_owner=self.owner_kind)
        child = _Child(process, next(self._generations), log)

        def read():
            try:
                for raw in process.stdout:
                    try:
                        message = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(message, dict):
                        self._on_message(child, message)
            except (OSError, ValueError):
                pass
            finally:
                self._on_eof(child)
        child.reader = threading.Thread(target=read, name='vox-movie-read-%d' % self.index, daemon=True)
        child.reader.start()
        return child

    def _on_message(self, child, message):
        if message.get('ready') is True:
            child.payload = message
            child.ready.set()
            return
        inflight = self._inflight
        if inflight is not None and inflight['child'] is child and message.get('id') == inflight['id']:
            inflight['response'] = message
            inflight['done'].set()
            return
        if message.get('id') is None:
            if message.get('event'):
                child.exit_event = message
            elif message.get('error') and child.failure is None:
                child.failure = message

    def _on_eof(self, child):
        inflight = self._inflight
        if inflight is not None and inflight['child'] is child and not inflight['done'].is_set():
            last = child.exit_event or child.failure or {}
            inflight['response'] = {'id': inflight['id'], 'error': last.get('error', 'Movie Vox process closed its stream'),
                                    'kind': last.get('kind', 'ChildExited'), 'eof': True,
                                    'fault_kind': last.get('fault_kind', 'fatal'),
                                    'fault_code': last.get('fault_code', 'vox_child_exited')}
            inflight['done'].set()

    def _kill(self, child, reason):
        if child is None:
            return
        if child.alive():
            try:
                child.process.kill()      # TerminateProcess on Windows: synthesis stops immediately
            except OSError:
                pass
            self.stats['kills'] += 1
            try:
                child.process.wait(timeout=.05)
            except subprocess.TimeoutExpired:
                pass
        self.log('child_killed', instance=self.index, reason=reason, generation=child.generation,
                 pid=child.process.pid, exited=child.process.poll() is not None)

    def _finish(self, child, reason):
        if child is None:
            return
        for stream in (child.process.stdin, child.process.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        if child.reader is not None and child.reader is not threading.current_thread():
            child.reader.join(timeout=1)
        if child.log is not None:
            try:
                child.log.close()
            except OSError:
                pass
        with self._lock:
            if self._child is child:
                self._child = None
            # R-INT-1: a killed/finished process is remembered until it has really EXITED (poll() is not None), so
            # drain() can prove it; _kill() waits only 50 ms after TerminateProcess.
            if child.process.poll() is None:
                self._exiting.append(child.process)

    def _finish_timed_out_child(self, child):
        """Fallback is safe only after both the launcher and the ready-reporting engine have exited."""
        engine_pid = (child.payload or {}).get('pid')
        engine_exited = engine_pid == child.process.pid
        engine_handle = None
        kernel = None
        if not engine_exited and isinstance(engine_pid, int) and engine_pid > 0 and os.name == 'nt':
            # A Windows venv python.exe can be a redirector with a different PID from the engine. Its
            # Popen exit alone is not proof that the engine stopped. Retain an exit-wait handle before killing.
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.CloseHandle.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            engine_handle = kernel.OpenProcess(0x00100000, False, int(engine_pid))  # SYNCHRONIZE only
            if not engine_handle:
                engine_exited = ctypes.get_last_error() == 87  # ERROR_INVALID_PARAMETER: PID no longer exists
        try:
            self._kill(child, 'request_timeout')
            self._finish(child, 'request_timeout')
            if engine_handle:
                engine_exited = kernel.WaitForSingleObject(engine_handle, 1000) == 0  # WAIT_OBJECT_0
            return child.process.poll() is not None and engine_exited
        finally:
            if engine_handle:
                kernel.CloseHandle(engine_handle)

    def processes(self):
        """Every engine Popen of this instance not yet proven exited: the current child plus finished/killed ones."""
        with self._lock:
            self._exiting = [p for p in self._exiting if p.poll() is None]
            current = [self._child.process] if self._child is not None else []
            return current + [p for p in self._exiting if p not in current]

    def _unload(self, child, reason, *, graceful_seconds=.3):
        if child is None:
            return
        started = self.clock()
        if child.alive():
            if graceful_seconds > 0:
                try:
                    child.process.stdin.write('{"action": "close"}\n')
                    child.process.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    child.process.wait(timeout=graceful_seconds)
                except subprocess.TimeoutExpired:
                    pass
            if child.alive():
                self._kill(child, reason + (':close_timeout' if graceful_seconds > 0 else ''))
        self._finish(child, reason)
        self.stats['releases'] += 1
        self.log('released', instance=self.index, reason=reason, generation=child.generation,
                 seconds=round(self.clock() - started, 4))

    def _check_ready_payload(self, payload):
        problems = []
        if payload.get('protocol') != approved.PROTOCOL:
            problems.append('protocol')
        if payload.get('model_sha256') != self.expected_model_sha256:
            problems.append('model_sha256')
        if payload.get('quantization') != approved.QUANTIZATION:
            problems.append('quantization')
        if payload.get('backend') != approved.SETTINGS['backend'] or payload.get('device') != approved.SETTINGS['device']:
            problems.append('backend')
        if payload.get('actor_voice_sha256') != approved.ACTOR_VOICE_SHA256:
            problems.append('actor_voice_sha256')
        if payload.get('settings') != approved.SETTINGS:
            problems.append('settings')
        if float(payload.get('capacity_seconds', -1)) != self.capacity_seconds:
            problems.append('capacity_seconds')
        return problems

    def ensure_loaded(self, cancelled, required_mib=None):
        if self.loaded():
            return self._child
        if self._child is not None:
            self._finish(self._child, 'exited')
        required = int(required_mib or self.required_mib)
        self._preempted = None
        if self.host_closed():
            raise HostHold('host_closed')
        reason = self.sem_activity()
        if reason:
            raise HostHold('sem_active', reason)
        try:
            free = int(self.gpu_free())
        except Exception as error:   # noqa: BLE001
            raise HostHold('vram_unreadable', str(error)) from error
        if free - required < self.reserve_mib:
            self.log('headroom_hold', instance=self.index, stage='before_load', free_mib=free, required_mib=required,
                     reserve_mib=self.reserve_mib)
            raise HostHold('vram_headroom', 'free %d MiB - required %d MiB < reserve %d MiB'
                           % (free, required, self.reserve_mib))
        self._loading = True
        started = self.clock()
        try:
            child = self._spawn()
            with self._lock:
                self._child = child
            while not child.ready.wait(.02):
                if not child.alive() and not child.ready.is_set():
                    child.reader.join(timeout=1)
                    failure = child.failure or child.exit_event or {}
                    self._finish(child, 'load_failed')
                    if self._preempted is not None:
                        raise HostHold('sem_active', self._preempted)
                    kind = failure.get('kind')
                    if kind == 'ResourceYield' or failure.get('fault_kind') == 'resource':
                        raise HostHold('child_yield', failure.get('error'), fault_code=failure.get('fault_code'))
                    if kind == 'ModelUnverified':
                        raise EngineError('model_unverified', failure.get('error'), child_kind=kind,
                                          fault_code=failure.get('fault_code'))
                    raise EngineError('start_failed', failure.get('error') or 'child exited before ready', child_kind=kind,
                                      fault_code=failure.get('fault_code'))
                if cancelled():
                    self._kill(child, 'cancelled_during_load')
                    self._finish(child, 'cancelled_during_load')
                    raise ActorCancelled('Movie Vox load cancelled')
                activity = self.sem_activity()
                if activity:
                    self._kill(child, 'sem_active_during_load')
                    self._finish(child, 'sem_active_during_load')
                    raise HostHold('sem_active', activity)
                if self.clock() - started > self.ready_timeout:
                    self._kill(child, 'ready_timeout')
                    self._finish(child, 'ready_timeout')
                    raise EngineError('ready_timeout', 'movie Vox child not ready in %.0fs' % self.ready_timeout)
            if self.host_closed():
                self._kill(child, 'host_closed_during_load')
                self._finish(child, 'host_closed_during_load')
                raise HostHold('host_closed')
            problems = self._check_ready_payload(child.payload)
            if problems:
                self._kill(child, 'ready_payload_mismatch')
                self._finish(child, 'ready_payload_mismatch')
                raise EngineError('model_unverified', 'ready payload mismatch: %s' % ', '.join(problems))
            try:
                after = int(self.gpu_free())
            except Exception as error:   # noqa: BLE001
                self._unload(child, 'post_load_vram_unreadable')
                raise HostHold('vram_unreadable', str(error)) from error
            self.last_vram = {'device_free_before_load_mib': free, 'device_free_after_load_mib': after,
                              'device_delta_mib': free - after, 'child_reported': dict(child.payload.get('vram') or {}),
                              'reserve_mib': self.reserve_mib, 'required_mib': required, 'pid': child.payload.get('pid')}
            self.stats['loads'] += 1
            self.log('loaded', instance=self.index, generation=child.generation,
                     load_seconds=round(self.clock() - started, 4), child_load_seconds=child.payload.get('load_seconds'),
                     **self.last_vram)
            if after < self.reserve_mib:
                self.log('headroom_hold', instance=self.index, stage='after_load', free_mib=after,
                         reserve_mib=self.reserve_mib)
                self._unload(child, 'post_load_reserve')
                raise HostHold('vram_headroom', 'after load free %d MiB < reserve %d MiB' % (after, self.reserve_mib))
            self.probe_async('after_load')
            return child
        finally:
            self._loading = False

    def probe_async(self, stage):
        child = self._child
        if self._probing or child is None:
            return
        self._probing = True
        self._last_probe = self.clock()
        pid = (child.payload or {}).get('pid') or child.process.pid

        def run():
            try:
                measured = self.vram_probe(pid)
                self.last_vram = {**self.last_vram, 'process': {k: v for k, v in measured.items() if k != 'rows'},
                                  'process_stage': stage}
                mib = measured.get('process_dedicated_mib')
                if isinstance(mib, (int, float)) and mib > 0:
                    mib = float(mib)
                    self.measured_dedicated_mib = max(self.measured_dedicated_mib or 0, mib)
                    if stage == 'in_take':
                        self.peak_mib = max(self.peak_mib or 0, mib)
                        self.take_probes += 1
                    else:
                        self.idle_footprint_mib = max(self.idle_footprint_mib or 0, mib)
                self.log('vram_process', instance=self.index, stage=stage, generation=child.generation,
                         **{k: v for k, v in measured.items() if k != 'rows'})
            except Exception as error:   # noqa: BLE001 - measurement only; never affects rendering
                self.log('vram_process_error', instance=self.index, stage=stage, error=str(error)[:300])
            finally:
                self._probing = False
        threading.Thread(target=run, name='vox-movie-vram-%d' % self.index, daemon=True).start()

    # -- requests -------------------------------------------------------------------
    def synthesize(self, request, *, cancelled=lambda: False, token=None, required_mib=None):
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError('instance %d is already serving a request' % self.index)
        self.busy_since = self.clock()
        try:
            child = self.ensure_loaded(cancelled, required_mib)
            output = Path(request['output'])
            marker = Path(str(output) + '.cancel')
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
            rid = 'i%d-r%d' % (self.index, next(self._ids))
            inflight = {'id': rid, 'child': child, 'marker': marker, 'token': token, 'done': threading.Event(),
                        'response': None, 'cancel_reason': None, 'started': self.clock()}
            with self._lock:
                self._inflight = inflight
            self.stats['requests'] += 1
            line = json.dumps({'action': 'synthesize', 'id': rid, 'text': request['text'],
                               'reference': str(request['reference']),
                               'reference_sha256': request.get('reference_sha256'),
                               'reference_text': request.get('reference_text', ''),
                               'seed': int(request['seed']), 'output': str(output)})
            try:
                child.process.stdin.write(line + '\n')
                child.process.stdin.flush()
            except (OSError, ValueError) as error:
                if inflight['cancel_reason'] is None:
                    self._kill(child, 'stdin_failed')
                    self._finish(child, 'stdin_failed')
                    raise EngineError('child_exited', 'request write failed: %s' % (error,)) from error
            while not inflight['done'].wait(.01):
                if cancelled():
                    self.cancel_inflight(inflight, 'owner_cancelled')
                    break
                if self.clock() - inflight['started'] > self.request_timeout:
                    exited = self._finish_timed_out_child(child)
                    if inflight['cancel_reason'] is not None:
                        break  # A concurrent owner cancel or lease preemption keeps its distinct outcome.
                    fault_kind = 'recoverable' if exited else 'fatal'
                    fault_code = 'vox_timeout' if exited else 'vox_timeout_drain_unconfirmed'
                    self.log('request_timeout', instance=self.index, generation=child.generation, request_id=rid,
                             pid=child.process.pid, engine_pid=(child.payload or {}).get('pid'),
                             child_exit_confirmed=exited, fault_kind=fault_kind, fault_code=fault_code)
                    raise EngineError('timeout', 'no response in %.0fs; child exit %s' %
                                      (self.request_timeout, 'confirmed' if exited else 'unconfirmed'),
                                      fault_kind=fault_kind, fault_code=fault_code)
            response = inflight['response'] or {}
            reason = inflight['cancel_reason']
            if reason is not None:
                try:
                    marker.unlink()
                except OSError:
                    pass
                if not child.alive():
                    self._finish(child, 'cancel_killed')
                if reason.startswith('sem_activity'):
                    raise HostHold('sem_active', reason)
                raise ActorCancelled('Movie Vox request cancelled: %s' % reason)
            if response.get('receipt') is not None:
                wall = self.clock() - inflight['started']
                self.stats['takes'] += 1
                self.stats['busy_seconds'] = round(self.stats['busy_seconds'] + wall, 6)
                self.stats['audio_seconds'] = round(self.stats['audio_seconds'] + float(response.get('seconds') or 0), 6)
                return {**response, 'wall_seconds': wall, 'instance': self.index,
                        'pid': (child.payload or {}).get('pid'), 'generation': child.generation}
            if response.get('cancelled'):
                raise ActorCancelled('Movie Vox request cancelled by its marker')
            kind, first = response.get('kind'), response.get('first_pcm_seconds')
            if not response.get('eof'):
                try:
                    child.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._kill(child, 'error_exit_timeout')
            self._finish(child, 'request_error')
            if kind == 'ResourceYield' or response.get('fault_kind') == 'resource':
                raise HostHold('child_yield', response.get('error'), fault_code=response.get('fault_code'))
            raise EngineError('child_exited' if response.get('eof') else 'child_error', response.get('error'),
                              first_pcm_seconds=first, child_kind=kind,
                              fault_kind=response.get('fault_kind', 'fatal'), fault_code=response.get('fault_code'))
        finally:
            with self._lock:
                self._inflight = None
            self.busy_since = None
            self.last_used = self.clock()
            self._request_lock.release()

    def mark_cancel(self, inflight=None, reason='cancelled', token=None):
        """Fast path, in the caller's thread (one GIL hop): set the reason and write the cancel marker. From here the
        CHILD stops by itself: it acknowledges in its next PCM callback, or its hard-cancel watchdog exits it
        hard_cancel_ms after the marker appeared. Neither needs this process's GIL (review P1-2)."""
        inflight = inflight or self._inflight
        if inflight is None or (token is not None and inflight['token'] is not token) or inflight['done'].is_set():
            return None
        if inflight['cancel_reason'] is None:
            inflight['cancel_reason'] = reason
        inflight.setdefault('marked_at', self.clock())
        try:
            inflight['marker'].touch()
        except OSError:
            try:
                inflight['marker'].parent.mkdir(parents=True, exist_ok=True)
                inflight['marker'].touch()
            except OSError:
                pass
        return inflight

    def cancel_inflight(self, inflight=None, reason='cancelled', *, immediate=False):
        """Backup path: after mark_cancel, wait cancel_ack_seconds for the child (ack, or its own hard exit), else kill
        it from this process. Returns stop seconds."""
        inflight = inflight or self._inflight
        if inflight is None:
            return None
        started = self.clock()
        if inflight['done'].is_set() and not immediate:
            return 0.0
        if inflight['cancel_reason'] is None:
            inflight['cancel_reason'] = reason
        acknowledged = False
        if not immediate:
            self.mark_cancel(inflight, reason)
            acknowledged = inflight['done'].wait(self.cancel_ack_seconds)
        if not acknowledged:
            self._kill(inflight['child'], ('preempt:' if immediate else 'cancel_ack_timeout:') + reason)
            if not inflight['done'].is_set():
                inflight['response'] = {'id': inflight['id'], 'cancelled': True, 'killed': True}
                inflight['done'].set()
        seconds = self.clock() - started
        self.stats['cancels'] += 1
        self.stats['cancel_stop_seconds'].append(round(seconds, 4))
        self.log('cancelled', instance=self.index, reason=reason, acknowledged=acknowledged, immediate=immediate,
                 stop_seconds=round(seconds, 4), generation=inflight['child'].generation)
        return seconds

    def cancel(self, token, reason):
        inflight = self.mark_cancel(None, reason, token=token)
        if inflight is None:
            return None
        return self.cancel_inflight(inflight, reason)

    def preempt(self, reason):
        """Sem activity: kill now (no acknowledgement wait). Returns seconds until the process is gone."""
        started = self.clock()
        self._preempted = reason
        inflight = self._inflight
        if inflight is not None:
            inflight['cancel_reason'] = 'sem_activity:' + reason      # Sem wins over any earlier cancel reason
            self.cancel_inflight(inflight, 'sem_activity:' + reason, immediate=True)
        child = self._child
        if child is not None:
            self._unload(child, 'sem_activity:' + reason, graceful_seconds=0)
        return self.clock() - started

    def release_idle(self, reason='idle_grace'):
        child = self._child
        if child is not None and self._inflight is None:
            self._unload(child, reason)

    def tick(self, vram_interval_seconds):
        child = self._child
        if child is None or self._loading or not child.ready.is_set():
            return
        if not child.alive():
            if self._inflight is None:
                self.log('child_exited', instance=self.index, generation=child.generation, code=child.process.poll(),
                         event=(child.exit_event or {}).get('event'))
                self._finish(child, 'exited')
            return
        now = self.clock()
        inflight = self._inflight
        if inflight is not None:
            # Peak under load (review P2-2): sample during takes, at most every vram_busy_interval_seconds, once a
            # take has run for 0.5 s (past the reference encode / prefill).
            if now - inflight['started'] >= .5 and (self._last_take_probe is None
                                                    or now - self._last_take_probe >= self.vram_busy_interval_seconds):
                self._last_take_probe = now
                self.probe_async('in_take')
        elif self._last_probe is not None and now - self._last_probe > vram_interval_seconds:
            self.probe_async('periodic')


class Lease:
    def __init__(self, host):
        self.host, self.released = host, False

    def release(self):
        if not self.released:
            self.released = True
            self.host._release_lease()


class PoolJob:
    """One line's engine request in the pool queue."""
    _seq = itertools.count(1)

    def __init__(self, request, *, job_id, deadline, priority, token, cancelled):
        self.request, self.job_id, self.token = request, str(job_id), token
        self.deadline = None if deadline is None else float(deadline)
        self.priority = int(priority)
        self.seq = next(PoolJob._seq)
        self.owner_cancelled = cancelled
        self.cancel_reason = None
        self.done = threading.Event()
        self.response = None
        self.error = None
        self.state = 'queued'
        self.instance = None
        self.submitted_at = time.time()
        self.started_at = None
        self.finished_at = None

    @property
    def key(self):
        return (self.deadline is None, self.deadline if self.deadline is not None else 0.0, -self.priority, self.seq)

    def should_stop(self):
        return self.cancel_reason is not None or bool(self.owner_cancelled())

    def scheduling(self):
        finished = self.finished_at or time.time()
        return {'job_id': self.job_id, 'deadline': self.deadline, 'priority': self.priority, 'instance': self.instance,
                'submitted_at': self.submitted_at, 'started_at': self.started_at, 'finished_at': self.finished_at,
                'queue_wait_seconds': None if self.started_at is None else round(self.started_at - self.submitted_at, 6),
                'deadline_slack_seconds': None if self.deadline is None else round(self.deadline - finished, 6),
                'deadline_missed': None if self.deadline is None else finished > self.deadline}


class VoxEngineHost:
    """Process-level pool of movie-Vox instances (see module docstring). `instances` is the maximum pool size."""

    def __init__(self, *, manifest, relay_root, python=None, budget=None, capacity_seconds=10.0, reserve_mib=3072,
                 required_mib=2700, instances=1, idle_grace_seconds=150.0, sem_activity=None,
                 gpu_free=None, vram_probe=None, log_dir=None, child_command=None, poll_seconds=0.05,
                 cancel_ack_seconds=0.15, ready_timeout=120.0, request_timeout=180.0, vram_interval_seconds=60.0,
                 expected_model_sha256=approved.MODEL_SHA256, clock=time.perf_counter, deadline_grace_seconds=0.0,
                 wall_time=time.time, process_owner='auto', hard_cancel_ms=60.0, reserve_ram_mib=None,
                 gpu_sustain_seconds=1.5, sibling_pins=None, vram_busy_interval_seconds=5.0, initial_limit=None,
                 refusal_hold_seconds=300.0):
        if not isinstance(instances, int) or isinstance(instances, bool) or not 1 <= instances <= MAX_INSTANCES:
            raise ValueError('movie Vox instances must be 1..%d' % MAX_INSTANCES)
        if int(reserve_mib) < 3072:
            raise EngineError('start_failed', 'movie Vox requires a free GPU floor of at least 3072 MiB',
                              fault_code='vox_reserve_configuration')
        self.deadline_grace_seconds = float(deadline_grace_seconds)
        self.wall_time = wall_time
        # concurrency knob (set_limit); never above the constructed pool. FIX 6 auto mode: a fresh process starts at 1.
        self._limit = max(1, min(instances, int(initial_limit))) if initial_limit else instances
        self.refusal_hold_seconds = float(refusal_hold_seconds)
        self._refused_at = None      # last resource refusal (headroom, Sem/lease hold, child yield): k falls back to 1
        self.manifest, self.relay_root = str(manifest), str(relay_root)
        self.capacity_seconds = float(capacity_seconds)
        self.reserve_mib, self.required_mib = int(reserve_mib), int(required_mib)
        self.max_instances, self.idle_grace_seconds = instances, float(idle_grace_seconds)
        # Sem priority: the movie GPU lease (bind_lease), or an injected callable (tests, measurement). Never a file.
        self.sem_activity = sem_activity if sem_activity is not None else LeaseActivity()
        self.poll_seconds, self.vram_interval_seconds = float(poll_seconds), float(vram_interval_seconds)
        self.expected_model_sha256, self.clock = expected_model_sha256, clock
        self.log = _Log(log_dir)
        self._instance_options = dict(
            manifest=manifest, relay_root=relay_root, python=python or sys.executable, budget=dict(budget or {}),
            capacity_seconds=capacity_seconds, reserve_mib=reserve_mib, required_mib=required_mib,
            sem_activity=self.sem_activity, gpu_free=gpu_free if gpu_free is not None else nvidia_free_mib,
            vram_probe=vram_probe if vram_probe is not None else wddm_process_vram, log=self.log, log_dir=log_dir,
            child_command=child_command, cancel_ack_seconds=cancel_ack_seconds, ready_timeout=ready_timeout,
            request_timeout=request_timeout, idle_grace_seconds=idle_grace_seconds,
            expected_model_sha256=expected_model_sha256, clock=clock, process_owner=self._owner,
            lease_event=self._lease_event, host_closed=lambda: self._closed, hard_cancel_ms=hard_cancel_ms,
            reserve_ram_mib=reserve_ram_mib, gpu_sustain_seconds=gpu_sustain_seconds, sibling_pins=sibling_pins,
            vram_busy_interval_seconds=vram_busy_interval_seconds)
        # Process ownership (review P1-1, Codex note 2):
        #   'lease' (production, factory.host_options): every engine is spawned INTO the bound lease v2's Job via
        #           lease.spawn(); a lease without spawn() (or no lease) refuses to start an engine (HostHold);
        #   'auto'  (tests/compat): the lease's Job when it has spawn(), else this host's own kill-on-close Job;
        #   an object with spawn()/describe()/close() may be injected (tests).
        if not (process_owner in ('lease', 'auto', None) or hasattr(process_owner, 'spawn')):
            raise ValueError("process_owner must be 'lease', 'auto' or an owner object")
        self._owner_mode = process_owner
        self._own_job = None
        self._lease_owner = None
        self.instances = [VoxEngineInstance(i, **self._instance_options) for i in range(instances)]
        self._cv = threading.Condition(threading.RLock())
        self._queue = []
        self._running = {}          # instance index -> PoolJob
        self._cold = set()          # instance indexes dispatched while not loaded (load in progress)
        self._jobs = {}             # seq -> PoolJob (queued or running); several jobs may share a job_id
        self._vram_cap = instances  # instances allowed by the last headroom check
        self._leases = 0
        self._idle_since = clock()
        self._closed = False
        self._owner_lock = threading.Lock()
        self.stats = {'preemptions': 0, 'preempt_seconds': collections.deque(maxlen=50), 'jobs': 0, 'requeued': 0,
                      'deadline_expired': 0}
        self._dispatcher = threading.Thread(target=self._dispatch, name='vox-movie-dispatch', daemon=True)
        self._watcher = threading.Thread(target=self._watch, name='vox-movie-watch', daemon=True)
        self._dispatcher.start()
        self._watcher.start()

    # -- process ownership -------------------------------------------------------------
    def _bound_lease(self):
        return self.sem_activity.lease if isinstance(self.sem_activity, LeaseActivity) else None

    def _lease_event(self):
        return getattr(self._bound_lease(), 'cancel_event_name', None)

    def _owner(self):
        """The owner for the next engine process (see __init__)."""
        if self._owner_mode not in ('auto', None) and hasattr(self._owner_mode, 'spawn'):
            return self._owner_mode
        lease = self._bound_lease()
        with self._owner_lock:
            if lease is not None and callable(getattr(lease, 'spawn', None)):
                if self._lease_owner is None or self._lease_owner.lease is not lease:
                    self._lease_owner = _owners.LeaseProcessOwner(lease)
                return self._lease_owner
            if self._owner_mode == 'lease':
                raise EngineError('start_failed', 'no movie GPU lease with a Job (lease v2 spawn) is bound; engine not started',
                                  fault_code='vox_lease_not_bound')
            if self._own_job is None:
                self._own_job = _owners.OwnJobProcessOwner()
            return self._own_job

    def owner_status(self):
        with self._owner_lock:
            return {'mode': self._owner_mode if isinstance(self._owner_mode, str) else 'injected',
                    'own_job': self._own_job.describe() if self._own_job else None,
                    'lease_job': self._lease_owner.describe() if self._lease_owner else None}

    # -- movie GPU lease (Sem priority) ------------------------------------------------
    def bind_lease(self, lease):
        """Bind the process's movie GPU lease (W5 lease_client.Lease or a compatible object). Its preemption kills
        every instance at once: on_preempt starts host.preempt on its own thread (the lease watcher's callback stays
        fast); the host watcher also sees lease.cancel_token within poll_seconds. Returns True when newly bound."""
        if not isinstance(self.sem_activity, LeaseActivity):
            raise TypeError('this host was built with an injected Sem-activity callable, not the movie GPU lease')

        def on_preempt(event):
            reason = 'lease_%s' % (getattr(event, 'reason', None) or 'preempted')
            threading.Thread(target=self.preempt, args=(reason,), name='vox-movie-lease-preempt', daemon=True).start()
        bound = self.sem_activity.bind(lease, on_preempt)
        if bound:
            self.log('lease_bound', lease_id=getattr(lease, 'lease_id', None), budget_mib=getattr(lease, 'budget_mib', None))
        return bound

    # -- leases / health ----------------------------------------------------------
    def lease(self):
        with self._cv:
            if self._closed:
                raise HostHold('host_closed')
            self._leases += 1
            return Lease(self)

    def _release_lease(self):
        with self._cv:
            self._leases = max(0, self._leases - 1)
            if self._leases == 0:
                self._idle_since = self.clock()

    def loaded(self):
        return any(i.loaded() for i in self.instances)

    def capacity(self):
        """Scheduler view: pool size, loaded/busy instances, VRAM cap and the queue in dispatch order."""
        with self._cv:
            queued = sorted(self._queue)
            return {'instances_max': self.max_instances, 'limit': self._limit,
                    'instances_loaded': sum(i.loaded() for i in self.instances),
                    'instances_busy': len(self._running), 'vram_cap': self._vram_cap, 'leases': self._leases,
                    'queued': [{'job_id': j.job_id, 'deadline': j.deadline, 'priority': j.priority}
                               for _key, j in queued],
                    'running': {i: j.job_id for i, j in self._running.items()},
                    'required_mib_per_instance': self.required_per_instance(),
                    'per_instance_basis': 'measured_peak' if self.measured_per_instance() else 'configured_unmeasured',
                    'throughput': self.throughput()}

    def measured_per_instance(self):
        """Largest per-instance WDDM reading taken DURING a take (peak under load), else None. Idle footprints never
        count (review P2-2)."""
        peaks = [i.peak_mib for i in self.instances if i.peak_mib]
        return float(max(peaks)) if peaks else None

    def required_per_instance(self):
        """max(configured actor_required_gpu_mib, measured peak): a measurement can only tighten the headroom check."""
        measured = self.measured_per_instance()
        return max(self.required_mib, int(math.ceil(measured))) if measured else self.required_mib

    def throughput(self):
        """Measured per-instance and pool totals: takes, audio seconds, busy seconds, busy RTF (busy/audio)."""
        rows = []
        for i in self.instances:
            s = i.stats
            rows.append({'instance': i.index, 'takes': s['takes'], 'audio_seconds': s['audio_seconds'],
                         'busy_seconds': s['busy_seconds'],
                         'busy_rtf': round(s['busy_seconds'] / s['audio_seconds'], 4) if s['audio_seconds'] else None})
        audio = sum(r['audio_seconds'] for r in rows)
        busy = sum(r['busy_seconds'] for r in rows)
        return {'instances': rows, 'audio_seconds': round(audio, 6), 'busy_seconds': round(busy, 6),
                'busy_rtf': round(busy / audio, 4) if audio else None}

    def pool_capacity(self, vram_budget_mib):
        """pool_capacity() with this host's measured per-instance VRAM (else the configured actor_required_gpu_mib)."""
        measured = self.measured_per_instance()
        return pool_capacity(vram_budget_mib, per_instance_mib=self.required_per_instance(), measured=bool(measured),
                             max_instances=self.max_instances, limit=self._limit)

    def note_refusal(self, reason=''):
        """FIX 6: any resource refusal (VRAM headroom before/after a load, Sem/lease hold, child yield) drops the live
        limit to 1 at once; admit_more() refuses a 2nd instance for refusal_hold_seconds."""
        with self._cv:
            self._refused_at = self.clock()
            self._limit = 1
            self._cv.notify_all()
        self.log('refusal', reason=str(reason), limit=1)

    def recently_refused(self):
        return self._refused_at is not None and self.clock() - self._refused_at < self.refusal_hold_seconds

    def admit_more(self, budget_mib, other_residents_mib=None, cap=2):
        """FIX 6 admission from LIVE telemetry only: the measured in-take peak of a running instance (no estimate, no
        probe load, no cache). Returns the instance count k (1..cap) and sets the live limit to it.

        k = 1 + the extra instances for which (budget_mib - other measured movie residents) still holds one more
        measured peak: total = floor((budget_mib - other_residents_mib) / peak), deducting the residents once. The lease
        budget is already the admitted movie allocation above the 3072 MiB free floor (not deducted again). The
        existing before-load headroom check in ensure_loaded (fresh device free - required >= reserve, required =
        max(configured, measured peak)) still runs right before the extra instance loads, and a refusal there drops k
        back to 1 (note_refusal)."""
        peak = self.measured_per_instance()
        k = 1
        # other_residents_mib None = UNKNOWN (another live movie GPU resident without a usable measurement): stay at 1.
        if peak and budget_mib and other_residents_mib is not None and not self.recently_refused():
            k = int(max(0.0, float(budget_mib) - float(other_residents_mib)) // peak)
            k = max(1, min(int(cap), self.max_instances, k))
        with self._cv:
            self._limit = k
            self._cv.notify_all()
        self.log('admitted', instances=k, peak_mib=peak, budget_mib=budget_mib, other_residents_mib=other_residents_mib,
                 recently_refused=self.recently_refused())
        return k

    def set_limit(self, instances):
        """Concurrency knob for the governor: run at most `instances` (1..instances_max) engine processes. Lowering it
        never interrupts a running take; surplus idle instances are released by the watcher."""
        if not isinstance(instances, int) or isinstance(instances, bool) or not 1 <= instances <= self.max_instances:
            raise ValueError('limit must be 1..%d' % self.max_instances)
        with self._cv:
            self._limit = instances
            self._cv.notify_all()
        self.log('limit', instances=instances)
        return instances

    def health(self):
        return {'status': 'loaded' if self.loaded() else 'idle', 'engine': approved.ENGINE,
                'expected_model_sha256': self.expected_model_sha256, 'manifest': self.manifest,
                'model_sha256': next((i.health()['model_sha256'] for i in self.instances if i.loaded()), None),
                'capacity_seconds': self.capacity_seconds, 'instances_max': self.max_instances,
                'reserve_mib': self.reserve_mib, 'required_mib': self.required_mib,
                'instances': [i.health() for i in self.instances], 'leases': self._leases,
                'stats': {k: (list(v) if isinstance(v, collections.deque) else v) for k, v in self.stats.items()},
                'shared_sem_worker_used': False}

    # -- jobs ---------------------------------------------------------------------
    def submit(self, request, *, cancelled=lambda: False, token=None, job_id=None, deadline=None, priority=0):
        job = PoolJob(request, job_id=job_id or 'job-%d' % next(PoolJob._seq), deadline=deadline, priority=priority,
                      token=token, cancelled=cancelled)
        with self._cv:
            if self._closed:
                raise HostHold('host_closed')
            reason = self.sem_activity()
            if reason:
                raise HostHold('sem_active', reason)
            heapq.heappush(self._queue, (job.key, job))
            self._jobs[job.seq] = job
            self.stats['jobs'] += 1
            self._cv.notify_all()
        return job

    def wait(self, job):
        while not job.done.wait(.01):
            if job.state == 'queued' and job.owner_cancelled():
                self._cancel_job(job, 'owner_cancelled')
        if job.error is not None:
            raise job.error
        return {**job.response, 'scheduling': job.scheduling()}

    def synthesize(self, request, *, cancelled=lambda: False, token=None, job_id=None, deadline=None, priority=0):
        """Blocking: submit one request and wait for its take (raises HostHold, EngineError or InterruptedError)."""
        return self.wait(self.submit(request, cancelled=cancelled, token=token, job_id=job_id, deadline=deadline,
                                     priority=priority))

    def _finish_job(self, job, *, response=None, error=None, state):
        job.response, job.error, job.state = response, error, state
        job.finished_at = time.time()
        with self._cv:
            self._jobs.pop(job.seq, None)
        job.done.set()

    def _cancel_job(self, job, reason):
        """Queued: removed now. Running: marker + ack, else kill (<= 0.2 s). Returns stop seconds."""
        with self._cv:
            if job.done.is_set():
                return None
            job.cancel_reason = reason
            queued = any(j is job for _k, j in self._queue)
            if queued:
                self._queue = [(k, j) for k, j in self._queue if j is not job]
                heapq.heapify(self._queue)
        if queued:
            self._finish_job(job, error=ActorCancelled('Movie Vox job cancelled while queued: %s' % reason),
                             state='cancelled')
            return 0.0
        instance = self.instances[job.instance] if job.instance is not None else None
        if instance is None:
            return None
        return instance.cancel(job, reason)

    def cancel_marker(self, job_id):
        """Path of the cancel marker of the RUNNING line `job_id` (None if queued/unknown). Creating that file from any
        process cancels the line: the child acknowledges or exits within hard_cancel_ms (review P1-2)."""
        with self._cv:
            for job in self._jobs.values():
                if job.job_id == str(job_id) and job.instance is not None:
                    inflight = self.instances[job.instance]._inflight
                    if inflight is not None and inflight.get('token') is job:
                        return str(inflight['marker'])
        return None

    def cancel(self, token=None, job_id=None, reason='cancelled', wait=False):
        """Cancel the jobs of `token` (a renderer) and/or the job `job_id`.

        Non-blocking by default (review P1-2): queued jobs are removed and every running job's marker is written in the
        CALLER's thread right here; the child then stops by itself (ack, or its hard-cancel watchdog). The host's
        backup kill runs on a background thread. wait=True blocks for the backup and returns the slowest stop
        seconds (tests)."""
        if token is None and job_id is None:
            return None
        with self._cv:
            jobs = [j for j in self._jobs.values()
                    if (job_id is None or j.job_id == str(job_id)) and (token is None or j.token is token)]
        return self._cancel_jobs(jobs, reason, wait=wait)

    def _cancel_jobs(self, jobs, reason, wait=False):
        if not jobs:
            return None
        for job in jobs:                                  # fast path first, for all of them
            job.cancel_reason = job.cancel_reason or reason
            instance = self.instances[job.instance] if job.instance is not None else None
            if instance is not None:
                instance.mark_cancel(None, reason, token=job)
        stops = [None] * len(jobs)

        def run(k, job):
            stops[k] = self._cancel_job(job, reason)
        threads = [threading.Thread(target=run, args=(k, j), daemon=True, name='vox-movie-cancel-backup')
                   for k, j in enumerate(jobs)]
        for t in threads:
            t.start()
        if not wait:
            return 0.0
        for t in threads:
            t.join()
        values = [s for s in stops if s is not None]
        return max(values) if values else None

    def cancel_all(self, reason='cancel_all'):
        """Scheduler-wide cancel (not Sem): every queued job removed, every running job marker-cancelled."""
        with self._cv:
            jobs = list(self._jobs.values())
        return self._cancel_jobs(jobs, reason, wait=True)

    def preempt(self, reason):
        """Sem activity: fail every queued job and kill every instance at once. Returns seconds."""
        started = self.clock()
        with self._cv:
            queued = [j for _k, j in self._queue]
            self._queue = []
            for job in queued:
                job.cancel_reason = 'sem_activity:' + reason
        for job in queued:
            self._finish_job(job, error=HostHold('sem_active', reason), state='failed')
        targets = [i for i in self.instances if i._child is not None or i._inflight is not None]
        threads = [threading.Thread(target=i.preempt, args=(reason,), daemon=True) for i in targets]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        seconds = self.clock() - started
        if targets or queued:
            self.stats['preemptions'] += 1
            self.stats['preempt_seconds'].append(round(seconds, 4))
            self.log('preempted', reason=reason, instances=[i.index for i in targets], queued_failed=len(queued),
                     seconds=round(seconds, 4))
        with self._cv:
            self._vram_cap = self.max_instances
            self._cv.notify_all()
        return seconds

    # -- dispatch -------------------------------------------------------------------
    def _free_instance(self):
        allowed = min(self._vram_cap, self._limit)
        free = [i for i in self.instances if i.index not in self._running and i.index < self._limit]
        loaded = [i for i in free if i.loaded()]
        if loaded:
            return loaded[0]
        if any((i.index in self._cold or i._loading) and not i.loaded() for i in self.instances):
            return None                                             # cold loads are sequential (VRAM read stays true)
        if self._running:
            cold = [i for i in free if i.index < allowed]           # grow only while headroom allowed it
        else:
            cold = free[:1]                                         # nothing loaded: instance 0 first
        return cold[0] if cold else None

    def _expire_queued(self):
        """Queued lines whose playback deadline (+ grace) has passed are never started: DeadlineExpired."""
        now = self.wall_time()
        late = [j for _k, j in self._queue
                if j.deadline is not None and now > j.deadline + self.deadline_grace_seconds]
        if not late:
            return []
        self._queue = [(k, j) for k, j in self._queue if j not in late]
        heapq.heapify(self._queue)
        for job in late:
            job.cancel_reason = 'deadline_missed'
        self.stats['deadline_expired'] += len(late)
        return [(job, now) for job in late]

    def _dispatch(self):
        while not self._closed:
            expired = []
            with self._cv:
                instance = None
                if self._queue:
                    expired = self._expire_queued()
                if self._queue:
                    instance = self._free_instance()
            for job, now in expired:
                self.log('deadline_expired', job_id=job.job_id, deadline=job.deadline, late_seconds=round(now - job.deadline, 4))
                self._finish_job(job, error=DeadlineExpired(job.job_id, job.deadline, now), state='expired')
            with self._cv:
                if instance is not None and not self._queue:
                    instance = None
                if instance is None:
                    self._cv.wait(.05)
                    continue
                _key, job = heapq.heappop(self._queue)
                job.instance, job.state, job.started_at = instance.index, 'running', time.time()
                self._running[instance.index] = job
                if not instance.loaded():
                    self._cold.add(instance.index)
            threading.Thread(target=self._run, args=(instance, job), name='vox-movie-job-%d' % instance.index,
                             daemon=True).start()

    def _run(self, instance, job):
        requeue = False
        try:
            response = instance.synthesize(job.request, cancelled=job.should_stop, token=job,
                                           required_mib=self.required_per_instance())
            self._finish_job(job, response=response, state='done')
        except HostHold as error:
            if error.fault_kind == 'resource':
                self.note_refusal(error.reason)
            others = any(i.loaded() for i in self.instances if i is not instance)
            if error.reason == 'vram_headroom' and others and job.cancel_reason is None:
                requeue = True
                with self._cv:
                    self._vram_cap = max(1, min(self._vram_cap, instance.index))
                    job.instance, job.state, job.started_at = None, 'queued', None
                    heapq.heappush(self._queue, (job.key, job))
                    self.stats['requeued'] += 1
                self.log('pool_capped', instance=instance.index, vram_cap=self._vram_cap, job_id=job.job_id)
            else:
                self._finish_job(job, error=error, state='failed')
        except BaseException as error:   # noqa: BLE001 - delivered to the waiting caller
            self._finish_job(job, error=error, state='cancelled' if isinstance(error, InterruptedError) else 'failed')
        finally:
            with self._cv:
                self._cold.discard(instance.index)
                if self._running.get(instance.index) is job:
                    del self._running[instance.index]
                if requeue:
                    self._jobs[job.seq] = job
                self._cv.notify_all()

    def _watch(self):
        while not self._closed:
            time.sleep(self.poll_seconds)
            try:
                active = self._running or self._queue or any(i._child is not None for i in self.instances)
                if not active:
                    continue
                reason = self.sem_activity()
                if reason:
                    self.preempt(reason)
                    continue
                for instance in self.instances:
                    instance.tick(self.vram_interval_seconds)
                    with self._cv:
                        surplus = (instance.index >= self._limit and instance.index not in self._running
                                   and instance._child is not None)
                    if surplus:
                        instance.release_idle('over_limit')
                with self._cv:
                    idle = (self._leases == 0 and not self._running and not self._queue
                            and self.clock() - self._idle_since > self.idle_grace_seconds)
                if idle:
                    for instance in self.instances:
                        if instance._child is not None:
                            instance.release_idle('idle_grace')
                    with self._cv:
                        self._vram_cap = self.max_instances
            except Exception as error:   # noqa: BLE001 - the watcher must keep running
                self.log('watcher_error', error=str(error)[:300])

    def drain(self, reason='lease_release', timeout=2.0):
        """Stop every engine of this host and wait until their processes have EXITED (Codex note 1: a lease is released
        only after its engine children stopped and drained). Returns True when none is alive."""
        self.preempt('drain:%s' % reason)
        deadline = self.clock() + float(timeout)
        processes = [p for i in self.instances for p in i.processes()]   # kept Popens, not just the current child
        while True:
            alive = [p for p in processes if p.poll() is None]
            if not alive:
                self.log('drained', reason=reason, processes=[p.pid for p in processes])
                return True
            if self.clock() >= deadline:
                break
            for process in alive:
                try:
                    process.wait(timeout=.01)
                except subprocess.TimeoutExpired:
                    pass
        self.log('drain_timeout', reason=reason, surviving_pids=[p.pid for p in alive])
        return False

    def close(self):
        """Closed FIRST (under the lock: no submit or cold load is accepted afterwards, review R3), then preempt, then
        the host's own kill-on-close Job is closed, which ends any child that is still alive."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self.preempt('host_closed')
        with self._owner_lock:
            job, self._own_job = self._own_job, None
        if job is not None:
            job.close()


# -- process-level shared host ----------------------------------------------------------
_SHARED = {'key': None, 'host': None}
_SHARED_LOCK = threading.Lock()


def shared_host(options):
    """One movie-Vox host (pool) per relay process, rebuilt only when its options change."""
    key = json.dumps({k: v for k, v in options.items() if not callable(v)}, sort_keys=True, default=str)
    with _SHARED_LOCK:
        host = _SHARED['host']
        if host is not None and _SHARED['key'] == key and not host._closed:
            return host
        if host is not None:
            host.close()
        _SHARED['host'], _SHARED['key'] = VoxEngineHost(**options), key
        return _SHARED['host']


def close_shared():
    with _SHARED_LOCK:
        host, _SHARED['host'], _SHARED['key'] = _SHARED['host'], None, None
    if host is not None:
        host.close()


atexit.register(close_shared)
