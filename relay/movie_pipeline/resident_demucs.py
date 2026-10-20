"""Parent side of the resident one-pass Demucs worker (relay/first_pass_resident.py). relay.movie_pipeline, W10/V2-GPU.

Registered by MoviePipeline as relay.first_pass_parent.RESIDENT. first_pass_parent.run (movie-lease admission) writes
the usual first-pass-request.json, calls produce(request, owner, timeout=...), then validates receipt.json exactly as
for the per-window child. One model load serves every window; requests are serialized.

Process ownership:
  * The worker is started with lease.spawn() of a movie GPU lease view (relay.app_runtime, lease v2): created
    suspended, put in the lease's kill-on-close Job, resumed. The lease watcher ends it on preemption or release, and
    when the holder dies. Nothing else is ever killed: _stop() ends only this worker, through its own Popen handle.
  * The view is owned from the moment the hub returns it: every later failure (check, admission, spawn, init,
    readiness) releases it through _stop().
  * Fix 8: the per-job cancellation owner passed to produce() is used from the very start. The worker is registered
    with that owner IMMEDIATELY after spawn, BEFORE the readiness wait, so a cancelled job stops a model load promptly.
    The same owner is unregistered in cleanup on every exit path. A cancelled load leaves no worker behind.
  * _stop() kills and waits. The resident is forgotten and the view released ONLY after the worker has exited. A
    worker that cannot be confirmed exited stays owned (view + active-unknown resident) as a zombie; it is reaped by
    the next produce()/idle_check()/close(), and no new worker is started beside it (resource wait). The hub's lease
    release (synchronous Job drain) is the final backstop.
  * A stdin/protocol failure stops the broken worker; it is never merely unregistered. The stdout reader is bounded
    (MAX_LINE_CHARS pieces; a non-object JSON line is noise; noise beyond MAX_NOISE_LINES + 1 in a row is dropped).
  * The lease view is released when the worker stops, including the idle unload (idle_check).

Faults (relay.movie_faults metadata on every raised error):
  resource  only a real hold: lease unavailable/ended/preempted, guard yield (exit 75 / {"yield"}: lease cancel, Sem,
            RAM bound), a not-yet-exited previous worker
  fatal     invalid interpreter/script/path/permission/config, spawn failure, init refusal, protocol violation,
            unexpected exit, the load/window deadline (exit 76; a deadline is not resource evidence), guard failure (77)
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
READY_TIMEOUT_S = 180.0
IDLE_UNLOAD_S = 150.0
STOP_WAIT_S = 10.0
EOF_WAIT_S = 15.0
MAX_NOISE_LINES = 200
MAX_LINE_CHARS = 1 << 20                 # one protocol line is far smaller; a longer line is read as noise pieces
EXIT_RESOURCE, EXIT_DEADLINE, EXIT_GUARD_FAILURE = 75, 76, 77
_LEASE_ERRORS = ('ViewEnded', 'LeaseError', 'LeasePreempted', 'LeaseExpired', 'LeaseUnavailable')


def _faulted(error, kind, code, detail=None):
    error.fault_kind, error.fault_code = kind, str(code)[:120]
    error.detail = str(error if detail is None else detail)[:4000]
    return error


class ResidentDemucs:
    def __init__(self, *, python, relay_dir, model_dir, model_pins, threads, device, lease_source, log_path,
                 resource_yield, cancelled, emit=None, idle_seconds=IDLE_UNLOAD_S, vram_reader=None):
        self.python, self.script = str(python), str(Path(relay_dir) / 'first_pass_resident.py')
        self.model = {'model_dir': str(model_dir), 'model_pins': dict(model_pins), 'threads': int(threads),
                      'device': device, 'load_timeout_seconds': READY_TIMEOUT_S}
        self.lease_source = lease_source
        self.log_path = Path(log_path)
        self.ResourceYield, self.Cancelled = resource_yield, cancelled
        self.emit = emit or (lambda kind, **fields: None)
        self.idle_seconds = float(idle_seconds)
        self.child = self.view = self.lines = self._log = None
        self.ready_info = None
        self.worker_pid = None              # real interpreter identity from this owned child's private ready pipe
        self._last_window_memory = None       # only the current child's latest successful, cache-released window
        self.lock = threading.RLock()
        self.starts = self.served = 0
        self.last_used = time.monotonic()
        self.vram_reader = vram_reader       # W3 vox_engine_host.wddm_process_vram (pid -> reading) or None
        self.vram_method = None
        self.zombies = []                    # [(child, view, log)] not confirmed exited: still owned

    @property
    def last_window_memory(self):
        """An observational live-child sample. Busy, exited or preempted is unknown, never stale credit."""
        if not self.lock.acquire(blocking=False):
            return None
        try:
            child = self.child
            if (child is None or self.view is None or self.zombies or child.poll() is not None
                    or self._lease_ended() or self._last_window_memory is None
                    or self.worker_pid is None or self._last_window_memory.get('pid') != self.worker_pid):
                return None
            return dict(self._last_window_memory)
        finally:
            self.lock.release()

    @last_window_memory.setter
    def last_window_memory(self, value):
        with self.lock:
            self._last_window_memory = None if value is None else dict(value)

    # -- fault constructors -------------------------------------------------------------------------------------
    def _wait_error(self, code, detail):
        return _faulted(self.ResourceYield(detail), 'resource', code, detail)

    @staticmethod
    def _fatal(code, detail):
        from ..movie_faults import MovieConfigurationError
        return MovieConfigurationError(code, detail)

    def _cancelled(self, detail='Operation cancelled'):
        return _faulted(self.Cancelled(detail), 'cancelled', 'cancelled', detail)

    def _ensure_fault(self, error, code):
        """The same exception object with explicit fault metadata: kept when present; owner cancellation ->
        cancelled; a raw ResourceYield -> resource; anything else (I/O, permission, bug) -> fatal `code`."""
        if getattr(error, 'fault_kind', None) and getattr(error, 'fault_code', None):
            return error
        if isinstance(error, self.Cancelled):
            return _faulted(error, 'cancelled', 'cancelled')
        if isinstance(error, self.ResourceYield):
            return _faulted(error, 'resource', 'demucs_resource_yield')
        return _faulted(error, 'fatal', code, '%s: %s' % (type(error).__name__, error))

    # -- worker process -----------------------------------------------------------------------------------------
    def _env(self):
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
        if self.model['device'] == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            env.pop('CUDA_VISIBLE_DEVICES', None)   # a server-side blank must never hide the GPU from this worker
        return env

    def _start(self, owner):
        self.worker_pid = None
        owner.check()
        self._reap()
        if self.zombies:
            raise self._wait_error('demucs_previous_worker_alive',
                                   'a previous resident Demucs worker has not exited yet; no second worker beside it')
        if not Path(self.python).is_file() or not Path(self.script).is_file():
            raise self._fatal('demucs_runtime_missing', 'python %s or worker %s is missing' % (self.python, self.script))
        try:
            view = self.lease_source('kevflix movie Demucs')
        except Exception as error:  # noqa: BLE001
            from ..movie_faults import fault_record
            if fault_record(error) is not None:
                raise                            # the holder decided: a real hold, or a permanent (fatal) lease fault
            if type(error).__name__ in _LEASE_ERRORS:
                raise self._wait_error('demucs_lease_unavailable',
                                       'movie GPU lease unavailable for Demucs: %s' % error) from error
            raise self._fatal('demucs_lease_failed', '%s: %s' % (type(error).__name__, error)) from error
        self.view = view                         # owned from here: every failure below releases it through _stop()
        try:
            try:
                view.check()
            except Exception as error:  # noqa: BLE001
                if self._lease_ended() or type(error).__name__ in _LEASE_ERRORS:     # ended between grant and use
                    raise self._wait_error('demucs_lease_ended', 'movie GPU lease ended before Demucs started: %s'
                                           % error) from error
                raise self._fatal('demucs_lease_failed', '%s: %s' % (type(error).__name__, error)) from error
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log = open(self.log_path, 'ab')
            except OSError as error:            # path/permission: retrying unchanged cannot repair it
                raise self._fatal('demucs_log_unavailable', 'resident Demucs log %s: %s'
                                  % (self.log_path, error)) from error
            flags = {'creationflags': CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS} if os.name == 'nt' else {}
            try:
                child = view.spawn([self.python, '-I', '-B', self.script], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=self._log, cwd=str(Path(self.script).parent),
                                   env=self._env(), text=True, encoding='utf-8', bufsize=1, **flags)
            except Exception as error:  # noqa: BLE001
                if self._lease_ended() or type(error).__name__ in _LEASE_ERRORS:
                    raise self._wait_error('demucs_lease_ended', 'movie GPU lease ended before the Demucs worker '
                                           'started: %s' % error) from error
                raise self._fatal('demucs_spawn_failed', '%s: %s' % (type(error).__name__, error)) from error
            self.child, self.starts = child, self.starts + 1
            if hasattr(view, 'report_resident'):
                view.report_resident(child.pid, None)     # live GPU resident, not yet measured: active-unknown
            owner.register(child)        # fix 8: BEFORE the readiness wait (raises Cancelled + kills it if cancelled)
            lines = queue.Queue()

            def reader(stream=child.stdout):
                # Bounded: lines are read in MAX_LINE_CHARS pieces, a non-object JSON line is noise, and after
                # MAX_NOISE_LINES + 1 consecutive noise lines further noise is dropped (the pipe keeps draining; the
                # waiter has already failed the worker as a protocol violation), so the queue cannot grow unbounded.
                noise = 0
                try:
                    for raw in iter(lambda: stream.readline(MAX_LINE_CHARS), ''):
                        try:
                            message = json.loads(raw)
                        except ValueError:
                            message = None
                        if isinstance(message, dict) and 'noise' not in message:
                            noise = 0
                            lines.put(message)
                            continue
                        noise += 1
                        if noise <= MAX_NOISE_LINES + 1:
                            lines.put({'noise': raw[:200]})
                except (OSError, ValueError):
                    pass
                lines.put(None)
            threading.Thread(target=reader, name='resident-demucs-reader', daemon=True).start()
            self.lines = lines
            self._write({'init': {**self.model, 'lease': {
                'cancel_event_name': getattr(view, 'cancel_event_name', None),
                'sem_mode': getattr(view, 'sem_mode', None)}}}, owner, phase='load')
            reply = self._wait(owner, READY_TIMEOUT_S, phase='load')
            if not reply.get('ready'):
                raise self._fatal('demucs_load_refused', 'resident Demucs did not load: %s' % reply)
            worker_pid = reply.get('pid')
            if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
                raise self._fatal('demucs_protocol_violation', 'resident Demucs ready reply has no positive worker PID')
            # Windows venv python.exe can be a redirector. The private readiness pipe pins the
            # interpreter for this spawn; the Popen launcher remains the ownership/cleanup handle.
            self.worker_pid = worker_pid
            self.ready_info = reply
            self._measure()
            self.emit('resident_demucs_ready', pid=child.pid, load_seconds=reply.get('load_seconds'),
                      launcher_pid=child.pid, worker_pid=self.worker_pid,
                      device=self.model['device'], info=reply.get('info'), lease_id=getattr(view, 'lease_id', None))
        except BaseException as error:
            # ANY failure of the check/spawn/init/readiness phase (BrokenPipe, deadline, init error, cancellation after
            # register, a failing ready callback) ends this worker through its own handle, unregisters it from the
            # owner, closes its log and releases the lease view once the worker has exited. The ORIGINAL error is
            # re-raised, carrying fault metadata.
            self._stop(owner)
            if isinstance(error, Exception):
                self._ensure_fault(error, 'demucs_start_failed')
            raise

    def _write(self, message, owner, *, phase, abort_path=None):
        """One protocol line. A broken pipe stops the worker and raises the reason it ended."""
        try:
            self.child.stdin.write(json.dumps(message) + '\n')
            self.child.stdin.flush()
        except (OSError, ValueError) as error:
            raise self._ended_error(owner, phase, abort_path, 'stdin write failed: %s' % error) from error

    def _lease_ended(self):
        view = self.view
        return view is not None and (getattr(view, 'preempted', False) or view.cancel_token.is_set())

    def _ended_error(self, owner, phase, abort_path, detail):
        """The worker ended (or must be stopped): stop it, then the classified error for why it ended."""
        child = self.child
        code = None
        if child is not None:
            try:
                code = child.wait(EOF_WAIT_S)
            except subprocess.TimeoutExpired:
                code = None
        ended = self._lease_ended()
        self._stop(owner)
        if abort_path and Path(abort_path).is_file():
            try:
                from ..first_pass_parent import abort_error
                error = abort_error(json.loads(Path(abort_path).read_text(encoding='utf-8')))
            except (OSError, ValueError, ImportError):
                error = None
            if error is not None:
                return error
        if ended:
            return self._wait_error('demucs_lease_preempted', 'movie GPU lease ended during Demucs %s (exit %s)'
                                    % (phase, code))
        if code == EXIT_RESOURCE:
            return self._wait_error('demucs_guard_yield', 'resident Demucs guard yielded during %s' % phase)
        if code == EXIT_DEADLINE:
            return self._fatal('demucs_window_deadline', 'resident Demucs exceeded its %s deadline' % phase)
        if code == EXIT_GUARD_FAILURE:
            return self._fatal('lease_guard_failure', 'resident Demucs guard failed during %s' % phase)
        return self._fatal('demucs_worker_exited', 'resident Demucs exited %s during %s: %s' % (code, phase, detail))

    def _wait(self, owner, timeout, *, phase, abort_path=None):
        deadline = time.monotonic() + float(timeout)
        noise = 0
        while True:
            if owner.cancelled.is_set():
                self._stop(owner)
                raise self._cancelled()
            try:
                reply = self.lines.get(timeout=0.25)
            except queue.Empty:
                if self._lease_ended():
                    reason = getattr(getattr(self.view, 'event', None), 'reason', 'ended')
                    self._stop(owner)
                    raise self._wait_error('demucs_lease_preempted',
                                           'movie GPU lease preempted during Demucs %s: %s' % (phase, reason))
                if time.monotonic() > deadline:
                    # A deadline is not evidence of resource pressure: fatal, explicit.
                    self._stop(owner)
                    raise self._fatal('demucs_%s_deadline' % phase,
                                      'resident Demucs exceeded its %s bound (%.0f s)' % (phase, timeout))
                if self.child.poll() is None:
                    continue
                reply = None
            if reply is not None and 'noise' in reply:
                noise += 1
                if noise > MAX_NOISE_LINES:
                    self._stop(owner)
                    raise self._fatal('demucs_protocol_violation', 'resident Demucs wrote non-protocol output')
                continue
            if reply is None:
                raise self._ended_error(owner, phase, abort_path, 'end of output')
            if 'yield' in reply:
                self._stop(owner)
                raise self._wait_error('demucs_guard_yield', 'resident Demucs yielded: %s' % reply['yield'])
            if 'fatal' in reply:
                self._stop(owner)
                raise self._fatal('demucs_refused', 'resident Demucs refused: %s' % reply['fatal'])
            return reply

    def _stop(self, owner=None):
        """End ONLY this worker (own Popen handle), unregister it from the owner, and release its lease view once the
        worker has exited. An unconfirmed exit keeps the worker owned as a zombie (view + resident kept)."""
        child, view, log = self.child, self.view, self._log
        self.last_window_memory = None
        self.worker_pid = None
        self.child = self.view = self._log = None
        if child is not None and owner is not None:
            owner.unregister(child)
        exited = True
        if child is not None:
            if child.poll() is None:
                try:
                    child.kill()
                except OSError:
                    pass
                try:
                    child.wait(STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    pass
            exited = child.poll() is not None
        if not exited:
            self.zombies.append((child, view, log))
            self.emit('resident_demucs_unconfirmed', pid=child.pid)
            return False
        self._release(child, view, log)
        return True

    @staticmethod
    def _release(child, view, log):
        if child is not None and view is not None and hasattr(view, 'forget_resident'):
            view.forget_resident(child.pid)
        if log is not None:
            try:
                log.close()
            except OSError:
                pass
        if view is not None:
            try:
                view.release('resident_demucs_stopped')
            except Exception:  # noqa: BLE001 - release never raises by contract
                pass

    def _reap(self):
        for entry in list(self.zombies):
            child = entry[0]
            if child.poll() is None:
                try:
                    child.kill()
                    child.wait(1.0)
                except (OSError, subprocess.TimeoutExpired):
                    continue
            if child.poll() is not None:
                self.zombies.remove(entry)
                self._release(*entry)

    # -- first_pass_parent.RESIDENT protocol ---------------------------------------------------------------------
    def produce(self, request, owner, *, timeout):
        try:
            self._produce(request, owner, timeout=timeout)
        except Exception as error:  # noqa: BLE001 - every raised error carries fault metadata (same object)
            self._ensure_fault(error, 'demucs_window_failed')
            raise

    def _produce(self, request, owner, *, timeout):
        with self.lock:
            self.last_window_memory = None
            owner.check()
            if self.child is None or self.child.poll() is not None or self._lease_ended():
                self._stop(owner)
                self._start(owner)
            else:
                owner.register(self.child)
            try:
                path = Path(request['folder']) / 'first-pass-request.json'
                self._write({'request': str(path)}, owner, phase='window', abort_path=request.get('abort'))
                reply = self._wait(owner, timeout, phase='window', abort_path=request.get('abort'))
            finally:
                if self.child is not None:
                    owner.unregister(self.child)
                self.last_used = time.monotonic()
            if 'refused' in reply:
                return                            # receipt.json carries state 'refused'; the parent raises ValueError
            if 'done' not in reply:
                self._stop(owner)                 # a protocol violation: the broken worker is stopped
                raise self._fatal('demucs_protocol_violation', 'unexpected resident Demucs reply: %s' % reply)
            memory = reply.get('cuda_memory')
            if str(self.model['device']).startswith('cuda'):
                valid = (isinstance(memory, dict) and memory.get('version') == 'demucs-window-cuda-memory-v1'
                         and self.child is not None and self.worker_pid is not None and memory.get('pid') == self.worker_pid
                         and memory.get('device') == str(self.model['device'])
                         and memory.get('reset_peak_memory_stats') is True and memory.get('empty_cache_completed') is True)
                if valid:
                    peak, current = memory.get('peak_reserved_bytes'), memory.get('current_reserved_bytes')
                    valid = (isinstance(peak, int) and not isinstance(peak, bool) and peak >= 0
                             and isinstance(current, int) and not isinstance(current, bool) and 0 <= current <= peak
                             and memory.get('additional_peak_reserved_bytes') == peak - current)
                if not valid:
                    self._stop(owner)
                    raise self._fatal('demucs_memory_telemetry_invalid', 'Successful CUDA window has no valid post-cache-release measurement')
                self.last_window_memory = dict(memory)
                self.emit('resident_demucs_window_memory', **memory)
            self.served += 1
            self._measure()

    def _measure(self):
        """Report this worker's GPU memory to its lease view (LeaseView.report_resident -> other_residents_mib, read by
        Vox admission). First W3's WDDM per-process reader (vox_engine_host.wddm_process_vram, via vram_reader). If it
        returns nothing, nvidia-smi per-process used_memory. If neither gives a usable value, the pid is reported
        active-unknown (None): unknown is never zero."""
        child, view, worker_pid = self.child, self.view, self.worker_pid
        if child is None or view is None or not hasattr(view, 'report_resident'):
            return
        if worker_pid is None:
            view.report_resident(child.pid, None)
            return
        mib = None
        if self.vram_reader is not None:
            try:
                reading = self.vram_reader(worker_pid) or {}
                if reading.get('rows') and reading.get('process_dedicated_mib') is not None:
                    mib = int(round(float(reading['process_dedicated_mib'])))
                    self.vram_method = reading.get('method')
            except Exception:  # noqa: BLE001 - no reading: fall back, never assume zero
                mib = None
        if mib is None:
            try:
                out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10,
                                     **({'creationflags': CREATE_NO_WINDOW} if os.name == 'nt' else {})).stdout
            except (OSError, subprocess.SubprocessError):
                out = ''
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == worker_pid and parts[1].isdigit():
                    mib = int(parts[1])
                    self.vram_method = 'nvidia-smi --query-compute-apps used_memory'
        view.report_resident(child.pid, mib)              # None keeps the pid active-unknown
        if mib is not None:
            self.peak_mib = max(getattr(self, 'peak_mib', 0), mib)

    def unload(self, reason='unloaded'):
        """Stop the resident worker now (model out of RAM and VRAM, lease view released; non-blocking). Returns 'stopped', 'absent' or
        'busy' (a window is being produced or the lock is taken: ask again shortly)."""
        if not self.lock.acquire(blocking=False):
            return 'busy'
        try:
            self._reap()
            if self.child is None:
                return 'absent'
            self._close_locked(reason)
            return 'stopped'
        finally:
            self.lock.release()

    def idle_check(self):
        """Unload the model and release the lease view after idle_seconds without a request (non-blocking)."""
        if (self.child is None and not self.zombies) or time.monotonic() - self.last_used < self.idle_seconds:
            return False
        if not self.lock.acquire(blocking=False):
            return False
        try:
            self._reap()
            if self.child is not None and time.monotonic() - self.last_used >= self.idle_seconds:
                self._close_locked('idle')
                return True
            return False
        finally:
            self.lock.release()

    def _close_locked(self, why):
        child = self.child
        if child is not None and child.poll() is None:
            try:
                child.stdin.close()
                child.wait(15)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self._stop()
        self._reap()
        self.emit('resident_demucs_stopped', reason=why, unconfirmed=len(self.zombies))

    def describe(self):
        memory = self.last_window_memory
        return {'starts': self.starts, 'served': self.served, 'device': self.model['device'],
                'pid': self.child.pid if self.child is not None else None, 'unconfirmed': len(self.zombies),
                'worker_pid': self.worker_pid,
                'last_window_memory': memory,
                'ready': {k: v for k, v in (self.ready_info or {}).items() if k != 'ready'}}

    def close(self):
        with self.lock:
            self._close_locked('close')
