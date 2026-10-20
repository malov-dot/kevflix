"""AnalysisService: ONE warm GPU analysis worker for the app process (relay.movie_pipeline, V2-GPU). Source only.

    service = AnalysisService(config, lease_source, log_path=..., emit=None, vram_reader=None)
    pipeline.set_analysis_service(service)          # BOTH ActorPipeline instances share this one service
    tools = service.language(owner)                 # LanguageTools-compatible: speech / english / text_language
    analyzer = service.speakers(owner)              # SpeakerAnalyzer-compatible: analyze / embed / refine_utterances,
                                                    #   policy_metadata / device_report
    service.device_report(); service.idle_check(); service.close()

The models (Codex LanguageTools(config, device='cuda') and SpeakerAnalyzer(config, cinematic=False, device='cuda',
cache=AnalysisCache)) live ONLY in analysis_child.py, spawned through lease_source('kevflix movie analysis').spawn():
the same LeaseHub as the resident Demucs and Vox, inside the lease's kill-on-close Job. This module imports no model,
tensor or CUDA library (numpy only to write .npy exchange files for array arguments). The child holds the ONE
authoritative raw-inference cache (<cache>/analysis-cache: outside every disposable snapshot).

Ownership (process, view, files):
  * The view is taken BEFORE the spawn. After a successful Popen ONE cleanup boundary covers resident registration,
    the reader start, the init send and the event: any failure stops and tracks THAT process.
  * The child is an active-unknown resident (report_resident(pid, None)) from spawn, measured after load and at most
    every MEASURE_INTERVAL_S afterwards (W3's WDDM reader, else nvidia-smi). Unknown VRAM is reported as None.
  * Stopping kills through the child's own Popen handle and waits; pending requests fail at once with the stop
    reason. The resident is forgotten and the view released ONLY after the child has exited. A child that cannot be
    confirmed exited stays owned (view + resident) in `_unconfirmed`; NO new worker is spawned while any earlier
    generation is unconfirmed (resource wait). A pending lease drain surfaces from the hub as a resource wait.
  * A failed protocol send stops the child and fails its pending requests promptly (no wait for a stdout EOF).
  * Deadlines are service-owned: a per-generation watchdog thread enforces the model-load bound and every request's
    queue/run bound (and reacts to the lease ending) even after every caller has detached; the child's own LeaseGuard
    deadline (exit 76) backs it up. Fatal stops (startup, load, protocol, send, deadline) are sticky until close():
    no relaunch loop. The warm models are kept across windows otherwise.
  * Per-window owners are never registered with the shared child. A cancelled window marks its request abandoned
    (under the generation lock), sends {"op":"cancel"} and raises; the reader then discards the late reply and its
    exchange files. A reply that wins the race against cancellation is discarded too (never returned).
  * Exchange files live in this service's own folder <state>/movie-pipeline/analysis-io/svc-<pid>-<random>/. Audio
    inputs are staged there as immutable per-request copies (the caller's chunk WAV may change after cancellation);
    arrays as .npy, large args/results as JSON. A request's files are removed on its final reply; files of failed or
    abandoned requests are removed only once the child is confirmed exited. close() removes the folder only when
    every child of this service has exited. No other folder is ever touched (no age-based sweeping).

Faults (relay.movie_faults records; nothing is silently retried):
  resource   a real hold only: the hub/holder's MovieResourceWait (lock, VRAM, Sem, pending drain), the lease ended
             or was preempted, guard yield (exit 75: lease cancel, Sem present, RAM bound), CUDA OOM, the measured
             analysis peak does not fit the view budget, an earlier worker not yet confirmed exited
  fatal      configuration/path/model/CUDA/runtime: the holder's MovieConfigurationError, missing interpreter/script/
             config, spawn failure, init failure, protocol/send failure, unexpected exit, load/request deadline
  recoverable content uncertainty raised by the models (MovieContentFallback records)
  cancelled  the window's owner was cancelled, or the service is closing (also a relay.process.Cancelled)
A fatal init failure is sticky until close(): every later call fails at once with the same record. Three consecutive
unexpected child deaths are sticky too.
"""
from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from ..movie_faults import (MovieCancelled, MovieConfigurationError, MovieFault, MovieResourceWait, fault_record,
                            from_record as _from_record)
from ..process import Cancelled as _ProcessCancelled
from . import analysis_child as _proto

VERSION = 'kevflix-analysis-service-v2'
HOLDER_NAME = 'kevflix movie analysis'
CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
READY_TIMEOUT_S = 600.0          # a first start hashes model files once and loads three models
QUEUE_TIMEOUT_S = 1800.0         # waiting behind other (possibly abandoned) requests before this one started
STOP_WAIT_S = 15.0
POLL_S = 0.2
IDLE_UNLOAD_S = 150.0
MEASURE_INTERVAL_S = 10.0
MAX_CONSECUTIVE_DEATHS = 3
WATCH_S = 1.0                    # service-owned watchdog cadence (deadlines hold even when every caller detached)
CHILD_DEADLINE_MARGIN_S = 60.0   # the child's own guard deadline backs up the service watchdog
# The first request of an op after the worker started also loads that op's model. On a cold disk (the first start after a reboot or an
# install) the language-id, audio-event and speaker models took longer than the former 120 s and the whole pipeline stopped with
# 'Dubbing error'; they now get the same room as speech recognition.
OP_TIMEOUTS = {'text_language': 300.0, 'speech': 900.0, 'english': 300.0, 'english_batch': 600.0, 'analyze': 900.0,
               'embed': 300.0, 'refine_utterances': 900.0, 'device_report': 60.0, 'generic_vad': 300.0,
               'generic_synthesize': 300.0, 'language_id': 600.0, 'audio_events': 600.0, 'warmup': 900.0}
SLOW_OP_LOG_S = 3.0               # a request that took at least this long is logged with its queued and running seconds
PREWARM_RETRY_S = 15.0            # after a warm-up stopped (resource hold, fault) the next claim may try again after this
# A request that ran past its limit costs that worker only: the next request starts a fresh one. MAX_CONSECUTIVE_DEATHS in a row
# (no request answered in between) still stop the service until close(), like every other fatal fault.
RECOVERABLE_STOPS = frozenset({'analysis_timeout'})
ENGLISH_BATCH_RPC_ROWS = 64      # rows per english_batch RPC (a Qwen3 batch is 16 rows; each RPC bounds its own deadline)
EXIT_DEADLINE, EXIT_GUARD_FAILURE = 76, 77      # relay.first_pass_resident.LeaseGuard exit codes (fatal)
_ENDED_ERRORS = ('ViewEnded', 'LeasePreempted', 'LeaseExpired')


class AnalysisCancelled(MovieCancelled, _ProcessCancelled):
    """Cancelled fault that existing `except Cancelled` handlers (relay.process) also recognise."""


def from_record(record):
    """relay.movie_faults.from_record, with cancellation also a relay.process.Cancelled."""
    if isinstance(record, dict) and record.get('kind') == 'cancelled' and isinstance(record.get('code'), str) \
            and record.get('code') and isinstance(record.get('detail'), str):
        return AnalysisCancelled(record['code'][:120], record['detail'][:4000])
    return _from_record(record)


def _rec(kind, code, detail):
    return {'kind': kind, 'code': str(code)[:120], 'detail': str(detail)[:4000]}


def _ended(view):
    return view is not None and (bool(getattr(view, 'preempted', False)) or view.cancel_token.is_set())


class _NoOwner:
    def __init__(self):
        self.cancelled = threading.Event()


class _FileArg:
    __slots__ = ('path',)

    def __init__(self, path):
        self.path = path


class _ArrayArg:
    __slots__ = ('array',)

    def __init__(self, array):
        self.array = array


class _Slot:
    def __init__(self, temps, timeout=None):
        self.event = threading.Event()
        self.enqueued = time.monotonic()
        self.timeout = timeout
        self.started = None
        self.reply = None
        self.temps = list(temps)
        self.abandoned = False


class _Gen:
    """One child process generation: its Popen, view, log handle and pending requests (guarded by self.lock)."""

    def __init__(self, popen, view, log):
        self.popen, self.view, self.log = popen, view, log
        self.pid = popen.pid
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.pending = {}
        self.ready = _Slot(())
        self.ready_info = None
        self.init_failure = None
        self.death = None                # fault record given to every pending request when the child ends
        self.dead = threading.Event()    # set only after death, fatal latch and retirement were published (locked)
        self.eof = False                 # the reader saw the child's stdout end; its death is being published
        self.stopping = False
        self.violation = None
        self.released = False
        self.orphan_temps = []           # exchange files of failed requests: removed only once the child has exited
        self.spawned = time.monotonic()
        self.last_measure = 0.0


class AnalysisService:
    def __init__(self, config, lease_source, *, log_path, emit=None, vram_reader=None):
        self.config, self.lease_source = config, lease_source
        self.log_path = Path(log_path)
        self.emit = emit or (lambda kind, **fields: None)
        self.vram_reader = vram_reader
        self.python = str(config['python'] or sys.executable)
        self.script = Path(_proto.__file__).resolve()
        config_file = getattr(config, 'file', None)
        if not Path(self.python).is_file():
            raise MovieConfigurationError('analysis_python_missing', 'configured python is not a file: %s' % self.python)
        if not self.script.is_file():
            raise MovieConfigurationError('analysis_child_missing', 'analysis worker source missing: %s' % self.script)
        if not config_file or not Path(config_file).is_file():
            raise MovieConfigurationError('analysis_config_file_missing',
                                          'the analysis worker needs the app config file (%s)' % config_file)
        self.config_file = str(Path(config_file).resolve())
        self.cache_root = Path(config.cache) / 'analysis-cache'
        io_root = Path(config.state) / 'movie-pipeline' / 'analysis-io'
        self.io_dir = io_root / ('svc-%d-%s' % (os.getpid(), uuid.uuid4().hex))
        try:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            io_root.mkdir(parents=True, exist_ok=True)
            self.io_dir.mkdir()                      # this service's own folder; no other folder is ever removed
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise MovieConfigurationError('analysis_folders_unusable', str(error)) from error
        self._lock = threading.RLock()               # service state only; never held across an RPC wait
        self._ids = itertools.count(1)
        self._gen = None
        self._unconfirmed = []                       # earlier generations not yet confirmed exited (still owned)
        self._fatal = None
        self._closed = False
        self._deaths = 0
        self._warming = False                        # a background start + warm-up is in flight (see prewarm)
        self._prewarm_not_before = 0.0               # monotonic time before which a stopped warm-up is not retried
        self._ready_info = None
        self._stats = None
        self._devices = self._devices_at = None      # from the latest actual reply (not startup, never from config)
        self.starts = self.served = 0
        self.vram_mib = self.peak_mib = None
        self.vram_method = None
        self.last_used = time.monotonic()

    # ---- clients ------------------------------------------------------------------------------------------------
    def language(self, owner):
        return LanguageClient(self, owner)

    def speakers(self, owner):
        return SpeakerClient(self, owner)

    # ---- child lifecycle ------------------------------------------------------------------------------------------
    def _env(self):
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
        env.pop('CUDA_VISIBLE_DEVICES', None)        # a CUDA worker; an inherited blank must never hide the GPU
        return env

    def _live(self, gen):
        return (gen is not None and not gen.dead.is_set() and not gen.eof and not gen.stopping
                and gen.death is None and not _ended(gen.view))

    def _ensure(self):
        """The live generation, or a freshly spawned one. Under the service lock (spawn only; readiness and request
        waits happen outside it). The current generation is ALWAYS retired before a replacement, and fatal +
        unconfirmed are rechecked after retirement, so no worker is ever spawned beside a live old one or before a
        fatal is latched."""
        waited = False
        while True:
            ending = None
            with self._lock:
                if self._closed:
                    raise AnalysisCancelled('analysis_service_closed', 'the analysis service is closing')
                if self._fatal is not None:
                    raise from_record(self._fatal)
                gen = self._gen
                if self._live(gen):
                    return gen
                if gen is not None and gen.eof and not gen.dead.is_set() and not waited:
                    ending = gen                     # its reader is publishing the death: wait for that (unlocked)
                else:
                    if gen is not None:
                        self._retire(gen)
                    self._reap()
                    if self._fatal is not None:
                        raise from_record(self._fatal)
                    if self._unconfirmed:
                        pids = [g.pid for g in self._unconfirmed]
                        raise MovieResourceWait('analysis_previous_worker_alive',
                                                'earlier analysis worker(s) %s not confirmed exited; no worker beside '
                                                'them' % pids)
                    return self._spawn()
            ending.dead.wait(2 * STOP_WAIT_S + 5.0)
            waited = True

    def _retire(self, gen):
        """Under self._lock: take gen out of service before any replacement. A generation whose death is not yet
        recorded is stopped (resource: its lease ended / it is stopping); an already-dead one is only ended through
        its handle and settled, so its recorded fault (and the consecutive-death count) are kept."""
        if gen.death is None:
            self._stop_gen(gen, _rec('resource', 'analysis_lease_preempted',
                                     'the movie GPU lease ended under the analysis worker'))
            return
        with gen.lock:
            gen.stopping = True
        if self._gen is gen:
            self._gen = None
        if not gen.released and gen not in self._unconfirmed:
            self._unconfirmed.append(gen)            # retired, not yet confirmed exited
        self._reap()

    def _spawn(self):
        # A worker that is being started is in use: idle unload counts from here. Without this a worker spawned after an idle
        # gap longer than IDLE_UNLOAD_S was unloaded again the moment it became ready (stale last_used, no request registered
        # yet), and every following call spawned another one until a request happened to register in time (minutes lost).
        self.last_used = time.monotonic()
        try:
            view = self.lease_source(HOLDER_NAME)
        except MovieFault:
            raise                                    # the holder/hub decided: a real hold, or a permanent (fatal) fault
        except Exception as error:  # noqa: BLE001
            if fault_record(error) is not None:
                raise
            if type(error).__name__ in _ENDED_ERRORS:
                raise MovieResourceWait('analysis_lease_ended', 'the movie GPU lease ended while the analysis view was '
                                        'created: %s' % error) from error
            raise MovieConfigurationError('analysis_lease_failed', '%s: %s' % (type(error).__name__, error)) from error
        # Owned from here: every failure before a Popen exists releases this view (nothing else holds it).
        log = None
        try:
            try:
                view.check()
            except MovieFault:
                raise                                # explicit metadata decides (resource hold or fatal)
            except Exception as error:  # noqa: BLE001
                if fault_record(error) is not None:
                    raise
                if _ended(view) or type(error).__name__ in _ENDED_ERRORS:
                    raise MovieResourceWait('analysis_lease_ended', 'movie GPU lease ended before the analysis '
                                            'worker started: %s' % error) from error
                raise MovieConfigurationError('analysis_lease_check_failed',      # invalid config / programming fault
                                              '%s: %s' % (type(error).__name__, error)) from error
            budget = getattr(view, 'budget_mib', None)
            others = getattr(view, 'other_residents_mib', None)
            if budget is not None and others is not None and self.peak_mib is not None \
                    and int(budget) - int(others) < int(self.peak_mib):
                raise MovieResourceWait('analysis_vram_budget',
                                        'lease budget %s MiB minus %s MiB of residents is below the measured analysis '
                                        'peak %s MiB' % (budget, others, self.peak_mib))
            log = open(self.log_path, 'ab')
            flags = {'creationflags': CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS} if os.name == 'nt' else {}
            try:
                popen = view.spawn([self.python, '-I', '-B', str(self.script)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=log, cwd=str(self.script.parent), env=self._env(),
                                   text=True, encoding='utf-8', bufsize=1, **flags)
            except Exception as error:  # noqa: BLE001
                if _ended(view) or type(error).__name__ in _ENDED_ERRORS:
                    raise MovieResourceWait('analysis_lease_ended', 'the movie GPU lease ended before the analysis '
                                            'worker started: %s' % error) from error
                raise MovieConfigurationError('analysis_spawn_failed',
                                              '%s: %s' % (type(error).__name__, error)) from error
        except BaseException:
            self._close_log(log)
            try:
                view.release('analysis_spawn_refused')
            except Exception:  # noqa: BLE001 - release never raises by contract
                pass
            raise
        gen = _Gen(popen, view, log)
        self._gen = gen
        self.starts += 1
        try:                                         # ONE boundary: any failure stops and tracks THIS process
            if hasattr(view, 'report_resident'):
                view.report_resident(popen.pid, None)    # live GPU resident, not yet measured: active-unknown
            threading.Thread(target=self._read, args=(gen,), name='analysis-service-reader', daemon=True).start()
            threading.Thread(target=self._watch, args=(gen,), name='analysis-service-watchdog', daemon=True).start()
            self._send(gen, {'id': 0, 'op': 'init', 'protocol': _proto.PROTOCOL, 'config_file': self.config_file,
                             'io_dir': str(self.io_dir), 'cache_root': str(self.cache_root),
                             'load_deadline_s': READY_TIMEOUT_S + CHILD_DEADLINE_MARGIN_S,
                             'lease': {'cancel_event_name': getattr(view, 'cancel_event_name', None),
                                       'sem_mode': getattr(view, 'sem_mode', None)},
                             'budget': {'budget_mib': budget, 'other_residents_mib': others}})
            self.emit('analysis_worker_started', pid=popen.pid, lease_id=getattr(view, 'lease_id', None),
                      budget_mib=budget, other_residents_mib=others)
        except BaseException as error:
            # ANY post-spawn failure stops THIS generation with the original fault (idempotent _stop_gen): the init
            # wait is settled, the process is ended through its handle and tracked until it is confirmed exited.
            record = fault_record(error) or _rec('fatal', 'analysis_start_failed',
                                                 '%s: %s' % (type(error).__name__, error))
            self._stop_gen(gen, record)
            if isinstance(error, MovieFault) or not isinstance(error, Exception):
                raise
            raise from_record(record) from error
        return gen

    def _send(self, gen, message):
        """One protocol line. A failed send stops the child and fails its pending requests at once, then raises."""
        text = json.dumps(message, allow_nan=True, separators=(',', ':'))
        if len(text) > _proto.MAX_LINE_CHARS:
            raise MovieConfigurationError('analysis_protocol_violation', 'request line exceeds its bound')
        try:
            with gen.write_lock:
                gen.popen.stdin.write(text + '\n')
                gen.popen.stdin.flush()
        except (OSError, ValueError) as error:
            if gen.eof or gen.death is not None:
                # The child already ended: its reader publishes the real death record and settles every registered
                # request (this one included); no second, misleading fault is latched here.
                return
            if _ended(gen.view):
                record = _rec('resource', 'analysis_lease_preempted', 'movie GPU lease ended; send failed: %s' % error)
            else:
                record = _rec('fatal', 'analysis_send_failed', '%s: %s' % (type(error).__name__, error))
            self._stop_gen(gen, record)
            raise from_record(record) from error

    def _read(self, gen):
        """Reader thread of one generation: dispatch replies by id; on EOF settle every pending request."""
        stream = gen.popen.stdout
        try:
            while True:
                raw = stream.readline(_proto.MAX_LINE_CHARS + 1)
                if not raw:
                    break
                if len(raw) > _proto.MAX_LINE_CHARS:
                    gen.violation = 'reply line exceeds %d characters' % _proto.MAX_LINE_CHARS
                    break
                try:
                    message = json.loads(raw)
                except ValueError:
                    gen.violation = 'reply line is not JSON'
                    break
                if not isinstance(message, dict) or not isinstance(message.get('id'), int):
                    gen.violation = 'reply without an id'
                    break
                rid = message['id']
                final = 'result' in message or 'result_file' in message or 'failure' in message
                with gen.lock:
                    if rid == 0:
                        if gen.ready.event.is_set() or not final:
                            continue                 # one init reply per generation
                        slot = gen.ready
                        if isinstance(message.get('failure'), dict):
                            gen.init_failure = message['failure']
                    else:
                        slot = gen.pending.get(rid)
                        if slot is None:
                            self._drop_reply_audio(message)
                            if 'result_file' in message:
                                self._drop_result_file(message['result_file'])   # never leak an orphan result
                            continue
                        if not final:
                            if message.get('started'):
                                slot.started = time.monotonic()
                            continue
                        gen.pending.pop(rid, None)
                    slot.reply = message
                    abandoned = slot.abandoned
                if abandoned and 'result_file' in message:
                    self._drop_result_file(message['result_file'])
                if abandoned:
                    self._drop_reply_audio(message)
                self._clean(slot.temps)
                slot.event.set()
        except (OSError, ValueError):
            pass
        finally:
            gen.eof = True
            self._on_death(gen)

    def _watch(self, gen):
        """Service-owned deadlines of one generation (they hold even when every caller has detached): lease end,
        model load, each request's queue wait and run time. A breach stops the generation; fatal ones are sticky."""
        while not gen.dead.wait(WATCH_S):
            if gen.stopping:
                return
            now, record = time.monotonic(), None
            if _ended(gen.view):
                record = _rec('resource', 'analysis_lease_preempted', 'movie GPU lease ended under the analysis worker')
            elif not gen.ready.event.is_set() and now > gen.spawned + READY_TIMEOUT_S:
                record = _rec('fatal', 'analysis_load_timeout',
                              'the analysis models did not load within %.0f s' % READY_TIMEOUT_S)
            else:
                with gen.lock:
                    slots = list(gen.pending.values())
                for slot in slots:
                    if slot.started is None and now > slot.enqueued + QUEUE_TIMEOUT_S:
                        record = _rec('fatal', 'analysis_queue_timeout',
                                      'an analysis request did not start within %.0f s' % QUEUE_TIMEOUT_S)
                        break
                    if slot.started is not None and slot.timeout and now > slot.started + slot.timeout:
                        record = _rec('fatal', 'analysis_timeout',
                                      'an analysis request exceeded %.0f s' % slot.timeout)
                        break
            if record is not None:
                self._stop_gen(gen, record)
                return

    def _death_record(self, gen, code):
        if gen.violation:
            return _rec('fatal', 'analysis_protocol_violation', gen.violation)
        if gen.init_failure is not None:
            return dict(gen.init_failure)            # the worker's own structured setup/admission fault
        if _ended(gen.view):
            return _rec('resource', 'analysis_lease_preempted', 'movie GPU lease ended: %s' % getattr(
                getattr(gen.view, 'event', None), 'reason', 'ended'))
        if code == _proto.EXIT_RESOURCE:
            return _rec('resource', 'analysis_guard_yield', self._tail())
        if code == EXIT_GUARD_FAILURE:
            return _rec('fatal', 'analysis_guard_failure', self._tail())
        if code == EXIT_DEADLINE:
            return _rec('fatal', 'analysis_deadline', self._tail())
        return _rec('fatal', 'analysis_worker_exited', 'exit %s: %s' % (code, self._tail()))

    def _fail_pending(self, gen, record):
        """Fail the ready wait and every pending request of this generation with record (idempotent). Their exchange
        files may still be open in the child: they move to gen.orphan_temps and are removed only after it exited."""
        with gen.lock:
            if gen.death is None:
                gen.death = record
            death = gen.death
            pending = list(gen.pending.values())
            gen.pending.clear()
            if not gen.ready.event.is_set():
                gen.ready.reply = {'id': 0, 'failure': death}
                gen.ready.event.set()
            for slot in pending:
                slot.reply = {'failure': death}
                gen.orphan_temps.extend(slot.temps)
                slot.temps = []
        for slot in pending:
            slot.event.set()
        return death

    def _on_death(self, gen):
        """The child's stdout ended: wait for it (own handle), decide the fault, settle all its pending requests."""
        code = gen.popen.poll()
        if code is None:
            try:
                code = gen.popen.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                code = None
        # Published ATOMICALLY under the service lock (order self._lock -> gen.lock): the fatal latch, retirement
        # (out of service, tracked as unconfirmed until exit), the failed requests, and only then gen.dead. A caller
        # woken by its failed request therefore always sees the latched fatal / unconfirmed worker in _ensure.
        with self._lock:
            with gen.lock:
                record = gen.death if gen.death is not None else self._death_record(gen, code)
                stopping = gen.stopping
            loaded = isinstance(gen.ready.reply, dict) and isinstance(gen.ready.reply.get('result'), dict)
            if not stopping and record['kind'] == 'fatal':
                self._deaths += 1
                if (not loaded or gen.violation or self._deaths >= MAX_CONSECUTIVE_DEATHS) and not self._closed \
                        and self._fatal is None:
                    self._fatal = record             # setup faults stop production clearly; no reload loop
            if self._gen is gen:
                self._gen = None
            if not gen.released and gen not in self._unconfirmed:
                self._unconfirmed.append(gen)        # retired; released by _settle_gen once confirmed exited
            if code is None:
                try:
                    gen.popen.kill()
                    code = gen.popen.wait(STOP_WAIT_S)
                except (OSError, subprocess.TimeoutExpired):
                    code = gen.popen.poll()
            death = self._fail_pending(gen, record)
            self._settle_gen(gen)
            gen.dead.set()
        self.emit('analysis_worker_ended', pid=gen.pid, exit=code, fault=death)

    def _stop_gen(self, gen, record, *, graceful=False):
        """End ONE generation through its own handle; pending requests fail now with record. The view is released
        only once the child has exited; otherwise the generation stays owned in _unconfirmed. A fatal stop (startup,
        load, protocol, send, deadline) is sticky: no further spawn for that cause until close()."""
        # Latch + retire FIRST (under the service lock), then fail the waiters: a woken caller can never spawn
        # beside this still-live process or before a fatal stop is latched.
        with self._lock:
            with gen.lock:
                gen.stopping = True
                if gen.death is None:
                    gen.death = record
                death = gen.death
            if death['kind'] == 'fatal' and not self._closed and self._fatal is None:
                if death['code'] in RECOVERABLE_STOPS and self._deaths + 1 < MAX_CONSECUTIVE_DEATHS:
                    self._deaths += 1
                    self.emit('analysis_worker_timeout', fault=death, consecutive=self._deaths)
                else:
                    self._fatal = death
                    self.emit('analysis_service_failed', fault=death)
            if self._gen is gen:
                self._gen = None
            if not gen.released and gen not in self._unconfirmed:
                self._unconfirmed.append(gen)        # retired; released by _settle_gen once confirmed exited
        self._fail_pending(gen, death)
        popen = gen.popen
        if graceful and popen.poll() is None:
            try:
                with gen.write_lock:
                    popen.stdin.write(json.dumps({'op': 'shutdown'}) + '\n')
                    popen.stdin.flush()
                    popen.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                popen.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                pass
        if popen.poll() is None:
            try:
                popen.kill()
            except OSError:
                pass
            try:
                popen.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                pass
        with self._lock:
            if self._gen is gen:
                self._gen = None
            self._settle_gen(gen)

    def _settle_gen(self, gen):
        """Under self._lock: release an exited generation, or keep an unconfirmed one owned."""
        if gen.released:
            return
        if gen.popen.poll() is None:
            if gen not in self._unconfirmed:
                self._unconfirmed.append(gen)
                self.emit('analysis_worker_unconfirmed', pid=gen.pid)
            return
        gen.released = True
        if gen in self._unconfirmed:
            self._unconfirmed.remove(gen)
        with gen.lock:
            orphans, gen.orphan_temps = gen.orphan_temps, []
        self._clean(orphans)                         # the child has exited: nothing can still read them
        if hasattr(gen.view, 'forget_resident'):
            gen.view.forget_resident(gen.pid)
        self._close_log(gen.log)
        gen.log = None
        try:
            gen.view.release('analysis_worker_stopped')
        except Exception:  # noqa: BLE001 - release never raises by contract
            pass

    def _reap(self):
        """Under self._lock: retry ending every unconfirmed generation; release those that exited."""
        for gen in list(self._unconfirmed):
            if gen.popen.poll() is None:
                try:
                    gen.popen.kill()
                    gen.popen.wait(1.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._settle_gen(gen)

    @staticmethod
    def _close_log(log):
        if log is not None:
            try:
                log.close()
            except OSError:
                pass

    def _tail(self):
        try:
            with open(self.log_path, 'rb') as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 2000))
                return handle.read().decode('utf-8', 'replace')
        except OSError:
            return ''

    def _await_ready(self, gen, owner):
        deadline = gen.spawned + READY_TIMEOUT_S
        while not gen.ready.event.wait(POLL_S):
            if owner.cancelled.is_set():
                # The load continues for the next window; only this caller stops.
                raise AnalysisCancelled('window_cancelled', 'cancelled while the analysis models load')
            if self._closed:
                raise AnalysisCancelled('analysis_service_closed', 'the analysis service is closing')
            if _ended(gen.view):
                record = _rec('resource', 'analysis_lease_preempted', 'movie GPU lease ended during the model load')
                self._stop_gen(gen, record)
                raise from_record(record)
            if time.monotonic() > deadline:
                record = _rec('fatal', 'analysis_load_timeout',
                              'the analysis models did not load within %.0f s' % READY_TIMEOUT_S)
                self._stop_gen(gen, record)
                raise from_record(record)
        reply = gen.ready.reply or {}
        if 'failure' in reply:
            raise from_record(reply['failure'])
        if gen.ready_info is None:
            info = reply.get('result') or {}
            if not info.get('ready'):
                record = _rec('fatal', 'analysis_protocol_violation', 'init reply without ready')
                self._stop_gen(gen, record)
                raise from_record(record)
            gen.ready_info = self._ready_info = info
            self._measure(gen, force=True)
            self.emit('analysis_worker_ready', pid=gen.pid, load_seconds=info.get('load_seconds'),
                      devices=info.get('devices'), memory=info.get('memory'), vram_mib=self.vram_mib)

    # ---- start-up -------------------------------------------------------------------------------------------------
    def prewarm(self, translation=False):
        """Start the worker now and warm its models in the background, so the first real request of a dub does not pay the
        start (about 13 s) and the first call of every step (about 20 s of analysis, about 25 s more for the translator) while
        the TV shows 'Preparing'. Never blocks and never raises; real requests queue behind the warm-up in the worker's order.
        A resource hold or any fault only means the first real request starts the worker as before."""
        with self._lock:
            if self._closed or self._fatal is not None or self._warming or time.monotonic() < self._prewarm_not_before:
                return False
            gen = self._gen
            if gen is not None and getattr(gen, 'warmed', False) and self._live(gen):
                return False                         # the live worker was already warmed
            self._warming = True
        threading.Thread(target=self._prewarm, args=(bool(translation),), name='analysis-service-prewarm', daemon=True).start()
        return True

    def _prewarm(self, translation):
        began = time.monotonic()
        try:
            owner = _NoOwner()
            gen = self._ensure()
            self._await_ready(gen, owner)
            if getattr(gen, 'warmed', False):
                return
            gen.warmed = True
            result = self.call('warmup', {'translation': translation}, owner) or {}
            self.emit('analysis_prewarmed', seconds=round(time.monotonic() - began, 1), steps=result.get('steps'),
                      translation=translation)
        except Exception as error:  # noqa: BLE001 - a warm-up is an optimisation: whatever stops it, real requests go on
            record = fault_record(error)
            self._prewarm_not_before = time.monotonic() + PREWARM_RETRY_S
            self.emit('analysis_prewarm_skipped', code=(record or {}).get('code') or type(error).__name__,
                      seconds=round(time.monotonic() - began, 1))
        finally:
            with self._lock:
                self._warming = False

    # ---- requests -------------------------------------------------------------------------------------------------
    def call(self, op, args, owner=None, *, timeout=None):
        """One RPC; returns the decoded JSON result or raises a relay.movie_faults error. No service lock is held
        while waiting for the child."""
        owner = owner if owner is not None else _NoOwner()
        if owner.cancelled.is_set():
            raise AnalysisCancelled('window_cancelled', 'analysis %s cancelled' % op)
        requested = time.monotonic()
        timeout = float(timeout or OP_TIMEOUTS.get(op, 600.0))
        temps = []
        try:
            payload = self._encode_args(args, temps)
            while True:
                gen = self._ensure()
                self._await_ready(gen, owner)
                rid = next(self._ids)
                slot = _Slot(temps, timeout)
                with gen.lock:
                    if owner.cancelled.is_set():
                        raise AnalysisCancelled('window_cancelled', 'analysis %s cancelled' % op)
                    if gen.stopping or gen.dead.is_set() or gen.death is not None:
                        continue                     # idle unload/death raced this call: use a fresh generation
                    gen.pending[rid] = slot
                break
        except BaseException:
            self._clean(temps)
            raise
        # The slot owns the exchange files now: they are removed on its reply or the generation's end.
        self.last_used = time.monotonic()
        try:
            self._send(gen, {'id': rid, 'op': op, 'deadline_s': timeout + CHILD_DEADLINE_MARGIN_S, **payload})
        except BaseException:
            # Settle THIS registered request. If the send stopped the generation, its slot was already failed and its
            # files moved to the generation (removed after exit); otherwise the child never received it.
            with gen.lock:
                unsent = gen.pending.pop(rid, None)
            if unsent is not None:
                self._clean(unsent.temps)
            raise
        queued_until = time.monotonic() + QUEUE_TIMEOUT_S
        while not slot.event.wait(POLL_S):
            if owner.cancelled.is_set():
                with gen.lock:
                    completed = slot.reply is not None
                    if not completed:
                        slot.abandoned = True        # the reader drops the late reply and its files
                if not completed:
                    try:
                        self._send(gen, {'op': 'cancel', 'target': rid})
                    except MovieFault:
                        pass                         # the send failure already stopped the generation
                    raise AnalysisCancelled('window_cancelled', 'analysis %s cancelled' % op)
                break                                # completed meanwhile: handled below (discarded as cancelled)
            if self._closed:
                raise AnalysisCancelled('analysis_service_closed', 'the analysis service is closing')
            if _ended(gen.view):
                record = _rec('resource', 'analysis_lease_preempted', 'movie GPU lease ended during analysis %s' % op)
                self._stop_gen(gen, record)
                raise from_record(record)
            now = time.monotonic()
            if slot.started is None and now > queued_until:
                record = _rec('fatal', 'analysis_queue_timeout', 'analysis %s did not start within %.0f s'
                              % (op, QUEUE_TIMEOUT_S))
                self._stop_gen(gen, record)
                raise from_record(record)
            if slot.started is not None and now > slot.started + timeout:
                record = _rec('fatal', 'analysis_timeout', 'analysis %s exceeded %.0f s' % (op, timeout))
                self._stop_gen(gen, record)
                raise from_record(record)
        reply = slot.reply or {}
        self.last_used = time.monotonic()
        if isinstance(reply.get('stats'), dict):
            self._stats = reply['stats']
        if isinstance(reply.get('devices'), dict):
            self._devices, self._devices_at = reply['devices'], time.time()
        if owner.cancelled.is_set():
            self._drop_reply_audio(reply)
            if 'result_file' in reply:
                self._drop_result_file(reply['result_file'])
            raise AnalysisCancelled('window_cancelled', 'analysis %s cancelled' % op)
        if 'failure' in reply:
            raise from_record(reply['failure'])
        value = self._decode_result(reply)
        self.served += 1
        with self._lock:
            self._deaths = 0
        try:
            self._measure(gen)
        except BaseException:
            self._drop_reply_audio(reply)
            raise
        if owner.cancelled.is_set():                 # re-checked immediately before returning (after decode/measure)
            self._drop_reply_audio(reply)
            raise AnalysisCancelled('window_cancelled', 'analysis %s cancelled' % op)
        finished = time.monotonic()
        if finished - requested >= SLOW_OP_LOG_S:
            begun = slot.started if slot.started is not None else requested
            self.emit('analysis_op', op=op, total_s=round(finished - requested, 1), waited_s=round(begun - requested, 1),
                      ran_s=round(finished - begun, 1))
        return value

    def _encode_args(self, args, temps):
        value = self._jsonable(args, temps)
        text = json.dumps(value, allow_nan=True, separators=(',', ':'))
        if len(text) <= _proto.INLINE_MAX:
            return {'args': value}
        try:
            ref = _proto.write_owned_json(self.io_dir, 'a', value, _proto.MAX_ARGS_BYTES)
        except ValueError as error:
            raise MovieConfigurationError('analysis_args_too_large', str(error)) from error
        except OSError as error:
            raise MovieConfigurationError('analysis_io_failed', str(error)) from error
        temps.append(ref['path'])
        return {'args_file': ref}

    def _jsonable(self, value, temps):
        if isinstance(value, _FileArg):
            return {'__file__': self._file_ref(value.path, temps)}
        if isinstance(value, _ArrayArg):
            return {'__array__': self._array_ref(value.array, temps)}
        if isinstance(value, dict):
            return {str(k): self._jsonable(v, temps) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v, temps) for v in value]
        module = type(value).__module__
        if module == 'numpy' or module.startswith('numpy.'):
            return self._jsonable(value.tolist(), temps)
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise MovieConfigurationError('analysis_args_invalid', 'unsupported argument type %s' % type(value).__name__)

    def _file_ref(self, path, temps):
        """Stage the caller's (mutable) audio as an immutable per-request copy inside this service's exchange folder.
        The copy belongs to the request: removed on its final reply, or after the child is confirmed exited."""
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise MovieConfigurationError('analysis_input_missing', 'analysis input is not a file: %s' % path)
        try:
            ref = _proto.stage_copy(resolved, self.io_dir, _proto.MAX_AUDIO_BYTES)
        except ValueError as error:
            raise MovieConfigurationError('analysis_input_too_large', str(error)) from error
        except OSError as error:
            raise MovieConfigurationError('analysis_input_unreadable', str(error)) from error
        temps.append(ref['path'])
        return ref

    def _array_ref(self, array, temps):
        import numpy as np
        data = np.ascontiguousarray(np.asarray(array, dtype='<f4'))
        if data.ndim != 1:
            raise MovieConfigurationError('analysis_array_invalid', 'analysis audio arrays must be one-dimensional')
        if data.nbytes > _proto.MAX_ARRAY_BYTES:
            raise MovieConfigurationError('analysis_array_too_large',
                                          'analysis array of %d bytes exceeds its bound' % data.nbytes)
        path = self.io_dir / ('w-%s.npy' % uuid.uuid4().hex)
        temporary = path.with_suffix('.tmp')
        try:
            with open(temporary, 'wb') as handle:
                np.save(handle, data, allow_pickle=False)
            os.replace(temporary, path)
            temps.append(str(path))
            digest, size = _proto.sha256_file(path, _proto.MAX_ARRAY_BYTES + _proto.NPY_HEADER_SLACK)
        except (OSError, ValueError) as error:
            _proto.unlink_quiet(temporary)
            raise MovieConfigurationError('analysis_io_failed', str(error)) from error
        return {'path': str(path), 'sha256': digest, 'bytes': size, 'dtype': 'float32', 'shape': [int(data.shape[0])]}

    def _decode_result(self, reply):
        if 'result' in reply:
            return reply['result']
        ref = reply.get('result_file')
        try:
            return _proto.read_owned_json(ref, self.io_dir, _proto.MAX_RESULT_BYTES)
        except (OSError, ValueError, TypeError) as error:
            raise MovieConfigurationError('analysis_protocol_violation', 'result file refused: %s' % error) from error
        finally:
            if isinstance(ref, dict):
                self._drop_result_file(ref)

    def _drop_result_file(self, ref):
        path = ref.get('path') if isinstance(ref, dict) else None
        if isinstance(path, str) and _proto.directly_inside(path, self.io_dir):
            _proto.unlink_quiet(path)

    def _drop_reply_audio(self, reply):
        # Generic synthesis receipts are small inline objects, never unowned paths.
        result = reply.get('result') if isinstance(reply, dict) else None
        ref = result.get('audio_ref') if isinstance(result, dict) else None
        if isinstance(ref, dict):
            self._drop_result_file(ref)

    @staticmethod
    def _clean(paths):
        for path in paths or ():
            _proto.unlink_quiet(path)

    # ---- metadata -------------------------------------------------------------------------------------------------
    def ready_info(self, owner=None):
        """The child's init report (devices, speaker policy, memory); starts the worker when none is warm."""
        owner = owner if owner is not None else _NoOwner()
        if owner.cancelled.is_set():
            raise AnalysisCancelled('window_cancelled', 'analysis metadata cancelled')
        gen = self._ensure()
        self._await_ready(gen, owner)
        if owner.cancelled.is_set():
            raise AnalysisCancelled('window_cancelled', 'analysis metadata cancelled')
        info = json.loads(json.dumps(gen.ready_info))
        if owner.cancelled.is_set():                 # re-checked immediately before returning
            raise AnalysisCancelled('window_cancelled', 'analysis metadata cancelled')
        return info

    def _measure(self, gen, *, force=False):
        """Report the child's GPU memory to its view (LeaseView.report_resident; read by Vox admission). WDDM per-
        process reader first, then nvidia-smi; None (active-unknown) when neither gives a usable value."""
        now = time.monotonic()
        if gen.dead.is_set() or (not force and now - gen.last_measure < MEASURE_INTERVAL_S):
            return
        gen.last_measure = now
        mib, method = None, None
        if self.vram_reader is not None:
            try:
                reading = self.vram_reader(gen.pid) or {}
                if reading.get('rows') and reading.get('process_dedicated_mib') is not None:
                    mib, method = int(round(float(reading['process_dedicated_mib']))), reading.get('method')
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
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == gen.pid and parts[1].isdigit():
                    mib, method = int(parts[1]), 'nvidia-smi --query-compute-apps used_memory'
        if hasattr(gen.view, 'report_resident') and not gen.released:
            gen.view.report_resident(gen.pid, mib)   # None keeps the pid active-unknown
        self.vram_mib = mib
        if mib is not None:
            self.vram_method = method
            self.peak_mib = max(self.peak_mib or 0, mib)
            budget = getattr(gen.view, 'budget_mib', None)
            if budget is not None and mib > int(budget):
                self.emit('analysis_over_budget', pid=gen.pid, vram_mib=mib, budget_mib=budget)

    def device_report(self):
        """Never spawns. vram_mib is None whenever no usable measurement exists (unknown is never 0)."""
        gen = self._gen
        info = (gen.ready_info if gen is not None else None) or self._ready_info or {}
        if self._closed:
            state = 'closed'
        elif self._fatal is not None:
            state = 'failed'
        elif gen is None or gen.dead.is_set():
            state = 'idle'
        else:
            state = 'ready' if gen.ready_info is not None else 'loading'
        alive = gen is not None and not gen.dead.is_set()
        return {'version': VERSION, 'state': state, 'device': 'cuda', 'cpu_fallback': False,
                'process': 'lease-owned analysis worker (lease Job)', 'pid': gen.pid if alive else None,
                'lease_id': getattr(gen.view, 'lease_id', None) if alive else None,
                'budget_mib': getattr(gen.view, 'budget_mib', None) if alive else None,
                'vram_mib': self.vram_mib if alive else None, 'peak_mib': self.peak_mib,
                'vram_method': self.vram_method, 'starts': self.starts, 'served': self.served,
                # 'models' is the device_report() the worker attached to its latest actual reply (lazily loaded ASR/MT
                # included); 'startup_models' is only what was loaded at startup. Neither is inferred from config.
                'models': self._devices, 'models_reported_at': self._devices_at,
                'startup_models': info.get('devices'), 'speaker_policy': info.get('speaker_policy'),
                # The worker's shared CT2 policy identity (relay.ct2_policy): id, chosen compute type, the child's own
                # installed ctranslate2 version. From its latest reply, else its init report; never from config.
                'ct2_policy': (self._devices or {}).get('ct2_policy') or info.get('ct2_policy'),
                'memory': info.get('memory'), 'cache': self._stats, 'fatal': self._fatal,
                'unconfirmed_workers': [g.pid for g in list(self._unconfirmed)]}

    # ---- idle unload / close ----------------------------------------------------------------------------------------
    def unload(self, reason='unloaded'):
        """Stop the worker now: its models leave RAM and VRAM and its lease view is released (non-blocking). Returns 'stopped', 'absent'
        (no live worker) or 'busy' (a request is still running or the lock is taken: ask again shortly). A worker that is still loading
        its models is stopped too."""
        if not self._lock.acquire(blocking=False):
            return 'busy'
        try:
            self._reap()
            gen = self._gen
            if gen is None or gen.dead.is_set():
                return 'absent'
            with gen.lock:
                if gen.pending:
                    return 'busy'
                gen.stopping = True                  # a racing call() sees this and takes a fresh generation
            self._stop_gen(gen, _rec('cancelled', 'analysis_idle_unload', reason), graceful=True)
            self.emit('analysis_worker_stopped', reason=reason)
            return 'stopped'
        finally:
            self._lock.release()

    def idle_check(self):
        """Unload the models and release the lease view after IDLE_UNLOAD_S without a request (non-blocking lock).
        Also retries ending any unconfirmed earlier worker."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            self._reap()
            gen = self._gen
            if gen is None or gen.dead.is_set() or time.monotonic() - self.last_used < IDLE_UNLOAD_S:
                return False
            with gen.lock:
                if gen.pending or not gen.ready.event.is_set():
                    return False
                gen.stopping = True                  # a racing call() sees this and takes a fresh generation
            self._stop_gen(gen, _rec('cancelled', 'analysis_idle_unload', 'idle unload'), graceful=True)
            self.emit('analysis_worker_stopped', reason='idle')
            return True
        finally:
            self._lock.release()

    def close(self):
        self._closed = True
        with self._lock:
            gen = self._gen
            if gen is not None:
                self._stop_gen(gen, _rec('cancelled', 'analysis_service_closed', 'the analysis service closed'),
                               graceful=True)
            self._reap()
            remaining = [g.pid for g in self._unconfirmed]
        if remaining:
            # Ownership is kept (views + residents); the hub's lease release (synchronous Job drain) follows this.
            # The exchange folder stays until those workers are gone; it is this service's own folder only.
            self.emit('analysis_worker_unconfirmed', pids=remaining)
        else:
            shutil.rmtree(self.io_dir, ignore_errors=True)
        self.emit('analysis_service_closed', starts=self.starts, served=self.served)


def _relay_error(error):
    """Faults pass through; anything else from the client layer is an explicit fatal record."""
    if fault_record(error) is not None:
        return error
    return MovieConfigurationError('analysis_client_failed', '%s: %s' % (type(error).__name__, error))


class _Client:
    def __init__(self, service, owner):
        self._service, self._owner = service, owner
        self.config = service.config
        self.device = 'cuda'

    def _call(self, op, args):
        try:
            return self._service.call(op, args, self._owner)
        except MovieFault:
            raise
        except Exception as error:  # noqa: BLE001
            raise _relay_error(error) from error


class LanguageClient(_Client):
    """LanguageTools call signatures and return shapes, served by the lease-owned worker."""

    def text_language(self, text, declared=None):
        code, confidence, basis = self._call('text_language', {'text': str(text), 'declared': declared})
        return str(code), float(confidence), str(basis)

    def speech(self, wav, language=None, *, vad=True, beam_size=5, task='transcribe'):
        segments, info = self._call('speech', {'wav': _FileArg(wav), 'language': language, 'vad': bool(vad),
                                               'beam_size': int(beam_size), 'task': task})
        return list(segments), dict(info)

    def generic_vad(self, wav):
        result = self._call('generic_vad', {'wav': _FileArg(wav)})
        self.generic_vad_receipt = dict(result['receipt'])
        return list(result['regions'])

    def generic_synthesize(self, text, voice, output, window=None):
        """Child generates full PCM; parent only validates/copies it to this job's output."""
        output = Path(output).resolve()
        if not output.is_relative_to(Path(self.config.cache).resolve()):
            raise MovieConfigurationError('generic_output_outside_cache', 'Generic takes must remain inside the movie cache')
        receipt = self._call('generic_synthesize', {'text': text, 'voice': voice})
        ref, staged = receipt.get('audio_ref'), None
        try:
            self._owner.check()
            source = _proto.verify_ref(ref, _proto.MAX_AUDIO_BYTES, root=self._service.io_dir)
            if receipt.get('sha256') != ref['sha256']:
                raise MovieConfigurationError('generic_audio_receipt_mismatch', 'Synthesis receipt and transferred audio differ')
            output.parent.mkdir(parents=True, exist_ok=True)
            staged = _proto.stage_copy(source, output.parent, _proto.MAX_AUDIO_BYTES)
            if staged['sha256'] != ref['sha256']:
                raise MovieConfigurationError('generic_audio_copy_changed', 'Full PCM changed while copying from child')
            self._owner.check()
            try:
                os.link(staged['path'], output)
            except FileExistsError:
                digest, _ = _proto.sha256_file(output, _proto.MAX_AUDIO_BYTES)
                if digest != ref['sha256']:
                    raise MovieConfigurationError('generic_output_collision', 'Existing take has different bytes')
            self._owner.check()
            return {**{key: value for key, value in receipt.items() if key != 'audio_ref'},
                    'audio': str(output), 'window': window, 'fit_applied': False}
        except MovieFault:
            raise
        except Exception as error:
            raise _relay_error(error) from error
        finally:
            if isinstance(staged, dict):
                _proto.unlink_quiet(staged['path'])
            if isinstance(ref, dict):
                self._service._drop_result_file(ref)

    def english(self, text, language):
        output, basis = self._call('english', {'text': str(text), 'language': language})
        return str(output), str(basis)

    def language_id(self, wav, spans):
        """One {'language', 'probability', 'top'} (or None for a span under 0.3 s) per [start, end] span of a 16 kHz wav."""
        return list(self._call('language_id', {'wav': _FileArg(wav), 'spans': [[float(a), float(b)] for a, b in spans]}))

    def audio_events(self, wav, spans):
        """One {label: probability} (or None) per [start, end] span of a 16 kHz wav; all None without a classifier folder."""
        return list(self._call('audio_events', {'wav': _FileArg(wav), 'spans': [[float(a), float(b)] for a, b in spans]}))

    def english_batch(self, rows):
        """Translate several cues in one go (Qwen3: left-padded batches of at most 16 rows in the warm GPU child).

        rows: [{'text', 'language'}, ...] or [(text, language), ...]. Returns one entry per row, in order: either
        (english, basis) or a relay.movie_faults fault INSTANCE (MovieContentFallback for a cue that cannot be
        translated, ...) describing that row only. The instances are returned, never raised. Whole-call problems raise as
        for english(): the model cannot load, the lease ended, the owner cancelled, the child timed out."""
        rows = list(rows)
        payload = []
        for row in rows:
            if isinstance(row, dict):
                payload.append({'text': row.get('text'), 'language': row.get('language'),
                                **({'max_words': row['max_words']} if row.get('max_words') is not None else {})})
            elif isinstance(row, (list, tuple)) and len(row) in (2, 3):
                payload.append({'text': row[0], 'language': row[1], **({'max_words': row[2]} if len(row) == 3 else {})})
            else:
                raise MovieConfigurationError('translation_row_invalid', 'A translation row needs text and language')
            if not isinstance(payload[-1]['text'], str):
                raise MovieConfigurationError('translation_row_invalid', 'A translation row needs string text')
        results = []
        for start in range(0, len(payload), ENGLISH_BATCH_RPC_ROWS):
            chunk = payload[start:start + ENGLISH_BATCH_RPC_ROWS]
            answers = self._call('english_batch', {'rows': chunk})
            if not isinstance(answers, list) or len(answers) != len(chunk):
                raise MovieConfigurationError('analysis_batch_mismatch',
                                              'english_batch answered %s rows for %d' % (
                                                  len(answers) if isinstance(answers, list) else 'no', len(chunk)))
            for answer in answers:
                if isinstance(answer, dict) and 'failure' in answer:
                    results.append(from_record(answer['failure']))
                elif isinstance(answer, dict) and isinstance(answer.get('english'), str) \
                        and isinstance(answer.get('basis'), str):
                    results.append((answer['english'], answer['basis']))
                else:
                    results.append(MovieConfigurationError('analysis_batch_row_invalid',
                                                           'english_batch returned a malformed row'))
        return results

    def device_report(self):
        return (self._service.ready_info(self._owner).get('devices') or {}).get('language')


class SpeakerClient(_Client):
    """SpeakerAnalyzer call signatures and return shapes, served by the lease-owned worker."""

    def analyze(self, wav, offset=0.0):
        turns, aligned, vectors = self._call('analyze', {'wav': _FileArg(wav), 'offset': float(offset)})
        return list(turns), list(aligned), dict(vectors)

    def embed(self, waveform):
        return list(self._call('embed', {'waveform': _ArrayArg(waveform)}))

    def refine_utterances(self, waveform, rate, segments, vectors, offset=0, speech_regions=(), known_speakers=None, *,
                          whisper_no_speech=False):
        turns, templates, evidence = self._call('refine_utterances', {
            'waveform': _ArrayArg(waveform), 'rate': int(rate), 'segments': segments, 'vectors': vectors,
            'offset': offset, 'speech_regions': list(speech_regions or ()), 'known_speakers': known_speakers,
            'whisper_no_speech': bool(whisper_no_speech)})
        return list(turns), dict(templates), list(evidence)

    def policy_metadata(self):
        """SpeakerAnalyzer.policy_metadata() of the live worker (effective clustering/device/embedding policy)."""
        return self._service.ready_info(self._owner).get('speaker_policy')

    def device_report(self):
        return (self._service.ready_info(self._owner).get('devices') or {}).get('speakers')
