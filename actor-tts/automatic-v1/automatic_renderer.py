"""Program-owned movie TTS factory, Vox engine: dynamic B admission, no per-film operator token.

A changed copy of A's frozen automatic renderer (sha256 d79327f5...). Changes, nothing else:
  * renderer class VoxActorRenderer (<app>/actor-tts/runtime-adapter, package vox_actor_adapter);
  * model read from actor_runtime_manifest (the approved Q4_K runtime), not breeze_runtime_manifest;
    breeze_busy_attempts removed; admission scope vox-actor-tts;
  * a process-level movie-Vox engine host lease per AutomaticVoxRenderer, released in close();
  * no global request lock: lines may render concurrently (the engine host queues them earliest-deadline-first over
    its pool); submit(...) returns a RenderJob with job_id/deadline/priority;
  * one natural-speed render per request, using duration_fit's recorded single-attempt branch;
  * program-owned references only: every voice must carry the provenance record the pipeline wrote when it cut the
    reference from this job's own dialogue (runtime-adapter/reference_provenance.py), bound to the CURRENT job folder;
  * Vox only (owner decision 2026-10-17). There is no Breeze delegate; actor_engine other than "vox" is refused;
  * Sem priority through the movie GPU lease (lease contract): MovieLeaseGuard replaces A's
    SemPriorityGuard, which read Sem's status.json every 0.2 s (an open handle makes Sem's own status write fail).
    This file never opens Sem's status. The engine host is bound to the window's lease; its preemption stops synthesis.
_Owner is A's code except that an abort caused by a non-lease fault re-raises that fault; _check_reference keeps the
gate's specific refusal reason; staging and monitoring follow A, while final takes are published immutably.
v2 fault rules: explicit fault metadata (fault_kind/fault_code/detail) is always preserved; only a real lease/VRAM/
preemption condition becomes a wait (LeaseHold, an InterruptedError with fault_kind 'resource'); a missing lease
source or failed engine-host setup is a terminal SetupFault; every other error propagates unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ADAPTER = CANDIDATE / 'runtime-adapter'         # <app>/actor-tts/runtime-adapter
PACKAGE = 'vox_actor_adapter'
SCOPE = 'vox-actor-tts'


def _load(name, path):
    """Load a NON-candidate module (B's gate) by path, exactly as A did."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- verified loading of this package (review P3-4) -------------------------------------------------------------
# The hook execs this file from bytes it verified against actor_beta.vox.sources_sha256 and sets __vox_pins__ in its
# globals before exec. Every module of the adapter package is then exec'd from bytes whose sha256 equals its pin
# (never a __pycache__ .pyc, no re-read after the check). Loaded without pins (tests), the bytes are still exec'd from
# source; only the pin comparison is skipped.
PINS = globals().get('__vox_pins__')


class _AdapterLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, pins):
        self.pins = pins

    def find_spec(self, fullname, path=None, target=None):
        if fullname == PACKAGE:
            return importlib.util.spec_from_loader(fullname, self, origin=str(ADAPTER / '__init__.py'), is_package=True)
        if fullname.startswith(PACKAGE + '.') and '.' not in fullname[len(PACKAGE) + 1:]:
            file = ADAPTER / (fullname[len(PACKAGE) + 1:] + '.py')
            if file.is_file():
                return importlib.util.spec_from_loader(fullname, self, origin=str(file))
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__spec__.name
        file = ADAPTER / ('__init__.py' if name == PACKAGE else name[len(PACKAGE) + 1:] + '.py')
        data = file.read_bytes()
        if self.pins is not None:
            want = self.pins.get('runtime-adapter/' + file.name)
            if want is None or hashlib.sha256(data).hexdigest() != want:
                raise ImportError('%s is not the pinned file (sha256)' % file.name)
        module.__file__ = str(file)
        if name == PACKAGE:
            module.__path__ = []             # submodules resolve only through this loader
        exec(compile(data, str(file), 'exec'), module.__dict__)


def load_adapter():
    """Import <app>/actor-tts/runtime-adapter as the package `vox_actor_adapter` from verified bytes."""
    mod = sys.modules.get(PACKAGE)
    if mod is not None:
        if pathlib.Path(mod.__file__).resolve().parent != ADAPTER.resolve():
            raise ImportError('%s already loaded from %s, not %s' % (PACKAGE, mod.__file__, ADAPTER))
        return mod
    finder = next((f for f in sys.meta_path if isinstance(f, _AdapterLoader)), None)
    if finder is None:
        finder = _AdapterLoader(PINS)
        sys.meta_path.insert(0, finder)
    elif PINS is not None:
        finder.pins = PINS
    try:
        return importlib.import_module(PACKAGE)
    except BaseException:
        sys.modules.pop(PACKAGE, None)
        raise


def adapter_module(name):
    load_adapter()
    return importlib.import_module('%s.%s' % (PACKAGE, name))


# ---- fault metadata (relay.movie_faults record shape, without importing relay) ------------------------------------
FAULT_KINDS = frozenset({'fatal', 'resource', 'recoverable', 'cancelled'})
# Lease-v2 / lease-view exception types that ARE a real lease/VRAM/preemption hold (matched by class name over the
# MRO: the client is exec'd from pinned bytes, so its classes are not importable here) ...
LEASE_HOLD_TYPES = frozenset({'LeaseUnavailable', 'LeaseModeUnavailable', 'LeasePreempted', 'ViewEnded',
                              'NoSampleYet'})
# ... except these reasons (LeaseUnavailable.reason, or the preemption event's reason), which retrying cannot repair:
# configuration, pinned-source drift, and a GPU probe that could not produce a valid reading (broken/invalid reader).
# Director decision: only a VALID probe state is a wait: NoSampleYet (the holder-fed probe has not delivered its
# bounded first sample yet) and insufficient_vram (a real measured reading). Any other GpuProbeError is fatal.
LEASE_FATAL_REASONS = frozenset({'bad_config', 'source_drift', 'gpu_probe_failed'})


def explicit_fault(error):
    """(kind, code, detail) when `error` carries explicit fault metadata, else None."""
    kind, code = getattr(error, 'fault_kind', None), getattr(error, 'fault_code', None)
    if kind in FAULT_KINDS and isinstance(code, str) and code:
        return kind, code, str(getattr(error, 'detail', None) or error)
    return None


def is_lease_hold(error):
    """True only for a real lease/VRAM/preemption condition. Explicit metadata wins; anything else (configuration,
    path, model, CUDA, programming errors) is not a hold and must propagate unchanged."""
    fault = explicit_fault(error)
    if fault is not None:
        return fault[0] == 'resource'
    if isinstance(error, InterruptedError):
        return True
    names = {cls.__name__ for cls in type(error).__mro__}
    if not names & LEASE_HOLD_TYPES:
        return False              # includes GpuProbeError other than NoSampleYet: a broken reader is fatal
    reasons = {getattr(error, 'reason', None), getattr(getattr(error, 'event', None), 'reason', None)}
    return not reasons & LEASE_FATAL_REASONS


class LeaseHold(InterruptedError):
    """A real movie GPU lease hold/preemption: InterruptedError (waiting_resources) with fault_kind 'resource'."""
    fault_kind = 'resource'

    def __init__(self, code, detail):
        self.fault_code, self.detail = str(code), str(detail)
        super().__init__('%s: %s' % (self.fault_code, self.detail))


class SetupFault(RuntimeError):
    """Terminal setup failure (no lease source configured; engine host options/creation failed): never a wait."""
    fault_kind = 'fatal'

    def __init__(self, code, detail):
        self.fault_code, self.detail = str(code), str(detail)
        super().__init__('%s: %s' % (self.fault_code, self.detail))


# ---- movie GPU lease kept across the windows of a job (review P2-1) ---------------------------------------------
class JobLeaseKeeper:
    """Holds ONE movie GPU lease for the process across windows, chunks and verification retries, so engines stay warm
    under one lease (lease v2: one lease object per process until release()).

    acquire() returns the live lease or acquires one (a preempted/expired lease is released first, as lease v2
    requires). Windows attach/detach; a background thread renews the lease every renew_every_s (no Sem I/O: a watcher
    round trip) and releases it after idle_seconds with no attached window. idle_seconds=0 releases at the last detach
    (per-window behaviour)."""

    def __init__(self, source, *, idle_seconds=150.0, renew_every_s=5.0, clock=time.perf_counter):
        self.source, self.idle_seconds, self.renew_every_s, self.clock = source, float(idle_seconds), float(renew_every_s), clock
        self._lock = threading.RLock()            # lease acquire / release / renew
        self._users_lock = threading.Lock()       # attached windows (never waits behind an acquire)
        self._lease, self._users, self._idle_since, self._renewed = None, set(), clock(), clock()
        self.acquired = 0
        self.released = []
        self._drains = []                         # host.drain callables: engines stopped BEFORE any release
        self._stop = threading.Event()
        self._thread = None

    @property
    def lease(self):
        return self._lease

    @staticmethod
    def live(lease):
        return lease is not None and not lease.cancel_token.is_set()

    def acquire(self, holder, user=None):
        if user is not None:
            with self._users_lock:
                self._users.add(user)
        with self._lock:
            lease = self._lease
            if self.live(lease) or (lease is not None and getattr(lease, 'preempted', False)):
                # A PREEMPTED lease is returned as is: the caller's check() reports the preemption (the render in
                # progress aborts) and releases it; only the next acquire takes a new lease.
                return lease
            if lease is not None:
                self._release_locked('replaced:%s' % getattr(getattr(lease, 'event', None), 'reason', 'ended'))
                if self._lease is not None:
                    raise LeaseHold('movie_lease_draining', 'Movie GPU lease: previous engines still draining')
            if self.source is None:
                # No hub registered / no gpu_lease configured: configuration, not a resource hold.
                raise SetupFault('movie_lease_source_missing',
                                 'Movie GPU lease source is not configured (no registered lease hub)')
            try:
                lease = self.source(holder)
            except Exception as error:   # noqa: BLE001 - only a real lease/VRAM hold becomes a wait
                if not is_lease_hold(error):
                    raise                 # configuration/path/model/CUDA/programming: the original fault, unchanged
                if explicit_fault(error) is not None:
                    raise                 # an explicit resource fault keeps its own code/detail
                raise LeaseHold('movie_lease_unavailable',
                                'Movie GPU lease unavailable: %s' % MovieLeaseGuard._reason(error)) from error
            self._lease, self._renewed = lease, self.clock()
            self.acquired += 1
            self._start()
            return lease

    def detach(self, user):
        with self._users_lock:
            self._users.discard(user)
            idle = not self._users
            if idle:
                self._idle_since = self.clock()
        if idle and self.idle_seconds <= 0:
            self.release('window_closed')         # per-window keeper (waits for an acquire in progress, then frees)

    def release(self, reason):
        with self._lock:
            self._release_locked(reason)

    def register_drain(self, drain):
        """drain(reason) -> bool: stop every engine child running under the lease and return once they have exited."""
        with self._lock:
            if drain not in self._drains:
                self._drains.append(drain)

    def _release_locked(self, reason):
        """Codex note 1: the lease is released only AFTER every registered engine has stopped and drained."""
        lease = self._lease
        if lease is None:
            return
        drained = True
        for drain in list(self._drains):
            try:
                drained = bool(drain(str(reason))) and drained
            except Exception:   # noqa: BLE001 - a failed drain keeps the lease (fail closed: the Job still owns them)
                drained = False
        if not drained:
            self.released.append('drain_pending:%s' % reason)
            return                                # retried by the keeper thread / next acquire
        self._lease = None
        self.released.append(str(reason))
        lease.release(str(reason))

    def renew_due(self):
        with self._lock:
            lease = self._lease
            if not self.live(lease) or self.clock() - self._renewed < self.renew_every_s:
                return
            self._renewed = self.clock()
        try:
            lease.renew()
        except Exception as error:   # noqa: BLE001 - preempted/expired: release so the next window re-acquires
            self.release('renew_failed:%s' % MovieLeaseGuard._reason(error))

    def _start(self):                            # called with self._lock held
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='vox-job-lease-keeper', daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.wait(min(1.0, max(.05, self.renew_every_s / 5))):
            with self._lock:
                lease = self._lease
                if lease is None:
                    self._thread = None                # a later acquire() starts a new one (same lock: no race)
                    return
            try:
                if lease.cancel_token.is_set() and getattr(lease, 'preempted', False):
                    self.release('preempted:%s' % getattr(getattr(lease, 'event', None), 'reason', 'preempted'))
                    continue
                with self._users_lock:
                    idle = not self._users and self.clock() - self._idle_since >= self.idle_seconds
                if idle:
                    self.release('idle_grace')
                    continue
                self.renew_due()
            except Exception as error:   # noqa: BLE001 - the keeper must keep running (renew / idle release)
                self.released.append('keeper_error:%s' % type(error).__name__)

    def close(self):
        self._stop.set()
        self.release('keeper_closed')


_KEEPERS = {}
_KEEPERS_LOCK = threading.Lock()


def shared_keeper(key, source, *, idle_seconds=150.0, renew_every_s=5.0):
    """One process-level JobLeaseKeeper per lease configuration (the hook keys it on the pinned lease client)."""
    with _KEEPERS_LOCK:
        keeper = _KEEPERS.get(key)
        if keeper is None:
            keeper = _KEEPERS[key] = JobLeaseKeeper(source, idle_seconds=idle_seconds, renew_every_s=renew_every_s)
        return keeper


class MovieLeaseGuard:
    """Program resource guard over the movie GPU lease. No file I/O of its own (never opens Sem's status.json).

    keeper: a JobLeaseKeeper (the hook passes the process-level one, so the lease and warm engines survive window
    close and verification retries; review P2-1). lease_source alone builds a per-guard keeper that releases at
    window close (v1 behaviour). lease: an external lease (W4) used but never released. check() raises
    InterruptedError (-> waiting_resources) when the lane is unavailable, no lease can be acquired, or the lease is
    preempted, expired or released."""

    def __init__(self, lane_available, *, gpu_free=None, lease_source=None, lease=None, keeper=None,
                 holder='kevflix movie Vox', renew_every_s=5.0, clock=time.perf_counter):
        if not callable(lane_available):
            raise TypeError('A runtime-owned lane scheduler is required')
        self.lane_available = lane_available
        self.gpu_free = gpu_free            # kept for the hook's call signature; the lease watcher owns GPU reserve
        self.lease_source, self.holder = lease_source, holder
        self._external = lease
        if keeper is None and lease is None:
            keeper = JobLeaseKeeper(lease_source, idle_seconds=0.0, renew_every_s=renew_every_s, clock=clock)
        self.keeper = keeper
        self.renew_every_s, self.clock = float(renew_every_s), clock
        self._renewed = clock()
        self._lock = threading.Lock()

    @property
    def allow_real_transport(self):
        return self.lane_available() is True

    @property
    def lease(self):
        return self._external if self._external is not None else (self.keeper.lease if self.keeper else None)

    @property
    def acquired(self):
        return self.keeper.acquired if self.keeper else 0

    @staticmethod
    def _reason(error):
        event = getattr(error, 'event', None)
        return getattr(event, 'reason', None) or getattr(error, 'reason', None) or str(error)

    def check(self):
        if self.lane_available() is not True:
            raise LeaseHold('actor_gpu_lane_unavailable', 'Movie actor GPU lane is unavailable')
        lease = self._external if self._external is not None else self.keeper.acquire(self.holder, user=self)
        try:
            lease.check()
            if self._external is not None and self.clock() - self._renewed >= self.renew_every_s:
                lease.renew()
                self._renewed = self.clock()
            elif self._external is None:
                self.keeper.renew_due()
        except Exception as error:   # noqa: BLE001 - only LeasePreempted / ViewEnded / lease holds become a wait
            if not is_lease_hold(error):
                raise                 # anything else keeps its own (fatal) classification and metadata
            reason = self._reason(error)
            if self._external is None:
                self.keeper.release('preempted:%s' % reason)   # lease v2: release before the next acquire
            if explicit_fault(error) is not None:
                raise
            raise LeaseHold('movie_lease_preempted', 'Movie GPU lease preempted: %s' % reason) from error
        return True

    def release(self, reason='window_closed'):
        """Window closed: detach from the keeper (the shared keeper keeps the lease warm; a per-guard one releases)."""
        if self._external is None and self.keeper is not None:
            self.keeper.detach(self)


class _Owner:
    def __init__(self, owner, guard):
        self.upstream = owner
        self.guard = guard
        self.cancelled = threading.Event()
        self.reason = None
        self.error = None

    def check(self):
        if self.cancelled.is_set() or self.upstream.cancelled.is_set():
            # v2: an abort caused by a fault that is not a lease hold (or carries explicit metadata) re-raises THAT
            # error, so a fatal fault seen by the monitor is never flattened into an InterruptedError wait.
            if self.error is not None and (explicit_fault(self.error) is not None or not is_lease_hold(self.error)):
                raise self.error
            raise InterruptedError(self.reason or 'Movie actor request cancelled')
        self.upstream.check()
        if self.guard.check() is False:
            raise InterruptedError('Movie actor guard refused')

    def abort(self, reason):
        self.reason = str(reason)
        if isinstance(reason, BaseException):
            self.error = reason
        self.cancelled.set()


class AutomaticVoxRenderer:
    def __init__(self, config, owner, *, bundle, resource_guard, job_id, source_id, transport=None, engine_host=None,
                 pipeline_context=None):
        if not callable(getattr(resource_guard, 'check', None)):
            raise TypeError('Program resource guard required')
        if not job_id or not source_id:
            raise ValueError('Current job/source binding required')
        gate = bundle.get('reference_gate')
        if not callable(gate):
            raise ValueError('Dynamic B reference gate missing')
        self.runtime = adapter_module('vox_renderer')
        self.errors = adapter_module('errors')
        self.jobs = adapter_module('jobs')
        self.fit = adapter_module('duration_fit')
        self.provenance = adapter_module('reference_provenance')
        self.owner = owner
        self.guard = resource_guard
        self.bundle = bundle
        self.context = {'job_id': job_id, 'source_id': source_id}
        self._gate = gate
        self._rejected = {}
        self._epoch = 0
        self._active = set()
        self._active_lock = threading.Lock()
        self._jobs = set()
        self._engine_host = engine_host if engine_host is not None else transport
        self._lease = None
        self._host_lock = threading.Lock()      # first concurrent submit()s create exactly one host lease
        self._cache = {}
        self._config = dict(config)
        vox = self.runtime.vox_options(self._config)
        self._fit = self.fit.options(vox.get('duration_fit'))
        self._class = self.runtime.VoxActorRenderer
        manifest = self._config.get('actor_runtime_manifest')
        if not manifest:
            raise ValueError('actor_runtime_manifest (the Vox runtime manifest on this machine) is not configured')
        self._model = json.loads(pathlib.Path(manifest).read_text(encoding='utf-8-sig'))['modelSha256']
        context = pipeline_context or {}
        # The current job's folder (<cache_dir>/<job_id>), from the pipeline context, never from a voice.
        receipt_dir = context.get('receipt_dir')
        self._job_dir = pathlib.Path(receipt_dir).parent if receipt_dir else None
        selection_hashes = context.get('selection_module_sha256')
        self._selection_sha256 = dict(selection_hashes) if isinstance(selection_hashes, dict) else selection_hashes
        self._revision = context.get('revision')
        self._source_identity = context.get('source_identity') if isinstance(context.get('source_identity'), dict) else None
        self._cues = {c.get('id'): c for c in list(context.get('cues') or []) + list(context.get('all_cues') or [])
                      if isinstance(c, dict) and c.get('id') is not None}

    @property
    def model(self):
        return self._model

    @property
    def fallbacks(self):
        return {'upstream': self.bundle.get('fallbacks', []), 'preparation': dict(self._rejected)}

    def _key(self, v):
        return hashlib.sha256(json.dumps(v, sort_keys=True, default=str).encode()).hexdigest()

    def _error(self, detail):
        return self.errors.ActorUnavailable('reference_unverified', str(detail))

    def _check_reference(self, voice, context_key=None):
        resolve = self.bundle.get('resolve_cue')
        if context_key is None and self.bundle.get('voices'):
            admitted = self.bundle['voices']
            hint = voice.get('actor_id') or voice.get('speaker') or (voice.get('trust') or {}).get('actor_id')
            choices = [admitted[hint]] if hint in admitted else [
                v for v in admitted.values() if all(voice.get(k) == v.get(k) for k in ('reference', 'reference_sha256', 'reference_text'))]
            if len(choices) != 1 or any(voice.get(k) != choices[0].get(k) for k in ('reference', 'reference_sha256', 'reference_text')):
                raise self._error('Voice has no unique dynamic reference binding')
            voice = choices[0]
        if context_key is not None and callable(resolve):
            cue = context_key.get('cue_id', context_key.get('cue')) if isinstance(context_key, dict) else context_key
            decision = resolve(cue)
            if not isinstance(decision, dict) or decision.get('mode') != 'actor-reference':
                # Keep the gate's SPECIFIC refusal reason (e.g. 'speaker reference not admitted: <why>').
                reason = decision.get('reason') if isinstance(decision, dict) else None
                raise self._error('cue %s not admitted (%s): %s' % (
                    cue, decision.get('mode') if isinstance(decision, dict) else 'no decision',
                    reason or 'the gate returned no reason'))
            expected = decision.get('voice') or {}
            if any(voice.get(k) != expected.get(k) for k in ('reference', 'reference_sha256', 'reference_text')):
                raise self._error('Cue/reference mismatch')
            voice = expected
        params = inspect.signature(self._gate).parameters.values()
        contextual = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        self._gate(voice, context_key) if contextual and context_key is not None else self._gate(voice)
        return voice

    def _provenance(self, voice):
        """Program provenance of a B-resolved voice in this job (raises; VoxActorRenderer maps to reference_unverified)."""
        return self.provenance.verify(voice, self._job_dir, selection_module_sha256=self._selection_sha256,
                                      job_id=self.context['job_id'], revision=self._revision,
                                      source_identity=self._source_identity)

    def prepare_references(self, voices):
        """One unadmitted voice is retained as original by its render failure, not a window failure."""
        self.owner.check()
        for voice in voices:
            try:
                self._check_reference(voice)
                self._rejected.pop(self._key(voice), None)
            except Exception as e:   # noqa: BLE001
                self._rejected[self._key(voice)] = str(e)
        return self.fallbacks

    # -- engine ---------------------------------------------------------------------
    def _host(self, epoch=None):
        with self._host_lock:
            if epoch is not None and epoch != self._epoch:
                # close() ran while this line was acquiring the lease: never take a host lease nobody will release
                # (review P2-4).
                raise InterruptedError('Actor renderer closed before the engine was attached')
            if self._engine_host is None:
                try:
                    factory = adapter_module('factory')
                    host_module = adapter_module('vox_engine_host')
                    self._engine_host = host_module.shared_host(factory.host_options(self._config))
                except Exception as error:   # noqa: BLE001 - setup is terminal; explicit adapter metadata is kept
                    if explicit_fault(error) is not None:
                        raise
                    raise SetupFault('vox_host_setup_failed', '%s: %s' % (type(error).__name__, error)) from error
            if self._lease is None and hasattr(self._engine_host, 'lease'):
                self._lease = self._engine_host.lease()
            host = self._engine_host
        gpu_lease = getattr(self.guard, 'lease', None)
        if gpu_lease is not None and hasattr(host, 'bind_lease'):
            host.bind_lease(gpu_lease)          # Sem priority + process ownership for the engine: the movie GPU lease
        keeper = getattr(self.guard, 'keeper', None)
        if keeper is not None and hasattr(host, 'drain'):
            keeper.register_drain(host.drain)   # the lease is released only after this host's engines have exited
        return host

    def concurrency(self):
        """Scheduler view of the engine pool (None before the engine host exists)."""
        host = self._engine_host
        return host.capacity() if host is not None and hasattr(host, 'capacity') else None

    def admitted_concurrency(self):
        """FIX 6: lines this window may have in flight. max_instances 'auto' (default, cap max_instances_cap 2) or an
        explicit integer. A fresh process runs 1 instance; a 2nd is admitted only from the RUNNING instance's measured
        in-take peak (host live telemetry) when (lease budget_mib - other movie residents) holds it
        (host.admit_more); the host's fresh before-load headroom check (free >= 3072 MiB floor + peak) runs right
        before that load, and any refusal drops back to 1. Acquires the window's lease through the guard (raises
        InterruptedError -> waiting_resources, as a first render would)."""
        mode, bound = self.runtime.concurrency_mode(self.runtime.vox_options(self._config))
        if bound <= 1:
            return 1                                   # explicit 1 (or cap 1): always serial
        self.guard.check()
        host = self._host(self._epoch)
        lease = getattr(self.guard, 'lease', None)
        budget = getattr(lease, 'budget_mib', None)    # already the movie allocation above the 3072 MiB floor
        if not budget or not hasattr(host, 'admit_more'):
            return 1
        # Other measured movie residents under the same lease (e.g. the resident Demucs child), deducted once.
        # LeaseView.other_residents_mib is int | None: None (or a missing attribute, old view) = UNKNOWN -> stay at 1;
        # an unknown is never credited as 0 and never treated as a peak.
        others = getattr(lease, 'other_residents_mib', None)
        return host.admit_more(float(budget), None if others is None else int(others), cap=bound)

    def note_refusal(self, reason=''):
        """A resource refusal seen by the window (Sem/lease hold, yield): k falls back to 1."""
        host = self._engine_host
        if host is not None and hasattr(host, 'note_refusal'):
            host.note_refusal(reason)

    def _cue_for(self, context_key):
        cue_id = context_key.get('cue_id', context_key.get('cue')) if isinstance(context_key, dict) else context_key
        return self._cues.get(cue_id)

    # -- render ---------------------------------------------------------------------
    def render(self, text, voice, directory, window, *, seed=42, reference_only=False, context_key=None):
        return self._render_one(text, voice, directory, window, seed=seed, reference_only=reference_only,
                                context_key=context_key, job=None)

    def submit(self, text, voice, directory, window, *, seed=42, reference_only=False, context_key=None,
               job_id=None, deadline=None, priority=0):
        """Concurrency API: one line -> RenderJob (job_id defaults to the cue id); see RESULTS.md."""
        if job_id is None and context_key is not None:
            cue = context_key.get('cue_id', context_key.get('cue')) if isinstance(context_key, dict) else context_key
            job_id = 'cue-%s' % (cue,)
        job = self.jobs.RenderJob(job_id, deadline=deadline, priority=priority, context_key=context_key)
        # Direct engine cancel in the CALLER's thread (review P1-2): the host writes the marker synchronously and the
        # child stops by itself (ack or hard-cancel watchdog); no polling hop through a holder thread.
        job.add_canceller(lambda reason, j=job: self._cancel_engine(j, reason))
        self._jobs.add(job)
        return job.start(lambda j: self._render_one(text, voice, directory, window, seed=seed,
                                                    reference_only=reference_only, context_key=context_key, job=j),
                         on_done=self._jobs.discard)

    def _cancel_engine(self, job, reason):
        host = self._engine_host
        if host is not None and hasattr(host, 'cancel'):
            host.cancel(job_id=job.job_id, reason=reason)

    def cancel_job(self, job_id, reason='scheduler_cancel'):
        jobs = [j for j in list(self._jobs) if j.job_id == str(job_id)]
        for job in jobs:
            job.cancel(reason)
        return len(jobs)

    def _render_one(self, text, voice, directory, window, *, seed, reference_only, context_key, job):
        epoch = self._epoch
        # Preparation reports are advisory; the current cue's B decision is authoritative.
        try:
            voice = self._check_reference(voice, context_key)
        except Exception as e:   # noqa: BLE001
            if explicit_fault(e) is not None:
                raise                 # keeps its kind/code/detail (a recoverable refusal keeps its specific reason)
            raise self._error(e) from e
        owned = _Owner(self.owner, self.guard)
        if job is not None:
            job.add_canceller(lambda reason: owned.abort('Movie Vox job cancelled: %s' % reason))
            if job.cancel_requested:
                owned.abort('Movie Vox job cancelled: %s' % job.cancel_reason)
        stop = threading.Event()
        thread = None
        base = None
        staging = None
        temp_root = pathlib.Path(tempfile.gettempdir()).resolve()
        try:
            if epoch != self._epoch:
                raise InterruptedError('Actor renderer closed before admission')
            owned.check()
            if self._engine_host is None and getattr(self.guard, 'allow_real_transport', False) is not True:
                raise self.errors.ActorUnavailable('gpu_admission_hold', 'Runtime has not released the actor transport')
            # The movie pipeline schedules the complete take or retains original speech; it never searches
            # seeds or alternate wording here to manufacture a fit.
            fit_options = {**self._fit, 'enabled': False}
            cache_key = self._key([text, voice, seed, reference_only, self.context, context_key, round(float(window), 6),
                                   fit_options])
            cached = self._cache.get(cache_key)
            if cached and pathlib.Path(cached[0]).is_file() and hashlib.sha256(pathlib.Path(cached[0]).read_bytes()).hexdigest() == cached[1]['sha256']:
                owned.check()
                return cached[0], {**cached[1], 'cache_hit': True, 'window_seconds': window,
                                   'timing_overrun': cached[1]['seconds'] > window}
            admission = type('ProgramAdmission', (), {'scope': SCOPE, 'granted': True})()
            owned.check()
            base = self._class(self._config, owned, transport=self._host(epoch), admission=admission,
                               reference_gate=self._gate, provenance=self._provenance)
            with self._active_lock:
                self._active.add(base)

            def monitor():
                while not stop.wait(.2):
                    try:
                        if epoch != self._epoch:
                            raise InterruptedError('Actor renderer closed during generation')
                        owned.check()
                    except Exception as e:   # noqa: BLE001
                        owned.abort(e)
                        base.close()
                        return
            thread = threading.Thread(target=monitor, name='automatic-actor-priority', daemon=True)
            thread.start()
            staging = pathlib.Path(tempfile.mkdtemp(prefix='movie-actor-inflight-', dir=temp_root))
            cue = self._cue_for(context_key)
            alternatives = self.fit.alternatives_for(cue, text, max_chars=int(base.options['max_text_chars'])) \
                if self._fit['alternatives'] else []
            base_context = {**self.context, 'cue': context_key}

            def attempt(attempt_text, attempt_seed):
                return base._render(attempt_text, voice, staging, window, seed=attempt_seed,
                                    reference_only=reference_only, context_key=base_context, job=job)
            result, fit_record = self.fit.fit(attempt, text=text, seed=seed, window=window, alternatives=alternatives,
                                              opts=fit_options, rate=fit_options['chars_per_second'])
            if epoch != self._epoch:
                raise InterruptedError('Actor renderer closed before returning audio')
            owned.check()
            target = pathlib.Path(directory)
            target.mkdir(parents=True, exist_ok=True)
            wav, receipt = result
            receipt = {**receipt, 'context_key': context_key, 'binding': dict(self.context), 'duration_fit': fit_record,
                       'requested_text': text, 'requested_seed': seed}
            digest = hashlib.sha256(pathlib.Path(wav).read_bytes()).hexdigest()
            if digest != receipt.get('sha256'):
                raise self.errors.ActorUnavailable('output_invalid', 'Staged take does not match its receipt')
            final = target / ('%s-%s.wav' % (pathlib.Path(wav).stem, digest[:16]))

            def publish_exclusive(destination, *, source=None, payload=None, expected_sha=None):
                # Copy onto the destination volume first. link() publishes complete bytes without replacing
                # a name that another render has already returned. A prior receipt sidecar is immutable too.
                fd, temporary_name = tempfile.mkstemp(prefix='.' + destination.name + '-', suffix='.part', dir=target)
                temporary = pathlib.Path(temporary_name)
                try:
                    with os.fdopen(fd, 'wb') as output:
                        if source is not None:
                            with pathlib.Path(source).open('rb') as incoming:
                                shutil.copyfileobj(incoming, output)
                        else:
                            output.write(payload)
                    if expected_sha and hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_sha:
                        raise self.errors.ActorUnavailable('output_invalid', 'Copied take does not match its receipt')
                    owned.check()
                    if epoch != self._epoch:
                        raise InterruptedError('Actor renderer closed before immutable publication')
                    try:
                        os.link(temporary, destination)
                    except FileExistsError:
                        if expected_sha and hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha:
                            raise self.errors.ActorUnavailable('output_invalid', 'Immutable take name contains different bytes')
                    except OSError as error:
                        raise self.errors.ActorUnavailable('output_invalid', 'Immutable take publication failed: %s' % error) from error
                finally:
                    temporary.unlink(missing_ok=True)

            # Once visible, immutable files survive a later cancellation: another caller may already have
            # reused them. Failed calls never return/cache them; unreferenced files remain with the job cache.
            publish_exclusive(final, source=wav, expected_sha=digest)
            owned.check()
            if epoch != self._epoch:
                raise InterruptedError('Actor renderer closed during commit')
            record = final.with_suffix('.json')
            publish_exclusive(record, payload=json.dumps(receipt, ensure_ascii=False, default=str).encode('utf8'))
            owned.check()
            if epoch != self._epoch:
                raise InterruptedError('Actor renderer closed after immutable publication')
            self._cache[cache_key] = (final, receipt)
            return final, receipt
        finally:
            stop.set()
            if base:
                base.close()
                with self._active_lock:
                    self._active.discard(base)
            if thread:
                thread.join(2)
            if staging and staging.resolve().parent == temp_root and staging.name.startswith('movie-actor-inflight-'):
                shutil.rmtree(staging, ignore_errors=True)

    def close(self):
        with self._host_lock:
            self._epoch += 1                    # under the lock: a racing _host(epoch) refuses (review P2-4)
        for job in list(self._jobs):
            job.cancel('renderer_closed')
        with self._active_lock:
            active = list(self._active)
        for base in active:
            base.close()
        with self._host_lock:
            lease, self._lease = self._lease, None
        if lease is not None:
            lease.release()
        release = getattr(self.guard, 'release', None)
        if callable(release):
            release('window_closed')


def actor_engine(config):
    """Always 'vox' (absent or "vox"); anything else is refused (Breeze was removed from the movie path)."""
    return adapter_module('factory').actor_engine(config)


def make_automatic_renderer(config, owner, *, pipeline_context, resource_guard, transport=None, gate_builder=None,
                            engine_host=None):
    """Context is populated by ActorPipeline from its actual source/ref/ASR, not an operator. Vox only."""
    actor_engine(config)
    if gate_builder is None:
        # The app's relay hook always supplies B's packaged gate (relay/beta_stable_gate or beta_caption_gate) with the
        # app's reference_cache; nothing is loaded from outside the app tree.
        raise ValueError('make_automatic_renderer needs the gate_builder supplied by relay.actor_beta_hook')
    bundle = gate_builder(pipeline_context)
    if not isinstance(bundle, dict):
        raise TypeError('B dynamic admission must return a mapping')
    source = pipeline_context.get('source_identity') or {}
    source_id = ((bundle.get('receipt') or {}).get('source_id') or source.get('source_id')
                 or source.get('source_fingerprint') or pipeline_context.get('source_id'))
    return AutomaticVoxRenderer(config, owner, bundle=bundle, resource_guard=resource_guard,
                                job_id=pipeline_context.get('job_id'), source_id=source_id,
                                transport=transport, engine_host=engine_host, pipeline_context=pipeline_context)
