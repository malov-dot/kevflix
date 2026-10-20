"""HOLDER INTERFACE (INTERFACES.md "HOLDER INTERFACE") for the movie GPU lease, used INSIDE the server process by
W10's LeaseHub (relay/movie_pipeline/leaseview.py). Stdlib + the bundled lease v2 client.

    holder(config) -> Holder                  # process singleton
    Holder.sem_mode: str                      # 'absent' (owner: Sem is off for movies) or 'running'
    Holder.acquire(*, holder: str) -> Lease   # LeaseClient(sem_mode, reserve_mib=3072, grace_s=0.15)
                                              #   .acquire(budget_mib, ttl_s=30, wait_s=0, holder=holder); ONE attempt;
                                              # budget_mib 'auto' (default): fresh free VRAM - reserve - admission
                                              # margin, capped by budget_cap_mib; an explicit int is honoured
                                              # raises MovieResourceWait (real lock/VRAM/Sem/drain hold) or
                                              # MovieConfigurationError (permanent lease fault) when not granted;
                                              # keeps the lease renewed (ttl) while held
    Holder.release(lease, reason) -> dict     # lease.release(reason): synchronous drain; never raises; returns
                                              # {drain_pending, drained, error (fault record|None), ...};
                                              # drain_pending is True unless the drain is CONFIRMED
    Holder.confirm_drain() -> dict | None     # re-check an unconfirmed drain (also done by every acquire)
    Holder.request_kill(lease) -> bool        # lease.request_kill(); asynchronous
    Holder.other_residents_mib(exclude=None)  # MiB of GPU residents reported to the HOLDER (not via a view);
                                              # the view adds its hub's residents (LeaseView.other_residents_mib)

Placement: the lease (and its kill-on-close Job) belongs to the lease-v2 WATCHER process, a child of the server.
Only GPU children created with lease.spawn() (resident Demucs, Vox engines) are in that Job; the server never is.
If the watcher dies the OS ends the Job (the GPU children); if the server dies the watcher ends and drains it.
Every PreemptEvent is recorded (Holder.events()). The holder never opens a Sem file itself.

Budget (v2 contract): gpu_lease.budget_mib 'auto' = one fresh read-only free-VRAM sample (the lease client's
available_vram(): NVML, else nvidia-smi) - reserve_mib (>= 3072) - admission_margin_mib (default 512), then
min(.., budget_cap_mib) when an integer cap is set. An unreadable sample or a budget <= 0 is a resource wait
(MovieResourceWait), never a guess. The frozen client still validates baseline free - reserve >= budget.

Drain (v2 contract): a lease stays an owned entry until its drain is CONFIRMED by evidence: lease.release returned,
the client did not flag release_drain_pending, and the named lease Job has no active process (or no longer exists;
winjob.named_job_active). Otherwise the entry moves to a pending-drain slot and every acquire refuses with a
resource wait until confirm_drain() (run first by each acquire) sees that evidence; each re-check also asks the
watcher to end the Job's processes (request_kill), so a failed drain is retried, never wedged and never assumed.

Faults: permanent lease failures (source_drift, mode_unavailable, bad_config, watcher start/answer failure, an
invalid budget or client pin, any non-lease error) are MovieConfigurationError; only an actual lock, VRAM, Sem or
drain hold is MovieResourceWait (relay.movie_faults).
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from pathlib import Path

from ..movie_faults import MovieConfigurationError, MovieFault, MovieResourceWait
from . import leasehold, winjob
from .paths import INSTALL_ROOT

GUARD_SOURCE = Path(__file__).resolve().parent / 'guard-source'   # contains relay/<5 pinned guard files>
TTL_S, RENEW_S = 30.0, 5.0
ADMISSION_MARGIN_MIB = 512
# LeaseUnavailable reasons that retrying the unchanged install/config cannot repair (watcher_failed/_timeout: our
# own watcher did not start or answer). Every other LeaseUnavailable reason is a real lock/VRAM/Sem hold.
PERMANENT_REASONS = frozenset({'source_drift', 'mode_unavailable', 'bad_config', 'watcher_failed', 'watcher_timeout'})
BREEZE_PORT = 17863
log = logging.getLogger('kevflix.movie_lease')
_HOLDER = None
_HOLDER_LOCK = threading.Lock()


def _data(config):
    return config.data if hasattr(config, 'data') else dict(config)


def holder(config):
    """Process singleton built from the config's gpu_lease block (sem_mode explicit; reserve >= 3072 MiB)."""
    global _HOLDER
    with _HOLDER_LOCK:
        if _HOLDER is None:
            _HOLDER = Holder(config)
        return _HOLDER


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('%s must be a positive integer MiB' % name)
    return value


def budget_policy(cfg):
    """(budget, cap, margin) from a gpu_lease block: budget 'auto' or a positive int; cap None or a positive int."""
    budget = cfg.get('budget_mib', 'auto')
    if budget in (None, '', 'auto'):
        budget = 'auto'
    elif isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError('gpu_lease.budget_mib must be "auto" or a positive integer MiB')
    cap = cfg.get('budget_cap_mib')
    cap = None if cap in (None, '') else _positive_int(cap, 'gpu_lease.budget_cap_mib')
    margin = cfg.get('admission_margin_mib', ADMISSION_MARGIN_MIB)
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError('gpu_lease.admission_margin_mib must be a non-negative integer MiB')
    return budget, cap, margin


def acquire_fault(lc, error):
    """Explicit movie fault for a lease acquire failure (see PERMANENT_REASONS)."""
    if isinstance(error, MovieFault):
        return error
    reason = getattr(error, 'reason', None)
    unavailable = getattr(lc, 'LeaseUnavailable', ())
    mode = getattr(lc, 'LeaseModeUnavailable', ())
    if (mode and isinstance(error, mode)) or (unavailable and isinstance(error, unavailable)
                                             and reason in PERMANENT_REASONS):
        return MovieConfigurationError('lease_%s' % (reason or 'mode_unavailable'), str(error))
    if unavailable and isinstance(error, unavailable):
        return MovieResourceWait('lease_%s' % (reason or 'unavailable'), str(error))
    if isinstance(error, (ValueError, TypeError)):
        return MovieConfigurationError('lease_invalid_config', '%s: %s' % (type(error).__name__, error))
    if isinstance(error, OSError):
        return MovieConfigurationError('lease_watcher_start_failed', '%s: %s' % (type(error).__name__, error))
    return MovieConfigurationError('lease_acquire_failed', '%s: %s' % (type(error).__name__, error))


def _event_dict(event):
    if event is None:
        return None
    if dataclasses.is_dataclass(event):
        return dataclasses.asdict(event)
    return {k: getattr(event, k, None) for k in ('reason', 'detail', 'lease_id', 'detected_at', 'received_at')}


class Holder:
    def __init__(self, config, *, lc=None, install_root=INSTALL_ROOT):
        data = _data(config)
        cfg = dict(data.get('gpu_lease') or {})
        self.sem_mode = cfg.get('sem_mode', 'absent')
        if self.sem_mode not in ('running', 'absent'):
            raise MovieConfigurationError('lease_invalid_config', 'gpu_lease.sem_mode must be "running" or "absent"')
        status_file = cfg.get('status_file') or None
        if self.sem_mode == 'running' and not status_file:
            raise MovieConfigurationError('lease_invalid_config', 'gpu_lease.sem_mode "running" needs gpu_lease.status_file')
        try:
            self.reserve_mib = max(3072, int(cfg.get('reserve_mib', 3072)))
            self.grace_s = float(cfg.get('grace_s', 0.15))
            self.budget_mib, self.budget_cap_mib, self.admission_margin_mib = budget_policy(cfg)
        except (TypeError, ValueError) as error:
            raise MovieConfigurationError('lease_invalid_config', str(error)) from error
        self.last_budget = None          # the latest computed budget record (state/receipt)
        state = Path(data.get('state_dir') or '.') / 'movie-lease'
        state.mkdir(parents=True, exist_ok=True)
        try:
            self.lc = lc or leasehold.load_client()
        except leasehold.LeasePinRefused as error:
            raise MovieConfigurationError('lease_client_pin_refused', str(error)) from error
        # The watcher loads the frozen guard closure in every sem_mode. It comes from the PRIVATE guard-package root
        # (relay/app_runtime/guard-source/relay/: the five pinned files, byte-identical to the client's GUARD_CLOSURE),
        # never from the app's own relay/, so app version bumps or module changes cannot make the lease refuse.
        try:
            self.client = self.lc.LeaseClient(status_file=status_file or self.lc.DEFAULT_STATUS, sem_mode=self.sem_mode,
                                              reserve_mib=self.reserve_mib, state_dir=str(state),
                                              grace_s=self.grace_s, guard_packages=[str(GUARD_SOURCE)])
        except (TypeError, ValueError) as error:
            raise MovieConfigurationError('lease_invalid_config', str(error)) from error
        self._lock = threading.Lock()
        self._held = {}                  # id(lease) -> {'lease', 'stop': Event, 'thread', 'budget'}
        self._pending_drain = None       # the owned entry of a released lease whose drain is NOT yet confirmed
        self._events = []
        self._residents = {}             # name -> MiB
        self.telemetry = {'acquires': 0, 'grants': 0, 'refusals': 0, 'releases': 0, 'renew_failures': 0,
                          'drain_pending': 0, 'drain_confirmed_late': 0, 'post_grant_failures': 0}

    # -- interface ---------------------------------------------------------------------------------------------------
    def acquire(self, *, holder):
        """ONE attempt (wait_s=0). Raises MovieResourceWait (lock/VRAM/Sem/pending drain) or MovieConfigurationError
        (permanent lease fault) when no lease is granted now."""
        if self._pending_drain is not None and self.confirm_drain() is not None:
            self.telemetry['refusals'] += 1
            pending = self._pending_drain or {}
            raise MovieResourceWait('lease_drain_pending', 'the previous movie GPU lease %s is not confirmed drained '
                                    '(%s); no new lease until it is' % (pending.get('lease_id'), pending.get('evidence')))
        with self._lock:
            if self._held:
                raise MovieResourceWait('lease_already_held', 'this process already holds the movie GPU lease; '
                                        'release it before re-acquiring')
            self.telemetry['acquires'] += 1
        if winjob.tcp_listeners().get(BREEZE_PORT):
            self.telemetry['refusals'] += 1
            raise MovieResourceWait('breeze_listener', 'a Breeze listener exists on port %d; KevFLIX never runs GPU '
                                    'work beside it' % BREEZE_PORT)
        try:
            budget = self._budget_now()
            lease = self.client.acquire(budget['budget_mib'], ttl_s=TTL_S, wait_s=0.0, holder=str(holder))
        except Exception as error:  # noqa: BLE001 -- converted to an explicit fault
            self.telemetry['refusals'] += 1
            fault = acquire_fault(self.lc, error)
            if fault is error:
                raise
            raise fault from error
        stop = threading.Event()
        with self._lock:
            self._held[id(lease)] = {'lease': lease, 'stop': stop, 'thread': None, 'budget': budget}
            self.telemetry['grants'] += 1
        try:
            lease.on_preempt(self._on_preempt)
            thread = threading.Thread(target=self._renew_loop, args=(lease, stop), daemon=True,
                                      name='kevflix-lease-renew')
            with self._lock:
                self._held[id(lease)]['thread'] = thread
            thread.start()
        except BaseException as error:
            # A granted lease is never left behind: release + drain (pending-drain entry if unconfirmed), then raise
            # the explicit fault (acquire_fault) so callers get a fault record, never the raw exception.
            self.telemetry['post_grant_failures'] += 1
            stop.set()
            record = self.release(lease, 'post_grant_failure: %s' % type(error).__name__)
            log.error('movie GPU lease post-grant step failed (%s); released, drained=%s', error, record['drained'])
            if not isinstance(error, Exception):
                raise                                   # KeyboardInterrupt/SystemExit pass through after cleanup
            fault = acquire_fault(self.lc, error)
            if fault is error:
                raise
            raise fault from error
        log.info('movie GPU lease granted (%s, %d MiB, %s, sem_mode %s)', getattr(lease, 'lease_id', '?'),
                 budget['budget_mib'], budget['mode'], self.sem_mode)
        return lease

    def _budget_now(self):
        """The budget for this acquire, recorded in self.last_budget. 'auto' reads free VRAM once, read-only."""
        record = {'mode': 'fixed' if isinstance(self.budget_mib, int) else 'auto', 'reserve_mib': self.reserve_mib,
                  'admission_margin_mib': self.admission_margin_mib, 'budget_cap_mib': self.budget_cap_mib,
                  'free_mib': None, 'total_mib': None, 'budget_mib': None, 'at': time.time()}
        if record['mode'] == 'fixed':
            record['budget_mib'] = self.budget_mib
            self.last_budget = record
            return record
        try:
            sample = self.client.available_vram()
            free, total = int(sample.free_mib), int(sample.total_mib)
        except Exception as error:  # noqa: BLE001 -- unreadable free VRAM is a resource wait, never a guess
            record['error'] = '%s: %s' % (type(error).__name__, error)
            self.last_budget = record
            raise MovieResourceWait('lease_gpu_probe_failed',
                                    'free VRAM is unreadable for the auto lease budget: %s' % record['error'])
        budget = free - self.reserve_mib - self.admission_margin_mib
        if self.budget_cap_mib is not None:
            budget = min(budget, self.budget_cap_mib)
        record.update(free_mib=free, total_mib=total, budget_mib=budget)
        self.last_budget = record
        if budget <= 0:
            raise MovieResourceWait('lease_insufficient_vram', 'free %d MiB - reserve %d MiB - margin %d MiB leaves '
                                    'no movie budget' % (free, self.reserve_mib, self.admission_margin_mib))
        return record

    def release(self, lease, reason):
        """Synchronous drain, then the machine lock is free. Never raises. Returns {'drain_pending', 'drained',
        'error': fault record or None, ...}; drain_pending is True unless the drain is CONFIRMED (module docstring).
        The lease stays an owned entry (held, then pending-drain) until that confirmation."""
        with self._lock:
            entry = self._held.get(id(lease))          # kept until the drain outcome is known
        if entry is not None:
            entry['stop'].set()
        began = time.time()
        problems = []
        try:
            lease.release(reason)
        except Exception as error:  # noqa: BLE001 -- contract: release never raises; recorded, drain unconfirmed
            problems.append('release raised %s: %s' % (type(error).__name__, error))
        telemetry = {}
        try:
            telemetry = dict(lease.telemetry)
        except Exception as error:  # noqa: BLE001
            problems.append('telemetry unreadable %s: %s' % (type(error).__name__, error))
        drained, evidence = self._drain_evidence(lease, telemetry, problems)
        self.telemetry['releases'] += 1
        record = {'reason': str(reason), 'seconds': round(time.time() - began, 3),
                  'budget': entry.get('budget') if entry is not None else None,
                  'drain_pending': not drained, 'drained': drained, 'evidence': evidence, 'error': None,
                  'final_event': _event_dict(getattr(lease, 'event', None)), 'telemetry': telemetry}
        with self._lock:
            self._held.pop(id(lease), None)
            if not drained:
                record['error'] = MovieResourceWait('lease_drain_pending', evidence).record()
                self._pending_drain = {'lease': lease, 'lease_id': getattr(lease, 'lease_id', None),
                                       'job_name': getattr(lease, 'job_name', None), 'reason': str(reason),
                                       'since': time.time(), 'checks': 0, 'evidence': evidence}
                self.telemetry['drain_pending'] += 1
        if not drained:
            log.error('movie GPU lease drain NOT confirmed (%s); no new lease until it is', evidence)
        return record

    def confirm_drain(self):
        """Re-check an unconfirmed drain. Returns None when no drain is pending (now or any more), else the pending
        record. Each unconfirmed check asks the watcher to end the lease Job's processes again (request_kill)."""
        with self._lock:
            pending = self._pending_drain
        if pending is None:
            return None
        lease = pending['lease']
        drained, evidence = self._drain_evidence(lease, None, [])
        with self._lock:
            if self._pending_drain is not pending:
                return self._pending_drain and {k: v for k, v in self._pending_drain.items() if k != 'lease'}
            pending['checks'] += 1
            pending['evidence'] = evidence
            if drained:
                self._pending_drain = None
                self.telemetry['drain_confirmed_late'] += 1
        if drained:
            log.info('movie GPU lease %s drain confirmed after %d check(s)', pending['lease_id'], pending['checks'])
            return None
        self.request_kill(lease)
        return {k: v for k, v in pending.items() if k != 'lease'}

    def _drain_evidence(self, lease, telemetry, problems):
        """(confirmed, evidence text): confirmed only when the named lease Job has no active process (or is gone)
        and, at release time, the client did not flag release_drain_pending and nothing failed."""
        if telemetry is not None and telemetry.get('release_drain_pending'):
            problems = problems + ['client flagged release_drain_pending (watcher did not exit in time)']
        job_name = getattr(lease, 'job_name', None)
        active = None
        if job_name:
            try:
                active = winjob.named_job_active(job_name)
            except Exception as error:  # noqa: BLE001 -- unreadable evidence is not a drain
                problems = problems + ['lease Job unreadable %s: %s' % (type(error).__name__, error)]
        else:
            problems = problems + ['lease has no job_name']
        if active is None:
            problems = problems + ['lease Job active-process count unknown']
        elif active:
            problems = problems + ['lease Job still has %d active process(es)' % active]
        if telemetry is None and active == 0:
            return True, 'lease Job has no active process (re-check)'
        if problems:
            return False, '; '.join(problems)
        return True, 'lease released; lease Job has no active process'

    def request_kill(self, lease):
        try:
            return bool(lease.request_kill())
        except Exception:  # noqa: BLE001
            return False

    def other_residents_mib(self, exclude=None):
        with self._lock:
            return int(sum(v for k, v in self._residents.items() if k != exclude))

    def report_resident(self, name, mib):
        with self._lock:
            self._residents[str(name)] = max(int(mib), self._residents.get(str(name), 0))

    def forget_resident(self, name):
        with self._lock:
            self._residents.pop(str(name), None)

    def events(self):
        with self._lock:
            return list(self._events)

    def state(self):
        with self._lock:
            return {'sem_mode': self.sem_mode, 'budget_mib': self.budget_mib, 'budget_cap_mib': self.budget_cap_mib,
                    'admission_margin_mib': self.admission_margin_mib, 'last_budget': self.last_budget,
                    'reserve_mib': self.reserve_mib,
                    'pending_drain': None if self._pending_drain is None else
                    {k: v for k, v in self._pending_drain.items() if k != 'lease'},
                    'held': len(self._held), 'telemetry': dict(self.telemetry), 'last_events': self._events[-5:]}

    # -- internals ---------------------------------------------------------------------------------------------------
    def _on_preempt(self, event):
        # holder thread; set flags only. The watcher already ends the GPU children after grace_s.
        record = {**(_event_dict(event) or {}), 'recorded_at': time.time()}
        with self._lock:
            self._events = self._events[-99:] + [record]
        log.warning('movie GPU lease preempted: %s (%s)', record.get('reason'), record.get('detail'))

    def _renew_loop(self, lease, stop):
        while not stop.wait(RENEW_S):
            try:
                lease.renew()
            except Exception as error:  # noqa: BLE001 -- LeasePreempted: the views end through on_preempt
                self.telemetry['renew_failures'] += 1
                log.warning('movie GPU lease renew stopped: %s', error)
                return
