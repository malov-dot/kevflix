"""Movie GPU lease VIEWS for the app process (W10). This is the LEASE VIEW API of INTERFACES.md.

One process holds at most ONE real movie GPU lease (lease v2: one Lease per process, a machine-wide mutex). Two
in-process consumers need it at the same time:
- the Vox engine host: W3's actor_beta_hook -> JobLeaseKeeper/MovieLeaseGuard/VoxEngineHost('lease' owner);
- the resident Demucs worker: resident_demucs.ResidentDemucs.
They never construct a LeaseClient. They take VIEWS from the process LeaseHub, which is fed by W7's holder
(relay.app_runtime.movie_lease):

    hub = LeaseHub(holder)            # holder: W7 HOLDER INTERFACE (INTERFACES.md)
    source = hub.source()             # callable(holder_name: str) -> LeaseView ; source.key (hashable, stable)
    view = source('kevflix movie Vox')

How views and the real lease relate:
- The first open view makes the hub call holder.acquire(holder=name), which returns a real lease-v2 Lease.
- Every view shares that lease; view.spawn() puts the child into its Job.
- Real-lease preemption (lease.on_preempt) ends EVERY open view with that event and runs their callbacks.
- When the last view is released, the hub calls holder.release(lease, reason). That drains synchronously (lease v2:
  returns after every Job process exited) and then frees the machine lock.
- Before a new acquire after a preemption, the old lease is released first, as lease v2 requires.
"""
from __future__ import annotations

import threading
import uuid


class ViewEnded(Exception):
    """Raised by check()/renew()/spawn() once this view has ended (lease preempted, expired or the view released)."""

    def __init__(self, event):
        super().__init__('movie GPU lease view %s ended: %s' % (event.lease_id, event.reason))
        self.event = event


class ViewEvent:
    def __init__(self, reason, detail='', lease_id=''):
        self.reason, self.detail, self.lease_id = str(reason), str(detail), lease_id


class LeaseView:
    """Lease-shaped view of the hub's real lease. Exact API (INTERFACES.md, LEASE VIEW API):

    lease_id: str  budget_mib: int  sem_mode: 'absent'|'running'  job_name / cancel_event_name / kill_event_name: str
    cancel_token: threading.Event (set when this view ended)     preempted: bool     event: ViewEvent | None
    check() -> None            raises ViewEnded once ended; else lease.check() of the real lease (LeasePreempted)
    renew() -> float           seconds left (real lease.renew()); raises ViewEnded / LeasePreempted
    on_preempt(callback)       callback(event) on preemption (not on release); immediately if already preempted
    spawn(args, **popen_kwargs) -> subprocess.Popen   the child is created INSIDE the real lease's Job
                                                       (lease.spawn); raises ViewEnded if this view ended
    adopt(popen) -> popen      own already-running child into the Job (real lease.adopt)
    request_kill() -> bool     REQUEST relayed to the holder (holder.request_kill(lease)); asynchronous,
                               not a confirmation that anything ended
    release(reason='done')     ends THIS view only (idempotent, never raises); the hub releases the real lease
                               (synchronous drain) when no view remains
    """

    def __init__(self, hub, lease, name):
        self._hub, self._lease, self.holder = hub, lease, name
        self.lease_id = 'view-%s-%s' % (getattr(lease, 'lease_id', 'lease'), uuid.uuid4().hex[:8])
        self.budget_mib = getattr(lease, 'budget_mib', None)
        self.sem_mode = getattr(lease, 'sem_mode', None) or hub.sem_mode
        self.job_name = getattr(lease, 'job_name', None)
        self.cancel_event_name = getattr(lease, 'cancel_event_name', None)
        self.kill_event_name = getattr(lease, 'kill_event_name', None)
        self.cancel_token = threading.Event()
        self._event = None
        self._callbacks = []
        self._mutex = threading.Lock()
        # pid -> measured peak MiB (int) of a live GPU resident this view spawned, or None = active, not yet measured
        self._residents = {}

    @property
    def other_residents_mib(self):
        """int | None. The MiB held by GPU residents registered by the hub's OTHER open views (e.g. the resident Demucs
        while Vox admits an engine). None when ANY such live resident has no usable measurement yet (unknown is never
        zero). Otherwise the int sum of their measured peaks; 0 only when there are truly no other live residents.
        Computed by the hub on every read."""
        return self._hub.residents_mib(exclude=self)

    def report_resident(self, pid, peak_mib):
        """Register a live GPU child of THIS view. peak_mib None marks it active-unknown (until a measurement arrives);
        an int keeps the maximum measured peak for that pid."""
        with self._mutex:
            pid = int(pid)
            if peak_mib is None:
                self._residents.setdefault(pid, None)
            else:
                self._residents[pid] = max(int(peak_mib), self._residents.get(pid) or 0)

    def forget_resident(self, pid):
        """The child stopped (call before releasing the view)."""
        with self._mutex:
            self._residents.pop(int(pid), None)

    def residents_mib(self):
        """(sum of measured peaks, any_unknown) of THIS view's live residents; (0, False) once the view ended."""
        with self._mutex:
            if self._event is not None:
                return 0, False
            values = list(self._residents.values())
        return sum(v for v in values if v is not None), any(v is None for v in values)

    @property
    def preempted(self):
        return self._event is not None and self._event.reason != 'released'

    @property
    def event(self):
        return self._event

    def on_preempt(self, callback):
        with self._mutex:
            if self._event is None:
                self._callbacks.append(callback)
                return
            event = self._event
        if event.reason != 'released':
            callback(event)

    def _end(self, reason, detail=''):
        with self._mutex:
            if self._event is not None:
                return []
            self._event = ViewEvent(reason, detail, self.lease_id)
            self.cancel_token.set()
            callbacks = [] if reason == 'released' else list(self._callbacks)
        return [(cb, self._event) for cb in callbacks]

    def check(self):
        if self._event is not None:
            raise ViewEnded(self._event)
        self._lease.check()

    def renew(self):
        self.check()
        return self._lease.renew()

    def wait_preempted(self, timeout=None):
        return self.cancel_token.wait(timeout)

    def spawn(self, args, **popen_kwargs):
        self.check()
        return self._lease.spawn(args, **popen_kwargs)

    def adopt(self, popen):
        self.check()
        return self._lease.adopt(popen)

    def request_kill(self):
        return self._hub.request_kill(self)

    def release(self, reason='done'):
        calls = self._end('released', reason)
        for cb, event in calls:
            cb(event)
        self._hub._view_released(self, str(reason))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release('context-exit')
        return False


class LeaseHub:
    """One per process. Feeds views from W7's holder; see the module doc and INTERFACES.md.

    Locking (no blocking call under the state lock):
      _lock  (RLock)  hub state only: the current lease, its views, detached leases awaiting release, last state.
                      Held briefly; the preemption callback takes ONLY this lock and never waits on _gate.
      _gate  (Lock)   serializes every holder.acquire() and holder.release(). A lease is DETACHED under _lock (the
                      hub forgets it and ends its views), then released, which drains, under _gate, OUTSIDE _lock.
                      view() first drains every detached lease under _gate before it may acquire, so
                      release-before-reacquire holds and at most one real lease ever exists.

    Drain confirmation (W7/V2-CONFIG Holder contract): holder.release(lease, reason) never raises and returns
    {drain_pending, drained, evidence, error, ...}. A detached entry leaves the queue only AFTER release returned:
      * drained                  -> done;
      * drain_pending            -> the holder keeps the lease in its own pending drain; the hub records it and,
                                    on every later view(), calls holder.confirm_drain() under _gate (which re-requests
                                    the Job kill). Until it returns None, view() raises
                                    MovieResourceWait('lease_drain_pending') and NEVER acquires beside it;
      * release raised (contract breach) -> the entry stays queued and is retried on the next view()/close();
      * drain_pending but the holder has no confirm_drain() -> invalid holder API: view() raises
                                    MovieConfigurationError('lease_holder_api_invalid') (fatal), never a success.
    Permanent lease faults from holder.acquire (MovieConfigurationError) propagate unchanged: they are fatal.
    """

    def __init__(self, holder):
        self.holder = holder
        self.sem_mode = getattr(holder, 'sem_mode', None)
        self._lock = threading.RLock()
        self._gate = threading.Lock()
        self._lease = None
        self._views = set()
        self._detached = []              # leases forgotten by the hub but not yet released (drained)
        self._pending_drain = None       # holder.release said drain_pending: no acquire until confirm_drain() clears it
        self._last = {'state': 'idle', 'reason': None}
        self.key = ('kevflix-movie-lease-hub', id(self))

    def source(self):
        source = lambda name: self.view(name)   # noqa: E731
        source.key = self.key
        return source

    # -- detach (under _lock) and drain (under _gate, outside _lock) ----------------------------------------------
    def _detach_locked(self, reason):
        """Under _lock: forget the current lease, end its views, queue it for release. Never blocks."""
        lease, self._lease = self._lease, None
        views, self._views = list(self._views), set()
        if lease is not None:
            self._detached.append((lease, str(reason)))
            if self._last.get('state') != 'preempted':
                self._last = {'state': 'idle', 'reason': str(reason)}
        return views

    def _drain_detached(self):
        """Under _gate (caller holds it), outside _lock: release every detached lease (synchronous drain), then
        re-confirm a pending drain. Returns the pending-drain record (None when nothing blocks a new acquire)."""
        while True:
            with self._lock:
                if not self._detached:
                    break
                lease, reason = self._detached[0]          # peek: the entry is dropped only after release returned
            try:
                result = self.holder.release(lease, reason)
            except Exception as error:  # noqa: BLE001 - contract breach: keep the entry queued, retry later
                with self._lock:
                    self._pending_drain = {'lease_id': getattr(lease, 'lease_id', None), 'reason': reason,
                                           'requeued': True,
                                           'error': {'kind': 'resource', 'code': 'lease_drain_pending',
                                                     'detail': 'holder.release raised %s: %s'
                                                               % (type(error).__name__, error)}}
                return self._pending_drain
            pending = not isinstance(result, dict) or bool(result.get('drain_pending', not result.get('drained')))
            with self._lock:
                if self._detached and self._detached[0][0] is lease:
                    self._detached.pop(0)
                if pending:
                    self._pending_drain = {'lease_id': getattr(lease, 'lease_id', None), 'reason': reason,
                                           'requeued': False,
                                           'evidence': (result or {}).get('evidence') if isinstance(result, dict)
                                           else None,
                                           'error': (result or {}).get('error') if isinstance(result, dict) else None}
                else:
                    self._pending_drain = None
        with self._lock:
            pending = self._pending_drain
        if pending is not None and not pending.get('requeued') and not pending.get('invalid_holder'):
            confirm = getattr(self.holder, 'confirm_drain', None)
            if not callable(confirm):
                # An unconfirmable drain is an invalid holder API: fatal, never treated as a confirmed success.
                with self._lock:
                    self._pending_drain = {**pending, 'invalid_holder': True}
                return self._pending_drain
            still = confirm()
            with self._lock:
                self._pending_drain = None if still is None else {**pending, 'holder_pending': still}
        with self._lock:
            return self._pending_drain

    def _release_outside(self, views, reason):
        for view in views:
            view._end('released', reason)
        with self._gate:
            self._drain_detached()

    # -- views ------------------------------------------------------------------------------------------------------
    def view(self, name):
        with self._gate:
            stale = []
            with self._lock:
                lease = self._lease
                if lease is not None and (getattr(lease, 'preempted', False) or lease.cancel_token.is_set()):
                    stale = self._detach_locked('replaced_after:%s' % getattr(getattr(lease, 'event', None),
                                                                            'reason', 'ended'))
            for view in stale:
                view._end('released', 'lease replaced')
            pending = self._drain_detached()             # release-before-reacquire (lease v2)
            if pending is not None:
                if pending.get('invalid_holder'):
                    from ..movie_faults import MovieConfigurationError
                    raise MovieConfigurationError('lease_holder_api_invalid',
                                                  'the movie lease holder reported an unconfirmed drain but has no '
                                                  'confirm_drain(); no replacement lease is acquired')
                # Never acquire beside an unconfirmed drain: a real hold, surfaced as a resource wait.
                from ..movie_faults import MovieResourceWait
                raise MovieResourceWait('lease_drain_pending',
                                        'the previous movie GPU lease %s is not confirmed drained: %s'
                                        % (pending.get('lease_id'), pending.get('evidence') or pending.get('error')))
            with self._lock:
                lease = self._lease
            if lease is None:
                lease = self.holder.acquire(holder=name)     # one attempt; raises when unavailable
                with self._lock:
                    self._lease = lease
                    self._last = {'state': 'held', 'reason': None, 'lease_id': getattr(lease, 'lease_id', None)}
                lease.on_preempt(lambda event, _lease=lease: self._on_preempt(_lease, event))
            with self._lock:
                if self._lease is not lease:             # preempted and detached meanwhile: report unavailable
                    raise ViewEnded(ViewEvent('preempted', 'lease ended while the view was being created',
                                              getattr(lease, 'lease_id', '')))
                view = LeaseView(self, lease, name)
                self._views.add(view)
                return view

    def _on_preempt(self, lease, event):
        """Lease-reader thread: end every view of THAT lease. Takes only _lock briefly; never waits on _gate."""
        with self._lock:
            if self._lease is not lease:
                return
            views = list(self._views)
            self._last = {'state': 'preempted', 'reason': getattr(event, 'reason', 'preempted'),
                          'detail': getattr(event, 'detail', '')}
        calls = []
        for view in views:
            calls += view._end(getattr(event, 'reason', 'preempted'), getattr(event, 'detail', ''))
        for cb, ev in calls:
            try:
                cb(ev)
            except Exception:  # noqa: BLE001
                pass

    def _view_released(self, view, reason):
        with self._lock:
            self._views.discard(view)
            if self._views or self._lease is None:
                return
            views = self._detach_locked('last_view_released:%s' % reason)
        self._release_outside(views, reason)

    def request_kill(self, view):
        with self._lock:
            lease = self._lease
        if lease is None or view.event is not None:
            return False
        return bool(self.holder.request_kill(lease))

    def residents_mib(self, exclude=None):
        """int sum of the measured peaks of the live residents of every open view except exclude, or None when any
        of them is active without a usable measurement (unknown is never reported as zero)."""
        with self._lock:
            views = [v for v in self._views if v is not exclude]
        total, unknown = 0, False
        for view in views:
            mib, missing = view.residents_mib()
            total += mib
            unknown = unknown or missing
        return None if unknown else int(total)

    def state(self):
        with self._lock:
            views = list(self._views)
            last = dict(self._last, views=len(views), detached=len(self._detached),
                        pending_drain=dict(self._pending_drain) if self._pending_drain else None)
        last['residents_mib'] = self.residents_mib()
        return last

    def close(self, reason='close'):
        with self._lock:
            views = self._detach_locked(reason)
        self._release_outside(views, reason)
