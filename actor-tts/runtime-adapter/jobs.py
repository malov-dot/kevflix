"""Per-line render jobs for the concurrency API (submit -> RenderJob). Stdlib only."""
from __future__ import annotations

import itertools
import threading
import time

_ids = itertools.count(1)


class DeadlineExpired(Exception):
    """A queued line's playback deadline passed before an engine was free, so it was never started.

    Not a resource wait and not an engine fault: the caller keeps that cue's ORIGINAL audio (never silence). The take
    is not rendered later for this deadline; a new submit (for example after a seek) is a new job."""
    reason = 'deadline_missed'
    fallback = 'original_audio'
    fault_kind = 'recoverable'
    fault_code = 'vox_deadline_missed'

    def __init__(self, job_id, deadline, now):
        self.job_id, self.deadline, self.now = job_id, deadline, now
        self.detail = 'line %s missed its deadline by %.3fs before starting' % (job_id, now - deadline)
        super().__init__(self.detail)


def new_job_id(prefix='vox-line'):
    return '%s-%d-%d' % (prefix, int(time.time() * 1000), next(_ids))


class RenderJob:
    """Handle for one submitted line. The render runs on its own thread; the engine queue orders it by `deadline`
    (UNIX time the take is needed for playback; None = after every deadlined job) and then `priority` (higher first).

    result(timeout=None) -> (path, receipt) or raises the render's exception (ActorUnavailable, ActorBusy,
    InterruptedError ...); cancel(reason) stops it (queued: at once; running: <= 0.2 s)."""

    def __init__(self, job_id=None, *, deadline=None, priority=0, context_key=None):
        self.job_id = str(job_id) if job_id is not None else new_job_id()
        self.deadline = None if deadline is None else float(deadline)
        self.priority = int(priority)
        self.context_key = context_key
        self.cancel_reason = None
        self.submitted_at = time.time()
        self._cancel_event = threading.Event()
        self._done = threading.Event()
        self._result = None
        self._error = None
        self._cancellers = []
        self._thread = None

    # -- runner side --------------------------------------------------------------
    def start(self, function, *, on_done=None):
        def run():
            try:
                self._result = function(self)
            except BaseException as error:   # noqa: BLE001 - delivered by result()
                self._error = error
            finally:
                self._done.set()
                if on_done is not None:
                    on_done(self)
        self._thread = threading.Thread(target=run, name='vox-job-%s' % self.job_id, daemon=True)
        self._thread.start()
        return self

    def add_canceller(self, canceller):
        self._cancellers.append(canceller)

    @property
    def cancel_requested(self):
        return self._cancel_event.is_set()

    # -- caller side ----------------------------------------------------------------
    @property
    def state(self):
        if not self._done.is_set():
            return 'cancelling' if self._cancel_event.is_set() else 'active'
        if self._error is None:
            return 'done'
        return 'cancelled' if isinstance(self._error, InterruptedError) else 'failed'

    def done(self):
        return self._done.is_set()

    def wait(self, timeout=None):
        return self._done.wait(timeout)

    def result(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError('render job %s still running' % self.job_id)
        if self._error is not None:
            raise self._error
        return self._result

    def exception(self, timeout=None):
        self.wait(timeout)
        return self._error

    def cancel(self, reason='scheduler_cancel'):
        if self._done.is_set():
            return
        if self.cancel_reason is None:
            self.cancel_reason = str(reason)
        self._cancel_event.set()
        for canceller in list(self._cancellers):
            try:
                canceller(self.cancel_reason)
            except Exception:   # noqa: BLE001
                pass
