"""Bounded tolerant read of Sem's status.json (candidate guard-status-read-v1; staged, not live).

Why: Sem writes C:/Sem-Data/profile-v1/local-coding-inbox/status.json every 500 ms as
writeFileSync(status.json.tmp) + renameSync(tmp -> status.json) (sem4-local-coding-inbox.cjs). On Windows a reader
that opens the file while that replace is in progress can get ERROR_ACCESS_DENIED / ERROR_SHARING_VIOLATION, which
Python raises as PermissionError [Errno 13]. The frozen guard treats any exception as fail-closed, so one such
collision aborted the 20261017T143131Z HLS proof at 120.4 s.

What this changes: ONLY a transient lock is retried, a bounded number of times, inside one guard check.
Each attempt is a complete call of the FROZEN check (relay/demucs_bed.py sem_idle_pid, sha256 49ec437a...), which is
neither copied nor modified. So freshness (status age <= 15 s), clock skew (age >= -2 s), serverReady, busy,
codexVoiceActive, voice.active, voice.closing and the returned PID are evaluated exactly as before, on freshly read
bytes. The caller still compares the PID for continuity.

What it never does:
* treat an unreadable file as idle. Nothing is cached; a PID is returned only by a successful frozen check;
* retry anything other than a lock. Not-idle, stale, skewed, missing file, malformed JSON and missing keys are raised
  unchanged on the attempt that saw them (also when an earlier attempt was locked);
* exceed the bound. No attempt starts more than BUDGET_SECONDS (300 ms) after the first one. When every attempt in
  the window is locked it raises StatusUnreadable, a PermissionError subclass, so every existing fail-closed handler
  (GuardedRun._sem, the HLS wrapper loop, demucs Guard._run's OSError -> ResourceYield) classifies it as before;
* change the 200 ms cadence. The guard loops keep SEM_INTERVAL = 0.2 s; a lock adds at most ~300 ms to that one check.
"""
import errno
import time

RETRY_DELAYS = (0.010, 0.020, 0.040, 0.080, 0.120)  # up to 5 retries after the first attempt; 270 ms of waiting
BUDGET_SECONDS = 0.300                              # hard ceiling; constructor refuses anything larger
TRANSIENT_WINERRORS = frozenset({5, 32, 33})        # ACCESS_DENIED, SHARING_VIOLATION, LOCK_VIOLATION


class StatusUnreadable(PermissionError):
    """status.json stayed locked for the whole bounded window: fail closed (never idle)."""


def is_transient_lock(error):
    """True only for an open/read lock. A missing file is not a lock: Sem stopped or the path is wrong."""
    if isinstance(error, (FileNotFoundError, NotADirectoryError)):
        return False
    if isinstance(error, PermissionError):
        return True
    return isinstance(error, OSError) and getattr(error, 'winerror', None) in TRANSIENT_WINERRORS


class TolerantSemCheck:
    """Callable with the frozen signature: check(status_file) -> Sem PID, or raise (fail closed)."""

    def __init__(self, check, *, delays=RETRY_DELAYS, budget=BUDGET_SECONDS, clock=time.monotonic, sleep=time.sleep):
        delays = tuple(float(d) for d in delays)
        if not 0 < budget <= BUDGET_SECONDS + 1e-9:
            raise ValueError('retry budget must be in (0, %.3f] s' % BUDGET_SECONDS)
        if any(d <= 0 for d in delays) or sum(delays) > budget + 1e-9:
            raise ValueError('retry delays must be positive and fit inside the budget')
        self.check, self.delays, self.budget, self.clock, self.sleep = check, delays, float(budget), clock, sleep
        self.telemetry = {'calls': 0, 'attempts': 0, 'transient_errors': 0, 'recovered_calls': 0,
                          'max_recovery_ms': 0.0, 'unreadable_aborts': 0, 'last_transient': None}

    def __call__(self, status_file):
        t = self.telemetry
        t['calls'] += 1
        started, locked, last = self.clock(), 0, None
        for delay in (0.0,) + self.delays:
            if delay:
                if self.clock() - started + delay > self.budget:
                    break
                self.sleep(delay)
            t['attempts'] += 1
            try:
                pid = self.check(status_file)
            except Exception as error:  # BaseException (KeyboardInterrupt, SystemExit) is never swallowed
                if not is_transient_lock(error):
                    raise  # unchanged frozen verdict: not idle / stale / skew / missing / malformed
                locked, last = locked + 1, error
                t['transient_errors'] += 1
                t['last_transient'] = '%s: %s' % (type(error).__name__, getattr(error, 'winerror', None) or error.errno)
                continue
            if locked:
                t['recovered_calls'] += 1
                t['max_recovery_ms'] = max(t['max_recovery_ms'], round((self.clock() - started) * 1000.0, 1))
            return pid
        t['unreadable_aborts'] += 1
        raise StatusUnreadable(errno.EACCES, 'Sem status unreadable: %d consecutive locked reads in %.0f ms (bound %.0f ms); last %s: %s'
                               % (locked, (self.clock() - started) * 1000.0, self.budget * 1000.0, type(last).__name__, last),
                               str(status_file))


def wrap_frozen_guard(guard_module, **options):
    """TolerantSemCheck around the frozen module's own sem_idle_pid (load it with the caller's sha-pinned loader)."""
    return TolerantSemCheck(guard_module.sem_idle_pid, **options)
