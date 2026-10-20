"""Explicit active-instance registration and stop (fix 3). Stdlib + ctypes only.

<profile>/runtime/instance.json = {pid, created_filetime, image, run_id, stop_event_name, install_root, profile,
started_utc}. Written atomically by the running guard (A) after it holds the single-instance mutex; removed by it on
exit (only when the file still names its own run_id).

stop(): stops ONLY a validated live owned instance: the recorded pid must still have exactly the recorded creation
time and image, and the stop-event name must be a KevFLIX app stop event. A stale entry (process gone or the PID
reused by another program) is reported and removed; no process is signalled. Nothing outside <profile>/runtime is
read or written; media and other folders are never touched.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from . import winjob

NAME = 'instance.json'
LOCK_NAME = 'instance.lock'


class AlreadyRunning(Exception):
    pass


def _path(profile):
    return Path(profile.runtime) / NAME


class RegistryLock:
    """Exclusive byte-range lock on <profile>/runtime/instance.lock around every registry read-modify-write.
    The OS frees it if the holder dies. Waits up to timeout_s."""

    def __init__(self, profile, timeout_s=10.0):
        self.path = Path(profile.runtime) / LOCK_NAME
        self.timeout_s = float(timeout_s)
        self.fd = None

    def __enter__(self):
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_BINARY', 0), 0o644)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                self.fd = fd
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError('the KevFLIX instance registry is busy')
                time.sleep(0.05)

    def __exit__(self, *exc):
        import msvcrt
        fd, self.fd = self.fd, None
        if fd is not None:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            finally:
                os.close(fd)
        return False


def _same_run(a, b):
    return (isinstance(a, dict) and isinstance(b, dict) and a.get('run_id') == b.get('run_id')
            and a.get('pid') == b.get('pid') and a.get('created_filetime') == b.get('created_filetime'))


def _remove_if_same(profile, expected):
    """Under the registry lock: re-read and unlink ONLY if the file still holds exactly the expected run."""
    with RegistryLock(profile):
        current = read(profile)
        if _same_run(current, expected) or (isinstance(current, dict) and current.get('unreadable')
                                            and isinstance(expected, dict) and expected.get('unreadable')):
            try:
                _path(profile).unlink()
                return True
            except OSError:
                return False
        return False


def read(profile):
    path = _path(profile)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'unreadable': True}


def validate(entry):
    """(ok, reason, identity). ok only for the exact recorded live process."""
    if not isinstance(entry, dict) or entry.get('unreadable'):
        return False, 'instance file unreadable', None
    try:
        pid = int(entry['pid'])
    except (KeyError, TypeError, ValueError):
        return False, 'instance file has no pid', None
    if not str(entry.get('stop_event_name', '')).startswith(winjob.STOP_EVENT_PREFIX):
        return False, 'instance file names no KevFLIX stop event', None
    identity = winjob.process_identity(pid)
    if identity is None:
        return False, 'recorded process %d is not running' % pid, None
    if identity.get('created_filetime') != entry.get('created_filetime') or \
            os.path.normcase(str(identity.get('image'))) != os.path.normcase(str(entry.get('image'))):
        return False, 'PID %d now belongs to another process (stale entry)' % pid, None
    return True, 'live', identity


def register(profile, *, install_root):
    """Called by A while holding the instance mutex. Returns the entry. A valid live entry refuses (AlreadyRunning);
    a stale one is replaced (recorded in the new entry)."""
    me = winjob.process_identity(os.getpid())
    if me is None:
        raise OSError('cannot read this process identity')
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    with RegistryLock(profile):
        current = read(profile)
        stale = None
        if current is not None:
            ok, reason, _identity = validate(current)
            if ok and int(current.get('pid', -1)) != os.getpid():
                raise AlreadyRunning('KevFLIX is already running (pid %s, run %s)'
                                     % (current.get('pid'), current.get('run_id')))
            stale = {'entry': current, 'reason': reason}
        entry = {'pid': me['pid'], 'created_filetime': me['created_filetime'], 'image': me['image'], 'run_id': run_id,
                 'stop_event_name': winjob.stop_event_name(run_id), 'install_root': str(install_root),
                 'profile': str(profile.root), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'replaced_stale': stale}
        path = _path(profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.%s.%d.tmp' % (NAME, os.getpid()))
        tmp.write_text(json.dumps(entry, indent=2), encoding='utf-8')
        os.replace(tmp, path)                       # under the lock: the stale entry it replaces was re-validated
    return entry


def unregister(profile, run_id):
    with RegistryLock(profile):
        current = read(profile)
        if isinstance(current, dict) and current.get('run_id') == run_id and current.get('pid') == os.getpid():
            try:
                _path(profile).unlink()
            except OSError:
                pass
            return True
    return False


def stop(profile, *, timeout_s=60.0):
    """Ask the validated running instance to stop and wait. Honest result dict, never raises for normal cases."""
    entry = read(profile)
    if entry is None:
        return {'ok': False, 'result': 'no active instance', 'profile': str(profile.root)}
    ok, reason, identity = validate(entry)
    if not ok:
        # stale: the recorded process is gone or its PID is reused. Signal nothing; remove the entry only if, under
        # the registry lock, it is still exactly the same run (a new registration in between is never removed).
        removed = False
        if not (isinstance(entry, dict) and entry.get('unreadable')):
            removed = _remove_if_same(profile, entry)
        return {'ok': False, 'result': 'no active instance', 'reason': reason, 'stale_entry_removed': removed}
    if not winjob.signal_stop_name(entry['stop_event_name']):
        return {'ok': False, 'result': 'not stopped', 'reason': 'the instance stop event does not exist',
                'pid': entry['pid'], 'run_id': entry.get('run_id')}
    gone = winjob.wait_identity_exit(identity, timeout_s)
    return {'ok': bool(gone), 'result': 'stopped' if gone else 'not stopped',
            'reason': None if gone else 'still running after %.0f s' % timeout_s,
            'pid': entry['pid'], 'run_id': entry.get('run_id')}
