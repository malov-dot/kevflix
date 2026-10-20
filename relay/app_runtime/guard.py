"""KevFLIX app guard (process A) for the NORMAL installed app (fix 2 startup side, fix 3). Stdlib + ctypes only.

Started by scripts/run_worker.py (Windows startup, the desktop launcher and the installer all use it):
  1. integrity: the installed app-manifest.json must verify (app_runtime/integrity.py); otherwise nothing starts
  2. profile: the persistent user profile (app_runtime/paths.py); a legacy in-install config is migrated by COPY
  3. single instance: named mutex + the active-instance registry (app_runtime/instances.py), one stop event per run
  3b. GPU-first migration of the app-owned CPU-only movie keys, then the startup PREFLIGHT (app_runtime/preflight.py):
     a root/model/manifest path under a protected Sem root, or (when jobs can run) a missing model, runtime manifest
     or CUDA backend setting, refuses the start with a MovieConfigurationError fault record (exit 78, not retried)
     before the server or any job exists. Recorded in <profile>/runtime/startup-preflight.json and the receipt.
  4. S = the app server (relay.server via scripts/serve.py) with the USER's settings as they are (Plex, folders, UI
     settings, LAN bind/port, Windows-startup preference untouched): created suspended -> guard Job -> resumed.
  5. stop (scripts/stop_worker.py / Stop-KevFLIX.cmd): the registered stop event (or <profile>/runtime/STOP-<run>)
     -> the guard Job ends S and its descendants -> the registry entry is removed. No other stop path exists.

Job placement (Codex / W10 INTERFACES "Placement"): the server is ONLY in this guard (app lifetime) Job. The movie
GPU lease v2 is held INSIDE the server by relay.app_runtime.movie_lease (W10's LeaseHub feeds the views); only the
GPU children (resident Demucs, Vox engines) are created with lease.spawn() inside the lease's own kill-on-close Job.
A preemption, request_kill or release() drain therefore ends only those GPU children, never the server, which
keeps serving cached, published and original playback. The guard Job is terminated only when the app stops.
The guard Job is kill-on-close: if A dies, the OS ends S (and S's lease watcher then drains the lease Job).
No service lifetime bound, no forced loopback or port, no cleared settings. A never opens a Sem file and never
starts Breeze; no child command line carries a Sem path.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from . import instances, integrity, winjob
from .paths import INSTALL_ROOT, Profile

LOOP_S = 0.25
EXIT = {'stopped': 0, 'failed': 2, 'integrity': 70, 'refused': 75, 'configuration': 78}
log = logging.getLogger('kevflix.guard')


def utc(ts=None):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)
    return path


def child_python(install_root):
    """The app's own environment (.venv, created by Setup-Worker), else the bundled portable runtime."""
    for candidate in (install_root / '.venv' / 'Scripts' / 'python.exe', install_root / 'runtime' / 'python' / 'python.exe'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('no KevFLIX Python found (run Setup-Worker.ps1)')


# Exact denylist: unsafe trial/injection switches, the retired lease-v1 control variable, and interpreter path
# overrides. Every other variable passes through, including Config's supported overrides (MOVIEDUB_API_TOKEN,
# MOVIEDUB_NAS_TOKEN, MOVIEDUB_STREAM_SECRET, MOVIEDUB_WEBHOOK_SECRET, MOVIEDUB_PLEX_TOKEN and the updates
# token_env, default MOVIEDUB_GITHUB_TOKEN). Values are never printed or logged.
ENV_DENYLIST = frozenset({'SEM_TRIAL_INJECT', 'KEVFLIX_GPU_LEASE_STATE', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP',
                          'PYTHONINSPECT', 'PYTHONUSERBASE'})


def child_env(install_root, profile):
    env = {k: v for k, v in os.environ.items() if k.upper() not in ENV_DENYLIST}
    tools = install_root / 'tools' / 'ffmpeg-9.0.2-full_build-shared' / 'bin'
    torch_lib = install_root / '.venv' / 'Lib' / 'site-packages' / 'torch' / 'lib'
    extra = [str(p) for p in (tools, torch_lib) if p.is_dir()]
    if extra:
        env['PATH'] = os.pathsep.join(extra) + os.pathsep + env.get('PATH', '')
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', TOKENIZERS_PARALLELISM='false',
               PYANNOTE_METRICS_ENABLED='0', KEVFLIX_PROFILE=str(profile.root), KEVFLIX_INSTALL_ROOT=str(install_root))
    return env


class AppGuard:
    def __init__(self, *, install_root=INSTALL_ROOT, profile=None, setup_only=False, clock=time.monotonic,
                 sleep=time.sleep):
        self.setup_only = bool(setup_only)            # passes --setup-only to the server (pairing/settings only)
        self.install_root = Path(install_root)
        self.profile = profile or Profile(install_root=self.install_root)
        self.clock, self.sleep = clock, sleep
        self.mutex = self.stop_event = self.job = self.entry = self.server = None
        self.install_lock = None
        self.acl_from = None
        self.files = []
        self.receipt = {'kind': 'kevflix app guard receipt', 'started_utc': utc(), 'guard_pid': os.getpid(),
                        'install_root': str(self.install_root), 'profile': str(self.profile.root), 'outcome': None,
                        'exit': None}

    def _finish(self, outcome, code, **extra):
        self.receipt.update(outcome=outcome, exit=code, **extra)
        return code

    def _stop_requested(self):
        return (self.stop_event is not None and self.stop_event.is_set()) or \
            (self.profile.runtime / ('STOP-%s' % self.entry['run_id'])).exists()

    def _hold_install_lock(self):
        """<install root>/state/install-guard.lock held for A's lifetime: KevFLIX Setup opens it exclusively, so files
        are never replaced under a running app (an update waits for this process to exit). It is NOT the server's
        own state/worker.lock (a migrated profile may keep state/ in the install folder)."""
        import msvcrt
        path = self.install_root / 'state' / 'install-guard.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open('a+b')
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            stream.close()
            raise FileExistsError('KevFLIX Setup is running or another KevFLIX instance holds the install lock')
        self.install_lock = stream

    def _start_server(self, python):
        out = open(self.profile.logs / 'server.stdout.log', 'ab')
        err = open(self.profile.logs / 'server.stderr.log', 'ab')
        self.files += [out, err]
        argv = [python, '-I', '-B', self.install_root / 'scripts' / 'serve.py', '--config', self.profile.config]
        if self.setup_only:
            argv.append('--setup-only')
        self.server = winjob.launch_suspended(argv, cwd=self.install_root, env=child_env(self.install_root, self.profile),
                                              stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                              assign=(self.job.assign,))        # guard Job ONLY, never the lease Job
        self.receipt['server'] = {'pid': self.server.pid, 'identity': winjob.process_identity(self.server.pid),
                                  'jobs': ['guard (app lifetime)']}

    def _end_server(self):
        record = {'job_terminated': self.job.terminate() if self.job else None}
        if self.server is not None:
            try:
                record['server_exit'] = self.server.wait(15)
            except subprocess.TimeoutExpired:
                record['server_exit'] = None
        record['guard_job_active_processes'] = self.job.active_processes() if self.job else None
        self.receipt['server_end'] = record
        return record

    def run(self):
        manifest, problems = integrity.verify(self.install_root)
        if problems:
            self.receipt['integrity_problems'] = problems[:60]
            log.error('KevFLIX integrity check failed; nothing was started: %s', problems[:5])
            return self._finish('integrity_refused', EXIT['integrity'])
        self.receipt['manifest'] = {'version': manifest.get('version'), 'revision': manifest.get('revision'),
                                    'files': len(manifest.get('files') or {})}
        self.profile.ensure()
        try:
            from ..settings import preserve_private_acl
        except Exception:  # noqa: BLE001
            preserve_private_acl = None
        self.acl_from = preserve_private_acl
        self.receipt['migration'] = self.profile.migrate_legacy(acl_from=preserve_private_acl)
        if not self.profile.config.is_file():
            return self._finish('refused_no_settings', EXIT['refused'],
                                reason='no settings file at %s (run KevFLIX Setup)' % self.profile.config)
        try:
            self.mutex = winjob.InstanceMutex()
        except FileExistsError as error:
            return self._finish('refused_already_running', EXIT['refused'], reason=str(error))
        try:
            self._hold_install_lock()
            self.entry = instances.register(self.profile, install_root=self.install_root)
        except (instances.AlreadyRunning, FileExistsError) as error:
            self.mutex.close()
            if self.install_lock is not None:
                self.install_lock.close()
            return self._finish('refused_already_running', EXIT['refused'], reason=str(error))
        self.receipt['run_id'] = self.entry['run_id']
        try:
            return self._run_registered()
        finally:
            self._close()

    def _preflight(self):
        """GPU-first migration, then the preflight. Returns the server python, or None after a refused start."""
        from ..movie_faults import MovieConfigurationError, fault_record
        from . import preflight
        try:
            self.receipt['gpu_first_migration'] = preflight.migrate_gpu_first(self.profile, acl_from=self.acl_from)
        except (OSError, ValueError) as error:       # not fatal: Config converts in memory; invalid JSON fails below
            self.receipt['gpu_first_migration'] = {'changes': [], 'error': '%s: %s' % (type(error).__name__, error)}
        try:
            try:
                python = child_python(self.install_root)
            except FileNotFoundError as error:
                raise MovieConfigurationError('missing_runtime', str(error)) from error
            record = preflight.preflight(self.install_root, self.profile, setup_only=self.setup_only,
                                         child_python=python)
        except MovieConfigurationError as error:
            fault = fault_record(error)
            preflight.write_record(self.profile, {'at_utc': utc(), 'outcome': 'refused', 'fault': fault,
                                                  'run_id': self.entry['run_id']})
            log.error('KevFLIX startup refused (configuration, not retried): %s', error)
            self._finish('refused_configuration', EXIT['configuration'], fault=fault)
            return None
        self.receipt['preflight'] = record
        preflight.write_record(self.profile, {**record, 'outcome': 'passed', 'run_id': self.entry['run_id']})
        return python

    def _run_registered(self):
        # Validates the user's settings (Config) and every startup path; only the app-owned GPU-first keys are
        # rewritten (migrate_gpu_first), nothing else.
        python = self._preflight()
        if python is None:
            return self.receipt['exit']
        self.stop_event = winjob.StopEvent(self.entry['run_id'])
        self.job = winjob.KillOnCloseJob()
        self._start_server(python)
        outcome = None
        while True:
            if self._stop_requested():
                outcome = 'stopped'
                break
            if self.server.poll() is not None:
                outcome = 'server_exited'
                break
            self.sleep(LOOP_S)
        self._end_server()
        if outcome == 'server_exited':
            log.error('KevFLIX server exited unexpectedly (exit %s); see %s', self.server.returncode,
                      self.profile.logs / 'server.stderr.log')
            return self._finish(outcome, EXIT['failed'], server_exit=self.server.returncode)
        return self._finish(outcome, EXIT['stopped'])

    def _close(self):
        try:
            if self.server is not None and self.server.poll() is None:
                self._end_server()
        finally:
            for stream in self.files:
                try:
                    stream.close()
                except OSError:
                    pass
            if self.job is not None:
                self.job.close()
            if self.stop_event is not None:
                self.stop_event.close()
            if self.entry is not None:
                instances.unregister(self.profile, self.entry['run_id'])
            if self.install_lock is not None:
                self.install_lock.close()
            if self.mutex is not None:
                self.mutex.close()
            self.receipt['finished_utc'] = utc()
            try:
                name = 'guard-%s.json' % (self.entry['run_id'] if self.entry else utc().replace(':', ''))
                atomic_json(self.profile.logs / name, self.receipt)
            except OSError:
                pass
