"""Start KevFLIX when the user logs in to Windows, through Windows scheduled tasks (stdlib only).

Two per-user tasks, both "run only when the user is logged on", least privilege, no time limit:
  KevFLIX Server  -> <install>\\.venv\\Scripts\\pythonw.exe (else runtime\\python\\pythonw.exe) scripts\\run_worker.py
  KevFLIX Tray    -> powershell.exe -File scripts\\tray.ps1 (the notification-area icon: running / stopped, open, start, stop)
The earlier HKCU\\...\\Run entry is removed: on a PC it was enumerated at login but never launched (Windows started only the
other entries), a logon task is launched by the Task Scheduler and can be checked with `schtasks /Query`."""
import os
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

SERVER_TASK = 'KevFLIX Server'
TRAY_TASK = 'KevFLIX Tray'
TASKS = (SERVER_TASK, TRAY_TASK)
SERVER_DELAY = 'PT15S'
TRAY_DELAY = 'PT10S'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def current_user():
    return '%s\\%s' % (os.environ.get('USERDOMAIN') or os.environ.get('COMPUTERNAME') or '', os.environ.get('USERNAME') or '')


def server_python(root):
    root = Path(root)
    for candidate in (root / '.venv' / 'Scripts' / 'pythonw.exe', root / 'runtime' / 'python' / 'pythonw.exe'):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('no KevFLIX Python found in %s (run Setup-Worker.ps1)' % root)


def powershell_exe():
    return Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'


def task_xml(*, user, command, arguments, working_directory, delay, description, restart=False):
    """Task Scheduler XML (schema 1.2) for a logon task of `user`."""
    restart_xml = '<RestartOnFailure><Interval>PT1M</Interval><Count>3</Count></RestartOnFailure>' if restart else ''
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        '  <RegistrationInfo><Description>%s</Description></RegistrationInfo>\n'
        '  <Triggers><LogonTrigger><Enabled>true</Enabled><UserId>%s</UserId><Delay>%s</Delay></LogonTrigger></Triggers>\n'
        '  <Principals><Principal id="Author"><UserId>%s</UserId><LogonType>InteractiveToken</LogonType>'
        '<RunLevel>LeastPrivilege</RunLevel></Principal></Principals>\n'
        '  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>'
        '<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><AllowHardTerminate>true</AllowHardTerminate><StartWhenAvailable>true</StartWhenAvailable>'
        '<RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable><AllowStartOnDemand>true</AllowStartOnDemand><Enabled>true</Enabled><Hidden>false</Hidden>'
        '<ExecutionTimeLimit>PT0S</ExecutionTimeLimit><Priority>7</Priority>%s</Settings>\n'
        '  <Actions Context="Author"><Exec><Command>%s</Command><Arguments>%s</Arguments><WorkingDirectory>%s</WorkingDirectory></Exec></Actions>\n'
        '</Task>\n') % (escape(description), escape(user), delay, escape(user), restart_xml,
                        escape(str(command)), escape(arguments), escape(str(working_directory)))


def definitions(root, user=None):
    """[(task name, xml)] for this install."""
    root = Path(root)
    user = user or current_user()
    server = task_xml(user=user, command=server_python(root), arguments='"%s"' % (root / 'scripts' / 'run_worker.py'),
                      working_directory=root, delay=SERVER_DELAY, restart=True,
                      description='Starts the KevFLIX PC worker (dubbing server for the TV) when you log in.')
    tray = task_xml(user=user, command=powershell_exe(),
                    arguments='-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%s"' % (root / 'scripts' / 'tray.ps1'),
                    working_directory=root, delay=TRAY_DELAY,
                    description='Shows the KevFLIX icon in the notification area while you are logged in.')
    return [(SERVER_TASK, server), (TRAY_TASK, tray)]


def _schtasks(*args):
    return subprocess.run(['schtasks.exe', *args], capture_output=True, text=True, creationflags=NO_WINDOW, timeout=60)


def exists(name):
    return _schtasks('/Query', '/TN', name).returncode == 0


def enabled():
    return exists(SERVER_TASK)


def install(root, user=None):
    """Create (or replace) both tasks. Raises ValueError with the scheduler's own message on failure."""
    for name, xml in definitions(root, user):
        handle, path = tempfile.mkstemp(suffix='.xml', prefix='kevflix-task-')
        try:
            with os.fdopen(handle, 'wb') as stream:
                stream.write(b'\xff\xfe' + xml.encode('utf-16-le'))
            done = _schtasks('/Create', '/TN', name, '/XML', path, '/F')
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if done.returncode != 0:
            raise ValueError('Windows could not create the "%s" scheduled task: %s' % (name, (done.stderr or done.stdout).strip()[:300]))


def remove():
    """Delete both tasks; a task that is not there is fine."""
    for name in TASKS:
        if exists(name):
            _schtasks('/Delete', '/TN', name, '/F')


def run_now(name):
    return _schtasks('/Run', '/TN', name).returncode == 0
