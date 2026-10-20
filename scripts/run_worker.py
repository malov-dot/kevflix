"""KevFLIX background entry point (Windows startup, desktop launcher, installer).

Starts the app guard (relay/app_runtime/guard.py), which verifies the installed integrity manifest, uses the
persistent user profile, registers the active instance, then runs the app server and the GPU dubbing producer under
its Jobs and the movie GPU lease. Stdlib only in this process. The user's settings are used as they are.
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['PYTHONUTF8'] = '1'

from relay.app_runtime.paths import Profile  # noqa: E402

profile = Profile(install_root=ROOT).ensure()
logging.basicConfig(level=logging.INFO, handlers=[
    logging.handlers.RotatingFileHandler(profile.logs / 'worker.log', maxBytes=2 * 1024 * 1024, backupCount=3,
                                         encoding='utf-8')],
    format='%(asctime)s %(levelname)s %(message)s')

from relay.desktop_updates import apply_pending  # noqa: E402

try:
    if apply_pending(profile.config):
        raise SystemExit(0)                 # the verified installer replaces the app and starts it again
except SystemExit:
    raise
except Exception:
    logging.exception('Desktop update deferred; keeping the installed app')

from relay.app_runtime.guard import AppGuard  # noqa: E402

guard = AppGuard(install_root=ROOT, profile=profile, setup_only='--setup-only' in sys.argv[1:])
code = guard.run()
logging.info('KevFLIX guard finished: %s (exit %s)', guard.receipt.get('outcome'), code)
if guard.receipt.get('integrity_problems'):
    logging.error('Integrity problems: %s', guard.receipt['integrity_problems'][:10])
if guard.receipt.get('fault'):
    # Startup preflight refusal (configuration; fault_kind fatal): not retried until the settings/paths are fixed.
    logging.error('Startup refused: %s', guard.receipt['fault'])
raise SystemExit(code)
