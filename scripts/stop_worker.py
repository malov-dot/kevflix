"""Stop the running KevFLIX app (fix 3).

Reads the explicit active-instance registration in the user profile, validates that the recorded process is still
the same live KevFLIX guard (pid + creation time + image), sets its registered stop event, and waits. Prints an
honest result: "stopped", "not stopped" (with the reason), or "no active instance" (a stale entry is reported and
removed only if it is still the same run; no process is signalled). Media and other folders are never touched.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relay.app_runtime import instances  # noqa: E402
from relay.app_runtime.paths import Profile  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args(argv)
    result = instances.stop(Profile(install_root=ROOT), timeout_s=args.timeout)
    print(json.dumps(result, indent=1))
    return 0 if result.get('ok') else (1 if result.get('result') == 'no active instance' else 2)


if __name__ == '__main__':
    raise SystemExit(main())
