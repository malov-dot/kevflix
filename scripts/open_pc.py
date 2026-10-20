"""Open the local PC control panel, starting only setup mode when it is stopped."""
import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib import request
from urllib.parse import urlparse
import ipaddress

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("The worker redirected unexpectedly")


def preflight_refusal(profile, since=None):
    """The detail of a startup-preflight refusal written after `since` (wall time), else None."""
    path = profile.runtime / "startup-preflight.json"
    try:
        if not path.is_file() or (since is not None and path.stat().st_mtime < since):
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("outcome") != "refused":
        return None
    fault = record.get("fault") or {}
    return "%s: %s" % (fault.get("code"), fault.get("detail"))


def main():
    from relay.app_runtime.paths import Profile
    profile = Profile(install_root=ROOT).ensure()
    from relay.settings import preserve_private_acl
    profile.migrate_legacy(acl_from=preserve_private_acl)
    config_path = profile.config
    if not config_path.is_file():
        raise SystemExit("KevFLIX settings were not found at %s. Run KevFLIX Setup." % config_path)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    base = str(config["public_url"]).rstrip("/")
    parsed = urlparse(base)
    try:
        private = ipaddress.ip_address(parsed.hostname).is_private
    except ValueError:
        private = parsed.hostname == "localhost" or str(parsed.hostname).endswith(".local")
    if parsed.scheme not in ("http", "https") or not private or parsed.username or parsed.password:
        raise SystemExit("Configure a private PC worker URL first")
    opener = request.build_opener(NoRedirect)
    def version():
        with opener.open(base + "/tv-api/version", timeout=2) as response:
            value = json.load(response)
            if value.get("product") != "KevFLIX":
                raise ValueError("The configured port is not the current KevFLIX worker")
            return value
    try:
        version()
    except OSError:
        from relay.desktop_updates import apply_pending
        if apply_pending(config_path):
            print("Installing the downloaded KevFLIX update. Open KevFLIX PC again when installation finishes.")
            return
        log_dir = profile.logs
        log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYANNOTE_METRICS_ENABLED="0", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS} if os.name == "nt" else {"start_new_session": True}
        started = time.time()
        with (log_dir / "worker-desktop.log").open("ab") as log:
            subprocess.Popen([sys.executable, str(ROOT / "scripts/run_worker.py")],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kwargs)
        until = time.monotonic() + 45      # the guard verifies the integrity manifest before the server starts
        while True:
            try:
                version()
                break
            except OSError:
                if time.monotonic() >= until:
                    raise SystemExit("Worker did not start. Inspect %s and worker.log there." % (log_dir / "worker-desktop.log"))
                refused = preflight_refusal(profile, since=started)
                if refused:
                    raise SystemExit("KevFLIX did not start (configuration): " + refused)
                time.sleep(.3)
    req = request.Request(base + "/api/pc-launch", data=b"{}", method="POST",
        headers={"Authorization":"Bearer " + config["api_token"],"Content-Type":"application/json"})
    with opener.open(req, timeout=5) as response:
        launch = json.load(response)["url"]
    webbrowser.open(launch, new=2)
    print("KevFLIX PC opened. The saved setup/playback setting is preserved.")


if __name__ == "__main__":
    main()
