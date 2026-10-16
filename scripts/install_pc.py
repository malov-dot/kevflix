"""First-run installation; generates new private settings and desktop shortcuts."""
import argparse
import ctypes
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--startup", choices=("on", "off"), default="on")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    config = ROOT / "relay-config.json"
    if not config.exists() or config.stat().st_size == 0:
        settings = json.loads((ROOT / "config.example.json").read_text())
        for name in ("api_token", "nas_token", "stream_secret", "webhook_secret"):
            settings[name] = secrets.token_urlsafe(48)
        addresses = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
        hosts = [a[4][0] for a in addresses if ipaddress.ip_address(a[4][0]).is_private
                 and not ipaddress.ip_address(a[4][0]).is_loopback
                 and not ipaddress.ip_address(a[4][0]).is_link_local]
        settings.update(setup_only=True, bind="0.0.0.0", public_url="http://" + (hosts[0] if hosts else "127.0.0.1") + ":8787",
                        start_with_windows=args.startup == "on")
        settings["media_roots"] = [str(Path.home() / "Videos")]
        settings["tv"]["roots"] = settings["media_roots"]
        config.touch(exist_ok=True)
        # Apply the current user's SID, not their name, before writing secrets.
        sid = json.loads(subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
            "[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value | ConvertTo-Json"],
            creationflags=subprocess.CREATE_NO_WINDOW, text=True, encoding="utf-8-sig"))
        subprocess.run(["icacls.exe", str(config), "/inheritance:r", "/grant:r", "*" + sid + ":(F)"],
                       check=True, stdout=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        config.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    from relay.settings import set_startup
    set_startup(args.startup == "on")
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "Create-Desktop-Shortcut.ps1")], check=True, creationflags=subprocess.CREATE_NO_WINDOW)


if __name__ == "__main__":
    main()
