"""Install/update step run by KevFLIX Setup (fix 4). Never resets the user's data.

* The user's settings live in the persistent profile (relay/app_runtime/paths.py), not in the install folder.
* Existing settings are KEPT as they are: Plex account/server, media folders, UI settings, LAN bind/port and the
  Windows-startup preference are never rewritten from config.example.json, and the startup registration is not
  touched on update.
* An old in-install relay-config.json (earlier layout) is COPIED into the profile once (the original stays).
* Only a brand-new PC (no settings anywhere) gets a new settings file with new local secrets; that file is published
  atomically only if none exists, with a private ACL applied before the secrets are written. Only then is the
  --startup choice applied.
"""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relay.app_runtime.paths import Profile  # noqa: E402


def private_acl(path):
    """Owner-only ACL for the current user's SID (applied to the temp file before secrets are written)."""
    sid = json.loads(subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
        "[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value | ConvertTo-Json"],
        creationflags=subprocess.CREATE_NO_WINDOW, text=True, encoding="utf-8-sig"))
    subprocess.run(["icacls.exe", str(path), "/inheritance:r", "/grant:r", "*" + sid + ":(F)"],
                   check=True, stdout=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)


def lan_url(port):
    addresses = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
    hosts = [a[4][0] for a in addresses if ipaddress.ip_address(a[4][0]).is_private
             and not ipaddress.ip_address(a[4][0]).is_loopback and not ipaddress.ip_address(a[4][0]).is_link_local]
    return "http://" + (hosts[0] if hosts else "127.0.0.1") + ":" + str(port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--startup", choices=("on", "off"), default="on",
                        help="Windows startup choice for a BRAND-NEW install only; updates keep the user's choice")
    parser.add_argument("--commit-app-pins", action="store_true",
                        help="KevFLIX Setup's LAST required step: one atomic write of this build's app-owned fields")
    args = parser.parse_args()
    os.chdir(ROOT)
    profile = Profile(install_root=ROOT).ensure()
    from relay.settings import preserve_private_acl, set_startup
    if args.commit_app_pins:
        print(json.dumps({"app_pins": refresh_app_pins(profile, preserve_private_acl)}))
        return
    result = {"profile": str(profile.root)}
    migrated = profile.migrate_legacy(acl_from=preserve_private_acl)
    if migrated:
        result["migrated_legacy_settings"] = migrated
    if profile.config.is_file():
        result["settings"] = "kept (existing user settings and Windows-startup preference unchanged)"
    else:
        settings = profile.new_config(ROOT / "config.example.json", secrets_factory=lambda: secrets.token_urlsafe(48),
                                      bind="0.0.0.0", public_url=lan_url(8787), startup=args.startup == "on",
                                      media_roots=[str(Path.home() / "Videos")])
        created = profile.write_new_config(settings, acl=private_acl)
        if created:
            set_startup(args.startup == "on")
            result["settings"] = "created (new local secrets)"
        else:
            result["settings"] = "kept (created concurrently by another setup step)"
    result["app_pins"] = "deferred: committed by KevFLIX Setup after every required step succeeded (--commit-app-pins)"
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "Create-Desktop-Shortcut.ps1")], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    print(json.dumps(result))


def refresh_app_pins(profile, acl_from):
    """Update ONLY the app-owned code pins in the user's settings to this build: actor_beta.vox.sources_sha256 (the
    build-generated Vox adapter hashes in the packaged config.example.json) and actor_beta.vox.gpu_lease.client /
    client_sha256 (the bundled lease client). Every user setting is left as it is."""
    from relay.app_runtime.leasehold import LEASE_PATH, LEASE_SHA256
    from relay.app_runtime.vox_provision import merge_config
    example = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8-sig"))
    pins = ((example.get("actor_beta") or {}).get("vox") or {}).get("sources_sha256")
    if not pins or not profile.config.is_file():
        return "not applicable"

    def update(raw):
        beta = raw.get("actor_beta")
        if isinstance(beta, dict):
            vox = beta.setdefault("vox", {})
            vox["sources_sha256"] = pins
            lease = vox.setdefault("gpu_lease", {})
            lease["client"] = str(LEASE_PATH)
            lease["client_sha256"] = LEASE_SHA256
        return raw
    merge_config(profile.config, update, acl_from=acl_from)
    return "Vox adapter pins and lease client updated to this build"


if __name__ == "__main__":
    main()
