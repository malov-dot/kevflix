"""Create the private KevFLIX settings file in the user profile ONLY if none exists; never contact Plex, Sem or the
NAS, never print a secret. Normally KevFLIX Setup does this (scripts/install_pc.py); this helper is for manual
setups. An existing settings file is never replaced (create-if-absent)."""
import argparse
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relay.app_runtime.paths import Profile  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    profile = Profile(install_root=ROOT).ensure()
    if profile.config.exists() or profile.legacy_config().exists():
        raise SystemExit("Settings already exist (%s); they are kept. Edit them in KevFLIX PC." % profile.config)
    settings = profile.new_config(ROOT / "config.example.json", secrets_factory=lambda: secrets.token_urlsafe(48),
                                  bind=args.bind, public_url="http://%s:%d" % (args.bind, args.port),
                                  startup=False, media_roots=[str(Path.home() / "Videos")])
    settings["port"] = args.port
    from install_pc import private_acl          # same owner-only ACL as KevFLIX Setup
    if not profile.write_new_config(settings, acl=private_acl):
        raise SystemExit("Settings were created concurrently; they are kept.")
    print("Private settings created at %s. No services contacted; no secrets printed." % profile.config)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
