"""Write private configuration files only; never contact Plex, Sem or the NAS."""
import argparse
import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "relay-config.json"))
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.exists():
        raise SystemExit("Configuration exists. Edit it directly; secrets will not be replaced.")
    cfg = json.loads((ROOT / "config.example.json").read_text())
    for key in ("api_token","nas_token","stream_secret","webhook_secret"):
        cfg[key] = secrets.token_urlsafe(48)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg,indent=2))
    private = target.parent / "private-activation"
    private.mkdir(exist_ok=True)
    (private / "movie-dub-relay.json").write_text(json.dumps(
        {"url":"http://127.0.0.1:8787","api_token":cfg["api_token"]},indent=2))
    print("Private configuration created. Replace address/share/client placeholders before activation.")
    print("No services contacted; no secrets printed.")


if __name__ == "__main__":
    main()
