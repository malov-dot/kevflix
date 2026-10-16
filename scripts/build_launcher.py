"""Keep cached TizenBrew launcher addresses pointed at the canonical bootstrap.

Only source in the next release changes. Existing Git tags and release artifacts
are retained. Old unversioned CDN paths are purged by the release workflow.
"""
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parent.parent
    tv = root / "tv"
    version = json.loads((tv / "package.json").read_text())["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit("Expected a three-part product version")
    versions = {version}
    legacy = tv / "launchers"
    if legacy.exists():
        versions.update(p.name for p in legacy.iterdir() if p.is_dir() and re.fullmatch(r"\d+\.\d+\.\d+", p.name))
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KevFLIX</title><meta http-equiv="refresh" content="0;url=../../bootstrap/index.html">
<style>body{background:#111216;color:#fff;font:28px Arial;padding:60px}a{color:#efad35}</style></head>
<body><h1>KevFLIX</h1><p>Opening KevFLIX…</p><a href="../../bootstrap/index.html">Open KevFLIX</a>
<script>location.replace('../../bootstrap/index.html'+location.hash);</script></body></html>
'''
    for previous in sorted(versions):
        destination = legacy / previous / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html, encoding="utf-8")
    print(json.dumps({"canonical_launcher": "tv/bootstrap/index.html", "compatibility_paths": sorted(versions),
                      "version": version, "application_executed": False}))


if __name__ == "__main__":
    main()
