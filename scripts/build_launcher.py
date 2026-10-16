"""Create one versioned launcher document so CDN caches cannot mix JS/HTML releases."""
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parent.parent
    tv = root / "tv"
    version = json.loads((tv / "package.json").read_text())["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit("Expected a three-part product version")
    html = (tv / "bootstrap/index.html").read_text(encoding="utf-8")
    css = (tv / "bootstrap/style.css").read_text(encoding="utf-8")
    js = (tv / "bootstrap/bootstrap.js").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="style.css">', '<style>' + css + '</style>')
    html = html.replace('<script src="bootstrap.js"></script>', '<script>' + js.replace('</script', '<\\/script') + '</script>')
    destination = tv / "launchers" / version / "index.html"
    if destination.exists() and destination.read_text(encoding="utf-8") != html:
        raise SystemExit("Published launcher paths are immutable; increase the version first")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(json.dumps({"launcher": str(destination), "version": version, "application_executed": False}))


if __name__ == "__main__":
    main()
