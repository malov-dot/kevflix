"""Construct TV distribution archives without running the player or a compiler."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    tv = root / "tv"
    metadata = json.loads((tv / "package.json").read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / ("KevFLIX-" + metadata["version"] + "-unsigned.wgt")
    if target.exists():
        raise SystemExit("Package already exists; preserve it and choose a new output folder")
    names = ["config.xml", "bootstrap/index.html", "bootstrap/style.css", "bootstrap/bootstrap.js"]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, (tv / name).read_bytes())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".wgt.sha256").write_text(digest + "  " + target.name + "\n")
    print(json.dumps({"file": str(target), "sha256": digest, "signed": False,
                      "application_executed": False, "tests_run": False}))


if __name__ == "__main__":
    main()
