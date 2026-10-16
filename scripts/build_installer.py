"""Build an unsigned installer from a clean publication tree. Never run it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

RUNTIME_SHA256 = "54a38c5e7cc1a5fd2b0182b495f29606dc1455d18ebfd3b5bf0a3c09253b5031"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    source, runtime, output = Path(args.source).resolve(), Path(args.runtime), Path(args.output).resolve()
    if hashlib.sha256(runtime.read_bytes()).hexdigest() != RUNTIME_SHA256:
        raise SystemExit("Portable Python differs from the pinned upstream release")
    output.mkdir(parents=True, exist_ok=True)
    version = json.loads((source / "package.json").read_text())["version"]
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit("Invalid release identity")
    payload = output / "payload.zip"
    allowed_dirs = {"relay", "web", "tv", "scripts", "docs", "installer"}
    allowed_root = {"config.example.json", "requirements.txt", "THIRD-PARTY.md", "README.md", "package.json",
                    "Setup-Worker.ps1", "Setup-FFmpeg.ps1", "Create-Desktop-Shortcut.ps1", "Open-KevFLIX-PC.cmd"}
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("desktop-build.json", json.dumps({"version": version, "revision": args.revision}))
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source)
            if ".git" in rel.parts or "__pycache__" in rel.parts or path.suffix == ".pyc":
                continue
            if rel.parts[0] in allowed_dirs or rel.as_posix() in allowed_root or rel.as_posix() == "legacy/scripts/render_sapi.ps1":
                archive.write(path, rel.as_posix())
        # Copy only regular upstream files. Keep all runtime license notices.
        with tarfile.open(runtime, "r:gz") as upstream:
            for entry in upstream:
                if not entry.isfile():
                    continue
                name = entry.name
                if not name.startswith("python/") or ".." in Path(name).parts or ":" in name:
                    raise SystemExit("Unexpected runtime archive path")
                archive.writestr("runtime/" + name, upstream.extractfile(entry).read())
    target = output / ("KevFLIX-Setup-" + version + ".exe")
    compiler = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    identity = output / "BuildInfo.cs"
    identity.write_text('internal static class BuildInfo { public const string Version="'+version+'"; public const string Revision="'+args.revision+'"; }')
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/platform:x64", "/optimize+", "/debug-",
        "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", "/r:System.IO.Compression.dll", "/r:System.Core.dll",
        "/resource:" + str(payload) + ",payload.zip", "/out:" + str(target), str(source / "installer/Setup.cs"), str(identity)], check=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".exe.sha256").write_text(digest + "  " + target.name + "\n")
    (output / "desktop-release.json").write_text(json.dumps({"product": "KevFLIX", "version": version,
        "revision": args.revision, "asset": target.name, "sha256": digest, "size": target.stat().st_size,
        "runtime": "python-3.11.17-20261013", "dependencies": "requirements.txt"}, indent=2))
    print(json.dumps({"file": str(target), "version": version, "sha256": digest,
                      "installer_executed": False, "tests_run": False}))


if __name__ == "__main__":
    main()
