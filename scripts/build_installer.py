"""Build the unsigned KevFLIX installer from a clean app tree (fix 1). The integrity manifest is generated HERE,
automatically, for every packaged file; there is no separate sealing step.

  python scripts/build_installer.py --source <app tree> --runtime <python-runtime.tar.gz> --output <dist dir>
                                    [--base-revision <40-hex revision of the publication base>]
                                    [--legacy-tree <the earlier published app tree, e.g. the 0.4.6 publication>]

Identity (honest): this assembled source is not a Git commit. The build's content identity ("revision" in
desktop-build.json, desktop-release.json and the installer marker) is the sha256 of its integrity manifest;
base_revision only names the publication base it was assembled from; status is "unpublished local build".

Steps (any failure stops the build; nothing is skipped silently):
  1. the pinned portable Python runtime (sha256) and one consistent version in package.json, relay/__init__.py and
     tv/package.json
  2. the file set: relay/** (incl. movie_pipeline, app_runtime with the bundled lease client, its tolerant reader
     and the private guard-package root), actor-tts/** (incl. the native adapter source), web/**, tv/** (incl. the
     player), scripts/**, docs/**, installer/** and the root launchers -- by relay/app_runtime/integrity.py
     select_package_files(); .git/.github/__pycache__, private config, tokens, media, recordings and model weights
     are excluded
  3. byte-identity of the pinned dependencies (lease client, tolerant reader, the 5 guard files, W3's
     relay/actor_voice.py and relay/resource_budget.py)
  4. no development path in any packaged text file (exact reviewed allowlist only: integrity.DEV_REFERENCE_ALLOWLIST)
  5. the packaged default config = config.example.json + actor_beta.vox.sources_sha256 GENERATED from the app tree
     (W3's pins rule: actor-tts/automatic-v1/automatic_renderer.py, reference-cache/reference_cache.py, every
     runtime-adapter/*.py)
  6. app-manifest.json (sha256 + size of every packaged file and every runtime file), desktop-build.json
     (version, revision, manifest_sha256) and app-files.txt (the installer removes files the previous build owned
     and this one no longer ships)
  7. payload.zip -> csc.exe compile of installer/Setup.cs (as before) -> KevFLIX-Setup-<version>.exe + .sha256 +
     desktop-release.json, which states honestly what is embedded and what is provisioned on the PC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

RUNTIME_SHA256 = "54a38c5e7cc1a5fd2b0182b495f29606dc1455d18ebfd3b5bf0a3c09253b5031"
RUNTIME_NAME = "python-3.11.17-20261013"
VERSION_RE = r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?"
COMPILER = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
# Byte-identical pins the running app depends on (author pins; W3 fix 6).
PINNED = {
    "relay/app_runtime/lease/lease_client.py": "5cbac8a4d33c114bf811bab50e5957142dfa46fa8e749ab482bf78749fefcd29",
    "relay/app_runtime/guard-status-read-v1/tolerant_status.py": "d58b4ec2fcb37d3ad0f45dd24c18a9fbd3d6c1b5dae5aca8b7e9910c6aaf587f",
    "relay/app_runtime/guard-source/relay/__init__.py": "04c1165f92486791c29e7f5a7fb19046396d52c090e943ae98e271bcb9a1ec24",
    "relay/app_runtime/guard-source/relay/short_utterance_policy.py": "4d8b55e27c649aa560bf1593558de85eb936b870ab56aa8845943aa559150204",
    "relay/app_runtime/guard-source/relay/original_audio_policy.py": "16c379652457953a7c42dea5546041037aed0d8891daedb369735bdcade441a2",
    "relay/app_runtime/guard-source/relay/cinematic.py": "ef39c3954c39aba9c08b5bdf4c1dd4cc99148285c0434d9e0b6cb9ece9a6c206",
    "relay/app_runtime/guard-source/relay/demucs_bed.py": "49ec437aae77f56db41fad5ec913eff61b9baa3d44e787bd5e4dcb3bff1ba32e",
    "relay/actor_voice.py": "8a97e12544b18776c5da442ec851fcc6d8591f61fc1ae5ad4f483f10ac67f96f",
    "relay/resource_budget.py": "f8f068962c45db45bd515b7e833392ae72e1cbfc21042d694e9e148f95d0f819",
}
REQUIRED_DIRS = ("relay/movie_pipeline", "relay/app_runtime", "actor-tts/runtime-adapter", "actor-tts/automatic-v1",
                 "actor-tts/reference-cache", "tv/app", "web")
EMBEDDED = ["KevFLIX app code and assets (relay, actor-tts adapter source, web, tv player, scripts, docs)",
            "pinned portable Python " + RUNTIME_NAME, "movie GPU lease v2 client (5cbac8a4) and its guard sources",
            "integrity manifest (verified at every start)"]
PROVISIONED = [
    {"item": "Python dependencies (.venv)", "how": "Setup-Worker.ps1 (requirements.txt; torch/torchaudio cu128)",
     "embedded": False},
    {"item": "Vox model voxcpm2-actor-q4_k.gguf (1,697,232,640 bytes, sha256 1d0b2970...)",
     "how": "Setup-Worker.ps1 -VoxModelSource <folder> or KevFLIX Setup /vox-model <folder> (verified, copied into "
            "the profile or referenced in place)", "embedded": False},
    {"item": "Vox native runtime (audiocpp.dll + ggml/CUDA 13/cufft 12/MSVC DLLs; sha256 list in "
             "actor-tts/runtime-adapter/approved.py)",
     "how": "Setup-Worker.ps1 -VoxRuntimeSource <folder> or KevFLIX Setup /vox-runtime <folder>", "embedded": False},
    {"item": "ASR/translation/diarization/separation models", "how": "scripts/provision_models.py (upstream terms)",
     "embedded": False},
    {"item": "FFmpeg", "how": "Setup-FFmpeg.ps1", "embedded": False},
    {"item": "X-Translator/X-Voice WSL runtime and model assets",
     "how": "PC settings X-Translator installer or python -m relay.app_runtime.xt_provision --config <profile>",
     "embedded": False},
]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# The packaging rule of the earlier published installer (0.4.6 build_installer.py), used ONLY to enumerate what an
# old install provably received; never used to choose this build's files.
LEGACY_DIRS = {"relay", "web", "tv", "scripts", "docs", "installer"}
LEGACY_ROOT = {"config.example.json", "requirements.txt", "THIRD-PARTY.md", "README.md", "package.json",
               "Setup-Worker.ps1", "Setup-FFmpeg.ps1", "Create-Desktop-Shortcut.ps1", "Open-KevFLIX-PC.cmd"}


def legacy_shipped(tree):
    """'sha256  path' lines for every file the earlier published package shipped from `tree` (read only)."""
    tree = tree.resolve()
    if not (tree / "package.json").is_file():
        raise SystemExit("--legacy-tree is not an app tree: %s" % tree)
    lines = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(tree)
        if ".git" in rel.parts or "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        posix = rel.as_posix()
        if rel.parts[0] in LEGACY_DIRS or posix in LEGACY_ROOT or posix == "legacy/scripts/render_sapi.ps1":
            lines.append("%s  %s\n" % (sha256_bytes(path.read_bytes()), posix))
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-tree", default=None,
                        help="earlier published tree whose shipped files (path + sha256) an update may remove when "
                             "the previous install predates app-files.txt")
    parser.add_argument("--base-revision", default=None,
                        help="40-hex revision of the publication base (recorded as the base, not as this build)")
    args = parser.parse_args()
    source, runtime, output = Path(args.source).resolve(), Path(args.runtime), Path(args.output).resolve()
    sys.path.insert(0, str(source))
    from relay.app_runtime import integrity
    from relay.config import VOX_SOURCE_FILES, vox_source_paths

    # 1. runtime pin and one version
    if hashlib.sha256(runtime.read_bytes()).hexdigest() != RUNTIME_SHA256:
        raise SystemExit("Portable Python differs from the pinned upstream release")
    version = integrity.package_version(source)
    relay_version = integrity.relay_version(source)
    tv_version = json.loads((source / "tv" / "package.json").read_text(encoding="utf-8-sig")).get("version")
    if args.base_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", args.base_revision):
        raise SystemExit("--base-revision must be a 40-hex revision")
    if not re.fullmatch(VERSION_RE, str(version)):
        raise SystemExit("Invalid version %r" % version)
    if not (version == relay_version == tv_version):
        raise SystemExit("Version mismatch: package.json %s, relay/__init__.py %s, tv/package.json %s"
                         % (version, relay_version, tv_version))

    # 2. file set
    files, skipped = integrity.select_package_files(source)
    chosen = dict(files)
    missing_dirs = [d for d in REQUIRED_DIRS if not any(p.startswith(d + "/") for p in chosen)]
    if missing_dirs:
        raise SystemExit("Required app parts are missing from the source tree: %s" % missing_dirs)

    # The app manifest alone cannot detect an exclusion that contradicts the
    # vendor's own identity manifest. Verify that contract before packaging.
    vendor_prefix = "relay/vendor/xtranslator/"
    provenance_name = vendor_prefix + "PROVENANCE.json"
    if ("relay/xt_services.py" in chosen or any(name.startswith(vendor_prefix) for name in chosen)) and provenance_name not in chosen:
        raise SystemExit("X-Translator source is packaged without its provenance manifest")
    if provenance_name in chosen:
        provenance = json.loads(chosen[provenance_name].read_text(encoding="utf-8-sig"))
        bad_vendor = []
        for record in provenance["files"]:
            name = vendor_prefix + record["path"]
            if name not in chosen or sha256_bytes(chosen[name].read_bytes()) != record["sha256"]:
                bad_vendor.append(name)
        wrapper = "relay/xt_services.py"
        if wrapper not in chosen or sha256_bytes(chosen[wrapper].read_bytes()) != provenance["xt_services_sha256"]:
            bad_vendor.append(wrapper)
        if bad_vendor:
            raise SystemExit("X-Translator provenance conflicts with packaged files: %s" % bad_vendor[:20])

    # 3. byte-identical pins
    bad = {rel: (sha256_bytes(chosen[rel].read_bytes()) if rel in chosen else None)
           for rel, digest in PINNED.items() if rel not in chosen or sha256_bytes(chosen[rel].read_bytes()) != digest}
    if bad:
        raise SystemExit("Pinned dependency files are missing or changed: %s" % bad)

    # 4. no development paths
    hits = integrity.dev_reference_hits(files)
    output.mkdir(parents=True, exist_ok=True)
    (output / "build-files.json").write_text(json.dumps({"packaged": sorted(chosen), "excluded": skipped,
                                                         "dev_reference_hits": hits}, indent=1), encoding="utf-8")
    if hits:
        raise SystemExit("Development paths in packaged files (see build-files.json): %s"
                         % [h["path"] for h in hits][:20])

    # 5. packaged default config with GENERATED Vox source pins
    example = json.loads((source / "config.example.json").read_text(encoding="utf-8-sig"))
    actor_root = source / "actor-tts"
    pins = {}
    for path in vox_source_paths(source):
        if not path.is_file():
            raise SystemExit("Vox adapter source missing: %s" % path.relative_to(source))
        pins[path.relative_to(actor_root).as_posix()] = sha256_bytes(path.read_bytes())
    if not all(name in pins for name in VOX_SOURCE_FILES):
        raise SystemExit("Vox pins rule did not find %s" % [n for n in VOX_SOURCE_FILES if n not in pins])
    vox = example.setdefault("actor_beta", {}).setdefault("vox", {})
    vox["sources_sha256"] = pins
    config_bytes = (json.dumps(example, indent=2) + "\n").encode("utf-8")
    chosen.pop("config.example.json", None)

    # 6. manifest over everything packaged, incl. the runtime
    entries = [(rel, path) for rel, path in chosen.items()] + [("config.example.json", config_bytes)]
    runtime_entries = []
    with tarfile.open(runtime, "r:gz") as upstream:
        for entry in upstream:
            if not entry.isfile():
                continue
            name = entry.name
            if not name.startswith("python/") or ".." in Path(name).parts or ":" in name:
                raise SystemExit("Unexpected runtime archive path")
            if name.endswith((".pyc", ".pyo")) or "/__pycache__/" in name:
                continue
            runtime_entries.append(("runtime/" + name, upstream.extractfile(entry).read()))
    manifest, manifest_bytes, manifest_sha = integrity.build_manifest(entries + runtime_entries, version=version,
                                                                      base_revision=args.base_revision)
    # content identity = the manifest digest; the base revision is a separate, labelled field
    build_info = {"version": version, "revision": manifest_sha, "identity": "integrity-manifest sha256",
                  "manifest_sha256": manifest_sha, "manifest_files": len(manifest["files"]),
                  "base_revision": args.base_revision, "status": "unpublished local build"}
    # "sha256  path" lines: an update removes an obsolete file only if both its path and its sha256 match
    file_list = "".join("%s  %s\n" % (manifest["files"][rel]["sha256"], rel) for rel in sorted(manifest["files"]))
    legacy_list = legacy_shipped(Path(args.legacy_tree)) if args.legacy_tree else ""

    # 7. payload, compile, release metadata
    payload = output / "payload.zip"
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(integrity.BUILD_NAME, json.dumps(build_info, indent=2))
        archive.writestr(integrity.MANIFEST_NAME, manifest_bytes)
        archive.writestr("app-files.txt", file_list)
        archive.writestr("legacy-shipped.txt", legacy_list)
        for rel, item in sorted(entries + runtime_entries, key=lambda e: e[0]):
            if isinstance(item, (bytes, bytearray)):
                archive.writestr(rel, item)
            else:
                archive.write(item, rel)
    target = output / ("KevFLIX-Setup-" + version + ".exe")
    identity = output / "BuildInfo.cs"
    identity.write_text('internal static class BuildInfo { public const string Version="' + version +
                        '"; public const string Revision="' + manifest_sha + '"; public const string ManifestSha256="' +
                        manifest_sha + '"; public const string BaseRevision="' + (args.base_revision or "") + '"; }')
    subprocess.run([str(COMPILER), "/nologo", "/target:winexe", "/platform:x64", "/optimize+", "/debug-",
        "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", "/r:System.IO.Compression.dll", "/r:System.Core.dll",
        "/resource:" + str(payload) + ",payload.zip", "/out:" + str(target), str(source / "installer/Setup.cs"),
        str(identity)], check=True)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".exe.sha256").write_text(digest + "  " + target.name + "\n")
    release = {"product": "KevFLIX", "version": version, "revision": manifest_sha,
               "identity": "integrity-manifest sha256 (content-addressed; not a Git commit)",
               "base_revision": args.base_revision, "status": "unpublished local build", "asset": target.name,
               "sha256": digest, "size": target.stat().st_size, "runtime": RUNTIME_NAME,
               "dependencies": "requirements.txt", "manifest_sha256": manifest_sha,
               "manifest_files": len(manifest["files"]), "embedded": EMBEDDED, "provisioned_on_pc": PROVISIONED,
               "not_covered_by_manifest": manifest["not_covered"],
               "legacy_update_allowlist": ("files of %s (path + sha256)" % Path(args.legacy_tree).name)
               if args.legacy_tree else "none: an update over a build without app-files.txt keeps every unknown "
                                        "file and stops if one sits in a verified code folder",
               "installer_executed": False, "tests_run": False}
    (output / "desktop-release.json").write_text(json.dumps(release, indent=2))
    print(json.dumps({"file": str(target), "version": version, "sha256": digest, "manifest_sha256": manifest_sha,
                      "manifest_files": len(manifest["files"]), "installer_executed": False, "tests_run": False}))


if __name__ == "__main__":
    main()
