"""Integrity manifest for the installed KevFLIX app (fix 1). Stdlib only.

BUILD (scripts/build_installer.py, automatically, no manual step): select_package_files() picks every transitive app
module, asset and dependency the app ships (relay/** incl. movie_pipeline and app_runtime, actor-tts/** incl. the
native adapter source, web/**, tv/** incl. the player, scripts/**, docs, root launchers), build_manifest() hashes
them plus the pinned portable Python runtime files, and the manifest is packaged as app-manifest.json; its sha256
goes into desktop-build.json and desktop-release.json.

STARTUP (relay/app_runtime/guard.py, every start, never disabled): verify() refuses unless desktop-build.json names
this manifest, every listed file has its recorded size and sha256, and no unlisted code or asset file exists in a
covered folder (arbitrary runtime bytes are never accepted). There is no bypass and no developer override.

NOT covered (provisioned explicitly, never faked; listed in the manifest and the release JSON): the .venv created by
Setup-Worker.ps1 from requirements.txt, models (scripts/provision_models.py), FFmpeg (Setup-FFmpeg.ps1), the Vox
engine runtime/model named by the user config, and everything in the user profile.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

MANIFEST_NAME = 'app-manifest.json'
BUILD_NAME = 'desktop-build.json'
FORMAT = 'kevflix-app-manifest-1'
COVERED_DIRS = ('relay', 'actor-tts', 'web', 'tv', 'scripts', 'runtime')       # unlisted files here are refused
PACKAGED_DIRS = ('relay', 'actor-tts', 'web', 'tv', 'scripts', 'docs', 'installer')
PACKAGED_ROOT_FILES = ('config.example.json', 'requirements.txt', 'THIRD-PARTY.md', 'README.md', 'package.json',
                       'Setup-Worker.ps1', 'Setup-FFmpeg.ps1', 'Create-Desktop-Shortcut.ps1', 'Open-KevFLIX-PC.cmd',
                       'Start-KevFLIX.cmd', 'Start-KevFLIX-Setup.cmd', 'Stop-KevFLIX.cmd', 'Allow-KevFLIX-LAN.cmd',
                       'Allow-KevFLIX-LAN.ps1')
PACKAGED_EXTRA = ('legacy/scripts/render_sapi.ps1',)
EXCLUDED_PARTS = ('.git', '.github', '__pycache__', '.venv', 'state', 'cache', 'models', 'dist', 'receipts', 'runs',
                  'scratch', 'logs', 'recordings', 'voice-compare', 'measurements', 'private-activation', 'private',
                  'setup-records', 'setup-backups', 'node_modules', '.pytest_cache')
EXCLUDED_SUFFIXES = ('.pyc', '.pyo', '.tmp', '.log', '.sqlite3', '.sqlite3-wal', '.sqlite3-shm', '.db',
                     # media and recordings
                     '.mkv', '.mp4', '.m4v', '.avi', '.mov', '.webm', '.ts', '.m2ts', '.wav', '.flac', '.mp3', '.aac',
                     '.m4a', '.opus', '.ogg', '.srt', '.ass', '.ssa',
                     # model weights (provisioned separately)
                     '.gguf', '.safetensors', '.ckpt', '.pt', '.pth', '.onnx', '.bin', '.th')
EXCLUDED_NAMES = ('relay-config.json', '.env', 'kevflix-install.txt', 'proof-device-session.json',
                  'desktop-update-result.json', MANIFEST_NAME, BUILD_NAME)
SECRET_MARKERS = ('token', 'secret', 'credential', 'password', '.pem', '.pfx', '.key')
# Internal coordination records: never product files. EXACT relative paths only (compared case-insensitively, as on
# Windows); product docs under docs/ are unaffected. Root entries are already outside PACKAGED_ROOT_FILES and are
# listed so that a later change to that list can never ship them.
INTERNAL_RECORDS = frozenset(p.lower() for p in (
    'docs/INTEGRATION-V2.md', 'docs/V2-SOURCE-RECEIPT.json',
    'PLAN.md', 'OWNERSHIP.md', 'OWNERSHIP-V2.md', 'ASSEMBLY-BASE.json', 'INTERFACES.md', 'V3-DIRECTION.md'))
# Text that must never be packaged (development locations); a hit fails the build with the file list.
DEV_MARKERS = ('sem-migration-staging', 'candidate-beta-hour', 'runtime-20261017', 'kevflix-beta\\runtime-',
               'kevflix-beta/runtime-', 'proof-hls', 'player-seek-reload-v1', 'movie-dub-relay\\.venv',
               'movie-dub-relay/.venv')
TEXT_SUFFIXES = ('.py', '.js', '.mjs', '.html', '.htm', '.css', '.json', '.md', '.ps1', '.cmd', '.bat', '.cs',
                 '.txt', '.xml', '.cjs', '.c', '.cpp', '.h', '.hpp')
NOT_COVERED = ['.venv (Setup-Worker.ps1 from requirements.txt; pinned direct versions, transitive versions recorded '
               'in <profile>/state/installed-requirements.txt by setup)',
               'models (scripts/provision_models.py; upstream licenses/gated terms)',
               'FFmpeg (Setup-FFmpeg.ps1)',
               'Vox engine runtime and model (paths from the user config: actor_runtime_manifest)',
               'X-Translator WSL environments, models and private phonemizer (explicit xt_provision installation)',
               'the user profile (settings, state, cache, actor bank)']


def sha256_file(path, block=4 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(rel):
    parts = [p.lower() for p in rel.parts]
    name = parts[-1]
    if any(part in EXCLUDED_PARTS for part in parts[:-1]):
        return 'excluded folder'
    if name in EXCLUDED_NAMES:
        return 'private or generated file'
    if any(name.endswith(s) for s in EXCLUDED_SUFFIXES):
        return 'media, model, database or generated file'
    stem = name.rsplit('.', 1)[0]
    if any(m in stem for m in SECRET_MARKERS) and not name.endswith(('.py', '.js', '.md', '.cs', '.html', '.css')):
        return 'possible credential file'
    return None


def select_package_files(source_root):
    """THE packaging rule (build): [(relative posix path, absolute path)], deterministic, plus the excluded list."""
    source_root = Path(source_root)
    chosen, skipped = [], []
    for path in sorted(source_root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(source_root)
        posix = rel.as_posix()
        wanted = rel.parts[0] in PACKAGED_DIRS or posix in PACKAGED_ROOT_FILES or posix in PACKAGED_EXTRA
        if not wanted:
            continue
        if posix.lower() in INTERNAL_RECORDS:
            skipped.append({'path': posix, 'reason': 'internal coordination record'})
            continue
        reason = _excluded(rel)
        if reason:
            skipped.append({'path': posix, 'reason': reason})
            continue
        chosen.append((posix, path))
    return chosen, skipped


# Reviewed exceptions: PROTECTION rules (denylists of Sem locations), not operational dependencies. A hit is waived
# only for these exact source strings (lower-case, as they appear in the file text) and, for a pinned upstream
# file, only while the file still has its pinned sha256 (it must not be edited cosmetically). Any other occurrence
# of a dev marker in the same file still fails the build.
DEV_REFERENCE_ALLOWLIST = {
    'relay/app_runtime/lease/lease_client.py': {
        'sha256': '5cbac8a4d33c114bf811bab50e5957142dfa46fa8e749ab482bf78749fefcd29',
        'why': 'lease v2 absent-mode Sem-process detection denylist (SEM_PROCESS_RULES exe_roots / cmd_markers)',
        'exact': ('c:\\\\sem-migration-staging\\\\sem-5.0\\\\', 'c:\\\\sem-migration-staging\\\\sem-4.0\\\\',
                  '\\\\sem-migration-staging\\\\sem-5.0\\\\', '\\\\sem-migration-staging\\\\sem-4.0\\\\')},
    'relay/app_runtime/vox_provision.py': {
        'sha256': None,
        'why': 'refusal list: Sem private roots are never used as a Vox provisioning source',
        'exact': ("'c:\\\\sem-migration-staging\\\\sem-'",)},
    'tv/app/player.js': {
        'sha256': '745741e0bf22b9a18bac383643f325f999d03822c65c657593345e1daa94d7cd',
        'why': 'pinned 6c0dd972 player release bytes: line-73 provenance comment naming its fix package',
        'exact': ('// player-seek-reload-v1: loading is not playing.',)},
    'relay/beta_stable_gate.py': {
        'sha256': 'e9a5be1d7b7090ea3bdac5cbc600b7aa0d748039760076fa3116c25755898bdf',
        'why': 'pinned legacy-compatibility DEFAULT_ROOTS; the installed app passes explicit allowed_roots and '
               'reference_cache_module and never uses these defaults',
        'exact': ("default_roots = ('c:/sem-migration-staging/movie-dub-relay', 'c:/sem-data/movie-dub-cache')",)},
    'relay/app_runtime/integrity.py': {
        'sha256': None,
        'why': 'this module defines the marker and allowlist strings themselves',
        'exact': ()},
}


def dev_reference_hits(files):
    """[(posix, absolute)] -> files whose text names a development location (the build refuses them), after
    removing only the exact reviewed allowlist strings of that file."""
    hits = []
    for posix, path in files:
        if not posix.lower().endswith(TEXT_SUFFIXES):
            continue
        raw = Path(path).read_bytes()
        text = raw.decode('utf-8', 'replace').lower()
        allowed = DEV_REFERENCE_ALLOWLIST.get(posix)
        if posix == 'relay/app_runtime/integrity.py':
            # only the marker/allowlist definitions may name markers: strip those two blocks, scan the rest
            start, end = text.find('dev_markers = ('), text.find('def dev_reference_hits')
            text = text[:start] + text[end:] if 0 <= start < end else text
        elif allowed:
            pinned = allowed.get('sha256')
            if pinned is None or hashlib.sha256(raw).hexdigest() == pinned:
                for exact in allowed['exact']:
                    text = text.replace(exact, '')
        found = [m for m in DEV_MARKERS if m in text]
        if found:
            hits.append({'path': posix, 'markers': found,
                         'allowlist': 'void: pinned sha256 differs' if allowed and allowed.get('sha256') and
                         hashlib.sha256(raw).hexdigest() != allowed['sha256'] else None})
    return hits


def build_manifest(entries, *, version, base_revision=None, status='unpublished local build'):
    """entries: [(posix, bytes_or_path)]. Returns (manifest dict, manifest bytes, sha256). The build's content
    identity IS the returned sha256; base_revision only names the publication base it was assembled from."""
    files = {}
    for posix, item in sorted(entries, key=lambda e: e[0]):
        if isinstance(item, (bytes, bytearray)):
            files[posix] = {'sha256': hashlib.sha256(item).hexdigest(), 'size': len(item)}
        else:
            files[posix] = {'sha256': sha256_file(item), 'size': Path(item).stat().st_size}
    manifest = {'format': FORMAT, 'product': 'KevFLIX', 'version': version, 'status': status,
                'base_revision': base_revision, 'base_revision_note': 'publication base only, not this build\'s source',
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'hash': 'sha256',
                'covered_dirs': list(COVERED_DIRS), 'not_covered': list(NOT_COVERED), 'files': files}
    data = json.dumps(manifest, indent=1, sort_keys=True).encode('utf-8')
    return manifest, data, hashlib.sha256(data).hexdigest()


def verify(install_root):
    """Startup check. Returns (manifest or None, problems). Any problem refuses the start."""
    root = Path(install_root)
    problems = []
    build_path, manifest_path = root / BUILD_NAME, root / MANIFEST_NAME
    if not build_path.is_file() or not manifest_path.is_file():
        return None, ['not an installed KevFLIX build: %s or %s is missing (the integrity manifest is created by '
                      'scripts/build_installer.py and installed by KevFLIX Setup)' % (BUILD_NAME, MANIFEST_NAME)]
    build = json.loads(build_path.read_text(encoding='utf-8-sig'))
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != build.get('manifest_sha256'):
        return None, ['%s does not match the manifest_sha256 recorded in %s' % (MANIFEST_NAME, BUILD_NAME)]
    manifest = json.loads(raw.decode('utf-8'))
    # The build identity is the manifest digest itself (content-addressed); version and base must agree too.
    if manifest.get('format') != FORMAT or manifest.get('version') != build.get('version') \
            or build.get('revision') != build.get('manifest_sha256') \
            or manifest.get('base_revision') != build.get('base_revision'):
        problems.append('manifest identity differs from the installed build identity')
    files = manifest.get('files') or {}
    for posix, entry in sorted(files.items()):
        path = root / posix
        try:
            if not path.is_file():
                problems.append('missing: %s' % posix)
            elif path.stat().st_size != entry.get('size') or sha256_file(path) != entry.get('sha256'):
                problems.append('changed: %s' % posix)
        except OSError as error:
            problems.append('unreadable: %s (%s)' % (posix, error))
        if len(problems) > 50:
            problems.append('... more problems not listed')
            return manifest, problems
    listed = set(files)
    for top in manifest.get('covered_dirs') or COVERED_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for path in base.rglob('*'):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if '__pycache__' in rel.parts or path.suffix.lower() in ('.pyc', '.pyo'):
                continue
            if rel.as_posix() not in listed:
                problems.append('unlisted file in %s/: %s' % (top, rel.as_posix()))
                if len(problems) > 50:
                    return manifest, problems
    return manifest, problems


def package_version(source_root):
    return json.loads((Path(source_root) / 'package.json').read_text(encoding='utf-8-sig'))['version']


def relay_version(source_root):
    """relay/__init__.py __version__, read as text (the build never imports the app)."""
    import re
    text = (Path(source_root) / 'relay' / '__init__.py').read_text(encoding='utf-8')
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return match.group(1) if match else None
