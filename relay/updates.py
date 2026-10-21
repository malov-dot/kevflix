"""GitHub-delivered TV interface updates. No Plex or Sem code is changed."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from urllib import request, error, parse

from . import __version__
from .media import atomic_json

FILES = ("index.html", "player.css", "player.js")
# Program player additions (tv/app, W10). The movie worker's player REQUIRES them: a release is used only when its
# manifest pins every one of REQUIRED_FILES and they are present with those hashes.
PROGRAM_FILES = ("no-native-hls.js", "program-audio.js", "program-status.js", "youtube.js", "controls.js", "controls.css")
REQUIRED_FILES = FILES + PROGRAM_FILES
BUNDLED_PACKAGE = Path(__file__).resolve().parent.parent / "tv" / "package.json"


def _bundled_metadata():
    try:
        data = json.loads(BUNDLED_PACKAGE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    player = data.get("semPlayer") or {}
    return {"version": str(data.get("version") or "0"), "capabilities": list(player.get("capabilities") or [])}


def version_key(text):
    """'0.4.8' > '0.4.8-beta.5' > '0.4.7'. Unparseable -> lowest."""
    main, _, pre = str(text or "").partition("-")
    try:
        numbers = tuple(int(part) for part in main.split("."))
    except ValueError:
        return ((-1,), 0, ())
    # Numeric prerelease identifiers precede textual identifiers; tagged values also avoid int/string comparison
    # errors when checking a minimum worker such as 0.4.11-1 against an installed 0.4.11-beta.1.
    pre_key = tuple((0, int(p)) if p.isdigit() else (1, p) for p in pre.split(".")) if pre else ()
    return (numbers, 0 if pre else 1, pre_key)


def worker_incompatibility(api_version, minimum_worker):
    """The fetched/saved player's declared worker contract, checked against this installed worker.

    Missing metadata is not compatibility evidence. Older cached releases without these fields stay on disk while
    the known bundled player is used; fetching a package cannot silently invent the missing requirement.
    """
    if isinstance(api_version, bool) or not isinstance(api_version, int) or api_version != 2:
        return "player API %r is not supported by this worker (API 2)" % (api_version,)
    if not isinstance(minimum_worker, str) or not re.fullmatch(
            r"\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?", minimum_worker):
        return "minimumWorker is missing or is not a supported version"
    if version_key(__version__) < version_key(minimum_worker):
        return "requires worker %s or newer; installed worker is %s" % (minimum_worker, __version__)
    return None


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Update redirects are not permitted")


class PlayerUpdates:
    def __init__(self, config, tv):
        self.config, self.tv = config, tv
        self.directory = config.state / "player-ui"
        self.bundled = Path(__file__).resolve().parent.parent / "tv" / "app"
        bundled = _bundled_metadata()
        self.bundled_version = bundled["version"]
        self.required_capabilities = bundled["capabilities"]
        self.stop_event = threading.Event()
        self.thread = None
        self.last_error = None
        self.pending = None
        self.active = "bundled"
        self.blocked = []
        self.ignored = {}
        self.opener = request.build_opener(NoRedirect)
        pointer = self.directory / "active.json"
        if pointer.is_file():
            try:
                saved = json.loads(pointer.read_text())
                revision = saved["revision"]
                self.blocked = saved.get("blocked", [])
                if revision != "bundled":
                    problem = self.incompatibility(revision)
                    if problem is None:
                        self.active = revision
                    else:
                        # Kept on disk (files, pointer and pairing untouched); the bundled program player is used.
                        self.ignored[revision] = problem
                        self.last_error = "Saved player release %s not used: %s" % (revision[:12], problem)
            except (OSError, ValueError, KeyError):
                self.last_error = "Saved player update unavailable; using the bundled interface"

    def valid_release(self, revision):
        """Every file the manifest pins (and at least the base FILES) is present with its sha256."""
        if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            return False
        root = self.directory / "releases" / revision
        try:
            manifest = json.loads((root / "manifest.json").read_text())
            pinned = manifest["sha256"]
            return all(name in pinned for name in FILES) and all(
                hashlib.sha256((root / name).read_bytes()).hexdigest() == digest for name, digest in pinned.items())
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def incompatibility(self, revision):
        """None when the movie worker may activate this release; else why not. Required: valid hashes, EVERY
        REQUIRED_FILES entry pinned and present, a supported API/minimumWorker contract, every required capability,
        and a version not older than bundled."""
        if not self.valid_release(revision):
            return "files missing or changed"
        root = self.directory / "releases" / revision
        manifest = json.loads((root / "manifest.json").read_text())
        problem = worker_incompatibility(manifest.get("apiVersion"), manifest.get("minimumWorker"))
        if problem is not None:
            return problem
        missing = [name for name in REQUIRED_FILES if name not in (manifest.get("sha256") or {})]
        if missing:
            return "program player files not pinned: %s" % ", ".join(missing)
        lacking = [c for c in self.required_capabilities if c not in (manifest.get("capabilities") or [])]
        if lacking:
            return "lacks capabilities %s" % ", ".join(lacking)
        if version_key(manifest.get("version")) < version_key(self.bundled_version):
            return "version %s is older than the bundled %s" % (manifest.get("version"), self.bundled_version)
        return None

    def file(self, revision, name):
        if name not in REQUIRED_FILES:
            raise ValueError("Unknown player asset")
        if revision == "bundled":
            return self.bundled / name
        if self.incompatibility(revision) is not None:
            raise ValueError("Player update is unavailable, changed or not a program player")
        return self.directory / "releases" / revision / name

    def status(self):
        return {"enabled": self.config["updates"]["enabled"], "active": self.active,
                "pending": self.pending, "error": self.last_error, "bundled_version": self.bundled_version,
                "required_capabilities": list(self.required_capabilities), "ignored": dict(self.ignored),
                "scope": "TV interface only; engine, dependencies and Sem integration require separate updates"}

    def start(self):
        if self.config["updates"]["enabled"]:
            self.thread = threading.Thread(target=self.loop, name="player-ui-updates", daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)

    def get(self, path):
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "Sem-Movie-Player-Updater/0.3.0",
                   "X-GitHub-Api-Version": "2022-11-28"}
        token = os.environ.get(self.config["updates"]["token_env"])
        if token:
            headers["Authorization"] = "Bearer " + token
        req = request.Request("https://api.github.com" + path, headers=headers)
        try:
            with self.opener.open(req, timeout=12) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError("Update response exceeds the allowed size")
                return json.loads(raw)
        except error.HTTPError as exc:
            raise ValueError("GitHub update returned HTTP %s; private repositories need a read-only token" % exc.code) from None

    def contents(self, repository, path, revision):
        value = self.get("/repos/" + repository + "/contents/" + parse.quote(path, safe="/") +
                         "?ref=" + parse.quote(revision, safe=""))
        if value.get("type") != "file" or value.get("encoding") != "base64":
            raise ValueError("Unexpected update asset")
        content = base64.b64decode(value["content"])
        if len(content) > 512 * 1024:
            raise ValueError("Player asset exceeds its size limit")
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != value.get("sha"):
            raise ValueError("GitHub player asset identity mismatch")
        return content

    def fetch_update(self):
        settings = self.config["updates"]
        repository, branch, prefix = settings["repository"], settings["branch"], settings["path"].strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid update repository")
        if any(part in ("", ".", "..") for part in prefix.split("/")):
            raise ValueError("Invalid update path")
        commit = self.get("/repos/" + repository + "/commits/" + parse.quote(branch, safe=""))
        revision = commit["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Invalid update commit")
        if revision == self.active or revision == self.pending or revision in self.blocked or revision in self.ignored:
            return
        metadata = json.loads(self.contents(repository, prefix + "/package.json", revision))
        player = metadata.get("semPlayer") or {}
        if not isinstance(player, dict):
            raise ValueError("Player semPlayer metadata is not an object; existing player retained")
        problem = worker_incompatibility(player.get("apiVersion"), player.get("minimumWorker"))
        if problem is not None:
            self.ignored[revision] = problem
            raise ValueError("Player update %s ignored: %s; existing player retained" % (revision[:12], problem))
        lacking = [c for c in self.required_capabilities if c not in (player.get("capabilities") or [])]
        if lacking:
            self.ignored[revision] = "lacks capabilities %s" % ", ".join(lacking)
            raise ValueError("Player update %s is not a program player (%s); bundled player retained"
                             % (revision[:12], self.ignored[revision]))
        if version_key(metadata.get("version")) < version_key(self.bundled_version):
            self.ignored[revision] = "version %s is older than the bundled %s" % (metadata.get("version"),
                                                                                 self.bundled_version)
            raise ValueError("Player update %s ignored: %s" % (revision[:12], self.ignored[revision]))
        # ALL required files or nothing: download every one before anything is written.
        files = {name: self.contents(repository, prefix + "/app/" + name, revision) for name in REQUIRED_FILES}
        releases = self.directory / "releases"
        if releases.is_dir() and len([p for p in releases.iterdir() if not p.name.startswith(".")]) >= 32:
            raise ValueError("Player update cache limit reached; remove old inactive UI releases manually")
        root = releases / revision
        if root.exists():
            if self.incompatibility(revision) is None:
                self.pending = revision
                return
            raise ValueError("A different cached copy of player release %s exists; it is kept and not used" % revision[:12])
        staging = releases / (".staging-%s-%d" % (revision, os.getpid()))
        staging.mkdir(parents=True, exist_ok=False)
        manifest = {"revision": revision, "version": metadata["version"], "repository": repository,
                    "apiVersion": player["apiVersion"], "minimumWorker": player["minimumWorker"],
                    "capabilities": list(player.get("capabilities") or []),
                    "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        for name, content in files.items():
            (staging / name).write_bytes(content)
        atomic_json(staging / "manifest.json", manifest)
        for name in REQUIRED_FILES:              # verify the staged bytes against the manifest before activation
            if hashlib.sha256((staging / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
                raise ValueError("Staged player file %s does not match its manifest; release not activated" % name)
        os.replace(staging, root)                # one atomic rename: the release appears complete or not at all
        if self.incompatibility(revision) is not None:
            raise ValueError("Downloaded player release %s failed verification; not activated" % revision[:12])
        self.pending = revision

    def promote(self):
        if not self.pending:
            return
        with self.tv.lock:
            if self.tv.sessions:
                return
        problem = self.incompatibility(self.pending)
        if problem is not None:
            self.ignored[self.pending] = problem
            self.pending = None
            raise ValueError("Downloaded player update not activated: %s" % problem)
        atomic_json(self.directory / "active.json", {"revision": self.pending, "previous": self.active, "blocked": self.blocked})
        self.active, self.pending = self.pending, None

    def rollback(self):
        with self.tv.lock:
            if self.tv.sessions:
                raise ValueError("Stop the TV player session before rolling back its interface")
        pointer = self.directory / "active.json"
        previous = json.loads(pointer.read_text()).get("previous", "bundled") if pointer.is_file() else "bundled"
        if previous != "bundled" and self.incompatibility(previous) is not None:
            previous = "bundled"
        if self.active != "bundled":
            self.blocked = list(dict.fromkeys(self.blocked + [self.active]))
        atomic_json(pointer, {"revision": previous, "previous": "bundled", "blocked": self.blocked})
        self.active, self.pending = previous, None
        return self.status()

    def loop(self):
        remaining = 0
        while not self.stop_event.is_set():
            try:
                if remaining <= 0:
                    self.fetch_update()
                    remaining = max(300, int(self.config["updates"]["interval_seconds"]))
                self.promote()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                remaining = max(300, int(self.config["updates"]["interval_seconds"]))
            self.stop_event.wait(30)
            remaining -= 30
