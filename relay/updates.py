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

from .media import atomic_json

FILES = ("index.html", "player.css", "player.js")


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Update redirects are not permitted")


class PlayerUpdates:
    def __init__(self, config, tv):
        self.config, self.tv = config, tv
        self.directory = config.state / "player-ui"
        self.bundled = Path(__file__).resolve().parent.parent / "tv" / "app"
        self.stop_event = threading.Event()
        self.thread = None
        self.last_error = None
        self.pending = None
        self.active = "bundled"
        self.blocked = []
        self.opener = request.build_opener(NoRedirect)
        pointer = self.directory / "active.json"
        if pointer.is_file():
            try:
                saved = json.loads(pointer.read_text())
                revision = saved["revision"]
                self.blocked = saved.get("blocked", [])
                if self.valid_release(revision):
                    self.active = revision
            except (OSError, ValueError, KeyError):
                self.last_error = "Saved player update unavailable; using the bundled interface"

    def valid_release(self, revision):
        if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            return False
        root = self.directory / "releases" / revision
        manifest = json.loads((root / "manifest.json").read_text())
        return all(hashlib.sha256((root / name).read_bytes()).hexdigest() == manifest["sha256"][name] for name in FILES)

    def file(self, revision, name):
        if name not in FILES:
            raise ValueError("Unknown player asset")
        if revision == "bundled":
            return self.bundled / name
        if not self.valid_release(revision):
            raise ValueError("Player update is unavailable or changed")
        return self.directory / "releases" / revision / name

    def status(self):
        return {"enabled": self.config["updates"]["enabled"], "active": self.active,
                "pending": self.pending, "error": self.last_error,
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
        if revision == self.active or revision == self.pending or revision in self.blocked:
            return
        metadata = json.loads(self.contents(repository, prefix + "/package.json", revision))
        if metadata.get("semPlayer", {}).get("apiVersion") != 1:
            raise ValueError("This player update needs a newer worker; existing player retained")
        files = {name: self.contents(repository, prefix + "/app/" + name, revision) for name in FILES}
        existing = self.directory / "releases"
        if existing.is_dir() and len(list(existing.iterdir())) >= 32:
            raise ValueError("Player update cache limit reached; remove old inactive UI releases manually")
        root = self.directory / "releases" / revision
        root.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (root / name).write_bytes(content)
        atomic_json(root / "manifest.json", {"revision": revision, "version": metadata["version"],
                    "repository": repository, "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}})
        self.pending = revision

    def promote(self):
        if not self.pending:
            return
        with self.tv.lock:
            if self.tv.sessions:
                return
        if not self.valid_release(self.pending):
            raise ValueError("Downloaded player update is incomplete")
        atomic_json(self.directory / "active.json", {"revision": self.pending, "previous": self.active, "blocked": self.blocked})
        self.active, self.pending = self.pending, None

    def rollback(self):
        with self.tv.lock:
            if self.tv.sessions:
                raise ValueError("Stop the TV player session before rolling back its interface")
        pointer = self.directory / "active.json"
        previous = json.loads(pointer.read_text()).get("previous", "bundled") if pointer.is_file() else "bundled"
        if previous != "bundled" and not self.valid_release(previous):
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
