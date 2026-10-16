"""Download GitHub desktop releases; install on the next KevFLIX start only."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib import request
from urllib.parse import urlsplit

from . import __version__
from .media import atomic_json, file_hash

ROOT = Path(__file__).resolve().parent.parent
REPO = "malov-dot/kevflix"
MARKER = ROOT / "kevflix-install.txt"


def managed():
    return MARKER.is_file() and MARKER.read_text().startswith("KevFLIX\n")


def download(url, destination=None, maximum=256 * 1024 * 1024):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in ("api.github.com", "github.com"):
        raise ValueError("Unexpected release source")
    req = request.Request(url, headers={"User-Agent": "KevFLIX-Desktop-Updates", "Accept": "application/vnd.github+json"})
    # No credentials, private configuration or media data are sent to GitHub/CDN.
    with request.urlopen(req, timeout=30) as response:
        final = urlsplit(response.url)
        if final.scheme != "https" or not (final.hostname in ("api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com")):
            raise ValueError("Unexpected release download redirect")
        length = 0
        if destination:
            with destination.open("xb") as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    length += len(chunk)
                    if length > maximum:
                        raise ValueError("Release exceeds the allowed download size")
                    stream.write(chunk)
            return
        content = response.read(maximum + 1)
        if len(content) > maximum:
            raise ValueError("Release metadata is too large")
        return json.loads(content)


def apply_pending():
    """Called before starting a worker. Never stop a running application."""
    if os.name != "nt" or not managed():
        return False
    config = json.loads((ROOT / "relay-config.json").read_text(encoding="utf-8-sig"))
    if config.get("setup_only", True) or not config.get("desktop_updates", {}).get("enabled", True):
        return False
    directory = ROOT / "state/desktop-updates"
    pointer = directory / "pending.json"
    if not pointer.is_file():
        return False
    pending = json.loads(pointer.read_text())
    current = MARKER.read_text().splitlines()
    if len(current) > 2 and current[2] == pending.get("revision"):
        pointer.unlink()
        return False
    result = ROOT / "state/desktop-update-result.json"
    if result.is_file():
        previous = json.loads(result.read_text())
        if not previous.get("ok") and previous.get("revision") == pending.get("revision"):
            return False
    asset = str(pending.get("asset", ""))
    if not re.fullmatch(r"KevFLIX-Setup-\d+\.\d+\.\d+\.exe", asset):
        raise ValueError("Unexpected update package name")
    installer = directory / asset
    if not installer.is_file() or file_hash(installer) != pending.get("sha256"):
        raise ValueError("Downloaded update is incomplete or changed")
    subprocess.Popen([str(installer), "/silent", "/update", "/target", str(ROOT), "/wait-pid", str(os.getpid())],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    return True


class DesktopUpdates:
    def __init__(self, config):
        self.config = config
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.pending = None
        self.directory = ROOT / "state/desktop-updates"

    def status(self):
        return {"managed": managed(), "enabled": self.config.data.get("desktop_updates", {}).get("enabled", True),
                "pending": self.pending, "error": self.error,
                "policy": "Downloads automatically; installs on the next KevFLIX start. Active viewing is never interrupted."}

    def start(self):
        if managed():
            self.thread = threading.Thread(target=self.loop, name="desktop-updates", daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()

    def check(self):
        releases = download("https://api.github.com/repos/" + REPO + "/releases?per_page=10", maximum=2 * 1024 * 1024)
        candidates = [r for r in releases if not r.get("draft") and
                      (not r.get("prerelease") or self.config.data.get("desktop_updates", {}).get("preview", True))]
        for release in candidates:
            assets = {a["name"]: a for a in release.get("assets", [])}
            if "desktop-release.json" not in assets:
                continue
            metadata = download(assets["desktop-release.json"]["browser_download_url"], maximum=16384)
            version = str(metadata.get("version", ""))
            revision = str(metadata.get("revision", ""))
            if metadata.get("product") != "KevFLIX" or not re.fullmatch(r"\d+\.\d+\.\d+", version) or not re.fullmatch(r"[a-f0-9]{40}", revision):
                raise ValueError("Unexpected desktop release metadata")
            if tuple(map(int, version.split("."))) < tuple(map(int, __version__.split("."))):
                continue
            current = MARKER.read_text().splitlines()
            if len(current) > 2 and current[2] == revision:
                return
            name = "KevFLIX-Setup-" + version + ".exe"
            if metadata.get("asset") != name or name not in assets or not re.fullmatch(r"[a-f0-9]{64}", str(metadata.get("sha256", ""))):
                raise ValueError("Desktop release is missing its checked installer")
            if int(metadata.get("size", 0)) != int(assets[name].get("size", -1)) or int(metadata["size"]) > 256 * 1024 * 1024:
                raise ValueError("Unexpected installer size")
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / name
            if not target.exists() or file_hash(target) != metadata["sha256"]:
                temporary = self.directory / (name + "." + str(time.time_ns()) + ".download")
                try:
                    download(assets[name]["browser_download_url"], temporary)
                    if temporary.stat().st_size != metadata["size"] or file_hash(temporary) != metadata["sha256"]:
                        raise ValueError("Installer checksum mismatch; current installation retained")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            atomic_json(self.directory / "pending.json", metadata)
            self.pending = version
            return

    def loop(self):
        while not self.stop_event.is_set():
            try:
                if self.config.data.get("desktop_updates", {}).get("enabled", True):
                    self.check()
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            self.stop_event.wait(3600)
