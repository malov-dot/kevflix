from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from urllib import error, parse, request

from .language import iso
from .streaming import ticket
from . import __version__


class NoCrossOriginRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        before, after = parse.urlparse(req.full_url), parse.urlparse(newurl)
        if (before.scheme, before.netloc) != (after.scheme, after.netloc):
            raise ValueError("Plex redirected to a different origin; credentials were not forwarded")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Plex:
    def __init__(self, config):
        self.config = config
        self.command_id = int(time.time())
        self.lock = threading.Lock()
        self.opener = request.build_opener(NoCrossOriginRedirect)
        self.headers = {
            "Accept": "application/json", "X-Plex-Token": config["plex"]["token"],
            "X-Plex-Client-Identifier": "kevflix",
            "X-Plex-Product": "KevFLIX", "X-Plex-Version": __version__,
            "X-Plex-Platform": "Windows",
        }

    def call(self, path, params=None, method="GET", target=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid Plex API path")
        if not self.config["plex"]["url"] or not self.config["plex"]["token"]:
            raise ValueError("Plex URL and token are not configured")
        url = self.config["plex"]["url"].rstrip("/") + path
        if params:
            url += "?" + parse.urlencode(params)
        headers = dict(self.headers)
        if target:
            headers["X-Plex-Target-Client-Identifier"] = target
        req = request.Request(url, headers=headers, method=method,
                              data=b"" if method in ("POST", "PUT") else None)
        try:
            with self.opener.open(req, timeout=20) as response:
                raw = response.read(8 * 1024 * 1024)
                return json.loads(raw) if raw.strip().startswith(b"{") else {}
        except error.HTTPError as exc:
            raise RuntimeError(f"Plex returned HTTP {exc.code} for {path}") from None
        except error.URLError:
            raise RuntimeError("The configured Plex server could not be reached") from None

    @staticmethod
    def entries(result, name):
        value = result.get("MediaContainer", {}).get(name, [])
        return value if isinstance(value, list) else [value] if value else []

    def sections(self):
        return self.entries(self.call("/library/sections"), "Directory")

    def library(self, section, query="", start=0):
        if not str(section).isdigit():
            raise ValueError("Choose a Plex library section")
        values = {"X-Plex-Container-Start": start, "X-Plex-Container-Size": 100,
                  "includeMedia": 1}
        if query:
            values["title"] = query
        return self.entries(self.call(f"/library/sections/{section}/all", values), "Metadata")

    def metadata(self, key):
        if not str(key).isdigit():
            raise ValueError("Invalid Plex item ID")
        values = self.entries(self.call(f"/library/metadata/{key}", {"includeMedia": 1}), "Metadata")
        if not values:
            raise ValueError("Plex media item was not found")
        return values[0]

    def children(self, key):
        if not str(key).isdigit():
            raise ValueError("Invalid Plex item ID")
        return self.entries(self.call(f"/library/metadata/{key}/children"), "Metadata")

    def clients(self):
        return self.entries(self.call("/clients"), "Server")

    def sessions(self):
        return self.entries(self.call("/status/sessions"), "Metadata")

    def control(self, action, params=None):
        if not self.config["tv"]["allow_plex_controls"]:
            raise ValueError("Plex playback controls are disabled; use the separate TV player")
        if action not in ("pause", "play", "stop", "seekTo", "playMedia", "setStreams"):
            raise ValueError("Unsupported Plex playback control")
        target = self.config["plex"]["client_id"]
        if not target:
            raise ValueError("Choose the target TV client in configuration first")
        with self.lock:
            self.command_id += 1
            number = self.command_id
        return self.call("/player/playback/" + action,
                         {"type": "video", "commandID": number, **(params or {})},
                         target=target)

    def play(self, key, position=0, force_transcode=False, media_index=None):
        identity = self.call("/").get("MediaContainer", {}).get("machineIdentifier")
        server = parse.urlparse(self.config["plex"]["url"])
        result = self.call("/playQueues", {
            "type": "video", "uri": f"server://{identity}/com.plexapp.plugins.library/library/metadata/{key}",
            "shuffle": 0, "repeat": 0, "includeChapters": 1}, "POST")
        queue = result.get("MediaContainer", {}).get("playQueueID")
        if not queue:
            raise ValueError("Plex did not return a playback queue")
        params = {"key": f"/library/metadata/{key}", "containerKey": f"/playQueues/{queue}?own=1&window=200",
                  "machineIdentifier": identity, "address": server.hostname,
                  "port": server.port or (443 if server.scheme == "https" else 80),
                  "protocol": server.scheme, "offset": max(0, round(position * 1000)),
                  "token": self.config["plex"]["token"]}
        if force_transcode:
            params.update({"directPlay": 0, "directStream": 0, "audioCodec": "aac",
                           "audioChannels": 2})
        if media_index is not None:
            params["mediaIndex"] = int(media_index)
        return self.control("playMedia", params)

    def refresh(self):
        section = self.config["plex_publish_section"]
        if not str(section).isdigit():
            raise ValueError("Configure the derived-media Plex library section")
        values = {"path": self.config["plex_publish_path"]} if self.config["plex_publish_path"] else None
        return self.call(f"/library/sections/{section}/refresh", values)


def selected_part(metadata):
    media = metadata.get("Media", [])
    if not media:
        raise ValueError("Plex item has no playable media")
    choice = next((x for x in media if x.get("selected")), media[0])
    parts = choice.get("Part", [])
    if len(parts) != 1:
        raise ValueError("Select a single-file movie version; stacked multipart media is not supported")
    part = parts[0]
    audio = [s for s in part.get("Stream", []) if int(s.get("streamType", 0)) == 2]
    if not audio:
        raise ValueError("Plex did not describe the audio streams")
    selected = next((x for x in audio if x.get("selected")), audio[0])
    return part, selected, audio


class PlexController:
    """Read-only observation of an explicitly configured Plex client."""
    def __init__(self, config, store, coordinator):
        self.config, self.store, self.coordinator = config, store, coordinator
        self.plex = Plex(config)
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.thread = None
        self.last_error = None
        self.watch_enabled = False

    def start(self):
        if self.config["plex"]["client_id"] and self.config["plex"]["token"]:
            self.thread = threading.Thread(target=self.loop, name="plex-observer", daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=25)

    def enqueue(self, key, mode=None, force=False):
        metadata = self.plex.metadata(key)
        part, audio, streams = selected_part(metadata)
        source = self.config.media_path(part["file"], nas=True)
        subtitle = next((s for s in part.get("Stream", []) if int(s.get("streamType", 0)) == 3
                         and s.get("selected") and s.get("index") is not None), None)
        return self.coordinator.enqueue(source, audio_index=int(audio["index"]),
            subtitle_index=int(subtitle["index"]) if subtitle else None,
            mode=mode or "prepared", title=metadata.get("title"), force_redub=force,
            plex={"rating_key": str(key), "part_id": str(part["id"]), "nas_path": part["file"]})

    def loop(self):
        previous = {}
        while not self.stop_event.is_set():
            try:
                seen = set()
                for item in self.plex.sessions():
                    player = item.get("Player", {})
                    if player.get("machineIdentifier") != self.config["plex"]["client_id"]:
                        continue
                    owner = self.config["plex"]["owner_id"]
                    if owner and str(item.get("User", {}).get("id", "")) != str(owner):
                        continue
                    ident = str(item.get("Session", {}).get("id") or item.get("sessionKey"))
                    part, audio, _ = selected_part(item)
                    position = float(item.get("viewOffset", 0)) / 1000
                    old = previous.get(ident)
                    state = player.get("state", "unknown")
                    expected = old["position"] + (time.time() - old["observed"]) if old and old["state"] == "playing" else old["position"] if old else position
                    epoch = old.get("epoch", 0) if old else 0
                    if old and (abs(position - expected) > 3 or old["rating_key"] != str(item["ratingKey"])):
                        epoch += 1
                    snapshot = {"source_kind": "plex", "session_id": ident,
                        "client_id": player["machineIdentifier"], "title": item.get("title"),
                        "rating_key": str(item["ratingKey"]), "part_id": str(part["id"]),
                        "nas_path": part["file"], "audio_index": int(audio["index"]), "state": state,
                        "position": position, "duration": float(item.get("duration", 0)) / 1000,
                        "observed": time.time(), "epoch": epoch, "transcode_session": ""}
                    self.store.put_record("sessions", ident, snapshot)
                    previous[ident] = snapshot
                    seen.add(ident)
                for record in self.store.records("sessions"):
                    if record["payload"].get("source_kind") != "tizenbrew" and record["id"] not in seen:
                        self.store.expire_record("sessions", record["id"])
                        previous.pop(record["id"], None)
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
            self.wake.wait(max(2, float(self.config["plex"]["poll_seconds"])))
            self.wake.clear()
