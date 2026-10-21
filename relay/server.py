from __future__ import annotations

import argparse
import hmac
import json
import logging
import mimetypes
import os
import re
import signal
import secrets
import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__
from .config import Config
from .coordinator import Coordinator
from .media import file_hash
from .plex import PlexController
from .store import Store
from .streaming import VirtualAudio, byte_range, valid_ticket
from .watch import WatchContext
from .tv import TVService
from .updates import PlayerUpdates
from .settings import PCSettings
from .desktop_updates import DesktopUpdates

ROOT = Path(__file__).resolve().parent.parent
IDENT = re.compile(r"^[a-f0-9]{32}$")


class Application:
    def __init__(self, config):
        self.config = config
        config.state.mkdir(parents=True, exist_ok=True)
        config.cache.mkdir(parents=True, exist_ok=True)
        self.store = Store(config.state)
        self.jobs = Coordinator(config, self.store)
        self.plex = PlexController(config, self.store, self.jobs)
        self.tv = TVService(config, self.store, self.jobs, self.plex.plex)
        self.updates = PlayerUpdates(config, self.tv)
        self.watch = WatchContext(config, self.store, self.jobs, self.plex)
        self.publications = set()
        self.publication_threads = set()
        self.lock = threading.Lock()
        self.pc_launch_codes = {}
        self.settings = PCSettings(self)
        self.desktop_updates = DesktopUpdates(config)
        self._youtube, self._youtube_lock = None, threading.Lock()

    def youtube_host(self, create=True):
        """The single background YouTube host. Imported lazily: the local/Plex startup path never loads it."""
        with self._youtube_lock:
            if self._youtube is None and create:
                from .youtube_host import YouTubeHost
                self._youtube = YouTubeHost(self.config["youtube"])
            return self._youtube

    def start(self):
        if self.config["setup_only"]:
            return
        self.jobs.start()
        self.plex.start()
        self.tv.start()
        self.updates.start()
        self.desktop_updates.start()

    def close(self):
        host = self.youtube_host(create=False)
        if host is not None:
            host.close()
        self.tv.close()
        self.updates.close()
        self.desktop_updates.close()
        self.plex.close()
        self.jobs.close()
        for thread in list(self.publication_threads):
            thread.join(timeout=10)

    def public_job(self, job):
        if not job:
            raise KeyError("Job not found")
        result = dict(job)
        result["chunks"] = self.store.chunks(job["id"])
        result["speakers"] = {key: {k: v for k, v in value.items() if k != "embedding"}
                              for key, value in self.store.speakers(job["id"]).items()}
        return result

    def create_pc_launch(self):
        value = secrets.token_urlsafe(32)
        digest = hashlib.sha256(value.encode()).hexdigest()
        with self.lock:
            self.pc_launch_codes = {k: v for k, v in self.pc_launch_codes.items() if v > time.time()}
            self.pc_launch_codes[digest] = time.time() + 60
        return {"url": self.config["public_url"].rstrip("/") + "/#launch=" + value}

    def redeem_pc_launch(self, value):
        digest = hashlib.sha256(str(value).encode()).hexdigest()
        with self.lock:
            expires = self.pc_launch_codes.pop(digest, 0)
        if expires < time.time():
            raise ValueError("This desktop login link expired. Open the KevFLIX PC shortcut again.")
        return {"api_token": self.config["api_token"], "setup_only": self.config["setup_only"]}

    def publish(self, job_id):
        with self.lock:
            if job_id in self.publications:
                return
            self.publications.add(job_id)
        def work():
            try:
                self.store.update_job(job_id, publication_state="working", publication_error=None)
                result = self.jobs.publish(job_id)
                if self.config["plex_publish_path"]:
                    rel = Path(result["path"]).relative_to(self.config["publish_dir"])
                    result["nas_path"] = self.config["plex_publish_path"].rstrip("/") + "/" + rel.as_posix()
                    self.store.update_job(job_id, published=result)
                self.plex.plex.refresh()
                self.store.update_job(job_id, publication_state="published")
            except Exception as error:
                self.store.update_job(job_id, publication_state="error", publication_error=str(error))
            finally:
                with self.lock:
                    self.publications.discard(job_id)
                    self.publication_threads.discard(threading.current_thread())
        thread = threading.Thread(target=work, name="publish-derived-movie", daemon=True)
        with self.lock:
            self.publication_threads.add(thread)
        thread.start()

    def published_key(self, job_id):
        job = self.store.job(job_id)
        if not job or not job["payload"].get("published"):
            raise ValueError("Publish the completed job first")
        expected = job["payload"]["published"].get("nas_path")
        if not expected:
            raise ValueError("Configure plex_publish_path so the server can identify the derived media")
        items = self.plex.plex.library(self.config["plex_publish_section"], job["payload"]["title"])
        for item in items:
            for index, media in enumerate(item.get("Media", [])):
                if any(p.get("file") == expected for p in media.get("Part", [])):
                    return str(item["ratingKey"]), index
        raise ValueError("Plex has not indexed the generated copy yet; wait for its library scan")


class Handler(BaseHTTPRequestHandler):
    server_version = "KevFLIXWorker/" + __version__
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):
        # Never log query strings, stream tickets, media text or Plex credentials.
        path = urlparse(self.path).path
        logging.info("%s %s", self.command, "/hooks/plex/[redacted]" if path.startswith("/hooks/plex/") else path)

    def authorized(self, nas=False):
        provided = self.headers.get("Authorization", "")
        token = provided.removeprefix("Bearer ").strip()
        expected = self.app.config["nas_token" if nas else "api_token"]
        return provided.startswith("Bearer ") and hmac.compare_digest(token, expected)

    def cors(self):
        origin = self.headers.get("Origin", "")
        if origin in self.app.config["tv"]["allow_origins"]:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self):
        origin = self.headers.get("Origin", "")
        if origin not in self.app.config["tv"]["allow_origins"]:
            return self.json_response({"error": "Origin not allowed"}, 403)
        self.send_response(204)
        self.cors()
        self.send_header("Access-Control-Allow-Methods", "GET,POST,HEAD,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization,Content-Type,Range")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def tv_device(self):
        if not self.app.config["tv"]["enabled"]:
            raise ValueError("TV player is disabled")
        token = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        device = self.app.tv.device(token)
        if not device:
            raise ValueError("Pair this TV with the worker first")
        return device

    def json_response(self, value, code=200):
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def body(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size < 0 or size > 2 * 1024 * 1024:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(size)
        return json.loads(raw or b"{}")

    def do_HEAD(self):
        self.do_GET()

    def youtube_api(self, path, data):
        """/tv-api/youtube/{status,open,key,close}: the device token was already checked by the caller."""
        action, enabled = path.rsplit("/", 1)[-1], bool(self.app.config["youtube"]["enabled"])
        if self.app.config["setup_only"]:
            return self.json_response({"error": "Setup-only mode: YouTube is disabled"}, 423)
        host = self.app.youtube_host(create=False)
        if action == "status":
            status = host.status() if host else {"state": "stopped", "clients": 0, "frame": 0}
            status.pop("pid", None)
            status["enabled"] = enabled
            if not enabled:
                status["state"] = "disabled"
            return self.json_response(status)
        if not enabled:
            return self.json_response({"error": "YouTube is turned off (youtube.enabled)"}, 403)
        try:
            if action == "open":
                return self.json_response(self.app.youtube_host().open())
            if action == "key":
                if host is None:
                    raise RuntimeError("YouTube is not open")
                reply = {"ok": True}
                play = host.key(str(data["key"]), str(data.get("action", "press")))
                if play:
                    reply["play"] = play                          # the page just navigated to a video: the TV hands it to its own player
                return self.json_response(reply)
            if action == "resume":
                self.app.tv.youtube_cancel()
                return self.json_response(host.resume() if host is not None else {"state": "stopped", "handoff": False})
            if action == "cancel":
                return self.json_response({"cancelled": bool(self.app.tv.youtube_cancel())})
            if action == "close":
                if host is not None:
                    host.close()
                return self.json_response({"state": "stopped"})
        except RuntimeError as error:                      # HostError: Chrome missing / not open / dropped
            return self.json_response({"error": str(error)}, 409)
        return self.json_response({"error": "Unknown TV endpoint"}, 404)

    def youtube_screen(self, query):
        """Long-lived multipart/x-mixed-replace MJPEG of the latest frames. An <img> cannot send an Authorization header, so
        this also accepts the screen_ticket returned by /tv-api/youtube/open (valid while that Chrome instance runs)."""
        if self.app.config["setup_only"] or not self.app.config["youtube"]["enabled"]:
            return self.json_response({"error": "YouTube is turned off (youtube.enabled)"}, 403)
        if self.command == "HEAD":
            return self.json_response({"error": "GET only"}, 405)
        host = self.app.youtube_host(create=False)
        token = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not ((host is not None and host.ticket_valid(query.get("ticket"))) or (token and self.app.tv.device(token))):
            return self.json_response({"error": "Pair this TV with the worker first"}, 401)
        if host is None or host.state != "running":
            return self.json_response({"error": "YouTube is not open"}, 409)
        from .youtube_host import mjpeg_headers, stream_screen
        self.send_response(200)
        self.cors()
        for name, value in mjpeg_headers().items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True

        def write(block):
            self.wfile.write(block)
            self.wfile.flush()
        stream_screen(host, write, int(self.app.config["youtube"]["max_fps"]))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
        try:
            if self.app.config["setup_only"] and (path.startswith(("/tv-media/", "/stream/"))
                                                 or path == "/api/watch/context"):
                return self.json_response({"error": "Setup-only mode: playback and model processing are disabled"}, 423)
            if path in ("/tv", "/tv/"):
                self.send_response(302)
                self.send_header("Location", "/tv-ui/" + self.app.updates.active + "/index.html")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            # The program player (tv/app: derived from release 6c0dd972 with its seek-reload fix) adds three scripts.
            ui_asset = re.fullmatch(r"/tv-ui/(bundled|[0-9a-f]{40})/(index\.html|player\.js|player\.css|"
                                    r"no-native-hls\.js|program-audio\.js|program-status\.js|youtube\.js)", path)
            if ui_asset:
                return self.file(self.app.updates.file(ui_asset[1], ui_asset[2]), tv_asset=True)
            if path == "/tv-api/version":
                return self.json_response({"product": "KevFLIX", "version": __version__, "validation": "untested",
                                           "player_release": self.app.updates.active,
                                           "setup_only": self.app.config["setup_only"]})
            if path == "/tv-api/youtube/screen":
                return self.youtube_screen(query)
            if path.startswith("/tv-api/"):
                device = self.tv_device()
                if path == "/tv-api/youtube/status":
                    return self.youtube_api(path, {})
                if path == "/tv-api/library":
                    return self.json_response(self.app.tv.browse(
                        query.get("folder"), query.get("offset", 0), query.get("query", "")))
                if path == "/tv-api/session":
                    return self.json_response(self.app.tv.status(query["id"], device))
                if path == "/tv-api/probe/status":
                    return self.json_response(self.app.tv.probe.status())
                probe_clip = re.fullmatch(r"/tv-api/probe/clip/([a-z0-9_]{1,40})\.mp4", path)
                if probe_clip:
                    clip = self.app.tv.probe.path(probe_clip[1])
                    if clip is None:
                        return self.json_response({"error": "That test clip is not ready"}, 404)
                    return self.file(clip, "video/mp4")
                if path == "/tv-api/producer":
                    # Dubbing producer state (MoviePipeline): running / preempted / failed / stopped.
                    return self.json_response(self.app.jobs.producer_state())
                if path == "/tv-api/plex/sections":
                    return self.json_response(self.app.tv.plex_sections())
                if path == "/tv-api/plex/metadata":
                    return self.json_response(self.app.tv.catalog.metadata(query["key"]))
                if path == "/tv-api/plex/continue":
                    return self.json_response(self.app.tv.catalog.continuing(query["section"]))
                if path == "/tv-api/plex/library":
                    return self.json_response(self.app.tv.plex_library(**{k: v for k, v in query.items()
                        if k in ("section", "query", "offset", "genre", "year", "collection", "sort", "parent")}))
                if path == "/tv-api/plex/options":
                    return self.json_response(self.app.tv.catalog.options(query["section"], query["field"]))
                if path == "/tv-api/plex/alphabet":
                    return self.json_response(self.app.tv.catalog.alphabet(**{k: v for k, v in query.items()
                        if k in ("section", "query", "genre", "year", "collection")}))
                if path == "/tv-api/plex/poster":
                    content, kind = self.app.tv.plex_poster(query["key"])
                    return self.bytes_response(content, kind)
                if path == "/tv-api/settings":
                    return self.json_response({"plex_url": self.app.config["plex"]["url"],
                        "plex_configured": bool(self.app.config["plex"]["token"]),
                        "local_only": self.app.config["tv"]["local_only"], "pc_process_all": self.app.config["tv"]["pc_process_all"]})
                return self.json_response({"error": "Unknown TV endpoint"}, 404)
            tv_media = re.fullmatch(r"/tv-media/([a-f0-9]{32})/(index\.m3u8|[0-9]+\.ts|original|captions\.vtt)", path)
            if tv_media:
                session = self.app.tv.authorize_media(tv_media[1], query)
                if tv_media[2] == "original":
                    return self.ranged_file(self.app.config.media_path(session["source"]))
                if tv_media[2] == "captions.vtt":
                    return self.bytes_response(self.app.tv.captions(session), "text/vtt; charset=utf-8")
                if tv_media[2] == "index.m3u8":
                    content = self.app.tv.playlist(session)
                    self.send_response(200)
                    self.cors()
                    self.send_header("Content-Type", "application/vnd.apple.mpegurl")
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(content)
                    return
                return self.file(self.app.tv.segment(session, int(tv_media[2][:-3])), "video/mp2t")
            if path in ("/", "/app.js", "/style.css"):
                name = "index.html" if path == "/" else path[1:]
                return self.file(ROOT / "web" / name)
            stream = re.fullmatch(r"/stream/([a-f0-9]{32})/audio.wav", path)
            if stream:
                if not valid_ticket(self.app.config, stream[1], query):
                    return self.json_response({"error": "Invalid or expired stream ticket"}, 401)
                return self.audio(stream[1], int(query["revision"]))
            if not self.authorized():
                return self.json_response({"error": "Authentication required"}, 401)
            if path == "/api/status":
                return self.json_response({"version": __version__, "setup_only": self.app.config["setup_only"], "config": self.app.config.public(),
                    "jobs": self.app.store.jobs(), "sessions": self.app.store.records("sessions"),
                    "leases": self.app.store.records("leases"), "plex_error": self.app.plex.last_error,
                    "watch_enabled": self.app.plex.watch_enabled, "validation": "Unvalidated build; owner deferred testing"})
            if path == "/api/tv/devices":
                return self.json_response(self.app.tv.devices())
            if path == "/api/settings":
                return self.json_response(self.app.settings.view())
            if path == "/api/settings/plex/servers":
                return self.json_response(self.app.settings.resources())
            if path == "/api/settings/plex/folders":
                return self.json_response(self.app.settings.plex_folders())
            if path == "/api/settings/plex/link-status":
                return self.json_response(self.app.settings.link_status(query["id"]))
            if path == "/api/player-updates":
                return self.json_response(self.app.updates.status())
            if path == "/api/desktop-updates":
                return self.json_response(self.app.desktop_updates.status())
            if path == "/api/events":
                return self.json_response(self.app.store.events(int(query.get("after", 0))))
            if path == "/api/plex/sections":
                return self.json_response(self.app.plex.plex.sections())
            if path == "/api/plex/clients":
                return self.json_response(self.app.plex.plex.clients())
            if path == "/api/plex/library":
                return self.json_response(self.app.plex.plex.library(
                    query["section"], query.get("query", ""), int(query.get("start", 0))))
            if path == "/api/plex/children":
                return self.json_response(self.app.plex.plex.children(query["key"]))
            if path == "/api/watch/context":
                return self.json_response(self.app.watch.current(
                    query.get("image") == "1", query.get("audio") == "1"))
            job = re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path)
            if job:
                return self.json_response(self.app.public_job(self.app.store.job(job[1])))
            artifact = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/chunks/(\d+)/artifact", path)
            if artifact:
                kind = query.get("kind", "audio")
                if kind not in ("audio", "original", "bed"):
                    raise ValueError("Choose audio, original or bed")
                chunk = self.app.store.chunk(artifact[1], int(artifact[2]))
                if not chunk or not chunk["payload"].get(kind):
                    raise ValueError("That audio artifact is not available")
                p = Path(chunk["payload"][kind]).resolve()
                allowed = (self.app.config.cache / artifact[1]).resolve()
                if allowed not in p.parents:
                    raise ValueError("Artifact does not belong to this job")
                if kind == "audio" and file_hash(p) != chunk["payload"].get("sha256"):
                    raise ValueError("Audio changed since generation")
                return self.file(p, "audio/wav")
            self.json_response({"error": "Endpoint not found"}, 404)
        except (ValueError, KeyError, FileNotFoundError) as error:
            self.json_response({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.json_response({"error": str(error)}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if self.app.config["setup_only"] and not path.startswith("/api/settings") and path not in (
                    "/tv-api/pair", "/api/tv/pair-code", "/api/tv/revoke",
                    "/api/pc-launch", "/pc-api/redeem"):
                self.close_connection = True
                return self.json_response({"error": "Setup-only mode: only TV pairing is enabled; playback and model processing are disabled"}, 423)
            if path == "/pc-api/redeem":
                data = self.body()
                return self.json_response(self.app.redeem_pc_launch(data.get("code", "")))
            if path == "/tv-api/pair":
                if not self.app.config["tv"]["enabled"]:
                    raise ValueError("TV player is disabled")
                data = self.body()
                return self.json_response(self.app.tv.pair(data.get("code", ""),
                    data.get("name", "Samsung TV"), self.client_address[0]))
            if path.startswith("/tv-api/"):
                device, data = self.tv_device(), self.body()
                if path == "/tv-api/plex/watched":
                    return self.json_response(self.app.tv.catalog.watched(data["key"], data.get("enabled", False)))
                if path == "/tv-api/open":
                    if data.get("youtube"):
                        return self.json_response(self.app.tv.open_youtube(device, data["youtube"]))
                    if data.get("plex_key"):
                        return self.json_response(self.app.tv.open_plex(device, data["plex_key"], data.get("resume", True)))
                    return self.json_response(self.app.tv.open_movie(device, data["node"], data.get("audio_index")))
                if path == "/tv-api/control":
                    return self.json_response(self.app.tv.control(data["id"], device, data["action"],
                        data.get("position"), data.get("enabled"), data.get("audio_index"),
                        data.get("quality"), data.get("upscale"), data.get("hdr_mode"),
                        data.get("subtitles"), data.get("watch_mode"), data.get("encoder"), data.get("encoder_threads"),
                        data.get("surround")))
                if path == "/tv-api/heartbeat":
                    return self.json_response(self.app.tv.heartbeat(data["id"], device, data["position"],
                        data["state"], data["generation"]))
                if path == "/tv-api/diagnostic":
                    return self.json_response(self.app.tv.diagnostic(device, data))
                if path == "/tv-api/probe/prepare":
                    return self.json_response(self.app.tv.probe.prepare())
                if path.startswith("/tv-api/youtube/"):
                    return self.youtube_api(path, data)
                return self.json_response({"error": "Unknown TV endpoint"}, 404)
            if path.startswith("/hooks/plex/"):
                secret = path.removeprefix("/hooks/plex/")
                if not hmac.compare_digest(secret, self.app.config["webhook_secret"]):
                    return self.json_response({"error": "Unauthorized webhook"}, 401)
                # The public Plex event only wakes an authenticated session poll.
                size = int(self.headers.get("Content-Length", "0"))
                if size < 0 or size > 4 * 1024 * 1024:
                    raise ValueError("Webhook too large")
                self.rfile.read(size)
                self.app.plex.wake.set()
                return self.json_response({"accepted": True}, 202)
            if path.startswith("/nas/"):
                return self.json_response({"error": "Retired transcoder adapter; use the separate TV player"}, 410)
            if not self.authorized():
                return self.json_response({"error": "Authentication required"}, 401)
            data = self.body()
            if path == "/api/settings":
                return self.json_response(self.app.settings.save(data))
            if path == "/api/settings/install":
                return self.json_response(self.app.settings.install_tools(str(data.get("component", "")), data.get("accept_terms", False)))
            if path == "/api/settings/stop":
                self.app.settings.idle_required()
                if self.app.jobs.active_job or self.app.publication_threads or (self.app.settings.setup_process and self.app.settings.setup_process.poll() is None):
                    raise ValueError("Wait for current KevFLIX work to finish before closing it")
                self.json_response({"stopped": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if path == "/api/settings/plex/link":
                return self.json_response(self.app.settings.link_start())
            if path == "/api/settings/plex/select":
                return self.json_response(self.app.settings.select_server(str(data["id"])))
            if path == "/api/settings/plex/disconnect":
                return self.json_response(self.app.settings.disconnect())
            if path == "/api/pc-launch":
                return self.json_response(self.app.create_pc_launch())
            if path == "/api/tv/pair-code":
                return self.json_response(self.app.tv.pair_code())
            if path == "/api/tv/revoke":
                self.app.tv.revoke(str(data["id"]))
                return self.json_response({"ok": True})
            if path == "/api/player-updates/rollback":
                return self.json_response(self.app.updates.rollback())
            if path == "/api/jobs":
                mode = data.get("mode", "prepared")
                if mode not in ("prepared", "buffered"):
                    raise ValueError("Choose prepared or buffered delivery")
                result = self.app.jobs.enqueue(data["source"], audio_index=data.get("audio_index"),
                    subtitles="off", subtitle_index=None,   # owner rule 2026-10-19: subtitles never feed dubbing (viewing only)
                    mode=mode, title=data.get("title"), force_redub=data.get("force_redub", False))
                return self.json_response(result, 202)
            if path == "/api/plex/prepare":
                return self.json_response(self.app.plex.enqueue(data["key"], "prepared",
                                                                 data.get("force_redub", False)), 202)
            if path == "/api/plex/buffered":
                return self.json_response({"error": "Use on-demand dubbing in the TizenBrew player"}, 410)
            if path == "/api/plex/control":
                action = data["action"]
                if action == "seekTo":
                    params = {"offset": max(0, round(float(data["position"]) * 1000))}
                elif action in ("pause", "play", "stop"):
                    params = None
                else:
                    raise ValueError("Choose play, pause, stop or seekTo")
                return self.json_response(self.app.plex.plex.control(action, params))
            if path == "/api/watch/start":
                self.app.plex.watch_enabled = True
                return self.json_response({"watch_enabled": True})
            if path == "/api/watch/stop":
                self.app.plex.watch_enabled = False
                return self.json_response({"watch_enabled": False})
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/(cancel|resume|publish|play|review)", path)
            if match:
                ident, action = match.groups()
                if action == "cancel":
                    self.app.jobs.cancel(ident)
                elif action == "resume":
                    self.app.jobs.resume(ident)
                elif action == "publish":
                    self.app.publish(ident)
                elif action == "play":
                    key, media_index = self.app.published_key(ident)
                    self.app.plex.plex.play(key, data.get("position", 0), media_index=media_index)
                elif action == "review":
                    self.app.jobs.review(ident, int(data["chunk"]),
                                        approve_hash=data.get("approve_hash"),
                                        corrections=data.get("corrections"))
                return self.json_response({"ok": True}, 202)
            self.json_response({"error": "Endpoint not found"}, 404)
        except (ValueError, KeyError, FileNotFoundError) as error:
            self.json_response({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.json_response({"error": str(error)}, 500)

    def audio(self, job_id, revision):
        audio = VirtualAudio(self.app.jobs, job_id, revision)
        try:
            start, end, partial = byte_range(self.headers.get("Range"), audio.length)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{audio.length}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "private, no-store")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{audio.length}")
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            for block in audio.blocks(start, end):
                self.wfile.write(block)
                self.wfile.flush()
        except Exception:
            # Headers have already been sent. End the stream rather than append
            # JSON or fabricated silence to a requested English audio track.
            self.close_connection = True

    def bytes_response(self, content, content_type):
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def ranged_file(self, path):
        path = Path(path)
        length = path.stat().st_size
        try:
            start, end, partial = byte_range(self.headers.get("Range"), length)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", "bytes */" + str(length))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if partial else 200)
        self.cors()
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-store")
        if partial:
            self.send_header("Content-Range", "bytes %s-%s/%s" % (start, end, length))
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                data = source.read(min(256 * 1024, remaining))
                if not data:
                    self.close_connection = True
                    return
                self.wfile.write(data)
                remaining -= len(data)

    def file(self, path, content_type=None, tv_asset=False):
        path = Path(path)
        size = path.stat().st_size
        self.send_response(200)
        self.cors()
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if tv_asset:
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; media-src 'self' http: https: blob:; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        else:
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(256 * 1024), b""):
                    self.wfile.write(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="the user's profile relay-config.json (the app guard always passes it); config, state and "
                             "cache all come from this file. Without it: relay-config.json in the current folder.")
    parser.add_argument("--setup-only", action="store_true",
                        help="Serve the interface and pairing without starting media, model or update workers")
    args = parser.parse_args()
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.is_file():
            raise SystemExit("KevFLIX config not found: %s" % config_path)   # never falls back to the install root
    else:
        config_path = Path("relay-config.json").resolve()
    config = Config(str(config_path))
    config.state.mkdir(parents=True, exist_ok=True)
    worker_lock = (config.state / "worker.lock").open("a+b")
    if os.name == "nt":
        import msvcrt
        worker_lock.seek(0)
        if not worker_lock.read(1):
            worker_lock.write(b"0")
            worker_lock.flush()
        worker_lock.seek(0)
        try:
            msvcrt.locking(worker_lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            worker_lock.close()
            raise SystemExit("KevFLIX is already running or its installer is active")
    if args.setup_only:
        config.data["setup_only"] = True
        config.data["setup_locked"] = True
    application = Application(config)
    server = ThreadingHTTPServer((config["bind"], int(config["port"])), Handler)
    server.daemon_threads = True
    server.app = application
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    application.start()
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        logging.info("KevFLIX %s listening on configured interface, port %s; setup_only=%s",
                     __version__, config["port"], config["setup_only"])
        server.serve_forever()
    finally:
        application.close()
        server.server_close()
        worker_lock.close()


if __name__ == "__main__":
    main()
