"""Separate TV player service. Never modifies Plex binaries or library databases."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import subprocess
import shutil
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from .media import probe, fingerprint, file_hash, subtitle_cues, wav_info, RATE, CHANNELS
from .process import ProcessOwner, Cancelled
from .plex_browse import PlexBrowse
from .tv_probe import ProbeClips
from .buffer_policy import buffer_metrics, next_request, note_heartbeat
from . import producer_policy as _producer
from . import embedded_subtitles as _subs
from . import youtube_stream as _ystream


# A stereo dub of a 5.1 film that has no surround mix for that chunk. The stereo dub is on the ITU-normalised scale (the downmix is divided
# by 1+sqrt(2)); the surround mixes and the film's own 5.1 are on the native scale. +5.8 dB (measured with ebur128 on front/center
# material, where the center is added on top of FL/FR) keeps a fallback chunk at the loudness of its neighbours.
SURROUND_UPMIX = "volume=5.8dB,pan=5.1|FL=FL|FR=FR|FC=0.5*FL+0.5*FR|LFE=0*FL|BL=0*FL|BR=0*FL"
# The two rear (surround) channels silenced, every other channel passed through (channels by index, so 5.1 and 5.1(side) both work).
# Owner request 2026-10-20: the rear channels of the film still carry the reverb/echo of the ORIGINAL language speech (they follow its
# loudness pattern at 0.95 in the published mix), which is heard as a Russian echo behind the English voice. tv.surround_rear = "on"
# brings them back.
SURROUND_REAR_OFF = "pan=5.1|FL=c0|FR=c1|FC=c2|LFE=c3|BL=0*c0|BR=0*c0"
def passthrough_mode(mode):
    """True for a chunk whose english.wav is just the original soundtrack: no-dialogue, original-unverified, original-trial-fallback,
    original-English... (every producer mode that starts with "original")."""
    return mode == "no-dialogue" or str(mode or "").startswith("original")
DIAGNOSTIC_KEYS = ("stage", "engine", "code", "message", "ready_state", "network_state", "native_hls", "hls_js", "ua",
                   "session", "error", "hls", "codecs", "probe")
_URL = re.compile(r"https?://\S+", re.I)


def diagnostic_record(data):
    """What the TV's player reports about a failure (its console cannot be read): whitelisted fields, short values, any URL (they
    carry signatures) replaced. One line of the server log; nothing is stored."""
    def clean(value):
        if isinstance(value, str):
            return _URL.sub("[url]", value)[:160]
        if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
            return value
        return None
    data = data if isinstance(data, dict) else {}
    record = {}
    for key in DIAGNOSTIC_KEYS:
        value = data.get(key)
        if isinstance(value, dict):
            value = {str(k)[:24]: clean(v) for k, v in list(value.items())[:24]}
        else:
            value = clean(value)
        record[key] = value
    return record


def _actor_beta_on(config):
    try:
        return config["actor_beta"]["enabled"] is True
    except (KeyError, TypeError):
        return False


def _generic_on(config):
    try:
        return config["speech_engine"] == "generic"
    except (KeyError, TypeError):
        return False


def _playback_buffering_on(config):
    return _full_pcm_on(config) or _actor_beta_on(config)


def _full_pcm_on(config):
    try:
        return config["speech_engine"] in ("generic", "xtranslator")
    except (KeyError, TypeError):
        return False


def _loudness_on(config):
    """Config key ``tv.loudness.enabled`` (bool). Absent or anything but True: off, the mux is exactly as before."""
    try:
        return config["tv"]["loudness"]["enabled"] is True
    except (KeyError, TypeError):
        return False


MEDIA_EXTENSIONS = {".mp4", ".mkv", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".webm", ".mpg", ".mpeg"}
# P1-1: a job blocked by one failed/held chunk keeps playing its already-ready parts; the failure is reported as the
# session error only when playback reaches the first part that cannot play (within BOUNDARY_SECONDS of it).
# 'cancelled' is not included: a cancelled job stays blocked everywhere, as before.
KEEP_READY_STATES = ("error", "needs_review")
READY_CHUNK_STATES = ("ready", "approved")
BOUNDARY_SECONDS = 2.0


class ResourceBusy(Exception):
    pass


class TVService:
    def __init__(self, config, store, coordinator, plex=None):
        self.config, self.store, self.jobs = config, store, coordinator
        self.lock = threading.RLock()
        self.sessions, self.nodes, self.codes, self.attempts = {}, {}, {}, {}
        self.active_owner, self.active_session = None, None
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.thread = None
        self.plex = plex
        self.catalog = PlexBrowse(plex)
        # Playback loudness (tv.loudness): in-process copies of each job's stored gain, the measuring job id for
        # status(), and the excerpts a cancelled measurement had finished. Only the mux thread writes them.
        self._loudness_memo, self._loudness_resume, self._loudness_active = {}, {}, None
        # Concurrent job-gain measurements must not both run (the gain of a job can never change mid-job).
        self._loudness_lock = threading.Lock()
        # Run-ahead mux (tv.mux_ahead_unlimited): the encodes running now, per session id (every generation still
        # unwinding included); the per-session encoder choice; one cached `ffmpeg -encoders` listing per ffmpeg path.
        self._encodes, self._encoder_choice, self._encoder_listing = {}, {}, {}
        self._encoder_lock = threading.Lock()
        self._premeasure_thread = None
        self._vram_probe = None            # tests inject: callable() -> free MiB | None
        self._yt_source, self._yt_lock = None, threading.Lock()   # the YouTube hand-off downloader (relay/youtube_source.py), created on first use
        self.probe = ProbeClips(config)    # the "Test this TV" clips (relay/tv_probe.py); nothing runs until the TV asks

    def start(self):
        if self.config["tv"]["enabled"]:
            self.thread = threading.Thread(target=self.loop, name="tv-segment-mux", daemon=True)
            self.thread.start()

    def close(self):
        self.stop_event.set()
        with self.lock:
            ids = list(self.sessions)
        for ident in ids:
            self.stop(ident)
        if self._yt_source is not None:
            self._yt_source.cancel()                    # a download or start-up no session owns any more
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=20)

    def token_hash(self, value):
        return hmac.new(self.config["api_token"].encode(), value.encode(), hashlib.sha256).hexdigest()

    def pair_code(self):
        code = str(secrets.randbelow(900000) + 100000)
        with self.lock:
            self.codes = {k: v for k, v in self.codes.items() if v > time.time()}
            self.codes[self.token_hash(code)] = time.time() + 300
        return {"code": code, "expires_in": 300, "worker_url": self.config["public_url"],
                "instruction": "Enter this one-use code in the TV player"}

    def pair(self, code, name, address):
        with self.lock:
            now = time.time()
            self.attempts = {k: v for k, v in self.attempts.items() if v[0] > now - 60}
            since, count = self.attempts.get(address, (now, 0))
            if count >= 6 or len(self.attempts) > 1000:
                raise ValueError("Pairing attempts limited; wait one minute")
            self.attempts[address] = (since, count + 1)
            expires = self.codes.pop(self.token_hash(str(code)), 0)
            if expires < now:
                raise ValueError("Pairing code is invalid or expired")
        ident, token = uuid.uuid4().hex, secrets.token_urlsafe(36)
        self.store.put_record("tv_devices", ident, {"name": str(name or "Samsung TV")[:80],
            "token_hash": self.token_hash(token), "created": time.time(), "revoked": False})
        return {"device_id": ident, "token": token, "worker_url": self.config["public_url"]}

    def device(self, token):
        if not token:
            return None
        digest = self.token_hash(token)
        for record in self.store.records("tv_devices"):
            value = record["payload"]
            if not value.get("revoked") and hmac.compare_digest(value["token_hash"], digest):
                return record["id"]
        return None

    def devices(self):
        return [{"id": r["id"], "name": r["payload"]["name"],
                 "revoked": r["payload"].get("revoked", False)}
                for r in self.store.records("tv_devices")]

    def revoke(self, ident):
        for record in self.store.records("tv_devices"):
            if record["id"] == ident:
                self.store.put_record("tv_devices", ident, {**record["payload"], "revoked": True})
        with self.lock:
            sessions = [k for k, v in self.sessions.items() if v["device_id"] == ident]
        for session in sessions:
            self.stop(session)

    def _roots(self):
        values = self.config["tv"]["roots"] or self.config["media_roots"]
        return [self.config.resolve(value) for value in values]

    def _node(self, path):
        path = Path(path).resolve()
        roots = self._roots() + self.config.extra_media_roots()
        if not any(path == r or r in path.parents for r in roots):
            raise ValueError("This item is outside the TV library")
        key = hmac.new(self.config["stream_secret"].encode(), str(path).encode(), hashlib.sha256).hexdigest()[:32]
        self.nodes[key] = path
        return {"id": key, "title": path.name or str(path), "folder": path.is_dir()}

    def browse(self, node=None, offset=0, query=""):
        with self.lock:
            if not node:
                return {"items": [self._node(root) for root in self._roots() if root.is_dir()],
                        "parent": None, "more": False}
            root = self.nodes.get(node)
            if root is None or not root.is_dir():
                raise ValueError("Open the library again; this folder reference expired")
            self._node(root)
            entries = sorted((p for p in root.iterdir() if
                p.is_dir() or p.suffix.lower() in MEDIA_EXTENSIONS),
                key=lambda p: (not p.is_dir(), p.name.lower()))
            if query:
                entries = [p for p in entries if str(query).casefold() in p.name.casefold()]
            offset = max(0, int(offset))
            items = []
            for path in entries[offset:offset + 60]:
                try:
                    items.append(self._node(path))
                except (ValueError, OSError):
                    continue
            parent = None if root in self._roots() else self._node(root.parent)["id"]
            return {"items": items, "parent": parent, "more": offset + 60 < len(entries),
                    "offset": offset, "folder": node}

    def open_movie(self, device, node, audio_index=None, _preset=None):
        """_preset (YouTube progressive sessions only): {metadata, fingerprint, spec}. The growing file cannot be probed or fingerprinted
        (its length and bytes change), so the caller supplies the probe result, a stable id-based fingerprint and spec overrides."""
        with self.lock:
            source = self.nodes.get(node)
        if source is None or source.suffix.lower() not in MEDIA_EXTENSIONS:
            raise ValueError("Choose a movie from the library")
        self._node(source)
        source = self.config.media_path(source)
        metadata = probe(self.config, ProcessOwner(), source) if _preset is None else _preset["metadata"]
        streams = metadata["audio"]
        dub_unsupported = None
        if audio_index is None:
            # Fix 7: the same default policy as Coordinator.enqueue (embedded_subtitles.select_audio). A source with no
            # supported track (e.g. 7.1-only) still plays its original audio, but dubbing is explicitly unsupported.
            audio_index, dub_unsupported = _subs.select_audio(streams)
            if audio_index is None:
                audio_index = next((s["index"] for s in streams if not _subs.is_commentary(s)), streams[0]["index"])
        else:
            audio_index = int(audio_index)
        if not any(s["index"] == audio_index for s in streams):
            raise ValueError("Selected audio stream is absent")
        with self.lock:
            old = [k for k, v in self.sessions.items() if v["device_id"] == device]
        for ident in old:
            self.stop(ident)
        ident = uuid.uuid4().hex
        spec = {"id": ident, "device_id": device, "source": str(source),
                "fingerprint": fingerprint(source) if _preset is None else _preset["fingerprint"], "title": source.stem,
                "duration": metadata["duration"], "streams": metadata["streams"],
                "audio_index": audio_index, "segment_seconds": int(self.config["segment_seconds"]),
                "position": 0.0, "start_index": 0, "generation": 1, "epoch": 0,
                "dub": False, "compatibility": bool(self.config["tv"]["pc_process_all"]),
                "quality": int(self.config["tv"]["max_height"]), "upscale": bool(self.config["tv"]["upscale"]),
                "hdr_mode": self.config["tv"]["hdr_mode"], "surround": self.config["tv"].get("surround", "auto"),
                "subtitles": "off", "watch_mode": "balanced",
                "encoder": self.config["tv"]["video_encoder"], "encoder_threads": self.config["tv"]["encoder_threads"],
                "plex_key": None, "sync_progress": bool(self.config["tv"]["sync_plex_progress"]),
                "job_id": None, "job_revision": None, "watch": False, "state": "paused", "resource_wait": None,
                "touched": time.time(), "segments": {}, "error": None, "dub_unsupported": dub_unsupported,
                "job_attached_at": 0.0}
        spec["expires"] = int(time.time()) + 12 * 3600
        if _full_pcm_on(self.config):
            settings = getattr(self.config, "data", self.config)
            spec.update(speech_engine=self.config["speech_engine"], generic_prebuffer_seconds=settings.get("generic_prebuffer_seconds", 120.0),
                        generic_producer_queue=settings.get("generic_producer_queue", 4))
        if _preset is not None:
            spec.update(_preset["spec"])
        with self.lock:
            self.sessions[ident] = spec
        self.wake.set()
        return self.status(ident, device)

    def youtube_source(self):
        with self._yt_lock:
            if self._yt_source is None:
                from .youtube_source import YouTubeSource
                self._yt_source = YouTubeSource(self.config["youtube"], python=self.config["python"], ffmpeg=self.config["ffmpeg"],
                                                in_use=self._youtube_in_use, notify=self.wake.set)
            return self._yt_source

    def _yt_opt(self, key):
        """A YouTube fast-start setting: youtube.<key> of the config, else its default in relay/youtube_source.STREAM_DEFAULTS."""
        from .youtube_source import STREAM_DEFAULTS
        value = self.config["youtube"].get(key)
        return STREAM_DEFAULTS[key] if value is None else value

    def _youtube_in_use(self):
        with self.lock:
            return {v["youtube"].get("id") or Path(v["source"]).parent.name for v in self.sessions.values() if v.get("youtube")}

    def youtube_prefetch(self, video_id):
        """Begin resolving a video the YouTube view just navigated to, before the TV's open request arrives (it comes about a second
        later), so the start of playback does not wait for yt-dlp twice. Safe to call for any id; never raises."""
        try:
            settings = self.config["youtube"]
            if settings.get("enabled") and settings.get("handoff", True):
                return bool(self.youtube_source().prefetch(video_id))
        except Exception:
            logging.exception("YouTube prefetch failed")
        return False

    def youtube_cancel(self):
        """Stop a running YouTube download or start-up (the TV's Back while 'Starting the video' / 'Downloading from YouTube'), and the
        sessions that were created for it but never handed to the TV."""
        with self.lock:
            pending = [ident for ident, v in self.sessions.items() if v.get("yt_pending")]
        for ident in pending:
            self.stop(ident)
        return bool((self._yt_source is not None and self._yt_source.cancel()) or pending)

    def open_youtube(self, device, video_id, wait=None):
        """Hand a video picked in the YouTube view to the player.

        Ordinary (English or otherwise not-to-be-dubbed) videos START WHILE THEY ARRIVE: as soon as the first seconds of the video are in
        the growing local file a session is created with the YouTube profile (4 s segments, a ~6 s start lead, 1080p / 6 Mbit/s cap) and
        the reply is the session status once its first segment exists. A live stream gets a per-session live relay (2 s segments, status
        `live: true`). A video that defaults to dubbing, or one the progressive path cannot read, is downloaded completely first and
        then opens like a library movie (dubbing needs the whole file).

        Until a session can start the reply is {preparing: true, youtube: {video_id, state, percent, text}}; the request itself waits up to
        `wait` seconds (youtube.open_wait_seconds) for the answer to change, so the TV does not lose time between its once-a-second asks."""
        settings = self.config["youtube"]
        if not settings.get("enabled") or not settings.get("handoff", True):
            raise ValueError("YouTube hand-off is turned off (youtube.enabled / youtube.handoff)")
        deadline = time.monotonic() + max(0.0, float(self._yt_opt("open_wait_seconds") if wait is None else wait))
        while True:
            reply = self._open_youtube_once(device, video_id)
            if not reply.get("preparing") or time.monotonic() >= deadline:
                return reply
            time.sleep(0.04)

    def _open_youtube_once(self, device, video_id):
        value = self._youtube_pending(device, video_id)
        if value is None:
            info = self.youtube_source().ensure(video_id)
            if info["state"] == "error":
                raise ValueError(info["error"])
            if info["state"] == "live":
                value = self._open_live(device, info)
            elif info["state"] in ("streaming", "ready") and self._youtube_fast(info):
                if not (info["complete"] or info["readable"] >= self._first_segment_end(info)):
                    return self._youtube_preparing(info)
                value = self._open_youtube_stream(device, info)
            elif info["state"] == "ready":
                return self._open_youtube_library(device, info)
            else:
                return self._youtube_preparing(info)
        return self._youtube_announce(device, value)

    @staticmethod
    def _youtube_preparing(info):
        if info["state"] in ("streaming", "ready", "live"):
            return {"preparing": True, "youtube": {"video_id": info["video_id"], "state": "starting", "percent": 0,
                                                   "text": "Starting the live stream…" if info["state"] == "live" else "Starting the video…"}}
        return {"preparing": True, "youtube": {key: info[key] for key in ("video_id", "state", "percent", "text")}}

    def _youtube_fast(self, info):
        """True when this video plays with the YouTube profile: fast start enabled and not a video that defaults to dubbing."""
        return bool(self._yt_opt("stream")) and (bool(self._yt_opt("stream_foreign")) or not info["dub_default"])

    def _first_segment_end(self, info):
        return float(self._yt_opt("stream_segment_seconds")) + _ystream.READABLE_MARGIN

    def _youtube_pending(self, device, video_id):
        """The session an earlier ask of this same open created and the TV has not been given yet (None when there is none)."""
        with self.lock:
            for value in self.sessions.values():
                if value["device_id"] == device and value.get("yt_pending") and value["youtube"].get("id") == video_id:
                    value["touched"] = time.time()
                    return value
        return None

    def _youtube_announce(self, device, value):
        """The session's status once the viewer can start (first segment published / live relay has its first segments), else 'preparing'."""
        with self.lock:
            if value.get("live"):
                ready = len(value["segments"]) >= max(1, int(self._yt_opt("live_start_segments")))
            else:
                # the first reply the TV gets is already startable: first segment published and the start lead (about 6 s) prepared
                lead = max(0.0, self.ready_through(value) - value["position"])
                required = min(float(self._yt_opt("stream_start_seconds")), max(0.0, value["duration"] - value["position"]))
                ready = value["start_index"] in value["segments"] and (lead + 1e-9 >= required or lead + value["position"] >= value["duration"] - 1e-9)
            if ready:
                value["yt_pending"] = False
            error = value["error"]
        if error and not ready:
            self.stop(value["id"])
            raise ValueError(error)
        if not ready:
            return {"preparing": True, "youtube": {"video_id": value["youtube"]["id"], "state": "starting", "percent": 0,
                                                   "text": "Starting the live stream…" if value.get("live") else "Starting the video…"}}
        return self.status(value["id"], device)

    def _open_youtube_library(self, device, info):
        """The original hand-off: a completely downloaded file opens exactly like a library movie (30 s segments, dubbing default)."""
        opened = self.open_movie(device, self._node(Path(info["path"]))["id"], None)
        value = self.session(opened["id"], device)
        with self.lock:
            value["title"] = info["title"] or value["title"]
            value["youtube"] = {"id": info["video_id"], "language": info["language"], "uploader": info["uploader"],
                                "dub_default": bool(info["dub_default"])}
        return self.status(opened["id"], device)

    @staticmethod
    def _yt_stream_dicts(formats):
        """ffprobe-style stream dicts for the file the Streamer writes (video is stream 0, audio stream 1)."""
        video = next((f for f in formats if f.get("kind") == "video"), {})
        audio = next((f for f in formats if f.get("kind") == "audio"), {})
        channels = int(audio.get("channels") or 2)
        fps = float(video.get("fps") or 0)
        streams = [{"index": 0, "codec_type": "video", "codec_name": video.get("codec") or "h264", "width": int(video.get("width") or 0),
                    "height": int(video.get("height") or 0), "avg_frame_rate": "%d/1" % round(fps) if fps else "0/0", "disposition": {}},
                   {"index": 1, "codec_type": "audio", "codec_name": audio.get("codec") or "aac", "channels": channels,
                    "channel_layout": {1: "mono", 2: "stereo"}.get(channels, ""), "sample_rate": str(int(audio.get("rate") or 48000)),
                    "tags": {"language": audio.get("language") or "und"}, "disposition": {}}]
        return streams

    def _open_youtube_stream(self, device, info):
        """A YouTube-profile session over media_dir/<id>/<id>.mkv, which may still be growing. Returns the (pending) session."""
        source = Path(info["path"])
        node = self._node(source)
        tv = self.config["tv"]
        requested = int(tv["max_height"])
        if info["complete"]:
            metadata = probe(self.config, ProcessOwner(), source)             # a finished file: the real probe, like a library movie
        else:
            streams = self._yt_stream_dicts(info.get("formats") or [])
            metadata = {"duration": float(info["duration"]), "streams": streams, "audio": [streams[1]]}
        spec = {"segment_seconds": max(2, int(self._yt_opt("stream_segment_seconds"))), "compatibility": True,
                "quality": min(requested, int(self._yt_opt("stream_max_height"))), "yt_quality_request": requested,
                "title": info["title"] or source.stem, "yt_stream": True, "yt_pending": True,
                "youtube": {"id": info["video_id"], "language": info["language"], "uploader": info["uploader"],
                            "dub_default": bool(info["dub_default"])}}
        opened = self.open_movie(device, node["id"], None, _preset={
            "metadata": metadata, "fingerprint": "youtube:" + info["video_id"], "spec": spec})
        return self.session(opened["id"], device)

    # ---- YouTube live ----------------------------------------------------------------------------------------------
    def _open_live(self, device, info):
        """A live session: ONE ffmpeg (the encoder the library path would choose) turns the YouTube live HLS into a sliding HLS of short
        segments in cache/tv-live/<session>/, which the signed /tv-media/<id>/ URLs serve. Returns the (pending) session."""
        video_id = info["video_id"]
        live = self.youtube_source().live_inputs(video_id)
        if not live:
            raise ValueError("The live stream is no longer available")
        with self.lock:
            old = [k for k, v in self.sessions.items() if v["device_id"] == device]
        for ident in old:
            self.stop(ident)
        tv = self.config["tv"]
        ident = uuid.uuid4().hex
        height = min(int(tv["max_height"]), int(self._yt_opt("live_max_height")))
        seconds = max(1, int(self._yt_opt("live_segment_seconds")))
        value = {"id": ident, "device_id": device, "source": "", "fingerprint": "youtube-live:" + video_id,
                 "title": live.get("title") or info["title"] or "Live", "duration": 0.0, "streams": self._yt_stream_dicts(info.get("formats") or []),
                 "audio_index": 1, "segment_seconds": seconds, "position": 0.0, "start_index": 0, "generation": 1, "epoch": 0,
                 "dub": False, "compatibility": True, "quality": height, "upscale": False, "hdr_mode": "sdr", "surround": "stereo",
                 "subtitles": "off", "watch_mode": "balanced", "encoder": tv["video_encoder"], "encoder_threads": tv["encoder_threads"],
                 "plex_key": None, "sync_progress": False, "job_id": None, "job_revision": None, "watch": False, "state": "paused",
                 "resource_wait": None, "touched": time.time(), "segments": {}, "error": None,
                 "dub_unsupported": "Dubbing is not available for live streams.", "job_attached_at": 0.0,
                 "expires": int(time.time()) + 12 * 3600, "live": True, "yt_pending": True, "live_window": [], "live_inputs": live["inputs"],
                 "youtube": {"id": video_id, "language": "", "uploader": live.get("uploader") or info["uploader"], "dub_default": False}}
        self._start_live_process(value, value["encoder"])
        with self.lock:
            self.sessions[ident] = value
        self.wake.set()
        return value

    def _start_live_process(self, value, requested):
        """(Re)start the live ffmpeg of a session with the encoder `requested` ('auto' / 'nvenc' / 'software')."""
        probe_value = dict(value, encoder=requested)
        encoder, _ = self._resolve_encoder(probe_value, False)
        codec, options = self._video_codec(encoder, False, value["encoder_threads"])
        folder = self.config.cache / "tv-live" / value["id"]
        command = _ystream.live_command(
            self.config["ffmpeg"], [item["url"] for item in value["live_inputs"]], folder, codec, options,
            threads=value["encoder_threads"], segment_seconds=value["segment_seconds"], list_size=int(self._yt_opt("live_list_size")),
            height=value["quality"], maxrate=self._yt_opt("live_maxrate"), bufsize=self._yt_opt("live_bufsize"))
        value["_live"] = _ystream.LiveHls(folder, command, segment_seconds=value["segment_seconds"]).start()
        value["_live_encoder"] = encoder

    # ---- YouTube-profile sessions: status fields, segment scheduling, leaving the profile --------------------------------
    def _yt_progress(self, value):
        """{readable, complete, error, running} of the file a progressive YouTube session plays from."""
        source = self._yt_source
        if source is None:
            return {"readable": float(value["duration"]), "complete": True, "error": "", "running": False}
        return source.progress(value["youtube"]["id"])

    def _youtube_fields(self, value, result):
        """The start/buffer fields of a YouTube-profile session (progressive or live), computed here instead of by the dubbing gate: the
        viewer may start once `buffer_required_seconds` (youtube.stream_start_seconds, about 6 s; live: live_start_segments segments) are
        prepared ahead of the playhead. Caller holds the lock."""
        position, duration, seg = value["position"], value["duration"], value["segment_seconds"]
        paused = bool(value.get("paused_intent", value["state"] == "paused"))
        youtube = dict(value["youtube"])
        if value.get("live"):
            lead = float(sum(seconds for _, seconds in value.get("live_window") or ()))
            required = max(1, int(self._yt_opt("live_start_segments"))) * float(seg)
            at_end = False
            youtube["live"] = True
        else:
            lead = max(0.0, self.ready_through(value) - position)
            required = min(float(self._yt_opt("stream_start_seconds")), max(0.0, duration - position))
            at_end = lead > 0 and position + lead >= duration - 1e-9
            progress = self._yt_progress(value)
            youtube.update(streaming=not progress["complete"], complete=bool(progress["complete"]),
                           readable=round(float(progress["readable"]), 2),
                           download_percent=100 if progress["complete"] else min(99, int(100 * progress["readable"] / duration)) if duration else 0)
        startable = bool(result["ready"]) and (lead + 1e-9 >= required or at_end)
        return {"startable": startable, "buffer_required_seconds": float(required), "contiguous_lead_seconds": float(lead),
                "paused_intent": paused, "seek_epoch": int(value.get("epoch", 0)), "buffer_seconds": round(lead, 3),
                "start_threshold_seconds": float(required), "underruns": int(value.get("underruns", 0)),
                "lead_kind": "original", "lead_reason": "live YouTube stream" if value.get("live") else "YouTube video",
                "live": bool(value.get("live")), "youtube": youtube}

    def _youtube_tick(self, ident, value):
        """One scheduler pass for a YouTube-profile session (called by loop(); never blocks on a render)."""
        if value.get("live"):
            return self._live_tick(ident, value)
        now = time.time()
        seg = value["segment_seconds"]
        generation = value["generation"]
        progress = self._yt_progress(value)
        readable, complete = float(progress["readable"]), bool(progress["complete"])
        total = math.ceil(value["duration"] / seg)
        current = max(value["start_index"], int(value["position"] // seg))
        horizon = min(total, current + max(2, math.ceil(float(self._yt_opt("stream_ahead_seconds")) / seg)))
        state = self._ahead_state(value)
        muxed = value["segments"]

        def renderable(index):
            # a segment is started only when the file really covers it (plus a margin for reordering); the last one waits for the end
            return complete or (index + 1) * seg + _ystream.READABLE_MARGIN <= readable

        first = next((i for i in range(current, total) if i not in muxed), None)
        waiting = str(value.get("resource_wait") or "").startswith("Waiting for the video")
        if first is not None and not renderable(first) and first <= current + 1:
            # the viewer is about to run out of prepared video because the download has not reached the next part
            if progress.get("error") and not progress.get("running"):
                value["error"] = progress["error"]
                return
            percent = min(99, int(100 * readable / value["duration"])) if value["duration"] else 0
            value["resource_wait"] = "Waiting for the video to download that far (%d%%)…" % percent
        elif waiting:
            value["resource_wait"] = None
        if now < state["hold_until"]:
            return
        failed = state["failed"]
        record = failed.get(first)
        if record and record["count"] >= _producer.MUX_ATTEMPTS and first - current < _producer.URGENT_SPAN:
            with self.lock:
                if value["generation"] == generation:
                    value["error"] = record["message"]
            return
        parked = {i for i, r in failed.items() if r["count"] >= _producer.MUX_ATTEMPTS or now < r["next_try"]}
        ready = {i for i in range(current, horizon) if renderable(i)} - parked
        with self.lock:
            inflight = set(state["inflight"])
        picks = _producer.plan_mux(current=current, total=horizon, muxed=muxed, inflight=inflight, ready=ready,
                                   workers=_producer.mux_workers(self._tv_cfg()), active=self._encode_total(), ahead_allowed=True)
        for index in picks:
            owner = ProcessOwner()
            with self.lock:
                if value["generation"] != generation or self.sessions.get(ident) is not value:
                    return
                state["inflight"][index] = owner
                self._encodes.setdefault(ident, set()).add(owner)
                state["launched"] += 1
            threading.Thread(target=self._ahead_encode, args=(ident, value, index, generation, owner, state),
                             name="tv-mux-%d" % index, daemon=True).start()

    def _live_tick(self, ident, value):
        """Follow the live ffmpeg: publish its finished segments, grow the duration, restart in software when NVENC refused, stop at
        the 12 h expiry or when the stream ends."""
        live = value.get("_live")
        if live is None:
            return
        if time.time() > value["expires"]:
            return self.stop(ident)
        segments, target = live.segments()
        with self.lock:
            if self.sessions.get(ident) is not value:
                return
            if segments:
                value["segments"] = {sequence: str(path) for sequence, _, path in segments}
                value["live_window"] = [(sequence, seconds) for sequence, seconds, _ in segments]
                value["start_index"] = segments[0][0]
                value["live_target"] = max(int(target), int(math.ceil(max(seconds for _, seconds, _ in segments))))
                last = segments[-1]
                value["duration"] = last[0] * float(value["segment_seconds"]) + last[1]
                if value["resource_wait"] and str(value["resource_wait"]).startswith("Starting"):
                    value["resource_wait"] = None
            else:
                value["resource_wait"] = value["resource_wait"] or "Starting the live stream…"
        if not live.alive():
            elapsed = time.monotonic() - live.started_at
            if not segments and elapsed < 30 and value.get("_live_encoder") == "nvenc":
                live.stop()
                self._pin_software(value, "live h264_nvenc encode failed: " + live.reason())
                self._start_live_process(value, "software")
                return
            with self.lock:
                value["error"] = "The live stream has ended." if segments else "The live stream could not be started: " + live.reason()

    def _youtube_quality_unchanged(self, value, quality, upscale, hdr_mode, surround):
        """True when a 'quality' request leaves a YouTube-profile session exactly as it is (the capped height, upscaling, HDR and
        surround choices all equal): then no restart is needed. Anything invalid or different takes the normal path (and its checks)."""
        try:
            quality = int(quality)
        except (TypeError, ValueError):
            return False
        if quality not in (480, 720, 1080, 2160) or hdr_mode not in ("auto", "sdr") or (surround is not None and surround not in ("auto", "stereo")):
            return False
        capped = min(quality, int(self._yt_opt("stream_max_height")))
        if (capped, bool(upscale), hdr_mode) != (value["quality"], bool(value["upscale"]), value["hdr_mode"]):
            return False
        if surround is not None and surround != value.get("surround", "auto"):
            return False
        value["yt_quality_request"] = quality
        return True

    def _youtube_to_library(self, value):
        """Dubbing needs the complete file: leave the YouTube profile (4 s segments, 1080p cap) for the normal movie profile (the
        configured segment_seconds, the asked quality) through the usual generation restart that control() performs next. Raises
        ValueError while the download is still running."""
        progress = self._yt_progress(value)
        if not progress["complete"]:
            raise ValueError("Dubbing needs the whole video and it is still downloading (%d%%). Try again in a moment."
                             % (min(99, int(100 * progress["readable"] / value["duration"])) if value["duration"] else 0))
        source = Path(value["source"])
        metadata = probe(self.config, ProcessOwner(), source)
        audio_index, dub_unsupported = _subs.select_audio(metadata["audio"])
        if audio_index is None:
            audio_index = next((s["index"] for s in metadata["audio"] if not _subs.is_commentary(s)), metadata["audio"][0]["index"])
        self._drop_stream_segments(value)
        with self.lock:
            value.update(segment_seconds=int(self.config["segment_seconds"]), yt_stream=False, fingerprint=fingerprint(source),
                         duration=metadata["duration"], streams=metadata["streams"], audio_index=audio_index,
                         dub_unsupported=dub_unsupported, quality=int(value.get("yt_quality_request") or self.config["tv"]["max_height"]))
            if _full_pcm_on(self.config):
                settings = getattr(self.config, "data", self.config)
                value.update(speech_engine=self.config["speech_engine"],
                             generic_prebuffer_seconds=settings.get("generic_prebuffer_seconds", 120.0),
                             generic_producer_queue=settings.get("generic_producer_queue", 4))
            value.pop("_mux_ahead", None)

    def session(self, ident, device=None, touch=True):
        with self.lock:
            value = self.sessions.get(ident)
            if value is None or (device and value["device_id"] != device):
                raise ValueError("TV playback session has ended")
            if touch:
                value["touched"] = time.time()
            return value

    def control(self, ident, device, action, position=None, enabled=None, audio_index=None,
                quality=None, upscale=None, hdr_mode=None, subtitles=None, watch_mode=None,
                encoder=None, encoder_threads=None, surround=None):
        value = self.session(ident, device)
        if action == "stop":
            self.stop(ident)
            return {"stopped": True}
        if action == "watch":
            value["watch"] = bool(enabled)
        elif action == "watch_mode":
            if watch_mode not in ("balanced", "quiet", "ask-only"):
                raise ValueError("Choose balanced, quiet or ask-only commentary")
            value["watch_mode"] = watch_mode
        elif action == "subtitles":
            selection = str(subtitles)
            allowed = ["off", "auto", "dub"] + [str(s["index"]) for s in value["streams"]
                if s.get("codec_type") == "subtitle" and s.get("codec_name") in ("subrip", "ass", "ssa", "webvtt", "mov_text", "text")]
            if selection not in allowed:
                raise ValueError("Choose an available text subtitle track")
            if selection == "dub" and not value["job_id"]:
                raise ValueError("Turn English dubbing on before choosing its subtitles")
            value["subtitles"] = selection
        elif action == "progress_sync":
            value["sync_progress"] = bool(enabled)
        elif action in ("seek", "dub", "audio", "compatibility", "quality", "processing"):
            if value.get("live"):
                # A live picture cannot be seeked, re-timed, dubbed or re-encoded on the fly: only dubbing says so, the rest is ignored
                # (the TV sends quality / encoder choices right after every open).
                if action == "dub" and bool(enabled):
                    raise ValueError("Dubbing is not available for live streams.")
                return self.status(ident, device)
            if action == "quality" and value.get("yt_stream") and self._youtube_quality_unchanged(value, quality, upscale, hdr_mode, surround):
                return self.status(ident, device)       # the TV sends its saved quality right after every open: nothing to restart
            if action == "dub" and bool(enabled) and value.get("yt_stream"):
                self._youtube_to_library(value)           # dubbing needs the whole file: leave the short-segment YouTube profile
            old_job = value["job_id"]
            if action == "processing":
                if encoder not in ("auto", "nvenc", "software") or int(encoder_threads or 0) not in (1, 2, 4, 8):
                    raise ValueError("Choose automatic, NVIDIA or software encoding and 1, 2, 4 or 8 CPU threads")
                # The chosen encoder is kept; _resolve_encoder decides per playback generation whether NVENC is usable
                # (h264_nvenc listed + the nvenc_min_free_mib VRAM check) and records software otherwise.
                value.update(encoder=encoder, encoder_threads=int(encoder_threads), compatibility=True)
            if action == "audio":
                audio_index = int(audio_index)
                if not any(s.get("codec_type") == "audio" and s["index"] == audio_index for s in value["streams"]):
                    raise ValueError("Selected audio stream is absent")
                value["audio_index"] = audio_index
                value["compatibility"] = True
            if action == "quality":
                quality = int(quality)
                if quality not in (480, 720, 1080, 2160):
                    raise ValueError("Choose a resolution up to 4K")
                if hdr_mode not in ("auto", "sdr"):
                    raise ValueError("Choose automatic HDR or SDR conversion")
                if surround is not None and surround not in ("auto", "stereo"):
                    raise ValueError("Choose 5.1 surround (automatic) or stereo sound")
                value.update(quality=quality, upscale=bool(upscale), hdr_mode=hdr_mode, compatibility=True)
                if value.get("yt_stream"):
                    # the YouTube profile never exceeds youtube.stream_max_height (a Wi-Fi TV); the asked height is kept for dubbing
                    value["yt_quality_request"] = quality
                    value["quality"] = min(quality, int(self._yt_opt("stream_max_height")))
                if surround is not None:
                    value["surround"] = surround
            new_position = value["position"] if position is None else float(position)
            if not math.isfinite(new_position):
                raise ValueError("Invalid playback position")
            job = None
            if (action == "dub" and bool(enabled)) or (action == "audio" and value["dub"]):
                if action == "dub" and value.get("dub_unsupported") and audio_index is None:
                    chosen = next((s for s in value["streams"] if s.get("index") == value["audio_index"]), {})
                    if _subs.audio_support(chosen) is not None:
                        raise _subs.UnsupportedAudio(value["dub_unsupported"])
                # Subtitles are for viewing only and never feed translation: the dub job is always created with
                # subtitles OFF (no sidecar file, no embedded track, no automatic English choice). The viewer's own
                # subtitle track is served separately by captions() from the session's "subtitles" selection.
                job = self.jobs.enqueue(value["source"], audio_index=value["audio_index"],
                                        mode="ondemand", title=value["title"], subtitles=_subs.OFF,
                                        **({"resume_cancelled": True}
                                           if getattr(self.jobs, "RESUME_CANCELLED_VERSION", 0) >= 1 else {}),
                                        **({"initial_index": int(min(max(0, new_position), max(0, value["duration"] - .1))
                                                                // value["segment_seconds"])}
                                           if _playback_buffering_on(self.config) else {}))
            with self.lock:
                value["position"] = min(max(0, new_position), max(0, value["duration"] - .1))
                value["start_index"] = int(value["position"] // value["segment_seconds"])
                value["generation"] += 1
                value["expires"] = int(time.time()) + 12 * 3600
                value["epoch"] += 1
                value["error"] = None
                value["resource_wait"] = None
                value["segments"] = {}
                value["state"] = "buffering"
                if action == "compatibility" and not self.config["tv"]["pc_process_all"] and not value.get("yt_stream"):
                    value["compatibility"] = bool(enabled)
                if action in ("dub", "audio"):
                    value["dub"] = bool(enabled) if action == "dub" else value["dub"]
                    value["job_id"] = job["id"] if job else None
                    value["job_revision"] = job["payload"].get("revision", 1) if job else None
                    # Production-rate samples (producer_policy.production_samples) ignore completions older than this.
                    value["job_attached_at"] = time.time() if job else 0.0
                    if not value["dub"] and value["subtitles"] == "dub":
                        value["subtitles"] = "off"
            if self.active_session == ident and self.active_owner:
                self.active_owner.cancel()
            self._cancel_encodes(ident)     # run-ahead encodes of the old generation (epoch rule); no-op otherwise
            if value["job_id"]:
                try:
                    self.jobs.request(value["job_id"], value["start_index"], -int(time.time()))
                except ValueError:
                    # P1-1: seeking within a blocked job's ready parts is allowed; status reports the boundary.
                    if not self._keeps_ready(value["job_id"]):
                        raise
            if _playback_buffering_on(self.config) and action in ("seek", "dub", "audio"):
                _producer.on_seek(value, self.store, self.jobs, self.config["tv"], time.time())
            if old_job and old_job != value["job_id"]:
                self.cancel_unshared_job(old_job)
        elif action in ("play", "pause"):
            value["state"] = "playing" if action == "play" else "paused"
            if _playback_buffering_on(self.config) or value.get("yt_stream") or value.get("live"):
                value["paused_intent"] = action == "pause"
        else:
            raise ValueError("Unsupported TV action")
        self.wake.set()
        return self.status(ident, device)

    def diagnostic(self, device, data):
        logging.info("tv_diagnostic %s", json.dumps(diagnostic_record(data), ensure_ascii=False))
        return {"ok": True}

    def heartbeat(self, ident, device, position, state, generation):
        value = self.session(ident, device)
        if int(generation) != value["generation"]:
            return self.status(ident, device)
        position = float(position)
        if not math.isfinite(position):
            raise ValueError("Invalid playback position")
        with self.lock:
            value["position"] = max(0, min(value["duration"], position))
            value["state"] = state if state in ("playing", "paused", "buffering", "ended") else "paused"
            self._note_playback_evidence(value)
            if _playback_buffering_on(self.config) or value.get("yt_stream") or value.get("live"):
                if not value.get("live"):
                    note_heartbeat(value, value["position"], value["state"], time.time())
                if value["state"] in ("playing", "paused"):
                    value["paused_intent"] = value["state"] == "paused"
        self.store.put_record("sessions", "tv-" + ident, {
            "source_kind": "tizenbrew", "session_id": ident, "client_id": device,
            "source": value["source"], "title": value["title"], "nas_path": "",
            "rating_key": "", "transcode_session": "", "audio_index": value["audio_index"],
            "state": value["state"], "position": value["position"], "duration": value["duration"],
            "observed": time.time(), "epoch": value["epoch"], "job_id": value["job_id"],
            "dub": value["dub"], "watch": value["watch"], "watch_mode": value["watch_mode"]})
        if value["sync_progress"] and value["plex_key"] and (state == "ended" or time.time() - value.get("progress_at", 0) >= 15):
            value["progress_at"] = time.time()
            try:
                self.plex.call("/:/timeline", {"ratingKey": value["plex_key"],
                    "key": "/library/metadata/" + value["plex_key"],
                    "state": "stopped" if value["state"] == "ended" else value["state"],
                    "time": round(value["position"] * 1000), "duration": round(value["duration"] * 1000)})
                if state == "ended" and value["position"] >= value["duration"] - 5 and not value.get("scrobbled"):
                    self.catalog.watched(value["plex_key"], True)
                    value["scrobbled"] = True
            except Exception:
                value["progress_warning"] = "Plex progress could not be saved"
        self.wake.set()
        return self.status(ident, device)

    def stop(self, ident):
        with self.lock:
            value = self.sessions.pop(ident, None)
        if value and value["sync_progress"] and value["plex_key"]:
            try:
                self.plex.call("/:/timeline", {"ratingKey": value["plex_key"],
                    "key": "/library/metadata/" + value["plex_key"], "state": "stopped",
                    "time": round(value["position"] * 1000), "duration": round(value["duration"] * 1000)})
            except Exception:
                pass
        self.store.expire_record("sessions", "tv-" + ident)
        if self.active_session == ident and self.active_owner:
            self.active_owner.cancel()
        self._cancel_encodes(ident)
        self._encoder_choice.pop(ident, None)
        if value and value["job_id"]:
            self.cancel_unshared_job(value["job_id"])
        if value and (value.get("yt_stream") or value.get("live")):
            self._youtube_session_closed(value)

    def _youtube_session_closed(self, value):
        """A YouTube session ended (Back, stop, idle, revoke, 12 h expiry, server close): stop its live relay or its still-running
        download and drop the short segments it made. Nothing it started outlives it."""
        live = value.pop("_live", None)
        if live is not None:
            live.stop()
        video_id = value["youtube"].get("id")
        with self.lock:
            shared = any(other.get("youtube", {}).get("id") == video_id for other in self.sessions.values())
        if value.get("yt_stream") and not shared:
            if self._yt_source is not None:
                self._yt_source.cancel_video(video_id)
            self._drop_stream_segments(value)

    def _drop_stream_segments(self, value):
        """Delete the segment files (and receipts) of a YouTube-profile session: 4 s segments of a video nobody replays would pile up."""
        with self.lock:
            paths = {Path(p) for p in value["segments"].values()} | {Path(p) for p in value.pop("yt_files", ())}
            value["segments"] = {}
        for path in paths:
            for victim in (path, path.with_suffix(".json")):
                try:
                    victim.unlink()
                except OSError:
                    pass

    def cancel_unshared_job(self, job_id):
        with self.lock:
            shared = any(s["job_id"] == job_id for s in self.sessions.values())
        job = self.store.job(job_id)
        if not shared and job and job["payload"].get("mode") == "ondemand":
            self.jobs.cancel(job_id)

    def status(self, ident, device=None):
        value = self.session(ident, device)
        with self.lock:
            current = int(value["position"] // value["segment_seconds"])
            direct = not value["dub"] and not value["compatibility"]
            ready = direct or value["start_index"] in value["segments"]
            issue = value["error"]
            dub_issue, stops_at, job_payload = None, None, None
            if value["job_id"]:
                job = self.store.job(value["job_id"])
                job_payload = job["payload"] if job else None
                if job and job["state"] in ("needs_review", "error", "cancelled"):
                    dub_issue = "Dubbing " + job["state"] + ". Review it on the PC, or turn dubbing off."
                    if job["state"] in KEEP_READY_STATES:
                        # P1-1: ready parts before the first unplayable part stay playable; the error is the
                        # session error once playback is at that boundary (or there is no ready part left).
                        stops_at = self.dub_boundary(value)
                        if stops_at is not None and stops_at - value["position"] <= BOUNDARY_SECONDS:
                            issue = dub_issue
                    else:
                        issue = dub_issue
            result = {"id": ident, "title": value["title"], "duration": value["duration"],
                "position": value["position"], "base": 0 if direct else value["start_index"] * value["segment_seconds"],
                "state": value["state"], "generation": value["generation"], "epoch": value["epoch"],
                "dub": value["dub"], "compatibility": value["compatibility"], "watch": value["watch"], "job_id": value["job_id"],
                "quality": value["quality"], "upscale": value["upscale"], "hdr_mode": value["hdr_mode"],
                "surround": value.get("surround", "auto"), "surround_active": self._surround_stream(value) is not None,
                "encoder": value["encoder"], "encoder_threads": value["encoder_threads"],
                "subtitles": value["subtitles"], "watch_mode": value["watch_mode"], "sync_progress": value["sync_progress"],
                "resource_wait": value["resource_wait"], "progress_warning": value.get("progress_warning"),
                "caption_url": self.media_url(value, "captions.vtt") if value["subtitles"] != "off" else None,
                "ready": ready, "ready_through": value["duration"] if direct else self.ready_through(value),
                "error": issue, "segment_seconds": value["segment_seconds"],
                "url": self.media_url(value, "original") if direct else self.media_url(value, "index.m3u8") if ready else None,
                "profile": "Original file" if direct else "PC processing · up to " + str(value["quality"]) + "p" + (" · upscaling" if value["upscale"] else ""),
                "subtitle_tracks": [{"index": s["index"], "language": s.get("tags", {}).get("language", "und"),
                                    "title": s.get("tags", {}).get("title", s.get("codec_name", ""))}
                                    for s in value["streams"] if s.get("codec_type") == "subtitle"
                                    and s.get("codec_name") in ("subrip", "ass", "ssa", "webvtt", "mov_text", "text")],
                "audio_tracks": [{"index": s["index"], "language": s.get("tags", {}).get("language", "und"),
                                  "title": s.get("tags", {}).get("title", s.get("codec_name", ""))}
                                 for s in value["streams"] if s.get("codec_type") == "audio"],
                "audio_index": value["audio_index"], "dub_unsupported": value.get("dub_unsupported")}
            if value.get("youtube"):
                result["youtube"] = dict(value["youtube"])
            youtube_fast = bool(value.get("yt_stream") or value.get("live"))
            if youtube_fast:
                result.update(self._youtube_fields(value, result))
            if _loudness_on(self.config):
                result["loudness"] = self._loudness_status(value, job_payload)
            # A detached snapshot: later muxes/heartbeats must not mutate a status response.
            evidence = json.loads(json.dumps(self._playback_evidence(value)))
            for prepared in evidence["prepared_segments"]:
                ranges = [[max(prepared["start"], a), min(prepared["end"], b)]
                          for a, b in evidence["played_ranges"]
                          if min(prepared["end"], b) > max(prepared["start"], a)]
                prepared["heartbeat_played_ranges"] = ranges
                prepared["heartbeat_played_seconds"] = round(sum(b - a for a, b in ranges), 6)
            result["playback_evidence"] = evidence
            if value.get("speech_engine") in ("generic", "xtranslator"):
                evidence["generic_take_evidence"] = self._generic_take_evidence(value, evidence)
                result["speech_engine"] = value["speech_engine"]
                evidence["speech_engine"] = value["speech_engine"]
                if value["speech_engine"] == "xtranslator":
                    candidates = [item for item in evidence["prepared_segments"] if item.get("speech_engine") == "xtranslator"]
                    latest = candidates[-1] if candidates else {}
                    result["xt_status"] = {"scope": "latest prepared chunk receipt; not current service health",
                        **{key: latest.get(key) for key in ("xt_version", "xt_session_uid", "detected_language",
                            "source_language", "speaker_count", "speaker_state_reset_count", "backend_identity",
                            "receipt_path", "receipt_sha256")}}
            if _playback_buffering_on(self.config) and not youtube_fast:
                lead = self.store.chunk(value["job_id"], current) if value["job_id"] else None
                result.update(buffer_metrics(value, time.time(), self.config["tv"], lead))
                result.update(_producer.status_fields(value, self.store, self.config["tv"], time.time()))
                result["run_ahead"] = self._run_ahead_status(value)
            if dub_issue:
                # Additive, informational: the blocked job's reason and where its playable parts end (None: they don't).
                result.update(dub_issue=dub_issue, dub_stops_at=stops_at)
                through = self.ready_through(value)
                if (not _full_pcm_on(self.config) and stops_at is not None
                        and through >= stops_at and through > value["position"]):
                    # Everything up to the boundary is muxed: start/continue to it (the player's eof_ready rule)
                    # instead of waiting for a lead that a blocked job can never produce.
                    result["eof_ready"] = True
                    if "startable" in result:
                        result["startable"] = True
            return result

    def _playback_evidence(self, value):
        """Current-generation observations only; never an input to playback decisions. Caller holds lock."""
        evidence = value.get("playback_evidence")
        if not evidence or evidence["generation"] != value["generation"]:
            evidence = {"version": "tv-prepared-media-v1", "generation": value["generation"],
                "scope": "current generation; discarded on seek, revision or playback settings change",
                "source_audio_language": "unknown", "foreign_dubbing_seconds": None,
                "prepared_segments": [], "served_seconds": None,
                "http_delivery_status": "unobserved: byte writes are owned by server.py",
                "played_ranges": [], "played_seconds": 0.0,
                "played_scope": "adjacent playing heartbeat positions only; client-reported, not hearing proof",
                "human_listening_acceptance": False}
            value["playback_evidence"] = evidence
            value.pop("_evidence_heartbeat", None)
        return evidence

    @staticmethod
    def _evidence_union(intervals):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return merged

    def _note_playback_evidence(self, value):
        evidence = self._playback_evidence(value)
        previous = value.get("_evidence_heartbeat")
        sample = {"position": value["position"], "state": value["state"], "at": time.monotonic()}
        # No range from a control request, first heartbeat, pause/buffer/ended transition,
        # backwards jump, or a forward jump faster than elapsed time plus two seconds.
        # Even accepted ranges are a client's reported progression, not proof of output.
        if previous and previous["state"] == sample["state"] == "playing":
            distance = sample["position"] - previous["position"]
            if 0 < distance <= sample["at"] - previous["at"] + 2:
                ranges = self._evidence_union(evidence["played_ranges"] +
                                              [[previous["position"], sample["position"]]])
                evidence["played_ranges"] = ranges
                evidence["played_seconds"] = round(sum(end - start for start, end in ranges), 6)
        value["_evidence_heartbeat"] = sample

    def _prepared_media_evidence(self, value, index, audio_hash, start, duration):
        """Describe the exact selected audio hash; do not call preparation HTTP delivery.

        Only installed in the session after render succeeds and its existing generation/job
        checks pass. Accounting failures stay visible and cannot change the selected audio.
        """
        end = start + duration
        receipt = {"segment_index": index, "generation": value["generation"],
            "job_id": value["job_id"], "job_revision": value["job_revision"],
            "audio_index": value["audio_index"], "start": start, "end": end,
            "media_state": "prepared", "prepared_seconds": duration,
            "chunk_audio_sha256": None if audio_hash == "original" else audio_hash,
            "uses_source_audio_stream": audio_hash == "original",
            "uses_original_audio": True if audio_hash == "original" else None,
            "uses_fallback_audio": False if audio_hash == "original" else None,
            "generated_intervals": [], "generated_union": [], "english_generated_seconds": None,
            "served_seconds": None, "source_audio_language": "unknown", "foreign_dubbing_seconds": None,
            "accounting_status": "original-source-stream" if audio_hash == "original" else "unverified"}
        if audio_hash == "original":
            receipt["english_generated_seconds"] = 0.0
            return receipt
        try:
            chunk = self.store.chunk(value["job_id"], index)
            payload = chunk["payload"] if chunk else {}
            if payload.get("sha256") != audio_hash:
                receipt["accounting_status"] = "chunk-payload-hash-mismatch-or-missing"
                return receipt
            if value.get("speech_engine") in ("generic", "xtranslator") or _full_pcm_on(self.config):
                saved = self._generic_chunk_payload(chunk)
                if saved["accounting_engine"] != value.get("speech_engine", self.config["speech_engine"]):
                    raise ValueError("Prepared audio engine differs from playback session")
                receipt.update(receipt_path=saved["receipt_path"], receipt_sha256=saved["receipt_sha256"])
                return self._generic_prepared_media_evidence(receipt, saved, start, end)
            from .actor_beta_timing import publication_accounting
            accounting = publication_accounting(payload)
            generated = []
            for item in accounting["generated_intervals"]:
                lo, hi = max(start, item["start"]), min(end, item["end"])
                if hi > lo:
                    generated.append(dict(item, start=lo, end=hi))
            union = self._evidence_union([[item["start"], item["end"]] for item in generated])
            mode = str(payload.get("mode") or "")
            whole_original = mode.startswith("original") or mode == "no-dialogue" or \
                payload.get("pipeline_state") == "original_fallback"
            fallback_ids = set(payload.get("beta_original_fallback") or {}) | \
                set(payload.get("beta_runtime_fallback") or {})
            fallback_spans = list(payload.get("beta_fallback_intervals") or ())
            fallback_spans += [[cue["start"], cue["end"]] for cue in payload.get("cues") or ()
                               if cue.get("id") in fallback_ids]
            fallback = mode == "original-unverified" or payload.get("pipeline_state") == "original_fallback" or \
                any(min(end, b) > max(start, a) for a, b in fallback_spans)
            protected = (payload.get("original_audio_protection") or {}).get("applied_protected_intervals") or ()
            receipt.update(accounting_status=accounting["publication_status"], generated_intervals=generated,
                generated_union=union, english_generated_seconds=round(sum(b - a for a, b in union), 6),
                uses_fallback_audio=fallback,
                uses_original_audio=whole_original or fallback or any(min(end, b) > max(start, a) for a, b in protected),
                original_audio_scope="source speech preservation from chunk receipts; not a claim about soundtrack beds",
                publication_accounting_version=accounting["version"])
        except Exception as error:
            # Evidence must not turn a playable segment into a mux/playback error.
            receipt["accounting_status"] = "unavailable"
            receipt["accounting_error"] = type(error).__name__
        return receipt

    @staticmethod
    def _generic_chunk_payload(chunk):
        """Follow the atomically published unique-claim receipt, never a shared chunk receipt filename."""
        recorded = chunk["payload"]
        path = Path(recorded["receipt_path"])
        if not path.is_absolute():
            raise ValueError("Generic receipt path is not absolute")
        raw = path.read_bytes()
        saved = json.loads(raw)
        if saved.get("generic_version") != "generic-dub-v1":
            raise ValueError("Unsupported generic receipt")
        engine = recorded.get("speech_engine", recorded.get("engine", "generic"))
        if engine not in ("generic", "xtranslator"):
            raise ValueError("Unsupported full-PCM engine identity")
        if engine == "xtranslator" or saved.get("xt_version") is not None:
            for key in ("speech_engine", "engine", "xt_version"):
                if not recorded.get(key) or saved.get(key) != recorded[key]:
                    raise ValueError("XTranslator saved receipt engine/session differs from its published chunk")
            if "xt_session_uid" not in recorded or saved.get("xt_session_uid") != recorded["xt_session_uid"]:
                raise ValueError("XTranslator saved receipt session differs from its published chunk")
            if not recorded["xt_session_uid"] and saved.get("mode") != "original-English":
                raise ValueError("XTranslator receipt session is missing")
            if recorded["speech_engine"] != "xtranslator" or recorded["engine"] != "xtranslator":
                raise ValueError("XTranslator receipt is mislabeled as another engine")
        for key in ("audio", "sha256", "start", "end", "revision", "generic_version"):
            if key not in recorded or saved.get(key) != recorded[key]:
                raise ValueError("Generic saved receipt identity differs from its published chunk")
        return dict(saved, receipt_path=str(path), receipt_sha256=hashlib.sha256(raw).hexdigest(), accounting_engine=engine)

    @staticmethod
    def _generic_span(span):
        if not isinstance(span, (list, tuple)) or len(span) != 2 or any(isinstance(v, bool) for v in span):
            raise ValueError("Invalid generic span")
        start, end = map(float, span)
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError("Invalid generic span")
        return start, end

    def _generic_prepared_media_evidence(self, receipt, payload, start, end):
        """Generic PCM fragments only. Full publication is resolved from all chunk receipts in status()."""
        if payload.get("generic_version") != "generic-dub-v1":
            receipt["accounting_status"] = "generic-receipt-missing-or-unsupported"
            return receipt
        engine = payload["accounting_engine"]
        fragments, seen = [], set()
        for take in payload.get("generic_takes") or ():
            if engine == "xtranslator" and not isinstance(take.get("origin_session_uid"), str):
                raise ValueError("XTranslator take origin session is missing")
            if engine == "xtranslator" and not take["origin_session_uid"]:
                raise ValueError("XTranslator take origin session is empty")
            full_start, full_end = self._generic_span([take["start"], take["end"]])
            mixed_start, mixed_end = self._generic_span(take["mixed_interval"])
            digest, ident = take["sha256"], take["id"]
            if not isinstance(ident, str) or not ident or not re.fullmatch(r"[a-fA-F0-9]{64}", str(digest)):
                raise ValueError("Invalid generic take identity")
            if mixed_start < max(full_start, float(payload["start"])) - 1e-6 or mixed_end > min(full_end, float(payload["end"])) + 1e-6:
                raise ValueError("Generic fragment exceeds its take or chunk")
            lo, hi = max(start, mixed_start), min(end, mixed_end)
            if hi <= lo or (ident, digest, lo, hi) in seen:
                continue
            seen.add((ident, digest, lo, hi))
            sources = [list(self._generic_span(span)) for span in take["source_spans"]]
            fragments.append({"take_id": ident, "sha256": digest, "source_spans": sources,
                "source_start": take["source_start"], "source_end": take["source_end"],
                "start": lo, "end": hi, "full_placement": [full_start, full_end],
                "owner_chunk": take["owner_chunk"], "required_chunk_indices": list(take["required_chunk_indices"]),
                "overflow_past_end_seconds": take.get("overflow_past_end_seconds", 0.0),
                "completeness": "prepared-fragment; full-take publication is evaluated separately",
                "source_audio_language": "unknown", "speech_engine": engine,
                **({"origin_session_uid": take.get("origin_session_uid"), "seg_uid": take.get("seg_uid"),
                    "speaker_id": take.get("speaker_id")} if engine == "xtranslator" else {})})
        union = self._evidence_union([[item["start"], item["end"]] for item in fragments])
        originals = [self._generic_span(span) for span in payload["generic_original_intervals"]]
        clipped_originals = [[max(start, a), min(end, b)] for a, b in originals if min(end, b) > max(start, a)]
        receipt.update(accounting_status=engine + "-mixed-fragment-receipts", speech_engine=engine, engine=engine,
            generic_version="generic-dub-v1", generated_intervals=fragments, generated_union=union,
            english_generated_seconds=round(sum(b - a for a, b in union), 6),
            generated_seconds_scope="PCM fragments included in prepared media; partial takes do not earn full source credit",
            generic_original_intervals=clipped_originals, uses_original_audio=bool(clipped_originals),
            uses_fallback_audio=bool(clipped_originals),
            original_audio_scope="source speech kept per generic chunk receipt; soundtrack bed is separate")
        if engine == "xtranslator":
            receipt.update({key: payload.get(key) for key in ("xt_version", "xt_session_uid", "detected_language",
                "source_language", "speaker_count", "speaker_ids", "speaker_state_reset_count", "backend_identity")})
            receipt["original_audio_scope"] = "explicit receipt source-voice intervals; failed translation/synthesis is bed only"
            if payload.get("mode") == "original-English":
                receipt["uses_fallback_audio"] = False
        return receipt

    def _generic_take_evidence(self, value, evidence):
        """Receipt-backed completeness and client-reported progression, never human hearing evidence."""
        takes, conflicts, prepared_hashes = {}, set(), {}
        for segment in evidence["prepared_segments"]:
            prepared_hashes.setdefault(segment["segment_index"], set()).add(segment.get("chunk_audio_sha256"))
            if segment.get("generic_version") != "generic-dub-v1":
                continue
            if segment.get("speech_engine") != value.get("speech_engine"):
                continue
            for take in segment["generated_intervals"]:
                ident = take["take_id"]
                if ident in takes and any(takes[ident].get(key) != take.get(key) for key in
                        ("sha256", "full_placement", "source_spans", "required_chunk_indices", "owner_chunk",
                         "origin_session_uid", "seg_uid", "speaker_id")):
                    conflicts.add(ident)
                takes[ident] = take
        rows, chunks = [], {}
        for ident, take in sorted(takes.items()):
            row = {"take_id": ident, "sha256": take["sha256"], "source_spans": take["source_spans"],
                   "speech_engine": value.get("speech_engine"),
                   "full_placement": take["full_placement"], "owner_chunk": take["owner_chunk"],
                   "published_full": False, "prepared_full": False, "played_full": False,
                   "source_credit_seconds": 0.0, "scope": "chunk receipts and heartbeat positions; not hearing proof"}
            rows.append(row)
            if value.get("speech_engine") == "xtranslator":
                row.update(origin_session_uid=take.get("origin_session_uid"), seg_uid=take.get("seg_uid"),
                           speaker_id=take.get("speaker_id"))
            try:
                a, b = self._generic_span(take["full_placement"])
                seg = value["segment_seconds"]
                expected = list(range(math.floor(a / seg), math.ceil(b / seg)))
                row["required_chunk_indices"] = expected
                overflow = max(float(take.get("overflow_past_end_seconds", 0.0)), b - value["duration"], 0.0)
                row["overflow_past_end_seconds"] = overflow
                if not math.isfinite(overflow) or overflow > 0:
                    row["publication_state"] = "overflow-past-end; no full credit"
                    continue
                if ident in conflicts or take["required_chunk_indices"] != expected:
                    row["publication_state"] = "conflicting-take-receipts"
                    continue
                missing, prepared_all = [], True
                row["published_chunk_receipts"] = []
                for index in expected:
                    if index not in chunks:
                        chunk = self.store.chunk(value["job_id"], index)
                        payload = (chunk or {}).get("payload") or {}
                        if chunk and chunk["state"] in READY_CHUNK_STATES:
                            payload = self._generic_chunk_payload(chunk)
                        chunks[index] = (chunk, payload)
                    chunk, payload = chunks[index]
                    if (not chunk or chunk["state"] not in READY_CHUNK_STATES
                            or payload.get("generic_version") != "generic-dub-v1"
                            or payload.get("accounting_engine") != value.get("speech_engine")
                            or payload.get("revision") != value["job_revision"]
                            or not re.fullmatch(r"[a-fA-F0-9]{64}", str(payload.get("sha256", "")))):
                        missing.append(index)
                        prepared_all = False
                        continue
                    row["published_chunk_receipts"].append({"segment_index": index,
                        "receipt_path": payload["receipt_path"], "receipt_sha256": payload["receipt_sha256"],
                        "chunk_audio_sha256": payload["sha256"]})
                    matched = [item for item in payload.get("generic_takes") or ()
                               if item.get("id") == ident and item.get("sha256") == take["sha256"]
                               and [item.get("start"), item.get("end")] == [a, b]
                               and item.get("source_spans") == take["source_spans"]
                               and item.get("required_chunk_indices") == expected
                               and (value.get("speech_engine") != "xtranslator" or
                                    all(item.get(key) == take.get(key) for key in
                                        ("origin_session_uid", "seg_uid", "speaker_id", "owner_chunk")))]
                    if len(matched) != 1:
                        missing.append(index)
                        prepared_all = False
                        continue
                    lo, hi = self._generic_span(matched[0]["mixed_interval"])
                    wanted_lo, wanted_hi = max(a, index * seg), min(b, (index + 1) * seg)
                    if (abs(lo - wanted_lo) > 1e-6 or abs(hi - wanted_hi) > 1e-6
                            or float(payload["start"]) > lo + 1e-6 or float(payload["end"]) < hi - 1e-6):
                        missing.append(index)
                    if payload["sha256"] not in prepared_hashes.get(index, set()):
                        prepared_all = False
                row["missing_or_partial_chunk_indices"] = missing
                row["published_full"] = not missing
                row["prepared_full"] = not missing and prepared_all
                row["publication_state"] = "published-full-receipts" if not missing else "published-partial-or-pending"
                covered = any(lo <= a + 1e-6 and hi >= b - 1e-6 for lo, hi in evidence["played_ranges"])
                row["heartbeat_covers_full_span"] = covered
                row["played_full"] = row["prepared_full"] and covered
                if row["published_full"]:
                    source_union = self._evidence_union([list(self._generic_span(s)) for s in take["source_spans"]])
                    row["source_credit_seconds"] = round(sum(hi - lo for lo, hi in source_union), 6)
            except Exception as error:
                row.update(publication_state="unverified", accounting_error=type(error).__name__,
                           published_full=False, prepared_full=False, played_full=False, source_credit_seconds=0.0)
        return rows

    def _keeps_ready(self, job_id):
        job = self.store.job(job_id)
        return bool(job and job["state"] in KEEP_READY_STATES)

    def dub_boundary(self, value):
        """P1-1: start (s) of the first part from the playhead that can no longer play while the job is blocked
        (neither muxed nor backed by ready audio; a blocked job claims no new chunks). None: all remaining parts can."""
        seg = value["segment_seconds"]
        total = math.ceil(value["duration"] / seg)
        states = {int(c["idx"]): c["state"] for c in self.store.chunks(value["job_id"])}
        index = max(value["start_index"], int(value["position"] // seg))
        while index < total and (index in value["segments"] or states.get(index) in READY_CHUNK_STATES):
            index += 1
        return None if index >= total else index * seg

    def _ready_audio(self, job, index):
        """P1-1: the verified, already-ready audio of one chunk of a blocked job (same checks as
        Coordinator.audio_chunk). Never requests work, never relabels a chunk, never substitutes audio."""
        chunk = self.store.chunk(job["id"], index)
        if not chunk or chunk["state"] not in READY_CHUNK_STATES:
            raise ValueError("Playback is blocked: " + job["state"])
        payload = chunk["payload"]
        if int(payload.get("revision", job["payload"].get("revision", 1))) != job["payload"].get("revision", 1):
            raise ValueError("Playback revision changed; reopen the current stream")
        path = Path(payload["audio"])
        if file_hash(path) != payload["sha256"]:
            raise ValueError("A generated chunk changed; playback refused")
        data = wav_info(path)
        if data["rate"] != RATE or data["channels"] != CHANNELS or data["width"] != 2:
            raise ValueError("Generated chunk has an incompatible format")
        if data["frames"] != round(payload["end"] * RATE) - round(payload["start"] * RATE):
            raise ValueError("Generated chunk duration does not match its timeline")
        return path

    def ready_through(self, value):
        index = value["start_index"]
        while index in value["segments"]:
            index += 1
        return min(value["duration"], index * value["segment_seconds"])

    def media_url(self, value, suffix):
        expires = value["expires"]
        message = "%s:%s:%s" % (value["id"], value["generation"], expires)
        signature = hmac.new(self.config["stream_secret"].encode(), message.encode(), hashlib.sha256).hexdigest()
        return self.config["public_url"].rstrip("/") + "/tv-media/" + value["id"] + "/" + suffix + "?" + urlencode({
            "generation": value["generation"], "expires": expires, "signature": signature})

    def authorize_media(self, ident, query):
        value = self.session(ident, touch=False)
        generation, expires = int(query["generation"]), int(query["expires"])
        message = "%s:%s:%s" % (ident, generation, expires)
        expected = hmac.new(self.config["stream_secret"].encode(), message.encode(), hashlib.sha256).hexdigest()
        if generation != value["generation"] or expires < time.time() or expires > time.time() + 13 * 3600:
            raise ValueError("Expired TV media session")
        if not hmac.compare_digest(expected, query.get("signature", "")):
            raise ValueError("Invalid TV media signature")
        if value["job_id"]:
            job = self.store.job(value["job_id"])
            if not job or job["state"] == "cancelled":
                # Coordinator.cancel keeps the revision: signed links of a cancelled/missing job are not served.
                raise ValueError("Dubbing was stopped; refresh the playback session")
            if job["payload"].get("revision", 1) != value["job_revision"]:
                raise ValueError("Dubbing was revised; refresh the playback session")
        return value

    def _mux_job_current(self, value):
        """A finished mux of a dub session is published only while its job still exists, is not cancelled and is at
        the session's revision (error/needs_review keep their ready parts: KEEP_READY_STATES)."""
        if not value["job_id"]:
            return True
        job = self.store.job(value["job_id"])
        return bool(job and job["state"] != "cancelled"
                    and job["payload"].get("revision", 1) == value["job_revision"])

    # --- playback loudness (config tv.loudness) -------------------------------------------------------------
    # ONE constant gain per dub job, measured once on the original movie (loudness_job.compute_job_gain), stored in
    # the job's own payload under loudness_job.PAYLOAD_KEY = {"key", "result"} and applied identically to every segment
    # by render(). Reads: render()/status() read job["payload"]; Writes: Store.update_job(job_id, playback_loudness=...)
    # (one atomic merge, state untouched). A seek, resume, second session or server restart reuses the stored record
    # while its key (source fingerprint + sampling plan + target + audio track + layout) is unchanged.

    def _surround_stream(self, value):
        """The selected source audio stream when this playback is 5.1 (surround on, 6-channel 5.1 track), else None. One answer
        for a whole playback: every segment of a stream has the same channel layout."""
        if (value.get("surround") or self.config["tv"].get("surround", "auto")) == "stereo":
            return None
        stream = next((s for s in value["streams"] if s.get("index") == value["audio_index"]
                       and s.get("codec_type") == "audio"), None)
        if stream and int(stream.get("channels") or 0) == 6 and stream.get("channel_layout") in ("5.1", "5.1(side)"):
            return stream
        return None

    def _loudness_cfg(self, value):
        """tv.loudness for this playback. A 5.1 stream is set by its DIALOGUE: the center channel alone is measured (EBU R 128 s4) and the
        film's gain brings it to tv.loudness.dialogue_lufs (default -22 LUFS), so a film whose dialogue is quiet and one whose dialogue is
        loud are heard at the same spoken volume. A whole-film target (-14) left the dialogue of one measured film at -17.6 LUFS, far too
        loud, while another film needed that much. A stereo stream keeps the whole-film target, measured on the downmix the dub is built
        from."""
        from . import loudness_job as _lj
        cfg = _lj.settings(self.config["tv"]["loudness"])
        if self._surround_stream(value) is not None:
            cfg = {**cfg, "layout": "center", "target_lufs": cfg["dialogue_lufs"]}
        return cfg

    def _surround_mix(self, audio, audio_hash):
        """The chunk's 5.1 mix (english-surround.flac beside english.wav) when its receipt binds it to exactly this english.wav and
        its own hash holds, else None."""
        folder = Path(audio).parent
        mix, receipt = folder / "english-surround.flac", folder / "english-surround.json"
        try:
            saved = json.loads(receipt.read_text())
            if saved.get("english_sha256") == audio_hash and mix.is_file() and file_hash(mix) == saved.get("surround_sha256"):
                return mix
        except (OSError, ValueError):
            pass
        return None

    def _loudness_key(self, value):
        from . import loudness_job as _lj
        cfg = self._loudness_cfg(value)
        return _lj.job_key(value["fingerprint"], value["audio_index"], value["duration"], cfg)[0]

    def _stored_loudness(self, value, job):
        """The job's current gain record ({"key", "result"}) or None. In-process copy first, then the job payload."""
        from . import loudness_job as _lj
        key = self._loudness_key(value)
        for record in (self._loudness_memo.get(job["id"]), (job["payload"] or {}).get(_lj.PAYLOAD_KEY)):
            if isinstance(record, dict) and record.get("key") == key and _lj.valid_result(record.get("result")):
                return record
        return None

    def _remember_loudness(self, job_id, record):
        self._loudness_memo[job_id] = record
        while len(self._loudness_memo) > 16:
            self._loudness_memo.pop(next(iter(self._loudness_memo)))

    def _playback_loudness(self, value, job, owner):
        """The job gain result for this render: stored, else measured now (cancellable through ``owner``) and stored.
        None when tv.loudness is off. Never raises except Cancelled; a failed measurement is a stored 0 dB 'unmeasured'
        result, so the gain can never change in the middle of a job."""
        if not _loudness_on(self.config):
            return None
        from . import loudness_job as _lj
        record = self._stored_loudness(value, job)
        if record is not None:
            self._remember_loudness(job["id"], record)
            return record["result"]
        # Run-ahead muxing encodes several segments at once: only ONE thread may measure the job gain, the others wait
        # for its stored result (a cancelled waiter gives up through owner.check()), so the gain stays constant.
        while not self._loudness_lock.acquire(timeout=0.5):
            owner.check()
        try:
            record = self._stored_loudness(value, job)
            if record is not None:
                self._remember_loudness(job["id"], record)
                return record["result"]
            cfg = self._loudness_cfg(value)
            key = self._loudness_key(value)
            stream = next((s for s in value["streams"] if s.get("index") == value["audio_index"]
                           and s.get("codec_type") == "audio"), None)
            self._loudness_active = job["id"]
            try:
                result = _lj.compute_job_gain(
                    value["source"], self.config["ffmpeg"], self.config["ffprobe"], audio_index=value["audio_index"],
                    stream=stream, duration=value["duration"], owner=owner, budget_seconds=cfg["budget_seconds"],
                    samples=cfg["samples"], excerpt_seconds=cfg["excerpt_seconds"], target_lufs=cfg["target_lufs"],
                    layout=cfg["layout"], key=key, resume=self._loudness_resume.setdefault(job["id"], {}))
            finally:
                self._loudness_active = None
            record = {"key": key, "result": result}
            self._remember_loudness(job["id"], record)
            self._loudness_resume.pop(job["id"], None)
            try:
                self.store.update_job(job["id"], **{_lj.PAYLOAD_KEY: record})
            except Exception:
                pass   # e.g. the job was removed meanwhile; the in-process copy keeps this session's gain constant
            return result
        finally:
            self._loudness_lock.release()

    def _premeasure_loudness(self, ident, value):
        """While a dub job's first chunk is still being produced the mux thread has nothing to do: measure the job gain
        now (same thread, same cancellable owner kind as a mux, so a seek or stop interrupts it and the next call
        resumes from the excerpts already measured), so the first segment is not delayed by the measurement."""
        if not value["job_id"] or not _loudness_on(self.config):
            return
        job = self.store.job(value["job_id"])
        if job is None or self._stored_loudness(value, job) is not None:
            return
        owner = ProcessOwner()
        self.active_owner, self.active_session = owner, ident
        try:
            self._playback_loudness(dict(value), job, owner)
        except Cancelled:
            pass
        except Exception:
            pass   # a measurement problem is stored as 0 dB 'unmeasured' by _playback_loudness; render() decides again
        finally:
            self.active_owner, self.active_session = None, None

    def _loudness_status(self, value, job_payload):
        """status()['loudness']: the job's stored gain and status, read only (never measures)."""
        from . import loudness_job as _lj
        if not value["job_id"]:
            return {"enabled": True, "status": "not-dubbing", "gain_db": None, "applied": False}
        key = self._loudness_key(value)
        result = None
        for record in (self._loudness_memo.get(value["job_id"]), (job_payload or {}).get(_lj.PAYLOAD_KEY)):
            if isinstance(record, dict) and record.get("key") == key and _lj.valid_result(record.get("result")):
                result = record["result"]
                break
        return _lj.public_status(result, enabled=True, measuring=self._loudness_active == value["job_id"])

    def _lead_limit(self, value, limit):
        """Seconds-based mux horizon (actor beta): the window always reaches position + required_seconds, the lead
        status_fields requires, for every legal segment size and off-grid position. producer_policy.mux_limit's fixed
        +4 segment cap cannot (15 s segments at 14 s with a 60 s lead need 5). Bounded by required_seconds plus the
        current partial segment, and by EOF (required_seconds is already capped at the remaining duration)."""
        seg = value["segment_seconds"]
        total = math.ceil(value["duration"] / seg)
        required, _ = _producer.required_seconds(value, self.config["tv"])
        if _producer.gate_active(value, self.config["tv"]):
            # The throughput gate may ask for a longer lead (a slow machine); this window must be able to reach it or the
            # viewer would wait for a lead that can never be built (only matters while mux_ahead_unlimited is off).
            required = max(required, float(value.get("_gate_required_lead") or 0.0))
        need = math.ceil((value["position"] + required) / seg - 1e-9)
        return min(total, max(limit, need))

    def _live_playlist(self, value):
        """The sliding live playlist: the finished segments of the live relay, newest last, no ENDLIST (a live playlist only moves forward)."""
        with self.lock:
            window = list(value.get("live_window") or ())
            if not window:
                raise ValueError("The first live segment is still being prepared")
            lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:" + str(int(value.get("live_target") or value["segment_seconds"])),
                     "#EXT-X-MEDIA-SEQUENCE:" + str(window[0][0]), "#EXT-X-INDEPENDENT-SEGMENTS"]
            for sequence, seconds in window:
                lines += ["#EXTINF:%.6f," % seconds, self.media_url(value, str(sequence) + ".ts")]
            return ("\n".join(lines) + "\n").encode()

    def playlist(self, value):
        if value.get("live"):
            return self._live_playlist(value)
        with self.lock:
            indexes = []
            index = value["start_index"]
            while index in value["segments"]:
                indexes.append(index)
                index += 1
            if not indexes:
                raise ValueError("The first playback segment is still being prepared")
            lines = ["#EXTM3U", "#EXT-X-VERSION:6", "#EXT-X-TARGETDURATION:" + str(value["segment_seconds"]),
                     "#EXT-X-MEDIA-SEQUENCE:" + str(value["start_index"]), "#EXT-X-PLAYLIST-TYPE:EVENT",
                     "#EXT-X-INDEPENDENT-SEGMENTS", "#EXT-X-START:TIME-OFFSET=0,PRECISE=YES"]
            for number in indexes:
                seconds = min(value["segment_seconds"], value["duration"] - number * value["segment_seconds"])
                lines += ["#EXTINF:%.6f," % seconds, self.media_url(value, str(number) + ".ts")]
            if index * value["segment_seconds"] >= value["duration"]:
                lines.append("#EXT-X-ENDLIST")
            return ("\n".join(lines) + "\n").encode()

    def segment(self, value, index):
        with self.lock:
            path = value["segments"].get(index)
        if path is None:
            raise ValueError("That playback segment is not ready")
        return Path(path)

    # --- run-ahead mux (config tv.mux_ahead_unlimited, default ON for dubbed jobs) ---------------------------------
    # The scheduler tick (loop thread) never encodes: it calls producer_policy.produce() (dubbed audio toward EOF, only
    # `producer_queue` chunks in flight), asks the store for the playback chunk first, then starts up to
    # tv.mux_ahead_workers concurrent encodes (plus one reserved slot for the playback segment) of every segment whose
    # audio is ready, current position first then ascending, with no distance cap and regardless of pause. Each encode
    # is a ProcessOwner registered under the session; a seek/stop/revision change cancels them all, and a finished
    # encode is published only while its session, generation and job are still current (the existing epoch rules).
    # Switching the key off restores the legacy buffer_segments/MAX_EXTRA_MUX_SEGMENTS serial loop untouched.
    SNAPSHOT_TTL = 1.0

    def _tv_cfg(self):
        try:
            return self.config["tv"]
        except (KeyError, TypeError):
            return {}

    def _ahead_on(self, value):
        return bool(_playback_buffering_on(self.config) and _producer.ahead_enabled(value, self._tv_cfg()))

    def _ahead_state(self, value):
        state = value.get("_mux_ahead")
        if state is None or state["generation"] != value["generation"]:
            state = {"generation": value["generation"], "inflight": {}, "failed": {}, "hold_until": 0.0,
                     "disk": "ok", "launched": 0, "completed": 0, "snapshot": None, "request": None}
            value["_mux_ahead"] = state
        return state

    def _cancel_encodes(self, ident):
        with self.lock:
            owners = list(self._encodes.get(ident, ()))
        for owner in owners:          # outside the lock: cancelling kills the ffmpeg process tree
            owner.cancel()

    def _encode_total(self):
        with self.lock:
            return sum(len(owners) for owners in self._encodes.values())

    @staticmethod
    def _note_mux(value, index, started, finished, render_value):
        """Record one real (non-cached) encode: the machine's mux capacity feeds the start gate's rho."""
        if render_value.get("_mux_cached"):
            return
        seg = value["segment_seconds"]
        log = value.setdefault("_mux_log", [])
        log.append((started, finished, min(seg, value["duration"] - index * seg), index))
        del log[:-64]

    def _publish_segment(self, ident, value, generation, index, path, render_value, current_job, started, finished):
        """Install a finished mux only while the session, its generation and its job are current. Caller holds no lock."""
        with self.lock:
            if (self.sessions.get(ident) is value and value["generation"] == generation and current_job):
                value["segments"][index] = str(path)
                value["resource_wait"] = None
                if value.get("yt_stream"):
                    value.setdefault("yt_files", set()).add(str(path))       # every short segment this session made (removed when it ends)
                evidence = self._playback_evidence(value)
                prepared = render_value.get("_prepared_media_evidence")
                if prepared is not None:
                    evidence["prepared_segments"].append(prepared)
                self._note_mux(value, index, started, finished, render_value)
                return True
        return False

    def _ahead_tick(self, ident, value):
        now = time.time()
        cfg = self._tv_cfg()
        seg = value["segment_seconds"]
        total = math.ceil(value["duration"] / seg)
        generation = value["generation"]
        state = self._ahead_state(value)
        job_id = value["job_id"]
        # 1. Production first, on this tick, so no mux (they run on worker threads) can ever delay a request.
        _producer.produce(value, self.store, self.jobs, cfg, now, lookahead=int(self.config["lookahead_segments"]))
        job = self.store.job(job_id)
        if job is None or job["state"] == "cancelled":
            return
        position = value["position"]
        current = max(value["start_index"], int(position // seg))
        muxed = value["segments"]

        def chunk_ready(max_age):
            # The whole chunk table is decoded per read (receipts are large): reuse it for max_age seconds.
            snapshot = state["snapshot"]
            if snapshot is None or now - snapshot[0] >= max_age:
                states, _ = _producer.chunk_states(self.store, job_id)
                state["snapshot"] = snapshot = (now, states)
            return {i for i, s in snapshot[1].items() if s in _producer.READY}

        ready = chunk_ready(self.SNAPSHOT_TTL)
        first = next((i for i in range(current, total) if i not in muxed), None)
        if first is None:
            return
        if first not in ready:
            ready = chunk_ready(0.25)       # the playback segment's audio is awaited: look again promptly
        # 2. The chunk playback needs next is requested ahead of the far-ahead producer (buffer_policy convention).
        gap = next((i for i in range(current, total) if i not in ready), None)
        last = state["request"]
        if gap is not None and (last is None or last[0] != gap or now - last[1] >= 1.0):
            try:
                self.jobs.request(job_id, gap, -_producer.priority_anchor(value, now) + int(max(0.0, gap * seg - position)))
            except ValueError:
                pass            # a blocked job (P1-1) still muxes its ready chunks
            state["request"] = (gap, now)
        # 3. Disk: below tv.min_free_disk_gib nothing renders (the guard in render() stays); within the margin above
        #    it only the playback segment and the next one may encode.
        try:
            free = shutil.disk_usage(self.config.cache).free
        except OSError:
            free = None
        gate = "ok" if free is None else _producer.disk_gate(
            free, _producer._tv(cfg, "min_free_disk_gib", 10), _producer._tv(cfg, "mux_ahead_disk_margin_gib", 5.0))
        state["disk"] = gate
        if gate == "floor":
            value["resource_wait"] = "PC cache space is low. Free space before continuing playback."
            return
        if str(value.get("resource_wait") or "").startswith("PC cache space is low"):
            value["resource_wait"] = None
        if now < state["hold_until"]:
            return
        failed = state["failed"]
        record = failed.get(first)
        if record and record["count"] >= _producer.MUX_ATTEMPTS and first - current < _producer.URGENT_SPAN:
            with self.lock:         # playback reached a segment that could not be encoded: the legacy session error
                if value["generation"] == generation:
                    value["error"] = record["message"]
            return
        exhausted = [i for i, r in failed.items() if r["count"] >= _producer.MUX_ATTEMPTS]
        # The start gate never asks for a lead beyond a segment that can no longer be encoded (it would wait forever).
        value["_mux_parked_at"] = min(exhausted) * seg if exhausted else None
        parked = set(exhausted) | {i for i, r in failed.items() if now < r["next_try"]}
        with self.lock:
            inflight = set(state["inflight"])
        picks = _producer.plan_mux(current=current, total=total, muxed=muxed, inflight=inflight,
                                   ready=ready - parked, workers=_producer.mux_workers(cfg),
                                   active=self._encode_total(), ahead_allowed=gate == "ok")
        if first not in ready and not inflight:
            self._ensure_premeasure(ident, value)       # idle until the first chunk exists: measure the job gain
        for index in picks:
            owner = ProcessOwner()
            with self.lock:
                if value["generation"] != generation or self.sessions.get(ident) is not value:
                    return
                state["inflight"][index] = owner
                self._encodes.setdefault(ident, set()).add(owner)
                state["launched"] += 1
            threading.Thread(target=self._ahead_encode, args=(ident, value, index, generation, owner, state),
                             name="tv-mux-%d" % index, daemon=True).start()

    def _ahead_encode(self, ident, value, index, generation, owner, state):
        try:
            render_value = dict(value)
            started = time.time()
            path = self.render(render_value, index, owner)
            finished = time.time()
            current_job = self._mux_job_current(value)
            if self._publish_segment(ident, value, generation, index, path, render_value, current_job, started, finished):
                state["completed"] += 1
                state["failed"].pop(index, None)
        except Cancelled:
            pass
        except TimeoutError as error:
            if "not ready" not in str(error):       # an ffmpeg that exceeded its time budget is a real failure
                self._ahead_failure(ident, value, generation, index, error, state)
            # else: the chunk's audio was not ready after all; the next tick looks again
        except ResourceBusy as error:
            with self.lock:
                if self.sessions.get(ident) is value and value["generation"] == generation:
                    value["resource_wait"] = str(error)
            state["hold_until"] = time.time() + 3
        except Exception as error:
            self._ahead_failure(ident, value, generation, index, error, state)
        finally:
            with self.lock:
                state["inflight"].pop(index, None)
                owners = self._encodes.get(ident)
                if owners is not None:
                    owners.discard(owner)
                    if not owners:
                        self._encodes.pop(ident, None)
            self.wake.set()

    def _ahead_failure(self, ident, value, generation, index, error, state):
        """A far-ahead segment that fails is retried (MUX_ATTEMPTS, backed off) and then parked; it becomes the session
        error only when playback needs it. The segment playback needs next fails the session at once (legacy rule)."""
        with self.lock:
            if self.sessions.get(ident) is not value or value["generation"] != generation:
                return
            record = state["failed"].setdefault(index, {"count": 0})
            record["count"] += 1
            record.update(message=str(error), at=time.time(),
                          next_try=time.time() + _producer.MUX_RETRY_SECONDS * record["count"])
            seg = value["segment_seconds"]
            current = max(value["start_index"], int(value["position"] // seg))
            first = next((i for i in range(current, math.ceil(value["duration"] / seg)) if i not in value["segments"]), None)
            if index == first and first - current < _producer.URGENT_SPAN:
                value["error"] = str(error)         # playback is at (or next to) the segment: the legacy rule

    def _ensure_premeasure(self, ident, value):
        if not value["job_id"] or not _loudness_on(self.config):
            return
        thread = self._premeasure_thread
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(target=self._premeasure_loudness, args=(ident, value),
                                  name="tv-loudness-premeasure", daemon=True)
        self._premeasure_thread = thread
        thread.start()

    def _sweep_partials(self, max_age=3600):
        """Stale .partial.ts files of a crashed run (a live encode's file is far younger)."""
        try:
            now = time.time()
            for path in (self.config.cache / "tv-media").glob("*.partial.ts"):
                try:
                    if now - path.stat().st_mtime > max_age:
                        path.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    def _run_ahead_status(self, value):
        """status()['run_ahead']: what the mux scheduler and the encoder choice are doing (additive, informational)."""
        cfg = self._tv_cfg()
        state = value.get("_mux_ahead") if value.get("_mux_ahead", {}).get("generation") == value["generation"] else None
        choice = self._encoder_choice.get(value["id"])
        if choice is not None and choice.get("generation") != value["generation"]:
            choice = None
        floor = _producer._tv(cfg, "min_free_disk_gib", 10)
        return {"mux_ahead_unlimited": _producer.ahead_enabled(value, cfg), "workers": _producer.mux_workers(cfg),
                "muxed_segments": len(value["segments"]),
                "inflight_segments": sorted(state["inflight"]) if state else [],
                "encodes_running": self._encode_total(),
                "disk": state["disk"] if state else "unknown",
                "disk_floor_gib": floor,
                "disk_floor_behaviour": "ahead-only stop at floor + mux_ahead_disk_margin_gib; below the floor nothing "
                                        "renders (resource_wait is reported) and muxing resumes by itself when space is freed",
                "failed_segments": {str(i): {"attempts": r["count"], "message": r["message"]}
                                    for i, r in (state["failed"].items() if state else ())},
                "encoder": ({k: choice[k] for k in ("requested", "used", "reason", "fallback", "free_vram_mib")}
                            if choice else {"requested": str(value.get("encoder") or "software"), "used": None,
                                            "reason": "not chosen yet (first encode of this playback)"})}

    # --- video encoder choice ----------------------------------------------------------------------------------
    def _ffmpeg_has_nvenc(self):
        path = str(self.config["ffmpeg"])
        if path not in self._encoder_listing:
            try:
                self._encoder_listing[path] = b"h264_nvenc" in ProcessOwner().run([path, "-hide_banner", "-encoders"], timeout=30)
            except Exception:
                self._encoder_listing[path] = False
        return self._encoder_listing[path]

    def _free_vram_mib(self):
        if self._vram_probe is not None:
            return self._vram_probe()
        try:
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, timeout=5, check=True, **flags)
            return int(result.stdout.decode().strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _resolve_encoder(self, value, hdr_preserve):
        """(encoder, choice): 'nvenc' | 'software', decided ONCE per playback generation so one HLS stream never mixes
        encoders (a failure pins software for the rest of the generation). 'auto' prefers NVENC when h264_nvenc is
        listed and free VRAM >= tv.nvenc_min_free_mib; recorded in status()['run_ahead']['encoder']."""
        ident, generation = value["id"], value["generation"]
        with self._encoder_lock:
            choice = self._encoder_choice.get(ident)
            if choice and choice["generation"] == generation:
                return choice["used"], choice
            requested = str(value.get("encoder") or "software")
            wanted = requested in ("auto", "nvenc") and not hdr_preserve
            listed = self._ffmpeg_has_nvenc() if wanted else False
            free = self._free_vram_mib() if listed else None
            used, reason = _producer.choose_encoder(
                requested, nvenc_listed=listed, free_vram_mib=free, hdr_preserve=hdr_preserve,
                min_free_mib=int(_producer._tv(self._tv_cfg(), "nvenc_min_free_mib", 2048)))
            choice = {"generation": generation, "requested": requested, "used": used, "reason": reason,
                      "free_vram_mib": free, "checked_at": time.time(), "fallback": None}
            self._encoder_choice[ident] = choice
            return used, choice

    def _pin_software(self, value, reason):
        with self._encoder_lock:
            choice = self._encoder_choice.get(value["id"])
            if choice and choice["generation"] == value["generation"]:
                choice.update(used="software", fallback=reason, reason="software after NVENC failure")

    @staticmethod
    def _video_codec(encoder, preserve_hdr, threads):
        """(codec, options). Software options are exactly the v4 ones; NVENC is a quality-equivalent constant-quality VBR."""
        if preserve_hdr:
            return "libx265", ["-preset", "veryfast", "-crf", "22",
                               "-x265-params", "pools=%s:frame-threads=1:log-level=error" % threads]
        if encoder == "nvenc":
            return "h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0", "-profile:v", "high",
                                  "-forced-idr", "1"]
        return "libx264", ["-preset", "veryfast", "-crf", "22"]

    def _sweep_live(self):
        """Folders of live relays a crashed run left behind (no session exists yet when the loop starts)."""
        shutil.rmtree(self.config.cache / "tv-live", ignore_errors=True)

    def loop(self):
        self._sweep_partials()
        self._sweep_live()
        while not self.stop_event.is_set():
            with self.lock:
                snapshot = list(self.sessions.items())
            worked = False
            for ident, value in snapshot:
                if time.time() - value["touched"] > self.config["tv"]["idle_seconds"]:
                    self.stop(ident)
                    continue
                if value["error"]:
                    continue
                if value.get("yt_stream") or value.get("live"):
                    try:
                        self._youtube_tick(ident, value)     # YouTube profile: its own scheduler (progressive file / live relay)
                    except Exception as error:               # one session's problem must not stop every session's scheduler
                        logging.exception("YouTube scheduler failed")
                        value["error"] = "YouTube playback failed: " + (str(error) or error.__class__.__name__)[:160]
                    continue
                if not value["dub"] and not value["compatibility"]:
                    continue
                if value["job_id"]:
                    job = self.store.job(value["job_id"])
                    if job and job["payload"].get("revision", 1) != value["job_revision"]:
                        with self.lock:
                            value["job_revision"] = job["payload"].get("revision", 1)
                            value["generation"] += 1
                            value["epoch"] += 1
                            value["start_index"] = int(value["position"] // value["segment_seconds"])
                            value["segments"] = {}
                        self._cancel_encodes(ident)
                if self._ahead_on(value):
                    # tv.mux_ahead_unlimited: production and mux run continuously toward EOF on their own threads.
                    self._ahead_tick(ident, value)
                    continue
                current = max(value["start_index"], int(value["position"] // value["segment_seconds"]))
                limit = min(math.ceil(value["duration"] / value["segment_seconds"]),
                            current + max(2, int(self.config["tv"]["buffer_segments"])))
                if _playback_buffering_on(self.config):
                    limit = self._lead_limit(value, _producer.mux_limit(value, limit, self.config["tv"]))
                missing = next((i for i in range(current, limit) if i not in value["segments"]), None)
                beta_priority = None
                if _playback_buffering_on(self.config) and value["job_id"]:
                    states = {}
                    for i in range(current, limit):
                        record = self.store.chunk(value["job_id"], i)
                        states[i] = record["state"] if record else None
                    # P0-1: plan over the mux_limit-extended window (same `current`), so a seek/resume lead larger
                    # than buffer_segments (60 s from an off-grid position) can actually be muxed.
                    plan = next_request(value, states, time.time(), buffer_segments=limit - current)
                    missing = None if plan is None else plan["index"]
                    beta_priority = None if plan is None else plan["priority"]
                if _playback_buffering_on(self.config) and value["job_id"]:
                    _producer.produce(value, self.store, self.jobs, self.config["tv"], time.time(),
                                      lookahead=int(self.config["lookahead_segments"]))
                if missing is None:
                    continue
                if value["job_id"]:
                    try:
                        self.jobs.request(value["job_id"], missing,
                                          -int(time.time()) if beta_priority is None else beta_priority)
                    except ValueError:
                        # P1-1: a blocked (error/needs_review) job still muxes its already-ready chunks.
                        if not self._keeps_ready(value["job_id"]):
                            continue
                    chunk = self.store.chunk(value["job_id"], missing)
                    if not chunk or chunk["state"] not in ("ready", "approved"):
                        # Idle until the chunk exists: measure the job's loudness gain now (tv.loudness; no-op when off
                        # or already stored) instead of delaying the first segment's mux by it.
                        self._premeasure_loudness(ident, value)
                        continue
                generation = value["generation"]
                owner = ProcessOwner()
                self.active_owner, self.active_session = owner, ident
                try:
                    render_value = dict(value)
                    mux_started = time.time()
                    path = self.render(render_value, missing, owner)
                    mux_finished = time.time()
                    current_job = self._mux_job_current(value)   # the mux is long: re-check the job before publishing
                    self._publish_segment(ident, value, generation, missing, path, render_value, current_job,
                                          mux_started, mux_finished)
                except Cancelled:
                    pass
                except ResourceBusy as error:
                    value["resource_wait"] = str(error)
                    self.stop_event.wait(3)
                except Exception as error:
                    with self.lock:
                        if self.sessions.get(ident) is value and value["generation"] == generation:
                            value["error"] = str(error)
                finally:
                    self.active_owner, self.active_session = None, None
                worked = True
                break
            if not worked:
                self.wake.wait(.5)
                self.wake.clear()

    def render(self, value, index, owner):
        if shutil.disk_usage(self.config.cache).free < self.config["tv"]["min_free_disk_gib"] * 1024**3:
            raise ResourceBusy("PC cache space is low. Free space before continuing playback.")
        youtube = bool(value.get("yt_stream"))
        # A YouTube-profile session reads a file that is still growing: its fingerprint is id-based and stable (the bytes change), and a
        # segment is rendered only once the file covers it (the scheduler checks too; a truncated segment must never be published).
        if not youtube and fingerprint(value["source"]) != value["fingerprint"]:
            raise ValueError("The source movie changed; reopen it")
        start = index * value["segment_seconds"]
        duration = min(value["segment_seconds"], value["duration"] - start)
        if youtube:
            progress = self._yt_progress(value)
            if not progress["complete"] and start + duration + _ystream.READABLE_MARGIN > progress["readable"]:
                raise TimeoutError("The downloaded part of the video is not ready yet")
        audio, audio_hash = None, "original"
        if value["job_id"]:
            job = self.store.job(value["job_id"])
            if job and job["state"] in KEEP_READY_STATES:
                audio = self._ready_audio(job, index)
            else:
                try:
                    audio = self.jobs.audio_chunk(job["id"], index, job["payload"].get("revision", 1), wait=False)
                except ValueError:
                    job = self.store.job(value["job_id"])   # the job became blocked meanwhile
                    if not job or job["state"] not in KEEP_READY_STATES:
                        raise
                    audio = self._ready_audio(job, index)
            audio_hash = file_hash(audio)
        value["_prepared_media_evidence"] = self._prepared_media_evidence(value, index, audio_hash, start, duration)
        # Playback loudness: only dub audio (a job's chunk) is levelled, with the job's ONE constant gain. Off (the
        # default), 0 dB, or no dub: gain_db stays 0.0 and the mux identity and filter are byte-identical to before.
        loudness = self._playback_loudness(value, job, owner) if audio and _loudness_on(self.config) else None
        gain_db = float(loudness["gain_db"]) if loudness else 0.0
        leveled = abs(gain_db) >= 0.0005
        if loudness:
            value["_prepared_media_evidence"]["playback_loudness"] = {
                "status": loudness["status"], "gain_db": gain_db, "applied": leveled, "target": loudness.get("target"),
                "measured_lufs": loudness.get("measured_lufs"), "key": loudness.get("key")}
        video = next((s for s in value["streams"] if s.get("codec_type") == "video"
                      and not s.get("disposition", {}).get("attached_pic")), None)
        hdr = bool(video) and video.get("color_transfer") in ("smpte2084", "arib-std-b67")
        preserve_hdr = hdr and value["hdr_mode"] == "auto"
        # 'auto' prefers NVENC when ffmpeg lists h264_nvenc and the nvenc_min_free_mib check passes; software otherwise
        # (and always for HDR-preserving libx265). Chosen once per playback generation, recorded in status().
        encoder, _ = self._resolve_encoder(value, preserve_hdr)
        identity_parts = [value["fingerprint"], value["audio_index"], index, audio_hash,
                          value["quality"], value["upscale"], value["hdr_mode"],
                          encoder, value["encoder_threads"], "tv-mux-cpu-v4" if encoder == "software" else "tv-mux-nvenc-v1"]
        if youtube:
            # segment length and rate cap are part of what a cached segment is (a library segment's index implies 30 s; this one does not)
            identity_parts.append(["youtube", value["segment_seconds"], str(self._yt_opt("stream_maxrate")), str(self._yt_opt("stream_bufsize"))])
        audio_filter = "apad"
        if leveled:
            from . import loudness_job as _lj, playback_loudness as _pl
            curve = {k: v for k, v in _lj.settings(self.config["tv"]["loudness"]).items()
                     if k in ("knee_dbfs", "ceiling_dbfs")}    # defaults -3 / -1 dBFS
            identity_parts.append(_pl.mux_identity(gain_db, **curve))   # a different gain/curve is a different mux output
            audio_filter = _pl.mux_filter(gain_db, **curve) + ",apad"   # constant gain, stateless soft knee, then pad
        # 5.1 playback: every segment is six-channel AAC. A dub chunk uses its bound surround mix; a chunk whose english.wav is just the
        # original soundtrack (and a playback without dub) uses the film's own 5.1; a dub chunk without a surround mix is upmixed from
        # the stereo dub so the stream's layout never changes mid-way.
        surround = self._surround_stream(value)
        audio_input, audio_prefix, channel_args, audio_rate = audio, "", ["-ac", "2"], "192k"
        if surround is not None:
            channel_args, audio_rate = [], "384k"
            bound = self._surround_mix(audio, audio_hash) if audio else None
            mode = ((self.store.chunk(job["id"], index) or {}).get("payload") or {}).get("mode") if audio and job else None
            if bound is not None:
                audio_input = bound
                identity_parts.append(["surround", "mix", file_hash(bound)])
            elif audio and not passthrough_mode(mode):
                audio_prefix = SURROUND_UPMIX + ","
                identity_parts.append(["surround", "upmix"])
            else:
                audio_input = None
                identity_parts.append(["surround", "original"])
            if str(self.config["tv"].get("surround_rear", "off")).lower() != "on":
                identity_parts.append(["surround", "rear-off"])                  # a different audio chain is a different cached segment
                if not audio_prefix.startswith(SURROUND_UPMIX):                  # the stereo upmix already leaves the rear channels empty
                    audio_prefix = SURROUND_REAR_OFF + "," + audio_prefix
        identity = json.dumps(identity_parts)
        if value.get("speech_engine") == "xtranslator":
            identity = json.dumps(["xtranslator", identity])
        key = hashlib.sha256(identity.encode()).hexdigest()
        directory = self.config.cache / "tv-media"
        directory.mkdir(parents=True, exist_ok=True)
        output, receipt = directory / (key + ".ts"), directory / (key + ".json")
        if output.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            if file_hash(output) == saved.get("sha256"):
                value["_mux_cached"] = True      # a reuse is instant: not a measurement of the mux rate
                return output
        # One file per attempt: a cancelled encode of this segment (seek back to it) may still be unwinding.
        temporary = directory / ("%s.%s.partial.ts" % (key, uuid.uuid4().hex[:8]))
        args = [self.config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", str(start), "-i", value["source"]]
        if audio_input:
            args += ["-i", str(audio_input)]
        height, width = value["quality"], value["quality"] * 16 // 9
        scale = ("scale=w=%s:h=%s" % (width, height) if value["upscale"] else
                 "scale=w='min(%s,iw)':h='min(%s,ih)'" % (width, height))
        scale += ":force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,setsar=1"
        if not video:
            raise ValueError("No playable video stream")
        if hdr and not preserve_hdr:
            scale = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv," + scale
        codec, codec_options = self._video_codec(encoder, preserve_hdr, value["encoder_threads"])
        rate, buffer_size = ("30M" if height == 2160 else "8M"), ("60M" if height == 2160 else "16M")
        if youtube:
            rate, buffer_size = str(self._yt_opt("stream_maxrate")), str(self._yt_opt("stream_bufsize"))    # a 10 Mbit/s Wi-Fi TV
        color = (["-color_primaries", "bt2020", "-color_trc", video["color_transfer"],
                  "-colorspace", "bt2020nc"] if preserve_hdr else [])
        args += ["-map", "0:" + str(video["index"]), "-map", "1:a:0" if audio_input else "0:" + str(value["audio_index"]),
                 "-t", "%.6f" % duration, "-vf", scale, "-c:v", codec, *codec_options,
                 "-threads", str(value["encoder_threads"]), "-filter_threads", str(min(2, value["encoder_threads"])),
                 "-pix_fmt", "yuv420p10le" if preserve_hdr else "yuv420p",
                 "-maxrate", rate, "-bufsize", buffer_size, *color,
                 "-force_key_frames", "expr:gte(t,n_forced*2)",
                 "-c:a", "aac", "-b:a", audio_rate, "-ar", "48000", *channel_args,
                 "-af", audio_prefix + audio_filter, "-sn", "-dn", "-output_ts_offset", str(start),
                 "-muxdelay", "0", "-muxpreload", "0", "-mpegts_flags", "+resend_headers",
                 "-f", "mpegts", "-y", str(temporary)]
        try:
            try:
                owner.run(args, timeout=max(180, duration * 20))
                owner.check()
                fell_back = False
            except (Cancelled, TimeoutError):
                raise
            except RuntimeError as error:
                if encoder != "nvenc":
                    raise
                # NVENC refused (sessions, VRAM, driver): software for the rest of this playback generation, then
                # this segment again under the software identity. One stream never mixes encoders afterwards.
                self._pin_software(value, "h264_nvenc encode failed: " + str(error)[-240:])
                fell_back = True
            if fell_back:
                temporary.unlink(missing_ok=True)
                return self.render(value, index, owner)
            if not temporary.is_file() or temporary.stat().st_size < 188:
                raise ValueError("TV mux produced no transport stream")
            os.replace(temporary, output)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)       # never leave a cancelled/failed attempt's partial output
            except OSError:
                pass
            raise
        saved = {"sha256": file_hash(output), "start": start, "duration": duration, "audio": audio_hash}
        if leveled:
            saved["loudness_gain_db"] = gain_db
        receipt.write_text(json.dumps(saved))
        return output

    def captions(self, value):
        if value.get("live"):
            return b"WEBVTT\n\n"                     # a live stream has no subtitle file
        selection, base = value["subtitles"], value["start_index"] * value["segment_seconds"]
        if not value["dub"] and not value["compatibility"]:
            base = 0
        if selection == "dub" and value["job_id"]:
            cues = []
            seen = set()
            for chunk in self.store.chunks(value["job_id"]):
                if chunk["state"] in ("ready", "approved"):
                    for cue in chunk["payload"].get("cues", []):
                        if cue.get("id") not in seen:
                            seen.add(cue.get("id"))
                            cues.append({**cue, "text": cue.get("english") or cue.get("text", "")})
        elif selection == "off":
            cues = []
        else:
            index = None if selection == "auto" else int(selection)
            cues, _ = subtitle_cues(self.config, ProcessOwner(), value["source"],
                                     stream_index=index, discover=index is None)
        def stamp(seconds):
            ms = round(max(0, seconds) * 1000)
            h, ms = divmod(ms, 3600000)
            m, ms = divmod(ms, 60000)
            s, ms = divmod(ms, 1000)
            return "%02d:%02d:%02d.%03d" % (h, m, s, ms)
        rows = ["WEBVTT", ""]
        for cue in sorted(cues, key=lambda c: c["start"]):
            if cue["end"] <= base:
                continue
            text = str(cue["text"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            rows += [stamp(cue["start"] - base) + " --> " + stamp(cue["end"] - base), text, ""]
        return "\n".join(rows).encode("utf-8")

    def plex_sections(self):
        return self.catalog.sections()

    def plex_library(self, **values):
        return self.catalog.page(**values)

    def open_plex(self, device, key, resume=True):
        from .plex import selected_part
        metadata = self.plex.metadata(key)
        part, _, _ = selected_part(metadata)
        source = self.config.media_path(part["file"], nas=True)
        node = self._node(source)
        # Plex's selected/first audio stream is only Plex's default, not a choice made in this app, and it ignores
        # layout (it can be 7.1). open_movie applies the app's own default to the PROBED streams (select_audio:
        # 5.1, then stereo, then mono; never 7.1) and records dub_unsupported when none qualifies. The viewer's
        # explicit choice remains the TV 'audio' action. Nothing is written back to Plex.
        opened = self.open_movie(device, node["id"], None)
        value = self.session(opened["id"], device)
        value["plex_key"], value["title"] = str(key), metadata.get("title", source.stem)
        if resume and metadata.get("viewOffset"):
            return self.control(value["id"], device, "seek", float(metadata["viewOffset"]) / 1000)
        return self.status(value["id"], device)

    def plex_poster(self, key):
        metadata = self.plex.metadata(key)
        path = str(metadata.get("thumb", ""))
        if not re.fullmatch(r"/library/metadata/[0-9]+/thumb(?:/[0-9]+)?", path):
            raise ValueError("Plex poster is not available")
        from urllib.request import Request
        req = Request(self.config["plex"]["url"].rstrip("/") + path, headers=self.plex.headers)
        with self.plex.opener.open(req, timeout=10) as response:
            content = response.read(5 * 1024 * 1024 + 1)
            kind = response.headers.get_content_type()
            if len(content) > 5 * 1024 * 1024 or kind not in ("image/jpeg", "image/png", "image/webp"):
                raise ValueError("Unsupported poster response")
            return content, kind
