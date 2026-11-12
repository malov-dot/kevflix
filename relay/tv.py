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
import struct
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from .media import probe, fingerprint, file_hash, subtitle_cues, wav_info, RATE, CHANNELS
from .process import ProcessOwner, Cancelled
from .plex_browse import PlexBrowse
from .tv_probe import ProbeClips, SPEC_VERSION as PROBE_SPEC
from .buffer_policy import buffer_metrics, next_request, note_heartbeat
from . import producer_policy as _producer
from . import embedded_subtitles as _subs
from . import youtube_stream as _ystream
from . import segment_join as _join


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
# F19 picture path -------------------------------------------------------------------------------------------------
FMP4_FLAGS = "+frag_keyframe+empty_moov+default_base_moof+frag_discont+skip_trailer"
# Version of the non-default picture path: part of the cache key of every such segment and of a TV's HEVC failure mark.
# Bump it whenever the HEVC arguments, the fMP4 flags or a GPU chain change: old segments are then not reused and a TV
# that failed on the old stream tries HEVC again by itself.
PICTURE_VERSION = "picture-v1"
# What the TV's player reports (tv/app/player.js reportFailure) and how much each counts toward HEVC_FAILURES: a failed
# load or a stall is followed by the player's own try with its other decoder (1 each); after 'playback-error' the
# player tries nothing more (2).
PLAYER_FAILURE_WEIGHT = {"hls.js-failed": 1, "native-failed": 1, "stalled": 1, "playback-error": 2}


def hevc_codec_string(height):
    """RFC 6381 codec string of the HEVC stream for the master playlist: Main10, main tier, level 5.1 above 1080 lines,
    else 4.1 (an upper bound; the encoder chooses the real level)."""
    return "hvc1.2.4.L153.B0" if int(height) > 1080 else "hvc1.2.4.L123.B0"


def video_range(transfer):
    """HLS VIDEO-RANGE of a transfer characteristic."""
    return {"smpte2084": "PQ", "arib-std-b67": "HLG"}.get(transfer, "SDR")


SELFCHECK_PSNR_DB = 26.0          # the same frame through another codec or scaler
SELFCHECK_TONE_PSNR_DB = 14.0     # another tone-map curve: only a gross error (black, green, garbage) is below this
SELFCHECK_SIZE = (320, 180)


def gray_psnr(a, b):
    """PSNR in dB between two 8-bit gray pictures of the same size given as bytes; 99.0 when they are identical."""
    if not a or len(a) != len(b):
        raise ValueError("the two pictures differ in size")
    error = sum((x - y) * (x - y) for x, y in zip(a, b))
    return 99.0 if error == 0 else 10.0 * math.log10(255.0 * 255.0 * len(a) / error)


def split_fmp4(source, media):
    """Split ffmpeg's fragmented MP4 `source` (ftyp, moov, then moof/mdat pairs) in two: returns the init segment (every
    box before the first moof) as bytes and writes the rest (the media segment) to `media`. ValueError when the file
    is not that shape."""
    data = Path(source).read_bytes()
    position, kinds, first = 0, [], None
    while position + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[position:position + 8])
        if size == 1:
            if position + 16 > len(data):
                raise ValueError("TV mux produced a truncated MP4 box")
            size = struct.unpack(">Q", data[position + 8:position + 16])[0]
        elif size == 0:
            size = len(data) - position
        if size < 8 or position + size > len(data):
            raise ValueError("TV mux produced a damaged MP4 box")
        if kind == b"moof" and first is None:
            first = position
        kinds.append(kind)
        position += size
    if first is None or position != len(data) or kinds[:2] != [b"ftyp", b"moov"] or b"mdat" not in kinds:
        raise ValueError("TV mux produced no fragmented MP4")
    Path(media).write_bytes(data[first:])
    return data[:first]


DIAGNOSTIC_KEYS = ("stage", "engine", "code", "message", "ready_state", "network_state", "native_hls", "hls_js", "ua",
                   "session", "error", "hls", "codecs", "probe", "youtube", "open", "controls", "joins", "picture")
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
PLEX_REPORT_TIMEOUT = 5.0        # one Plex progress report (F24: sent by its own thread, never by a TV request)
PLEX_NEIGHBOUR_TIMEOUT = 3.0     # each Plex request of the previous / next episode look-up (at most four)
# The code a held chunk's receipt names: "MovieResourceWait: lease_sem_started: sem_started: ..." -> lease_sem_started.
_RESOURCE_CODE = re.compile(r"(?:^|:\s)([a-z][a-z0-9]*(?:_[a-z0-9]+)+):")
STATUS_SLOW_MS = 500             # F09: a status() that took this long is reported in the log (tv_status_slow)


class ResourceBusy(Exception):
    pass


class TVService:
    def __init__(self, config, store, coordinator, plex=None):
        self.config, self.store, self.jobs = config, store, coordinator
        _producer.LIGHT_ROWS = self._tv_cfg().get("light_chunk_states", True) is not False     # F09 switch
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
        # F19: one cached hevc_nvenc listing per ffmpeg path; picture-path pins of this server run (a GPU filter chain
        # or hevc_nvenc that failed once is not tried again until the server restarts).
        self._hevc_listing, self._picture_pins = {}, {}
        self._picture_checked = {}         # picture plan -> {"ok", "psnr_db", "limit_db"}: one self-check per plan and server run
        self._encoder_lock = threading.Lock()
        self._premeasure_thread = None
        self._vram_probe = None            # tests inject: callable() -> free MiB | None
        self._yt_source, self._yt_lock = None, threading.Lock()   # the YouTube hand-off downloader (relay/youtube_source.py), created on first use
        self._joins = None                 # F07 self-check per TV: {device id: {"fails": n, "off_until": unix time}} (lazy, from tv_devices)
        self.probe = ProbeClips(config)    # the "Test this TV" clips (relay/tv_probe.py); nothing runs until the TV asks
        # F24: Plex progress reports wait here for their own thread: newest per session, {session id: report}.
        self._plex_reports, self._plex_report_wake, self._plex_report_thread = {}, threading.Event(), None
        # Linger (tv.dub_linger_seconds): on-demand dub jobs whose last session ended, {job_id: time.time() at which the
        # job is cancelled unless a session picked it up again}; _enqueueing > 0 while control() attaches a job (no
        # expiry meanwhile). Both under self.lock.
        self._linger, self._enqueueing = {}, 0
        if hasattr(coordinator, "playback_active"):
            coordinator.playback_active = lambda: bool(self.sessions)      # background finish pauses during any playback

    def start(self):
        self._forget_stale_records()
        if self.config["tv"]["enabled"]:
            self.thread = threading.Thread(target=self.loop, name="tv-segment-mux", daemon=True)
            self.thread.start()

    def _forget_stale_records(self):
        """F11. A run that crashed never expired its 'tv-<session>' rows (they stayed in /api/status and in the watch
        context for ever); a revoked TV pairing is dropped 30 days after it was revoked."""
        records = getattr(self.store, "records", None)
        if not callable(records):
            return
        try:
            for record in records("sessions"):
                if str(record["id"]).startswith("tv-"):
                    self.store.expire_record("sessions", record["id"])
            cutoff = time.time() - 30 * 86400
            for record in records("tv_devices"):
                if record["payload"].get("revoked") and float(record.get("updated") or 0) < cutoff:
                    self.store.expire_record("tv_devices", record["id"])
        except Exception:
            logging.exception("Stale TV records were not cleared")

    def live_job_ids(self):
        """Dub jobs attached to a current session (housekeeping never prunes these)."""
        with self.lock:
            return {value["job_id"] for value in self.sessions.values() if value.get("job_id")}

    def live_segment_paths(self):
        """Every muxed file a current session may still serve (housekeeping never evicts these)."""
        with self.lock:
            return {str(path) for value in self.sessions.values() for path in value["segments"].values()}

    def session_count(self):
        """Open TV sessions (housekeeping deletes nothing while a film is on screen)."""
        with self.lock:
            return len(self.sessions)

    def close(self):
        self.stop_event.set()
        with self.lock:
            ids = list(self.sessions)
        for ident in ids:
            self.stop(ident)
        self._flush_linger()                            # server close: a lingering dub job is cancelled now
        if self._yt_source is not None:
            self._yt_source.cancel()                    # a download or start-up no session owns any more
        self.wake.set()
        self._plex_report_wake.set()
        reporter = self._plex_report_thread
        if reporter is not None:
            reporter.join(timeout=PLEX_REPORT_TIMEOUT + 1)      # the 'stopped' reports of the sessions just closed
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
                "container": str((metadata.get("format") or {}).get("format_name") or ""),
                "seamless": self._seamless_for(device),
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
            value["yt_channel"], value["yt_captions"] = info.get("channel") or info["uploader"], list(info.get("captions") or [])
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
                            "dub_default": bool(info["dub_default"])},
                "yt_channel": info.get("channel") or info["uploader"], "yt_captions": list(info.get("captions") or [])}
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
                 "youtube": {"id": video_id, "language": "", "uploader": live.get("uploader") or info["uploader"], "dub_default": False},
                 "yt_channel": info.get("channel") or live.get("uploader") or info["uploader"], "yt_captions": []}
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
        youtube = dict(value["youtube"], channel=value.get("yt_channel") or value["youtube"].get("uploader", ""))
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
            allowed += ["yt:" + item["code"] for item in value.get("yt_captions") or []]
            if selection not in allowed:
                raise ValueError("Choose an available text subtitle track")
            if selection.startswith("yt:"):
                try:
                    self.youtube_source().captions(value["youtube"]["id"], selection[3:])      # fetched once (a few seconds), then cached
                except Exception as error:
                    raise ValueError(str(error) or "Those captions could not be loaded")
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
                with self.lock:
                    self._enqueueing += 1       # a lingering job is not cancelled while it is being picked up again
                try:
                    job = self.jobs.enqueue(value["source"], audio_index=value["audio_index"],
                                            mode="ondemand", title=value["title"], subtitles=_subs.OFF,
                                            **({"resume_cancelled": True}
                                               if getattr(self.jobs, "RESUME_CANCELLED_VERSION", 0) >= 1 else {}),
                                            **({"initial_index": int(min(max(0, new_position), max(0, value["duration"] - .1))
                                                                    // value["segment_seconds"])}
                                               if _playback_buffering_on(self.config) else {}))
                    self._job_attached(job["id"])
                finally:
                    with self.lock:
                        self._enqueueing -= 1
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
                # One log line per seek: which window the viewer went to and how many requests of the position he left were parked.
                logging.info("tv_seek %s", json.dumps({"session": ident, "job_id": value["job_id"], "action": action,
                                                       "index": value["start_index"], "parked": value.get("seek_demoted", 0)}))
            if old_job and old_job != value["job_id"]:
                # Replaced by another dub job (audio track): that one needs the GPU now. Dubbing switched off: linger,
                # the viewer may switch it on again.
                self.cancel_unshared_job(old_job, linger=not value["job_id"])
        elif action in ("play", "pause"):
            value["state"] = "playing" if action == "play" else "paused"
            if _playback_buffering_on(self.config) or value.get("yt_stream") or value.get("live"):
                value["paused_intent"] = action == "pause"
        else:
            raise ValueError("Unsupported TV action")
        self.wake.set()
        return self.status(ident, device)

    def diagnostic(self, device, data):
        record = diagnostic_record(data)
        logging.info("tv_diagnostic %s", json.dumps(record, ensure_ascii=False))
        self._note_joins(device, record)                # F07: the TV's report of three joins
        try:
            self._note_diagnostic(device, record)
        except Exception:                       # bookkeeping must never fail the TV's report
            logging.exception("tv diagnostic bookkeeping failed")
        return {"ok": True}

    def _device_video(self, device):
        """The stored video record of one paired TV ({} when none): {"probe": {...}, "probe_at", "hevc_failed", ...}."""
        records = getattr(self.store, "records", None)
        if not device or not callable(records):
            return {}
        for record in records("tv_devices"):
            if record["id"] == device:
                return dict(record["payload"].get("video") or {})
        return {}

    def _store_device_video(self, device, **fields):
        """Merge fields into the TV's stored video record; a field given as None is removed."""
        records = getattr(self.store, "records", None)
        if not device or not callable(records):
            return
        for record in records("tv_devices"):
            if record["id"] == device:
                video = dict(record["payload"].get("video") or {})
                video.update(fields)
                self.store.put_record("tv_devices", device, {**record["payload"],
                                      "video": {k: v for k, v in video.items() if v is not None}})
                return

    def video_probe_needed(self, device):
        """True while this TV should run its silent decode test by itself (asked once per app start through
        /tv-api/settings): tv.video_codec is 'auto' and no result for the current clip set is stored for this TV."""
        if str(_producer._tv(self._tv_cfg(), "video_codec", "h264")) != "auto":
            return False
        return self._device_video(device).get("probe_spec") != PROBE_SPEC

    def _note_diagnostic(self, device, record):
        """F19: keep what the TV proved about itself with its record (the decode test, an HEVC playback that ran for
        30 s), and send a playback that fails on the HEVC stream (both decoders tried, a failure the player does not
        retry, or more than a tenth of the frames dropped) back to H.264 MPEG-TS. The TV itself is marked when the HEVC
        stream never played in this playback or dropped frames."""
        stage = record.get("stage")
        if stage == "probe" and isinstance(record.get("probe"), dict):
            self._store_device_video(device, probe=record["probe"], probe_at=time.time(), probe_spec=PROBE_SPEC,
                                     hevc_failed=None, hevc_failed_stage=None, hevc_failed_picture=None)
            return
        weight, dropping = PLAYER_FAILURE_WEIGHT.get(stage), False
        picture = record.get("picture") if stage == "picture" and isinstance(record.get("picture"), dict) else None
        if picture is not None:
            if picture.get("codec") != "hevc":
                return
            decoded, dropped = picture.get("dec"), picture.get("drop")
            dropping = (isinstance(decoded, (int, float)) and isinstance(dropped, (int, float)) and decoded > 0
                        and dropped * 10 > decoded)
            if not dropping and (picture.get("adv") or 0) >= 25 and (picture.get("w") or 0) > 0:
                self._store_device_video(device, hevc_played=time.time(), hevc_played_engine=record.get("engine"),
                                         hevc_played_picture="%sx%s dec %s drop %s" % (
                                             picture.get("w"), picture.get("h"), decoded, dropped))
                return
            weight = _producer.HEVC_FAILURES if dropping else None
        if not weight:
            return
        ident = str(record.get("session") or "")
        with self.lock:
            value = self.sessions.get(ident)
            if value is None or value["device_id"] != device:
                return
            generation = value["generation"]
            played = bool(value.get("beta_started")) and value.get("beta_epoch") == value.get("epoch")
        with self._encoder_lock:
            choice = self._encoder_choice.get(ident)
            if not choice or choice.get("generation") != generation or choice.get("codec") != "hevc":
                return
            choice["player_failures"] = int(choice.get("player_failures") or 0) + weight
            failures = choice["player_failures"]
        if failures >= _producer.HEVC_FAILURES and self._restart_picture(
                ident, generation, "the TV reported '%s' on the HEVC stream" % stage, codec_pin="tv-playback") \
                and (dropping or not played):
            self._store_device_video(device, hevc_failed=time.time(), hevc_failed_stage=stage,
                                     hevc_failed_picture=PICTURE_VERSION)

    def _restart_picture(self, ident, generation, reason, codec_pin=None):
        """The stream format or look must change (HEVC goes back to H.264, a tone-map chain failed): one new playback
        generation, exactly like a dub revision change, so one playlist never mixes containers. False when the session
        ended or already moved to another generation."""
        with self.lock:
            value = self.sessions.get(ident)
            if value is None or value["generation"] != generation:
                return False
            if codec_pin:
                value["codec_pinned"] = codec_pin
            value["generation"] += 1
            value["epoch"] += 1
            value["start_index"] = int(value["position"] // value["segment_seconds"])
            value["segments"] = {}
        logging.warning("tv picture path restarted for %s: %s", ident[:8], reason)
        self._cancel_encodes(ident)
        self.wake.set()
        return True

    # --- F07 self-check: the TV reports what its player did at three joins; nobody has to listen ------------------------
    JOINS_FAIL_STAGES = ("stalled", "playback-error", "hls.js-failed", "native-failed")
    JOINS_PIN_SECONDS = 14 * 86400

    def _joins_state(self):
        if self._joins is None:
            try:
                self._joins = {r["id"]: dict(r["payload"]["joins"]) for r in self.store.records("tv_devices")
                               if isinstance(r["payload"].get("joins"), dict)}
            except Exception:
                self._joins = {}
        return self._joins

    def _seamless_for(self, device):
        """Decided once, when a session opens, and kept for its whole life (one playlist never mixes the two segment formats):
        tv.seamless_joins, unless this TV's own reports failed in two sessions in a row; then the previous format for
        JOINS_PIN_SECONDS, after which the new one is tried again."""
        if not _producer._flag(self._tv_cfg(), "seamless_joins", True):
            return False
        return time.time() >= float((self._joins_state().get(device) or {}).get("off_until") or 0)

    def _note_joins(self, device, record):
        """One verdict per session: a "joins" report decides it; a failure report before any passing joins report counts as
        a failure. Applied when the session ends (_settle_joins)."""
        with self.lock:
            value = self.sessions.get(str(record.get("session") or ""))
            if value is None or value.get("device_id") != device or value.get("yt_stream") or value.get("live"):
                return
            stage = record.get("stage")
            if stage == "joins":
                ok, why = _join.joins_verdict(record.get("joins"))
                if ok is None:
                    return
            elif stage in self.JOINS_FAIL_STAGES and value.get("_joins") != "pass":
                ok, why = False, [str(stage)]
            else:
                return
            value["_joins"] = "pass" if ok else "fail"
            seamless = self._seamless_on(value)
        logging.info("tv_joins %s", json.dumps({"session": value["id"][:8], "device": str(device)[:8], "seamless": seamless,
                                                "ok": bool(ok), "why": why}))

    def _settle_joins(self, value):
        """A session ended: count its verdict for its TV. Only sessions that used the new format count."""
        verdict = value.get("_joins")
        if verdict not in ("pass", "fail") or not self._seamless_on(value):
            return
        device = value["device_id"]
        state = self._joins_state().get(device) or {}
        fails = 0 if verdict == "pass" else int(state.get("fails") or 0) + 1
        state = {"fails": fails, "off_until": float(state.get("off_until") or 0)}
        if fails >= 2:
            state = {"fails": 0, "off_until": time.time() + self.JOINS_PIN_SECONDS}
            logging.warning("tv_joins_pinned %s", json.dumps({"device": str(device)[:8], "days": self.JOINS_PIN_SECONDS // 86400,
                                                              "format": "previous segment format (tv.seamless_joins off for this TV)"}))
        self._joins_state()[device] = state
        try:
            for record in self.store.records("tv_devices"):
                if record["id"] == device:
                    self.store.put_record("tv_devices", device, {**record["payload"], "joins": state})
        except Exception:
            pass                            # kept in memory; a store problem must never break a stop

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
            # F11: the HTTP log no longer lists every heartbeat; one line per state change and per minute says where
            # the viewer is (the old lines carried neither position nor state).
            mark = value.get("_progress_log")
            if mark is None or mark[0] != value["state"] or time.time() - mark[1] >= 60:
                value["_progress_log"] = (value["state"], time.time())
                logging.info("tv_progress %s", json.dumps({"session": ident[:8], "state": value["state"],
                                                           "position": round(value["position"], 1),
                                                           "generation": value["generation"]}))
            self._note_playback_evidence(value)
            if _playback_buffering_on(self.config) or value.get("yt_stream") or value.get("live"):
                if not value.get("live"):
                    note_heartbeat(value, value["position"], value["state"], time.time())
                    self._note_standstill(value, time.time())
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
            self._queue_plex_report(value, "stopped" if value["state"] == "ended" else value["state"],
                                    scrobble=state == "ended" and value["position"] >= value["duration"] - 5)
        self.wake.set()
        return self.status(ident, device)

    def _queue_plex_report(self, value, state, scrobble=False):
        """Hand one Plex progress report (timeline; scrobble: also 'watched') to the reporter thread. The TV's request
        never waits for Plex. Only the newest report of a session is kept while Plex is slow."""
        report = {"value": value, "scrobble": bool(scrobble),
                  "params": {"ratingKey": value["plex_key"], "key": "/library/metadata/" + value["plex_key"],
                             "state": state, "time": round(value["position"] * 1000),
                             "duration": round(value["duration"] * 1000)}}
        with self.lock:
            waiting = self._plex_reports.pop(value["id"], None)
            report["scrobble"] = report["scrobble"] or bool(waiting and waiting["scrobble"])
            self._plex_reports[value["id"]] = report
            if self._plex_report_thread is None or not self._plex_report_thread.is_alive():
                self._plex_report_thread = threading.Thread(target=self._plex_report_loop, name="tv-plex-progress",
                                                            daemon=True)
                self._plex_report_thread.start()
        self._plex_report_wake.set()

    def _plex_report_loop(self):
        while True:
            with self.lock:
                ident = next(iter(self._plex_reports), None)
                report = self._plex_reports.pop(ident) if ident is not None else None
            if report is None:
                if self.stop_event.is_set():
                    return                           # nothing left to send at shutdown
                self._plex_report_wake.wait(5)
                self._plex_report_wake.clear()
                continue
            value = report["value"]
            try:
                self.plex.call("/:/timeline", report["params"], timeout=PLEX_REPORT_TIMEOUT)
                if report["scrobble"] and not value.get("scrobbled"):
                    self.catalog.watched(value["plex_key"], True)
                    value["scrobbled"] = True
                value["progress_warning"] = None
            except Exception as error:
                if value.get("progress_warning") is None:
                    logging.info("tv plex_report_failed %s", json.dumps({"session": str(value["id"])[:8],
                                                                         "error": type(error).__name__}))
                value["progress_warning"] = "Plex progress could not be saved"

    def stop(self, ident):
        with self.lock:
            value = self.sessions.pop(ident, None)
        timing = value.get("_status_timing") if value else None
        if timing and timing[0]:
            # F09: what status() cost during this session (it runs on every heartbeat, under the session lock).
            logging.info("tv_status_timing %s", json.dumps({"session": ident[:8], "calls": timing[0],
                                                            "mean_ms": round(timing[1] / timing[0], 1),
                                                            "max_ms": round(timing[2], 1)}))
        if value and value["sync_progress"] and value["plex_key"]:
            self._queue_plex_report(value, "stopped")       # F24: Back never waits for Plex
        self.store.expire_record("sessions", "tv-" + ident)
        if value:
            self._settle_joins(value)
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

    def _linger_seconds(self):
        """tv.dub_linger_seconds (0..1800): how long an on-demand dub job outlives its last session. Absent or 0: the
        job is cancelled at once, exactly as before."""
        value = self._tv_cfg().get("dub_linger_seconds", 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return 0.0
        return min(1800.0, max(0.0, float(value)))

    def _cancel_if_unshared(self, job_id):
        with self.lock:
            shared = any(s["job_id"] == job_id for s in self.sessions.values())
        job = self.store.job(job_id)
        if shared or not job or job["payload"].get("mode") != "ondemand":
            return False
        if job["state"] == "complete":
            # F24: a finished dub has nothing left to cancel; cancelling it only made the next open re-hash every
            # window and bump every claim token (Coordinator._reopen_cancelled).
            logging.info("tv dub_kept_complete %s", json.dumps({"job_id": job_id}))
            return False
        self.jobs.cancel(job_id)
        return True

    def _release_job(self, job_id):
        """The linger of the film just watched is over (or there is none). tv.background_finish on: hand the job to
        Coordinator.background (it is finished at the lowest priority while nobody watches or uses the PC); otherwise,
        or when it cannot continue, cancel it as before."""
        if self._tv_cfg().get("background_finish") is True and callable(getattr(self.jobs, "background", None)):
            with self.lock:
                shared = any(s["job_id"] == job_id for s in self.sessions.values())
            if shared:
                return False
            if self.jobs.background(job_id):
                logging.info("tv dub_background %s", json.dumps({"job_id": job_id}))
                return True
        return self._cancel_if_unshared(job_id)

    def cancel_unshared_job(self, job_id, *, linger=True):
        """The last session of an on-demand dub job ended (Back, stop, idle, revoke, switch to another video, dubbing
        switched off). linger True: the job keeps running for tv.dub_linger_seconds, so its windows in flight are kept,
        the models stay loaded (the pipeline unloads them only when no job is unfinished) and a reopen of the same film
        attaches to the live job; the scheduler loop cancels it afterwards (_expire_linger). linger False, or the
        setting at 0: cancelled now. The GPU lease is not involved: the other app still preempts at once."""
        seconds = self._linger_seconds() if linger else 0.0
        if seconds <= 0:
            with self.lock:
                self._linger.pop(job_id, None)
            if linger:
                self._release_job(job_id)           # no linger configured: background hand-over or cancel
            else:
                self._cancel_if_unshared(job_id)    # replaced by another job / server close: always cancel
            return
        with self.lock:
            if any(s["job_id"] == job_id for s in self.sessions.values()):
                return
            self._linger[job_id] = time.time() + seconds
        logging.info("tv dub_linger %s", json.dumps({"job_id": job_id, "seconds": seconds}))
        self.wake.set()

    def _job_attached(self, job_id):
        """A session is attaching dub job `job_id`: it no longer lingers, and every OTHER lingering job is cancelled now
        (the film being watched never shares the GPU with one nobody watches)."""
        with self.lock:
            self._linger.pop(job_id, None)
            others, self._linger = list(self._linger), {}
        for other in others:
            if self._cancel_if_unshared(other):
                logging.info("tv dub_linger_ended %s", json.dumps({"job_id": other, "reason": "another_dub_started"}))

    def _expire_linger(self, now=None):
        """Scheduler loop, once per pass: cancel the jobs whose linger is over and that no session picked up again."""
        now = time.time() if now is None else now
        with self.lock:
            if self._enqueueing:
                return
            due = [job_id for job_id, at in self._linger.items() if now >= at]
            for job_id in due:
                del self._linger[job_id]
        for job_id in due:
            if self._release_job(job_id):
                logging.info("tv dub_linger_ended %s", json.dumps({"job_id": job_id, "reason": "expired"}))

    def _flush_linger(self):
        with self.lock:
            jobs, self._linger = list(self._linger), {}
        for job_id in jobs:
            self._cancel_if_unshared(job_id)

    def status(self, ident, device=None, evidence=False):
        """evidence=True (GET /tv-api/session?id=..&evidence=1) or tv.heartbeat_evidence adds the full per-segment
        playback_evidence; every other reply carries its small summary (the full record is about 1.9 KB per muxed
        segment)."""
        began = time.perf_counter()
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
                "url": self.media_url(value, "original") if direct else self.media_url(
                    value, "master.m3u8" if self._container(value) == "fmp4" else "index.m3u8") if ready else None,
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
                result["youtube"] = dict(value["youtube"], channel=value.get("yt_channel") or value["youtube"].get("uploader", ""))
                result["subtitle_tracks"] = result["subtitle_tracks"] + [{"index": "yt:" + item["code"], "language": item["code"], "title": item["name"]}
                                                                          for item in value.get("yt_captions") or []]      # YouTube's own captions
            if value.get("plex"):
                result["plex"] = dict(value["plex"])             # the Plex-style controls: show / season / episode / previous / next
            youtube_fast = bool(value.get("yt_stream") or value.get("live"))
            if youtube_fast:
                result.update(self._youtube_fields(value, result))
            if _loudness_on(self.config):
                result["loudness"] = self._loudness_status(value, job_payload)
            full_evidence = (bool(evidence) or self._tv_cfg().get("heartbeat_evidence") is True
                             or value.get("speech_engine") in ("generic", "xtranslator"))
            if full_evidence:
                # A detached snapshot: later muxes/heartbeats must not mutate a status response.
                evidence = json.loads(json.dumps(self._playback_evidence(value)))
                for prepared in evidence["prepared_segments"]:
                    ranges = [[max(prepared["start"], a), min(prepared["end"], b)]
                              for a, b in evidence["played_ranges"]
                              if min(prepared["end"], b) > max(prepared["start"], a)]
                    prepared["heartbeat_played_ranges"] = ranges
                    prepared["heartbeat_played_seconds"] = round(sum(b - a for a, b in ranges), 6)
            else:
                live = self._playback_evidence(value)
                evidence = {"version": live["version"], "generation": live["generation"], "summary": True,
                            "prepared_segment_count": len(live["prepared_segments"]),
                            "played_seconds": live["played_seconds"],
                            "played_range_count": len(live["played_ranges"]),
                            "human_listening_acceptance": False,
                            "detail": "GET /tv-api/session?id=<session>&evidence=1 returns the full record"}
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
                # Original-audio windows of this generation and whether the dub is held (additive, for the TV's notice).
                ahead = value.get("_mux_ahead")
                played_original = sorted(ahead["original"]) if ahead and ahead["generation"] == value["generation"] else []
                blocked = value.get("dub_blocked") if value["job_id"] else None
                result.update(original_fallback=_producer._flag(self._tv_cfg(), "original_when_dub_late", True),
                              original_fallback_segments=played_original,
                              original_audio_now=current in played_original,
                              dub_blocked={"code": blocked["code"], "since": blocked["since"]} if blocked else None,
                              standstills=int(value.get("standstills", 0)),
                              original_lead_bonus=float(value.get("original_lead_bonus") or 0.0))
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
            elapsed = (time.perf_counter() - began) * 1000
            timing = value.setdefault("_status_timing", [0, 0.0, 0.0])      # calls, total ms, slowest ms
            timing[0], timing[1], timing[2] = timing[0] + 1, timing[1] + elapsed, max(timing[2], elapsed)
            if elapsed >= STATUS_SLOW_MS:
                logging.warning("tv_status_slow %s", json.dumps({"session": ident[:8], "ms": round(elapsed)}))
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
        states, _ = _producer.chunk_states(self.store, value["job_id"])
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
            fmp4 = self._container(value) == "fmp4"
            lines = ["#EXTM3U", "#EXT-X-VERSION:" + ("7" if fmp4 else "6"), "#EXT-X-TARGETDURATION:" + str(value["segment_seconds"]),
                     "#EXT-X-MEDIA-SEQUENCE:" + str(value["start_index"]), "#EXT-X-PLAYLIST-TYPE:EVENT",
                     "#EXT-X-INDEPENDENT-SEGMENTS", "#EXT-X-START:TIME-OFFSET=0,PRECISE=YES"]
            if fmp4:
                lines.append('#EXT-X-MAP:URI="%s"' % self.media_url(value, "init.mp4"))
            for number in indexes:
                seconds = min(value["segment_seconds"], value["duration"] - number * value["segment_seconds"])
                lines += ["#EXTINF:%.6f," % seconds, self.media_url(value, str(number) + (".m4s" if fmp4 else ".ts"))]
            if index * value["segment_seconds"] >= value["duration"]:
                lines.append("#EXT-X-ENDLIST")
            return ("\n".join(lines) + "\n").encode()

    def segment(self, value, index, suffix="ts"):
        with self.lock:
            path = value["segments"].get(index)
        if path is None or Path(path).suffix != "." + suffix:
            raise ValueError("That playback segment is not ready")
        return Path(path)

    def _container(self, value):
        """'fmp4' when this playback generation is muxed as HEVC in fragmented MP4, else 'ts'."""
        choice = self._encoder_choice.get(value["id"])
        return choice.get("container", "ts") if choice and choice.get("generation") == value["generation"] else "ts"

    def init_segment(self, value):
        """The init segment (EXT-X-MAP) every fMP4 segment of this playback generation shares."""
        choice = self._encoder_choice.get(value["id"])
        if not choice or choice.get("generation") != value["generation"] or not choice.get("init"):
            raise ValueError("That playback segment is not ready")
        return self.config.cache / "tv-media" / choice["init"]

    def master_playlist(self, value):
        """One-variant master playlist of an HEVC playback: codec, size and dynamic range before any segment is loaded."""
        with self.lock:
            if self._container(value) != "fmp4":
                raise ValueError("This playback has no HEVC stream")
            video = next((s for s in value["streams"] if s.get("codec_type") == "video"
                          and not s.get("disposition", {}).get("attached_pic")), {})
            transfer = video.get("color_transfer")
            preserve = transfer in ("smpte2084", "arib-std-b67") and value["hdr_mode"] == "auto"
            height = int(value["quality"])
            box_w, box_h = height * 16 // 9, height
            source_w, source_h = int(video.get("width") or box_w), int(video.get("height") or box_h)
            if not value["upscale"]:
                box_w, box_h = min(box_w, source_w), min(box_h, source_h)
            factor = min(box_w / max(1, source_w), box_h / max(1, source_h))
            out_w, out_h = max(2, int(source_w * factor) // 2 * 2), max(2, int(source_h * factor) // 2 * 2)
            bandwidth = (20000000 if height == 2160 else 6000000) + (384000 if self._surround_stream(value) is not None else 192000)
            lines = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-INDEPENDENT-SEGMENTS",
                     '#EXT-X-STREAM-INF:BANDWIDTH=%d,CODECS="%s,mp4a.40.2",RESOLUTION=%dx%d,VIDEO-RANGE=%s' % (
                         bandwidth, hevc_codec_string(out_h), out_w, out_h, video_range(transfer) if preserve else "SDR"),
                     self.media_url(value, "index.m3u8")]
            return ("\n".join(lines) + "\n").encode()

    # --- run-ahead mux (config tv.mux_ahead_unlimited, default ON for dubbed jobs) ---------------------------------
    # The scheduler tick (loop thread) never encodes: it calls producer_policy.produce() (dubbed audio toward EOF, only
    # `producer_queue` chunks in flight), asks the store for the playback chunk first, then starts up to
    # tv.mux_ahead_workers concurrent encodes (plus one reserved slot for the playback segment) of every segment whose
    # audio is ready, current position first then ascending, with no distance cap and regardless of pause. Each encode
    # is a ProcessOwner registered under the session; a seek/stop/revision change cancels them all, and a finished
    # encode is published only while its session, generation and job are still current (the existing epoch rules).
    # Switching the key off restores the legacy buffer_segments/MAX_EXTRA_MUX_SEGMENTS serial loop untouched.
    SNAPSHOT_TTL = 1.0
    # Original audio for a window the dub cannot deliver in time (tv.original_when_dub_late, default ON): when playback
    # is this close to a window whose dub is not ready, that ONE window is muxed with the film's own audio (levelled
    # with the job's gain) instead of letting the picture freeze. The TV re-reads an unchanged playlist every 15 s and
    # fetches a segment 29 s before it is needed (2026-10-25 log), so the window must be published about 23 s ahead:
    # 15 s playlist + 4 s mux + 2 s heartbeat age + 2 s fetch.
    ORIGINAL_BLOCKED_LEAD = 25.0     # default of tv.original_blocked_lead_seconds: the dub job is held (no graphics card)
    ORIGINAL_LATE_LEAD = 20.0        # default of tv.original_late_lead_seconds: the dub is merely late (0 = never)
    DUB_BLOCKED_GRACE = 45.0         # status 'dub_blocked' outlives a retry round of the held job by this long
    # Self-check: a standstill the heartbeats show at an original-audio window means that window reached the TV too
    # late; this playback's two leads are then raised by ORIGINAL_LEAD_STEP, at most ORIGINAL_LEAD_BONUS_MAX in all.
    STANDSTILL_MIN_SECONDS = 3.0     # shorter ones are not logged (heartbeats are 2 s apart)
    ORIGINAL_LEAD_STEP = 10.0
    ORIGINAL_LEAD_BONUS_MAX = 30.0

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
                     "disk": "ok", "launched": 0, "completed": 0, "snapshot": None, "request": None,
                     "original": set()}     # segments of this generation muxed with the film's own audio
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
        self._note_dub_blocked(value, job, state["snapshot"][1], now)
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
        reason = self._original_due(value, cfg, job, first, ready, position)
        if (reason and first not in inflight and first not in parked
                and self._encode_total() < _producer.mux_workers(cfg) + 1):
            # The dub of the window playback needs next cannot reach the TV in time: the film's own audio for that one
            # window. The store chunk is not touched, so the window is still dubbed and used the next time.
            if not self._start_encode(ident, value, first, generation, state, original=True):
                return
            inflight.add(first)
            logging.info("tv original audio for segment %d (%s, %.0f s before it is needed, lead bonus %.0f s)", first,
                         reason, max(0.0, first * seg - position), float(value.get("original_lead_bonus") or 0.0))
        dubbing = gap is not None           # the job still has windows to dub from the playhead to the end
        picks = _producer.plan_mux(current=current, total=total, muxed=muxed, inflight=inflight,
                                   ready=ready - parked, workers=_producer.mux_workers(cfg, dubbing=dubbing),
                                   active=self._encode_total(), ahead_allowed=gate == "ok")
        low_priority = dubbing and _producer._flag(cfg, "mux_ahead_low_priority", True)
        if first not in ready and not inflight:
            self._ensure_premeasure(ident, value)       # idle until the first chunk exists: measure the job gain
        for index in picks:
            # nobody waits for a segment URGENT_SPAN or more ahead yet: its ffmpeg yields the CPU to the dub (F25)
            if not self._start_encode(ident, value, index, generation, state,
                                      low_priority=low_priority and index - current >= _producer.URGENT_SPAN):
                return

    def _start_encode(self, ident, value, index, generation, state, original=False, low_priority=False):
        """Start one mux on its own thread, registered under the session so a seek or stop cancels it. False (nothing
        started) when the session or its generation changed meanwhile. original: see render(). low_priority: the
        encode's ffmpeg runs below normal priority (F25; never for the segment playback needs, nor an original window)."""
        owner = ProcessOwner()
        if low_priority:
            owner.priority = "below_normal"
        with self.lock:
            if value["generation"] != generation or self.sessions.get(ident) is not value:
                return False
            state["inflight"][index] = owner
            self._encodes.setdefault(ident, set()).add(owner)
            state["launched"] += 1
        threading.Thread(target=self._ahead_encode, args=(ident, value, index, generation, owner, state, original),
                         name="tv-mux-%d" % index, daemon=True).start()
        return True

    def _note_dub_blocked(self, value, job, states, now):
        """value['dub_blocked'] = {'code', 'since', 'seen'} while the dub job is held for a resource (job state
        waiting_resources: none of its windows is claimed until the graphics card is free), else None. The code comes
        from the receipt of the first held chunk ('resource_wait' until one is read: the chunk snapshot can be a second
        older than the job row). Kept DUB_BLOCKED_GRACE seconds after the hold was last seen, so a retry round of the
        job does not make the TV's notice blink."""
        current = value.get("dub_blocked")
        if job["state"] != "waiting_resources":
            if current is not None and now - current["seen"] > self.DUB_BLOCKED_GRACE:
                value["dub_blocked"] = None
            return
        if current is None:
            current = value["dub_blocked"] = {"code": "resource_wait", "since": now, "seen": now}
        current["seen"] = now
        if current["code"] != "resource_wait":
            return
        held = min((i for i, s in states.items() if s == "waiting_resources"), default=None)
        if held is None:
            return
        chunk = self.store.chunk(value["job_id"], held) or {}
        found = _RESOURCE_CODE.search(str((chunk.get("payload") or {}).get("resource_wait") or ""))
        if found:
            current["code"] = found.group(1)

    def _original_due(self, value, cfg, job, first, ready, position):
        """'start' | 'blocked' | 'late' when segment `first` (the first one from the playhead that is not muxed) has to
        be muxed with the film's own audio NOW, else None. Never while the viewer paused, never for an errored /
        needs-review job (its own boundary rule, P1-1, stays).
        'blocked': playback runs and the dub job is held for a resource (no graphics card).
        'late':    playback runs and the dub is only slow.
        'start':   playback has not started in this seek epoch and the dub job is held: nothing can be dubbed until
                   the card is free, so the lead the start gate asks for is filled with the film's own audio and the
                   film starts now. With the card available the start gate waits for the dub: the film opens dubbed."""
        if first is None or first in ready or not _producer._flag(cfg, "original_when_dub_late", True):
            return None
        if job["state"] in KEEP_READY_STATES:
            return None
        held = job["state"] == "waiting_resources"
        started = bool(value.get("beta_started")) and value.get("beta_epoch") == int(value.get("epoch", 0))
        if not started and not held:
            return None
        if value.get("paused_intent", value.get("state") == "paused"):
            return None
        lead = max(0.0, first * value["segment_seconds"] - position)
        bonus = float(value.get("original_lead_bonus") or 0.0)          # raised by _note_standstill, per playback
        if held and not started:
            # exactly the lead the start gate asks for (status 'startable'), never more: the gate then lets the film start
            required = max(_producer.required_seconds(value, cfg)[0], float(value.get("_gate_required_lead") or 0.0))
            return "start" if lead < min(600.0, required) - 1e-6 else None
        if held:
            limit = min(120.0, bonus + _producer._number(
                _producer._tv(cfg, "original_blocked_lead_seconds", self.ORIGINAL_BLOCKED_LEAD),
                self.ORIGINAL_BLOCKED_LEAD, 5.0, 120.0))
            return "blocked" if lead <= limit else None
        limit = _producer._number(_producer._tv(cfg, "original_late_lead_seconds", self.ORIGINAL_LATE_LEAD),
                                  self.ORIGINAL_LATE_LEAD, 0.0, 60.0)
        return "late" if limit > 0 and lead <= min(60.0, limit + bonus) else None

    def _note_standstill(self, value, now):
        """Standstill meter, fed by every heartbeat (the TV reports every 2 s): playback has started in this seek
        epoch, two heartbeats in a row say 'playing' or 'buffering', and the position did not move between them. When
        it moves again, a standstill of at least STANDSTILL_MIN_SECONDS (first to last heartbeat at that position) is
        counted in the session and written to the log ('tv standstill {...}').
        If the picture stood inside a window that was muxed with the film's own audio, or within 1.5 s of the start
        of one, that window reached the TV too late: this playback's leads for original-audio windows are raised by
        ORIGINAL_LEAD_STEP (ORIGINAL_LEAD_BONUS_MAX in all), so the next one is published earlier."""
        epoch = int(value.get("epoch", 0))
        position = float(value["position"])
        previous = value.get("_standstill")
        started = bool(value.get("beta_started")) and value.get("beta_epoch") == epoch
        sample = {"position": position, "at": now, "since": None, "epoch": epoch,
                  "moving": started and value["state"] in ("playing", "buffering")}
        value["_standstill"] = sample
        if previous is None or previous["epoch"] != epoch:
            return
        if (sample["moving"] and previous["moving"] and abs(position - previous["position"]) < 0.05
                and value["duration"] - position > 1.0):
            sample["since"] = previous["since"] if previous["since"] is not None else previous["at"]
            return
        if previous["since"] is None:
            return
        seconds = previous["at"] - previous["since"]        # first to last heartbeat at the same position: a lower bound
        if seconds < self.STANDSTILL_MIN_SECONDS:
            return
        seg = value["segment_seconds"]
        index = int(previous["position"] // seg)
        ahead = value.get("_mux_ahead")
        original = set(ahead["original"]) if ahead and ahead["generation"] == value["generation"] else set()
        at_original = index in original or (index + 1 in original and (index + 1) * seg - previous["position"] <= 1.5)
        value["standstills"] = int(value.get("standstills", 0)) + 1
        bonus = float(value.get("original_lead_bonus") or 0.0)
        if at_original and bonus < self.ORIGINAL_LEAD_BONUS_MAX:
            bonus = value["original_lead_bonus"] = min(self.ORIGINAL_LEAD_BONUS_MAX, bonus + self.ORIGINAL_LEAD_STEP)
        logging.info("tv standstill %s", json.dumps({
            "session": value["id"], "position": round(previous["position"], 1), "seconds_at_least": round(seconds, 1),
            "segment": index, "original_window": at_original, "original_lead_bonus": bonus,
            "standstills": value["standstills"]}))

    def _ahead_encode(self, ident, value, index, generation, owner, state, original=False):
        try:
            render_value = dict(value)
            started = time.time()
            # The extra argument is passed only for an original-audio window, so a render() replaced by a
            # three-argument callable keeps working.
            path = (self.render(render_value, index, owner, original=True) if original
                    else self.render(render_value, index, owner))
            finished = time.time()
            # One line per segment: the only record of what the encodes cost (low_priority/mux_workers are judged from it).
            logging.info("tv_mux %s", json.dumps({
                "session": ident[:8], "index": index, "seconds": round(finished - started, 2),
                "cached": bool(render_value.get("_mux_cached")), "below_normal": getattr(owner, "priority", None) == "below_normal",
                "running": self._encode_total()}))
            current_job = self._mux_job_current(value)
            if self._publish_segment(ident, value, generation, index, path, render_value, current_job, started, finished):
                state["completed"] += 1
                state["failed"].pop(index, None)
                if original:
                    with self.lock:
                        state["original"].add(index)
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
            for path in (self.config.cache / "tv-media").glob("*.partial*"):
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
                "workers_while_dubbing": _producer.mux_workers(cfg, dubbing=True),
                "muxed_segments": len(value["segments"]),
                "inflight_segments": sorted(state["inflight"]) if state else [],
                "encodes_running": self._encode_total(),
                "disk": state["disk"] if state else "unknown",
                "disk_floor_gib": floor,
                "disk_floor_behaviour": "ahead-only stop at floor + mux_ahead_disk_margin_gib; below the floor nothing "
                                        "renders (resource_wait is reported) and muxing resumes by itself when space is freed",
                "failed_segments": {str(i): {"attempts": r["count"], "message": r["message"]}
                                    for i, r in (state["failed"].items() if state else ())},
                "encoder": ({k: choice.get(k) for k in ("requested", "used", "reason", "fallback", "free_vram_mib",
                                                         "codec", "container", "codec_reason", "filters", "tonemap")}
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

    def _ffmpeg_has_hevc_nvenc(self):
        path = str(self.config["ffmpeg"])
        if path not in self._hevc_listing:
            try:
                self._hevc_listing[path] = b"hevc_nvenc" in ProcessOwner().run([path, "-hide_banner", "-encoders"], timeout=30)
            except Exception:
                self._hevc_listing[path] = False
        return self._hevc_listing[path]

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
            cfg, pins = self._tv_cfg(), self._picture_pins
            setting = "h264" if pins.get("codec") == "h264" else str(_producer._tv(cfg, "video_codec", "h264"))
            hevc_setting = setting in ("auto", "hevc")
            minimum = int(_producer._tv(cfg, "nvenc_min_free_mib", 2048))
            wanted = requested in ("auto", "nvenc") and (hevc_setting or not hdr_preserve)
            listed = self._ffmpeg_has_nvenc() if wanted else False
            free = self._free_vram_mib() if listed else None
            nvenc_ok = bool(listed and free is not None and free >= minimum)
            codec, codec_reason = "h264", "tv.video_codec is h264"
            if hevc_setting:
                device = self._device_video(value.get("device_id")) if setting == "auto" else {}
                codec, codec_reason = _producer.choose_codec(
                    setting, encoder="nvenc" if nvenc_ok else "software",
                    hevc_nvenc_listed=self._ffmpeg_has_hevc_nvenc() if nvenc_ok else False,
                    device_ok=_producer.probe_passes_hevc(device.get("probe")),
                    device_failed=device.get("hevc_failed_picture") == PICTURE_VERSION,
                    pinned=value.get("codec_pinned"), short_segments=bool(value.get("yt_stream") or value.get("live")))
            if codec == "hevc":
                used = "nvenc" if nvenc_ok and self._ffmpeg_has_hevc_nvenc() else "software"
                reason = codec_reason
            else:
                used, reason = _producer.choose_encoder(
                    requested, nvenc_listed=listed, free_vram_mib=free, hdr_preserve=hdr_preserve, min_free_mib=minimum)
            gpu = used == "nvenc" and str(_producer._tv(cfg, "gpu_filters", "off")) == "auto"
            filters = "cuda" if gpu and pins.get("filters") != "cpu" else "cpu"
            tonemap = (pins.get("tonemap") or ("placebo" if filters == "cuda" else "opencl")) if gpu else "cpu"
            choice = {"generation": generation, "requested": requested, "used": used, "reason": reason,
                      "free_vram_mib": free, "checked_at": time.time(), "fallback": None,
                      "codec": codec, "container": "fmp4" if codec == "hevc" else "ts", "codec_reason": codec_reason,
                      "filters": filters, "tonemap": tonemap, "init": None, "player_failures": 0}
            self._encoder_choice[ident] = choice
            logging.info("tv_encoder %s", json.dumps({"session": str(ident)[:8], "generation": generation, "requested": requested,
                                                      "used": used, "reason": reason, "free_vram_mib": free,
                                                      "codec": codec, "filters": filters}))
            return used, choice

    def _pin_software(self, value, reason):
        with self._encoder_lock:
            choice = self._encoder_choice.get(value["id"])
            if choice and choice["generation"] == value["generation"]:
                choice.update(used="software", fallback=reason, reason="software after NVENC failure")
                logging.warning("tv_encoder_fallback %s", json.dumps({"session": str(value["id"])[:8],
                                                                      "generation": value["generation"], "reason": reason[-300:]}))

    def _degrade_picture(self, value, choice, encoder, detail):
        """An ffmpeg run of this plan failed. Take the next step down: True = render this segment again with the changed
        plan; False = nothing left to change (the error is real). A step that changes the container or the tone-map look
        restarts the playback generation and raises Cancelled for this encode."""
        tone_step = None
        with self._encoder_lock:
            if self._encoder_choice.get(value["id"]) is not choice or choice["generation"] != value["generation"]:
                raise Cancelled("picture plan changed")
            if value.get("_tone_mapped") and choice.get("tonemap") in ("placebo", "opencl"):
                tone_step = "opencl" if choice["tonemap"] == "placebo" else "cpu"
                self._picture_pins["tonemap"] = tone_step
            elif choice.get("filters") == "cuda":
                self._picture_pins["filters"] = "cpu"
                choice.update(filters="cpu", fallback="GPU filters failed: " + detail)
                return True
            elif choice.get("codec") == "hevc":
                self._picture_pins["codec"] = "h264"
        if tone_step:
            self._restart_picture(value["id"], value["generation"], "GPU tone mapping failed, next is %s: %s" % (tone_step, detail))
            raise Cancelled("picture plan changed")
        if choice.get("codec") == "hevc":
            self._restart_picture(value["id"], value["generation"], "HEVC encode failed: " + detail, codec_pin="encode-failed")
            raise Cancelled("picture plan changed")
        if encoder != "nvenc":
            return False
        self._pin_software(value, "h264_nvenc encode failed: " + detail)
        return True

    def _video_filters(self, value, choice, hdr, preserve_hdr, encoder, transfer):
        """(ffmpeg options placed before -ss/-i, the video filter chain, the -pix_fmt arguments) of this generation's plan.
        With the defaults (H.264, gpu_filters off) the chain and pixel format are exactly the v4 ones."""
        height, width = value["quality"], value["quality"] * 16 // 9
        hevc, tone = choice.get("codec") == "hevc", hdr and not preserve_hdr
        size = ("w=%s:h=%s" % (width, height) if value["upscale"] else "w='min(%s,iw)':h='min(%s,ih)'" % (width, height))
        if choice.get("filters") == "cuda" and (not tone or choice.get("tonemap") == "placebo"):
            fit = size + ":force_original_aspect_ratio=decrease:force_divisible_by=2:reset_sar=1"
            if tone:
                return (["-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk", "-hwaccel", "cuda",
                         "-hwaccel_output_format", "cuda"],
                        "hwupload=derive_device=vulkan,libplacebo=" + fit + ":upscaler=ewa_lanczossharp:tonemapping=bt.2390"
                        ":gamut_mode=perceptual:colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv:format=nv12,"
                        "hwupload=derive_device=cuda", [])
            return (["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
                    "scale_cuda=" + fit + ":interp_algo=lanczos:format=" + ("p010le" if hevc and preserve_hdr else "nv12"), [])
        chain = "scale=" + size + ":force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,setsar=1"
        options = []
        if tone and choice.get("tonemap") == "opencl":
            options = ["-init_hw_device", "opencl=ocl", "-filter_hw_device", "ocl"]
            chain = ("format=p010le,hwupload,tonemap_opencl=tonemap=hable:desat=0:t=bt709:m=bt709:p=bt709:r=tv:format=nv12,"
                     "hwdownload,format=nv12," + chain)
        elif tone:
            chain = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv," + chain
        if hevc and preserve_hdr:
            chain += ",setparams=color_primaries=bt2020:color_trc=%s:colorspace=bt2020nc" % transfer
        if hevc and encoder == "nvenc":
            pixel = "p010le" if preserve_hdr else "yuv420p"          # 8-bit input is encoded as Main10 by -highbitdepth 1
        elif hevc:
            pixel = "yuv420p10le"                                     # libx265 takes its Main10 profile from the pixel format
        else:
            pixel = "yuv420p10le" if preserve_hdr else "yuv420p"
        return options, chain, ["-pix_fmt", pixel]

    @staticmethod
    def _container_args(container, start_ts, index, seamless=True):
        """Muxer arguments. fMP4: true timestamps in every fragment (frag_discont + avoid_negative_ts disabled), no edit
        list, a fragment at every forced key frame, fragment numbers that rise across segments. MPEG-TS: F07's tail
        (restart of the continuity counters, start_ts = start + its 10 s base); seamless False (tv.seamless_joins off):
        today's tail, byte for byte."""
        if container == "fmp4":
            return ["-output_ts_offset", str(start_ts), "-avoid_negative_ts", "disabled", "-movflags", FMP4_FLAGS,
                    "-video_track_timescale", "90000", "-fragment_index", str(index * 64 + 1), "-f", "mp4"]
        if seamless:
            return ["-output_ts_offset", "%.6f" % float(start_ts), "-muxdelay", "0", "-muxpreload", "0",
                    "-mpegts_flags", "+resend_headers+initial_discontinuity", "-f", "mpegts"]
        return ["-output_ts_offset", str(start_ts), "-muxdelay", "0", "-muxpreload", "0", "-mpegts_flags", "+resend_headers",
                "-f", "mpegts"]

    def _install_fmp4(self, temporary, output, directory):
        """Split ffmpeg's file into the shared init segment (<sha256>.init.mp4, written once) and this segment's media
        file at `output`; returns the init file name."""
        media = temporary.with_name(temporary.name + ".m4s")
        try:
            init = split_fmp4(temporary, media)
            name = hashlib.sha256(init).hexdigest() + ".init.mp4"
            target = directory / name
            if not target.exists():
                partial = directory / ("%s.%s.partial" % (name, uuid.uuid4().hex[:8]))
                partial.write_bytes(init)
                os.replace(partial, target)
            os.replace(media, output)
            return name
        finally:
            media.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)

    def _picture_selfcheck(self, value, choice, encoder, owner, start, output, init_name, hdr, preserve_hdr, transfer):
        """Once per server run and picture plan that is not the v4 default (HEVC, GPU scaling, GPU tone mapping): decode
        the first frame of the segment just made and compare it with the same source frame through the v4 CPU chain
        (both 320x180 gray, PSNR). None when the plan is fine or was already found fine, else the problem text. The
        numbers go to the server log as one 'tv picture self-check' line."""
        tone = bool(hdr and not preserve_hdr)
        plan = (choice.get("codec"), encoder, choice.get("filters"), choice.get("tonemap") if tone else None, bool(preserve_hdr))
        with self._encoder_lock:
            known = self._picture_checked.get(plan)
        if known is not None:
            return None if known["ok"] else "the picture self-check of this plan failed earlier"
        width, height = SELFCHECK_SIZE
        small = "scale=%d:%d:flags=bilinear,format=gray" % (width, height)
        _, reference_chain, _ = self._video_filters(value, {"codec": "h264", "filters": "cpu", "tonemap": "cpu"}, hdr,
                                                    preserve_hdr, "software", transfer)
        made = "concat:%s|%s" % (output.parent / init_name, output) if init_name else str(output)
        head = [self.config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin"]
        tail = ["-an", "-sn", "-frames:v", "1", "-f", "rawvideo", "-"]
        psnr, problem = -1.0, None
        try:
            reference = owner.run(head + ["-ss", str(start), "-i", value["source"], "-vf", reference_chain + "," + small] + tail,
                                  timeout=120) or b""
            produced = owner.run(head + ["-i", made, "-vf", small] + tail, timeout=120) or b""
            psnr = gray_psnr(reference[:width * height], produced[:width * height])
        except Cancelled:
            raise
        except Exception as error:
            problem = "%s: %s" % (type(error).__name__, str(error)[-160:])
        limit = SELFCHECK_TONE_PSNR_DB if tone and choice.get("tonemap") != "cpu" else SELFCHECK_PSNR_DB
        result = {"ok": problem is None and psnr >= limit, "psnr_db": round(psnr, 1), "limit_db": limit}
        with self._encoder_lock:
            self._picture_checked[plan] = result
        logging.info("tv picture self-check %s", json.dumps({"codec": plan[0], "encoder": plan[1], "filters": plan[2],
                                                            "tonemap": plan[3], "hdr": plan[4], "error": problem, **result}))
        if result["ok"]:
            return None
        return "picture self-check failed: " + (problem or "PSNR %.1f dB, limit %.1f dB" % (psnr, limit))

    def _adopt_init(self, value, choice, name, fresh):
        """Every fMP4 segment of one playback generation must share one init segment. True when `name` is that one (or
        becomes it). A cached segment with another init is simply encoded again (False); a fresh encode that disagrees
        sends the playback back to H.264."""
        with self._encoder_lock:
            if choice.get("init") in (None, name):
                choice["init"] = name
                return True
        if not fresh:
            return False
        self._restart_picture(value["id"], value["generation"], "the HEVC init segment changed between segments",
                              codec_pin="init-mismatch")
        raise Cancelled("picture plan changed")

    @staticmethod
    def _video_codec(encoder, preserve_hdr, threads, codec="h264"):
        """(codec, options). Software options are exactly the v4 ones; NVENC is a quality-equivalent constant-quality VBR.
        codec 'hevc' (F19): HEVC Main10 without B-frames (their reorder delay would shift an independently made fMP4
        segment), hvc1 sample entry."""
        if codec == "hevc":
            if encoder == "nvenc":
                return "hevc_nvenc", ["-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "24", "-b:v", "0",
                                      "-profile:v", "main10", "-highbitdepth", "1", "-spatial-aq", "1", "-bf", "0",
                                      "-forced-idr", "1", "-tag:v", "hvc1"]
            return "libx265", ["-preset", "veryfast", "-crf", "22", "-tag:v", "hvc1", "-x265-params",
                               "pools=%s:frame-threads=1:log-level=error:bframes=0:repeat-headers=0" % threads]
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
            self._expire_linger()
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

    def _seamless_on(self, value):
        """tv.seamless_joins (default True): library films get the gapless argument list of relay/segment_join.py. Only an
        explicit False restores the old list and cache key. The YouTube profile keeps the old list."""
        latched = value.get("seamless")
        if not isinstance(latched, bool):                # a session made before the latch existed (tests): the setting itself
            latched = _producer._flag(self._tv_cfg(), "seamless_joins", True)
        return latched and not value.get("yt_stream")

    def _previous_window_audio(self, value, job, index, surround):
        """(source, identity) of the audio window index - 1 has in this playback, decided the way render() decides it for
        that window now: the pre-roll of segment `index`. The movie's own audio when that window has no ready dub.
        Reads only; never requests work."""
        rear_off = str(self.config["tv"].get("surround_rear", "off")).lower() != "on"
        prefix = SURROUND_REAR_OFF if surround is not None and rear_off else ""
        movie = ({"kind": "source", "path": value["source"], "stream": value["audio_index"], "prefix": prefix}, ["source"])
        if not job:
            return movie
        try:
            chunk = self.store.chunk(job["id"], index - 1)
            if not chunk or chunk["state"] not in READY_CHUNK_STATES:
                return movie
            payload = chunk["payload"]
            if int(payload.get("revision", job["payload"].get("revision", 1))) != job["payload"].get("revision", 1):
                return movie
            path = Path(payload["audio"])
            digest = file_hash(path)
            if digest != payload["sha256"]:
                return movie
        except (OSError, KeyError, TypeError, ValueError):
            return movie
        if surround is None:
            return {"kind": "file", "path": str(path), "prefix": ""}, ["wav", digest]
        bound = self._surround_mix(path, digest)
        if bound is not None:
            return {"kind": "file", "path": str(bound), "prefix": prefix}, ["mix", file_hash(bound)]
        if not passthrough_mode(payload.get("mode")):
            return {"kind": "file", "path": str(path), "prefix": SURROUND_UPMIX}, ["upmix", digest]
        return movie

    def render(self, value, index, owner, original=False):
        """original=True (tv.original_when_dub_late): this window of a dubbed playback is muxed with the film's OWN
        audio because its dub cannot arrive in time. Nothing is asked of the dub job; the job's one loudness gain is
        applied like on every other window; in a 5.1 playback it takes the existing 'surround original' chain."""
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
        audio, audio_hash, job = None, "original", None
        if value["job_id"] and original:
            job = self.store.job(value["job_id"])       # only for the job's loudness gain below
        elif value["job_id"]:
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
        loudness = (self._playback_loudness(value, job, owner)
                    if (audio or (original and job)) and _loudness_on(self.config) else None)
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
        encoder, choice = self._resolve_encoder(value, preserve_hdr)
        codec_name, container = choice.get("codec", "h264"), choice.get("container", "ts")
        value["_tone_mapped"] = bool(hdr and not preserve_hdr)
        seamless = self._seamless_on(value)
        identity_parts = [value["fingerprint"], value["audio_index"], index, audio_hash,
                          value["quality"], value["upscale"], value["hdr_mode"],
                          encoder, value["encoder_threads"],
                          _join.MUX_VERSION["software" if encoder == "software" else "nvenc"] if seamless
                          else "tv-mux-cpu-v4" if encoder == "software" else "tv-mux-nvenc-v1"]
        gpu_tone = value["_tone_mapped"] and choice.get("tonemap") in ("placebo", "opencl")
        if codec_name == "hevc" or choice.get("filters") == "cuda" or gpu_tone:
            # F19: another codec, container, scaler or tone-map chain is another cached segment (the default path adds nothing)
            identity_parts.append([PICTURE_VERSION, codec_name, container, choice.get("filters"),
                                   choice.get("tonemap") if value["_tone_mapped"] else None])
        if youtube:
            # segment length and rate cap are part of what a cached segment is (a library segment's index implies 30 s; this one does not)
            identity_parts.append(["youtube", value["segment_seconds"], str(self._yt_opt("stream_maxrate")), str(self._yt_opt("stream_bufsize"))])
        audio_filter, gain_filter = "apad", ""
        if leveled:
            from . import loudness_job as _lj, playback_loudness as _pl
            curve = {k: v for k, v in _lj.settings(self.config["tv"]["loudness"]).items()
                     if k in ("knee_dbfs", "ceiling_dbfs")}    # defaults -3 / -1 dBFS
            identity_parts.append(_pl.mux_identity(gain_db, **curve))   # a different gain/curve is a different mux output
            gain_filter = _pl.mux_filter(gain_db, **curve)              # constant gain, stateless soft knee
            audio_filter = gain_filter + ",apad"                        # ... then pad (old argument list)
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
        plan = pre = pre_identity = None
        ts_source = seamless and _join.ts_like(value.get("container"))
        if seamless:
            plan = _join.audio_join_plan(index, math.ceil(value["duration"] / value["segment_seconds"]),
                                         value["segment_seconds"], value["duration"])
            pre, pre_identity = (None, ["none"]) if plan["first"] else self._previous_window_audio(value, job, index, surround)
            # What the bytes depend on beyond the old identity: the sample plan, the pre-roll audio, the PTS base, the early seek.
            identity_parts.append(["join", plan["pre"], plan["tail"], plan["keep"], _join.TS_BASE_SECONDS,
                                   bool(ts_source), pre_identity])
        identity = json.dumps(identity_parts)
        if value.get("speech_engine") == "xtranslator":
            identity = json.dumps(["xtranslator", identity])
        key = hashlib.sha256(identity.encode()).hexdigest()
        directory = self.config.cache / "tv-media"
        directory.mkdir(parents=True, exist_ok=True)
        fmp4 = container == "fmp4"
        output, receipt = directory / (key + (".m4s" if fmp4 else ".ts")), directory / (key + ".json")
        if output.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            if file_hash(output) == saved.get("sha256") and (not fmp4 or (
                    saved.get("init") and (directory / saved["init"]).is_file()
                    and self._adopt_init(value, choice, saved["init"], fresh=False))):
                value["_mux_cached"] = True      # a reuse is instant: not a measurement of the mux rate
                try:
                    os.utime(output, None)       # F11: "used now", for the least-recently-used eviction of tv-media
                except OSError:
                    pass
                return output
        # One file per attempt: a cancelled encode of this segment (seek back to it) may still be unwinding.
        temporary = directory / ("%s.%s.partial.%s" % (key, uuid.uuid4().hex[:8], "mp4" if fmp4 else "ts"))
        if not video:
            raise ValueError("No playable video stream")
        input_options, scale, pixel = self._video_filters(value, choice, hdr, preserve_hdr, encoder,
                                                          video.get("color_transfer"))
        height = value["quality"]
        codec, codec_options = self._video_codec(encoder, preserve_hdr, value["encoder_threads"], codec_name)
        rate, buffer_size = ("30M" if height == 2160 else "8M"), ("60M" if height == 2160 else "16M")
        if codec_name == "hevc":
            rate, buffer_size = ("20M" if height == 2160 else "6M"), ("40M" if height == 2160 else "12M")
        if youtube:
            # optional caps (youtube.stream_maxrate / stream_bufsize); unset: the same rate rule as a library film at this height
            rate, buffer_size = str(self._yt_opt("stream_maxrate") or rate), str(self._yt_opt("stream_bufsize") or buffer_size)
        color = (["-color_primaries", "bt2020", "-color_trc", video["color_transfer"],
                  "-colorspace", "bt2020nc"] if preserve_hdr else [])
        if seamless:
            prefix = audio_prefix.rstrip(",")
            main = ({"kind": "file", "path": str(audio_input), "prefix": prefix} if audio_input else
                    {"kind": "source", "path": value["source"], "stream": value["audio_index"], "prefix": prefix})
            args = _join.mux_arguments(
                ffmpeg=self.config["ffmpeg"], source=value["source"], video_stream=video["index"], start=start,
                duration=duration, scale=scale, codec=codec, codec_options=codec_options, threads=value["encoder_threads"],
                pix_fmt="yuv420p10le" if preserve_hdr else "yuv420p", maxrate=rate, bufsize=buffer_size, color=color,
                audio_rate=audio_rate, channel_args=channel_args, plan=plan, pre=pre, main=main,
                layout="5.1" if surround is not None else "stereo", gain=gain_filter, ts_source=ts_source, output=temporary,
                input_options=input_options, pixel=pixel,      # F19: the plan's decoder options and pixel format
                container_args=self._container_args(container, start + _join.TS_BASE_SECONDS, index, seamless=True))
        else:
            args = [self.config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", *input_options,
                    "-ss", str(start), "-i", value["source"]]
            if audio_input:
                args += ["-i", str(audio_input)]
            args += ["-map", "0:" + str(video["index"]), "-map", "1:a:0" if audio_input else "0:" + str(value["audio_index"]),
                     "-t", "%.6f" % duration, "-vf", scale, "-c:v", codec, *codec_options,
                     "-threads", str(value["encoder_threads"]), "-filter_threads", str(min(2, value["encoder_threads"])),
                     *pixel,
                     "-maxrate", rate, "-bufsize", buffer_size, *color,
                     "-force_key_frames", "expr:gte(t,n_forced*2)",
                     "-c:a", "aac", "-b:a", audio_rate, "-ar", "48000", *channel_args,
                     "-af", audio_prefix + audio_filter, "-sn", "-dn",
                     # fMP4 cannot hold segment 0's negative priming timestamp: +10 s base also without F07 (seamless off)
                     *self._container_args(container, start + (10 if fmp4 else 0), index, seamless=False),
                     "-y", str(temporary)]
        try:
            try:
                owner.run(args, timeout=max(180, duration * 20))
                owner.check()
                fell_back = False
            except (Cancelled, TimeoutError):
                raise
            except RuntimeError as error:
                # NVENC refused (sessions, VRAM, driver): software for the rest of this playback generation, then
                # this segment again under the software identity. One stream never mixes encoders afterwards.
                # F19: a GPU filter chain or the HEVC path that fails steps down first (_degrade_picture).
                if not self._degrade_picture(value, choice, encoder, str(error)[-240:]):
                    raise
                fell_back = True
            if fell_back:
                temporary.unlink(missing_ok=True)
                return self.render(value, index, owner, original)
            if not temporary.is_file() or temporary.stat().st_size < 188:
                raise ValueError("TV mux produced no transport stream")
            init_name = None
            if fmp4:
                init_name = self._install_fmp4(temporary, output, directory)
                self._adopt_init(value, choice, init_name, fresh=True)
            else:
                os.replace(temporary, output)
            if codec_name == "hevc" or choice.get("filters") == "cuda" or gpu_tone:
                problem = self._picture_selfcheck(value, choice, encoder, owner, start, output, init_name, hdr, preserve_hdr,
                                                  video.get("color_transfer"))
                if problem:
                    output.unlink(missing_ok=True)
                    if not self._degrade_picture(value, choice, encoder, problem):
                        raise ValueError(problem)
                    return self.render(value, index, owner, original)
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)       # never leave a cancelled/failed attempt's partial output
            except OSError:
                pass
            raise
        saved = {"sha256": file_hash(output), "start": start, "duration": duration, "audio": audio_hash}
        if init_name:
            saved["init"] = init_name
        if leveled:
            saved["loudness_gain_db"] = gain_db
        if seamless:
            saved["join"] = {"version": 1, "pre_frames": plan["pre"], "tail": plan["tail"], "keep": plan["keep"],
                             "pts_base": _join.TS_BASE_SECONDS, "pre_roll": pre_identity, "early_seek": bool(ts_source)}
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
        elif selection.startswith("yt:") and value.get("youtube"):
            from .youtube_source import parse_json3
            try:
                cues = parse_json3(self.youtube_source().captions(value["youtube"]["id"], selection[3:]).read_text(encoding="utf-8"))
            except Exception:
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
        from .plex_neighbors import fill_neighbours, plex_info
        # F24: show / season / episode at once; the previous / next episode (up to four Plex requests) are looked up by
        # a thread and appear in the session's `plex` block one status later.
        value["plex"] = plex_info(self.plex, metadata, lookup=False)
        if metadata.get("type") == "episode":
            threading.Thread(target=fill_neighbours, args=(self.plex, metadata, value["plex"]),
                             kwargs={"timeout": PLEX_NEIGHBOUR_TIMEOUT}, name="tv-plex-neighbours", daemon=True).start()
        if resume and metadata.get("viewOffset"):
            return self.control(value["id"], device, "seek", float(metadata["viewOffset"]) / 1000)
        return self.status(value["id"], device)

    def apply_open_prefs(self, ident, device, prefs):
        """F24. The preferences the TV used to send as separate control requests right after `open` (player.js
        openMovie), applied in the same order through the same control() code, in the open request itself. A value a
        control would refuse raises the same ValueError; the session is then stopped (no orphan session or dub job).
        Returns the final status with prefs_applied True (an older server ignores "prefs": the TV then sends the
        controls itself)."""
        value = self.session(ident, device)
        status = None
        try:
            position = prefs.get("resume_position")
            if position is not None and float(position) > 10:
                status = self.control(ident, device, "seek", position=float(position))
            if prefs.get("quality") is not None and (
                    int(prefs["quality"]) != value["quality"] or bool(prefs.get("upscale")) != value["upscale"]
                    or prefs.get("hdr_mode", value["hdr_mode"]) != value["hdr_mode"]
                    or (prefs.get("surround") or "auto") != value.get("surround", "auto")):
                status = self.control(ident, device, "quality", position=value["position"], quality=prefs["quality"],
                                      upscale=prefs.get("upscale"), hdr_mode=prefs.get("hdr_mode", value["hdr_mode"]),
                                      surround=prefs.get("surround") or "auto")
            if prefs.get("encoder") not in (None, "default"):
                status = self.control(ident, device, "processing", position=value["position"], encoder=prefs["encoder"],
                                      encoder_threads=prefs.get("encoder_threads") or 2)
            if prefs.get("dub"):
                status = self.control(ident, device, "dub", position=value["position"], enabled=True)
            if prefs.get("watch"):
                status = self.control(ident, device, "watch", enabled=True)
            if prefs.get("watch_mode"):
                status = self.control(ident, device, "watch_mode", watch_mode=prefs["watch_mode"])
            if "progress_sync" in prefs:
                status = self.control(ident, device, "progress_sync", enabled=bool(prefs["progress_sync"]))
            if prefs.get("subtitles") and prefs["subtitles"] != "off":
                status = self.control(ident, device, "subtitles", subtitles=prefs["subtitles"])
        except Exception:
            self.stop(ident)
            raise
        status = status or self.status(ident, device)
        status["prefs_applied"] = True
        return status

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
