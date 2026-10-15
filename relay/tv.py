"""Separate TV player service. Never modifies Plex binaries or library databases."""
from __future__ import annotations

import hashlib
import hmac
import json
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

from .media import probe, fingerprint, file_hash, subtitle_cues
from .process import ProcessOwner, Cancelled

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".webm", ".mpg", ".mpeg"}


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
        roots = self._roots()
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

    def open_movie(self, device, node, audio_index=None):
        with self.lock:
            source = self.nodes.get(node)
        if source is None or source.suffix.lower() not in MEDIA_EXTENSIONS:
            raise ValueError("Choose a movie from the library")
        self._node(source)
        source = self.config.media_path(source)
        metadata = probe(self.config, ProcessOwner(), source)
        streams = metadata["audio"]
        audio_index = streams[0]["index"] if audio_index is None else int(audio_index)
        if not any(s["index"] == audio_index for s in streams):
            raise ValueError("Selected audio stream is absent")
        with self.lock:
            old = [k for k, v in self.sessions.items() if v["device_id"] == device]
        for ident in old:
            self.stop(ident)
        ident = uuid.uuid4().hex
        spec = {"id": ident, "device_id": device, "source": str(source),
                "fingerprint": fingerprint(source), "title": source.stem,
                "duration": metadata["duration"], "streams": metadata["streams"],
                "audio_index": audio_index, "segment_seconds": int(self.config["segment_seconds"]),
                "position": 0.0, "start_index": 0, "generation": 1, "epoch": 0,
                "dub": False, "compatibility": bool(self.config["tv"]["pc_process_all"]),
                "quality": int(self.config["tv"]["max_height"]), "upscale": bool(self.config["tv"]["upscale"]),
                "hdr_mode": self.config["tv"]["hdr_mode"], "subtitles": "off", "watch_mode": "balanced",
                "plex_key": None, "sync_progress": bool(self.config["tv"]["sync_plex_progress"]),
                "job_id": None, "job_revision": None, "watch": False, "state": "paused", "resource_wait": None,
                "touched": time.time(), "segments": {}, "error": None}
        spec["expires"] = int(time.time()) + 12 * 3600
        with self.lock:
            self.sessions[ident] = spec
        self.wake.set()
        return self.status(ident, device)

    def session(self, ident, device=None, touch=True):
        with self.lock:
            value = self.sessions.get(ident)
            if value is None or (device and value["device_id"] != device):
                raise ValueError("TV playback session has ended")
            if touch:
                value["touched"] = time.time()
            return value

    def control(self, ident, device, action, position=None, enabled=None, audio_index=None,
                quality=None, upscale=None, hdr_mode=None, subtitles=None, watch_mode=None):
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
        elif action in ("seek", "dub", "audio", "compatibility", "quality"):
            old_job = value["job_id"]
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
                value.update(quality=quality, upscale=bool(upscale), hdr_mode=hdr_mode, compatibility=True)
            new_position = value["position"] if position is None else float(position)
            if not math.isfinite(new_position):
                raise ValueError("Invalid playback position")
            job = None
            if (action == "dub" and bool(enabled)) or (action == "audio" and value["dub"]):
                job = self.jobs.enqueue(value["source"], audio_index=value["audio_index"],
                                        mode="ondemand", title=value["title"])
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
                if action == "compatibility" and not self.config["tv"]["pc_process_all"]:
                    value["compatibility"] = bool(enabled)
                if action in ("dub", "audio"):
                    value["dub"] = bool(enabled) if action == "dub" else value["dub"]
                    value["job_id"] = job["id"] if job else None
                    value["job_revision"] = job["payload"].get("revision", 1) if job else None
            if self.active_session == ident and self.active_owner:
                self.active_owner.cancel()
            if value["job_id"]:
                self.jobs.request(value["job_id"], value["start_index"], -int(time.time()))
            if old_job and old_job != value["job_id"]:
                self.cancel_unshared_job(old_job)
        elif action in ("play", "pause"):
            value["state"] = "playing" if action == "play" else "paused"
        else:
            raise ValueError("Unsupported TV action")
        self.wake.set()
        return self.status(ident, device)

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
        self.store.put_record("sessions", "tv-" + ident, {
            "source_kind": "tizenbrew", "session_id": ident, "client_id": device,
            "source": value["source"], "title": value["title"], "nas_path": "",
            "rating_key": "", "transcode_session": "", "audio_index": value["audio_index"],
            "state": value["state"], "position": value["position"], "duration": value["duration"],
            "observed": time.time(), "epoch": value["epoch"], "job_id": value["job_id"],
            "dub": value["dub"], "watch": value["watch"], "watch_mode": value["watch_mode"]})
        if value["sync_progress"] and value["plex_key"] and time.time() - value.get("progress_at", 0) >= 15:
            value["progress_at"] = time.time()
            try:
                self.plex.call("/:/timeline", {"ratingKey": value["plex_key"],
                    "key": "/library/metadata/" + value["plex_key"],
                    "state": "stopped" if value["state"] == "ended" else value["state"],
                    "time": round(value["position"] * 1000), "duration": round(value["duration"] * 1000)})
            except Exception:
                value["progress_warning"] = "Plex progress could not be saved"
        self.wake.set()
        return self.status(ident, device)

    def stop(self, ident):
        with self.lock:
            value = self.sessions.pop(ident, None)
        self.store.expire_record("sessions", "tv-" + ident)
        if self.active_session == ident and self.active_owner:
            self.active_owner.cancel()
        if value and value["job_id"]:
            self.cancel_unshared_job(value["job_id"])

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
            if value["job_id"]:
                job = self.store.job(value["job_id"])
                if job and job["state"] in ("needs_review", "error", "cancelled"):
                    issue = "Dubbing " + job["state"] + ". Review it on the PC, or turn dubbing off."
            return {"id": ident, "title": value["title"], "duration": value["duration"],
                "position": value["position"], "base": 0 if direct else value["start_index"] * value["segment_seconds"],
                "state": value["state"], "generation": value["generation"], "epoch": value["epoch"],
                "dub": value["dub"], "compatibility": value["compatibility"], "watch": value["watch"], "job_id": value["job_id"],
                "quality": value["quality"], "upscale": value["upscale"], "hdr_mode": value["hdr_mode"],
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
                "audio_index": value["audio_index"]}

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
            if not job or job["payload"].get("revision", 1) != value["job_revision"]:
                raise ValueError("Dubbing was revised; refresh the playback session")
        return value

    def playlist(self, value):
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

    def loop(self):
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
                current = max(value["start_index"], int(value["position"] // value["segment_seconds"]))
                limit = min(math.ceil(value["duration"] / value["segment_seconds"]),
                            current + max(2, int(self.config["tv"]["buffer_segments"])))
                missing = next((i for i in range(current, limit) if i not in value["segments"]), None)
                if missing is None:
                    continue
                if value["job_id"]:
                    try:
                        self.jobs.request(value["job_id"], missing, -int(time.time()))
                    except ValueError:
                        continue
                    chunk = self.store.chunk(value["job_id"], missing)
                    if not chunk or chunk["state"] not in ("ready", "approved"):
                        continue
                generation = value["generation"]
                owner = ProcessOwner()
                self.active_owner, self.active_session = owner, ident
                try:
                    path = self.render(dict(value), missing, owner)
                    with self.lock:
                        if self.sessions.get(ident) is value and value["generation"] == generation:
                            value["segments"][missing] = str(path)
                            value["resource_wait"] = None
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
        if fingerprint(value["source"]) != value["fingerprint"]:
            raise ValueError("The source movie changed; reopen it")
        start = index * value["segment_seconds"]
        duration = min(value["segment_seconds"], value["duration"] - start)
        audio, audio_hash = None, "original"
        if value["job_id"]:
            job = self.store.job(value["job_id"])
            audio = self.jobs.audio_chunk(job["id"], index, job["payload"].get("revision", 1), wait=False)
            audio_hash = file_hash(audio)
        identity = json.dumps([value["fingerprint"], value["audio_index"], index, audio_hash,
                               value["quality"], value["upscale"], value["hdr_mode"],
                               self.config["tv"]["video_encoder"], "tv-mux-v2"])
        key = hashlib.sha256(identity.encode()).hexdigest()
        directory = self.config.cache / "tv-media"
        directory.mkdir(parents=True, exist_ok=True)
        output, receipt = directory / (key + ".ts"), directory / (key + ".json")
        if output.exists() and receipt.exists():
            saved = json.loads(receipt.read_text())
            if file_hash(output) == saved.get("sha256"):
                return output
        temporary = directory / (key + ".partial.ts")
        args = [self.config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
                "-ss", str(start), "-i", value["source"]]
        if audio:
            args += ["-i", str(audio)]
        height, width = value["quality"], value["quality"] * 16 // 9
        scale = ("scale=w=%s:h=%s" % (width, height) if value["upscale"] else
                 "scale=w='min(%s,iw)':h='min(%s,ih)'" % (width, height))
        scale += ":force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,setsar=1"
        video = next((s for s in value["streams"] if s.get("codec_type") == "video"
                      and not s.get("disposition", {}).get("attached_pic")), None)
        if not video:
            raise ValueError("No playable video stream")
        hdr = video.get("color_transfer") in ("smpte2084", "arib-std-b67")
        preserve_hdr = hdr and value["hdr_mode"] == "auto"
        if hdr and not preserve_hdr:
            scale = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv," + scale
        use_gpu = self.config["tv"]["video_encoder"] != "software"
        if use_gpu:
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            try:
                inventory = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, timeout=5, check=True, **flags)
                free = int(inventory.stdout.decode().splitlines()[0].strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                raise ResourceBusy("PC NVIDIA encoder is unavailable. Choose software encoding on the PC or restore GPU availability.")
            if free < int(self.config["tv"]["nvenc_min_free_mib"]):
                raise ResourceBusy("Waiting for PC GPU memory; no other application or model will be unloaded.")
        codec = ("hevc_nvenc" if preserve_hdr else "h264_nvenc") if use_gpu else ("libx265" if preserve_hdr else "libx264")
        codec_options = ["-preset", "p4", "-rc", "vbr", "-cq", "21", "-b:v", "0"] if use_gpu else ["-preset", "veryfast", "-crf", "22"]
        if use_gpu and preserve_hdr:
            codec_options += ["-profile:v", "main10"]
        if not use_gpu and preserve_hdr:
            codec_options += ["-x265-params", "pools=2:frame-threads=2:log-level=error"]
        rate = "30M" if height == 2160 else "8M"
        color = (["-color_primaries", "bt2020", "-color_trc", video["color_transfer"],
                  "-colorspace", "bt2020nc"] if preserve_hdr else [])
        args += ["-map", "0:" + str(video["index"]), "-map", "1:a:0" if audio else "0:" + str(value["audio_index"]),
                 "-t", "%.6f" % duration, "-vf", scale, "-c:v", codec, *codec_options,
                 "-threads", str(self.config["tv"]["encoder_threads"]), "-filter_threads", "2",
                 "-pix_fmt", "p010le" if preserve_hdr and use_gpu else "yuv420p10le" if preserve_hdr else "yuv420p",
                 "-maxrate", rate, "-bufsize", "60M" if height == 2160 else "16M", *color,
                 "-force_key_frames", "expr:gte(t,n_forced*2)",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                 "-af", "apad", "-sn", "-dn", "-output_ts_offset", str(start),
                 "-muxdelay", "0", "-muxpreload", "0", "-mpegts_flags", "+resend_headers",
                 "-f", "mpegts", "-y", str(temporary)]
        owner.run(args, timeout=max(180, duration * 20))
        owner.check()
        if not temporary.is_file() or temporary.stat().st_size < 188:
            raise ValueError("TV mux produced no transport stream")
        os.replace(temporary, output)
        receipt.write_text(json.dumps({"sha256": file_hash(output), "start": start,
                                      "duration": duration, "audio": audio_hash}))
        return output

    def captions(self, value):
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
        return [{"id": str(s["key"]), "title": s["title"]} for s in self.plex.sections() if s.get("type") == "movie"]

    def plex_library(self, section, query="", offset=0):
        items = self.plex.library(section, query, int(offset))
        return {"items": [{"id": str(item["ratingKey"]), "title": item.get("title", ""),
            "year": item.get("year"), "summary": item.get("summary", ""),
            "resume": float(item.get("viewOffset", 0)) / 1000, "folder": False,
            "plex": True, "poster": bool(item.get("thumb"))} for item in items if item.get("type") == "movie"],
            "more": len(items) == 100, "offset": int(offset)}

    def open_plex(self, device, key, resume=True):
        from .plex import selected_part
        metadata = self.plex.metadata(key)
        part, audio, _ = selected_part(metadata)
        source = self.config.media_path(part["file"], nas=True)
        node = self._node(source)
        opened = self.open_movie(device, node["id"], int(audio["index"]))
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
