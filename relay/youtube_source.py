"""YouTube hand-off source (stage 3): resolve and download ONE picked YouTube video with yt-dlp so KevFLIX's own player (and its dubbing /
upscaling pipeline) can play it like any film. Stdlib only; the heavy work runs in a yt-dlp subprocess of the project interpreter.

Rules enforced here and asserted by tests/engine/test_youtube_source.py:
  * only the 11-character video id is ever used; the TV never supplies a URL, and the id is never passed through a shell;
  * metadata first (title, length, language, live flag); live streams and videos longer than youtube.max_minutes are refused with a
    message the TV shows; public videos only (no cookies, no sign-in: a sign-in wall is reported as such);
  * ONE download at a time, cancellable (the whole process tree is stopped), progress parsed from yt-dlp's own output;
  * a stable folder per video id (media_dir/<id>/<id>.mkv + meta.json) so a repeat play is instant and the dubbing cache (keyed by the
    source file) is reused; a size-capped cache (youtube.max_cache_gb) deletes the least recently used folders first.

Fast start (relay/youtube_stream.py does the work): an ordinary video is NOT downloaded before it plays. yt-dlp only resolves the direct
stream addresses; one hidden `ffmpeg -c copy` then remuxes them into the same media_dir/<id>/<id>.mkv while the playback session already
reads from it (job state "streaming", `readable` seconds growing). When ffmpeg finishes the file is the normal cached one (meta.json
complete) and the job is "ready". A live stream is not downloaded at all: the job only resolves it (state "live") and relay/tv.py
starts a per-session live relay. A video that will be dubbed (foreign or unknown language) is still downloaded completely first, because
dubbing needs the whole file; so is any video the progressive path cannot read (then the original yt-dlp download runs).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import youtube_stream as _stream

CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
PROGRESS = re.compile(r"^KFPROG\|([a-z]+)\|([0-9.]+|NA)\|([0-9.]+|NA)\|([0-9.]+|NA)\s*$")
PROGRESS_TEMPLATE = ("download:KFPROG|%(progress.status)s|%(progress.downloaded_bytes)s|"
                     "%(progress.total_bytes)s|%(progress.total_bytes_estimate)s")
LIVE_STATES = {"is_live", "is_upcoming", "post_live"}
MEDIA_SUFFIXES = (".mkv", ".mp4", ".webm", ".m4v", ".mov")
SIGN_IN = ("YouTube needs a sign-in to play this video. KevFLIX does not sign in to YouTube yet, "
           "so only public videos can be played.")
# Fast-start settings. They live here (not in config.DEFAULTS / youtube_host.DEFAULT_SETTINGS, which must stay equal to each other) and
# a youtube.<key> in relay-config.json overrides them; relay/config.py validates the ranges of the ones that are present.
CAPTION_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,15}")
STREAM_DEFAULTS = {
    "stream": True,                  # an ordinary (not-to-be-dubbed) video plays while it downloads
    "stream_foreign": False,         # True: videos that default to dubbing (foreign or unknown language) stream too (dubbing then waits for the file)
    "stream_segment_seconds": 4,     # playback segment length of a streaming YouTube session (library films keep segment_seconds)
    "stream_start_seconds": 6.0,     # lead the TV is asked to buffer before it starts
    "stream_ahead_seconds": 48.0,    # how far past the playhead segments are prepared
    "stream_max_height": 2160,       # output height cap of a streaming session: by default the TV's own quality / upscale setting decides (lower it for a slow link)
    "stream_maxrate": None, "stream_bufsize": None,     # None: the same rate rule as a library film at that height (30M / 60M at 2160, else 8M / 16M)
    "open_wait_seconds": 1.5,        # the TV's open request waits this long for the first segment before answering "preparing"
    "live": True,                    # live streams play
    "live_segment_seconds": 2, "live_list_size": 8, "live_start_segments": 2,
    "live_max_height": 1080, "live_maxrate": "6M", "live_bufsize": "12M",
}
CODEC_NAMES = {"avc1": "h264", "avc3": "h264", "hev1": "hevc", "hvc1": "hevc", "vp09": "vp9", "vp9": "vp9", "av01": "av1",
                   "mp4a": "aac", "opus": "opus", "vorbis": "vorbis", "ac-3": "ac3", "ec-3": "eac3"}


class SourceError(ValueError):
    """The message is shown on the TV as is."""


def clean_id(value):
    """The 11-character video id, or SourceError. Nothing else (URL, longer text, shell characters) is accepted."""
    if not isinstance(value, str) or not VIDEO_ID.fullmatch(value):
        raise SourceError("That is not a YouTube video id")
    return value


def dub_default(language):
    """English dubbing is ON unless the video is known to be English (an unknown language is dubbed: the pipeline detects the speech)."""
    code = str(language or "").strip().lower().replace("_", "-")
    return not (code in ("en", "eng") or code.startswith("en-"))


def parse_progress(line):
    """One KFPROG line -> (status, downloaded_bytes, total_bytes|None), else None."""
    found = PROGRESS.match(str(line).strip())
    if not found:
        return None

    def number(text):
        return None if text == "NA" else float(text)
    status, done, exact, estimate = found.groups()
    return status, number(done) or 0.0, number(exact) or number(estimate)


def explain(text):
    """A short viewer-facing reason from yt-dlp's error output."""
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    blob = " ".join(lines).lower()
    if any(word in blob for word in ("sign in", "log in", "login required", "confirm your age", "age-restricted",
                                     "members-only", "private video", "cookies")):
        return SIGN_IN
    if "node" in blob and ("runtime" in blob or "not found" in blob or "no such file" in blob):
        return "Node.js was not found; yt-dlp needs it to read YouTube (install Node.js or set youtube.node_path)."
    if any(word in blob for word in ("video unavailable", "is unavailable", "not available", "has been removed", "does not exist")):
        return "YouTube says this video is not available."
    last = lines[-1] if lines else "no details"
    return "YouTube could not provide this video: " + last[:160]


def _kill_tree(proc):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=CREATE_NO_WINDOW, timeout=15)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _folder_bytes(folder):
    total = 0
    for root, _, names in os.walk(folder):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _codec_name(value):
    """ffprobe-style codec name from a yt-dlp codec string ('avc1.640028' -> 'h264', 'mp4a.40.2' -> 'aac')."""
    head = str(value or "").split(".", 1)[0].lower()
    return CODEC_NAMES.get(head, head)


def _clean_headers(headers):
    keep = ("user-agent", "accept", "accept-language", "sec-fetch-mode", "referer", "origin")
    return {str(k): str(v) for k, v in (headers or {}).items() if str(k).lower() in keep}


def caption_list(info):
    """[{code, name, auto}] of the captions YouTube offers for a video: every uploaded track, plus the automatic track in the video's own
    language and English. Addresses are not kept (they expire): captions() asks yt-dlp again when one is picked."""
    def label(tracks, code):
        first = next((t for t in (tracks or []) if isinstance(t, dict) and t.get("name")), {})
        return re.sub(r"[\x00-\x1f\x7f]", " ", str(first.get("name") or code))[:60]
    found = []
    for code, tracks in (info.get("subtitles") or {}).items():
        if CAPTION_CODE.fullmatch(str(code)) and str(code) != "live_chat":
            found.append({"code": str(code), "name": label(tracks, code), "auto": False})
    # Automatic captions: only the speech recognition of the video's own audio ('<lang>-orig', English first). The other automatic
    # entries are YouTube's machine TRANSLATIONS of it, which YouTube rate-limits (HTTP 429), so they are offered only when there is no original.
    auto = info.get("automatic_captions") or {}
    originals = sorted((str(key) for key in auto if str(key).endswith("-orig") and CAPTION_CODE.fullmatch(str(key))),
                       key=lambda code: (not code.startswith("en"), code))
    for code in (originals[:3] or (["en"] if "en" in auto else [])):
        if not any(item["code"] == code for item in found):
            found.append({"code": code, "name": label(auto[code], code) + " (automatic)", "auto": True})
    return found[:12]


def parse_json3(text):
    """[{start, end, text}] (seconds) from a YouTube json3 caption file; empty lines and bare line-break events are skipped."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    cues = []
    for event in (data.get("events") or []) if isinstance(data, dict) else []:
        if not isinstance(event, dict) or not event.get("segs"):
            continue
        line = " ".join("".join(str(seg.get("utf8") or "") for seg in event["segs"] if isinstance(seg, dict)).split())
        if line:
            start = float(event.get("tStartMs") or 0) / 1000
            cues.append({"start": start, "end": start + float(event.get("dDurationMs") or 2000) / 1000, "text": line})
    return cues


def stream_inputs(info):
    """The inputs the range proxy can read for a resolved video: [{url, headers, length, kind}] (one video and one audio, or one
    muxed file), or None when any selected format is not a plain https googlevideo file (HLS, DASH fragments, DRM...)."""
    formats = info.get("requested_formats") or ([info] if info.get("url") else [])
    inputs, kinds = [], []
    for fmt in formats:
        if not isinstance(fmt, dict):
            return None
        url = str(fmt.get("url") or "")
        parts = urlsplit(url)
        if (fmt.get("protocol") != "https" or parts.scheme != "https" or not _stream.host_allowed(parts.hostname)
                or fmt.get("has_drm") or fmt.get("fragments")):
            return None
        video, audio = str(fmt.get("vcodec") or "none") != "none", str(fmt.get("acodec") or "none") != "none"
        length = 0
        try:
            length = int(parse_qs(parts.query).get("clen", ["0"])[0] or 0) or int(fmt.get("filesize") or 0)
        except (TypeError, ValueError):
            pass
        inputs.append({"url": url, "headers": _clean_headers(fmt.get("http_headers") or info.get("http_headers")),
                       "length": length or None, "kind": "muxed" if video and audio else "video" if video else "audio"})
        kinds.append((video, audio))
    if len(inputs) == 1 and kinds[0] == (True, True):
        return inputs
    if len(inputs) == 2 and kinds[0] == (True, False) and kinds[1] == (False, True):
        return inputs
    return None


def stream_formats(info):
    """A small, URL-free description of the selected formats (kept in meta.json and used to describe the growing file's streams)."""
    out = []
    for fmt in info.get("requested_formats") or ([info] if info.get("url") else []):
        if not isinstance(fmt, dict):
            continue
        if str(fmt.get("vcodec") or "none") != "none":
            out.append({"kind": "video", "codec": _codec_name(fmt.get("vcodec")), "width": int(fmt.get("width") or 0),
                        "height": int(fmt.get("height") or 0), "fps": float(fmt.get("fps") or 0)})
        if str(fmt.get("acodec") or "none") != "none":
            out.append({"kind": "audio", "codec": _codec_name(fmt.get("acodec")), "channels": int(fmt.get("audio_channels") or 2),
                        "rate": int(fmt.get("asr") or 48000), "language": str(fmt.get("language") or "")[:16]})
    return out


class _Job:
    def __init__(self, video_id):
        self.id, self.state, self.percent, self.text = video_id, "checking", 0, "Checking the video on YouTube…"
        self.error, self.meta, self.proc, self.cancelled, self.reported, self.path = "", {}, None, False, False, ""
        self.mode, self.readable, self.streamer, self.inputs, self.live, self.live_at = "full", 0.0, None, None, None, 0.0
        self.t0, self.timing = time.monotonic(), {}            # seconds since the job started: resolve / first readable / complete (diagnostics)

    @property
    def running(self):
        return self.state in ("checking", "downloading", "streaming")

    def view(self):
        meta = self.meta
        done = self.state == "ready"
        return {"state": self.state, "video_id": self.id, "percent": int(self.percent), "text": self.text, "error": self.error,
                "path": self.path, "title": meta.get("title", ""), "duration": meta.get("duration", 0),
                "language": meta.get("language", ""), "uploader": meta.get("uploader", ""),
                "channel": meta.get("channel") or meta.get("uploader", ""), "captions": list(meta.get("captions") or []),
                "dub_default": dub_default(meta.get("language")) and not meta.get("live"),      # a live stream is never dubbed
                # fast start: how the file is obtained, how many seconds of it are readable now, whether it is the finished cached file
                "mode": self.mode, "readable": float(meta.get("duration", 0)) if done else float(self.readable), "complete": done,
                "live": self.state == "live", "formats": list(meta.get("formats") or []), "timing": dict(self.timing)}


class YouTubeSource:
    """Downloads picked videos. `prefix` replaces the yt-dlp command (tests use a fake script); `in_use()` returns the ids whose files a
    playback session still reads (never evicted)."""

    def __init__(self, settings, python="", ffmpeg="", prefix=None, in_use=None, env=None, notify=None, streamer=None):
        self.settings = settings
        self.media_dir = Path(str(settings["media_dir"]))
        self.python = python or sys.executable
        self.prefix = list(prefix) if prefix else [self.python, "-I", "-m", "yt_dlp"]
        self.ffmpeg, self.in_use, self.env = ffmpeg, in_use or (lambda: set()), env
        self.notify = notify or (lambda: None)             # called whenever readable seconds / the state change (wakes the segment producer)
        self.streamer = streamer or _stream.Streamer       # tests substitute a stand-in that writes a growing file
        self._lock = threading.RLock()
        self._job = None
        self._done = {}                                    # video id -> (seconds, checked at): files seen complete (progress() asks often)

    def opt(self, key):
        """A fast-start setting: the youtube.<key> of the settings if present, else STREAM_DEFAULTS[key]."""
        value = self.settings.get(key)
        return STREAM_DEFAULTS[key] if value is None else value

    # ---- public API --------------------------------------------------------------------------------------------
    def ensure(self, video_id):
        """Idempotent and non-blocking. Returns the view dict: state ready (path set, complete file) | streaming (path set, `readable`
        seconds of the growing file) | live (a live stream was resolved; nothing is downloaded) | checking | downloading | error (error
        set). An error is reported once; asking again for the same video starts a new attempt."""
        video_id = clean_id(video_id)
        busy = set(self.in_use())                            # asked before the lock: the player's lock is never taken inside this one
        with self._lock:
            ready = self._ready(video_id)
            if ready:
                return ready
            job = self._job
            if job and job.running:
                if job.id == video_id:
                    return job.view()
                if job.id in busy:
                    raise SourceError("Another YouTube video is still downloading")
                self.cancel()                                # nobody plays that video any more: the new pick wins
            if job and job.id == video_id and job.state == "live" and time.time() - job.live_at < 120:
                return job.view()                            # just resolved: the caller opens the live relay from it
            if job and job.id == video_id and job.state == "error" and not job.reported:
                job.reported = True
                return job.view()
            self._job = job = _Job(video_id)
            threading.Thread(target=self._run, args=(job,), daemon=True, name="youtube-source").start()
            return job.view()

    def cancel(self):
        """Stop the running download (its whole process tree). True when one was running."""
        with self._lock:
            job = self._job
            if not job or not job.running:
                return False
            job.cancelled = True
            proc, streamer = job.proc, job.streamer
        if proc is not None:
            _kill_tree(proc)
        if streamer is not None:
            streamer.stop()
        return True

    def cancel_video(self, video_id):
        """cancel(), but only when the running job is for `video_id` (a playback session of that video ended)."""
        with self._lock:
            job = self._job
            if not job or not job.running or job.id != video_id:
                return False
        return self.cancel()

    def progress(self, video_id):
        """{readable, complete, error, running} for the file a playback session of `video_id` reads while it is being written.
        Takes no lock (the player calls it with its own lock held, every status poll and scheduler pass)."""
        job = self._job
        if job is not None and job.id == video_id and job.mode == "stream":
            if job.state == "ready":
                return {"readable": float(job.meta.get("duration", 0)), "complete": True, "error": "", "running": False}
            if job.state == "error":
                return {"readable": float(job.readable), "complete": False, "error": job.error, "running": False}
            return {"readable": float(job.readable), "complete": False, "error": "", "running": job.running}
        known = self._done.get(video_id)
        if known is not None and time.monotonic() - known[1] < 5.0:
            return {"readable": known[0], "complete": True, "error": "", "running": False}
        ready = self._ready(video_id)
        if ready:
            self._done[video_id] = (float(ready["duration"]), time.monotonic())
            return {"readable": float(ready["duration"]), "complete": True, "error": "", "running": False}
        self._done.pop(video_id, None)
        return {"readable": 0.0, "complete": False, "error": "The video is no longer available on the PC", "running": False}

    def prefetch(self, video_id):
        """Start resolving `video_id` now (the pick is known before the TV asks to open it). Does nothing when the video is cached, is
        already being handled, or another video's job is running. True when a job was started."""
        try:
            video_id = clean_id(video_id)
            job = self._job
            if (job is not None and (job.running or job.id == video_id)) or self._ready(video_id):
                return False
            self.ensure(video_id)
            return True
        except SourceError:
            return False

    def captions(self, video_id, code):
        """The json3 captions of one language as a file: downloaded with yt-dlp the first time (a few seconds), then kept next to the
        video. SourceError (shown on the TV) when YouTube has none or does not answer."""
        video_id = clean_id(video_id)
        code = str(code)
        if not CAPTION_CODE.fullmatch(code):
            raise SourceError("Those captions are not available")
        folder = Path(self.settings["media_dir"]) / video_id
        target = folder / ("%s.%s.json3" % (video_id, code))
        if target.is_file():
            return target
        folder.mkdir(parents=True, exist_ok=True)
        command = self.prefix + self._common() + ["--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", code,
                                                  "--sub-format", "json3", "-o", str(folder / "%(id)s"), "--",
                                                  "https://www.youtube.com/watch?v=" + video_id]
        flags = {"creationflags": CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS} if os.name == "nt" else {}
        try:
            done = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=90, env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **(self.env or {})}, **flags)
        except subprocess.TimeoutExpired:
            raise SourceError("YouTube did not answer in time")
        if not target.is_file():
            if "429" in (done.stderr or ""):
                raise SourceError("YouTube is limiting caption requests right now. Try again in a few minutes.")
            raise SourceError("YouTube has no captions in that language for this video")
        return target

    def live_inputs(self, video_id):
        """The resolved live stream of the last job ({inputs: [{url, headers, kind}], title, ...}) or None. Addresses never leave the PC."""
        with self._lock:
            job = self._job
            return dict(job.live) if job and job.id == video_id and job.live else None

    def evict(self, keep=()):
        """Delete the least recently used folders until the cache is within youtube.max_cache_gb. Never deletes `keep` or an in-use id."""
        cap = int(float(self.settings.get("max_cache_gb", 30)) * 1e9)
        removed, folders = [], []
        if not self.media_dir.is_dir():
            return removed
        root = self.media_dir.resolve()
        for path in self.media_dir.iterdir():
            if path.is_dir() and not path.is_symlink() and VIDEO_ID.fullmatch(path.name) and path.resolve().parent == root:
                folders.append((path.stat().st_mtime, path, _folder_bytes(path)))
        total = sum(item[2] for item in folders)
        busy = set(keep) | set(self.in_use())
        for _, path, size in sorted(folders, key=lambda item: item[0]):
            if total <= cap:
                break
            if path.name in busy:
                continue
            shutil.rmtree(path, ignore_errors=True)
            self._done.pop(path.name, None)
            total -= size
            removed.append(path.name)
        return removed

    # ---- internals ---------------------------------------------------------------------------------------------
    def _ready(self, video_id):
        folder = self.media_dir / video_id
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            path = folder / str(meta["file"])
            if not meta.get("complete") or path.parent != folder or not path.is_file():
                return None
            os.utime(folder, None)                               # most recently used -> last to be evicted
        except (OSError, ValueError, KeyError):
            return None
        job = _Job(video_id)
        job.state, job.percent, job.text, job.meta, job.path = "ready", 100, "Ready", meta, str(path)
        job.mode = str(meta.get("mode") or "full")
        return job.view()

    def _format(self):
        height = int(self.settings.get("max_height", 1080))
        return ("bv*[height<=%d]+ba[format_note*=original]/bv*[height<=%d]+ba/b[height<=%d]/bv*+ba/b" % (height, height, height))

    def _format_stream(self):
        """The selector of the fast-start resolve: plain https H.264 + AAC files first (what the range proxy can read and a TV decodes
        cheaply), the original-audio track before any other, then the same fallbacks as the full download (_format)."""
        height = int(self.settings.get("max_height", 1080))
        plain = "[protocol=https]"
        return ("bv*[height<=%(h)d][vcodec^=avc1]%(p)s+ba[acodec^=mp4a][format_note*=original]%(p)s/"
                "bv*[height<=%(h)d][vcodec^=avc1]%(p)s+ba[acodec^=mp4a]%(p)s/"
                "bv*[height<=%(h)d]%(p)s+ba[format_note*=original]%(p)s/bv*[height<=%(h)d]%(p)s+ba%(p)s/" % {"h": height, "p": plain}
                + self._format())

    def _common(self):
        node = str(self.settings.get("node_path") or "") or shutil.which("node") or ""
        args = ["--ignore-config", "--no-playlist", "--no-warnings", "--no-colors", "--socket-timeout", "30", "--retries", "5",
                "--js-runtimes", ("node:" + node) if node else "node"]
        ffmpeg = str(self.ffmpeg or "")
        if ffmpeg and Path(ffmpeg).is_file():
            args += ["--ffmpeg-location", ffmpeg]
        return args

    def _spawn(self, job, command, stderr):
        kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": stderr, "text": True, "encoding": "utf-8",
                  "errors": "replace", "env": {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **(self.env or {})}}
        if os.name == "nt":
            kwargs["creationflags"] = CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
        with self._lock:
            if job.cancelled:
                raise SourceError("Download cancelled")
            job.proc = subprocess.Popen(command, **kwargs)
        return job.proc

    def _run(self, job):
        try:
            job.meta, info = self._metadata(job)
            job.timing["resolve_s"] = round(time.monotonic() - job.t0, 2)
            if job.state == "live":                                  # resolved only: relay/tv.py starts the live relay; nothing to download
                return
            path = self._stream(job) if job.mode == "stream" else None
            if path is None:                                         # not streamable (dubbing, HLS-only, remux failed to start): download it all first
                job.mode, job.readable = "full", 0.0
                job.state, job.percent, job.text = "downloading", 0, "Downloading from YouTube… 0%"
                path = self._download(job)
            meta = dict(job.meta, complete=True, file=path.name, bytes=path.stat().st_size, at=time.time(), mode=job.mode)
            (path.parent / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            job.meta, job.path = meta, str(path)
            self.evict(keep={job.id})
            job.state, job.percent, job.text = "ready", 100, "Ready"
            job.timing["complete_s"] = round(time.monotonic() - job.t0, 2)
            logging.info("YouTube %s %s: %s", job.mode, job.id, json.dumps(job.timing))
            self.notify()
        except SourceError as error:
            self._fail(job, str(error))
        except Exception as error:                                # never leave the job running: the TV would wait forever
            logging.exception("YouTube source failed")
            self._fail(job, "The YouTube download failed: " + (str(error) or error.__class__.__name__)[:160])

    def _fail(self, job, message):
        with self._lock:
            job.error = "Download cancelled" if job.cancelled else message
            job.state, job.text = "error", job.error
        if job.mode == "stream" and job.state == "error":
            shutil.rmtree(self.media_dir / job.id, ignore_errors=True)      # an unfinished growing file is never reused
        self.notify()

    def _stream(self, job):
        """Remux the resolved streams into media_dir/<id>/<id>.mkv while the session plays from it. Returns the finished file's path;
        None when ffmpeg could not start reading at all (the caller falls back to the full download)."""
        folder = self.media_dir / job.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").unlink(missing_ok=True)
        self._done.pop(job.id, None)
        out = folder / (job.id + ".mkv")
        streamer = self.streamer(self.ffmpeg or "ffmpeg", job.inputs, out, notify=self.notify)
        with self._lock:
            if job.cancelled:
                raise SourceError("Download cancelled")
            job.streamer = streamer
            job.path = str(out)
            streamer.start()
        job.state, job.text = "streaming", "Starting the video…"
        duration = float(job.meta.get("duration") or 0)
        while not streamer.done.wait(0.1):
            job.readable = streamer.readable
            if job.readable >= 4.5 and "first_readable_s" not in job.timing:
                job.timing["first_readable_s"] = round(time.monotonic() - job.t0, 2)
            job.percent = max(job.percent, min(99, int(100 * streamer.readable / duration))) if duration else job.percent
            if job.cancelled:
                streamer.stop()
                break
        job.readable = streamer.readable
        if job.cancelled:
            raise SourceError("Download cancelled")
        if streamer.complete:
            if not out.is_file() or out.stat().st_size <= 0:
                raise SourceError("The download finished but no video file was found")
            return out
        if streamer.readable < 1.0:
            logging.warning("YouTube progressive remux did not start (%s); downloading the video completely instead", streamer.failed)
            job.streamer, job.path = None, ""
            shutil.rmtree(folder, ignore_errors=True)
            return None
        raise SourceError("The YouTube download stopped: " + (streamer.failed or "connection lost"))

    def _metadata(self, job):
        url = "https://www.youtube.com/watch?v=" + job.id
        # fast start off: the original selector, byte for byte (it resolves a live stream's HLS formats too)
        proc = self._spawn(job, self.prefix + self._common() + [
            "--dump-json", "-f", self._format_stream() if self.opt("stream") else self._format(), "--", url], subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            raise SourceError("YouTube did not answer in time")
        if job.cancelled:
            raise SourceError("Download cancelled")
        if proc.returncode != 0:
            raise SourceError(explain(err))
        try:
            info = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise SourceError("YouTube returned no details for this video")
        title = re.sub(r"[\x00-\x1f\x7f]", " ", str(info.get("title") or "YouTube video")).strip()[:200] or "YouTube video"
        uploader = re.sub(r"[\x00-\x1f\x7f]", " ", str(info.get("uploader") or ""))[:120]
        channel = re.sub(r"[\x00-\x1f\x7f]", " ", str(info.get("channel") or info.get("uploader") or ""))[:120]
        status = info.get("live_status")
        if info.get("is_live") or status in LIVE_STATES:
            if status == "is_upcoming":
                raise SourceError("This live stream has not started yet.")
            if status == "post_live":
                raise SourceError("This live stream has ended and YouTube is still processing the recording.")
            if not self.opt("live"):
                raise SourceError("Live streams cannot be played from YouTube yet. Pick a normal video.")
            inputs = self._live_inputs(info)
            if inputs is None:
                raise SourceError("YouTube did not offer a live picture this PC can play.")
            job.live = {"inputs": inputs, "title": title, "uploader": uploader}
            job.meta = {"id": job.id, "title": title, "duration": 0.0, "language": "", "uploader": uploader, "channel": channel, "live": True,
                        "formats": stream_formats(info)}
            job.live_at = time.time()
            job.state, job.percent, job.text = "live", 0, "Starting the live stream…"
            return job.meta, info
        duration = float(info.get("duration") or 0)
        limit = int(self.settings.get("max_minutes", 240))
        if duration <= 0:
            raise SourceError("YouTube did not report this video's length, so it cannot be played.")
        if duration > limit * 60:
            raise SourceError("This video is %d minutes long; KevFLIX plays YouTube videos up to %d minutes." % (round(duration / 60), limit))
        # the video's language, else the language of the downloaded audio format (YouTube often leaves the former empty), else the
        # language YouTube's own speech recognition reports as the original ('en-orig' in the automatic captions)
        language = info.get("language") or next((f.get("language") for f in reversed(info.get("requested_formats") or [])
                                                 if isinstance(f, dict) and f.get("language")), "")
        if not language:
            language = next((str(key)[:-5] for key in (info.get("automatic_captions") or {}) if str(key).endswith("-orig")), "")
        meta = {"id": job.id, "title": title, "duration": duration, "language": str(language or "")[:16], "uploader": uploader,
                "channel": channel, "captions": caption_list(info),
                "streams": max(1, len(info.get("requested_formats") or [])), "formats": stream_formats(info)}
        inputs = stream_inputs(info) if self.opt("stream") else None
        if inputs and (self.opt("stream_foreign") or not dub_default(meta["language"])):
            job.mode, job.inputs = "stream", inputs
        return meta, info

    def _live_inputs(self, info):
        """[{url, headers, kind}] of the selected live format(s) (HLS playlists ffmpeg reads itself), or None."""
        formats = info.get("requested_formats") or ([info] if info.get("url") else [])
        inputs = []
        for fmt in formats:
            parts = urlsplit(str((fmt or {}).get("url") or ""))
            if parts.scheme != "https" or not _stream.host_allowed(parts.hostname) or not str(fmt.get("protocol") or "").startswith("m3u8"):
                return None
            inputs.append({"url": str(fmt["url"]), "headers": _clean_headers(fmt.get("http_headers") or info.get("http_headers")),
                           "kind": "video" if str(fmt.get("vcodec") or "none") != "none" else "audio"})
        return inputs or None

    def _download(self, job):
        folder = self.media_dir / job.id
        folder.mkdir(parents=True, exist_ok=True)
        command = self.prefix + self._common() + [
            "-f", self._format(), "--merge-output-format", "mkv", "--newline", "--no-mtime", "--progress-template", PROGRESS_TEMPLATE,
            "-o", str(folder / (job.id + ".%(ext)s")), "--", "https://www.youtube.com/watch?v=" + job.id]
        proc = self._spawn(job, command, subprocess.STDOUT)
        finished, streams, tail = 0, max(1, int(job.meta.get("streams", 1))), []
        for line in iter(proc.stdout.readline, ""):
            parsed = parse_progress(line)
            if not parsed:
                if line.strip():
                    tail = (tail + [line.strip()])[-12:]
                    if line.startswith("[Merger]") or line.startswith("[VideoRemuxer]"):
                        job.text = "Finishing the download…"
                continue
            status, done, total = parsed
            if status == "finished":
                finished, fraction = finished + 1, 0.0
            else:
                fraction = min(1.0, done / total) if total else 0.0
            job.percent = max(job.percent, min(99, int(100 * (finished + fraction) / streams)))   # never goes backwards; 100 only when staged
            job.text = "Downloading from YouTube… %d%%" % job.percent
        proc.stdout.close()
        proc.wait()
        if job.cancelled:
            raise SourceError("Download cancelled")
        if proc.returncode != 0:
            shutil.rmtree(folder, ignore_errors=True)
            raise SourceError(explain("\n".join(tail)))
        for candidate in sorted(folder.iterdir()):
            if candidate.suffix.lower() in MEDIA_SUFFIXES and candidate.stem == job.id and candidate.stat().st_size > 0:
                return candidate
        shutil.rmtree(folder, ignore_errors=True)
        raise SourceError("The download finished but no video file was found")
