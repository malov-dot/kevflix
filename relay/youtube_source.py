"""YouTube hand-off source (stage 3): resolve and download ONE picked YouTube video with yt-dlp so KevFLIX's own player (and its dubbing /
upscaling pipeline) can play it like any film. Stdlib only; the heavy work runs in a yt-dlp subprocess of the project interpreter.

Rules enforced here and asserted by tests/engine/test_youtube_source.py:
  * only the 11-character video id is ever used; the TV never supplies a URL, and the id is never passed through a shell;
  * metadata first (title, length, language, live flag); live streams and videos longer than youtube.max_minutes are refused with a
    message the TV shows; public videos only (no cookies, no sign-in: a sign-in wall is reported as such);
  * ONE download at a time, cancellable (the whole process tree is stopped), progress parsed from yt-dlp's own output;
  * a stable folder per video id (media_dir/<id>/<id>.mkv + meta.json) so a repeat play is instant and the dubbing cache (keyed by the
    source file) is reused; a size-capped cache (youtube.max_cache_gb) deletes the least recently used folders first.
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


class _Job:
    def __init__(self, video_id):
        self.id, self.state, self.percent, self.text = video_id, "checking", 0, "Checking the video on YouTube…"
        self.error, self.meta, self.proc, self.cancelled, self.reported, self.path = "", {}, None, False, False, ""

    @property
    def running(self):
        return self.state in ("checking", "downloading")

    def view(self):
        meta = self.meta
        return {"state": self.state, "video_id": self.id, "percent": int(self.percent), "text": self.text, "error": self.error,
                "path": self.path, "title": meta.get("title", ""), "duration": meta.get("duration", 0),
                "language": meta.get("language", ""), "uploader": meta.get("uploader", ""),
                "dub_default": dub_default(meta.get("language"))}


class YouTubeSource:
    """Downloads picked videos. `prefix` replaces the yt-dlp command (tests use a fake script); `in_use()` returns the ids whose files a
    playback session still reads (never evicted)."""

    def __init__(self, settings, python="", ffmpeg="", prefix=None, in_use=None, env=None):
        self.settings = settings
        self.media_dir = Path(str(settings["media_dir"]))
        self.python = python or sys.executable
        self.prefix = list(prefix) if prefix else [self.python, "-I", "-m", "yt_dlp"]
        self.ffmpeg, self.in_use, self.env = ffmpeg, in_use or (lambda: set()), env
        self._lock = threading.RLock()
        self._job = None

    # ---- public API --------------------------------------------------------------------------------------------
    def ensure(self, video_id):
        """Idempotent and non-blocking. Returns the view dict: state ready (path set) | checking | downloading | error (error set).
        An error is reported once; asking again for the same video starts a new attempt."""
        video_id = clean_id(video_id)
        with self._lock:
            ready = self._ready(video_id)
            if ready:
                return ready
            job = self._job
            if job and job.running:
                if job.id == video_id:
                    return job.view()
                raise SourceError("Another YouTube video is still downloading")
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
            proc = job.proc
        if proc is not None:
            _kill_tree(proc)
        return True

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
        return job.view()

    def _format(self):
        height = int(self.settings.get("max_height", 1080))
        return ("bv*[height<=%d]+ba[format_note*=original]/bv*[height<=%d]+ba/b[height<=%d]/bv*+ba/b" % (height, height, height))

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
            job.meta = self._metadata(job)
            job.state, job.text = "downloading", "Downloading from YouTube… 0%"
            path = self._download(job)
            meta = dict(job.meta, complete=True, file=path.name, bytes=path.stat().st_size, at=time.time())
            (path.parent / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            job.meta, job.path = meta, str(path)
            self.evict(keep={job.id})
            job.state, job.percent, job.text = "ready", 100, "Ready"
        except SourceError as error:
            self._fail(job, str(error))
        except Exception as error:                                # never leave the job running: the TV would wait forever
            logging.exception("YouTube source failed")
            self._fail(job, "The YouTube download failed: " + (str(error) or error.__class__.__name__)[:160])

    def _fail(self, job, message):
        with self._lock:
            job.error = "Download cancelled" if job.cancelled else message
            job.state, job.text = "error", job.error

    def _metadata(self, job):
        url = "https://www.youtube.com/watch?v=" + job.id
        proc = self._spawn(job, self.prefix + self._common() + ["--dump-json", "-f", self._format(), "--", url], subprocess.PIPE)
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
        if info.get("is_live") or info.get("live_status") in LIVE_STATES:
            raise SourceError("Live streams cannot be played from YouTube yet. Pick a normal video.")
        duration = float(info.get("duration") or 0)
        limit = int(self.settings.get("max_minutes", 240))
        if duration <= 0:
            raise SourceError("YouTube did not report this video's length, so it cannot be played.")
        if duration > limit * 60:
            raise SourceError("This video is %d minutes long; KevFLIX plays YouTube videos up to %d minutes." % (round(duration / 60), limit))
        title = re.sub(r"[\x00-\x1f\x7f]", " ", str(info.get("title") or "YouTube video")).strip()[:200] or "YouTube video"
        # the video's language, else the language of the downloaded audio format (YouTube often leaves the former empty)
        language = info.get("language") or next((f.get("language") for f in reversed(info.get("requested_formats") or [])
                                                 if isinstance(f, dict) and f.get("language")), "")
        return {"id": job.id, "title": title, "duration": duration, "language": str(language or "")[:16],
                "uploader": re.sub(r"[\x00-\x1f\x7f]", " ", str(info.get("uploader") or ""))[:120],
                "streams": max(1, len(info.get("requested_formats") or []))}

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
