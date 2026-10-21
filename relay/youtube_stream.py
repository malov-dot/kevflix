"""YouTube progressive playback and live relay (stdlib only; ffmpeg and yt-dlp run as hidden, below-normal-priority child processes).

Why this exists. relay/youtube_source.py used to download the WHOLE video before the TV could start. Measured on the installed
0.4.65 server: about 30 s from the OK press to the first picture of a 12-minute video, minutes for long ones, and a live stream never
played. Here a video starts while it is still arriving:

  * RangeProxy   a loopback HTTP server. googlevideo throttles any read longer than 10 MiB to about 1.5x real time (measured: ranges
                 of 10 MiB or less run at 30+ MB/s, one 20 MiB range or an open-ended one at 0.4 MB/s), which is exactly why yt-dlp
                 reads in 10 MiB chunks. ffmpeg cannot do that by itself, so it reads this proxy, and the proxy asks YouTube for
                 consecutive <= 10 MiB ranges. The proxy only serves the URLs registered for one download, on 127.0.0.1, behind a
                 random path token, and only to googlevideo hosts over https.
  * Streamer     one `ffmpeg -c copy` (no transcoding) that remuxes the video and audio inputs into ONE growing Matroska file at
                 network speed. The segment producer of relay/tv.py plays from that file while it grows.
  * MatroskaTail how many seconds of the growing file are really readable: an incremental reader of the file's own clusters (the
                 smaller of the video and the audio track's newest block). The producer never starts a segment past this point.
  * LiveHls      for a live stream: ONE ffmpeg reads the YouTube live HLS and writes a short-segment sliding HLS playlist into a
                 per-session folder; relay/tv.py serves it through the signed /tv-media/<id>/ URLs.
"""
from __future__ import annotations

import collections
import http.client
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin, urlsplit

CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CHUNK_BYTES = 10 * 1024 * 1024               # the largest range googlevideo serves at full speed (what yt-dlp's http_chunk_size is)
UPSTREAM_HOSTS = (".googlevideo.com",)       # nothing else is ever fetched by the proxy
READABLE_MARGIN = 0.5                        # seconds past a segment's end that must be readable (B-frame reordering, audio frame)
RETRIES = 5


def host_allowed(host):
    host = str(host or "").lower()
    return any(host.endswith(suffix) for suffix in UPSTREAM_HOSTS)


_CHILD_JOB = None


def _child_job():
    """One Windows Job Object (KILL_ON_JOB_CLOSE) for every ffmpeg this module starts: if the server is killed, the OS ends them too,
    so an orphaned live encode can never keep running. None off Windows or when the job cannot be made."""
    global _CHILD_JOB
    if _CHILD_JOB is None:
        _CHILD_JOB = False
        if os.name == "nt":
            try:
                from .app_runtime.winjob import KillOnCloseJob
                _CHILD_JOB = KillOnCloseJob()
            except Exception:
                logging.warning("YouTube relay: no Windows Job Object for its ffmpeg children", exc_info=True)
    return _CHILD_JOB or None


def spawn_background(command, stderr=subprocess.PIPE):
    """A hidden, below-normal-priority child with no stdin/stdout (ffmpeg), tied to the server's lifetime. stderr is a pipe unless told otherwise."""
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": stderr}
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
    proc = subprocess.Popen([str(part) for part in command], **kwargs)
    job = _child_job()
    if job is not None:
        try:
            job.assign(proc)
        except OSError:                                      # already in a job that forbids nesting: the explicit stop paths still apply
            pass
    return proc


def kill_tree(proc):
    """Stop a child and everything it started (idempotent; never raises)."""
    try:
        if proc.poll() is not None:
            return
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


def _drain(proc, sink):
    """Keep the last stderr lines of a child (so a failure can be explained) and never let the pipe fill up."""
    def run():
        try:
            for line in iter(proc.stderr.readline, b""):
                text = line.decode("utf-8", "replace").strip()
                if text:
                    sink.append(text)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True, name="ffmpeg-stderr").start()


# ============================================================================================================================
# Loopback range proxy
# ============================================================================================================================
class Upstream:
    """One persistent HTTPS connection to googlevideo for ONE proxy request; `range(a, b)` yields the bytes of [a, b] inclusive."""

    def __init__(self, source, timeout=30):
        self.url, self.headers, self.timeout = str(source["url"]), dict(source.get("headers") or {}), timeout
        self.total = source.get("length")
        self.conn, self.target = None, ""

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = None

    def _open(self):
        parts = urlsplit(self.url)
        if parts.scheme != "https" or not host_allowed(parts.hostname):
            raise OSError("The stream address is not a YouTube media server")
        self.close()
        self.conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=self.timeout)
        self.target = parts.path + ("?" + parts.query if parts.query else "")

    def range(self, start, end):
        response = None
        for _ in range(4):                                   # a media server may redirect to a nearer cache node
            if self.conn is None:
                self._open()
            headers = {**self.headers, "Range": "bytes=%d-%d" % (start, end), "Accept-Encoding": "identity"}
            try:
                self.conn.request("GET", self.target, headers=headers)
                response = self.conn.getresponse()
            except (OSError, http.client.HTTPException) as error:
                self.close()
                raise OSError(str(error) or error.__class__.__name__)
            if response.status in (301, 302, 303, 307, 308) and response.getheader("Location"):
                self.url = urljoin(self.url, response.getheader("Location"))
                response.read()
                self.close()
                continue
            break
        if response is None or response.status not in (200, 206) or (response.status == 200 and start > 0):
            status = getattr(response, "status", 0)
            self.close()
            raise OSError("YouTube answered HTTP %s" % status)
        found = re.fullmatch(r"bytes \d+-\d+/(\d+)", response.getheader("Content-Range") or "")
        if found:
            self.total = int(found.group(1))
        elif response.status == 200 and response.getheader("Content-Length"):
            self.total = int(response.getheader("Content-Length"))
        remaining = end - start + 1
        try:
            while remaining > 0:
                piece = response.read(min(65536, remaining))
                if not piece:
                    break
                remaining -= len(piece)
                yield piece
        except (OSError, http.client.HTTPException) as error:
            self.close()
            raise OSError(str(error) or error.__class__.__name__)
        if remaining > 0:
            self.close()
            raise OSError("YouTube closed the connection early")

    def size(self):
        if self.total is None:
            for _ in self.range(0, 0):
                pass
        return int(self.total)


def parse_range(header, total):
    """(start, end, partial) for a Range header over `total` bytes, or ValueError when it cannot be satisfied."""
    if not header:
        return 0, total - 1, False
    found = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not found or (not found.group(1) and not found.group(2)):
        raise ValueError("bad range")
    if not found.group(1):                                   # the last n bytes
        count = min(int(found.group(2)), total)
        return total - count, total - 1, True
    start = int(found.group(1))
    end = total - 1 if not found.group(2) else min(int(found.group(2)), total - 1)
    if start >= total or end < start:
        raise ValueError("range not satisfiable")
    return start, end, True


class RangeProxy:
    """See the module doc. `upstream` builds the connection object (tests pass a fake with `range(a, b)`, `size()` and `close()`)."""

    def __init__(self, upstream=Upstream, chunk=CHUNK_BYTES):
        self.token = secrets.token_urlsafe(12)
        self.sources, self._upstream, self.chunk = {}, upstream, int(chunk)
        self.closed = threading.Event()
        self.server, self.port, self.bytes_served = None, 0, 0

    def add(self, name, url, headers=None, length=None, content_type="application/octet-stream"):
        self.sources[str(name)] = {"url": url, "headers": headers or {}, "length": length, "type": content_type}

    def url_for(self, name):
        return "http://127.0.0.1:%d/%s/%s" % (self.port, self.token, name)

    def start(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):                    # no request log: the paths carry a token
                pass

            def do_HEAD(self):
                proxy._serve(self, False)

            def do_GET(self):
                proxy._serve(self, True)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 16
        self.server = Server(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True, name="youtube-range-proxy").start()
        return self

    def stop(self):
        self.closed.set()
        server, self.server = self.server, None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass

    def _fail(self, handler, code):
        try:
            handler.send_response(code)
            handler.send_header("Content-Length", "0")
            handler.send_header("Connection", "close")
            handler.end_headers()
        except OSError:
            pass
        handler.close_connection = True

    def _serve(self, handler, body):
        prefix = "/%s/" % self.token
        path = handler.path.split("?", 1)[0]
        source = self.sources.get(path[len(prefix):]) if path.startswith(prefix) else None
        if source is None:
            return self._fail(handler, 404)
        upstream = self._upstream(source)
        try:
            total = int(source.get("length") or 0) or upstream.size()
            try:
                start, end, partial = parse_range(handler.headers.get("Range"), total)
            except ValueError:
                handler.send_response(416)
                handler.send_header("Content-Range", "bytes */%d" % total)
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
            handler.send_response(206 if partial else 200)
            handler.send_header("Content-Type", source["type"])
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Content-Length", str(end - start + 1))
            if partial:
                handler.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, total))
            handler.end_headers()
            if not body:
                return
            position, failures = start, 0
            while position <= end and not self.closed.is_set():
                stop = min(position + self.chunk - 1, end)
                before = position
                try:
                    for piece in upstream.range(position, stop):
                        try:
                            handler.wfile.write(piece)
                        except OSError:                       # ffmpeg went away (stopped, or finished): nothing more to do
                            return
                        position += len(piece)
                        self.bytes_served += len(piece)
                except OSError as error:
                    upstream.close()
                    failures = failures + 1 if position == before else 1
                    if failures > RETRIES:
                        logging.warning("YouTube range proxy gave up: %s", error)
                        handler.close_connection = True
                        return
                    self.closed.wait(min(3.0, 0.4 * failures))
        except OSError as error:
            logging.warning("YouTube range proxy: %s", error)
            handler.close_connection = True
        finally:
            upstream.close()


# ============================================================================================================================
# Readable seconds of a growing Matroska file
# ============================================================================================================================
_SEGMENT, _INFO, _CLUSTER, _BLOCKGROUP = 0x18538067, 0x1549A966, 0x1F43B675, 0xA0
_TIMECODE, _SIMPLEBLOCK, _BLOCK, _TIMESCALE = 0xE7, 0xA3, 0xA1, 0x2AD7B1
_CONTAINERS = (_SEGMENT, _INFO, _CLUSTER, _BLOCKGROUP)       # parsed flat: their children follow the header directly


def _read_id(data, pos):
    """(element id, length) or None when the buffer ends inside the id."""
    if pos >= len(data):
        return None
    first = data[pos]
    length = 1
    mask = 0x80
    while length <= 4 and not first & mask:
        length += 1
        mask >>= 1
    if length > 4:
        raise ValueError("bad element id")
    if pos + length > len(data):
        return None
    return int.from_bytes(data[pos:pos + length], "big"), length


def _read_size(data, pos):
    """(size or -1 for unknown, length) or None when the buffer ends inside the number."""
    if pos >= len(data):
        return None
    first = data[pos]
    length = 1
    mask = 0x80
    while length <= 8 and not first & mask:
        length += 1
        mask >>= 1
    if length > 8:
        raise ValueError("bad element size")
    if pos + length > len(data):
        return None
    value = first & (mask - 1)
    for byte in data[pos + 1:pos + length]:
        value = (value << 8) | byte
    return (-1 if value == (1 << (7 * length)) - 1 else value), length


class MatroskaTail:
    """Reads, incrementally, a Matroska file that ffmpeg is still writing, and says how many seconds of EVERY track are complete in it.

    `poll()` returns the smaller of the newest block time of each wanted track (video is track 1 and audio track 2 in the file the
    Streamer writes). A block counts only when all of its bytes are in the file. The cluster layout is parsed flat (Segment, Cluster
    and BlockGroup are containers), so unknown sizes of the still-open Segment and Cluster do not matter."""

    def __init__(self, path, tracks=(1, 2)):
        self.path, self.tracks = Path(path), tuple(tracks)
        self.cursor, self.cluster_time, self.scale_ms = 0, 0, 1.0
        self.last, self.readable, self.broken = {}, 0.0, False

    def poll(self):
        if self.broken:
            return self.readable
        try:
            size = self.path.stat().st_size
            if size <= self.cursor:
                return self.readable
            with open(self.path, "rb") as handle:
                handle.seek(self.cursor)
                data = handle.read(min(size - self.cursor, 16 * 1024 * 1024))
        except OSError:
            return self.readable
        pos = 0
        try:
            while True:
                ident = _read_id(data, pos)
                if ident is None:
                    break
                element, id_len = ident
                sized = _read_size(data, pos + id_len)
                if sized is None:
                    break
                size, size_len = sized
                head = id_len + size_len
                if element in _CONTAINERS:
                    pos += head
                    continue
                if size < 0 or size > 256 * 1024 * 1024:
                    raise ValueError("unusable element")
                if pos + head + size > len(data):
                    break                                     # not completely written yet (or beyond this read window)
                payload = data[pos + head:pos + head + size]
                if element == _TIMECODE:
                    self.cluster_time = int.from_bytes(payload, "big")
                elif element == _TIMESCALE:
                    self.scale_ms = int.from_bytes(payload, "big") / 1e6
                elif element in (_SIMPLEBLOCK, _BLOCK) and len(payload) >= 4:
                    number = _read_size(payload, 0)
                    if number is not None:
                        track, used = number[0], number[1]
                        if len(payload) >= used + 2:
                            relative = int.from_bytes(payload[used:used + 2], "big", signed=True)
                            moment = (self.cluster_time + relative) * self.scale_ms
                            if moment > self.last.get(track, -1.0):
                                self.last[track] = moment
                pos += head + size
        except ValueError:
            self.broken = True                                # not a file this reader understands: keep the last answer
        self.cursor += pos
        if all(track in self.last for track in self.tracks):
            self.readable = max(self.readable, min(self.last[track] for track in self.tracks) / 1000.0)
        return self.readable


# ============================================================================================================================
# ffmpeg command lines
# ============================================================================================================================
def remux_command(ffmpeg, urls, out_path):
    """Copy (never transcode) the video and audio inputs into one Matroska file that is readable while it grows."""
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-probesize", "2M", "-analyzeduration", "1000000"]
    for url in urls:
        command += ["-i", url]
    if len(urls) >= 2:
        command += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        command += ["-map", "0:v:0", "-map", "0:a:0"]
    command += ["-c", "copy", "-sn", "-dn", "-max_interleave_delta", "2000000", "-flush_packets", "1",
                "-cluster_time_limit", "2000", "-cluster_size_limit", "2000000", "-f", "matroska", "-y", str(out_path)]
    return command


def live_command(ffmpeg, urls, folder, codec, codec_options, *, threads, segment_seconds, list_size, height, maxrate, bufsize):
    """ONE ffmpeg: the live HLS in, a TV-compatible sliding HLS (H.264 + AAC stereo, `segment_seconds` segments) out."""
    folder = Path(folder)
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-rw_timeout", "15000000"]
    for url in urls:
        command += ["-i", url]
    command += ["-map", "0:v:0", "-map", "1:a:0" if len(urls) >= 2 else "0:a:0"]
    width = int(height) * 16 // 9
    scale = ("scale=w='min(%d,iw)':h='min(%d,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2:flags=bilinear,setsar=1"
             % (width, int(height)))
    command += ["-vf", scale, "-c:v", codec, *codec_options, "-threads", str(int(threads)), "-pix_fmt", "yuv420p",
                "-maxrate", str(maxrate), "-bufsize", str(bufsize),
                "-force_key_frames", "expr:gte(t,n_forced*%g)" % float(segment_seconds),
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-sn", "-dn",
                "-f", "hls", "-hls_time", "%g" % float(segment_seconds), "-hls_list_size", str(int(list_size)),
                "-hls_delete_threshold", "3", "-start_number", "0",
                "-hls_flags", "delete_segments+independent_segments+omit_endlist+temp_file",
                "-hls_segment_type", "mpegts", "-hls_segment_filename", str(folder / "seg%06d.ts"), "-y", str(folder / "live.m3u8")]
    return command


def parse_hls_playlist(text):
    """{'sequence': first media sequence, 'target': target duration, 'segments': [(name, seconds), ...]} of an HLS media playlist."""
    sequence, target, segments, pending = 0, 0, [], None
    for line in str(text).splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            sequence = int(line.split(":", 1)[1] or 0)
        elif line.startswith("#EXT-X-TARGETDURATION:"):
            target = int(float(line.split(":", 1)[1] or 0))
        elif line.startswith("#EXTINF:"):
            pending = float(line.split(":", 1)[1].split(",", 1)[0])
        elif line and not line.startswith("#") and pending is not None:
            segments.append((line, pending))
            pending = None
    return {"sequence": sequence, "target": target, "segments": segments}


# ============================================================================================================================
# The two supervisors
# ============================================================================================================================
class Streamer:
    """Copy-remux of one video's inputs into the growing file `out_path` (see the module doc).

    `inputs` are dicts {url, headers, length, kind}; `popen(command)` starts the hidden child (tests pass a stand-in). After `start()`:
    `readable` (seconds complete in the file), `complete` (ffmpeg finished cleanly), `failed` (a reason, or ''), `stop()`."""

    def __init__(self, ffmpeg, inputs, out_path, popen=spawn_background, notify=None, proxy=None):
        self.ffmpeg, self.inputs, self.out_path = ffmpeg, list(inputs), Path(out_path)
        self.popen, self.notify, self.proxy = popen, notify or (lambda: None), proxy
        self.proc, self.readable, self.complete, self.failed = None, 0.0, False, ""
        self.stopped, self.done, self.log = threading.Event(), threading.Event(), collections.deque(maxlen=30)
        self.started_at = 0.0

    def start(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.unlink(missing_ok=True)
        self.proxy = self.proxy or RangeProxy()
        self.proxy.start()
        urls = []
        for number, item in enumerate(self.inputs):
            name = "input%d" % number
            self.proxy.add(name, item["url"], item.get("headers"), item.get("length"))
            urls.append(self.proxy.url_for(name))
        self.started_at = time.monotonic()
        self.proc = self.popen(remux_command(self.ffmpeg, urls, self.out_path))
        if getattr(self.proc, "stderr", None) is not None:
            _drain(self.proc, self.log)
        threading.Thread(target=self._watch, daemon=True, name="youtube-remux").start()
        return self

    def _watch(self):
        tail = MatroskaTail(self.out_path)
        try:
            while True:
                running = self.proc.poll() is None
                value = tail.poll()
                if value != self.readable:
                    self.readable = value
                    self.notify()
                if not running:
                    break
                self.stopped.wait(0.15)
            self.readable = max(self.readable, tail.poll())
            code = self.proc.returncode
            if self.stopped.is_set():
                self.failed = "stopped"
            elif code != 0:
                self.failed = (self.log[-1] if self.log else "ffmpeg exited with code %s" % code)[:200]
            else:
                self.complete = True
        except Exception as error:                                  # never leave the job waiting for ever
            logging.exception("YouTube remux watcher failed")
            self.failed = str(error)[:200] or error.__class__.__name__
        finally:
            if self.proxy is not None:
                self.proxy.stop()
            self.done.set()
            self.notify()

    def stop(self):
        self.stopped.set()
        if self.proc is not None:
            kill_tree(self.proc)
        if self.proxy is not None:
            self.proxy.stop()

    def view(self):
        return {"readable": self.readable, "complete": self.complete, "failed": self.failed}


class LiveHls:
    """The per-session live ffmpeg and its output folder. `command` is built by the caller (so it can choose the encoder)."""

    def __init__(self, folder, command, popen=spawn_background, segment_seconds=2.0):
        self.folder, self.command, self.popen, self.segment_seconds = Path(folder), list(command), popen, float(segment_seconds)
        self.proc, self.log, self.started_at = None, collections.deque(maxlen=30), 0.0

    def start(self):
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.started_at = time.monotonic()
        self.proc = self.popen(self.command)
        if getattr(self.proc, "stderr", None) is not None:
            _drain(self.proc, self.log)
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def returncode(self):
        return None if self.proc is None else self.proc.poll()

    def reason(self):
        return (self.log[-1] if self.log else "ffmpeg exited")[:200]

    def segments(self):
        """[(sequence, seconds, path)] of the finished segments the playlist lists right now (oldest first)."""
        try:
            text = (self.folder / "live.m3u8").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return [], 0
        parsed = parse_hls_playlist(text)
        out = []
        for offset, (name, seconds) in enumerate(parsed["segments"]):
            path = self.folder / Path(name).name
            if path.suffix == ".ts" and path.is_file():
                out.append((parsed["sequence"] + offset, seconds, path))
        return out, parsed["target"]

    def stop(self, remove=True):
        if self.proc is not None:
            kill_tree(self.proc)
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
        if remove:
            shutil.rmtree(self.folder, ignore_errors=True)
