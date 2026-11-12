"""YouTube menu as low-latency video (F20). The screen of the background YouTube page (relay/youtube_host.py) is encoded as
H.264 on the PC and pushed to the TV over ONE WebSocket, which also carries the remote keys back: a key press and the
picture it causes travel on a connection that is already open.

    Chrome screencast (JPEG, a frame only when the page changes)
      -> one ffmpeg child: MPJPEG parts on stdin, libx264 ultrafast zerolatency on the CPU, FLV on stdout
         (measured with ffmpeg 9.0.2: one coded frame out per JPEG in, 4-6 ms later, nothing held back)
      -> FLV tags -> binary WebSocket messages: format "avc" (for the TV's WebCodecs decoder) or "fmp4" (Media Source)
      <- JSON text messages from the TV: key, reset, hb

Standard library only. One WebSocket = one viewer = one encoder process (BELOW_NORMAL priority, no window, never the
graphics card unless youtube.video_encoder says nvenc). While the page is still Chrome sends nothing and the encoder idles.
"""
from __future__ import annotations

import base64
import collections
import hashlib
import json
import logging
import os
import struct
import subprocess
import threading
import time

from .youtube_host import BELOW_NORMAL_PRIORITY_CLASS, CREATE_NO_WINDOW, mjpeg_part, video_size

WS_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0, 1, 2, 8, 9, 10
MAX_CLIENT_MESSAGE = 64 * 1024        # a TV sends keys and heartbeats: a few dozen bytes each
SOCKET_TIMEOUT = 20.0                 # the TV sends a heartbeat every 5 s; a silent or stuck TV is dropped after this
SEND_BUFFER = 256 * 1024              # what may wait in the PC's socket for a slow link (the encoder is held back beyond it)
FORMATS = ("avc", "fmp4")
KIND_INIT, KIND_FRAME, FLAG_KEY = 1, 2, 1
HEADER = struct.Struct(">BBHI")       # kind, flags, stream epoch, frame number: in front of every binary message
START_LIMIT = 3                       # encoder processes in a row that died without one coded frame: give up, the TV falls back
POPEN = subprocess.Popen              # replaced by the tests


class ProtocolError(Exception):
    """The peer broke the WebSocket protocol, or sent more than a TV ever sends."""


# ---- WebSocket (RFC 6455), server side ---------------------------------------------------------------------------------
def accept_key(key):
    return base64.b64encode(hashlib.sha1(str(key).strip().encode("ascii", "replace") + WS_GUID).digest()).decode("ascii")


def ws_frame(opcode, payload=b""):
    """One complete, unmasked server-to-client frame."""
    size = len(payload)
    if size < 126:
        head = struct.pack(">BB", 0x80 | opcode, size)
    elif size < 65536:
        head = struct.pack(">BBH", 0x80 | opcode, 126, size)
    else:
        head = struct.pack(">BBQ", 0x80 | opcode, 127, size)
    return head + payload


def read_exact(read, size):
    data = b""
    while len(data) < size:
        block = read(size - len(data))
        if not block:
            raise ConnectionError("the stream ended")
        data += block
    return data


class MessageReader:
    """Client messages, one per next() -> (opcode, payload). A control frame (close, ping, pong) that arrives between the
    fragments of a data message is returned first; the data message follows when its last fragment has come.
    `read(n)` returns up to n bytes and b"" at the end of the stream."""

    def __init__(self, read):
        self._read, self._opcode, self._parts, self._total = read, None, [], 0

    def next(self):
        read = self._read
        while True:
            first, second = read_exact(read, 2)
            code, size = first & 0x0F, second & 0x7F
            if first & 0x70 or not second & 0x80:
                raise ProtocolError("reserved bits set, or a client frame without a mask")
            if size == 126:
                size = struct.unpack(">H", read_exact(read, 2))[0]
            elif size == 127:
                size = struct.unpack(">Q", read_exact(read, 8))[0]
            if self._total + size > MAX_CLIENT_MESSAGE:
                raise ProtocolError("message too large")
            mask = read_exact(read, 4)
            data = bytes(value ^ mask[index & 3] for index, value in enumerate(read_exact(read, size)))
            if code >= 8:
                if not first & 0x80 or size > 125:
                    raise ProtocolError("bad control frame")
                return code, data
            if (code == OP_CONT) == (self._opcode is None):
                raise ProtocolError("fragments out of order")
            self._opcode = code if self._opcode is None else self._opcode
            self._parts.append(data)
            self._total += size
            if first & 0x80:
                message = (self._opcode, b"".join(self._parts))
                self._opcode, self._parts, self._total = None, [], 0
                return message


# ---- what ffmpeg writes: FLV tags ----------------------------------------------------------------------------------------
def read_flv(read):
    """Yield ("config", avcC record) and ("frame", is_key, sample) from an FLV stream of H.264. A sample is the frame in
    MP4 form (each NAL unit behind its 4-byte length), complete the moment its tag has been read: nothing waits for the
    next frame. Ends when the stream ends."""
    try:
        head = read_exact(read, 9)
        if head[:3] != b"FLV":
            raise ValueError("the encoder did not write FLV")
        read_exact(read, struct.unpack(">I", head[5:9])[0] - 9 + 4)
        while True:
            tag = read_exact(read, 11)
            body = read_exact(read, int.from_bytes(tag[1:4], "big"))
            read_exact(read, 4)
            if tag[0] != 9 or len(body) < 5 or body[0] & 0x0F != 7:
                continue                                        # only H.264 video tags matter
            if body[1] == 0:
                yield ("config", body[5:])
            elif body[1] == 1:
                yield ("frame", body[0] >> 4 == 1, body[5:])
    except ConnectionError:
        return


def codec_string(avcc):
    """RFC 6381 name of the stream from its avcC record, for example avc1.42c028."""
    return "avc1.%02x%02x%02x" % (avcc[1], avcc[2], avcc[3])


# ---- fragmented MP4 for the Media Source path: one frame per fragment ---------------------------------------------------
FRAME_TICKS = 1000
_MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


def _box(kind, *parts):
    body = b"".join(parts)
    return struct.pack(">I", 8 + len(body)) + kind + body


def _full(kind, version, flags, *parts):
    return _box(kind, struct.pack(">I", version << 24 | flags), *parts)


def init_segment(avcc, width, height, fps):
    """ftyp + moov of one H.264 track without samples and without a duration (a live stream). Timescale fps * 1000: every
    frame lasts FRAME_TICKS."""
    entry = _box(b"avc1", bytes(6), struct.pack(">H", 1), bytes(16), struct.pack(">HH", width, height),
                 struct.pack(">II", 0x480000, 0x480000), bytes(4), struct.pack(">H", 1), bytes(32),
                 struct.pack(">Hh", 0x18, -1), _box(b"avcC", avcc))
    stbl = _box(b"stbl", _full(b"stsd", 0, 0, struct.pack(">I", 1), entry), _full(b"stts", 0, 0, bytes(4)),
                _full(b"stsc", 0, 0, bytes(4)), _full(b"stsz", 0, 0, bytes(8)), _full(b"stco", 0, 0, bytes(4)))
    minf = _box(b"minf", _full(b"vmhd", 0, 1, bytes(8)),
                _box(b"dinf", _full(b"dref", 0, 0, struct.pack(">I", 1), _full(b"url ", 0, 1))), stbl)
    mdia = _box(b"mdia", _full(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, int(fps) * 1000, 0, 0x55C4, 0)),
                _full(b"hdlr", 0, 0, bytes(4), b"vide", bytes(12), b"KevFLIX\x00"), minf)
    trak = _box(b"trak", _full(b"tkhd", 0, 3, struct.pack(">IIIII", 0, 0, 1, 0, 0), bytes(8), bytes(8), _MATRIX,
                                struct.pack(">II", width << 16, height << 16)), mdia)
    mvhd = _full(b"mvhd", 0, 0, struct.pack(">IIII", 0, 0, int(fps) * 1000, 0), struct.pack(">IH", 0x10000, 0x100),
                 bytes(10), _MATRIX, bytes(24), struct.pack(">I", 2))
    mvex = _box(b"mvex", _full(b"trex", 0, 0, struct.pack(">IIIII", 1, 1, FRAME_TICKS, 0, 0x01010000)))
    return (_box(b"ftyp", b"isom", struct.pack(">I", 0x200), b"isom", b"iso6", b"avc1", b"mp41")
            + _box(b"moov", mvhd, trak, mvex))


def media_segment(number, sample, key):
    """moof + mdat holding ONE frame. Its decode time is number * FRAME_TICKS: the timeline counts frames, not seconds, so it
    has no hole while the page is still (the TV's player then simply waits at the end of what it has)."""
    def moof(offset):
        trun = _full(b"trun", 0, 0x000001 | 0x000200 | 0x000400,
                     struct.pack(">IiII", 1, offset, len(sample), 0x02000000 if key else 0x01010000))
        traf = _box(b"traf", _full(b"tfhd", 0, 0x020000, struct.pack(">I", 1)),
                    _full(b"tfdt", 1, 0, struct.pack(">Q", number * FRAME_TICKS)), trun)
        return _box(b"moof", _full(b"mfhd", 0, 0, struct.pack(">I", number + 1)), traf)
    return moof(len(moof(0)) + 8) + _box(b"mdat", sample)


# ---- the encoder ---------------------------------------------------------------------------------------------------------
def encoder_command(ffmpeg, settings, fmt):
    """ffmpeg arguments: MPJPEG parts on stdin -> H.264 in FLV on stdout. No B-frames and no look-ahead, so each picture
    comes out as soon as it went in. (-fflags nobuffer must NOT be added: it drops the first picture.) Every picture is
    scaled to one fixed size (video_size), so a picture of another size never restarts the encoder, and converted from
    full-range JPEG to TV-range BT.709, which every hardware decoder shows correctly."""
    width, height = video_size(settings)
    fps = int(settings.get("video_fps", 30))
    rate = int(settings.get("video_maxrate_kbps", 12000))
    quality = str(int(settings.get("video_crf", 23)))
    # Media Source can only drop old pictures up to a key frame, so it gets one every two seconds of pictures; the
    # WebCodecs decoder needs only the first (a lost or broken stream is restarted, which begins with a key frame).
    gop = fps * 2 if fmt == "fmp4" else 100000
    if str(settings.get("video_encoder", "cpu")) == "nvenc":
        codec = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", "-zerolatency", "1", "-delay", "0", "-bf", "0",
                 "-rc", "vbr", "-cq", quality, "-b:v", "0"]
    else:
        codec = ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                 "-threads", str(int(settings.get("video_threads", 4))), "-crf", quality]
    return [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-probesize", "32", "-analyzeduration", "0",
            "-f", "mpjpeg", "-r", str(fps), "-i", "pipe:0", "-an",
            "-vf", "scale=%d:%d:in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p" % (width, height),
            *codec, "-maxrate", "%dk" % rate, "-bufsize", "%dk" % (rate // 2), "-g", str(gop), "-fps_mode", "passthrough",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-flush_packets", "1", "-f", "flv", "pipe:1"]


class ScreenEncoder:
    """One ffmpeg child. write(jpeg) feeds a picture; on_config(avcC) and on_frame(is_key, sample, milliseconds since that
    picture was written) are called from this object's reader thread, in order."""

    def __init__(self, command, size, on_config, on_frame):
        flags = (BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW) if os.name == "nt" else 0
        self.proc = POPEN(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        self.size, self.frames, self.error = size, 0, ""
        self._on_config, self._on_frame, self._written = on_config, on_frame, collections.deque()
        self._threads = [threading.Thread(target=self._read, daemon=True, name="youtube-video-out"),
                         threading.Thread(target=self._errors, daemon=True, name="youtube-video-err")]
        for thread in self._threads:
            thread.start()

    def write(self, jpeg):
        self._written.append(time.perf_counter())
        self.proc.stdin.write(mjpeg_part(jpeg))
        self.proc.stdin.flush()

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        for step in (self.proc.stdin.close, self.proc.kill):
            try:
                step()
            except OSError:
                pass
        try:
            self.proc.wait(5)
        except Exception:
            pass
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(1)
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except Exception:
                pass

    def _read(self):
        try:
            for item in read_flv(self.proc.stdout.read):
                if item[0] == "config":
                    self._on_config(self, item[1])
                    continue
                self.frames += 1
                wrote = self._written.popleft() if self._written else time.perf_counter()
                self._on_frame(item[1], item[2], (time.perf_counter() - wrote) * 1000.0)
        except Exception as error:                                # a closed socket or a closed pipe ends this stream
            self.error = self.error or str(error)

    def _errors(self):
        try:
            for line in self.proc.stderr:
                self.error = (self.error + " " + line.decode("utf-8", "replace").strip())[-300:].strip()
        except Exception:
            pass


# ---- one viewer ----------------------------------------------------------------------------------------------------------
class VideoSession:
    """One TV on the WebSocket: pumps the page's pictures into an encoder, sends what the encoder makes, and takes the keys.
    `send(bytes)` writes to the socket (it may block on a slow link: the encoder and the pump then wait, so the newest
    picture is always the next one encoded and nothing queues beyond the socket's buffer)."""

    def __init__(self, host, send, fmt, settings, ffmpeg, on_play=None):
        self.host, self._send, self.fmt, self.settings, self.ffmpeg, self.on_play = host, send, fmt, settings, ffmpeg, on_play
        self._lock = threading.Lock()                             # the socket has one writer at a time
        self._stop, self._restart = threading.Event(), threading.Event()
        self.epoch, self.encoder, self.starts, self.error = 0, None, 0, ""
        self._number, self._key_at = 0, None
        self._began = self._window = time.monotonic()
        self.total = {"frames": 0, "bytes": 0, "keys": 0}
        self._stats = self._fresh()

    @staticmethod
    def _fresh():
        return {"frames": 0, "bytes": 0, "encode": [], "page": []}

    # ---- to the TV ---------------------------------------------------------------------------------------------------
    def _write(self, data, epoch=None):
        with self._lock:
            if epoch is None or epoch == self.epoch:              # nothing of a stream that was restarted meanwhile
                try:
                    self._send(data)
                except OSError:
                    self._stop.set()                              # the TV is gone or stuck: end the session
                    raise

    def text(self, value):
        self._write(ws_frame(OP_TEXT, json.dumps(value, separators=(",", ":")).encode("utf-8")))

    def _binary(self, epoch, kind, flags, number, payload):
        self._write(ws_frame(OP_BINARY, HEADER.pack(kind, flags, epoch & 0xFFFF, number) + payload), epoch)

    def _config(self, epoch, encoder, avcc):
        width, height = encoder.size
        fps = int(self.settings.get("video_fps", 30))
        message = {"t": "config", "epoch": epoch, "format": self.fmt, "codec": codec_string(avcc), "width": width,
                   "height": height, "fps": fps}
        if self.fmt == "avc":
            message["description"] = base64.b64encode(avcc).decode("ascii")
        self._write(ws_frame(OP_TEXT, json.dumps(message, separators=(",", ":")).encode("utf-8")), epoch)
        if self.fmt == "fmp4":
            self._binary(epoch, KIND_INIT, 0, 0, init_segment(avcc, width, height, fps))

    def _frame(self, epoch, key, sample, encode_ms):
        if epoch != self.epoch:
            return
        number, self._number = self._number, self._number + 1
        payload = media_segment(number, sample, key) if self.fmt == "fmp4" else sample
        self._binary(epoch, KIND_FRAME, FLAG_KEY if key else 0, number, payload)
        stats = self._stats
        stats["frames"] += 1
        stats["bytes"] += len(payload)
        stats["encode"].append(encode_ms)
        self.total["frames"] += 1
        self.total["bytes"] += len(payload)
        now = time.monotonic()
        if now - self._window >= 10.0:
            self._stats, seconds, self._window = self._fresh(), now - self._window, now
            encode, page = sorted(stats["encode"]), sorted(stats["page"])
            logging.info("YouTube video (%s): %d frames in %.0f s (%.1f per second), %.1f KB each, encode %.0f ms median / %.0f ms worst, "
                         "key to next page picture %s", self.fmt, stats["frames"], seconds, stats["frames"] / seconds,
                         stats["bytes"] / stats["frames"] / 1024, encode[len(encode) // 2], encode[-1],
                         ("%.0f ms median / %.0f ms worst (%d keys)" % (page[len(page) // 2], page[-1], len(page))) if page else "(no key)")

    # ---- pictures in ---------------------------------------------------------------------------------------------------
    def _start_encoder(self):
        old, self.encoder = self.encoder, None
        if old is not None:
            old.close()
        with self._lock:
            self.epoch += 1
            epoch = self.epoch
        self._number, self.starts = 0, self.starts + 1
        self._restart.clear()
        self.encoder = ScreenEncoder(encoder_command(self.ffmpeg, self.settings, self.fmt), video_size(self.settings),
                                     lambda encoder, avcc: self._config(epoch, encoder, avcc),
                                     lambda key, sample, ms: self._frame(epoch, key, sample, ms))

    def _pump(self):
        last, barren = 0, 0
        try:
            while not self._stop.is_set() and self.host.state == "running":
                encoder = self.encoder
                if encoder is None or self._restart.is_set() or not encoder.alive():
                    if encoder is not None and not encoder.alive() and encoder.frames == 0:
                        barren += 1
                        self.error = encoder.error or "the video encoder stopped"
                        if barren >= START_LIMIT:
                            raise RuntimeError("The video encoder does not start: " + self.error)
                    elif encoder is not None and encoder.frames:
                        barren = 0
                    self._start_encoder()
                    last = 0                                      # the picture on screen now opens the new stream
                found = self.host.next_frame(last, 0.5)
                if found is None:
                    continue
                last, jpeg = found
                if self._key_at is not None:
                    self._stats["page"].append((time.perf_counter() - self._key_at) * 1000.0)
                    self._key_at = None
                try:
                    self.encoder.write(jpeg)
                except (OSError, ValueError):
                    time.sleep(0.05)                              # the encoder died: the next turn starts another
                    self._restart.set()
        except Exception as error:
            self.error = str(error)
            try:
                self.text({"t": "error", "error": self.error[:200]})
                self._write(ws_frame(OP_CLOSE, struct.pack(">H", 1011)))
            except OSError:
                pass
        finally:
            self._stop.set()

    # ---- from the TV ---------------------------------------------------------------------------------------------------
    def _message(self, message):
        kind = message.get("t")
        if kind == "key":
            self.total["keys"] += 1
            self._key_at = time.perf_counter()
            try:
                play = self.host.key(str(message.get("key")), str(message.get("action", "press")))
            except (KeyError, ValueError) as error:               # an unknown key name: say so, keep the stream
                self.text({"t": "error", "error": str(error)[:120], "key": True})
                return
            if play:
                self.text({"t": "play", "play": play})
                if self.on_play:
                    self.on_play(play["video_id"])
        elif kind == "reset":
            logging.info("YouTube video: the TV asked for a new stream (%s)", str(message.get("why", ""))[:60])
            self._restart.set()
        elif kind == "hb":
            self.text({"t": "hb", "n": message.get("n")})

    def run(self, read):
        """Serve until the TV closes the socket, the page's host stops, or the encoder cannot be started."""
        self.host.video_enter()
        pump = threading.Thread(target=self._pump, daemon=True, name="youtube-video-in")
        pump.start()
        reason, messages = "the TV closed it", MessageReader(read)
        try:
            while not self._stop.is_set():
                opcode, payload = messages.next()
                if opcode == OP_CLOSE:
                    break
                if opcode == OP_PING:
                    self._write(ws_frame(OP_PONG, payload))
                elif opcode == OP_TEXT:
                    message = json.loads(payload.decode("utf-8"))
                    if isinstance(message, dict):
                        self._message(message)
        except (ConnectionError, OSError, ProtocolError, ValueError, RuntimeError) as error:
            reason = str(error) or error.__class__.__name__
        finally:
            self._stop.set()
            try:
                self._write(ws_frame(OP_CLOSE, struct.pack(">H", 1000)))
            except OSError:
                pass
            pump.join(3)
            if self.encoder is not None:
                self.encoder.close()
            self.host.video_leave()
            logging.info("YouTube video (%s): closed after %.0f s (%s): %d frames, %.0f KB, %d keys, %d encoder starts",
                         self.fmt, time.monotonic() - self._began, self.error or reason, self.total["frames"],
                         self.total["bytes"] / 1024, self.total["keys"], self.starts)


def serve(host, connection, read, fmt, settings, ffmpeg, on_play=None):
    """Run one viewer on an upgraded connection (the 101 answer has been sent). Returns when the viewer is gone."""
    import socket
    try:
        connection.settimeout(SOCKET_TIMEOUT)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)       # small frames must not wait for more data
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER)
    except OSError:
        pass
    logging.info("YouTube video (%s): a TV connected, encoder %s", fmt, settings.get("video_encoder", "cpu"))
    VideoSession(host, connection.sendall, fmt, settings, ffmpeg, on_play).run(read)
