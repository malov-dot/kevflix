from __future__ import annotations

import hashlib
import hmac
import math
import struct
import time
import wave
from urllib.parse import urlencode

from .media import RATE, CHANNELS, BYTES_PER_FRAME


def wave_header(frames):
    size = frames * BYTES_PER_FRAME
    fmt = struct.pack("<HHIIHH", 1, CHANNELS, RATE, RATE * BYTES_PER_FRAME,
                      BYTES_PER_FRAME, 16)
    if size + 36 <= 0xFFFFFFFF:
        return b"RIFF" + struct.pack("<I", size + 36) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", size)
    # RF64 keeps finite duration and seek support for tracks beyond the RIFF size limit.
    ds64 = struct.pack("<QQQI", size + 72, size, frames, 0)
    return b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEds64" + struct.pack("<I", 28) + ds64 + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", 0xFFFFFFFF)


def ticket(config, job_id, revision, expires=None):
    expires = int(expires or time.time() + 12 * 3600)
    value = f"{job_id}:{revision}:{expires}"
    signature = hmac.new(config["stream_secret"].encode(), value.encode(), hashlib.sha256).hexdigest()
    return urlencode({"revision": revision, "expires": expires, "signature": signature})


def valid_ticket(config, job_id, values):
    try:
        revision, expires = int(values["revision"]), int(values["expires"])
        if expires < time.time() or expires > time.time() + 24 * 3600:
            return False
        value = f"{job_id}:{revision}:{expires}"
        expected = hmac.new(config["stream_secret"].encode(), value.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, values["signature"])
    except (KeyError, TypeError, ValueError):
        return False


def byte_range(header, total):
    if not header:
        return 0, total - 1, False
    if not header.startswith("bytes=") or "," in header:
        raise ValueError("Only one byte range is supported")
    left, right = header[6:].split("-", 1)
    if left:
        start = int(left)
        end = min(total - 1, int(right)) if right else total - 1
    else:
        suffix = int(right)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        start, end = max(0, total - suffix), total - 1
    if start < 0 or start >= total or end < start:
        raise ValueError("Range lies outside the audio")
    return start, end, True


class VirtualAudio:
    """A finite, seekable PCM file assembled from validated chunks on demand."""
    def __init__(self, coordinator, job_id, revision):
        self.coordinator, self.job_id, self.revision = coordinator, job_id, revision
        job = coordinator.store.job(job_id)
        if not job or job["payload"].get("revision", 1) != revision:
            raise ValueError("Unknown audio revision")
        self.spec = job["payload"]
        self.frames = round(self.spec["duration"] * RATE)
        self.header = wave_header(self.frames)
        self.length = len(self.header) + self.frames * BYTES_PER_FRAME
        self.segment_frames = self.spec["segment_seconds"] * RATE

    def blocks(self, start, end):
        position = start
        if position < len(self.header):
            last = min(end + 1, len(self.header))
            yield self.header[position:last]
            position = last
        while position <= end:
            pcm_offset = position - len(self.header)
            frame, sub_byte = divmod(pcm_offset, BYTES_PER_FRAME)
            index, in_chunk = divmod(frame, self.segment_frames)
            path = self.coordinator.audio_chunk(self.job_id, index, self.revision)
            with wave.open(str(path), "rb") as stream:
                stream.setpos(in_chunk)
                remain_in_chunk = (stream.getnframes() - in_chunk) * BYTES_PER_FRAME - sub_byte
                wanted = min(end - position + 1, remain_in_chunk, 256 * 1024)
                if wanted <= 0:
                    raise ValueError("Generated chunk is shorter than its advertised timeline")
                data = stream.readframes(math.ceil((wanted + sub_byte) / BYTES_PER_FRAME))
                data = data[sub_byte:sub_byte + wanted]
                if len(data) != wanted:
                    raise ValueError("Generated audio ended unexpectedly")
                yield data
                position += wanted
