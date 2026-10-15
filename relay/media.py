from __future__ import annotations

import hashlib
import json
import math
import os
import re
import wave
from pathlib import Path

RATE = 48000
CHANNELS = 2
BYTES_PER_FRAME = CHANNELS * 2


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    info = path.stat()
    digest = hashlib.sha256()
    digest.update(str(path).encode())
    digest.update(f":{info.st_size}:{info.st_mtime_ns}:".encode())
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if info.st_size > 1024 * 1024:
            stream.seek(max(0, info.st_size - 1024 * 1024))
            digest.update(stream.read(1024 * 1024))
    return digest.hexdigest()


def probe(config, owner, path):
    data = json.loads(owner.run([config["ffprobe"], "-v", "error", "-show_format",
                                "-show_streams", "-of", "json", path], timeout=60))
    duration = float(data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Media has no finite positive duration")
    data["duration"] = duration
    data["audio"] = [s for s in data["streams"] if s.get("codec_type") == "audio"]
    if not data["audio"]:
        raise ValueError("Media contains no audio track")
    return data


def extract_audio(config, owner, source, audio_index, start, seconds, dest, rate=RATE, channels=CHANNELS):
    owner.run([config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{start:.6f}", "-i", source, "-t", f"{seconds:.6f}",
               "-map", f"0:{audio_index}", "-vn", "-sn", "-ac", channels, "-ar", rate,
               "-c:a", "pcm_s16le", "-y", dest], timeout=max(60, seconds * 4))


def wav_info(path):
    with wave.open(str(path), "rb") as w:
        return {"rate": w.getframerate(), "channels": w.getnchannels(),
                "width": w.getsampwidth(), "frames": w.getnframes(),
                "seconds": w.getnframes() / w.getframerate()}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def find_sidecar(source):
    source = Path(source)
    candidates = [p for p in source.parent.iterdir() if p.is_file()
                  and p.suffix.lower() in (".srt", ".ass", ".ssa") and
                  (p.stem == source.stem or p.stem.startswith(source.stem + "."))]
    candidates.sort(key=lambda p: (not bool(re.search(r"\.(en|eng)(\.|$)", p.name, re.I)),
                                   p.name.lower()))
    return candidates[0] if candidates else None


def subtitle_cues(config, owner, source, explicit=None, stream_index=None, discover=True):
    """Parse bounded text captions outside Electron's UI thread."""
    source = Path(source)
    picked = Path(explicit) if explicit else find_sidecar(source) if discover and stream_index is None else None
    if picked:
        picked = config.media_path(picked)
        import pysubs2
        parsed = pysubs2.load(str(picked), encoding="utf-8")
        suffix = picked.stem[len(source.stem):] if picked.stem.startswith(source.stem) else ""
        lang_match = re.search(r"^\.([a-z]{2,3})(?:\.|$)", suffix, re.I)
        language = lang_match.group(1).lower() if lang_match else None
        return [{"start": x.start / 1000, "end": x.end / 1000,
                 "text": x.plaintext.strip(), "actor": x.name or None,
                 "language": language}
                for x in parsed if x.plaintext.strip()], str(picked)
    if stream_index is not None:
        import pysubs2
        text = owner.run([config["ffmpeg"], "-hide_banner", "-loglevel", "error",
                          "-nostdin", "-i", source, "-map", f"0:{stream_index}",
                          "-f", "srt", "-"], timeout=120).decode("utf-8", "replace")
        parsed = pysubs2.SSAFile.from_string(text, format_="srt")
        return [{"start": x.start / 1000, "end": x.end / 1000,
                 "text": x.plaintext.strip(), "actor": x.name or None,
                 "language": None} for x in parsed if x.plaintext.strip()], f"embedded:{stream_index}"
    return [], None


def safe_name(title):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(title)).strip(" .")[:100]
    return name or "Movie"
