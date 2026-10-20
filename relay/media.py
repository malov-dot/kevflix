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


def extract_audio(config, owner, source, audio_index, start, seconds, dest, rate=RATE, channels=CHANNELS,
                  *, preserve_layout=None):
    """Decode one audio stream to 16-bit PCM WAV.

    Default (legacy streaming path): ``-ac channels`` downmixes/remaps to the
    requested count. ``preserve_layout`` (actor path) names the selected
    source stream's own layout; ``-ac`` is then omitted so FFmpeg keeps the
    decoder's channel order, and the written WAV must prove that layout: the
    channel count must match and a six-channel file must carry the exact
    speaker mask (0x3F for 5.1, 0x60F for 5.1(side)). A mismatch raises; the
    output is never relabelled. Returns the ``resolve_channel_layout``
    evidence when preserving, else None.
    """
    if preserve_layout is not None:
        if preserve_layout not in CHANNEL_LAYOUTS:
            raise ValueError(f"Cannot preserve unsupported channel layout {preserve_layout!r}")
        if channels not in (CHANNELS, CHANNEL_LAYOUTS[preserve_layout]):
            raise ValueError("Requested channel count contradicts the preserved layout")
        mix = []
    else:
        mix = ["-ac", channels]
    owner.run([config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{start:.6f}", "-i", source, "-t", f"{seconds:.6f}",
               "-map", f"0:{audio_index}", "-vn", "-sn", *mix, "-ar", rate,
               "-c:a", "pcm_s16le", "-y", dest], timeout=max(60, seconds * 4))
    if preserve_layout is None:
        return None
    expected = CHANNEL_LAYOUTS[preserve_layout]
    header = wav_format(dest)
    if header is None or header["channels"] != expected:
        raise ValueError(f"Extracted audio does not keep the source's {expected} channels")
    if expected > 2 and header.get("channel_mask") != _LAYOUT_MASKS[preserve_layout]:
        raise ValueError(f"Extracted audio speaker mask {header.get('channel_mask')!r} does not prove "
                         f"the source {preserve_layout} layout; refusing to relabel remapped channels")
    return resolve_channel_layout(expected, source=preserve_layout, wav=header)


# Supported source layouts and their WAVE_FORMAT_EXTENSIBLE speaker masks.
# FFmpeg naming: "5.1" is FL+FR+FC+LFE+BL+BR (back) and "5.1(side)" is
# FL+FR+FC+LFE+SL+SR. Both share six channels, so a count never decides them.
CHANNEL_LAYOUTS = {"mono": 1, "stereo": 2, "5.1": 6, "5.1(side)": 6}
WAV_MASK_LAYOUTS = {0x4: "mono", 0x3: "stereo", 0x3F: "5.1", 0x60F: "5.1(side)"}
_LAYOUT_MASKS = {layout: mask for mask, layout in WAV_MASK_LAYOUTS.items()}
_COUNT_ONLY_LAYOUTS = {1: "mono", 2: "stereo"}
_UNSPECIFIED_LAYOUT = re.compile(r"^(?:|unknown|\d+ channels?|\d+c)$", re.I)
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def wav_format(path, *, header_limit=1024 * 1024):
    """Read only the RIFF/RF64 ``fmt `` header of a decoded WAV file.

    Returns None for a non-WAVE container (no channel-mask evidence). A WAVE
    header that is truncated, lacks ``fmt `` before ``data`` or is malformed
    raises ValueError instead of being treated as an unassigned layout.
    """
    import struct
    with open(path, "rb") as stream:
        head = stream.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64", b"BW64") or head[8:12] != b"WAVE":
            return None
        consumed = 12
        while consumed < header_limit:
            chunk = stream.read(8)
            if len(chunk) < 8:
                raise ValueError("WAV header ended before its fmt chunk")
            ident, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            consumed += 8
            if ident == b"fmt ":
                if not 16 <= size <= 4096:
                    raise ValueError("WAV fmt chunk has an invalid size")
                body = stream.read(size)
                if len(body) < size:
                    raise ValueError("WAV fmt chunk is truncated")
                tag, channels, rate, _, align, bits = struct.unpack("<HHIIHH", body[:16])
                result = {"format_tag": tag, "channels": channels, "rate": rate,
                          "block_align": align, "bits": bits, "extensible": tag == _WAVE_FORMAT_EXTENSIBLE,
                          "channel_mask": None, "subformat": None}
                if tag == _WAVE_FORMAT_EXTENSIBLE:
                    if size < 40 or struct.unpack("<H", body[16:18])[0] < 22:
                        raise ValueError("WAVE_FORMAT_EXTENSIBLE header lacks its channel mask")
                    result["channel_mask"] = struct.unpack("<I", body[20:24])[0]
                    result["subformat"] = body[24:40].hex()
                if not channels:
                    raise ValueError("WAV header declares zero channels")
                return result
            if ident == b"data":
                raise ValueError("WAV data precedes its fmt chunk")
            skip = size + (size & 1)
            stream.seek(skip, os.SEEK_CUR)
            consumed += skip
    raise ValueError("WAV fmt chunk was not found within the header limit")


def resolve_channel_layout(decoded_channels, *, source=None, wav=None):
    """Decide the actual decoded PCM layout from explicit evidence only.

    ``source`` is the selected ffprobe audio stream (a dict with ``channels``
    and ``channel_layout``) or its ``channel_layout`` string. ``wav`` is the
    ``wav_format`` result for the decoded file. Mono and stereo may follow from
    the decoded count; six channels require an explicit 5.1 or 5.1(side) layout
    from ffprobe or a WAV channel mask. Unsupported, unknown or contradictory
    multichannel evidence raises ValueError before any synthesis.
    """
    if isinstance(decoded_channels, bool) or not isinstance(decoded_channels, int) or decoded_channels < 1:
        raise ValueError("Decoded PCM channel count is invalid")
    evidence, basis = {}, []
    probe_layout = probe_channels = None
    probe_unspecified = False
    if isinstance(source, dict):
        probe_layout, probe_channels = source.get("channel_layout"), source.get("channels")
    elif source is not None:
        probe_layout = source
    if probe_channels is not None:
        if isinstance(probe_channels, bool) or int(probe_channels) != probe_channels or int(probe_channels) != decoded_channels:
            raise ValueError("Selected source channel count does not match the decoded PCM")
    if probe_layout is not None:
        if not isinstance(probe_layout, str):
            raise ValueError("Selected source channel layout must be a string")
        text = probe_layout.strip()
        if text in CHANNEL_LAYOUTS:
            evidence["ffprobe"] = text
        elif _UNSPECIFIED_LAYOUT.match(text):
            probe_unspecified = True
        else:
            raise ValueError(f"Unsupported source channel layout {probe_layout!r}; supported: mono, stereo, 5.1, 5.1(side)")
    if probe_unspecified and decoded_channels not in _COUNT_ONLY_LAYOUTS:
        # A decoder/extractor may assign a default order to unknown input; the
        # extracted WAV mask would then describe that assignment, not the movie.
        raise ValueError("The selected source declares an unspecified multichannel order; "
                         "an extracted WAV mask cannot establish 5.1 versus 5.1(side)")
    mask = None
    if wav is not None:
        if wav.get("channels") != decoded_channels:
            raise ValueError("WAV header channel count does not match the decoded PCM")
        mask = wav.get("channel_mask")
        if mask:
            if mask not in WAV_MASK_LAYOUTS:
                raise ValueError(f"Unsupported WAV channel mask 0x{mask:X}; supported: mono, stereo, 5.1, 5.1(side)")
            evidence["wav_channel_mask"] = WAV_MASK_LAYOUTS[mask]
    named = set(evidence.values())
    if len(named) > 1:
        raise ValueError(f"Contradictory channel layout evidence: {evidence}")
    if named:
        layout = named.pop()
        basis = sorted(evidence)
    elif decoded_channels in _COUNT_ONLY_LAYOUTS:
        layout, basis = _COUNT_ONLY_LAYOUTS[decoded_channels], ["decoded_channel_count"]
    else:
        raise ValueError(f"{decoded_channels}-channel PCM needs an explicit 5.1 or 5.1(side) layout; "
                         "the channel count alone cannot decide it")
    if CHANNEL_LAYOUTS[layout] != decoded_channels:
        raise ValueError(f"Channel layout {layout} does not match {decoded_channels} decoded channels")
    return {"version": "channel-layout-v1", "layout": layout, "channels": decoded_channels,
            "basis": basis, "ffprobe_channel_layout": probe_layout,
            "wav_channel_mask": mask, "lfe_index": 3 if decoded_channels == 6 else None}


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
    available = [p for p in source.parent.iterdir() if p.is_file()
                 and p.suffix.lower() in (".srt", ".ass", ".ssa")]
    stem = source.stem.casefold()
    candidates = [p for p in available if p.stem.casefold() == stem
                  or p.stem.casefold().startswith(stem + '.')]
    if not candidates:
        # Different release groups often use different basenames. Require the
        # same complete title and year; a sole unrelated subtitle is not a match.
        def identity(name):
            match = re.match(r'^(.*?)[ ._(\[]+((?:19|20)\d{2})(?=[ ._)\]]|$)', name)
            if not match:
                return None
            return (re.sub(r'[\W_]+', ' ', match[1].casefold()).strip(), match[2])
        key = identity(source.stem)
        candidates = [p for p in available if key and identity(p.stem) == key]
    english = [p for p in candidates if re.search(r'\.(en|eng)(\.|$)', p.name, re.I)]
    preferred = english or candidates
    # Ambiguous alternate subtitle cuts require explicit selection.
    return preferred[0] if len(preferred) == 1 else None


def subtitle_cues(config, owner, source, explicit=None, stream_index=None, discover=True):
    """Parse bounded text captions outside Electron's UI thread."""
    source = Path(source)
    picked = Path(explicit) if explicit else find_sidecar(source) if discover and stream_index is None else None
    if picked:
        picked = config.media_path(picked)
        import pysubs2
        parsed = pysubs2.load(str(picked), encoding="utf-8")
        suffix = picked.stem[len(source.stem):] if picked.stem.startswith(source.stem) else picked.stem
        lang_match = re.search(r"\.(en|eng|ru|rus|es|spa|fr|fra|de|deu|ja|jpn|zh|zho|ko|kor|it|ita|pt|por|ar|ara|hi|hin)(?=\.|$)", suffix, re.I)
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
