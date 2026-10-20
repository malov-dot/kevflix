"""Subtitle stream selection and window-bounded caption extraction (embedded-subtitles-v1).

Selection happens once, when a job is created, from the streams that job already probed:
an explicit file, then an explicit stream, then a sidecar file, then the single
unambiguous English text stream. ``"off"`` disables subtitles. Ambiguous, image-only
and untagged candidates give an explicit reason and no subtitles; English is never
assumed for a stream that does not declare it.

Extraction happens per chunk for the analysed window only. FFmpeg reads aligned
blocks of the selected stream with ``-copyts`` so cue times stay absolute source
seconds (without it FFmpeg shifts and clips a cue that straddles the seek point).
Each block starts PREROLL_SECONDS early, so a caption that starts before the window
and ends inside it keeps its original start and end. Blocks are cached immutably
under ``config.cache/subtitle-blocks``, keyed by source fingerprint, stream, block,
options and VERSION, and are content-checked on every read.

Captions are evidence about text, never about speech: an empty window is not "no
dialogue", and nothing here admits or removes audio. Any failure other than
cancellation returns no captions with an explicit reason so the existing speech
recognition route continues.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

from .language import iso
from .process import Cancelled

VERSION = "embedded-subtitles-v1"
OFF = "off"
TEXT_CODECS = frozenset(("subrip", "ass", "ssa", "webvtt", "mov_text", "text"))
BLOCK_SECONDS = 60.0
PREROLL_SECONDS = 30.0
MAX_WINDOW_SECONDS = 600.0
TIME_BASIS = "absolute-source-seconds"
_UNDETERMINED = ("", "und", "unk", "mis", "mul", "zxx", "qaa")
_FILE_LANGUAGE = re.compile(r"\.(en|eng|ru|rus|es|spa|fr|fra|de|deu|ja|jpn|zh|zho|ko|kor|it|ita|pt|por|ar|ara|hi|hin)(?=\.|$)", re.I)
_LOCKS, _LOCKS_GUARD = {}, threading.Lock()


class CaptionList(list):
    """A plain list of cue dicts (JSON-serialisable) that also carries ``subtitle_source``."""
    subtitle_source = None


def normalize_language(tag):
    """(normalized, raw) using relay.language.iso; undetermined tags normalize to None."""
    raw = str(tag).strip() if tag is not None else ""
    if raw.lower() in _UNDETERMINED:
        return None, raw or None
    return iso(raw), raw


def _tags(stream):
    return stream.get("tags") or {}


def _disposition(stream, name):
    return bool((stream.get("disposition") or {}).get(name))


def is_commentary(stream):
    return _disposition(stream, "comment") or "commentary" in str(_tags(stream).get("title", "")).lower()


SUPPORTED_SIX_CHANNEL_LAYOUTS = ("5.1", "5.1(side)")
AUDIO_POLICY = "primary-non-commentary-5.1-else-stereo-else-mono-v1"


class UnsupportedAudio(ValueError):
    """No audio track the dubbing path supports (e.g. a 7.1-only source). Explicit; nothing is downmixed."""


def audio_support(stream):
    """None when the dubbing path supports this stream (mono, stereo, 5.1 / 5.1(side)), else the reason."""
    channels = int(stream.get("channels") or 0)
    if channels in (1, 2):
        return None
    if channels == 6 and stream.get("channel_layout") in SUPPORTED_SIX_CHANNEL_LAYOUTS:
        return None
    return "%s channels (%s) is not supported for dubbing; 7.1 and other layouts are never downmixed" % (
        channels or "unknown", stream.get("channel_layout") or "no layout")


def select_audio(audio_streams):
    """Fix-7 default track policy, shared by every entry path (Coordinator.enqueue, TVService.open_movie).

    The primary (first) non-commentary 5.1 / 5.1(side) track; if there is no such 5.1 track, the first non-commentary
    stereo track, then mono. Commentary tracks are considered only when no main track is supported. Returns
    (index, None), or (None, reason) when no track is supported (e.g. 7.1-only): an explicit unsupported state, never a
    downmix and never an unwanted first track."""
    if not audio_streams:
        raise ValueError("Media contains no audio track")
    main = [s for s in audio_streams if not is_commentary(s)]
    for group in (main, [s for s in audio_streams if is_commentary(s)]):
        for channels in (6, 2, 1):
            for stream in group:
                if int(stream.get("channels") or 0) == channels and audio_support(stream) is None:
                    return stream["index"], None
    layouts = sorted({"%s/%s" % (s.get("channels"), s.get("channel_layout")) for s in audio_streams})
    return None, "No supported audio track (only %s): 7.1 and other layouts are not supported for dubbing" % ", ".join(layouts)


def default_audio_index(audio_streams):
    """The fix-7 default (select_audio); raises UnsupportedAudio when the source has no supported track."""
    index, reason = select_audio(audio_streams)
    if index is None:
        raise UnsupportedAudio(reason)
    return index


def require_supported_audio(audio_streams, audio_index):
    """An explicit user track choice is respected when the dubbing path supports it; otherwise a clear error."""
    stream = next((s for s in audio_streams if s.get("index") == audio_index), None)
    if stream is None:
        raise ValueError("The requested audio stream is not present")
    reason = audio_support(stream)
    if reason is not None:
        raise UnsupportedAudio("Audio track %s: %s" % (audio_index, reason))
    return audio_index


def _stream_facts(stream):
    language, raw = normalize_language(_tags(stream).get("language"))
    return {"stream_index": stream["index"], "codec": stream.get("codec_name"),
            "declared_language": language, "declared_language_raw": raw,
            "language_basis": "stream-language-tag" if language else None,
            "forced": _disposition(stream, "forced"), "title": _tags(stream).get("title")}


def _file_facts(path):
    match = _FILE_LANGUAGE.search(Path(path).stem)
    language, raw = normalize_language(match.group(1)) if match else (None, None)
    return {"path": str(path), "codec": Path(path).suffix.lower().lstrip(".") or None,
            "declared_language": language, "declared_language_raw": raw,
            "language_basis": "file-name-tag" if language else None}


def _choice(kind, selection, reason=None, **facts):
    choice = {"version": VERSION, "kind": kind, "selection": selection, "reason": reason,
              "stream_index": None, "path": None, "codec": None, "declared_language": None,
              "declared_language_raw": None, "language_basis": None}
    choice.update(facts)
    return choice


def select_subtitles(streams, *, explicit_path=None, explicit_index=None, sidecar=None):
    """Decide which subtitle source a job uses, from already-probed ``streams``."""
    if explicit_path == OFF:
        return _choice("none", "explicit-off", "subtitles were turned off")
    if explicit_path:
        return _choice("external", "explicit-file", **_file_facts(explicit_path))
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
    if explicit_index is not None:
        stream = next((s for s in subtitle_streams if s["index"] == explicit_index), None)
        if stream is None:
            raise ValueError("The requested subtitle stream is not present")
        facts = _stream_facts(stream)
        if facts["codec"] not in TEXT_CODECS:
            return _choice("none", "explicit-stream",
                           "the selected subtitle stream is image-only (%s); text cues are unavailable" % facts["codec"],
                           **facts)
        return _choice("embedded", "explicit-stream", **facts)
    if sidecar:
        return _choice("external", "sidecar", **_file_facts(sidecar))
    facts = [_stream_facts(s) for s in subtitle_streams]
    english_text = [f for f in facts if f["codec"] in TEXT_CODECS and f["declared_language"] == "en"
                    and "commentary" not in str(f["title"] or "").lower()]
    full = [f for f in english_text if not f["forced"]]
    if len(full) == 1:
        return _choice("embedded", "auto-unambiguous-english", **full[0])
    if len(full) > 1:
        reason = "ambiguous: %d English text subtitle streams (%s)" % (
            len(full), ", ".join(str(f["stream_index"]) for f in full))
    elif english_text:
        reason = "only forced English text subtitle streams; they do not carry full dialogue"
    elif any(f["declared_language"] == "en" for f in facts):
        reason = "English subtitles are image-only; text cues are unavailable"
    elif any(f["codec"] in TEXT_CODECS and f["declared_language"] is None for f in facts):
        reason = "text subtitle streams declare no language; English is not assumed"
    elif facts:
        reason = "no English text subtitle stream"
    else:
        reason = "no subtitle streams"
    return _choice("none", "auto", reason)


def spec_choice(spec):
    """The subtitle choice for a job payload, including jobs created before this overlay."""
    if spec.get("subtitles"):
        stored = spec.get("subtitle_choice") or {}
        selection = stored.get("selection") if stored.get("kind") == "external" else "file"
        return _choice("external", selection, **_file_facts(spec["subtitles"]))
    streams = (spec.get("media") or {}).get("streams") or []
    if spec.get("subtitle_index") is not None:
        return select_subtitles(streams, explicit_index=spec["subtitle_index"])
    stored = spec.get("subtitle_choice")
    if not stored:
        return _choice("none", "legacy", "the job predates automatic subtitle selection")
    if stored.get("kind") == "embedded":
        stream = next((s for s in streams if s.get("index") == stored.get("stream_index")
                       and s.get("codec_type") == "subtitle"), None)
        if stream is None:
            return _choice("none", stored.get("selection"), "the selected subtitle stream is missing from the probed media")
        return _choice("embedded", stored["selection"], **_stream_facts(stream))
    return _choice("none", stored.get("selection"), stored.get("reason"))


# --- bounded, cached extraction ---------------------------------------------------------

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def cache_root(config):
    """``config.cache/subtitle-blocks``; refuses links or junctions that leave the cache."""
    base = Path(config.cache).resolve()
    root = base / "subtitle-blocks"
    if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
        raise ValueError("The subtitle cache directory must not be a link")
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root or base not in root.resolve().parents:
        raise ValueError("The subtitle cache directory resolves outside the movie cache")
    return root


def block_identity(source_fingerprint, stream_index, block, codec):
    return {"version": VERSION, "source_fingerprint": source_fingerprint, "stream_index": int(stream_index),
            "block": int(block), "block_seconds": BLOCK_SECONDS, "preroll_seconds": PREROLL_SECONDS,
            "codec": codec, "output": "srt", "timestamps": "copyts"}


def _read_block(path, identity):
    if path.is_symlink():
        raise ValueError("A subtitle cache entry must not be a link")
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if entry.get("identity") != identity or entry.get("content_sha256") != _digest(entry.get("cues")):
        return None
    return entry


def _write_block(path, entry):
    temp = path.with_name(".%s.%d.%s.tmp" % (path.name, os.getpid(), uuid.uuid4().hex))
    temp.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    try:
        os.replace(temp, path)
    except PermissionError:
        # Windows: a concurrent reader holds the identical entry another writer replaced.
        temp.unlink(missing_ok=True)
        if _read_block(path, entry["identity"]) is None:
            raise


def _srt_cues(text):
    import pysubs2
    parsed = pysubs2.SSAFile.from_string(text, format_="srt") if text.strip() else []
    return [{"start": x.start / 1000, "end": x.end / 1000, "text": x.plaintext.strip(), "actor": x.name or None}
            for x in parsed if x.plaintext.strip()]


def extract_block(config, owner, source, stream_index, block):
    """One FFmpeg call: [block*B - preroll, (block+1)*B) of one stream, absolute cue times."""
    start, stop = max(0.0, block * BLOCK_SECONDS - PREROLL_SECONDS), (block + 1) * BLOCK_SECONDS
    out = owner.run([config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin", "-copyts",
                     "-ss", "%.3f" % start, "-i", str(source), "-map", "0:%d" % int(stream_index),
                     "-to", "%.3f" % stop, "-f", "srt", "-"], timeout=120)
    return _srt_cues(out.decode("utf-8", "replace"))


def cached_block(config, owner, source, source_fingerprint, stream_index, codec, block):
    identity = block_identity(source_fingerprint, stream_index, block, codec)
    key = _digest(identity)
    path = cache_root(config) / (key + ".json")
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:  # overlapping requests in this process extract a block once
        entry = _read_block(path, identity) if path.exists() else None
        if entry is not None:
            return entry, "hit"
        owner.check()
        cues = extract_block(config, owner, source, stream_index, block)
        entry = {"identity": identity, "cues": cues, "content_sha256": _digest(cues)}
        _write_block(path, entry)
        return entry, "extracted"


def window_cues(config, owner, source, source_fingerprint, stream_index, codec, start, end):
    """Cues overlapping [start, end), original absolute times, deduplicated across blocks."""
    start, end = float(start), float(end)
    if not 0 <= start < end or end - start > MAX_WINDOW_SECONDS:
        raise ValueError("Subtitle window must be finite, ordered and at most %d seconds" % MAX_WINDOW_SECONDS)
    seen, cues, blocks = set(), [], []
    for block in range(int(start // BLOCK_SECONDS), int((end - 1e-9) // BLOCK_SECONDS) + 1):
        entry, status = cached_block(config, owner, source, source_fingerprint, stream_index, codec, block)
        blocks.append({"block": block, "cache_key": _digest(entry["identity"]),
                       "content_sha256": entry["content_sha256"], "cache": status})
        for cue in entry["cues"]:
            ident = (cue["start"], cue["end"], cue["text"], cue["actor"])
            if cue["end"] > start and cue["start"] < end and ident not in seen:
                seen.add(ident)
                cues.append(dict(cue))
    cues.sort(key=lambda c: (c["start"], c["end"], c["text"]))
    return cues, blocks


def _file_cues(path):
    import pysubs2
    parsed = pysubs2.load(str(path), encoding="utf-8")
    return [{"start": x.start / 1000, "end": x.end / 1000, "text": x.plaintext.strip(), "actor": x.name or None}
            for x in parsed if x.plaintext.strip()]


def chunk_captions(config, owner, spec, start, end, directory=None):
    """Worker hook: captions for one analysed window plus provenance.

    Returns ``(captions, legacy_source)``: ``captions`` is a CaptionList whose
    ``subtitle_source`` holds the provenance contract; ``legacy_source`` is the string
    the worker stored before (subtitle path, ``"embedded:N"`` or None).
    """
    choice = spec_choice(spec)
    meta = {"version": VERSION, "kind": choice["kind"], "selection": choice["selection"],
            "selected_stream_index": choice["stream_index"], "path": choice["path"], "codec": choice["codec"],
            "declared_language": choice["declared_language"], "declared_language_raw": choice["declared_language_raw"],
            "language_basis": choice["language_basis"], "source_fingerprint": spec.get("source_fingerprint"),
            "subtitle_fingerprint": spec.get("subtitle_fingerprint") if choice["kind"] == "external" else None,
            "cue_time_basis": TIME_BASIS, "window": [float(start), float(end)], "blocks": [],
            "status": "none", "reason": choice["reason"], "completeness": None, "coverage_limit": None}
    if choice["kind"] == "embedded":
        # Never asserted complete: only cues starting at or after (first block start - preroll) are read.
        meta.update(block_seconds=BLOCK_SECONDS, preroll_seconds=PREROLL_SECONDS, completeness="not-guaranteed",
                    coverage_limit="a cue that starts more than %d s before the start of the 60 s block containing "
                                   "the window start may be missing; VAD/ASR remains the speech route" % PREROLL_SECONDS)
    elif choice["kind"] == "external":
        meta["completeness"] = "whole-file-parsed"
    legacy = (choice["path"] if choice["kind"] == "external" else
              "embedded:%d" % choice["stream_index"] if choice["kind"] == "embedded" else None)
    cues = []
    try:
        if choice["kind"] == "external":
            cues = [c for c in _file_cues(config.media_path(choice["path"])) if c["end"] > start and c["start"] < end]
        elif choice["kind"] == "embedded":
            cues, meta["blocks"] = window_cues(config, owner, spec["source"], spec["source_fingerprint"],
                                               choice["stream_index"], choice["codec"], start, end)
        if choice["kind"] != "none":
            meta["status"] = "ok" if cues else "empty-window"
            meta["reason"] = None if cues else "no caption overlaps this window; this is not evidence of no speech"
    except Cancelled:
        raise
    except Exception as error:
        cues = []
        meta.update(status="unavailable", reason="%s: %s" % (type(error).__name__, str(error)[:300]))
    for cue in cues:
        cue.update(language=choice["declared_language"], language_raw=choice["declared_language_raw"],
                   language_basis=choice["language_basis"])
    meta["cue_count"], meta["content_sha256"] = len(cues), _digest(cues)
    captions = CaptionList(cues)
    captions.subtitle_source = meta
    if directory is not None:
        path = Path(directory) / "subtitle-source.json"
        temp = path.with_name(".%s.%d.%s.tmp" % (path.name, os.getpid(), uuid.uuid4().hex))
        temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    return captions, legacy
