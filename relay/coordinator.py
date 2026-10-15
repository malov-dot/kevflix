from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from .media import (atomic_json, fingerprint, probe, file_hash, wav_info, find_sidecar,
                    safe_name, RATE, CHANNELS)
from .process import ProcessOwner, Cancelled


class Coordinator:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.active_owner = None
        self.active_job = None
        self.thread = None
        self.hash_cache = {}
        self.publication_lock = threading.Lock()
        self.publication_owners = {}

    def start(self):
        self.store.recover()
        self.thread = threading.Thread(target=self.loop, name="movie-dub-jobs", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.active_owner:
            self.active_owner.cancel()
        for owner in list(self.publication_owners.values()):
            owner.cancel()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=30)

    def enqueue(self, source, *, audio_index=None, subtitles=None, subtitle_index=None,
                mode="prepared", title=None, plex=None, force_redub=False):
        source = self.config.media_path(source)
        if not subtitles and subtitle_index is None:
            subtitles = find_sidecar(source)
        if subtitles:
            subtitles = str(self.config.media_path(subtitles))
        owner = ProcessOwner()
        info = probe(self.config, owner, source)
        if audio_index is None:
            audio_index = info["audio"][0]["index"]
        if audio_index not in [x["index"] for x in info["audio"]]:
            raise ValueError("The requested audio stream is not present")
        if subtitle_index is not None and not any(s["index"] == subtitle_index
                and s.get("codec_type") == "subtitle" for s in info["streams"]):
            raise ValueError("The requested subtitle stream is not present")
        source_fingerprint = fingerprint(source)
        engine = self.config.engine_fingerprint()
        identity = [source_fingerprint, engine, audio_index, fingerprint(subtitles) if subtitles else None,
                    subtitle_index, bool(force_redub)]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        old = self.store.cached_job(key)
        if old:
            if mode == "prepared":
                self.request_all(old["id"])
            return old
        ident = uuid.uuid4().hex
        payload = {"source": str(source), "source_fingerprint": source_fingerprint,
                   "engine_fingerprint": engine, "audio_index": audio_index,
                   "subtitles": subtitles, "subtitle_index": subtitle_index, "duration": info["duration"],
                   "subtitle_fingerprint": fingerprint(subtitles) if subtitles else None,
                   "title": title or source.stem, "mode": mode, "plex": plex or {},
                   "force_redub": bool(force_redub), "revision": 1, "overrides": {},
                   "segment_seconds": int(self.config["segment_seconds"]),
                   "chunks_total": math.ceil(info["duration"] / self.config["segment_seconds"]),
                   "media": {"streams": info["streams"]}, "published": None}
        self.store.put_job(ident, key, payload)
        self.store.event(ident, "queued", title=payload["title"])
        if mode == "prepared":
            self.request_all(ident)
        else:
            self.request(ident, 0, 0)
        return self.store.job(ident)

    def request(self, job_id, index, priority=0):
        job = self.store.job(job_id)
        if not job or job["state"] in ("cancelled", "error"):
            raise ValueError("This job is unavailable")
        spec = job["payload"]
        if index < 0 or index >= spec["chunks_total"]:
            raise ValueError("Chunk is outside this movie")
        seconds = spec["segment_seconds"]
        self.store.request_chunk(job_id, index, {"start": index * seconds,
                                  "end": min(spec["duration"], (index + 1) * seconds)}, priority)
        for next_index in range(index + 1, min(spec["chunks_total"], index + 1 + self.config["lookahead_segments"])):
            self.store.request_chunk(job_id, next_index,
                                     {"start": next_index * seconds,
                                      "end": min(spec["duration"], (next_index + 1) * seconds)}, priority + 10)
        self.wake.set()

    def request_all(self, job_id):
        job = self.store.job(job_id)
        for index in range(job["payload"]["chunks_total"]):
            self.request(job_id, index, 100 + index)

    def cancel(self, job_id):
        self.store.update_job(job_id, "cancelled", message="Cancelled; completed chunks retained")
        if self.active_job == job_id and self.active_owner:
            self.active_owner.cancel()
        if job_id in self.publication_owners:
            self.publication_owners[job_id].cancel()
        self.store.event(job_id, "cancelled")

    def resume(self, job_id):
        job = self.store.job(job_id)
        if not job:
            raise KeyError(job_id)
        if fingerprint(job["payload"]["source"]) != job["payload"]["source_fingerprint"]:
            raise ValueError("Source changed; start a new job")
        self.store.update_job(job_id, "queued", message="Resuming retained work")
        for chunk in self.store.chunks(job_id):
            if chunk["state"] in ("error", "working", "cancelled"):
                self.store.update_chunk(job_id, chunk["idx"], "pending")
        if job["payload"]["mode"] == "prepared":
            self.request_all(job_id)
        self.summarize(job_id)
        self.wake.set()

    def review(self, job_id, index, *, approve_hash=None, corrections=None):
        chunk = self.store.chunk(job_id, index)
        job = self.store.job(job_id)
        if not chunk or chunk["state"] != "needs_review":
            raise ValueError("This chunk is not awaiting review")
        if corrections:
            overrides = dict(job["payload"].get("overrides", {}))
            valid = {c["id"] for c in chunk["payload"].get("cues", [])}
            for ident, correction in corrections.items():
                if ident not in valid:
                    raise ValueError("Correction belongs to a different cue")
                allowed = {"source_text", "source_language", "english_text", "speaker"}
                if not isinstance(correction, dict) or not set(correction) <= allowed:
                    raise ValueError("Invalid cue correction")
                speaker = correction.get("speaker")
                if speaker and speaker not in self.store.speakers(job_id):
                    raise ValueError("Choose an existing stable speaker identity")
                overrides[ident] = {**overrides.get(ident, {}), **correction}
            revision = job["payload"].get("revision", 1) + 1
            self.store.update_job(job_id, "queued", overrides=overrides, revision=revision, published=None)
            # A line can cross chunk boundaries, so invalidate this chunk and its neighbors.
            for item in self.store.chunks(job_id):
                if abs(item["idx"] - index) <= 1:
                    self.store.update_chunk(job_id, item["idx"], "pending")
                elif item["state"] in ("ready", "approved"):
                    self.store.update_chunk(job_id, item["idx"], item["state"], revision=revision)
            self.wake.set()
            return
        payload = chunk["payload"]
        if not payload.get("audio") or not approve_hash or approve_hash != payload.get("sha256"):
            raise ValueError("Approval must name the current generated audio hash")
        if file_hash(payload["audio"]) != approve_hash:
            raise ValueError("Audio changed since review")
        non_waivable = {"translation_unavailable", "speaker_unresolved", "language_uncertain",
                        "voice_pool_exhausted", "speech_without_transcript"}
        if any(x["kind"] in non_waivable for x in payload.get("issues", [])):
            raise ValueError("Correct unresolved text, language or speaker information before approval")
        self.store.update_chunk(job_id, index, "approved", owner_review={"sha256": approve_hash,
                                                                       "at": time.time()})
        self.store.update_job(job_id, "queued")
        self.summarize(job_id)
        self.wake.set()

    def audio_chunk(self, job_id, index, revision, *, wait=True):
        self.request(job_id, index)
        until = time.monotonic() + self.config["stream_wait_seconds"]
        while True:
            job = self.store.job(job_id)
            if not job or job["state"] in ("cancelled", "error", "needs_review"):
                raise ValueError("Playback is blocked: " + (job["state"] if job else "missing job"))
            if job["payload"].get("revision", 1) != revision:
                raise ValueError("Playback revision changed; reopen the current stream")
            chunk = self.store.chunk(job_id, index)
            if chunk and chunk["state"] in ("ready", "approved"):
                payload = chunk["payload"]
                path = Path(payload["audio"])
                stat = path.stat()
                cache_key = (str(path), stat.st_mtime_ns, stat.st_size, payload["sha256"])
                if cache_key not in self.hash_cache:
                    if file_hash(path) != payload["sha256"]:
                        raise ValueError("A generated chunk changed; playback refused")
                    data = wav_info(path)
                    if data["rate"] != RATE or data["channels"] != CHANNELS or data["width"] != 2:
                        raise ValueError("Generated chunk has an incompatible format")
                    expected_frames = round(payload["end"] * RATE) - round(payload["start"] * RATE)
                    if data["frames"] != expected_frames:
                        raise ValueError("Generated chunk duration does not match its timeline")
                    self.hash_cache[cache_key] = True
                return path
            if not wait or time.monotonic() >= until or self.stop_event.is_set():
                raise TimeoutError("English audio is not ready at this position")
            self.wake.wait(.3)
            self.wake.clear()

    def summarize(self, job_id):
        job = self.store.job(job_id)
        if not job or job["state"] == "cancelled":
            return
        chunks = self.store.chunks(job_id)
        ready = sum(x["state"] in ("ready", "approved") for x in chunks)
        if any(x["state"] == "needs_review" for x in chunks):
            state = "needs_review"
        elif any(x["state"] == "error" for x in chunks):
            state = "error"
        elif ready == job["payload"]["chunks_total"]:
            state = "complete"
        else:
            state = "buffering"
        self.store.update_job(job_id, state, chunks_ready=ready)

    def loop(self):
        while not self.stop_event.is_set():
            item = self.store.claim()
            if not item:
                self.wake.wait(1)
                self.wake.clear()
                continue
            job_id, index = item["job_id"], item["idx"]
            self.active_job = job_id
            self.active_owner = ProcessOwner()
            self.store.update_job(job_id, "processing")
            try:
                python = self.config["python"] or sys.executable
                env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                self.active_owner.run(
                    [python, "-m", "relay.worker", "--config", self.config.file,
                     "--job", job_id, "--chunk", index], timeout=3600,
                    cwd=Path(__file__).resolve().parent.parent, env=env)
                result = self.store.chunk(job_id, index)
                if result["state"] == "working":
                    raise RuntimeError("Worker exited without a chunk receipt")
            except Cancelled:
                self.store.update_chunk(job_id, index, "cancelled")
            except Exception as error:
                self.store.update_chunk(job_id, index, "error", error=str(error))
                self.store.event(job_id, "error", chunk=index, message=str(error))
            finally:
                self.active_owner = None
                self.active_job = None
                self.summarize(job_id)
                self.wake.set()

    def publish(self, job_id):
        """Assemble incrementally and publish a derived file; never overwrite the source."""
        owner = ProcessOwner()
        self.publication_owners[job_id] = owner
        try:
            with self.publication_lock:
                return self._publish(job_id, owner)
        finally:
            self.publication_owners.pop(job_id, None)

    def _publish(self, job_id, owner):
        job = self.store.job(job_id)
        if not job or job["state"] != "complete":
            raise ValueError("Every chunk must be ready or explicitly reviewed before publishing")
        if not self.config["publish_dir"]:
            raise ValueError("Configure publish_dir on the shared NAS folder")
        spec = job["payload"]
        existing = spec.get("published")
        if existing and existing.get("revision") == spec.get("revision", 1):
            if Path(existing["path"]).is_file() and file_hash(existing["path"]) == existing["sha256"]:
                return existing
        directory = self.config.cache / job_id
        concat = directory / "audio-concat.txt"
        paths = [self.audio_chunk(job_id, i, spec.get("revision", 1), wait=False)
                 for i in range(spec["chunks_total"])]
        def escaped(p):
            return str(p).replace("\\", "/").replace("'", "'\\''")
        concat.write_text("".join("file '" + escaped(p) + "'\n" for p in paths), encoding="utf-8")
        output = directory / f"english-r{spec.get('revision', 1)}.mkv"
        subtitle_options = []
        subtitles = [s for s in spec["media"]["streams"] if s.get("codec_type") == "subtitle"]
        for number, stream in enumerate(subtitles):
            if stream.get("codec_name") in ("mov_text", "text", "webvtt"):
                subtitle_options.extend([f"-c:s:{number}", "srt"])
        owner.run([self.config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-i", spec["source"], "-f", "concat", "-safe", 0, "-i", concat,
                   "-map", "0:v?", "-map", "1:a:0", "-map", "0:a?", "-map", "0:s?", "-map", "0:t?",
                   "-map_metadata", "0", "-map_chapters", "0", "-c", "copy",
                   "-c:a:0", "aac", "-b:a:0", "192k", "-metadata:s:a:0", "language=eng",
                   "-metadata:s:a:0", "title=Movie Dub English",
                   "-disposition:a", "0", "-disposition:a:0", "default",
                   *subtitle_options, "-y", output], timeout=max(600, spec["duration"] * 2))
        metadata = probe(self.config, owner, output)
        if abs(metadata["duration"] - spec["duration"]) > .5:
            raise ValueError("Published version has a different movie duration")
        destination_dir = Path(self.config["publish_dir"]) / (safe_name(spec["title"]) + " - Movie Dub")
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / (safe_name(spec["title"]) + f" - English {job_id[:8]}-r{spec.get('revision', 1)}.mkv")
        if destination.exists():
            raise ValueError("A published destination already exists; originals are never replaced")
        temporary = destination.with_suffix(".partial")
        with output.open("rb") as src, temporary.open("wb") as dst:
            for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                owner.check()
                dst.write(block)
        digest = file_hash(output)
        if file_hash(temporary) != digest:
            raise ValueError("Published copy does not match the prepared file")
        owner.check()
        current = self.store.job(job_id)
        if current["state"] == "cancelled" or current["payload"].get("revision", 1) != spec.get("revision", 1):
            raise Cancelled("Publication was cancelled or revised")
        os.replace(temporary, destination)
        published = {"path": str(destination), "sha256": digest,
                     "duration": metadata["duration"], "at": time.time(),
                     "revision": spec.get("revision", 1)}
        self.store.update_job(job_id, published=published)
        self.store.event(job_id, "published", filename=destination.name)
        return published
