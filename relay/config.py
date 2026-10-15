from __future__ import annotations

import copy
import hashlib
import json
import os
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

DEFAULTS = {
    "setup_only": False,
    "bind": "127.0.0.1", "port": 8787,
    "state_dir": "./state", "cache_dir": "./state/cache",
    "api_token": "", "nas_token": "", "stream_secret": "", "webhook_secret": "",
    "public_url": "http://127.0.0.1:8787",
    "python": "", "ffmpeg": "ffmpeg", "ffprobe": "ffprobe",
    "cpu_threads": 2,
    "media_roots": [], "path_mappings": [],
    "segment_seconds": 30, "context_seconds": 3, "lookahead_segments": 2,
    "max_workers": 1, "stream_wait_seconds": 900,
    "device": "cpu", "gpu_min_free_mib": 10000,
    "sem_status_file": "",
    "asr_model": "./models/faster-whisper-small", "asr_compute_type": "int8",
    "translation_model": "./models/m2m100_418M",
    "diarization_model": "./models/pyannote-community-1",
    "speaker_similarity": 0.70, "speaker_margin": 0.06,
    "language_confidence": 0.65, "speaker_overlap": 0.55,
    "speech_engine": "sapi", "voices": [],
    "piper_models": [], "max_speech_speed": 1.35,
    "min_dialogue_suppression_db": 12.0,
    "residual_word_overlap": 0.35,
    "separation_model": "htdemucs",
    "separation_model_dir": "./models/htdemucs",
    "auto_dub": False, "delivery_mode": "prepared",
    "publish_dir": "", "plex_publish_path": "", "plex_publish_section": "",
    "plex": {"url": "", "token": "", "client_id": "", "owner_id": "",
             "poll_seconds": 2, "route_timeout_seconds": 45},
    "tv": {"enabled": True, "roots": [], "segment_seconds": 30,
           "local_only": True,
           "max_height": 2160, "encoder_threads": 2, "buffer_segments": 2,
           "video_encoder": "auto", "nvenc_min_free_mib": 2048, "upscale": False,
           "hdr_mode": "auto", "pc_process_all": True, "sync_plex_progress": False,
           "min_free_disk_gib": 10,
           "idle_seconds": 45, "allow_origins": ["http://127.0.0.1:8081", "http://localhost:8081"],
           "allow_plex_controls": False},
    "updates": {"enabled": True, "repository": "malov-dot/kevflix",
                "branch": "main", "path": "tv", "interval_seconds": 3600,
                "token_env": "MOVIEDUB_GITHUB_TOKEN"},
}


class Config:
    def __init__(self, filename):
        self.file = Path(filename).resolve()
        raw = json.loads(self.file.read_text(encoding="utf-8-sig"))
        self.data = copy.deepcopy(DEFAULTS)
        self.data.update(raw)
        self.data["plex"] = {**DEFAULTS["plex"], **raw.get("plex", {})}
        self.data["tv"] = {**DEFAULTS["tv"], **raw.get("tv", {})}
        self.data["updates"] = {**DEFAULTS["updates"], **raw.get("updates", {})}
        if not 1 <= int(self.data["tv"]["encoder_threads"]) <= 8:
            raise ValueError("TV encoder threads must be between 1 and 8")
        if int(self.data["tv"]["max_height"]) not in (480, 720, 1080, 2160):
            raise ValueError("TV output height must be 480, 720, 1080 or 2160")
        if self.data["tv"]["video_encoder"] not in ("auto", "nvenc", "software"):
            raise ValueError("TV video_encoder must be auto, nvenc or software")
        for key, env in (("api_token", "MOVIEDUB_API_TOKEN"),
                         ("nas_token", "MOVIEDUB_NAS_TOKEN"),
                         ("stream_secret", "MOVIEDUB_STREAM_SECRET"),
                         ("webhook_secret", "MOVIEDUB_WEBHOOK_SECRET")):
            self.data[key] = os.environ.get(env, self.data[key])
        self.data["plex"]["token"] = os.environ.get("MOVIEDUB_PLEX_TOKEN",
                                                    self.data["plex"]["token"])
        for key in ("state_dir", "cache_dir", "asr_model", "translation_model",
                    "diarization_model", "separation_model_dir", "publish_dir"):
            if self.data[key]:
                self.data[key] = str(self.resolve(self.data[key]))
        if not 5 <= int(self.data["segment_seconds"]) <= 120:
            raise ValueError("segment_seconds must be between 5 and 120")
        if self.data["device"] not in ("cpu", "cooperative-cuda"):
            raise ValueError("device must be cpu or cooperative-cuda")
        if self.data["speech_engine"] not in ("sapi", "piper"):
            raise ValueError("Choose sapi or piper for intelligible English speech")
        if self.data["delivery_mode"] not in ("prepared", "buffered"):
            raise ValueError("delivery_mode must be prepared or buffered")
        for name in ("api_token", "nas_token", "stream_secret", "webhook_secret"):
            if len(str(self.data[name])) < 32:
                raise ValueError(name + " must be generated during setup (at least 32 characters)")
        for name, value in (("public_url", self.data["public_url"]),
                            ("plex.url", self.data["plex"]["url"])):
            if value and (urlparse(value).scheme not in ("http", "https")
                          or not urlparse(value).hostname):
                raise ValueError(name + " must be an HTTP(S) URL")
        if self.data["tv"]["enabled"] and self.data["tv"]["local_only"]:
            host = urlparse(self.data["public_url"]).hostname or ""
            try:
                local = ipaddress.ip_address(host).is_private
            except ValueError:
                local = host == "localhost" or host.endswith(".local")
            if not local:
                raise ValueError("TV streaming is local-only; public_url must use the PC's private LAN address")
        self.state = Path(self.data["state_dir"])
        self.cache = Path(self.data["cache_dir"])

    def resolve(self, value):
        p = Path(os.path.expandvars(str(value))).expanduser()
        return p.resolve() if p.is_absolute() else (self.file.parent / p).resolve()

    def __getitem__(self, key):
        return self.data[key]

    def media_path(self, value, *, nas=False):
        value = str(value)
        if nas:
            match = None
            for entry in self["path_mappings"]:
                prefix = entry["nas"].rstrip("/")
                if value == prefix or value.startswith(prefix + "/"):
                    if match is None or len(prefix) > len(match[0]):
                        match = (prefix, entry["pc"])
            if match is None:
                raise ValueError("No NAS-to-PC mapping is configured for this media")
            suffix = value[len(match[0]):].lstrip("/")
            value = str(Path(match[1]) / Path(suffix))
        p = self.resolve(value)
        allowed = [self.resolve(x) for x in self["media_roots"]]
        if not any(p == r or r in p.parents for r in allowed):
            raise ValueError("Media path is outside configured media_roots")
        if not p.is_file():
            raise ValueError("Media file is unavailable to the PC worker")
        return p

    def engine_fingerprint(self):
        keys = ("segment_seconds", "context_seconds", "asr_model",
                "translation_model", "diarization_model", "speech_engine",
                "voices", "piper_models", "max_speech_speed",
                "min_dialogue_suppression_db", "speaker_similarity",
                "speaker_margin", "speaker_overlap", "language_confidence", "separation_model",
                "separation_model_dir", "residual_word_overlap")
        values = {k: self.data[k] for k in keys}
        # Replacement model assets must not reuse PCM from an older engine.
        assets = []
        roots = [self.resolve(self[k]) for k in
                 ("asr_model", "translation_model", "diarization_model", "separation_model_dir")]
        roots.extend(self.resolve(m if isinstance(m, str) else m["path"]) for m in self["piper_models"])
        for root in roots:
            for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
                if path.is_file() and ".cache" not in path.parts:
                    stat = path.stat()
                    assets.append((str(path), stat.st_size, stat.st_mtime_ns))
        values["assets"] = assets
        from . import __version__
        values["version"] = __version__
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def public(self):
        return {
            "device": self["device"], "speech_engine": self["speech_engine"],
            "segment_seconds": self["segment_seconds"], "auto_dub": self["auto_dub"],
            "delivery_mode": self["delivery_mode"],
            "plex_configured": bool(self["plex"]["url"] and self["plex"]["token"]),
            "plex_client_selected": bool(self["plex"]["client_id"]),
            "publish_configured": bool(self["publish_dir"] and self["plex_publish_section"]),
        }
