from __future__ import annotations

import copy
import hashlib
import json
import os
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

from .ct2_policy import CUDA_COMPUTE_TYPE, POLICY_ID

# Movie CUDA CTranslate2 path (ASR + translation): the compute type is fixed by relay/ct2_policy.py. These settings
# keys may only be unset or exactly the policy value; int8/int8_float16/auto/float32 and any CPU fallback are refused.
# Explicit CPU tooling (e.g. the cinematic/subtitle CPU helpers) has its own settings and is not governed here.
CT2_COMPUTE_KEYS = ("ct2_compute_type", "compute_type", "asr_compute_type", "translation_compute_type")
# Translators of the Whisper-transcribe movie path (relay/language.py TRANSLATION_ENGINES must list the same names).
TRANSLATION_ENGINES = ("m2m100", "qwen3")
# Weight precision of the qwen3 translator (relay/language.py TRANSLATION_QUANTIZATIONS must list the same names).
QUANTIZATION_MODES = ("none", "int8")

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
    # v2 GPU-first: the normal movie path runs its analysis (ASR, diarization/embeddings, translation) and actor
    # models on CUDA inside lease-owned children. 'cuda' is the only accepted value; the obsolete CPU-only values are
    # migrated (gpu_first_migration). gpu_min_free_mib is read only by the legacy worker's choose_device.
    "device": "cuda", "gpu_min_free_mib": 10000,
    "sem_status_file": "",
    "asr_model": "./models/faster-whisper-small",
    "asr_caption_gap_rescue": True,
    "translation_model": "./models/m2m100_418M",
    "translation_ct2_model": "",
    # Translator of the Whisper-transcribe movie path: 'm2m100' (default; every existing settings file) or 'qwen3'
    # (Qwen3-8B bf16 on CUDA, ~16 GB; translation_llm_model names the local folder, which is never downloaded here).
    "translation_engine": "m2m100", "translation_llm_model": "",
    # AudioSet event classifier folder (relay/audio_events.py); "" = no classifier, the nonverbal rules use text and pitch only.
    "audio_event_model": "",
    # qwen3 only: 'none' keeps the bf16 weights (default, ~16.4 GB resident); 'int8' quantizes the decoder linears to
    # weight-only INT8 while the same local bf16 safetensors are loaded (~9.4 GB resident). Ignored with m2m100.
    "translation_quantization": "none",
    "diarization_model": "./models/pyannote-community-1",
    "speaker_onnx_model": "",
    "overlap_checkpoint": "", "overlap_code_root": "", "overlap_dependency_root": "",
    "speaker_similarity": 0.70, "speaker_margin": 0.06,
    "language_confidence": 0.65, "speaker_overlap": 0.55,
    "speech_engine": "sapi", "voices": [],
    "generic_voice_model": "./models/piper/en_US-libritts_r-medium.onnx",
    "generic_speaker_id": 0, "generic_device": "cuda", "generic_source_language": None,
    "generic_max_speed": 1.35, "generic_max_lag_seconds": 2.0, "generic_early_seconds": .3,
    "generic_context_seconds": 6.0, "generic_prebuffer_seconds": 120.0,
    "generic_producer_queue": 4, "generic_readback": False,
    "xt_wsl_distro": "Ubuntu-24.04", "xt_source_language": "auto",
    "xt_python": "", "xt_asr_python": "", "xt_mt_python": "", "xt_tts_python": "",
    "xt_vendor_dir": str(Path(__file__).resolve().parent / "vendor" / "xtranslator"),
    "xt_asr_model": "", "xt_mt_model": "", "xt_mt_tokenizer": "", "xt_xvoice_ckpt": "", "xt_rate_ckpt": "",
    "xt_mt_format": "gguf-q8_0", "xt_mt_dtype": "half", "xt_mt_gguf_overhead_mib": 1536,
    "xt_vocab": "", "xt_vocos_dir": "", "xt_campplus_model": "",
    "xt_nltk_data": "",
    "xt_espeak_library": "", "xt_espeak_data": "",
    "xt_asr_port": 18791, "xt_mt_port": 18792, "xt_tts_port": 18793, "xt_app_port": 18794,
    "xt_asr_max_model_len": 8192, "xt_asr_kv_cache_mib": 1024,
    "xt_mt_max_model_len": 4096, "xt_mt_kv_cache_mib": 768,
    "xt_vllm_max_num_seqs": 1, "xt_vllm_enforce_eager": True,
    "xt_vllm_overhead_mib": 1024, "xt_xvoice_allowance_mib": 2048,
    "xt_campplus_allowance_mib": 256, "xt_demucs_growth_allowance_mib": 0,
    "xt_start_timeout_seconds": 600, "xt_idle_seconds": 150, "xt_request_timeout_seconds": 900,
    "xt_supervisor_first_status_seconds": 90,
    "xt_asr_request_timeout_seconds": 90,
    "xt_mt_request_timeout_seconds": 90,
    "xt_barrier_timeout_seconds": 120,
    "xt_max_session_resets": 3,
    "xt_lead_seconds": 60, "xt_extend_max": 2, "xt_seek_pad_seconds": 14.3,
    "xt_model_manifest": "", "xt_runtime_receipt": "",
    "actor_runtime_manifest": "", "actor_backend": "cuda",
    "actor_reference_seconds": 8, "actor_reserve_gpu_mib": 3072,
    "actor_required_gpu_mib": 2700, "resource_status_files": [],
    "resource_process_names": [],
    "resource_require_process_absence": False,
    "resource_coordination_file": "", "resource_hold_file": "",
    "cinematic_checkpoint": "", "cinematic_code_root": "",
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
           "hdr_mode": "auto", "surround": "auto", "pc_process_all": True, "sync_plex_progress": False,
           "min_free_disk_gib": 10,
           "idle_seconds": 45, "allow_origins": ["http://127.0.0.1:8081", "http://localhost:8081"],
           "allow_plex_controls": False,
           # One constant -23 LUFS gain per dubbed movie (owner decision 2026-10-19); set {"enabled": false} to turn it off.
           "loudness": {"enabled": True},
           # No-stall playback (owner decision 2026-10-19): dub + video are prepared toward the end of the film whether or
           # not playback is paused; playback starts only when it is safe not to stall. See producer_policy.py.
           "mux_ahead_unlimited": True, "mux_ahead_workers": 2, "mux_ahead_disk_margin_gib": 5,
           "adaptive_start_gate": True, "actor_beta_start_seconds": 60, "start_gate_fast_rho": 1.25,
           "low_water_seconds": 10, "rho_window_chunks": 6, "actor_beta_producer_queue": 4},
    "updates": {"enabled": True, "repository": "malov-dot/kevflix",
                "branch": "main", "path": "tv", "interval_seconds": 3600,
                "token_env": "MOVIEDUB_GITHUB_TOKEN"},
    # Persistent profile data (fix 4): actor references owned by the app; '' = <config folder>/actor-bank.
    "actor_bank_dir": "",
    # Movie GPU lease held by the app guard (relay/app_runtime). sem_mode is explicit: 'absent' when Sem is shut
    # down for movies, 'running' (needs status_file) when Sem stays up and must be proved idle.
    # budget_mib 'auto' (v2): fresh free VRAM - reserve_mib - admission_margin_mib at acquire, capped by an optional
    # integer budget_cap_mib (relay/app_runtime/movie_lease.py); an explicit integer budget_mib is honoured.
    "gpu_lease": {"sem_mode": "absent", "status_file": "", "reserve_mib": 3072, "budget_mib": "auto",
                  "admission_margin_mib": 512, "budget_cap_mib": None, "grace_s": 0.15, "retry_seconds": 30},
}
# The earlier app default for gpu_lease.budget_mib (Demucs/Vox-only design); only this exact value is migrated.
OLD_DEFAULT_BUDGET_MIB = 6144


def gpu_first_migration(raw):
    """Convert the obsolete CPU-only movie settings of a raw settings dict IN PLACE to the GPU-first equivalents and
    return the list of changes ({'key', 'from', 'to'}). Only these app-owned keys are touched: device,
    actor_backend, actor_beta.demucs_device, gpu_lease.budget_mib (the old 6144 default only). Plex/account, media,
    UI, LAN, startup and personal data are never read or changed; nothing is reset from config.example.json."""
    changes = []
    if not isinstance(raw, dict):
        return changes
    if raw.get("device") in ("cpu", "cooperative-cuda"):
        changes.append({"key": "device", "from": raw["device"], "to": "cuda"})
        raw["device"] = "cuda"
    if raw.get("actor_backend") == "cpu":
        changes.append({"key": "actor_backend", "from": "cpu", "to": "cuda"})
        raw["actor_backend"] = "cuda"
    beta = raw.get("actor_beta")
    if isinstance(beta, dict) and beta.get("demucs_device") == "cpu" \
            and beta.get("first_pass_admission", "movie-lease") == "movie-lease":
        changes.append({"key": "actor_beta.demucs_device", "from": "cpu", "to": "cuda"})
        beta["demucs_device"] = "cuda"
    lease = raw.get("gpu_lease")
    if isinstance(lease, dict) and not isinstance(lease.get("budget_mib"), bool) \
            and lease.get("budget_mib") == OLD_DEFAULT_BUDGET_MIB:
        changes.append({"key": "gpu_lease.budget_mib", "from": OLD_DEFAULT_BUDGET_MIB, "to": "auto"})
        lease["budget_mib"] = "auto"
    return changes


def _mib(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError("%s must be a %s integer MiB" % (name, "non-negative" if allow_zero else "positive"))
    return value
APP_ROOT = Path(__file__).resolve().parent.parent
LEASE_CLIENT = Path(__file__).resolve().parent / "app_runtime" / "lease" / "lease_client.py"
LEASE_CLIENT_SHA256 = "5cbac8a4d33c114bf811bab50e5957142dfa46fa8e749ab482bf78749fefcd29"
VOX_SOURCE_FILES = ("automatic-v1/automatic_renderer.py", "reference-cache/reference_cache.py")


def _interpreter(value, data, key):
    """A configured interpreter path that exists is kept as is; otherwise this process's interpreter is used and the
    reason is recorded in data['python_resolution'] (never an empty or missing executable)."""
    import sys
    if isinstance(value, str) and value.strip():
        path = Path(os.path.expandvars(value.strip())).expanduser()
        if path.is_file():
            return str(path)
        data.setdefault("python_resolution", {})[key] = "configured interpreter %s not found; using %s" % (path, sys.executable)
    elif key == "actor_beta.vox.python" and data.get("python"):
        return str(data["python"])                 # W3 rule: vox.python, else top-level python
    return str(Path(sys.executable))


def vox_source_paths(app_root=APP_ROOT):
    """W3's pins rule over the APP TREE: automatic_renderer, reference_cache and every runtime-adapter/*.py."""
    root = Path(app_root) / "actor-tts"
    return [root / n for n in VOX_SOURCE_FILES] + sorted((root / "runtime-adapter").glob("*.py"))


class Config:
    def __init__(self, filename):
        self.file = Path(filename).resolve()
        raw = json.loads(self.file.read_text(encoding="utf-8-sig"))
        # GPU-first: an un-migrated file is read with the same conversion the startup migration persists
        # (relay/app_runtime/preflight.py); the file itself is never rewritten here.
        migrated = gpu_first_migration(raw)
        self.data = copy.deepcopy(DEFAULTS)
        self.data.update(raw)
        self.data["plex"] = {**copy.deepcopy(DEFAULTS["plex"]), **raw.get("plex", {})}
        self.data["tv"] = {**copy.deepcopy(DEFAULTS["tv"]), **raw.get("tv", {})}
        self.data["updates"] = {**copy.deepcopy(DEFAULTS["updates"]), **raw.get("updates", {})}
        for key in CT2_COMPUTE_KEYS:
            if raw.get(key) not in (None, "", CUDA_COMPUTE_TYPE):
                raise ValueError('%s must be "%s" or unset (CT2 policy %s): the movie CUDA path has no int8/auto '
                                 'compute type and no CPU fallback' % (key, CUDA_COMPUTE_TYPE, POLICY_ID))
            self.data.pop(key, None)
        self.data["ct2_compute_type"] = CUDA_COMPUTE_TYPE       # fixed by policy; never a user choice
        # The raw configured interpreter setting (before the fallback below), so the startup preflight resolves the
        # analysis child's interpreter as the SERVER process will, not as this (possibly bootstrap) process does.
        self.python_setting = raw.get("python") if isinstance(raw.get("python"), str) else ""
        # Engine python: an explicitly configured, existing interpreter is kept; empty/missing -> this app's own
        # interpreter (the installed .venv / bundled runtime running the server), never an empty executable.
        self.data["python"] = _interpreter(self.data.get("python"), self.data, "python")
        self.data["gpu_lease"] = {**DEFAULTS["gpu_lease"], **raw.get("gpu_lease", {})}
        if self.data["gpu_lease"]["sem_mode"] not in ("running", "absent"):
            raise ValueError('gpu_lease.sem_mode must be "running" or "absent"')
        if self.data["gpu_lease"]["sem_mode"] == "running" and not self.data["gpu_lease"]["status_file"]:
            raise ValueError('gpu_lease.sem_mode "running" needs gpu_lease.status_file')
        if int(self.data["gpu_lease"]["reserve_mib"]) < 3072:
            raise ValueError("gpu_lease.reserve_mib must be at least 3072")
        if self.data["gpu_lease"]["budget_mib"] in (None, "", "auto"):
            self.data["gpu_lease"]["budget_mib"] = "auto"
        elif isinstance(self.data["gpu_lease"]["budget_mib"], bool) or \
                not isinstance(self.data["gpu_lease"]["budget_mib"], int) or self.data["gpu_lease"]["budget_mib"] <= 0:
            raise ValueError('gpu_lease.budget_mib must be "auto" or a positive integer MiB')
        if self.data["gpu_lease"]["budget_cap_mib"] not in (None, ""):
            _mib(self.data["gpu_lease"]["budget_cap_mib"], "gpu_lease.budget_cap_mib")
        _mib(self.data["gpu_lease"]["admission_margin_mib"], "gpu_lease.admission_margin_mib", allow_zero=True)
        if migrated:
            self.data["gpu_first_migration"] = migrated      # in-memory record; the guard persists the same change
        beta = self.data.get("actor_beta")
        if isinstance(beta, dict) and beta.get("enabled") is True:
            # W10 (INTERFACES "Config keys read by W10 code"): the in-server MoviePipeline, the resident first pass
            # admitted only by the movie lease, GPU Demucs. Defaults only; a user value is kept.
            beta = self.data["actor_beta"] = dict(beta)
            pipeline = dict(beta.get("movie_pipeline") or {})
            pipeline.setdefault("enabled", True)
            pipeline.setdefault("queues", {"separation": 3, "analysis": 1})
            pipeline.setdefault("poll_seconds", 0.5)
            beta["movie_pipeline"] = pipeline
            beta.setdefault("first_pass_admission", "movie-lease")
            beta.setdefault("demucs_device", "cuda")
            if beta.get("demucs_device") == "cuda" and beta.get("first_pass_admission") != "movie-lease":
                raise ValueError('actor_beta.demucs_device "cuda" is accepted only with first_pass_admission "movie-lease"')
            if beta.get("actor_engine", "vox") == "vox":
                beta.setdefault("actor_engine", "vox")
                beta["vox"] = dict(beta.get("vox") or {})
        if isinstance(beta, dict) and isinstance(beta.get("vox"), dict):
            # W3 fix 6: unset = 'auto' (measured-peak admission, capped at 2); an explicit int (1 included) is kept.
            # The lease client is the bundled one.
            vox = beta["vox"] = dict(beta["vox"])
            if vox.get("max_instances") in (None, ""):
                vox["max_instances"] = "auto"
            vox.setdefault("max_instances_cap", 2)
            vox["python"] = _interpreter(vox.get("python"), self.data, "actor_beta.vox.python")
            lease = vox["gpu_lease"] = dict(vox.get("gpu_lease") or {})
            lease.setdefault("client", str(LEASE_CLIENT))
            lease.setdefault("client_sha256", LEASE_CLIENT_SHA256)
            lease.setdefault("sem_mode", self.data["gpu_lease"]["sem_mode"])
            lease.setdefault("reserve_mib", int(self.data["gpu_lease"]["reserve_mib"]))
        if not 1 <= int(self.data["tv"]["encoder_threads"]) <= 8:
            raise ValueError("TV encoder threads must be between 1 and 8")
        if int(self.data["tv"]["max_height"]) not in (480, 720, 1080, 2160):
            raise ValueError("TV output height must be 480, 720, 1080 or 2160")
        if self.data["tv"]["video_encoder"] not in ("auto", "nvenc", "software"):
            raise ValueError("TV video_encoder must be auto, nvenc or software")
        # Owner decision 2026-10-19: everything on the GPU where possible. 'auto' picks NVENC when ffmpeg has h264_nvenc and
        # enough VRAM is free (tv.nvenc_min_free_mib), otherwise software; an NVENC failure retries in software (tv.py).
        for key, env in (("api_token", "MOVIEDUB_API_TOKEN"),
                         ("nas_token", "MOVIEDUB_NAS_TOKEN"),
                         ("stream_secret", "MOVIEDUB_STREAM_SECRET"),
                         ("webhook_secret", "MOVIEDUB_WEBHOOK_SECRET")):
            self.data[key] = os.environ.get(env, self.data[key])
        self.data["plex"]["token"] = os.environ.get("MOVIEDUB_PLEX_TOKEN",
                                                    self.data["plex"]["token"])
        # A null/empty engine means "unset" (the default), like the CT2 keys above; any other value must be exact.
        if self.data["translation_engine"] in (None, ""):
            self.data["translation_engine"] = "m2m100"
        if self.data["translation_llm_model"] is None:
            self.data["translation_llm_model"] = ""
        if self.data["audio_event_model"] is None:
            self.data["audio_event_model"] = ""
        if not isinstance(self.data["audio_event_model"], str):
            raise ValueError("audio_event_model must be a folder path")
        if not isinstance(self.data["translation_engine"], str) or \
                self.data["translation_engine"] not in TRANSLATION_ENGINES:
            raise ValueError('translation_engine must be "m2m100" or "qwen3", not %r' % (self.data["translation_engine"],))
        if not isinstance(self.data["translation_llm_model"], str):
            raise ValueError("translation_llm_model must be a folder path")
        # null/empty = unset = "none" (bf16 weights); only "none" and "int8" exist, anything else is refused, never guessed.
        if self.data["translation_quantization"] in (None, ""):
            self.data["translation_quantization"] = "none"
        if not isinstance(self.data["translation_quantization"], str) or \
                self.data["translation_quantization"] not in QUANTIZATION_MODES:
            raise ValueError('translation_quantization must be "none" or "int8", not %r'
                             % (self.data["translation_quantization"],))
        for key in ("state_dir", "cache_dir", "asr_model", "translation_model", "translation_ct2_model",
                    "translation_llm_model", "audio_event_model",
                    "diarization_model", "speaker_onnx_model", "separation_model_dir", "publish_dir",
                    "actor_runtime_manifest", "cinematic_checkpoint", "cinematic_code_root",
                    "resource_coordination_file", "resource_hold_file", "overlap_checkpoint", "overlap_code_root", "overlap_dependency_root"):
            if self.data[key]:
                self.data[key] = str(self.resolve(self.data[key]))
        if self.data["translation_engine"] == "qwen3":
            if not self.data["translation_llm_model"]:
                raise ValueError('translation_engine "qwen3" needs translation_llm_model (a local Qwen3-8B folder)')
            if not Path(self.data["translation_llm_model"]).is_dir():
                raise ValueError("translation_llm_model %s is not an existing folder" % self.data["translation_llm_model"])
        if self.data["audio_event_model"] and not Path(self.data["audio_event_model"]).is_dir():
            raise ValueError("audio_event_model %s is not an existing folder" % self.data["audio_event_model"])
        if not 5 <= int(self.data["segment_seconds"]) <= 120:
            raise ValueError("segment_seconds must be between 5 and 120")
        if self.data["device"] != "cuda":
            raise ValueError('device must be "cuda": the movie path runs its analysis and actor models on the GPU '
                             '(CPU inference is opt-in tooling only)')
        if self.data["speech_engine"] not in ("sapi", "piper", "actor", "generic", "xtranslator"):
            raise ValueError("Choose X-Translator, generic English, actor cloning, sapi or piper for speech")
        if self.data['speech_engine'] in ('generic', 'xtranslator'):
            if self.data['generic_device'] != 'cuda':
                raise ValueError('Generic English inference requires CUDA')
            if isinstance(self.data['generic_speaker_id'], bool) or int(self.data['generic_speaker_id']) < 0:
                raise ValueError('generic_speaker_id must be a nonnegative integer')
            if int(self.data['generic_speaker_id']) != float(self.data['generic_speaker_id']):
                raise ValueError('generic_speaker_id must be integral')
            lang = self.data['generic_source_language']
            if lang in (None, '', 'auto'):
                self.data['generic_source_language'] = None
            elif not isinstance(lang, str) or len(lang) not in (2, 3) or not lang.isalpha():
                raise ValueError('generic_source_language must be a language code or auto')
            else:
                self.data['generic_source_language'] = lang.lower()
            for key, low, high in (('generic_max_speed', 1., 1.35), ('generic_max_lag_seconds', 0., 30.),
                    ('generic_early_seconds', 0., .75), ('generic_context_seconds', 6., 30.),
                    ('generic_prebuffer_seconds', 1., 300.), ('generic_producer_queue', 1., 8.)):
                if not low <= float(self.data[key]) <= high:
                    raise ValueError('%s outside supported range' % key)
            if int(self.data['generic_producer_queue']) != float(self.data['generic_producer_queue']):
                raise ValueError('generic_producer_queue must be integral')
        if self.data['speech_engine'] == 'xtranslator':
            if self.data['xt_mt_format'] != 'gguf-q8_0' or self.data['xt_mt_dtype'] != 'half':
                raise ValueError('X-Translator MT requires Q8_0 GGUF with FP16 computation')
            overhead = self.data['xt_mt_gguf_overhead_mib']
            if isinstance(overhead, bool) or not isinstance(overhead, int) or not 1536 <= overhead <= 4096:
                raise ValueError('xt_mt_gguf_overhead_mib must be 1536..4096 MiB')
            if self.data['xt_source_language'] not in ('auto', 'ru', 'es', 'en'):
                raise ValueError('Choose auto, ru, es or en for X-Translator source language')
            distro = self.data['xt_wsl_distro']
            if distro != 'Ubuntu-24.04':
                raise ValueError('This X-Translator integration uses the isolated Ubuntu-24.04 distribution')
            ports = [self.data[k] for k in ('xt_asr_port', 'xt_mt_port', 'xt_tts_port', 'xt_app_port')]
            if any(isinstance(p, bool) or not isinstance(p, int) or not 1024 <= p <= 65535 for p in ports) or len(set(ports)) != 4:
                raise ValueError('X-Translator needs four distinct unprivileged local ports')
            for key, low, high in (('xt_lead_seconds', 30, 600), ('xt_extend_max', 1, 8),
                                  ('xt_seek_pad_seconds', 14.3, 60), ('xt_idle_seconds', 30, 1800),
                                  ('xt_start_timeout_seconds', 30, 1800),
                                  ('xt_supervisor_first_status_seconds', 30, 120),
                                  ('xt_asr_request_timeout_seconds', 30, 90),
                                  ('xt_mt_request_timeout_seconds', 30, 90),
                                  ('xt_barrier_timeout_seconds', 30, 120),
                                  ('xt_request_timeout_seconds', 30, 3600)):
                if not low <= float(self.data[key]) <= high:
                    raise ValueError('%s outside supported range' % key)
            if int(self.data['xt_extend_max']) != float(self.data['xt_extend_max']):
                raise ValueError('xt_extend_max must be integral')
            if float(self.data['xt_request_timeout_seconds']) < float(self.data['xt_barrier_timeout_seconds']):
                raise ValueError('The X-Translator overall deadline must cover its no-progress timeout')
            resets = self.data['xt_max_session_resets']
            if isinstance(resets, bool) or not isinstance(resets, int) or not 0 <= resets <= 3:
                raise ValueError('xt_max_session_resets must be an integer from zero to three')
            for key, value in (('xt_asr_max_model_len', 8192), ('xt_mt_max_model_len', 4096),
                               ('xt_vllm_max_num_seqs', 1)):
                if isinstance(self.data[key], bool) or not isinstance(self.data[key], int) or self.data[key] != value:
                    raise ValueError('%s is fixed at %d for this single-stream engine' % (key, value))
            if self.data['xt_vllm_enforce_eager'] is not True:
                raise ValueError('X-Translator requires eager inference without CUDA graph reservations')
            for key, low, high in (('xt_asr_kv_cache_mib', 1024, 2048), ('xt_mt_kv_cache_mib', 768, 1536),
                                  ('xt_vllm_overhead_mib', 1024, 4096), ('xt_xvoice_allowance_mib', 2048, 4096),
                                  ('xt_campplus_allowance_mib', 256, 1024), ('xt_demucs_growth_allowance_mib', 0, 4096)):
                number = self.data[key]
                if isinstance(number, bool) or not isinstance(number, int) or not low <= number <= high:
                    raise ValueError('%s outside the supported MiB range' % key)
        if self.data['actor_backend'] != 'cuda':
            raise ValueError('actor_backend must be "cuda" (no CPU actor inference on the movie path)')
        _beta=self.data.get('actor_beta')
        if (isinstance(_beta,dict) and _beta.get('enabled') is True and _beta.get('demucs_first_pass') is True
                and _beta.get('demucs_mix_bed') is True):
            raise ValueError('actor_beta.demucs_first_pass and demucs_mix_bed are mutually exclusive')
        if not 1.0 <= float(self.data['actor_reference_seconds']) <= 15:
            raise ValueError('Actor reference limit must be between 1 and 15 seconds')
        self.data['resource_status_files'] = [str(self.resolve(p)) for p in self.data['resource_status_files']]
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
        self.data["actor_bank_dir"] = str(self.resolve(self.data["actor_bank_dir"] or "./actor-bank"))
        if self.data["gpu_lease"]["status_file"]:
            self.data["gpu_lease"]["status_file"] = str(self.resolve(self.data["gpu_lease"]["status_file"]))
        self.actor_bank = Path(self.data["actor_bank_dir"])

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
        keys = ("segment_seconds", "context_seconds", "asr_model", "asr_caption_gap_rescue",
                "translation_model", "translation_ct2_model", "diarization_model", "speaker_onnx_model", "speech_engine",
                "voices", "piper_models", "max_speech_speed",
                "min_dialogue_suppression_db", "speaker_similarity",
                "speaker_margin", "speaker_overlap", "language_confidence", "separation_model",
                "separation_model_dir", "residual_word_overlap", "actor_runtime_manifest",
                "actor_backend", "actor_reference_seconds", "cinematic_checkpoint", "cinematic_code_root",
                "overlap_checkpoint", "overlap_code_root", "overlap_dependency_root")
        values = {k: self.data[k] for k in keys}
        qwen3 = self.data.get('translation_engine') == 'qwen3'
        if qwen3:
            # Only a qwen3 selection adds to the fingerprint: m2m100 (every existing settings file) keeps its own,
            # while another engine or another Qwen3 folder (its files are listed under "assets") never reuses a dub.
            values['translation_engine'] = 'qwen3'
            values['translation_llm_model'] = self.data['translation_llm_model']
            # int8 weights translate differently from bf16 weights: a dub made with one never answers the other. The
            # default 'none' adds nothing, so fingerprints of existing bf16 qwen3 settings stay exactly as they were.
            if self.data.get('translation_quantization') not in (None, '', 'none'):
                values['translation_quantization'] = self.data['translation_quantization']
        if self['speech_engine'] == 'xtranslator':
            values.pop('language_confidence', None)
        if self['speech_engine'] in ('generic', 'xtranslator'):
            values['generic'] = {k: v for k, v in self.data.items() if k.startswith('generic_')}
            names = ('generic_dub.py', 'generic_policy.py', 'language.py', 'speech.py', 'media.py', 'first_pass_parent.py',
                     'first_pass_child.py', 'first_pass.py', 'first_pass_key.py', 'store.py',
                     'coordinator.py', 'movie_pipeline/analysis_service.py', 'movie_pipeline/analysis_child.py')
            here = Path(__file__).resolve().parent
            values['generic_implementation'] = {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                                 for name in names}
            model = self.resolve(self['generic_voice_model'])
            values['generic_voice_assets'] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                for p in (model, Path(str(model) + '.json')) if p.is_file()]
            if self['speech_engine'] == 'xtranslator':
                values['xtranslator'] = {k: v for k, v in self.data.items() if k.startswith('xt_')}
                paths = [here / name for name in ('xt_client.py', 'xt_services.py', 'xt_dub.py',
                                                  'app_runtime/xt_provision.py')]
                vendor = self.resolve(self['xt_vendor_dir'])
                paths.extend(p for p in vendor.rglob('*') if p.is_file() and
                             p.suffix in ('.py', '.json', '.toml', '.yaml', '.yml', '.sh'))
                values['xt_implementation'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                for key in ('xt_model_manifest', 'xt_runtime_receipt'):
                    value = self[key]
                    path = self.resolve(value) if value else None
                    values[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
        # Replacement model assets must not reuse PCM from an older engine.
        assets = []
        roots = [self.resolve(self[k]) for k in
                 ("asr_model", "translation_model", "diarization_model", "separation_model_dir")]
        roots.extend(self.resolve(m if isinstance(m, str) else m["path"]) for m in self["piper_models"])
        roots.extend(self.resolve(self[k]) for k in ('actor_runtime_manifest','cinematic_checkpoint') if self[k])
        if self['translation_ct2_model']:roots.append(self.resolve(self['translation_ct2_model']))
        if qwen3:roots.append(self.resolve(self['translation_llm_model']))
        if self['audio_event_model']:roots.append(self.resolve(self['audio_event_model']))
        if self['speaker_onnx_model']:roots.append(self.resolve(self['speaker_onnx_model']))
        if self['overlap_checkpoint']:roots.append(self.resolve(self['overlap_checkpoint']))
        for root in roots:
            for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
                if path.is_file() and ".cache" not in path.parts:
                    stat = path.stat()
                    assets.append((str(path), stat.st_size, stat.st_mtime_ns))
        values["assets"] = assets
        if self['speech_engine']=='actor':
            # Staged code may change before a release version is assigned.
            # Do not reuse canonical speech/chunks from a different algorithm.
            names=('actor_pipeline','actor_voice','actor_render','actor_client','actor_child',
                   'actor_reference','cinematic','dialogue','dialogue_units','dialogue_ledger','language',
                   'speakers','speaker_onnx','overlap','timing','dubbing_timing','prepared_actor',
                   'verified_speech','progressive_actor','live_mux','asr_rescue',
                   'original_audio_policy','short_utterance_policy','media','store',
                   # owner-approved dubbing behaviour (2026-10-19): speed fit, speech-only removal, non-speech lines
                   'take_fit','speech_activity','nonverbal','loudness_reference','audio_evidence','audio_events','borrowed_reference','take_deess','actor_surround')
            values['actor_implementation']={name:hashlib.sha256(Path(__file__).with_name(name+'.py').read_bytes()).hexdigest()
                                            for name in names}
            # embedded-subtitles-v1: caption selection/extraction changes chunk captions.
            captions=Path(__file__).with_name('embedded_subtitles.py')
            if captions.is_file():
                values['caption_implementation']={'embedded_subtitles':hashlib.sha256(captions.read_bytes()).hexdigest()}
            # B combined-caption-firstpass-v1 helper files (firstpass-runtime-glue-v2); missing -> None.
            b_caption=('actor_beta_timing.py', 'beta_caption_authority.py', 'beta_caption_flow.py', 'beta_caption_gate.py', 'beta_caption_planner.py', 'beta_caption_provenance.py', 'beta_first_pass_flow.py', 'beta_stable_gate.py')
            for name in b_caption:
                helper=Path(__file__).with_name(name)
                values.setdefault('caption_implementation',{})[name]=hashlib.sha256(helper.read_bytes()).hexdigest() if helper.is_file() else None
        from . import __version__
        values["version"] = __version__
        beta = self.data.get("actor_beta")
        if isinstance(beta, dict) and beta.get("enabled") is True:
            here = Path(__file__).resolve().parent
            sources = [here / n for n in ("actor_beta_hook.py", "beta_policy_validator.py", "subtitle_runtime.py",
                                          "actor_beta_timing.py")]
            sources += vox_source_paths(here.parent)          # app tree, never a development folder
            if beta.get("demucs_mix_bed") is True:
                sources += [here / n for n in ("demucs_bed_parent.py", "demucs_bed_child.py", "demucs_bed.py")]
            if beta.get("demucs_first_pass") is True:
                # Same set first_pass_key.CODE_FILES binds into every first-pass bundle key.
                sources += [here / n for n in ('first_pass_parent.py', 'first_pass_child.py', 'first_pass.py', 'first_pass_key.py', 'demucs_bed.py')]
            # Paths relative to the app root, so a moved install keeps its fingerprint (and its completed dubs).
            values["actor_beta"] = {"options": beta, "sources": {
                (p.relative_to(here.parent).as_posix() if here.parent in p.parents else str(p)):
                hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None for p in sources}}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    # ---- fix 4: reuse of completed dubs only when media + engine + config + format all match ------------------------
    FORMAT_KEYS = ("segment_seconds", "context_seconds")
    TV_FORMAT_KEYS = ("segment_seconds", "max_height", "video_encoder", "upscale", "hdr_mode")

    def format_fingerprint(self):
        """Output format: segment geometry, TV output settings and the FFmpeg build (by size/mtime) used to mux."""
        values = {k: self.data[k] for k in self.FORMAT_KEYS}
        values["tv"] = {k: self.data["tv"].get(k) for k in self.TV_FORMAT_KEYS}
        tools = {}
        for key in ("ffmpeg", "ffprobe"):
            path = Path(os.path.expandvars(str(self.data[key])))
            if not path.is_absolute() and ("/" in str(self.data[key]) or "\\" in str(self.data[key])):
                path = self.resolve(self.data[key])
            if path.is_file():
                stat = path.stat()
                tools[key] = [path.name, stat.st_size, stat.st_mtime_ns]
            else:
                tools[key] = [str(self.data[key]), None, None]      # a PATH command: identified by name only
        values["tools"] = tools
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def config_fingerprint(self):
        """Processing-relevant settings that are not already in the engine fingerprint."""
        keys = ("device", "delivery_mode", "lookahead_segments", "auto_dub")
        values = {k: self.data.get(k) for k in keys}
        values["actor_beta"] = self.data.get("actor_beta")
        return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()

    def dub_reuse_key(self, media_fingerprint, *, audio_index=None, subtitle_fingerprint=None):
        """The single key under which a completed dub may be reused: media + engine + config + format. Any change in
        one of them gives a new key, so stale dubs are never served; matching ones are reused across restarts."""
        parts = {"media": media_fingerprint, "audio_index": audio_index, "subtitles": subtitle_fingerprint,
                 "engine": self.engine_fingerprint(), "config": self.config_fingerprint(),
                 "format": self.format_fingerprint()}
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def public(self):
        return {
            "device": self["device"], "speech_engine": self["speech_engine"],
            "segment_seconds": self["segment_seconds"], "auto_dub": self["auto_dub"],
            "delivery_mode": self["delivery_mode"],
            "plex_configured": bool(self["plex"]["url"] and self["plex"]["token"]),
            "plex_client_selected": bool(self["plex"]["client_id"]),
            "publish_configured": bool(self["publish_dir"] and self["plex_publish_section"]),
        }
