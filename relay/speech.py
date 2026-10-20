from __future__ import annotations

import hashlib
import json
import os
import sys
import wave
from pathlib import Path

from .media import atomic_json, wav_info, RATE


class Speech:
    def __init__(self, config, owner):
        self.config, self.owner = config, owner
        self.loaded = {}
        self.sapi_script = Path(__file__).resolve().parent.parent / "legacy/scripts/render_sapi.ps1"

    def pool(self):
        if self.config["speech_engine"] == "sapi":
            if os.name != "nt":
                raise ValueError("SAPI is available only on Windows; configure Piper here")
            output = self.owner.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                                     "-ExecutionPolicy", "Bypass", "-File",
                                     self.sapi_script, "-Mode", "voices"], timeout=30)
            available = [x.strip() for x in output.decode("utf-8-sig").splitlines() if x.strip()]
            requested = self.config["voices"] or ["Microsoft David Desktop", "Microsoft Zira Desktop"]
            if any(x not in available for x in requested):
                raise ValueError("A configured English SAPI voice is not installed")
            return [{"engine": "sapi", "name": x} for x in requested]
        pool = []
        for entry in self.config["piper_models"]:
            entry = {"path": entry} if isinstance(entry, str) else entry
            path = self.config.resolve(entry["path"])
            metadata = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
            code = metadata.get("language", {}).get("code", "")
            if not code.lower().startswith("en"):
                raise ValueError("Movie Dub output requires an English Piper voice")
            count = int(metadata.get("num_speakers", 1))
            ids = entry.get("speakers", list(range(count)))
            for ident in ids:
                if not 0 <= int(ident) < count:
                    raise ValueError("Piper speaker ID is outside the model's voice range")
                pool.append({"engine": "piper", "path": str(path),
                             "speaker_id": int(ident), "name": f"{path.stem} #{ident}"})
        if not pool:
            raise ValueError("No English voice models are configured")
        return pool

    def render(self, text, voice, directory, window):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps([text, voice, window, self.config["max_speech_speed"]],
                                       sort_keys=True).encode()).hexdigest()
        result = directory / (key + ".wav")
        receipt = directory / (key + ".json")
        if result.exists() and receipt.exists():
            from .media import file_hash
            saved = json.loads(receipt.read_text())
            if file_hash(result) == saved.get("sha256"):
                return result, saved
        raw = directory / (key + ".raw.wav")
        if voice["engine"] == "sapi":
            spec = directory / (key + ".sapi.json")
            atomic_json(spec, [{"out": str(raw), "voice": voice["name"], "text": text,
                               "pitch": 0, "rate": 0}])
            self.owner.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                            "Bypass", "-File", self.sapi_script, "-Mode", "render",
                            "-JobsFile", spec], timeout=180)
        else:
            from piper.voice import PiperVoice
            from piper.config import SynthesisConfig
            path = voice["path"]
            if path not in self.loaded:
                self.loaded[path] = PiperVoice.load(path, use_cuda=False)
            synth = self.loaded[path]
            synthesis = SynthesisConfig(speaker_id=voice["speaker_id"])
            with wave.open(str(raw), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(synth.config.sample_rate)
                for chunk in synth.synthesize(text, syn_config=synthesis):
                    self.owner.check()
                    output.writeframes(chunk.audio_int16_bytes)
        duration = wav_info(raw)["seconds"]
        if duration <= .05:
            raise ValueError("Speech synthesis returned no usable audio")
        needed_speed = max(1.0, duration / max(.15, window))
        applied_speed = min(needed_speed, float(self.config["max_speech_speed"]))
        self.owner.run([self.config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-i", raw, "-af", f"atempo={applied_speed:.6f}", "-ar", RATE,
                        "-ac", 1, "-c:a", "pcm_s16le", "-y", result], timeout=90)
        from .media import file_hash
        meta = {"engine": voice["engine"], "voice": voice, "text": text,
                "seconds": wav_info(result)["seconds"], "speed": applied_speed,
                "timing_overrun": needed_speed > self.config["max_speech_speed"] * 1.03,
                "sha256": file_hash(result)}
        atomic_json(receipt, meta)
        return result, meta


class GenericSpeech:
    """Full-length Piper PCM, instantiated only inside the lease-owned analysis child."""
    def __init__(self, config):
        from .movie_faults import MovieConfigurationError
        from .media import file_hash
        self.config, self.loaded = config, None
        self.path = Path(config.resolve(config['generic_voice_model'])).resolve()
        self.speaker_id = config['generic_speaker_id']
        if config['generic_device'] != 'cuda':
            raise MovieConfigurationError('generic_device_invalid', 'Generic Piper requires CUDA; CPU is not substituted')
        if self.path.name != 'en_US-libritts_r-medium.onnx' or self.speaker_id != 0 \
                or not isinstance(self.speaker_id, int) or isinstance(self.speaker_id, bool):
            raise MovieConfigurationError('generic_voice_invalid', 'Generic baseline requires libritts_r medium speaker 0')
        try:
            self.metadata_path = Path(str(self.path) + '.json')
            metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
            self.model_sha256 = file_hash(self.path)
            self.metadata_sha256 = file_hash(self.metadata_path)
            self.sample_rate = int(metadata['audio']['sample_rate'])
            if not str(metadata['language']['code']).lower().startswith('en') or self.sample_rate <= 0 \
                    or not 0 <= self.speaker_id < int(metadata.get('num_speakers', 1)):
                raise ValueError('Piper metadata is not an English voice with this speaker/sample rate')
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise MovieConfigurationError('generic_voice_assets_invalid', str(error)) from error

    def identity(self):
        return {'kind': 'generic-piper', 'path': str(self.path), 'model_sha256': self.model_sha256,
                'metadata_sha256': self.metadata_sha256, 'speaker_id': self.speaker_id,
                'sample_rate': self.sample_rate, 'device': 'cuda', 'loaded': self.loaded is not None,
                'providers': self.loaded.session.get_providers() if self.loaded is not None else []}

    def _providers(self):
        from .movie_faults import MovieConfigurationError
        providers = self.loaded.session.get_providers()
        if not providers or providers[0] != 'CUDAExecutionProvider':
            raise MovieConfigurationError('generic_piper_provider_failed', 'Piper actual providers: %r; CUDA required' % providers)
        return list(providers)

    def synthesize(self, text, voice, output):
        from .language import backend_fault
        from .media import file_hash
        from .movie_faults import MovieConfigurationError, MovieContentFallback, MovieFault
        if not isinstance(text, str) or not text.strip():
            raise MovieContentFallback('generic_tts_empty_text', 'No English text to synthesize')
        if len(text) > 50000:
            raise MovieConfigurationError('generic_tts_text_too_large', 'One synthesis request exceeds 50000 characters')
        if not isinstance(voice, dict) or voice.get('engine') != 'generic-piper' \
                or voice.get('speaker_id') != self.speaker_id or not isinstance(voice.get('speaker_id'), int) \
                or isinstance(voice.get('speaker_id'), bool) \
                or Path(voice.get('path', self.path)).resolve() != self.path:
            raise MovieConfigurationError('generic_tts_voice_mismatch', 'Request must use the configured fixed generic voice')
        output = Path(output)
        try:
            if self.loaded is None:
                import onnxruntime as ort
                from piper.voice import PiperVoice
                if 'CUDAExecutionProvider' not in ort.get_available_providers():
                    raise MovieConfigurationError('generic_piper_cuda_unavailable', 'ONNX Runtime CUDA provider is unavailable')
                if file_hash(self.path) != self.model_sha256 or file_hash(self.metadata_path) != self.metadata_sha256:
                    raise MovieConfigurationError('generic_voice_changed', 'Configured Piper assets changed after identification')
                self.loaded = PiperVoice.load(self.path, use_cuda=True, download_dir=output.parent)
                self._providers()
                self.loaded.session.disable_fallback()
            from piper.config import SynthesisConfig
            self._providers()
            frames = 0
            with wave.open(str(output), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                for chunk in self.loaded.synthesize(text, syn_config=SynthesisConfig(speaker_id=self.speaker_id)):
                    self._providers()
                    data = chunk.audio_int16_bytes
                    if len(data) % 2:
                        raise MovieContentFallback('generic_tts_invalid_pcm', 'Piper returned incomplete PCM frames')
                    wav.writeframes(data)
                    frames += len(data) // 2
                    if frames * 2 > 1 << 30:
                        raise MovieConfigurationError('generic_tts_audio_too_large', 'Full take exceeds exchange bound; not truncated')
            if frames <= self.sample_rate * .05:
                raise MovieContentFallback('generic_tts_empty', 'Piper returned no usable full PCM')
            return {'audio': str(output), 'sha256': file_hash(output), 'seconds': frames / self.sample_rate,
                    'sample_rate': self.sample_rate, 'frames': frames, 'channels': 1, 'sample_width': 2,
                    'device': 'cuda', 'providers': self._providers(), 'voice': self.identity(),
                    'speed': 1.0, 'truncated': False, 'pcm_scope': 'full-untruncated-take'}
        except MovieFault:
            output.unlink(missing_ok=True)
            raise
        except Exception as error:
            output.unlink(missing_ok=True)
            raise backend_fault('generic_piper', error, 'cuda') from error
