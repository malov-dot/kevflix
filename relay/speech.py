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
