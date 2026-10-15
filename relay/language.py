from __future__ import annotations

import re
from pathlib import Path

ISO = {"eng": "en", "spa": "es", "fra": "fr", "fre": "fr", "deu": "de",
       "ger": "de", "jpn": "ja", "zho": "zh", "chi": "zh", "kor": "ko",
       "ita": "it", "por": "pt", "rus": "ru", "ara": "ar", "hin": "hi",
       "nld": "nl", "dut": "nl", "pol": "pl", "swe": "sv", "tur": "tr"}


def iso(code):
    value = str(code or "").lower().replace("_", "-").split("-")[0]
    return ISO.get(value, value)


class LanguageTools:
    def __init__(self, config, device):
        from lingua import LanguageDetectorBuilder
        self.detector = LanguageDetectorBuilder.from_all_languages().build()
        self.config = config
        self.device = device
        self.asr = None
        self.translator = None
        self.tokenizer = None

    def text_language(self, text, declared=None):
        declared = iso(declared)
        if declared and declared not in ("und", "unknown"):
            return declared, 1.0, "subtitle-metadata"
        values = self.detector.compute_language_confidence_values(str(text))
        if not values:
            return "und", 0.0, "text-detection"
        best = values[0]
        return best.language.iso_code_639_1.name.lower(), float(best.value), "text-detection"

    def speech(self, wav, language=None):
        if self.asr is None:
            from faster_whisper import WhisperModel
            path = self.config["asr_model"]
            if not Path(path).is_dir():
                raise ValueError("Multilingual ASR assets are missing; run the separate model setup step")
            self.asr = WhisperModel(path, device=self.device,
                                   compute_type="int8_float16" if self.device == "cuda" else "int8",
                                   cpu_threads=int(self.config["cpu_threads"]),
                                   local_files_only=True)
        segments, info = self.asr.transcribe(
            str(wav), language=iso(language) or None, task="transcribe", beam_size=5,
            vad_filter=True, word_timestamps=True, condition_on_previous_text=False)
        values = [{"start": float(x.start), "end": float(x.end), "text": x.text.strip(),
                   "log_probability": float(x.avg_logprob),
                   "no_speech_probability": float(x.no_speech_prob),
                   "words": [{"start": float(w.start), "end": float(w.end),
                              "text": w.word, "probability": float(w.probability)}
                             for w in (x.words or [])]}
                  for x in segments if x.text.strip()]
        return values, {"language": iso(info.language),
                        "probability": float(info.language_probability)}

    def english(self, text, language):
        language = iso(language)
        if language == "en":
            return text, "already-English"
        if not language or language in ("und", "unknown"):
            raise ValueError("The source language has not been resolved")
        if self.translator is None:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            path = self.config["translation_model"]
            if not Path(path).is_dir():
                raise ValueError("Local multilingual translation assets are missing")
            self.tokenizer = M2M100Tokenizer.from_pretrained(path, local_files_only=True)
            self.translator = M2M100ForConditionalGeneration.from_pretrained(
                path, local_files_only=True).eval()
        if language not in self.tokenizer.lang_code_to_id:
            raise ValueError(f"Local translation does not support detected language {language}")
        import torch
        self.tokenizer.src_lang = language
        encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[-1] > 900:
            raise ValueError("Subtitle line is too long for one translation; split it before dubbing")
        with torch.inference_mode():
            result = self.translator.generate(
                **encoded, forced_bos_token_id=self.tokenizer.get_lang_id("en"),
                max_new_tokens=512, num_beams=5)
        output = self.tokenizer.batch_decode(result, skip_special_tokens=True)[0].strip()
        if not output:
            raise ValueError("Translation returned no English text")
        return output, "m2m100-to-English"


def words(text):
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


def word_error(expected, heard):
    a, b = words(expected), words(heard)
    if not a:
        return 0.0 if not b else 1.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / len(a)
