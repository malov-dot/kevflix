"""AudioSet event labels (singing, laughter, cough, ...) for short excerpts of the separated vocals.

The Audio Spectrogram Transformer `MIT/ast-finetuned-audioset-10-10-0.4593` (BSD-3-Clause, 87 M parameters, 527 AudioSet
classes) is supplied as a local folder named by the setting `audio_event_model` (config.json, preprocessor_config.json,
model.safetensors). KevFLIX never downloads it. Nothing is loaded until the first request, and an unset or unusable folder
means "no events": the nonverbal rules then use only the text and pitch evidence (relay/nonverbal.py).

Runs inside the lease-owned analysis child (relay/movie_pipeline/analysis_child.py, op `audio_events`), on the same CUDA
device as the recognizer. The excerpt is peak-normalized before classification, because the stems are separated at film level.
"""
from __future__ import annotations

import threading
from pathlib import Path

VERSION = 'audio-events-v1'
RATE = 16000
MIN_SECONDS = 0.1
KEEP = 25                       # labels returned per excerpt (highest first)
REQUIRED_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')


def model_problems(folder):
    """Reasons the folder cannot be used (empty list = usable). Pure file checks."""
    if not folder:
        return ['audio_event_model is not set']
    path = Path(folder)
    if not path.is_dir():
        return ['audio_event_model %s is not an existing folder' % folder]
    return ['%s is missing from %s' % (name, folder) for name in REQUIRED_FILES if not (path / name).is_file()]


class AudioEventClassifier:
    def __init__(self, folder, device='cuda:0'):
        self.folder, self.device = str(folder), device
        self._lock = threading.RLock()
        self._model = self._extractor = self._labels = None

    @property
    def available(self):
        return not model_problems(self.folder)

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            problems = model_problems(self.folder)
            if problems:
                raise FileNotFoundError('; '.join(problems))
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            self._extractor = ASTFeatureExtractor.from_pretrained(self.folder, local_files_only=True)
            self._model = ASTForAudioClassification.from_pretrained(self.folder, local_files_only=True).to(self.device).eval()
            self._labels = dict(self._model.config.id2label)

    def classify(self, samples):
        """{label: probability} of the KEEP most likely labels for a mono 16 kHz float excerpt, or None when it is too short."""
        import numpy as np
        import torch
        x = np.asarray(samples, dtype=np.float32).ravel()
        if len(x) < RATE * MIN_SECONDS:
            return None
        peak = float(np.max(np.abs(x))) or 1.0
        with self._lock:
            self.load()
            inputs = self._extractor(x / peak * 0.5, sampling_rate=RATE, return_tensors='pt')
            with torch.no_grad():
                logits = self._model(**{k: v.to(self.device) for k, v in inputs.items()}).logits[0]
            probs = torch.sigmoid(logits).float().cpu().numpy()
        top = np.argsort(-probs)[:KEEP]
        return {self._labels[int(i)]: round(float(probs[i]), 3) for i in top}

    def classify_spans(self, wav, spans):
        """One {label: probability} (or None) per [start, end] second span of a 16 kHz wav file."""
        import soundfile as sf
        x, rate = sf.read(str(wav), dtype='float32', always_2d=True)
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        if rate != RATE:
            raise ValueError('audio_events expects a 16 kHz excerpt, got %s Hz' % rate)
        out = []
        for start, end in spans:
            a, b = max(0, int(float(start) * RATE)), min(len(x), int(float(end) * RATE))
            out.append(self.classify(x[a:b]) if b > a else None)
        return out
