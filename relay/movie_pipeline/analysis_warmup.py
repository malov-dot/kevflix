"""One-time warm-up of the analysis worker (relay.movie_pipeline). Runs inside the worker's own process.

The models themselves load in seconds; what made the first dub of a session slow was the first CALL of each step (CUDA and
cuDNN kernel set-up, the language-detector tables, the speaker pipeline's first batch, Qwen3's first generate): about 20 s of
speech analysis plus about 25 s of translation, paid by the first chunk while the TV showed "Preparing". The same calls on a
generated voice-like clip take about 7 s and leave the real first chunk about 1 s of analysis.

Nothing here is cached, counted or reported as analysis: each tool is called directly, below the worker's raw-inference
cache, so no row exists for the generated clip. A step that fails only means that step stays cold; it never fails the worker.
"""
from __future__ import annotations

import time
import uuid

SAMPLE_ROW = ('Esto es una prueba de una frase corta para el detector de idioma.', 'es')
CLIP_SECONDS = 10.0
RATE = 16000


def voice_like_clip(seconds=CLIP_SECONDS, rate=RATE):
    """A repeatable synthetic voice: gliding pitch, harmonics, a slow formant sweep and a 4 Hz syllable gate, plus faint noise."""
    import numpy as np
    t = np.arange(int(seconds * rate)) / rate
    pitch = 110 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    voice = sum(np.sin(h * phase) / h for h in range(1, 30))
    shape = 0.6 + 0.4 * np.sin(2 * np.pi * 3.3 * t)
    gate = np.sin(2 * np.pi * 4.0 * t) > -0.3
    noise = 0.01 * np.random.default_rng(1).standard_normal(len(t))
    return (0.25 * voice * shape * gate + noise).astype('float32')


def run(worker, translation, io_dir):
    """Warm every step the first real chunk uses. `worker` is the analysis child's worker object; `translation` also loads
    the translator and generates once. Returns {'seconds', 'steps': {name: seconds | 'skipped' | 'Type: message'}}."""
    import soundfile as sf
    started = time.perf_counter()
    steps = {}

    def step(name, call):
        begun = time.perf_counter()
        try:
            call()
            steps[name] = round(time.perf_counter() - begun, 2)
        except Exception as error:  # noqa: BLE001 - one cold step is not a failure of the worker
            steps[name] = '%s: %s' % (type(error).__name__, str(error)[:160])

    path = io_dir / ('warmup-%s.wav' % uuid.uuid4().hex)
    try:
        sf.write(str(path), voice_like_clip(), RATE, subtype='PCM_16')
        spans = [[0.0, 6.0], [6.0, 9.0]]
        language = worker.language
        step('text_language', lambda: language.text_language(SAMPLE_ROW[0], None))
        if getattr(worker, 'generic', False):
            steps['speech'] = steps['language_id'] = steps['audio_events'] = steps['speakers'] = 'skipped'
        else:
            step('speech', lambda: list(language.speech(str(path), None, vad=False, beam_size=5)))
            step('language_id', lambda: language.language_id(str(path), spans))
            if worker.audio_events is not None and worker.audio_events.available:
                step('audio_events', lambda: worker.audio_events.classify_spans(str(path), spans))
            else:
                steps['audio_events'] = 'skipped'
            if worker.speakers is not None:
                from relay.speakers import diarize
                speakers = worker.speakers
                step('speakers', lambda: diarize(speakers.config, str(path), 0, pipeline=speakers.pipeline, device=speakers.device))
            else:
                steps['speakers'] = 'skipped'
        if translation and not getattr(worker, 'generic', False):
            step('translation', lambda: language.english_batch([SAMPLE_ROW]))
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    return {'seconds': round(time.perf_counter() - started, 2), 'steps': steps}
