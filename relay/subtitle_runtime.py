"""Subtitle + VAD scoped BandIt separation for ActorPipeline.process (installed as relay/subtitle_runtime.py).

Active only when config['actor_beta']['enabled'] is True AND config['actor_beta']['subtitle_scoped'] is True.
Otherwise plan() returns None and separate() makes exactly the original full-window call.

Evidence of speech = dialogue subtitle cues (SDH/music captions excluded) UNION real local VAD
(faster_whisper.vad.get_speech_timestamps, bundled silero_vad_v6.onnx, CPU) run PER non-LFE channel and
unioned (no downmix cancellation). Those spans (+padding, +separator context) are separated; the rest of the
window is NOT separated:
  * its speech-stem samples are 0, so replace_dialogue removes nothing there (original preserved exactly);
  * it is reported as unprocessed and returned by timing_blocked()/mix_walls(), so no English line is placed or
    mixed there. It is never labelled verified silence.
Any VAD failure, or no speech evidence at all, separates the FULL window (explicit reason), never false quiet.

VAD route (receipt 'vad_route'): 'cpu-silero-prefilter' = the small bundled Silero ONNX run on CPU through
faster_whisper.vad. It is ONLY a documented prefilter to skip expensive GPU separation of spans with no speech
evidence; it is not ASR VAD (faster-whisper applies its own internal VAD during recognition) and it decides no speech,
speaker or English. No other model framework is used for it. A VAD exception is never treated as "no speech": the
window separates in full AND the scope receipt carries an explicit relay.movie_faults record ('vad_fault', fatal kind =
retrying the unchanged setup cannot repair the asset/runtime), so the fault is visible rather than hidden.
Per-cue original fallbacks for unprocessed (not separated) spans carry recoverable fault records
(report['beta_runtime_faults']): English is never authorized outside the verified speech stem.
"""
import hashlib
import json
import time

from .movie_faults import MovieConfigurationError, MovieContentFallback   # stdlib-only metadata classes

RATE_VAD = 16000
VAD_ROUTE = 'cpu-silero-prefilter'
DEFAULTS = {'subtitle_offset': 0.0, 'subtitle_pad_before': .35, 'subtitle_pad_after': .5, 'merge_gap': .5,
            'vad_pad': .25, 'context_seconds': 1.0, 'min_slice_seconds': 2.0}
VAD = None  # tests inject: VAD(mono16k ndarray) -> [(start_s, end_s)]


def _beta(config):
    try:
        value = config['actor_beta']
    except (KeyError, TypeError):
        value = (getattr(config, 'data', None) or {}).get('actor_beta')
    return value if isinstance(value, dict) and value.get('enabled') is True else None


def _opt(beta, key):
    value = beta.get(key)
    return DEFAULTS[key] if value is None else float(value)


def _merge(spans, gap=0.0):
    out = []
    for s, e in sorted(spans):
        if e <= s:
            continue
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out]


def _subtract(base, remove):
    out = []
    for s, e in base:
        cur = s
        for rs, re_ in sorted(remove):
            if re_ <= cur or rs >= e:
                continue
            if rs > cur:
                out.append((cur, rs))
            cur = max(cur, re_)
        if cur < e:
            out.append((cur, e))
    return out


def _default_vad(mono16k):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    rows = get_speech_timestamps(mono16k, VadOptions(min_silence_duration_ms=300, speech_pad_ms=0), sampling_rate=RATE_VAD)
    return [(r['start'] / RATE_VAD, r['end'] / RATE_VAD) for r in rows]


def _vad_union(original, rate, lfe):
    """Per non-LFE channel VAD at 16 kHz; union of all channels (window-relative seconds)."""
    import numpy as np
    from scipy.signal import resample_poly
    detect = VAD or _default_vad
    spans = []
    for channel in range(original.shape[1]):
        if channel == lfe:
            continue
        mono = np.ascontiguousarray(resample_poly(original[:, channel], 1, rate // RATE_VAD).astype('float32'))
        spans.extend(detect(mono))
    return _merge(spans)


MODEL_MIN_SECONDS = 4  # literal in cinematic.CinematicSeparator.separate (max(4*sample_rate, ...)); not an attribute


def _precision(quantized):
    return {True: 'qint8-dynamic(GRU,Linear)-cpu', False: 'fp32-cpu'}.get(quantized, 'unknown')


def separator_identity(config, separator=None):
    """Actual separator compute settings for the cache key/metrics. Live instance when it exists; otherwise the values
    actor_pipeline constructs it with (threads=min(8,cpu_threads); quantized/window/hop = CinematicSeparator class
    defaults read via inspect, since the call site does not override them). Both paths give identical values."""
    if separator is not None:
        window, hop, quantized = getattr(separator, 'window', None), getattr(separator, 'hop', None), \
            getattr(separator, 'quantized', None)
        return {'quantized': quantized, 'threads': getattr(separator, 'threads', None), 'precision': _precision(quantized),
                'window_seconds': None if window is None else window / 48000,
                'hop_seconds': None if hop is None else hop / 48000, 'source': 'instance'}
    import inspect
    from .cinematic import CinematicSeparator
    params = inspect.signature(CinematicSeparator.__init__).parameters
    quantized = params['quantized'].default
    try:
        threads = min(8, int(config['cpu_threads']))
    except (KeyError, TypeError, ValueError):
        threads = None
    return {'quantized': quantized, 'threads': threads, 'precision': _precision(quantized),
            'window_seconds': float(params['window_seconds'].default), 'hop_seconds': float(params['hop_seconds'].default),
            'source': 'config+class-defaults'}


def plan(config, original, rate, origin, captions, lfe, separator=None):
    beta = _beta(config)
    if beta is None or beta.get('subtitle_scoped') is not True:
        return None
    import numpy as np
    window = (float(origin), float(origin) + len(original) / rate)
    offset = _opt(beta, 'subtitle_offset')
    dialogue, sdh = [], []
    for cue in captions or ():
        start, end = float(cue['start']) + offset, float(cue['end']) + offset
        if end <= window[0] or start >= window[1]:
            continue
        text = (cue.get('text') or '').strip()
        nonspeech = not text or all(part.strip() == '' or part.strip()[0] in '[(♪' for part in text.splitlines())
        (sdh if nonspeech else dialogue).append((max(window[0], start - _opt(beta, 'subtitle_pad_before')),
                                                 min(window[1], end + _opt(beta, 'subtitle_pad_after'))))
    reason, vad_fault = None, None
    vad_route = VAD_ROUTE if VAD is None else 'injected'
    try:
        vad = [(window[0] + s - _opt(beta, 'vad_pad'), window[0] + e + _opt(beta, 'vad_pad'))
               for s, e in _vad_union(original, rate, lfe)]
    except Exception as error:  # missing asset/runtime: never false quiet; the fault is recorded, not hidden
        vad, reason = [], 'full_window: VAD failed (%s)' % type(error).__name__
        vad_fault = MovieConfigurationError('subtitle_vad_failed', '%s via %s: %s'
                                            % (type(error).__name__, vad_route, error)).record()
    evidence = _merge([(max(window[0], s), min(window[1], e)) for s, e in dialogue + vad], _opt(beta, 'merge_gap'))
    if reason is None and not evidence:
        reason = 'full_window: no subtitle or VAD speech evidence; full separation verifies the window'
    if reason:
        separate = [window]
    else:
        minimum, grown = _opt(beta, 'min_slice_seconds'), []
        for s, e in evidence:
            if e - s < minimum:
                mid = (s + e) / 2
                s, e = max(window[0], mid - minimum / 2), min(window[1], mid + minimum / 2)
            grown.append((s, e))
        separate = _merge(grown, _opt(beta, 'merge_gap'))
    unprocessed = _subtract([window], separate)
    frames = [(round((s - origin) * rate), round((e - origin) * rate)) for s, e in separate]
    from .cinematic import CHECKPOINT_SHA256
    ident = separator_identity(config, separator)
    key = hashlib.sha256(json.dumps({
        'pcm': hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest(), 'rate': rate,
        'shape': list(original.shape), 'origin': origin, 'frames': frames, 'checkpoint': CHECKPOINT_SHA256,
        'separator': {k: v for k, v in ident.items() if k != 'source'}, 'lfe': lfe,
        'context': _opt(beta, 'context_seconds'), 'version': 2},
        sort_keys=True).encode()).hexdigest()
    return {'key': key, 'window': list(window), 'separate': [list(s) for s in separate], 'frames': frames,
            'unprocessed': [list(s) for s in unprocessed], 'sdh_nonspeech': [list(s) for s in _merge(sdh)],
            'subtitle_dialogue': [list(s) for s in _merge(dialogue)], 'vad': [list(s) for s in _merge(vad)],
            'offset': offset, 'context_seconds': _opt(beta, 'context_seconds'), 'reason': reason,
            'vad_route': vad_route, 'vad_fault': vad_fault,
            'separator': ident, 'cross_window_cache': 'not implemented'}


def effective_compute_seconds(separator, frames, rate=48000):
    """Model frames run for one call of `frames` input samples, mirroring CinematicSeparator.separate (hop steps;
    each chunk padded to min(window, max(4 s, chunk))). None when window/hop are not exposed."""
    window, hop = getattr(separator, 'window', None), getattr(separator, 'hop', None)
    if not isinstance(window, int) or not isinstance(hop, int) or hop <= 0:
        return None
    total = 0
    for start in range(0, frames, hop):  # same loop as cinematic.py separate(), including its right==count break
        right = min(frames, start + window)
        if right <= start:
            continue
        total += min(window, max(MODEL_MIN_SECONDS * rate, right - start))
        if right == frames:
            break
    return total / rate


def _compute(separator, lengths, rate):
    values = [effective_compute_seconds(separator, n, rate) for n in lengths]
    return 'unavailable (separator exposes no window/hop)' if any(v is None for v in values) else sum(values)


def separate(separator, original, rate, origin, lfe, owner, scope):
    if scope is None:
        return separator.separate(original, lfe_index=lfe, cancelled=owner.cancelled.is_set)
    import numpy as np
    seconds = len(original) / rate
    if scope['reason']:
        started = time.perf_counter()
        speech, metrics = separator.separate(original, lfe_index=lfe, cancelled=owner.cancelled.is_set)
        wall = time.perf_counter() - started
        metrics = dict(metrics)
        metrics['subtitle_scope'] = {**scope, 'selected_seconds': seconds, 'processed_input_seconds': seconds,
                                     'duplicate_input_seconds': 0.0, 'window_seconds': seconds, 'calls': 1,
                                     'effective_window_compute_seconds': _compute(separator, [len(original)], rate),
                                     'slices': [{'wall': wall}], 'scoped': False}
        return speech, metrics
    speech = np.zeros_like(original)
    context = int(round(scope['context_seconds'] * rate))
    groups = []  # merged context intervals [left, right, cores]: no input sample is processed twice in this window
    for a, b in sorted(tuple(f) for f in scope['frames']):
        left, right = max(0, a - context), min(len(original), b + context)
        if groups and left <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], right)
            groups[-1][2].append((a, b))
        else:
            groups.append([left, right, [(a, b)]])
    slices, total_wall, last = [], 0.0, {}
    for left, right, cores in groups:
        owner.check()
        if owner.cancelled.is_set():
            raise InterruptedError('Separation cancelled')
        started = time.perf_counter()
        part, last = separator.separate(original[left:right], lfe_index=lfe, cancelled=owner.cancelled.is_set)
        wall = time.perf_counter() - started
        total_wall += wall
        if part.shape != (right - left, original.shape[1]):
            raise ValueError('Scoped separation geometry changed')
        for a, b in cores:
            speech[a:b] = part[a - left:b - left]
        slices.append({'cores': [[origin + a / rate, origin + b / rate] for a, b in cores],
                       'context_frames': [left, right], 'wall': wall})
    inputs = [right - left for left, right, _ in groups]
    union = sum(e - s for s, e in _merge([(l, r) for l, r, _ in groups]))
    metrics = {**dict(last or {}), 'seconds': total_wall,
               'subtitle_scope': {**scope, 'selected_seconds': sum(b - a for a, b in scope['frames']) / rate,
                                  'processed_input_seconds': sum(inputs) / rate,
                                  'duplicate_input_seconds': (sum(inputs) - union) / rate, 'window_seconds': seconds,
                                  'effective_window_compute_seconds': _compute(separator, inputs, rate),
                                  'calls': len(groups), 'slices': slices, 'scoped': True,
                                  'note': 'unprocessed spans: speech stem 0 (no removal), timing/mix blocked, not silence'}}
    return speech, metrics


def _scope(report):
    return ((report or {}).get('separation') or {}).get('subtitle_scope') or {}


def timing_blocked(report):
    """Unprocessed (not separated) spans: never free time for English placement."""
    return [tuple(s) for s in _scope(report).get('unprocessed') or ()]


def _overlaps(cue, spans):
    s0, e0 = float(cue['start']), float(cue['end'])
    return any(min(e, e0) > max(s, s0) for s, e in spans)


def fallback_unprocessed(report, cues):
    """Call right after _beta.window_fallback, before the actor call. Never drops a wall: ALL unprocessed spans are
    returned as walls. Each synthesis cue overlapping a wall becomes a per-cue ORIGINAL fallback
    (report['beta_runtime_fallback'][id] += 'overlaps_unprocessed_separation'), is removed from cues, and its own
    source interval joins the wall (repeated until no remaining cue overlaps any wall). No scope -> (cues, [])."""
    walls = _merge(timing_blocked(report))
    if not walls:
        return cues, []
    runtime = report.setdefault('beta_runtime_fallback', {})
    faults = report.setdefault('beta_runtime_faults', {})
    remaining = list(cues)
    while True:
        hit = [c for c in remaining if _overlaps(c, walls)]
        if not hit:
            break
        for cue in hit:
            runtime.setdefault(cue['id'], []).append('overlaps_unprocessed_separation')
            faults.setdefault(cue['id'], []).append(MovieContentFallback(
                'overlaps_unprocessed_separation', 'unit [%s, %s] touches speech that was not separated; original kept'
                % (cue['start'], cue['end'])).record())
        gone = {id(c) for c in hit}
        remaining = [c for c in remaining if id(c) not in gone]
        walls = _merge(walls + [(float(c['start']), float(c['end'])) for c in hit])
    report['subtitle_unprocessed_walls'] = [list(w) for w in walls]
    return remaining, walls


def mix_walls(report, cues=None):
    """Compatibility: ALL unprocessed spans (never filtered by cues). Runtime uses fallback_unprocessed."""
    return list(timing_blocked(report))
