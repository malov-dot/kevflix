"""Vox overrun handling before a take reaches scheduling (plan 5.4). Pure Python, deterministic.

Owner rule: "Prefer fitting translated wording into natural gaps to artificial speed-up". Neither engine has a duration
target, so this never uses tempo, speed-up, max_tokens capping or truncation. It only chooses among takes the engine
renders at natural speed:

1. Requested text, requested seed. If that take fits the pipeline's window (seconds <= window) it is returned unchanged:
   identical to an engine without duration fit.
2. Before synthesis, the requested text's length is predicted (characters / chars-per-second). If it is predicted to
   overrun by more than `predict_margin` and an EXISTING shorter alternative translation is predicted to fit, that
   alternative is tried first (shortest first), so an overrunning take is not rendered at all.
3. For each text, at most `seeds_per_text` deterministic seeds: the requested seed, then the approved seeds (42, 7).
   The first take that fits is kept ("shortest passing seed": a later seed is only rendered when the earlier overran).
4. Nothing fits within `max_renders`: the shortest take is returned with timing_overrun=True, so the EXISTING path
   decides (actor_beta_timing reschedule without tempo or truncation, then the protected-original per-cue fallback).

Existing alternatives (never generated here): the cue's aligned English caption (`caption_fallback`, only when the
caption track is declared English; the subtitle line dialogue.py aligned to this cue) and producer-supplied
`english_alternatives` entries that carry `"faithful": true`. Every alternative must be shorter than the requested text.
"""
from __future__ import annotations

VERSION = 'vox-duration-fit-v1'
# est.: median Vox characters per second (Sem Q8 log, 2026-10-17; plan 5.4). A fixed constant, not learned from earlier
# takes: with concurrent lines the completion order varies, and the choice must not depend on it (determinism).
DEFAULT_CHARS_PER_SECOND = 15.6
ENGLISH = ('en', 'eng', 'english')
DEFAULTS = {'enabled': True, 'alternatives': True, 'seeds': [42, 7], 'seeds_per_text': 2, 'max_renders': 4,
            'predict_margin': 0.10, 'chars_per_second': DEFAULT_CHARS_PER_SECOND}


def options(raw=None):
    """Validated options (actor_beta.vox.duration_fit). `false` disables fitting entirely (one render, as Breeze)."""
    if raw is False:
        raw = {'enabled': False}
    merged = {**DEFAULTS, **dict(raw or {})}
    unknown = set(merged) - set(DEFAULTS)
    if unknown:
        raise ValueError('unknown duration_fit options: %s' % sorted(unknown))
    seeds = merged['seeds']
    if (not isinstance(seeds, (list, tuple)) or not seeds or len(set(seeds)) != len(seeds)
            or not all(isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 2 ** 31 for s in seeds)):
        raise ValueError('duration_fit.seeds must be distinct non-negative integers')
    for key, low, high in (('seeds_per_text', 1, 4), ('max_renders', 1, 8)):
        value = merged[key]
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError('duration_fit.%s must be an integer in %d..%d' % (key, low, high))
    for key in ('enabled', 'alternatives'):
        if not isinstance(merged[key], bool):
            raise ValueError('duration_fit.%s must be true or false' % key)
    if not 0 <= float(merged['predict_margin']) <= 1 or not 4 <= float(merged['chars_per_second']) <= 40:
        raise ValueError('duration_fit.predict_margin/chars_per_second out of range')
    return {**merged, 'seeds': list(seeds), 'predict_margin': float(merged['predict_margin']),
            'chars_per_second': float(merged['chars_per_second'])}


def _iso(value):
    return str(value or '').strip().lower()


def _valid_text(text, max_chars):
    return (isinstance(text, str) and text.strip() and len(text) <= max_chars
            and not any(ord(c) < 32 and c not in '\t\n\r' for c in text))


def alternatives_for(cue, requested, *, max_chars=400):
    """Existing, shorter alternative English texts for `cue` (list of {'text', 'basis'}), shortest first."""
    if not isinstance(cue, dict) or not isinstance(requested, str):
        return []
    found = []
    caption = cue.get('caption') if isinstance(cue.get('caption'), dict) else {}
    fallback = cue.get('caption_fallback')
    if fallback and _iso(caption.get('language')) in ENGLISH:
        found.append({'text': fallback, 'basis': 'aligned-english-caption'})
    for item in cue.get('english_alternatives') or []:
        if isinstance(item, dict) and item.get('faithful') is True and item.get('text'):
            found.append({'text': item['text'], 'basis': str(item.get('basis') or 'producer-alternative')})
    seen, out = {requested.strip()}, []
    for item in found:
        text = str(item['text']).strip()
        if text in seen or not _valid_text(text, max_chars) or len(text) >= len(requested.strip()):
            continue
        seen.add(text)
        out.append({'text': text, 'basis': item['basis']})
    return sorted(out, key=lambda a: (len(a['text']), a['text']))


def predict_seconds(text, chars_per_second):
    return len(text.strip()) / float(chars_per_second)


def seed_order(requested, seeds, per_text):
    order = []
    for seed in [int(requested)] + list(seeds):
        if seed not in order:
            order.append(seed)
    return order[:per_text]


def text_order(requested, alternatives, window, rate, margin):
    """(ordered [(text, basis)], predicted seconds of requested, switched_before_synthesis)."""
    predicted = predict_seconds(requested, rate)
    alts = [(a['text'], a['basis']) for a in alternatives]
    fitting = [a for a in alts if predict_seconds(a[0], rate) <= window]
    if window > 0 and predicted > window * (1 + margin) and fitting:
        rest = [a for a in alts if a not in fitting]
        return fitting + [(requested, 'requested')] + rest, predicted, True
    return [(requested, 'requested')] + alts, predicted, False


def fit(render, *, text, seed, window, alternatives, opts, rate):
    """Run the bounded fit. render(text, seed) -> (path, receipt) with receipt['seconds'].

    Returns ((path, receipt), record). Engine errors propagate unchanged (never swallowed)."""
    window = float(window)
    record = {'version': VERSION, 'enabled': opts['enabled'], 'requested_text': text, 'requested_seed': int(seed),
              'window_seconds': window, 'chars_per_second': round(float(rate), 4),
              'alternatives': [dict(a) for a in alternatives], 'attempts': [], 'text_changed': False,
              'seed_changed': False, 'switched_before_synthesis': False, 'outcome': None}
    if not opts['enabled'] or window <= 0:
        result = render(text, int(seed))
        record['attempts'].append({'text': text, 'basis': 'requested', 'seed': int(seed),
                                   'seconds': result[1]['seconds'], 'fits': result[1]['seconds'] <= window})
        record['outcome'] = 'disabled' if not opts['enabled'] else 'no_window_existing_fallback'
        record['chosen'] = {'text': text, 'seed': int(seed), 'basis': 'requested'}
        return result, record
    texts, predicted, switched = text_order(text, alternatives if opts['alternatives'] else [], window, rate,
                                            opts['predict_margin'])
    record['predicted_seconds'] = round(predicted, 4)
    record['switched_before_synthesis'] = switched
    best, renders = None, 0
    for candidate, basis in texts:
        for s in seed_order(seed, opts['seeds'], opts['seeds_per_text']):
            if renders >= opts['max_renders']:
                break
            result = render(candidate, s)
            renders += 1
            seconds = float(result[1]['seconds'])
            fits = seconds <= window
            record['attempts'].append({'text': candidate, 'basis': basis, 'seed': s, 'seconds': seconds, 'fits': fits,
                                       'sha256': result[1].get('sha256')})
            if best is None or seconds < best[0]:
                best = (seconds, result, candidate, basis, s)
            if fits:
                best = (seconds, result, candidate, basis, s)
                record['outcome'] = 'fits' if renders == 1 else 'fits_after_search'
                break
        if record['outcome'] is not None or renders >= opts['max_renders']:
            break
    if record['outcome'] is None:
        record['outcome'] = 'no_fit_existing_fallback'
    _seconds, result, chosen, basis, s = best
    record['chosen'] = {'text': chosen, 'seed': s, 'basis': basis}
    record['text_changed'] = chosen != text
    record['seed_changed'] = s != int(seed)
    return result, record
