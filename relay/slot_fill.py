"""Slot fill: slow a take that ends before the original speaker stopped, with ONE pitch-preserving ffmpeg ``atempo`` < 1.

Owner report 2026-10-20 ("some scenes dubbing speed is rushed when there is no reason", "speakers silenced but nothing
spoken", "hard cutoff of all volume"). Measured on the live Spanish film run (625 dubbed lines, 73 minutes, read-only replay
of the job database): the English take is shorter than the original speech span by more than 0.3 s in 193 lines (31 %), by
more than 0.6 s in 129 and by more than 1.0 s in 88 -- about 238 s in which the original voice is already removed, the actor
is still talking and the English is over. relay.cinematic.fill_room_tone keeps the ambience in that hole but cannot make the
English fill the actor's mouth time.

Owner decision 2026-10-20: a fitted take that is shorter than the slot the original speaker occupied is slowed down so that it
ends at (or just before) the original span end. This is the mirror image of relay.take_fit (which only ever speeds a take up):
one ``atempo`` pass over the untouched generated PCM, never repeated, never a trim, never a pad.

The rules, each of them tested (tests/fit/test_slot_fill.py):
  * target = min(original span, allowed window): never past the span, never past the placement's allowed window (next onset,
    silence slack, committed-window limit -- the ``window`` the take was rendered and fitted against);
  * a take already within ``slot_fill_tolerance_seconds`` (0.25 s) of the target, or longer than it, is not touched;
  * the slowdown is bounded by ``slot_fill_max_slowdown`` (duration factor 1.18, atempo 0.847; ceiling 1.35). Evidence (same
    replay, no listening): the English pace of the 414 long takes (>= 4 words, >= 1.5 s) is 3.24 words/s (P10 2.26, P5 1.99)
    and the actors' own Spanish pace is 2.84 words/s (P25 2.14). Slowing the 166 comparable short lines by up to x1.18 puts
    their median at 2.82 words/s -- the actors' own median -- and leaves 33 of them (20 %) below the natural English P10,
    against 14 (8 %) before; x1.25 would leave 41 and x1.35 54 (33 %). The factor needed to fill the span exactly is
    1.16 / 1.23 / 1.39 / 1.61 / 1.85 at P10 / 25 / 50 / 75 / 90 of the 193 short lines, so x1.18 closes 28 of them in full,
    x1.25 53 and x1.35 85; 108 lines (184 of the 238 s) need more than x1.35, which slowing cannot honestly give (that is the
    second lever, ``fuller_retranslation``). The cap is the owner's knob, to be raised only after listening;
  * a take under 0.8 s that is a one- or two-word interjection is left alone unless the gap is large (>= 0.6 s) and then slowed
    by at most x1.10: a drawn-out "no" sounds worse than a short silence, and x1.18 of 0.5 s is only 0.09 s anyway;
  * a take is never slowed below ``slot_fill_min_words_per_second`` (2.0, the P5 of natural English takes), so an
    already-slow take is not dragged further;
  * the slowed take is read back by the SAME gate the take passed (``check``): critical meaning change, ordinary word error
    above the larger of the take's own error and 0.25, or a lost last word reject it, and the unslowed take stays. A take
    whose readback already carries a critical change is not slowed at all (it keeps the original audio anyway);
  * the slowed take is a NEW file named by the unslowed take's sha256 and the exact factor, with its own sha256 and receipt
    (``slot_fill``, ``unslowed_seconds``, ``unslowed_sha256``) so claims, private copies, readback, ledger and mix see one
    consistent take, exactly as relay.actor_pipeline.fit_rendered_take does for the speed-up.

The gap that remains after the cap keeps the room-tone fill. Rubber Band, repeated passes, re-synthesis and padding are
deliberately not used. This module is also the home of the second, optional lever (``fuller_retranslation``, default OFF):
a line whose English is far shorter than its slot is translated again with a minimum word budget (see ``fuller_retry``).

Pure planning functions first, then the audio step, then the file/readback wrapper, then the optional fuller retry. numpy and
soundfile are imported inside the functions that need them; the module is safe to import from relay.config.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from . import take_fit as _fit

VERSION = 'slot-fill-v1'
RATE = _fit.RATE
DEFAULT_MAX_SLOWDOWN = 1.18         # duration factor (atempo 1 / 1.18 = 0.847)
MAX_SLOWDOWN_CEILING = 1.35         # hard ceiling: 1 / 1.35 = atempo 0.741, still far inside ffmpeg's 0.5 floor
MIN_SLOWDOWN = 1.04                 # a slowdown under 4 % is inaudible and not worth a file, a readback and a receipt
TOLERANCE_SECONDS = .25             # a take this close to the target already fills it
SHORT_TAKE_SECONDS = .8             # under this a one/two-word take is an interjection
SHORT_WORDS = 2
SHORT_MIN_GAP_SECONDS = .6          # an interjection is slowed only when this much of its slot is empty ...
SHORT_MAX_SLOWDOWN = 1.10           # ... and then by at most this
MIN_WORDS_PER_SECOND = 2.0          # P5 of the English pace of long takes in the measured film (1.99)
MARGIN_SECONDS = .03                # aim inside the target: atempo's output frames end up to ~20 ms long (take_fit.py)
ORDINARY_ERROR = .25                # the readback gate's word-error limit (actor_pipeline.READBACK_ORDINARY_ERROR)
_QUIET_REASONS = frozenset({'disabled', 'fills_slot', 'unusable_slot'})   # the take is returned as is, with no receipt

# Fuller retranslation (second lever, default OFF): lines whose English is much shorter than the slot.
FULLER_RATIO = .7                   # take / span below this is a target (21 % of the measured lines)
FULLER_PACE_SHARE = .95             # aim a little under the take's own pace so the fuller wording does not overshoot
FULLER_MAX_WORDS_PER_SECOND = 3.2   # never ask for more words per slot second than a brisk natural English pace
FULLER_MIN_EXTRA_WORDS = 2          # fewer extra words than this is not "fuller", it is noise
FULLER_MIN_GAIN_SECONDS = .1        # the fuller take must be at least this much closer to the target
FULLER_SEEDS = (42, 7, 1234)
FULLER_MAX_WORDS = 60               # relay.language.QWEN3_MAX_WORDS


def _number(source, key, default, low, high):
    """A finite number in [low, high] from ``source[key]``, else ``default`` (the overlap_policy convention)."""
    try:
        value = float(source.get(key, default))
    except (TypeError, ValueError, OverflowError, AttributeError):
        return default
    if isinstance(source.get(key), bool) or not math.isfinite(value) or not low <= value <= high:
        return default
    return value


def settings(beta):
    """Validated slot-fill / fuller settings from the ``actor_beta`` mapping (``None`` or ``{}``: every default).

    ``slot_fill`` (default False until this module has its own tests - 2026-10-20; True turns it on), ``slot_fill_max_slowdown`` (1.0-1.35, default 1.18; 1.0 disables
    the slowdown), ``slot_fill_tolerance_seconds`` (0-2, .25), ``slot_fill_short_seconds`` (0-3, .8),
    ``slot_fill_short_min_gap_seconds`` (0-5, .6), ``slot_fill_short_max_slowdown`` (1.0-1.35, 1.10),
    ``slot_fill_min_words_per_second`` (0-4, 2.0; 0 switches the floor off), ``fuller_retranslation`` (default False),
    ``fuller_ratio`` (.3-.95, .7). An invalid value keeps its default."""
    beta = beta if isinstance(beta, dict) else {}
    return {'enabled': beta.get('slot_fill') is True,
            'max_slowdown': _number(beta, 'slot_fill_max_slowdown', DEFAULT_MAX_SLOWDOWN, 1.0, MAX_SLOWDOWN_CEILING),
            'tolerance': _number(beta, 'slot_fill_tolerance_seconds', TOLERANCE_SECONDS, 0., 2.),
            'short_seconds': _number(beta, 'slot_fill_short_seconds', SHORT_TAKE_SECONDS, 0., 3.),
            'short_min_gap': _number(beta, 'slot_fill_short_min_gap_seconds', SHORT_MIN_GAP_SECONDS, 0., 5.),
            'short_max_slowdown': _number(beta, 'slot_fill_short_max_slowdown', SHORT_MAX_SLOWDOWN, 1.0, MAX_SLOWDOWN_CEILING),
            'min_wps': _number(beta, 'slot_fill_min_words_per_second', MIN_WORDS_PER_SECOND, 0., 4.),
            'fuller': beta.get('fuller_retranslation') is True,
            'fuller_ratio': _number(beta, 'fuller_ratio', FULLER_RATIO, .3, .95)}


def fingerprint(beta):
    """What config.engine_fingerprint hashes for this step: the code version and every EFFECTIVE setting (defaults filled
    in), so a changed cap, a switched-off step or a newly enabled fuller retranslation never reuses an older chunk."""
    return {'version': VERSION, 'margin_seconds': MARGIN_SECONDS, 'min_slowdown': MIN_SLOWDOWN, 'short_words': SHORT_WORDS,
            'settings': settings(beta)}


def count_words(text):
    """Words of an English line, counted like actor_pipeline.concise_budget does."""
    return len(re.findall(r"[\w']+", str(text or '')))


def _round(value):
    return round(float(value), 4)


def plan(take_seconds, span_seconds, window_seconds, *, words=None, cfg=None):
    """Decide the slowdown of one take. Pure; returns a JSON-compatible record.

    ``take_seconds``: the fitted take's length; ``span_seconds``: the original speech span (cue end - cue start);
    ``window_seconds``: the seconds the placement allows from the cue start (next onset, silence slack, window limit);
    ``words``: English word count, or None when unknown. ``cfg``: settings().

    The record always has ``applied`` and ``reason``. Applied: ``reason`` 'slot_fill', ``duration_factor`` (output / input
    length, 1.0 < f <= cap), ``atempo`` (1 / duration_factor), ``clamped`` (the cap stopped a full closure) and the
    geometry. Not applied: 'disabled', 'unusable_slot', 'fills_slot' (within tolerance of the target, or longer),
    'short_interjection', 'already_slow' (the words/second floor), 'gain_too_small'."""
    cfg = cfg or settings(None)
    take, span, window = (float(x) for x in (take_seconds, span_seconds, window_seconds))
    record = {'version': VERSION, 'applied': False, 'reason': 'disabled', 'take_seconds': _round(take),
              'span_seconds': _round(span), 'window_seconds': _round(window), 'duration_factor': 1.0, 'atempo': 1.0}
    if not cfg['enabled']:
        return record
    if not all(math.isfinite(x) and x > 0 for x in (take, span, window)):
        return dict(record, reason='unusable_slot')
    target = min(span, window)
    gap = target - take
    record.update(target_seconds=_round(target), gap_seconds=_round(gap), max_slowdown=cfg['max_slowdown'])
    if gap <= cfg['tolerance']:
        return dict(record, reason='fills_slot')
    cap = cfg['max_slowdown']
    if take < cfg['short_seconds'] and (words is None or words <= SHORT_WORDS):
        if gap < cfg['short_min_gap']:
            return dict(record, reason='short_interjection')
        cap = min(cap, cfg['short_max_slowdown'])
    if words and cfg['min_wps'] > 0:
        cap = min(cap, words / (take * cfg['min_wps']))          # the slowdown at which the pace reaches the floor
        if cap < MIN_SLOWDOWN:
            return dict(record, reason='already_slow', words=int(words), words_per_second=_round(words / take))
    aim = target - min(MARGIN_SECONDS, target / 4)
    wanted = aim / take
    factor = min(cap, wanted)
    if factor < MIN_SLOWDOWN:
        return dict(record, reason='gain_too_small')
    record.update(applied=True, reason='slot_fill', duration_factor=factor, atempo=1.0 / factor, clamped=wanted > cap,
                  requested_duration_factor=wanted, cap=cap)
    if words:
        record.update(words=int(words), words_per_second_after=_round(words / (take * factor)))
    return record


def slow_take(samples, rate, decision, *, ffmpeg, runner=None, workdir=None):
    """``(slowed_samples, info)``: ONE ``atempo`` (< 1) over the untouched PCM; ``decision`` is an applied plan() record.

    Raises ValueError (input or decision unusable) / RuntimeError (ffmpeg failed or its output is unusable) like
    relay.take_fit; TimeoutError and Cancelled from ``runner`` pass through. ``info`` is the decision plus the measured
    result: ``before_seconds``, ``after_seconds``, ``tail_seconds`` (atempo's frame tail), ``gap_after_seconds`` (what
    still separates the take from the target), ``overshoot`` (True when the result is longer than the target; the caller
    must then keep the unslowed take), ``input_sha256`` and ``output_sha256`` (float32 LE PCM)."""
    if not decision.get('applied') or not 0 < float(decision.get('atempo', 0)) < 1.0:
        raise ValueError('slow_take needs an applied slot-fill decision with atempo < 1')
    if ffmpeg is None or not str(ffmpeg).strip():
        raise ValueError('ffmpeg executable path is required')
    audio = _fit._validated_take(samples, rate)
    before = len(audio) / RATE
    if abs(before - float(decision['take_seconds'])) > .001:
        raise ValueError('The decision does not describe this take (%.4f s, not %.4f s)' % (before, decision['take_seconds']))
    slowed = _fit._atempo(audio, float(decision['atempo']), ffmpeg=ffmpeg, runner=runner, workdir=workdir)
    after = len(slowed) / RATE
    target = float(decision['target_seconds'])
    return slowed, dict(decision, factor=float(decision['atempo']), before_seconds=before, after_seconds=after,
                        tail_seconds=after - before * float(decision['duration_factor']),
                        gap_after_seconds=_round(target - after), overshoot=after > target + 1e-6,
                        input_sha256=_fit.pcm_sha256(audio), output_sha256=_fit.pcm_sha256(slowed))


def summarize(verdict, receipt=None, tail_missing=None):
    """A readback verdict (``check`` result: critical, error, optional readback / intended) as the facts the gate uses.
    ``tail_missing(intended, heard) -> list | None`` is actor_pipeline.missing_tail_words; ``tail_ok`` is None when it cannot
    be judged."""
    verdict = verdict or {}
    summary = {'critical': bool(verdict.get('critical')), 'error': _round(verdict.get('error') or 0.), 'tail_ok': None}
    intended = verdict.get('intended') or (receipt or {}).get('requested_text')
    if tail_missing is not None and verdict.get('readback') is not None:
        missing = tail_missing(intended, verdict.get('readback'))
        if missing is not None:
            summary['tail_ok'] = not missing
            if missing:
                summary['tail_missing'] = list(missing)
    return summary


def readback_accepted(summary, ordinary_error=ORDINARY_ERROR):
    """choose_take's readback part of 'accepted': no critical change, word error <= ordinary_error, last words heard."""
    return not summary['critical'] and summary['error'] <= ordinary_error and summary.get('tail_ok') is not False


def not_worse(before, after, ordinary_error=ORDINARY_ERROR):
    """True when the slowed take reads back at least as well as the unslowed one: a take the gate accepted stays accepted
    (no critical change, error <= ordinary_error, tail heard); a take it did not accept may not get worse."""
    if after['critical'] and not before['critical']:
        return False
    if after['error'] > max(before['error'], ordinary_error) + 1e-9:
        return False
    return not (after.get('tail_ok') is False and before.get('tail_ok') is not False)


def _store(target, samples, rate):
    """Write ``samples`` to ``target`` atomically unless an identical file is already there (libsndfile stamps a wall-clock
    PEAK chunk, so identical samples written seconds apart differ in bytes). True when a file was written."""
    import numpy as np
    import soundfile as sf
    temp = target.with_name(target.name + '.%d.tmp' % os.getpid())
    try:
        if target.exists():
            try:
                existing, existing_rate = sf.read(str(target), dtype='float32')
                if existing_rate == rate and existing.shape == samples.shape and bool(np.array_equal(existing, samples)):
                    return False
            except (RuntimeError, ValueError, OSError):
                pass                                    # unreadable stale file: replace it
        sf.write(str(temp), samples, rate, subtype='FLOAT', format='WAV')
        os.replace(temp, target)
        return True
    finally:
        if temp.exists():
            temp.unlink()


def slot_fill_rendered_take(path, receipt, *, span_seconds, window, cfg=None, ffmpeg, runner=None, words=None, check=None,
                            ordinary_error=ORDINARY_ERROR, tail_missing=None):
    """Owner decision 2026-10-20: a take that ends well before the original speaker stopped is slowed to fill the slot.
    Returns ``(path, receipt)``: the same objects when the take fills its slot or the step is off; the NEW slowed take
    (file named by the unslowed sha256 and the exact factor, receipt with ``seconds``, ``sha256``, ``timing_overrun``,
    ``slot_fill``, ``unslowed_seconds``, ``unslowed_sha256``) when it was slowed; or the unchanged path with a copy of the
    receipt carrying ``slot_fill`` = {applied: False, reason} when the slowdown was considered and refused
    ('short_interjection', 'already_slow', 'gain_too_small', 'readback_critical', 'readback_rejected', 'overshoot',
    'unsupported_take_format', 'fill_failed').

    ``check(path, receipt) -> {'critical', 'error', 'readback'?, 'intended'?}`` is the readback gate of the take search
    (actor_pipeline.take_check); ``tail_missing`` is actor_pipeline.missing_tail_words and ``ordinary_error`` its limit.
    Without ``check`` the slowed take is not re-read (tests, offline tools). Cancellation, resource waits, timeouts and
    errors of ``check`` propagate; ffmpeg / file failures leave the take unchanged with reason 'fill_failed'."""
    import soundfile as sf
    from .media import file_hash
    from .process import Cancelled
    from .resource_budget import ResourceYield
    cfg = cfg or settings(None)
    seconds = float(receipt.get('seconds') or 0)
    decision = plan(seconds, span_seconds, window, words=words, cfg=cfg)

    def refused(reason, **extra):
        planned = {'planned_duration_factor': decision['duration_factor']} if decision['applied'] else {}
        return path, dict(receipt, slot_fill=dict(decision, applied=False, reason=reason, duration_factor=1.0, atempo=1.0,
                                                  **planned, **extra))
    if not decision['applied']:
        return (path, receipt) if decision['reason'] in _QUIET_REASONS else (path, dict(receipt, slot_fill=decision))
    before = summarize(check(path, receipt), receipt, tail_missing) if check is not None else None
    if before is not None and before['critical']:
        return refused('readback_critical', readback={'before': before})        # the line keeps the original audio anyway
    try:
        samples, take_rate = sf.read(str(path), dtype='float32')
        if take_rate != RATE or samples.ndim != 1:
            return refused('unsupported_take_format')
        if abs(len(samples) / take_rate - seconds) > .001:                       # the receipt's length is not the file's: trust the file
            decision = plan(len(samples) / take_rate, span_seconds, window, words=words, cfg=cfg)
            if not decision['applied']:
                return refused(decision['reason'])
        slowed, info = slow_take(samples, take_rate, decision, ffmpeg=ffmpeg, runner=runner)
        if info['overshoot']:
            return refused('overshoot', after_seconds=_round(info['after_seconds']), gap_after_seconds=info['gap_after_seconds'])
        unslowed = str(receipt.get('sha256') or file_hash(Path(path)))
        target = Path(path).with_name('%s-fill-%s-%04d.wav' % (Path(path).stem[:48], unslowed[:16], round(info['factor'] * 1000)))
        written = _store(target, slowed, take_rate)
    except (Cancelled, ResourceYield, TimeoutError):
        raise
    except (ValueError, RuntimeError, OSError) as error:
        return refused('fill_failed', error='%s: %s' % (type(error).__name__, str(error)[:200]))
    new_seconds = len(slowed) / take_rate
    info.update(reason='slot_fill', applied=True, unslowed_sha256=unslowed)
    slowed_receipt = dict(receipt, seconds=new_seconds, sha256=file_hash(target), timing_overrun=new_seconds > window,
                          slot_fill=info, unslowed_seconds=seconds, unslowed_sha256=unslowed)
    if 'audio' in receipt:
        slowed_receipt['audio'] = str(target)
    if check is not None:
        after = summarize(check(target, slowed_receipt), slowed_receipt, tail_missing)
        if not not_worse(before, after, ordinary_error):
            if written and target.exists():
                target.unlink()                                  # nothing else holds this file: it was written just now
            return refused('readback_rejected', readback={'before': before, 'after': after})
        info['readback'] = {'before': before, 'after': after}
    return target, slowed_receipt


# ---- second lever: fuller retranslation (actor_beta.fuller_retranslation, default OFF) ------------------------------------

def fuller_min_words(english, take_seconds, span_seconds, window_seconds, cfg=None):
    """Minimum word budget for a fuller wording of ``english``, or None when the line is not a target.

    Targets: the step is on, the take is shorter than ``fuller_ratio`` (0.7) of the original span, and the words the same
    voice would speak in the slot at ITS OWN pace (words x slot / take, 5 % under) exceed the present words by at least
    two. The budget never asks for more than 3.2 words per slot second or 60 words."""
    cfg = cfg or settings(None)
    take, span, window = (float(x) for x in (take_seconds, span_seconds, window_seconds))
    if not cfg['fuller'] or not all(math.isfinite(x) and x > 0 for x in (take, span, window)):
        return None
    slot = min(span, window)
    if take / span >= cfg['fuller_ratio'] or take >= slot:
        return None
    count = count_words(english)
    if count < 1:
        return None
    budget = min(int(count * slot / take * FULLER_PACE_SHARE), int(slot * FULLER_MAX_WORDS_PER_SECOND), FULLER_MAX_WORDS)
    return budget if budget >= count + FULLER_MIN_EXTRA_WORDS else None


def fuller_retry(chosen, *, english, span_seconds, window, cfg=None, ask, render, fit, accept, recoverable=lambda error: False,
                 seeds=FULLER_SEEDS, preferred_factor=1.12):
    """Owner proposal 2026-10-20 (default OFF): a line whose English is far shorter than its slot is translated again with a
    minimum word budget, "keep exactly the same meaning with fuller natural phrasing, no new information", and rendered like
    any other take. ``chosen`` is the (path, receipt) the take search kept.

    ``ask(min_words) -> str | None`` is the translator (actor_pipeline.fuller_english); ``render(seed, text) -> raw take``;
    ``fit(path, receipt) -> take`` (the same speed-up fit as every take); ``accept(path, receipt) -> bool`` is the readback
    gate (critical / ordinary error / tail). A candidate is adopted only when it fits the window, needs no more than
    ``preferred_factor`` of speed-up, is accepted by the gate AND is closer to min(span, window) than the present take by at
    least 0.1 s. Otherwise ``chosen`` is returned with ``fuller_attempt`` in a copy of its receipt (what was asked, what
    happened). The adopted receipt carries ``fuller`` {text, requested, min_words, seed, before_seconds, after_seconds,
    attempts}; actor_pipeline.adopt_spoken_text makes that wording the cue's English. A take that is itself unaccepted, or the
    product of the concise retranslation, is never touched (its problem is not length)."""
    cfg = cfg or settings(None)
    path, receipt = chosen
    seconds = float(receipt.get('seconds') or 0)
    minimum = fuller_min_words(english, seconds, span_seconds, window, cfg)
    if minimum is None or receipt.get('concise') or receipt.get('fuller') or not accept(path, receipt):
        return chosen
    goal = min(float(span_seconds), float(window))
    text = ask(minimum)
    text = text.strip() if isinstance(text, str) else ''
    attempt = {'applied': False, 'min_words': minimum, 'requested': english}
    if not text or text == str(english).strip() or count_words(text) < count_words(english) + 1:
        return path, dict(receipt, fuller_attempt=dict(attempt, reason='no_fuller_wording', text=text or None))
    attempt.update(text=text, attempts=[])
    for seed in seeds:
        try:
            got = fit(*render(seed, text))
        except Exception as error:
            if not recoverable(error):
                raise
            attempt['attempts'].append({'seed': seed, 'unusable': True, 'accepted': False})
            continue
        new_seconds = float(got[1].get('seconds') or 0)
        factor = float((got[1].get('fit') or {}).get('factor') or 1.)
        entry = {'seed': seed, 'seconds': _round(new_seconds), 'factor': _round(factor),
                 'overrun': new_seconds > window + 1e-6, 'closer': abs(goal - new_seconds) < abs(goal - seconds) - FULLER_MIN_GAIN_SECONDS}
        entry['accepted'] = (not entry['overrun'] and factor <= preferred_factor and entry['closer'] and bool(accept(*got)))
        attempt['attempts'].append(entry)
        if entry['accepted']:
            record = {'text': text, 'requested': english, 'min_words': minimum, 'seed': seed, 'before_seconds': _round(seconds),
                      'after_seconds': _round(new_seconds), 'attempts': attempt['attempts']}
            return got[0], dict(got[1], fuller=record)
    return path, dict(receipt, fuller_attempt=dict(attempt, reason='no_accepted_take'))
