"""Speech-activity analysis for the dubbing mix (pure numpy; no torch, no I/O).

The owner approved this behaviour on the demo film (step4_clone_mix.py ``speech_env`` / ``active_rms_db``):

  * the original talking is removed ONLY while somebody is actually speaking, never for the whole cue span.
    Pauses, footsteps and whispers between the words stay bit-exact original;
  * a line is dubbed as loudly as the original line was spoken, measured over the frames that carry speech
    (a quiet speaker who then yells is dubbed quiet, then loud).

Method (all constants are module level and can be overridden per call):

  1. Analyse the separated-dialogue stem inside ``[begin - LEAD_SECONDS, end + TRAIL_SECONDS]`` of one cue,
     clamped to the stem. Nothing outside that window is ever reported.
  2. Cut it into FRAME_SECONDS (10 ms) frames; a frame is ACTIVE when its RMS is within ``-rel_db`` (28 dB) of the loudest
     frame of THIS cue (so a whispered line is judged against itself) and above the absolute FLOOR_DB.
  3. Active frames at most ``hang`` (40 ms) apart (start to start) are bridged, then the mask is dilated by ``edge``
     (20 ms) on each side so consonant onsets/decays are covered.
  4. Adjacent active frames become one interval; intervals shorter than MIN_INTERVAL_SECONDS are dropped.

The intervals are absolute (start_s, end_s) tuples that plug into ``cinematic.replacement_envelope`` with
``fade_seconds=ACTIVITY_FADE_SECONDS`` (10 ms raised-sine fades). Deviation from the demo: for a multichannel stem the
per-frame level is the strongest non-LFE channel (the demo averaged a stereo pair), so speech panned to one channel
or carried by the centre channel alone is still found; for mono or identical channels the result is identical.

Silent, empty or too-short stems return no intervals and never raise; non-finite samples or invalid geometry raise
ValueError.

Speech-only removal of a cue/take (``span_activity_intervals``; the mix calls it through ``cinematic.speech_only_spans``)
fixes a failure of the method above when it is applied to an already PADDED span ([start-.12, end+.25]): the level
reference was the loudest frame of that padded window, so a whisper next to a shout, or the quiet half of a cue that
turns loud, fell more than 28 dB under the reference and the original speech stayed audible under the dub. It differs in
three ways:

  * the window searched is padded, but the level REFERENCE comes only from the frames inside the UNPADDED source span
    (a neighbouring loud cue, even one that starts inside the pad, never sets this cue's reference);
  * the reference is the 90th percentile (REFERENCE_PERCENTILE) of the frame levels of that span that are above the noise
    floor, not the single loudest frame: one click or one shouted syllable cannot raise the bar for the whole line, and
    pauses cannot lower it;
  * threshold_dB = max(NOISE_FLOOR_DB, min(reference_dB - 28, ABS_SPEECH_DB)). Any frame above ABS_SPEECH_DB (-55 dBFS)
    is speech whatever the line's own level (so the quiet half of a cue that turns loud is still found; a relative
    threshold cannot span more than 28 dB inside one line); frames at or below NOISE_FLOOR_DB (-65 dBFS) never are
    (a very quiet line is not judged against its own noise, and room tone / separator residue at -70 dBFS is not
    speech). Bridging (<= 40 ms) and dilation (20 ms) are the same as above, and the result is clamped to the padded
    span.

Deliberate consequence: the owner rule "original speech is removed only while somebody is speaking" now depends on the
separated stem being below -55 dBFS between words. Residue above that level inside a pause is treated as speech and
removed (it is the separator's estimate of speech, so removing it subtracts little); the safe direction is to remove too
much of the ESTIMATE, never to leave a spoken passage under the dub.
"""
from __future__ import annotations

import math

import numpy as np

FRAME_SECONDS = .01            # analysis frame (10 ms)
REL_DB = -28.0                 # a frame is speech when within 28 dB of the cue's loudest frame
HANG_SECONDS = .04             # active frames closer than this (start to start) are bridged
EDGE_SECONDS = .02             # the mask is dilated by this on each side
LEAD_SECONDS = .03             # analysis window starts this long before the cue start
TRAIL_SECONDS = .06            # ... and ends this long after the cue end
MIN_INTERVAL_SECONDS = .01     # shorter intervals are dropped
ACTIVITY_FADE_SECONDS = .01    # fade to hand cinematic.replacement_envelope for activity intervals
FLOOR_DB = -90.0               # frames at or below -90 dBFS never count as speech (digital-silence guard)
SILENCE_RMS = 1e-6             # a signal whose loudest frame is below this (-120 dBFS) is silent
SILENCE_DB = -120.0            # what active_rms_db reports for silent/empty signals

# Speech-only removal of one cue/take (span_activity_intervals).
SPAN_PAD_BEFORE_SECONDS = .12  # padded window = [start - .12, end + .25]; the padding of beta_caption_flow / DialogueLedger
SPAN_PAD_AFTER_SECONDS = .25
REFERENCE_PERCENTILE = 90.0    # level reference = p90 of the above-noise frames INSIDE the unpadded span
ABS_SPEECH_DB = -55.0          # ceiling of the threshold: a frame above this is speech whatever the line's own level
NOISE_FLOOR_DB = -65.0         # lower bound of the threshold: frames at or below this are never speech
MERGE_JOIN_SECONDS = .002      # merge_intervals joins intervals closer than this (2 ms)


def _rate(rate):
    value = float(rate)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Sample rate must be positive and finite')
    return value


def _seconds(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('%s must be finite' % name)
    return value


def _hop(rate):
    return max(1, int(round(FRAME_SECONDS * rate)))


def _frame_rms(window, hop):
    """RMS of every full ``hop``-sample frame; a (n, ch) window reduces to the strongest channel per frame."""
    frames = len(window) // hop
    if frames <= 0:
        return np.zeros(0, dtype=np.float64)
    body = window[:frames * hop]
    if body.ndim == 1:
        return np.sqrt(np.mean(np.square(body.reshape(frames, hop)), axis=1))
    channels = body.shape[1]
    return np.sqrt(np.mean(np.square(body.reshape(frames, hop, channels)), axis=1)).max(axis=1)


def _analysis_window(count, rate, begin, end, origin):
    """(left, right, origin_sample): stem sample indices of the clamped analysis window."""
    begin, end, origin = _seconds(begin, 'begin'), _seconds(end, 'end'), _seconds(origin, 'origin')
    if end <= begin:
        raise ValueError('Cue end must follow its start')
    origin_sample = round(origin * rate)
    left = max(0, round((begin - LEAD_SECONDS) * rate) - origin_sample)
    right = min(count, round((end + TRAIL_SECONDS) * rate) - origin_sample)
    return left, right, origin_sample


def _stem_window(stem, left, right, lfe_index):
    """Float64 copy of ``stem[left:right]`` restricted to non-LFE channels; validates finiteness."""
    window = np.asarray(stem[left:right], dtype=np.float64)
    if window.ndim == 2:
        if lfe_index is not None:
            if not 0 <= int(lfe_index) < window.shape[1]:
                raise ValueError('LFE index is outside the stem channels')
            keep = [c for c in range(window.shape[1]) if c != int(lfe_index)]
            if not keep:
                raise ValueError('The stem has no non-LFE channel')
            window = window[:, keep]
        if window.shape[1] == 0:
            raise ValueError('The stem has no channels')
    if not np.isfinite(window).all():
        raise ValueError('Speech stem contains nonfinite samples')
    return window


def _check_stem(stem):
    stem = np.asarray(stem)
    if stem.ndim not in (1, 2):
        raise ValueError('Speech stem must be mono (n,) or sample-major (n, channels)')
    return stem


def _bridge_and_dilate(active, hang, edge):
    """Bridge gaps of at most ``hang`` seconds (start to start) between active frames, then dilate by ``edge``."""
    bridge = max(0, int(round(float(hang) / FRAME_SECONDS)))
    closed = active.copy()
    indices = np.flatnonzero(active)
    if bridge > 1 and len(indices) > 1:
        gaps = np.diff(indices)
        gap = (gaps > 1) & (gaps <= bridge)
        for first, second in zip(indices[:-1][gap], indices[1:][gap]):
            closed[first:second] = True
    reach = max(0, int(round(float(edge) / FRAME_SECONDS)))
    if reach:
        total = np.concatenate(([0], np.cumsum(closed, dtype=np.int64)))
        position = np.arange(len(closed))
        high = np.minimum(len(closed), position + reach + 1)
        low = np.maximum(0, position - reach)
        closed = (total[high] - total[low]) > 0
    return closed


def _active_frames(rms, rel_db, hang, edge, floor_db):
    """Boolean per-frame speech mask (threshold, bridge short gaps, dilate)."""
    if not len(rms):
        return np.zeros(0, dtype=bool)
    db = 20 * np.log10(rms + 1e-9)
    loudest = float(db.max())
    if loudest <= floor_db or rms.max() < SILENCE_RMS:
        return np.zeros(len(rms), dtype=bool)
    active = db > max(loudest + float(rel_db), floor_db)
    return _bridge_and_dilate(active, hang, edge)


def _analyse(stem_mono, rate, begin, end, origin, rel_db, hang, edge, lfe_index, floor_db):
    """-> (frame mask, left sample, right sample, hop, origin sample). Empty mask when there is nothing to analyse."""
    rate = _rate(rate)
    stem = _check_stem(stem_mono)
    left, right, origin_sample = _analysis_window(len(stem), rate, begin, end, origin)
    hop = _hop(rate)
    if right - left < hop:
        # Still validate the geometry of whatever is there, but a short/empty window has no speech to find.
        _stem_window(stem, left, max(left, right), lfe_index)
        return np.zeros(0, dtype=bool), left, right, hop, origin_sample
    window = _stem_window(stem, left, right, lfe_index)
    return _active_frames(_frame_rms(window, hop), rel_db, hang, edge, floor_db), left, right, hop, origin_sample


def activity_mask(stem_mono, rate, begin, end, origin, rel_db=REL_DB, hang=HANG_SECONDS, edge=EDGE_SECONDS,
                  *, lfe_index=None, floor_db=FLOOR_DB):
    """Per-frame speech mask of one cue: bool array, one entry per FRAME_SECONDS frame.

    ``stem_mono`` is the separated-dialogue stem starting at absolute time ``origin`` (seconds): mono ``(n,)`` or
    sample-major ``(n, channels)`` (strongest non-LFE channel per frame). Frame ``i`` covers
    ``[start + i*hop/rate, start + (i+1)*hop/rate)`` where ``start`` is the first element of
    ``activity_window(...)``. Returns an empty array for a silent, empty or too-short stem.
    """
    return _analyse(stem_mono, rate, begin, end, origin, rel_db, hang, edge, lfe_index, floor_db)[0]


def activity_window(stem_length, rate, begin, end, origin):
    """Absolute (start_s, end_s) of the clamped window ``activity_mask`` analyses ([begin-0.03, end+0.06] on the stem)."""
    rate = _rate(rate)
    left, right, origin_sample = _analysis_window(int(stem_length), rate, begin, end, origin)
    return (origin_sample + left) / rate, (origin_sample + max(left, right)) / rate


def activity_intervals(stem_mono, rate, begin, end, origin, rel_db=REL_DB, hang=HANG_SECONDS, edge=EDGE_SECONDS,
                       *, lfe_index=None, floor_db=FLOOR_DB, min_seconds=MIN_INTERVAL_SECONDS):
    """Absolute ``(start_s, end_s)`` intervals during which the stem carries speech, inside [begin-0.03, end+0.06].

    Ready for ``cinematic.replacement_envelope(frames, rate, origin, intervals, fade_seconds=ACTIVITY_FADE_SECONDS)``:
    interval edges land on exact sample positions of that function's grid and intervals never touch each other
    (adjacent frames are merged, so remaining gaps are longer than ``hang`` plus the dilation). ``[]`` for a silent,
    empty or too-short stem or a window outside the stem.
    """
    rate = _rate(rate)
    mask, left, right, hop, origin_sample = _analyse(stem_mono, rate, begin, end, origin, rel_db, hang, edge,
                                                     lfe_index, floor_db)
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    floor_samples = max(1, round(float(min_seconds) * rate))
    result = []
    for first, last in zip(starts, stops):
        a = left + int(first) * hop
        b = min(right, left + int(last) * hop)
        if b - a >= floor_samples:
            result.append(((origin_sample + a) / rate, (origin_sample + b) / rate))
    return result


def _frame_levels_with_tail(window, hop):
    """Linear RMS per ``hop``-sample frame (strongest non-LFE channel); a final partial frame of >= hop/2 samples is kept."""
    full = len(window) // hop
    parts = []
    if full:
        parts.append(_frame_rms(window[:full * hop], hop))
    rest = len(window) - full * hop
    if rest * 2 >= hop:
        parts.append(_frame_rms(window[full * hop:], rest))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def span_activity_details(stem, rate, begin, end, origin, *, pad_before=SPAN_PAD_BEFORE_SECONDS,
                          pad_after=SPAN_PAD_AFTER_SECONDS, lfe_index=None, rel_db=REL_DB, hang=HANG_SECONDS,
                          edge=EDGE_SECONDS, abs_speech_db=ABS_SPEECH_DB, noise_floor_db=NOISE_FLOOR_DB,
                          percentile=REFERENCE_PERCENTILE, min_seconds=MIN_INTERVAL_SECONDS):
    """Speech-only analysis of ONE unpadded source span; see ``span_activity_intervals``.

    Returns ``{'intervals': [(start_s, end_s), ...], 'reference_db', 'threshold_db', 'window': (start_s, end_s)}``;
    ``reference_db`` and ``threshold_db`` are None when no frame of the unpadded span is above the noise floor
    (``intervals`` is then []). ``window`` is the clamped padded window that was analysed.
    """
    rate = _rate(rate)
    stem = _check_stem(stem)
    begin, end, origin = _seconds(begin, 'begin'), _seconds(end, 'end'), _seconds(origin, 'origin')
    before, after = _seconds(pad_before, 'pad_before'), _seconds(pad_after, 'pad_after')
    pct = _seconds(percentile, 'percentile')
    if end <= begin:
        raise ValueError('Span end must follow its start')
    if before < 0 or after < 0:
        raise ValueError('Span padding must not be negative')
    if not 0 <= pct <= 100:
        raise ValueError('Reference percentile must be within 0..100')
    origin_sample = round(origin * rate)
    begin_sample = round(begin * rate) - origin_sample
    end_sample = round(end * rate) - origin_sample
    left = max(0, round((begin - before) * rate) - origin_sample)
    right = min(len(stem), round((end + after) * rate) - origin_sample)
    left = min(left, len(stem))                        # a span past the end of the stem gives an empty window, not a negative one
    right = max(right, left)
    hop = _hop(rate)
    result = {'intervals': [], 'reference_db': None, 'threshold_db': None,
              'window': ((origin_sample + left) / rate, (origin_sample + max(left, right)) / rate)}
    if right - left < hop:
        _stem_window(stem, left, max(left, right), lfe_index)          # validate whatever is there, then: nothing to find
        return result
    db = 20 * np.log10(_frame_levels_with_tail(_stem_window(stem, left, right, lfe_index), hop) + 1e-9)
    starts = left + np.arange(len(db)) * hop
    stops = np.minimum(right, starts + hop)
    centres = (starts + stops) / 2.0
    floor = max(float(noise_floor_db), FLOOR_DB)
    inside = (db > floor) & (centres >= begin_sample) & (centres < end_sample)
    if not inside.any():
        return result
    reference = float(np.percentile(db[inside], pct))
    threshold = max(floor, min(reference + float(rel_db), float(abs_speech_db)))
    result.update(reference_db=reference, threshold_db=threshold)
    active = db > threshold
    if not active.any():
        return result
    closed = _bridge_and_dilate(active, hang, edge)
    edges = np.diff(np.concatenate(([0], closed.astype(np.int8), [0])))
    first_frames, last_frames = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    floor_samples = max(1, round(float(min_seconds) * rate))
    for first, last in zip(first_frames, last_frames):
        a, b = int(starts[first]), int(stops[last - 1])
        if b - a >= floor_samples:
            result['intervals'].append(((origin_sample + a) / rate, (origin_sample + b) / rate))
    return result


def span_activity_intervals(stem, rate, begin, end, origin, **options):
    """Absolute ``(start_s, end_s)`` intervals where the separated stem is speaking, for ONE unpadded source span.

    ``begin``/``end`` are the UNPADDED source speech span (absolute seconds). Activity is searched inside
    ``[begin - pad_before, end + pad_after]`` (defaults .12 / .25 s) clamped to the stem; the level reference is the
    ``REFERENCE_PERCENTILE`` of the above-noise 10 ms frame levels inside the unpadded span only; a frame is speech when
    above ``max(NOISE_FLOOR_DB, min(reference_db - 28, ABS_SPEECH_DB))``; gaps up to 40 ms are bridged and the mask is
    dilated by 20 ms, never beyond the padded window. A span partly or wholly outside the stem is clamped.
    ``[]`` when the stem is silent (<= -90 dBFS), too short, outside the span, or nothing in the unpadded span rises above
    the noise floor: the CALLER then keeps the padded span (removing what is, at most, noise from the stem is harmless;
    leaving speech is not). Non-finite samples or invalid geometry raise ValueError. See the module docstring.
    """
    return span_activity_details(stem, rate, begin, end, origin, **options)['intervals']


def merge_intervals(intervals, join_gap=MERGE_JOIN_SECONDS):
    """Sorted, non-overlapping ``(start, end)`` tuples: overlapping, touching and near-touching ones are joined.

    Intervals less than ``join_gap`` seconds apart (default 2 ms) are joined, which removes the one-sample notch that
    ``replacement_envelope`` cuts where two intervals touch (each one fades to zero at its own edge). Intervals with
    ``end <= start`` carry no samples and are dropped; non-finite values or a negative ``join_gap`` raise ValueError.
    """
    gap = _seconds(join_gap, 'join_gap')
    if gap < 0:
        raise ValueError('join_gap must not be negative')
    pairs = []
    for first, last in intervals:
        first, last = _seconds(first, 'interval start'), _seconds(last, 'interval end')
        if last > first:
            pairs.append((first, last))
    pairs.sort()
    merged = []
    for first, last in pairs:
        if merged and first - merged[-1][1] <= gap + 1e-12:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def active_rms_db(x, rate, rel_db=REL_DB):
    """Loudness (dB re full scale) of the frames of one mono signal that actually carry speech.

    Frames are FRAME_SECONDS long; a frame is active when within ``-rel_db`` (28 dB) of the loudest frame. The
    result is the RMS (dB) over the active frames only, so pauses do not understate the line. A clip shorter than one
    frame is measured as a single frame. Empty or silent input (loudest frame below -120 dBFS) returns SILENCE_DB.
    """
    rate = _rate(rate)
    signal = np.asarray(x, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError('active_rms_db measures one channel')
    if not np.isfinite(signal).all():
        raise ValueError('Loudness measurement contains nonfinite samples')
    if not len(signal):
        return SILENCE_DB
    hop = _hop(rate)
    rms = _frame_rms(signal, hop) if len(signal) >= hop else np.asarray([math.sqrt(float(np.mean(np.square(signal))))])
    if float(rms.max()) < SILENCE_RMS:
        return SILENCE_DB
    level = rms + 1e-9                      # same epsilon as the approved demo
    db = 20 * np.log10(level)
    active = db > db.max() + float(rel_db)
    return float(20 * np.log10(math.sqrt(float(np.mean(np.square(level[active]))))))
