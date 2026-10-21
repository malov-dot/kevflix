"""Speech that no take replaces stays the film's own sound: the removal of a dubbed line stops short of it.

A dubbed line removes the original voice over its source span PLUS a pad (``take_removal_intervals``: 0.12 s before, 0.25 s after,
only where the separated stem is active). The pad exists to catch the first and last syllable that the recognizer's times miss. When
another voice follows or precedes the line without a pause, a word the recognizer never transcribed (a one-word answer, an aside,
the continuation of a sentence that was cut at an analysis edge) lies inside that pad, and its edge was removed too: heard as a word
that starts or ends in a hole.

Rule (owner, 2026-10-20: a word that is left original must not be cut):

  * cores   the source speech spans of every take mixed in the window (new, reused and carried). Whatever lies inside a core is
            replaced by its dub and is removed as before; a second voice inside a core cannot be kept without its language
            bleeding under the English (not handled here).
  * words   diarized speech (``raw_turns``) outside every core, grown by CORE_GRACE_* around it so the last syllable of a dubbed line
            is still removed, and at least WORD_MIN_SECONDS long: shorter leftovers beside a core are word tails the recognizer's
            end time missed, not words. Given the separated stem, a leftover counts only where the stem is really speaking
            (speech_activity.span_activity_intervals, bridged over pauses of up to WORD_BRIDGE_SECONDS), so a diarizer turn that
            runs longer than the voice does not turn a silent margin into a word.
  * zones   each word grown by KEEP_PAD_SECONDS on both sides, minus the cores and their grace. The removal intervals of the mix
            lose exactly the zones, so the word and a margin of KEEP_PAD_SECONDS around it are the original, bit for bit apart from
            the 10 ms envelope fade, and every core is removed as it was.

Stdlib only; spans are absolute seconds.
"""
from __future__ import annotations

import math

VERSION = 'kept-speech-v1'
WORD_MIN_SECONDS = .30           # a leftover this long is a word; shorter beside a core is the tail of the dubbed word
KEEP_PAD_SECONDS = .12           # margin kept around a word (the pad that is not the word's own is never removed)
CORE_GRACE_AFTER_SECONDS = .08   # the tail of a dubbed line that is still removed after its source end
CORE_GRACE_BEFORE_SECONDS = .03  # and the onset before its source start
MIN_PIECE_SECONDS = .002         # a removal remnant shorter than this is dropped (merge_intervals joins at .002)
WORD_BRIDGE_SECONDS = .25        # stem activity separated by less than this belongs to one word / phrase


def _clean(spans):
    out = []
    for span in spans or ():
        try:
            a, b = float(span[0]), float(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(a) and math.isfinite(b) and b > a:
            out.append((a, b))
    return out


def merge(spans):
    out = []
    for a, b in sorted(_clean(spans)):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(spans, holes):
    out = []
    holes = merge(holes)
    for a, b in merge(spans):
        cursor = a
        for c, d in holes:
            if d <= cursor or c >= b:
                continue
            if c > cursor:
                out.append((cursor, c))
            cursor = max(cursor, d)
            if cursor >= b:
                break
        if cursor < b:
            out.append((cursor, b))
    return out


def record_spans(record):
    """The source speech a take (or the cue behind it) replaces, with no pad, in the order ``take_removal_intervals`` reads a
    record: authorized spans, else source start/end, else the recorded replacement span (taken whole: nothing inside it is carved),
    else start/end. Empty when the record names none."""
    if record.get('authorized_audio_spans'):
        return _clean(record['authorized_audio_spans'])
    for first, last in (('source_start', 'source_end'), ('replacement_start', 'replacement_end'), ('start', 'end')):
        if record.get(first) is not None and record.get(last) is not None:
            return _clean([(record[first], record[last])])
    return []


def _active_extents(pieces, stem, rate, origin, lfe_index):
    """The parts of ``pieces`` in which ``stem`` is speaking, over pauses shorter than WORD_BRIDGE_SECONDS joined."""
    from .speech_activity import merge_intervals, span_activity_intervals
    found = []
    for a, b in pieces:
        try:
            found.extend(span_activity_intervals(stem, rate, a, b, origin, pad_before=0., pad_after=0., lfe_index=lfe_index))
        except ValueError:
            continue                 # an unusable stem proves nothing: the removal stays as it was
    return merge_intervals(found, WORD_BRIDGE_SECONDS) if found else []


def keep_zones(cores, turns, *, stem=None, rate=None, origin=0., lfe_index=None, word_min=WORD_MIN_SECONDS, pad=KEEP_PAD_SECONDS,
               grace_after=CORE_GRACE_AFTER_SECONDS, grace_before=CORE_GRACE_BEFORE_SECONDS):
    """[(start, end)] absolute spans where no dub removal may reach: the undubbed words of ``turns`` with their margin.
    ``stem`` (n, channels) from ``origin`` at ``rate``: the stem the removal subtracts; leftovers are then cut down to where it speaks."""
    cores = merge(cores)
    graced = merge([(a - grace_before, b + grace_after) for a, b in cores])
    speech = merge([(t['start'], t['end']) if isinstance(t, dict) else t for t in turns or ()])
    pieces = subtract(speech, graced)
    if stem is not None and pieces:
        pieces = subtract(_active_extents(pieces, stem, rate, origin, lfe_index), graced)
    words = [(a, b) for a, b in pieces if b - a >= word_min]
    return subtract([(a - pad, b + pad) for a, b in words], graced)


def carve(intervals, cores, turns, **options):
    """``intervals`` (the removal spans of every take in the window) without the keep zones. Returns (intervals, zones, seconds):
    the new sorted list, the zones that cut into it, and how many seconds of removal they took back. Nothing to keep (no turns, no
    undubbed word) returns the intervals unchanged, in their order. ``options`` as keep_zones."""
    zones = keep_zones(cores, turns, **options)
    if not zones:
        return list(intervals), [], 0.
    before = sum(b - a for a, b in merge(intervals))
    carved = [(a, b) for a, b in subtract(intervals, zones) if b - a >= MIN_PIECE_SECONDS]
    after = sum(b - a for a, b in merge(carved))
    touched = [z for z in zones if any(min(b, z[1]) > max(a, z[0]) for a, b in merge(intervals))]
    return carved, touched, before - after
