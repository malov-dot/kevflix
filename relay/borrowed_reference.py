"""Borrowed line references (owner decision 2026-10-19: every spoken line is dubbed).

A line whose own audio cannot be a voice reference (too short, overlapped by laughter, no usable embedding) may borrow the
admitted own-line reference of another cue of the same window. The reviewed stable gate (beta_stable_gate.py, pinned by
sha256 in three places) is NOT changed: it still refuses such a cue with 'line reference belongs to another cue'. This
wrapper re-admits exactly the cues that declare a borrow AND whose donor the reviewed gate itself admitted, and nothing else.

A borrow is admitted only when all of these hold (anything else keeps the reviewed gate's refusal):
  * the reviewed gate refused the cue for exactly that reason (no other refusal is overridden);
  * the cue declares voice_source.mode == 'borrowed-reference' and names a donor cue of this window, never itself;
  * the cue's speaker is the donor's own-line ident ('line-' + donor id), which the reviewed gate admitted with the
    donor's own reference (so the reference is a validated source excerpt of the donor's own span and raw turn);
  * that admitted voice's line_reference names the donor, and the donor cue carries the same speaker;
  * the cue is not unclear or overlapped (the checks the reviewed gate runs after the reference check).
"""
from __future__ import annotations

import math

REFUSAL = 'line reference belongs to another cue'
MODE = 'borrowed-reference'

# Donor choice (who lends a reference to a refused line). choose_reference_donors below replaces the ranking of
# actor_pipeline.choose_reference_donors (same arguments, same pick format, additive fields only).
DONOR_MAX_GAP_SECONDS = 36.0       # same value as actor_pipeline.REFERENCE_DONOR_MAX_GAP_SECONDS
DONOR_PITCH_SEMITONES = 5.0        # same value as actor_pipeline.REFERENCE_DONOR_PITCH_SEMITONES
DONOR_ADJACENT_SECONDS = 2.0       # a donor this close (or closer) to the borrowing line counts as adjacent in time
PREFERENCE_VERSION = 'borrow-preference-v1'


def _refined_label(cue):
    """The cue's utterance-refined window speaker label (cue['local_speaker']), None when it has none."""
    label = cue.get('local_speaker')
    return label if isinstance(label, str) and label.strip() else None


def _dominant_raw_label(cue, speech_regions):
    from .actor_pipeline import dominant_reference_label
    try:
        return dominant_reference_label(cue, speech_regions)
    except ValueError:
        return None


def choose_reference_donors(cues, refused_ids, usable, speech_regions, *, max_gap=DONOR_MAX_GAP_SECONDS, raw_label=None,
                            strict=False):
    """Who lends a reference to each refused (too short, overlapped, no embedding) line of this window. Every refused line
    that has ANY usable donor within max_gap gets one (owner rule: every line is dubbed); a refused line with none is absent
    and keeps its original audio. Returns {cue_id: {'donor', 'same_raw_label', 'gap_seconds', 'pitch_mismatch'} + additive
    'label_relation', 'adjacent', 'borrower_local_speaker', 'donor_local_speaker', 'preference'}.

    Ranking, first difference wins (usable(cue) -> bool says which cues own an admitted voice):
      1. never a donor whose refined speaker label CONTRADICTS the borrower's (both known and different) while any other
         donor exists; a lone contradicting donor is still used (label_relation 'contradicts' in the pick),
      2. a donor whose median pitch is within DONOR_PITCH_SEMITONES of the line's (when both are known),
      3. a donor with the borrower's speaker label (equal refined label; with an unknown label on either side, the same
         dominant raw diarization label) who is also adjacent in time (gap <= DONOR_ADJACENT_SECONDS),
      4. any donor with an equal refined label, 5. the same dominant raw label, 6. the nearest in time, 7. the longer cue.
    ``raw_label(cue, speech_regions) -> label|None`` defaults to actor_pipeline.dominant_reference_label.

    strict=True (owner rule 2026-10-20: every line owns its clone; a borrow is the LAST resort and never a guessed voice):
    only a PURE donor qualifies - the same dominant raw diarization label as the borrower (an unknown label on either side
    is not a match), a refined speaker label that does not contradict the borrower's, and a median pitch that is not far
    from the borrower's. With no pure donor the line is absent from the result and keeps its original audio."""
    label_of = raw_label or _dominant_raw_label

    def span(cue):
        return float(cue.get('speech_start') or cue['start']), float(cue.get('speech_end') or cue['end'])
    donors = [c for c in cues if c['id'] not in refused_ids and not (c.get('overlapped') or c.get('turn_overlap') is True)
              and usable(c)]
    picks = {}
    for cue in cues:
        if cue['id'] not in refused_ids:
            continue
        mine, mine_refined = label_of(cue, speech_regions), _refined_label(cue)
        begin, stop = span(cue)
        ranked = []
        for donor in donors:
            x, y = span(donor)
            gap = max(0., x - stop, begin - y)
            if gap > max_gap:
                continue
            same = mine is not None and label_of(donor, speech_regions) == mine
            theirs = _refined_label(donor)
            relation = 'unknown' if mine_refined is None or theirs is None else 'same' if mine_refined == theirs else 'contradicts'
            adjacent = gap <= DONOR_ADJACENT_SECONDS
            f1, f2 = cue.get('f0_median_hz'), donor.get('f0_median_hz')
            far = bool(f1 and f2 and abs(12 * math.log2(float(f1) / float(f2))) > DONOR_PITCH_SEMITONES)
            if strict and (not same or relation == 'contradicts' or far):
                continue
            preferred = adjacent and (relation == 'same' or (relation == 'unknown' and same))
            key = (1 if relation == 'contradicts' else 0, 1 if far else 0, 0 if preferred else 1,
                   0 if relation == 'same' else 1, 0 if same else 1, round(gap, 6), -(y - x), donor['id'])
            ranked.append((key, same, gap, far, relation, adjacent, theirs, donor))
        if ranked:
            key, same, gap, far, relation, adjacent, theirs, donor = min(ranked, key=lambda item: item[0])
            picks[cue['id']] = {'donor': donor, 'same_raw_label': same, 'gap_seconds': round(gap, 3), 'pitch_mismatch': bool(far),
                                'label_relation': relation, 'adjacent': adjacent, 'borrower_local_speaker': mine_refined,
                                'donor_local_speaker': theirs, 'preference': PREFERENCE_VERSION, 'strict': bool(strict)}
    return picks


def borrowed_voice(cue, voices, all_cues):
    """The donor's admitted voice when `cue` legitimately borrows it, else None."""
    if not isinstance(cue, dict) or cue.get('unclear') or cue.get('overlapped'):
        return None
    source = cue.get('voice_source')
    if not isinstance(source, dict) or source.get('mode') != MODE:
        return None
    donor_id, ident = source.get('donor_cue_id'), cue.get('speaker')
    if not isinstance(donor_id, str) or not donor_id or donor_id == cue.get('id') or ident != 'line-' + donor_id:
        return None
    voice = voices.get(ident)
    if not isinstance(voice, dict) or (voice.get('line_reference') or {}).get('cue_id') != donor_id:
        return None
    donor = all_cues.get(donor_id)
    if not isinstance(donor, dict) or donor.get('speaker') != ident:
        return None
    return voice


def admit_declared_borrows(context, bundle):
    """`bundle` is the reviewed gate's make_pipeline_gate result; returns it with the legitimate borrows admitted."""
    fallbacks = bundle.get('fallbacks')
    if not isinstance(fallbacks, dict) or REFUSAL not in fallbacks.values():
        return bundle
    all_cues = {c.get('id'): c for c in (context.get('all_cues') or ()) if isinstance(c, dict)}
    voices = bundle.get('voices') or {}
    admitted = {}
    for cue in context.get('cues') or ():
        cue_id = cue.get('id') if isinstance(cue, dict) else None
        if fallbacks.get(cue_id) != REFUSAL:
            continue
        voice = borrowed_voice(cue, voices, all_cues)
        if voice is not None:
            admitted[cue_id] = {'mode': 'actor-reference', 'voice': voice, 'borrowed_reference': True}
    if not admitted:
        return bundle
    base_resolve = bundle['resolve_cue']

    def resolve_cue(key):
        return dict(admitted[key]) if key in admitted else base_resolve(key)

    remaining = {k: v for k, v in fallbacks.items() if k not in admitted}
    receipt = dict(bundle.get('receipt') or {})
    receipt.update(admitted_cues=list(receipt.get('admitted_cues') or []) + sorted(admitted), fallbacks=dict(remaining),
                   borrowed_references={k: v['voice'].get('actor_id') for k, v in admitted.items()})
    return {**bundle, 'resolve_cue': resolve_cue, 'fallbacks': remaining,
            'admitted_cues': list(bundle.get('admitted_cues') or []) + sorted(admitted), 'receipt': receipt}


class BorrowAwareGate:
    """The reviewed stable gate module with borrow admission added to make_pipeline_gate; every other attribute is the
    reviewed module's own."""

    def __init__(self, stable):
        self._stable = stable

    def __getattr__(self, name):
        return getattr(self._stable, name)

    def make_pipeline_gate(self, context, **kwargs):
        return admit_declared_borrows(context, self._stable.make_pipeline_gate(context, **kwargs))
