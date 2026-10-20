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

REFUSAL = 'line reference belongs to another cue'
MODE = 'borrowed-reference'


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
