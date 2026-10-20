"""Caption-target gate, packaged relay module (derived from caption-target-v1 gate aef09667...; same decisions).

make_caption_target_gate(context) = stable gate (byte-identical beta_stable_gate.py, 95a33d8a...) for STRICT
source/stem/model/actor/reference admission, plus subtitle-primary TARGET decisions recomputed here:
  * the caption authority is RE-VERIFIED from the host receipt in context['caption_inputs'] (never trusted from a
    caller dict); the target plan is recomputed from raw all_cues/speaker_evidence/window issues;
  * a target renders only if eligible, owned by this window and its actor admitted by the stable gate; ownership is
    whole-utterance by the FIRST source cue start (planner owned_by_this_window; beta_caption_flow utterance-start-v1),
    so a later source cue starting in the next window resolves there to target-owned-by-other-window (carried once);
  * the SAME pure rules as the caller flow (beta_caption_flow.utterance_eligibility + kept_source_refusal) are applied
    to this gate's own recomputed plan, so the gate never admits a target the flow refused and never marks a raw cue
    superseded-by-target when no target renders (a refused target's raw cues keep the stable decision);
  * fallback decisions carry a relay.movie_faults record ('fault'; recoverable = cue/actor uncertainty);
  * resolve_cue(target_id) returns the target decision (A's automatic renderer resolves context_key['cue_id']);
    resolve_cue(raw source cue) returns superseded-by-target / target-owned-by-other-window / the stable decision.
Reference text stays the stable gate's Russian source transcript; English is target wording only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import beta_caption_authority as AU
from . import beta_caption_flow as ctf      # pure shared eligibility rules (flow imports this gate only lazily)
from . import beta_caption_planner as tp
from .movie_faults import MovieContentFallback

GATE_VERSION = 'caption-target-gate-v1.4-packaged'
STABLE_SHA256 = 'e9a5be1d7b7090ea3bdac5cbc600b7aa0d748039760076fa3116c25755898bdf'
FALLBACK = 'original-audio-fallback'
REQUIRED = ('caption_inputs', 'all_cues', 'speaker_evidence', 'window_issues', 'window')


def load_stable():
    path = Path(__file__).with_name('beta_stable_gate.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != STABLE_SHA256:
        raise RuntimeError('packaged stable gate drifted from the reviewed version')
    from . import beta_stable_gate
    return beta_stable_gate


def authority_from_inputs(context):
    inputs = context.get('caption_inputs') or {}
    sid = context.get('source_identity') or {}
    return AU.verify(receipt=inputs.get('receipt'), captions=inputs.get('captions') or [],
                     expected_choice=inputs.get('expected_choice'), job_id=context.get('job_id'),
                     revision=context.get('revision'), source_fingerprint=sid.get('source_fingerprint'),
                     window=inputs.get('window'), cache_root=inputs.get('cache_root'),
                     resolve_media=inputs.get('resolve_media'), load_file=inputs.get('load_file'))


def make_caption_target_gate(context, *, caption_text, stable_gate_module=None, reference_cache_module=None):
    if 'trusted' in context or 'target_plan' in context or 'caption_authority' in context:
        raise ValueError('caller trust flags / caller plans / caller authorities are refused')
    stable = stable_gate_module or load_stable()
    bundle = stable.make_pipeline_gate(context, reference_cache_module=reference_cache_module)
    missing = [k for k in REQUIRED if context.get(k) is None]
    sid = context.get('source_identity') or {}
    authority = None
    if missing:
        plan = {'version': tp.VERSION, 'status': 'context_refused', 'reason': 'missing caller fields: %s' % missing,
                'targets': [], 'captions': [], 'superseded_cue_ids': [], 'carry_target_ids': []}
    else:
        authority = authority_from_inputs(context)
        if authority.get('status') != 'ok':
            plan = {'version': tp.VERSION, 'status': 'authority_' + authority.get('status', 'refused'),
                    'reason': authority.get('reason'), 'targets': [], 'captions': [], 'superseded_cue_ids': [],
                    'carry_target_ids': []}
        else:
            plan = tp.plan_targets(authority=authority,
                                   context={'job_id': context.get('job_id'), 'revision': context.get('revision'),
                                            'source_fingerprint': sid.get('source_fingerprint')},
                                   captions=None, all_cues=context['all_cues'], speaker_evidence=context['speaker_evidence'],
                                   window_issues=context['window_issues'], window=context['window'], caption_text=caption_text)
    actors = (bundle.get('receipt') or {}).get('actors') or {}
    stable_refused = (bundle.get('receipt') or {}).get('context_refused')
    decisions, eligible = {}, []
    # The flow's own pure rules, applied to THIS gate's independently recomputed plan (never a caller plan):
    # utterance_eligibility (first-start role, bounded/complete, overlap) and kept_source_refusal (settle).
    by_id = {c['id']: c for c in context.get('all_cues') or () if isinstance(c, dict) and 'id' in c}
    synthesis_ids = {c['id'] for c in context.get('cues') or () if isinstance(c, dict) and 'id' in c}
    win = context.get('window') if isinstance(context.get('window'), dict) else {}
    # Kept-original set: the caller's settle set when supplied (sorted list of str cue ids; malformed -> refused),
    # else inferred (own-start source in the render range but absent from the window's synthesis cues).
    kept_supplied = context.get('kept_original_cue_ids')
    if kept_supplied is not None and (not isinstance(kept_supplied, list)
                                      or not all(isinstance(c, str) and c for c in kept_supplied)):
        raise ValueError('kept_original_cue_ids must be a list of cue id strings')
    kept_set = None if kept_supplied is None else set(kept_supplied)

    def kept_of(t):
        if kept_set is not None:
            return [c for c in t['source_cue_ids'] if c in kept_set]
        return [c for c in t['source_cue_ids'] if c in by_id and c not in synthesis_ids
                and float(win['render_start']) <= float(by_id[c]['start']) < float(win['render_end'])]

    for t in plan['targets']:
        rule = ctf.utterance_eligibility(t, by_id, render_start=win['render_start'], render_end=win['render_end'],
                                         origin=win['origin'], end=win['end'])
        # Both shared refusals first, for owned AND carry targets (flow.settle checks kept sources for both roles);
        # a refused target's raw cues keep the stable decision. Carry only when neither refusal applies.
        refusal = rule['refusal'] or ctf.kept_source_refusal(t, kept_of(t))
        if refusal is not None:
            decisions[t['target_id']] = {'mode': FALLBACK, 'reason': refusal['detail'],
                                         'fault': MovieContentFallback(refusal['code'], refusal['detail']).record()}
            continue
        if rule['window_role'] != 'owned':
            decisions[t['target_id']] = {'mode': 'target-owned-by-other-window', 'reason': 'carry; rendered once by its owner'}
            continue
        eligible.append(t)
        actor = t['actor']['actor_id']
        rec = actors.get(actor) or {}
        if stable_refused or not rec.get('admitted') or actor not in bundle['voices']:
            reason = 'actor not admitted by stable gate: %s' % (stable_refused or rec.get('reason') or 'unknown actor')
            decisions[t['target_id']] = {'mode': FALLBACK, 'reason': reason,
                                         'fault': MovieContentFallback('target_actor_not_admitted', reason).record()}
        else:
            decisions[t['target_id']] = {'mode': 'actor-reference', 'voice': bundle['voices'][actor], 'target': t}
    by_cue = {}
    for t in plan['targets']:
        for cid in t['source_cue_ids']:
            by_cue[cid] = t
    base_resolve = bundle['resolve_cue']

    def resolve_target(target_id):
        return decisions.get(target_id) or {'mode': FALLBACK, 'reason': 'unknown target',
                                            'fault': MovieContentFallback('unknown_target', 'unknown target').record()}

    def resolve_cue(key):
        if key in decisions:                             # a target unit rendered under its stable target id
            return dict(decisions[key])
        t = by_cue.get(key)
        if t is None:
            return base_resolve(key)
        d = decisions[t['target_id']]
        if d['mode'] == 'actor-reference':
            return {'mode': 'superseded-by-target', 'target_id': t['target_id']}
        if d['mode'] == 'target-owned-by-other-window':
            return {'mode': 'target-owned-by-other-window', 'target_id': t['target_id']}
        return base_resolve(key)                         # target refused -> the raw cue keeps its existing decision

    receipt = {'version': GATE_VERSION, 'stable_gate_sha256': STABLE_SHA256, 'stable_receipt': bundle.get('receipt'),
               'authority': None if authority is None else {k: v for k, v in authority.items() if k != 'captions'},
               'plan': plan, 'target_decisions': {k: {kk: vv for kk, vv in v.items() if kk not in ('voice', 'target')}
                                                  for k, v in decisions.items()},
               'meaning': 'target text = verified caption part; voice/reference admission = stable gate (unchanged); '
                          'provenance only, not identity truth or quality'}
    return {'voices': bundle['voices'], 'reference_gate': bundle['reference_gate'], 'resolve_cue': resolve_cue,
            'resolve_target': resolve_target, 'targets': eligible,
            'admitted_cues': bundle.get('admitted_cues'), 'fallbacks': bundle.get('fallbacks') or [], 'receipt': receipt}
