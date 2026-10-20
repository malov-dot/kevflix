"""Caption-target caller flow for ActorPipeline (actor_beta.caption_targets; default off). Stdlib only at import.

Explicit policy layer between the SOURCE view (raw cues; untouched) and the TARGET view:
  1. plan_window: program-verified caption authority -> packaged planner (coverage union, same-actor parts).
  2. release_word_uncertainty: for source cues of usable targets ONLY, remove the source-WORD uncertainty issues
     ('uncertain_transcript', 'unclear_dialogue_without_aligned_caption'); raw flags/text stay untouched.
  3. Whole-utterance ownership (utterance-start-v1): a target is ONE utterance owned by the render range holding its
     first source cue start (the planner's owned_by_this_window and B's gate). A complete, bounded utterance is not
     refused because its caption or later source cues cross a render boundary: the owner ('owned') renders it once and
     the later window ('carry') consumes its own-start sources of it, so English is never produced twice. The owner
     authorizes English only for a bounded utterance: every source cue and authorized span lies strictly inside its
     analysed window (a cue touching the window edge may be truncated -> original), no source cue overlaps another
     voice, and every source cue starting after its render range carries the owner-side simultaneous-voices check
     (cue['turn_overlap'] is False; missing/True -> original). A carry window authorizes nothing. Refusals carry a
     relay.movie_faults record (recoverable = cue uncertainty) in plan['target_faults'].
     settle (after the policy recompute): a target whose source cue is still kept original (policy keep, persisted
     protection, speaker/reference fallback ...) is refused and its released issues restored -> its sources stay on
     the existing path. Protected/unknown/overlap spans are never overwritten by a target.
  4. Superseded source cues skip MT and the old aligned-caption branch; unclear superseded cues never feed a
     reference (Russian source text only, clean cues only).
  5. render_view: synthesis units = owned target units (id = stable target id, english = verified caption part,
     authorized spans) + raw cues not consumed; sources of targets owned by another window are not synthesized
     (no duplicate speech; the owner's canonical take carries). The raw lists are never mutated.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .movie_faults import MovieContentFallback   # stdlib-only metadata classes

VERSION = 'caption-target-flow-v2-utterance-start'
UTTERANCE_RULE = 'utterance-start-v1'
EDGE_EPS = 1e-3          # a source cue within 1 ms of the analysed-window edge may be truncated (incomplete evidence)
WORD_KINDS = ('uncertain_transcript', 'unclear_dialogue_without_aligned_caption')
MODULES = ('beta_caption_flow.py', 'beta_caption_planner.py', 'beta_caption_gate.py', 'beta_caption_authority.py',
           'beta_caption_provenance.py', 'beta_stable_gate.py')


def enabled(config):
    data = getattr(config, 'data', config)
    beta = data.get('actor_beta') if isinstance(data, dict) else None
    return isinstance(beta, dict) and beta.get('enabled') is True and beta.get('caption_targets') is True


def load_file(path):
    """relay.embedded_subtitles._file_cues parse (pysubs2 plaintext, actor) for external sidecars."""
    import pysubs2
    parsed = pysubs2.load(str(path), encoding='utf-8')
    return [{'start': x.start / 1000, 'end': x.end / 1000, 'text': x.plaintext.strip(), 'actor': x.name or None}
            for x in parsed if x.plaintext.strip()]


def caption_inputs(config, folder, captions, subtitle_choice, origin, end):
    """Host receipt + exact captions + job selection; None when the flag is off or no selection was supplied."""
    if not enabled(config) or subtitle_choice is None:
        return None
    receipt = getattr(captions, 'subtitle_source', None)
    path = Path(folder) / 'subtitle-source.json'
    on_disk = None
    if path.is_file() and not path.is_symlink():
        try:
            on_disk = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            on_disk = {'unreadable': True}
    if receipt is None:
        receipt = on_disk
    elif on_disk is not None and on_disk != receipt:
        receipt = {'version': None, 'reason': 'host receipt attribute and subtitle-source.json disagree'}
    return {'receipt': copy.deepcopy(receipt), 'captions': [dict(c) for c in captions or ()],
            'expected_choice': copy.deepcopy(subtitle_choice), 'window': [float(origin), float(end)],
            'cache_root': str(Path(config.cache) / 'subtitle-blocks'),
            'resolve_media': getattr(config, 'media_path', None), 'load_file': load_file}


def module_hashes():
    here = Path(__file__).resolve().parent
    return {n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in MODULES}


def key_part(inputs):
    """Prepared-plan key contribution: module hashes + receipt digest + job selection (captions already keyed)."""
    if inputs is None:
        return None
    receipt = inputs.get('receipt') or {}
    return {'flow': VERSION, 'modules': module_hashes(), 'expected_choice': inputs.get('expected_choice'),
            'receipt_content_sha256': receipt.get('content_sha256'), 'receipt_blocks': receipt.get('blocks')}


def plan_window(inputs, *, job_id, revision, source_identity, all_cues, speaker_evidence, issues, origin, end,
                render_start, render_end, caption_text):
    from . import beta_caption_gate as gate
    from . import beta_caption_planner as planner
    if inputs is None:
        return None
    context = {'job_id': job_id, 'revision': revision, 'source_identity': source_identity or {}, 'caption_inputs': inputs}
    authority = gate.authority_from_inputs(context)
    if authority.get('status') != 'ok':
        plan = {'version': planner.VERSION, 'status': 'authority_' + authority.get('status', 'refused'),
                'reason': authority.get('reason'), 'targets': [], 'captions': [], 'superseded_cue_ids': [],
                'carry_target_ids': []}
    else:
        plan = planner.plan_targets(authority=authority,
                                    context={'job_id': job_id, 'revision': revision,
                                             'source_fingerprint': (source_identity or {}).get('source_fingerprint')},
                                    captions=None, all_cues=all_cues, speaker_evidence=speaker_evidence,
                                    window_issues=issues, window={'origin': origin, 'end': end, 'render_start': render_start,
                                                                  'render_end': render_end}, caption_text=caption_text)
    plan['flow'] = VERSION
    plan['utterance_rule'] = UTTERANCE_RULE
    plan['render_range'] = [float(render_start), float(render_end)]
    plan['analysed_range'] = [float(origin), float(end)]
    plan['policy_refused'] = {}
    plan['target_faults'] = {}
    plan['released_word_uncertainty'] = []
    by_id = {c['id']: c for c in all_cues}
    rs, re_ = float(render_start), float(render_end)
    for t in plan['targets']:
        _assign_role(plan, t, by_id, rs, re_, float(origin), float(end))
    _recompute(plan)
    return plan


def _refuse(plan, target_id, code, detail):
    """Record a target refusal: legacy reason string + explicit recoverable fault (sources keep the existing path)."""
    plan['policy_refused'][target_id] = detail
    plan['target_faults'][target_id] = MovieContentFallback(code, detail).record()


def utterance_eligibility(target, by_id, *, render_start, render_end, origin, end):
    """PURE (no mutation): the ONE utterance-start-v1 rule shared by this flow (_assign_role) and the caption gate.

    target: a planner target; by_id: source cue id -> cue (the window's all_cues, with ownership 'turn_overlap').
    Returns {'window_role': 'owned'|'carry', 'utterance_start', 'later_window_source_cue_ids',
    'carried_own_start_cue_ids', 'crosses_render_boundary', 'refusal': None | {'code', 'detail'}}.
    Role is by the FIRST source cue start. Only an owned utterance can be refused (the owner authorizes English;
    a carry window never does): no authorized speech, authorized speech outside the analysed window, a source cue
    touching the analysed-window edge (possibly truncated), a source cue overlapping another voice, or a
    later-window source cue without the simultaneous-voices check. A source cue missing from by_id refuses any role."""
    rs, re_, origin, end = float(render_start), float(render_end), float(origin), float(end)
    ids = list(target['source_cue_ids'])
    missing = [c for c in ids if c not in by_id]
    if missing or not ids:
        return {'window_role': 'owned' if target.get('owned_by_this_window') else 'carry', 'utterance_start': None,
                'later_window_source_cue_ids': [], 'carried_own_start_cue_ids': [], 'crosses_render_boundary': None,
                'refusal': {'code': 'utterance_source_missing',
                            'detail': 'source cue missing from the window: %s' % missing}}
    starts = [float(by_id[c]['start']) for c in ids]
    first = min(starts)
    role = 'owned' if rs <= first < re_ else 'carry'
    # owned: sources whose start lies in a later render range (their window carries this take);
    # carry: this window's own-start sources consumed by the owner's take (never synthesized here).
    later = [c for c, s in zip(ids, starts) if role == 'owned' and s >= re_]
    carried = [c for c, s in zip(ids, starts) if role == 'carry' and rs <= s < re_]
    out = {'window_role': role, 'utterance_start': first, 'later_window_source_cue_ids': later,
           'carried_own_start_cue_ids': carried,
           'crosses_render_boundary': bool(later or carried or any(float(by_id[c]['end']) > re_
                                                                   or float(by_id[c]['start']) < rs for c in ids)),
           'refusal': None}
    if role != 'owned':
        return out                  # the owner authorizes English; a carry window never does (existing carry path)
    edge = [c for c in ids if float(by_id[c]['start']) <= origin + EDGE_EPS or float(by_id[c]['end']) >= end - EDGE_EPS]
    spans = target.get('authorized_audio_spans') or []
    outside = [list(s) for s in spans if float(s[0]) < origin or float(s[1]) > end]
    overlapped = [c for c in ids if by_id[c].get('turn_overlap') is True]
    unchecked = [c for c in later if by_id[c].get('turn_overlap') is not False]
    if not spans:
        refusal = ('utterance_without_authorized_speech', 'target has no authorized source speech span')
    elif outside:
        refusal = ('utterance_outside_analysed_window',
                   'authorized speech outside the analysed window %s: %s' % ([origin, end], outside))
    elif edge:
        refusal = ('utterance_incomplete_at_window_edge',
                   'source cue touches the analysed-window edge (possibly truncated): %s' % edge)
    elif overlapped:
        refusal = ('utterance_overlapping_voices', 'source cue overlaps another voice: %s' % overlapped)
    elif unchecked:
        refusal = ('utterance_overlap_unverified',
                   'later-window source cue lacks the simultaneous-voices check: %s' % unchecked)
    else:
        refusal = None
    out['refusal'] = None if refusal is None else {'code': refusal[0], 'detail': refusal[1]}
    return out


def kept_source_refusal(target, kept):
    """PURE: settle's rule, shared with the caption gate. kept = the target's source cue ids this window keeps
    original (flow: policy kept_ids; gate: own-start sources absent from the window's synthesis cues)."""
    kept = list(kept)
    return None if not kept else {'code': 'utterance_source_kept_original',
                                  'detail': 'source cue kept original by policy: %s' % kept}


def _assign_role(plan, t, by_id, rs, re_, origin, end):
    """Apply utterance_eligibility to a flow target (role fields on the target; refusal -> fault record)."""
    rule = utterance_eligibility(t, by_id, render_start=rs, render_end=re_, origin=origin, end=end)
    for key in ('window_role', 'utterance_start', 'later_window_source_cue_ids', 'carried_own_start_cue_ids',
                'crosses_render_boundary'):
        t[key] = rule[key]
    if rule['refusal'] is not None:
        _refuse(plan, t['target_id'], rule['refusal']['code'], rule['refusal']['detail'])


def _usable(plan):
    return [t for t in plan['targets'] if t['target_id'] not in plan['policy_refused']]


def _recompute(plan):
    usable = _usable(plan)
    plan['superseded_cue_ids'] = [c for t in usable if t['window_role'] == 'owned' for c in t['source_cue_ids']]
    plan['carry_source_cue_ids'] = [c for t in usable if t['window_role'] == 'carry' for c in t['source_cue_ids']]
    plan['carry_target_ids'] = [t['target_id'] for t in usable if t['window_role'] == 'carry']


def consumed_ids(plan):
    return set() if not plan else set(plan['superseded_cue_ids']) | set(plan['carry_source_cue_ids'])


def release_word_uncertainty(issues, plan):
    """In place: drop word-uncertainty issues of usable target sources; remember them for restoration."""
    if not plan:
        return
    ids = consumed_ids(plan)
    keep = []
    for issue in issues:
        if isinstance(issue, dict) and issue.get('kind') in WORD_KINDS and issue.get('cue') in ids:
            plan['released_word_uncertainty'].append(copy.deepcopy(issue))
        else:
            keep.append(issue)
    issues[:] = keep


def settle(plan, kept_ids, issues):
    """After the policy recompute: refuse targets whose sources are still kept original; restore their issues.
    Returns True when issues were restored (caller re-runs its policy)."""
    if not plan:
        return False
    restored = False
    plan.setdefault('target_faults', {})
    for t in _usable(plan):
        refusal = kept_source_refusal(t, [c for c in t['source_cue_ids'] if c in kept_ids])
        if refusal is not None:
            _refuse(plan, t['target_id'], refusal['code'], refusal['detail'])
    refused_sources = {c for t in plan['targets'] if t['target_id'] in plan['policy_refused'] for c in t['source_cue_ids']}
    back = [i for i in plan['released_word_uncertainty'] if i.get('cue') in refused_sources]
    if back:
        issues.extend(copy.deepcopy(back))
        plan['released_word_uncertainty'] = [i for i in plan['released_word_uncertainty'] if i not in back]
        restored = True
    _recompute(plan)
    return restored


def skip_source_path(plan, cue):
    """True for raw cues consumed by a usable target: no MT, no old aligned-caption branch."""
    return bool(plan) and cue['id'] in consumed_ids(plan)


def reference_cues(plan, cues):
    """Keep raw source cues eligible even when a caption target supplies their uncertain words.

    Speaker/overlap/excerpt quality is checked by make_reference and the source-bound admission gate. An unclear
    contributing cue makes make_reference return an empty SOURCE transcript, selecting reference-only cloning.
    No target/caption text is copied into these raw cues or used as the reference transcript.
    """
    return list(cues)


PLACEMENT_POLICY = 'verified-same-actor-quiet-gap-v1'


def _gaps(spans):
    return [[a[1], b[0]] for a, b in zip(spans, spans[1:]) if b[0] > a[1]]


def _unit(t, by_id):
    """Target unit. Two time notions, kept explicit:
      * authorized_audio_spans = counted SOURCE speech and the ONLY removal (replacement) mask;
      * [start, end] = permitted PLACEMENT envelope: natural English may span the quiet same-actor gaps inside it
        (the planner proved full coverage and no foreign/unresolved/unrepresented speech there). The gaps are never
        removed, never counted as dubbed speech, and protected/foreign original speech keeps blocking via the walls.
    speech_windows (= authorized spans) keep the source pause visible to dubbing_timing, so the progressive plan still
    reports pause_alignment_required for this unit; nothing clears that issue (the progressive renderer refuses it)."""
    spans = [list(s) for s in t['authorized_audio_spans']]
    return {'id': t['target_id'], 'target_id': t['target_id'], 'start': spans[0][0], 'end': spans[-1][1],
            'speech_start': spans[0][0], 'speech_end': spans[-1][1], 'speaker': t['actor']['actor_id'],
            'english': t['english'], 'translation': t['basis'],
            'text': ' '.join(by_id[c].get('text', '') for c in t['source_cue_ids'] if c in by_id),
            'source_cue_ids': list(t['source_cue_ids']), 'authorized_audio_spans': spans,
            'placement_policy': PLACEMENT_POLICY, 'quiet_gaps': _gaps(spans),
            'speech_windows': [{'start': a, 'end': b} for a, b in spans],
            'caption_target': {'caption': t['caption'], 'source_word_uncertain': t['source_word_uncertain'],
                               'source_speech': t.get('source_speech')},
            'overlapped': False, 'singing': False, 'unclear': False}


def render_view(plan, cues, all_cues):
    """(synthesis units, all-cues view). No plan -> the original list objects unchanged."""
    if not plan or not (plan['superseded_cue_ids'] or plan['carry_source_cue_ids']):
        return cues, all_cues
    by_id = {c['id']: c for c in all_cues}
    owned = {c: t for t in _usable(plan) if t['window_role'] == 'owned' for c in t['source_cue_ids']}
    consumed = consumed_ids(plan)
    units, emitted = [], set()
    for cue in cues:
        t = owned.get(cue['id'])
        if t is not None:
            if t['target_id'] not in emitted:
                units.append(_unit(t, by_id)); emitted.add(t['target_id'])
        elif cue['id'] not in consumed:
            units.append(cue)
    view = [c for c in all_cues if c['id'] not in consumed]
    view += [_unit(t, by_id) for t in _usable(plan)]          # owned AND carry units keep their speech walls/budgets
    view.sort(key=lambda c: (c['start'], c['end'], c['id']))
    return units, view


def replacement_intervals(cue):
    """Removal envelope: target units replace only their authorized spans; raw cues keep the existing rule."""
    if cue.get('authorized_audio_spans'):
        return [(a - .12, b + .25) for a, b in cue['authorized_audio_spans']]
    return [(cue['start'] - .12, cue['end'] + .25)]


def line_fields(cue):
    """Extra fields on a rendered line (-> ledger item): a target take remembers its authorized spans so canonical
    reuse and cross-window carry also remove ONLY source speech. Raw cues: nothing (existing behaviour)."""
    if not cue.get('authorized_audio_spans'):
        return {}
    return {'target_id': cue['target_id'], 'source_cue_ids': list(cue['source_cue_ids']),
            'authorized_audio_spans': [list(s) for s in cue['authorized_audio_spans']], 'placement_policy': PLACEMENT_POLICY}


def canonical_intervals(item):
    """Envelope for a canonical reused/carried take: authorized spans when recorded, else the existing replacement span."""
    if item.get('authorized_audio_spans'):
        return [(a - .12, b + .25) for a, b in item['authorized_audio_spans']]
    return [(item['replacement_start'], item['replacement_end'])]


def mix_levels(measure, speech, voice, cue, origin, rate, lfe):
    """Pan/gain for a target from its authorized SOURCE speech only (a quiet gap would understate the level).
    Raw cues: exactly the existing call."""
    spans = cue.get('authorized_audio_spans')
    if not spans:
        return measure(speech, voice, cue, origin, rate, lfe_index=lfe)
    import numpy as np
    pieces = [speech[max(0, round((a - origin) * rate)):min(len(speech), round((b - origin) * rate))] for a, b in spans]
    joined = np.concatenate([p for p in pieces if len(p)]) if any(len(p) for p in pieces) else speech[:0]
    pan, gain, levels = measure(joined, voice, {'start': 0.0, 'end': len(joined) / rate}, 0.0, rate, lfe_index=lfe)
    levels = dict(levels, level_basis='authorized-source-speech-spans', authorized_audio_spans=[list(s) for s in spans],
                  quiet_gaps=_gaps([list(s) for s in spans]), source_start=spans[0][0], source_end=spans[-1][1])
    return pan, gain, levels


def annotate_publication(report, source_all_cues):
    """After stamp_published: every published/placement row of a target carries target id, source cue ids and
    authorized spans; the raw SOURCE view is attached for an independent scorer. Attempted rows stay separate."""
    plan = report.get('caption_target_plan')
    if not plan:
        return
    targets = {t['target_id']: t for t in plan.get('targets') or ()}
    for row in list(report.get('published_utterances') or ()) + list(report.get('placements') or ()):
        t = targets.get(row.get('cue_id'))
        if t is not None:
            spans = [list(s) for s in t['authorized_audio_spans']]
            gaps = _gaps(spans)
            row.update(target_id=t['target_id'], source_cue_ids=list(t['source_cue_ids']), authorized_audio_spans=spans,
                       basis=t['basis'], placement_policy=PLACEMENT_POLICY, quiet_gaps=gaps,
                       window_role=t.get('window_role'), utterance_start=t.get('utterance_start'),
                       later_window_source_cue_ids=list(t.get('later_window_source_cue_ids') or ()),
                       dubbed_source_speech_seconds=round(sum(b - a for a, b in spans), 6),
                       quiet_gap_seconds_not_counted=round(sum(b - a for a, b in gaps), 6))
    report['caption_target_source_view'] = [
        {'id': c['id'], 'start': c['start'], 'end': c['end'], 'text': c.get('text'), 'speaker': c.get('speaker'),
         'local_speaker': c.get('local_speaker'), 'unclear': bool(c.get('unclear')), 'overlapped': bool(c.get('overlapped')),
         'word_count': len(c.get('words') or ()), 'consumed_by_target': c['id'] in consumed_ids(plan)}
        for c in source_all_cues]
