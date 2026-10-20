"""Subtitle-primary TARGET planner, packaged relay module (derived from caption-target-v1 planner 78f9d15a...).

Packaging changes only: relative imports; caption authority may be the program-verified host authority
(relay.beta_caption_authority.verify: embedded stream or external sidecar) or the reviewed v2 sidecar binding;
stable target ids use the authority namespace + absolute cue identity instead of a file index.

Two explicit views:
  SOURCE view  = the pipeline's raw cues (ids, times, Russian ASR text/words, unclear, overlapped, speaker binding,
                 speaker evidence). Never modified; it stays the authority for original-audio policy and for clean
                 reference creation (Russian reference_text). English never flows into it.
  TARGET view  = derived lines: verbatim English caption part, stable target id, source cue refs with their exact
                 authorized audio spans, actor binding provenance, caption authority, eligibility/refusal.

A caption supplies target wording only when, generically (no film/index/quote rules):
  * the caption authority (expected job/revision/source/subtitle FULL sha/language from the pipeline's own source
    selection) verifies against the subtitle file and the window context;
  * the caption lies inside the analyzed window (no unknown head/tail speech);
  * every raw cue mapped to it has a bound job-stable actor id, is not overlapped/singing, and every speaker-evidence
    unit inside that cue is resolved and consistent with the cue's local label, with no bind problem for it;
  * no speech-evidence unit lying mostly inside the caption is left unrepresented by the mapped cues;
  * cues lie within caption +/- 0.65 s (existing plan_dialogue grouping gap);
  * one caption part per contiguous same-actor run: whole caption for a single run; dash parts only for ordered
    distinct-actor runs. A run may span any pause between its fragments when both fragments' speech is fully
    covered and the gap holds no intervening raw cue / foreign / unresolved / pooled / unrepresented evidence
    (target grouping, not identity relaxation; the gap is never authorized). Ordinal sentence splits are never used.
ASR word uncertainty alone does not block a target (the caption supplies the words); speaker/overlap/reference
uncertainty does. Ineligible captions leave their raw cues on the existing path (MT/original) unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path

VERSION = 'subtitle-primary-target-plan-v1.3-packaged'
TARGET_BASIS = 'subtitle-primary-v1'
GROUP_GAP = .65                      # caption-EDGE alignment tolerance (existing plan_dialogue value); NOT a pause limit
DASH_RE = r'(?:^|\s)-\s+'           # relay/dialogue.py dash split (verbatim)
from . import beta_caption_provenance as CT   # byte-identical reviewed v2 helper (e2c329e2...)
from . import beta_caption_authority as AU    # program-owned host caption authority
_PROPOSAL = Path(CT.__file__)


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def stable_target_id(*, namespace, caption_key, part_index, part_count, actor_id, job_id, revision, source_fingerprint):
    """Deterministic across windows/blocks: authority namespace (embedded source+stream+codec, or external FULL sha)
    + absolute cue identity (start, end, text digest) + part + job-stable actor + job/source revision."""
    body = json.dumps([namespace, caption_key, part_index, part_count, actor_id, job_id, revision, source_fingerprint],
                      separators=(',', ':'))
    return 'tgt-' + hashlib.sha256(body.encode('utf-8')).hexdigest()[:24]


def _verified_problem(authority, context):
    if authority.get('version') != AU.VERSION or authority.get('status') != 'ok':
        return 'caption authority not verified: %s' % authority.get('reason')
    for key in ('job_id', 'revision', 'source_fingerprint'):
        if context.get(key) != authority.get(key) or type(context.get(key)) is not type(authority.get(key)):
            return 'caption authority does not match the window context (%s)' % key
    if not isinstance(authority.get('captions'), list) or not isinstance(authority.get('namespace'), str):
        return 'caption authority incomplete'
    return None


def _authority_problem(authority, context, path_allowed):
    if isinstance(authority, dict) and authority.get('version') == AU.VERSION:
        return _verified_problem(authority, context)
    bad = CT._binding_problem(authority, path_allowed)
    if bad:
        return bad
    for key in ('job_id', 'revision', 'source_fingerprint'):
        if context.get(key) != authority.get(key) or type(context.get(key)) is not type(authority.get(key)):
            return 'caption authority does not match the window context (%s)' % key
    loaded = context.get('subtitle_source')   # the file relay.media.subtitle_cues actually loaded (worker caption_source)
    if not isinstance(loaded, str) or not loaded or loaded.startswith('embedded:'):
        return 'loaded subtitle source missing or embedded (unsupported without a persisted stream-tag receipt)'
    if os.path.normcase(os.path.abspath(loaded)) != os.path.normcase(os.path.abspath(authority['subtitle_path'])):
        return 'caption authority is not the subtitle file the pipeline loaded'
    return None


def _cue_evidence(cue, evidence):
    s, e = float(cue['start']), float(cue['end'])
    return [u for u in evidence if _num(u.get('start')) and _num(u.get('end')) and min(e, u['end']) > max(s, u['start'])]


def _unit_label(unit):
    if unit.get('speaker'):
        return unit['speaker']
    cands = unit.get('candidates') or []
    return cands[0][1] if cands and isinstance(cands[0], (list, tuple)) and len(cands[0]) == 2 else None


EPS = 1e-6                                     # float comparison only; not a tuning threshold
OWN_BASES = (None, 'saved-actor-reference')   # unit decided on its OWN score (relay.speakers.refine_utterances)


def _own_evidence(unit, label):
    """Resolved on the unit's own interval and score, with this label. Pooled/contextual decisions do not count:
    context_supported (nearby anchors) and 'reference-confirmed-utterance-cluster' (cluster-pooled) are excluded."""
    return (unit.get('resolved') is True and unit.get('context_supported') is not True
            and unit.get('basis') in OWN_BASES and _unit_label(unit) == label)


def _merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _speech_intervals(cue):
    """Actual source speech of the cue: its ASR word spans (clipped to the cue). None when unavailable."""
    s, e = float(cue['start']), float(cue['end'])
    spans = []
    for w in cue.get('words') or ():
        if isinstance(w, dict) and _num(w.get('start')) and _num(w.get('end')) and w['end'] > w['start']:
            a, b = max(s, float(w['start'])), min(e, float(w['end']))
            if b > a:
                spans.append([a, b])
    return _merge(spans) or None


def _uncovered(spans, cover):
    gaps = []
    for a, b in spans:
        cursor = a
        for c0, c1 in cover:
            if c1 <= cursor + EPS:
                continue
            if c0 > cursor + EPS:
                gaps.append([cursor, min(c0, b)])
            cursor = max(cursor, c1)
            if cursor >= b - EPS:
                break
        if cursor < b - EPS:
            gaps.append([cursor, b])
    return [g for g in gaps if g[1] > g[0]]          # unrounded decision (EPS is float noise only, < 1 sample at 48 kHz)


def _fmt(spans):
    return [[round(a, 6), round(b, 6)] for a, b in spans]   # diagnostics only, after the decision


def _actor_check(cue, evidence, bind_problem_labels):
    """(problem, authorized_spans). Every source speech interval must lie inside the UNION of this cue's own-label,
    own-score resolved evidence units; any unresolved, other-label or pooled unit touching the cue is competing.
    Authorized spans = covered pieces of the cue that contain speech (never whole-cue extension)."""
    if not isinstance(cue.get('speaker'), str) or not cue['speaker']:
        return 'actor unbound (no job-stable speaker id)', None
    if cue.get('overlapped'):
        return 'overlapped source speech', None
    if cue.get('singing'):
        return 'singing (pitch/tempo path)', None
    local = cue.get('local_speaker')
    if not isinstance(local, str) or not local:
        return 'local speaker label missing', None
    if local in bind_problem_labels:
        return 'speaker bank binding problem for %s' % local, None
    if local.startswith('KNOWN::') and local[len('KNOWN::'):] != cue['speaker']:
        return 'KNOWN label disagrees with bound actor', None
    speech = _speech_intervals(cue)
    if speech is None:
        return 'no word timing: cannot distinguish speech from silence in the cue', None
    units = _cue_evidence(cue, evidence)
    if not units:
        return 'no speaker evidence unit for the cue', None
    for u in units:
        if not _own_evidence(u, local):
            why = ('unresolved (margin %s)' % (round(u['margin'], 3) if _num(u.get('margin')) else 'n/a')
                   if u.get('resolved') is not True else 'pooled/contextual decision' if _unit_label(u) == local
                   else 'label %s' % _unit_label(u))
            return 'competing speaker evidence inside the cue [%s-%s]: %s' % (u['start'], u['end'], why), None
    s, e = float(cue['start']), float(cue['end'])
    cover = _merge([[max(s, float(u['start'])), min(e, float(u['end']))] for u in units])
    gaps = _uncovered(speech, cover)
    if gaps:
        return 'source speech not covered by resolved same-speaker evidence: %s' % _fmt(gaps), None
    authorized = [c for c in cover if any(min(c[1], b) > max(c[0], a) for a, b in speech)]
    return None, authorized


def _binding_provenance(cue, evidence):
    return {'actor_id': cue['speaker'], 'local_speaker': cue['local_speaker'],
            'evidence': [{'start': u['start'], 'end': u['end'], 'resolved': u['resolved'],
                          'basis': u.get('basis', 'window-diarization-cluster'), 'margin': u.get('margin'),
                          'context_supported': u.get('context_supported'),
                          'best': (u.get('candidates') or [[None, None]])[0]} for u in _cue_evidence(cue, evidence)],
            'bind_policy': 'relay.speakers.SpeakerBank.bind (similarity/margin from config); bind score not persisted',
            'identity_scope': 'job-local acoustic actor binding; not human identity truth'}


def _same_actor_gap_problem(runs, src, member_ids, evidence):
    """Target-grouping rule (not identity policy): consecutive raw cues of ONE actor inside one caption may form one
    target across ANY pause, provided each side's speech is fully covered (checked per cue) and the gap holds no
    intervening raw cue, no unresolved/pooled/other-label evidence, and no evidence unit outside both fragments
    (unrepresented speech). An own-score same-label unit that spans the pause is allowed; the gap is never authorized."""
    for run in runs:
        for a, b in zip(run['cues'], run['cues'][1:]):
            g0, g1 = float(a['end']), float(b['start'])
            if g1 <= g0:
                continue
            intervening = [c['id'] for c in src if c['id'] not in member_ids
                           and _overlap(float(c['start']), float(c['end']), g0, g1) > EPS]
            if intervening:
                return 'intervening raw speech between same-actor fragments %s: %s' % (_fmt([[g0, g1]]), intervening)
            for u in evidence:
                if not (_num(u.get('start')) and _num(u.get('end'))) or _overlap(u['start'], u['end'], g0, g1) <= EPS:
                    continue
                if not (_own_evidence(u, a['local_speaker']) or _own_evidence(u, b['local_speaker'])):
                    return 'foreign, unresolved or pooled evidence between same-actor fragments %s' % _fmt([[g0, g1]])
                touches_side = any(_overlap(u['start'], u['end'], float(c['start']), float(c['end'])) > EPS for c in (a, b))
                if not touches_side:
                    return 'unrepresented speech evidence between same-actor fragments %s' % _fmt([[g0, g1]])
    return None


def plan_targets(*, authority, context, captions, all_cues, speaker_evidence, window_issues, window, caption_text,
                 path_allowed=None, load=None):
    """Return an immutable-by-construction plan dict. Inputs are deep-copied; the source view is never modified.

    authority: CT.job_binding(...) built by the pipeline's own source selection (expected, not per cue).
    context:   {'job_id','revision','source_fingerprint','subtitle_source'} of the CURRENT window (from the pipeline;
               subtitle_source = the path relay.media.subtitle_cues loaded, i.e. the worker's caption_source).
    captions:  ignored for authority; the planner reparses the verified subtitle file (load injectable for tests).
    window:    {'origin','end','render_start','render_end'} (analyzed range and owned render range).
    """
    src = copy.deepcopy(list(all_cues))
    evidence = copy.deepcopy(list(speaker_evidence or ()))
    plan = {'version': VERSION, 'planner_source_sha256': CT.file_sha256(__file__),
            'v2_helper_sha256': CT.file_sha256(_PROPOSAL), 'targets': [], 'captions': [],
            'existing_path_cue_ids': [], 'superseded_cue_ids': [], 'carry_target_ids': [], 'status': 'ok'}
    bad = _authority_problem(authority, context, path_allowed)
    if bad:
        plan.update(status='authority_refused', reason=bad, existing_path_cue_ids=[c['id'] for c in src])
        return plan
    if authority.get('version') == AU.VERSION:
        plan['authority'] = {k: authority.get(k) for k in ('version', 'kind', 'job_id', 'revision', 'source_fingerprint',
                                                           'selection', 'codec', 'stream_index', 'subtitle_path',
                                                           'subtitle_sha256', 'language', 'language_basis', 'namespace',
                                                           'blocks', 'completeness', 'window')}
        file_caps = [dict(c) for c in authority['captions']]
        namespace = authority['namespace']
    else:
        plan['authority'] = {k: authority[k] for k in ('job_id', 'revision', 'source_fingerprint', 'subtitle_path',
                                                        'subtitle_sha256', 'language', 'language_basis')}
        try:
            file_caps = (load or CT.load_sidecar)(authority['subtitle_path'])
        except Exception as exc:
            plan.update(status='authority_refused', reason='subtitle parse failed: %s' % type(exc).__name__,
                        existing_path_cue_ids=[c['id'] for c in src])
            return plan
        namespace = AU.digest({'kind': 'external', 'subtitle_sha256': authority['subtitle_sha256'], 'version': AU.VERSION})
    plan['authority']['namespace'] = namespace
    try:
        origin, end = float(window['origin']), float(window['end'])
        rs, re_ = float(window['render_start']), float(window['render_end'])
        if not all(map(_num, (origin, end, rs, re_))) or not (origin <= rs < re_ <= end):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        plan.update(status='window_refused', reason='window bounds malformed', existing_path_cue_ids=[c['id'] for c in src])
        return plan
    bind_problem_labels = {i.get('local_speaker') for i in window_issues or () if isinstance(i, dict) and i.get('local_speaker')}
    ids = [c.get('id') for c in src]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        plan.update(status='source_refused', reason='missing or duplicate source cue ids', existing_path_cue_ids=[])
        return plan

    # map each raw cue to its FILE caption via the pipeline's own caption dict (exact time/text), never by text search alone
    def file_index_of(cap):
        hits = [i for i, f in enumerate(file_caps) if abs(f['start'] - cap.get('start', -1)) <= .001
                and abs(f['end'] - cap.get('end', -1)) <= .001 and f['text'] == cap.get('text')]
        return hits[0] if len(hits) == 1 else None

    groups = {}
    for cue in src:
        cap = cue.get('caption')
        if isinstance(cap, dict):
            fi = file_index_of(cap)
            if fi is not None:
                groups.setdefault(fi, []).append(cue)
    consumed, seen_ids = set(), set()
    for fi in sorted(groups):
        cap = file_caps[fi]
        members = sorted(groups[fi], key=lambda c: (float(c['start']), c['id']))
        row = {'file_index': fi, 'caption_start': cap['start'], 'caption_end': cap['end'],
               'caption_text_sha256': CT.sha256_text(cap['text']), 'source_cue_ids': [c['id'] for c in members],
               'eligible': False, 'reason': None, 'targets': []}
        plan['captions'].append(row)

        def refuse(reason):
            row['reason'] = reason
        if len({c.get('caption_index') for c in members}) != 1:
            refuse('mapped cues disagree on caption index'); continue
        if cap['start'] < origin - 1e-9 or cap['end'] > end + 1e-9:
            refuse('caption extends outside the analyzed window (unknown %s speech)' %
                   ('head' if cap['start'] < origin - 1e-9 else 'tail')); continue
        authorized, problem = {}, None
        for c in members:
            p, spans = _actor_check(c, evidence, bind_problem_labels)
            if p:
                problem = (c['id'], p); break
            authorized[c['id']] = spans
        if problem:
            refuse('cue %s: %s' % problem); continue
        outside = [c['id'] for c in members if float(c['start']) < cap['start'] - GROUP_GAP or float(c['end']) > cap['end'] + GROUP_GAP]
        if outside:
            refuse('source cue outside the caption envelope: %s' % outside); continue
        member_ids = {c['id'] for c in members}
        foreign = [c['id'] for c in src if c['id'] not in member_ids
                   and _overlap(float(c['start']), float(c['end']), cap['start'], cap['end']) > EPS
                   and not (isinstance(c.get('caption'), dict) and file_index_of(c['caption']) not in (None, fi))]
        if foreign:
            refuse('unmapped source speech touches the caption: %s' % foreign); continue
        raw_cover = _merge([[float(c['start']), float(c['end'])] for c in src if float(c['end']) > float(c['start'])])
        touching = [u for u in evidence if _num(u.get('start')) and _num(u.get('end')) and u['end'] > u['start']
                    and _overlap(u['start'], u['end'], cap['start'], cap['end']) > EPS]
        bad_unit = None
        for u in touching:
            if u.get('resolved') is not True or u.get('context_supported') is True or u.get('basis') not in OWN_BASES:
                bad_unit = 'unresolved or pooled speaker evidence touches the caption [%s-%s]' % (u['start'], u['end']); break
            if any(_overlap(u['start'], u['end'], float(c['start']), float(c['end'])) > EPS for c in members):
                continue    # already validated inside its member cue: own-score, same label (a pause it spans is not speech)
            inside = [[max(u['start'], cap['start']), min(u['end'], cap['end'])]]
            if _uncovered(inside, raw_cover):
                bad_unit = 'speech evidence in the caption without any source cue [%s-%s]' % (u['start'], u['end']); break
            if not any(
                    c['id'] not in member_ids and isinstance(c.get('caption'), dict) and file_index_of(c['caption']) not in (None, fi)
                    and _overlap(u['start'], u['end'], float(c['start']), float(c['end'])) > EPS for c in src):
                bad_unit = 'speech evidence in the caption not represented by mapped cues [%s-%s]' % (u['start'], u['end']); break
        if bad_unit:
            refuse(bad_unit); continue
        runs = []
        for c in members:
            if runs and runs[-1]['actor'] == c['speaker']:
                runs[-1]['cues'].append(c)
            else:
                runs.append({'actor': c['speaker'], 'cues': [c]})
        gap_problem = _same_actor_gap_problem(runs, src, member_ids, evidence)
        if gap_problem:
            refuse(gap_problem); continue
        parts = [caption_text(x) for x in re.split(DASH_RE, cap['text']) if caption_text(x)]
        if len(parts) == 1 and len(runs) == 1:
            assignment = [(runs[0], caption_text(cap['text']), 'whole')]
        elif len(parts) > 1 and len(parts) == len(runs) and all(a['actor'] != b['actor'] for a, b in zip(runs, runs[1:])):
            assignment = [(r, p, 'dash') for r, p in zip(runs, parts)]   # same-actor runs already passed the gap rule
        elif len(parts) == 1:
            refuse('single-part caption spoken by %d distinct actors (no guessed split)' % len(runs)); continue
        else:
            refuse('%d dash parts vs %d actor runs (ambiguous mapping)' % (len(parts), len(runs))); continue
        targets = []
        for k, (run, english, split) in enumerate(assignment):
            tid = stable_target_id(namespace=namespace, caption_key=AU.caption_key({'namespace': namespace}, cap), part_index=k,
                                   part_count=len(assignment), actor_id=run['actor'], job_id=authority['job_id'],
                                   revision=authority['revision'], source_fingerprint=authority['source_fingerprint'])
            if tid in seen_ids:
                refuse('duplicate target id'); targets = []; break
            seen_ids.add(tid)
            cues = run['cues']
            first = float(cues[0]['start'])
            targets.append({
                'target_id': tid, 'basis': TARGET_BASIS, 'english': english,
                'caption': {'file_index': fi, 'caption_key': AU.caption_key({'namespace': namespace}, cap),
                            'start': cap['start'], 'end': cap['end'],
                            'text_sha256': CT.sha256_text(cap['text']), 'part_index': k, 'part_count': len(assignment), 'split': split},
                'actor': _binding_provenance(cues[0], evidence) | {'cue_bindings': [_binding_provenance(c, evidence) for c in cues]},
                'source_cue_ids': [c['id'] for c in cues],
                'authorized_audio_spans': [list(s) for c in cues for s in authorized[c['id']]],
                'source_speech': {'word_count': sum(len(c.get('words') or ()) for c in cues),
                                  'speech_seconds': round(sum(b - a for c in cues for a, b in _speech_intervals(c)), 6),
                                  'authorized_seconds': round(sum(b - a for c in cues for a, b in authorized[c['id']]), 6),
                                  'speech_basis': 'ASR word spans; identity coverage = union of own-score resolved units'},
                'speech_span': [min(float(c.get('speech_start', c['start'])) for c in cues),
                                max(float(c.get('speech_end', c['end'])) for c in cues)],
                'safe_caption_span': [cap['start'], cap['end']],
                'source_word_uncertain': any(bool(c.get('unclear')) for c in cues),
                'owned_by_this_window': rs <= first < re_})
        if not targets:
            continue
        row.update(eligible=True, targets=[t['target_id'] for t in targets])
        for t in targets:
            plan['targets'].append(t)
            if t['owned_by_this_window']:
                plan['superseded_cue_ids'].extend(t['source_cue_ids'])
            else:
                plan['carry_target_ids'].append(t['target_id'])
            consumed.update(t['source_cue_ids'])
    plan['existing_path_cue_ids'] = [c['id'] for c in src if c['id'] not in consumed]
    return plan


def reference_source_cues(all_cues, actor_id):
    """Independent target->reference adapter: the RAW source cues usable for this actor's clean reference.

    Russian ASR text only, never target English; excludes unclear and overlapped cues even when a caption target
    covered them. Returns deep copies; the caller's source list is untouched.
    """
    return [copy.deepcopy(c) for c in all_cues if c.get('speaker') == actor_id and not c.get('unclear')
            and not c.get('overlapped') and isinstance(c.get('text'), str)]
