"""Pure dialogue grouping and timing policy; no model calls or media mutation.

Caption wording is admitted only for a complete, single-speaker unit supported by
the existing alignment. Speaker labels and source words remain explicit so the
admission gate can verify the decision independently.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from .dialogue import caption_text

POLICY_VERSION = 'dialogue-unit-v1'
VOICE_POLICY_VERSION = 'bank-first-line-fallback-v1'
SLACK_VERSION = 'silence-slack-v1'
BANK_REFERENCE_MIN_SECONDS = 3.0


def _span(item):
    a, b = float(item['start']), float(item['end'])
    if not math.isfinite(a) or not math.isfinite(b) or b <= a:
        raise ValueError('dialogue unit has invalid source geometry')
    return a, b


def _intersect(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def _union(spans):
    result = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def annotate_overlaps(cues, turns, *, span_seconds=.12, span_fraction=0.0):
    """Preserve recognition flags and add the existing raw-turn overlap rule."""
    result, issues = copy.deepcopy(list(cues)), []
    shared = []
    for index, left in enumerate(turns):
        a, b = _span(left)
        for right in turns[index + 1:]:
            c, d = _span(right)
            if left.get('speaker') != right.get('speaker') and _intersect(a, b, c, d) > 0:
                shared.append((max(a, c), min(b, d)))
    for cue in result:
        a, b = _span(cue)
        spans = [(max(a, c), min(b, d)) for c, d in shared if _intersect(a, b, c, d) > 0]
        cue['overlap_intervals'] = _union(spans)
        cue['overlap_seconds'] = sum(y - x for x, y in cue['overlap_intervals'])
        cue['cue_seconds'] = b - a
        if any(y - x > max(span_seconds, span_fraction * (b - a)) for x, y in spans):
            cue['overlapped'] = True
            cue['turn_overlap'] = True
            issues.append({'kind': 'overlapping_dialogue', 'cue': cue['id']})
    return result, issues


def _caption_key(cue):
    value = cue.get('caption')
    if not isinstance(value, dict):
        return None
    a, b = _span(value)
    return (a, b, str(value.get('text', '')), str(value.get('language', '')).lower())


def _raw_label(cue, turns):
    scores = {}
    for word in cue.get('words') or []:
        a, b = _span(word)
        for turn in turns:
            c, d = _span(turn)
            amount = _intersect(a, b, c, d)
            label = turn.get('speaker')
            if amount and isinstance(label, str) and label:
                scores[label] = scores.get(label, 0.0) + amount
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if not ranked or (len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= 1e-9):
        return None
    return ranked[0][0]


_SENTENCE_FINAL = re.compile(r'[.!?…][\"”’\x27]*$')
_SENTENCE_FINAL_INSIDE = re.compile(r'[.!?…][\"”’\x27]*\s+\S')


def _sentence_final(text):
    """True when the text ends a sentence (terminal punctuation, optionally followed by closing quotes)."""
    return bool(_SENTENCE_FINAL.search(str(text).strip()))


def _completes_sentence(text):
    """True when this fragment is the tail of ONE sentence: it ends with terminal punctuation, carries none before its
    last word, does not open a new question/exclamation (inverted mark) and does not start with a capital letter."""
    value = str(text).strip()
    if not value or not _sentence_final(value) or _SENTENCE_FINAL_INSIDE.search(value):
        return False
    if value[0] in '¿¡':
        return False
    first = next((ch for ch in value if ch.isalpha()), '')
    return not first.isupper()


_DIARIZER_LABEL = re.compile(r'SPEAKER_\d+|DIART_D\d+')       # DIART_Dnnn: film-global label from relay/diart_turns.py


def _same_raw_speaker(left, right, turns):
    """The raw diarization labels of both cues are equal and not None (refined labels are ignored)."""
    label = _raw_label(left, turns)
    return label is not None and label == _raw_label(right, turns)


def _bare_diarizer_label(cue, turns):
    """The cue's refined label is just its own raw diarizer label (SPEAKER_nn): the refinement found no embedding
    identity (KNOWN::x, UTTERANCE_nn) for this fragment, so it contradicts nothing about the neighbouring cue."""
    label = cue.get('speaker')
    return isinstance(label, str) and bool(_DIARIZER_LABEL.fullmatch(label)) and label == _raw_label(cue, turns)


def _dominant_refined_label(members):
    """Label of a bridged unit: among the labels that carry embedding evidence (not a bare SPEAKER_nn) the one with the
    most member speech, else the bare label with the most speech; earliest wins a tie. None when no member has one."""
    weight = {}
    for cue in members:
        label = cue.get('speaker')
        if label is not None:
            a, b = _span(cue)
            weight[label] = weight.get(label, 0.0) + (b - a)
    pool = {k: v for k, v in weight.items() if not _DIARIZER_LABEL.fullmatch(str(k))} or weight
    best = None
    for label, seconds in pool.items():
        if best is None or seconds > pool[best]:
            best = label
    return best


def _safe(members, turns, merged_labels, origin, end, render_start, render_end, caption=None, raw_only=False):
    labels = {c.get('speaker') for c in members}
    # raw_only (stub joins): the diarizer's raw turn labels alone must show ONE speaker across the whole span; the
    # refined/merged-label evidence is skipped because a sub-second fragment has no usable embedding of its own.
    if not raw_only and (len(labels) != 1 or None in labels or next(iter(labels)) in merged_labels):
        return None
    if any(c.get('overlapped') or c.get('turn_overlap') or c.get('singing') for c in members):
        return None
    a, b = min(_span(c)[0] for c in members), max(_span(c)[1] for c in members)
    if not origin <= a < b <= end - .5 or a < render_start:
        return None
    if any(float(c['start']) >= render_end for c in members):
        return None
    if caption is not None and not origin <= _span(caption)[0] < _span(caption)[1] <= end - .5:
        return None
    raw = {_raw_label(c, turns) for c in members}
    if None in raw or len(raw) != 1:
        return None
    raw_label = next(iter(raw))
    # Refined labels may be KNOWN:: identities; compare raw labels with raw labels.
    if any(t.get('speaker') != raw_label and _intersect(a, b, *_span(t)) > .12 for t in turns):
        return None
    return raw_label


def _unit(members, basis, raw_label, *, caption=None, text=None, dash_part=None, evidence=None, speaker=None):
    ids = [c['id'] for c in members]
    ident = hashlib.sha256(json.dumps([POLICY_VERSION, ids], separators=(',', ':')).encode()).hexdigest()[:24]
    words = [{**copy.deepcopy(w), 'source_cue_id': c['id']} for c in members for w in c.get('words', [])]
    words.sort(key=lambda w: (float(w['start']), float(w['end'])))
    a, b = min(_span(c)[0] for c in members), max(_span(c)[1] for c in members)
    gaps = [max(0.0, float(right['start']) - float(left['end'])) for left, right in zip(members, members[1:])]
    result = copy.deepcopy(members[0])
    for key in ('caption_fallback', 'english', 'translation', 'line_reference', 'voice_source'):
        result.pop(key, None)
    if speaker is not None:
        result['speaker'] = speaker
    elif result.get('speaker') is None:
        result['speaker'] = next((c['speaker'] for c in members if c.get('speaker') is not None), None)
    result.update(id=ident, start=a, end=b, speech_start=a, speech_end=b, words=words,
                  text=' '.join(c['text'].strip() for c in members).strip(),
                  source_cue_ids=ids, source_intervals=[list(_span(c)) for c in members],
                  unclear=any(c.get('unclear') for c in members), overlapped=False,
                  turn_overlap=False, overlap_intervals=[], overlap_seconds=0.0, cue_seconds=b - a)
    if words:
        result['probability'] = sum(float(w.get('probability', 0.0)) for w in words) / len(words)
    cap = None
    if caption is not None:
        result['caption'] = copy.deepcopy(caption)
        result['caption_fallback'] = text
        cap = {'start': float(caption['start']), 'end': float(caption['end']),
               'text_sha256': hashlib.sha256(str(caption['text']).encode('utf-8')).hexdigest()}
    result['dialogue_unit'] = {
        'version': POLICY_VERSION, 'basis': basis, 'caption': cap, 'dash_part': dash_part,
        'local_speaker': result.get('speaker'), 'member_cue_ids': ids,
        'max_internal_gap_seconds': max(gaps, default=0.0),
        'evidence': {'member_spans': [list(_span(c)) for c in members], 'raw_speaker': raw_label,
                     **(evidence or {})}}
    return result


def group_units(cues, captions, turns, *, origin, end, render_start, render_end, issues=(), merged_labels=(),
                caption_edge_pause=.25, fragment_seconds=0.0, fragment_gap=1.0, fragment_unit_seconds=6.0):
    """Return replacement cues and remapped issues before ownership is assigned.

All source cues for a caption must be visible and consecutive. Dash assignments
also require pre-existing per-cue wording evidence and raw speaker transitions.
Unresolved dash parts remain original through an existing cue fallback reason.

fragment_seconds > 0 (owner decision 2026-10-19): a transcript fragment shorter than this is a stub ("Доча!", "Вот!") that
cannot be cloned or read back reliably on its own. It joins its same-speaker neighbour across a pause of up to
fragment_gap seconds, even after sentence punctuation, while the unit stays within fragment_unit_seconds. Every other
condition (one raw speaker, no unclear member, no caption, render window) is unchanged; 0 keeps the earlier rule.

Fragment repairs (hotfix 2026-10-20; stubs only, so they need fragment_seconds > 0). Both apply only while the earlier cue
is NOT sentence-final, i.e. when one sentence is demonstrably still open, and never across a raw speaker change:
  * refined-label flip: the refined labels of the two cues differ (e.g. KNOWN::x -> SPEAKER_00 for a 0.8 s tail, where the
    short fragment got no embedding identity and fell back to its bare raw label) but the RAW diarization labels of both
    cues are equal and not None, and at least one side is only its own bare SPEAKER_nn label. That is no speaker change;
    the whole span must still show one raw speaker (_safe raw_only). Two cues that BOTH carry an identity (KNOWN::a vs
    KNOWN::b, UTTERANCE_01 vs UTTERANCE_07, ...) keep the conflict. The unit takes the label that has embedding evidence
    and records every bridged label in evidence['refined_labels_bridged'].
  * completing tail: the later cue is the tail of ONE sentence (ends with terminal punctuation, none before its last
    word, no leading inverted mark or capital letter). It then joins across a silent hole up to fragment_unit_seconds
    (a Whisper word-time hole inside one segment: 'Como' ... 'la encuentra?') and the unit may reach 12 s, the same
    ceiling as an ordinary multi-cue unit. Cue data carries no Whisper segment id, so this text shape plus one raw
    speaker over the whole span stands in for 'same segment'. Joins across sentence-final punctuation keep the 6 s cap.
"""
    raw = sorted(copy.deepcopy(list(cues)), key=lambda c: (float(c['start']), float(c['end']), c['id']))
    by_caption = {}
    for index, cue in enumerate(raw):
        key = _caption_key(cue)
        if key is not None:
            by_caption.setdefault(key, []).append(index)
    replacements, consumed, mapping, extra = {}, set(), {}, []
    caption_keys = {_caption_key({'caption': c}) for c in captions}

    def commit(indices, unit):
        replacements[indices[0]] = unit
        consumed.update(indices[1:])
        for index in indices:
            mapping[raw[index]['id']] = unit['id']

    for key, indices in by_caption.items():
        if key not in caption_keys or key[3] not in ('en', 'eng'):
            continue
        if indices != list(range(indices[0], indices[-1] + 1)):
            if re.search(r'(?:^|\s)-\s+', key[2]):
                for index in indices:
                    extra.append({'kind': 'unclear_dialogue_without_aligned_caption', 'cue': raw[index]['id'],
                                  'reason': 'dash_caption_unresolved'})
            continue
        members = [raw[i] for i in indices]
        caption = members[0]['caption']
        text = caption_text(caption['text'])
        if not text:
            continue
        is_dash = bool(re.search(r'(?:^|\s)-\s+', str(caption['text'])))
        if not is_dash:
            label = _safe(members, turns, merged_labels, origin, end, render_start, render_end, caption)
            if label is not None:
                commit(indices, _unit(members, 'aligned-caption', label, caption=caption, text=text))
            continue
        parts = [caption_text(p) for p in re.split(r'(?:^|\s)-\s+', str(caption['text'])) if caption_text(p)]
        runs = []
        for i in indices:
            if runs and raw[runs[-1][-1]].get('speaker') == raw[i].get('speaker'):
                runs[-1].append(i)
            else:
                runs.append([i])
        planned = []
        if len(parts) == len(runs) and len(parts) > 1:
            for part_index, (part, run) in enumerate(zip(parts, runs)):
                chunk = [raw[i] for i in run]
                label = _safe(chunk, turns, merged_labels, origin, end, render_start, render_end, caption)
                known = [caption_text(c.get('caption_fallback', '')) for c in chunk if c.get('caption_fallback')]
                valid = label is not None and bool(known) and all(value == part for value in known)
                pause = None
                if part_index:
                    left = raw[runs[part_index - 1][-1]]
                    pause = float(chunk[0]['start']) - float(left['end'])
                    left_label = _raw_label(left, turns)
                    valid = valid and pause >= caption_edge_pause and left_label is not None and left_label != label
                    # Both raw turn endpoints must support the observed change in the gap.
                    valid = valid and any(t.get('speaker') == left_label and
                        float(left['end']) - .12 <= float(t['end']) <= float(chunk[0]['start']) + .12 for t in turns)
                    valid = valid and any(t.get('speaker') == label and
                        float(left['end']) - .12 <= float(t['start']) <= float(chunk[0]['start']) + .12 for t in turns)
                if not valid:
                    planned = []
                    break
                unit = _unit(chunk, 'caption-dash-part', label, caption=caption, text=part, dash_part=part_index,
                             evidence={'dash_boundary_pause_seconds': pause, 'dash_part_count': len(parts),
                                       'caption_part_verified': True})
                planned.append((run, unit))
        if planned and len(planned) == len(parts):
            for run, unit in planned:
                commit(run, unit)
        else:
            for member in members:
                extra.append({'kind': 'unclear_dialogue_without_aligned_caption', 'cue': member['id'],
                              'reason': 'dash_caption_unresolved'})

    # A caption relation that is unresolved must not be replaced by a partial
    # transcript unit that happens to reuse the full caption wording.
    index = 0
    while index < len(raw):
        if index in consumed or index in replacements or _caption_key(raw[index]) is not None:
            index += 1
            continue
        indices = [index]
        joined_stub = bridged = False
        while indices[-1] + 1 < len(raw):
            previous, following = raw[indices[-1]], raw[indices[-1] + 1]
            stub = fragment_seconds > 0 and min(_span(previous)[1] - _span(previous)[0],
                                                _span(following)[1] - _span(following)[0]) < fragment_seconds
            unresolved = following.get('speaker') is None or previous.get('speaker') is None
            open_head = stub and not _sentence_final(previous['text'])
            # A refined-label flip inside a sentence that is still open is no speaker change when the RAW diarization
            # labels of both cues are equal; _safe(raw_only) below still checks one raw speaker over the whole span.
            bridge = (open_head and not unresolved and following.get('speaker') != previous.get('speaker')
                      and _same_raw_speaker(previous, following, turns)
                      and (_bare_diarizer_label(previous, turns) or _bare_diarizer_label(following, turns)))
            conflict = following.get('speaker') != previous.get('speaker') and not (stub and unresolved) and not bridge
            # The later cue completes the open sentence: it may follow a longer silent hole and the unit may use the
            # ordinary 12 s ceiling instead of the stub ceiling.
            completes = open_head and _completes_sentence(following['text'])
            gap_limit = (max(fragment_gap, fragment_unit_seconds) if completes else fragment_gap) if stub else .6
            span_limit = (12 if completes else fragment_unit_seconds) if stub else 12
            if (_caption_key(following) is not None or conflict
                    or following.get('unclear') or previous.get('unclear')
                    or (not stub and re.search(r'[.!?…][\"\u201d\u2019\x27]*$', previous['text'].strip()))
                    or float(following['start']) - float(previous['end']) > gap_limit
                    or float(following['end']) - float(raw[index]['start']) > span_limit):
                break
            candidate = [raw[i] for i in indices + [indices[-1] + 1]]
            if _safe(candidate, turns, merged_labels, origin, end, render_start, render_end, raw_only=stub) is None:
                break
            indices.append(indices[-1] + 1)
            joined_stub = joined_stub or stub
            bridged = bridged or bridge
        members = [raw[i] for i in indices]
        label = _safe(members, turns, merged_labels, origin, end, render_start, render_end, raw_only=joined_stub)
        if (label is not None and not any(c.get('unclear') for c in members)
                and float(members[-1]['end']) - float(members[0]['start']) <= 12):
            if bridged:
                refined = sorted({c['speaker'] for c in members if c.get('speaker') is not None})
                commit(indices, _unit(members, 'transcript-utterance', label, speaker=_dominant_refined_label(members),
                                      evidence={'refined_labels_bridged': refined}))
            else:
                commit(indices, _unit(members, 'transcript-utterance', label))
        index = indices[-1] + 1
    units = [replacements.get(i, cue) for i, cue in enumerate(raw) if i not in consumed]
    remapped = []
    for original in list(issues) + extra:
        issue = copy.deepcopy(original)
        if issue.get('cue') in mapping:
            old = issue['cue']
            issue.update(cue=mapping[old], member_cue_id=old)
            # The complete supported caption now supplies the wording. No other
            # uncertainty, overlap or identity reason is silently removed.
            unit = next(c for c in units if c['id'] == issue['cue'])
            if (issue.get('kind') == 'unclear_dialogue_without_aligned_caption'
                    and unit.get('caption_fallback') and unit.get('dialogue_unit', {}).get('basis')
                    in ('aligned-caption', 'caption-dash-part')):
                continue
        if issue not in remapped:
            remapped.append(issue)
    return units, remapped


def voice_mode(label, bind_evidence, merged_labels, *, cue_seconds=None, prefer_own_seconds=None):
    """Use only an actual accepted bank decision, never a guessed local ID.

    Owner decision 2026-10-19: a line long enough to be its own reference (cue_seconds >= prefer_own_seconds) is cloned
    from ITS OWN audio, so a quiet line is dubbed quiet and a shouted line is dubbed from shouting. Shorter lines keep
    the frozen per-speaker bank reference. Default (prefer_own_seconds None) is the unchanged bank-first rule."""
    if prefer_own_seconds is not None and (prefer_own_seconds <= 0 if cue_seconds is None else cue_seconds >= prefer_own_seconds):   # 0.0 = owner rule 2026-10-20: every line is its own clone
        return 'line-own'
    decision = (bind_evidence or {}).get(label) or {}
    if (label not in merged_labels and decision.get('decision')
            and decision.get('reason') in ('bound_existing', 'new_identity', 'known_ok')):
        return 'bank'
    return 'line-own'


def silence_slack(cue, *, raw_turns=(), transcript=(), all_cues=(), protected=(), placements=(),
                  origin=0.0, analysis_end, window_limit=None):
    """Bound a proposed tail by all detected speech, existing takes and horizon.

Transcript intervals are relative to ``origin``; other intervals are absolute.
The caller still enforces scheduler, ownership and committed-window limits.
"""
    start, natural_end = _span(cue)
    limit = float(analysis_end)
    if window_limit is not None:
        limit = min(limit, float(window_limit))
    if not math.isfinite(limit):
        raise ValueError('nonfinite dialogue slack limit')
    next_onset, source = None, None
    def consider(a, b, kind):
        nonlocal next_onset, source
        if not math.isfinite(a) or not math.isfinite(b) or b <= a:
            raise ValueError('invalid dialogue slack obstacle')
        if b <= natural_end + 1e-9:
            return
        candidate = max(natural_end, a)
        if next_onset is None or candidate < next_onset:
            next_onset, source = candidate, kind
    raw_label = ((cue.get('dialogue_unit') or {}).get('evidence') or {}).get('raw_speaker')
    raw_label = raw_label or _raw_label(cue, raw_turns)
    for turn in raw_turns:
        a, b = _span(turn)
        if b <= natural_end + 1e-9:
            continue
        if raw_label is not None and turn.get('speaker') == raw_label and a < natural_end:
            continue  # the current speaker's already-active tail is not a new onset
        consider(a, b, 'raw_speech')
    for segment in transcript:
        a, b = _span(segment)
        a, b = a + float(origin), b + float(origin)
        if a >= natural_end:
            consider(a, b, 'transcript')
    for other in all_cues:
        if other.get('id') != cue.get('id'):
            consider(*_span(other), 'other_cue')
    for kind, rows in (('protected_original', protected), ('published_placement', placements)):
        for item in rows:
            if isinstance(item, dict):
                if kind == 'published_placement' and item.get('cue_id') == cue.get('id'):
                    continue
                a, b = _span(item)
            else:
                a, b = map(float, item)
            consider(a, b, kind)
    if next_onset is not None:
        limit = min(limit, max(natural_end, next_onset - .15))
    return {'version': SLACK_VERSION, 'natural_end': natural_end,
            'allowed_end': limit, 'extension_available': limit > natural_end,
            'natural_end_within_bound': natural_end <= limit, 'next_onset': next_onset,
            'onset_source': source, 'guard': .15, 'cap': None}
