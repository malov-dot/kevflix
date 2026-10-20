"""Deterministic dialogue ownership across render windows (beta5-asr-gain-fix-v4; actor_beta.caption_targets only).

Stdlib only. Inert unless ActorPipeline calls it with caption_targets on; every legacy path is unchanged.

ONE ownership rule, by speech start (the rule of B's packaged caption gate, beta_caption_planner
owned_by_this_window = render_start <= first source cue start < render_end):
  * A cue (raw ASR line, caption-bound or not) is owned by the window whose render range [render_start, render_end)
    holds the cue's START. Caption times play no part in cue ownership, so a caption that starts after a boundary
    never moves its line to the later window, and a caption that starts before a boundary never strands its line.
  * A caption-target unit is one whole utterance, owned where its FIRST source cue starts (exactly B's gate and the
    planner's owned_by_this_window). It is not refused because its caption or later source cues cross a render
    boundary (utterance-start-v1, beta_caption_flow): the owner renders the whole bounded utterance once (inside its
    analysed window) and the later window carries the committed canonical take. A later source cue that starts in
    the next window is selected there by its own start but consumed by that carry (no MT, no second synthesis), so the
    gate, the flow roles and the ledger agree; where no committed take covers it, original audio plays.
    The owner never selects such a later cue, so plan() annotates every cue with 'turn_overlap' (the simultaneous-
    voices rule ActorPipeline applies only to selected cues): the flow keeps original for a whole utterance whose
    later source overlaps another voice, exactly as the later window would.
  * A caption with no transcribed cue (speech_without_transcript) is owned where its speech starts: the earliest
    speech region inside the caption (the regions plan_dialogue matched it with, overlap > .1 s). Only that window
    keeps the caption issue.
  Because an owned cue starts inside the render range, an owner always reaches its own line. The owner never places a
  take before its own render start (placement floor, ActorPipeline) and never publishes a take that a committed
  neighbour window rendered without (beta_line_ledger).

Every other window overlapping a non-owned cue or caption accounts for that speech span (per span, at prepare and
again at render against the canonical takes that exist then):
  * owner_published_dub_covers       canonical DialogueLedger takes OF THIS DIALOGUE (cue id, caption key or the
                                     same-line source overlap rule; never an unrelated take) remove the whole span.
                                     The existing carry path mixes them; nothing is protected.
  * protected_original_fallback      otherwise (owner not run, owner fell back, take covers only part): the uncovered
                                     speech plays exact original audio through the runtime protection walls.
  * not_protected_under_carried_take uncovered speech that lies under a carried canonical take (its placement or
                                     removal envelope). It cannot be protected without cutting that take; Russian and
                                     the carried take overlap there. Reported, never labelled as protected or covered.
  * speech_inside_owned_cue / no_speech_in_render_range: accounted, nothing to protect.

Dedupe does not rest on keys. It rests on (1) the single cue-start owner, (2) canonical_conflicts at render: an owned
cue whose source overlaps a canonical take of another window (>= half the shorter span, v4) is not synthesized
again, and (3) the atomic line-ledger commit (beta_line_ledger), which re-checks the same overlap rule and the
canonical snapshot inside one SQLite IMMEDIATE transaction. A line id (source, owner window, span, caption key) only
makes a window's re-render idempotent; ASR jitter gives different ids in different windows by design.
"""
from __future__ import annotations

import hashlib
import json

VERSION = 'beta-dialogue-ownership-v4-cue-start'   # also digested into caption keys / line ids: unchanged on purpose
# Whole-utterance rule (caption-target units owned by their first source cue start; cue rows unchanged).
UTTERANCE_RULE = 'utterance-start-v1'
# ActorPipeline's simultaneous-voices rule: two diarization turns of different speakers share time that overlaps the
# cue by more than this (relay.actor_pipeline, selected cues -> 'overlapping_dialogue').
TURN_OVERLAP_SECONDS = .12
# Coverage tolerance (seconds) when deciding that published takes cover a span.
COVER_TOLERANCE = 1e-3
# protect_dialogue_mix fades over 10 ms outside a protected span; protected fallback keeps twice that away from a
# carried canonical take so its generated PCM never enters a protected span or its transition.
TAKE_GUARD_SECONDS = .02
# Owned synthesis cues and their own removal envelope stay out of caption-speech fallback pieces:
# transition .01 + schedule gap .04 + margin.
OWNED_GUARD_SECONDS = .06
# Two source spans are the same line (a second claim) when they share at least half of the SHORTER span (ASR jitter
# of one line across a boundary; a fragment contained in a merged cue). v4 (W2 P2-6):
# the former absolute rule (any 0.25 s shared) merged two DIFFERENT adjacent lines whose ASR spans overlap a little
# across a boundary, so the second was never dubbed; it is removed. Every cross-window comparison uses this one rule.
CANONICAL_OVERLAP_FRACTION = .5
# plan_dialogue matches a caption to speech regions by more than this overlap (speech_without_transcript).
CAPTION_SPEECH_OVERLAP_SECONDS = .1
EPS = 1e-6


def _r(value, places=3):
    return round(float(value), places)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def merge(spans):
    out = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if float(b) > float(a)):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def clip(spans, low, high):
    return merge([max(float(a), low), min(float(b), high)] for a, b in spans)


def subtract(spans, holes):
    out = []
    holes = merge(holes)
    for a, b in merge(spans):
        cursor = a
        for c, d in holes:
            if d <= cursor or c >= b:
                continue
            if c > cursor:
                out.append([cursor, c])
            cursor = max(cursor, d)
            if cursor >= b:
                break
        if cursor < b:
            out.append([cursor, b])
    return [s for s in out if s[1] - s[0] > 1e-9]


def intersect(spans, others):
    return subtract(spans, subtract(spans, others))


def total(spans):
    return sum(b - a for a, b in merge(spans))


def owns(time, render_start, render_end):
    return float(render_start) <= float(time) < float(render_end)


def significant_overlap(a, b):
    """Same line: shared source time >= half of the shorter span (see CANONICAL_OVERLAP_FRACTION)."""
    shared = min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0]))
    shorter = min(float(a[1]) - float(a[0]), float(b[1]) - float(b[0]))
    return shared > 0 and shorter > 0 and shared >= CANONICAL_OVERLAP_FRACTION * shorter


def turn_overlap(cue, speech_regions):
    """True when two diarization turns of different speakers share time overlapping the cue by more than
    TURN_OVERLAP_SECONDS (the same rule ActorPipeline applies to its selected cues)."""
    s0, e0 = float(cue['start']), float(cue['end'])
    turns = [t for t in speech_regions or () if t.get('speaker') is not None]
    for i, left in enumerate(turns):
        for right in turns[i + 1:]:
            if left['speaker'] == right['speaker']:
                continue
            a, b = max(float(left['start']), float(right['start'])), min(float(left['end']), float(right['end']))
            if min(b, e0) - max(a, s0) > TURN_OVERLAP_SECONDS:
                return True
    return False


def caption_key(source_identity, caption):
    """Caption identity shared by every window: source + caption start/end (one subtitle stream, ms)."""
    source = source_identity or {}
    return 'ck1-' + _digest([VERSION, source.get('source_fingerprint'), source.get('audio_index'),
                             _r(caption['start']), _r(caption['end'])])


def caption_speech_start(caption, speech_regions):
    """Earliest speech inside the caption among the regions plan_dialogue matched it with; None without any."""
    starts = [max(float(t['start']), float(caption['start'])) for t in speech_regions or ()
              if min(float(t['end']), float(caption['end'])) - max(float(t['start']), float(caption['start']))
              > CAPTION_SPEECH_OVERLAP_SECONDS]
    return min(starts) if starts else None


def caption_line_start(caption, bound_cues, speech_regions):
    """(time, basis): the first bound cue start (the caption's line = B's target owner), else its speech start."""
    if bound_cues:
        return min(float(c['start']) for c in bound_cues), 'first_bound_cue_start'
    start = caption_speech_start(caption, speech_regions)
    return (start, 'caption_speech_start') if start is not None else (None, 'no_speech')


def keep_caption_issue(caption, speech_regions, render_start, render_end):
    """A cue-less caption issue counts only in the window holding the caption's speech start."""
    start = caption_speech_start(caption, speech_regions)
    return start is not None and owns(start, render_start, render_end)


def unit_span(unit):
    spans = unit.get('authorized_audio_spans')
    if spans:
        return [float(spans[0][0]), float(spans[-1][1])]
    return [float(unit['start']), float(unit['end'])]


def unit_caption(unit):
    target = unit.get('caption_target')
    if isinstance(target, dict) and isinstance(target.get('caption'), dict):
        return target['caption']
    return unit.get('caption')


def line_record(unit, *, render_start, render_end, source_identity):
    """Ledger fields of a new take: stable line id (source, owner window, span, caption key)."""
    source = source_identity or {}
    span = unit_span(unit)
    cap = unit_caption(unit)
    key = caption_key(source_identity, cap) if cap else None
    window = [_r(render_start), _r(render_end)]
    ident = 'ln1-' + _digest([VERSION, source.get('source_fingerprint'), source.get('audio_index'), window,
                              [_r(span[0]), _r(span[1])], key])
    return {'line_id': ident, 'line_window': [float(render_start), float(render_end)], 'line_caption_key': key,
            'line_owner_basis': 'cue_start', 'line_source_span': span}


def plan(all_cues, captions, speech_regions, *, render_start, render_end, source_identity, caption_text):
    """Ownership of every analysed cue and caption, and the boundary items this window must account for.
    Annotates each cue dict with 'turn_overlap' (bool, see turn_overlap); nothing else on a cue is changed."""
    rs, re_ = float(render_start), float(render_end)
    rows = {}
    for cue in all_cues:
        start, end = float(cue['start']), float(cue['end'])
        caption = cue.get('caption')
        cue['turn_overlap'] = turn_overlap(cue, speech_regions)
        rows[cue['id']] = {'owner_time': start, 'owner_basis': 'cue_start', 'owned': owns(start, rs, re_),
                           'overlaps_render_range': end > rs and start < re_, 'turn_overlap': cue['turn_overlap'],
                           'caption_key': caption_key(source_identity, caption) if caption else None}
    owned_spans = [[float(c['start']) - OWNED_GUARD_SECONDS, float(c['end']) + OWNED_GUARD_SECONDS]
                   for c in all_cues if rows[c['id']]['owned']]
    owned_cue_spans = [[float(c['start']), float(c['end'])] for c in all_cues if rows[c['id']]['owned']]
    turns = [[float(t['start']), float(t['end'])] for t in speech_regions or ()]
    groups = {}
    for cue in all_cues:
        row = rows[cue['id']]
        if row['owned'] or not row['overlaps_render_range']:
            continue
        caption = cue.get('caption')
        group = ('caption', row['caption_key']) if caption else ('asr', cue['id'])
        groups.setdefault(group, {'caption': dict(caption) if caption else None, 'cues': [],
                                  'caption_speech': False})['cues'].append(cue)
    accounting = []
    for caption in captions or ():
        if not caption_text(caption.get('text', '')):
            continue
        if not (float(caption['end']) > rs and float(caption['start']) < re_):
            continue
        key = caption_key(source_identity, caption)
        bound = [c for c in all_cues if rows[c['id']]['caption_key'] == key]
        start, basis = caption_line_start(caption, bound, speech_regions)
        owned = start is not None and owns(start, rs, re_)
        entry = {'caption_start': float(caption['start']), 'caption_end': float(caption['end']), 'caption_key': key,
                 'line_start': start, 'line_start_basis': basis, 'owned': owned,
                 'bound_cue_ids': [c['id'] for c in bound]}
        if owned:
            entry['handling'] = 'owner_window'
        else:
            group = groups.setdefault(('caption', key), {'caption': dict(caption), 'cues': [], 'caption_speech': False})
            group['caption_speech'] = True
            entry['handling'] = 'boundary_item'
            entry['item_id'] = key
        accounting.append(entry)

    def line_start(value):
        if value['cues']:
            return min(float(c['start']) for c in value['cues'])
        return caption_speech_start(value['caption'], speech_regions) or float(value['caption']['start'])

    items = []
    for group, value in sorted(groups.items(), key=lambda kv: (line_start(kv[1]), kv[0])):
        caption, cues = value['caption'], value['cues']
        raw_pieces = clip([[float(c['start']), float(c['end'])] for c in cues], rs, re_)
        # v4: where this window's own (owned) cue covers the same time, that cue dubs or keeps it; the neighbour's
        # overlapping ASR span is not protected against it (adjacent lines jittering across a boundary). Only the
        # actually shared time is deferred (no guard): a merely adjacent owned cue leaves the wall unchanged.
        cue_pieces = subtract(raw_pieces, owned_cue_spans)
        cue_inside_owned = intersect(raw_pieces, owned_cue_spans)
        caption_pieces, inside_owned = [], []
        if caption and value['caption_speech']:
            speech = clip([[max(a, float(caption['start'])), min(b, float(caption['end']))] for a, b in turns], rs, re_)
            caption_pieces = subtract(speech, owned_spans)
            inside_owned = subtract(intersect(speech, owned_spans), cue_pieces)
        items.append({'item_id': group[1] if caption else 'asr-' + group[1],
                      'kind': 'caption' if caption else 'asr_cue', 'line_start': line_start(value),
                      'owner': 'other_window',
                      'caption': None if not caption else {'start': float(caption['start']), 'end': float(caption['end'])},
                      'caption_key': group[1] if caption else None,
                      'cue_ids': [c['id'] for c in cues], 'cue_spans': [[float(c['start']), float(c['end'])] for c in cues],
                      'speech_pieces': merge(cue_pieces + caption_pieces),
                      'speech_inside_owned_cues': merge(inside_owned + cue_inside_owned),
                      'speech_basis': 'non-owned cue spans' + (' + speech regions inside the caption, minus owned cues'
                                                                if caption_pieces or inside_owned else '')})
    return {'version': VERSION, 'rule': 'cue_start', 'utterance_rule': UTTERANCE_RULE, 'render_range': [rs, re_],
            'cues': rows,
            'owned_cue_ids': [c['id'] for c in all_cues if rows[c['id']]['owned']],
            'items': items, 'captions': accounting}


def source_spans(take):
    spans = take.get('authorized_audio_spans')
    if spans:
        return [[float(a), float(b)] for a, b in spans]
    return [[float(take['source_start']), float(take['source_end'])]]


def related(item, take):
    """How a canonical take belongs to this item's dialogue, or None: same cue id, same caption key, or the
    same-line source overlap rule against one of the item's cues. Mere proximity never counts."""
    ids = set(item.get('cue_ids') or ())
    if take.get('cue') in ids or ids & set(take.get('source_cue_ids') or ()):
        return 'cue_id'
    if item.get('caption_key') and take.get('line_caption_key') == item['caption_key']:
        return 'caption_key'
    for span in item.get('cue_spans') or ():
        if any(significant_overlap(span, s) for s in source_spans(take)):
            return 'same_line_source_overlap'
    return None


def decide(items, canonical, *, render_start, render_end, canonical_intervals, take_span, evaluated_at):
    """Per-span decisions against the canonical takes that exist now. Returns (decisions, protected spans)."""
    rs, re_ = float(render_start), float(render_end)
    guards = [(i['id'], [take_span(i)[0] - TAKE_GUARD_SECONDS, take_span(i)[1] + TAKE_GUARD_SECONDS]) for i in canonical]
    guard = merge(span for _, span in guards)
    decisions, protected = [], []
    for item in items:
        pieces = clip(item['speech_pieces'], rs, re_)
        links = {i['id']: related(item, i) for i in canonical}
        mine = [i for i in canonical if links[i['id']]]
        cover = merge([a, b] for i in mine for a, b in canonical_intervals(i))
        uncovered = subtract(pieces, cover)
        row = {'item_id': item['item_id'], 'kind': item['kind'], 'line_start': item.get('line_start'),
               'cue_ids': item['cue_ids'], 'speech_pieces': pieces, 'evaluated_at': evaluated_at,
               'canonical_ids': sorted(i['id'] for i in mine),
               'related_basis': {i['id']: links[i['id']] for i in mine}}
        for key in ('claim_status', 'blocking_claim', 'conflicting_canonical_ids'):
            if key in item:
                row[key] = item[key]
        if not pieces and item.get('speech_inside_owned_cues'):
            row.update(decision='speech_inside_owned_cue', protected=[],
                       speech_inside_owned_cues=item['speech_inside_owned_cues'],
                       note="this window's own cue covers the speech; a second claim on a published take is refused at render")
        elif not pieces:
            row.update(decision='no_speech_in_render_range', protected=[])
        elif total(uncovered) <= COVER_TOLERANCE:
            row.update(decision='owner_published_dub_covers', protected=[],
                       proof='removal intervals of canonical takes of this dialogue (related_basis) cover the span')
        else:
            walls = subtract(uncovered, guard)
            under = subtract(uncovered, walls)
            under_ids = sorted(ident for ident, span in guards if intersect(under, [span]))
            reason = ('owner take covers only part of the span' if mine and total(uncovered) < total(pieces) - COVER_TOLERANCE
                      else 'owner take does not cover the span' if mine else 'no published owner take')
            row.update(protected=walls, reason=reason, uncovered=uncovered,
                       not_protected_under_carried_take=under, under_canonical_ids=under_ids)
            if walls and under:
                row['decision'] = 'mixed'
                row['note'] = ('part plays protected original (protected); part lies under a carried canonical take and '
                               'is NOT protected (not_protected_under_carried_take); see decisions_by_span')
            elif walls:
                row['decision'] = 'protected_original_fallback'
            else:
                row['decision'] = 'not_protected_under_carried_take'
                row['note'] = 'uncovered source speech lies under a carried canonical take; it is not protected'
            protected.extend(walls)
        covered = intersect(pieces, cover)
        row['decisions_by_span'] = sorted(
            [{'span': s, 'decision': 'owner_published_dub_covers'} for s in covered] +
            [{'span': s, 'decision': 'protected_original_fallback'} for s in row.get('protected') or ()] +
            [{'span': s, 'decision': 'not_protected_under_carried_take'} for s in row.get('not_protected_under_carried_take') or ()] +
            [{'span': s, 'decision': 'speech_inside_owned_cue'} for s in item.get('speech_inside_owned_cues') or ()],
            key=lambda r: (r['span'][0], r['decision']))
        decisions.append(row)
    return decisions, merge(protected)


def own_prior(cue, canonical, *, window, line_ids, match):
    """Reuse only this window's own canonical take of the line: the existing text match restricted to own-window
    takes, else the same line id. Another window's take is never reused (it is carried, never republished)."""
    own = [i for i in canonical if list(i.get('line_window') or ()) == [float(window[0]), float(window[1])]]
    if not own:
        return None
    found = match(cue, own) if 'english' in cue else None
    if found is not None:
        return found
    ident = line_ids.get(cue['id'])
    same = [i for i in own if ident and i.get('line_id') == ident]
    return same[0] if len(same) == 1 else None


def canonical_conflicts(cues, canonical, *, prior):
    """Owned synthesis cues without a reusable own take that would be a second claim on canonical dialogue
    (significant_overlap against any canonical take's source span). Returns [(cue, [canonical ids])]."""
    out = []
    for cue in cues:
        if not canonical or prior(cue) is not None:
            continue
        span = unit_span(cue)
        hits = sorted(i['id'] for i in canonical if any(significant_overlap(span, s) for s in source_spans(i)))
        if hits:
            out.append((cue, hits))
    return out


def overrun_limit(windows, *, window_key, render_end, end, take_id=None):
    """Latest end of a take: the analysed window end, or the start of the first OTHER committed window after
    render_end that did not mix this take (it rendered without it, so a tail there would be cut)."""
    limit = float(end)
    for w in windows:
        if w['window_key'] == window_key or (take_id is not None and take_id in (w.get('mixed_canonical_ids') or ())):
            continue
        a, b = float(w['render_start']), float(w['render_end'])
        if b > render_end and a < limit:
            limit = min(limit, max(float(render_end), a))
    return limit


def audibility(placement, *, render_start, render_end, limit):
    """Where a placed take is heard: inside this window's output, and after render_end through a later window's
    carry of the committed take. Problems name a placement that would be cut."""
    s, e = float(placement['start']), float(placement['end'])
    rs, re_ = float(render_start), float(render_end)
    problems = []
    if s < rs - EPS:
        problems.append('placed_before_owner_render_start')
    if e > float(limit) + 1e-4:
        problems.append('overruns_into_window_rendered_without_it')
    inside = [max(s, rs), min(e, re_)] if min(e, re_) > max(s, rs) else None
    after = [max(s, re_), e] if e > re_ else None
    return {'in_window': inside, 'after_render_end': after, 'carry_required': after is not None, 'limit': float(limit),
            'problems': problems}
