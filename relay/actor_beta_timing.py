"""actor_beta-only scheduling repair: no phantom placements for takes that will keep original audio.

Installed next to relay/actor_pipeline.py and imported lazily from the beta branch only. Pure Python.

The legacy beta path schedules every rendered take, then after_schedule() moves readback/timing take issues
to original fallback. An overrunning (or readback-invalid) take still occupied its placement while its
neighbours were scheduled, so a later cue could be pushed into a "phantom" overrun although it fits once the
discarded cue keeps its original speech. This helper:

1. Moves readback-invalid takes (the issue rows the existing verification loop already emitted with its exact
   semantics: speech_readback_mismatch / critical_readback_mismatch) to ordinary original fallback BEFORE
   scheduling, and protects their original source spans.
2. Schedules the remaining takes with the unchanged scheduler. While timing overruns remain, moves ONLY the
   earliest offending cue to original fallback, protects its original source span and reschedules.

Durations, gaps, max-early, walls and canonical/carry handling are unchanged; no tempo, clipping or truncation.
Any malformed state (cue/line mismatch, duplicate ids, unmapped or unknown issue, scheduler error, no progress)
returns the exact legacy single-pass schedule with no mutation, recorded in report['beta_timing_reschedule'].
"""
from __future__ import annotations

import math

VERSION = 'beta-timing-reschedule-v1'
READBACK_KINDS = ('speech_readback_mismatch', 'critical_readback_mismatch')
TIMING_KIND = 'speech_timing_overrun'
CASCADE_REASON = 'overlaps_original_fallback'  # same reason string as actor_beta_hook._cascade


class _Abort(Exception):
    pass


def _canonical_override(cues, lines, placements, timing_issues):
    """Exactly the existing per-iteration canonical handling in _render_prepared."""
    placements = list(placements)
    for index, line in enumerate(lines):
        if line.get('canonical_id'):
            placements[index] = {k: line[k] for k in ('start', 'end', 'source_start', 'source_end',
                                                      'early_seconds', 'speed', 'timing_overrun')}
            if line.get('timing_slack') is not None:
                placements[index]['timing_slack'] = line['timing_slack']
            timing_issues = [i for i in timing_issues if i.get('cue') != cues[index]['id']]
    return placements, timing_issues


def _overlaps(cue, span):
    return min(float(cue['end']), span[1]) > max(float(cue['start']), span[0])


def schedule_with_reschedule(issues, report, cues, lines, protected, *, floor_for, end, schedule, merge):
    """Return (cues, lines, placements, timing_issues, protected) for the actor_beta path.

    issues: cumulative issue list (readback rows for moved cues are removed on success).
    floor_for(cues): the existing floor formula for the CURRENT first cue (carry-aware).
    schedule: relay.actor_pipeline.schedule_around_protection. merge: dialogue_ledger.merge_spans.
    """
    def legacy(reason):
        placements, timing_issues = schedule(cues, [line['seconds'] for line in lines],
                                             origin=floor_for(cues), end=end, protected=protected)
        report['beta_timing_reschedule'] = {'version': VERSION, 'status': 'legacy', 'reason': reason}
        return cues, lines, placements, timing_issues, protected

    try:
        if len(cues) != len(lines):
            raise _Abort('cue/line count mismatch')
        ids = [c.get('id') for c in cues]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise _Abort('missing or duplicate cue ids')
        if any(line.get('cue') != c['id'] for c, line in zip(cues, lines)):
            raise _Abort('line is not associated with its cue')
        readback = [i for i in issues if isinstance(i, dict) and i.get('kind') in READBACK_KINDS]
        if any(i.get('cue') not in ids for i in readback):
            raise _Abort('readback issue does not map to a scheduled cue')

        rows = list(zip(cues, lines))
        walls = list(protected)
        fallback, dropped = {}, []

        def drop(cue_id, reasons, stage):
            """Remove cue_id and, transitively, every remaining cue overlapping a newly kept original span.

            Explicit worklist over CURRENT membership: each cue is removed at most once (transitive and cyclic
            overlaps terminate). The requested cue itself must be bound now; anything else is an identity error.
            """
            nonlocal rows, walls
            if not any(c['id'] == cue_id for c, _ in rows):
                raise _Abort('drop of a cue that is not currently scheduled: %r' % (cue_id,))
            work = [(cue_id, list(reasons), stage)]
            while work:
                current, why, at = work.pop(0)
                cue = next((c for c, _ in rows if c['id'] == current), None)
                if cue is None:
                    if current not in fallback:
                        raise _Abort('cascade reached an unbound cue: %r' % (current,))
                    fallback[current].extend(r for r in why if r not in fallback[current])  # removed earlier in this cascade
                    continue
                span = (float(cue['start']), float(cue['end']))
                fallback.setdefault(current, []).extend(why)
                dropped.append({'cue': current, 'stage': at, 'reasons': list(why), 'source': list(span)})
                rows = [(c, l) for c, l in rows if c['id'] != current]
                walls = merge(walls + [span])
                # A remaining cue overlapping kept original speech cannot be scheduled (same as _cascade).
                work.extend((c['id'], [CASCADE_REASON], at + '-cascade') for c, _ in rows if _overlaps(c, span))

        for cue_id in dict.fromkeys(i['cue'] for i in readback):
            kinds = [i['kind'] for i in readback if i['cue'] == cue_id]
            if cue_id in fallback:          # already kept original by an earlier readback cascade: record its own reason
                fallback[cue_id].extend(k for k in kinds if k not in fallback[cue_id])
            else:                           # bound (checked above) and not yet removed
                drop(cue_id, kinds, 'readback')

        limit = len(cues)
        for iteration in range(limit + 1):
            cur_cues = [c for c, _ in rows]
            cur_lines = [l for _, l in rows]
            try:
                placements, timing_issues = schedule(cur_cues, [l['seconds'] for l in cur_lines],
                                                     origin=floor_for(cur_cues), end=end, protected=walls)
            except ValueError as error:
                raise _Abort('scheduler refused: %s' % error)
            placements, timing_issues = _canonical_override(cur_cues, cur_lines, placements, timing_issues)
            if not timing_issues:
                break
            live = {c['id']: (float(c['start']), n) for n, (c, l) in enumerate(rows) if not l.get('canonical_id')}
            if any(not isinstance(i, dict) or i.get('kind') != TIMING_KIND or i.get('cue') not in live
                   for i in timing_issues):
                raise _Abort('unmapped or unknown timing issue')
            if iteration == limit:
                raise _Abort('no convergence within candidate bound')
            earliest = min({i['cue'] for i in timing_issues}, key=lambda k: live[k])
            before = len(rows)
            drop(earliest, [TIMING_KIND], 'timing')
            if len(rows) >= before:
                raise _Abort('no progress')
    except _Abort as error:
        return legacy(str(error))

    # Commit: move the readback rows that became ordinary original fallback (same as take_fallbacks).
    moved = set(fallback)
    issues[:] = [i for i in issues if not (isinstance(i, dict) and i.get('kind') in READBACK_KINDS
                                           and i.get('cue') in moved)]
    runtime = report.setdefault('beta_runtime_fallback', {})
    for cue_id, reasons in fallback.items():
        runtime.setdefault(cue_id, []).extend(reasons)
    report['beta_timing_reschedule'] = {'version': VERSION, 'status': 'ok', 'initial_candidates': len(cues),
                                        'iterations': iteration + 1, 'dropped': dropped,
                                        'scheduled_cue_ids': [c['id'] for c in cur_cues]}
    return cur_cues, cur_lines, placements, timing_issues, walls


PLACEMENT_KEYS = ('start', 'end', 'source_start', 'source_end', 'early_seconds', 'speed', 'timing_overrun', 'timing_slack')


def stamp_published(report, pending, cues, lines, placements, carry=(), window_frames=None):
    """After a SUCCESSFUL mix: explicit cue mapping of what this window actually wrote into its output.

    pending: the (cue, line, placement, pan, gain) rows mixed into `generated`, after after_mix_takes.
    Canonical reused lines (canonical_id) of this window were added via add_utterance.
    carry: canonical takes owned by OTHER window receipts, mixed here; listed separately, never as published here.
    Every scheduled placement is stamped with cue_id and whether it was published, so a placement moved to
    original by after_mix_takes can't be read as played. Attempted takes stay in report['utterances'].
    On any identity inconsistency the published lists are NOT emitted (status 'inconsistent'); fail closed.
    """
    runtime = report.get('beta_runtime_fallback') or {}
    by_id = {}
    for cue, line, placement in zip(cues, lines, placements):
        if line.get('cue') != cue['id'] or cue['id'] in by_id:
            report['published_mapping'] = {'version': VERSION, 'status': 'inconsistent',
                                           'reason': 'cue/line association or duplicate id', 'cue': cue['id']}
            return None
        by_id[cue['id']] = (cue, line, placement)
    published, problems = [], []
    for cue_id, (cue, line, placement) in by_id.items():
        if line.get('canonical_id'):
            if cue_id in runtime:
                problems.append('canonical reuse %s is also an original fallback' % cue_id)
            published.append(('canonical_reused', cue, line, placement))
    for cue, line, placement, _, _ in pending:
        row = by_id.get(cue['id'])
        if row is None or row[1] is not line or line.get('canonical_id'):
            problems.append('pending take %s does not map to its scheduled cue/line' % cue['id'])
        elif cue['id'] in runtime:
            problems.append('pending take %s is also an original fallback' % cue['id'])
        else:
            published.append(('generated', cue, line, placement))
    if problems:
        report['published_mapping'] = {'version': VERSION, 'status': 'inconsistent', 'reason': problems}
        return None
    published.sort(key=lambda row: float(row[3]['start']))
    ids = {cue['id'] for _, cue, _, _ in published}
    report['placements'] = [{**{k: p.get(k) for k in PLACEMENT_KEYS}, 'cue_id': c['id'], 'published': c['id'] in ids,
                             'fallback': None if c['id'] in ids else list(runtime.get(c['id'], ['not_mixed']))}
                            for c, _, p in by_id.values() if p is not None]
    report['published_cue_ids'] = [cue['id'] for _, cue, _, _ in published]
    report['published_utterances'] = [{
        'cue_id': cue['id'], 'kind': kind, 'audio': line.get('audio'), 'sha256': line.get('sha256'),
        'source_cue_ids': list(cue.get('source_cue_ids') or [cue['id']]),
        'seconds': line.get('seconds'), 'text': cue.get('english'),
        'canonical_id': line.get('canonical_id'), 'placement': {k: placement.get(k) for k in PLACEMENT_KEYS}}
        for kind, cue, line, placement in published]
    report['carried_from_other_windows'] = [{
        'canonical_id': item.get('id'), 'cue_id': item.get('cue', item.get('id')), 'revision': item.get('revision'),
        'start': item.get('start'), 'end': item.get('end'), 'source_start': item.get('source_start'),
        'note': 'owned by another window receipt; mixed here for continuity, not published by this window'}
        for item in carry]
    report['published_mapping'] = {'version': VERSION, 'status': 'ok', 'published': len(published),
                                   'scheduled': len(by_id), 'carried': len(report['carried_from_other_windows'])}
    _label_all_original(report, published, carry, window_frames)
    return report['published_cue_ids']


MAX_REASONS = 12


def _label_all_original(report, published, carry, window_frames):
    """A successful candidate that replaced NOTHING is labelled original-unverified (worker keeps result['mode']).

    Only when: no mode was set earlier (no-dialogue etc. preserved), nothing was published (no generated take and
    no canonical reuse), no carried canonical take was mixed, and either every selected cue is an explicit original
    fallback or the mix proves every frame unchanged. Unaccounted cue IDs stay visible. Audio/denominators are untouched.
    """
    if 'mode' in report or published or list(carry):
        return
    original = set(report.get('beta_original_fallback') or {}) | set(report.get('beta_runtime_fallback') or {}) \
        | set(report.get('kept_original_cue_ids') or ())
    selected = [c.get('id') for c in report.get('cues') or () if isinstance(c, dict)]
    plan = report.get('caption_target_plan') or {}
    superseded = set(plan.get('superseded_cue_ids') or ())
    replacements = {}
    for target in plan.get('targets') or ():
        target_id = target.get('target_id')
        if not isinstance(target_id, str) or not target_id:
            continue
        for source_id in target.get('source_cue_ids') or ():
            if source_id in superseded:
                replacements.setdefault(source_id, set()).add(target_id)
    accounted = original | {cid for cid, targets in replacements.items() if targets <= original}
    unaccounted = [cid for cid in selected if cid not in accounted]
    mix = report.get('mix') or {}
    unchanged = mix.get('unchanged_frames')
    all_unchanged = isinstance(window_frames, int) and unchanged == window_frames
    report['beta_original_check'] = {
        'version': VERSION, 'status': 'all-selected-original' if not unaccounted else
        'unchanged-with-unaccounted-cues' if all_unchanged else 'not-all-selected-original',
        'unaccounted': sorted(str(cid) for cid in unaccounted)[:MAX_REASONS],
        'fallback_targets_by_source': {cid: sorted(targets) for cid, targets in replacements.items()
                                       if targets <= original}}
    if unaccounted and not all_unchanged:
        return
    kinds = set()
    for table in (report.get('beta_original_fallback') or {}, report.get('beta_runtime_fallback') or {}):
        for cid in selected:
            for key in {cid} | replacements.get(cid, set()):
                kinds.update(str(k).split(':', 1)[0][:80] for k in table.get(key, ()))
    if any(cid in (report.get('kept_original_cue_ids') or ()) for cid in selected):
        kinds.add('policy_kept_original')
    head = ['unchanged_audio_no_published_takes'] if unaccounted else \
        ['all_selected_cues_original'] if selected else ['no_selected_cues']
    report['mode'] = 'original-unverified'
    report['beta_original_reason'] = (head + sorted(kinds) + ([] if all_unchanged else ['no_replacement_candidate']))[:MAX_REASONS]
    report['beta_original_evidence'] = {
        'version': VERSION, 'published': 0, 'carried_mixed': 0, 'selected_cues': len(selected),
        'mix_unchanged_frames': unchanged, 'window_frames': window_frames, 'mix_all_frames_unchanged': all_unchanged,
        'scope': 'mix receipt only; the written WAV and the later english.wav extraction are not re-verified here'}


def publication_accounting(report):
    """Receipt-backed PCM mixed into a chunk, never inferred from an English-candidate label.

    This does not establish source language, linguistic correctness, served media or actual playback.
    Selected take receipts and retry files are counted separately, not as fresh native inference calls.
    Carried tails are reported separately and do not earn new generated credit in the receiving chunk.
    """
    def span(start, end):
        try:
            if isinstance(start, bool) or isinstance(end, bool):
                return None
            start, end = float(start), float(end)
            return (start, end) if math.isfinite(start) and math.isfinite(end) and 0 <= start < end else None
        except (TypeError, ValueError, OverflowError):
            return None

    def rows(value):
        return [item for item in (value or ()) if isinstance(item, dict)]

    plan = report.get('caption_target_plan') or {}
    targets = {t['target_id']: t for t in rows(plan.get('targets')) if t.get('target_id')}
    superseded = set(plan.get('superseded_cue_ids') or ())
    planned = (set(report.get('synthesis_cue_ids') or ()) - superseded) | set(targets)
    utterances = [u for u in rows(report.get('utterances')) if not u.get('canonical_id')]
    # Actor receipts also have an analysis-context 'end'. Only a settled chunk's paired start/end supersede render bounds.
    bounds = span(report.get('start'), report.get('end')) if 'start' in report else \
        span(report.get('render_start'), report.get('render_end'))
    mapping = report.get('published_mapping') or {}
    # An original-window override may retain an earlier actor receipt. It cannot prove a fresh generated mix.
    overridden = report.get('pipeline_state') in ('needs_review', 'original_fallback') or \
        str(report.get('mode') or '').startswith('original')
    verified = mapping.get('status') == 'ok' and bounds is not None and bool(report.get('mix')) and not overridden
    published_ids = set(report.get('published_cue_ids') or ())
    not_audible = set((report.get('beta_line_ledger') or {}).get('published_not_audible') or ())
    audibility_unverified = set(not_audible)
    generated, reused, invalid, seen = [], [], [], set()
    for take in rows(report.get('published_utterances')):
        if take.get('kind') not in ('generated', 'canonical_reused'):
            continue
        cue_id = take.get('cue_id')
        placement = take.get('placement') or {}
        interval = span(placement.get('start'), placement.get('end'))
        audibility = take.get('rendered_audibility') or {}
        digest = take.get('sha256')
        if (not isinstance(audibility, dict) or audibility.get('audible') is not True
                or audibility.get('heard_over_mix') is not True or cue_id in not_audible):
            audibility_unverified.add(str(cue_id))
            continue
        if (not verified or not cue_id or cue_id not in published_ids or interval is None
                or not isinstance(digest, str) or len(digest) != 64 or not take.get('audio')):
            invalid.append(str(cue_id))
            continue
        clipped = span(max(bounds[0], interval[0]), min(bounds[1], interval[1]))
        if clipped is None:
            continue
        key = (cue_id, digest, *clipped)
        if key in seen:
            continue
        seen.add(key)
        source_ids = take.get('source_cue_ids') or (targets.get(cue_id) or {}).get('source_cue_ids')
        if not source_ids:
            source_ids = [] if cue_id in targets else [cue_id]
        item = {'cue_id': cue_id, 'source_cue_ids': list(source_ids), 'sha256': digest,
                'start': clipped[0], 'end': clipped[1], 'source_audio_language': 'unknown'}
        (generated if take['kind'] == 'generated' else reused).append(item)
    intervals = []
    for take in sorted(generated, key=lambda item: (item['start'], item['end'])):
        start, end = take['start'], take['end']
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    retry_files = {u['retry_evidence']['retry_audio'] for u in utterances
                   if isinstance(u.get('retry_evidence'), dict) and u['retry_evidence'].get('retry_audio')}
    source_ids = [c['id'] for c in rows(report.get('cues')) if c.get('id')]
    mapped_publications = published_ids if mapping.get('status') == 'ok' and not overridden else set()
    return {
        'version': 'published-pcm-accounting-v1',
        'publication_status': 'receipt-backed' if verified else 'unverified-or-original-override',
        'english_generated_seconds': round(sum(end - start for start, end in intervals), 6),
        'generated_intervals': generated, 'generated_union': intervals, 'canonical_reused_intervals': reused,
        'planned_cues': len(planned), 'selected_take_receipts': len(utterances),
        'selected_takes_not_published': sum(u.get('cue') not in mapped_publications for u in utterances),
        'retry_take_receipts': len(retry_files), 'published_generated': len(generated),
        'carried': len(rows(report.get('carried_from_other_windows'))),
        'invalid_published_cue_ids': invalid,
        'audibility_unverified_cue_ids': sorted(audibility_unverified),
        'source_audio_languages': {cid: 'unknown' for cid in source_ids},
        'original_english_cues': [], 'source_language_status': 'no-per-cue-audio-language-evidence',
        'window_language_context': report.get('language'),
        'foreign_dubbing_seconds': None, 'served_seconds': None, 'played_seconds': None,
        'scope': 'Accepted generated PCM placements clipped to this chunk; not foreign-language, served or heard proof'}
