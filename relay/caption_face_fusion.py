"""Pure caption/visual activity alignment; never establishes acoustic identity.

Caption and detection-cell times are absolute movie seconds. ``origin`` is
the clip origin; ``first_pts[face]`` is the relative PTS of its first dense
TalkNet crop. Scores are uncalibrated TalkNet logits on a fixed ``fps`` grid.
Only real detector cells contribute evidence, even when crops interpolate gaps.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from copy import deepcopy

from .caption_speakers import caption_speaker_evidence, implied_speaker_turns

# Provisional (uncalibrated) defaults. Named so identity consumers can refuse a
# report built with looser overrides; these values are not retuned here.
VISUAL_SUPPORT_FLOORS = dict(min_detection_coverage=.8, min_scored_coverage=.8,
                             min_active_fraction=.7, min_support_seconds=.4)


def _number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a timestamp or score')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite value')
    return number


def _duration(spans):
    return sum(b-a for a, b in spans)


def _intersection_duration(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0.0, min(b, d)-max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def _face_cells(track, cells, fps):
    """Require cells to name real valid detections; never synthesize support."""
    detections = {}
    invalid_geometry = 0
    for row in track:
        try:
            t, x1, y1, x2, y2 = map(_number, row[:5])
            if t < 0 or not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                raise ValueError('Invalid normalized box')
            detections[t] = True
        except (ValueError, TypeError, OverflowError):
            invalid_geometry += 1
    valid, invalid_cells = [], 0
    last_end = -math.inf
    for cell in cells:
        try:
            a, b = _number(cell['start']), _number(cell['end'])
            if a < 0 or b <= a or b-a > 2.5/fps+1e-8 or a < last_end-1e-8:
                raise ValueError('Invalid or overlapping detector cells')
            # A dense crop timestamp alone cannot grant detection support.
            if not any(abs(t-a) <= 1e-7 for t in detections):
                raise ValueError('No valid detection at cell start')
            valid.append((a, b))
            last_end = b
        except (ValueError, TypeError, KeyError, OverflowError):
            invalid_cells += 1
    return valid, invalid_geometry, invalid_cells


def _face_evidence(face, cells, scores, score_start, start, end, fps, active_score):
    visible, scored, active = [], [], []
    weighted_score = 0.0
    for a, b in cells:
        a, b = max(a, start), min(b, end)
        if b <= a:
            continue
        visible.append([a, b])
        if score_start is None:
            continue
        first = max(0, math.floor((a-score_start)*fps+1e-8))
        last = min(len(scores), math.ceil((b-score_start)*fps-1e-8))
        for index in range(first, last):
            left = max(a, score_start+index/fps)
            right = min(b, score_start+(index+1)/fps)
            if right <= left:
                continue
            try:
                value = _number(scores[index])
            except (ValueError, TypeError, OverflowError):
                continue
            scored.append([left, right])
            weighted_score += value*(right-left)
            if value > active_score:
                active.append([left, right])
    visible_seconds, scored_seconds = _duration(visible), _duration(scored)
    active_seconds = _duration(active)
    duration = end-start
    return dict(face_id=face, detection_seconds=visible_seconds,
                detection_coverage=visible_seconds/duration,
                scored_seconds=scored_seconds, scored_coverage=scored_seconds/duration,
                active_seconds=active_seconds,
                active_fraction=active_seconds/scored_seconds if scored_seconds else 0.0,
                active_caption_coverage=active_seconds/duration,
                weighted_mean_score=weighted_score/scored_seconds if scored_seconds else None,
                detection_spans=visible, scored_spans=scored, active_spans=active,
                score_grid_start=score_start)


def match_caption_faces(captions, face_tracks, frame_cells, talknet_scores, *, origin,
                       first_pts, fps=25.0,
                       min_detection_coverage=VISUAL_SUPPORT_FLOORS['min_detection_coverage'],
                       min_scored_coverage=VISUAL_SUPPORT_FLOORS['min_scored_coverage'],
                       min_active_fraction=VISUAL_SUPPORT_FLOORS['min_active_fraction'],
                       min_support_seconds=VISUAL_SUPPORT_FLOORS['min_support_seconds'], active_score=0.0,
                       min_mean_score=0.0, score_margin=.3,
                       max_competing_active_seconds=.12):
    """Return every input caption, with a provisional face candidate or unknown.

    No actor names, face labels, subtitles or audio speaker predictions are used
    as identity truth. Parent bounds prevent a clipped fragment from masquerading
    as a fully evaluated caption. Unknown results retain source pending resolution;
    only the separate short-utterance policy can approve short-line passthrough.
    This evaluation helper never edits audio or approves downstream playback.
    """
    origin, fps = _number(origin), _number(fps)
    if origin < 0 or fps <= 0:
        raise ValueError('Invalid clip origin or score frame rate')
    min_detection_coverage, min_scored_coverage, min_active_fraction = map(
        _number, (min_detection_coverage, min_scored_coverage, min_active_fraction))
    if any(not 0 < v <= 1 for v in (min_detection_coverage, min_scored_coverage, min_active_fraction)):
        raise ValueError('Coverage and activity fractions must be in (0,1]')
    min_support_seconds, score_margin, max_competing_active_seconds = map(
        _number, (min_support_seconds, score_margin, max_competing_active_seconds))
    if min_support_seconds <= 0 or score_margin < 0 or max_competing_active_seconds < 0:
        raise ValueError('Invalid support or ambiguity threshold')
    active_score, min_mean_score = _number(active_score), _number(min_mean_score)
    captions = list(captions)
    prepared, timing = {}, []
    for face, track in face_tracks.items():
        cells, bad_geometry, bad_cells = _face_cells(track, frame_cells.get(face, []), fps)
        try:
            relative_pts = _number(first_pts[face])
            if relative_pts < 0:
                raise ValueError('Negative first PTS')
            score_start = origin+relative_pts
        except (KeyError, TypeError, ValueError, OverflowError):
            score_start = None
        prepared[face] = (cells, talknet_scores.get(face, []), score_start,
                          bad_geometry, bad_cells)
    for caption in captions:
        try:
            a, b = _number(caption['start']), _number(caption['end'])
            if a < 0 or b <= a:
                raise ValueError('Invalid caption interval')
            timing.append((a, b))
        except (KeyError, TypeError, ValueError, OverflowError):
            timing.append(None)
    results = []
    for index, caption in enumerate(captions):
        result = dict(caption_index=index, caption=deepcopy(caption), candidate_only=True,
                      face_id=None, acoustic_identity=None, state='unknown',
                      true_offscreen_speaker=None,
                      original_audio_policy='retain_source_pending_resolution',
                      hints=[], candidates=[], reasons=[], coverage_scope='caption_interval')
        results.append(result)
        if timing[index] is None:
            result['reasons'] = ['invalid_caption_interval']
            continue
        start, end = timing[index]
        try:
            parent_start = _number(caption.get('parent_start', start))
            parent_end = _number(caption.get('parent_end', end))
            if not 0 <= parent_start <= start < end <= parent_end:
                raise ValueError('Invalid parent caption bounds')
        except (ValueError, TypeError, OverflowError):
            result['reasons'] = ['invalid_parent_caption_bounds']
            continue
        result.update(start=start, end=end, parent_start=parent_start, parent_end=parent_end)
        clipped = parent_start < start-1e-8 or parent_end > end+1e-8
        if clipped or any(caption.get(flag) for flag in ('clipped', 'source_clipped', 'left_clipped', 'right_clipped', 'truncated')):
            result['reasons'].append('clipped_caption_requires_full_parent_review')
            result['coverage_scope'] = 'clipped_fragment_not_full_parent'
        try:
            result['hints'] = caption_speaker_evidence(caption, caption_index=index)
        except (TypeError, ValueError):
            result['reasons'].append('invalid_caption_text')
        text = caption.get('text')
        if isinstance(text, str) and (not text.strip() or re.fullmatch(r'\s*(?:\[[^\]]*\]|\([^)]*\))\s*', text)):
            result['reasons'].append('no_caption_dialogue_text')
        turns = [h for h in result['hints'] if h['kind'] == 'turn_marker']
        labels = {h['label'].casefold() for h in result['hints'] if h.get('label')}
        if len(turns) > 1:
            result['reasons'].append('multiple_dashed_caption_turns')
        else:
            # "Hello.\n- Bye." marks only the second speaker but is still mixed.
            try:
                if implied_speaker_turns(caption, result['hints']) > 1:
                    result['reasons'].append('implied_caption_speaker_change')
            except (TypeError, ValueError):
                pass
        if len(labels) > 1:
            result['reasons'].append('multiple_caption_speaker_hints')
        if caption.get('overlapped') or caption.get('overlap'):
            result['reasons'].append('overlapping_speech_reported')
        overlapping = [j for j, interval in enumerate(timing)
                       if j != index and interval and min(end, interval[1])-max(start, interval[0]) > 1e-8]
        if overlapping:
            result['reasons'].append('overlapping_caption_intervals')
            result['overlapping_caption_indices'] = overlapping
        for face, (cells, scores, score_start, bad_geometry, bad_cells) in prepared.items():
            evidence = _face_evidence(face, cells, scores, score_start, start, end, fps, active_score)
            if not evidence['detection_seconds']:
                continue
            evidence.update(invalid_geometry_rows=bad_geometry, invalid_frame_cells=bad_cells,
                            parent_scored_coverage=evidence['scored_seconds']/(parent_end-parent_start))
            evidence['sufficient_support'] = (
                evidence['detection_coverage']+1e-8 >= min_detection_coverage
                and evidence['scored_coverage']+1e-8 >= min_scored_coverage
                and evidence['scored_seconds']+1e-8 >= min_support_seconds)
            evidence['positive_activity'] = (evidence['weighted_mean_score'] is not None
                and evidence['weighted_mean_score'] > min_mean_score
                and evidence['active_fraction']+1e-8 >= min_active_fraction)
            result['candidates'].append(evidence)
        result['candidates'].sort(key=lambda row: (-(row['weighted_mean_score'] if row['weighted_mean_score'] is not None else -math.inf), str(row['face_id'])))
        eligible = [c for c in result['candidates'] if c['sufficient_support'] and c['positive_activity']]
        if not result['candidates']:
            result['reasons'].append('no_retained_face_track' if not face_tracks
                                     else 'no_valid_observed_face_support')
        elif not any(c['scored_seconds'] for c in result['candidates']):
            result['reasons'].append('missing_talknet_evidence')
        elif not eligible:
            result['reasons'].append('insufficient_positive_face_support')
        else:
            best = eligible[0]
            for other in result['candidates']:
                if other is best or other['scored_seconds']+1e-8 < min_support_seconds:
                    continue
                # Simultaneous activity blocks even when the other face's mean
                # is too low to be eligible itself (possible overlapping speech).
                simultaneous = _intersection_duration(best['active_spans'], other['active_spans'])
                if simultaneous > max_competing_active_seconds+1e-8:
                    result['reasons'].append('competing_active_faces')
                if (other['positive_activity'] and
                        abs(best['weighted_mean_score']-other['weighted_mean_score']) <= score_margin+1e-8):
                    result['reasons'].append('competing_near_scores')
            if not result['reasons']:
                result.update(state='candidate', face_id=best['face_id'],
                              original_audio_policy='candidate_requires_audio_identity_and_dub_validation')
                result['reasons'] = ['unambiguous_positive_visual_activity']
        result['reasons'] = list(dict.fromkeys(result['reasons']))
        result['true_offscreen_speaker'] = None
    policy = dict(version='caption-face-fusion-v1', thresholds_status='provisional_not_scene_tuned',
                  fps=fps, origin=origin, min_detection_coverage=min_detection_coverage,
                  min_scored_coverage=min_scored_coverage, min_active_fraction=min_active_fraction,
                  min_support_seconds=min_support_seconds, active_score=active_score,
                  min_mean_score=min_mean_score, score_margin=score_margin,
                  max_competing_active_seconds=max_competing_active_seconds,
                  score_semantics='TalkNet logits, not identity probabilities',
                  timing='absolute detector cells intersected with origin+first_pts+n/fps score cells',
                  scope='caption timing and visual activity candidates; no audio identity truth')
    return dict(evaluation_only=True, audio_modified=False, policy=policy, captions=results,
                face_validation={face: dict(valid_detection_cells=len(value[0]),
                    invalid_geometry_rows=value[3], invalid_frame_cells=value[4],
                    score_grid_start=value[2]) for face, value in prepared.items()},
                summary=dict(input_captions=len(captions), output_captions=len(results),
                             states=dict(Counter(row['state'] for row in results))))
