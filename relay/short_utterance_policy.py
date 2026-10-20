"""Classify naturally short source utterances for an explicit original-audio policy.

No inference, audio editing, issue clearing, or automatic sentence passthrough.
Eligibility is independent of issue kind: the caller must still restrict use to
the owner's permitted unresolved short utterances and preserve other blockers.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be finite seconds')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite seconds')
    return result


def _bounds(row, label):
    start, end = _number(row['start'], label), _number(row['end'], label)
    if start < 0 or end <= start:
        raise ValueError(f'{label} must be a positive absolute source interval')
    return start, end


def _intersection(a, b):
    return min(a[1], b[1]) > max(a[0], b[0])


def _distance(a, b):
    return max(0., max(a[0], b[0]) - min(a[1], b[1]))


def _source_bounds(cue):
    """Do not shrink a cue to its ASR words when parent evidence is broader."""
    entries = [('cue', _bounds(cue, 'cue'))]
    for first, last in (('speech_start', 'speech_end'), ('parent_start', 'parent_end'),
                        ('source_start', 'source_end'),
                        ('source_segment_start', 'source_segment_end')):
        if first in cue or last in cue:
            if first not in cue or last not in cue:
                raise ValueError('Source boundary pair is incomplete')
            entries.append((first.removesuffix('_start'), _bounds(
                {'start': cue[first], 'end': cue[last]}, first)))
    for field in ('parent_bounds', 'source_segment_bounds'):
        if field in cue:
            entries.append((field, _bounds(cue[field], field)))
    return entries


def plan_short_original(cues, speech_regions=(), *, max_seconds=1.0,
                        origin=None, end=None, continuation_gap_seconds=.35,
                        boundary_margin_seconds=.15, max_words=2):
    """Return one decision per cue without dropping or mutating any cue data.

    All input times are absolute movie seconds. ``speech_regions`` must be full
    source acoustic speech regions from the observed analysis window, not copies
    of cropped ASR words or caption timestamps. ``origin`` and ``end`` describe
    that actual observation window. Both source boundaries need observed context;
    a region touching an analysis cut never proves a standalone utterance.

    The natural interval covers cue/speech/parent bounds and every intersecting
    source region. A long parent, nearby continuation, actual/flagged overlap or
    singing prevents eligibility. The default two-word cap is a conservative
    fragment filter, not semantic sentence detection. Unknown/empty transcripts
    can qualify on acoustic evidence; a valid full sentence must not be passed
    through merely because its duration fits. Caller policy selects permitted
    unresolved cases; this function cannot clear translation/identity issues.

    Missing source/window evidence returns review decisions. Invalid policy or
    interval data raises ValueError, never silently discarding bad evidence.
    """
    maximum = _number(max_seconds, 'max_seconds')
    continuation = _number(continuation_gap_seconds, 'continuation gap')
    margin = _number(boundary_margin_seconds, 'boundary margin')
    if not 0 < maximum <= 2 or not 0 <= continuation <= 2 or not 0 < margin <= 2:
        raise ValueError('Invalid short-utterance policy limits')
    edge_context = max(margin, continuation)
    if isinstance(max_words, bool) or not isinstance(max_words, int) or not 1 <= max_words <= 6:
        raise ValueError('max_words must be an integer from 1 through 6')
    if (origin is None) != (end is None):
        raise ValueError('Both observed window boundaries must be supplied')
    window = None if origin is None else _bounds({'start': origin, 'end': end}, 'observed window')
    cues = list(cues)
    ids = [cue['id'] for cue in cues]
    if any(not isinstance(ident, str) or not ident for ident in ids) or len(set(ids)) != len(ids):
        raise ValueError('Cue IDs must be unique nonempty strings')
    cue_evidence = [_source_bounds(cue) for cue in cues]
    cue_bounds = [(min(b[0] for _, b in entries), max(b[1] for _, b in entries))
                  for entries in cue_evidence]
    regions = []
    for region in speech_regions:
        entries = _source_bounds(region)
        regions.append((deepcopy(region), (min(bound[0] for _, bound in entries),
                                            max(bound[1] for _, bound in entries))))
    decisions = []
    for index, cue in enumerate(cues):
        reasons = []
        def reject(reason):
            if reason not in reasons:
                reasons.append(reason)
        interval = cue_bounds[index]
        evidence = [{'basis': label, 'start': bound[0], 'end': bound[1]}
                    for label, bound in cue_evidence[index]]
        support = [(number, row, bound) for number, (row, bound) in enumerate(regions)
                   if _intersection(interval, bound)]
        if not support:
            reject('natural_source_boundaries_unproven')
        else:
            interval = (min(interval[0], *(bound[0] for _, _, bound in support)),
                        max(interval[1], *(bound[1] for _, _, bound in support)))
            evidence.extend({'basis': 'source-speech-region', 'region_index': number,
                             'start': bound[0], 'end': bound[1], 'speaker': row.get('speaker')}
                            for number, row, bound in support)
            # Boundary coverage must come from acoustic source support, not
            # from an ASR extension that happens to be close to that support.
            if min(bound[0] for _, _, bound in support) > interval[0] or max(bound[1] for _, _, bound in support) < interval[1]:
                reject('natural_source_boundaries_unproven')
        duration = interval[1] - interval[0]
        if duration > maximum + 1e-9:
            reject('natural_utterance_exceeds_cutoff')
        # Each edge needs enough observed context to see a continuation:
        # a margin shorter than the continuation gap hides speech beyond a cut.
        if window is None:
            reject('observed_window_boundaries_missing')
        elif interval[0] - window[0] < edge_context - 1e-9 or window[1] - interval[1] < edge_context - 1e-9:
            reject('analysis_cut_or_insufficient_boundary_context')
        if any(cue.get(flag) for flag in ('clipped', 'truncated', 'partial', 'source_clipped',
                                          'left_clipped', 'right_clipped', 'benchmark_cut')):
            reject('explicitly_clipped_source')
        if cue.get('singing'):
            reject('singing_requires_review')
        if cue.get('overlapped'):
            reject('overlapping_speech_requires_review')
        if cue.get('sentence_complete') is True:
            reject('complete_sentence_requires_normal_resolution')
        text = str(cue.get('text') or '').strip()
        word_count = len(re.findall(r"\w+(?:['’]\w+)*", text, flags=re.UNICODE))
        if word_count > max_words:
            reject('source_text_exceeds_fragment_word_limit')
        # Non-whitespace languages cannot use this tokenizer as proof of a
        # one-word fragment. Preserve unknown lexical segmentation for review.
        if text and re.search(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]', text):
            reject('source_word_segmentation_unverified')
        for other_index, other_interval in enumerate(cue_bounds):
            if other_index == index:
                continue
            if _intersection(interval, other_interval):
                reject('overlapping_cue_requires_review')
            elif _distance(interval, other_interval) <= continuation + 1e-9:
                reject('adjacent_cue_may_continue_utterance')
        support_indices = {number for number, _, _ in support}
        for number, (row, bound) in enumerate(regions):
            if number not in support_indices and _distance(interval, bound) <= continuation + 1e-9:
                reject('adjacent_source_speech_may_continue_utterance')
            if row.get('singing') and _intersection(interval, bound):
                reject('singing_requires_review')
            if row.get('overlapped') and _intersection(interval, bound):
                reject('overlapping_speech_requires_review')
            if _intersection(interval, bound) and any(row.get(flag) for flag in (
                'clipped', 'truncated', 'partial', 'source_clipped', 'left_clipped',
                'right_clipped', 'benchmark_cut'
            )):
                reject('explicitly_clipped_source_region')
        for position, (_, left, left_bound) in enumerate(support):
            for _, right, right_bound in support[position + 1:]:
                if _intersection(left_bound, right_bound) and (
                    not left.get('speaker') or not right.get('speaker') or left['speaker'] != right['speaker']
                ):
                    reject('overlapping_source_regions_require_review')
        eligible = not reasons
        decisions.append({'id': cue['id'], 'cue': deepcopy(cue), 'eligible': eligible,
                          'reason': 'standalone_short_source_utterance' if eligible else reasons[0],
                          'reasons': reasons, 'natural_interval': {'start': interval[0], 'end': interval[1]},
                          'natural_seconds': duration, 'source_word_count': word_count,
                          'original_keep_interval': {'start': interval[0], 'end': interval[1]} if eligible else None,
                          'evidence': evidence, 'automatic_passthrough': False})
    return {'version': 'standalone-short-original-v1', 'scope': 'source-eligibility-only',
            'policy': {'max_seconds': maximum, 'max_words': max_words,
                       'continuation_gap_seconds': continuation, 'boundary_margin_seconds': margin},
            'decisions': decisions}
