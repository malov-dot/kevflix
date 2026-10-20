"""Evaluation-only movie identity fusion; persistent acoustic IDs stay primary.

Consumes SpeakerIdentityEngine.handle({'op': 'analyze', ...}) output without
importing the worker, loading a model or writing its reference bank. Candidate
identities never authorize cloning. Exact audio, movie and encoder scope are
retained; unavailable human-face detection is not proof of offscreen speech.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re

from .caption_face_fusion import VISUAL_SUPPORT_FLOORS
from .caption_speakers import caption_speaker_evidence, implied_speaker_turns

# v2: one acoustic observation is (analyzed PCM hash, worker local speaker), so
# cue splits/retimings of identical PCM neither add support nor evade replay
# conflicts. v1 checkpoints keyed observations by cue interval; they are
# refused rather than mixed with v2 keys.
CHECKPOINT_VERSION = 2


def _number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not timing or identity evidence')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite timing or identity evidence')
    return value


def _text(value, maximum=160):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError('Expected a bounded nonempty identity/provenance string')
    return value


def _interval(start, end):
    start, end = _number(start), _number(end)
    if start < 0 or end <= start:
        raise ValueError('Expected a positive bounded interval')
    return start, end


def _union(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]+1e-8:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _overlap_seconds(left, right):
    return sum(max(0., min(b, d)-max(a, c)) for a, b in left for c, d in right)


def _meets_visual_floors(candidate):
    """Candidate face evidence must meet the matcher's default floors."""
    try:
        return (_number(candidate['detection_coverage'])+1e-8 >= VISUAL_SUPPORT_FLOORS['min_detection_coverage']
                and _number(candidate['scored_coverage'])+1e-8 >= VISUAL_SUPPORT_FLOORS['min_scored_coverage']
                and _number(candidate['active_fraction'])+1e-8 >= VISUAL_SUPPORT_FLOORS['min_active_fraction']
                and _number(candidate['scored_seconds'])+1e-8 >= VISUAL_SUPPORT_FLOORS['min_support_seconds'])
    except (KeyError, TypeError, ValueError):
        return False


def _gaps(start, end, spans):
    gaps, cursor = [], start
    for begin, stop in spans:
        if begin > cursor+1e-8:
            gaps.append([cursor, begin])
        cursor = max(cursor, stop)
    if cursor < end-1e-8:
        gaps.append([cursor, end])
    return gaps


class MovieSpeakerIdentity:
    """Bounded source/encoder-scoped candidate state, separate from people.

    resolve(analysis, *, origin, audio_duration, cues=None, captions=(),
            visual_report=None, visual_scope=None, face_identities=None)

    ``origin`` is the absolute start of the actual cropped WAV supplied to the
    worker (include request.start if it cropped again). Default cues are its
    exact turns. A supplied longer cue retains uncovered spans and abstains.
    Visual rows come from match_caption_faces and must cover the exact cue;
    callers should run that helper on the same intervals. A raw face track is
    scoped by visual_scope; persistent face IDs must be supplied explicitly.
    """
    def __init__(self, scope_id, model_sha256, *, max_voices=256, max_observations=2048, state=None):
        self.scope_id = _text(scope_id)
        if not isinstance(model_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', model_sha256):
            raise ValueError('Exact lowercase encoder SHA-256 required')
        if not isinstance(max_voices, int) or not 1 <= max_voices <= 512:
            raise ValueError('max_voices must be between 1 and 512')
        if not isinstance(max_observations, int) or not 1 <= max_observations <= 8192:
            raise ValueError('max_observations must be between 1 and 8192')
        self.model_sha256 = model_sha256
        self.max_voices, self.max_observations = max_voices, max_observations
        self.voices, self.observations, self.bindings, self.proposals = {}, set(), [], []
        self.audio_assignments = {}
        if state is not None:
            self._restore(state)

    def add_binding(self, kind, value, voice_id, *, verification, evidence_id):
        """Record an external independent anchor; never infer one from hints.

        This is an explicit caller assertion, not verification performed here.
        Even a binding cannot resolve unknown/weak audio or authorize cloning.
        Conflicting bindings are retained and force abstention when encountered.
        """
        if kind not in ('caption_label', 'face_identity') or verification not in ('manual', 'independent_audio'):
            raise ValueError('Bindings require explicit manual or independent-audio provenance')
        item = dict(kind=kind, value=_text(value), voice_id=_text(voice_id, 80),
                    verification=verification, evidence_id=_text(evidence_id))
        if item not in self.bindings:
            if len(self.bindings) >= 512:
                raise ValueError('Binding capacity reached')
            self.bindings.append(item)

    def _bound_voices(self, kind, value):
        normalized = value.casefold().strip() if kind == 'caption_label' else value
        return {item['voice_id'] for item in self.bindings if item['kind'] == kind and
                (item['value'].casefold().strip() if kind == 'caption_label' else item['value']) == normalized}

    def _voice_key(self, voice_id):
        data = json.dumps(['movie', self.scope_id, self.model_sha256, voice_id], separators=(',', ':'))
        return 'movie-voice-'+hashlib.sha256(data.encode()).hexdigest()[:24]

    def _audio_candidate(self, row, policy, duration):
        identity = row.get('identity') or {}
        if identity.get('state') != 'candidate' or not identity.get('candidate_id'):
            return None, ['audio_'+str(identity.get('reason') or 'identity_unknown')], {}
        voice = _text(identity['candidate_id'], 80)
        candidates = identity.get('candidates', [])
        if not candidates or candidates[0].get('speaker_id') != voice:
            return None, ['audio_candidate_ranking_mismatch'], {}
        try:
            scores = [_number(item['score']) for item in candidates]
            if any(not -1 <= score <= 1 for score in scores) or scores != sorted(scores, reverse=True):
                raise ValueError('Invalid ranking')
            spans = _union([_interval(*span) for span in row.get('solo_spans', [])])
            if any(end > duration+1e-8 for _, end in spans):
                raise ValueError('Solo speech extends outside analyzed audio')
            solo = sum(end-start for start, end in spans)
            if abs(solo-_number(row.get('solo_seconds', 0))) > 1e-4:
                raise ValueError('Solo duration does not match spans')
            margin = scores[0]-scores[1] if len(scores) > 1 else None
            if margin is not None and abs(margin-_number(identity.get('margin'))) > 1e-8:
                raise ValueError('Margin inconsistent with ranking')
        except (ValueError, TypeError, KeyError):
            return None, ['audio_evidence_invalid'], {}
        reasons = []
        if solo < policy['minimum_solo_seconds']:
            reasons.append('audio_insufficient_solo_speech')
        if scores[0] < policy['threshold']:
            reasons.append('audio_below_source_threshold')
        if margin is not None and margin < policy['margin']:
            reasons.append('audio_ambiguous_candidates')
        evidence = dict(score=scores[0], margin=margin, solo_seconds=solo,
                        score_semantics='cosine similarity, not calibrated probability',
                        identity_verified=False, policy_calibrated=policy['calibrated'])
        return (None if reasons else voice), reasons, evidence

    def resolve(self, analysis, *, origin, audio_duration, cues=None, captions=(),
                visual_report=None, visual_scope=None, face_identities=None):
        if analysis.get('scope') != {'kind': 'movie', 'id': self.scope_id}:
            raise ValueError('Movie scope mismatch; personal/cross-movie evidence refused')
        if analysis.get('model_sha256') != self.model_sha256:
            raise ValueError('Identity encoder mismatch')
        audio_hash = analysis.get('audio_sha256')
        if not isinstance(audio_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', audio_hash):
            raise ValueError('Exact analyzed PCM hash required')
        origin, audio_duration = _number(origin), _number(audio_duration)
        if origin < 0 or not 0 < audio_duration <= 30:
            raise ValueError('Supply absolute crop origin and bounded analyzed duration (<=30s)')
        supplied_policy = analysis.get('policy') or {}
        policy = {key: _number(supplied_policy[key]) for key in ('threshold', 'margin', 'minimum_solo_seconds')}
        if not -1 <= policy['threshold'] <= 1 or not 0 <= policy['margin'] <= 2 or policy['minimum_solo_seconds'] <= 0:
            raise ValueError('Invalid worker identity policy')
        policy['calibrated'] = supplied_policy.get('calibrated') is True
        turns = []
        for turn in analysis.get('turns', []):
            start, end = _interval(turn['start'], turn['end'])
            if end > audio_duration+1e-8:
                raise ValueError('Audio turn lies outside analyzed crop')
            turns.append(dict(start=start+origin, end=end+origin, speaker=_text(turn['speaker'], 80)))
        speakers = {}
        for row in analysis.get('speakers', []):
            local = _text(row['local_speaker'], 80)
            if local in speakers:
                raise ValueError('Duplicate local audio speaker')
            speakers[local] = row
        if cues is None:
            cues = [dict(id=f'{audio_hash[:16]}:turn-{index}', start=turn['start'], end=turn['end'])
                    for index, turn in enumerate(turns)]
        cues, captions = list(cues), list(captions)
        if len(cues) > 512 or len(captions) > 512:
            raise ValueError('Resolve calls are bounded to 512 cues/captions')
        if len({cue['id'] for cue in cues}) != len(cues):
            raise ValueError('Cue IDs must be unique within an observation')
        face_identities = face_identities or {}
        visual_rows = []
        if visual_report is not None:
            _text(visual_scope)
            if visual_report.get('policy', {}).get('version') != 'caption-face-fusion-v1':
                raise ValueError('Expected caption_face_fusion report, not a CMIA correlation as identity truth')
            visual_rows = visual_report.get('captions', [])
            if not isinstance(visual_rows, list) or len(visual_rows) > 512:
                raise ValueError('Invalid bounded visual report')
        intervals = [_interval(cue['start'], cue['end']) for cue in cues]
        results, created_now = [], set()
        for cue_index, cue in enumerate(cues):
            cue_id = _text(cue['id'], 256)
            start, end = intervals[cue_index]
            result = dict(id=cue_id, start=start, end=end, state='unresolved', voice_id=None,
                          persistent_voice_key=None, audio_candidate_id=None, reasons=[], notes=[],
                          caption_hints=[], visual_identity=None, visual_track=None,
                          true_offscreen_speaker=None, candidate_only=True, identity_verified=False,
                          clone_binding_ready=False, original_audio_policy='retain_source_pending_resolution')
            results.append(result)
            reasons, notes = result['reasons'], result['notes']
            if start < origin-1e-8 or end > origin+audio_duration+1e-8:
                reasons.append('cue_outside_analyzed_audio')
            # Simultaneous subtitle lines (mixed cues) or retimed duplicates
            # cannot both be attributed to one diarized voice; abstain.
            if any(other != cue_index and min(end, b)-max(start, a) > 1e-8
                   for other, (a, b) in enumerate(intervals)):
                reasons.append('overlapping_cue_intervals')
            supports = {}
            for turn in turns:
                left, right = max(start, turn['start']), min(end, turn['end'])
                if right > left:
                    supports.setdefault(turn['speaker'], []).append((left, right))
            union = _union([span for spans in supports.values() for span in spans])
            result['audio_support_spans'] = union
            result['uncovered_spans'] = _gaps(start, end, union)
            if result['uncovered_spans']:
                reasons.append('incomplete_audio_turn_support')
            local = None
            if len(supports) != 1:
                reasons.append('multiple_audio_speakers_in_cue' if supports else 'no_audio_speaker_support')
                voice = None
            else:
                local = next(iter(supports))
                result['local_audio_speaker'] = local
                if local not in speakers:
                    voice = None; reasons.append('missing_audio_identity_evidence')
                else:
                    voice, audio_reasons, evidence = self._audio_candidate(speakers[local], policy, audio_duration)
                    reasons.extend(audio_reasons); result['audio_evidence'] = evidence
            result['audio_candidate_id'] = voice
            proposed_hints = []
            observed_caption_labels = set()
            for index, caption in enumerate(captions):
                try:
                    cstart, cend = _interval(caption['start'], caption['end'])
                    if min(end, cend) <= max(start, cstart):
                        continue
                    hints = caption_speaker_evidence(caption, caption_index=index)
                except (ValueError, TypeError, KeyError):
                    reasons.append('invalid_caption_evidence'); continue
                result['caption_hints'].extend(hints)
                if implied_speaker_turns(caption, hints) > 1:
                    reasons.append('multiple_caption_turns_require_split')
                labels = {hint['label'].strip().casefold(): hint['label'] for hint in hints if hint.get('label')}
                observed_caption_labels.update(labels)
                if len(labels) > 1:
                    reasons.append('conflicting_caption_labels')
                for label in labels.values():
                    if cstart > start+1e-8 or cend < end-1e-8:
                        notes.append('caption_label_partial_interval'); continue
                    bound = self._bound_voices('caption_label', label)
                    if len(bound) > 1 or (bound and voice is not None and voice not in bound):
                        reasons.append('caption_audio_identity_conflict')
                    elif not bound:
                        notes.append('caption_label_unverified')
                        proposed_hints.append(('caption_label', label))
            if len(observed_caption_labels) > 1:
                reasons.append('conflicting_caption_labels')
            relevant = [row for row in visual_rows if abs(_number(row.get('start', -1))-start) <= 1e-8
                        and abs(_number(row.get('end', -1))-end) <= 1e-8]
            visual_candidates = []
            for row in relevant:
                if any(reason in row.get('reasons', []) for reason in
                       ('competing_active_faces', 'competing_near_scores', 'overlapping_speech_reported')):
                    reasons.append('conflicting_visual_activity')
                if row.get('state') == 'candidate' and row.get('face_id'):
                    supporting = [item for item in row.get('candidates', [])
                                  if item.get('face_id') == row['face_id'] and
                                  item.get('sufficient_support') is True and item.get('positive_activity') is True]
                    if len(supporting) != 1:
                        reasons.append('invalid_visual_activity_evidence')
                    elif not _meets_visual_floors(supporting[0]):
                        # A report built with looser overrides (e.g. a briefly
                        # visible face) may not name the face for the line.
                        notes.append('visual_support_below_default_floor')
                    else:
                        visual_candidates.append(_text(row['face_id']))
            if len(set(visual_candidates)) > 1:
                reasons.append('multiple_visual_candidates')
            elif visual_candidates:
                face = visual_candidates[0]
                visual_id = _text(face_identities.get(face, visual_scope+':'+face))
                result.update(visual_track=face, visual_identity=visual_id)
                bound = self._bound_voices('face_identity', visual_id)
                if len(bound) > 1 or (bound and voice is not None and voice not in bound):
                    reasons.append('visual_audio_identity_conflict')
                elif not bound:
                    notes.append('visual_identity_binding_unverified')
                    proposed_hints.append(('face_identity', visual_id))
            else:
                notes.append('visual_identity_unavailable_not_proven_offscreen')
            observation = None if local is None else hashlib.sha256(
                json.dumps(['acoustic-observation', audio_hash, local]).encode()).hexdigest()
            if voice is not None and observation in self.audio_assignments and self.audio_assignments[observation] != voice:
                reasons.append('audio_replay_identity_conflict')
            if voice is not None and voice not in self.voices and len(self.voices) >= self.max_voices:
                reasons.append('persistent_identity_capacity_reached')
            result['reasons'] = list(dict.fromkeys(reasons))
            result['notes'] = list(dict.fromkeys(notes))
            if result['reasons'] or voice is None:
                continue
            key = self._voice_key(voice)
            result.update(state='audio_identity_candidate', voice_id=voice, persistent_voice_key=key,
                          reasons=['persistent_audio_identity_supported'],
                          original_audio_policy='candidate_requires_independent_reference_and_dub_validation')
            if observation not in self.observations and len(self.observations) < self.max_observations:
                self.observations.add(observation)
                created_now.add(observation)
                self.audio_assignments[observation] = voice
                state = self.voices.setdefault(voice, dict(persistent_voice_key=key, observations=0,
                                                         first_seen=start, last_seen=end))
                state['observations'] += 1
            elif observation not in self.observations:
                result['notes'].append('evidence_capacity_reached_no_new_state_retained')
            # More cues of the same local speaker in this same call extend its
            # span and hints; a later replay of identical PCM adds nothing.
            if observation in created_now:
                state = self.voices[voice]
                state['first_seen'], state['last_seen'] = min(state['first_seen'], start), max(state['last_seen'], end)
                for kind, value in proposed_hints:
                    proposal = dict(kind=kind, value=value, voice_id=voice, observation=observation,
                                    independent_anchor=False)
                    if proposal not in self.proposals and len(self.proposals) < self.max_observations:
                        self.proposals.append(proposal)
        # Report-only coverage diagnostics: speech without any cue (e.g. a
        # missing subtitle line) and cue time without speech (e.g. an offset or
        # drifted subtitle edit). No threshold is applied; uncovered cues
        # already abstain individually.
        low, high = origin, origin+audio_duration
        speech = _union([[max(t['start'], low), min(t['end'], high)] for t in turns])
        cued = _union([[max(a, low), min(b, high)] for a, b in intervals if min(b, high) > max(a, low)])
        cue_spans = _union([list(span) for span in intervals])
        uncued = [gap for begin, stop in speech for gap in
                  _gaps(begin, stop, [[max(a, begin), min(b, stop)] for a, b in cued if b > begin and a < stop])]
        cue_seconds = sum(b-a for a, b in cue_spans)
        alignment = dict(cue_seconds=cue_seconds,
                         cue_seconds_without_audio=max(0., cue_seconds-_overlap_seconds(cue_spans, speech)),
                         audio_turn_seconds=sum(b-a for a, b in speech),
                         audio_seconds_without_cue=sum(b-a for a, b in uncued),
                         semantics='diagnostic only; offset, drift or missing-line decisions remain with the caller')
        return dict(evaluation_only=True, scope={'kind': 'movie', 'id': self.scope_id},
                    model_sha256=self.model_sha256, audio_sha256=audio_hash,
                    origin=origin, audio_duration=audio_duration, source_policy=policy,
                    identity_verified=False, clone_binding_ready=False, results=results,
                    uncued_audio_spans=uncued, cue_alignment=alignment,
                    summary=dict(input_cues=len(cues), output_cues=len(results),
                                 states=dict(Counter(row['state'] for row in results))))

    def snapshot(self):
        """Private JSON-compatible metadata; no waveforms, embeddings or names learned as truth."""
        return deepcopy(dict(version=CHECKPOINT_VERSION, scope_id=self.scope_id, model_sha256=self.model_sha256,
                             max_voices=self.max_voices, max_observations=self.max_observations,
                             voices=self.voices, observations=sorted(self.observations),
                             audio_assignments=self.audio_assignments,
                             bindings=self.bindings, proposals=self.proposals))

    def _restore(self, state):
        # Malformed checkpoints fail closed with ValueError, never a partial state.
        if not isinstance(state, dict):
            raise ValueError('Movie identity checkpoint must be a mapping')
        try:
            self._restore_checked(state)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError('Invalid movie identity checkpoint structure') from error

    def _restore_checked(self, state):
        if (state.get('version') != CHECKPOINT_VERSION or state.get('scope_id') != self.scope_id or
                state.get('model_sha256') != self.model_sha256 or state.get('max_voices') != self.max_voices or
                state.get('max_observations') != self.max_observations):
            raise ValueError('Movie identity checkpoint scope/model/policy mismatch')
        voices, observations = state.get('voices'), state.get('observations')
        bindings, proposals = state.get('bindings'), state.get('proposals')
        assignments = state.get('audio_assignments')
        if (not isinstance(voices, dict) or len(voices) > self.max_voices or
                not isinstance(observations, list) or len(observations) > self.max_observations or
                len(set(observations)) != len(observations) or
                any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value) for value in observations) or
                not isinstance(bindings, list) or len(bindings) > 512 or
                not isinstance(assignments, dict) or set(assignments) != set(observations) or
                any(voice not in voices for voice in assignments.values()) or
                not isinstance(proposals, list) or len(proposals) > self.max_observations):
            raise ValueError('Invalid bounded movie identity checkpoint')
        for voice, value in voices.items():
            _text(voice, 80)
            _interval(value['first_seen'], value['last_seen'])
            if (value['persistent_voice_key'] != self._voice_key(voice) or isinstance(value['observations'], bool)
                    or not isinstance(value['observations'], int) or not 1 <= value['observations'] <= len(observations)):
                raise ValueError('Invalid persistent voice checkpoint')
        if sum(value['observations'] for value in voices.values()) != len(observations):
            raise ValueError('Checkpoint observation counts disagree')
        if any(sum(value == voice for value in assignments.values()) != info['observations'] for voice, info in voices.items()):
            raise ValueError('Checkpoint acoustic assignment counts disagree')
        self.voices, self.observations = deepcopy(voices), set(observations)
        self.audio_assignments = dict(assignments)
        for binding in bindings:
            self.add_binding(**binding)
        for proposal in proposals:
            if (proposal.get('kind') not in ('caption_label', 'face_identity') or
                    proposal.get('voice_id') not in voices or proposal.get('observation') not in self.observations or
                    assignments[proposal['observation']] != proposal['voice_id'] or
                    proposal.get('independent_anchor') is not False):
                raise ValueError('Invalid unverified hint proposal')
            _text(proposal['value'])
        self.proposals = deepcopy(proposals)
