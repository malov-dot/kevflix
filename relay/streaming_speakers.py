"""Staged Diart adapter. Importing this module does not load a model.

The caller supplies offline Diart model wrappers and explicitly opts in. Stream
identities are provisional acoustic clusters, never accepted actor references.
See docs/SPEAKER-STREAMING-REVIEW.md for integration and validation boundaries.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class StreamingSpeakerOptions:
    enabled: bool = False
    sample_rate: int = 16000
    duration: float = 5.0
    step: float = 0.5
    activity: float = 0.6
    similarity: float = 0.70
    margin: float = 0.10
    new_speaker_similarity: float = 0.45
    minimum_speech_seconds: float = 1.5
    minimum_solo_seconds: float = 1.0
    max_speakers: int = 64

    def validate(self):
        if self.enabled is not True:
            raise ValueError('Incremental speaker tracking requires explicit opt-in')
        numbers = [v for k, v in asdict(self).items() if k != 'enabled']
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError('Speaker tracking options must be finite')
        if self.sample_rate != 16000 or not 0 < self.step <= self.duration:
            raise ValueError('Expected 16 kHz audio and 0 < step <= duration')
        if not 0 < self.activity < 1 or not 0 <= self.new_speaker_similarity < self.similarity <= 1:
            raise ValueError('Invalid speaker activity or similarity thresholds')
        if not 0 < self.margin <= 2:
            raise ValueError('Speaker margin must be positive')
        if not 0 < self.minimum_solo_seconds <= self.minimum_speech_seconds <= self.duration:
            raise ValueError('Invalid minimum speaker evidence duration')
        if not isinstance(self.max_speakers, int) or not 1 <= self.max_speakers <= 512:
            raise ValueError('max_speakers must be an integer between 1 and 512')


class IncrementalSpeakerAnalyzer:
    """One instance per movie job and encoder revision; CPU only.

    analyze_window returns a *replacement observation* for its full window,
    not append-only finalized turns. It preserves all local speaker tracks,
    including unresolved speech. No output authorizes synthesis by itself.
    """
    def __init__(self, *, segmentation_model, embedding_model, stream_id,
                 encoder_id, options=None, state=None):
        self.options = options or StreamingSpeakerOptions()
        self.options.validate()
        if not stream_id or not encoder_id:
            raise ValueError('A movie-specific stream_id and exact encoder_id are required')
        if segmentation_model is None or embedding_model is None:
            raise ValueError('Supply both offline model wrappers; there are no remote defaults')
        for model in (segmentation_model, embedding_model):
            loaded = getattr(model, 'is_in_memory', None)
            if not callable(loaded) or loaded() is not True:
                raise ValueError('Supply already-loaded Diart wrappers; lazy remote loading is refused')
        import numpy as np
        import torch
        from diart.blocks.clustering import OnlineSpeakerClustering
        from diart.blocks.embedding import OverlapAwareSpeakerEmbedding
        from diart.blocks.segmentation import SpeakerSegmentation
        from diart.blocks.utils import Binarize

        self.np, self.torch = np, torch
        self.stream_id, self.encoder_id = str(stream_id), str(encoder_id)
        cpu = torch.device('cpu')
        self.segmentation = SpeakerSegmentation(segmentation_model, device=cpu)
        self.embedding = OverlapAwareSpeakerEmbedding(embedding_model, device=cpu)
        self.binarize = Binarize(self.options.activity)
        # Reuse upstream centroid storage, but never identify/__call__: these
        # contain a closest-free-center fallback for short unknown voices.
        self.clusters = OnlineSpeakerClustering(
            tau_active=self.options.activity, rho_update=0.3,
            delta_new=1-self.options.similarity,
            max_speakers=self.options.max_speakers)
        self.sequence = 0
        self.last_offset = None
        self.updated_until = {}
        if state is not None:
            self._restore(state)

    def _global_id(self, center):
        return f'{self.stream_id}:stream-speaker-{center+1:03d}'

    def _scores(self, vector):
        np = self.np
        result = []
        for center in sorted(self.clusters.active_centers):
            prototype = self.clusters.centers[center]
            norm = np.linalg.norm(prototype)
            if norm > 1e-8:
                result.append((float(np.dot(vector, prototype / norm)), center))
        return sorted(result, reverse=True)

    def _assign(self, scores, vectors, offset):
        np, opts = self.np, self.options
        active = scores > opts.activity
        frame_seconds = opts.duration / len(scores)
        solo = active.sum(axis=1) == 1
        decisions, proposals = {}, {}
        for local in range(scores.shape[1]):
            speech = float(active[:, local].sum() * frame_seconds)
            if not speech:
                continue
            solo_seconds = float((active[:, local] & solo).sum() * frame_seconds)
            decision = dict(local_speaker=local, speaker=None, speech_seconds=speech,
                            solo_seconds=solo_seconds, reason='invalid_embedding', candidates=[])
            decisions[local] = decision
            vector = vectors[local]
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or not math.isfinite(float(norm)) or norm <= 1e-8:
                continue
            vectors[local] = vector / norm
            ranking = self._scores(vectors[local])
            decision['candidates'] = [(score, self._global_id(center)) for score, center in ranking[:3]]
            best = ranking[0][0] if ranking else -1.0
            margin = best - ranking[1][0] if len(ranking) > 1 else 2.0
            decision['margin'] = margin
            # A short or fully overlapping utterance remains unresolved even
            # when its embedding happens to resemble a known actor.
            if speech < opts.minimum_speech_seconds or solo_seconds < opts.minimum_solo_seconds:
                decision['reason'] = 'insufficient_clean_speech'
            elif best >= opts.similarity and margin >= opts.margin:
                proposals[local] = ranking[0][1]
                decision['reason'] = 'candidate_match'
            elif best < opts.new_speaker_similarity:
                decision['reason'] = 'candidate_new'
            else:
                decision['reason'] = 'ambiguous_embedding'

        counts = Counter(proposals.values())
        new_locals = [local for local, item in decisions.items() if item['reason'] == 'candidate_new']
        # Do not use iteration order to pick among indistinguishable local
        # tracks; two local speakers cannot claim the same global identity.
        new_conflicts = {local for local in new_locals for other in new_locals
                         if local != other and float(np.dot(vectors[local], vectors[other]))
                         >= opts.new_speaker_similarity}
        for local, item in decisions.items():
            center = proposals.get(local)
            if center is not None:
                if counts[center] != 1:
                    item['reason'] = 'local_identity_collision'
                    continue
                item['reason'] = 'matched_cluster'
                # Rolling windows overlap. Update a center only after its last
                # contributing window has ended; do not repeatedly count it.
                if offset >= self.updated_until.get(center, float('-inf')) - 1e-6:
                    self.clusters.update([(local, center)], vectors)
                    self.updated_until[center] = offset + opts.duration
            elif item['reason'] == 'candidate_new':
                if local in new_conflicts:
                    item['reason'] = 'new_identity_collision'
                    continue
                if self.clusters.num_free_centers <= 0:
                    item['reason'] = 'speaker_capacity_exhausted'
                    continue
                center = self.clusters.add_center(vectors[local])
                self.updated_until[center] = offset + opts.duration
                item['reason'] = 'new_provisional_cluster'
            else:
                continue
            item['speaker'] = self._global_id(center)
        return decisions

    def analyze_window(self, waveform, *, sample_rate, offset):
        """Accept consecutive overlapping mono windows; do not pad final audio.

        Caller retains/reconciles observations until its commit horizon and
        blocks synthesis anywhere speaker=None or overlapping tracks occur.
        """
        np, torch, opts = self.np, self.torch, self.options
        from pyannote.core import SlidingWindow, SlidingWindowFeature
        if sample_rate != opts.sample_rate or not math.isfinite(offset) or offset < 0:
            raise ValueError('Expected 16 kHz audio and a finite nonnegative offset')
        if self.last_offset is not None and not math.isclose(offset-self.last_offset, opts.step, abs_tol=1e-6):
            raise ValueError('Windows must advance by exactly step; restore the matching state after a seek')
        data = np.asarray(waveform, dtype=np.float32)
        if data.ndim != 1 or len(data) != round(opts.duration * sample_rate) or not np.isfinite(data).all():
            raise ValueError('Expected one complete finite mono window; padding is not accepted')
        batch = torch.from_numpy(data.copy())[None, :, None]
        with torch.inference_mode():
            segmentation = self.segmentation(batch)
            embeddings = self.embedding(batch, segmentation)
        scores = np.asarray(segmentation.detach().cpu(), dtype=np.float32)
        if scores.ndim != 3 or scores.shape[0] != 1 or not scores.shape[1] or not scores.shape[2]:
            raise ValueError('Unexpected Diart segmentation shape')
        scores = scores[0]
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('Expected finite per-speaker probabilities; verify powerset conversion')
        vectors = np.asarray(embeddings.detach().cpu(), dtype=np.float64)
        # Diart squeezes singleton batch and speaker dimensions.
        if vectors.ndim == 3 and vectors.shape[0] == 1:
            vectors = vectors[0]
        if vectors.ndim == 1 and scores.shape[1] == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or vectors.shape[0] != scores.shape[1] or not vectors.shape[1]:
            raise ValueError('Unexpected Diart embedding shape')
        if self.clusters.centers is None:
            self.clusters.init_centers(vectors.shape[1])
        elif self.clusters.centers.shape[1] != vectors.shape[1]:
            raise ValueError('Speaker embedding dimension changed within a stream')
        decisions = self._assign(scores, vectors.copy(), offset)
        resolution = opts.duration / len(scores)
        feature = SlidingWindowFeature(scores, SlidingWindow(start=offset, duration=resolution, step=resolution))
        annotation = self.binarize(feature)
        turns = []
        for segment, track, _label in annotation.itertracks(yield_label=True):
            local = int(track)
            # Reviewed Binarize also uses > activity. If a future upstream
            # revision emits a threshold-only track, retain it as unresolved
            # instead of either inventing an identity or raising KeyError.
            decision = decisions.setdefault(local, dict(
                local_speaker=local, speaker=None,
                reason='segmentation_boundary_mismatch', candidates=[]))
            begin, end = max(offset, float(segment.start)), min(offset+opts.duration, float(segment.end))
            if end > begin:
                turns.append(dict(start=begin, end=end, speaker=decision['speaker'],
                                  local_speaker=local, reason=decision['reason']))
        self.sequence += 1
        self.last_offset = float(offset)
        return dict(stream_id=self.stream_id, encoder_id=self.encoder_id,
                    sequence=self.sequence, start=offset, end=offset+opts.duration,
                    mode='replace_window_observation', finalized=False,
                    turns=turns, decisions=list(decisions.values()))

    def snapshot(self):
        """JSON-compatible private checkpoint; excludes audio and actor references."""
        centers = self.clusters.centers
        return dict(version=1, stream_id=self.stream_id, encoder_id=self.encoder_id,
                    options=asdict(self.options), sequence=self.sequence, last_offset=self.last_offset,
                    centers=[] if centers is None else [centers[c].tolist() for c in sorted(self.clusters.active_centers)],
                    updated_until={str(k): v for k, v in self.updated_until.items()})

    def _restore(self, state):
        np = self.np
        if (state.get('version') != 1 or state.get('stream_id') != self.stream_id or
                state.get('encoder_id') != self.encoder_id or state.get('options') != asdict(self.options)):
            raise ValueError('Speaker checkpoint stream, encoder or options mismatch')
        centers = state.get('centers')
        sequence, offset = state.get('sequence'), state.get('last_offset')
        if not isinstance(sequence, int) or sequence < 0 or not isinstance(centers, list):
            raise ValueError('Invalid speaker checkpoint')
        if (sequence == 0 and offset is not None) or (sequence > 0 and (
                not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0)):
            raise ValueError('Invalid checkpoint timeline')
        if centers:
            matrix = np.asarray(centers, dtype=np.float64)
            if (matrix.ndim != 2 or not matrix.shape[1] or len(matrix) > self.options.max_speakers or
                    not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) <= 1e-8).any()):
                raise ValueError('Invalid speaker checkpoint centers')
            self.clusters.init_centers(matrix.shape[1])
            for vector in matrix:
                self.clusters.add_center(vector)
        updates = state.get('updated_until', {})
        if set(updates) != {str(i) for i in range(len(centers))}:
            raise ValueError('Invalid checkpoint update positions')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in updates.values()):
            raise ValueError('Invalid checkpoint update times')
        self.updated_until = {int(k): float(v) for k, v in updates.items()}
        self.sequence, self.last_offset = sequence, offset


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


class CommitHorizonReconciler:
    """Pure (Diart/numpy-free) reconciliation of overlapping window revisions.

    Feed consecutive ``analyze_window`` results. The commit horizon is the
    next window's start (last start + step): no later window can revise time
    before it, so only that time is finalized. A span keeps a cluster ID only
    when every window that covered it reported exactly one track there with
    that same non-None ID. Any unresolved track, overlap, speaker disagreement
    or missing activity in any covering window keeps the speech as
    ``speaker=None`` with reasons; a later window emitting fewer tracks never
    deletes earlier speech. Segments are cut at each horizon; consumers may
    join adjacent identical pieces. IDs remain acoustic clusters, not actors.
    """
    MAX_TURNS = 1024

    def __init__(self, *, stream_id, encoder_id, duration, step, state=None):
        if not isinstance(stream_id, str) or not stream_id or not isinstance(encoder_id, str) or not encoder_id:
            raise ValueError('A movie-specific stream_id and exact encoder_id are required')
        self.stream_id, self.encoder_id = stream_id, encoder_id
        self.duration, self.step = _finite(duration, 'duration'), _finite(step, 'step')
        if not 0 < self.step <= self.duration:
            raise ValueError('Expected 0 < step <= duration')
        self.windows, self.committed_until = [], None
        self.last_start = self.last_sequence = None
        self.finished = False
        if state is not None:
            self._restore(state)

    def _validated(self, observation):
        if not isinstance(observation, dict):
            raise ValueError('Window observation must be a mapping')
        if (observation.get('stream_id') != self.stream_id or observation.get('encoder_id') != self.encoder_id
                or observation.get('mode') != 'replace_window_observation' or observation.get('finalized') is not False):
            raise ValueError('Observation stream, encoder or mode mismatch')
        sequence = observation.get('sequence')
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError('Invalid observation sequence')
        start, end = _finite(observation.get('start'), 'start'), _finite(observation.get('end'), 'end')
        if start < 0 or not math.isclose(end-start, self.duration, abs_tol=1e-6):
            raise ValueError('Observation does not span one full window')
        if self.last_start is not None and (sequence != self.last_sequence+1 or
                                            not math.isclose(start-self.last_start, self.step, abs_tol=1e-6)):
            raise ValueError('Observations must be consecutive and advance by exactly step')
        turns = observation.get('turns')
        if not isinstance(turns, list) or len(turns) > self.MAX_TURNS:
            raise ValueError('Invalid bounded window turns')
        rows = []
        for turn in turns:
            a, b = _finite(turn.get('start'), 'turn start'), _finite(turn.get('end'), 'turn end')
            speaker, local, reason = turn.get('speaker'), turn.get('local_speaker'), turn.get('reason')
            if not start-1e-6 <= a < b <= end+1e-6:
                raise ValueError('Turn lies outside its window or is empty')
            if speaker is not None and (not isinstance(speaker, str) or not speaker):
                raise ValueError('Turn speaker must be a cluster ID or None')
            if isinstance(local, bool) or not isinstance(local, (int, str)):
                raise ValueError('Turn local_speaker must be an int or string')
            if reason is not None and not isinstance(reason, str):
                raise ValueError('Turn reason must be a string')
            rows.append([a, b, local, speaker, reason])
        return dict(sequence=sequence, start=start, end=end, turns=rows)

    def add(self, observation):
        """Retain one window revision; return newly finalized segments."""
        if self.finished:
            raise ValueError('Stream already finished; start a new reconciler')
        window = self._validated(observation)
        self.windows.append(window)
        self.last_start, self.last_sequence = window['start'], window['sequence']
        if self.committed_until is None:
            self.committed_until = window['start']
        return self._finalize(window['start']+self.step)

    def finish(self):
        """End of stream: finalize through the last window's end."""
        if self.finished or not self.windows:
            self.finished = True
            return []
        self.finished = True
        return self._finalize(self.windows[-1]['end'])

    def _finalize(self, horizon):
        low = self.committed_until
        if horizon <= low+1e-9:
            return []
        points = {round(low, 6), round(horizon, 6)}
        for window in self.windows:
            points.update(round(t, 6) for t in (window['start'], window['end']) if low < t < horizon)
            for a, b, *_ in window['turns']:
                points.update(round(t, 6) for t in (a, b) if low < t < horizon)
        points = sorted(points)
        segments = []
        for a, b in zip(points, points[1:]):
            if b-a <= 1e-9:
                continue
            middle, reasons, speakers, covering, any_speech = (a+b)/2, set(), set(), 0, False
            for window in self.windows:
                if not window['start'] <= middle < window['end']:
                    continue
                covering += 1
                tracks = [turn for turn in window['turns'] if turn[0] <= middle < turn[1]]
                if not tracks:
                    reasons.add('activity_disagreement')
                    continue
                any_speech = True
                if len({str(turn[2]) for turn in tracks}) > 1:
                    reasons.add('overlapping_tracks')
                for _a, _b, _local, speaker, reason in tracks:
                    if speaker is None:
                        reasons.add('unresolved:'+(reason or 'unknown'))
                    else:
                        speakers.add(speaker)
            if not any_speech:
                continue
            if len(speakers) > 1:
                reasons.add('speaker_disagreement')
            speaker = next(iter(speakers)) if len(speakers) == 1 and not reasons else None
            piece = dict(stream_id=self.stream_id, start=a, end=b, speaker=speaker, reasons=sorted(reasons),
                         windows=covering, finalized=True, acoustic_cluster_only=True,
                         identity_verified=False, clone_binding_ready=False)
            last = segments[-1] if segments else None
            if (last and abs(last['end']-a) <= 1e-9 and last['speaker'] == speaker
                    and last['reasons'] == piece['reasons']):
                last['end'], last['windows'] = b, min(last['windows'], covering)
            else:
                segments.append(piece)
        self.committed_until = horizon
        self.windows = [window for window in self.windows if window['end'] > horizon+1e-9]
        return segments

    def snapshot(self):
        """JSON-compatible private state (cluster IDs and times only; no audio)."""
        return dict(version=1, kind='commit-horizon-reconciler', stream_id=self.stream_id,
                    encoder_id=self.encoder_id, duration=self.duration, step=self.step,
                    committed_until=self.committed_until, last_start=self.last_start,
                    last_sequence=self.last_sequence, finished=self.finished,
                    windows=[dict(sequence=w['sequence'], start=w['start'], end=w['end'],
                                  turns=[list(turn) for turn in w['turns']]) for w in self.windows])

    def _restore(self, state):
        if not isinstance(state, dict):
            raise ValueError('Reconciler checkpoint must be a mapping')
        if (state.get('version') != 1 or state.get('kind') != 'commit-horizon-reconciler' or
                state.get('stream_id') != self.stream_id or state.get('encoder_id') != self.encoder_id or
                state.get('duration') != self.duration or state.get('step') != self.step):
            raise ValueError('Reconciler checkpoint stream, encoder or timing mismatch')
        windows, finished = state.get('windows'), state.get('finished')
        if not isinstance(windows, list) or not isinstance(finished, bool):
            raise ValueError('Invalid reconciler checkpoint')
        committed = state.get('committed_until')
        if committed is None:
            if windows or state.get('last_start') is not None or state.get('last_sequence') is not None:
                raise ValueError('Invalid empty reconciler checkpoint')
            self.finished = finished
            return
        committed = _finite(committed, 'committed_until')
        if committed < 0:
            raise ValueError('Invalid reconciler commit position')
        probe = CommitHorizonReconciler(stream_id=self.stream_id, encoder_id=self.encoder_id,
                                        duration=self.duration, step=self.step)
        restored = []
        try:
            for window in windows:
                rows = [dict(start=t[0], end=t[1], local_speaker=t[2], speaker=t[3], reason=t[4])
                        for t in window['turns']]
                restored.append(probe._validated(dict(window, turns=rows, stream_id=self.stream_id,
                                                      encoder_id=self.encoder_id, finalized=False,
                                                      mode='replace_window_observation')))
                probe.last_start, probe.last_sequence = restored[-1]['start'], restored[-1]['sequence']
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError('Invalid reconciler checkpoint windows') from error
        last_start, last_sequence = state.get('last_start'), state.get('last_sequence')
        if (isinstance(last_sequence, bool) or not isinstance(last_sequence, int) or
                _finite(last_start, 'last_start') < 0 or
                (restored and (restored[-1]['start'] != last_start or restored[-1]['sequence'] != last_sequence)) or
                committed > last_start+(self.duration if finished else self.step)+1e-6 or
                any(window['end'] <= committed+1e-9 for window in restored)):
            raise ValueError('Inconsistent reconciler checkpoint timeline')
        self.windows, self.committed_until = restored, committed
        self.last_start, self.last_sequence, self.finished = float(last_start), last_sequence, finished
