"""CPU numerical CMIA adapter; imports no inference framework or model.

Upstream numerical source is hash-checked and executed unchanged. Feature
extraction belongs to the caller. Results are candidates, not actor acceptance.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import math
from pathlib import Path
import pickle
from types import ModuleType
import warnings


SOURCE_REVISION = '1cc3f1632222ea77d88da08b03dcce9003d3a55e'
MAX_FACE_SAMPLE_GAP_SECONDS = 0.10
SOURCE_HASHES = {
    'src/ASD/asd_utils.py': '26783a9b1c8f10eca01aeef8e6b9cce65cf061e504db1936d8ccaed0de185a9f',
    'src/ASD/speech_face_association.py': 'c1da072a61de2650193da7e25a6b023d713faad722a6b481f249fb262019f6c9',
}


def _face_support(track, candidate, speech, frame_cells=None):
    """Conservative sampled-track support; do not bridge a cut or lost face.

    No evidence is extrapolated before the first/after the last sample. Sparse
    tracks with gaps above 100 ms need a separately reviewed timing contract.
    """
    begin = max(candidate[1], speech[0])
    end = min(candidate[2], speech[1])
    spans = []
    if frame_cells is None:
        intervals = [(left[0], right[0]) for left, right in zip(track, track[1:])
                     if 0 < right[0]-left[0] <= MAX_FACE_SAMPLE_GAP_SECONDS+1e-9]
    else:
        intervals = [(cell['start'], cell['end']) for cell in frame_cells]
    for left, right in intervals:
        start, stop = max(begin, left), min(end, right)
        if stop <= start:
            continue
        if spans and start <= spans[-1][1] + 1e-9:
            spans[-1][1] = max(spans[-1][1], stop)
        else:
            spans.append([start, stop])
    return spans


def _validated_frame_cells(frame_cells, face_tracks):
    """Validate explicit presentation intervals against observed face rows.

    Timing provenance (for example hashed FFprobe PTS output) belongs to the
    caller. There is no inferred final-frame duration or fill across a gap.
    """
    if frame_cells is None:
        return None
    if not isinstance(frame_cells, dict):
        raise ValueError('frame_cells must map face IDs to presentation intervals')
    result = {}
    for face, cells in frame_cells.items():
        if face not in face_tracks or not isinstance(cells, (list, tuple)):
            raise ValueError('Frame cells require a known face track and interval list')
        try:
            timestamps = [float(row[0]) for row in face_tracks[face]]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError('Invalid frame-cell face track timestamps') from exc
        validated, used = [], set()
        for cell in cells:
            try:
                if not isinstance(cell, dict):
                    raise ValueError('Invalid cell')
                start, end = float(cell['start']), float(cell['end'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Frame cell must contain numeric start and end') from exc
            if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or
                    not 0 < end-start <= MAX_FACE_SAMPLE_GAP_SECONDS+1e-9):
                raise ValueError('Frame cell must have a finite positive duration no greater than 0.10 seconds')
            matches = [index for index, stamp in enumerate(timestamps) if abs(stamp-start) <= 1e-8]
            if len(matches) != 1 or matches[0] in used:
                raise ValueError('Each frame cell must match one distinct observed face-row timestamp')
            next_rows = [stamp for stamp in timestamps if stamp > start+1e-8]
            if next_rows and end > min(next_rows)+1e-8:
                raise ValueError('Frame cell extends across a later observed presentation timestamp')
            used.add(matches[0])
            if validated and start < validated[-1]['end']-1e-9:
                raise ValueError('Frame cells must be ordered and nonoverlapping')
            validated.append(dict(start=start, end=end))
        result[face] = validated
    return result


def _uncovered_spans(speech, support):
    cursor, end = speech
    gaps = []
    for begin, stop in support:
        if begin > cursor + 1e-9:
            gaps.append([cursor, begin])
        cursor = max(cursor, stop)
    if cursor < end - 1e-9:
        gaps.append([cursor, end])
    return gaps


def _read_sources(upstream_root):
    root = Path(upstream_root).resolve(strict=True)
    sources = {}
    for relative, digest in SOURCE_HASHES.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('CMIA source must remain inside the supplied checkout')
        source = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError(f'CMIA source differs from reviewed revision: {relative}')
        sources[relative] = (path, source)
    return sources


def _load_core(sources, cache):
    """Use module-local import shims; never replace sys.modules or sys.path."""
    helper = ModuleType('local_utils')
    cache_pickle = ModuleType('pickle')
    allowed_output = cache / 'faceDistancesAll.pkl'

    def write_pickle(value, filename):
        if Path(filename).resolve() != allowed_output:
            raise ValueError('Unexpected upstream cache write')
        with allowed_output.open('xb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def refuse_pickle_read(*args, **kwargs):
        raise ValueError('Reading upstream pickle caches is prohibited')

    helper.writeToPickleFile = write_pickle
    cache_pickle.load = refuse_pickle_read
    normal_import = builtins.__import__

    def core_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == 'local_utils':
            return helper
        if level == 0 and name == 'pickle':
            return cache_pickle
        return normal_import(name, globals, locals, fromlist, level)

    modules = []
    for index, (path, source) in enumerate(sources.values()):
        spec = importlib.util.spec_from_file_location(f'_kevflix_cmia_core_{index}', path)
        if spec is None or spec.loader is None:
            raise ValueError('Cannot create a CMIA source module')
        module = importlib.util.module_from_spec(spec)
        module.__dict__['__builtins__'] = {**vars(builtins), '__import__': core_import}
        # Execute exactly the bytes checked above, avoiding a second file read.
        exec(compile(source, str(path), 'exec'), module.__dict__)
        modules.append(module)
    return modules


def associate_movie_speakers(upstream_root, cache_dir, speech_face_tracks,
                             face_tracks, face_features, speech_features, guides=None, *, frame_cells=None):
    """Associate supplied real speech/face features using pinned upstream CMIA.

    cache_dir must not exist. Matrices execute on CPU through NumPy/SciPy;
    process priority/thread caps remain the caller's responsibility. Returns a
    JSON-compatible report. Input/identity failures remain unresolved. Path,
    revision, malformed guide and unexpected upstream failures raise exceptions.
    The caller must preserve speech omitted from speech_face_tracks itself.
    """
    import numpy as np

    sources = _read_sources(upstream_root)
    for name, value in [('speech_face_tracks', speech_face_tracks), ('face_tracks', face_tracks),
                        ('face_features', face_features), ('speech_features', speech_features)]:
        if not isinstance(value, dict) or any(not isinstance(key, str) or not key for key in value):
            raise ValueError(f'{name} must be a dictionary with nonempty string IDs')
    frame_cells = _validated_frame_cells(frame_cells, face_tracks)
    cache = Path(cache_dir).absolute()
    # No prior cache is trusted, even if it appears empty. Atomic mkdir refuses
    # a concurrent claimant; the pickle shim also forbids all upstream reads.
    cache.mkdir(parents=True, exist_ok=False)
    cache = cache.resolve(strict=True)
    report = dict(source_revision=SOURCE_REVISION, source_hashes=dict(SOURCE_HASHES),
                  status='unresolved', candidate_only=True, device='cpu', optimizer_ran=False,
                  associations={}, candidate_associations={}, association_spans=[],
                  coverage={}, unresolved_spans=[],
                  correlations={key: None for key in speech_face_tracks},
                  excluded=[], unresolved=[], warnings=[],
                  adaptations=['writer-only local_utils import shim; no global module changes',
                               'upstream pickle reads prohibited; exclusive cache writes',
                               'direct optimizer; marginal-neighbor refinement omitted',
                               'invalid correlation matrices score negative infinity',
                               'whole-cue associations require complete sampled-face coverage'],
                  temporal_support_policy=dict(maximum_sample_gap_seconds=MAX_FACE_SAMPLE_GAP_SECONDS,
                                               extrapolate_track_ends=False,
                                               explicit_frame_cells=frame_cells is not None),
                  cache_dir=str(cache), input_speech_count=len(speech_face_tracks))
    unresolved = {}
    speech, vectors_speech = {}, {}

    def reject_speech(key, reason):
        unresolved[key] = reason
        report['excluded'].append(dict(kind='speech', id=key, reason=reason))

    def finish(reason=None):
        report['candidate_associations'] = dict(report['associations'])
        for key in speech_face_tracks:
            item = speech_face_tracks[key]
            times = interval(item.get('speech')) if isinstance(item, dict) else None
            face = report['candidate_associations'].get(key)
            support = []
            if face is not None:
                candidate = next(c for c in speech[key]['face_tracks'] if c[0] == face)
                cells = None if frame_cells is None else frame_cells.get(face, [])
                support = _face_support(valid_faces[face], candidate, times, cells)
            if times is None:
                unresolved.setdefault(key, reason or 'invalid_speech_interval')
                continue
            gaps = _uncovered_spans(times, support)
            supported = sum(stop-start for start, stop in support)
            report['coverage'][key] = dict(
                candidate_face=face, speech_start=times[0], speech_end=times[1],
                speech_seconds=times[1]-times[0], supported_seconds=supported,
                supported_fraction=supported/(times[1]-times[0]),
                full_temporal_support=bool(face is not None and not gaps),
                supported_spans=support, uncovered_spans=gaps)
            for start, stop in support:
                report['association_spans'].append(dict(
                    speech_id=key, face_id=face, start=start, end=stop, candidate_only=True))
            if gaps:
                report['associations'].pop(key, None)
                unresolved.setdefault(key, 'partial_face_visibility' if face is not None
                                      else reason or 'no_candidate_association')
                for start, stop in gaps:
                    report['unresolved_spans'].append(dict(
                        speech_id=key, start=start, end=stop, reason=unresolved[key]))
        report['unresolved'] = [dict(speech_id=k, reason=v) for k, v in unresolved.items()]
        report['complete_association_count'] = len(report['associations'])
        report['status'] = 'candidates' if report['candidate_associations'] else 'unresolved'
        return report

    def valid_vector(value):
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            return None
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-8:
            return None
        return vector.copy() / norm

    def interval(value):
        try:
            if len(value) != 2:
                return None
            start, end = map(float, value)
        except (TypeError, ValueError):
            return None
        return (start, end) if math.isfinite(start) and math.isfinite(end) and 0 <= start < end else None

    valid_faces, vectors_faces = {}, {}
    for key, track in face_tracks.items():
        vector = valid_vector(face_features.get(key))
        reason = None
        boxes = []
        try:
            for box in track:
                values = list(map(float, box[:5]))
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError('invalid box')
                stamp, x1, y1, x2, y2 = values
                if stamp < 0 or not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                    raise ValueError('invalid box')
                boxes.append(values)
        except (TypeError, ValueError):
            reason = 'invalid_face_track'
        if not boxes or any(b[0] < a[0] for a, b in zip(boxes, boxes[1:])):
            reason = 'invalid_face_track'
        if vector is None:
            reason = 'invalid_face_embedding'
        if reason:
            report['excluded'].append(dict(kind='face', id=key, reason=reason))
        else:
            valid_faces[key], vectors_faces[key] = boxes, vector
    if len({len(v) for v in vectors_faces.values()}) > 1:
        return finish('inconsistent_face_embedding_dimensions')

    for key, item in speech_face_tracks.items():
        times = interval(item.get('speech')) if isinstance(item, dict) else None
        vector = valid_vector(speech_features.get(key))
        if times is None or vector is None:
            reject_speech(key, 'invalid_speech_interval' if times is None else 'invalid_speech_embedding')
            continue
        candidates, seen = [], set()
        supplied = item.get('face_tracks', [])
        if not isinstance(supplied, (list, tuple)):
            reject_speech(key, 'invalid_candidate_list')
            continue
        for candidate in supplied:
            if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
                continue
            face, begin, end = candidate
            if not isinstance(face, str) or face not in valid_faces or face in seen:
                continue
            overlap = interval([begin, end])
            if overlap is None:
                continue
            track = valid_faces[face]
            cells = None if frame_cells is None else frame_cells.get(face, [])
            if cells is not None and not cells:
                continue
            track_begin = track[0][0] if cells is None else cells[0]['start']
            track_end = track[-1][0] if cells is None else cells[-1]['end']
            start = max(times[0], overlap[0], track_begin)
            stop = min(times[1], overlap[1], track_end)
            if stop > start:
                candidates.append([face, start, stop])
                seen.add(face)
        if not candidates:
            reject_speech(key, 'no_usable_overlapping_face')
            continue
        speech[key] = dict(speech=list(times), face_tracks=candidates)
        vectors_speech[key] = vector
    report['eligible_speech_count'] = len(speech)
    if len(speech) < 3:
        return finish('insufficient_observations')
    if len({len(v) for v in vectors_speech.values()}) > 1:
        return finish('inconsistent_speech_embedding_dimensions')

    sanitized_guides = None
    if guides is not None:
        try:
            positive, negative = float(guides['posTh']), float(guides['negTh'])
            if not math.isfinite(positive) or not math.isfinite(negative) or negative > positive:
                raise ValueError('invalid thresholds')
            guide_scores = {}
            for key, item in speech.items():
                guide_scores[key] = {}
                for face, _, _ in item['face_tracks']:
                    score = float(guides['scores'][key][face])
                    if not math.isfinite(score):
                        raise ValueError('nonfinite guide')
                    guide_scores[key][face] = score
            sanitized_guides = dict(scores=guide_scores, posTh=positive, negTh=negative)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Guides require finite thresholds and scores for every eligible candidate') from exc

    utils, core = _load_core(sources, cache)
    base_similarity = utils.Similarity(measure='correlation')

    def usable_matrix(matrix):
        values = np.asarray(matrix)
        return (values.ndim == 2 and values.shape[0] == values.shape[1] and
                len(values) >= 3 and np.isfinite(values).all() and
                bool((np.ptp(values, axis=1) > 1e-10).all()))

    class GuardedSimilarity:
        def computeAvgSimilarity(self, audioDistances, faceDistances, avg=True):
            if not usable_matrix(audioDistances) or not usable_matrix(faceDistances):
                return float('-inf') if avg else [float('nan')] * len(audioDistances)
            result = base_similarity.computeAvgSimilarity(audioDistances, faceDistances, avg=avg)
            if avg and not math.isfinite(float(result)):
                return float('-inf')
            return result

    distances = utils.Distances(vectors_faces, vectors_speech, str(cache), verbose=False)
    similarity = GuardedSimilarity()
    association = core.SpeechFaceAssociation(str(cache), speech, {}, similarity,
                                             distances, valid_faces, sanitized_guides, verbose=False)
    initial, positive, negative = association.initializeASD(list(speech))
    for key in speech:
        if key not in initial:
            unresolved[key] = 'excluded_by_guides'
    initial = {key: face for key, face in initial.items()
               if any(candidate[0] not in negative[key] for candidate in speech[key]['face_tracks'])}
    if len(initial) < 3:
        return finish('insufficient_observations_after_guides')
    keys = list(initial)
    audio_matrix = distances.computeDistanceMatrix(keys, modality='speech')
    face_matrix = distances.computeDistanceMatrix(keys, asd=initial, modality='face')
    if not usable_matrix(audio_matrix) or not usable_matrix(face_matrix):
        return finish('degenerate_initial_distance_matrix')
    with warnings.catch_warnings(record=True) as captured, np.errstate(divide='warn', invalid='warn'):
        warnings.simplefilter('always')
        candidate = association.findSpeechFaceAssociationPartion(initial, positive, negative)
        report['optimizer_ran'] = True
        keys = list(candidate)
        audio_matrix = distances.computeDistanceMatrix(keys, modality='speech')
        face_matrix = distances.computeDistanceMatrix(keys, asd=candidate, modality='face')
        if usable_matrix(audio_matrix) and usable_matrix(face_matrix):
            correlations = similarity.computeAvgSimilarity(audio_matrix, face_matrix, avg=False)
            for key, correlation in zip(keys, correlations):
                value = float(correlation)
                if math.isfinite(value):
                    report['correlations'][key] = value
            # Upstream's final < 0.1 comparison does not remove NaNs. Refuse
            # such a result before invoking its actual offscreen correction.
            if all(report['correlations'][key] is not None for key in keys):
                kept = association.offscreenSpeakercorrection(candidate)
                for key, face in kept.items():
                    report['associations'][key] = face
                for key in candidate:
                    if key not in kept:
                        unresolved[key] = 'offscreen_or_low_cross_modal_correlation'
                collisions = set()
                retained = list(kept)
                for i, left in enumerate(retained):
                    for right in retained[i+1:]:
                        if kept[left] != kept[right]:
                            continue
                        a, b = speech[left]['speech'], speech[right]['speech']
                        if min(a[1], b[1]) > max(a[0], b[0]):
                            collisions.update((left, right))
                for key in collisions:
                    del report['associations'][key]
                    unresolved[key] = 'simultaneous_speech_face_collision'
            else:
                for key in candidate:
                    unresolved[key] = 'nonfinite_cross_modal_correlation'
        else:
            for key in candidate:
                unresolved[key] = 'degenerate_final_distance_matrix'
    report['warnings'] = list(dict.fromkeys(str(item.message) for item in captured))
    return finish()
