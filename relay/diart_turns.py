"""Opt-in reader for a per-film Diart pre-pass artifact (stdlib + numpy; Diart is never imported here).

The pre-pass (scripts/diart_prepass.py, run in the held WSL environment) streams the WHOLE film through Diart
(segmentation + overlap-aware embedding + online clustering) and writes one JSON artifact with film-global speaker ids
(D000, D001, ...). This module only reads that artifact and answers ``SpeakerAnalyzer.analyze``-shaped questions for one
analysis window, so the same real person keeps one label across windows (offline pyannote relabels per window).

Setting: ``actor_beta.diart_turns`` (needs ``actor_beta.enabled``) = path to an artifact file, or to a folder holding
``<source_fingerprint>.json``. Absent or empty = unchanged behaviour. Every failure raises DiartTurnsUnavailable with a
short reason; the caller falls back to the offline pyannote window analysis and records the reason.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path

from .movie_faults import MovieContentFallback

SCHEMA = 'kevflix-diart-turns'
VERSION = 1
NAMESPACE = 'diart-turns-v1'
LABEL_PREFIX = 'DIART_'
SAMPLE_RATE = 16000
COVERAGE_TOLERANCE = .25          # seconds a window may overhang the artifact's covered range (rounding at film end)
MIN_PIECE = .05                   # clipped turn pieces shorter than this are dropped
MIN_VECTOR_SEGMENT = .3           # only non-overlapped stretches at least this long feed a label's embedding
MIN_VECTOR_SECONDS = 1.            # less clean speech than this in the window -> vectors[label] = []
MAX_VECTOR_SECONDS = 15.           # cap of speech embedded per label per window
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_ID = re.compile(r'D\d{3,}')
_FINGERPRINT = re.compile(r'[0-9A-Fa-f]{8,128}')
_CACHE = {}
_CACHE_LOCK = threading.Lock()
_CACHE_LIMIT = 4


class DiartTurnsUnavailable(Exception):
    """The artifact cannot answer for this window; ``reason`` is a short machine-readable code."""

    def __init__(self, reason, detail=''):
        super().__init__(reason + (': ' + detail if detail else ''))
        self.reason = reason


def policy():
    return {'version': VERSION, 'schema': SCHEMA, 'label_prefix': LABEL_PREFIX,
            'coverage_tolerance': COVERAGE_TOLERANCE, 'min_piece': MIN_PIECE,
            'vector_min_segment': MIN_VECTOR_SEGMENT, 'vector_min_seconds': MIN_VECTOR_SECONDS,
            'vector_max_seconds': MAX_VECTOR_SECONDS, 'vectors': 'community1-native from non-overlapped speech',
            'fallback': 'pyannote-community-1 per window'}


def setting(config):
    """The EFFECTIVE setting: the configured artifact path/folder, or None (absent, empty or actor_beta not enabled)."""
    data = getattr(config, 'data', config)
    beta = data.get('actor_beta') if hasattr(data, 'get') else None
    if not isinstance(beta, dict) or beta.get('enabled') is not True:
        return None
    value = beta.get('diart_turns')
    return value.strip() if isinstance(value, str) and value.strip() else None


def _resolve(config, value):
    resolver = getattr(config, 'resolve', None)
    return Path(resolver(value)) if callable(resolver) else Path(value)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(config):
    """Value bound into config.engine_fingerprint(): None when off, else the setting plus the artifact identity."""
    value = setting(config)
    if value is None:
        return None
    record = {'version': VERSION, 'setting': value}
    path = _resolve(config, value)
    try:
        if path.is_file():
            record.update(kind='file', sha256=_file_sha256(path))
        elif path.is_dir():
            # Folder mode: the artifact is chosen per film at analysis time. Binding the listing here would change
            # every film's engine fingerprint (and re-dub it) whenever ANY film's artifact is added, so only the
            # setting is bound; the analysis cache key carries the artifact sha256 (file mode binds it here).
            record['kind'] = 'folder'
        else:
            record['kind'] = 'missing'
    except OSError:
        record['kind'] = 'unreadable'
    return record


def analyze_options(config, source_identity):
    """Extra keyword arguments for SpeakerClient.analyze: empty unless the setting is effective."""
    if setting(config) is None:
        return {}
    fingerprint_value = (source_identity or {}).get('source_fingerprint')
    return {'source_fingerprint': str(fingerprint_value)} if fingerprint_value else {}


def _number(value, reason):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DiartTurnsUnavailable('artifact_invalid', reason)
    return round(float(value), 3)


def split_exclusive(turns):
    """(aligned, overlaps) from possibly overlapping turns, by one sweep over the boundaries.

    aligned: stretches with exactly ONE active speaker (adjacent stretches of one speaker merged).
    overlaps: stretches with two or more active speakers, {'start','end','speakers':[sorted labels]}.
    """
    events = []
    for turn in turns:
        events.append((turn['start'], 1, turn['speaker']))
        events.append((turn['end'], -1, turn['speaker']))
    events.sort(key=lambda event: (event[0], event[1]))      # ends before starts at the same instant
    active, aligned, overlaps, previous = {}, [], [], None
    for time, delta, speaker in events:
        if previous is not None and time - previous > 1e-9 and active:
            who = sorted(active)
            if len(who) == 1:
                last = aligned[-1] if aligned else None
                if last and last['speaker'] == who[0] and abs(last['end'] - previous) <= 1e-9:
                    last['end'] = time
                else:
                    aligned.append({'start': previous, 'end': time, 'speaker': who[0]})
            else:
                last = overlaps[-1] if overlaps else None
                if last and last['speakers'] == who and abs(last['end'] - previous) <= 1e-9:
                    last['end'] = time
                else:
                    overlaps.append({'start': previous, 'end': time, 'speakers': who})
        active[speaker] = active.get(speaker, 0) + delta
        if active[speaker] == 0:
            del active[speaker]
        previous = time
    return aligned, overlaps


class DiartTurns:
    """A validated artifact. ``window`` clips it to one analysis window; labels are stable across windows."""

    def __init__(self, meta, turns, sha256):
        self.sha256 = sha256
        self.meta = meta
        self.turns = turns
        self.coverage = (meta['coverage']['start'], meta['coverage']['end'])

    def require_cover(self, start, end):
        if start < self.coverage[0] - COVERAGE_TOLERANCE or end > self.coverage[1] + COVERAGE_TOLERANCE:
            raise DiartTurnsUnavailable('window_not_covered', '%.3f-%.3f outside %.3f-%.3f'
                                        % (start, end, self.coverage[0], self.coverage[1]))

    def window(self, start, end):
        """Turns clipped to [start, end] (film seconds, labels DIART_Dnnn) plus the derived aligned/overlap views."""
        turns = []
        for turn in self.turns:
            a, b = max(turn['start'], start), min(turn['end'], end)
            if b - a >= MIN_PIECE:
                turns.append({'start': round(a, 3), 'end': round(b, 3), 'speaker': LABEL_PREFIX + turn['speaker']})
        turns.sort(key=lambda row: (row['start'], row['end'], row['speaker']))
        aligned, overlaps = split_exclusive(turns)
        return {'turns': turns, 'aligned': aligned, 'overlaps': overlaps}


def _parse(raw, sha256, expected_fingerprint):
    try:
        data = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as error:
        raise DiartTurnsUnavailable('artifact_unreadable', str(error)) from error
    if not isinstance(data, dict) or data.get('schema') != SCHEMA or data.get('version') != VERSION:
        raise DiartTurnsUnavailable('artifact_schema', 'schema/version mismatch')
    if str(data.get('source_fingerprint') or '').lower() != str(expected_fingerprint or '').lower() or not expected_fingerprint:
        raise DiartTurnsUnavailable('fingerprint_mismatch')
    coverage = data.get('coverage')
    if not isinstance(coverage, dict):
        raise DiartTurnsUnavailable('artifact_invalid', 'coverage')
    meta = {'coverage': {'start': _number(coverage.get('start'), 'coverage.start'),
                         'end': _number(coverage.get('end'), 'coverage.end')},
            'speaker_count': data.get('speaker_count'), 'params': data.get('params'), 'models': data.get('models'),
            'audio_sha256': data.get('audio_sha256'), 'source_fingerprint': data.get('source_fingerprint')}
    if meta['coverage']['end'] <= meta['coverage']['start'] or not isinstance(data.get('turns'), list):
        raise DiartTurnsUnavailable('artifact_invalid', 'coverage/turns')
    turns = []
    for row in data['turns']:
        if not isinstance(row, dict) or not isinstance(row.get('speaker'), str) or not _ID.fullmatch(row['speaker']):
            raise DiartTurnsUnavailable('artifact_invalid', 'turn speaker')
        a, b = _number(row.get('start'), 'turn.start'), _number(row.get('end'), 'turn.end')
        if b <= a:
            raise DiartTurnsUnavailable('artifact_invalid', 'turn order')
        turns.append({'start': a, 'end': b, 'speaker': row['speaker']})
    turns.sort(key=lambda row: (row['start'], row['end'], row['speaker']))
    return DiartTurns(meta, turns, sha256)


def load(path, expected_fingerprint):
    """Read and validate one artifact file (cached by path/size/mtime while unchanged)."""
    path = Path(path)
    try:
        info = path.stat()
    except OSError as error:
        raise DiartTurnsUnavailable('artifact_missing', str(error)) from error
    if not path.is_file() or info.st_size > MAX_ARTIFACT_BYTES:
        raise DiartTurnsUnavailable('artifact_missing' if not path.is_file() else 'artifact_too_large')
    key = (str(path), info.st_size, info.st_mtime_ns, str(expected_fingerprint or '').lower())
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DiartTurnsUnavailable('artifact_unreadable', str(error)) from error
    artifact = _parse(raw, hashlib.sha256(raw).hexdigest(), expected_fingerprint)
    with _CACHE_LOCK:
        while len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = artifact
    return artifact


def open_for(config, source_fingerprint):
    """The artifact for this film under the effective setting; DiartTurnsUnavailable when it cannot be used."""
    value = setting(config)
    if value is None:
        raise DiartTurnsUnavailable('not_configured')
    if not source_fingerprint:
        raise DiartTurnsUnavailable('source_fingerprint_missing')
    path = _resolve(config, value)
    if path.is_dir():
        if not _FINGERPRINT.fullmatch(str(source_fingerprint)):
            raise DiartTurnsUnavailable('fingerprint_invalid')
        path = path / (str(source_fingerprint).lower() + '.json')
    return load(path, source_fingerprint)


def _choose(pieces, cap):
    """Longest stretches first until ``cap`` seconds are reached (the last one trimmed); returned in time order."""
    chosen, total = [], 0.
    for a, b in sorted(pieces, key=lambda piece: (piece[0] - piece[1], piece[0])):
        take = min(b - a, cap - total)
        if take < MIN_VECTOR_SEGMENT:
            break
        chosen.append((a, a + take))
        total += take
    return sorted(chosen)


def label_vectors(window, waveform, window_start, embed, *, rate=SAMPLE_RATE):
    """{label: community-1-native embedding list, or [] when the clean evidence is insufficient}.

    ``waveform`` is the window's mono 16 kHz audio starting at film second ``window_start``; ``embed`` is
    SpeakerAnalyzer.embed. Only stretches where the label is the ONLY active speaker count as evidence.
    """
    import numpy as np
    vectors = {}
    for label in sorted({turn['speaker'] for turn in window['turns']}):
        pieces = [(row['start'], row['end']) for row in window['aligned']
                  if row['speaker'] == label and row['end'] - row['start'] >= MIN_VECTOR_SEGMENT]
        chosen = _choose(pieces, MAX_VECTOR_SECONDS)
        if sum(b - a for a, b in chosen) < MIN_VECTOR_SECONDS:
            vectors[label] = []
            continue
        parts = [waveform[max(0, round((a - window_start) * rate)):round((b - window_start) * rate)] for a, b in chosen]
        samples = np.concatenate(parts).astype(np.float32, copy=False)
        try:
            vector = [float(x) for x in embed(samples)]
        except MovieContentFallback:
            vector = []
        vectors[label] = vector if vector and all(math.isfinite(x) for x in vector) else []
    return vectors
