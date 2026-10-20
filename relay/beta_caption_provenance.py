"""PROPOSAL v2 (not wired; stable gate 95a33d8a... unchanged): verified aligned-English-caption TARGET wording.

Separates source-word uncertainty (ASR `unclear`) from speaker/reference certainty. An unclear cue may use an
aligned English caption as its TARGET text only when an immutable provenance record re-verifies against
(1) the EXPECTED job/source/subtitle binding produced by the pipeline's own trusted source selection (the job
spec: job_id, revision, source_fingerprint, selected sidecar path + full sha + language), and (2) the subtitle
file itself. Voice/reference admission stays STRICT (pipeline_gate._check_actor and clean contributing reference
cues are unchanged); this concerns target wording only.

v2 (review findings): expected job/source/subtitle binding (a correctly hashed subtitle of another film is
refused; the caller's source_path is no longer trusted), strict int/finite/bounded field validation, every
grouped cue must map to the same caption index/time/text and lie inside the caption envelope (dash parts too).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path

VERSION = 'caption-target-provenance-v2'
BINDING_VERSION = 'job-subtitle-binding-v1'
BASIS = 'aligned-English-caption-for-unclear-speech'
ENGLISH_TAGS = ('en', 'eng')
GROUP_GAP = .65            # existing relay.dialogue grouping gap; no new threshold
MAX_SECONDS = 24 * 3600.0  # bounded timestamp domain (a movie timeline), not a tuning threshold
LANG_RE = re.compile(r"\.(en|eng|ru|rus|es|spa|fr|fra|de|deu|ja|jpn|zh|zho|ko|kor|it|ita|pt|por|ar|ara|hi|hin)(?=\.|$)", re.I)
DASH_RE = r'(?:^|\s)-\s+'           # relay/dialogue.py dash split (verbatim)
SENTENCE_RE = r'(?<=[.!?])\s+'      # relay/dialogue.py sentence split (verbatim)
_HEX64 = re.compile(r'^[0-9a-f]{64}$')


def sha256_text(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _norm(p):
    return os.path.normcase(os.path.abspath(str(p)))


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _time(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= MAX_SECONDS


def load_sidecar(path):
    """Exactly relay/media.subtitle_cues for an explicit sidecar: pysubs2 plaintext, empty entries dropped."""
    import pysubs2
    parsed = pysubs2.load(str(path), encoding='utf-8')
    return [{'start': x.start / 1000, 'end': x.end / 1000, 'text': x.plaintext.strip()} for x in parsed if x.plaintext.strip()]


def filename_language(subtitle_path, source_path):
    """relay/media.subtitle_cues language rule from the sidecar file name."""
    picked, source = Path(subtitle_path), Path(source_path)
    suffix = picked.stem[len(source.stem):] if picked.stem.startswith(source.stem) else picked.stem
    m = LANG_RE.search(suffix)
    return m.group(1).lower() if m else None


def caption_parts(text, n_cues, caption_text):
    """relay/dialogue.py caption_fallback assignment, returning (parts, split_kind) or (None, reason)."""
    parts = [caption_text(x) for x in re.split(DASH_RE, text) if caption_text(x)]
    kind = 'dash'
    if len(parts) != n_cues and n_cues > 1:
        sentences = [x.strip() for x in re.split(SENTENCE_RE, caption_text(text)) if x.strip()]
        if len(sentences) == n_cues:
            parts, kind = sentences, 'sentence'
    if n_cues == 1:
        return [caption_text(text)], 'whole'
    if len(parts) == n_cues and n_cues > 1:
        return parts, kind
    return None, 'caption parts do not match mapped cues'


def job_binding(*, job_id, revision, source_fingerprint, subtitle_path, source_path):
    """Expected binding, built ONCE by the pipeline from the job spec (worker spec['subtitles'], spec['source'],
    spec['source_fingerprint'], job_id, revision) when it selects the sidecar; held in the gate context, never
    read from a cue. Full sha (media.fingerprint is partial)."""
    return {'version': BINDING_VERSION, 'job_id': job_id, 'revision': revision, 'source_fingerprint': source_fingerprint,
            'subtitle_path': str(Path(subtitle_path).resolve()), 'subtitle_sha256': file_sha256(subtitle_path),
            'source_path': str(Path(source_path).resolve()), 'language': filename_language(subtitle_path, source_path),
            'language_basis': 'filename-tag', 'subtitle_kind': 'sidecar'}


def _ordered_caption_cues(all_cues, caption_index):
    return sorted((c for c in all_cues if c.get('caption_index') == caption_index), key=lambda c: (float(c['start']), c['id']))


def emit(cue, *, binding, all_cues, captions, window_captions, caption_text):
    """Reference emitter for C (called where actor_pipeline sets english from caption_fallback).

    binding: the job_binding() for this job. captions: the FULL list from subtitle_cues; window_captions: the
    `selected` list passed to plan_dialogue (cue['caption_index'] is window-local, actor_pipeline.py:293).
    The file index is found by object identity, never by text search. Records only; never decides admission.
    """
    idx = cue['caption_index']
    cap = window_captions[idx]
    file_index = next(i for i, c in enumerate(captions) if c is cap)
    ordered = _ordered_caption_cues(all_cues, idx)
    ids = [c['id'] for c in ordered]
    parts, kind = caption_parts(cap['text'], len(ordered), caption_text)
    return {'version': VERSION, 'basis': BASIS,
            'binding': {k: binding[k] for k in ('job_id', 'revision', 'source_fingerprint', 'subtitle_path', 'subtitle_sha256')},
            'caption_index': idx, 'caption_file_index': file_index, 'caption_start': cap['start'], 'caption_end': cap['end'],
            'caption_text_sha256': sha256_text(cap['text']), 'cue_ids_in_caption': ids, 'part_index': ids.index(cue['id']),
            'part_count': len(parts) if parts else None, 'split': kind,
            'part_text': parts[ids.index(cue['id'])] if parts else None}


def _has_trust_flag(value):
    if isinstance(value, dict):
        return any(k == 'trusted' or _has_trust_flag(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_trust_flag(v) for v in value)
    return False


def _binding_problem(expected, path_allowed):
    if not isinstance(expected, dict) or expected.get('version') != BINDING_VERSION:
        return 'expected job/subtitle binding missing'
    if _has_trust_flag(expected):
        return 'caller trust flags are refused'
    if not isinstance(expected.get('job_id'), str) or not expected['job_id'] or not _int(expected.get('revision')) \
            or not isinstance(expected.get('source_fingerprint'), str) or not expected['source_fingerprint']:
        return 'expected binding job/revision/source malformed'
    if expected.get('subtitle_kind') != 'sidecar':
        return 'embedded subtitle language is unproven in v2 (needs a persisted stream-tag receipt)'
    path = expected.get('subtitle_path')
    if not path or not Path(path).is_file() or not path_allowed(path):
        return 'expected subtitle missing or outside movie roots'
    if not _HEX64.match(str(expected.get('subtitle_sha256') or '')) or file_sha256(path) != expected['subtitle_sha256']:
        return 'expected subtitle sha256 mismatch'
    language = filename_language(path, expected.get('source_path') or '')
    if language not in ENGLISH_TAGS or expected.get('language') != language or expected.get('language_basis') != 'filename-tag':
        return 'subtitle language is not provably English'
    return None


def verify(cue, *, expected, all_cues, caption_text, path_allowed, load=load_sidecar):
    """None when the cue's TARGET wording is a verified aligned English caption part; else a refusal reason.

    `expected` comes from the pipeline's trusted source selection (gate context), never from the cue.
    Everything else is recomputed from the subtitle file and the pipeline's own cue list.
    """
    if not cue.get('unclear'):
        return 'not an unclear cue (ordinary path applies)'
    if cue.get('translation') != BASIS:
        return 'english is not from the aligned caption basis'
    bad = _binding_problem(expected, path_allowed)
    if bad:
        return bad
    rec = cue.get('caption_target')
    if not isinstance(rec, dict) or rec.get('version') != VERSION or rec.get('basis') != BASIS:
        return 'missing or unknown caption provenance'
    if _has_trust_flag(rec) or _has_trust_flag(cue.get('caption')):
        return 'caller trust flags are refused'
    if not cue.get('speaker') or cue.get('overlapped'):
        return 'speaker unresolved or overlapped'
    b = rec.get('binding') or {}
    for key, want in (('job_id', expected['job_id']), ('revision', expected['revision']),
                      ('source_fingerprint', expected['source_fingerprint']), ('subtitle_sha256', expected['subtitle_sha256'])):
        if b.get(key) != want or type(b.get(key)) is not type(want):
            return 'provenance is not bound to this job/source/subtitle (%s)' % key
    if not isinstance(b.get('subtitle_path'), str) or _norm(b['subtitle_path']) != _norm(expected['subtitle_path']):
        return 'provenance is not bound to this job/source/subtitle (subtitle_path)'
    for key in ('caption_index', 'caption_file_index', 'part_index', 'part_count'):
        if not _int(rec.get(key)):
            return 'malformed %s' % key
    if rec['part_index'] >= rec['part_count']:
        return 'malformed part_index'
    if not (_time(rec.get('caption_start')) and _time(rec.get('caption_end')) and rec['caption_start'] < rec['caption_end']):
        return 'malformed caption timestamps'
    if not _HEX64.match(str(rec.get('caption_text_sha256') or '')):
        return 'malformed caption text hash'
    try:
        captions = load(expected['subtitle_path'])
        cap = captions[rec['caption_file_index']]
    except Exception as exc:
        return 'caption index not found: %s' % type(exc).__name__
    if abs(cap['start'] - rec['caption_start']) > .001 or abs(cap['end'] - rec['caption_end']) > .001 \
            or sha256_text(cap['text']) != rec['caption_text_sha256']:
        return 'caption time/text does not match the subtitle file'
    if cue.get('caption_index') != rec['caption_index']:
        return 'cue caption index does not match provenance'
    ordered = _ordered_caption_cues(all_cues, rec['caption_index'])
    ids = [c.get('id') for c in ordered]
    if ids != rec.get('cue_ids_in_caption') or cue['id'] not in ids or ids.index(cue['id']) != rec['part_index']:
        return 'ambiguous cue to caption-part mapping'
    for member in ordered:                       # every grouped cue must truly map to THIS caption and lie inside it
        own = member.get('caption') or {}
        if not (_time(own.get('start')) and _time(own.get('end'))) or abs(own['start'] - cap['start']) > .001 \
                or abs(own['end'] - cap['end']) > .001 or own.get('text') != cap['text']:
            return 'grouped cue %s does not map to the same caption' % member.get('id')
        try:
            s, e = float(member['start']), float(member['end'])
        except (KeyError, TypeError, ValueError):
            return 'grouped cue %s has malformed times' % member.get('id')
        if not (_time(s) and _time(e) and s < e) or s < cap['start'] - GROUP_GAP or e > cap['end'] + GROUP_GAP:
            return 'grouped cue %s lies outside the caption envelope' % member.get('id')
    parts, kind = caption_parts(cap['text'], len(ordered), caption_text)
    if parts is None:
        return kind
    if kind == 'sentence':
        return 'ordinal sentence split has no per-part timing (unverifiable)'
    if kind == 'whole' and len([x for x in re.split(DASH_RE, cap['text']) if caption_text(x)]) > 1:
        return 'multi-speaker (dash) caption mapped to a single cue'
    if kind != rec.get('split') or len(parts) != rec['part_count']:
        return 'caption split does not match provenance'
    part = parts[rec['part_index']]
    if part != rec.get('part_text') or cue.get('english') != part:
        return 'cue english is not the exact allowed caption part'
    if kind == 'dash':
        speakers = [c.get('speaker') for c in ordered]
        if any(s is None for s in speakers) or any(a == b for a, b in zip(speakers, speakers[1:])):
            return 'dash turns are not distinct resolved speakers'
    return None
