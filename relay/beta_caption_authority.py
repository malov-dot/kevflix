"""Program-owned caption authority for subtitle-primary targets (actor_beta.caption_targets; stdlib only).

Verifies the HOST's own subtitle receipt for one analysed window against the job's own subtitle selection, never a
caller boolean, an arbitrary receipt path, or a path-only binding:

  * expected  = relay.embedded_subtitles.spec_choice(spec) computed by the worker from the job payload, plus the
                job's source_fingerprint / revision / job_id (pipeline arguments);
  * receipt   = CaptionList.subtitle_source (also <chunk>/subtitle-source.json), contract 'embedded-subtitles-v1';
  * captions  = the exact cue list the pipeline received.

Embedded: every 60 s block of the requested window must be named in the receipt with the cache key recomputed here
from (source_fingerprint, stream index, block, codec), its cache entry must exist under config.cache/subtitle-blocks,
carry that identity and a matching content digest, and the deduplicated sorted window cues rebuilt from those blocks
must equal the captions received (and the receipt content digest). No FFmpeg, no NAS or whole-movie rescan.
External sidecar: FULL file sha256 + the same parse/overlap rule, compared with the received captions.
Language must be English by an explicit stream tag (embedded) or file-name tag (external). Forced/image/ambiguous/
untagged selections arrive as kind 'none' and yield no authority (existing path). Mirrors C's embedded_subtitles
(pinned 5822192d...) digest/identity/window rules exactly; tests compare against C's functions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

VERSION = 'caption-authority-v1'
RECEIPT_VERSION = 'embedded-subtitles-v1'
TEXT_CODECS = frozenset(('subrip', 'ass', 'ssa', 'webvtt', 'mov_text', 'text'))
BLOCK_SECONDS = 60.0
PREROLL_SECONDS = 30.0
EMBEDDED_SELECTIONS = ('explicit-stream', 'auto-unambiguous-english')
EXTERNAL_SELECTIONS = ('explicit-file', 'sidecar', 'file')


def digest(value):
    """Same canonical digest as relay.embedded_subtitles._digest."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def block_identity(source_fingerprint, stream_index, block, codec):
    return {'version': RECEIPT_VERSION, 'source_fingerprint': source_fingerprint, 'stream_index': int(stream_index),
            'block': int(block), 'block_seconds': BLOCK_SECONDS, 'preroll_seconds': PREROLL_SECONDS,
            'codec': codec, 'output': 'srt', 'timestamps': 'copyts'}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _plain(cues):
    return [{'start': c['start'], 'end': c['end'], 'text': c['text'], 'actor': c.get('actor')} for c in cues]


def _received(captions):
    out = []
    for c in captions:
        if not isinstance(c, dict) or not _num(c.get('start')) or not _num(c.get('end')) or not isinstance(c.get('text'), str):
            raise ValueError('malformed received caption')
        out.append(c)
    return out


def _with_language(cues, choice):
    return [dict(c, language=choice.get('declared_language'), language_raw=choice.get('declared_language_raw'),
                 language_basis=choice.get('language_basis')) for c in cues]


def _refuse(reason):
    return {'version': VERSION, 'status': 'refused', 'reason': reason}


def verify(*, receipt, captions, expected_choice, job_id, revision, source_fingerprint, window, cache_root,
           resolve_media=None, load_file=None):
    """Return {'status': 'ok', ...authority...} | {'status': 'none', reason} | {'status': 'refused', reason}.

    resolve_media(path) -> actual file path (relay Config.media_path); load_file(path) -> [{start,end,text,actor}]
    (pysubs2 plaintext, the same parse relay.embedded_subtitles._file_cues uses).
    """
    if not isinstance(expected_choice, dict) or not isinstance(receipt, dict):
        return _refuse('subtitle receipt or expected job selection missing')
    kind = expected_choice.get('kind')
    if kind == 'none' or receipt.get('kind') == 'none':
        return {'version': VERSION, 'status': 'none', 'reason': expected_choice.get('reason') or receipt.get('reason')}
    if receipt.get('version') != RECEIPT_VERSION or receipt.get('kind') != kind:
        return _refuse('receipt contract/kind does not match the job selection')
    if not isinstance(job_id, str) or not job_id or not _int(revision) or not isinstance(source_fingerprint, str) or not source_fingerprint:
        return _refuse('job/revision/source identity malformed')
    if receipt.get('source_fingerprint') != source_fingerprint:
        return _refuse('receipt source fingerprint differs from the job source')
    if receipt.get('cue_time_basis') != 'absolute-source-seconds':
        return _refuse('cue time basis is not absolute source seconds')
    try:
        start, end = float(window[0]), float(window[1])
    except (TypeError, ValueError, IndexError):
        return _refuse('window malformed')
    if not (_num(start) and _num(end) and 0 <= start < end):
        return _refuse('window malformed')
    rw = receipt.get('window')
    if not isinstance(rw, list) or len(rw) != 2 or rw[0] != start or rw[1] != end:
        return _refuse('receipt window differs from the requested window')
    if receipt.get('status') != 'ok':
        return {'version': VERSION, 'status': 'none', 'reason': 'receipt status %s: %s' % (receipt.get('status'), receipt.get('reason'))}
    for key in ('selection', 'codec', 'declared_language', 'declared_language_raw', 'language_basis'):
        if receipt.get(key) != expected_choice.get(key):
            return _refuse('receipt %s differs from the job selection' % key)
    if receipt.get('declared_language') != 'en':
        return {'version': VERSION, 'status': 'none', 'reason': 'selected subtitles are not declared English'}
    try:
        received = _received(captions)
    except ValueError as error:
        return _refuse(str(error))
    if receipt.get('cue_count') != len(received) or receipt.get('content_sha256') != digest(received):
        return _refuse('received captions differ from the receipt content digest')
    base = {'version': VERSION, 'status': 'ok', 'kind': kind, 'job_id': job_id, 'revision': revision,
            'source_fingerprint': source_fingerprint, 'selection': receipt['selection'], 'codec': receipt['codec'],
            'language': 'en', 'language_basis': receipt['language_basis'], 'window': [start, end]}
    if kind == 'embedded':
        idx = expected_choice.get('stream_index')
        if not _int(idx) or receipt.get('selected_stream_index') != idx or type(receipt.get('selected_stream_index')) is not int:
            return _refuse('selected stream index differs from the job selection')
        if receipt['selection'] not in EMBEDDED_SELECTIONS or receipt['codec'] not in TEXT_CODECS:
            return _refuse('embedded selection or codec is not an eligible text stream')
        if receipt['language_basis'] != 'stream-language-tag':
            return _refuse('embedded English must come from the stream language tag')
        blocks = receipt.get('blocks')
        wanted = list(range(int(start // BLOCK_SECONDS), int((end - 1e-9) // BLOCK_SECONDS) + 1))
        if not isinstance(blocks, list) or [b.get('block') if isinstance(b, dict) else None for b in blocks] != wanted:
            return _refuse('receipt blocks do not cover exactly the requested window')
        root = Path(cache_root)
        rebuilt, seen, verified = [], set(), []
        for b in blocks:
            identity = block_identity(source_fingerprint, idx, b['block'], receipt['codec'])
            key = digest(identity)
            if b.get('cache_key') != key:
                return _refuse('block %s cache key is not the recomputed identity key' % b['block'])
            path = root / (key + '.json')
            if path.is_symlink() or not path.is_file():
                return _refuse('block %s cache entry missing' % b['block'])
            try:
                entry = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                return _refuse('block %s cache entry unreadable' % b['block'])
            if entry.get('identity') != identity or entry.get('content_sha256') != digest(entry.get('cues')) \
                    or b.get('content_sha256') != entry['content_sha256']:
                return _refuse('block %s cache entry identity/content does not verify' % b['block'])
            verified.append({'block': b['block'], 'cache_key': key, 'content_sha256': entry['content_sha256']})
            for cue in entry['cues']:
                ident = (cue['start'], cue['end'], cue['text'], cue['actor'])
                if cue['end'] > start and cue['start'] < end and ident not in seen:
                    seen.add(ident); rebuilt.append(dict(cue))
        rebuilt.sort(key=lambda c: (c['start'], c['end'], c['text']))
        if _with_language(rebuilt, expected_choice) != received:
            return _refuse('received captions differ from the verified subtitle blocks')
        namespace = digest({'kind': 'embedded', 'source_fingerprint': source_fingerprint, 'stream_index': idx,
                            'codec': receipt['codec'], 'version': RECEIPT_VERSION})
        return dict(base, stream_index=idx, blocks=verified, captions=_plain(rebuilt), namespace=namespace,
                    completeness='not-guaranteed (receipt preroll limit)')
    if kind == 'external':
        if receipt['selection'] not in EXTERNAL_SELECTIONS or receipt.get('path') != expected_choice.get('path'):
            return _refuse('external selection/path differs from the job selection')
        if receipt['language_basis'] != 'file-name-tag':
            return _refuse('external English must come from the file-name tag')
        if resolve_media is None or load_file is None:
            return _refuse('external subtitle resolver unavailable')
        try:
            path = Path(resolve_media(receipt['path']))
        except Exception as exc:
            return _refuse('external subtitle path refused: %s' % type(exc).__name__)
        if not path.is_file() or path.is_symlink():
            return _refuse('external subtitle file missing')
        full = file_sha256(path)
        parsed = [c for c in load_file(path) if c['end'] > start and c['start'] < end]
        if _with_language(parsed, expected_choice) != received:
            return _refuse('received captions differ from the verified subtitle file')
        namespace = digest({'kind': 'external', 'subtitle_sha256': full, 'version': VERSION})
        return dict(base, subtitle_path=str(path), subtitle_sha256=full, captions=_plain(parsed), namespace=namespace,
                    completeness='whole-file-parsed')
    return _refuse('unknown subtitle kind')


def caption_key(authority, caption):
    """Stable across windows/blocks: authority namespace + absolute cue identity (no window index, no list hash)."""
    return digest({'ns': authority['namespace'], 'start': caption['start'], 'end': caption['end'],
                   'text_sha256': hashlib.sha256(caption['text'].encode('utf-8')).hexdigest()})
