"""Program-owned in-film actor references: verification of the provenance the pipeline writes (stdlib only).

Writer: relay-files/actor_beta_hook.py write_reference_provenance(), called at E3 (persist_reference_receipt) right after
relay.actor_reference.make_reference or relay.actor_pipeline.make_line_reference cut the reference from this film's
own dialogue and B's
source_bound_receipt proved it is an exact excerpt of the window's speech stem. Nothing else writes these records.

Three records, kept apart on purpose:
  <job_dir>/reference-provenance/<reference_sha256>[-<line_identity_hash>].json   WHERE the voice came from:
      source WAV + speech stem (path, sha256), source fingerprint/audio index, job/revision, speaker id, time spans
      (movie seconds), channel, selection rule (function + module sha256 + limit), B admission receipt (path, sha256),
      and a pointer (path, sha256) to the source-text record.
  <job_dir>/reference-source-text/<reference_sha256>[-<line_identity_hash>].json  WHAT the actor said (the exact
      transcript Vox continuation needs). No target (English) text.
  take receipts (<cache>/<job>/actor-utterances/*.json)    the target (English) text that was spoken.

verify(voice, job_dir, ...) is the renderer's gate: a reference without a matching program record (hand-picked,
agent-prepared, copied from a fixture folder, made for another job, edited after creation) is refused, and that cue keeps its
original audio. v2 (review P3-1) BINDS the record instead of trusting it:
  * the voice must carry actor_id, equal to the record's speaker and to B's receipt's actor_id;
  * the B receipt must sit under <job_dir>/reference-receipts/, have B's version and match the record and the voice:
    reference_sha256, job_id, revision, actor_id, intervals; the record's job_id must be the CURRENT job;
  * the source WAV and speech stem named by the receipt must exist with their sha256, and the source fingerprint /
    audio index must equal the current source identity;
  * the reference must be an EXACT excerpt of the receipt's own speech stem at the recorded intervals and channel (the
    rebuild B's gate performs), so a re-encoded hand-picked clip placed in the job folder with a written record is
    refused even if B's gate is stubbed out.
Admission integrity still rests on B's gate first; this check is independent defence in depth.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROVENANCE_VERSION = 'vox-actor-reference-provenance/v1'
SOURCE_TEXT_VERSION = 'vox-actor-reference-source-text/v1'
PROVENANCE_DIR = 'reference-provenance'
SOURCE_TEXT_DIR = 'reference-source-text'
SELECTION_FUNCTION = 'relay.actor_reference.make_reference'
LINE_SELECTION_FUNCTION = 'relay.actor_pipeline.make_line_reference'
SELECTION_FUNCTIONS = frozenset({SELECTION_FUNCTION, LINE_SELECTION_FUNCTION})
CREATED_BY = 'program'
HEX64 = re.compile(r'^[0-9a-f]{64}$')
B_RECEIPT_VERSION = 'supervisor-b-source-bound-reference/v2'
B_RECEIPT_DIR = 'reference-receipts'
GAP_SECONDS = .12            # relay/actor_reference.make_reference joins utterances with 0.12 s of silence
# A provenance record must not carry text: any key naming text other than the pointer object 'source_text'.
_TEXT_KEY = re.compile(r'text|english|caption|transcript|translation', re.IGNORECASE)


class ProvenanceError(ValueError):
    """The reference has no valid program provenance (caller: ActorUnavailable('reference_unverified'))."""


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _norm(path):
    return str(Path(path).resolve()).replace('\\', '/').lower()


def _under(path, folder):
    child, parent = _norm(path), _norm(folder).rstrip('/')
    return child.startswith(parent + '/')


def text_keys(value, prefix=''):
    """Every key path in `value` that names text (the 'source_text' pointer itself is allowed: path + sha256 only)."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = prefix + '/' + str(key)
            if key == 'source_text' and isinstance(item, dict) and set(item) <= {'path', 'sha256', 'version'}:
                continue
            if _TEXT_KEY.search(str(key)):
                found.append(here)
            found.extend(text_keys(item, here))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(text_keys(item, '%s[%d]' % (prefix, index)))
    return found


def record_paths(job_dir, reference_sha256, *, job_id=None, revision=None, actor_id=None):
    job_dir = Path(job_dir)
    stem = reference_sha256
    if any(value is not None for value in (job_id, revision, actor_id)):
        if (not isinstance(job_id, str) or not job_id or type(revision) is not int
                or not isinstance(actor_id, str) or not actor_id.startswith('line-')):
            raise ProvenanceError('line reference needs the current job, integer revision and admitted line actor')
        identity = json.dumps([job_id, revision, actor_id], separators=(',', ':'), ensure_ascii=True).encode('ascii')
        stem += '-' + hashlib.sha256(identity).hexdigest()[:16]
    return (job_dir / PROVENANCE_DIR / (stem + '.json'),
            job_dir / SOURCE_TEXT_DIR / (stem + '.json'))


def _load(path, what):
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw.decode('utf-8')), _sha_bytes(raw)
    except FileNotFoundError:
        raise ProvenanceError('%s record missing (reference was not cut by the program)' % what) from None
    except (OSError, ValueError) as error:
        raise ProvenanceError('%s record unreadable: %s' % (what, error)) from error


def _spans(value):
    try:
        return [[float(a), float(b)] for a, b in value]
    except (TypeError, ValueError):
        raise ProvenanceError('spans malformed') from None


def exact_excerpt(reference_path, stem_path, intervals, origin, rate, channel):
    """True when the reference PCM equals the stem excerpt at `intervals` (movie seconds) on `channel`, joined with
    0.12 s gaps: the same reconstruction as B's gate (beta_stable_gate._reconstruct / _match_channel)."""
    import numpy as np
    import soundfile as sf
    ref, ref_rate = sf.read(str(reference_path), dtype='float32', always_2d=True)
    stem, stem_rate = sf.read(str(stem_path), dtype='float32', always_2d=True)
    if ref_rate != rate or stem_rate != rate or ref.shape[1] != 1 or not 0 <= int(channel) < stem.shape[1]:
        return False
    gap, parts = np.zeros(round(GAP_SECONDS * rate), dtype=np.float32), []
    for start, end in intervals:
        a, b = round((float(start) - origin) * rate), round((float(end) - origin) * rate)
        if a < 0 or b > len(stem) or b <= a:
            return False
        if parts:
            parts.append(gap)
        parts.append(stem[a:b, int(channel)].astype(np.float32))
    built = np.concatenate(parts)
    return built.shape == ref[:, 0].shape and bool(np.array_equal(built, ref[:, 0]))


def _bound_file(entry, what):
    if not isinstance(entry, dict) or not entry.get('path') or not HEX64.match(str(entry.get('sha256') or '')):
        raise ProvenanceError('%s path/sha256 missing' % what)
    if not Path(entry['path']).is_file() or _sha_file(entry['path']) != entry['sha256']:
        raise ProvenanceError('%s missing or changed' % what)
    return Path(entry['path'])


def verify(voice, job_dir, *, selection_module_sha256=None, job_id=None, revision=None, source_identity=None,
           exact=True):
    """Return the program provenance summary for `voice`, or raise ProvenanceError.

    voice: B's admitted voice ({'reference', 'reference_sha256', 'reference_text', 'source_intervals', 'actor_id'}).
    job_dir: <cache_dir>/<job_id> of the CURRENT job (from the pipeline context, never from the voice).
    job_id / revision / source_identity ({'source_fingerprint', 'audio_index'}): the current window's binding (from
    the pipeline context); each is enforced when given."""
    if not isinstance(voice, dict):
        raise ProvenanceError('voice must be a mapping')
    if job_dir is None:
        raise ProvenanceError('no job folder bound; program provenance cannot be checked')
    job_dir = Path(job_dir)
    reference = voice.get('reference')
    sha = str(voice.get('reference_sha256') or '').lower()
    if not reference or not HEX64.match(sha):
        raise ProvenanceError('voice lacks reference/reference_sha256')
    if not _under(reference, job_dir):
        raise ProvenanceError('reference is not under this job folder (hand-picked or foreign reference)')
    if not Path(reference).is_file() or _sha_file(reference) != sha:
        raise ProvenanceError('reference file missing or changed since creation')
    line_reference = voice.get('line_reference')
    if line_reference is not None:
        if (not isinstance(line_reference, dict) or line_reference.get('version') != 'line-own-reference-v1'
                or not isinstance(line_reference.get('cue_id'), str) or not line_reference['cue_id']
                or voice.get('actor_id') != 'line-' + line_reference['cue_id']):
            raise ProvenanceError('line reference is not bound to its admitted cue actor')
        prov_path, text_path = record_paths(job_dir, sha, job_id=job_id, revision=revision,
                                           actor_id=voice.get('actor_id'))
    else:
        prov_path, text_path = record_paths(job_dir, sha)
    record, record_sha = _load(prov_path, 'provenance')
    if not isinstance(record, dict) or record.get('version') != PROVENANCE_VERSION or record.get('created_by') != CREATED_BY:
        raise ProvenanceError('provenance record version/creator mismatch')
    leaked = text_keys(record)
    if leaked:
        raise ProvenanceError('provenance record carries text fields: %s' % leaked)
    ref = record.get('reference') or {}
    if ref.get('sha256') != sha or _norm(ref.get('path', '')) != _norm(reference):
        raise ProvenanceError('provenance reference path/sha256 mismatch')
    speaker = record.get('speaker_id')
    if not isinstance(speaker, str) or not speaker:
        raise ProvenanceError('provenance speaker id missing')
    if not voice.get('actor_id'):
        raise ProvenanceError('voice has no actor_id (B binds every admitted voice to its actor)')
    if voice.get('actor_id') != speaker:
        raise ProvenanceError('provenance speaker %r != admitted actor %r' % (speaker, voice.get('actor_id')))
    if job_id is not None and str(record.get('job_id')) != str(job_id):
        raise ProvenanceError('provenance record belongs to job %r, not the current job %r' % (record.get('job_id'), job_id))
    spans = _spans(record.get('spans_movie_seconds') or [])
    if not spans or spans != _spans(voice.get('source_intervals') or []):
        raise ProvenanceError('provenance time spans != voice source_intervals')
    if any(b <= a for a, b in spans):
        raise ProvenanceError('provenance time spans malformed')
    source = record.get('source') or {}
    wav = source.get('source_wav') or {}
    if not HEX64.match(str(wav.get('sha256') or '')) or not wav.get('path'):
        raise ProvenanceError('provenance source file hash missing')
    rule = record.get('selection_rule') or {}
    if isinstance(selection_module_sha256, dict):
        if set(selection_module_sha256) != SELECTION_FUNCTIONS or any(
                not HEX64.match(str(value)) for value in selection_module_sha256.values()):
            raise ProvenanceError('installed reference selector map is incomplete or malformed')
        selectors = selection_module_sha256
    elif isinstance(selection_module_sha256, str) and HEX64.match(selection_module_sha256):
        selectors = {SELECTION_FUNCTION: selection_module_sha256}
    else:
        raise ProvenanceError('installed reference selector hashes are missing')
    expected_function = LINE_SELECTION_FUNCTION if line_reference is not None else SELECTION_FUNCTION
    if (rule.get('function') != expected_function or rule.get('function') not in selectors
            or not HEX64.match(str(rule.get('module_sha256') or ''))):
        raise ProvenanceError('provenance selection rule is not the program reference cutter')
    if rule['module_sha256'] != selectors[rule['function']]:
        raise ProvenanceError('reference was cut by a different installed reference selector')
    admission = record.get('admission_receipt') or {}
    receipt_path = _bound_file(admission, 'B admission receipt')
    if not _under(receipt_path, job_dir / B_RECEIPT_DIR):
        raise ProvenanceError('B admission receipt is not under this job\'s %s/' % B_RECEIPT_DIR)
    receipt, _receipt_sha = _load(receipt_path, 'B admission receipt')
    if not isinstance(receipt, dict) or receipt.get('version') != B_RECEIPT_VERSION:
        raise ProvenanceError('B admission receipt version mismatch')
    for key, want in (('reference_sha256', sha), ('job_id', record.get('job_id')), ('revision', record.get('revision')),
                      ('actor_id', speaker)):
        if receipt.get(key) != want:
            raise ProvenanceError('B receipt %s != provenance/voice' % key)
    if _spans(receipt.get('intervals') or []) != spans:
        raise ProvenanceError('B receipt intervals != provenance spans')
    if revision is not None and receipt.get('revision') != revision:
        raise ProvenanceError('B receipt revision %r != current revision %r' % (receipt.get('revision'), revision))
    for what in ('source_wav', 'speech_stem'):
        if (receipt.get(what) or {}) != (source.get(what) or {}):
            raise ProvenanceError('provenance %s != B receipt %s' % (what, what))
    source_wav = _bound_file(receipt.get('source_wav'), 'source WAV')
    stem = _bound_file(receipt.get('speech_stem'), 'speech stem')
    for what, path in (('source WAV', source_wav), ('speech stem', stem)):
        if not _under(path, job_dir):              # the pipeline writes the window's files inside the job folder
            raise ProvenanceError('%s named by the B receipt is outside this job folder' % what)
    for key in ('source_fingerprint', 'audio_index'):
        if receipt.get(key) != source.get(key):
            raise ProvenanceError('provenance %s != B receipt' % key)
        if source_identity is not None and source_identity.get(key) != receipt.get(key):
            raise ProvenanceError('reference comes from another source (%s)' % key)
    if exact and not exact_excerpt(reference, stem, spans, float(receipt.get('origin')), int(receipt.get('rate')),
                                   receipt.get('channel', -1)):
        raise ProvenanceError('reference is not an exact excerpt of its receipt\'s speech stem (not program-cut)')
    pointer = record.get('source_text') or {}
    if _norm(pointer.get('path', '')) != _norm(text_path):
        raise ProvenanceError('source-text pointer is not this reference record')
    text_record, text_sha = _load(text_path, 'source-text')
    if text_sha != pointer.get('sha256'):
        raise ProvenanceError('source-text record changed since creation')
    if not isinstance(text_record, dict) or text_record.get('version') != SOURCE_TEXT_VERSION \
            or text_record.get('reference_sha256') != sha:
        raise ProvenanceError('source-text record version/reference mismatch')
    spoken = text_record.get('reference_text')
    if not isinstance(spoken, str) or spoken != voice.get('reference_text', ''):
        raise ProvenanceError('source-text record != voice reference_text')
    if text_record.get('reference_text_sha256') != _sha_bytes(spoken.encode('utf-8')):
        raise ProvenanceError('source-text sha256 mismatch')
    for key in ('english', 'target_text', 'translation', 'caption'):
        if key in text_record:
            raise ProvenanceError('source-text record carries target text (%s)' % key)
    return {'path': str(prov_path), 'sha256': record_sha, 'version': PROVENANCE_VERSION, 'speaker_id': speaker,
            'job_id': record.get('job_id'), 'revision': record.get('revision'), 'exact_excerpt_checked': bool(exact),
            'admission_receipt': {'path': str(receipt_path), 'sha256': admission.get('sha256')},
            'spans_movie_seconds': spans, 'source_wav_sha256': wav['sha256'],
            'source_fingerprint': source.get('source_fingerprint'), 'audio_index': source.get('audio_index'),
            'selection_rule': {k: rule.get(k) for k in ('function', 'module_sha256', 'limit_seconds')},
            'source_text': {'path': str(text_path), 'sha256': text_sha}}
