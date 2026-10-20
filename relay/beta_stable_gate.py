"""Program-owned dynamic reference admission for the actor pipeline (Supervisor B), v2.

Every call creates a fresh in-memory TrustIssuer and mints a TrustAdmission only after
(1) the window context is proven (source_wav file sha256 + decoded PCM == supplied original PCM,
speech stem file sha256 + decoded PCM == supplied speech PCM, actual model provenance file sha256 +
policy, stable source fingerprint/audio index), and (2) the reference is an exact source-bound excerpt
of this window's speech stem, or carries a persisted source-bound receipt whose OWN creation-window
stem rebuilds it exactly. Missing/tampered metadata fails closed: cues fall back to original audio.
No persisted secret, no manual token, no caller trust flag, no hash whitelist. Provenance only:
identity_verified is always False; nothing here measures voice similarity or accuracy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import sys
from pathlib import Path

GATE_VERSION = 'supervisor-b-pipeline-gate/v2'
RECEIPT_VERSION = 'supervisor-b-source-bound-reference/v2'
DEFAULT_ROOTS = ('C:/Sem-Migration-Staging/Movie-Dub-Relay', 'C:/Sem-Data/Movie-Dub-Cache')
PRIVATE_SEGMENTS = {'sem-voice', 'voice', 'voices', 'profile', 'profiles', 'sem-data', 'memory', 'memories',
                    'personal', 'legacy512', 'speaker-bank-512', 'continuity-protection', 'sem-4.0', 'sem-5.0'}
PERSON_FIELDS = ('person', 'person_id', 'profile', 'profile_id', 'owner', 'owner_id')
REFUSED_FAMILIES = {'legacy512', 'personal512', 'personal-512'}
_CACHE_DIR = Path(__file__).resolve().parents[2] / 'actor-tts' / 'reference-cache'
GAP = .12
PAD = .06
_HEX64 = re.compile(r'^[0-9a-f]{64}$')


def _load_reference_cache():
    # Same import style as A's runner so TrustAdmission identity matches A's renderer.
    if str(_CACHE_DIR) not in sys.path:
        sys.path.insert(0, str(_CACHE_DIR))
    import reference_cache
    return reference_cache


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _norm(p):
    return os.path.normcase(os.path.abspath(str(p).replace('/', os.sep)))


def path_allowed(path, allowed_roots=None):
    """True only under a movie root with no private Sem segment below that root."""
    roots = list(DEFAULT_ROOTS) + [str(r) for r in (allowed_roots or ())]
    full = _norm(Path(path).resolve()) if Path(path).exists() else _norm(path)
    for root in roots:
        r = _norm(root).rstrip(os.sep)
        if not full.startswith(r + os.sep):
            continue
        below = [s.lower() for s in full[len(r):].split(os.sep) if s][:-1]
        if not any(s in PRIVATE_SEGMENTS for s in below):
            return True
    return False


def _samples(t, origin, rate):
    # make_reference stores origin+a/rate, so round((t-origin)*rate) recovers a.
    return round((float(t) - origin) * rate)


def _reconstruct(speech, intervals, origin, rate):
    """Return (per-channel reconstructions, sample spans) or None if intervals leave the stem."""
    import numpy as np
    spans = []
    for s, e in intervals:
        a, b = _samples(s, origin, rate), _samples(e, origin, rate)
        if a < 0 or b > len(speech) or b <= a:
            return None
        spans.append((a, b))
    gap = np.zeros(round(GAP * rate), dtype=np.float32)
    out = []
    for c in range(speech.shape[1]):
        parts = []
        for a, b in spans:
            if parts:
                parts.append(gap)
            parts.append(speech[a:b, c].astype(np.float32))
        out.append(np.concatenate(parts))
    return out, spans


def _match_channel(built, data):
    if built is None or data.shape[1] != 1:
        return None
    mono = data[:, 0]
    return next((c for c, r in enumerate(built[0]) if r.shape == mono.shape and (r == mono).all()), None)


def _decoded(path):
    import numpy as np
    import soundfile as sf
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    return data, rate, _sha(np.ascontiguousarray(data).tobytes())


def _source_id(context):
    # Stable movie source (not the per-window WAV hash), so windows of one source share a namespace.
    sid = context['source_identity']
    return f"movie-source:{sid['source_fingerprint']}:a{sid['audio_index']}"


def _job(context):
    return f"movie-dub:{context['job_id']}:r{context['revision']}"


def _model_identity(prov):
    return {'family': prov.get('family'), 'model_sha256': prov.get('model_sha256')}


def _file_bound(entry, roots, what):
    """entry {'path','sha256'}: under movie roots and file sha256 equals the declared sha256."""
    if not isinstance(entry, dict) or not entry.get('path') or not _HEX64.match(str(entry.get('sha256') or '')):
        return None, f'{what} path/sha256 missing or malformed'
    path = Path(entry['path'])
    if not path.is_file():
        return None, f'{what} file missing'
    if not path_allowed(path, roots):
        return None, f'{what} outside movie roots'
    if file_sha256(path) != entry['sha256']:
        return None, f'{what} file sha256 != declared sha256'
    return path, None


def _pcm_matches(path, array, rate, what):
    import numpy as np
    import soundfile as sf
    data, file_rate = sf.read(str(path), dtype='float32', always_2d=True)
    if file_rate != rate:
        return f'{what} sample rate != context rate'
    if data.shape != array.shape or not np.array_equal(data, array):
        return f'{what} decoded PCM != supplied PCM'
    return None


def validate_context(context):
    """Fail-closed window metadata check. None when usable, else a reason (every cue -> original audio).

    Speech-stem reconstruction alone is NOT source validation: the source WAV file and its decoded PCM
    must equal the supplied original PCM, and the stem file must equal the supplied speech PCM.
    """
    import numpy as np
    try:
        if not isinstance(context.get('job_id'), str) or not context['job_id'].strip():
            return 'job_id missing'
        for key in ('revision', 'rate'):
            if isinstance(context.get(key), bool) or not isinstance(context.get(key), int) or context[key] <= 0:
                return f'{key} missing'
        float(context['origin'])
        original, speech = context.get('original'), context.get('speech')
        if not isinstance(original, np.ndarray) or not isinstance(speech, np.ndarray) or original.ndim != 2 \
                or speech.ndim != 2 or len(original) != len(speech) or len(original) == 0:
            return 'original/speech PCM missing or geometry mismatch'
        sid = context.get('source_identity')
        if not isinstance(sid, dict) or not isinstance(sid.get('source_fingerprint'), str) \
                or not sid['source_fingerprint'].strip() or isinstance(sid.get('audio_index'), bool) \
                or not isinstance(sid.get('audio_index'), int):
            return 'source_identity {source_fingerprint, audio_index} missing'
        roots = context.get('allowed_roots') or ()
        for what, array in (('source_wav', original), ('speech_stem', speech)):
            path, bad = _file_bound(context.get(what), roots, what)
            if bad:
                return bad
            bad = _pcm_matches(path, array, context['rate'], what)
            if bad:
                return bad
        prov = context.get('model_provenance')
        if not isinstance(prov, dict) or not isinstance(prov.get('family'), str) or not prov['family'].strip():
            return 'model_provenance.family missing'
        fam = prov['family'].lower()
        if fam in REFUSED_FAMILIES or 'legacy512' in fam or 'personal' in fam:
            return 'refused speaker model family ' + prov['family']
        if not prov.get('model_path') or not Path(prov['model_path']).is_file():
            return 'model_provenance.model_path missing'
        if not _HEX64.match(str(prov.get('model_sha256') or '')) or file_sha256(prov['model_path']) != prov['model_sha256']:
            return 'model_provenance.model_sha256 != model file'
        if not isinstance(prov.get('policy'), dict) or not prov['policy']:
            return 'model_provenance.policy (actual existing thresholds) missing'
    except Exception as exc:  # malformed metadata never admits
        return f'context malformed: {type(exc).__name__}: {exc}'
    return None


class _LineReferenceRefused(ValueError):
    pass


def _line_expected_text(context, ident, spans, voice):
    """Re-derive a line reference from original cue/turn evidence, never its claimed geometry."""
    try:
        metadata = voice.get('line_reference')
        if not isinstance(metadata, dict) or metadata.get('version') != 'line-own-reference-v1':
            return None
        cue_id = metadata.get('cue_id')
        if not isinstance(cue_id, str) or ident != 'line-' + cue_id or len(spans) != 1:
            return None
        matches = [c for c in context.get('all_cues') or () if isinstance(c, dict) and c.get('id') == cue_id]
        if len(matches) != 1:
            return None
        cue = matches[0]
        if cue.get('speaker') != ident or cue.get('overlapped') or cue.get('turn_overlap') is True:
            return None
        if not isinstance(cue.get('text'), str):
            return None
        origin, rate, n = float(context['origin']), int(context['rate']), len(context['speech'])
        start, end = float(cue['start']), float(cue['end'])
        begin = max(start, float(cue.get('speech_start') or start))
        stop = min(end, float(cue.get('speech_end') or end))
        limit = float(context['actor_reference_seconds'])
        if not all(math.isfinite(v) for v in (origin, start, end, begin, stop, limit)):
            return None
        if not origin <= begin < stop <= origin + n / rate or limit < 1.0:
            return None
        if metadata.get('cue_speech_span') != [begin, stop]:
            return None
        a, b = spans[0]
        if not 0 <= a < b <= n or not math.ceil(rate) <= b - a <= math.floor(limit * rate):
            return None
        left, right = origin + a / rate, origin + b / rate
        declared = metadata.get('source_interval')
        if not isinstance(declared, (list, tuple)) or len(declared) != 2:
            return None
        if not all(math.isfinite(float(v)) for v in declared):
            return None
        if abs(float(declared[0]) - left) > 1e-8 or abs(float(declared[1]) - right) > 1e-8:
            return None
        # Validate every raw turn before using labels to exclude another voice.
        turns = []
        raw_turns = context['raw_turns']
        if not isinstance(raw_turns, (list, tuple)):
            return None
        for turn in raw_turns:
            x, y, label = float(turn['start']), float(turn['end']), turn['speaker']
            if not math.isfinite(x) or not math.isfinite(y) or x >= y or not isinstance(label, str) or not label:
                return None
            turns.append((x, y, label))
        words = cue.get('words')
        if not isinstance(words, (list, tuple)) or not words:
            raise _LineReferenceRefused('reference_not_one_raw_speaker')
        scores = {}
        for word in words:
            try:
                word_start, word_end = float(word['start']), float(word['end'])
            except (KeyError, TypeError, ValueError, OverflowError):
                raise _LineReferenceRefused('reference_not_one_raw_speaker')
            if not math.isfinite(word_start) or not math.isfinite(word_end) or word_end <= word_start:
                raise _LineReferenceRefused('reference_not_one_raw_speaker')
            for x, y, raw_label in turns:
                amount = max(0.0, min(word_end, y) - max(word_start, x))
                if amount > 0:
                    scores[raw_label] = scores.get(raw_label, 0.0) + amount
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        if not ranked or (len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= 1e-9):
            raise _LineReferenceRefused('reference_not_one_raw_speaker')
        dominant = ranked[0][0]
        labels = {label for x, y, label in turns if min(y, right) > max(x, left)}
        if labels != {dominant}:
            raise _LineReferenceRefused('reference_not_one_raw_speaker')
        label = dominant
        if any(other != label and min(y, right) > max(x, left) for x, y, other in turns):
            return None
        # Derive every different-label intersection, without an issue-list threshold.
        overlaps = [(max(x, u), min(y, v)) for index, (x, y, label_a) in enumerate(turns)
                    for u, v, label_b in turns[index + 1:]
                    if label_a != label_b and min(y, v) > max(x, u)]
        for source_cue in context.get('all_cues') or ():
            overlaps.extend(source_cue.get('overlap_intervals') or ())
        for pair in overlaps:
            x, y = map(float, pair)
            if not math.isfinite(x) or not math.isfinite(y) or x >= y:
                return None
            if min(y, right) > max(x, left):
                return None
        own = (math.ceil((begin - origin) * rate - 1e-7),
               math.floor((stop - origin) * rate + 1e-7))
        padded = (max(0, round((begin - origin - PAD) * rate)),
                  min(n, round((stop - origin + PAD) * rate)))
        asserted_turn = metadata.get('raw_turn')
        # A short-cue extension may coincidentally equal PAD geometry. Its real
        # containing-turn evidence must still be checked and its text kept empty.
        exact = (a, b) == own or ((a, b) == padded and asserted_turn is None)
        trimmed = a >= own[0] and b <= own[1]
        if exact or trimmed:
            if asserted_turn is not None:
                return None
        else:
            # An extension must contain the whole cue and remain in one real solo turn.
            if a > own[0] or b < own[1] or not isinstance(asserted_turn, dict):
                return None
            actual = (float(asserted_turn['start']), float(asserted_turn['end']), asserted_turn['speaker'])
            if actual not in turns or actual[2] != label:
                return None
            x, y, _ = actual
            if not x <= begin < stop <= y or left < max(origin, x) or right > min(origin + n / rate, y):
                return None
        basis = metadata.get('transcript_basis')
        if cue.get('unclear'):
            expected = '' if basis == 'reference-only-unclear' else None
        # The ASR text belongs to the complete cue, not an inward speech subset.
        elif exact and begin == start and stop == end:
            expected = cue['text'] if basis == 'exact-cue-asr' else None
        else:
            expected = '' if basis == 'reference-only-expanded-or-trimmed' else None
        if 'sample_span' in metadata and metadata['sample_span'] != [a, b]:
            return None
        if 'reference_text' in metadata and metadata['reference_text'] != expected:
            return None
        return expected
    except _LineReferenceRefused:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


def _expected_text(context, ident, spans):
    """make_reference text semantics from the contributing cues (never English fields)."""
    voice = (context.get('voices') or {}).get(ident) or {}
    if str(ident).startswith('line-') or 'line_reference' in voice:
        return _line_expected_text(context, ident, spans, voice)
    origin, rate, n = float(context['origin']), int(context['rate']), len(context['speech'])
    found = []
    for a, b in spans:
        hit = None
        for cue in context.get('all_cues') or ():
            if cue.get('speaker') != ident or cue.get('overlapped'):
                continue
            begin, end = cue.get('speech_start', cue.get('start')), cue.get('speech_end', cue.get('end'))
            if begin is None or end is None:
                continue
            x = max(0, round((begin - origin - PAD) * rate))
            y = min(n, round((end - origin + PAD) * rate))
            if (x, y) == (a, b):
                hit = cue
                break
        if hit is None:
            return None
        found.append(hit)
    if any(c.get('unclear') for c in found):
        return ''
    return ' '.join(c['text'] for c in found)


def source_bound_receipt(context, ident):
    """Receipt to persist right after make_reference/attach_reference (creation window only).

    Binds the reference to the stable source (fingerprint/audio index), job/revision, model identity and
    the creation window's OWN source_wav + speech stem files, origin, exact ranges, channel and text.
    """
    bad = validate_context(context)
    if bad:
        raise ValueError('no receipt: ' + bad)
    voice = context['voices'][ident]
    if (str(ident).startswith('line-') or 'line_reference' in voice) and 'raw_turns' not in context:
        raise ValueError('admission_context_missing_raw_turns')
    data, rate, pcm = _decoded(voice['reference'])
    built = _reconstruct(context['speech'], voice['source_intervals'], float(context['origin']), int(context['rate']))
    channel = _match_channel(built, data)
    if channel is None:
        raise ValueError('reference is not an exact excerpt of this speech stem; no receipt')
    expected = _expected_text(context, ident, built[1])
    if expected is None or expected != voice.get('reference_text', ''):
        raise ValueError('reference_text != contributing cue transcript; no receipt')
    sid = context['source_identity']
    receipt = {'version': RECEIPT_VERSION, 'job_id': context['job_id'], 'revision': context['revision'],
            'actor_id': ident, 'source_fingerprint': sid['source_fingerprint'], 'audio_index': sid['audio_index'],
            'model': _model_identity(context['model_provenance']),
            'origin': float(context['origin']), 'rate': int(context['rate']),
            'source_wav': {'path': str(context['source_wav']['path']), 'sha256': context['source_wav']['sha256']},
            'speech_stem': {'path': str(context['speech_stem']['path']), 'sha256': context['speech_stem']['sha256']},
            'reference_sha256': file_sha256(voice['reference']), 'reference_pcm_sha256': pcm,
            'reference_text': voice.get('reference_text', ''),
            'intervals': [list(map(float, i)) for i in voice['source_intervals']], 'channel': channel}
    if str(ident).startswith('line-') or 'line_reference' in voice:
        # Preserve the original creation-window evidence for independent replay on carry.
        # Do not capture mutable translation, rendering or reference-admission fields.
        cue_fields = ('id', 'start', 'end', 'speech_start', 'speech_end', 'speaker',
                      'text', 'unclear', 'overlapped', 'turn_overlap', 'overlap_intervals')
        evidence_cues = []
        for cue in sorted(context['all_cues'], key=lambda item: item['id']):
            projected = {key: cue[key] for key in cue_fields if key in cue}
            projected['words'] = [{key: word[key] for key in ('start', 'end') if key in word}
                                  for word in cue.get('words') or ()]
            evidence_cues.append(projected)
        receipt['line_reference'] = json.loads(json.dumps(voice['line_reference']))
        receipt['line_reference_evidence'] = json.loads(json.dumps({
            'all_cues': evidence_cues,
            'raw_turns': [{key: turn[key] for key in ('start', 'end', 'speaker')}
                          for turn in context['raw_turns']],
            'actor_reference_seconds': context['actor_reference_seconds']}))
    return receipt


def persist_receipt(receipt, directory):
    """Program-owned automatic persistence (atomic, one file per actor). No approval step."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{receipt['actor_id']}.json"
    tmp = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    os.replace(tmp, target)
    return target


def load_receipts(directory, idents):
    out = {}
    for ident in idents:
        f = Path(directory) / f'{ident}.json'
        if f.is_file():
            try:
                out[ident] = json.loads(f.read_text(encoding='utf-8'))
            except Exception:
                pass  # unreadable receipt -> treated as absent -> fallback
    return out


def _receipt_problem(context, ident, voice, receipt, data, pcm, ref_sha, roots):
    """Cross-window proof: stable source/job/revision/model match, then exact rebuild from the receipt's
    OWN creation-window stem at the receipt's OWN origin (never reinterpreted against the current origin)."""
    if not isinstance(receipt, dict) or receipt.get('version') != RECEIPT_VERSION:
        return 'cross-window reference without source-bound receipt'
    sid = context['source_identity']
    want = {'job_id': context['job_id'], 'revision': context['revision'], 'actor_id': ident,
            'source_fingerprint': sid['source_fingerprint'], 'audio_index': sid['audio_index'],
            'model': _model_identity(context['model_provenance']), 'rate': int(context['rate']),
            'reference_sha256': ref_sha, 'reference_pcm_sha256': pcm, 'reference_text': voice.get('reference_text', ''),
            'intervals': [list(map(float, i)) for i in voice['source_intervals']]}
    for key, value in want.items():
        if receipt.get(key) != value:
            return f'receipt {key} mismatch'
    for what in ('source_wav', 'speech_stem'):
        _, bad = _file_bound(receipt.get(what), roots, 'receipt ' + what)
        if bad:
            return bad
    import soundfile as sf
    stem, rate = sf.read(str(receipt['speech_stem']['path']), dtype='float32', always_2d=True)
    if rate != receipt['rate']:
        return 'receipt stem rate mismatch'
    built = _reconstruct(stem, voice['source_intervals'], float(receipt['origin']), rate)
    channel = _match_channel(built, data)
    if channel is None or channel != receipt.get('channel'):
        return 'reference is not an exact excerpt of the receipt stem'
    if str(ident).startswith('line-') or 'line_reference' in voice:
        if receipt.get('line_reference') != voice.get('line_reference'):
            return 'receipt line_reference mismatch'
        evidence = receipt.get('line_reference_evidence')
        if not isinstance(evidence, dict):
            return 'line reference without creation-window cue/turn evidence'
        if 'raw_turns' not in evidence:
            return 'admission_context_missing_raw_turns'
        # Existing source/job/model/hash checks and exact PCM reconstruction above
        # still apply. Recompute geometry/text from the receipt's original evidence.
        creation = {**evidence, 'origin': receipt['origin'], 'rate': rate,
                    'speech': stem, 'voices': {ident: voice}}
        try:
            expected = _expected_text(creation, ident, built[1])
        except _LineReferenceRefused as error:
            return str(error)
        if expected is None or expected != receipt['reference_text']:
            return 'receipt line reference geometry or transcript invalid'
        try:
            limit = float(context['actor_reference_seconds'])
            if not math.isfinite(limit) or limit < 1.0 or len(data) > math.floor(limit * rate):
                return 'line reference exceeds current configured duration limit'
        except (KeyError, TypeError, ValueError, OverflowError):
            return 'line reference current duration limit missing or malformed'
    return None


def _check_actor(context, ident, voice, roots):
    rec = {'actor_id': ident, 'identity_verified': False, 'admitted': False}
    if not isinstance(voice, dict):
        return {**rec, 'reason': 'voice is not a mapping'}
    if (str(ident).startswith('line-') or 'line_reference' in voice) and 'raw_turns' not in context:
        return {**rec, 'reason': 'admission_context_missing_raw_turns'}
    if any(k in voice for k in PERSON_FIELDS):
        return {**rec, 'reason': 'voice carries person/profile/owner field'}
    if voice.get('engine') != 'actor':
        return {**rec, 'reason': 'voice engine is not actor'}
    path = voice.get('reference')
    if not path or not Path(path).is_file():
        return {**rec, 'reason': 'reference file missing'}
    if not path_allowed(path, roots):
        return {**rec, 'reason': 'reference path outside movie roots or under a private segment'}
    ref_sha = file_sha256(path)
    rec.update(reference_sha256=ref_sha, intervals=voice.get('source_intervals'))
    if ref_sha != str(voice.get('reference_sha256') or '').lower():
        return {**rec, 'reason': 'file sha256 != voice reference_sha256'}
    try:
        data, rate, pcm = _decoded(path)
    except Exception as exc:
        return {**rec, 'reason': f'reference unreadable: {exc}'}
    rec['reference_pcm_sha256'] = pcm
    intervals = voice.get('source_intervals') or []
    if data.shape[1] != 1 or rate != int(context['rate']) or not intervals:
        return {**rec, 'reason': 'reference geometry/intervals malformed'}
    built = _reconstruct(context['speech'], intervals, float(context['origin']), int(context['rate']))
    channel = _match_channel(built, data)
    if channel is None:
        receipt = (context.get('reference_receipts') or {}).get(ident)
        if receipt is None and context.get('receipt_dir'):
            receipt = load_receipts(context['receipt_dir'], [ident]).get(ident)
        bad = _receipt_problem(context, ident, voice, receipt, data, pcm, ref_sha, roots)
        if bad:
            return {**rec, 'reason': bad}
        rec.update(channel=receipt['channel'], proof='persisted receipt + exact rebuild from receipt stem')
        expected = receipt['reference_text']  # bound and checked at creation; cues may lie outside this window
    else:
        rec.update(channel=channel, proof='exact rebuild from current validated speech stem')
        try:
            expected = _expected_text(context, ident, built[1])
        except _LineReferenceRefused as error:
            return {**rec, 'reason': str(error)}
        if expected is None:
            return {**rec, 'reason': 'reference intervals do not map to cues of this speaker'}
    if voice.get('reference_text', '') != expected:
        return {**rec, 'reason': 'reference_text != contributing cue transcript'}
    rec.update(admitted=True, reason=None, transcript_empty=expected == '')
    return rec


def _bank_caption_unit(cue, voice, context):
    """A complete English unit may use an already-admitted bank donor's true transcript."""
    try:
        from .dialogue import caption_text
        if not isinstance(context, dict) or not isinstance(voice, dict) or 'line_reference' in voice:
            return False
        ident = cue['speaker']
        if not isinstance(ident, str) or not re.fullmatch(r'speaker-[0-9]+', ident) \
                or voice.get('actor_id') != ident or voice.get('engine') != 'actor':
            return False
        source = cue['voice_source']
        local = source['local_speaker']
        if source.get('version') != 'voice-source-v1' or source.get('mode') != 'bank' \
                or source.get('bank_ident') != ident or not isinstance(local, str) or not local:
            return False
        bindings = context['speaker_binding_evidence']
        decision = bindings['local_decisions'][local]
        bound_cue = bindings['cues'][cue['id']]
        if decision.get('reason') not in {'bound_existing', 'new_identity', 'known_ok'} \
                or decision.get('decision') != ident or source.get('bind_reason') != decision['reason'] \
                or bound_cue.get('local_speaker') != local or bound_cue.get('mapped_actor') != ident \
                or bound_cue.get('attempted') is not True:
            return False
        unit = cue['dialogue_unit']
        basis = unit.get('basis')
        if unit.get('version') != 'dialogue-unit-v1' or unit.get('local_speaker') != local \
                or basis not in {'aligned-caption', 'caption-dash-part'} or cue.get('overlapped') \
                or cue.get('turn_overlap') or cue.get('singing'):
            return False
        members = unit['member_cue_ids']
        if not isinstance(members, list) or not members or len(set(members)) != len(members) \
                or cue.get('source_cue_ids') != members:
            return False
        expected_id = _sha(json.dumps(['dialogue-unit-v1', members], separators=(',', ':')).encode())[:24]
        if cue['id'] != expected_id:
            return False

        def caption_key(caption):
            start, end = float(caption['start']), float(caption['end'])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError('invalid caption span')
            return start, end, _sha(str(caption['text']).encode('utf-8')), str(caption.get('language', '')).lower()

        caption = cue['caption']
        key = caption_key(caption)
        if key[3] not in {'en', 'eng'} or unit['caption'] != {'start': key[0], 'end': key[1], 'text_sha256': key[2]}:
            return False
        raw_cues = sorted(context['dialogue_raw_cues'], key=lambda c: (c['start'], c['end'], c['id']))
        indices = [index for index, c in enumerate(raw_cues)
                   if isinstance(c.get('caption'), dict) and caption_key(c['caption']) == key]
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            return False
        related = [raw_cues[index] for index in indices]
        by_id = {c['id']: c for c in related}
        if len(by_id) != len(related) or any(ident not in by_id for ident in members):
            return False
        chosen = [by_id[ident] for ident in members]
        if chosen != sorted(chosen, key=lambda c: (c['start'], c['end'], c['id'])) \
                or any(c.get('speaker') != local or c.get('overlapped') or c.get('turn_overlap')
                       or c.get('singing') for c in chosen):
            return False
        start, end = min(c['start'] for c in chosen), max(c['end'] for c in chosen)
        if cue['start'] != start or cue['end'] != end \
                or cue.get('speech_start') != start or cue.get('speech_end') != end \
                or cue['text'] != ' '.join(c['text'].strip() for c in chosen).strip():
            return False
        words = [{**word, 'source_cue_id': member['id']} for member in chosen for word in member.get('words') or ()]
        words.sort(key=lambda word: (float(word['start']), float(word['end'])))
        if cue.get('words') != words or bool(cue.get('unclear')) != any(c.get('unclear') for c in chosen):
            return False
        window = context['window']
        render_end = float(window['render_end'])
        if not math.isfinite(render_end) or any(float(member['start']) >= render_end for member in chosen):
            return False
        owner_start = max(float(context['origin']), float(window['render_start']))
        visible_end = float(window['end']) - .5
        if not float(context['origin']) <= key[0] < key[1] <= visible_end \
                or not owner_start <= start < end <= visible_end:
            return False
        if cue.get('translation') != 'aligned-English-caption' or not isinstance(cue.get('english'), str) \
                or not cue['english'].strip() or cue['english'] != cue.get('caption_fallback'):
            return False
        turns = [(float(t['start']), float(t['end']), t['speaker']) for t in context['raw_turns']]
        if any(not math.isfinite(a) or not math.isfinite(b) or b <= a
               or not isinstance(label, str) or not label for a, b, label in turns):
            return False

        def dominant(group):
            scores = {}
            words = [w for member in group for w in member.get('words') or ()]
            if not words:
                raise ValueError('missing raw speaker evidence')
            for word in words:
                a, b = float(word['start']), float(word['end'])
                if not math.isfinite(a) or not math.isfinite(b) or b <= a:
                    raise ValueError('invalid word span')
                for x, y, label in turns:
                    amount = max(0., min(b, y) - max(a, x))
                    if amount:
                        scores[label] = scores.get(label, 0.) + amount
            ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            if not ranked or (len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= 1e-9):
                raise ValueError('ambiguous raw speaker')
            return ranked[0][0]

        raw_labels = {dominant([member]) for member in chosen}
        if len(raw_labels) != 1:
            return False
        raw_label = next(iter(raw_labels))
        evidence = unit['evidence']
        if evidence.get('member_spans') != [[member['start'], member['end']] for member in chosen] \
                or evidence.get('raw_speaker') != raw_label:
            return False
        if any(label != raw_label and min(b, end) - max(a, start) > .12 for a, b, label in turns):
            return False
        if basis == 'aligned-caption':
            if [c['id'] for c in related] != members or unit.get('dash_part') is not None \
                    or re.search(r'(?:^|\s)-\s+', str(caption['text'])):
                return False
            expected = caption_text(caption['text'])
        else:
            parts = [caption_text(part) for part in re.split(r'(?:^|\s)-\s+', str(caption['text']))
                     if caption_text(part)]
            runs = []
            for member in related:
                if not runs or runs[-1][0]['speaker'] != member['speaker']:
                    runs.append([])
                runs[-1].append(member)
            part = unit.get('dash_part')
            if isinstance(part, bool) or not isinstance(part, int) or not 0 <= part < len(parts) \
                    or len(parts) < 2 or len(parts) != len(runs) or runs[part] != chosen:
                return False
            for index, group in enumerate(runs):
                fallbacks = [caption_text(c['caption_fallback']) for c in group if c.get('caption_fallback')]
                if not fallbacks or any(value != parts[index] for value in fallbacks):
                    return False
                labels = {dominant([member]) for member in group}
                if len(labels) != 1 or any(c.get('overlapped') or c.get('turn_overlap') or c.get('singing') for c in group):
                    return False
                group_start, group_end = min(c['start'] for c in group), max(c['end'] for c in group)
                if not owner_start <= group_start < group_end <= visible_end:
                    return False
                label = next(iter(labels))
                if any(other != label and min(b, group_end) - max(a, group_start) > .12 for a, b, other in turns):
                    return False
            for left, right in zip(runs, runs[1:]):
                if right[0]['start'] - left[-1]['end'] < .25:
                    return False
                left_label, right_label = dominant([left[-1]]), dominant([right[0]])
                if left_label == right_label:
                    return False
                lo, hi = left[-1]['end'] - .12, right[0]['start'] + .12
                if not any(label == left_label and lo <= b <= hi for a, b, label in turns) \
                        or not any(label == right_label and lo <= a <= hi for a, b, label in turns):
                    return False
            pause = None if part == 0 else runs[part][0]['start'] - runs[part - 1][-1]['end']
            if evidence.get('dash_boundary_pause_seconds') != pause or evidence.get('dash_part_count') != len(parts) \
                    or evidence.get('caption_part_verified') is not True:
                return False
            expected = parts[part]
        return bool(expected) and cue['english'] == expected
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False


def _cue_bad(cue, voice=None, context=None):
    if not isinstance(cue, dict) or not cue.get('id') or not isinstance(cue.get('text'), str):
        return 'malformed cue'
    try:
        if float(cue['end']) <= float(cue['start']):
            return 'malformed cue'
    except (KeyError, TypeError, ValueError):
        return 'malformed cue'
    if not cue.get('speaker'):
        return 'missing speaker'
    if isinstance(voice, dict) and (str(cue['speaker']).startswith('line-') or 'line_reference' in voice):
        metadata = voice.get('line_reference')
        if not isinstance(metadata, dict) or metadata.get('cue_id') != cue['id'] \
                or cue['speaker'] != 'line-' + str(cue['id']):
            return 'line reference belongs to another cue'
    if cue.get('unclear') and not (
            cue.get('translation') == 'aligned-English-caption'
            and isinstance(voice, dict) and voice.get('reference_text') == '') \
            and not _bank_caption_unit(cue, voice, context):
        return 'unclear cue'
    if cue.get('overlapped'):
        return 'overlapped cue'
    return None


def make_pipeline_gate(context, *, reference_cache_module=None):
    """See evidence/supervisor-b-movie-beta/dynamic-admission/INTERFACE.md."""
    if 'trusted' in context:
        raise ValueError('caller trust flags are refused')
    rc = reference_cache_module or _load_reference_cache()
    issuer = rc.TrustIssuer(secrets.token_bytes(32), 'supervisor-b-pipeline-gate-ephemeral')
    roots = context.get('allowed_roots') or ()
    refused = validate_context(context)
    envelopes, admissions, voices, actors = {}, {}, {}, {}
    for ident, voice in (context.get('voices') or {}).items():
        if refused:
            actors[ident] = {'actor_id': ident, 'identity_verified': False, 'admitted': False,
                             'reason': 'context refused: ' + refused}
            continue
        rec = _check_actor(context, ident, voice, roots)
        actors[ident] = rec
        if not rec['admitted']:
            continue
        env = {'source_id': _source_id(context), 'job_id': _job(context), 'actor_id': ident,
               'reference_path': voice['reference'], 'reference_sha256': rec['reference_sha256'],
               'reference_text': voice.get('reference_text', ''),
               'verified_ranges': [list(i) for i in voice['source_intervals']], 'status': 'verified'}
        envelopes[ident] = env
        admissions[ident] = issuer.mint(env['source_id'], env['job_id'], ident, env['reference_sha256'],
                                        rc.text_sha256(env['reference_text']), env['verified_ranges'])
        voices[ident] = {**voice, 'actor_id': ident}
    gate = rc.make_reference_gate(envelopes, admissions, issuer)
    decisions = {}
    for cue in context.get('cues') or ():
        cid = cue.get('id') if isinstance(cue, dict) else None
        voice = voices.get(cue.get('speaker')) if isinstance(cue, dict) else None
        bad = _cue_bad(cue, voice, context)
        if bad is None and cue['speaker'] not in actors:
            bad = 'unknown speaker'
        elif bad is None and not actors[cue['speaker']]['admitted']:
            bad = 'speaker reference not admitted: ' + actors[cue['speaker']]['reason']
        decisions[cid] = ({'mode': 'original-audio-fallback', 'reason': bad} if bad
                          else {'mode': 'actor-reference', 'voice': voices[cue['speaker']]})

    def resolve_cue(cue_id):
        return decisions.get(cue_id) or {'mode': 'original-audio-fallback', 'reason': 'unknown cue'}

    admitted = [k for k, v in decisions.items() if v['mode'] == 'actor-reference']
    fallbacks = {k: v['reason'] for k, v in decisions.items() if v['mode'] != 'actor-reference'}
    prov = context.get('model_provenance') if isinstance(context.get('model_provenance'), dict) else None
    receipt = {'version': GATE_VERSION, 'status': 'provenance-only', 'context_refused': refused,
               'meaning': 'reference is an exact source-bound excerpt; speaker identity, quality and accuracy NOT verified',
               'issuer_id': issuer.issuer_id, 'issuer_scope': 'ephemeral-per-call',
               'job_id': None if refused else _job(context), 'source_id': None if refused else _source_id(context),
               'source_identity': context.get('source_identity'),
               'source_wav': context.get('source_wav'), 'speech_stem': context.get('speech_stem'),
               'model_provenance': prov, 'actors': actors, 'admitted_cues': admitted, 'fallbacks': fallbacks}
    return {'voices': voices, 'reference_gate': gate, 'admitted_cues': admitted, 'fallbacks': fallbacks,
            'receipt': receipt, 'resolve_cue': resolve_cue}
