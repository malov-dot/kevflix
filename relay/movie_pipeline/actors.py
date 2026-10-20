"""Actor identity for speculative window preparation (relay.movie_pipeline, W10; moved from movie-trial-v1/pipeline).

Policy: the stage's own relay.speakers.SpeakerBank. That covers configured
speaker_similarity and speaker_margin, one-to-one per window, invalid/near-zero vectors refused, KNOWN:: reference
evidence, speaker-NNN ids, the 4:1 running average and the first reference frozen. Nothing here adds a matching rule.

Two parts:

capture()
  Thread-local recorder installed around SpeakerBank.bind / attach_reference in process P. The analysis worker runs
  ActorPipeline.process(prepare_only=True) on a PRIVATE snapshot (snapshot.py) with a capture active. It records:
  * the exact vectors passed to bind (dict order kept);
  * the mapping and problems bind returned;
  * every first reference attached (ident -> path, sha256, text, source intervals).
  Those writes go only to the snapshot.

commit_actors()
  The coordinator thread only, before any render of that window. W4's ActorBank.commit transaction
  (parallel-pipeline-v1/ppipe/actors.py, ppipe-actor-bank-v2) is adapted to the real stage relay and the v4 chunk
  fence. In ONE `BEGIN IMMEDIATE` on the job Store, with the fence checked (beta_line_ledger._begin):
  1. Base check. The committed bank's reference set (ident -> reference sha256) must equal the snapshot base's.
     Preparation's speaker refinement read those references, so a changed set means the analysis itself would differ.
  2. Merge validated schema-3 reference evidence, including a changed embedding space even when the reference WAV
     is unchanged. A legacy centroid migrates from reference evidence only; an already-current running centroid
     stays intact. The real SpeakerBank then re-binds the captured vectors against the CURRENT committed bank.
  3. Compare. For each label, (speaker id, reference sha256) is compared with what the prepared plan used.
  4. Equal: the bind updates are written, the plan's first references are attached for the new speakers (and
     reference-vector caches computed for identical reference bytes are copied), then commit. The plan is valid
     against the bank.
  5. Different: rollback, so nothing is written, and the prepared plan is renamed. The render then re-decides the
     window on the real bank through the relay's own serial path (ActorPipeline._process without a reusable plan).
  The render additionally re-validates the plan against the bank: ActorPipeline._process reuses a plan only when its
  key AND every voice equal store.speakers(job). That is the relay's own check, kept as the final guard.
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path

VERSION = 'movie-trial-actor-commit-v2'
_LOCAL = threading.local()
_INSTALLED = {}


class Capture:
    def __init__(self, base_speakers):
        self.base = copy.deepcopy(base_speakers)
        self.binds = []          # [(vectors, mapping, problems)]
        self.attached = {}       # ident -> {'path','sha256','text','source_intervals'}

    def as_record(self):
        return {'binds': [{'labels': list(v), 'mapping': m, 'problems': p} for v, m, p in self.binds],
                'attached': {k: {kk: vv for kk, vv in r.items()} for k, r in self.attached.items()},
                'base_reference_sha256': reference_set(self.base)}


def reference_set(speakers):
    return {ident: (value.get('voice') or {}).get('reference_sha256') for ident, value in sorted(speakers.items())}


def install(speakers_module):
    """Wrap SpeakerBank.bind / attach_reference once; recording only happens while a capture is active."""
    cls = speakers_module.SpeakerBank
    if _INSTALLED.get(id(cls)):
        return
    real_bind, real_attach = cls.bind, cls.attach_reference

    def bind(self, vectors):
        mapping, problems = real_bind(self, vectors)
        capture = getattr(_LOCAL, 'capture', None)
        if capture is not None:
            capture.binds.append((copy.deepcopy(dict(vectors)), dict(mapping), copy.deepcopy(problems)))
        return mapping, problems

    def attach_reference(self, ident, path, transcript='', *, source_intervals=()):
        frozen = bool(((self.known.get(ident) or {}).get('voice') or {}).get('reference'))
        voice = real_attach(self, ident, path, transcript, source_intervals=source_intervals)
        capture = getattr(_LOCAL, 'capture', None)
        if capture is not None and not frozen:
            capture.attached[ident] = {'path': str(path), 'sha256': voice.get('reference_sha256'),
                                       'text': transcript, 'source_intervals': list(source_intervals)}
        return voice

    bind._trial_capture = attach_reference._trial_capture = True
    cls.bind, cls.attach_reference = bind, attach_reference
    _INSTALLED[id(cls)] = True


class capture:
    """with capture(base_speakers) as cap: ... (analysis worker thread only)."""

    def __init__(self, base_speakers):
        self.cap = Capture(base_speakers)

    def __enter__(self):
        _LOCAL.capture = self.cap
        return self.cap

    def __exit__(self, *exc):
        _LOCAL.capture = None
        return False


class TxStore:
    """store.speakers()/put_speaker() inside the open transaction, with relay/store.py's own SQL (W4 ppipe TxStore)."""

    def __init__(self, db):
        self.db = db
        self.writes = []

    def speakers(self, job_id):
        return {r['id']: json.loads(r['payload']) for r in self.db.execute(
            'SELECT * FROM speakers WHERE job_id=?', (job_id,))}

    def put_speaker(self, job_id, ident, payload):
        self.db.execute('INSERT INTO speakers VALUES(?,?,?) ON CONFLICT(job_id,id) '
                        'DO UPDATE SET payload=excluded.payload', (job_id, ident, json.dumps(payload)))
        self.writes.append(ident)


class _Changed(Exception):
    def __init__(self, detail):
        super().__init__(detail)
        self.detail = detail


def _view(mapping, known, references):
    """label -> (speaker id or None, reference sha256 the line renders with) (W4 ppipe _view)."""
    out = {}
    for label, ident in mapping.items():
        if ident is None:
            out[label] = (None, None)
            continue
        voice = (known.get(ident) or {}).get('voice') or {}
        out[label] = (ident, voice.get('reference_sha256') or (references.get(label) or {}).get('sha256'))
    return out


CACHE_IDENTITY_KEYS = ('reference_vectors_sha256', 'reference_vectors_version', 'reference_vectors_space')
CACHE_KEYS = ('reference_vectors',) + CACHE_IDENTITY_KEYS


def _valid_vectors(vectors):
    """Only consistent finite nonzero evidence may cross from a speculative snapshot."""
    if not isinstance(vectors, (list, tuple)) or not vectors:
        return False
    if not isinstance(vectors[0], (list, tuple)) or not vectors[0]:
        return False
    width = len(vectors[0])
    try:
        return all(isinstance(row, (list, tuple)) and len(row) == width
                   and all(math.isfinite(x) for x in row)
                   and 1e-8 <= sum(x * x for x in row) < math.inf for row in vectors)
    except (TypeError, ValueError, OverflowError):
        return False


def _reference_refresh(ident, real, prepared, embedding_space):
    """Copy observation metadata only; never copy a snapshot's post-bind running centroid or voice."""
    ref = (real.get('voice') or {}).get('reference_sha256')
    expected = (ref, 3, embedding_space)
    if (not ref or (prepared.get('voice') or {}).get('reference_sha256') != ref
            or tuple(prepared.get(k) for k in CACHE_IDENTITY_KEYS) != expected):
        return {}
    if not _valid_vectors(prepared.get('reference_vectors')):
        raise _Changed({'kind': 'reference_vectors_invalid', 'ident': ident})
    changes = {}
    if (tuple(real.get(k) for k in CACHE_IDENTITY_KEYS) != expected
            or not _valid_vectors(real.get('reference_vectors'))):
        changes.update({k: copy.deepcopy(prepared[k]) for k in CACHE_KEYS})
    if real.get('embedding_space') != embedding_space:
        # Snapshot centroids already include its bind. Rebuild only the initial
        # migration mean, matching SpeakerBank.reference_vectors, so the bind
        # below applies this window's 4:1 running update exactly once.
        vectors = changes['reference_vectors'] if 'reference_vectors' in changes else real['reference_vectors']
        centroid = [sum(column) / len(vectors) for column in zip(*vectors)]
        if not _valid_vectors([centroid]):
            raise _Changed({'kind': 'reference_centroid_invalid', 'ident': ident})
        changes.update(embedding=centroid, embedding_space=embedding_space)
    return changes


def commit_actors(store, fence, job_id, config, *, speakers_module, ledger_module, prepared):
    """-> {'status': 'committed'|'changed'|'no_plan'|'not_bound', ...}. 'changed' = nothing written; the caller
    invalidates the plan. Raises beta_line_ledger.Superseded when the chunk fence no longer holds."""
    record = {'version': VERSION, 'at': time.time()}
    if prepared.state != 'prepared' or not prepared.plan_path:
        return {**record, 'status': 'no_plan', 'reason': 'preparation stopped at %s' % prepared.state}
    cap = prepared.capture
    if not cap.binds and prepared.plan_reused:
        # ActorPipeline reused an existing valid plan (same key; every voice equal to the snapshot bank) and never
        # bound. Read-only check against the committed bank; the render re-checks it again (_process plan reuse).
        with store.connect() as db:
            current = TxStore(db).speakers(job_id)
        if reference_set(current) != reference_set(cap.base) or any(
                (current.get(i) or {}).get('voice') != v for i, v in prepared.plan_voices.items()
                if not (i.startswith('line-') and
                        (v.get('line_reference') or {}).get('version') == 'line-own-reference-v1')):
            return {**record, 'status': 'changed', 'reason': {'kind': 'reused_plan_differs_from_bank'}}
        return {**record, 'status': 'committed', 'mapping': {}, 'problems': [], 'attached': [], 'writes': [],
                'labels': 0, 'new_speakers': [], 'plan_reused': True}
    if len(cap.binds) != 1:
        return {**record, 'status': 'changed', 'reason': 'expected exactly one SpeakerBank.bind, saw %d' % len(cap.binds)}
    vectors, used_mapping, _ = cap.binds[0]
    voices = prepared.plan_voices
    references = {label: cap.attached[ident] for label, ident in used_mapping.items() if ident in cap.attached}
    used_view = {label: (ident, (voices.get(ident) or {}).get('reference_sha256') if ident else None)
                 for label, ident in used_mapping.items()}
    # Resolve/warm the backend's existing content-identity memo before opening
    # the write transaction; the merge itself performs no model I/O or inference.
    embedding_space = speakers_module.embedding_space_identity(config)
    try:
        with store.connect() as db:
            ledger_module._begin(db, fence, job_id)                  # BEGIN IMMEDIATE + v4 chunk fence
            tx = TxStore(db)
            current = tx.speakers(job_id)
            if reference_set(current) != reference_set(cap.base):
                raise _Changed({'kind': 'bank_references_changed_since_snapshot',
                                'now': reference_set(current), 'base': reference_set(cap.base)})
            for ident, value in (prepared.snapshot_speakers or {}).items():
                real = current.get(ident)
                if real is None:
                    continue
                changes = _reference_refresh(ident, real, value, embedding_space)
                if changes:
                    tx.put_speaker(job_id, ident, {**real, **changes})
            bank = speakers_module.SpeakerBank(tx, job_id, config)
            mapping, problems = bank.bind(vectors)
            now = _view(mapping, bank.known, references)
            if {k: tuple(v) for k, v in now.items()} != {k: tuple(v) for k, v in used_view.items()}:
                changed = sorted(l for l in set(now) | set(used_view) if tuple(now.get(l) or ()) != tuple(used_view.get(l) or ()))
                raise _Changed({'kind': 'actor_assignment_changed', 'labels': changed,
                                'now': {l: now.get(l) for l in changed}, 'used': {l: used_view.get(l) for l in changed}})
            attached = []
            for label, ident in mapping.items():
                ref = references.get(label)
                if ident is not None and ref and not bank.known[ident]['voice'].get('reference'):
                    if not Path(ref['path']).is_file():
                        raise _Changed({'kind': 'reference_file_missing', 'ident': ident})
                    voice = bank.attach_reference(ident, ref['path'], ref.get('text', ''),
                                                  source_intervals=ref.get('source_intervals', ()))
                    if voice.get('reference_sha256') != ref.get('sha256'):
                        raise _Changed({'kind': 'reference_bytes_changed', 'ident': ident})
                    attached.append(ident)
            for ident, voice in voices.items():
                # A line reference belongs to its source cue, not the persistent speaker bank. The render
                # gate and provenance verifier independently check its identity and exact source excerpt.
                if (ident.startswith('line-') and
                        (voice.get('line_reference') or {}).get('version') == 'line-own-reference-v1'):
                    continue
                if (bank.known.get(ident) or {}).get('voice') != voice:
                    raise _Changed({'kind': 'plan_voice_differs_from_bank', 'ident': ident})
            writes = list(tx.writes)
    except _Changed as change:
        return {**record, 'status': 'changed', 'reason': change.detail}
    return {**record, 'status': 'committed', 'mapping': mapping, 'problems': problems, 'attached': attached,
            'writes': writes, 'labels': len(mapping), 'new_speakers': sorted(set(attached))}


def invalidate_plan(plan_path, reason):
    """Rename the prepared plan so ActorPipeline._process cannot reuse it; the window is re-decided on the real bank."""
    path = Path(plan_path)
    if not path.is_file():
        return None
    n = 1
    while True:
        target = path.with_name('actor-prepared-plan.invalidated-%d.json' % n)
        if not target.exists():
            break
        n += 1
    path.replace(target)
    (target.with_suffix('.reason.json')).write_text(json.dumps({'reason': reason, 'at': time.time()}, default=str),
                                                     encoding='utf-8')
    return str(target)
