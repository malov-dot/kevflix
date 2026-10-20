"""Atomic cross-chunk line ledger (beta5-asr-gain-fix-v4; actor_beta.caption_targets only). Stdlib only (sqlite3).

Inert unless ActorPipeline calls it with caption_targets on. It adds one table, beta_line_claims, to the job Store's
SQLite file and never alters an existing table's schema. Contract for other callers (parallel pipeline): LEDGER-API.md.

Every write is ONE `BEGIN IMMEDIATE` transaction, so it is atomic and serialised across threads AND processes that
share the Store file; a process that dies inside a transaction leaves nothing (SQLite rolls it back).

Line claims (exactly-once synthesis and publication per line):
  claim_lines      BEFORE synthesis. A line (line_id, source span) is granted to one claimant. A claim of another
                   line whose source span is the same line (beta_dialogue_ownership.significant_overlap: ASR jitter
                   across a boundary) and that is published, or claimed by a live holder, blocks it ('published' /
                   'held'): the caller must not synthesize. A claim whose holder is dead (same host, process gone)
                   or whose lease expired is released and taken over (token + 1); its recorded artifacts stay.
  record_artifact  after synthesis, fenced by (holder run, token): the take file + receipt are stored in the claim,
                   so a retry, resume or takeover reuses them and never synthesizes the same line again.
  commit           publishes: the claim must still be held by this run with its token (else race 'claim_not_held');
                   then protection rows, canonical takes, published line claims and the window record are written
                   together. This run's remaining claims are released in the same transaction.
  release / renew  release this run's unpublished claims and window intent (kept artifacts); renew its leases.
  register_window  a window's intent row, written when its worker starts (before preparation), so a neighbour can
                   wait for the owner's decision (await_lines) instead of committing first and limiting its take.
  fenced_write     runs a file write (receipts, output) inside a fence-checked IMMEDIATE transaction.

Chunk fence (Codex review; W2 P2-1, P3-2): fence={'job_id','index','token','revision'} is the coordinator's chunk claim.
Every ledger write and fenced_write re-checks inside its own transaction that the job is not cancelled/error, still at
that revision, and that the chunk is 'working' with that claim token; otherwise nothing is written and Superseded (a
relay.process.Cancelled) is raised. Every claim and intent STORES its fence: once that fence fails, the claim counts as
stale for everyone (released and taken over by the next claimant), so a superseded worker can neither keep a line
'held' nor renew it.

Artifacts (W2 P2-3): a recorded take file must carry the per-claim name artifact_name(line_id, token, run_id, key), and
commit publishes a take only if its audio is that claim's recorded artifact with the recorded sha256 and the file still
hashes to it. A late write by an old holder to a shared renderer path can therefore never change a published take.

Commit race checks against other windows' commits (unchanged from the first v3 draft):
  canonical_takes_changed      a canonical take (DialogueLedger row of this job revision) that was not in the
                               snapshot read at render start and that this window's decisions depend on: its extent
                               (placement, removal, source) touches the render range (carry, covers, protection), or
                               its source is the same line as one of this window's owned units (dedupe).
  line_already_claimed         another claim on the same line is published or live-held (backstop; claim_lines
                               normally prevents this).
  take_overruns_into_committed_window
                               a new take ends after render_end inside the render range of another window that has
                               already committed; that window's output was made without the take.
  claim_not_held               this run no longer holds the line claim it is publishing (taken over or released).
  take_not_claimed_artifact    a new take's audio/sha256 is not an artifact recorded in its claim.
  take_file_changed            the claimed take file no longer hashes to its recorded sha256.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
import uuid

from .process import Cancelled
from .beta_dialogue_ownership import significant_overlap

VERSION = 'beta-line-ledger-v3'
TABLE = 'beta_line_claims'
# A claim is held this long without renew(); a holder that is alive renews it before each synthesis and at commit.
LEASE_SECONDS = 600.
HISTORY_LIMIT = 16
SCHEMA = ("CREATE TABLE IF NOT EXISTS beta_line_claims("
          "job_id TEXT NOT NULL, revision INTEGER NOT NULL, claim_id TEXT NOT NULL, kind TEXT NOT NULL, "
          "window_key TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL, state TEXT NOT NULL, "
          "holder TEXT, holder_pid INTEGER, holder_host TEXT, token INTEGER NOT NULL DEFAULT 0, "
          "lease_expires REAL NOT NULL DEFAULT 0, payload TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, "
          "PRIMARY KEY(job_id,revision,claim_id))",
          "CREATE INDEX IF NOT EXISTS beta_line_claims_span ON beta_line_claims(job_id,revision,kind,start,end)")
LIVE_STATES = ('claimed',)
LIVE_KINDS = ('line', 'intent')
FENCE_KEYS = ('job_id', 'index', 'token', 'revision')
CHUNK_LIVE_JOB_STATES_EXCLUDED = ('cancelled', 'error')


class Superseded(Cancelled):
    """The caller's chunk claim was cancelled, requeued or re-claimed; nothing was written."""

    def __init__(self, problem):
        super().__init__('chunk claim superseded: %s' % (problem,))
        self.problem = problem


def window_key(render_start, render_end):
    return 'w-%.3f-%.3f' % (float(render_start), float(render_end))


def new_claimant(window_key, run_id=None):
    """Identity of one render of one window (all its attempts): window key, unique run id, process id and host."""
    return {'window_key': str(window_key), 'run_id': run_id or uuid.uuid4().hex, 'pid': os.getpid(),
            'host': socket.gethostname()}


def window_range(key):
    """(render_start, render_end) of a window_key."""
    _, a, b = str(key).split('-')
    return float(a), float(b)


def fence_of(job_id, index, token, revision):
    return {'job_id': job_id, 'index': int(index), 'token': int(token), 'revision': int(revision)}


def artifact_name(line_id, token, run_id, key, suffix='.wav'):
    """The only file name record_artifact accepts: unique per line claim, token, holder run and artifact key."""
    return '%s-t%d-%s-%s%s' % (line_id, int(token), str(run_id)[:16], key, suffix)


def pid_alive(pid):
    """True unless the process is known to be gone. Unknown/denied counts as alive (wait for the lease instead)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                  # ERROR_ACCESS_DENIED: it exists
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                             # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def stale(row, *, now, alive=pid_alive, host=None, db=None):
    """A claimed row whose stored chunk fence failed (with ``db``: read in the caller's transaction), whose lease
    expired, or whose holder process on this host is gone."""
    if row.get('state') not in LIVE_STATES:
        return False
    stored = (row.get('payload') or {}).get('claim_fence')
    if db is not None and stored is not None and _check_fence(db, stored) is not None:
        return True
    if float(row.get('lease_expires') or 0) < float(now):
        return True
    return row.get('holder_host') == (host or socket.gethostname()) and not alive(row.get('holder_pid'))


def _decode(row):
    out = dict(row)
    out['payload'] = json.loads(out['payload'])
    return out


def _ensure(db):
    for statement in SCHEMA:
        db.execute(statement)


def _rows(db, job_id, revision, start, end, kind=None):
    query = "SELECT * FROM beta_line_claims WHERE job_id=? AND revision=? AND end>? AND start<?"
    args = [job_id, int(revision), float(start), float(end)]
    if kind is not None:
        query += " AND kind=?"
        args.append(kind)
    try:
        rows = db.execute(query + " ORDER BY start,claim_id", args).fetchall()
    except sqlite3.OperationalError as error:
        if 'no such table' in str(error).lower():
            return []
        raise
    return [_decode(r) for r in rows]


def _row(db, job_id, revision, claim_id):
    try:
        row = db.execute("SELECT * FROM beta_line_claims WHERE job_id=? AND revision=? AND claim_id=?",
                         (job_id, int(revision), claim_id)).fetchone()
    except sqlite3.OperationalError as error:
        if 'no such table' in str(error).lower():
            return None
        raise
    return _decode(row) if row is not None else None


def _history(payload, event):
    payload['history'] = (list(payload.get('history') or ()) + [event])[-HISTORY_LIMIT:]
    return payload


def _check_fence(db, fence):
    """None, or why the caller's chunk claim no longer allows writes (read inside the caller's transaction)."""
    if fence is None:
        return None
    missing = [k for k in FENCE_KEYS if k not in fence]
    if missing:
        raise ValueError('A chunk fence needs %s' % missing)
    job = db.execute("SELECT state,payload FROM jobs WHERE id=?", (fence['job_id'],)).fetchone()
    if job is None:
        return {'kind': 'job_missing', 'job_id': fence['job_id']}
    if job['state'] in CHUNK_LIVE_JOB_STATES_EXCLUDED:
        return {'kind': 'job_' + job['state'], 'job_id': fence['job_id']}
    revision = int(json.loads(job['payload']).get('revision', 1) or 1)
    if revision != int(fence['revision']):
        return {'kind': 'job_revision_changed', 'revision': revision, 'expected_revision': int(fence['revision'])}
    chunk = db.execute("SELECT state,payload FROM chunks WHERE job_id=? AND idx=?",
                       (fence['job_id'], int(fence['index']))).fetchone()
    if chunk is None:
        return {'kind': 'chunk_missing', 'index': int(fence['index'])}
    token = int(json.loads(chunk['payload']).get('claim_token') or 0)
    if chunk['state'] != 'working' or token != int(fence['token']):
        return {'kind': 'chunk_claim_superseded', 'index': int(fence['index']), 'state': chunk['state'],
                'token': token, 'expected_token': int(fence['token'])}
    return None


def _begin(db, fence, job_id=None):
    """BEGIN IMMEDIATE, then the fence check. W2 P3-1: without a fence, a write for a job that EXISTS in the store's
    jobs table is refused (fence_required): only a store without that job (an off-path tool, a unit fixture) may
    write unfenced."""
    db.execute("BEGIN IMMEDIATE")
    _ensure(db)
    if fence is None and job_id is not None and db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
        raise Superseded({'kind': 'fence_required', 'job_id': job_id})
    problem = _check_fence(db, fence)
    if problem is not None:
        raise Superseded(problem)


def check_fence(store, fence):
    """Raise Superseded when ``fence`` no longer holds (read-only)."""
    if fence is None:
        return
    with store.connect() as db:
        problem = _check_fence(db, fence)
    if problem is not None:
        raise Superseded(problem)


def fenced_write(store, fence, write, job_id=None):
    """Run ``write()`` (a file write) while holding the Store's write lock with ``fence`` verified, so a chunk claim
    cannot be cancelled, requeued or re-claimed between the check and the write. Without a fence: write() only, and
    only if ``job_id`` (when given) is not a stored job."""
    if fence is None:
        if job_id is not None:
            with store.connect() as db:
                if db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                    raise Superseded({'kind': 'fence_required', 'job_id': job_id})
        return write()
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        problem = _check_fence(db, fence)
        if problem is not None:
            raise Superseded(problem)
        return write()


def _upsert(db, job_id, revision, claim_id, kind, window, start, end, state, claimant, token, lease, payload, now):
    db.execute("INSERT INTO beta_line_claims VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
               "ON CONFLICT(job_id,revision,claim_id) DO UPDATE SET kind=excluded.kind,window_key=excluded.window_key,"
               "start=excluded.start,end=excluded.end,state=excluded.state,holder=excluded.holder,"
               "holder_pid=excluded.holder_pid,holder_host=excluded.holder_host,token=excluded.token,"
               "lease_expires=excluded.lease_expires,payload=excluded.payload,updated=excluded.updated",
               (job_id, int(revision), claim_id, kind, window, float(start), float(end), state,
                claimant and claimant['run_id'], claimant and claimant['pid'], claimant and claimant['host'],
                int(token), float(lease), json.dumps(payload, sort_keys=True, default=str), now, now))


def _set_state(db, row, state, *, payload=None, claimant=None, token=None, lease=0., now):
    holder = claimant or {'run_id': row.get('holder'), 'pid': row.get('holder_pid'), 'host': row.get('holder_host')}
    db.execute("UPDATE beta_line_claims SET state=?,holder=?,holder_pid=?,holder_host=?,token=?,lease_expires=?,"
               "payload=?,updated=? WHERE job_id=? AND revision=? AND claim_id=?",
               (state, holder['run_id'], holder['pid'], holder['host'],
                int(row['token'] if token is None else token), float(lease),
                json.dumps(row['payload'] if payload is None else payload, sort_keys=True, default=str), now,
                row['job_id'], int(row['revision']), row['claim_id']))


def _public(row):
    """Claim row as returned to callers (no internal columns renamed)."""
    return {'claim_id': row['claim_id'], 'kind': row['kind'], 'window_key': row['window_key'],
            'start': row['start'], 'end': row['end'], 'state': row['state'], 'holder': row.get('holder'),
            'holder_pid': row.get('holder_pid'), 'holder_host': row.get('holder_host'), 'token': row.get('token'),
            'lease_expires': row.get('lease_expires'), 'payload': row['payload']}


# --------------------------------------------------------------------------------------------- reads
def claims(store, job_id, revision, start, end, kind=None):
    """Every row (any state) of ``kind`` ('line', 'window', 'protected' or None) overlapping [start, end)."""
    with store.connect() as db:
        return [_public(r) for r in _rows(db, job_id, revision, start, end, kind)]


def windows(store, job_id, revision, start, end):
    """Committed window records overlapping [start, end): payload fields plus window_key."""
    return [{**r['payload'], 'window_key': r['window_key']} for r in claims(store, job_id, revision, start, end, 'window')]


def live_claims(store, job_id, revision, start, end, *, exclude_window=None, before=None, kinds=LIVE_KINDS,
                now=None, alive=pid_alive):
    """Line claims and window intents overlapping [start, end) whose holder may still publish (not stale: fence,
    lease and holder process checked). ``before``: only rows of windows whose render range starts before it."""
    now = time.time() if now is None else now
    with store.connect() as db:
        rows = [r for kind in kinds for r in _rows(db, job_id, revision, start, end, kind)]
        live = [r for r in rows if r['state'] in LIVE_STATES and r['window_key'] != exclude_window
                and (before is None or window_range(r['window_key'])[0] < float(before))
                and not stale(r, now=now, alive=alive, db=db)]
    return [_public(r) for r in sorted(live, key=lambda r: (r['start'], r['claim_id']))]


def await_lines(store, job_id, revision, start, end, *, timeout, exclude_window=None, before=None, poll=.05,
                alive=pid_alive, cancelled=None):
    """Block until no live line claim and no live window intent overlaps [start, end), or ``timeout`` seconds pass
    (or cancelled() is true). Returns what is still live ([] = resolved).

    Intents are registered when a window's worker starts, before its preparation, so this waits for the OWNER's
    decision (its commit or release), not only for claims it already made. Pass before=<own render_start> to wait only
    for earlier windows (whose takes can run into this one); two neighbours then never wait for each other."""
    deadline = time.monotonic() + max(0., float(timeout))
    while True:
        pending = live_claims(store, job_id, revision, start, end, exclude_window=exclude_window, before=before,
                              alive=alive)
        if not pending or time.monotonic() >= deadline or (cancelled is not None and cancelled()):
            return pending
        time.sleep(poll)


# --------------------------------------------------------------------------------------------- claim lifecycle
def claim_lines(store, *, job_id, revision, claimant, lines, same_line=None, lease_seconds=LEASE_SECONDS,
                fence=None, now=None, alive=pid_alive, fault=None):
    """Atomically claim ``lines`` ([{'line_id', 'source_span', ...extra payload}]) for ``claimant`` before synthesis.

    Returns {line_id: {'line_id', 'status', 'token', 'artifacts', 'blocking'}}; status:
      acquired    new claim (token 1)
      reacquired  this claim again: same run (same token, lease renewed) or a released claim (token + 1)
      taken_over  the previous holder was dead or its lease had expired (token + 1)
      held        another live holder claims this line or the same line: do NOT synthesize
      published   this line or the same line is already published: do NOT synthesize
    Only acquired / reacquired / taken_over may synthesize. Artifacts recorded under an earlier token are returned.
    """
    same_line = same_line or significant_overlap
    now = time.time() if now is None else float(now)
    lease = now + float(lease_seconds)
    out = {}
    with store.connect() as db:
        _begin(db, fence, job_id)
        for line in sorted(lines, key=lambda l: (float(l['source_span'][0]), l['line_id'])):
            lid, span = line['line_id'], [float(line['source_span'][0]), float(line['source_span'][1])]
            near = _rows(db, job_id, revision, span[0] - 1., span[1] + 1., 'line')
            own = next((r for r in near if r['claim_id'] == lid), None)
            blocking = None
            for row in near:
                if row['claim_id'] == lid or not same_line(span, [row['start'], row['end']]):
                    continue
                if row['state'] == 'published':
                    blocking = ('published', row)
                    break
                if row['state'] in LIVE_STATES and row.get('holder') != claimant['run_id']:
                    if stale(row, now=now, alive=alive, host=claimant['host'], db=db):
                        payload = _history(dict(row['payload']), {'event': 'released_stale', 'at': now,
                                                                  'by': claimant['run_id'], 'for': lid})
                        _set_state(db, row, 'released', payload=payload, now=now)
                        continue
                    blocking = ('held', row)
                    break
            result = {'line_id': lid, 'token': None, 'artifacts': {}, 'blocking': None}
            if blocking is not None:
                kind, row = blocking
                result.update(status=kind, blocking={'claim_id': row['claim_id'], 'window_key': row['window_key'],
                                                     'state': row['state'], 'holder': row.get('holder'),
                                                     'span': [row['start'], row['end']]})
            elif own is None:
                payload = _history({**{k: v for k, v in line.items() if k not in ('line_id',)}, 'version': VERSION,
                                    'line_id': lid, 'source_span': span, 'artifacts': {}, 'claim_fence': fence},
                                   {'event': 'acquired', 'at': now, 'by': claimant['run_id'], 'token': 1})
                _upsert(db, job_id, revision, lid, 'line', claimant['window_key'], span[0], span[1], 'claimed',
                        claimant, 1, lease, payload, now)
                result.update(status='acquired', token=1)
            elif own['state'] == 'published':
                result.update(status='published', token=own['token'], artifacts=own['payload'].get('artifacts') or {},
                              blocking={'claim_id': lid, 'window_key': own['window_key'], 'state': 'published',
                                        'holder': own.get('holder'), 'span': [own['start'], own['end']]})
            else:
                mine = own['state'] in LIVE_STATES and own.get('holder') == claimant['run_id']
                if own['state'] in LIVE_STATES and not mine and not stale(own, now=now, alive=alive, host=claimant['host'],
                                                                         db=db):
                    result.update(status='held', blocking={'claim_id': lid, 'window_key': own['window_key'],
                                                           'state': own['state'], 'holder': own.get('holder'),
                                                           'span': [own['start'], own['end']]})
                else:
                    status = 'reacquired' if mine or own['state'] == 'released' else 'taken_over'
                    token = int(own['token']) if mine else int(own['token']) + 1
                    payload = _history({**own['payload'], 'claim_fence': fence},
                                       {'event': status, 'at': now, 'by': claimant['run_id'], 'token': token,
                                        'previous_holder': own.get('holder')})
                    _set_state(db, own, 'claimed', payload=payload, claimant=claimant, token=token, lease=lease, now=now)
                    result.update(status=status, token=token, artifacts=payload.get('artifacts') or {})
            out[lid] = result
        if fault is not None:
            fault('claim_lines')
    return out


def _held(row, claimant, token):
    return (row is not None and row['state'] in LIVE_STATES and row.get('holder') == claimant['run_id']
            and int(row['token']) == int(token))


def record_artifact(store, *, job_id, revision, claimant, line_id, token, key, artifact, fence=None,
                    lease_seconds=LEASE_SECONDS, now=None, fault=None):
    """Store a synthesized take (file + receipt) in a held claim and renew its lease. False when the claim is no
    longer held by this run with this token (it was released or taken over): the caller must not publish it.
    artifact['audio'] must be named artifact_name(line_id, token, claimant run, key) (ValueError otherwise)."""
    from pathlib import Path
    expected = artifact_name(line_id, token, claimant['run_id'], key, Path(str(artifact.get('audio'))).suffix or '.wav')
    now = time.time() if now is None else float(now)
    with store.connect() as db:
        _begin(db, fence, job_id)                               # a superseded worker gets Superseded first
        if Path(str(artifact.get('audio'))).name != expected:
            raise ValueError('A recorded take must use its per-claim file name %s' % expected)
        row = _row(db, job_id, revision, line_id)
        if not _held(row, claimant, token):
            return False
        payload = dict(row['payload'])
        payload['artifacts'] = {**(payload.get('artifacts') or {}), str(key): artifact}
        _history(payload, {'event': 'artifact', 'key': str(key), 'at': now, 'token': int(token)})
        _set_state(db, row, 'claimed', payload=payload, claimant=claimant, token=token,
                   lease=now + float(lease_seconds), now=now)
        if fault is not None:
            fault('record_artifact')
        return True


def renew(store, *, job_id, revision, claimant, lease_seconds=LEASE_SECONDS, fence=None, now=None):
    """Heartbeat: extend the lease of every line claim and intent this run holds. Returns how many. Fenced: a
    superseded worker raises Superseded and renews nothing."""
    now = time.time() if now is None else float(now)
    with store.connect() as db:
        _begin(db, fence, job_id)
        cursor = db.execute("UPDATE beta_line_claims SET lease_expires=?,updated=? WHERE job_id=? AND revision=? "
                            "AND kind IN ('line','intent') AND state='claimed' AND holder=?",
                            (now + float(lease_seconds), now, job_id, int(revision), claimant['run_id']))
        return cursor.rowcount


def release(store, *, job_id, revision, claimant, line_ids=None, reason='released', now=None):
    """Release this run's unpublished claims (all, or ``line_ids``) and, when releasing all, its window intent.
    Artifacts stay for a later re-claim."""
    now = time.time() if now is None else float(now)
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        _ensure(db)
        return _release(db, job_id, revision, claimant, line_ids, reason, now)


def _release(db, job_id, revision, claimant, line_ids, reason, now, keep=(), kinds=LIVE_KINDS, state='released'):
    rows = [_decode(r) for r in db.execute(
        "SELECT * FROM beta_line_claims WHERE job_id=? AND revision=? AND state='claimed' AND holder=?",
        (job_id, int(revision), claimant['run_id'])) if r['kind'] in kinds]
    count = 0
    for row in rows:
        if (line_ids is not None and row['claim_id'] not in line_ids) or row['claim_id'] in keep:
            continue
        payload = _history(dict(row['payload']), {'event': state, 'reason': reason, 'at': now,
                                                  'by': claimant['run_id']})
        _set_state(db, row, state, payload=payload, now=now)
        count += 1
    return count


def register_window(store, *, job_id, revision, claimant, render_range, analysis_range, fence=None,
                    lease_seconds=LEASE_SECONDS, now=None):
    """Record that this run of a window has started (before preparation): an 'intent' row over [render_start,
    analysis_end) that await_lines waits for until the window commits ('committed') or releases. Fenced."""
    now = time.time() if now is None else float(now)
    rs, re_ = float(render_range[0]), float(render_range[1])
    with store.connect() as db:
        _begin(db, fence, job_id)
        _upsert(db, job_id, revision, 'intent:%s:%s' % (claimant['window_key'], claimant['run_id']), 'intent',
                claimant['window_key'], rs, max(re_, float(analysis_range[1])), 'claimed', claimant, 1,
                now + float(lease_seconds),
                {'version': VERSION, 'render': [rs, re_], 'analysis': [float(analysis_range[0]), float(analysis_range[1])],
                 'claim_fence': fence, 'history': [{'event': 'registered', 'at': now, 'by': claimant['run_id']}]}, now)


# --------------------------------------------------------------------------------------------- commit
def _file_sha(path):
    try:
        with open(str(path), 'rb') as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _canonical(payloads, revision):
    return [p for p in payloads if p.get('ledger_version') == 1 and p.get('revision') == revision]


def take_extent(take):
    """dialogue_ledger.take_span: complete source, placement and removal extent of a canonical take."""
    return (min(float(take['start']), float(take['replacement_start']), float(take['source_start'])),
            max(float(take['end']), float(take['replacement_end']), float(take['source_end'])))


def _take_sources(take):
    spans = take.get('authorized_audio_spans')
    return [[float(a), float(b)] for a, b in spans] if spans else [[float(take['source_start']), float(take['source_end'])]]


def races(*, window, snapshot_ids, new_takes, line_claims, canonical_now, committed, same_line, claimant=None,
          now=None, alive=pid_alive, is_stale=None):
    """Pure race check (see module docstring). new_takes: ledger payloads this window commits; line_claims: line
    claims {'line_id', 'source_span', 'take_id', 'token'} it publishes; canonical_now: canonical payloads in
    [origin, end) read inside the transaction; committed: beta_line_claims rows (any window); same_line(a, b) on
    source spans. window['owned_unit_spans']: source spans of every unit this window owns at render."""
    found = []
    known = set(snapshot_ids)
    now = time.time() if now is None else float(now)
    rs, re_ = float(window['render_start']), float(window['render_end'])
    owned = [list(s) for s in window.get('owned_unit_spans') or ()]
    run = (claimant or {}).get('run_id')

    def relevant(take):
        a, b = take_extent(take)
        return (b > rs and a < re_) or any(same_line(s, o) for s in _take_sources(take) for o in owned)
    changed = sorted(p['id'] for p in canonical_now if p['id'] not in known and relevant(p))
    if changed:
        found.append({'kind': 'canonical_takes_changed', 'canonical_ids': changed})
    by_id = {r['claim_id']: r for r in committed if r.get('kind') == 'line'}
    for claim in line_claims:
        row = by_id.get(claim['line_id'])
        held = (row is not None and claimant is not None and row.get('holder') == run
                and int(row.get('token') or 0) == int(claim.get('token') or -1) and row['state'] in LIVE_STATES)
        if not held:
            found.append({'kind': 'claim_not_held', 'line_id': claim['line_id'], 'take_id': claim['take_id'],
                          'state': None if row is None else row['state'],
                          'holder_window': None if row is None else row['window_key']})
        for other in by_id.values():
            if other['claim_id'] == claim['line_id'] or not same_line(claim['source_span'], [other['start'], other['end']]):
                continue
            dead = (is_stale(other) if is_stale is not None else
                    stale(other, now=now, alive=alive, host=(claimant or {}).get('host')))
            live = other['state'] in LIVE_STATES and other.get('holder') != run and not dead
            if other['state'] == 'published' or live:
                found.append({'kind': 'line_already_claimed', 'line_id': claim['line_id'], 'take_id': claim['take_id'],
                              'claimed_by_window': other['window_key'], 'claim_id': other['claim_id'],
                              'state': other['state']})
    for take in new_takes:
        if float(take['end']) <= re_:
            continue
        for other in committed:
            if other.get('kind') != 'window' or other['window_key'] == window['window_key']:
                continue
            a, b = float(other['start']), float(other['end'])
            if b > re_ and min(float(take['end']), b) > max(re_, a, float(take['start'])):
                found.append({'kind': 'take_overruns_into_committed_window', 'take_id': take['id'],
                              'take_end': float(take['end']), 'window_key': other['window_key'], 'window': [a, b]})
    unique = []
    for row in found:
        if row not in unique:
            unique.append(row)
    return unique


def commit(store, *, job_id, revision, window, snapshot_ids, utterances, line_claims, protected, same_line=None,
           protection=None, protection_rows=(), claimant=None, fence=None, now=None, alive=pid_alive, fault=None):
    """Atomically commit this window. Returns (protection_problems, races); nothing is written unless both are empty.

    window: {'window_key','render_start','render_end','origin','end', ...receipt fields}
    utterances: (ident, begin, end, payload) rows exactly as DialogueLedger.entry builds them
    line_claims: [{'line_id','source_span','take_id','token','placement','audible'}]; each must be held by
      ``claimant`` with that token (claim_lines). Required whenever line_claims is not empty.
    protected: [{'item_id','spans','decision'}] boundary walls of this window
    protection/protection_rows: the window's ProtectionLedger and protection_rows() exactly as passed to
      ProtectionLedger.commit (None/() when the window has no source identity).
    fence: chunk claim {'job_id','index','token','revision'}; a superseded claim raises Superseded, nothing written.
    Every new take's audio must be an artifact recorded in its claim (record_artifact) whose file still hashes to it.
    On success this run's other unpublished claims are released in the same transaction."""
    same_line = same_line or significant_overlap
    if line_claims and claimant is None:
        raise ValueError('Publishing line claims requires the claimant that holds them')
    revision = int(revision)
    now = time.time() if now is None else float(now)
    utterances = list(utterances)
    if sorted(u[0] for u in utterances) != sorted(c['take_id'] for c in line_claims):
        raise ValueError('Every new take needs exactly one held line claim (and every claim one take)')
    stored = []
    if protection is not None:
        stored = [{'cue': r['cue'], 'start': float(r['start']), 'end': float(r['end']),
                   'payload': {**r, 'scope': protection.scope, 'revision': protection.revision}} for r in protection_rows]
    with store.connect() as db:
        _begin(db, fence, job_id)
        # 1. relay.store.Store.commit_protection, unchanged semantics (same reads, same conflict callback)
        problems = []
        if protection is not None:
            spans = [(r['start'], r['end']) for r in stored] + [(u[1], u[2]) for u in utterances]
            if spans:
                low, high = min(a for a, _ in spans) - 1., max(b for _, b in spans) + 1.
                existing = [store.decode(r) for r in db.execute(
                    "SELECT * FROM protected_originals WHERE job_id=? AND end>? AND start<?", (job_id, low, high))]
                takes = [json.loads(r["payload"]) for r in db.execute(
                    "SELECT payload FROM utterances WHERE job_id=? AND end>? AND start<?", (job_id, low, high))]
            else:
                existing, takes = [], []
            problems = protection.conflicts(existing, takes, stored, utterances)
            if problems:
                return problems, []
        # 2. races against other windows' commits and claims
        origin, end = float(window['origin']), float(window['end'])
        canonical_now = _canonical([json.loads(r['payload']) for r in db.execute(
            "SELECT payload FROM utterances WHERE job_id=? AND end>? AND start<?", (job_id, origin, end))], revision)
        lo = min([origin] + [float(c['source_span'][0]) for c in line_claims]) - 1.
        hi = max([end] + [float(c['source_span'][1]) for c in line_claims] + [float(u[3]['end']) for u in utterances]) + 1.
        committed = [r for r in _rows(db, job_id, revision, lo, hi) if r['kind'] != 'intent']
        found = races(window=window, snapshot_ids=snapshot_ids, new_takes=[u[3] for u in utterances],
                      line_claims=line_claims, canonical_now=canonical_now, committed=committed,
                      same_line=same_line, claimant=claimant, now=now, alive=alive,
                      is_stale=lambda r: stale(r, now=now, alive=alive, host=(claimant or {}).get('host'), db=db))
        by_claim = {r['claim_id']: r for r in committed if r['kind'] == 'line'}
        takes = {u[0]: u[3] for u in utterances}
        for claim in line_claims:
            take, row = takes[claim['take_id']], by_claim.get(claim['line_id'])
            if row is None:
                continue                                         # already reported as claim_not_held
            recorded = [a for a in (row['payload'].get('artifacts') or {}).values()
                        if str(a.get('audio')) == str(take.get('audio'))
                        and (a.get('receipt') or {}).get('sha256') == take.get('sha256')]
            if not recorded:
                found.append({'kind': 'take_not_claimed_artifact', 'line_id': claim['line_id'], 'take_id': claim['take_id'],
                              'audio': take.get('audio')})
            elif _file_sha(take.get('audio')) != take.get('sha256'):
                found.append({'kind': 'take_file_changed', 'line_id': claim['line_id'], 'take_id': claim['take_id'],
                              'audio': take.get('audio')})
        if found:
            return [], found
        # 3. writes: Store.commit_protection's two INSERTs, then the claims
        if protection is not None:
            for row in stored:
                db.execute("INSERT INTO protected_originals VALUES(?,?,?,?,?,?,?,?) "
                           "ON CONFLICT(job_id,revision,scope_key,cue_id) DO NOTHING",
                           (job_id, revision, protection.key, row["cue"], row["start"], row["end"],
                            json.dumps(row["payload"], sort_keys=True), now))
        for ident, start, stop, payload in utterances:
            db.execute("INSERT INTO utterances VALUES(?,?,?,?,?) ON CONFLICT(job_id,id) "
                       "DO UPDATE SET start=excluded.start,end=excluded.end,payload=excluded.payload",
                       (job_id, ident, start, stop, json.dumps(payload)))
        key = window['window_key']
        rows = {r['claim_id']: r for r in committed if r['kind'] == 'line'}
        for claim in line_claims:
            row = rows[claim['line_id']]
            payload = _history({**row['payload'], **{k: v for k, v in claim.items() if k not in ('line_id', 'token')},
                                'published_window': key, 'published_at': now},
                               {'event': 'published', 'at': now, 'by': claimant['run_id'], 'token': int(claim['token'])})
            _set_state(db, row, 'published', payload=payload, claimant=claimant, token=claim['token'], now=now)
        _upsert(db, job_id, revision, 'win:' + key, 'window', key, window['render_start'], window['render_end'],
                'committed', claimant, 0, 0., {**window, 'version': VERSION}, now)
        for p in protected:
            if p.get('spans'):
                _upsert(db, job_id, revision, 'prot:%s:%s' % (key, p['item_id']), 'protected', key,
                        min(a for a, _ in p['spans']), max(b for _, b in p['spans']), 'committed', claimant, 0, 0.,
                        {**p, 'version': VERSION, 'window_key': key}, now)
        if claimant is not None:
            _release(db, job_id, revision, claimant, None, 'not_published_by_window_commit', now,
                     keep={c['line_id'] for c in line_claims}, kinds=('line',))
            _release(db, job_id, revision, claimant, None, 'window_committed', now, kinds=('intent',),
                     state='committed')
        if fault is not None:
            fault('commit')
        return [], []
