from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "relay.sqlite3"
        # One IMMEDIATE transaction for schema creation and the B6 migration:
        # executescript's deferred statements let concurrent first openers
        # deadlock on the read->write upgrade ("database is locked").
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for statement in """
            CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, state TEXT NOT NULL,
                payload TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS job_fingerprint ON jobs(fingerprint);
            CREATE TABLE IF NOT EXISTS chunks(
                job_id TEXT NOT NULL, idx INTEGER NOT NULL, state TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100, payload TEXT NOT NULL,
                updated REAL NOT NULL, PRIMARY KEY(job_id,idx));
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT,
                at REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS speakers(
                job_id TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(job_id,id));
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS leases(
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS tv_devices(
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS utterances(
                job_id TEXT NOT NULL, id TEXT NOT NULL, start REAL NOT NULL,
                end REAL NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id,id));
            CREATE TABLE IF NOT EXISTS protected_originals(
                job_id TEXT NOT NULL, revision INTEGER NOT NULL, scope_key TEXT NOT NULL,
                cue_id TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
                payload TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY(job_id,revision,scope_key,cue_id));
            CREATE INDEX IF NOT EXISTS protected_original_span ON protected_originals(job_id,start,end);
            CREATE TABLE IF NOT EXISTS sample_carries(
                job_id TEXT NOT NULL, revision INTEGER NOT NULL, scope_key TEXT NOT NULL,
                session TEXT NOT NULL, edge_frame INTEGER NOT NULL, digest TEXT NOT NULL,
                state TEXT NOT NULL, payload TEXT NOT NULL, updated REAL NOT NULL,
                token INTEGER NOT NULL DEFAULT 0, claimant TEXT,
                PRIMARY KEY(job_id,revision,scope_key,session,edge_frame));
            CREATE TABLE IF NOT EXISTS output_streams(
                session TEXT PRIMARY KEY, sink_id TEXT UNIQUE, state TEXT NOT NULL,
                token INTEGER NOT NULL, owner TEXT, frames INTEGER NOT NULL DEFAULT 0,
                write_start INTEGER, write_end INTEGER, payload TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS xt_journal_events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, revision INTEGER NOT NULL,
                session_uid TEXT NOT NULL, seg_uid TEXT NOT NULL, digest TEXT NOT NULL,
                payload TEXT NOT NULL, created REAL NOT NULL,
                UNIQUE(job_id,revision,session_uid,seg_uid,digest));
            CREATE TABLE IF NOT EXISTS xt_stems(
                job_id TEXT NOT NULL, revision INTEGER NOT NULL, window_idx INTEGER NOT NULL,
                payload TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY(job_id,revision,window_idx));
            """.split(";"):
                if statement.strip():
                    db.execute(statement)
            # B6: B5-era carry tables lack the fencing columns. Added
            # idempotently; existing rows keep their state and start at token 0.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(sample_carries)")}
            if "token" not in columns:
                db.execute("ALTER TABLE sample_carries ADD COLUMN token INTEGER NOT NULL DEFAULT 0")
            if "claimant" not in columns:
                db.execute("ALTER TABLE sample_carries ADD COLUMN claimant TEXT")

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=30000")
            self._journal_wal(db)
        except BaseException:
            db.close()
            raise
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    WAL_SWITCH_SECONDS = 30.

    @classmethod
    def _journal_wal(cls, db):
        """``PRAGMA journal_mode=WAL`` with a bounded retry.

        Switching a not-yet-WAL file needs exclusive access, and SQLite fails
        that switch at once with "database is locked" without consulting the
        busy handler, so concurrent first openers collided (B6 gap 7). Only
        that lock error is retried, for at most WAL_SWITCH_SECONDS; the
        result is otherwise ignored exactly as before.
        """
        deadline, pause = time.monotonic() + cls.WAL_SWITCH_SECONDS, .005
        while True:
            try:
                db.execute("PRAGMA journal_mode=WAL").fetchone()
                return
            except sqlite3.OperationalError as error:
                text = str(error).lower()
                if ("locked" not in text and "busy" not in text) or time.monotonic() >= deadline:
                    raise
            time.sleep(pause)
            pause = min(.1, pause * 2)

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        return result

    def put_job(self, ident, fingerprint, payload):
        now = time.time()
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                       (ident, fingerprint, "queued", json.dumps(payload), now, now))

    def job(self, ident):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (ident,)).fetchone())

    def _xt_claim(self, db, job_id, revision, claim):
        from .process import Cancelled
        if claim is None:
            raise ValueError('X-Translator writes require a chunk claim')
        index, token = claim
        job = self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
        row = self.decode(db.execute('SELECT * FROM chunks WHERE job_id=? AND idx=?',
                                     (job_id, int(index))).fetchone())
        if (not job or job['state'] in ('cancelled', 'error') or not row or row['state'] != 'working'
                or int(row['payload'].get('claim_token', 0)) != int(token)
                or int(job['payload'].get('revision', 1)) != int(revision)
                or job['payload'].get('speech_engine') != 'xtranslator'):
            raise Cancelled('X-Translator writer lost its job/revision/chunk claim')
        return job

    def xt_journal_put(self, job_id, revision, session_uid, seg_uid, payload, *, claim=None):
        """Append an immutable event; never let an old session publish into its successor."""
        if not session_uid or not isinstance(session_uid, str) or not seg_uid or not isinstance(seg_uid, str):
            raise ValueError('X-Translator journal identity missing')
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self._xt_claim(db, job_id, revision, claim)
            active = job['payload'].get('xt_session_uid')
            if seg_uid == '@session' and payload.get('event') == 'session_open':
                if active != session_uid:
                    old = db.execute('SELECT 1 FROM xt_journal_events WHERE job_id=? AND revision=? AND session_uid=? LIMIT 1',
                                     (job_id, revision, session_uid)).fetchone()
                    if old:
                        raise ValueError('A superseded X-Translator session cannot be reopened')
                    job['payload']['xt_session_uid'] = session_uid
                    db.execute('UPDATE jobs SET payload=?,updated=? WHERE id=?',
                               (json.dumps(job['payload']), time.time(), job_id))
            elif active != session_uid:
                from .process import Cancelled
                raise Cancelled('X-Translator event belongs to a superseded session')
            terminal = {'ok', 'translation_failed', 'tts_failed', 'empty_output',
                        'controller_commit_timeout', 'owner_chunk_already_published', 'error', 'pad_output_discarded',
                        'source_english', 'language_unknown', 'language_unsupported'}
            if seg_uid != '@session' and payload.get('status') in terminal:
                rows = db.execute('SELECT digest,payload FROM xt_journal_events WHERE job_id=? AND revision=? AND session_uid=? AND seg_uid=?',
                                  (job_id, revision, session_uid, seg_uid)).fetchall()
                if any(json.loads(r['payload']).get('status') in terminal and r['digest'] != digest for r in rows):
                    raise ValueError('A terminal X-Translator outcome cannot be replaced')
            db.execute('INSERT OR IGNORE INTO xt_journal_events(job_id,revision,session_uid,seg_uid,digest,payload,created) VALUES(?,?,?,?,?,?,?)',
                       (job_id, int(revision), session_uid, seg_uid, digest, encoded, time.time()))
            return db.execute('SELECT seq FROM xt_journal_events WHERE job_id=? AND revision=? AND session_uid=? AND seg_uid=? AND digest=?',
                              (job_id, int(revision), session_uid, seg_uid, digest)).fetchone()['seq']

    def xt_journal(self, job_id, revision, session_uid):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute(
                'SELECT * FROM xt_journal_events WHERE job_id=? AND revision=? AND session_uid=? ORDER BY seq',
                (job_id, int(revision), session_uid))]

    def xt_stem_put(self, job_id, revision, window_index, payload, *, claim=None):
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self._xt_claim(db, job_id, revision, claim)
            for key in ('source_fingerprint', 'engine_fingerprint'):
                if payload.get(key) != job['payload'].get(key):
                    raise ValueError('X-Translator stem has a different ' + key)
            old = db.execute('SELECT payload FROM xt_stems WHERE job_id=? AND revision=? AND window_idx=?',
                             (job_id, int(revision), int(window_index))).fetchone()
            if old and old['payload'] != encoded:
                raise ValueError('An immutable X-Translator source window cannot be replaced')
            db.execute('INSERT OR IGNORE INTO xt_stems VALUES(?,?,?,?,?)',
                       (job_id, int(revision), int(window_index), encoded, time.time()))

    def xt_stem(self, job_id, revision, window_index):
        with self.connect() as db:
            row = db.execute('SELECT payload FROM xt_stems WHERE job_id=? AND revision=? AND window_idx=?',
                             (job_id, int(revision), int(window_index))).fetchone()
            return json.loads(row['payload']) if row else None

    def retry_xt_session(self, job_id, index, token, reason, *, limit=3, wait_resources=False):
        """After confirmed cleanup, fence the old claim and budget one internal recovery."""
        from .process import Cancelled
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
            if not job:
                return None
            revision = int(job['payload'].get('revision', 1))
            try:
                job = self._xt_claim(db, job_id, revision, (index, token))
            except Cancelled:
                return None  # owner cancellation/seek superseded this claim during cleanup
            count = int(job['payload'].get('xt_recovery_count', 0))
            if count >= limit:
                return False
            old_uid = job['payload'].get('xt_session_uid')
            job['payload'].update(xt_session_uid=None, xt_recovery_count=count + 1,
                                  xt_last_recovery_reason=str(reason))
            row = self.decode(db.execute('SELECT * FROM chunks WHERE job_id=? AND idx=?',
                                         (job_id, index)).fetchone())
            row['payload']['claim_token'] = int(token) + 1
            row['payload']['xt_recovery_count'] = count + 1
            retry_state = 'waiting_resources' if wait_resources else 'pending'
            if wait_resources:
                row['payload']['resource_wait'] = str(reason)
            now = time.time()
            db.execute("UPDATE chunks SET state=?,payload=?,updated=? WHERE job_id=? AND idx=?",
                       (retry_state, json.dumps(row['payload']), now, job_id, index))
            db.execute("UPDATE jobs SET state='queued',payload=?,updated=? WHERE id=?",
                       (json.dumps(job['payload']), now, job_id))
            db.execute('INSERT INTO events(job_id,at,kind,payload) VALUES(?,?,?,?)',
                       (job_id, now, 'speaker_state_reset', json.dumps({'reason': str(reason),
                        'old_session_uid': old_uid, 'chunk': index, 'attempt': count + 1,
                        'maximum': limit, 'cleanup_confirmed': True, 'retry_state': retry_state})))
            return True

    def jobs(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 100")]

    def cached_job(self, fingerprint):
        with self.connect() as db:
            return self.decode(db.execute(
                "SELECT * FROM jobs WHERE fingerprint=? AND state NOT IN ('cancelled','error') "
                "ORDER BY created DESC LIMIT 1", (fingerprint,)).fetchone())

    def resumable_job(self, fingerprint):
        """Explicit reopen lookup: newest non-error job, an active one before a cancelled one.

        One snapshot, so a concurrent opener that already reactivated the cancelled job
        is found as active instead of starting a duplicate job.
        """
        with self.connect() as db:
            return self.decode(db.execute(
                "SELECT * FROM jobs WHERE fingerprint=? AND state<>'error' "
                "ORDER BY state='cancelled', created DESC LIMIT 1", (fingerprint,)).fetchone())

    def reactivate_cancelled_job(self, job_id, revision, verified, invalid, base_priority=100):
        """Atomically requeue a cancelled job for an explicit reopen, keeping valid ready chunks.

        ``verified`` {idx: sha256} and ``invalid`` {idx: (sha256_or_None, reason)} together
        cover every ready/approved chunk; their audio was hashed by the caller outside this
        lock. Inside one IMMEDIATE transaction the job must still be cancelled at
        ``revision`` and every ready/approved chunk must still carry the hashed sha256,
        otherwise nothing is written ('changed'). Invalid chunks return to pending with
        their audio receipt removed (the file itself is kept); cancelled/working/
        waiting_resources chunks return to pending. Every reset chunk takes a new claim
        token, so a worker still unwinding an older claim cannot settle it. Stale pending
        priorities are demoted to ``base_priority + idx`` so the caller's target is
        claimed first. Returns 'reactivated', 'active' (not cancelled: another opener
        won), 'changed' or 'unresumable' (a chunk is in error).
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            if job is None:
                raise KeyError(job_id)
            if job["state"] != "cancelled":
                return "active"
            if int(job["payload"].get("revision", 1)) != int(revision):
                return "changed"
            rows = {r["idx"]: self.decode(r) for r in db.execute("SELECT * FROM chunks WHERE job_id=?", (job_id,))}
            if any(r["state"] == "error" for r in rows.values()):
                return "unresumable"
            ready = {i: r["payload"].get("sha256") for i, r in rows.items() if r["state"] in ("ready", "approved")}
            if set(ready) != set(verified) | set(invalid) or set(verified) & set(invalid):
                return "changed"
            if any(ready[i] != sha for i, sha in verified.items()) or any(ready[i] != v[0] for i, v in invalid.items()):
                return "changed"
            now = time.time()
            for i, row in sorted(rows.items()):
                old = row["payload"]
                if i in invalid:
                    payload = {"start": old.get("start"), "end": old.get("end"),
                               "cache_invalidated": {"reason": invalid[i][1], "previous_sha256": invalid[i][0],
                                                     "previous_audio": old.get("audio"),
                                                     "previous_state": row["state"], "at": now}}
                elif row["state"] in ("pending", "cancelled", "working", "waiting_resources"):
                    payload = dict(old)
                else:
                    continue
                payload["claim_token"] = int(old.get("claim_token") or 0) + 1
                db.execute("UPDATE chunks SET state='pending',priority=max(priority,?),payload=?,updated=? "
                           "WHERE job_id=? AND idx=?",
                           (int(base_priority) + i, json.dumps(payload), now, job_id, i))
            job["payload"].update(message="Resumed retained work for an explicit reopen", resumed_at=now)
            db.execute("UPDATE jobs SET state='queued',payload=?,updated=? WHERE id=? AND state='cancelled'",
                       (json.dumps(job["payload"]), now, job_id))
            return "reactivated"

    def settle_claimed_chunk(self, job_id, index, token, state, **updates):
        """Write a claimed chunk's outcome only while it still carries this claim's token.

        ``token`` is the "claim_token" returned by claim(). Any later claim, requeue or
        reopen reset has advanced the token, so a stale claimant's receipt, cancel, error
        or resource write changes nothing and returns False. Returns True when written.
        """
        if isinstance(token, bool) or not isinstance(token, int):
            return False
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.decode(db.execute("SELECT * FROM chunks WHERE job_id=? AND idx=?",
                                         (job_id, index)).fetchone())
            if not row:
                raise KeyError((job_id, index))
            if int(row["payload"].get("claim_token") or 0) != token:
                return False
            row["payload"].update(updates)
            row["payload"]["claim_token"] = token
            db.execute("UPDATE chunks SET state=?,payload=?,updated=? WHERE job_id=? AND idx=?",
                       (state, json.dumps(row["payload"]), time.time(), job_id, index))
            return True

    def update_job(self, ident, state=None, **updates):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (ident,)).fetchone())
            if not row:
                raise KeyError(ident)
            row["payload"].update(updates)
            db.execute("UPDATE jobs SET state=?,payload=?,updated=? WHERE id=?",
                       (state or row["state"], json.dumps(row["payload"]), time.time(), ident))

    def request_chunk(self, job_id, index, payload, priority=100):
        with self.connect() as db:
            db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?) "
                       "ON CONFLICT(job_id,idx) DO UPDATE SET priority=min(priority,excluded.priority)",
                       (job_id, index, "pending", priority, json.dumps(payload), time.time()))

    def chunk(self, job_id, index):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM chunks WHERE job_id=? AND idx=?",
                                          (job_id, index)).fetchone())

    def chunks(self, job_id):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM chunks WHERE job_id=? ORDER BY idx", (job_id,))]

    def update_chunk(self, job_id, index, state, **updates):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.decode(db.execute("SELECT * FROM chunks WHERE job_id=? AND idx=?",
                                         (job_id, index)).fetchone())
            if not row:
                raise KeyError((job_id, index))
            row["payload"].update(updates)
            if state == "pending" and row["state"] != "pending":
                # A requeue starts a new claim generation; an older claimant can no longer settle it.
                row["payload"]["claim_token"] = int(row["payload"].get("claim_token") or 0) + 1
            db.execute("UPDATE chunks SET state=?,payload=?,updated=? WHERE job_id=? AND idx=?",
                       (state, json.dumps(row["payload"]), time.time(), job_id, index))

    def claim(self, *, ordered=False, engine=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT c.* FROM chunks c JOIN jobs j ON j.id=c.job_id
                WHERE c.state='pending' AND j.state NOT IN ('cancelled','error','needs_review','waiting_resources')
                AND (? IS NULL OR json_extract(j.payload,'$.speech_engine')=?)
                AND (?=0 OR c.idx>=coalesce(json_extract(j.payload,'$.generic_active_index'),
                                            json_extract(j.payload,'$.generic_start_index'),0))
                ORDER BY c.priority,c.updated LIMIT 1""", (engine, engine, int(ordered))).fetchone()
            if not row:
                return None
            if ordered:
                row = db.execute("""SELECT c.* FROM chunks c JOIN jobs j ON j.id=c.job_id
                    WHERE c.job_id=? AND c.state='pending'
                    AND c.idx>=coalesce(json_extract(j.payload,'$.generic_active_index'),
                                        json_extract(j.payload,'$.generic_start_index'),0)
                    ORDER BY c.idx LIMIT 1""",
                                 (row['job_id'],)).fetchone()
            # Each claim takes the next claim token; the returned row carries it as "claim_token".
            claimed = self.decode(row)
            claimed["claim_token"] = int(claimed["payload"].get("claim_token") or 0) + 1
            claimed["payload"]["claim_token"] = claimed["claim_token"]
            db.execute("UPDATE chunks SET state='working',payload=?,updated=? WHERE job_id=? AND idx=?",
                       (json.dumps(claimed["payload"]), time.time(), row["job_id"], row["idx"]))
            return claimed

    def retry_resources(self,delay=60):
        """Resume deferred owned jobs without treating a resource hold as failure."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute("SELECT c.job_id,c.idx FROM chunks c JOIN jobs j ON j.id=c.job_id WHERE c.state='waiting_resources' AND j.state='waiting_resources' AND c.updated<=?",
                            (time.time()-delay,)).fetchall()
            for row in rows:
                db.execute("UPDATE chunks SET state='pending',updated=? WHERE job_id=? AND idx=?",
                           (time.time(),row['job_id'],row['idx']))
                db.execute("UPDATE jobs SET state='queued',updated=? WHERE id=? AND state='waiting_resources'",
                           (time.time(),row['job_id']))

    def recover(self):
        with self.connect() as db:
            db.execute("UPDATE chunks SET state='pending' WHERE state='working'")
            db.execute("UPDATE jobs SET state='queued' WHERE state='processing'")

    def event(self, job_id, kind, **payload):
        with self.connect() as db:
            db.execute("INSERT INTO events(job_id,at,kind,payload) VALUES(?,?,?,?)",
                       (job_id, time.time(), kind, json.dumps(payload)))
            db.execute("DELETE FROM events WHERE seq < (SELECT COALESCE(MAX(seq),0)-10000 FROM events)")

    def events(self, after=0):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT 300", (after,))]

    def speakers(self, job_id):
        with self.connect() as db:
            return {r["id"]: json.loads(r["payload"]) for r in db.execute(
                "SELECT * FROM speakers WHERE job_id=?", (job_id,))}

    def put_speaker(self, job_id, ident, payload):
        with self.connect() as db:
            db.execute("INSERT INTO speakers VALUES(?,?,?) ON CONFLICT(job_id,id) "
                       "DO UPDATE SET payload=excluded.payload", (job_id, ident, json.dumps(payload)))

    def put_record(self, table, ident, payload):
        if table not in ("sessions", "leases", "tv_devices"):
            raise ValueError("Unsupported record table")
        with self.connect() as db:
            db.execute(f"INSERT INTO {table} VALUES(?,?,?) ON CONFLICT(id) "
                       "DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
                       (ident, json.dumps(payload), time.time()))

    def records(self, table):
        if table not in ("sessions", "leases", "tv_devices"):
            raise ValueError("Unsupported record table")
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY updated DESC")]

    def expire_record(self, table, ident):
        if table not in ("sessions", "leases", "tv_devices"):
            raise ValueError("Unsupported record table")
        with self.connect() as db:
            db.execute(f"DELETE FROM {table} WHERE id=?", (ident,))

    def utterances(self, job_id, start, end):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM utterances WHERE job_id=? AND end>? AND start<? ORDER BY start",
                (job_id, start, end))]

    def put_utterance(self, job_id, ident, start, end, payload):
        with self.connect() as db:
            db.execute("INSERT INTO utterances VALUES(?,?,?,?,?) ON CONFLICT(job_id,id) "
                       "DO UPDATE SET start=excluded.start,end=excluded.end,payload=excluded.payload",
                       (job_id, ident, start, end, json.dumps(payload)))

    # Protected-original decisions are stored apart from generated takes. A
    # row records that exact source PCM was kept for a short cue; it carries
    # no generated audio and is never returned by utterances().
    def protected_originals(self, job_id, start, end):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM protected_originals WHERE job_id=? AND end>? AND start<? "
                "ORDER BY start,cue_id", (job_id, start, end))]

    def commit_protection(self, job_id, revision, scope_key, rows, utterances=(), *, conflicts):
        """Atomically record kept-original rows and generated takes.

        ``conflicts(db_rows, db_takes, rows, utterances)`` receives the
        protected rows and canonical take payloads already committed for this
        job (read inside the same IMMEDIATE transaction) and returns a list of
        conflict issues. Nothing is written when it returns any conflict.
        Identical re-entry rows are idempotent; ``rows`` are dicts with
        ``cue``, ``start``, ``end`` and ``payload``; ``utterances`` are
        ``(ident, start, end, payload)`` tuples as for put_utterance.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            spans = [(r["start"], r["end"]) for r in rows]+[(u[1], u[2]) for u in utterances]
            if spans:
                low, high = min(a for a, _ in spans)-1., max(b for _, b in spans)+1.
                existing = [self.decode(r) for r in db.execute(
                    "SELECT * FROM protected_originals WHERE job_id=? AND end>? AND start<?",
                    (job_id, low, high))]
                takes = [json.loads(r["payload"]) for r in db.execute(
                    "SELECT payload FROM utterances WHERE job_id=? AND end>? AND start<?",
                    (job_id, low, high))]
            else:
                existing, takes = [], []
            problems = conflicts(existing, takes, rows, utterances)
            if problems:
                return problems
            now = time.time()
            for row in rows:
                db.execute("INSERT INTO protected_originals VALUES(?,?,?,?,?,?,?,?) "
                           "ON CONFLICT(job_id,revision,scope_key,cue_id) DO NOTHING",
                           (job_id, int(revision), scope_key, row["cue"], row["start"], row["end"],
                            json.dumps(row["payload"], sort_keys=True), now))
            for ident, start, end, payload in utterances:
                db.execute("INSERT INTO utterances VALUES(?,?,?,?,?) ON CONFLICT(job_id,id) "
                           "DO UPDATE SET start=excluded.start,end=excluded.end,payload=excluded.payload",
                           (job_id, ident, start, end, json.dumps(payload)))
            return []

    # Release markers for generated PCM carried across a window edge (B5).
    # A row holds carry metadata only; the samples stay in canonical takes.
    def sample_carry(self, job_id, revision, scope_key, session, edge_frame):
        with self.connect() as db:
            return self.decode(db.execute(
                "SELECT * FROM sample_carries WHERE job_id=? AND revision=? AND scope_key=? "
                "AND session=? AND edge_frame=?",
                (job_id, int(revision), scope_key, session, int(edge_frame))).fetchone())

    _CARRY_KEY = "job_id=? AND revision=? AND scope_key=? AND session=? AND edge_frame=?"
    _CARRY_COLUMNS = "job_id,revision,scope_key,session,edge_frame,digest,state,payload,updated"

    def transition_sample_carry(self, job_id, revision, scope_key, session, edge_frame, decide):
        """Atomic read-decide-write of one carry release marker (generic).

        ``decide(row_or_None)`` returns ``(new_or_None, result)``; ``new`` is
        ``{'digest', 'state', 'payload'}`` and is written in the same
        IMMEDIATE transaction. ``result`` is returned unchanged. Any rewrite
        of an existing row advances its fencing token, so a claimant holding
        an older token can no longer release or confirm it.
        """
        key = (job_id, int(revision), scope_key, session, int(edge_frame))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.decode(db.execute(
                f"SELECT * FROM sample_carries WHERE {self._CARRY_KEY}", key).fetchone())
            new, result = decide(row)
            if new is not None:
                db.execute(f"INSERT INTO sample_carries({self._CARRY_COLUMNS}) VALUES(?,?,?,?,?,?,?,?,?) "
                           "ON CONFLICT(job_id,revision,scope_key,session,edge_frame) DO UPDATE SET "
                           "digest=excluded.digest,state=excluded.state,payload=excluded.payload,"
                           "updated=excluded.updated,token=sample_carries.token+1,claimant=NULL",
                           key+(new["digest"], new["state"], json.dumps(new["payload"], sort_keys=True),
                                time.time()))
            return result

    # B6 fenced transitions. Each is one conditional statement inside a
    # BEGIN IMMEDIATE transaction with a rowcount check, so two independent
    # connections (threads or processes) can never both win the same step.
    def publish_sample_carry(self, job_id, revision, scope_key, session, edge_frame, digest, payload):
        """Insert a pending marker; identical re-publication is idempotent.

        Returns None on success, else the existing row's state (conflict).
        """
        key = (job_id, int(revision), scope_key, session, int(edge_frame))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            inserted = db.execute(
                f"INSERT INTO sample_carries({self._CARRY_COLUMNS}) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(job_id,revision,scope_key,session,edge_frame) DO NOTHING",
                key+(digest, "pending", json.dumps(payload, sort_keys=True), time.time())).rowcount
            if inserted == 1:
                return None
            row = db.execute(f"SELECT state,digest FROM sample_carries WHERE {self._CARRY_KEY}", key).fetchone()
            if row["state"] == "pending" and row["digest"] == digest:
                return None
            return row["state"]

    def claim_sample_carry(self, job_id, revision, scope_key, session, edge_frame, digest, claimant):
        """Take ownership of a pending (or not yet releasing) carry.

        Returns ``{'token': n}`` with a new, strictly larger fencing token
        on success. Taking over a ``claimed`` row fences its previous
        claimant: that claimant never wrote a carried frame (writing needs
        the ``releasing`` step, which checks the token). On refusal returns
        ``{'token': None, 'state': row_state_or_None, 'digest_matches': bool}``.
        """
        key = (job_id, int(revision), scope_key, session, int(edge_frame))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE sample_carries SET state='claimed',token=token+1,claimant=?,updated=? "
                f"WHERE {self._CARRY_KEY} AND digest=? AND state IN ('pending','claimed')",
                (claimant, time.time())+key+(digest,)).rowcount
            row = db.execute(f"SELECT state,digest,token FROM sample_carries WHERE {self._CARRY_KEY}",
                             key).fetchone()
            if changed == 1:
                return {"token": int(row["token"])}
            return {"token": None, "state": None if row is None else row["state"],
                    "digest_matches": row is not None and row["digest"] == digest}

    def advance_sample_carry(self, job_id, revision, scope_key, session, edge_frame, digest, token,
                             from_states, to_state):
        """Move a carry the caller owns (current ``token``) between states.

        True only when the row has this digest, this exact token and one of
        ``from_states``; a stale token, another state or a missing row
        changes nothing and returns False.
        """
        key = (job_id, int(revision), scope_key, session, int(edge_frame))
        states = tuple(from_states)
        if not states or isinstance(token, bool) or not isinstance(token, int):
            return False
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return db.execute(
                "UPDATE sample_carries SET state=?,updated=? "
                f"WHERE {self._CARRY_KEY} AND digest=? AND token=? AND state IN ({','.join('?'*len(states))})",
                (to_state, time.time())+key+(digest, token)+states).rowcount == 1

    def discard_sample_carry(self, job_id, revision, scope_key, session, edge_frame, digest):
        """Invalidate a pending carry (fencing any later stale claim)."""
        key = (job_id, int(revision), scope_key, session, int(edge_frame))
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return db.execute(
                "UPDATE sample_carries SET state='discarded',token=token+1,claimant=NULL,updated=? "
                f"WHERE {self._CARRY_KEY} AND digest=? AND state='pending'",
                (time.time(),)+key+(digest,)).rowcount == 1

    # B6b: whole-stream ownership of one continuous output (session + sink).
    # States: open (an owner holds ``token``; no sink call in flight),
    # writing (a sink call for frames [write_start, write_end) is in flight;
    # persisted BEFORE the call), uncertain (that call raised: 0..all of its
    # frames may be on the sink), closed (owner finished with no call in
    # flight; ``frames`` were acknowledged by the sink returning). A dead
    # owner leaves open/writing forever. Only a closed stream that never
    # acknowledged a frame can be acquired again; nothing else is resumed.
    def output_stream(self, session):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM output_streams WHERE session=?", (session,)).fetchone())

    @staticmethod
    def _stream_refusal(row, conflict):
        return {"token": None, "conflict": conflict, "state": row["state"], "frames": int(row["frames"]),
                "write": None if row["write_start"] is None else [int(row["write_start"]), int(row["write_end"])],
                "session": row["session"]}

    def acquire_output_stream(self, session, sink_id, owner, payload):
        """Take ownership of ``session`` (and ``sink_id`` when given).

        One BEGIN IMMEDIATE transaction, so of any number of independent
        connections at most one wins. Returns ``{'token': n}`` or a refusal
        ``{'token': None, 'conflict': 'session'|'sink'|'session_sink_changed',
        'state', 'frames', 'write', 'session'}`` describing the blocking row.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM output_streams WHERE session=?", (session,)).fetchone()
            if row is not None:
                if row["sink_id"] != sink_id:
                    return self._stream_refusal(row, "session_sink_changed")
                changed = db.execute(
                    "UPDATE output_streams SET state='open',token=token+1,owner=?,payload=?,updated=? "
                    "WHERE session=? AND state='closed' AND frames=0 AND write_start IS NULL",
                    (owner, json.dumps(payload, sort_keys=True), time.time(), session)).rowcount
                if changed != 1:
                    return self._stream_refusal(row, "session")
                return {"token": int(db.execute("SELECT token FROM output_streams WHERE session=?",
                                                (session,)).fetchone()["token"])}
            if sink_id is not None:
                other = db.execute("SELECT * FROM output_streams WHERE sink_id=?", (sink_id,)).fetchone()
                if other is not None:
                    return self._stream_refusal(other, "sink")
            db.execute("INSERT INTO output_streams(session,sink_id,state,token,owner,frames,payload,updated) "
                       "VALUES(?,?,'open',1,?,0,?,?)",
                       (session, sink_id, owner, json.dumps(payload, sort_keys=True), time.time()))
            return {"token": 1}

    def _stream_step(self, session, token, sql, where, params):
        if isinstance(token, bool) or not isinstance(token, int):
            return False
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return db.execute(f"UPDATE output_streams SET {sql},updated=? WHERE session=? AND token=? AND {where}",
                              params[0]+(time.time(), session, token)+params[1]).rowcount == 1

    def begin_output_write(self, session, token, start, end):
        """Persist the intent to hand frames [start, end) to the sink. Contiguous only."""
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or not 0 <= start < end:
            return False
        return self._stream_step(session, token, "state='writing',write_start=?,write_end=?",
                                 "state='open' AND frames=?", ((start, end), (start,)))

    def confirm_output_write(self, session, token, end):
        """The sink call returned: [write_start, end) is acknowledged."""
        return self._stream_step(session, token, "state='open',frames=write_end,write_start=NULL,write_end=NULL",
                                 "state='writing' AND write_end=?", ((), (end,)))

    def fail_output_write(self, session, token):
        """The sink call raised: the in-flight interval is uncertain for good."""
        return self._stream_step(session, token, "state='uncertain'", "state='writing'", ((), ()))

    def close_output_stream(self, session, token):
        """The owner finished with no sink call in flight."""
        return self._stream_step(session, token, "state='closed'", "state='open'", ((), ()))
