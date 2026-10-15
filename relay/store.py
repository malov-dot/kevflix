from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "relay.sqlite3"
        with self.connect() as db:
            db.executescript("""
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
            """)

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

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

    def jobs(self):
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 100")]

    def cached_job(self, fingerprint):
        with self.connect() as db:
            return self.decode(db.execute(
                "SELECT * FROM jobs WHERE fingerprint=? AND state NOT IN ('cancelled','error') "
                "ORDER BY created DESC LIMIT 1", (fingerprint,)).fetchone())

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
            db.execute("UPDATE chunks SET state=?,payload=?,updated=? WHERE job_id=? AND idx=?",
                       (state, json.dumps(row["payload"]), time.time(), job_id, index))

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT c.* FROM chunks c JOIN jobs j ON j.id=c.job_id
                WHERE c.state='pending' AND j.state NOT IN ('cancelled','error','needs_review')
                ORDER BY c.priority,c.updated LIMIT 1""").fetchone()
            if row:
                db.execute("UPDATE chunks SET state='working',updated=? WHERE job_id=? AND idx=?",
                           (time.time(), row["job_id"], row["idx"]))
            return self.decode(row)

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
