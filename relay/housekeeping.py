"""Housekeeping (F11): prune the job store and the caches, with no one's approval. Stdlib only; never touches the source
film, the actor bank, the settings file or anything outside <state>/relay.sqlite3 and <cache>.

Three lists are built by plan(); nothing else is ever deleted, and verify() re-checks every entry before a deletion:

1. DEAD JOBS (store rows + <cache>/<job id>/). A job is dead when it can never be opened again:
     error         its state is 'error' (Coordinator.enqueue never returns an errored job), untouched for 24 h;
     other-engine  its payload.engine_fingerprint differs from config.engine_fingerprint() (the job key contains the
                   engine fingerprint, so no enqueue can match it any more), untouched for 14 days;
     superseded    another job with the same fingerprint is the one Store.resumable_job() returns, untouched for 24 h.
   NEVER dead, whatever else is true: a FINISHED dub (every window ready), the job Store.resumable_job() returns for
   its fingerprint, a job that is not terminal, a job attached to a TV session, a job with a published file.
   Also listed: <cache>/<32 hex>/ folders older than 24 h that have no job row at all ('orphan').
2. INTERMEDIATES of kept jobs, per window folder <cache>/<job id>/chunk-NNNNNN/: first-pass/, original-layout.wav,
   actor-English.wav, actor-surround.flac, vocals-16k.wav, dialogue-16k.wav, language-redecode-*.wav. Only when the
   window and both neighbours are ready/approved and the window was settled more than 24 h ago. Never: english.wav,
   english-surround.flac/.json, line-*-reference.wav, any .json receipt, original-carry.*, and the job-level folders
   (actor-utterances/, reference-*/).
3. TV SEGMENTS in <cache>/tv-media: a <key>.ts or <key>.m4s (with its <key>.json receipt) unused for
   tv_media_max_age_days, then the least recently used ones while the folder is above tv_media_cap_gib or the disk is
   below tv.min_free_disk_gib + 20 GiB; and a <sha256>.init.mp4 that no remaining receipt names. Never: a segment a
   current TV session may still serve, an init segment named by the receipt of any segment that stays, a file younger
   than one hour, *.partial* (tv._sweep_partials).

housekeeping.store_prune:
  'auto' (default)  the first 24 h after the first listing: list only (<profile>/logs/housekeeping-plan.json and one log
                    line); after that, and only while verify() finds nothing wrong: delete. No one has to switch it on.
  'dry-run'         list only, for ever.      'on'  delete from the first start.      'off'  do not even look.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path

VERSION = "kevflix-housekeeping-v1"
MODES = ("auto", "dry-run", "on", "off")
DEFAULTS = {"store_prune": "auto", "tv_media_cap_gib": 40, "tv_media_max_age_days": 30,
            "keep_intermediates": False, "log_to_file": True}
JOB_TABLES = ("chunks", "speakers", "utterances", "events", "protected_originals", "sample_carries",
              "xt_journal_events", "xt_stems", "beta_line_claims")
TERMINAL = ("complete", "cancelled", "error", "needs_review")
READY = ("ready", "approved")
INTERMEDIATE_DIRS = ("first-pass",)
INTERMEDIATE_FILES = ("original-layout.wav", "actor-English.wav", "actor-surround.flac", "vocals-16k.wav",
                      "dialogue-16k.wav")
INTERMEDIATE_PATTERN = re.compile(r"^language-redecode-\d+\.wav$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")
CHUNK_DIR = re.compile(r"^chunk-(\d{6})$")
SEGMENT_SUFFIXES = (".ts", ".m4s")
INIT_SUFFIX = ".init.mp4"
MIN_AGE_SECONDS = 24 * 3600
OTHER_ENGINE_MIN_AGE_SECONDS = 14 * 86400
SEGMENT_MIN_AGE_SECONDS = 3600
AUTO_ON_AFTER_SECONDS = 24 * 3600
DISK_MARGIN_GIB = 20
PERIOD_SECONDS = 3600
BUSY_RETRY_SECONDS = 600
STATE_FILE = "housekeeping-state.json"
GIB = 1024 ** 3
log = logging.getLogger("kevflix.housekeeping")


def settings(config):
    data = getattr(config, "data", config)
    block = data.get("housekeeping") if hasattr(data, "get") else None
    return {**DEFAULTS, **(block if isinstance(block, dict) else {})}


def validate(block):
    """Called by relay.config.Config: raises ValueError for a bad housekeeping block, returns the merged block."""
    merged = {**DEFAULTS, **(block or {})}
    if merged["store_prune"] not in MODES:
        raise ValueError('housekeeping.store_prune must be "auto", "dry-run", "on" or "off"')
    cap = merged["tv_media_cap_gib"]
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 5 <= cap <= 2000:
        raise ValueError("housekeeping.tv_media_cap_gib must be a number between 5 and 2000")
    age = merged["tv_media_max_age_days"]
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not 0 <= age <= 3650:
        raise ValueError("housekeeping.tv_media_max_age_days must be a number between 0 (off) and 3650")
    for key in ("keep_intermediates", "log_to_file"):
        if not isinstance(merged[key], bool):
            raise ValueError("housekeeping.%s must be true or false" % key)
    return merged


def _tree_bytes(path):
    total = files = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, files


def _stamp(seconds):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(seconds)) if seconds else None


# ---- the 'auto' rule ------------------------------------------------------------------------------------------------

def _state_path(store):
    return Path(store.path).parent / STATE_FILE


def read_state(store):
    try:
        value = json.loads(_state_path(store).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def note_listing(store, listing, now):
    """Remember that a listing was written and logged (the 'auto' rule counts 24 h from the first one)."""
    state = read_state(store)
    if state.get("version") != VERSION or not isinstance(state.get("first_listing_at"), (int, float)):
        state = {"version": VERSION, "first_listing_at": now}
    state.update(last_listing_at=now, verified=not listing.get("refused"))
    target = _state_path(store)
    try:
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        os.replace(temporary, target)
    except OSError as error:
        log.warning("housekeeping: state not written (%s)", error)


def effective_mode(config, store, now=None):
    """'dry-run' | 'on' | 'off'. 'auto' is 'dry-run' until 24 h after the first listing was written and logged and the
    latest listing passed verify(); then it is 'on' by itself."""
    mode = settings(config)["store_prune"]
    if mode != "auto":
        return mode
    now = time.time() if now is None else now
    state = read_state(store)
    first = state.get("first_listing_at")
    if (state.get("version") == VERSION and isinstance(first, (int, float)) and state.get("verified") is True
            and now - first >= AUTO_ON_AFTER_SECONDS):
        return "on"
    return "dry-run"


# ---- 1. dead jobs ---------------------------------------------------------------------------------------------------

def dead_jobs(store, engine_fingerprint, now, keep=()):
    """[{id, title, state, updated, reason, chunks, ready}] of the jobs that can never be opened again (module docstring)."""
    keep = set(keep)
    with store.connect() as db:
        rows = [dict(r) for r in db.execute(
            "SELECT id,fingerprint,state,created,updated,"
            "json_extract(payload,'$.engine_fingerprint') AS engine,json_extract(payload,'$.title') AS title,"
            "json_extract(payload,'$.published') AS published,"
            "json_extract(payload,'$.chunks_total') AS total FROM jobs")]
        counts = {r["job_id"]: r["n"] for r in db.execute("SELECT job_id,count(*) AS n FROM chunks GROUP BY job_id")}
        ready = {r["job_id"]: r["n"] for r in db.execute(
            "SELECT job_id,count(*) AS n FROM chunks WHERE state IN ('ready','approved') GROUP BY job_id")}
    survivors = {}
    # Store.resumable_job(): "WHERE fingerprint=? AND state<>'error' ORDER BY state='cancelled', created DESC LIMIT 1"
    for row in sorted((r for r in rows if r["state"] != "error"), key=lambda r: (r["state"] == "cancelled", -r["created"])):
        survivors.setdefault(row["fingerprint"], row["id"])
    dead = []
    for row in rows:
        age = now - float(row["updated"])
        finished = bool(row["total"]) and ready.get(row["id"], 0) >= int(row["total"])
        if (row["id"] in keep or row["published"] or row["state"] not in TERMINAL or age < MIN_AGE_SECONDS
                or (finished and row["state"] != "error")):
            continue                                      # a finished dub is never deleted
        if row["state"] == "error":
            reason = "error"
        elif row["engine"] != engine_fingerprint:
            if age < OTHER_ENGINE_MIN_AGE_SECONDS:
                continue                                  # a settings change may be undone: two weeks of grace
            reason = "other-engine"
        elif survivors.get(row["fingerprint"]) != row["id"]:
            reason = "superseded"
        else:
            continue                                      # the resumable job of its film
        dead.append({"id": row["id"], "title": row["title"], "state": row["state"], "updated": _stamp(row["updated"]),
                     "reason": reason, "chunks": counts.get(row["id"], 0), "ready": ready.get(row["id"], 0)})
    return sorted(dead, key=lambda item: item["updated"] or "")


def delete_jobs(store, job_ids):
    """Delete the rows of these jobs from every job table in ONE transaction. Returns {table: rows deleted}."""
    job_ids = list(job_ids)
    counts = {}
    if not job_ids:
        return counts
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        tables = {r["name"] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, column in [(t, "job_id") for t in JOB_TABLES] + [("jobs", "id")]:
            if table not in tables:
                continue
            counts[table] = 0
            for start in range(0, len(job_ids), 400):
                part = job_ids[start:start + 400]
                cursor = db.execute("DELETE FROM %s WHERE %s IN (%s)" % (table, column, ",".join("?" * len(part))), part)
                counts[table] += cursor.rowcount
    return counts


def vacuum(store):
    """Give the freed pages back to the disk. Needs the store to itself: called before the coordinator starts. A busy
    store is left as it is (the next start tries again). Returns (bytes before, bytes after)."""
    path = Path(store.path)
    before = path.stat().st_size
    db = sqlite3.connect(str(path), timeout=5, isolation_level=None)
    try:
        db.execute("PRAGMA busy_timeout=5000")
        db.execute("VACUUM")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError as error:
        log.warning("housekeeping: store not compacted (%s)", error)
    finally:
        db.close()
    return before, path.stat().st_size


# ---- 2. job folders and intermediates -------------------------------------------------------------------------------

def job_folders(cache, dead_ids, known_ids, now):
    """([{path, reason, bytes, files}]) of whole <cache>/<32 hex>/ folders to delete: a dead job's, or an orphan's."""
    out = []
    try:
        entries = sorted(p for p in Path(cache).iterdir() if p.is_dir() and HEX32.match(p.name))
    except OSError:
        return out
    for folder in entries:
        if folder.name in dead_ids:
            reason = "dead-job"
        elif folder.name not in known_ids and now - folder.stat().st_mtime > MIN_AGE_SECONDS:
            reason = "orphan"
        else:
            continue
        size, files = _tree_bytes(folder)
        out.append({"path": str(folder), "reason": reason, "bytes": size, "files": files})
    return out


def intermediates(cache, store, live_ids, now):
    """[{path, bytes}] of intermediate files/folders of settled windows of kept jobs (module docstring, list 2)."""
    out = []
    for job_id in sorted(live_ids):
        folder = Path(cache) / job_id
        if not folder.is_dir():
            continue
        job = store.job(job_id)
        if not job:
            continue
        total = int(job["payload"].get("chunks_total") or 0)
        rows = {int(r["idx"]): r for r in store.chunk_rows(job_id)}

        def settled(index):
            row = rows.get(index)
            return bool(row and row["state"] in READY)

        for chunk in sorted(folder.iterdir()):
            match = CHUNK_DIR.match(chunk.name)
            if not match or not chunk.is_dir():
                continue
            index = int(match.group(1))
            if not settled(index) or now - float(rows[index]["updated"]) < MIN_AGE_SECONDS:
                continue
            if (index > 0 and not settled(index - 1)) or (index + 1 < total and not settled(index + 1)):
                continue                                  # a neighbour may still be produced: keep everything
            for entry in chunk.iterdir():
                if entry.is_dir() and entry.name in INTERMEDIATE_DIRS:
                    size, _ = _tree_bytes(entry)
                elif entry.is_file() and (entry.name in INTERMEDIATE_FILES or INTERMEDIATE_PATTERN.match(entry.name)):
                    size = entry.stat().st_size
                else:
                    continue
                out.append({"path": str(entry), "bytes": size})
    return out


# ---- 3. TV segments -------------------------------------------------------------------------------------------------

def tv_segments(cache, now, cap_bytes, max_age_seconds, protected=(), free_bytes=None, floor_bytes=0):
    """([{path, bytes, reason}], total bytes) of <cache>/tv-media to evict: media segments (<key>.ts, <key>.m4s) unused
    for max_age_seconds, then least recently used while the folder is above cap_bytes or the disk is under floor_bytes;
    and init segments (<sha256>.init.mp4) that the receipt of no remaining .m4s segment names. Protected and young
    files stay; when a receipt cannot be read, every init segment stays."""
    folder = Path(cache) / "tv-media"
    protected = {os.path.normcase(str(p)) for p in protected}
    media, inits = [], []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if ".partial" in entry.name or not entry.is_file():
                    continue
                if entry.name.endswith(INIT_SUFFIX):
                    info = entry.stat()
                    inits.append((info.st_mtime, info.st_size, entry.path))
                elif entry.name.endswith(SEGMENT_SUFFIXES):
                    info = entry.stat()
                    media.append((info.st_mtime, info.st_size, entry.path))
    except OSError:
        return [], 0
    media.sort()
    total = sum(size for _, size, _ in media) + sum(size for _, size, _ in inits)
    evict, left, free = [], total, free_bytes
    for mtime, size, path in media:
        if now - mtime < SEGMENT_MIN_AGE_SECONDS or os.path.normcase(path) in protected:
            continue
        if max_age_seconds and now - mtime > max_age_seconds:
            reason = "unused-%d-days" % round(max_age_seconds / 86400)
        elif left > cap_bytes:
            reason = "over-cap"
        elif free is not None and free < floor_bytes:
            reason = "disk-low"
        else:
            continue
        evict.append({"path": path, "bytes": size, "reason": reason})
        left -= size
        if free is not None:
            free += size
    if inits:
        going = {item["path"] for item in evict}
        named, readable = set(), True
        for _, _, path in media:
            if path.endswith(".m4s") and path not in going:
                try:
                    name = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8")).get("init")
                except (OSError, ValueError, AttributeError):
                    readable = False                      # an unknown receipt: no init segment is touched this pass
                    break
                if isinstance(name, str):
                    named.add(name)
        if readable:
            for mtime, size, path in sorted(inits):
                if (os.path.basename(path) in named or now - mtime < SEGMENT_MIN_AGE_SECONDS
                        or os.path.normcase(path) in protected):
                    continue
                evict.append({"path": path, "bytes": size, "reason": "init-unused"})
    return evict, total


# ---- plan / verify / apply ------------------------------------------------------------------------------------------

def plan(config, store, *, now=None, live_job_ids=(), live_paths=(), store_part=True, files_part=True):
    """The complete list of what a deleting pass would delete. Reads only. store_part False: no dead-job selection (the
    hourly pass); files_part False: no walk of the cache (the synchronous store pass at start)."""
    now = time.time() if now is None else now
    opts = settings(config)
    cache = Path(config.cache)
    with store.connect() as db:
        known = {r["id"] for r in db.execute("SELECT id FROM jobs")}
        sizes = {r["job_id"]: r["n"] for r in db.execute(
            "SELECT job_id,sum(length(payload)) AS n FROM chunks GROUP BY job_id")}
    dead = dead_jobs(store, config.engine_fingerprint(), now, keep=live_job_ids) if store_part else []
    for item in dead:
        item["receipt_bytes"] = int(sizes.get(item["id"]) or 0)
    dead_ids = {item["id"] for item in dead}
    folders = job_folders(cache, dead_ids, known, now) if files_part else []
    leftovers = ([] if opts["keep_intermediates"] or not files_part
                 else intermediates(cache, store, known - dead_ids, now))
    try:
        free = shutil.disk_usage(cache).free
    except OSError:
        free = None
    try:
        floor = (float(config["tv"]["min_free_disk_gib"]) + DISK_MARGIN_GIB) * GIB
    except (KeyError, TypeError, ValueError):
        floor = DISK_MARGIN_GIB * GIB
    segments, segment_total = ([], 0) if not files_part else tv_segments(
        cache, now, float(opts["tv_media_cap_gib"]) * GIB, float(opts["tv_media_max_age_days"]) * 86400,
        protected=live_paths, free_bytes=free, floor_bytes=floor)
    reasons = {}
    for item in dead:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    return {"version": VERSION, "setting": opts["store_prune"], "mode": opts["store_prune"], "at": _stamp(now),
            "applied": False,
            "store": {"path": str(store.path), "bytes": Path(store.path).stat().st_size, "jobs_total": len(known),
                      "jobs_dead": len(dead), "by_reason": reasons,
                      "dead_receipt_bytes": sum(item["receipt_bytes"] for item in dead), "dead": dead,
                      "kept": sorted(known - dead_ids)},
            "job_folders": {"count": len(folders), "bytes": sum(f["bytes"] for f in folders), "folders": folders},
            "intermediates": {"count": len(leftovers), "bytes": sum(f["bytes"] for f in leftovers),
                              "paths": leftovers},
            "tv_media": {"bytes": segment_total, "evict_count": len(segments),
                         "evict_bytes": sum(s["bytes"] for s in segments), "segments": segments},
            "disk_free_bytes": free}


def verify(config, store, listing, live_job_ids=()):
    """Independent re-check of a listing, entry by entry, straight from the store and the paths. Returns the list of
    problems; a deleting pass runs only when it is empty. It can only make housekeeping delete less, never more."""
    problems = []
    cache = Path(config.cache).resolve()
    engine = config.engine_fingerprint()
    live = set(live_job_ids)
    dead_ids = set()
    for item in listing["store"]["dead"]:
        job = store.job(item["id"])
        if job is None:
            continue                                      # gone already: nothing to delete
        dead_ids.add(item["id"])
        payload = job["payload"]
        rows = store.chunk_rows(item["id"])
        total = int(payload.get("chunks_total") or 0)
        done = sum(1 for r in rows if r["state"] in READY)
        if job["state"] not in TERMINAL:
            problems.append("%s is not terminal (%s)" % (item["id"][:8], job["state"]))
        elif item["id"] in live:
            problems.append("%s is attached to a TV session" % item["id"][:8])
        elif payload.get("published"):
            problems.append("%s has a published file" % item["id"][:8])
        elif job["state"] != "error" and total and done >= total:
            problems.append("%s is a finished dub" % item["id"][:8])
        elif job["state"] != "error" and payload.get("engine_fingerprint") == engine:
            resumable = store.resumable_job(job["fingerprint"])
            if resumable and resumable["id"] == item["id"]:
                problems.append("%s is the resumable job of its film" % item["id"][:8])
    with store.connect() as db:
        known = {r["id"] for r in db.execute("SELECT id FROM jobs")}
    for item in listing["job_folders"]["folders"]:
        path = Path(item["path"]).resolve()
        if path.parent != cache or not HEX32.match(path.name):
            problems.append("folder outside the cache: %s" % path.name)
        elif item["reason"] == "dead-job" and path.name not in dead_ids:
            problems.append("folder of a job that is not dead: %s" % path.name[:8])
        elif item["reason"] == "orphan" and path.name in known:
            problems.append("folder of an existing job listed as orphan: %s" % path.name[:8])
    for item in listing["intermediates"]["paths"]:
        path = Path(item["path"]).resolve()
        chunk, job = path.parent, path.parent.parent
        if (job.parent != cache or job.name not in known or job.name in dead_ids or not CHUNK_DIR.match(chunk.name)
                or not (path.name in INTERMEDIATE_DIRS or path.name in INTERMEDIATE_FILES
                        or INTERMEDIATE_PATTERN.match(path.name))):
            problems.append("not an intermediate of a kept job: %s" % path.name)
    media = cache / "tv-media"
    for item in listing["tv_media"]["segments"]:
        path = Path(item["path"]).resolve()
        if path.parent != media or ".partial" in path.name or not path.name.endswith(SEGMENT_SUFFIXES + (INIT_SUFFIX,)):
            problems.append("not a TV segment: %s" % path.name)
    return problems


def _remove(path):
    path = Path(path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError as error:
        log.warning("housekeeping: could not delete %s (%s)", path, error)
        return False


def apply_store(store, listing):
    """Delete the dead jobs' rows and compact the store. Call before the coordinator starts."""
    ids = [item["id"] for item in listing["store"]["dead"]]
    rows = delete_jobs(store, ids)
    before, after = vacuum(store) if ids else (listing["store"]["bytes"], listing["store"]["bytes"])
    listing["store"].update(rows_deleted=rows, bytes_after=after)
    return rows


def apply_files(listing, still_protected=lambda: ()):
    """Delete the listed folders and files (dead-job folders, intermediates, TV segments)."""
    freed = 0
    for item in listing["job_folders"]["folders"] + listing["intermediates"]["paths"]:
        if _remove(item["path"]):
            freed += item["bytes"]
    protected = {os.path.normcase(str(p)) for p in still_protected()}
    for item in listing["tv_media"]["segments"]:
        path = Path(item["path"])
        try:
            if os.path.normcase(str(path)) in protected or time.time() - path.stat().st_mtime < SEGMENT_MIN_AGE_SECONDS:
                continue                                  # used again since the listing was made
        except OSError:
            continue
        if _remove(path):
            freed += item["bytes"]
            receipt = path.with_suffix(".json")
            if not path.name.endswith(INIT_SUFFIX) and receipt.exists():
                _remove(receipt)
    listing["freed_bytes"] = freed
    return freed


def write_listing(config, listing):
    folder = Path(config.file).parent / "logs"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / "housekeeping-plan.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(listing, indent=1), encoding="utf-8")
        os.replace(temporary, target)
        return target
    except OSError as error:
        log.warning("housekeeping: listing not written (%s)", error)
        return None


def summary(listing):
    gib = lambda n: round((n or 0) / GIB, 2)       # noqa: E731
    return ("housekeeping %s: store %d of %d jobs dead (%s), %.0f MB of receipts; job folders %d (%.2f GiB); "
            "intermediates %d (%.2f GiB); tv-media %d of %.2f GiB to evict (%.2f GiB)%s%s"
            % (listing["mode"], listing["store"]["jobs_dead"], listing["store"]["jobs_total"],
               listing["store"]["by_reason"], listing["store"]["dead_receipt_bytes"] / 1e6,
               listing["job_folders"]["count"], gib(listing["job_folders"]["bytes"]),
               listing["intermediates"]["count"], gib(listing["intermediates"]["bytes"]),
               listing["tv_media"]["evict_count"], gib(listing["tv_media"]["bytes"]),
               gib(listing["tv_media"]["evict_bytes"]),
               "; freed %.2f GiB" % gib(listing["freed_bytes"]) if "freed_bytes" in listing else "",
               "; REFUSED, nothing deleted: %s" % "; ".join(listing["refused"][:3]) if listing.get("refused") else ""))


class Housekeeper:
    """start() runs the store part at once (before the coordinator starts) and the file part on a daemon thread, then
    repeats the file part every hour. tv: the TVService (live_job_ids(), live_segment_paths(), session_count()) or
    None. Nothing is deleted while a TV session is open: the pass waits and tries again ten minutes later."""

    def __init__(self, config, store, tv=None):
        self.config, self.store, self.tv = config, store, tv
        self.stop_event = threading.Event()
        self.thread = None
        self.last = None

    def _live(self):
        if self.tv is None:
            return set(), set()
        return set(self.tv.live_job_ids()), set(self.tv.live_segment_paths())

    def _watching(self):
        return bool(self.tv is not None and self.tv.session_count())

    def run_once(self, store_part, now=None):
        """One pass. Returns the listing, or None when switched off or postponed because a film is playing."""
        now = time.time() if now is None else now
        mode = effective_mode(self.config, self.store, now)
        if mode == "off" or (mode == "on" and self._watching()):
            return None
        jobs, paths = self._live()
        listing = plan(self.config, self.store, now=now, live_job_ids=jobs, live_paths=paths, store_part=store_part)
        listing["mode"] = mode
        problems = verify(self.config, self.store, listing, live_job_ids=jobs)
        if problems:
            listing["refused"] = problems[:50]
            log.error("housekeeping refused its own listing; nothing is deleted: %s", "; ".join(problems[:5]))
        elif mode == "on":
            if store_part:
                apply_store(self.store, listing)
            apply_files(listing, still_protected=lambda: self._live()[1])
            listing["applied"] = True
        if store_part or mode == "on" or problems:
            written = write_listing(self.config, listing)
            if written is not None and settings(self.config)["store_prune"] == "auto":
                note_listing(self.store, listing, now)
        log.info(summary(listing))
        self.last = listing
        return listing

    def start(self):
        if settings(self.config)["store_prune"] == "off":
            return
        try:
            if effective_mode(self.config, self.store) == "on" and not self._watching():
                # Rows first, synchronously: the coordinator and the per-window snapshot never see the dead jobs.
                jobs = self._live()[0]
                listing = plan(self.config, self.store, live_job_ids=jobs, files_part=False)
                problems = verify(self.config, self.store, listing, live_job_ids=jobs)
                if problems:
                    log.error("housekeeping refused its own listing; nothing is deleted: %s", "; ".join(problems[:5]))
                elif listing["store"]["dead"]:
                    apply_store(self.store, listing)
                    log.info("housekeeping: %d dead jobs removed from the store, %d -> %d MB", listing["store"]["jobs_dead"],
                             listing["store"]["bytes"] >> 20, listing["store"]["bytes_after"] >> 20)
        except Exception:  # noqa: BLE001 - housekeeping never stops the server
            log.exception("housekeeping: store part failed")
        self.thread = threading.Thread(target=self._loop, name="kevflix-housekeeping", daemon=True)
        self.thread.start()

    def _loop(self):
        first = True
        while not self.stop_event.is_set():
            done = None
            try:
                # While listing only, the first pass of a run also lists the dead jobs; a deleting run removed their
                # rows in start() and finds their folders here as orphans.
                done = self.run_once(store_part=first and effective_mode(self.config, self.store) == "dry-run")
            except Exception:  # noqa: BLE001
                log.exception("housekeeping pass failed")
                done = False
            if done is None and effective_mode(self.config, self.store) == "on":
                self.stop_event.wait(BUSY_RETRY_SECONDS)   # a film is playing: try again in ten minutes
                continue
            first = False
            self.stop_event.wait(PERIOD_SECONDS)

    def close(self):
        self.stop_event.set()
