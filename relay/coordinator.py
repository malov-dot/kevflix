from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from .media import (atomic_json, fingerprint, probe, file_hash, wav_info, find_sidecar,
                    safe_name, RATE, CHANNELS)
from .process import ProcessOwner, Cancelled
from .resource_budget import ResourceBudget,ResourceYield
from .movie_faults import fault_record
from . import embedded_subtitles as _subs


class _ClaimedStore:
    """The store as one claimed chunk's worker sees it.

    Writes to that chunk go through Store.settle_claimed_chunk with the claim token, so a
    worker whose claim was superseded (requeue, reopen reset or a newer claim) can never
    overwrite the newer claim or its published receipt; the refusal raises Cancelled so
    the stale worker stops. Everything else is the real store, unchanged.
    """

    def __init__(self, store, job_id, index, token):
        self._store, self._key, self._token = store, (job_id, int(index)), token

    def __getattr__(self, name):
        return getattr(self._store, name)

    @property
    def claim_token(self):
        return self._token

    def _checked_row(self, db):
        job_id, index = self._key
        row = self._store.decode(db.execute('SELECT * FROM chunks WHERE job_id=? AND idx=?',
                                            (job_id, index)).fetchone())
        job = self._store.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
        if (not row or not job or job['state'] in ('cancelled', 'error') or row['state'] != 'working'
                or int(row['payload'].get('claim_token') or 0) != self._token):
            raise Cancelled('Generic chunk lost its current publication claim')
        return row

    def check_claim(self):
        with self._store.connect() as db:
            self._checked_row(db)

    def xt_journal_put(self, job_id, revision, session_uid, seg_uid, payload):
        if job_id != self._key[0]:
            raise ValueError('Journal write belongs to a different job')
        return self._store.xt_journal_put(job_id, revision, session_uid, seg_uid, payload,
                                         claim=(self._key[1], self._token))

    def xt_stem_put(self, job_id, revision, window_index, payload):
        if job_id != self._key[0]:
            raise ValueError('Stem write belongs to a different job')
        return self._store.xt_stem_put(job_id, revision, window_index, payload,
                                      claim=(self._key[1], self._token))

    def publish_generic(self, job_id, index, utterances, **updates):
        """Atomically publish immutable generic takes and this chunk's ready receipt."""
        if (job_id, int(index)) != self._key:
            raise ValueError('Generic publication targets a different chunk')
        if updates.get('generic_version') != 'generic-dub-v1':
            raise ValueError('Generic publication version missing')
        if file_hash(updates['audio']) != updates.get('sha256'):
            raise ValueError('Generic mixed audio changed before publication')
        receipt = json.loads(Path(updates['receipt_path']).read_text(encoding='utf-8-sig'))
        if any(receipt.get(key) != updates.get(key) for key in ('generic_version', 'audio', 'sha256', 'start', 'end', 'revision')):
            raise ValueError('Generic receipt does not match published chunk')
        for ident, start, end, payload in utterances:
            if (not isinstance(ident, str) or not ident or not math.isfinite(start) or not math.isfinite(end)
                    or end <= start or payload.get('generic_version') != 'generic-dub-v1'
                    or payload.get('owner_chunk') != int(index)
                    or file_hash(payload['audio']) != payload.get('sha256')):
                raise ValueError('Invalid generic take publication')
        with self._store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self._checked_row(db)
            job = self._store.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
            if job['payload'].get('speech_engine') == 'xtranslator':
                active_session = job['payload'].get('xt_session_uid')
                if not active_session or updates.get('xt_session_uid') != active_session:
                    raise Cancelled('X-Translator publication belongs to a superseded session')
                for ident, start, end, payload in utterances:
                    if (payload.get('origin_session_uid') != active_session
                            or int(payload.get('revision', -1)) != int(job['payload'].get('revision', 1))):
                        raise Cancelled('Unpublished X-Translator take belongs to a superseded session')
                    journal = db.execute('SELECT payload FROM xt_journal_events WHERE job_id=? AND revision=? AND session_uid=? AND seg_uid=?',
                        (job_id, payload['revision'], active_session, str(payload.get('seg_uid', '')))).fetchall()
                    if not any(json.loads(event['payload']).get('status') == 'ok' for event in journal):
                        raise ValueError('X-Translator take has no immutable successful outcome')
                for key in ('xt_detected_language', 'xt_speaker_count', 'xt_reset_count', 'xt_receipt_path'):
                    if key in updates:
                        job['payload'][key] = updates[key]
                db.execute('UPDATE jobs SET payload=?,updated=? WHERE id=?',
                           (json.dumps(job['payload']), time.time(), job_id))
            for ident, start, end, payload in utterances:
                prior = self._store.decode(db.execute('SELECT * FROM utterances WHERE job_id=? AND id=?',
                                                      (job_id, ident)).fetchone())
                if prior:
                    if prior['start'] != start or prior['end'] != end or prior['payload'] != payload:
                        raise ValueError('A published generic take cannot be replaced')
                else:
                    db.execute('INSERT INTO utterances VALUES(?,?,?,?,?)',
                               (job_id, ident, start, end, json.dumps(payload)))
            row['payload'].update(updates)
            row['payload']['claim_token'] = self._token
            db.execute("UPDATE chunks SET state='ready',payload=?,updated=? WHERE job_id=? AND idx=?",
                       (json.dumps(row['payload']), time.time(), job_id, int(index)))

    def update_chunk(self, job_id, index, state, **updates):
        if (job_id, int(index)) != self._key:
            return self._store.update_chunk(job_id, index, state, **updates)
        if not self._store.settle_claimed_chunk(job_id, int(index), self._token, state, **updates):
            raise Cancelled("This chunk was requeued or claimed again; the stale result was discarded")


class Coordinator:
    # enqueue(resume_cancelled=True): an explicit reopen may reactivate a matching cancelled job.
    RESUME_CANCELLED_VERSION = 1

    def __init__(self, config, store):
        self.config, self.store = config, store
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.active_owner = None
        self.active_job = None
        self.thread = None
        self.hash_cache = {}
        self.publication_lock = threading.Lock()
        self.publication_owners = {}
        self.actor_pipeline = None
        self.movie = None      # relay.movie_pipeline.MoviePipeline when actor_beta.movie_pipeline is enabled
        # The ONE AnalysisService of this server (GPU models only in its lease-owned child): MoviePipeline's, or the
        # serial loop's. The two producers are mutually exclusive (start), so at most one LeaseHub ever exists.
        self.analysis_service = None
        self._serial_hub = None
        self._serial_resident = None
        self.xt_services = None
        self.xt_state = {}
        self._xt_last_activity = 0.
        self._configuration_lock = threading.RLock()
        self._reconfiguring = False

    def start(self):
        self.store.recover()
        from . import movie_pipeline as _movie
        # Exclusive producer selection: movie_pipeline.enabled -> MoviePipeline ONLY; otherwise the serial loop ONLY.
        # Never both: each owns the process's single movie GPU LeaseHub and AnalysisService.
        if _movie.enabled(self.config):
            # Movie dubbing (actor engine): ONE coordinator thread + warm separation/analysis workers, queue-ahead from
            # the viewer's requests, publication by settling chunks 'ready' (the TV loop muxes HLS). Replaces the serial
            # loop below for this configuration only.
            self.movie = _movie.MoviePipeline(self.config, self.store, coordinator=self)
            self.analysis_service = self.movie.analysis
            self.thread = threading.Thread(target=self.movie.run, name="movie-dub-jobs", daemon=True)
        else:
            self.thread = threading.Thread(target=self.loop, name="movie-dub-jobs", daemon=True)
        self.thread.start()

    def _serial_xt_feeder(self):
        """Lease-owned separation only: XT recognition never constructs AnalysisService."""
        from . import first_pass_parent
        from .app_runtime import movie_lease
        from .movie_pipeline.leaseview import LeaseHub
        from .movie_pipeline.resident_demucs import ResidentDemucs
        from .movie_faults import MovieConfigurationError
        if self.movie is not None or self.analysis_service is not None:
            raise MovieConfigurationError('producer_conflict', 'XT separation requires exclusive serial ownership')
        if self._serial_resident is not None:
            return self._serial_resident
        if self._serial_hub is not None or first_pass_parent.RESIDENT is not None:
            raise MovieConfigurationError('producer_conflict', 'A movie GPU producer is already registered')
        beta = self.config['actor_beta']
        if (first_pass_parent.options(self.config) is None
                or beta.get('first_pass_admission') != 'movie-lease'
                or beta.get('demucs_device') != 'cuda'):
            raise MovieConfigurationError('xt_separation_config', 'XT requires lease-owned CUDA Demucs')
        hub = LeaseHub(movie_lease.holder(self.config))
        try:
            resident = ResidentDemucs(python=self.config['python'] or sys.executable,
                relay_dir=Path(first_pass_parent.HERE),
                model_dir=first_pass_parent.config_path(self.config, 'separation_model_dir'),
                model_pins=beta['demucs_model_pins'], threads=int(beta.get('demucs_threads', 4)),
                device='cuda', lease_source=hub.source(),
                log_path=Path(self.config.state) / 'xt-separation/logs/resident-demucs.stderr.log',
                resource_yield=ResourceYield, cancelled=Cancelled)
        except BaseException:
            hub.close('xt_feeder_init_failed')
            raise
        self._serial_hub, self._serial_resident = hub, resident
        first_pass_parent.RESIDENT = resident
        return resident

    def _serial_analysis(self):
        """The serial loop's AnalysisService (actor engine): built once, on the process's ONLY LeaseHub, which is
        registered with the Vox hook so Vox shares the same lease. Refused while MoviePipeline is active."""
        if self.movie is not None:
            from .movie_faults import MovieConfigurationError
            raise MovieConfigurationError("producer_conflict",
                                          "the serial loop cannot run beside MoviePipeline (one movie GPU lease)")
        if self.analysis_service is None:
            from . import actor_beta_hook
            from .app_runtime import movie_lease
            from .movie_pipeline.analysis_service import AnalysisService
            from .movie_pipeline.leaseview import LeaseHub
            hub = LeaseHub(movie_lease.holder(self.config))
            actor_beta_hook.register_lease_hub(hub)
            resident = None
            try:
                if self.config['speech_engine'] == 'generic':
                    from . import first_pass_parent
                    from .movie_pipeline.resident_demucs import ResidentDemucs
                    from .movie_faults import MovieConfigurationError
                    beta = self.config['actor_beta']
                    if (first_pass_parent.options(self.config) is None
                            or beta.get('first_pass_admission') != 'movie-lease'
                            or beta.get('demucs_device') != 'cuda'):
                        raise MovieConfigurationError('generic_separation_config', 'Generic requires lease-owned CUDA Demucs')
                    if first_pass_parent.RESIDENT is not None:
                        raise MovieConfigurationError('producer_conflict', 'A Demucs producer is already registered')
                    resident = ResidentDemucs(python=self.config['python'] or sys.executable,
                        relay_dir=Path(first_pass_parent.HERE),
                        model_dir=first_pass_parent.config_path(self.config, 'separation_model_dir'),
                        model_pins=beta['demucs_model_pins'], threads=int(beta.get('demucs_threads', 4)),
                        device='cuda', lease_source=hub.source(),
                        log_path=Path(self.config.state) / 'serial-analysis/logs/resident-demucs.stderr.log',
                        resource_yield=ResourceYield, cancelled=Cancelled)
                    first_pass_parent.RESIDENT = resident
                service = AnalysisService(self.config, hub.source(),
                                          log_path=Path(self.config.state) / "serial-analysis" / "logs"
                                          / "analysis-worker.stderr.log")
            except BaseException:
                try:
                    if resident is not None:
                        resident.close()
                finally:
                    if resident is not None and first_pass_parent.RESIDENT is resident:
                        first_pass_parent.RESIDENT = None
                    try:
                        actor_beta_hook.unregister_lease_hub(hub)
                    finally:
                        hub.close("serial_analysis_init_failed")
                raise
            self._serial_hub, self.analysis_service, self._serial_resident = hub, service, resident
        return self.analysis_service

    def _close_serial_analysis(self):
        """Stop the serial analysis worker, then unregister the hub from the Vox hook BEFORE closing it."""
        self._close_xt()
        service, hub = (self.analysis_service, self._serial_hub) if self._serial_hub is not None else (None, None)
        if hub is None:
            return
        self.analysis_service = self._serial_hub = None
        resident, self._serial_resident = self._serial_resident, None
        try:
            if service is not None:
                service.close()
        finally:
            from . import actor_beta_hook, first_pass_parent
            try:
                if resident is not None:
                    resident.close()
            finally:
                if resident is not None and first_pass_parent.RESIDENT is resident:
                    first_pass_parent.RESIDENT = None
                try:
                    actor_beta_hook.unregister_lease_hub(hub)
                finally:
                    hub.close("serial_loop_close")

    def producer_state(self):
        """Dubbing producer state for the player (/tv-api/producer)."""
        if self.movie is not None:
            return self.movie.state()
        return {"state": "running" if self.thread and self.thread.is_alive() else "stopped",
                "reason": None, "producer": "serial-loop"}

    def _close_xt(self):
        session = self.xt_state.get('session')
        previous_job = self.xt_state.get('job_id')
        if session is not None:
            session.close()
        # Do not forget the service/lease if Linux process exit is unconfirmed.
        if self.xt_services is not None:
            self.xt_services.close()
        resets = int(self.xt_state.get('reset_count', 0)) + int(bool(session))
        self.xt_state.clear()
        self.xt_state['reset_count'] = resets
        self.xt_state['job_id'] = previous_job
        self.xt_services = None

    def engine_status(self):
        health = self.xt_services.health() if self.xt_services is not None else None
        observed_services = None
        if health is not None:
            observed_services = {name: {'state': value.get('phase', 'unknown'), 'pid': value.get('pid')}
                                 for name, value in (health.get('status') or {}).get('services', {}).items()}
        job_id = self.active_job or self.xt_state.get('job_id')
        job = self.store.job(job_id) if job_id else None
        payload = job['payload'] if job else {}
        return {'engine': self.config['speech_engine'],
                'state': 'reconfiguring' if self._reconfiguring else self.producer_state()['state'],
                'service_health': observed_services, 'runtime_detail': health,
                'error': health.get('failure') if health else None,
                'versions': {'worker': __import__('relay').__version__},
                'job_id': job_id, 'detected_language': payload.get('xt_detected_language'),
                'speaker_count': payload.get('xt_speaker_count'),
                'speaker_state_reset_count': self.xt_state.get('reset_count', payload.get('xt_reset_count')),
                'observed_at': time.time(), 'receipt_path': payload.get('xt_receipt_path')}

    def reconfigure_engine(self, save_callback):
        """Switch only an idle producer. Saving settings never loads a model."""
        with self._configuration_lock:
            if self._reconfiguring or self.active_owner or self.publication_owners:
                raise ValueError('Wait for the movie worker to become idle before changing engines')
            if any(job['state'] not in ('complete', 'cancelled', 'error', 'needs_review') for job in self.store.jobs()):
                raise ValueError('Cancel queued movie jobs before changing the dubbing engine')
            self._reconfiguring = True
        self.close()
        if self.thread and self.thread.is_alive():
            raise RuntimeError('The old producer has not exited; engine change was not saved')
        try:
            save_callback()
        finally:
            self.movie = self.actor_pipeline = self.analysis_service = None
            self._serial_hub = self._serial_resident = None
            self.thread = None
            self.stop_event.clear()
            self._reconfiguring = False
            if not self.config['setup_only']:
                self.start()

    def close(self):
        self.stop_event.set()
        if self.movie is not None:
            self.movie.stop("server_close")
        if self.active_owner:
            self.active_owner.cancel()
        for owner in list(self.publication_owners.values()):
            owner.cancel()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=30)
        if self.movie is not None:
            self.movie.close()
        if not self.thread or not self.thread.is_alive():
            if self.actor_pipeline:
                self.actor_pipeline.close()
                self.actor_pipeline = None
            self._close_serial_analysis()

    def enqueue(self, source, **kwargs):
        with self._configuration_lock:
            if self._reconfiguring:
                raise ValueError('Dubbing engine settings are changing; wait for the worker')
            return self._enqueue(source, **kwargs)

    def _enqueue(self, source, *, audio_index=None, subtitles=None, subtitle_index=None,
                mode="prepared", title=None, plex=None, force_redub=False, initial_index=None,
                resume_cancelled=False):
        source = self.config.media_path(source)
        # Owner rule (2026-10-19): subtitles are for viewing only. No route (TV, prepared-dub API, Plex) may attach a
        # sidecar/embedded subtitle track to a dubbing job; the viewing path serves captions separately.
        subtitles, subtitle_index = _subs.OFF, None
        subtitles_off = subtitles == _subs.OFF
        if subtitles_off:
            subtitles = None
        explicit_file = bool(subtitles)
        if not subtitles and subtitle_index is None and not subtitles_off:
            subtitles = find_sidecar(source)
        if subtitles:
            subtitles = str(self.config.media_path(subtitles))
        owner = ProcessOwner()
        info = probe(self.config, owner, source)
        if audio_index is None:
            # Fix 7: the primary non-commentary 5.1 track, else stereo, else mono (embedded_subtitles.select_audio).
            # A source with no supported track (e.g. 7.1-only) raises UnsupportedAudio: explicit, never downmixed.
            audio_index = _subs.default_audio_index(info["audio"])
        if audio_index not in [x["index"] for x in info["audio"]]:
            raise ValueError("The requested audio stream is not present")
        if self.config["speech_engine"] == "actor":
            # An explicit user choice is respected only when the dubbing path supports its layout.
            _subs.require_supported_audio(info["audio"], audio_index)
        if subtitle_index is not None and not any(s["index"] == subtitle_index
                and s.get("codec_type") == "subtitle" for s in info["streams"]):
            raise ValueError("The requested subtitle stream is not present")
        source_fingerprint = fingerprint(source)
        engine = self.config.engine_fingerprint()
        identity = [source_fingerprint, engine, audio_index, fingerprint(subtitles) if subtitles else None,
                    subtitle_index, bool(force_redub)]
        subtitle_choice = _subs.select_subtitles(
            info["streams"], explicit_path=_subs.OFF if subtitles_off else subtitles if explicit_file else None,
            explicit_index=subtitle_index, sidecar=None if explicit_file else subtitles)
        if subtitle_choice["selection"] == "auto-unambiguous-english":
            # Only an automatic embedded choice extends the key; every other job keeps its key.
            identity.append({"embedded_subtitles": _subs.VERSION, "stream_index": subtitle_choice["stream_index"]})
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        old = self.store.resumable_job(key) if resume_cancelled else self.store.cached_job(key)
        if old and old["state"] == "cancelled":
            old = self._reopen_cancelled(old)
            if old is not None:
                if old["payload"]["mode"] == "prepared":
                    self.request_all(old["id"])
                elif initial_index is not None:
                    self.request(old["id"], max(0, min(int(initial_index), old["payload"]["chunks_total"] - 1)), 0)
                self.summarize(old["id"])
                self.wake.set()
                return self.store.job(old["id"])
        if old:
            if mode == "prepared":
                self.request_all(old["id"])
            elif initial_index is not None:
                self.request(old["id"], max(0, min(int(initial_index), old["payload"]["chunks_total"] - 1)), 0)
            return old
        ident = uuid.uuid4().hex
        payload = {"source": str(source), "source_fingerprint": source_fingerprint,
                   "engine_fingerprint": engine, "audio_index": audio_index,
                   "subtitles": subtitles, "subtitle_index": subtitle_index, "duration": info["duration"],
                   "subtitle_fingerprint": fingerprint(subtitles) if subtitles else None,
                   "title": title or source.stem, "mode": mode, "plex": plex or {},
                   "force_redub": bool(force_redub), "revision": 1, "overrides": {},
                   "speech_engine": self.config["speech_engine"],
                   "segment_seconds": int(self.config["segment_seconds"]),
                   "chunks_total": math.ceil(info["duration"] / self.config["segment_seconds"]),
                   "media": {"streams": info["streams"]}, "published": None,
                   "subtitle_choice": subtitle_choice}
        if self.config['speech_engine'] in ('generic', 'xtranslator'):
            payload['generic_start_index'] = max(0, min(int(initial_index or 0), payload['chunks_total'] - 1))
            payload['generic_active_index'] = payload['generic_start_index']
        self.store.put_job(ident, key, payload)
        self.store.event(ident, "queued", title=payload["title"])
        if mode == "prepared":
            self.request_all(ident)
        else:
            self.request(ident, 0 if initial_index is None else
                         max(0, min(int(initial_index), payload["chunks_total"] - 1)), 0)
        return self.store.job(ident)

    def request(self, job_id, index, priority=0):
        job = self.store.job(job_id)
        if not job or job["state"] in ("cancelled", "error"):
            raise ValueError("This job is unavailable")
        spec = job["payload"]
        if index < 0 or index >= spec["chunks_total"]:
            raise ValueError("Chunk is outside this movie")
        seconds = spec["segment_seconds"]
        self.store.request_chunk(job_id, index, {"start": index * seconds,
                                  "end": min(spec["duration"], (index + 1) * seconds)}, priority)
        for next_index in range(index + 1, min(spec["chunks_total"], index + 1 + self.config["lookahead_segments"])):
            self.store.request_chunk(job_id, next_index,
                                     {"start": next_index * seconds,
                                      "end": min(spec["duration"], (next_index + 1) * seconds)}, priority + 10)
        self.wake.set()

    def request_all(self, job_id):
        job = self.store.job(job_id)
        for index in range(job["payload"]["chunks_total"]):
            self.request(job_id, index, 100 + index)

    def cancel(self, job_id):
        self.store.update_job(job_id, "cancelled", message="Cancelled; completed chunks retained")
        if self.movie is not None:
            self.movie.cancel_job(job_id)
        if self.active_job == job_id and self.active_owner:
            self.active_owner.cancel()
        if job_id in self.publication_owners:
            self.publication_owners[job_id].cancel()
        self.store.event(job_id, "cancelled")

    def resume(self, job_id):
        job = self.store.job(job_id)
        if not job:
            raise KeyError(job_id)
        if fingerprint(job["payload"]["source"]) != job["payload"]["source_fingerprint"]:
            raise ValueError("Source changed; start a new job")
        self.store.update_job(job_id, "queued", message="Resuming retained work")
        for chunk in self.store.chunks(job_id):
            if chunk["state"] in ("error", "working", "cancelled", "waiting_resources"):
                self.store.update_chunk(job_id, chunk["idx"], "pending")
        if job["payload"]["mode"] == "prepared":
            self.request_all(job_id)
        self.summarize(job_id)
        self.wake.set()

    def _cached_audio_problem(self, payload, revision):
        """None when a ready/approved chunk's recorded audio exists and matches its sha256."""
        audio, sha = payload.get("audio"), payload.get("sha256")
        if not audio or not sha:
            return "missing audio receipt"
        if int(payload.get("revision", revision)) != revision:
            return "stale revision"
        path = Path(audio)
        try:
            stat = path.stat()
        except OSError:
            return "audio file missing"
        if (str(path), stat.st_mtime_ns, stat.st_size, sha) in self.hash_cache:
            return None
        try:
            digest = file_hash(path)
        except OSError:
            return "audio unreadable"
        return None if digest == sha else "audio sha256 mismatch"

    def _reopen_cancelled(self, job):
        """Explicit reopen of a matching cancelled job; None when it must not be resumed.

        Audio is hashed with no DB lock held; the store then rechecks job state, revision
        and every chunk hash inside one transaction and refuses if anything changed.
        """
        for _ in range(3):
            revision = int(job["payload"].get("revision", 1))
            verified, invalid = {}, {}
            for chunk in self.store.chunks(job["id"]):
                if chunk["state"] == "error":
                    return None
                if chunk["state"] in ("ready", "approved"):
                    problem = self._cached_audio_problem(chunk["payload"], revision)
                    if problem:
                        invalid[chunk["idx"]] = (chunk["payload"].get("sha256"), problem)
                    else:
                        verified[chunk["idx"]] = chunk["payload"]["sha256"]
            outcome = self.store.reactivate_cancelled_job(job["id"], revision, verified, invalid)
            if outcome == "unresumable":
                return None
            if outcome == "reactivated":
                self.store.event(job["id"], "resumed", kept=len(verified),
                                 invalidated={str(i): v[1] for i, v in sorted(invalid.items())})
            job = self.store.job(job["id"])
            if job is None or job["state"] == "error":
                return None
            if job["state"] != "cancelled":
                return job
        raise ValueError("Retained dubbing changed while reopening; try again")

    def review(self, job_id, index, *, approve_hash=None, corrections=None):
        chunk = self.store.chunk(job_id, index)
        job = self.store.job(job_id)
        if job and job['payload'].get('speech_engine') == 'xtranslator':
            raise ValueError('X-Translator is automatic audio-to-audio; manual dialogue/identity overrides are disabled')
        if not chunk or chunk["state"] != "needs_review":
            raise ValueError("This chunk is not awaiting review")
        if corrections:
            overrides = dict(job["payload"].get("overrides", {}))
            valid = {c["id"] for c in chunk["payload"].get("cues", [])}
            for ident, correction in corrections.items():
                if ident not in valid:
                    raise ValueError("Correction belongs to a different cue")
                allowed = {"source_text", "source_language", "english_text", "speaker"}
                if not isinstance(correction, dict) or not set(correction) <= allowed:
                    raise ValueError("Invalid cue correction")
                speaker = correction.get("speaker")
                if speaker and speaker not in self.store.speakers(job_id):
                    raise ValueError("Choose an existing stable speaker identity")
                overrides[ident] = {**overrides.get(ident, {}), **correction}
            revision = job["payload"].get("revision", 1) + 1
            self.store.update_job(job_id, "queued", overrides=overrides, revision=revision, published=None)
            # A line can cross chunk boundaries, so invalidate this chunk and its neighbors.
            for item in self.store.chunks(job_id):
                if abs(item["idx"] - index) <= 1:
                    self.store.update_chunk(job_id, item["idx"], "pending")
                elif item["state"] in ("ready", "approved"):
                    self.store.update_chunk(job_id, item["idx"], item["state"], revision=revision)
            self.wake.set()
            return
        payload = chunk["payload"]
        if not payload.get("audio") or not approve_hash or approve_hash != payload.get("sha256"):
            raise ValueError("Approval must name the current generated audio hash")
        if file_hash(payload["audio"]) != approve_hash:
            raise ValueError("Audio changed since review")
        non_waivable = {"translation_unavailable", "speaker_unresolved", "language_uncertain",
                        "voice_pool_exhausted", "speech_without_transcript", "possible_merged_speakers",
                        "speaker_embedding_invalid", "actor_reference_unavailable", "actor_gain_unresolved", "overlapping_dialogue",
                        "speech_timing_overrun", "speech_crosses_window", "uncertain_transcript",
                        "unclear_dialogue_without_aligned_caption", "singing_requires_pitch_tempo_path", "canonical_speech_collision", "speaker_reference_mismatch", "critical_readback_mismatch"}
        if any(x["kind"] in non_waivable for x in payload.get("issues", [])):
            raise ValueError("Correct unresolved text, language or speaker information before approval")
        self.store.update_chunk(job_id, index, "approved", owner_review={"sha256": approve_hash,
                                                                       "at": time.time()})
        self.store.update_job(job_id, "queued")
        self.summarize(job_id)
        self.wake.set()

    def audio_chunk(self, job_id, index, revision, *, wait=True):
        self.request(job_id, index)
        until = time.monotonic() + self.config["stream_wait_seconds"]
        while True:
            job = self.store.job(job_id)
            if not job or job["state"] in ("cancelled", "error", "needs_review"):
                raise ValueError("Playback is blocked: " + (job["state"] if job else "missing job"))
            if job["payload"].get("revision", 1) != revision:
                raise ValueError("Playback revision changed; reopen the current stream")
            chunk = self.store.chunk(job_id, index)
            if chunk and chunk["state"] in ("ready", "approved"):
                payload = chunk["payload"]
                path = Path(payload["audio"])
                stat = path.stat()
                cache_key = (str(path), stat.st_mtime_ns, stat.st_size, payload["sha256"])
                if cache_key not in self.hash_cache:
                    if file_hash(path) != payload["sha256"]:
                        raise ValueError("A generated chunk changed; playback refused")
                    data = wav_info(path)
                    if data["rate"] != RATE or data["channels"] != CHANNELS or data["width"] != 2:
                        raise ValueError("Generated chunk has an incompatible format")
                    expected_frames = round(payload["end"] * RATE) - round(payload["start"] * RATE)
                    if data["frames"] != expected_frames:
                        raise ValueError("Generated chunk duration does not match its timeline")
                    self.hash_cache[cache_key] = True
                return path
            if not wait or time.monotonic() >= until or self.stop_event.is_set():
                raise TimeoutError("English audio is not ready at this position")
            self.wake.wait(.3)
            self.wake.clear()

    def set_generic_start(self, job_id, index):
        """Move the serial generic producer's active range only on an explicit viewer seek."""
        job = self.store.job(job_id)
        if not job or job['payload'].get('speech_engine') not in ('generic', 'xtranslator'):
            return
        index = max(0, min(int(index), int(job['payload']['chunks_total']) - 1))
        self.store.update_job(job_id, generic_active_index=index)
        if job['payload'].get('speech_engine') == 'xtranslator':
            # The serial worker observes this generation at its next claim. Do
            # not mutate or close its session from the HTTP/seek thread.
            self.store.event(job_id, 'xt_seek_requested', index=index)
        self.wake.set()

    def summarize(self, job_id):
        job = self.store.job(job_id)
        if not job or job["state"] == "cancelled":
            return
        chunks = self.store.chunks(job_id)
        ready = sum(x["state"] in ("ready", "approved") for x in chunks)
        if any(x["state"] == "needs_review" for x in chunks):
            state = "needs_review"
        elif any(x["state"] == "error" for x in chunks):
            state = "error"
        elif any(x['state']=='waiting_resources' for x in chunks):
            state='waiting_resources'
        elif ready == job["payload"]["chunks_total"]:
            state = "complete"
        else:
            state = "buffering"
        self.store.update_job(job_id, state, chunks_ready=ready)

    def loop(self):
        resource_retry_at=0
        while not self.stop_event.is_set():
            if self._serial_hub is not None:
                if self.analysis_service is not None:
                    self.analysis_service.idle_check()   # non-blocking: releases the idle analysis view
                if self._serial_resident is not None:
                    self._serial_resident.idle_check()
            if time.monotonic()>=resource_retry_at:
                self.store.retry_resources()
                resource_retry_at=time.monotonic()+30
            if (self.xt_services is not None and self._xt_last_activity
                    and time.monotonic() - self._xt_last_activity > float(self.config['xt_idle_seconds'])):
                self._close_xt()
            with self._configuration_lock:
                if self._reconfiguring or self.stop_event.is_set():
                    break
                generic = self.config['speech_engine'] in ('generic', 'xtranslator')
                item = self.store.claim(ordered=True, engine=self.config['speech_engine']) if generic else self.store.claim()
                if item:
                    self.active_job = item['job_id']
                    self.active_owner = ProcessOwner()
            if not item:
                self.wake.wait(1)
                self.wake.clear()
                continue
            job_id, index = item["job_id"], item["idx"]
            token = item.get("claim_token")
            self.store.update_job(job_id, "processing")
            try:
                python = self.config["python"] or sys.executable
                env = dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                if self.config['speech_engine'] == 'xtranslator':
                    from .xt_dub import process_xt_chunk
                    from .xt_services import XTServices
                    service = self._serial_xt_feeder()
                    identity = (job_id, int(self.store.job(job_id)['payload'].get('revision', 1)),
                                self.config.engine_fingerprint())
                    if self.xt_state.get('identity') not in (None, identity):
                        self._close_xt()
                    if self.xt_state.get('job_id') not in (None, job_id):
                        self.xt_state['reset_count'] = 0
                    self.xt_state.update(identity=identity, job_id=job_id)
                    if self.xt_services is None:
                        self.xt_services = XTServices(self.config, lease_source=self._serial_hub.source())
                    self._xt_last_activity = time.monotonic()
                    process_xt_chunk(self.config, _ClaimedStore(self.store, job_id, index, token),
                                     job_id, index, self.active_owner, service, self.xt_services, self.xt_state)
                    self._xt_last_activity = time.monotonic()
                elif self.config['speech_engine']=='generic':
                    from .generic_dub import process_generic_chunk
                    service = self._serial_analysis()
                    process_generic_chunk(self.config, _ClaimedStore(self.store, job_id, index, token),
                                          job_id, index, self.active_owner, service)
                elif self.config['speech_engine']=='actor':
                    # Cold jobs can prepare on CPU while cloning is unavailable.
                    # A deferred prepared job checks admission before retrying.
                    if self.config['actor_backend']=='cuda' and item['payload'].get('preparation_seconds') is not None:
                        ResourceBudget.for_actor(self.config).admit_gpu()
                    from .actor_pipeline import ActorPipeline
                    from .worker import process_chunk
                    if self.actor_pipeline is None:
                        # GPU-first: the serial ActorPipeline gets the ONE AnalysisService (lease-owned GPU worker).
                        pipeline=ActorPipeline(self.config)
                        pipeline.set_analysis_service(self._serial_analysis())
                        self.actor_pipeline=pipeline
                    process_chunk(self.config,_ClaimedStore(self.store,job_id,index,token),job_id,index,
                                  self.active_owner,self.actor_pipeline)
                else:
                    self.active_owner.run(
                        [python, "-m", "relay.worker", "--config", self.config.file,
                         "--job", job_id, "--chunk", index], timeout=3600,
                        cwd=Path(__file__).resolve().parent.parent, env=env)
                result = self.store.chunk(job_id, index)
                if result["state"] == "working" and result["payload"].get("claim_token") == token:
                    raise RuntimeError("Worker exited without a chunk receipt")
            except Exception as error:
                # Explicit fault records (relay.movie_faults) decide first: cancelled, resource (a real lease/VRAM/
                # preemption hold) -> waiting_resources; fatal and recoverable -> an explicit error (this serial
                # loop has no original-window publication). Legacy exception types keep their old mapping.
                fault = fault_record(error)
                kind = fault["kind"] if fault else ("cancelled" if isinstance(error, Cancelled) else
                                                    "resource" if isinstance(error, ResourceYield) else "fatal")
                message = ("%s: %s" % (fault["code"], fault["detail"])) if fault else str(error)
                recovered_xt = False
                if self.config['speech_engine'] == 'xtranslator':
                    resets_before = int(self.xt_state.get('reset_count', 0))
                    try:
                        self._close_xt()
                    except Exception as cleanup_error:
                        kind = 'fatal'
                        message += '; X-Translator cleanup unconfirmed: ' + str(cleanup_error)
                    # Only explicit lease/floor/session holds may retry. A file-I/O or
                    # programming defect must never become an unbounded model restart.
                    retry_codes = {'xt_session_recoverable', 'xt_lease_ended', 'xt_resource_yield',
                                   'lease_insufficient_vram', 'lease_already_held',
                                   'demucs_lease_unavailable', 'demucs_lease_ended',
                                   'demucs_lease_preempted', 'demucs_guard_yield', 'demucs_resource_yield'}
                    code = (fault or {}).get('code')
                    if kind == 'resource' and code == 'xt_owner_cancelled':
                        kind = 'cancelled'
                    elif kind == 'resource' and code not in retry_codes:
                        kind = 'fatal'
                        message += '; this X-Translator resource code is not eligible for automatic recovery'
                    if kind == 'resource' and code in retry_codes:
                        recovered_xt = self.store.retry_xt_session(job_id, index, token, message,
                            limit=int(self.config['xt_max_session_resets']),
                            wait_resources=code != 'xt_session_recoverable')
                        if recovered_xt is None:
                            kind = 'cancelled'
                        elif not recovered_xt:
                            kind = 'fatal'
                            message += '; bounded X-Translator session recovery exhausted'
                        else:
                            self.xt_state['reset_count'] = max(resets_before + 1,
                                                              int(self.xt_state.get('reset_count', 0)))
                if recovered_xt:
                    pass
                elif kind == "cancelled":
                    self.store.settle_claimed_chunk(job_id, index, token, "cancelled")
                elif kind == "resource":
                    if self.store.settle_claimed_chunk(job_id,index,token,'waiting_resources',resource_wait=message):
                        self.store.event(job_id,'waiting_resources',chunk=index,message=message)
                elif self.store.settle_claimed_chunk(job_id, index, token, "error", error=message):
                    self.store.event(job_id, "error", chunk=index, message=message, fault=fault)
            finally:
                self.active_owner = None
                self.active_job = None
                self.summarize(job_id)
                self.wake.set()
        if self.actor_pipeline:
            self.actor_pipeline.close()
            self.actor_pipeline = None
        self._close_serial_analysis()

    def publish(self, job_id):
        """Assemble incrementally and publish a derived file; never overwrite the source."""
        owner = ProcessOwner()
        self.publication_owners[job_id] = owner
        try:
            with self.publication_lock:
                return self._publish(job_id, owner)
        finally:
            self.publication_owners.pop(job_id, None)

    def _publish(self, job_id, owner):
        job = self.store.job(job_id)
        if not job or job["state"] != "complete":
            raise ValueError("Every chunk must be ready or explicitly reviewed before publishing")
        if not self.config["publish_dir"]:
            raise ValueError("Configure publish_dir on the shared NAS folder")
        spec = job["payload"]
        existing = spec.get("published")
        if existing and existing.get("revision") == spec.get("revision", 1):
            if Path(existing["path"]).is_file() and file_hash(existing["path"]) == existing["sha256"]:
                return existing
        directory = self.config.cache / job_id
        concat = directory / "audio-concat.txt"
        paths = [self.audio_chunk(job_id, i, spec.get("revision", 1), wait=False)
                 for i in range(spec["chunks_total"])]
        def escaped(p):
            return str(p).replace("\\", "/").replace("'", "'\\''")
        concat.write_text("".join("file '" + escaped(p) + "'\n" for p in paths), encoding="utf-8")
        output = directory / f"english-r{spec.get('revision', 1)}.mkv"
        subtitle_options = []
        subtitles = [s for s in spec["media"]["streams"] if s.get("codec_type") == "subtitle"]
        for number, stream in enumerate(subtitles):
            if stream.get("codec_name") in ("mov_text", "text", "webvtt"):
                subtitle_options.extend([f"-c:s:{number}", "srt"])
        owner.run([self.config["ffmpeg"], "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-i", spec["source"], "-f", "concat", "-safe", 0, "-i", concat,
                   "-map", "0:v?", "-map", "1:a:0", "-map", "0:a?", "-map", "0:s?", "-map", "0:t?",
                   "-map_metadata", "0", "-map_chapters", "0", "-c", "copy",
                   "-c:a:0", "aac", "-b:a:0", "192k", "-metadata:s:a:0", "language=eng",
                   "-metadata:s:a:0", "title=Movie Dub English",
                   "-disposition:a", "0", "-disposition:a:0", "default",
                   *subtitle_options, "-y", output], timeout=max(600, spec["duration"] * 2))
        metadata = probe(self.config, owner, output)
        if abs(metadata["duration"] - spec["duration"]) > .5:
            raise ValueError("Published version has a different movie duration")
        destination_dir = Path(self.config["publish_dir"]) / (safe_name(spec["title"]) + " - Movie Dub")
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / (safe_name(spec["title"]) + f" - English {job_id[:8]}-r{spec.get('revision', 1)}.mkv")
        if destination.exists():
            raise ValueError("A published destination already exists; originals are never replaced")
        temporary = destination.with_suffix(".partial")
        with output.open("rb") as src, temporary.open("wb") as dst:
            for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                owner.check()
                dst.write(block)
        digest = file_hash(output)
        if file_hash(temporary) != digest:
            raise ValueError("Published copy does not match the prepared file")
        owner.check()
        current = self.store.job(job_id)
        if current["state"] == "cancelled" or current["payload"].get("revision", 1) != spec.get("revision", 1):
            raise Cancelled("Publication was cancelled or revised")
        os.replace(temporary, destination)
        published = {"path": str(destination), "sha256": digest,
                     "duration": metadata["duration"], "at": time.time(),
                     "revision": spec.get("revision", 1)}
        self.store.update_job(job_id, published=published)
        self.store.event(job_id, "published", filename=destination.name)
        return published
