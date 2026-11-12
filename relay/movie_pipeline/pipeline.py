"""MoviePipeline: the app's movie-dubbing producer. ONE coordinator thread plus two warm workers over bounded queues.
relay.movie_pipeline, W10; moved from movie-trial-v1/pipeline/coordinator.py.

Started by relay.coordinator.Coordinator.start() instead of the serial loop when actor_beta.movie_pipeline.enabled is
true. It runs inside the normal app server, so a dubbed TV/PC session (TVService enqueue/request, producer_policy
queue-ahead) is processed automatically:
  separation -> ASR/translation -> speaker matching -> Vox -> timing/mix -> settle 'ready' -> TV loop HLS mux.

Ownership (only this thread writes them):
  * chunk claims        the real Store.claim(): playback-priority order and the relay's job-state rules. The claim
                        token plus the job revision is the v4 chunk fence. A chunk is never admitted again while a
                        worker of an older claim of it still runs.
  * actor ids           actors.commit_actors() before ANY render of a prepared window: a fenced rebind against the
                        committed bank, then reconcile or invalidate (re-decide on the real bank).
  * reference receipts  speculative preparation and a live rebind never write them concurrently. Every ready
                        rebind-capable render holds new analysis dispatch; a running analysis finishes first.
                        Committed-plan renders stay parallel, with ActorPipeline's prepared-required guard preventing
                        an unexpected cache miss from falling through into fresh reference admission.
  * render + v4 ledger  worker.process_actor_chunk under a token-checked _ClaimedStore.
  * settlement          outcomes.classify: cancelled / waiting_resources / error (fatal), or a recoverable window
                        published 'ready' as original audio with the committed carried English (fix 5).
  * publication order   a chunk is published by settling it 'ready'; the TV loop muxes it into the viewer's contiguous
                        playlist. Render order is the best CURRENT store priority.

GPU: the resident Demucs worker, the ONE warm analysis worker (analysis_service.AnalysisService: ASR, translation,
diarization, speaker embeddings) and the Vox engines run inside the ONE movie GPU lease's Job, through views of
leaseview.LeaseHub fed by W7's holder relay.app_runtime.movie_lease.holder(config) (INTERFACES.md, LEASE VIEW API).
This server process loads no model. The one AnalysisService is constructed here and injected into BOTH ActorPipeline
instances (set_analysis_service); its child holds the one raw-inference cache (<cache>/analysis-cache).

Fault records (relay.movie_faults, through outcomes.classify) drive settlement: fatal (configuration/path/model/CUDA/
runtime) -> 'error'; resource (a real lease/VRAM/preemption hold) -> 'waiting_resources'; recoverable (cue
uncertainty) -> original window; cancelled. Every receipt line carries the fault record. Counters keep generated
English windows separate from original-audio windows: an original fallback never counts as rendered English.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import logging
import types

from . import actors as _actors
from . import outcomes as _outcomes
from . import snapshot as _snapshot
from . import stages as _stages
from ..actor_beta_timing import publication_accounting

LOG = logging.getLogger('relay.movie_pipeline')

RETRY_RESOURCES_S = 30.0          # default of gpu_lease.retry_seconds: how often a held dub looks whether the card is free
RETRY_RESOURCES_MAX_S = 300.0     # default of gpu_lease.retry_max_seconds: ceiling of the back-off after a retry round failed again
RETRY_HELD_DELAY_S = 60.0         # without a probe verdict a held window is requeued only after this long (the old rule)
PROBE_FRESH_S = 5.0               # an admission reuses a probe this young: one claim round costs one process scan
FENCE_POLL_S = 1.0
RELEASE_GRACE_S = 4.0              # no unfinished job for this long: the models leave RAM and VRAM (playback stopped / film done)
RELEASE_POLL_S = 1.0
REFUSED_FRESH_S = 150.0            # a refused lease acquire counts as 'waiting for the card' this long (blind retry rounds are 91 s apart)
TERMINAL_JOB_STATES = ('complete', 'error', 'cancelled', 'needs_review')


class _Record:
    def __init__(self, claim, owner):
        self.claim, self.owner = claim, owner
        # deferred -> separating -> separated -> analysing -> committed -> rendering
        self.state = 'separating'
        self.separated = None
        self.prepared = None
        self.actor_commit = None
        self.serial_reference_write = False  # a rejected prepared reuse must retry without speculative analysis
        self.warmed = False                  # its raw inference was pre-run during the previous window's render (once)
        self.snapshot_dir = None
        self.pending_settle = None       # (kind, reason, stage, detail, fault): settle once no worker of it runs
        self.requeue = False             # a background window that gave way: its claim returns to 'pending'
        self.outstanding = 0             # workers of THIS claim still running
        self.times = {'claimed': time.time()}


def _relay_modules():
    from .. import (actor_beta_hook, actor_pipeline, beta_line_ledger, coordinator, first_pass_parent, process,
                    resource_budget, speakers, store)
    return types.SimpleNamespace(actor_beta_hook=actor_beta_hook, actor_pipeline=actor_pipeline,
                                 beta_line_ledger=beta_line_ledger, coordinator=coordinator,
                                 first_pass_parent=first_pass_parent, process=process, resource_budget=resource_budget,
                                 speakers=speakers, store=store)


def options(config):
    data = getattr(config, 'data', config)
    beta = data.get('actor_beta') if hasattr(data, 'get') else None
    if not isinstance(beta, dict) or beta.get('enabled') is not True:
        return None
    movie = beta.get('movie_pipeline')
    return movie if isinstance(movie, dict) and movie.get('enabled') is True else None


def _lease_seconds(config, key, default, low, high):
    """gpu_lease.<key> in seconds, clamped; the default for anything unreadable (config.py validates the real file)."""
    try:
        data = getattr(config, 'data', config)
        value = float((data.get('gpu_lease') or {}).get(key, default))
    except (AttributeError, TypeError, ValueError):
        return float(default)
    return min(float(high), max(float(low), value)) if value == value else float(default)


class MoviePipeline:
    # GPU retry state (class defaults; __init__ reads gpu_lease.retry_seconds / retry_max_seconds).
    _retry_base, _retry_max = RETRY_RESOURCES_S, RETRY_RESOURCES_MAX_S
    _retry_wait, _retry_at, _retry_released = RETRY_RESOURCES_S, 0.0, 0
    _gpu_wait = None                 # {'code', 'detail', 'since', 'checked', 'probes'} while the card is known to be unavailable
    _probe_cache = None              # (monotonic time, record) of the latest holder.probe()
    _probe_enabled = True            # gpu_lease.probe_before_retry; False: the old blind rule, exactly
    _heat = None                     # relay.gpu_heat.HeatGuard (F28), built by __init__; None: no heat guard

    def __init__(self, config, store, *, coordinator, queues=None, poll_s=None):
        relay = _relay_modules()
        opts = options(config) or {}
        queues = dict(queues or opts.get('queues') or {'separation': 3, 'analysis': 1})
        self.config, self.store, self.relay = config, store, relay
        self.run_dir = Path(config.state) / 'movie-pipeline'
        self.poll_s = float(poll_s or opts.get('poll_seconds', 0.5))
        # actor_beta.movie_pipeline.warm_next_window (default on): pre-run the next window's raw inference during a render
        self.warm_next = opts.get('warm_next_window', True) is True
        self.failed = None
        self.stop_event = threading.Event()
        self.stop_reason = None
        self.results = queue.Queue()
        self.sep_bound = max(1, int(queues.get('separation', 3)))
        self.ana_bound = max(1, int(queues.get('analysis', 1)))
        self.sep_q = queue.Queue(maxsize=self.sep_bound)
        self.ana_q = queue.Queue(maxsize=self.ana_bound)
        self.max_inflight = self.sep_bound + self.ana_bound + 1
        self.records = {}                # (job, idx, token) -> _Record
        self.busy = {}                   # (job, idx) -> outstanding worker count over ALL claims of that chunk
        self.deferred = []               # claims of busy chunks, admitted once the older workers returned
        self.analysing = None            # (job, idx, token) whose analysis is running
        self.jobs = coordinator                     # the app's Coordinator: summarize(), stop_event
        self.claimed_store = _outcomes.trial_claimed_store(relay.coordinator._ClaimedStore)
        self.analysis_pipeline = relay.actor_pipeline.ActorPipeline(config)
        self.render_pipeline = relay.actor_pipeline.ActorPipeline(config)
        self.log_path = self.run_dir / 'pipeline-log.jsonl'
        self.snapshots = self.run_dir / 'snapshots'
        self.threads = []
        self.on_job = None
        self.jobs_seen = set()
        self._retry_base = _lease_seconds(config, 'retry_seconds', RETRY_RESOURCES_S, 5.0, 600.0)
        self._retry_max = max(self._retry_base,
                              _lease_seconds(config, 'retry_max_seconds', RETRY_RESOURCES_MAX_S, 30.0, 3600.0))
        self._retry_wait = self._retry_base
        self._probe_enabled = (getattr(config, 'data', config).get('gpu_lease') or {}).get('probe_before_retry', True) is not False
        from ..gpu_heat import HeatGuard
        limit = _lease_seconds(config, 'max_gpu_celsius', 85, 0, 95)
        self._heat = HeatGuard(limit, min(_lease_seconds(config, 'resume_gpu_celsius', 78, 40, 93), max(40.0, limit - 2)))
        self._idle_since = None          # monotonic time the pipeline last had no unfinished work (models are released after RELEASE_GRACE_S)
        self._models_released = False
        self._release_check_at = 0.0
        # Background finish (tv.background_finish): the gate (setting, no playback, PC idle, GPU temperature), whether
        # it is open now, and the keeper of the below-normal process priority.
        from ..background_gate import BackgroundGate, BackgroundPriority
        try:
            tv_cfg = config['tv']
        except (KeyError, TypeError):
            tv_cfg = None
        self.background_gate = BackgroundGate(
            tv_cfg, playback=lambda: bool(getattr(coordinator, 'playback_active', lambda: False)()))
        self.background_priority = BackgroundPriority()
        self._background_open, self._background_reason = False, None
        # english_windows counts only windows settled ready with generated English; original-audio outcomes are
        # counted apart and never as rendered English. fallback_original includes original-unverified windows.
        self.counts = {'claimed': 0, 'render_attempts': 0, 'english_windows': 0, 'english_generated_seconds': 0.0,
                       'english_scope': 'receipt-backed generated PCM mixed into settled chunks; not foreign-source, served or heard proof',
                       'fallback_original': 0,
                       'original_english_source': 0, 'no_dialogue': 0, 'settled': {}, 'actor_commits': {},
                       'faults': {}}
        self.analysis = None
        # The ONE movie GPU lease of this process: W7's holder (relay.app_runtime.movie_lease.holder) feeds the
        # LeaseHub, whose views (leaseview.py, LEASE VIEW API) are shared by Vox and the resident Demucs.
        from ..app_runtime import movie_lease
        from .leaseview import LeaseHub
        self.hub = LeaseHub(movie_lease.holder(config))
        self.lease_source = self.hub.source()
        self._closed = False
        # Vox (W3's actor_beta_hook, APP_RUNTIME_LEASE) takes views of THIS hub (keyed on source.key): no second lease.
        relay.actor_beta_hook.register_lease_hub(self.hub)
        try:
            _actors.install(relay.speakers)
            from .resident_demucs import ResidentDemucs
            beta = config['actor_beta']
            self.resident = ResidentDemucs(python=config['python'], relay_dir=Path(relay.first_pass_parent.HERE),
                                           model_dir=relay.first_pass_parent.config_path(config, 'separation_model_dir'),
                                           model_pins=beta['demucs_model_pins'],
                                           threads=int(beta.get('demucs_threads', 4)),
                                           device=beta.get('demucs_device', 'cpu'), lease_source=self.lease_source,
                                           log_path=self.run_dir / 'logs' / 'resident-demucs.stderr.log',
                                           resource_yield=relay.resource_budget.ResourceYield,
                                           cancelled=relay.process.Cancelled, emit=self.emit,
                                           vram_reader=self._wddm_reader)
            relay.first_pass_parent.RESIDENT = self.resident
            # The ONE AnalysisService of this process: GPU models only in its lease-owned child (same hub as Demucs
            # and Vox). Injected into BOTH ActorPipeline instances; the actor author's set_analysis_service stores it.
            from .analysis_service import AnalysisService
            self.analysis = AnalysisService(config, self.lease_source,
                                            log_path=self.run_dir / 'logs' / 'analysis-worker.stderr.log',
                                            emit=self.emit, vram_reader=self._wddm_reader)
            for pipeline in (self.analysis_pipeline, self.render_pipeline):
                inject = getattr(pipeline, 'set_analysis_service', None)
                if not callable(inject):
                    from ..movie_faults import MovieConfigurationError
                    raise MovieConfigurationError('actor_pipeline_without_analysis_service',
                                                  'ActorPipeline.set_analysis_service is missing')
                inject(self.analysis)
        except BaseException:
            if self.analysis is not None:
                self.analysis.close()
            if relay.first_pass_parent.RESIDENT is getattr(self, 'resident', None):
                relay.first_pass_parent.RESIDENT = None
            self._release_hub('pipeline_init_failed')
            raise

    def _wddm_reader(self, pid):
        """W3's WDDM per-process GPU memory reader (actor-tts/runtime-adapter vox_engine_host.wddm_process_vram),
        imported through W3's own pin-verified adapter loader (actor_beta_hook.automatic_module); loaded once."""
        module = getattr(self, '_vox_host_module', None)
        if module is None:
            module = self._vox_host_module = self.relay.actor_beta_hook.automatic_module(self.config).adapter_module(
                'vox_engine_host')
        return module.wddm_process_vram(pid)

    def _release_hub(self, reason):
        """Unregister the hub from the Vox hook BEFORE closing it (every exit path: close, init/run failure)."""
        try:
            self.relay.actor_beta_hook.unregister_lease_hub(self.hub)
        finally:
            self.hub.close(reason)                   # after the engines stopped: synchronous drain, lock freed

    # ---- events / state (player and logs) -------------------------------------------------------------------------
    def emit(self, kind, **fields):
        LOG.info('movie %s %s', kind, json.dumps(fields, default=str)[:2000])
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.run_dir / 'events.jsonl', 'a', encoding='utf-8') as handle:
                handle.write(json.dumps({'kind': kind, 't': time.time(), **fields}, default=str) + chr(10))
        except OSError:
            pass

    def _gpu_waiting(self, lease):
        """{'code', 'since'} while dub work is waiting for the graphics card, else None. Sources: the retry gate's
        probe (self._gpu_wait, when that change is installed), a refused lease acquire younger than REFUSED_FRESH_S,
        a preemption. Never reported when no dub job is unfinished: a preemption hours ago must not put a notice on
        the TV's library screen."""
        found = None
        gate = getattr(self, '_gpu_wait', None)
        if gate:
            found = {'code': gate.get('code'), 'since': gate.get('since')}
        elif lease.get('state') == 'refused' and time.time() - float(lease.get('at') or 0.0) <= REFUSED_FRESH_S:
            found = {'code': lease.get('reason'), 'since': lease.get('since')}
        elif lease.get('state') == 'preempted':
            found = {'code': lease.get('reason'), 'since': lease.get('since')}
        if found is None or not self._work_pending():
            return None
        return found

    def state(self):
        """Producer state for /tv-api/producer: running | preempted | failed | stopped. 'preempted' also covers a
        dub that was REFUSED the card (it never held it): every released TV player knows that name. waiting_gpu
        (bool) and since (unix seconds) are additive."""
        if self.failed:
            return {'state': 'failed', 'reason': self.failed, 'producer': 'movie-pipeline'}
        if self.stop_event.is_set():
            return {'state': 'stopped', 'reason': self.stop_reason, 'producer': 'movie-pipeline'}
        lease = self.hub.state()
        analysis = self.analysis.device_report() if self.analysis is not None else None
        waiting = self._gpu_waiting(lease)
        if waiting is not None:
            return {'state': 'preempted', 'reason': waiting['code'], 'waiting_gpu': True, 'since': waiting['since'],
                    'producer': 'movie-pipeline', 'lease': lease, 'analysis': analysis}
        return {'state': 'running', 'reason': None, 'waiting_gpu': False, 'producer': 'movie-pipeline', 'lease': lease,
                'analysis': analysis,
                'counts': {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.counts.items()}}

    def cancel_job(self, job_id):
        """Coordinator.cancel(): cancel this job's in-flight windows now (they then settle 'cancelled')."""
        for record in list(self.records.values()):
            if record.claim.job_id == job_id:
                record.owner.cancel()

    # ---- background finish ---------------------------------------------------------------------------------------
    def yield_background(self, background_jobs):
        """Any thread (Coordinator.request of a film being watched, the gate watcher): every window in flight for a
        BACKGROUND job gives way now. Its owner is cancelled; the claim then settles back to 'pending' (see _settle)
        and is produced again when the gate is open and no other job is unfinished. Returns how many gave way."""
        yielded = 0
        for record in list(self.records.values()):
            if record.claim.job_id in background_jobs and not record.requeue and record.pending_settle is None:
                record.requeue = True
                record.owner.cancel()
                yielded += 1
        if yielded:
            self.emit('background_yielded', windows=yielded)
        return yielded

    def _background_jobs(self):
        jobs = getattr(self.jobs, 'background_jobs', None)
        return jobs() if callable(jobs) else ()

    def _background_tick(self):
        """Coordinator thread, once a second: evaluate the gate for the background job (if any), log every change,
        make its windows give way when the gate closed, and keep the child processes at below-normal priority while
        ONLY background windows are in production."""
        gate = getattr(self, 'background_gate', None)
        jobs = self._background_jobs() if gate is not None else ()
        allowed, reason = gate.check() if jobs else (False, None)
        if (allowed, reason) != (getattr(self, '_background_open', False), getattr(self, '_background_reason', None)):
            self.emit('background_gate', open=allowed, reason=reason)
        self._background_open, self._background_reason = allowed, reason
        if jobs and not allowed:
            self.yield_background(jobs)
        priority = getattr(self, 'background_priority', None)
        if priority is not None:
            if allowed and self.records and all(r.claim.job_id in jobs for r in self.records.values()):
                priority.lower()
            elif priority.lowered:
                priority.restore()

    def _background_watch(self):
        """Own thread, because the coordinator thread is busy for a whole render: once a second, while background
        windows are in flight, re-check the gate and make them give way at once when it closed (keyboard or mouse
        input, a playback, the GPU at its limit). While it stays open and only background windows are in production,
        child processes started meanwhile (ffmpeg, a worker) are put at below-normal priority here too."""
        while not self.stop_event.wait(1.0):
            jobs = getattr(self.jobs, '_background', None) or ()
            records = list(self.records.values())
            if jobs and any(r.claim.job_id in jobs for r in records):
                allowed, _ = self.background_gate.check()
                if not allowed:
                    self._background_open = False
                    self.yield_background(jobs)
                elif all(r.claim.job_id in jobs for r in records):
                    self.background_priority.lower()

    # ---- lifecycle ------------------------------------------------------------------------------------------------
    def stop(self, why='stop'):
        if not self.stop_event.is_set():
            self.stop_reason = why
            self.stop_event.set()
        for record in list(self.records.values()):
            record.owner.cancel()

    def start_workers(self):
        for name, target in (('movie-separation', self._separation_worker), ('movie-analysis', self._analysis_worker),
                             ('movie-background-watch', self._background_watch)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)
        # F15: the film scout feeds Diart ahead of playback, only while the analysis worker is warm and no window waits for it
        from .. import film_scout
        self.scout = film_scout.ScoutHost(self.config, self.store, self.analysis, terminal=TERMINAL_JOB_STATES,
                                          busy=lambda: self.analysing is not None or not self.ana_q.empty())
        self.scout.start()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if getattr(self, 'scout', None) is not None:
            self.scout.stop()
        self.stop(self.stop_reason or 'close')
        for q in (self.sep_q, self.ana_q):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for thread in self.threads:
            thread.join(timeout=30)
        for pipeline in (self.analysis_pipeline, self.render_pipeline):
            try:
                pipeline.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            try:
                if self.analysis is not None:
                    self.analysis.close()            # stops its GPU child BEFORE the hub releases the lease
            finally:
                self.resident.close()
        finally:
            if self.relay.first_pass_parent.RESIDENT is self.resident:
                self.relay.first_pass_parent.RESIDENT = None
            self._release_hub('pipeline_close')

    # ---- workers (no shared-state writes; results are data) -------------------------------------------------------
    def _separation_worker(self):
        while not self.stop_event.is_set():
            item = self.sep_q.get()
            if item is None:
                return
            claim, owner = item
            self.results.put(('separated', _stages.separate_window(self.config, self.store, claim, owner)))

    def _analysis_worker(self):
        while not self.stop_event.is_set():
            item = self.ana_q.get()
            if item is None:
                return
            if item[0] == 'warm':
                _, claim, owner, separated = item
                self.results.put(('warmed', _stages.warm_window(self.config, self.store, claim, owner,
                                                                self.analysis_pipeline, separated)))
                continue
            claim, owner, separated, snapshot_claimed, base = item
            self.results.put(('prepared', _stages.prepare_window(self.config, snapshot_claimed, claim, owner,
                                                                 self.analysis_pipeline, separated, base, _actors)))

    # ---- the one coordinator thread ---------------------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except BaseException as error:  # noqa: BLE001 - visible to the player as 'failed', never silent
            self.failed = '%s: %s' % (type(error).__name__, str(error)[:800])
            LOG.exception('movie pipeline failed')
            self.emit('failed', error=self.failed)
            self.stop('failed')
            self.close()                             # stops workers/engines, unregisters the hub, then closes it

    def _run(self):
        self.start_workers()
        self.emit('coordinator_started', max_inflight=self.max_inflight,
                  queues={'separation': self.sep_bound, 'analysis': self.ana_bound})
        fence_at = 0.0
        while not self.stop_event.is_set() and not self.jobs.stop_event.is_set():
            now = time.monotonic()
            self.resident.idle_check()
            self.analysis.idle_check()
            self._release_idle_models(now)
            self._retry_resources(now)
            if now >= fence_at:
                self._cancel_superseded()
                self._background_tick()
                fence_at = now + FENCE_POLL_S
            worked = self._drain_results()
            worked = self._admit() or worked
            worked = self._dispatch_analysis() or worked
            worked = self._dispatch_warm() or worked
            if self._render_next():
                continue
            if not worked:
                try:
                    kind, result = self.results.get(timeout=self.poll_s)
                except queue.Empty:
                    continue
                self._on_result(kind, result)
        self.emit('coordinator_stopped', reason=self.stop_reason, counts=self.counts)

    @staticmethod
    def _key(claim):
        return (claim.job_id, claim.index, claim.token)

    @staticmethod
    def _chunk(claim):
        return (claim.job_id, claim.index)

    def _busy_add(self, record):
        record.outstanding += 1
        self.busy[self._chunk(record.claim)] = self.busy.get(self._chunk(record.claim), 0) + 1

    def _busy_done(self, record):
        record.outstanding = max(0, record.outstanding - 1)
        key = self._chunk(record.claim)
        left = self.busy.get(key, 1) - 1
        if left <= 0:
            self.busy.pop(key, None)
        else:
            self.busy[key] = left

    # ---- GPU availability: probe before retrying (gpu_lease.retry_seconds / retry_max_seconds) ------------------------
    def _lease_held(self):
        try:
            return self.hub.state().get('state') == 'held'
        except Exception:  # noqa: BLE001 -- no hub (tests) or an unreadable state: not held
            return False

    def _gpu_probe(self, now, max_age=0.0):
        """holder.probe() (relay.app_runtime.movie_lease.Holder.probe), reused for max_age seconds. None when the
        holder has no probe or it failed: every caller then behaves exactly as before this check existed."""
        if not self._probe_enabled:
            return None
        cached = self._probe_cache
        if cached is not None and now - cached[0] <= max_age:
            return cached[1]
        probe = getattr(getattr(getattr(self, 'hub', None), 'holder', None), 'probe', None)
        if not callable(probe):
            return None
        try:
            record = probe()
        except Exception as error:  # noqa: BLE001 -- a probe that cannot run never holds the dub back
            LOG.warning('movie GPU probe failed: %s: %s', type(error).__name__, error)
            return None
        if not isinstance(record, dict):
            return None
        self._probe_cache = (now, record)
        return record

    def _note_gpu_wait(self, record):
        """Remember that the card is unavailable (one log line when it starts or its reason changes, one when it
        ends). record: a probe record, or None / an obtainable one to clear."""
        previous = self._gpu_wait
        if record is None or record.get('obtainable', True):
            if previous is not None:
                self.emit('gpu_wait_ended', code=previous['code'], waited_s=round(time.time() - previous['since'], 1),
                          probes=previous['probes'])
            self._gpu_wait = None
            return
        now = time.time()
        code, detail = str(record.get('code')), str(record.get('detail') or '')[:300]
        if previous is None or previous['code'] != code:
            self.emit('gpu_wait', code=code, detail=detail, free_mib=record.get('free_mib'))
        self._gpu_wait = {'code': code, 'detail': detail, 'since': previous['since'] if previous is not None else now,
                          'checked': now, 'probes': (previous['probes'] if previous is not None else 0) + 1}

    def _waiting_count(self):
        count = getattr(self.store, 'waiting_resources', None)
        if not callable(count):
            return 1                                 # a store without the counter: always look (the old behaviour)
        try:
            return int(count())
        except Exception:  # noqa: BLE001
            return 1

    def _retry_resources(self, now):
        """Requeue the windows held in waiting_resources, once per period, and only when the card can be had.
        * nothing held: nothing is probed or requeued;
        * no lease and the probe names a hold (the other app is running, no VRAM, a pending drain): nothing is
          requeued, so no window is claimed, extracted from the NAS and refused again; the probe repeats every
          gpu_lease.retry_seconds, so dubbing resumes within one period of the hold ending;
        * the probe allows it: every held window is requeued at once (delay 0). If that round is refused again (a
          hold the probe cannot see) the next round waits twice as long, up to gpu_lease.retry_max_seconds;
        * a lease is held, or there is no probe: the old rule (windows held for 60 s are requeued)."""
        if now < self._retry_at:
            return
        if not self._waiting_count():
            self._note_gpu_wait(None)
            self._retry_wait, self._retry_released = self._retry_base, 0
            self._retry_at = now + self._retry_base
            return
        heat = self._heat_block(now)
        if heat is not None:                         # hot: nothing is requeued; looked at again after one base period
            self._note_gpu_wait(heat)
            self._retry_wait, self._retry_released = self._retry_base, 0
            self._retry_at = now + self._retry_base
            return
        held = self._lease_held()
        record = None if held else self._gpu_probe(now)
        if record is not None and not record.get('obtainable', True):
            self._note_gpu_wait(record)
            self._retry_wait, self._retry_released = self._retry_base, 0
            self._retry_at = now + self._retry_base
            return
        self._note_gpu_wait(None)
        if record is not None and self._retry_released and not held:   # the probe let the last round through and it still got no lease
            self._retry_wait = min(self._retry_max, max(self._retry_base, self._retry_wait * 2))
        else:
            self._retry_wait = self._retry_base
        self._retry_released = int(self.store.retry_resources(delay=0 if record is not None else RETRY_HELD_DELAY_S) or 0)
        self._retry_at = now + self._retry_wait

    def _heat_block(self, now):
        """The heat guard's hold record (code 'gpu_hot', relay/gpu_heat.py), or None. None without a guard (objects
        built with __new__, as the tests do) or when the guard itself fails."""
        heat = self._heat
        if heat is None:
            return None
        try:
            return heat.block(now)
        except Exception as error:  # noqa: BLE001 -- a guard that cannot run never stops a film someone is watching
            LOG.warning('movie heat guard failed: %s: %s', type(error).__name__, error)
            return None

    def _admission_block(self, now):
        """The hold record when a window claimed just now must not start: the card is hot (heat guard, also while the
        lease is held), or no lease is held and the probe names a hold. Else None."""
        heat = self._heat_block(now)
        if heat is not None:
            self._note_gpu_wait(heat)
            return heat
        if self._gpu_wait is not None and self._gpu_wait.get('code') == 'gpu_hot':
            self._note_gpu_wait(None)                # cooled down: the heat wait ends here (one gpu_wait_ended line)
        if self._lease_held():
            return None
        record = self._gpu_probe(now, PROBE_FRESH_S)
        if record is None or record.get('obtainable', True):
            return None
        self._note_gpu_wait(record)
        return record

    # ---- admission -------------------------------------------------------------------------------------------------
    def _admit(self):
        admitted = started = False
        for record in list(self.deferred):           # claims that waited for an older claim's workers
            if self._chunk(record.claim) not in self.busy and not self.sep_q.full():
                self.deferred.remove(record)
                self._start_separation(record)
                admitted = started = True
        while len(self.records) < self.max_inflight and not self.sep_q.full() and not self.stop_event.is_set():
            # Background chunks are asked for only while the gate is open (store.claim then still puts every other
            # unfinished job first).
            item = self.store.claim(background=True) if getattr(self, '_background_open', False) else self.store.claim()
            if not item:
                break
            job = self.store.job(item['job_id'])
            revision = int(((job or {}).get('payload') or {}).get('revision', 1))
            claim = _stages.Claimed(job_id=item['job_id'], index=int(item['idx']), token=int(item['claim_token']),
                                    revision=revision, priority=int(item['priority']), claimed_at=time.time())
            self.store.update_job(claim.job_id, 'processing')
            if claim.job_id not in self.jobs_seen:
                self.jobs_seen.add(claim.job_id)
                if self.on_job is not None:
                    self.on_job(job)
            record = _Record(claim, self.relay.process.ProcessOwner())
            self.records[self._key(claim)] = record
            self.counts['claimed'] += 1
            self.emit('chunk_claimed', job_id=claim.job_id, index=claim.index, token=claim.token, priority=claim.priority)
            admitted = True
            block = self._admission_block(time.monotonic())
            if block is not None:
                # The card cannot be had: hold the window now, before its audio is extracted from the NAS. The job
                # goes to waiting_resources (summarize), so this loop claims nothing more of it.
                self._settle(record, 'resource', '%s: %s' % (block['code'], block['detail']), 'admission',
                             fault={'kind': 'resource', 'code': block['code'], 'detail': block['detail']})
                continue
            if self._chunk(claim) in self.busy:
                record.state = 'deferred'
                self.deferred.append(record)
            else:
                self._start_separation(record)
            started = True
        if started:
            self._prewarm_analysis()
        return admitted

    def _work_pending(self):
        """True while any job has unfinished work (a dub being played or produced), or a window is in flight or queued. Unknown counts as
        pending: models are never dropped on a guess."""
        if self.records or self.deferred or not self.sep_q.empty() or not self.ana_q.empty():
            return True
        try:
            # A background job counts as work only while its gate is open: paused (PC in use, GPU hot, a playback), it
            # must not keep the models in RAM and VRAM.
            waiting = getattr(self, '_background_open', False)
            return any(job['state'] not in TERMINAL_JOB_STATES
                       and (waiting or (job.get('payload') or {}).get('background') is not True)
                       for job in self.store.jobs())
        except Exception:  # noqa: BLE001
            return True

    def _release_idle_models(self, now):
        """Models live only while a dub is being produced for playback: RELEASE_GRACE_S after the last job ended (playback stopped, the
        job was cancelled, or the whole film is dubbed) the speech/translation worker, the resident Demucs and the Vox engines exit and
        their RAM and VRAM are free. A seek or a new playback starts them again."""
        if now < self._release_check_at:
            return
        self._release_check_at = now + RELEASE_POLL_S
        if self._work_pending():
            self._idle_since, self._models_released = None, False
            return
        if self._idle_since is None:
            self._idle_since = now
        if self._models_released or now - self._idle_since < RELEASE_GRACE_S:
            return
        results = {}
        for name, release in (('analysis', self.analysis.unload), ('demucs', self.resident.unload)):
            try:
                results[name] = release('playback_stopped')
            except Exception as error:  # noqa: BLE001
                results[name] = 'error:' + type(error).__name__
        if 'busy' in results.values():
            return                                   # a request is still unwinding: ask again at the next poll
        threading.Thread(target=self._release_vox, name='movie-vox-release', daemon=True).start()
        self._models_released = True
        self.emit('models_released', reason='no unfinished dub work', **results)

    def _release_vox(self):
        try:
            released = self.relay.actor_beta_hook.release_vox_engines(self.config, 'playback_stopped')
            self.emit('vox_released', keepers=released)
        except Exception as error:  # noqa: BLE001 - the keeper's own idle release still follows
            self.emit('vox_release_failed', error=type(error).__name__)

    def _prewarm_analysis(self):
        """Start the analysis worker and warm its models while the first chunk is still being separated (see
        AnalysisService.prewarm). Best effort: the first real request starts the worker if this does not."""
        prewarm = getattr(self.analysis, 'prewarm', None)
        if not callable(prewarm):
            return
        try:
            prewarm(translation=self.config['translation_engine'] == 'qwen3')
        except Exception:  # noqa: BLE001 - an optimisation never stops claiming
            pass

    def _start_separation(self, record):
        record.state = 'separating'
        self._busy_add(record)
        self.sep_q.put((record.claim, record.owner))

    # ---- results ---------------------------------------------------------------------------------------------------
    def _drain_results(self):
        worked = False
        while True:
            try:
                kind, result = self.results.get_nowait()
            except queue.Empty:
                return worked
            self._on_result(kind, result)
            worked = True

    def _on_result(self, kind, result):
        key = self._key(result.claim)
        if kind in ('prepared', 'warmed') and self.analysing == key:
            self.analysing = None
        record = self.records.get(key)
        if record is None:
            return
        self._busy_done(record)
        if kind == 'prepared' and record.snapshot_dir is not None:
            _snapshot.discard(record.snapshot_dir)
            record.snapshot_dir = None
        if record.pending_settle is not None:          # the claim ended while this worker ran: settle now
            self._settle(record, *record.pending_settle)
            return
        if kind == 'warmed':                             # nothing was decided: the window stays 'separated'
            record.times['warmed'] = time.time()
            self.emit('stage', job_id=result.claim.job_id, index=result.claim.index, stage='warm', ok=result.ok,
                      state=result.state, seconds=result.seconds, timing=result.timing, error=result.error or None)
            # Three warm-ups in a row that did not run (wrong lane, a fault): stop trying until the server restarts.
            self._warm_failures = 0 if result.ok else getattr(self, '_warm_failures', 0) + 1
            if self._warm_failures >= 3 and getattr(self, 'warm_next', False):
                self.warm_next = False
                self.emit('warm_disabled', reason=result.error or result.state)
            return
        if kind == 'separated':
            record.separated = result
            record.times['separated'] = time.time()
            self.emit('stage', job_id=result.claim.job_id, index=result.claim.index, stage='separation',
                      ok=result.ok, seconds=result.seconds, producer=result.producer, error=result.error or None)
            if not result.ok:
                self._settle(record, result.error_kind, result.error, 'separation', fault=result.fault)
                return
            record.state = 'separated'
            return
        if kind == 'prepared':
            record.prepared = result
            record.times['prepared'] = time.time()
            self.emit('stage', job_id=result.claim.job_id, index=result.claim.index, stage='analysis',
                      state=result.state, seconds=result.seconds, timing=result.timing, error=result.error or None)
            if result.state == 'failed':
                self._settle(record, result.error_kind, result.error, 'analysis', fault=result.fault)
                return
            self._commit_actors(record)

    def _commit_actors(self, record):
        """Actor ids: fenced rebind of the prepared window against the committed bank BEFORE it can render."""
        claim, prepared = record.claim, record.prepared
        fence = self.relay.beta_line_ledger.fence_of(claim.job_id, claim.index, claim.token, claim.revision)
        try:
            outcome = _actors.commit_actors(self.store, fence, claim.job_id, self.config,
                                            speakers_module=self.relay.speakers,
                                            ledger_module=self.relay.beta_line_ledger, prepared=prepared)
        except Exception as error:  # noqa: BLE001 - Superseded -> cancelled; others classified
            kind, reason = _outcomes.classify(error)
            self._settle(record, kind, reason, 'actor_commit', fault=_stages.fault_of(error))
            return
        if outcome['status'] == 'changed' and prepared.plan_path:
            outcome['invalidated_plan'] = _actors.invalidate_plan(prepared.plan_path, outcome.get('reason'))
        record.actor_commit = outcome
        record.state = 'committed'
        self.counts['actor_commits'][outcome['status']] = self.counts['actor_commits'].get(outcome['status'], 0) + 1
        self.emit('actor_commit', job_id=claim.job_id, index=claim.index, status=outcome['status'],
                  reason=outcome.get('reason'), new_speakers=outcome.get('new_speakers'),
                  mapping=outcome.get('mapping'), invalidated_plan=outcome.get('invalidated_plan'))

    # ---- analysis dispatch -----------------------------------------------------------------------------------------
    def _ready_renders(self):
        return [r for r in self.records.values() if r.state == 'committed' and r.pending_settle is None]

    @staticmethod
    def _render_may_bind(record):
        """A committed plan is safe to overlap ONLY while the actor-side reuse guard is enabled for its render."""
        prepared = record.prepared
        return (record.serial_reference_write or (record.actor_commit or {}).get('status') != 'committed'
                or prepared is None or prepared.state != 'prepared' or not prepared.plan_path)

    PREDECESSOR_BLOCKS = ('separating', 'separated', 'analysing', 'committed')

    def _behind(self, record):
        """True while the previous window of the same film (index - 1) is still ahead of rendering in this pipeline. A window
        is analysed and rendered after it, so the previous window's takes are in the ledger (carry, removal) when this one is
        mixed; rendering out of order left original speech and clicks at the joins (owner review 2026-10-20)."""
        job, previous = record.claim.job_id, record.claim.index - 1
        return any(r is not record and r.claim.job_id == job and r.claim.index == previous and r.state in self.PREDECESSOR_BLOCKS
                   for r in self.records.values())

    def _dispatch_analysis(self):
        if self.analysing is not None or self.ana_q.full():
            return False
        # Inspect ALL ready windows, not only the next one. Otherwise rendering
        # one rebind-capable window could admit new analysis beside a second one.
        if any(self._render_may_bind(r) for r in self._ready_renders()):
            return False
        ready = [r for r in self.records.values() if r.state == 'separated' and r.pending_settle is None and not self._behind(r)]
        if not ready:
            return False
        record = min(ready, key=self._current_priority)
        claim = record.claim
        record.snapshot_dir = self.snapshots / ('%s-chunk-%06d-t%d' % (claim.job_id[:12], claim.index, claim.token))
        fence = self.relay.beta_line_ledger.fence_of(claim.job_id, claim.index, claim.token, claim.revision)
        try:
            snap = _snapshot.snapshot_store(self.store, record.snapshot_dir, self.relay.store.Store, fence=fence,
                                            check_fence=self.relay.beta_line_ledger.check_fence)
            # The capture base is EXACTLY what preparation reads: the snapshot's own bank, not a later real-store read.
            base = snap.speakers(claim.job_id)
        except Exception as error:  # noqa: BLE001
            kind, reason = _outcomes.classify(error)
            self._settle(record, kind, 'snapshot failed: %s' % reason, 'analysis', fault=_stages.fault_of(error))
            return True
        claimed = self.relay.coordinator._ClaimedStore(snap, claim.job_id, claim.index, claim.token)
        record.state = 'analysing'
        self.analysing = self._key(claim)
        self._busy_add(record)
        self.ana_q.put((claim, record.owner, record.separated, claimed, base))
        return True

    def _dispatch_warm(self):
        """While the best committed window is about to render (the coordinator thread is then busy for its ~20 s and the
        analysis worker would sit idle), pre-run the RAW inference of the next window of the same film on the analysis
        worker (stages.warm_window: ASR + speaker analysis). This is NOT a preparation: no snapshot, no ledger, no
        speaker bank, no plan. The window stays 'separated' and is prepared only after its predecessor was published
        (_behind, unchanged), because preparation reads the predecessor's published takes and its committed-window
        record. Never beside a render that may bind references, never twice for one claim."""
        if not getattr(self, 'warm_next', False) or self.analysing is not None or self.ana_q.full() \
                or self.stop_event.is_set():
            return False
        renders = self._ready_renders()
        if not renders or any(self._render_may_bind(r) for r in renders):
            return False
        ready = [r for r in renders if not self._behind(r)]
        if not ready:
            return False
        claim = min(ready, key=self._current_priority).claim            # the window _render_next renders now
        record = next((r for r in self.records.values()
                       if r.state == 'separated' and r.pending_settle is None and not r.warmed
                       and r.claim.job_id == claim.job_id and r.claim.index == claim.index + 1), None)
        if record is None:
            return False
        record.warmed = True
        self.analysing = self._key(record.claim)
        self._busy_add(record)
        self.ana_q.put(('warm', record.claim, record.owner, record.separated))
        return True

    def _current_priority(self, record):
        row = self.store.chunk(record.claim.job_id, record.claim.index)
        return (int(row['priority']) if row else 1 << 30, record.claim.index)

    # ---- render (coordinator thread) -------------------------------------------------------------------------------
    def _render_next(self):
        ready = [r for r in self._ready_renders() if not self._behind(r)]
        if not ready or self.stop_event.is_set():
            return False
        record = min(ready, key=self._current_priority)
        serial = self._render_may_bind(record)
        if serial and self.analysing is not None:
            # Keep the best-priority claim queued. The main loop continues
            # handling results/cancellation; once preparation has returned AND
            # its actor result is handled, self.analysing clears. Do not cancel
            # that worker or run a lower-priority render past this one.
            return False
        record.state = 'rendering'
        claim = record.claim
        record.times['render_start'] = time.time()
        claimed = self.claimed_store(self.store, claim.job_id, claim.index, claim.token)
        self.emit('stage', job_id=claim.job_id, index=claim.index, stage='render_start',
                  plan_status=(record.actor_commit or {}).get('status'))
        self.counts['render_attempts'] += 1             # an attempt, not rendered English (see _finish)
        previous_required = self.render_pipeline.require_prepared_plan
        self.render_pipeline.require_prepared_plan = not serial
        try:
            _stages.render_window(self.config, claimed, claim, record.owner, self.render_pipeline)
        except self.relay.actor_pipeline.PreparedPlanRequired as error:
            if serial:
                # This guard must be disabled for the one serialized retry.
                # An implementation disagreement is terminal, never a retry loop.
                self._settle(record, 'fatal', str(error), 'render',
                             fault={'kind': 'fatal', 'code': 'prepared_guard_not_released', 'detail': str(error)})
                return True
            record.serial_reference_write = True
            record.state = 'committed'
            record.times['reference_rebind_deferred'] = time.time()
            self.emit('reference_rebind_deferred', job_id=claim.job_id, index=claim.index, token=claim.token,
                      reason=str(error), waiting_for_analysis=self.analysing is not None)
            return True                               # retain this exact claim; retry through normal serialized flow
        except _outcomes.NeedsReviewIntercepted as review:
            record.times['render_end'] = time.time()
            self._settle(record, 'recoverable', str(review), 'render_needs_review',
                         detail={'blocked_issues': [i for i in (review.result.get('issues') or [])
                                                    if isinstance(i, dict)][:50],
                                 'pipeline_state': review.result.get('pipeline_state'),
                                 'blocked_reason': review.result.get('beta_blocked_reason')},
                         fault={'kind': 'recoverable', 'code': 'render_needs_review', 'detail': str(review)[:4000]})
            return True
        except Exception as error:  # noqa: BLE001 - classified; never blanket-relabelled
            record.times['render_end'] = time.time()
            kind, reason = _outcomes.classify(error)
            self._settle(record, kind, reason, 'render', fault=_stages.fault_of(error))
            return True
        finally:
            self.render_pipeline.require_prepared_plan = previous_required
        record.times['render_end'] = time.time()
        self.jobs.summarize(claim.job_id)
        self._finish(record, 'settled_by_pipeline', None)
        return True

    # ---- settlement ------------------------------------------------------------------------------------------------
    def _settle(self, record, kind, reason, stage, detail=None, fault=None):
        """Settle THIS claim (token-checked). If a worker of this claim is still running, the settle waits for it, so
        the chunk stays 'working' under this token and no new claim can start beside a stale worker. `fault` is the
        relay.movie_faults record {kind, code, detail} that decided `kind`; it is written to the receipt line."""
        if fault is None:
            fault = {'kind': kind, 'code': 'pipeline_%s' % stage, 'detail': str(reason)[:4000]}
        if record.outstanding > 0:
            record.pending_settle = (kind, reason, stage, detail, fault)
            record.owner.cancel()
            return
        self.counts['faults'][fault.get('code')] = self.counts['faults'].get(fault.get('code'), 0) + 1
        claim = record.claim
        store = self.store
        outcome = kind
        if record.requeue:                               # a background window that gave way (its owner was cancelled by
            # yield_background): whatever the interrupted stage reported, the claim goes back to the queue
            store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'pending')
            outcome = 'yielded'
        elif kind == 'cancelled':
            store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'cancelled')
        elif kind == 'resource':
            if store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'waiting_resources',
                                          resource_wait=str(reason)):
                store.event(claim.job_id, 'waiting_resources', chunk=claim.index, message=str(reason))
        elif kind == 'recoverable' and not record.owner.cancelled.is_set():
            try:
                _outcomes.publish_original(self.config, store, claim, record.owner,
                                           chunk_inputs=_stages._chunk_inputs, stage=stage, reason=reason, detail=detail)
                outcome = 'original_fallback'                # counted in _finish from the settled receipt
            except Exception as error:  # noqa: BLE001 - the fallback itself failed: classified like the relay
                second, why = _outcomes.classify(error)
                outcome = 'fallback_failed_' + second
                if second == 'cancelled':
                    store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'cancelled')
                elif second == 'resource':
                    store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'waiting_resources',
                                               resource_wait=why)
                elif store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'error',
                                                error='original fallback failed: %s (after %s)' % (why, reason)):
                    store.event(claim.job_id, 'error', chunk=claim.index, message=why)
                reason = '%s; fallback: %s' % (reason, why)
        elif kind == 'recoverable':                      # owner cancelled meanwhile: the claim is ending
            store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'cancelled')
            outcome = 'cancelled'
        else:                                            # fatal: explicit error, as coordinator.loop
            if store.settle_claimed_chunk(claim.job_id, claim.index, claim.token, 'error', error=str(reason)):
                store.event(claim.job_id, 'error', chunk=claim.index, message=str(reason))
            outcome = 'error'
        self.jobs.summarize(claim.job_id)
        self._finish(record, outcome, {'stage': stage, 'reason': reason, 'classified': kind, 'detail': detail,
                                       'fault': fault})

    def _finish(self, record, outcome, detail):
        claim = record.claim
        self.records.pop(self._key(claim), None)
        if record in self.deferred:
            self.deferred.remove(record)
        if record.snapshot_dir is not None:
            _snapshot.discard(record.snapshot_dir)
        chunk = self.store.chunk(claim.job_id, claim.index) or {}
        payload = chunk.get('payload') or {}
        state, mode = chunk.get('state'), payload.get('mode')
        ready = state in ('ready', 'approved')
        accounting = publication_accounting(payload)
        english_seconds = accounting['english_generated_seconds'] if ready else 0.0
        english_generated = english_seconds > 0
        if english_generated:
            self.counts['english_windows'] += 1
            self.counts['english_generated_seconds'] = round(
                self.counts['english_generated_seconds'] + english_seconds, 6)
        elif ready and (mode in ('original-trial-fallback', 'original-unverified')
                        or payload.get('pipeline_state') == 'original_fallback'):
            self.counts['fallback_original'] += 1
        elif ready and accounting['original_english_cues']:
            self.counts['original_english_source'] += 1
        elif ready and mode == 'no-dialogue':
            self.counts['no_dialogue'] += 1
        analysis = self.analysis.device_report() if self.analysis is not None else {}
        line = {'job_id': claim.job_id, 'index': claim.index, 'token': claim.token, 'outcome': outcome,
                'english_generated': english_generated,     # positive published PCM evidence, never a mode label
                'english_generated_seconds': english_seconds, 'generation_accounting': accounting,
                'served_audio_verified': False, 'played_audio_verified': False,
                'fault': (detail or {}).get('fault') if isinstance(detail, dict) else None,
                'detail': detail, 'chunk_state': state, 'mode': mode,
                'pipeline_state': payload.get('pipeline_state'), 'audio_sha256': payload.get('sha256'),
                'times': record.times,
                'separation': None if record.separated is None else {
                    'ok': record.separated.ok, 'seconds': record.separated.seconds,
                    'producer': record.separated.producer,          # observed: the producing worker's own record
                    'requested': record.separated.requested,        # the parent's request settings (not evidence)
                    'reused': record.separated.reused, 'bundle_sha256': record.separated.bundle_sha256},
                'analysis': None if record.prepared is None else {
                    'state': record.prepared.state, 'seconds': record.prepared.seconds,
                    'plan_sha256': record.prepared.plan_sha256, 'plan_reused': record.prepared.plan_reused,
                    'capture': record.prepared.capture.as_record() if record.prepared.capture is not None else None},
                'actor_commit': record.actor_commit,
                # Observed evidence only: the Demucs bundle's own producer record (None when it carries none) and the
                # analysis worker's latest reply-attached device_report. Requested settings never appear here.
                'devices': {'demucs': None if record.separated is None else {
                                'observed_device': (record.separated.producer or {}).get('device'),
                                'producer_pid': (record.separated.producer or {}).get('pid'),
                                'producer_version': (record.separated.producer or {}).get('version'),
                                'bundle_reused': record.separated.reused},
                            'analysis': {k: analysis.get(k) for k in ('state', 'pid', 'vram_mib', 'models',
                                                                       'models_reported_at')},
                            'vox': 'see the actor receipt (reported by the Vox host)'}}
        self.counts['settled'][state] = self.counts['settled'].get(state, 0) + 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(line, default=str) + '\n')
        self.emit('chunk_done', job_id=claim.job_id, index=claim.index, token=claim.token, outcome=outcome,
                  state=state, mode=payload.get('mode'))

    def _cancel_superseded(self):
        """Coordinator.cancel's active-owner rule for this process: a cancelled/errored or revised job, or a superseded
        claim, cancels that window's owner at once; its stages stop at their next check and it settles 'cancelled'."""
        for record in list(self.records.values()):
            if record.pending_settle is not None or record.state == 'rendering':
                continue
            claim = record.claim
            if record.requeue and record.outstanding == 0:   # gave way while no worker of it was running
                self._settle(record, 'cancelled', 'background window gave way', record.state)
                continue
            job = self.store.job(claim.job_id)
            chunk = self.store.chunk(claim.job_id, claim.index)
            stale = (not job or job['state'] in ('cancelled', 'error')
                     or int((job.get('payload') or {}).get('revision', 1)) != claim.revision
                     or not chunk or chunk['state'] != 'working'
                     or int((chunk.get('payload') or {}).get('claim_token') or 0) != claim.token)
            if stale:
                self.emit('owner_cancelled', job_id=claim.job_id, index=claim.index, state=record.state)
                record.owner.cancel()
                self._settle(record, 'cancelled', 'job cancelled/revised or claim superseded', record.state)
