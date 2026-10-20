"""Producer horizon policy (pure) and the run-ahead playback policy. relay/store.py claim orders by c.priority,c.updated
(smaller = sooner) and request_chunk keeps min(priority) on conflict, so priorities can only
be lowered by re-request. buffer_segments (relay/tv.py) is the playback prefetch, NOT this horizon.

Run-ahead without limit (owner, 2026-10-19: "I don't want any buffering to happen during play"):
  * production : produce() requests chunks from the viewer position to EOF; producer_queue only bounds how many are
                 IN FLIGHT (concurrency), never how far ahead. relay/tv.py calls it from a scheduler tick that no mux
                 can block when tv.mux_ahead_unlimited is on.
  * mux        : plan_mux() lists the segments to encode now (playback segment first, then ascending, no distance cap).
  * start gate : gate_decision() / rho_estimate() / computed_need(): the movie does not start until it is safe not to
                 stall (a slow machine waits longer instead of stalling mid-film).
Everything below the "run-ahead" banner is pure stdlib so tests/playback can drive it with fake stores and clocks."""
import math
import statistics

DEFAULTS = {"producer_queue": 4, "seek_lead_seconds": 60.0, "startup_seconds": 30.0}


# relay/store.py + coordinator.py actual chunk states.
READY = frozenset({"ready", "approved"})
INFLIGHT = frozenset({"pending", "queued", "working", "waiting_resources"})

def _cfg(cfg, key):
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return DEFAULTS[key]


def _n(session):
    return int(math.ceil(session["duration"] / session["segment_seconds"]))


def contiguous_ready_through(session, chunk_states, position):
    seg = session["segment_seconds"]
    i = int(position // seg)
    while chunk_states.get(i) in READY:
        i += 1
    return min(session["duration"], i * seg)


def plan_requests(session, chunk_states, now, cfg=None):
    """Ordered [(index, priority)]: playback chunk upward first, continues toward EOF.
    Paused viewer does not stop production. Bounded by producer_queue in-flight."""
    q = int(_cfg(cfg, "producer_queue"))
    seg = session["segment_seconds"]
    start = int(session["position"] // seg)
    inflight = sum(1 for v in chunk_states.values() if v in INFLIGHT)
    out = []
    for i in range(start, _n(session)):
        if len(out) + inflight >= q:
            break
        if chunk_states.get(i) in READY | INFLIGHT:
            continue
        out.append((i, i - start))
    return out


def seek(session, chunk_states, target, epoch, cfg=None, explicit_play=False):
    """Never clears cached audio; epoch only invalidates obsolete pending playback requests."""
    seg = session["segment_seconds"]
    idx = int(target // seg)
    lead = float(_cfg(cfg, "seek_lead_seconds"))
    need = min(session["duration"], target + lead)
    through = contiguous_ready_through(session, chunk_states, target)
    ready = chunk_states.get(idx) in READY
    resume = through >= need - 1e-9 and explicit_play
    return {"epoch": epoch + 1, "target": target, "target_index": idx,
            "play_immediately": ready and explicit_play,
            "reprioritize": [] if ready else [(idx, 0)],
            "required_ready_through": need, "contiguous_ready_through": through,
            "resume_allowed": ready and explicit_play or resume,
            "clear_cache": False}


def startup_ready(session, chunk_states, cfg=None):
    need = min(session["duration"], session["position"] + float(_cfg(cfg, "startup_seconds")))
    return contiguous_ready_through(session, chunk_states, session["position"]) >= need - 1e-9


def tracking(session, chunk_states, completions, position):
    """completions: [(wall_start, wall_end, processed_seconds)] injected timestamps."""
    seg = session["segment_seconds"]
    through = contiguous_ready_through(session, chunk_states, position)
    far = sum(1 for i, v in chunk_states.items() if v in READY and i * seg >= through)
    processed = sum(c[2] for c in completions)
    wall = (max(c[1] for c in completions) - min(c[0] for c in completions)) if completions else 0
    tput = processed / wall if wall > 0 else None
    remaining = session["duration"] - sum(min(seg, session["duration"] - i * seg)
                                          for i, v in chunk_states.items() if v in READY)
    return {"contiguous_ready_through": through, "far_ahead_complete": far,
            "processed_seconds": processed, "throughput": tput,
            "eta_eof_seconds": (remaining / tput) if tput else None,
            "realtime": bool(tput is not None and tput >= 1.0)}


# ---- runtime wiring used by relay/tv.py (producer_edits.py), actor_beta only ------------------------
# Priorities: smaller = sooner (store.claim ORDER BY priority,updated; request_chunk keeps the minimum).
# Existing playback requests use -int(now) (coordinator adds lookahead at +10). The seek target goes
# above them; far-ahead production goes below them.
SEEK_BOOST = 1000
FAR_OFFSET = 1000


def priority_anchor(session, now):
    """One timestamp per playback session and seek epoch. Chunk priorities were `-now + ...` at every request, so a request made
    a moment later outranked an earlier one for an EARLIER chunk and windows rendered out of order (55 of 260 pairs in the
    2026-10-20 full-film run), which leaves original speech and clicks at the joins. With the anchor, a newer session or
    seek still outranks everything older, but within one session chunks are served strictly by index."""
    if not session.get('priority_anchor') or session.get('priority_anchor_epoch') != session.get('epoch'):
        session['priority_anchor'] = int(now)
        session['priority_anchor_epoch'] = session.get('epoch')
    return session['priority_anchor']
PRODUCE_EVERY = 2.0
MAX_EXTRA_MUX_SEGMENTS = 4        # legacy mux window cap; used only while tv.mux_ahead_unlimited is OFF


def _tv(cfg, key, default):
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return default


def chunk_states(store, job_id):
    rows = store.chunks(job_id) if job_id else []
    return {int(r["idx"]): r["state"] for r in rows}, rows


def _direct(session):
    return not session.get("dub") and not session.get("compatibility")


def playable_through(session):
    """Contiguous MUXED segments of the current generation from the position (session['segments'])."""
    seg, duration = session["segment_seconds"], session["duration"]
    if _direct(session):
        return duration
    i = max(int(session.get("start_index", 0)), int(session["position"] // seg))
    if i not in session.get("segments", {}):
        return session["position"]
    while i in session["segments"]:
        i += 1
    return min(duration, i * seg)


def on_seek(session, store, jobs, cfg, now):
    """After the existing seek request. Keeps cached audio; latches whether the target audio was ready AT seek."""
    seg = session["segment_seconds"]
    index = int(session["position"] // seg)
    if session.get("speech_engine") in ("generic", "xtranslator") and session.get("job_id"):
        jobs.set_generic_start(session["job_id"], index)
    states, _ = chunk_states(store, session.get("job_id"))
    if session["position"] <= 1e-6:
        # initial dub / start of film: short configured startup threshold, never the 60 s seek lead
        session["seek_latch"] = None
    else:
        session["seek_latch"] = {"index": index, "position": session["position"], "epoch": session.get("epoch", 0),
                                 "target_ready_at_seek": states.get(index) in READY, "at": now}
    if session.get("job_id"):
        try:
            jobs.request(session["job_id"], index, -int(now) - SEEK_BOOST)
        except ValueError:
            pass


# ============================== run-ahead: configuration (tv.* keys) ======================================
# tv.mux_ahead_unlimited      bool  True   mux video+dub audio toward EOF independent of position/pause (dubbed jobs)
# tv.mux_ahead_workers        int   2      concurrent encodes (1..4); +1 extra slot is reserved for the playback segment
# tv.mux_ahead_disk_margin_gib float 5     ahead-only stop at tv.min_free_disk_gib + margin; at the floor nothing renders
# tv.adaptive_start_gate      bool  True   throughput-aware start gate (False: the legacy fixed startup lead)
# tv.actor_beta_start_seconds float 60     startup lead (legacy default 8 applies only while adaptive_start_gate is off)
# tv.start_gate_fast_rho      float 1.25   rho at or above which only the startup lead is needed
# tv.low_water_seconds        float 10     after start: not-startable only below this lead AND rho < 1
# tv.rho_window_chunks        int   6      last N speech-bearing chunks / muxes the rate estimate uses
# tv.actor_beta_producer_queue int  4      chunks IN FLIGHT at once (concurrency, not distance); 1..64
GIB = 1024 ** 3
MUX_WORKERS_BOUNDS = (1, 4)
MUX_ATTEMPTS = 3                  # a far-ahead segment is retried this many times before it is parked
MUX_RETRY_SECONDS = 10.0
URGENT_SPAN = 2                   # segments (playback + next) that may still encode at the soft disk limit
FAST_RHO = 1.25
LOW_WATER_SECONDS = 10.0
RHO_WINDOW = 6
RHO_MIN_SAMPLES = 2
RHO_RATE_CAP = 50.0               # per-chunk rate cap for the median only (a burst of two finishes is not 600x)
STARTUP_SECONDS = 60.0
LEGACY_STARTUP_SECONDS = 8.0


def _number(value, default, low=None, high=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isinstance(value, bool) or not math.isfinite(number):
        return default
    if low is not None:
        number = max(low, number)
    if high is not None:
        number = min(high, number)
    return number


def _flag(cfg, key, default):
    value = _tv(cfg, key, default)
    return value if isinstance(value, bool) else default      # only a real boolean switches a policy


def gate_enabled(cfg):
    return _flag(cfg, "adaptive_start_gate", True)


def gate_active(session, cfg):
    """The throughput-aware gate governs dubbed jobs only; a transcode-only (no dub) session keeps the legacy rule."""
    return bool(gate_enabled(cfg) and session.get("job_id"))


def ahead_enabled(session, cfg):
    """tv.mux_ahead_unlimited (default ON) applies to dubbed jobs only; OFF keeps the legacy buffer_segments window."""
    return bool(session.get("dub") and session.get("job_id") and _flag(cfg, "mux_ahead_unlimited", True))


def mux_workers(cfg):
    low, high = MUX_WORKERS_BOUNDS
    return int(_number(_tv(cfg, "mux_ahead_workers", 2), 2, low, high))


def producer_queue(session, cfg):
    if session.get("speech_engine") in ("generic", "xtranslator"):
        return int(_number(session.get("generic_producer_queue", DEFAULTS["producer_queue"]), DEFAULTS["producer_queue"], 1, 64))
    return int(_number(_tv(cfg, "actor_beta_producer_queue", DEFAULTS["producer_queue"]), DEFAULTS["producer_queue"], 1, 64))


def start_seconds(cfg, session=None):
    on = gate_enabled(cfg) if session is None else gate_active(session, cfg)
    default = STARTUP_SECONDS if on else LEGACY_STARTUP_SECONDS
    return _number(_tv(cfg, "actor_beta_start_seconds", default), default)


def required_seconds(session, cfg):
    remaining = max(0.0, session["duration"] - session["position"])
    if session.get("speech_engine") in ("generic", "xtranslator") and session.get("dub"):
        seconds = float(session.get("generic_prebuffer_seconds", 120.0))
        if not math.isfinite(seconds) or seconds <= 0:
            seconds = 120.0
        return min(seconds, remaining), session["speech_engine"] + "_prebuffer"
    latch = session.get("seek_latch")
    if latch and latch.get("epoch") == session.get("epoch", 0):
        if latch["target_ready_at_seek"]:  # immediate once the target segment itself is muxed
            return min(1.0, remaining), "seek_ready_target"
        lead = float(_tv(cfg, "actor_beta_seek_lead_seconds", DEFAULTS["seek_lead_seconds"]))
        return min(lead, remaining), "seek_unready_target"
    start = start_seconds(cfg, session)
    explicit = _tv(cfg, "actor_beta_max_prefill_segments", None)
    if gate_active(session, cfg):
        # The startup lead is the owner's ~60 s; the old prefill cap applies only when it is configured explicitly.
        cap = start if explicit is None else min(start, int(explicit) * session["segment_seconds"])
        return min(max(1.0, cap), remaining), "startup"
    prefill = int(_tv(cfg, "actor_beta_max_prefill_segments", 3))
    return min(max(1.0, min(start, prefill * session["segment_seconds"])), remaining), "startup"


def mux_limit(session, limit, cfg):
    """Extend the existing mux window to ceil((position+required)/seg), bounded.
    tv.mux_ahead_unlimited ON (dubbed job): no distance cap, the window is the whole film."""
    seg = session["segment_seconds"]
    total = int(math.ceil(session["duration"] / seg))
    if ahead_enabled(session, cfg):
        return total
    required, _ = required_seconds(session, cfg)
    current = max(int(session.get("start_index", 0)), int(session["position"] // seg))
    need = int(math.ceil((session["position"] + required) / seg - 1e-9))
    return min(total, max(limit, min(need, current + MAX_EXTRA_MUX_SEGMENTS)))


def produce(session, store, jobs, cfg, now, lookahead=0):
    """Bounded production toward EOF, independent of viewer pause. Each coordinator.request also queues
    `lookahead` following chunks; in-flight is re-read from the store after every request so the real
    pending count (including lookahead) never exceeds producer_queue. The bound is concurrency, never distance."""
    if not session.get("job_id") or now - session.get("producer_at", 0) < PRODUCE_EVERY:
        return []
    session["producer_at"] = now
    queue = producer_queue(session, cfg)
    seg = session["segment_seconds"]
    total = int(math.ceil(session["duration"] / seg))
    start = int(session["position"] // seg)
    requested = []
    states, _ = chunk_states(store, session["job_id"])
    for index in range(start, total):
        if states.get(index) in READY | INFLIGHT:
            continue
        inflight = sum(1 for v in states.values() if v in INFLIGHT)
        added = sum(1 for i in range(index, min(total, index + 1 + int(lookahead))) if i not in states)
        if inflight + added > queue and inflight > 0:   # an idle worker is always fed: a queue below lookahead+1 cannot deadlock
            break
        try:
            jobs.request(session["job_id"], index, -priority_anchor(session, now) + FAR_OFFSET + index)
        except ValueError:
            break
        requested.append(index)
        states, _ = chunk_states(store, session["job_id"])   # the request (and its lookahead) changed the store
    session["producer_requested"] = requested
    return requested


def _throughput(rows, seg, duration):
    """Measured only: processed seconds between the first and last ready completion timestamps."""
    done = sorted((float(r.get("updated") or 0), min(seg, duration - int(r["idx"]) * seg))
                  for r in rows if r["state"] in READY and r.get("updated"))
    if len(done) < 2 or done[-1][0] - done[0][0] < 1.0:  # <1 s span is not a measurement
        return None
    return sum(s for _, s in done[1:]) / (done[-1][0] - done[0][0])


# ============================== run-ahead: throughput measurement (pure) ====================================

def _take_evidence(payload):
    """True/False when the payload lists its generated takes, None when it carries no take list."""
    if "generic_takes" in payload:
        return bool(payload.get("generic_takes"))
    published = payload.get("published_utterances")
    if published is not None:
        return any(isinstance(t, dict) and t.get("kind") == "generated" for t in published)
    return None


def chunk_kind(row):
    """'speech' = a ready chunk with at least one generated dubbed take (the only kind that feeds rho),
    'free' = speech-free (no dialogue / music bed only / source voice kept): counts as ready lead, never toward rho,
    'other' = anything else (not ready, needs review, error, a dub label without a generated take)."""
    try:
        from .buffer_policy import classify_window_detail
    except ImportError:  # standalone import
        from buffer_policy import classify_window_detail
    kind, _ = classify_window_detail(row)
    if kind in ("no_dialogue", "bed_only", "original_english", "original_unverified"):
        return "free" if row and row.get("state") in READY else "other"
    if kind != "dub":
        return "other"
    takes = _take_evidence(row.get("payload") or {})
    return "other" if takes is False else "speech"


def production_samples(rows, seg, duration, kind_of=chunk_kind, baseline=0.0):
    """[(index, media_seconds, wall_seconds)] for SPEECH-BEARING ready chunks only, in completion order.
    A chunk's wall time is its completion minus the previous completion (of ANY ready chunk of this job after
    `baseline`, the moment the session attached to the job) because the store keeps only the completion stamp.
    The first completion has no predecessor, so it is never a sample. An idle gap makes a sample slower: pessimistic."""
    done = sorted((float(r.get("updated") or 0.0), int(r["idx"]), r) for r in rows
                  if r["state"] in READY and r.get("updated") and float(r["updated"]) >= baseline)
    samples = []
    for n in range(1, len(done)):
        end, index, row = done[n]
        if kind_of(row) != "speech":
            continue
        samples.append((index, min(float(seg), float(duration) - index * seg), max(1e-3, end - done[n - 1][0])))
    return samples


def mux_samples(log):
    """[(index, media_seconds, wall_seconds)] from mux records (start, end, seconds, index), non-cached encodes only.
    Overlapping concurrent encodes share the busy time (wall = end - max(start, previous end)), so the aggregate is the
    machine's real mux capacity."""
    out, previous_end = [], None
    for start, end, seconds, index in sorted(log, key=lambda r: r[1]):
        begin = start if previous_end is None else max(start, previous_end)
        out.append((int(index), float(seconds), max(1e-3, end - begin)))
        previous_end = end if previous_end is None else max(previous_end, end)
    return out


def rho_estimate(samples, window=RHO_WINDOW, min_samples=RHO_MIN_SAMPLES):
    """Pessimistic media-seconds-per-wall-second from the last `window` samples: the slower of the median per-chunk
    rate and the time-weighted mean (sum seconds / sum wall). `used` lists the chunk indexes behind the number.
    rho is None (unmeasured) below `min_samples` samples or under 1 s of total wall time."""
    use = list(samples)[-max(1, int(window)):]
    out = {"rho": None, "median": None, "mean": None, "n": len(use), "needed": int(min_samples),
           "used": [s[0] for s in use], "rates": [round(s[1] / s[2], 4) for s in use]}
    if len(use) < min_samples:
        return out
    wall = sum(s[2] for s in use)
    if wall < 1.0:
        return out
    median = statistics.median(min(RHO_RATE_CAP, s[1] / s[2]) for s in use)
    mean = sum(s[1] for s in use) / wall
    out.update(rho=min(median, mean), median=median, mean=mean)
    return out


def combine_rho(production, mux, *, audio_complete, remaining, dialogue_seconds=None, fast_rho=FAST_RHO):
    """The end-to-end rate: the slowest measured stage. Once all remaining audio exists production no longer limits.
    `dialogue_seconds` (remaining dialogue-bearing time, only when known) credits speech-free stretches: production
    must clone dialogue_seconds of the `remaining`, so its average rate over the rest of the film is higher.
    A measured-fast stage proves nothing about an unmeasured one (state 'unproven', rho None)."""
    stages = {} if audio_complete else {"production": production}
    stages["mux"] = mux
    values = {}
    for name, estimate in stages.items():
        value = estimate.get("rho")
        if value is not None and name == "production" and dialogue_seconds is not None \
                and 0 < dialogue_seconds < remaining:
            value = value * remaining / dialogue_seconds
        values[name] = value
    measured = {k: v for k, v in values.items() if v is not None}
    missing = sorted(k for k, v in values.items() if v is None)
    if not measured:
        return {"rho": None, "state": "unmeasured", "limiting": None, "missing": missing, "stages": values}
    limiting = min(measured, key=measured.get)
    rho = measured[limiting]
    if missing and rho >= fast_rho:
        return {"rho": None, "state": "unproven", "limiting": None, "missing": missing, "stages": values}
    return {"rho": rho, "state": "partial" if missing else "measured", "limiting": limiting, "missing": missing,
            "stages": values}


# ============================== run-ahead: start gate (pure) ================================================

def computed_need(remaining, rho, fast_rho=FAST_RHO):
    """Lead (s) that lets playback reach EOF without a stall when production+mux run at rho/fast_rho of the measured
    rate (a 20% safety margin at 1.25): remaining * (1 - rho/fast_rho), 0 when rho >= fast_rho, capped at remaining."""
    if rho is None or remaining <= 0 or rho >= fast_rho:
        return 0.0
    return min(float(remaining), max(0.0, remaining * (1.0 - max(0.0, rho) / fast_rho)))


def gate_decision(*, remaining, lead, base_required, base_rule, rho, started=False, rebuffering=False,
                  fast_rho=FAST_RHO, low_water=LOW_WATER_SECONDS, lead_ceiling=None):
    """Start/rebuffer decision for the status contract (startable + the lead the player compares against).
    Before start:   required = max(base_required, computed_need), startable = lead >= required (or EOF).
    After start:    startable stays True with required = low_water, unless lead < low_water AND rho < 1 (an honest
                    rebuffer): then startable False and required = the full requirement until it is met again.
    rho None (unmeasured): the base (startup / seek) lead only, plus the same low-water rule (which needs a rho).
    lead_ceiling: the most lead that can ever exist (a segment that failed for good is at that distance): the
    requirement never exceeds it, so a viewer is not held forever for a lead that cannot be built."""
    need = computed_need(remaining, rho, fast_rho)
    full = min(float(remaining), max(float(base_required), need))
    if lead_ceiling is not None and lead_ceiling > 0:
        full = min(full, float(lead_ceiling))
    rule = base_rule + ("+throughput" if need > base_required + 1e-9 else "")
    eof = remaining <= 1e-9 or lead >= remaining - 1e-9
    if not started:
        startable = remaining <= 1e-9 or (lead > 0 and lead >= full - 1e-9)
        return {"startable": bool(startable), "required": full, "full_required": full, "rule": rule,
                "rebuffering": False, "computed_need": need}
    relaxed = min(float(low_water), float(remaining))
    if rebuffering:
        if eof or lead >= full - 1e-9:
            return {"startable": True, "required": relaxed, "full_required": full, "rule": "started_low_water",
                    "rebuffering": False, "computed_need": need}
        return {"startable": False, "required": full, "full_required": full, "rule": "rebuffer",
                "rebuffering": True, "computed_need": need}
    if not eof and lead < low_water - 1e-9 and rho is not None and rho < 1.0:
        return {"startable": False, "required": full, "full_required": full, "rule": "rebuffer",
                "rebuffering": True, "computed_need": need}
    return {"startable": True, "required": relaxed, "full_required": full, "rule": "started_low_water",
            "rebuffering": False, "computed_need": need}


def expected_stall_free(*, remaining, lead, rho, fast_rho=FAST_RHO, low_water=LOW_WATER_SECONDS):
    """True / False, or None when it cannot be known yet (rho unmeasured). True means the measured rates plus the 20%
    margin carry playback to EOF from this lead (and the lead is not below the low-water mark)."""
    if remaining <= 1e-9 or lead >= remaining - 1e-9:
        return True
    if rho is None:
        return None
    return bool(lead >= max(computed_need(remaining, rho, fast_rho), min(float(low_water), float(remaining))) - 1e-9)


def projected_stall_seconds(*, lead, remaining, rho, paused=False):
    """Wall seconds until the lead runs out if playback proceeds now: lead / (1 - rho) when rho < 1 and that moment
    precedes EOF; None otherwise (rho >= 1, unmeasured, paused: production keeps filling the lead, or the film ends)."""
    if paused or rho is None or rho >= 1.0 or remaining <= 1e-9 or lead >= remaining - 1e-9:
        return None
    seconds = lead / (1.0 - max(0.0, rho))
    return float(seconds) if seconds < remaining else None


# ============================== run-ahead: mux scheduling, disk floor, encoder (pure) =======================

def plan_mux(*, current, total, muxed, inflight, ready, workers, active, urgent_span=URGENT_SPAN, ahead_allowed=True):
    """Segment indexes to start encoding now. The first segment not yet muxed from the playhead goes first (when it is
    within `urgent_span` segments of the playhead it may use one extra slot beyond `workers`, so far-ahead encodes can
    never delay the segment playback is about to need), then ascending toward EOF with NO distance cap. Only segments
    whose audio is ready are listed. `active` counts every encode already running (any session/generation still
    unwinding). ahead_allowed False (soft disk limit) restricts to the playhead's own segment and the next one."""
    first = next((i for i in range(current, total) if i not in muxed), None)
    if first is None:
        return []
    free = max(0, int(workers) - int(active))
    horizon = total if ahead_allowed else min(total, current + max(1, int(urgent_span)))
    urgent = first - current < int(urgent_span)
    picks = []
    if first < horizon and first in ready and first not in inflight and (free > 0 or (urgent and active < workers + 1)):
        picks.append(first)
        free = max(0, free - 1)
    for index in range(first + 1, horizon):
        if free <= 0:
            break
        if index in muxed or index in inflight or index not in ready:
            continue
        picks.append(index)
        free -= 1
    return picks


def disk_gate(free_bytes, floor_gib, margin_gib):
    """'ok' | 'ahead_hold' (below floor + margin: only the playback segment and the next may encode) |
    'floor' (below tv.min_free_disk_gib: nothing renders; resource_wait is reported; resumes by itself when freed)."""
    floor = float(floor_gib) * GIB
    if free_bytes < floor:
        return "floor"
    if free_bytes < floor + float(margin_gib) * GIB:
        return "ahead_hold"
    return "ok"


def choose_encoder(requested, *, nvenc_listed, free_vram_mib, min_free_mib, hdr_preserve=False):
    """('nvenc'|'software', reason). 'auto' prefers NVENC when ffmpeg lists h264_nvenc and the free-VRAM check passes;
    an explicit 'nvenc' needs the same two conditions (a GPU out-of-memory would hurt the dub, so the check is never
    skipped); 'software' (or anything else) is software. HDR-preserving output is libx265 software."""
    if requested not in ("auto", "nvenc"):
        return "software", "software requested" if requested == "software" else "unknown encoder %r: software" % (requested,)
    if hdr_preserve:
        return "software", "HDR-preserving output uses libx265 (software)"
    if not nvenc_listed:
        return "software", "h264_nvenc is not listed by this ffmpeg"
    if free_vram_mib is None:
        return "software", "free VRAM could not be read"
    if free_vram_mib < min_free_mib:
        return "software", "free VRAM %d MiB is below nvenc_min_free_mib %d" % (free_vram_mib, min_free_mib)
    return "nvenc", "h264_nvenc listed and %d MiB VRAM free (needs %d)" % (free_vram_mib, min_free_mib)


def status_fields(session, store, cfg, now):
    """Exact BACKEND-CONTRACT.json keys (unchanged) plus the run-ahead keys (additive). Playable lead = muxed
    segments of this generation, not store audio."""
    seg, duration, position = session["segment_seconds"], session["duration"], session["position"]
    states, rows = chunk_states(store, session.get("job_id"))
    remaining = max(0.0, duration - position)
    lead = max(0.0, playable_through(session) - position)
    base_required, base_rule = required_seconds(session, cfg)
    audio_through = contiguous_ready_through(session, states, position) if session.get("job_id") else duration
    total = int(math.ceil(duration / seg))
    tput = _throughput(rows, seg, duration)
    ready_seconds = sum(min(seg, duration - i * seg) for i, v in states.items() if v in READY)
    paused = bool(session.get("paused_intent", session.get("state") == "paused"))
    direct = _direct(session)
    latch = session.get("seek_latch")
    if gate_active(session, cfg):
        gate = _gate(session, rows, cfg, now, lead=lead, remaining=remaining, base_required=base_required,
                     base_rule=base_rule, audio_through=audio_through, paused=paused)
        startable = direct or gate["decision"]["startable"]
        required, rule = gate["decision"]["required"], gate["decision"]["rule"]
        if gate["started"] and latch and base_rule.startswith("seek"):
            session["seek_latch"] = None  # fulfilled by real playback; later rebuffers use the throughput rule
    else:
        gate = None
        required, rule = base_required, base_rule
        startable = direct or remaining == 0 or (lead > 0 and lead >= required - 1e-9)
        if startable and latch and rule.startswith("seek"):
            session["seek_latch"] = None  # fulfilled; later rebuffers use the startup rule
    fields = {"startable": bool(startable),
              "buffer_required_seconds": float(required),
              "contiguous_lead_seconds": float(lead),
              "paused_intent": paused,
              "seek_epoch": int(session.get("epoch", 0)),
              "producer_progress": {
                  "ready_chunks": sum(1 for v in states.values() if v in READY), "total_chunks": total,
                  "contiguous_ready_through_seconds": float(audio_through),
                  "audio_lead_seconds": float(max(0.0, audio_through - position)),
                  "far_ahead_ready_chunks": sum(1 for i, v in states.items() if v in READY and i * seg >= audio_through),
                  "inflight_chunks": sum(1 for v in states.values() if v in INFLIGHT),
                  "processed_seconds": float(ready_seconds),
                  "throughput_x": tput,
                  "eta_seconds": (max(0.0, duration - ready_seconds) / tput) if tput else None,
                  "realtime": bool(tput is not None and tput >= 1.0),
                  "startable_rule": rule}}
    if gate is not None:
        fields.update(gate["fields"])
    return fields


def _gate(session, rows, cfg, now, *, lead, remaining, base_required, base_rule, audio_through, paused):
    """Throughput-aware gate: rho from speech-bearing production samples and from non-cached mux timings."""
    seg, duration, position = session["segment_seconds"], session["duration"], session["position"]
    fast = _number(_tv(cfg, "start_gate_fast_rho", FAST_RHO), FAST_RHO, 1.0, 10.0)
    low_water = _number(_tv(cfg, "low_water_seconds", LOW_WATER_SECONDS), LOW_WATER_SECONDS, 0.0, 600.0)
    window = int(_number(_tv(cfg, "rho_window_chunks", RHO_WINDOW), RHO_WINDOW, 2, 64))
    cache = session.setdefault("_kind_cache", {})

    def kind_of(row):
        key = (int(row["idx"]), row.get("updated"))
        if key not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[key] = chunk_kind(row)
        return cache[key]

    production = rho_estimate(production_samples(rows, seg, duration, kind_of,
                                                 float(session.get("job_attached_at") or 0.0)), window)
    mux = rho_estimate(mux_samples(session.get("_mux_log") or ()), window)
    audio_complete = (not session.get("job_id")) or audio_through >= duration - 1e-9   # no job: no production stage
    dialogue = session.get("dialogue_remaining_hint")
    dialogue = None if isinstance(dialogue, bool) or not isinstance(dialogue, (int, float)) else float(dialogue)
    combined = combine_rho(production, mux, audio_complete=audio_complete, remaining=remaining,
                           dialogue_seconds=dialogue, fast_rho=fast)
    rho = combined["rho"]
    epoch = int(session.get("epoch", 0))
    started = bool(session.get("beta_started")) and session.get("beta_epoch") == epoch
    latched = session.get("_gate") or {}
    rebuffering = bool(latched.get("rebuffering")) and latched.get("epoch") == epoch and started
    parked_at = session.get("_mux_parked_at")
    ceiling = None if isinstance(parked_at, bool) or not isinstance(parked_at, (int, float)) else parked_at - position
    decision = gate_decision(remaining=remaining, lead=lead, base_required=base_required, base_rule=base_rule, rho=rho,
                             started=started, rebuffering=rebuffering, fast_rho=fast, low_water=low_water,
                             lead_ceiling=ceiling)
    session["_gate"] = {"epoch": epoch, "rebuffering": decision["rebuffering"]}
    session["_gate_required_lead"] = float(decision["full_required"])    # tv._lead_limit: the legacy window must reach it
    stall_free = expected_stall_free(remaining=remaining, lead=lead, rho=rho, fast_rho=fast, low_water=low_water)
    projected = projected_stall_seconds(lead=lead, remaining=remaining, rho=rho, paused=paused)
    kinds = {}
    for row in rows:
        if row["state"] in READY:
            kinds[kind_of(row)] = kinds.get(kind_of(row), 0) + 1
    fields = {
        "expected_stall_free": stall_free,
        "stall_free_state": "unknown" if stall_free is None else "yes" if stall_free else "no",
        "required_lead_seconds": float(decision["full_required"]),
        "lead_seconds": float(lead),
        "rho": rho,
        "rho_state": combined["state"],
        "projected_stall_seconds": projected,
        "playback_gate": {
            "version": "throughput-gate-v1", "rule": decision["rule"], "started": started,
            "rebuffering": decision["rebuffering"], "computed_need_seconds": decision["computed_need"],
            "base_required_seconds": float(base_required), "startup_seconds": start_seconds(cfg, session),
            "low_water_seconds": low_water, "fast_rho": fast, "rho_limiting_stage": combined["limiting"],
            "rho_missing_stages": combined["missing"], "audio_complete": audio_complete,
            "remaining_basis": "dialogue_hint" if dialogue is not None else "remaining_duration (conservative)",
            "start_eta_seconds": _eta(decision, lead, rho),
            "rho_production": {k: production[k] for k in ("rho", "median", "mean", "n", "needed", "used", "rates")},
            "rho_mux": {k: mux[k] for k in ("rho", "median", "mean", "n", "needed", "used", "rates")},
            "ready_chunk_kinds": kinds,
            "scope": "rho counts only speech-bearing chunks (>=1 generated take) and non-cached encodes; "
                     "speech-free chunks add ready lead but never rate"}}
    return {"decision": decision, "fields": fields, "started": started, "rho": rho}


def _eta(decision, lead, rho):
    """Wall seconds until the lead requirement is met at the measured rate; None when already met or unmeasured."""
    gap = decision["full_required"] - lead
    if decision["startable"] or gap <= 0 or rho is None or rho <= 0:
        return None
    return float(gap / rho)


