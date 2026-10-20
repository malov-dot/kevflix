"""Actor-beta TV buffer policy: pure stdlib helpers for relay/tv.py (opt-in overlay).

Operates on the real TVService session dict (relay/tv.py open_movie spec: position,
duration, segment_seconds, start_index, segments {index: path}, epoch, dub,
compatibility, job_id, state) and on real store chunk records (relay/store.py
chunks table: job_id, idx, state, priority, payload, updated).

Chunk states written by the program: pending, working, ready, approved,
needs_review, waiting_resources, error, cancelled. Payload "mode" values written
by relay/worker.py: "English-dub", "original-English", "no-dialogue" and, for the
actor path, "actor-English-candidate" (or "no-dialogue" from
relay/actor_pipeline.py when there were no cues, turns, transcript segments or
canonical speech).

Priority convention (relay/coordinator.py request -> store.request_chunk, claim
ORDER BY c.priority, c.updated): SMALLER number is served first, conflicts keep
min(priority). The TV loop passes -int(time.time()), so newer requests win.

Keys added to the session dict by this module (all prefixed "beta_" except the
requested "underruns"): underruns, beta_started, beta_in_underrun, beta_epoch.
Seek/cancel/generation stay owned by TVService.control().
"""
from __future__ import annotations

import math

READY_STATES = ("ready", "approved")
WAITING_STATES = ("pending", "working", "waiting_resources")
DEFAULT_START_SECONDS = 8.0          # legacy (tv.adaptive_start_gate off)
GATE_START_SECONDS = 60.0            # tv.adaptive_start_gate on (default)
DEFAULT_MAX_PREFILL_SEGMENTS = 3
START_SECONDS_BOUNDS = (1.0, 60.0)
GATE_START_BOUNDS = (1.0, 600.0)
PREFILL_BOUNDS = (1, 4)
UNDERRUN_EPSILON = 0.5


def _read(mapping, key, default):
    """config['tv'] is a plain dict; relay.config.Config has no .get, so use try/except."""
    try:
        return mapping[key]
    except (KeyError, TypeError, IndexError):
        return default


def _direct(session):
    return not session["dub"] and not session["compatibility"]


def _segment_index(session, position):
    return max(int(session["start_index"]), int(position // session["segment_seconds"]))


def contiguous_ready_end(session, position=None):
    """End time (s) of contiguous rendered segments starting at the playback segment."""
    position = session["position"] if position is None else position
    seg = session["segment_seconds"]
    index = _segment_index(session, position)
    if index * seg > position + 1e-9:
        return position  # playback is before start_index: nothing ready under it
    while index in session["segments"]:
        index += 1
    return min(session["duration"], index * seg)


def _gate_on(config_tv):
    """tv.adaptive_start_gate (default ON): the throughput-aware gate's 60 s startup lead replaces the legacy 8 s one."""
    value = _read(config_tv, "adaptive_start_gate", True)
    return value if isinstance(value, bool) else True


def startup_threshold(config_tv, seeked, segment_seconds=30, remaining=None):
    """Short bounded startup buffer in seconds.

    Initial start: min(actor_beta_start_seconds, actor_beta_max_prefill_segments * seg).
    After a seek/restart (seeked=True): at most one segment, so dialogue right
    after the seek point is never held behind a long prefill. Never assumes a
    long intro. Capped by the remaining duration when given.
    Adaptive start gate ON (default): the startup lead defaults to 60 s (up to 600 s) and the prefill cap applies only
    when actor_beta_max_prefill_segments is configured explicitly; OFF keeps the legacy 8 s / 3 segments exactly.
    """
    adaptive = _gate_on(config_tv)
    default_seconds = GATE_START_SECONDS if adaptive else DEFAULT_START_SECONDS
    seconds = float(_read(config_tv, "actor_beta_start_seconds", default_seconds))
    if not math.isfinite(seconds):
        seconds = default_seconds
    seconds = min(max(seconds, START_SECONDS_BOUNDS[0]), (GATE_START_BOUNDS if adaptive else START_SECONDS_BOUNDS)[1])
    explicit = _read(config_tv, "actor_beta_max_prefill_segments", None)
    try:
        prefill = int(DEFAULT_MAX_PREFILL_SEGMENTS if explicit is None else explicit)
    except (TypeError, ValueError):
        prefill = DEFAULT_MAX_PREFILL_SEGMENTS
    prefill = min(max(prefill, PREFILL_BOUNDS[0]), PREFILL_BOUNDS[1])
    seg = float(segment_seconds)
    if adaptive and explicit is None and not seeked:
        threshold = seconds
    else:
        threshold = min(seconds, (1 if seeked else prefill) * seg)
    if remaining is not None:
        threshold = min(threshold, max(0.0, float(remaining)))
    return threshold


def classify_window_detail(chunk_record):
    """(kind, reason) for one store chunk record (or None).

    kind is one of:
      'dub'                 ready/approved chunk with audio whose mode is a dub
                            ("English-dub" / "actor-English-candidate").
      'original_english'    ready chunk with mode "original-English": the program
                            detected English speech and kept it (not a dub, not unverified).
      'no_dialogue'         ONLY when payload mode == "no-dialogue" and state is ready/approved
                            (confirmed: no cues/turns/transcript/speech).
      'original_unverified' terminal non-dub outcome with an explicit reason
                            (needs_review, error, cancelled, or ready without audio).
                            A missing subtitle alone is NOT proof of no speech.
      'pending'             no record yet, or pending/working/waiting_resources.
    Zero cues in an early window is reported per window; it is never a whole-job failure.
    """
    if not chunk_record:
        return "pending", "not requested yet"
    state = chunk_record.get("state")
    payload = chunk_record.get("payload") or {}
    mode = payload.get("mode")
    if state in WAITING_STATES:
        return "pending", str(state)
    if state in READY_STATES:
        if payload.get("generic_version") == "generic-dub-v1":
            if payload.get("speech_engine", payload.get("engine")) == "xtranslator":
                if not payload.get("audio"):
                    return "pending", "XTranslator receipt has no prepared audio"
                if mode == "original-English":
                    return "original_english", "XTranslator detected English; source voice kept without dubbing"
                if payload.get("generic_takes"):
                    return "dub", "XTranslator PCM included; full publication and playback reported separately"
                if payload.get("generic_failures") or payload.get("generic_vad_regions"):
                    return "bed_only", "XTranslator music/effects bed only; no completed speech take in this chunk"
                if "generic_vad_regions" not in payload:
                    return "pending", "XTranslator speech-detection evidence missing"
                return "no_dialogue", "XTranslator receipt reports no detected speech and no mixed take"
            if not payload.get("audio"):
                return "original_unverified", "generic ready receipt without audio"
            if payload.get("generic_takes"):
                return "dub", "generic PCM included; full-take publication and playback reported separately"
            if payload.get("generic_failures") or payload.get("generic_original_intervals"):
                return "original_unverified", "generic source-voice fallback; zero generated credit"
            if payload.get("generic_vad_regions"):
                return "original_unverified", "generic detected speech without mixed takes"
            if "generic_vad_regions" not in payload:
                return "original_unverified", "generic speech-detection receipt missing"
            return "no_dialogue", "generic receipt reports no VAD speech and no mixed takes"
        if mode == "no-dialogue":
            return "no_dialogue", "receipt mode no-dialogue"
        if mode == "original-English":
            return "original_english", "detected English dialogue kept"
        if mode == "original-unverified":  # actor_beta worker fallback: original source audio, explicit reasons
            return "original_unverified", "beta original fallback: " + ",".join(payload.get("beta_original_reason") or ["unknown"])
        if payload.get("audio"):
            return "dub", str(mode or "dub")
        return "original_unverified", "ready without audio path"
    if state == "needs_review":
        issues = payload.get("issues") or (payload.get("quality") or {}).get("issues") or []
        kinds = sorted({str(i.get("kind")) for i in issues if isinstance(i, dict) and i.get("kind")})
        return "original_unverified", "needs_review" + (": " + ",".join(kinds) if kinds else "")
    if state in ("error", "cancelled"):
        return "original_unverified", str(state) + (": " + str(payload["error"]) if payload.get("error") else "")
    return "pending", "unknown state " + str(state)


def classify_window(chunk_record):
    return classify_window_detail(chunk_record)[0]


def buffer_metrics(session, now, config_tv=None, chunk_record=None):
    """Honest buffer/lead metrics for status().

    buffer_seconds: contiguous rendered segments from the playback position
    (direct original playback: the rest of the file). lead_kind: 'direct' when the
    session is not PC-processed; 'original' when processed without dubbing;
    otherwise classify_window(chunk_record) for the window under the position.
    seek_epoch: the session epoch (bumped by control() on seek/dub/audio/quality).
    """
    position = float(session["position"])
    remaining = max(0.0, session["duration"] - position)
    seeked = int(session.get("epoch", 0)) > 0
    seg = session["segment_seconds"]
    threshold = startup_threshold(config_tv or {}, seeked, seg, remaining)
    if session.get("speech_engine") in ("generic", "xtranslator") and session.get("dub"):
        threshold = float(session.get("generic_prebuffer_seconds", 120.0))
        if not math.isfinite(threshold) or threshold <= 0:
            threshold = 120.0
        threshold = min(threshold, remaining)
    if _direct(session):
        buffered, startable, kind, reason = remaining, True, "direct", "original file"
    else:
        buffered = max(0.0, contiguous_ready_end(session, position) - position)
        index = _segment_index(session, position)
        startable = index in session["segments"] and buffered + 1e-9 >= threshold
        if session.get("job_id"):
            kind, reason = classify_window_detail(chunk_record)
        else:
            kind, reason = "original", "dubbing off"
    return {"buffer_seconds": round(buffered, 3), "start_threshold_seconds": round(threshold, 3),
            "startable": bool(startable), "underruns": int(session.get("underruns", 0)),
            "seek_epoch": int(session.get("epoch", 0)), "lead_kind": kind, "lead_reason": reason}


def next_request(session, chunk_states, now, buffer_segments=2):
    """Deadline-ordered next missing segment, bounded to the existing TV window.

    Window and selection match TVService.loop exactly: current = max(start_index,
    position // seg), limit = min(ceil(duration / seg), current + max(2, buffer_segments)),
    first index not yet rendered (that is the earliest playback deadline). Returns
    None when nothing in the window is missing; never returns an index >= limit.
    chunk_states: {idx: store state} for the window (informational, returned as
    'chunk_state'). priority = -int(now) + seconds_until_needed, which keeps the
    program convention (smaller first, newer requests win) and ranks a nearer
    deadline ahead of a later one requested in the same second.
    Note: coordinator.request still adds its own lookahead_segments at priority+10;
    that is program behaviour this overlay does not change.
    """
    seg = session["segment_seconds"]
    position = float(session["position"])
    current = _segment_index(session, position)
    limit = min(math.ceil(session["duration"] / seg), current + max(2, int(buffer_segments)))
    for index in range(current, limit):
        if index in session["segments"]:
            continue
        lead = max(0.0, index * seg - position)
        return {"index": index, "priority": -int(now) + int(lead), "deadline_seconds": lead,
                "deadline_at": now + lead, "chunk_state": (chunk_states or {}).get(index), "limit": limit}
    return None


def note_heartbeat(session, position, state, now):
    """Count playback underruns from heartbeats. Returns the updated count.

    tv/app/player.js poll() sends 'ended' | 'buffering' (while loading or the
    loaded generation differs) | 'paused' | 'playing'. A mid-play stall fires the
    video 'waiting' event, which only changes the UI; the heartbeat then still says
    'playing'. So an underrun is counted once per stall when playback had started
    in this epoch and buffer_seconds <= 0.5 while the player reports 'buffering'
    or 'playing'. A seek/restart (epoch change) resets 'started', so seek
    buffering is not counted.
    """
    epoch = int(session.get("epoch", 0))
    if session.get("beta_epoch") != epoch:
        session["beta_epoch"], session["beta_started"], session["beta_in_underrun"] = epoch, False, False
    session.setdefault("underruns", 0)
    if _direct(session):
        return session["underruns"]
    buffered = max(0.0, contiguous_ready_end(session, float(position)) - float(position))
    if state == "playing" and buffered > UNDERRUN_EPSILON:
        session["beta_started"], session["beta_in_underrun"] = True, False
    elif (state in ("buffering", "playing") and session["beta_started"] and buffered <= UNDERRUN_EPSILON
          and session["duration"] - float(position) > UNDERRUN_EPSILON):
        if not session["beta_in_underrun"]:
            session["underruns"] += 1
            session["beta_in_underrun"] = True
    return session["underruns"]
