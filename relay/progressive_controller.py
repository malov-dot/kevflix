"""Opt-in chaining of prepared ADJACENT progressive windows into one sink (B6).

UNUSED BY PRODUCTION. ``worker.py`` and ``coordinator.py`` never call this
module; the production playback route serves per-chunk ``english.wav`` files
(see docs/MOVIE-PROGRESSIVE-CONTROLLER-B6.md for the route trace). This is
the smallest reviewable connection from prepared plans to the progressive
renderer and a continuous sink such as ``LiveMux.write``.

Contract:

- Windows are complete prepared plans (``render_start == origin`` and
  ``render_end == end``) that are exactly adjacent, both in seconds (1e-6 s)
  and in absolute frames. Overlapping offline ``actor_pipeline`` windows
  (context-padded, ``render_start > origin``) are refused, never re-cut.
- The next window is pulled and validated BEFORE the current window renders,
  so its lookahead and protection are ready before any output can cross the
  edge. When it is missing or invalid the current window renders without
  lookahead (the renderer then blocks any crossing take explicitly) and the
  stream stops at that edge with an explicit reason.
- Only evidence the plans carry is passed through. Confirmed pauses are never
  inferred: ``pause_evidence`` is optional caller evidence and defaults to
  none. Source identity comes only from the caller's ProtectionLedger and each
  plan's own ``protection_scope``; a plan without one is blocked.
- One controller run is one output stream session. A seek or replay needs a
  new run (new session). The carry chain stops at the first failure; a failed
  window's carry is never handed on.
- B6b binding: each plan is read and parsed once. The renderer must render
  exactly that validated object (``plan_digest``): the controller re-checks
  the file before rendering and the renderer checks the object it actually
  loaded, before any commit or PCM. Source PCM is bound through the bound
  plan's ``pcm_sha256`` (verified at load); the source frame count validated
  during lookahead must still hold. A broken chained carry fails before PCM.
- B6b ownership: a run owns its (session, sink) in the store
  (``output_streams``) before the first sink call and persists each sink
  call's intent before making it. Delivery is at most once per frame offset:
  a session/sink is never restarted or resumed, and a sink call that raised
  (or a run that died inside it) leaves its interval uncertain for good. It
  is not exactly once: the sink returning is the only acknowledgment.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path

from .dubbing_timing import timing_carry_scope
from .progressive_actor import LAYOUT_CHANNELS, next_window_lookahead, plan_digest, render_progressively

CONTROLLER_VERSION = 'progressive-controller-v2'
ADJACENT_TOLERANCE = 1e-6
DELIVERY = ('at most once per (session, sink): each frame offset is handed to the sink at most once and the '
            'session/sink is never restarted or resumed; a sink call that raised, or a run that died inside it, '
            'leaves its frame interval uncertain (none to all of it may have reached the output); '
            'not exactly once: the sink returning is the only acknowledgment')


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def read_plan(path):
    try:
        plan = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return plan if isinstance(plan, dict) else None


def load_plan(path):
    """``(plan, digest)`` read and parsed once; ``(None, None)`` when unreadable."""
    plan = read_plan(path)
    return plan, (None if plan is None else plan_digest(plan))


def _bound_reason(path, digest):
    """Why the plan file no longer holds the validated object, or None."""
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return 'progressive_plan_missing_since_validation'
    try:
        plan = json.loads(text)
    except ValueError:
        return 'progressive_plan_changed_since_validation'
    return None if plan_digest(plan) == digest else 'progressive_plan_changed_since_validation'


def _refusal(result):
    """Explicit reason for a refused stream ownership."""
    if result['conflict'] == 'session_sink_changed':
        return 'progressive_stream_session_sink_changed'
    if result['state'] in ('writing', 'uncertain'):
        return 'progressive_stream_sink_uncertain'
    if result['state'] == 'open':
        # An active owner, or one that died: indistinguishable, never taken over.
        return 'progressive_stream_owned_elsewhere'
    if result['frames']:
        return 'progressive_stream_already_delivered'
    return 'progressive_stream_sink_in_use'


def window_reason(config, plan, ledger, *, layout=None):
    """Why a prepared plan cannot join this stream, or None.

    JSON-level checks only; the renderer still verifies PCM, stem, voices and
    the original-audio policy when the window itself renders.
    """
    if plan is None:
        return 'progressive_plan_unreadable'
    if plan.get('engine_fingerprint') != config.engine_fingerprint():
        return 'progressive_plan_stale'
    report = plan.get('report')
    if not isinstance(report, dict) or not _finite(report.get('origin')) or not _finite(report.get('end')):
        return 'progressive_plan_timeline_invalid'
    origin, end = float(report['origin']), float(report['end'])
    if not 0 <= origin < end:
        return 'progressive_plan_timeline_invalid'
    if report.get('issues'):
        return 'progressive_plan_unresolved'
    if report.get('render_start') != report.get('origin') or report.get('render_end') != report.get('end'):
        # Offline worker plans observe context around their render range.
        return 'progressive_plan_not_complete_window'
    if plan.get('channel_layout') not in LAYOUT_CHANNELS:
        return 'progressive_plan_layout_unknown'
    if layout is not None and plan['channel_layout'] != layout:
        return 'progressive_layout_changed'
    if plan.get('protection_scope') is None:
        return 'progressive_plan_protection_scope_missing'
    if plan['protection_scope'] != ledger.scope:
        return 'progressive_plan_protection_scope_changed'
    if not isinstance(report.get('raw_turns'), list):
        return 'progressive_plan_coverage_unknown'
    for cue in plan.get('cues') or ():
        if not (_finite(cue.get('start')) and _finite(cue.get('end'))
                and origin <= cue['start'] < cue['end'] <= end):
            return 'progressive_cue_crosses_window'
    return None


def _source_frames(path):
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames, info.samplerate


def _tag(error, default):
    found = re.findall(r'\(([a-z0-9_]+)\)', str(error))
    return found[-1] if found else default


class ProgressiveController:
    """Render prepared adjacent windows in order into one continuous sink.

    ``windows`` (passed to ``run``) is an iterable of ``{'plan': path,
    'source': wav_path}``. It is advanced one window ahead of rendering, so a
    generator may prepare window i+1 lazily; it is then prepared before
    window i renders (serially). Real-time overlap of preparation and
    rendering is the caller's job (e.g. a producer thread feeding a queue).

    The sink receives float32 blocks of the plans' channel layout; the caller
    owns it (e.g. ``LiveMux(ffmpeg, source, folder, first_origin, last_end,
    channels, owner, channel_layout=layout).write``) and calls ``finish``.

    ``sink_id`` (optional, at most 200 characters) names the physical output,
    e.g. the LiveMux folder; that sink can then serve only one session ever.
    Without it the session alone stands for the sink, and reuse of one
    physical sink under another session cannot be detected.
    """

    def __init__(self, config, protection_ledger, sink, owner, *, session=None, language=None, hints=(),
                 event=lambda _: None, pause_evidence=None, render=render_progressively,
                 source_frames=_source_frames, sink_id=None):
        if protection_ledger is None:
            raise ValueError('The progressive controller needs the job protection ledger')
        if sink_id is not None and (not isinstance(sink_id, str) or not sink_id or len(sink_id) > 200):
            raise ValueError('The sink identity must be a nonempty string of at most 200 characters')
        self.config, self.ledger, self.sink, self.owner = config, protection_ledger, sink, owner
        self.session = session or 'stream-' + uuid.uuid4().hex
        self.sink_id, self.run_id = sink_id, uuid.uuid4().hex
        self._token = self._acknowledged = None
        self.language, self.hints, self.event = language, hints, event
        self.pause_evidence, self.render, self.source_frames = pause_evidence, render, source_frames
        self.scope = timing_carry_scope(**protection_ledger.scope, revision=protection_ledger.revision)

    def _receipt(self, windows, status, reason=None, error=None):
        stream = None
        if self._token is not None:
            row = self.ledger.store.output_stream(self.session)
            stream = {'sink_id': self.sink_id, 'state': row['state'], 'acknowledged_frames': int(row['frames']),
                      'unconfirmed_sink_write': None if row['write_start'] is None else {
                          'start_frame': int(row['write_start']), 'end_frame': int(row['write_end'])},
                      'delivery': DELIVERY}
        return {'version': CONTROLLER_VERSION, 'session': self.session, 'status': status, 'reason': reason,
                'error': error, 'windows': windows, 'frames': sum(w.get('frames') or 0 for w in windows),
                'stream': stream,
                'pause_evidence': 'caller' if self.pause_evidence else 'none',
                'production_caller': None,
                'scope': 'Opt-in controller; unused by worker/coordinator. Synthetic tests only; '
                         'full-film real-time behaviour is not established.'}

    def _acquire(self):
        """Own this (session, sink) before any sink call; a reason when refused."""
        result = self.ledger.store.acquire_output_stream(
            self.session, self.sink_id, self.run_id,
            {'controller': CONTROLLER_VERSION, 'job_id': self.ledger.job_id, 'revision': self.ledger.revision,
             'scope_key': self.ledger.key})
        if result['token'] is None:
            return _refusal(result)
        self._token, self._acknowledged = result['token'], 0
        return None

    def _write(self, block):
        """The renderer's sink: persist the call's intent, call, then acknowledge."""
        store, end = self.ledger.store, self._acknowledged + len(block)
        if not store.begin_output_write(self.session, self._token, self._acknowledged, end):
            raise ValueError('This run no longer owns its output stream (progressive_stream_ownership_lost)')
        try:
            self.sink(block)
        except BaseException:
            try:
                store.fail_output_write(self.session, self._token)
            except Exception:
                pass    # the row stays 'writing', which every reader treats as uncertain
            raise
        if not store.confirm_output_write(self.session, self._token, end):
            raise ValueError('This run no longer owns its output stream (progressive_stream_ownership_lost)')
        self._acknowledged = end

    def _lookahead(self, plan, frames, rate, nxt, layout):
        """``(lookahead, reason, next_plan, next_digest, next_frames)`` for the window after ``plan``."""
        plan_next, digest = load_plan(nxt['plan'])
        reason = window_reason(self.config, plan_next, self.ledger, layout=layout)
        if reason:
            return None, reason, plan_next, digest, None
        end, following = float(plan['report']['end']), float(plan_next['report']['origin'])
        if following < end - ADJACENT_TOLERANCE:
            return None, 'progressive_windows_overlap', plan_next, digest, None
        try:
            following_frames, following_rate = self.source_frames(nxt['source'])
        except (OSError, RuntimeError, ValueError):
            return None, 'progressive_source_unreadable', plan_next, digest, None
        if following_rate != rate or not following_frames:
            return None, 'progressive_source_rate_changed', plan_next, digest, None
        try:
            ahead = next_window_lookahead(plan_next, edge=end, channel_layout=layout,
                                          protection_scope=self.ledger.scope)
        except ValueError as error:
            return None, _tag(error, 'progressive_next_plan_invalid'), plan_next, digest, None
        if round(following * rate) != round(float(plan['report']['origin']) * rate) + frames:
            # A one-frame gap or overlap would shift the continuous sink.
            return None, 'progressive_edge_frame_mismatch', plan_next, digest, None
        if self.pause_evidence is not None:
            pauses = list(self.pause_evidence(edge=end, plan=plan, next_plan=plan_next) or ())
            ahead['pauses'] = [{'start': p['start'], 'end': p['end']} for p in pauses]
        return ahead, None, plan_next, digest, following_frames

    def run(self, windows):
        stream = iter(windows)
        records = []
        current = next(stream, None)
        if current is None:
            return self._receipt(records, 'complete')
        plan, digest = load_plan(current['plan'])
        reason = window_reason(self.config, plan, self.ledger)
        if reason:
            return self._receipt(records, 'blocked', reason)
        # Ownership precedes the first sink call; a refused stream writes nothing.
        reason = self._acquire()
        if reason:
            return self._receipt(records, 'blocked', reason)
        try:
            outcome = self._chain(stream, records, current, plan, digest)
        finally:
            # Nothing is in flight here unless a sink call raised (already
            # 'uncertain') or this process dies (the row stays open/writing).
            self.ledger.store.close_output_stream(self.session, self._token)
        return self._receipt(records, *outcome)

    def _chain(self, stream, records, current, plan, digest):
        layout = plan['channel_layout']
        previous = samples = validated_frames = None
        while True:
            try:
                frames, rate = self.source_frames(current['source'])
            except (OSError, RuntimeError, ValueError):
                return 'blocked', 'progressive_source_unreadable'
            if validated_frames is not None and frames != validated_frames:
                return 'blocked', 'progressive_source_changed_since_validation'
            # The next window is pulled (and possibly prepared) before this
            # one renders: its lookahead and protection precede any crossing.
            nxt = next(stream, None)
            ahead, ahead_reason, plan_next, digest_next, frames_next = (
                (None, None, None, None, None) if nxt is None else self._lookahead(plan, frames, rate, nxt, layout))
            # The file must still hold the validated object (B6b); the
            # renderer re-checks the object it actually loads.
            reason = _bound_reason(current['plan'], digest)
            if reason:
                return 'blocked', reason
            carry = {'scope': self.scope, 'previous': previous, 'samples': samples, 'session': self.session,
                     'chain': True}
            if ahead is not None:
                carry['lookahead'] = ahead
            record = {'plan': str(current['plan']), 'plan_digest': digest,
                      'origin': float(plan['report']['origin']), 'end': float(plan['report']['end']),
                      'lookahead': 'ready' if ahead is not None else ahead_reason or 'end_of_stream',
                      'frames': None}
            records.append(record)
            try:
                result = self.render(self.config, current['plan'], current['source'], self.owner, self._write,
                                     language=self.language, hints=self.hints, event=self.event,
                                     protection_ledger=self.ledger, timing_carry=carry, bound_plan=digest)
            except Exception as error:
                record['error'] = str(error)
                return 'failed', _tag(error, 'progressive_window_failed'), str(error)
            record.update(frames=result['frames'], sample_carry_in=result['sample_carry_in'],
                          sample_carry_out={'frames': result['sample_carry']['frames'],
                                            'edge_frame': result['sample_carry']['edge_frame'],
                                            'takes': [t['id'] for t in result['sample_carry']['takes']]})
            if result['frames'] != frames:
                return 'failed', 'progressive_window_incomplete'
            if result['sample_carry']['edge_frame'] != round(record['origin'] * rate) + frames:
                # Too late to retract audio, but nothing is handed on.
                return 'failed', 'progressive_window_timeline_mismatch'
            if nxt is None:
                return ('complete',)
            if ahead_reason:
                return 'blocked', ahead_reason
            # Handed on only after this window fully succeeded (B4/B5 duty).
            previous, samples = result['timing_carry']['carry_out'], result['sample_carry']
            current, plan, digest, validated_frames = nxt, plan_next, digest_next, frames_next
