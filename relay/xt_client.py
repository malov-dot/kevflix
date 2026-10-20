"""Client of the complete upstream TranslationSession; no ASR/MT/TTS substitute.

The receiver only collects events. Claimed-store journal writes belong to the
chunk owner. Readiness requires the patched speaker/ASR/TTS drain barrier.
"""
from __future__ import annotations

import json
import hashlib
import errno
import math
import threading
import time
import uuid

from .movie_faults import MovieConfigurationError, MovieResourceWait, MovieFault

RATE = 16000
BYTES_PER_SECOND = RATE * 2
TERMINAL = {'ok', 'translation_failed', 'tts_failed', 'empty_output',
            'source_english', 'language_unknown', 'language_unsupported'}


def recoverable_session(reason, detail):
    error = MovieResourceWait('xt_session_recoverable', reason + ': ' + str(detail))
    error.recovery_reason = reason
    return error


def _transport_error(error):
    """Recognized connection loss only; protocol, TLS/config and assets stay fatal."""
    import websocket
    if isinstance(error, (ConnectionError, TimeoutError, websocket.WebSocketConnectionClosedException,
                          websocket.WebSocketTimeoutException)):
        return True
    network_errno = {getattr(errno, name, None) for name in ('ECONNABORTED', 'ECONNREFUSED', 'ECONNRESET',
        'ENETRESET', 'ENETDOWN', 'ENETUNREACH', 'EHOSTDOWN', 'EHOSTUNREACH', 'ETIMEDOUT', 'EPIPE', 'ENOTCONN')}
    return isinstance(error, OSError) and error.errno is not None and error.errno in network_errno


class XTSession:
    def __init__(self, config, owner):
        self.config, self.owner = config, owner
        self.session_uid = uuid.uuid4().hex
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._socket = None
        self._thread = None
        self._closed = False
        self._failure = None
        self._ready = self._configured = False
        self._audio_bytes = 0
        self._mapping = []
        self._frontier_ms = 0.
        self._processed_ms = 0.
        self._outcomes = {}
        self._barriers = {}
        self._events = []
        self._progress_serial = 0
        self._progress_at = time.monotonic()
        self._progress_values = {'speaker_processed_ms': 0., 'asr_processed_ms': 0., 'terminal_count': 0}
        self._terminal_count = 0
        self._waiting_barriers = set()
        self._pending_audio = None
        self._audio_frames = {}
        self._io_timeout_seconds = 2.
        self.feed_epoch = 0
        self.server_session_id = None
        self.source_lang = None
        self.endpoints = {}

    @property
    def failed(self):
        return self._failure is not None or self._closed

    @property
    def sent_ms(self):
        return self._audio_bytes * 1000. / BYTES_PER_SECOND

    def _check(self):
        self.owner.check()
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise recoverable_session('transport_lost', 'session closed')

    def check(self):
        """Preserve the exact stored fault; the coordinator owns recovery policy."""
        self._check()

    def open(self, services, job_id, revision, source_lang):
        import websocket
        data = getattr(self.config, 'data', self.config)
        timeout = float(data.get('xt_barrier_timeout_seconds', 120.))
        if not math.isfinite(timeout) or timeout < 1.:
            raise MovieConfigurationError('xt_timeout_invalid', 'Socket I/O timeout must be finite and >=1 second')
        self._io_timeout_seconds = min(120., timeout)
        self.job_id, self.revision = job_id, int(revision)
        self.source_lang = source_lang
        self.endpoints = dict(services.ensure(self.owner))
        if self.endpoints.get('generation') is None:
            raise MovieConfigurationError('xt_generation_missing', 'Service generation tag is required')
        endpoint = self.endpoints['ws']
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        if parsed.scheme not in ('ws', 'wss') or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise MovieConfigurationError('xt_nonlocal_endpoint', 'XT session endpoint must be loopback')
        try:
            self._socket = websocket.create_connection(endpoint, timeout=2, enable_multithread=True,
                http_no_proxy=['127.0.0.1', 'localhost', '::1'])
            self._socket.settimeout(self._io_timeout_seconds)
            self._thread = threading.Thread(target=self._receive, name='xt-session-events', daemon=True)
            self._thread.start()
            self._wait(lambda: self._ready, 30, 'session_ready')
            self._send({'action': 'session_config', 'source_lang': source_lang,
                        'target_lang': 'en', 'sample_rate': RATE})
            self._wait(lambda: self._configured, 30, 'session_configured')
        except BaseException as error:
            self.close()
            if _transport_error(error):
                raise recoverable_session('transport_lost', error) from error
            raise
        return self.session_uid

    def _send(self, record):
        self._check()
        try:
            with self._send_lock:
                self._socket.send(json.dumps(record, ensure_ascii=True))
        except BaseException as error:
            self._transport_failure(error)
            if _transport_error(error):
                raise self._failure from error
            raise

    def _transport_failure(self, error):
        with self._condition:
            if _transport_error(error):
                self._failure = recoverable_session('transport_lost', '%s: %s' % (type(error).__name__, error))
            elif isinstance(error, MovieFault):
                self._failure = error
            else:
                self._failure = MovieConfigurationError('xt_protocol_integrity', '%s: %s' % (type(error).__name__, error))
            self._condition.notify_all()

    def _note_progress(self):
        self._progress_serial += 1
        self._progress_at = time.monotonic()

    def _wait_barrier(self, barrier_id, idle_seconds, overall_seconds):
        started = last_progress = time.monotonic()
        with self._condition:
            serial = self._progress_serial
            while barrier_id not in self._barriers:
                self._check()
                if self._progress_serial != serial:
                    serial = self._progress_serial
                    last_progress = max(started, self._progress_at)
                now = time.monotonic()
                if now - started >= overall_seconds:
                    raise recoverable_session('drain_request_timeout', 'Barrier exceeded %.1fs overall' % overall_seconds)
                if now - last_progress >= idle_seconds:
                    raise recoverable_session('drain_stalled', 'No processing progress for %.1fs' % idle_seconds)
                self._condition.wait(min(.25, overall_seconds - (now - started), idle_seconds - (now - last_progress)))
        self._check()

    def _wait(self, predicate, timeout, phase):
        deadline = time.monotonic() + timeout
        with self._condition:
            while not predicate():
                self._check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MovieConfigurationError('xt_event_timeout', 'Timed out waiting for ' + phase)
                self._condition.wait(min(remaining, .25))
        self._check()

    def _map_span(self, start_ms, end_ms):
        start_ms, end_ms = float(start_ms), float(end_ms)
        if not math.isfinite(start_ms) or not math.isfinite(end_ms) or end_ms <= start_ms:
            raise ValueError('invalid XT source span')
        spans = []
        for row in self._mapping:
            a, b = max(start_ms, row['session_start_ms']), min(end_ms, row['session_end_ms'])
            if b > a:
                lo = row['film_start'] + (a - row['session_start_ms']) / 1000.
                hi = row['film_start'] + (b - row['session_start_ms']) / 1000.
                if spans and abs(spans[-1][1] - lo) <= 1e-6:
                    spans[-1][1] = hi
                else:
                    spans.append([lo, hi])
        return spans

    def _film_point(self, milliseconds):
        point = None
        for row in self._mapping:
            if milliseconds < row['session_start_ms']:
                break
            point = row['film_start'] + (min(milliseconds, row['session_end_ms']) - row['session_start_ms']) / 1000.
        return point

    def _event(self, message):
        action = message.get('action')
        data = message.get('data') or {}
        if not isinstance(data, dict):
            raise ValueError('XT event data must be an object')
        if action == 'session_ready':
            self.server_session_id = data['session_id']
            if not isinstance(self.server_session_id, str) or not self.server_session_id:
                raise ValueError('upstream session identity is missing')
            self._ready = True
        elif action == 'session_configured':
            if (data.get('source_lang') != self.source_lang or data.get('target_lang') != 'en'
                    or int(data.get('sample_rate', 0)) != RATE):
                raise ValueError('XT session configuration acknowledgement differs from request')
            self._configured = True
        elif action == 'commit_frontier':
            # Diagnostic only: the readiness barrier below attests queue drain.
            self._events.append({'event': action, 'data': data, 'session_uid': self.session_uid})
        elif action == 'source_language_detected':
            code = data.get('code')
            if (not isinstance(code, str) or len(code) != 2 or not code.isalpha()
                    or code != code.lower() or data.get('language_source') != 'qwen3-asr-native'
                    or not isinstance(data.get('seg_uid'), int) or data['seg_uid'] < 1):
                raise ValueError('invalid native language event')
            self._events.append({'event': action, 'session_uid': self.session_uid,
                'seg_uid': str(data['seg_uid']), 'language_code': code,
                'language_raw': data.get('raw'), 'language_source': data['language_source']})
        elif action == 'xt_progress':
            advanced = False
            values = {}
            for key in self._progress_values:
                value = float(data[key])
                if (not math.isfinite(value) or value < self._progress_values[key]
                        or (key != 'terminal_count' and value > self.sent_ms + 1e-3)
                        or (key == 'terminal_count' and (value != int(value) or value > self._terminal_count))):
                    raise ValueError('invalid/nonmonotonic XT processing progress: ' + key)
                values[key] = value
                advanced = advanced or value > self._progress_values[key]
            self._progress_values.update(values)
            if advanced:
                self._note_progress()
                self._events.append({'event': action, 'data': values, 'session_uid': self.session_uid})
        elif action in ('drain_complete', 'barrier_done'):
            barrier_id = data['barrier_id']
            processed = float(data['processed_audio_ms'])
            frontier = float(data['commit_frontier_ms'])
            if (not math.isfinite(processed) or not math.isfinite(frontier)
                    or not 0 <= frontier <= processed + 1e-3 or processed > self.sent_ms + 1e-3
                    or data.get('open_seg_uids') != []):
                raise ValueError('invalid/incomplete XT drain acknowledgement')
            self._barriers[barrier_id] = dict(data)
            self._processed_ms, self._frontier_ms = processed, frontier
            self._events.append({'event': action, 'data': data, 'session_uid': self.session_uid})
        elif action in ('source_segment_ready', 'translation_ready', 'error'):
            if 'seg_uid' not in data:
                recovery_codes = {'xt_barrier_stalled': 'drain_stalled', 'xt_request_timeout': 'drain_request_timeout',
                                  'xt_transport_lost': 'transport_lost'}
                code = data.get('code')
                if (action == 'error' and code in recovery_codes and data.get('fault_kind') == 'resource'
                        and data.get('recoverable') is True and data.get('fatal') is False):
                    if code != 'xt_transport_lost' and data.get('barrier_id') not in self._waiting_barriers:
                        raise ValueError('XT failure names an unknown barrier')
                    self._failure = recoverable_session(recovery_codes[code], data.get('message') or str(data))
                    self._events.append({'event': 'session_recoverable', 'data': data, 'session_uid': self.session_uid})
                    return
                raise ValueError('XT unattributed session/controller error: ' + json.dumps(message))
            uid = str(data['seg_uid'])
            spans = self._map_span(data['source_start_ms'], data['source_end_ms'])
            existing = self._outcomes.get(uid)
            record = {**(existing or {}), 'session_uid': self.session_uid, 'seg_uid': uid,
                      'source_start_ms': data['source_start_ms'], 'source_end_ms': data['source_end_ms'],
                      'source_spans': spans, 'film_start': spans[0][0] if spans else None,
                      'film_end': spans[-1][1] if spans else None,
                      'speaker_id': data.get('speaker_id', (existing or {}).get('speaker_id')),
                      'source_text': data.get('source_text', (existing or {}).get('source_text', '')),
                      'language_code': data.get('language_code'), 'language_raw': data.get('language_raw'),
                      'language_source': data.get('language_source')}
            for key in ('commit_basis', 'language_context', 'language_observation_id', 'reference_refresh'):
                record[key] = data.get(key, (existing or {}).get(key))
            code = record['language_code']
            if code is not None and (not isinstance(code, str) or len(code) != 2
                                     or not code.isalpha() or code != code.lower()):
                raise ValueError('invalid segment language code')
            if existing and existing.get('status') in TERMINAL:
                raise ValueError('duplicate terminal/late source event for seg_uid ' + uid)
            if action == 'source_segment_ready':
                record['status'] = 'pending'
            elif action == 'translation_ready':
                record.update(status=data.get('status', 'ok'), translated_text=data.get('translated_text', ''),
                              tts_wav=data.get('tts_audio_path'), error=None,
                              tts_bytes=int(data['tts_bytes']), tts_sha256=data['tts_sha256'],
                              tts_sample_rate=int(data['tts_sample_rate']))
                if record['status'] != 'ok' or data.get('terminal') is not True:
                    raise ValueError('translation_ready is not a terminal full output')
                if not record['translated_text'].strip():
                    record.update(status='empty_output', error='empty translation')
                if self._pending_audio is not None:
                    raise ValueError('second XT audio announcement before the preceding binary frame')
                self._pending_audio = uid
                record['announced_status'] = record['status']
                record['status'] = 'audio_pending'
            else:
                status = data.get('status')
                if status not in TERMINAL - {'ok'}:
                    raise ValueError('XT error lacks a terminal per-segment status: ' + json.dumps(message))
                if data.get('terminal') is not True:
                    raise ValueError('XT per-segment terminal flag missing')
                if status == 'source_english' and code != 'en':
                    raise ValueError('source_english terminal without English language evidence')
                if status == 'language_unknown' and code is not None:
                    raise ValueError('language_unknown terminal has a language code')
                record.update(status=status, error=data.get('error') or data.get('message') or str(data))
            if len(spans) > 1:
                record['mapping_error'] = 'controller segment crosses a seek/padding discontinuity'
            epochs = {r['feed_epoch'] for r in self._mapping if min(float(data['source_end_ms']),r['session_end_ms'])
                      > max(float(data['source_start_ms']),r['session_start_ms'])}
            record['feed_epoch'] = next(iter(epochs)) if len(epochs) == 1 else None
            self._outcomes[uid] = record
            self._events.append({'event': action, **record})
            if record['status'] in TERMINAL:
                self._terminal_count += 1
                self._note_progress()

    def _binary(self, message):
        uid = self._pending_audio
        if uid is None:
            raise ValueError('XT binary frame without an attributed audio announcement')
        record = self._outcomes[uid]
        if (len(message) != record['tts_bytes'] or len(message) % 2
                or hashlib.sha256(message).hexdigest() != record['tts_sha256']
                or not 8000 <= record['tts_sample_rate'] <= 192000):
            raise ValueError('XT seg_uid %s binary length/SHA/rate mismatch' % uid)
        record['status'] = record.pop('announced_status')
        if not message or record.get('mapping_error'):
            record.update(status='empty_output', error=record.get('mapping_error') or 'empty upstream PCM')
        self._audio_frames[uid] = bytes(message)
        self._pending_audio = None
        self._terminal_count += 1
        self._note_progress()
        self._events.append({'event': 'audio_verified', **record})

    def _receive(self):
        import websocket
        try:
            while not self._closed:
                try:
                    # Match websocket-client's recv() read lock and decoding,
                    # retaining CLOSE data that recv() reduces to an empty string.
                    # Default control_frame=False keeps automatic ping/pong handling.
                    with self._socket.readlock:
                        opcode, payload = self._socket.recv_data()
                    if opcode == websocket.ABNF.OPCODE_CLOSE:
                        raw = payload.encode('utf-8') if isinstance(payload, str) else bytes(payload)
                        code = int.from_bytes(raw[:2], 'big') if len(raw) >= 2 else None
                        reason = raw[2:514].decode('utf-8', errors='replace') if len(raw) >= 2 else ''
                        raise ConnectionError('XT peer closed WebSocket: code=%s reason=%r' % (code, reason))
                    if opcode == websocket.ABNF.OPCODE_TEXT:
                        message = payload.decode('utf-8') if isinstance(payload, bytes) else payload
                    elif opcode == websocket.ABNF.OPCODE_BINARY:
                        message = payload
                    else:
                        message = ''
                except websocket.WebSocketTimeoutException:
                    continue
                if message in (None, ''):
                    raise ConnectionError('XT session disconnected before client close')
                if isinstance(message, bytes):
                    with self._condition:
                        self._binary(message)
                        self._condition.notify_all()
                    continue
                with self._condition:
                    self._event(json.loads(message))
                    self._condition.notify_all()
        except BaseException as error:
            if not self._closed:
                self._transport_failure(error)

    def _wait_pcm_writable(self, phase):
        """Check cancellation while waiting for initial socket writability.

        Writability is not a guarantee that a complete frame write cannot block.
        """
        import select
        waited_from = time.monotonic()
        while True:
            self._check()
            sock = self._socket.sock
            if sock is None:
                raise ConnectionError('XT socket closed while waiting to send PCM')
            writable = select.select([], [sock], [], .25)[1]
            self._check()
            if writable:
                return
            idle = time.monotonic() - max(waited_from, self._progress_at)
            if idle >= self._io_timeout_seconds:
                raise recoverable_session('drain_stalled',
                    '%s PCM socket not writable for %.3fs without processing progress (limit %.3fs)' %
                    (phase, idle, self._io_timeout_seconds))

    def _pcm_timeout(self, error, started, phase):
        """Only a real PCM send timeout is a backpressure stall, with no retry here."""
        import websocket
        if not isinstance(error, websocket.WebSocketTimeoutException):
            return
        self.owner.check()
        with self._condition:
            if self._failure is None:
                self._failure = recoverable_session('drain_stalled',
                    '%s PCM send timed out after %.3fs (socket timeout %.3fs)' %
                    (phase, time.monotonic() - started, self._io_timeout_seconds))
            self._condition.notify_all()
            raise self._failure from error

    def feed(self, film_start, pcm16):
        self._check()
        pcm16 = bytes(pcm16)
        if not pcm16 or len(pcm16) % 2:
            raise MovieConfigurationError('xt_pcm_geometry', 'Need nonempty PCM16 mono 16kHz')
        duration = len(pcm16) / BYTES_PER_SECOND
        with self._condition:
            if any(min(film_start + duration, row['film_end']) > max(film_start, row['film_start']) + 1e-6
                   for row in self._mapping):
                raise MovieConfigurationError('xt_source_refed', 'Mapped source audio must be fed only once per session')
            start_ms = self.sent_ms
            self._mapping.append({'session_start_ms': start_ms, 'session_end_ms': start_ms + duration * 1000.,
                                  'film_start': float(film_start), 'film_end': float(film_start) + duration,
                                  'feed_epoch': self.feed_epoch})
            self._audio_bytes += len(pcm16)
            self._events.append({'event': 'feed', 'session_uid': self.session_uid, 'mapping': dict(self._mapping[-1])})
        send_started = time.monotonic()
        try:
            with self._send_lock:
                for offset in range(0, len(pcm16), 3200):
                    self._wait_pcm_writable('feed')
                    send_started = time.monotonic()
                    self._socket.send_binary(pcm16[offset:offset + 3200])
        except BaseException as error:
            self._pcm_timeout(error, send_started, 'feed')
            self._transport_failure(error)
            if _transport_error(error):
                raise self._failure from error
            raise

    def pad(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid seek silence duration')
        pcm = bytes(round(seconds * RATE) * 2)
        with self._condition:
            start_ms = self.sent_ms
            self._audio_bytes += len(pcm)
            self._events.append({'event': 'padding', 'session_uid': self.session_uid,
                                 'seconds': len(pcm) / BYTES_PER_SECOND, 'mapped': False,
                                 'session_start_ms': start_ms, 'session_end_ms': self.sent_ms,
                                 'feed_epoch': self.feed_epoch})
        send_started = time.monotonic()
        try:
            with self._send_lock:
                for offset in range(0, len(pcm), 3200):
                    self._wait_pcm_writable('pad')
                    send_started = time.monotonic()
                    self._socket.send_binary(pcm[offset:offset + 3200])
        except BaseException as error:
            self._pcm_timeout(error, send_started, 'pad')
            self._transport_failure(error)
            if _transport_error(error):
                raise self._failure from error
            raise

    def drain(self, timeout=120):
        timeout = float(timeout)
        data = getattr(self.config, 'data', self.config)
        overall = float(data.get('xt_request_timeout_seconds', 900.))
        if not math.isfinite(timeout) or not math.isfinite(overall) or timeout < 1 or overall < 1:
            raise MovieConfigurationError('xt_timeout_invalid', 'Barrier deadlines must be finite and positive')
        timeout, overall = min(120., timeout), min(900., overall)
        barrier_id, through = uuid.uuid4().hex, self.sent_ms
        with self._condition:
            self._waiting_barriers.add(barrier_id)
        try:
            self._send({'action': 'xt_barrier', 'barrier_id': barrier_id, 'through_ms': through,
                        'timeout_seconds': timeout, 'no_progress_timeout_seconds': timeout,
                        'request_timeout_seconds': overall})
            self._wait_barrier(barrier_id, timeout, overall)
        finally:
            with self._condition:
                self._waiting_barriers.discard(barrier_id)
        with self._condition:
            ack = self._barriers.pop(barrier_id)
            maximum = int(ack['max_enqueued_uid'])
            observed = {int(uid) for uid in self._outcomes}
            if (float(ack['processed_audio_ms']) + 1e-3 < through or self.open_uids()
                    or maximum < 0 or maximum > len(observed)
                    or any(uid not in observed for uid in range(1, maximum + 1))):
                raise MovieConfigurationError('xt_drain_incomplete', 'Drain did not process all supplied PCM and terminal outcomes')
            return ack

    def audio_frame(self, seg_uid):
        with self._condition:
            return self._audio_frames[str(seg_uid)]

    def release_audio(self, seg_uid):
        with self._condition:
            self._audio_frames.pop(str(seg_uid), None)

    def frontier_film(self):
        with self._condition:
            return self._film_point(self._frontier_ms)

    def open_uids(self):
        with self._condition:
            return [uid for uid, row in self._outcomes.items() if row.get('status') not in TERMINAL]

    def outcomes(self, until_film):
        with self._condition:
            return [dict(row) for row in self._outcomes.values()
                    if row.get('film_start') is not None and row['film_start'] < until_film]

    def fed_range(self, begin, end):
        spans = sorted((r['film_start'], r['film_end']) for r in self._mapping)
        cursor = begin
        for a, b in spans:
            if a > cursor + 1e-6:
                break
            if b > cursor:
                cursor = b
        return cursor >= end - 1e-6

    def take_events(self):
        with self._condition:
            events, self._events = self._events, []
            return events

    def close(self):
        # No stop/config command: cancellation closes this socket, making all
        # unfinished outcomes unavailable; a normal owner drains before close.
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._socket is not None:
            try:
                # close() sends a CLOSE frame under the library send lock before
                # applying its short close deadline. Abort blocked I/O first;
                # shutdown never waits for that lock or the long I/O timeout.
                try:
                    self._socket.abort()
                finally:
                    self._socket.shutdown()
            except Exception as error:
                if self._failure is None:
                    self._failure = recoverable_session('transport_lost', 'socket close failed: ' + str(error))
        if self._thread is not None and self._thread.ident is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
