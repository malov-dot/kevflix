"""Live streaming Diart speaker step (stdlib + numpy; Diart is never imported here).

A stateful Diart sidecar (scripts/diart_stream_server.py, run inside WSL where torch + diart live) is fed the SAME 16 kHz
analysis windows the dubbing pipeline already extracts, in order, one stream per source. Per window it answers who speaks
when with ids that stay stable for the whole stream (the online clustering state persists between windows), so no
per-film file and no pre-pass is needed and any source works (a local file today, downloaded or streamed audio later):
the only input is the window audio.

Precedence inside SpeakerAnalyzer.analyze (actor_beta.diart_stream, default ON when actor_beta.enabled):
    streaming sidecar -> Diart file artifact (actor_beta.diart_turns) -> offline pyannote per window.
Every failure here raises DiartStreamUnavailable(reason); the analyzer records the reason once and uses the next
provider for that window. A call never waits longer than the call timeout. Streaming results depend on stream state, so
they are never stored in the analysis cache (the embeddings computed from them are content keyed and may be).

Wire format (both directions): 4-byte big-endian header length, JSON header (always carries ``payload_bytes``), payload.
Requests: open{stream} | feed{stream,start,n,create?} + little-endian float32 PCM | reset{stream} | close{stream} | ping.
Replies echo ``id`` and carry ``ok``; the server also sends one unsolicited ``{"event": "ready"}`` (or ``fatal``) frame.
"""
from __future__ import annotations

import atexit
import collections
import hashlib
import json
import logging
import math
import os
import queue
import re
import struct
import subprocess
import threading
import time
from pathlib import Path

from . import diart_turns

LOG = logging.getLogger(__name__)
VERSION = 1
RATE = diart_turns.SAMPLE_RATE
LABEL_PREFIX = diart_turns.LABEL_PREFIX
# Diart parameters. These are the pre-pass defaults; delta_new 0.7 kept one real speaker whole where 0.6 split it in two.
PARAMS = {'duration': 5.0, 'step': 0.5, 'latency': 5.0, 'tau_active': 0.5, 'rho_update': 0.3, 'delta_new': 0.7,
          'gamma': 3.0, 'beta': 10.0, 'max_speakers': 40, 'metric': 'cosine', 'aggregation': 'hamming/loose',
          'batch_size': 32}
SEEK_GAP = 35.0              # a window starting this far past the fed audio is a seek: the stream restarts cleanly
MERGE_GAP = .05              # same-speaker turns closer than this are one turn
HISTORY_SECONDS = 7200.      # turns older than this (behind the newest fed audio) are forgotten
EPS = .02
MAX_STREAMS = 4
CALL_TIMEOUT = 20.0
START_WAIT = 20.0
MAX_RESTARTS = 1
RULES = {'seek_gap_seconds': SEEK_GAP, 'merge_gap_seconds': MERGE_GAP, 'history_seconds': HISTORY_SECONDS,
         'overlapping_audio': 'only audio not yet fed is sent; fully covered windows are answered from stored turns',
         'gap_audio': 'zero-filled up to the seek gap, otherwise the stream restarts',
         'tail': 'provisional (state copy + flush) and replaced by the next feed',
         'vectors': 'community1-native from non-overlapped speech (relay.diart_turns.label_vectors)'}
DEFAULT_WSL = {
    'distro': 'Ubuntu-24.04',
    'python': '/root/cleanup-holding-20261020/kevflix-textless/tools/python/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10',
    'site_packages': '/root/cleanup-holding-20261020/kevflix-textless/env-speakers/lib/python3.10/site-packages'}
SERVER = Path(__file__).resolve().parents[1] / 'scripts' / 'diart_stream_server.py'
MAX_HEADER = 1 << 20
MAX_PAYLOAD = 256 << 20
_BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_CREATE_NO_WINDOW = 0x08000000


class DiartStreamUnavailable(Exception):
    """The sidecar cannot answer for this window; ``reason`` is a short machine-readable code."""

    def __init__(self, reason, detail=''):
        super().__init__(reason + (': ' + detail if detail else ''))
        self.reason = reason


class ProtocolError(Exception):
    pass


# --------------------------------------------------------------------------------------------------------- framing
def write_frame(stream, header, payload=b''):
    raw = json.dumps({**header, 'payload_bytes': len(payload)}, separators=(',', ':')).encode('utf-8')
    stream.write(struct.pack('>I', len(raw)) + raw)
    if payload:
        stream.write(payload)
    stream.flush()


def _read_exact(stream, count):
    chunks, left = [], count
    while left:
        block = stream.read(left)
        if not block:
            raise EOFError('stream closed inside a frame')
        chunks.append(block)
        left -= len(block)
    return b''.join(chunks)


def read_frame(stream):
    """(header dict, payload bytes), or None on a clean end of stream at a frame boundary."""
    first = stream.read(4)
    if not first:
        return None
    if len(first) < 4:
        first += _read_exact(stream, 4 - len(first))
    size = struct.unpack('>I', first)[0]
    if size > MAX_HEADER:
        raise ProtocolError('header too large')
    try:
        header = json.loads(_read_exact(stream, size).decode('utf-8'))
    except ValueError as error:
        raise ProtocolError('header is not JSON') from error
    length = header.get('payload_bytes', 0) if isinstance(header, dict) else None
    if not isinstance(length, int) or isinstance(length, bool) or not 0 <= length <= MAX_PAYLOAD:
        raise ProtocolError('invalid payload length')
    return header, (_read_exact(stream, length) if length else b'')


# --------------------------------------------------------------------------------------------------------- settings
def _beta(config):
    data = getattr(config, 'data', config)
    beta = data.get('actor_beta') if hasattr(data, 'get') else None
    return beta if isinstance(beta, dict) else None


def enabled(config):
    """EFFECTIVE setting: actor_beta.diart_stream, default TRUE when actor_beta.enabled, false = off."""
    beta = _beta(config)
    if beta is None or beta.get('enabled') is not True:
        return False
    value = beta.get('diart_stream', True)
    if isinstance(value, str):
        return value.strip().lower() not in ('', '0', 'false', 'off', 'no')
    return value is not False and value != 0


def server_sha256():
    try:
        return hashlib.sha256(SERVER.read_bytes()).hexdigest()
    except OSError:
        return None


def fingerprint(config):
    """Value bound into config.engine_fingerprint(): None when off, else the settings and code that decide the speakers."""
    if not enabled(config):
        return None
    return {'version': VERSION, 'params': dict(PARAMS), 'rules': dict(RULES), 'server_sha256': server_sha256()}


def analyze_options(config, source_identity):
    """Extra keyword arguments for SpeakerClient.analyze: the file-artifact options, else the stream's source identity."""
    options = diart_turns.analyze_options(config, source_identity)
    if not options and enabled(config):
        value = (source_identity or {}).get('source_fingerprint')
        if value:
            options = {'source_fingerprint': str(value)}
    return options


def wsl_path(path):
    text = str(path)
    if text.startswith('/'):
        return text
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', text)
    if not match:
        raise DiartStreamUnavailable('path_not_on_a_drive', text)
    return '/mnt/%s/%s' % (match.group(1).lower(), match.group(2).replace('\\', '/'))


def build_command(config):
    """The wsl.exe command line that starts the sidecar (defaults = the held speaker environment)."""
    beta = _beta(config) or {}
    wsl = {**DEFAULT_WSL, **(beta.get('diart_wsl') if isinstance(beta.get('diart_wsl'), dict) else {})}
    models = wsl.get('models')
    if not models:
        try:
            models = config['diarization_model']
        except (KeyError, TypeError) as error:
            raise DiartStreamUnavailable('models_not_configured') from error
        resolver = getattr(config, 'resolve', None)
        models = resolver(models) if callable(resolver) else models
    script = wsl.get('script') or SERVER
    return ['wsl.exe', '-d', str(wsl['distro']), '--exec', '/usr/bin/env', 'PYTHONPATH=' + str(wsl['site_packages']),
            'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
            str(wsl['python']), '-u', wsl_path(script), '--models', wsl_path(models),
            '--params', json.dumps(PARAMS, separators=(',', ':'))]


# --------------------------------------------------------------------------------------------------------- turn math
def _merge_rows(rows):
    """Same-speaker rows that touch (gap <= MERGE_GAP) become one; time ordered."""
    merged = []
    for row in sorted(rows, key=lambda r: (r['speaker'], r['start'], r['end'])):
        last = merged[-1] if merged else None
        if last and last['speaker'] == row['speaker'] and row['start'] - last['end'] <= MERGE_GAP:
            last['end'] = max(last['end'], row['end'])
        else:
            merged.append(dict(row))
    merged.sort(key=lambda r: (r['start'], r['end'], r['speaker']))
    return merged


def splice(history, rows, cut):
    """``history`` with everything at/after ``cut`` replaced by ``rows`` (the sidecar's fresh answer from ``cut`` on)."""
    head, tail = [], []
    for row in history:
        (head if row['end'] <= cut - MERGE_GAP - 1e-9 else tail).append(row)
    tail = [dict(row, end=min(row['end'], cut)) for row in tail if row['start'] < cut - 1e-9]
    return sorted(head + _merge_rows(tail + [dict(row) for row in rows]), key=lambda r: (r['start'], r['end'], r['speaker']))


def window_turns(history, start, end):
    """The window view (turns, aligned, overlaps) with DIART_Dnnn labels, same clipping as DiartTurns.window."""
    turns = []
    for row in history:
        a, b = max(row['start'], start), min(row['end'], end)
        if b - a >= diart_turns.MIN_PIECE:
            turns.append({'start': round(a, 3), 'end': round(b, 3), 'speaker': LABEL_PREFIX + row['speaker']})
    turns.sort(key=lambda row: (row['start'], row['end'], row['speaker']))
    aligned, overlaps = diart_turns.split_exclusive(turns)
    return {'turns': turns, 'aligned': aligned, 'overlaps': overlaps}


def read_mono(path):
    """(mono float32 samples, sample rate) of one analysis window."""
    import numpy as np
    import soundfile as sf
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1), dtype=np.float32), rate


class _Mirror:
    """The manager's view of one stream held by the sidecar: where the fed audio ends and the turns known so far."""

    def __init__(self, origin):
        self.origin, self.fed, self.history, self.holes, self.history_start = origin, 0, [], [], origin

    @property
    def fed_until(self):
        return self.origin + self.fed / RATE


# --------------------------------------------------------------------------------------------------------- manager
class DiartStream:
    """Owns ONE lazily spawned sidecar process (below-normal priority) and the per-source streams it serves."""

    def __init__(self, config=None, *, command=None, call_timeout=None, start_wait=None, max_restarts=MAX_RESTARTS):
        beta = (_beta(config) or {}) if config is not None else {}
        wsl = beta.get('diart_wsl') if isinstance(beta.get('diart_wsl'), dict) else {}
        self._config, self._command = config, command
        self.call_timeout = float(call_timeout if call_timeout is not None else wsl.get('call_timeout_seconds', CALL_TIMEOUT))
        self.start_wait = float(start_wait if start_wait is not None else wsl.get('start_wait_seconds', START_WAIT))
        self.max_restarts = int(max_restarts)
        self._lock = threading.RLock()
        self._proc = self._replies = None
        self._ready, self._waited, self._spawns = False, False, 0
        self._disabled = None
        self._closed = False
        self._ids = 0
        self._streams = collections.OrderedDict()
        self._stderr = collections.deque(maxlen=20)
        self.info = {}
        self.last_timing = None
        self.events = collections.deque(maxlen=64)       # (kind, stream, detail) for diagnostics and tests
        atexit.register(self.close)

    # ---- process
    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self):
        command = self._command if self._command is not None else build_command(self._config)
        flags = (_BELOW_NORMAL_PRIORITY_CLASS | _CREATE_NO_WINDOW) if os.name == 'nt' else 0
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=flags)
        except (OSError, ValueError) as error:
            self._disabled = 'sidecar_not_startable'
            raise DiartStreamUnavailable('sidecar_not_startable', str(error)[:200]) from error
        self._proc, self._replies, self._ready, self._waited = proc, queue.Queue(), False, False
        self._spawns += 1
        self._streams.clear()
        threading.Thread(target=self._pump, args=(proc, self._replies), daemon=True, name='diart-stream-out').start()
        threading.Thread(target=self._drain, args=(proc,), daemon=True, name='diart-stream-err').start()

    def _pump(self, proc, replies):
        try:
            while True:
                frame = read_frame(proc.stdout)
                if frame is None:
                    break
                replies.put(frame)
        except (OSError, EOFError, ProtocolError, ValueError) as error:
            replies.put(({'event': 'protocol_error', 'detail': str(error)[:200]}, b''))
        finally:
            replies.put(None)

    def _drain(self, proc):
        try:
            for line in proc.stderr:
                text = line.decode('utf-8', 'replace').rstrip()
                self._stderr.append(text[:300])
                LOG.debug('diart sidecar: %s', text)
        except (OSError, ValueError):
            pass

    def _kill(self, proc=None):
        proc = proc or self._proc
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc is self._proc:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
            self._proc, self._ready = None, False
            self._streams.clear()

    def _tail(self):
        return ' | '.join(list(self._stderr)[-3:])[:300]

    def _ensure_ready(self):
        """Spawn if needed and wait for the ready frame: the call that spawned waits up to start_wait, later calls
        do not wait for a sidecar that is still loading (the dub never stalls on it)."""
        if self._closed:
            raise DiartStreamUnavailable('closed')
        if self._disabled:
            raise DiartStreamUnavailable(self._disabled)
        if not self._alive():
            if self._proc is not None:
                self._kill()
            if self._spawns > self.max_restarts:
                self._disabled = 'sidecar_crashed'
                raise DiartStreamUnavailable('sidecar_crashed', self._tail())
            self._spawn()
        if self._ready:
            return
        wait = 0.0 if self._waited else self.start_wait
        self._waited = True
        deadline = time.monotonic() + wait
        while not self._ready:
            try:
                item = self._replies.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise DiartStreamUnavailable('sidecar_starting') from None
            if item is None:
                self._kill()
                raise DiartStreamUnavailable('sidecar_exited', self._tail())
            header = item[0]
            if header.get('event') == 'ready':
                self._ready, self.info = True, header
            elif header.get('event') == 'fatal':
                self._disabled = 'sidecar_fatal'
                self._kill()
                raise DiartStreamUnavailable('sidecar_fatal', str(header.get('error'))[:300])

    def _call(self, header, payload=b''):
        self._ids += 1
        header = {**header, 'id': self._ids}
        proc = self._proc
        watchdog = threading.Timer(self.call_timeout, self._kill, args=(proc,))
        watchdog.daemon = True
        watchdog.start()
        try:
            write_frame(proc.stdin, header, payload)
        except (OSError, ValueError):
            self._kill(proc)
            raise DiartStreamUnavailable('sidecar_crashed', self._tail()) from None
        finally:
            watchdog.cancel()
        deadline = time.monotonic() + self.call_timeout
        while True:
            try:
                item = self._replies.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                self._kill(proc)
                raise DiartStreamUnavailable('timeout', '%.0f s' % self.call_timeout) from None
            if item is None:
                self._kill(proc)
                raise DiartStreamUnavailable('sidecar_crashed', self._tail())
            reply, body = item
            if reply.get('event') == 'protocol_error':
                self._kill(proc)
                raise DiartStreamUnavailable('protocol_error', str(reply.get('detail')))
            if reply.get('id') != header['id']:
                continue                                       # unsolicited or stale frame
            if reply.get('ok') is not True:
                raise DiartStreamUnavailable('sidecar_error', str(reply.get('error'))[:300])
            return reply, body

    # ---- public
    def close(self):
        """Stop the sidecar (analysis worker unload/idle). The process also exits on its own when our pipe closes."""
        with self._lock:
            self._closed = True
            proc = self._proc
            if proc is not None:
                try:
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass
                try:
                    proc.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                self._kill(proc)
            self._streams.clear()

    def ping(self):
        """The sidecar's own report (streams held, torch memory); waits like any other call."""
        with self._lock:
            self._ensure_ready()
            return self._call({'op': 'ping'})[0]

    def reset(self, stream_id):
        """Forget one stream (a seek or a new job); the speaker bank re-identifies from the next window."""
        with self._lock:
            if self._streams.pop(stream_id, None) is not None and self._alive() and self._ready:
                try:
                    self._call({'op': 'reset', 'stream': stream_id})
                except DiartStreamUnavailable:
                    pass

    def analyze_window(self, stream_id, wav_path, offset, embed):
        """(turns, aligned, vectors) for one analysis window, shaped like speakers.diarize()."""
        try:
            mono, rate = read_mono(wav_path)
        except Exception as error:                                 # noqa: BLE001 - unreadable window -> next provider
            raise DiartStreamUnavailable('window_unreadable', str(error)[:200]) from error
        if rate != RATE:
            raise DiartStreamUnavailable('window_rate_unsupported', str(rate))
        if not len(mono) or not math.isfinite(float(offset)):
            raise DiartStreamUnavailable('window_empty')
        start, end = float(offset), float(offset) + len(mono) / rate
        stream_id = str(stream_id or 'default')
        with self._lock:
            self._ensure_ready()
            mirror = self._streams.get(stream_id)
            if mirror is not None and self._must_restart(mirror, start, end):
                self.events.append(('reset', stream_id, '%.3f' % start))
                self.reset(stream_id)
                mirror = None
            if mirror is None:
                while len(self._streams) >= MAX_STREAMS:
                    self._streams.popitem(last=False)
                mirror = self._streams[stream_id] = _Mirror(start)
                created = True
            else:
                created = False
                self._streams.move_to_end(stream_id)
            if end > mirror.fed_until + EPS:
                try:
                    self._feed(stream_id, mirror, mono, start, created)
                except DiartStreamUnavailable:
                    self._streams.pop(stream_id, None)             # state unknown: the next window starts the stream over
                    raise
            window = window_turns(mirror.history, start, end)
        return window['turns'], window['aligned'], diart_turns.label_vectors(window, mono, start, embed)

    def _must_restart(self, mirror, start, end):
        if start < mirror.history_start - EPS or start - mirror.fed_until > SEEK_GAP:
            return True
        if end <= mirror.fed_until + EPS:                          # answered from stored turns: no unanalysed hole inside
            return any(min(b, end) - max(a, start) > 1. for a, b in mirror.holes)
        return False

    def _feed(self, stream_id, mirror, mono, start, created):
        import numpy as np
        fed = mirror.fed_until
        bridge = 0
        if created:
            new = mono
        else:
            skip = min(len(mono), max(0, int(round((fed - start) * RATE))))
            new = mono[skip:]
            if start - fed > EPS:                                  # a short gap: keep the timeline, fill it with silence
                bridge = int(round((start - fed) * RATE))
                new = np.concatenate([np.zeros(bridge, dtype=np.float32), new])
                mirror.holes.append((fed, start))
        payload = np.ascontiguousarray(new, dtype='<f4').tobytes()
        reply, _ = self._call({'op': 'feed', 'stream': stream_id, 'start': mirror.fed_until, 'n': len(new),
                               'create': bool(created)}, payload)
        self.last_timing = reply.get('timing')
        rows = reply.get('turns')
        cut = reply.get('from')
        if not isinstance(rows, list) or not isinstance(cut, (int, float)):
            raise DiartStreamUnavailable('protocol_error', 'feed reply')
        clean = []
        for row in rows:
            if not (isinstance(row, dict) and isinstance(row.get('speaker'), str) and diart_turns._ID.fullmatch(row['speaker'])
                    and all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in ('start', 'end'))):
                raise DiartStreamUnavailable('protocol_error', 'turn row')
            clean.append({'start': float(row['start']), 'end': float(row['end']), 'speaker': row['speaker']})
        mirror.fed += len(new)
        mirror.history = splice(mirror.history, clean, float(cut))
        floor = mirror.fed_until - HISTORY_SECONDS
        if floor > mirror.history_start:
            mirror.history = [row for row in mirror.history if row['end'] > floor]
            mirror.holes = [hole for hole in mirror.holes if hole[1] > floor]
            mirror.history_start = floor
        self.events.append(('feed', stream_id, '%d samples%s' % (len(new), ' (+%d silence)' % bridge if bridge else '')))
