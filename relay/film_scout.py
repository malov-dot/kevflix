"""Film scout (F15, 2026-11-11): feeds the film's dialogue channel to a second Diart stream ahead of playback and keeps the
film-speaker map (relay/film_speakers.py) current.

One FilmScout per film. It owns no model and no process: every outside step is a callable given to it, so it runs the same
way in the app (ffmpeg + the analysis worker's Diart sidecar) and in a unit test (fakes).

    read_block(start, seconds) -> mono float32 samples at 16 kHz (a sequence; shorter at the film's end; empty = end of film)
    feed(samples, start)       -> {'rows': [{'start','end','speaker'}], 'from': seconds, 'committed_until': seconds,
                                    'created': bool}  (DiartStream.feed_ahead: the sidecar's fresh answer from 'from' on,
                                   scout labels 'Dnnn'; 'created' = the sidecar had to start the stream for this block)
    centers()                  -> {label: [float, ...]}  (DiartStream.centers: the clusterer's centre per label)
    pitch(samples)             -> median pitch in Hz of voiced speech, or None

Pacing (the card is shared and the dub comes first): one block per tick(); tick() does nothing while ``busy()`` says a window
of the dub is waiting, and after a block that took t seconds the scout rests t * (1 - DUTY) / DUTY seconds. Any failure of a
callable ends the scout quietly (``stopped`` names the reason): the map then simply stops growing and later windows behave
as they did before the map existed.
"""
from __future__ import annotations

import time

from . import film_speakers

RATE = 16000
BLOCK_SECONDS = 30.0          # audio fed per step
REBUILD_EVERY_BLOCKS = 10     # the map is rebuilt every 5 minutes of film (and at the end)
DUTY = .5                     # share of wall time the scout may keep the sidecar busy
MIN_PITCH_SECONDS = .6        # a turn at least this long, with nobody else speaking, gives one pitch sample
PITCH_SAMPLES_PER_LABEL = 200
KEEP_SECONDS = 90.0           # audio kept for pitch (turns are measured once they are final, a few seconds after they are fed)


def merge_turns(history, rows, cut, gap=.05):
    """``history`` ([(start, end, label)]) with everything at or after ``cut`` replaced by ``rows``; same-label neighbours joined."""
    kept = [(a, min(b, cut), label) for a, b, label in history if a < cut - 1e-9]
    fresh = [(float(r['start']), float(r['end']), str(r['speaker'])) for r in rows]
    merged = []
    for a, b, label in sorted(kept + fresh, key=lambda row: (row[2], row[0], row[1])):
        if merged and merged[-1][2] == label and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), label)
        elif b > a:
            merged.append((a, b, label))
    return sorted(merged)


def default_pitch(samples):
    """Median pitch of one stretch of speech with the app's own estimator (relay.audio_evidence, numpy, CPU)."""
    from . import audio_evidence
    try:
        import numpy as np
        return audio_evidence.cue_features(np.asarray(samples, dtype=np.float32), RATE).get('f0_median_hz')
    except Exception:  # noqa: BLE001 - a pitch that cannot be measured is simply unknown
        return None


class FilmScout:
    def __init__(self, artifact_path, duration, *, read_block, feed, centers, pitch=default_pitch, busy=lambda: False,
                 clock=time.monotonic, start=0.0, block_seconds=BLOCK_SECONDS, rebuild_every=REBUILD_EVERY_BLOCKS, duty=DUTY,
                 thresholds=None):
        self.path, self.duration = artifact_path, float(duration)
        self.read_block, self.feed, self.centers, self.pitch, self.busy, self.clock = read_block, feed, centers, pitch, busy, clock
        self.block_seconds, self.rebuild_every, self.duty = float(block_seconds), int(rebuild_every), float(duty)
        self.thresholds = dict(thresholds or {})
        self.position = float(start)          # film seconds fed so far
        self.committed = float(start)         # film seconds whose turns are final
        self.turns = []                       # [(start, end, label)] of the whole scout stream
        self.pitches = {}                     # label -> [Hz]
        self.measured_until = float(start)
        self.audio, self.audio_start = [], float(start)
        self.blocks = 0
        self.rest_until = 0.0
        self.stopped = None                   # None while running; 'done' or the reason it ended
        self.artifact = None                  # a scout always starts a new Diart stream: older labels mean nothing to it

    # ---- one step
    def tick(self):
        """Feed at most one block. True when a block was fed. Never raises."""
        if self.stopped or self.busy() or self.clock() < self.rest_until:
            return False
        if self.position >= self.duration - 1e-3:
            return self._finish('done')
        began = self.clock()
        try:
            samples = self.read_block(self.position, min(self.block_seconds, self.duration - self.position))
            if samples is None or len(samples) == 0:
                return self._finish('done')
            reply = self.feed(samples, self.position)
            if reply.get('created') and self.blocks:
                # the sidecar was restarted: its labels start again at D000 and no longer mean what the map says
                self.stopped = 'stream_restarted'
                return False
            self.turns = merge_turns(self.turns, reply['rows'], float(reply['from']))
            self.committed = max(self.committed, float(reply['committed_until']))
            self._remember(samples)
            self.position += len(samples) / RATE
            self._measure_pitch()
            self.blocks += 1
            if self.blocks % self.rebuild_every == 0:
                self.rebuild()
        except Exception as error:  # noqa: BLE001 - the scout is an extra: it ends, the dub goes on as before
            self.stopped = 'failed:%s' % type(error).__name__
            return False
        spent = self.clock() - began
        self.rest_until = self.clock() + spent * (1.0 - self.duty) / max(self.duty, 1e-3)
        return True

    def _finish(self, reason):
        try:
            self.committed = max(self.committed, self.position)      # nothing more will come: the tail is final
            self._measure_pitch()
            self.rebuild()
            self.stopped = reason
        except Exception as error:  # noqa: BLE001
            self.stopped = 'failed:%s' % type(error).__name__
        return False

    # ---- pitch of final, non-overlapped turns
    def _remember(self, samples):
        self.audio.extend(samples)
        extra = len(self.audio) - int(KEEP_SECONDS * RATE)
        if extra > 0:
            del self.audio[:extra]
            self.audio_start += extra / RATE

    def _measure_pitch(self):
        final = [row for row in self.turns if self.measured_until <= row[1] <= self.committed]
        for a, b, label in final:
            if b - a < MIN_PITCH_SECONDS or a < self.audio_start:
                continue
            if any(other != label and c < b and d > a for c, d, other in self.turns):
                continue                                              # somebody else speaks during it
            bucket = self.pitches.setdefault(label, [])
            if len(bucket) >= PITCH_SAMPLES_PER_LABEL:
                continue
            i, j = int((a - self.audio_start) * RATE), int((b - self.audio_start) * RATE)
            value = self.pitch(self.audio[i:j])
            if value:
                bucket.append(float(value))
        self.measured_until = max(self.measured_until, self.committed)

    # ---- the map
    def prints(self):
        """{label: {'centre', 'seconds', 'pitch_hz', 'first'}} of every label that has a centre."""
        centres = self.centers() or {}
        out = {}
        for a, b, label in self.turns:
            if label not in centres:
                continue
            entry = out.setdefault(label, {'centre': list(centres[label]), 'seconds': 0.0, 'pitch_hz': None, 'first': a})
            entry['seconds'] += b - a
            entry['first'] = min(entry['first'], a)
        for label, entry in out.items():
            values = sorted(self.pitches.get(label) or ())
            if len(values) >= 5:
                entry['pitch_hz'] = round(values[len(values) // 2], 1)
            entry['seconds'] = round(entry['seconds'], 2)
        return out

    def rebuild(self):
        """Consolidate what is known and write the artifact. Only final turns are published (covered_until = committed)."""
        final = [(a, min(b, self.committed), label) for a, b, label in self.turns if a < self.committed]
        self.artifact = film_speakers.build(self.prints(), final, self.committed, self.artifact, **self.thresholds)
        on_disk = film_speakers.load(self.path)
        if not on_disk or float(on_disk.get('covered_until') or 0.0) <= self.committed + 1e-6:
            film_speakers.save(self.path, self.artifact)              # a map that covers more of the film is never replaced by less
        return self.artifact


def ffmpeg_block_reader(ffmpeg, source, audio_index, centre):
    """read_block for a local file: the dialogue channel (front centre when the track has one, else a mono downmix) of
    [start, start + seconds) as 16 kHz mono float32, decoded by ffmpeg on the CPU at below-normal priority."""
    def read(start, seconds):
        import os
        import subprocess
        import numpy as np
        command = [str(ffmpeg), '-nostdin', '-v', 'error', '-ss', '%.3f' % start, '-t', '%.3f' % seconds, '-i', str(source),
                   '-map', '0:%d' % int(audio_index), '-vn', '-sn'] + (['-af', 'pan=mono|c0=FC'] if centre else ['-ac', '1']) + [
                   '-ar', str(RATE), '-f', 'f32le', '-']
        flags = (0x00004000 | 0x08000000) if os.name == 'nt' else 0        # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
        done = subprocess.run(command, capture_output=True, timeout=120, creationflags=flags)
        if done.returncode != 0:
            raise RuntimeError('ffmpeg: ' + done.stderr[-200:].decode('utf-8', 'replace'))
        return np.frombuffer(done.stdout, dtype='<f4')
    return read


def has_centre(stream):
    """True when the audio stream (an ffprobe stream dict) has a front-centre channel."""
    layout = str((stream or {}).get('channel_layout') or '').lower()
    return any(layout.startswith(name) for name in ('3.0', '3.1', '4.0', '4.1', '5.0', '5.1', '6.0', '6.1', '7.0', '7.1'))


class ScoutHost:
    """One daemon thread that runs the scout of the film being dubbed.

    It never starts a model: a block is fed only while the analysis worker of the dub is already warm
    (``analysis.scout_client()`` returns a client) and ``busy()`` says no window of the dub is waiting. A film whose scout
    ended (done, failed, sidecar restarted) is not scouted again in this run of the app."""

    def __init__(self, config, store, analysis, *, busy, terminal=('complete', 'error', 'cancelled', 'needs_review'), poll_seconds=.25):
        import threading
        self.config, self.store, self.analysis, self.busy, self.terminal, self.poll = config, store, analysis, busy, terminal, poll_seconds
        self._stop, self._thread = threading.Event(), None
        self._job_id = self._scout = self._client = None
        self.finished = {}                    # job id -> why its scout ended

    def start(self):
        import threading
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name='film-scout', daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.poll):
            try:
                self.step()
            except Exception:  # noqa: BLE001 - the scout is an extra and never disturbs the dub
                pass

    def _beta(self):
        data = getattr(self.config, 'data', self.config)
        beta = data.get('actor_beta') if hasattr(data, 'get') else None
        return beta if isinstance(beta, dict) else None

    def _job(self):
        for job in self.store.jobs():                                  # newest first
            if job.get('state') not in self.terminal:
                return job
        return None

    def step(self):
        """One poll: pick the film, build its scout once, feed at most one block. True when a block was fed."""
        from pathlib import Path
        if not film_speakers.enabled(self._beta()):
            return False
        job = self._job()
        if job is None or job['id'] in self.finished:
            return False
        client = self.analysis.scout_client()
        if client is None:
            return False
        self._client = client
        if self._job_id != job['id']:
            payload = job.get('payload') or {}
            fingerprint = str(payload.get('source_fingerprint') or job['id'])
            stream = next((s for s in ((payload.get('media') or {}).get('streams') or ())
                           if s.get('index') == payload.get('audio_index')), None)
            self._scout = FilmScout(
                film_speakers.path_for(Path(self.config.cache) / str(job['id'])), float(payload.get('duration') or 0.0),
                read_block=ffmpeg_block_reader(self.config['ffmpeg'], payload['source'], payload['audio_index'], has_centre(stream)),
                feed=lambda samples, start: self._client.scout_feed(samples, start, fingerprint),
                centers=lambda: self._client.scout_centers(fingerprint), busy=self.busy)
            self._job_id = job['id']
        fed = self._scout.tick()
        if self._scout.stopped:
            self.finished[job['id']] = self._scout.stopped
        return fed
