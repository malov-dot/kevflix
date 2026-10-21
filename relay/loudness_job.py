"""Job-level playback gain: a bounded, cancellable, film-wide measurement of the SOURCE audio.

``compute_job_gain`` decodes a deterministic plan of short excerpts (default 16 x 20 s, spread over
the whole film, <= 45 s wall budget) of the ORIGINAL movie's selected audio track with FFmpeg, measures
BS.1770-4 integrated loudness with ``playback_loudness`` and turns it into ONE constant gain in dB for
the whole job (target -23 LUFS, clamp -10..+30 dB, >= 60 s of gated material or 0 dB).

Never raises for a measurement problem: no duration, a short film, an unreadable track, a decode failure
or too little gated audio all return ``status='unmeasured'`` with ``gain_db == 0.0`` and a ``reason``.
Only cancellation (``process.Cancelled``) propagates, and nothing is stored for a cancelled run, so
the caller can simply run it again (``resume`` keeps the excerpts already measured).

The result is a small JSON object. ``relay.tv`` stores it in the job's own payload under ``PAYLOAD_KEY``
(Store.update_job merges it atomically), keyed by ``job_key`` (source fingerprint + plan + target + track
+ layout), so every segment, seek, resume and server restart of that job reuses the identical gain.

Decode layout. The default ``layout='stereo'`` decodes with ``-ac 2``: the same FFmpeg downmix the app
uses for the dubbing stem (media.extract_audio) and for the stereo mux, so the number describes the
signal the dub mix is actually built from. ``layout='native'`` measures mono, stereo and 5.1 / 5.1(side)
with the BS.1770 channel weights (LFE excluded); any other layout falls back to the stereo downmix.
"""
from __future__ import annotations

import math
import os
import subprocess
import time

import numpy as np

from . import playback_loudness as pl
from .process import Cancelled

PAYLOAD_KEY = 'playback_loudness'
STATUS_MEASURED = 'measured'
STATUS_UNMEASURED = 'unmeasured'
RATE = 48000
MAX_CONSECUTIVE_DECODE_FAILURES = 3

#: Defaults of the ``tv.loudness`` config object. Only ``enabled`` has to be present; the code default is off.
DEFAULTS = {'enabled': False, 'target_lufs': pl.DEFAULT_TARGET_LUFS, 'dialogue_lufs': -22.0, 'samples': 16,
            'excerpt_seconds': 20.0, 'budget_seconds': 45.0, 'layout': 'stereo',
            'knee_dbfs': pl.DEFAULT_KNEE_DBFS, 'ceiling_dbfs': pl.DEFAULT_CEILING_DBFS}
_BOUNDS = {'target_lufs': (-40.0, -10.0), 'dialogue_lufs': (-35.0, -12.0), 'samples': (4, 48), 'excerpt_seconds': (5.0, 60.0),
           'budget_seconds': (5.0, 120.0), 'knee_dbfs': (-12.0, -1.5), 'ceiling_dbfs': (-6.0, -0.5)}


def _bounded(value, name):
    low, high = _BOUNDS[name]
    default = DEFAULTS[name]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if isinstance(value, bool) or not math.isfinite(number) or not low <= number <= high:
        return default
    return int(number) if name == 'samples' else number


def settings(raw=None):
    """Normalised ``tv.loudness`` settings. Invalid values fall back to the defaults; never raises."""
    raw = raw if isinstance(raw, dict) else {}
    result = {'enabled': raw.get('enabled') is True}
    for name in _BOUNDS:
        result[name] = _bounded(raw.get(name, DEFAULTS[name]), name)
    result['layout'] = raw.get('layout') if raw.get('layout') in ('stereo', 'native', 'center') else 'stereo'
    if result['ceiling_dbfs'] - result['knee_dbfs'] < 0.5:      # an unusable pair falls back to the default pair
        result['knee_dbfs'], result['ceiling_dbfs'] = DEFAULTS['knee_dbfs'], DEFAULTS['ceiling_dbfs']
    return result


def job_key(fingerprint, audio_index, duration, cfg):
    """``(key, excerpts, plan_receipt)`` for one job. The key covers everything that changes the number."""
    excerpts, plan = pl.sample_plan(duration, count=cfg['samples'], excerpt_seconds=cfg['excerpt_seconds'])
    key, _ = pl.persist_key(fingerprint, excerpts, cfg['target_lufs'], audio_index=audio_index, layout=cfg['layout'])
    return key, excerpts, plan


def valid_result(result):
    """True for a stored result this code can reuse."""
    if not isinstance(result, dict) or result.get('version') != pl.VERSION:
        return False
    if result.get('status') not in (STATUS_MEASURED, STATUS_UNMEASURED):
        return False
    gain = result.get('gain_db')
    return isinstance(gain, (int, float)) and not isinstance(gain, bool) and math.isfinite(gain)


def public_status(result, *, enabled=True, measuring=False, applied=None):
    """The small status object TVService.status() exposes (``loudness``)."""
    if not enabled:
        return {'enabled': False}
    if not valid_result(result):
        return {'enabled': True, 'status': 'measuring' if measuring else 'pending', 'gain_db': None}
    gain = float(result['gain_db'])
    return {'enabled': True, 'status': result['status'], 'reason': result.get('reason'), 'gain_db': gain,
            'measured_lufs': result.get('measured_lufs'), 'gated_seconds': result.get('gated_seconds'),
            'samples': result.get('samples'), 'target': result.get('target'),
            'clamped': result.get('clamped'),
            'applied': (abs(gain) >= 0.0005) if applied is None else bool(applied)}


# --- decoding --------------------------------------------------------------------------------

def _cancelled(cancel):
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    return bool(cancel.is_set())


def _check(owner, cancel):
    if owner is not None:
        owner.check()
    if _cancelled(cancel):
        raise Cancelled('Operation cancelled')


def _run(argv, *, owner, cancel, timeout):
    """Run one FFmpeg/FFprobe command and return stdout bytes. A ProcessOwner makes it cancellable by the
    app (owner.cancel() kills the child); otherwise ``cancel`` is polled every 0.25 s and the child killed."""
    if owner is not None:
        _check(owner, cancel)
        return owner.run(argv, timeout=timeout)
    flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    child = subprocess.Popen([str(a) for a in argv], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, **flags)
    started = time.monotonic()
    try:
        while True:
            if _cancelled(cancel):
                raise Cancelled('Operation cancelled')
            try:
                out, err = child.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - started > timeout:
                    raise TimeoutError('Loudness decode exceeded its allowed duration') from None
    except BaseException:
        child.kill()
        try:
            child.communicate(timeout=5)
        except Exception:
            pass
        raise
    if child.returncode:
        raise RuntimeError('%s exited %s: %s' % (os.path.basename(str(argv[0])), child.returncode,
                                                 err.decode('utf-8', 'replace')[-400:]))
    return out


def decode_layout(stream, layout):
    """``(channels, weights, ffmpeg -ac args, label)`` for one source stream (a probe stream dict or None)."""
    channels = int((stream or {}).get('channels') or 0)
    name = (stream or {}).get('channel_layout')
    if layout == 'center' and channels == 6 and name in ('5.1', '5.1(side)'):
        # Dialogue loudness: only the CENTER channel counts (EBU R 128 s4: for a 5.1 programme the dialogue basis is the center channel).
        return 6, pl.SURROUND_51_CENTER_WEIGHTS, [], name + '-center'
    if layout in ('native', 'center'):
        if channels == 1:
            return 1, pl.MONO_WEIGHTS, [], 'mono'
        if channels == 2:
            return 2, pl.STEREO_WEIGHTS, [], 'stereo'
        if channels == 6 and name in ('5.1', '5.1(side)'):
            return 6, pl.SURROUND_51_WEIGHTS, [], name
    return 2, pl.STEREO_WEIGHTS, ['-ac', '2'], 'stereo-downmix-s16'


def _decode(ffmpeg, source, audio_index, start, seconds, ac_args, channels, *, owner, cancel, timeout, runner):
    argv = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-ss', '%.3f' % start, '-t', '%.3f' % seconds,
            '-i', str(source), '-map', '0:%d' % int(audio_index) if audio_index is not None else '0:a:0',
            '-vn', '-sn', '-dn', *ac_args, '-ar', str(RATE), '-f', 's16le' if ac_args else 'f32le', '-']
    # A downmix is decoded to 16-bit PCM, exactly like media.extract_audio makes the engine's stereo: FFmpeg normalizes an
    # integer-output 5.1 downmix (x 1/(1+sqrt2), 7.7 dB down) but not a float one. Measuring a float downmix while the dub is
    # built from the normalized one made the job gain ~7 dB too small (owner heard a dub far quieter than the film).
    raw = runner(argv, owner=owner, cancel=cancel, timeout=timeout)
    if ac_args:
        samples = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
    else:
        samples = np.frombuffer(raw, dtype='<f4')
    frames = samples.size // channels
    data = samples[:frames * channels].astype(np.float64).reshape(frames, channels)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _probe(ffprobe, source, audio_index, *, owner, cancel, runner):
    import json
    raw = runner([ffprobe, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(source)],
                 owner=owner, cancel=cancel, timeout=60)
    data = json.loads(raw)
    duration = float(data.get('format', {}).get('duration', 0))
    audio = [s for s in data.get('streams', []) if s.get('codec_type') == 'audio']
    stream = next((s for s in audio if s.get('index') == audio_index), None) if audio_index is not None \
        else (audio[0] if audio else None)
    return duration, stream


# --- the job gain ----------------------------------------------------------------------------

def compute_job_gain(source_path, ffmpeg, ffprobe=None, *, audio_index=None, stream=None, duration=None,
                     owner=None, cancel=None, budget_seconds=45.0, samples=16, excerpt_seconds=20.0,
                     target_lufs=pl.DEFAULT_TARGET_LUFS, layout='stereo', key=None,
                     min_gated_seconds=pl.DEFAULT_MIN_GATED_SECONDS, max_gain_db=pl.DEFAULT_MAX_GAIN_DB,
                     min_gain_db=pl.DEFAULT_MIN_GAIN_DB, resume=None, runner=None, clock=time.monotonic):
    """One constant job gain from a bounded film-wide sample of the SOURCE audio.

    ``source_path``/``audio_index``: the original movie and its selected audio stream (None: first audio).
    ``stream``/``duration``: the probed stream dict and film length when the caller has them; otherwise
    ``ffprobe`` is run once. ``owner``: a ``process.ProcessOwner`` (preferred; its cancel kills FFmpeg);
    ``cancel``: an Event-like or callable polled between and during decodes. ``budget_seconds``: wall
    budget for the whole call. ``resume``: a dict (memory only) that keeps already measured excerpts so a
    cancelled run continues instead of restarting; it is reset when ``key`` changes.

    Returns ``{'gain_db', 'measured_lufs', 'gated_seconds', 'samples', 'status', 'target', ...}``;
    ``status`` is ``'measured'`` or ``'unmeasured'`` (``gain_db`` 0.0 and a ``reason``). Raises only
    ``Cancelled``.
    """
    runner = runner or _run
    started = clock()
    deadline = started + float(budget_seconds)
    result = {'version': pl.VERSION, 'status': STATUS_UNMEASURED, 'reason': None, 'gain_db': 0.0,
              'measured_lufs': None, 'gated_seconds': 0.0, 'samples': 0, 'planned_samples': 0,
              'target': float(target_lufs), 'clamped': False, 'clamp': None, 'raw_gain_db': None,
              'layout': None, 'channels': None, 'audio_index': audio_index,
              'excerpt_seconds': float(excerpt_seconds), 'budget_seconds': float(budget_seconds),
              'budget_exhausted': False, 'partial': False, 'elapsed_seconds': 0.0, 'errors': [], 'key': key,
              'standard': pl.STANDARD, 'scope': 'one constant gain for the whole dub mix of this job'}

    def finish(reason):
        result['reason'] = result['reason'] or reason
        result['elapsed_seconds'] = round(clock() - started, 3)
        return result

    phase = 'probe'
    try:
        _check(owner, cancel)
        if duration is None or stream is None:
            if ffprobe is None and duration is None:
                return finish('no-duration')
            if ffprobe is not None:
                probed_duration, probed_stream = _probe(ffprobe, source_path, audio_index,
                                                        owner=owner, cancel=cancel, runner=runner)
                duration = probed_duration if duration is None else duration
                stream = probed_stream if stream is None else stream
        phase = 'measure'
        duration = float(duration or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            return finish('no-duration')
        channels, weights, ac_args, label = decode_layout(stream, layout)
        result.update(layout=label, channels=channels)
        excerpts, plan = pl.sample_plan(duration, count=samples, excerpt_seconds=excerpt_seconds)
        result['planned_samples'] = len(excerpts)
        if plan['total_seconds'] < float(min_gated_seconds):
            return finish('short-film')
        state = resume if isinstance(resume, dict) else {}
        if state.get('key') != key or 'energy' not in state:
            state.clear()
            state.update(key=key, energy={})
        energy = state['energy']
        failures = 0
        for index in pl.visit_order(len(excerpts)):
            if index in energy:
                continue
            _check(owner, cancel)
            remaining = deadline - clock()
            if remaining <= 0.5:
                result['budget_exhausted'] = True
                break
            start, end = excerpts[index]
            try:
                data = _decode(ffmpeg, source_path, audio_index, start, end - start, ac_args, channels,
                               owner=owner, cancel=cancel, timeout=max(1.0, min(remaining, 30.0)), runner=runner)
                energy[index] = pl.block_energies(data, RATE, weights)
                failures = 0
            except Cancelled:
                raise
            except TimeoutError:
                result['budget_exhausted'] = True
                break
            except Exception as error:   # unreadable track, EOF, FFmpeg failure: skip this excerpt
                failures += 1
                if len(result['errors']) < 3:
                    result['errors'].append('%s: %s' % (type(error).__name__, str(error)[:160]))
                if failures >= MAX_CONSECUTIVE_DECODE_FAILURES:
                    break
        parts = [energy[i] for i in sorted(energy) if energy[i].size]
        result['samples'] = len(parts)
        lufs, gated_seconds, _ = pl.integrate(parts, RATE)
        gain, receipt = pl.choose_job_gain(lufs, gated_seconds, target_lufs=target_lufs, max_gain_db=max_gain_db,
                                           min_gain_db=min_gain_db, min_gated_seconds=min_gated_seconds)
        result.update(measured_lufs=None if lufs is None else round(lufs, 3), gated_seconds=round(gated_seconds, 3),
                      raw_gain_db=None if receipt['raw_gain_db'] is None else round(receipt['raw_gain_db'], 3),
                      clamped=receipt['clamped'], clamp=receipt['clamp'])
        if receipt['reason'] == 'insufficient_gated_audio':
            if not parts and result['errors']:
                return finish('decode-failed')
            return finish('budget-exhausted' if result['budget_exhausted'] else 'insufficient_gated_audio')
        result.update(status=STATUS_MEASURED, gain_db=round(gain, 3), reason=receipt['reason'],
                      partial=result['samples'] < len(excerpts))
        return finish(receipt['reason'])
    except Cancelled:
        raise
    except Exception as error:       # a measurement problem is never a playback problem
        if len(result['errors']) < 3:
            result['errors'].append('%s: %s' % (type(error).__name__, str(error)[:160]))
        result.update(status=STATUS_UNMEASURED, gain_db=0.0)
        return finish('probe-failed' if phase == 'probe' else 'error')
