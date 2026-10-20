"""Fit one generated dubbing take into its dialogue slot with ONE pitch-preserving ffmpeg ``atempo``.

This is the deterministic, receipted equivalent of what the owner's approved demo did (step4_clone_mix.py):
a take is sped up only when it is longer than its slot, by ``min(max_speed, take_seconds / slot_seconds)``, in a
single ``atempo`` pass over the untouched generated PCM. A take that fits is never touched. A take that still does
not fit after the 1.35x cap keeps the remainder as ``overrun_seconds`` in the receipt: the overrun is REPORTED, never
hidden and never silently trimmed. What the pipeline then does with a reported overrun (keep it, spilling into the
quiet gap before the next cue, or fall back to original audio) is the caller's policy, not this module's.

Rubber Band, repeated passes, re-synthesis and truncation are deliberately not used.

One measured difference from the bare demo formula: atempo emits whole filter frames, so its output is 0.3-20 ms longer
(occasionally a few ms shorter) than take/factor. Aimed exactly at the slot, 55 of 60 synthetic fits ended a few
milliseconds over and would have been reported as overruns by the scheduler; aimed ``margin_seconds`` (default 0.03)
short of the slot, 1 of 160 did (a take already clamped at 1.35x). ``margin_seconds=0`` restores the plain formula.

Sample rate: takes are 48000 Hz mono float32 (the Vox actor sink and generic_dub both produce exactly that).
Any other rate raises ValueError. This module never resamples, because a silent resample would change the
gate-verified PCM and the receipt hashes; the caller must resample explicitly (and say so) before fitting.

Errors use the project's existing types: ValueError for rejected input (as media.py / actor_reference.py),
RuntimeError for an ffmpeg failure or unusable ffmpeg output (as ProcessOwner.run), TimeoutError / Cancelled from the
supplied runner are passed through untouched.

``runner`` is any callable ``runner(argv, *, timeout) -> object`` and is call-compatible with
``relay.process.ProcessOwner.run`` (so ``runner=owner.run`` keeps the fit cancellable with the rest of the job).
Without a runner a plain ``subprocess.run`` is used.
"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

FIT_VERSION = 'take-fit-atempo-v1'
RATE = 48000
DEFAULT_MAX_SPEED = 1.35
MAX_SPEED_CEILING = 1.6           # hard ceiling; the default stays 1.35 and actor_beta.take_fit_max_speed (1.0-1.6) is the owner knob
MAX_TAKE_SECONDS = 120.0          # one dialogue line; anything longer is a caller bug, not a take
OVERRUN_TOLERANCE_SECONDS = .0001  # the tolerance timing.schedule_lines already applies to its overrun decision
DEFAULT_MARGIN_SECONDS = .03       # aim this far inside the slot: atempo's output frames end up to ~20 ms long
MAX_MARGIN_SECONDS = .1
_LENGTH_SLACK_SECONDS = .1        # atempo frames its output; allow this plus 3 % before calling the output corrupt


def slot_seconds(cue_start, next_cue_start, window_end, guard=.05, floor=.3):
    """Seconds a take may occupy: from this cue's start to ``guard`` before the next cue's start.

    Mirrors the approved demo: ``slot = max(0.3, next_cue_start - 0.05 - this_start)``; the last cue (no next cue)
    runs to the window end. ``next_cue_start`` may be None for the last cue. When both a next cue and a window end
    are given the slot never extends past the window end. ``floor`` keeps a degenerate gap from demanding an
    impossible fit; it is never larger than what the caller allows."""
    start = _finite(cue_start, 'cue_start')
    guard = _finite(guard, 'guard')
    floor = _finite(floor, 'floor')
    if guard < 0 or floor < 0:
        raise ValueError('guard and floor must not be negative')
    end = None if window_end is None else _finite(window_end, 'window_end')
    if next_cue_start is None:
        if end is None:
            raise ValueError('The last cue needs a window_end')
        limit = end
    else:
        limit = _finite(next_cue_start, 'next_cue_start') - guard
        if end is not None:
            limit = min(limit, end)
    return max(floor, limit - start)


def pcm_sha256(samples):
    """sha256 of the little-endian float32 PCM bytes (the same bytes the receipt hashes); deterministic."""
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(samples, dtype='<f4').tobytes()).hexdigest()


def fit_take(samples, rate, slot_seconds, *, max_speed=DEFAULT_MAX_SPEED, min_speed=1.0, ffmpeg, runner=None,
             workdir=None, margin_seconds=DEFAULT_MARGIN_SECONDS):
    """Return ``(fitted_samples, receipt)``.

    ``samples``: finite mono float32 PCM at 48000 Hz (1-D, or (n, 1)). ``slot_seconds``: finite positive seconds the
    take may occupy. The take is sped up ONLY when it is longer than the slot (beyond the 0.1 ms tolerance of
    timing.schedule_lines): factor = clamp(take / aim, min_speed, max_speed) where aim = slot - margin (see below).
    ``min_speed`` (>= 1.0, default 1.0) is the lowest tempo ever applied when a fit is needed; this module never
    slows a take down. ``max_speed`` is at most 1.35. A take that fits is returned unchanged (a copy), with factor
    1.0 and no ffmpeg call.

    ``margin_seconds`` (default 0.03, 0..0.1): ffmpeg's atempo emits whole filter frames, so its output is
    0.3-20 ms (measured, ffmpeg 9.0.2, noise takes 0.4-6 s at 1.05-1.35x) LONGER than take/factor. Aiming exactly at the
    slot therefore leaves a few milliseconds of overrun on almost every fit, which the scheduler (0.1 ms tolerance)
    would report as ``speech_timing_overrun``. The tempo is aimed ``margin_seconds`` (capped at a quarter of the slot)
    short of the slot so the frame tail lands inside it. ``margin_seconds=0`` gives the plain demo formula
    ``min(max_speed, take / slot)``. The margin never causes a speed-up of a take that already fits. Near the cap the
    aim can exceed ``max_speed``; the tempo is then the cap and ``clamped`` is True.

    The receipt always contains: version, method ('atempo'), applied, factor, requested_factor (before / aim), clamped,
    before_seconds, after_seconds, slot_seconds, aim_seconds, margin_seconds (effective), tail_seconds (measured
    after - before/factor; 0.0 when not applied), overrun_seconds, overrun, max_speed, min_speed, rate,
    input_samples, output_samples, input_sha256, output_sha256 (float32 LE PCM bytes). ``overrun_seconds`` is
    how far the (fitted) take still exceeds the slot (0.0 within tolerance).

    ``workdir``: optional parent directory for the temporary directory (default: the system temp). The temporary
    directory is always removed in a ``finally``."""
    audio = _validated_take(samples, rate)
    slot = _finite(slot_seconds, 'slot_seconds')
    if slot <= 0:
        raise ValueError('slot_seconds must be positive')
    fastest = _finite(max_speed, 'max_speed')
    slowest = _finite(min_speed, 'min_speed')
    if not 1.0 <= slowest <= fastest <= MAX_SPEED_CEILING:
        raise ValueError('Speeds must satisfy 1.0 <= min_speed <= max_speed <= %g' % MAX_SPEED_CEILING)
    margin = _finite(margin_seconds, 'margin_seconds')
    if not 0 <= margin <= MAX_MARGIN_SECONDS:
        raise ValueError('margin_seconds must be between 0 and %g' % MAX_MARGIN_SECONDS)
    if ffmpeg is None or not str(ffmpeg).strip():
        raise ValueError('ffmpeg executable path is required')
    margin = min(margin, slot / 4)
    aim = slot - margin
    count = len(audio)
    before = count / RATE
    receipt = {'version': FIT_VERSION, 'method': 'atempo', 'applied': False, 'factor': 1.0,
               'requested_factor': 1.0, 'clamped': False, 'before_seconds': before,
               'after_seconds': before, 'slot_seconds': slot, 'aim_seconds': aim, 'margin_seconds': margin,
               'tail_seconds': 0.0, 'overrun_seconds': 0.0, 'overrun': False,
               'max_speed': fastest, 'min_speed': slowest, 'rate': RATE, 'input_samples': count,
               'output_samples': count, 'input_sha256': pcm_sha256(audio), 'output_sha256': None}
    if before <= slot + OVERRUN_TOLERANCE_SECONDS:
        receipt['output_sha256'] = receipt['input_sha256']
        return audio.copy(), receipt
    requested = before / aim
    factor = min(fastest, max(slowest, requested))
    receipt['requested_factor'] = requested
    receipt['clamped'] = requested > fastest
    if factor <= 1.0:
        # max_speed == 1.0 disables fitting: nothing is changed and the overrun is reported honestly.
        receipt.update(overrun_seconds=before - slot, overrun=True, output_sha256=receipt['input_sha256'])
        return audio.copy(), receipt
    fitted = _atempo(audio, factor, ffmpeg=ffmpeg, runner=runner, workdir=workdir)
    after = len(fitted) / RATE
    over = after - slot
    receipt.update(applied=True, factor=factor, after_seconds=after, output_samples=len(fitted),
                   tail_seconds=after - before / factor,
                   overrun_seconds=over if over > OVERRUN_TOLERANCE_SECONDS else 0.0,
                   overrun=over > OVERRUN_TOLERANCE_SECONDS, output_sha256=pcm_sha256(fitted))
    return fitted, receipt


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError('%s must be a number' % name)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('%s must be a finite number' % name) from error
    if not math.isfinite(number):
        raise ValueError('%s must be a finite number' % name)
    return number


def _validated_take(samples, rate):
    import numpy as np
    if isinstance(rate, bool):
        raise ValueError('Take sample rate must be %d Hz' % RATE)
    try:
        exact = float(rate) == RATE
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact:
        raise ValueError('Take sample rate must be %d Hz (resample explicitly before fitting)' % RATE)
    audio = np.asarray(samples)
    if audio.dtype != np.float32:
        raise ValueError('Take must be float32 PCM, got %s' % audio.dtype)
    if audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio[:, 0]
    if audio.ndim != 1:
        raise ValueError('Take must be mono audio')
    if not audio.size:
        raise ValueError('Take is empty')
    if audio.size / RATE > MAX_TAKE_SECONDS:
        raise ValueError('Take is longer than %g seconds' % MAX_TAKE_SECONDS)
    if not np.all(np.isfinite(audio)):
        raise ValueError('Take contains non-finite audio')
    return np.ascontiguousarray(audio)


def _default_run(argv, *, timeout):
    """Same contract and error shapes as relay.process.ProcessOwner.run, without ownership."""
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    try:
        done = subprocess.run([str(x) for x in argv], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError('Owned process exceeded its allowed duration') from error
    except OSError as error:
        raise RuntimeError('%s could not be started: %s' % (os.path.basename(str(argv[0])), error)) from error
    if done.returncode:
        raise RuntimeError('%s exited %d: %s' % (os.path.basename(str(argv[0])), done.returncode,
                                                 done.stderr.decode('utf-8', 'replace')[-4000:]))
    return done.stdout


def _atempo(audio, factor, *, ffmpeg, runner, workdir):
    import numpy as np
    import soundfile as sf
    if runner is None and not Path(str(ffmpeg)).is_file():
        raise RuntimeError('ffmpeg executable not found: %s' % ffmpeg)
    run = runner or _default_run
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='kevflix-take-fit-', dir=None if workdir is None else str(workdir)))
    try:
        source, target = directory / 'take-in.wav', directory / 'take-out.wav'
        sf.write(str(source), audio, RATE, subtype='FLOAT')
        argv = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', str(source),
                '-af', 'atempo=%.9f' % factor, '-ar', str(RATE), '-ac', '1', '-c:a', 'pcm_f32le', str(target)]
        try:
            run(argv, timeout=60 + 4 * len(audio) / RATE)
        except TimeoutError:
            raise
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError('ffmpeg atempo failed: %s' % error) from error
        if not target.is_file():
            raise RuntimeError('ffmpeg atempo produced no output')
        try:
            fitted, rate = sf.read(str(target), dtype='float32', always_2d=True)
        except (RuntimeError, OSError, ValueError) as error:
            raise RuntimeError('ffmpeg atempo output is unreadable: %s' % error) from error
        if rate != RATE or fitted.shape[1] != 1 or not len(fitted):
            raise RuntimeError('ffmpeg atempo output is not 48000 Hz mono audio')
        fitted = np.ascontiguousarray(fitted[:, 0])
        if not np.all(np.isfinite(fitted)):
            raise RuntimeError('ffmpeg atempo output contains non-finite audio')
        expected = len(audio) / factor
        if abs(len(fitted) - expected) > _LENGTH_SLACK_SECONDS * RATE + .03 * expected:
            raise RuntimeError('ffmpeg atempo output length %d is implausible for factor %.4f (expected about %d)'
                               % (len(fitted), factor, round(expected)))
        return fitted
    finally:
        shutil.rmtree(directory, ignore_errors=True)
