"""Movie-Vox engine process: one owned VoxCPM2 Q4_K actor session for KevFLIX movie dubbing.

This is NOT Sem's live Vox worker: it loads the movie actor model (models/actor-voxcpm2, Q4_K + AudioVAE F16) through
the pinned relay/actor_voice.py ActorVoice binding, in its own process with its own CUDA session. It never connects to
Sem's speech worker, never reads Sem's reference and never opens Sem's status.json.

Run by vox_engine_host.VoxEngineHost only (python -I -B; spawned inside the lease v2 Job or the host's own
kill-on-close Job, see process_owner.py):
  vox_movie_child.py --relay-root R --manifest M --budget B --capacity C --idle-seconds N --actor-voice-sha256 H
                     --sibling-pins JSON --parent-pid P [--cancel-event NAME] [--hard-cancel-ms 60]
                     [--reserve-ram-mib N] [--gpu-sustain-seconds 1.5]

Loaded code is exactly the verified bytes (review P3-4): approved.py and child_watchdog.py are exec'd from bytes whose
sha256 equals --sibling-pins; relay/actor_voice.py and relay/resource_budget.py are exec'd from bytes whose sha256
equals approved.ACTOR_VOICE_SHA256 / RESOURCE_BUDGET_SHA256, into a private `relay` package with an empty __path__
(any other relay import fails). No .pyc is used.

Watchdogs (child_watchdog.py, kernel waits on child threads; none needs the holder's GIL):
  parent process exit -> exit 70 (started before the model load); lease cancel event set -> exit 71;
  a cancel marker unacknowledged for --hard-cancel-ms while a request runs -> exit 72.

Protocol vox-movie-child-v2 (JSON lines; requests on stdin, responses on stdout):
  -> ready     {"ready": true, "protocol", "pid", "model_sha256", "quantization", "backend", "device", "threads",
                "capacity_seconds", "encoder_sample_capacity", "actor_voice_sha256", "settings", "load_seconds", "vram",
                "watchdogs"}
  <- {"action": "synthesize", "id", "text", "reference", "reference_sha256", "reference_text", "seed", "output"}
  -> {"id", "receipt", "seconds", "output"}            take written to `output` (48 kHz FLOAT); receipt.reference_sha256
                                                       is the hash of the exact bytes the engine read
  -> {"id", "cancelled": true, "kind": "InterruptedError", "first_pcm_seconds"}   marker acknowledged; stays warm
  -> {"id"?, "error", "kind": "ResourceYield"}          sustained GPU/RAM shortfall or hold; the process exits
  -> {"id", "error", "kind": "ReferenceUnverified"}     reference bytes != reference_sha256; the process exits
  -> {"id", "error", "kind", "first_pcm_seconds"}       any other failure; the process exits
  <- {"action": "close"}  -> {"event": "closed"}
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

MAX_LINE = 65536
HERE = Path(__file__).resolve().parent


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def exec_verified(name, path, expected_sha256, *, package=None):
    """Exec exactly the bytes of `path` whose sha256 equals `expected_sha256` (no .pyc, no re-read)."""
    data = Path(path).read_bytes()
    if expected_sha256 is not None and _sha(data) != expected_sha256:
        raise ImportError('%s is not the pinned file (sha256 mismatch)' % path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    if package is not None:
        module.__package__ = package
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def _sibling_pins():
    if '--sibling-pins' in sys.argv:
        return json.loads(sys.argv[sys.argv.index('--sibling-pins') + 1])
    return {}


_PINS = _sibling_pins()
approved = sys.modules.get('approved')
if approved is None or Path(getattr(approved, '__file__', '')).resolve() != HERE / 'approved.py':
    approved = exec_verified('approved', HERE / 'approved.py', _PINS.get('approved.py'))
child_watchdog = sys.modules.get('child_watchdog')
if child_watchdog is None or Path(getattr(child_watchdog, '__file__', '')).resolve() != HERE / 'child_watchdog.py':
    child_watchdog = exec_verified('child_watchdog', HERE / 'child_watchdog.py', _PINS.get('child_watchdog.py'))


def emit(value):
    print(json.dumps(value), flush=True)


def log_request_stage(rid, stage, started, clock):
    """Diagnostic receipts only: stderr, no speech/reference content, and no stdout protocol messages."""
    try:
        if sys.stderr is None:
            return
        print(json.dumps({'event': 'vox_request_stage', 'id': rid, 'pid': os.getpid(), 'stage': stage,
                          'elapsed_seconds': round(clock() - started, 6)}), file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - logging must not affect request handling
        pass  # Diagnostics must not turn a valid take into a failure if the log is unavailable.


def log_startup_stage(stage, started):
    """Record dependency initialization without adding a stdout protocol message."""
    try:
        if sys.stderr is None:
            return
        print(json.dumps({'event': 'vox_startup_stage', 'pid': os.getpid(), 'stage': stage,
                          'elapsed_seconds': round(time.perf_counter() - started, 6)}),
              file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - logging must not affect startup
        pass


def fault_fields(error, *, phase='request'):
    """Describe a failure without turning deployment/runtime bugs into resource waits.

    Explicit CUDA OOM and the resource-budget exception are resource failures. CUDA loader, kernel, dependency,
    model and protocol failures stay fatal. Only known input/output faults belong to one cue.
    """
    kind = type(error).__name__
    detail = str(error)
    if getattr(error, 'fault_kind', None) in ('fatal', 'resource', 'recoverable', 'cancelled'):
        return {'fault_kind': error.fault_kind, 'fault_code': getattr(error, 'fault_code', 'vox_' + kind.lower())}
    lower = detail.lower()
    if kind == 'ResourceYield':
        return {'fault_kind': 'resource', 'fault_code': 'vox_resource_yield'}
    if kind == 'OutOfMemoryError' or any(text in lower for text in (
            'cuda out of memory', 'cuda error: out of memory', 'cudaerrormemoryallocation', 'cuda_error_out_of_memory')):
        return {'fault_kind': 'resource', 'fault_code': 'vox_cuda_out_of_memory'}
    if isinstance(error, InterruptedError):
        return {'fault_kind': 'cancelled', 'fault_code': 'vox_cancelled'}
    if phase == 'request':
        if kind == 'ReferenceUnverified':
            return {'fault_kind': 'recoverable', 'fault_code': 'vox_reference_unverified'}
        if isinstance(error, ValueError):
            if detail.startswith('Invalid actor synthesis request'):
                return {'fault_kind': 'recoverable', 'fault_code': 'vox_text_invalid'}
            if detail.startswith('Actor reference'):
                return {'fault_kind': 'recoverable', 'fault_code': 'vox_reference_invalid'}
            if detail.startswith(('Invalid actor PCM', 'Actor PCM', 'Actor synthesis returned no audio')):
                return {'fault_kind': 'recoverable', 'fault_code': 'vox_output_invalid'}
    if isinstance(error, FileNotFoundError):
        code = 'vox_runtime_missing'
    elif isinstance(error, PermissionError):
        code = 'vox_runtime_permission'
    elif isinstance(error, ImportError) or 'artifact changed' in lower:
        code = 'vox_model_unverified'
    else:
        code = 'vox_startup_failed' if phase == 'startup' else 'vox_runtime_failed'
    return {'fault_kind': 'fatal', 'fault_code': code}


def check_engine_settings(synthesize, settings=approved.SETTINGS):
    """The pinned ActorVoice.synthesize defaults must equal the approved settings (no silent drift)."""
    params = inspect.signature(synthesize).parameters
    for key in ('max_tokens', 'steps', 'guidance'):
        if key not in params or params[key].default != settings[key]:
            raise ValueError('ActorVoice.synthesize %s default differs from the approved setting' % key)
    if params.get('seed') is None or params['seed'].default != approved.SEEDS[0]:
        raise ValueError('ActorVoice.synthesize seed default differs from the approved seed')


def write_take(output, audio):
    """Byte-for-byte the write relay/actor_child.py used for the approved takes."""
    import soundfile as sf
    output = Path(output)
    temporary = output.with_suffix('.tmp.wav')
    sf.write(temporary, audio, 48000, subtype='FLOAT')
    temporary.replace(output)


def load_relay(relay_root):
    """relay.actor_voice and relay.resource_budget from verified bytes into a private package (empty __path__)."""
    relay_root = Path(relay_root).resolve()
    package = types.ModuleType('relay')
    package.__path__ = []
    package.__file__ = str(relay_root / 'relay' / '__init__.py')
    sys.modules['relay'] = package
    voice = exec_verified('relay.actor_voice', relay_root / 'relay' / 'actor_voice.py', approved.ACTOR_VOICE_SHA256,
                          package='relay')
    budget = exec_verified('relay.resource_budget', relay_root / 'relay' / 'resource_budget.py',
                           approved.RESOURCE_BUDGET_SHA256, package='relay')
    package.actor_voice, package.resource_budget = voice, budget
    return voice, budget


def sustained_budget_class(ResourceBudget):
    """ResourceBudget whose GPU/RAM reserve yields only after a SUSTAINED shortfall (review P2-5; lease v2 uses the
    same 1.5 s rule). Sem's Infernix cache dips free VRAM for 0.4-0.6 s about every 30.5 s; one instantaneous
    sample every 2 s (relay default) would cut a take in ~25 % of dips. Hold/coordination/process checks are unchanged
    and immediate; this child is given no Sem status files."""

    class SustainedBudget(ResourceBudget):
        def __init__(self, *, gpu_sustain_seconds=1.5, sample_seconds=0.5, clock=time.perf_counter, **payload):
            super().__init__(**payload)
            self.gpu_sustain_seconds, self.sample_seconds, self.clock = float(gpu_sustain_seconds), float(sample_seconds), clock
            self._sampled, self._below_since, self.reason = None, None, None
            self.dips = []

        def _shortfall(self):
            try:
                gpu = self.gpu_free()
                ram = self.ram_free()
            except Exception:   # noqa: BLE001 - unreadable counts as short (fail closed after the sustain window)
                return 'Resource availability could not be read'
            if gpu < self.reserve_gpu_mib:
                return 'GPU memory below the voice reserve'
            if ram is not None and ram < self.reserve_ram_mib:
                return 'System memory below the voice reserve'
            return None

        def should_yield(self):
            reason = self.voice_busy()
            if reason:
                self.reason = reason
                return True
            now = self.clock()
            if self._sampled is None or now - self._sampled >= self.sample_seconds:
                self._sampled = now
                short = self._shortfall()
                if short and self._below_since is None:
                    self._below_since, self._short = now, short
                elif not short and self._below_since is not None:
                    self.dips.append(round(now - self._below_since, 3))
                    self._below_since = None
            if self._below_since is not None and now - self._below_since >= self.gpu_sustain_seconds:
                self.reason = '%s for %.1f s' % (self._short, now - self._below_since)
                return True
            return False

    return SustainedBudget


def _yielding(budget):
    return budget is not None and bool(budget.reason or budget.should_yield())


def marker_exists(request):
    try:
        return Path(str(request['output']) + '.cancel').exists()
    except (KeyError, TypeError, OSError):
        return False


class ReferenceUnverified(Exception):
    pass


def private_reference(request, folder):
    """Copy the reference to a child-private file after checking its bytes against reference_sha256, so the engine
    reads exactly the verified bytes (review P3-1 d: no TOCTOU between the renderer's check and the engine's read)."""
    expected = request.get('reference_sha256')
    try:
        data = Path(request['reference']).read_bytes()
    except OSError as error:
        raise ReferenceUnverified('reference file became unavailable: %s' % error) from error
    if not expected or _sha(data) != expected:
        raise ReferenceUnverified('reference bytes do not match reference_sha256')
    path = Path(folder) / (expected + '.wav')
    if not path.is_file():
        temporary = path.with_suffix('.part')
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return path


def serve(engine, lines, send, *, budget=None, idle_seconds=None, settings=approved.SETTINGS, clock=time.perf_counter,
          write=write_take, poll=.05, hard_cancel=None, reference_dir=None):
    """Serve requests from `lines` (a queue of raw lines; None = stdin closed) until close/EOF/yield/failure.

    Returns the exit reason. A cancelled request keeps the engine warm; a budget yield or any failure ends the process
    (relay/actor_child.py exits after a failure too)."""
    idle_since = clock()
    while True:
        try:
            line = lines.get(timeout=poll)
        except queue.Empty:
            if budget is not None and budget.should_yield():
                send({'event': 'yield', 'error': budget.reason, 'kind': 'ResourceYield',
                      'fault_kind': 'resource', 'fault_code': 'vox_resource_yield'})
                return 'yield'
            if idle_seconds and clock() - idle_since > idle_seconds:
                send({'event': 'idle-exit', 'error': 'Movie Vox engine released its idle model', 'kind': 'ResourceYield',
                      'fault_kind': 'resource', 'fault_code': 'vox_idle_release'})
                return 'idle'
            continue
        if line is None:
            return 'eof'
        if len(line) > MAX_LINE:
            send({'error': 'request exceeds bounded size', 'kind': 'ProtocolError',
                  'fault_kind': 'fatal', 'fault_code': 'vox_protocol_size'})
            return 'protocol'
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('request must be an object')
        except ValueError as error:
            send({'error': 'bad request: %s' % (error,), 'kind': 'ProtocolError',
                  'fault_kind': 'fatal', 'fault_code': 'vox_protocol_invalid'})
            return 'protocol'
        action = request.get('action', 'synthesize')
        if action == 'close':
            send({'event': 'closed'})
            return 'close'
        if action == 'ping':
            send({'id': request.get('id'), 'pong': True})
            continue
        if action != 'synthesize':
            send({'id': request.get('id'), 'error': 'unsupported action', 'kind': 'ProtocolError',
                  'fault_kind': 'fatal', 'fault_code': 'vox_protocol_action'})
            return 'protocol'
        rid = request.get('id')
        state = {'first': None, 'stage': 'dequeued'}
        started = clock()

        def reached(stage):
            state['stage'] = stage
            log_request_stage(rid, stage, started, clock)

        reached('dequeued')
        try:
            output = Path(request['output'])
            marker = Path(str(output) + '.cancel')

            def cancelled():
                return marker.exists()

            def first_pcm(_chunk):
                if state['first'] is None:
                    state['first'] = clock() - started
            reference = private_reference(request, reference_dir) if reference_dir is not None else request['reference']
            reached('reference_verified' if reference_dir is not None else 'reference_selected')
            if hard_cancel is not None:
                hard_cancel.arm(output)
            if cancelled():
                raise InterruptedError('cancel marker present before generation')
            # This enters ActorVoice.synthesize, including its lazy imports and request setup; it does not
            # establish that the native generate call has begun. That boundary is inside the binding.
            reached('synthesis_entry')
            audio, receipt = engine.synthesize(
                request['text'], str(reference), request.get('reference_text', ''), callback=first_pcm,
                cancelled=cancelled, max_tokens=settings['max_tokens'], steps=settings['steps'],
                guidance=settings['guidance'], seed=int(request.get('seed', approved.SEEDS[0])),
                progress=lambda stage: reached('actor_' + stage))
            if hard_cancel is not None:
                hard_cancel.disarm()
            reached('synthesis_return')
            if cancelled():
                raise InterruptedError('cancel marker present after generation')
            write(output, audio)
            reached('output_written')
            send({'id': rid, 'receipt': receipt, 'seconds': len(audio) / 48000, 'output': str(output)})
            idle_since = clock()
        except InterruptedError as error:
            if hard_cancel is not None:
                hard_cancel.disarm()
            if not marker_exists(request) and _yielding(budget):
                send({'id': rid, 'error': str(budget.reason or error), 'kind': 'ResourceYield',
                      'first_pcm_seconds': state['first'], 'request_stage': state['stage'],
                      'fault_kind': 'resource', 'fault_code': 'vox_resource_yield'})
                return 'yield'
            send({'id': rid, 'cancelled': True, 'kind': 'InterruptedError', 'error': str(error),
                  'first_pcm_seconds': state['first'], 'request_stage': state['stage'],
                  'fault_kind': 'cancelled', 'fault_code': 'vox_cancelled'})
            idle_since = clock()
        except Exception as error:   # noqa: BLE001 - reported to the host, then the process exits
            if hard_cancel is not None:
                hard_cancel.disarm()
            send({'id': rid, 'error': str(error)[:500], 'kind': type(error).__name__,
                  'first_pcm_seconds': state['first'], 'request_stage': state['stage'], **fault_fields(error)})
            return 'error'


def start_watchdogs(args):
    """Parent and lease-event watchdogs first (before any model load), then the hard-cancel watchdog."""
    started = []
    if args.parent_pid:
        child_watchdog.watch_parent(args.parent_pid)
        started.append('parent')
    if args.cancel_event:
        child_watchdog.watch_event(args.cancel_event)
        started.append('lease_event')
    hard = child_watchdog.HardCancel(args.hard_cancel_ms / 1000.0) if args.hard_cancel_ms > 0 else None
    if hard is not None:
        started.append('hard_cancel')
    return hard, started


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--relay-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--budget', required=True)
    parser.add_argument('--capacity', required=True, type=float)
    parser.add_argument('--idle-seconds', type=float, default=0.0)
    parser.add_argument('--actor-voice-sha256', required=True)
    parser.add_argument('--sibling-pins', default='{}')
    parser.add_argument('--parent-pid', type=int, default=0)
    parser.add_argument('--cancel-event', default=None)
    parser.add_argument('--hard-cancel-ms', type=float, default=60.0)
    parser.add_argument('--reserve-ram-mib', type=int, default=None)
    parser.add_argument('--gpu-sustain-seconds', type=float, default=1.5)
    args = parser.parse_args(argv)
    hard, watchdogs = start_watchdogs(args)
    engine, private = None, None
    dependency_stage = None
    try:
        relay_root = Path(args.relay_root).resolve()
        voice_file = relay_root / 'relay' / 'actor_voice.py'
        if args.actor_voice_sha256 != approved.ACTOR_VOICE_SHA256 or _sha(voice_file.read_bytes()) != approved.ACTOR_VOICE_SHA256:
            emit({'error': 'relay/actor_voice.py is not the approved ActorVoice binding', 'kind': 'ModelUnverified',
                  'fault_kind': 'fatal', 'fault_code': 'vox_actor_binding_unverified'})
            return 3
        try:
            voice_module, budget_module = load_relay(relay_root)
        except ImportError as error:
            emit({'error': str(error)[:500], 'kind': 'ModelUnverified',
                  'fault_kind': 'fatal', 'fault_code': 'vox_source_unverified'})
            return 3
        ActorVoice = voice_module.ActorVoice
        check_engine_settings(ActorVoice.synthesize)
        payload = json.loads(Path(args.budget).read_text(encoding='utf-8'))
        payload['status_files'] = []                       # never Sem's status.json (CONTRACT.md section 3)
        if args.reserve_ram_mib is not None:
            payload['reserve_ram_mib'] = int(args.reserve_ram_mib)
        budget = sustained_budget_class(budget_module.ResourceBudget)(gpu_sustain_seconds=args.gpu_sustain_seconds,
                                                                     **payload)
        # Complete the required Python/audio DLL initialization before Vox starts native workers and before
        # reporting ready. ActorVoice's first request then reuses these modules; no synthesis warmup is performed.
        dependency_started = time.perf_counter()
        dependency_stage = 'numpy_import'
        log_startup_stage(dependency_stage, dependency_started)
        import numpy  # noqa: F401 - eager dependency initialization for ActorVoice.synthesize
        dependency_stage = 'soundfile_import'
        log_startup_stage(dependency_stage, dependency_started)
        import soundfile  # noqa: F401 - eager dependency initialization for synthesis and write_take
        log_startup_stage('audio_dependencies_ready', dependency_started)
        dependency_stage = None
        try:
            free_before = budget.gpu_free()
        except Exception:   # noqa: BLE001 - recorded as unknown; admit_gpu() still decides
            free_before = None
        started = time.perf_counter()
        try:
            engine = ActorVoice(args.manifest, backend=approved.SETTINGS['backend'], threads=approved.SETTINGS['threads'],
                                max_reference_seconds=args.capacity, budget=budget)
        except ValueError as error:
            kind = 'ModelUnverified' if 'artifact changed' in str(error) else type(error).__name__
            emit({'error': str(error)[:500], 'kind': kind, **fault_fields(error, phase='startup')})
            return 3
        load_seconds = time.perf_counter() - started
        try:
            free_after = budget.gpu_free()
        except Exception:   # noqa: BLE001
            free_after = None
        private = tempfile.mkdtemp(prefix='vox-movie-ref-')
        emit({'ready': True, 'protocol': approved.PROTOCOL, 'pid': os.getpid(), 'model_sha256': engine.model_sha256,
              'quantization': engine.quantization, 'backend': approved.SETTINGS['backend'],
              'device': approved.SETTINGS['device'], 'threads': approved.SETTINGS['threads'],
              'capacity_seconds': args.capacity, 'encoder_sample_capacity': approved.encoder_sample_capacity(args.capacity),
              'actor_voice_sha256': approved.ACTOR_VOICE_SHA256, 'settings': dict(approved.SETTINGS),
              'load_seconds': round(load_seconds, 6), 'watchdogs': watchdogs,
              'reserve_ram_mib': budget.reserve_ram_mib, 'gpu_sustain_seconds': budget.gpu_sustain_seconds,
              'vram': {'device_free_before_mib': free_before, 'device_free_after_mib': free_after,
                       'device_delta_mib': (free_before - free_after) if None not in (free_before, free_after) else None,
                       'scope': 'device-wide nvidia-smi free memory around this process model load'}})
        lines = queue.Queue(maxsize=4)

        def receive():
            try:
                for value in sys.stdin:
                    lines.put(value)
            finally:
                lines.put(None)
        threading.Thread(target=receive, name='vox-movie-stdin', daemon=True).start()
        serve(engine, lines, emit, budget=budget, idle_seconds=args.idle_seconds or None, hard_cancel=hard,
              reference_dir=private)
        return 0
    except Exception as error:   # noqa: BLE001
        emit({'error': str(error)[:500], 'kind': type(error).__name__, **fault_fields(error, phase='startup'),
              **({'startup_stage': dependency_stage} if dependency_stage is not None else {})})
        return 1
    finally:
        if engine is not None:
            engine.close()
        if private:
            shutil.rmtree(private, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
