"""Resident one-pass Demucs worker + its movie-lease guard (W10).

    python -I <app>/relay/first_pass_resident.py      (started only by relay.movie_pipeline.resident_demucs, through
                                                      the movie GPU lease's spawn(), i.e. INSIDE the lease's Job)

The process loads the pinned htdemucs bag ONCE: relay.first_pass.DemucsInfer, on the device given in the init message
(actor_beta.demucs_device). It then serves window requests. A request is the SAME first-pass-request.json that
first_pass_parent.run writes, and the answer is the SAME first_pass.write_bundle (first-pass-bundle.json +
receipt.json), so the parent's validation (bundle_problems, stem hashes, geometry, code hashes) is unchanged.

Protocol (JSON lines; stdin -> stdout; nothing else is printed to stdout):
  in : {"init": {"model_dir", "model_pins", "threads", "device", "lease": {"cancel_event_name", "sem_mode"}}}
  out: {"ready": true, "info": {...}}            or {"fatal": "..."} then exit 2
  in : {"request": "<path of first-pass-request.json>"}
  out: {"done": "<receipt path>"} | {"refused": "<reason>"} (receipt written with state 'refused')
  A guard violation writes the request's abort.json and hard-exits (os._exit):
    real resource pressure or preemption (lease cancel event, Sem present/unverifiable, RAM bound)
        -> {'kind': 'ResourceYield', 'fault': {'kind': 'resource', ...}}           exit 75
    the per-request wall deadline (not evidence of resource pressure)
        -> {'kind': 'Fatal', 'fault': {'kind': 'fatal', 'code': 'demucs_window_deadline'}}  exit 76
    any other guard failure (a bug or an unreadable OS counter)
        -> {'kind': 'Fatal', 'fault': {'kind': 'fatal', 'code': 'lease_guard_failure'}}     exit 77
  Any other exception: an explicit CUDA out-of-memory -> {"yield": ...} exit 75 (real VRAM shortage); everything else
    (bad request/path/permission/protocol, a bug) -> {"fatal": ...} exit 2. Nothing ends without a classified reply.
  EOF on stdin -> exit 0.

LeaseGuard replaces demucs_bed.Guard, whose Sem status.json rule can never admit Sem OFF:
- Same bounds as that guard: MAX_WORKING_SET, MIN_AVAILABLE, BELOW_NORMAL priority, plus a per-request wall bound.
- Every 0.2 s it reads the movie GPU lease's cancel event: lease_client.cancel_is_set(name). The client is the app's
  pinned lease v2 client, relay.app_runtime.lease_client (W7).
- In sem_mode 'absent' it also requires verified Sem process absence: lease_client.sem_processes(); an unverifiable
  process counts as present.
- No Sem status file is opened or synthesized.
- The lease watcher also ends this process on any preemption, because the process is a member of the lease Job.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIT_OK, EXIT_REFUSED, EXIT_RESOURCE, EXIT_DEADLINE, EXIT_GUARD_FAILURE = 0, 2, 75, 76, 77
GUARD_INTERVAL_S = 0.2
VERSION = 'kevflix-first-pass-resident-v3'


class DeadlineExceeded(Exception):
    """The per-request wall bound elapsed: a fatal deadline, never a resource wait."""


class LeaseGuard:
    EXIT_CODE = EXIT_RESOURCE

    def __init__(self, lease, demucs_bed, lease_client):
        self.bed, self.lc = demucs_bed, lease_client
        self.cancel_event_name = lease.get('cancel_event_name')
        self.sem_mode = lease.get('sem_mode')
        if not self.cancel_event_name:
            raise demucs_bed.Refused('resident Demucs needs the movie lease cancel event name')
        self.abort_path, self.deadline = None, None
        self.done, self.ready = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.telemetry = {'guard_checks': 0, 'peak_working_set': 0, 'min_ram_available': None,
                          'priority': 'below_normal', 'interval_seconds': GUARD_INTERVAL_S,
                          'lease_rule': 'movie lease cancel event not set', 'sem_mode': self.sem_mode, 'sem_scans': 0}
        self.previous_priority = None

    def arm(self, abort_path, timeout):
        with self.lock:
            self.abort_path = Path(abort_path) if abort_path else None
            self.deadline = time.monotonic() + float(timeout)

    def disarm(self):
        with self.lock:
            self.deadline = None

    def check(self):
        bed = self.bed
        with self.lock:
            deadline = self.deadline
        if deadline is not None and time.monotonic() > deadline:
            raise DeadlineExceeded('per-window wall bound exceeded')
        if self.lc.cancel_is_set(self.cancel_event_name):
            raise bed.ResourceAbort('movie GPU lease cancelled (preempted or released)')
        if self.sem_mode == 'absent':
            found = self.lc.sem_processes()
            self.telemetry['sem_scans'] += 1
            if found:
                raise bed.ResourceAbort('Sem process present or unverifiable: %s' % (found[:3],))
        rss, avail = bed.working_set(), bed.available_ram()
        t = self.telemetry
        t['guard_checks'] += 1
        t['peak_working_set'] = max(t['peak_working_set'], rss)
        t['min_ram_available'] = min(avail, t['min_ram_available'] or avail)
        if rss > bed.MAX_WORKING_SET or avail < bed.MIN_AVAILABLE:
            raise bed.ResourceAbort('RAM reserve bound')

    def _run(self):
        try:
            while True:
                self.check()
                self.ready.set()
                if self.done.wait(GUARD_INTERVAL_S):
                    return
        except Exception as error:   # noqa: BLE001 - every guard failure stops this worker (fail closed)
            text = '%s: %s' % (type(error).__name__, error)
            if isinstance(error, self.bed.ResourceAbort):
                kind, fault, code = 'ResourceYield', {'kind': 'resource', 'code': 'lease_guard_yield'}, EXIT_RESOURCE
            elif isinstance(error, DeadlineExceeded):
                kind, fault, code = 'Fatal', {'kind': 'fatal', 'code': 'demucs_window_deadline'}, EXIT_DEADLINE
            else:
                kind, fault, code = 'Fatal', {'kind': 'fatal', 'code': 'lease_guard_failure'}, EXIT_GUARD_FAILURE
            fault['detail'] = text[:4000]
            with self.lock:
                abort = self.abort_path
            if abort is not None:
                try:
                    self.bed.write_json(abort, {'kind': kind, 'fault': fault, 'error': text,
                                                'telemetry': self.telemetry})
                except Exception:  # noqa: BLE001 - the exit code still carries the class
                    pass
            os._exit(code)

    def __enter__(self):
        kernel = self.bed._kernel32()
        handle = kernel.GetCurrentProcess()
        self.previous_priority = kernel.GetPriorityClass(handle)
        if not kernel.SetPriorityClass(handle, self.bed.BELOW_NORMAL):
            raise OSError('SetPriorityClass failed')
        self.check()                     # admission before any model work
        threading.Thread(target=self._run, daemon=True, name='first-pass-lease-guard').start()
        if not self.ready.wait(5):
            raise self.bed.Refused('lease guard did not become ready')   # a guard fault, not resource evidence
        return self

    def __exit__(self, *exc):
        self.done.set()
        if self.previous_priority:
            kernel = self.bed._kernel32()
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), self.previous_priority)
        return False


def _reply(stream, value):
    stream.write(json.dumps(value, default=str) + '\n')
    stream.flush()


def serve(stdin, stdout):
    if HERE.name != 'relay':
        _reply(stdout, {'fatal': 'resident worker must run from the relay folder'})
        return EXIT_REFUSED
    sys.path.insert(0, str(HERE.parent))
    from relay import demucs_bed, first_pass
    from relay.app_runtime import lease_client
    try:
        init = json.loads(stdin.readline())['init']
    except (ValueError, KeyError, TypeError):
        _reply(stdout, {'fatal': 'first message must be {"init": ...}'})
        return EXIT_REFUSED
    threads = int(init['threads'])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(threads)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
    device = init.get('device', 'cpu')
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.monotonic()
    try:
        guard = LeaseGuard(init.get('lease') or {}, demucs_bed, lease_client)
        demucs_bed._block_network()
        with guard:
            guard.arm(None, float(init.get('load_timeout_seconds', 180)))
            infer = first_pass.DemucsInfer(init['model_dir'], init['model_pins'], threads, device=device).load()
            guard.disarm()
            _reply(stdout, {'ready': True, 'version': VERSION, 'info': dict(infer.info),
                            'load_seconds': round(time.monotonic() - started, 3), 'pid': os.getpid()})
            for raw in iter(stdin.readline, ''):
                raw = raw.strip()
                if not raw:
                    continue
                message = json.loads(raw)
                request = json.loads(Path(message['request']).read_text(encoding='utf-8'))
                if request.get('device') != device or request.get('model_pins') != init['model_pins'] \
                        or int(request.get('threads')) != threads:
                    demucs_bed.write_json(request['receipt'], {'version': first_pass.VERSION, 'state': 'refused',
                                                               'reason': 'request differs from the loaded model'})
                    _reply(stdout, {'refused': 'request differs from the loaded model'})
                    continue
                guard.arm(request['abort'], float(request['timeout_seconds']))
                request_started = time.monotonic()
                cuda = None
                cache_released = False
                try:
                    if str(device).startswith('cuda'):
                        import torch
                        cuda = torch.cuda
                        # Exclude model-load and previous-window peaks; live weights remain in the baseline.
                        cuda.synchronize(device)
                        cuda.reset_peak_memory_stats(device)
                    import soundfile as sf
                    original, rate = sf.read(request['original_source_wav'], dtype='float32', always_2d=True)
                    if rate != first_pass.RATE:
                        raise first_pass.Refused('original source WAV is not 48 kHz')
                    bundle = first_pass.write_bundle(request['folder'], original_source_wav=request['original_source_wav'],
                                                     source_layout_evidence=request['source_layout_evidence'],
                                                     original=original, origin=request['origin'],
                                                     scope=request.get('scope'), infer=infer, sources=infer.sources,
                                                     model_pins=request['model_pins'], threads=request['threads'],
                                                     model_info=infer.info, code_directory=HERE, device=device)
                    memory = None
                    if cuda is not None:
                        cuda.synchronize(device)
                        peak_reserved = int(cuda.max_memory_reserved(device))
                        peak_allocated = int(cuda.max_memory_allocated(device))
                        cuda.empty_cache()
                        cuda.synchronize(device)
                        current_reserved = int(cuda.memory_reserved(device))
                        current_allocated = int(cuda.memory_allocated(device))
                        cache_released = True
                        growth = max(peak_reserved - current_reserved, 0)
                        memory = {'version': 'demucs-window-cuda-memory-v1', 'device': str(device),
                            'pid': os.getpid(), 'origin': float(request['origin']), 'frames': len(original),
                            'peak_reserved_bytes': peak_reserved, 'current_reserved_bytes': current_reserved,
                            'additional_peak_reserved_bytes': growth,
                            'peak_reserved_mib': peak_reserved / 1048576.,
                            'current_reserved_mib': current_reserved / 1048576.,
                            'additional_peak_reserved_mib': growth / 1048576.,
                            'peak_allocated_bytes': peak_allocated, 'current_allocated_bytes': current_allocated,
                            'reset_peak_memory_stats': True, 'empty_cache_completed': True,
                            'measured_at_unix': time.time(),
                            'scope': 'this process CUDA allocator; successful window peak and post-empty-cache current; not total GPU usage'}
                    bundle['telemetry'] = dict(guard.telemetry)
                    if memory is not None:
                        bundle['telemetry']['cuda_memory'] = memory
                        bundle.setdefault('parent', {})['cuda_memory'] = memory
                    bundle['child_seconds'] = time.monotonic() - request_started
                    bundle['producer'] = {'version': VERSION, 'resident': True, 'pid': os.getpid(), 'device': device}
                    demucs_bed.write_json(request['receipt'], bundle)
                    _reply(stdout, {'done': request['receipt'], 'cuda_memory': memory})
                except (first_pass.Refused, demucs_bed.Refused) as error:
                    demucs_bed.write_json(request['receipt'], {'version': first_pass.VERSION, 'state': 'refused',
                                                               'reason': str(error)})
                    _reply(stdout, {'refused': str(error)})
                finally:
                    try:
                        # A refused/failed window publishes no successful measurement, but releases cached blocks.
                        if cuda is not None and not cache_released:
                            cuda.empty_cache()
                    finally:
                        guard.disarm()
    except (demucs_bed.ResourceAbort, MemoryError) as error:
        _reply(stdout, {'yield': '%s: %s' % (type(error).__name__, error)})
        return EXIT_RESOURCE
    except (first_pass.Refused, demucs_bed.Refused) as error:
        _reply(stdout, {'fatal': str(error)})
        return EXIT_REFUSED
    except Exception as error:  # noqa: BLE001 - every other failure is classified for the parent, never silent
        text = '%s: %s' % (type(error).__name__, error)
        if _cuda_out_of_memory(error):           # an explicit CUDA allocator failure: real VRAM shortage
            _reply(stdout, {'yield': text[:4000]})
            return EXIT_RESOURCE
        _reply(stdout, {'fatal': text[:4000]})   # bad request/path/permission/protocol/bug: retrying cannot repair
        return EXIT_REFUSED
    return EXIT_OK


def _cuda_out_of_memory(error):
    """The explicit CUDA allocator errors (same strings relay.language.backend_fault treats as resource)."""
    if type(error).__name__ == 'OutOfMemoryError':
        return True
    lower = str(error).lower()
    return ('cuda out of memory' in lower or 'cuda failed with error out of memory' in lower
            or 'cuda_error_out_of_memory' in lower or ('cuda failure' in lower and 'out of memory' in lower))


def main():
    stdout = sys.stdout
    sys.stdout = sys.stderr            # library prints never reach the protocol pipe
    return serve(sys.stdin, stdout)


if __name__ == '__main__':
    sys.exit(main())
