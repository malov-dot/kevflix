"""VoxCPM2 (Vox) actor renderer: the Breeze actor-TTS contract (actor-tts/INTERFACES.md section 1) over the movie-Vox
engine host. Drop-in for runtime-adapter/breeze_renderer.BreezeActorRenderer.

Engine: the Kevin-approved first-dub Vox path, unchanged - relay/actor_voice.py ActorVoice (pinned), VoxCPM2 Q4_K +
AudioVAE F16 (sha256 1d0b2970...; located by config actor_runtime_manifest), reference-only or verified source-transcript
conditioning, seed 42 / retry 7,
max_tokens 350, steps 10, guidance 1.6 (approved.py). The child writes 48 kHz mono FLOAT. Before caching, the renderer
applies one bounded constant gain with a receipt; format and duration stay unchanged. The cached hash describes the
normalized bytes used by readback and mixing, not the raw native output.

Reference rules retain the mandatory callable reference gate, sha256 match, WAV of 1.0 s to actor_reference_seconds,
and non-silent finite audio. The source-language transcript is optional; it is never replaced with English subtitles.
Sem's own voice is never a dub reference. Program-owned references only: a
mandatory `provenance(voice)` callable (reference_provenance.verify bound to the current job) must accept the reference,
and hand-picked/agent-prepared references (approved.REFERENCES hashes, configured agent-prepared roots) are refused.

Takes: never sped up, padded or truncated. A silent take, or one that reaches the max_tokens cap (possibly cut), is
output_invalid, so the caller keeps that cue's original audio.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from . import approved
from .errors import ActorBusy, ActorCancelled, ActorUnavailable, wrap_fault
from .jobs import DeadlineExpired, RenderJob
from .vox_engine_host import EngineError, HostHold

ENGINE = approved.ENGINE                    # 'VoxCPM2/audio.cpp' (beta_common.ACCEPTED_CLONE_ENGINES)
CACHE_TAG = 'vox-native-v3-peak-normalized-reference-optional'
LEVEL_POLICY = 'generated-take-peak-minus3db-v1'
LEVEL_TARGET_DBFS = -3.0
LEVEL_MAX_BOOST_DB = 40.0
LEVEL_MIN_PEAK = 1e-5
SCOPE = 'vox-actor-tts'
SOURCE_RATE = 48000
TARGET_RATE = 48000
HEX64 = re.compile(r'^[a-f0-9]{64}$')
DEFAULTS = {
    'actor_runtime_manifest': '',     # machine location from config (required); its CONTENT is checked (approved.py)
    'actor_reference_seconds': 10,
    'actor_reserve_gpu_mib': 3072,
    'actor_required_gpu_mib': 2700,
}
VOX_DEFAULTS = {
    'max_text_chars': 400,
    'max_output_seconds': 60,
    'timeout_seconds': 180,
    'idle_grace_seconds': 150,
    'poll_seconds': 0.05,
    'cancel_ack_seconds': 0.15,   # host-side BACKUP kill only; the child's own hard cancel fires at hard_cancel_ms
    'hard_cancel_ms': 60,         # child exits by itself this long after an unacknowledged cancel marker
    'reserve_ram_mib': 12288,     # EXPLICIT child RAM reserve (the relay's existing value; unmeasured for Vox, see plan)
    'gpu_sustain_seconds': 1.5,   # child yields only after a sustained GPU/RAM shortfall (lease v2's rule)
    'job_lease_idle_seconds': 150,  # keep one GPU lease (and warm engines) across windows of a job for this long idle
    'max_instances': 'auto', # FIX 6: 'auto' (default) = a fresh process starts with 1 instance; later windows admit a 2nd
                             # only from the RUNNING instance's measured in-take peak (host live telemetry) and the
                             # lease budget; an explicit integer (incl. 1) overrides.
    'max_instances_cap': 2,  # the bounded cap in auto mode
    'refusal_hold_seconds': 300,  # after any resource refusal, k falls back to 1 for this long
    'instances': None,       # deprecated alias of max_instances
    'python': None,          # the Vox engine interpreter (machine location); else the relay config 'python'
    'sem_roots': [],         # extra Sem roots on this machine (e.g. Sem's install folder): refused, never on a child cmd
    'movie_roots': [],       # movie folders that may lie under a Sem root (the cache_dir is always allowed)
}


def concurrency_mode(options):
    """('auto', cap) or ('fixed', n) from actor_beta.vox.max_instances ('instances' accepted as an alias)."""
    value = options.get('max_instances')
    if value is None:
        value = options.get('instances') or 'auto'
    cap = max(1, int(options.get('max_instances_cap') or 2))
    if isinstance(value, str) and value.strip().lower() == 'auto':
        return 'auto', cap
    return 'fixed', max(1, int(value))


def max_instances(options):
    """The pool size: the auto cap, or the fixed integer bound."""
    return concurrency_mode(options)[1]


def configure_roots(config, options):
    """Machine roots from config (approved.configure): Sem roots refused, the cache and movie roots allowed."""
    allowed = [config.get('cache_dir')] + list(options.get('movie_roots') or [])
    approved.configure(sem_roots=options.get('sem_roots') or (), allowed_roots=[a for a in allowed if a],
                       agent_prepared_roots=options.get('agent_prepared_roots') or ())


def vox_options(config):
    """actor_beta.vox options (merged over VOX_DEFAULTS). They live inside actor_beta, so they are part of
    Config.engine_fingerprint()'s actor_beta options."""
    beta = config.get('actor_beta') if hasattr(config, 'get') else None
    vox = (beta or {}).get('vox') if isinstance(beta, dict) else None
    return {**VOX_DEFAULTS, **{k: v for k, v in dict(vox or {}).items() if v is not None}}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temp, path)


def _unlink(path):
    try:
        Path(path).unlink()
    except OSError:
        pass


class VoxActorRenderer:
    def __init__(self, config, owner, *, transport=None, admission=None, reference_gate=None, provenance=None):
        # `transport` keeps the Breeze parameter name: the injected engine (a VoxEngineHost or a test fake with
        # synthesize(request, cancelled=, token=), cancel(token=, reason=) and health()).
        # `provenance` (Vox addition, keyword-only): callable(voice) -> program provenance summary, raising for a
        # reference the pipeline did not cut. Mandatory at use, like reference_gate.
        self.provenance = provenance
        self.config = {**DEFAULTS, **{k: v for k, v in dict(config or {}).items() if v is not None}}
        self.options = vox_options(self.config)
        self.owner = owner
        self.admission = admission
        self.reference_gate = reference_gate
        self._transport = transport
        self._verified = False
        self._timed_out = None          # a confirmed engine request timeout in this renderer's window (see _synthesize)
        self._epoch = 0                 # close() bumps it: every request started before then is cancelled
        self._jobs = set()
        self._jobs_lock = threading.Lock()
        self.manifest_path = Path(self.config.get('actor_runtime_manifest') or '.') \
            if self.config.get('actor_runtime_manifest') else None
        self.capacity_seconds = float(self.config['actor_reference_seconds'])
        configure_roots(self.config, self.options)

    # -- admission / identity -------------------------------------------------
    def _admitted(self):
        a = self.admission
        return a is not None and getattr(a, 'granted', False) is True and getattr(a, 'scope', None) == SCOPE

    def _require_admission(self):
        if not self._admitted():
            raise ActorUnavailable('configuration_invalid', 'root-owned GPU admission for %s not granted' % SCOPE,
                                   fault_code='vox_admission_invalid')

    def _transport_or_create(self):
        if self._transport is None:
            from .factory import host_options
            from .vox_engine_host import shared_host
            try:
                self._transport = shared_host(host_options(self.config))
            except ActorUnavailable:
                raise
            except Exception as error:   # noqa: BLE001 - malformed paths/options cannot be solved by resource retries
                raise wrap_fault(error, reason='configuration_invalid', fault_code='vox_host_configuration') from error
        return self._transport

    def _manifest(self):
        if self.manifest_path is None:
            raise ActorUnavailable('model_unverified', 'actor_runtime_manifest is not configured')
        try:
            raw = self.manifest_path.read_bytes()
            data = json.loads(raw.decode('utf-8-sig'))
        except (OSError, ValueError) as error:
            raise ActorUnavailable('model_unverified', 'actor runtime manifest unreadable: %s' % (error,)) from error
        model = data.get('modelSha256') if isinstance(data, dict) else None
        if not isinstance(model, str) or not HEX64.match(model):
            raise ActorUnavailable('model_unverified', 'actor runtime manifest lacks modelSha256')
        manifest_sha = _sha256_bytes(raw)
        # Approved first-dub engine only, by CONTENT (the manifest's paths are machine-specific): the model hash and the
        # exact native artifact set. ActorVoice re-hashes every file at load.
        artifacts = {a.get('file'): a.get('sha256') for a in data.get('artifacts') or [] if isinstance(a, dict)}
        if model != approved.MODEL_SHA256 or artifacts != approved.RUNTIME_ARTIFACTS_SHA256:
            raise ActorUnavailable('model_unverified', 'actor runtime manifest is not the approved Q4_K first-dub runtime')
        if data.get('quantization') != approved.QUANTIZATION:
            raise ActorUnavailable('model_unverified', 'actor runtime quantization is not %s' % approved.QUANTIZATION)
        return data, model, manifest_sha

    def _verify_model(self, transport, manifest):
        if self._verified:
            return
        try:
            health = transport.health()
        except ActorUnavailable:
            raise
        except HostHold as error:
            if error.fault_kind == 'cancelled':
                raise ActorCancelled(str(error), fault_code=error.fault_code) from error
            if error.fault_kind == 'fatal':
                raise ActorUnavailable('runtime_failed', str(error), fault_code=error.fault_code) from error
            raise ActorBusy(str(error), fault_code=error.fault_code) from error
        except EngineError as error:
            raise self._engine_unavailable(error) from error
        except Exception as error:   # noqa: BLE001
            raise wrap_fault(error, fault_code='vox_health_failed', detail='engine health failed: %s' % error) from error
        if not isinstance(health, dict) or health.get('engine') != ENGINE:
            raise ActorUnavailable('server_unavailable', 'engine health not available')
        if health.get('expected_model_sha256') != approved.MODEL_SHA256:
            raise ActorUnavailable('model_unverified', 'engine host is not configured for the approved actor model')
        if Path(str(health.get('manifest', ''))).resolve() != self.manifest_path.resolve():
            raise ActorUnavailable('model_unverified', 'engine host manifest does not match actor_runtime_manifest')
        if health.get('model_sha256') not in (None, approved.MODEL_SHA256):
            raise ActorUnavailable('model_unverified', 'engine reports a different model')
        if float(health.get('capacity_seconds', -1)) != self.capacity_seconds:
            raise ActorUnavailable('model_unverified', 'engine reference capacity differs from actor_reference_seconds')
        self._verified = True

    # -- references -----------------------------------------------------------
    def _load_reference(self, voice, reference_only):
        if not callable(self.reference_gate):
            raise ActorUnavailable('configuration_invalid', 'no reference gate; admitted source association required',
                                   fault_code='vox_reference_gate_missing')
        if not isinstance(voice, dict):
            raise ActorUnavailable('reference_invalid', 'voice must be a mapping')
        reference = voice.get('reference')
        if not reference or not Path(reference).is_file():
            raise ActorUnavailable('reference_invalid', 'reference file missing')
        expected = voice.get('reference_sha256')
        if not isinstance(expected, str) or not HEX64.match(expected.lower()):
            raise ActorUnavailable('reference_unverified', 'voice has no reference_sha256')
        if expected.lower().startswith(approved.SEM_REFERENCE_PREFIXES):
            raise ActorUnavailable('untrusted_actor', "Sem's voice reference is never a dub reference")
        refused = approved.refused_reference_root(reference)         # component-wise, never a substring (P3-2)
        if refused == 'sem':
            raise ActorUnavailable('untrusted_actor', 'reference path is under a protected Sem/personal root')
        if expected.lower() in approved.REFERENCES or refused == 'agent-prepared':
            raise ActorUnavailable('untrusted_actor', 'hand-picked/agent-prepared reference; only program-cut references')
        try:
            data = Path(reference).read_bytes()
        except OSError as error:
            raise ActorUnavailable('reference_invalid', str(error)) from error
        if _sha256_bytes(data) != expected.lower():
            raise ActorUnavailable('reference_unverified', 'reference sha256 mismatch')
        try:
            import soundfile as sf
        except (ImportError, OSError) as error:
            raise ActorUnavailable('runtime_failed', 'soundfile runtime unavailable: %s' % error,
                                   fault_code='vox_audio_runtime_missing') from error
        try:
            audio, rate = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
            info = sf.info(io.BytesIO(data))
        except Exception as error:   # noqa: BLE001
            raise ActorUnavailable('reference_invalid', 'reference WAV unreadable: %s' % (error,)) from error
        if info.format != 'WAV' or rate <= 0 or audio.size == 0:
            raise ActorUnavailable('reference_invalid', 'reference must be a non-empty WAV')
        seconds = audio.shape[0] / float(rate)
        limit = float(self.config['actor_reference_seconds'])
        if seconds < 1.0 or seconds > limit:
            raise ActorUnavailable('reference_invalid', 'reference duration %.3fs outside 1..%gs' % (seconds, limit))
        if not np.all(np.isfinite(audio)) or float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) <= 1e-4:
            raise ActorUnavailable('reference_invalid', 'reference is silent or non-finite')
        source_text = voice.get('reference_text', '')
        if not isinstance(source_text, str):
            raise ActorUnavailable('reference_invalid', 'reference_text must be a string when provided')
        self.reference_gate(voice)
        if not callable(self.provenance):
            raise ActorUnavailable('configuration_invalid', 'no provenance check; program-cut reference required',
                                   fault_code='vox_provenance_check_missing')
        try:
            record = self.provenance(voice)
        except ActorUnavailable:
            raise
        except Exception as error:   # noqa: BLE001 - any provenance failure keeps that cue original
            raise ActorUnavailable('reference_unverified', 'program provenance: %s' % (error,)) from error
        if not isinstance(record, dict):
            raise ActorUnavailable('reference_unverified', 'program provenance returned no record')
        # Admission and provenance above always inspect the original voice record, including an empty source-text
        # record. Omitting text changes only Vox conditioning; it never bypasses reference identity or provenance.
        ref_text = '' if reference_only else source_text.strip()
        return data, ref_text, seconds, record

    def prepare_references(self, voices):
        for voice in voices:
            self._load_reference(voice, reference_only=False)

    # -- render ---------------------------------------------------------------
    def _check_text(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ActorUnavailable('text_invalid', 'empty text')
        if len(text) > int(self.options['max_text_chars']):
            raise ActorUnavailable('text_invalid', 'text exceeds %s chars' % self.options['max_text_chars'])
        if any(ord(c) < 32 and c not in '\t\n\r' for c in text):
            raise ActorUnavailable('text_invalid', 'control characters in text')
        return text.strip()

    def render(self, text, voice, directory, window, *, seed=42, reference_only=False, context_key=None):
        """Blocking, Breeze-contract render of one take (signature identical to BreezeActorRenderer.render)."""
        return self._render(text, voice, directory, window, seed=seed, reference_only=reference_only,
                            context_key=context_key, job=None)

    def submit(self, text, voice, directory, window, *, seed=42, reference_only=False, context_key=None,
               job_id=None, deadline=None, priority=0):
        """Concurrency API: start one line and return its RenderJob at once (see RESULTS.md, "Concurrency API")."""
        job = RenderJob(job_id, deadline=deadline, priority=priority, context_key=context_key)
        job.add_canceller(lambda reason: self._cancel_engine_job(job, reason))
        with self._jobs_lock:
            self._jobs.add(job)
        return job.start(lambda j: self._render(text, voice, directory, window, seed=seed, reference_only=reference_only,
                                                context_key=context_key, job=j),
                         on_done=self._job_done)

    def _job_done(self, job):
        with self._jobs_lock:
            self._jobs.discard(job)

    def cancel_job(self, job_id, reason='scheduler_cancel'):
        """Cancel this renderer's jobs with `job_id` (queued: at once; running: <= 0.2 s)."""
        with self._jobs_lock:
            jobs = [j for j in self._jobs if j.job_id == str(job_id)]
        for job in jobs:
            job.cancel(reason)
        return len(jobs)

    def _cancel_engine_job(self, job, reason):
        transport = self._transport
        if transport is not None and hasattr(transport, 'cancel'):
            transport.cancel(token=self, job_id=job.job_id, reason=reason)

    def _render(self, text, voice, directory, window, *, seed, reference_only, context_key, job):
        self.owner.check()
        epoch = self._epoch
        stop = lambda: (self.owner.cancelled.is_set() or self._epoch != epoch   # noqa: E731
                        or (job is not None and job.cancel_requested))
        spoken = self._check_text(text)
        self._require_admission()
        _ref_bytes, ref_text, _ref_seconds, provenance = self._load_reference(voice, reference_only)
        reference_only = not bool(ref_text)  # receipt/cache identity describe what is actually sent to Vox
        manifest, model, manifest_sha = self._manifest()
        identity = [text, voice, model, manifest_sha, CACHE_TAG, seed, reference_only, context_key,
                    approved.SETTINGS, self.capacity_seconds]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        path, receipt_path = folder / (key + '.wav'), folder / (key + '.json')
        meta = {'job_id': job.job_id if job else None, 'deadline': job.deadline if job else None,
                'priority': job.priority if job else 0}
        if path.is_file() and receipt_path.is_file():
            try:
                saved = json.loads(receipt_path.read_text(encoding='utf-8'))
                normalized = saved.get('level_normalization')
                if (saved.get('sha256') == _file_hash(path) and saved.get('engine') == ENGINE
                        and isinstance(normalized, dict) and normalized.get('policy') == LEVEL_POLICY):
                    return path, {**saved, 'timing_overrun': saved['seconds'] > window,
                                  'window_seconds': window, 'cache_hit': True,
                                  'scheduling': {**meta, 'engine_job': False}}
            except (OSError, ValueError, KeyError):
                pass
        # Unique per call: concurrent identical requests never share a partial file (identical bytes either way).
        part = folder / ('%s.%s.part.wav' % (key, uuid.uuid4().hex[:12]))
        if self._timed_out is not None:
            # The engine already timed out in this window (the actor pipeline makes one renderer per window): the
            # remaining cues keep their original audio instead of each waiting out another request timeout.
            raise ActorUnavailable('output_invalid', 'movie Vox timed out earlier in this window (%s); cue keeps '
                                   'its original audio' % self._timed_out, fault_kind='recoverable',
                                   fault_code='vox_timeout_window')
        transport = self._transport_or_create()
        self._verify_model(transport, manifest)
        admitted_sha = voice['reference_sha256'].lower()
        request = {'text': spoken, 'reference': str(Path(voice['reference']).resolve()), 'reference_text': ref_text,
                   'reference_sha256': admitted_sha, 'seed': int(seed), 'output': str(part)}
        started = time.perf_counter()
        try:
            response = self._synthesize(transport, request, stop, meta)
            if stop():
                raise ActorCancelled('render cancelled; late PCM dropped')
            seconds = self._validate_take(part, response)
            engine_sha = str((response.get('receipt') or {}).get('reference_sha256') or '').lower()
            if engine_sha != admitted_sha:
                # The engine must have read exactly the admitted bytes (review P3-1 d; the child copies the verified
                # bytes to a private file before synthesis).
                raise ActorUnavailable('reference_unverified', 'engine read reference %s, admitted %s'
                                       % (engine_sha[:12] or 'none', admitted_sha[:12]))
            level_normalization = self._normalize_take(part)
            if stop():
                raise ActorCancelled('render cancelled during level normalization; late PCM dropped')
            self.owner.check()                                      # last check BEFORE anything is committed
        except BaseException:
            for leftover in (part, part.with_suffix('.tmp.wav'), Path(str(part) + '.cancel')):
                if leftover.exists():
                    _unlink(leftover)
            raise
        for leftover in (part.with_suffix('.tmp.wav'), Path(str(part) + '.cancel')):
            if leftover.exists():
                _unlink(leftover)
        child = response.get('receipt') or {}
        result = {
            'seconds': seconds, 'sha256': None, 'text': text, 'voice': voice, 'engine': ENGINE,
            'model': model, 'model_manifest_sha256': manifest_sha,
            'reference_sha256': voice['reference_sha256'].lower(),
            'reference_text_sha256': _sha256_bytes(ref_text.encode('utf-8')),
            'server_url': None, 'timing_overrun': seconds > window, 'window_seconds': window,
            'cache_hit': False, 'seed': seed, 'reference_only': reference_only, 'context_key': context_key,
            'sample_rate': TARGET_RATE, 'source_rate': SOURCE_RATE,
            'chunks': int(child.get('stream_chunks') or 0),
            'first_audio_seconds': round(float(child.get('first_pcm_seconds') or 0.0), 6),
            'wall_seconds': round(time.perf_counter() - started, 6),
            'quantization': child.get('quantization'), 'cloning_mode': child.get('cloning_mode'),
            'real_time_factor': child.get('real_time_factor'), 'speed': 1.0,
            'settings': dict(approved.SETTINGS), 'native_fixed': dict(approved.NATIVE_FIXED),
            'encoder_capacity_seconds': self.capacity_seconds,
            'engine_process': {'kind': 'movie-vox-child', 'instance': response.get('instance'),
                               'pid': response.get('pid'), 'generation': response.get('generation'),
                               'shared_sem_worker_used': False, 'backend': approved.SETTINGS['backend'],
                               'device': approved.SETTINGS['device']},
            'scheduling': {**meta, 'engine_job': True, **dict(response.get('scheduling') or {})},
            'approved_reference': approved.reference_label(voice['reference_sha256']),
            'reference_provenance': dict(provenance),
            'level_normalization': level_normalization,
        }
        result['sha256'] = _file_hash(part)
        self._commit(part, path, receipt_path, result)
        return path, result

    @staticmethod
    def _commit(part, path, receipt_path, result):
        """WAV and receipt together (review P3-6): receipt staged first, then the WAV moved into place, then the receipt;
        any failure removes both, so a final WAV never exists without its receipt."""
        staged = receipt_path.with_name(receipt_path.name + '.%s.tmp' % uuid.uuid4().hex[:8])
        try:
            staged.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            os.replace(part, path)
            os.replace(staged, receipt_path)
        except BaseException:
            for leftover in (staged, part, path, receipt_path):
                _unlink(leftover)
            raise

    # Only a classified recoverable output failure may retry once. Startup, protocol, model, CUDA/runtime and
    # unclassified child failures remain fatal even if PCM had started; partial PCM never disguises a broken runtime.
    LINE_FAILURES = ('child_error', 'child_exited', 'timeout', 'protocol')
    LINE_ATTEMPTS = 2

    def _synthesize(self, transport, request, stop, meta):
        for attempt in range(1, self.LINE_ATTEMPTS + 1):
            try:
                return self._synthesize_once(transport, request, stop, meta)
            except EngineError as error:
                if stop():
                    raise ActorCancelled('render cancelled during an engine failure') from error
                mapped = self._engine_unavailable(error)
                if error.kind == 'timeout' and mapped.fault_kind == 'recoverable':
                    # The host classifies a timeout recoverable only after the owned child is confirmed dead (an
                    # unconfirmed drain stays fatal). No second full-timeout attempt on this line, and no further
                    # requests in this window: the cue (and the window's later cues) keep their original audio.
                    self._timed_out = str(error)[:200]
                    raise mapped from error
                if (mapped.fault_kind != 'recoverable' or error.kind not in self.LINE_FAILURES or error.first_pcm_seconds is not None
                        or mapped.reason in ('text_invalid', 'reference_invalid', 'reference_unverified')):
                    raise mapped from error
                if attempt == self.LINE_ATTEMPTS:
                    raise ActorUnavailable('output_invalid', 'engine failed on this line %d times: %s'
                                           % (attempt, error)) from error
                for leftover in (Path(request['output']), Path(request['output']).with_suffix('.tmp.wav')):
                    _unlink(leftover)

    def _engine_unavailable(self, error):
        self._verified = self._verified and error.kind not in ('model_unverified', 'start_failed', 'ready_timeout')
        if error.fault_kind == 'fatal':
            reason = 'model_unverified' if error.kind == 'model_unverified' else 'runtime_failed'
            return ActorUnavailable(reason, str(error), fault_kind='fatal', fault_code=error.fault_code)
        if error.fault_kind == 'resource':
            return ActorUnavailable('gpu_admission_hold', str(error), fault_kind='resource', fault_code=error.fault_code)
        if error.fault_kind == 'cancelled':
            return ActorCancelled(str(error), fault_code=error.fault_code)
        if error.first_pcm_seconds is not None:
            return ActorUnavailable('output_invalid', '%s after first audio' % (error,), fault_code=error.fault_code)
        detail = str(error.detail or '')
        if error.child_kind == 'ReferenceUnverified':
            return ActorUnavailable('reference_unverified', 'engine refused the reference bytes: %s' % detail)
        if error.child_kind == 'ValueError' and 'Invalid actor synthesis request' in detail:
            return ActorUnavailable('text_invalid', detail)
        if error.child_kind == 'ValueError' and 'Actor reference' in detail:
            return ActorUnavailable('reference_invalid', detail)
        return ActorUnavailable('output_invalid', str(error), fault_code=error.fault_code)

    def _synthesize_once(self, transport, request, stop, meta):
        try:
            response = transport.synthesize(request, cancelled=stop, token=self, job_id=meta['job_id'],
                                            deadline=meta['deadline'], priority=meta['priority'])
        except (InterruptedError, DeadlineExpired, EngineError):
            raise
        except HostHold as error:
            if error.fault_kind == 'cancelled':
                raise ActorCancelled(str(error), fault_code=error.fault_code) from error
            if error.fault_kind == 'fatal':
                raise ActorUnavailable('runtime_failed', str(error), fault_code=error.fault_code) from error
            raise ActorBusy('movie Vox engine held: %s' % (error,), fault_code=error.fault_code) from error
        except ActorUnavailable:
            raise
        except Exception as error:   # noqa: BLE001
            if stop():
                raise ActorCancelled('render cancelled') from error
            raise wrap_fault(error, fault_code='vox_transport_failed') from error
        if not isinstance(response, dict) or response.get('receipt') is None:
            raise ActorUnavailable('output_invalid', 'engine returned no take')
        mode = (response.get('receipt') or {}).get('cloning_mode')
        expected_mode = 'reference+transcript' if request.get('reference_text') else 'reference-only'
        if mode != expected_mode:
            raise ActorUnavailable('runtime_failed', 'engine cloning mode %r, expected %r' % (mode, expected_mode),
                                   fault_code='vox_cloning_mode_mismatch')
        return response

    def _validate_take(self, part, response):
        """Natural-duration 48 kHz mono FLOAT take; nothing is resampled, sped up, padded or truncated."""
        try:
            import soundfile as sf
        except (ImportError, OSError) as error:
            raise ActorUnavailable('runtime_failed', 'soundfile runtime unavailable: %s' % error,
                                   fault_code='vox_audio_runtime_missing') from error
        if not part.is_file():
            raise ActorUnavailable('output_invalid', 'engine wrote no take')
        try:
            info = sf.info(str(part))
            audio, rate = sf.read(str(part), dtype='float32', always_2d=True)
        except Exception as error:   # noqa: BLE001
            raise ActorUnavailable('output_invalid', 'take unreadable: %s' % (error,)) from error
        if info.format != 'WAV' or info.subtype != 'FLOAT' or rate != TARGET_RATE or audio.shape[1] != 1:
            raise ActorUnavailable('output_invalid', 'take is not 48 kHz mono FLOAT WAV')
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise ActorUnavailable('output_invalid', 'empty or non-finite audio')
        seconds = audio.shape[0] / float(TARGET_RATE)
        if seconds > float(self.options['max_output_seconds']):
            raise ActorUnavailable('output_invalid', 'take exceeds max_output_seconds')
        if float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) <= 1e-4:
            raise ActorUnavailable('output_invalid', 'silent take; the cue keeps its original audio')
        # Token cap (review P3-7): the child receipt's stream_chunks is exact (one frame per PCM event,
        # stream_frames_per_event 1); the 0.16 s/token duration bound stays as a fallback when it is missing.
        chunks = (response.get('receipt') or {}).get('stream_chunks')
        cap_tokens = approved.SETTINGS['max_tokens']
        if isinstance(chunks, int) and chunks >= cap_tokens:
            raise ActorUnavailable('output_invalid', 'take reached max_tokens (%d chunks); possibly truncated' % chunks)
        cap = cap_tokens * approved.SECONDS_PER_TOKEN_EST
        if seconds >= cap - approved.SECONDS_PER_TOKEN_EST:
            raise ActorUnavailable('output_invalid', 'take reached the max_tokens cap (%.2fs); possibly truncated' % cap)
        reported = response.get('seconds')
        if reported is not None and abs(float(reported) - seconds) > 1.0 / TARGET_RATE:
            raise ActorUnavailable('output_invalid', 'take length differs from the engine report')
        return seconds

    @staticmethod
    def _normalize_take(part):
        """One constant gain on a new validated take, before its cache hash/publication.

        Target -3 dBFS peak, at most +40 dB boost. Existing RMS silence admission remains in _validate_take;
        the extra peak floor refuses near-silence rather than amplifying it. No tempo, trimming or limiting.
        """
        import soundfile as sf
        audio, rate = sf.read(str(part), dtype='float32', always_2d=True)
        info = sf.info(str(part))
        if (rate != TARGET_RATE or info.format != 'WAV' or info.subtype != 'FLOAT'
                or audio.shape[1] != 1 or audio.size == 0 or not np.all(np.isfinite(audio))):
            raise ActorUnavailable('output_invalid', 'invalid take at level-normalization boundary')
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if peak < LEVEL_MIN_PEAK or rms <= 0:
            raise ActorUnavailable('output_invalid', 'near-silent take; normalization refused')
        input_peak_dbfs = 20.0 * float(np.log10(peak))
        wanted_gain_db = LEVEL_TARGET_DBFS - input_peak_dbfs
        applied_gain_db = min(LEVEL_MAX_BOOST_DB, wanted_gain_db)
        scale = np.float32(10.0 ** (applied_gain_db / 20.0))
        normalized = np.multiply(audio, scale, dtype=np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ActorUnavailable('output_invalid', 'non-finite normalized take')
        input_sha256 = _file_hash(part)
        temporary = part.with_name(part.stem + '.normalized.wav')
        try:
            sf.write(str(temporary), normalized, rate, format='WAV', subtype='FLOAT')
            written = sf.info(str(temporary))
            if (written.frames != audio.shape[0] or written.samplerate != rate or written.channels != 1
                    or written.format != 'WAV' or written.subtype != 'FLOAT'):
                raise ActorUnavailable('output_invalid', 'normalization changed audio geometry')
            os.replace(temporary, part)
        finally:
            _unlink(temporary)
        return {
            'policy': LEVEL_POLICY, 'input_sha256': input_sha256,
            'input_peak_dbfs': round(input_peak_dbfs, 6),
            'input_rms_dbfs': round(20.0 * float(np.log10(rms)), 6),
            'applied_gain_db': round(applied_gain_db, 6),
            'output_peak_dbfs': round(20.0 * float(np.log10(float(np.max(np.abs(normalized))))), 6),
            'target_peak_dbfs': LEVEL_TARGET_DBFS, 'max_boost_db': LEVEL_MAX_BOOST_DB,
            'boost_limited': wanted_gain_db > LEVEL_MAX_BOOST_DB,
            'frames': int(audio.shape[0]), 'sample_rate': rate, 'channels': 1, 'subtype': 'FLOAT'}

    def close(self):
        """Cancel only this renderer's own in-flight and queued requests; the engine stays warm. The renderer can be
        used again afterwards (as BreezeActorRenderer)."""
        self._epoch += 1
        with self._jobs_lock:
            jobs = list(self._jobs)
        for job in jobs:
            job.cancel('renderer_closed')
        transport = self._transport
        if transport is not None and hasattr(transport, 'cancel'):
            transport.cancel(token=self, reason='renderer_closed')
