"""Independent actor cloning with VoxCPM2 and the audio.cpp PCM callback ABI.

This owns its model/session. It never connects to Sem's speech worker or changes
Sem's accepted reference. Native binaries and actor references stay external.
"""
from __future__ import annotations
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time


class ModelConfig(C.Structure):
    _fields_ = [('family', C.c_char_p), ('config', C.c_char_p), ('weight', C.c_char_p), ('override', C.c_char_p)]


class BackendConfig(C.Structure):
    _fields_ = [('backend', C.c_char_p), ('device', C.c_int), ('threads', C.c_int)]


PCM_CALLBACK = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int, C.c_void_p)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ActorVoice:
    def __init__(self, manifest_path, *, backend='cpu', threads=2, max_reference_seconds=15, budget=None):
        if os.name != 'nt' or backend not in ('cpu', 'cuda') or threads not in (1, 2, 4):
            raise ValueError('Unsupported actor voice runtime configuration')
        self.registry = C.c_void_p()
        self.model = C.c_void_p()
        self.session = C.c_void_p()
        self.closed = False
        self.budget = budget
        if backend == 'cuda':
            if budget is None:
                raise ValueError('GPU actor cloning requires a resource admission guard')
            budget.admit_gpu()
        self.max_reference_seconds = max_reference_seconds
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
        self.quantization = manifest.get('quantization', 'Q8_0')
        self.model_sha256 = manifest['modelSha256']
        library = Path(manifest['library']).resolve()
        for entry in manifest['artifacts']:
            path = library.parent / entry['file']
            if path.parent != library.parent or sha256(path) != entry['sha256']:
                raise ValueError('Actor runtime artifact changed')
        if sha256(manifest['model']) != manifest['modelSha256']:
            raise ValueError('Actor model artifact changed')
        self.dll_directories = [os.add_dll_directory(str(library.parent))]
        self.dll_directories += [os.add_dll_directory(str(p)) for p in manifest.get('dependency_directories', [])]
        self.backend_library = C.CDLL(str(library.with_name('ggml.dll')))
        if backend == 'cuda':
            self.cuda_plugin = C.CDLL(str(library.with_name('ggml-cuda.dll')))
        self.backend_library.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
        self.backend_library.ggml_backend_load_all_from_path.restype = None
        self.backend_library.ggml_backend_load_all_from_path(str(library.parent).encode())
        self.lib = C.CDLL(str(library))

        def bind(name, result, *args):
            fn = getattr(self.lib, 'audiocpp_' + name)
            fn.restype, fn.argtypes = result, list(args)
            return fn
        self.error = bind('last_error', C.c_char_p)
        registry_create = bind('registry_create', C.c_int, C.c_char_p, C.POINTER(C.c_void_p))
        model_load = bind('model_load', C.c_int, C.c_void_p, C.c_char_p, C.POINTER(ModelConfig), C.c_void_p, C.POINTER(C.c_void_p))
        options_create = bind('options_create', C.c_void_p)
        options_set = bind('options_set', C.c_int, C.c_void_p, C.c_char_p, C.c_char_p)
        options_free = bind('options_free', None, C.c_void_p)
        session_create = bind('session_create', C.c_int, C.c_void_p, C.c_char_p, C.c_char_p, C.POINTER(BackendConfig), C.c_void_p, C.POINTER(C.c_void_p))
        self.request_create = bind('request_create', C.c_void_p)
        self.request_text = bind('request_set_text', C.c_int, C.c_void_p, C.c_char_p, C.c_char_p)
        self.request_option = bind('request_set_option', C.c_int, C.c_void_p, C.c_char_p, C.c_char_p)
        self.request_audio = bind('request_set_audio', C.c_int, C.c_void_p, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int)
        self.request_voice = bind('request_set_voice_audio', C.c_int, C.c_void_p, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int)
        self.generate = bind('sem_stream_generate', C.c_int, C.c_void_p, C.c_void_p, PCM_CALLBACK, C.c_void_p)
        self.request_free = bind('request_free', None, C.c_void_p)
        self.session_free = bind('session_free', None, C.c_void_p)
        self.model_free = bind('model_free', None, C.c_void_p)
        self.registry_free = bind('registry_free', None, C.c_void_p)
        try:
            self.check(registry_create(None, C.byref(self.registry)))
            self.check(model_load(self.registry, manifest['model'].encode(), C.byref(ModelConfig(b'voxcpm2', None, None, None)), None, C.byref(self.model)))
            options = options_create()
            try:
                capacity = ((int(max_reference_seconds * 16000) + 2559) // 2560) * 2560
                for key, value in {'weight_type': 'native', 'audiovae_weight_type': 'native',
                        'audiovae_encoder_sample_capacity': capacity, 'mem_saver': 'true', 'prompt_cache_slots': 1}.items():
                    self.check(options_set(options, ('voxcpm2.' + key).encode(), str(value).encode()))
                self.check(session_create(self.model, b'tts', b'streaming', C.byref(BackendConfig(backend.encode(), 0, threads)), options, C.byref(self.session)))
            finally:
                options_free(options)
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.error().decode('utf-8', 'replace'))

    def synthesize(self, text, reference_path, reference_text='', *, callback=None, cancelled=lambda: False,
                   max_tokens=350, steps=10, guidance=1.6, seed=42, progress=None):
        def report_progress(stage):
            # Diagnostics receive stage names only and cannot interrupt synthesis.
            if progress is not None:
                try:
                    progress(stage)
                except Exception:
                    pass

        report_progress('entry')
        import numpy as np
        import soundfile as sf
        report_progress('imports_ready')
        if self.closed or not isinstance(text, str) or not text.strip() or len(text) > 1500:
            raise ValueError('Invalid actor synthesis request')
        caller_cancelled = cancelled
        cancelled = lambda: caller_cancelled() or bool(self.budget and self.budget.should_yield())
        report_progress('budget_check_entry')
        yielded = cancelled()
        report_progress('budget_check_return')
        if yielded:
            raise InterruptedError('Actor synthesis yielded before generation: '+str(self.budget.reason if self.budget else 'cancelled'))
        report_progress('reference_read_entry')
        reference, rate = sf.read(reference_path, dtype='float32', always_2d=True)
        report_progress('reference_read_return')
        if not 0.25 <= len(reference) / rate <= self.max_reference_seconds or reference.shape[1] not in (1, 2):
            raise ValueError('Actor reference must contain 0.25 to configured maximum seconds')
        if not np.isfinite(reference).all() or np.max(np.abs(reference)) < 1e-5:
            raise ValueError('Actor reference is silent or invalid')
        reference = np.ascontiguousarray(reference)
        report_progress('request_configuration_entry')
        request = self.request_create()
        chunks, errors, first = [], [], None
        first_pcm_reported = False
        start = time.perf_counter()
        try:
            self.check(self.request_text(request, text.encode('utf-8'), None))
            audio = (reference.ctypes.data_as(C.POINTER(C.c_float)), len(reference), rate, reference.shape[1])
            self.check(self.request_voice(request, *audio))
            options = {'seed': seed, 'max_tokens': max_tokens, 'min_tokens': 2, 'guidance_scale': guidance,
                       'num_inference_steps': steps, 'retry_badcase': 'false', 'stream_frames_per_event': 1, 'stream_left_context': 3}
            if reference_text:
                self.check(self.request_audio(request, *audio))
                options['reference_text'] = reference_text
            for key, value in options.items():
                self.check(self.request_option(request, key.encode(), str(value).encode('utf-8')))
            report_progress('request_configuration_return')

            @PCM_CALLBACK
            def sink(samples, frames, sample_rate, channels, _):
                nonlocal first, first_pcm_reported
                try:
                    if not first_pcm_reported:
                        first_pcm_reported = True
                        report_progress('first_pcm')
                    if cancelled():
                        return 0
                    if sample_rate != 48000 or channels != 1 or not 0 < frames <= 48000:
                        raise ValueError('Invalid actor PCM geometry')
                    chunk = np.ctypeslib.as_array(samples, shape=(frames,)).copy()
                    if not np.isfinite(chunk).all():
                        raise ValueError('Actor PCM is not finite')
                    if first is None:
                        first = time.perf_counter() - start
                    chunks.append(chunk)
                    if callback:
                        callback(chunk)
                    return 0 if cancelled() else 1
                except BaseException as exc:
                    errors.append(exc)
                    return 0
            report_progress('native_generate_entry')
            result = self.generate(self.session, request, sink, None)
            report_progress('native_generate_return')
            if errors:
                raise errors[0]
            if cancelled():
                raise InterruptedError('Actor synthesis yielded: '+str(self.budget.reason if self.budget else 'cancelled'))
            self.check(result)
            if not chunks:
                raise ValueError('Actor synthesis returned no audio')
            audio = np.concatenate(chunks)
            elapsed = time.perf_counter() - start
            return audio, {'engine': 'VoxCPM2/audio.cpp', 'quantization': self.quantization,
                           'model_sha256': self.model_sha256, 'reference_sha256': sha256(reference_path),
                           'first_pcm_seconds': first, 'elapsed_seconds': elapsed, 'audio_seconds': len(audio) / 48000,
                           'real_time_factor': elapsed / (len(audio) / 48000), 'stream_chunks': len(chunks),
                           'cloning_mode': 'reference+transcript' if reference_text else 'reference-only'}
        finally:
            self.request_free(request)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if hasattr(self, 'session_free'):
            self.session_free(self.session)
            self.model_free(self.model)
            self.registry_free(self.registry)
        for handle in getattr(self, 'dll_directories', []):
            handle.close()
