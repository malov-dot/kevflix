"""Automatic full-session X-Translator/X-Voice movie producer.

Only isolated vocals enter the intact upstream controller. Subtitle text and
manual speaker/reference labels never enter this path. Claim-owner code journals
buffered events and publishes full immutable PCM through the shared renderer.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import wave

from .media import RATE, atomic_json, extract_audio, file_hash, fingerprint, resolve_channel_layout, wav_format
from .movie_faults import MovieConfigurationError, fault_record
from .generic_dub import VERSION, GAP_SECONDS, _require, _clip, _union, _seconds, _check, render_and_publish
from .xt_client import XTSession, TERMINAL, recoverable_session

XT_VERSION = 'xt-dub-v1'
SUPPORTED_LANGUAGES = frozenset('af zh en fr pt es ja ru ko th it de vi id pl cs nl'.split())
VAD_BASIS = 'energy-upstream-tracker-params-v1'
FEED_NORMALIZATION = 'xt-feed-active-rms-v1'


def _condition_feed(samples):
    """Bounded analysis-feed gain; never alter mix or voice-level reference stems."""
    import numpy as np
    _require(samples.ndim == 1 and len(samples) > 0 and np.isfinite(samples).all(),
             'xt_feed_geometry', 'Conditioning needs finite nonempty mono16k PCM')
    values = samples.astype(np.float64)
    frame_samples = 640
    powers = np.array([np.mean(values[i:i + frame_samples] ** 2)
                       for i in range(0, len(values), frame_samples)])
    count = max(1, math.ceil(.2 * len(powers)))
    selected = np.argsort(powers, kind='stable')[-count:]
    active_before = float(np.sqrt(np.mean(powers[selected])))
    gain = min(31.6, max(1., .1 / active_before)) if active_before > 0 else 1.
    amplified = values * gain
    limited_samples = int(np.count_nonzero(np.abs(amplified) > .99))
    limited = np.clip(amplified, -.99, .99)
    encoded = np.rint(limited * 32768.).astype('<i2')
    actual = encoded.astype(np.float64) / 32768.
    powers_after = [np.mean(actual[i * frame_samples:(i + 1) * frame_samples] ** 2) for i in selected]
    return encoded.tobytes(), {'version': FEED_NORMALIZATION, 'sample_rate': 16000,
        'frames_before': len(values), 'frames_after': len(encoded), 'frame_ms': 40,
        'active_fraction': .2, 'active_frame_count': count, 'frame_count': len(powers),
        'active_rms_basis': 'sqrt(mean(loudest20percent_frame_RMS_squared)); equal frame weights; ceil count',
        'last_partial_frame': 'included with equal frame weight', 'gain': gain, 'gain_db': 20. * math.log10(gain),
        'target_active_rms': .1, 'gain_min': 1., 'gain_max': 31.6,
        'active_rms_before': active_before, 'active_rms_after': float(np.sqrt(np.mean(powers_after))),
        'after_basis': 'same selected frames after clipping and PCM16 quantization',
        'limited_samples': limited_samples, 'clip_limit': .99, 'silent_input': active_before == 0.,
        'scope': 'ASR/speaker/energy feed only; raw vocals, background, processing and mix references unchanged'}


def _energy_regions(pcm, begin):
    """CPU energy coverage, not Silero or a claim of linguistic speech accuracy."""
    import numpy as np
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.
    frame_samples, noise_floor = 640, -60.
    regions = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset:offset + frame_samples]
        rms = float(np.sqrt(np.mean(frame * frame)))
        dbfs = 20. * math.log10(rms + 1e-8)
        if dbfs >= max(-42., noise_floor + 10.):
            regions.append([begin + offset / 16000., begin + (offset + len(frame)) / 16000.])
        elif dbfs <= -36.:
            noise_floor = .98 * noise_floor + .02 * dbfs
    return _union(regions), {'basis': VAD_BASIS, 'device': 'cpu', 'frame_ms': 40,
        'silence_dbfs': -42., 'margin_db': 10., 'initial_noise_floor_db': -60.,
        'final_noise_floor_db': noise_floor, 'noise_floor_scope': 'reset per central window',
        'last_partial_frame': 'included', 'padding_ms': 0, 'silence_merge_ms': 0,
        'coverage_comparable_to_silero': False,
        'disclosure': 'Energy-region denominator replaces Silero; coverage is not comparable to previous Silero runs'}


def _stem_window(config, store, job_id, revision, window, owner, services, spec):
    import numpy as np
    import soundfile as sf
    from . import first_pass_parent
    cached = store.xt_stem(job_id, revision, window)
    if cached is not None:
        _require(cached['source_fingerprint'] == spec['source_fingerprint'] and
                 cached['engine_fingerprint'] == spec['engine_fingerprint'], 'xt_stem_identity', 'Cached stem identity changed')
        for key in ('vocals', 'background', 'processing', 'feed_pcm'):
            _require(file_hash(cached[key]['path']) == cached[key]['sha256'], 'xt_stem_changed', key)
        _require(cached.get('vad_basis') == VAD_BASIS, 'xt_vad_basis_changed', 'Cached denominator is not energy based')
        if getattr(services, 'last_window_memory', None):
            return cached
        # A fresh resident needs measured current allocation; old cached bytes
        # remain authoritative, but their historical reserved memory does not.
    segment = float(spec['segment_seconds'])
    begin, end = window * segment, min((window + 1) * segment, float(spec['duration']))
    _require(end > begin, 'xt_window_outside_movie', str(window))
    origin, stop = max(0., begin - 6.), min(float(spec['duration']), end + 6.)
    directory = Path(config.cache) / job_id / 'xt-stems' / ('window-%06d-claim-%s-%s' % (window, store.claim_token, uuid.uuid4().hex))
    directory.mkdir(parents=True, exist_ok=False)
    stream = next((s for s in spec['media']['streams'] if s.get('index') == spec['audio_index']), None)
    _require(stream is not None, 'xt_source_stream_missing', 'Selected source audio is missing')
    channels = int(stream.get('channels', 0))
    layout = stream.get('channel_layout') if channels == 6 else {1: 'mono', 2: 'stereo'}.get(channels)
    _require(layout in ('mono', 'stereo', '5.1', '5.1(side)'), 'xt_source_layout', 'Unsupported source layout')
    decoded_path = directory / 'source.wav'
    _check(store, owner)
    extract_audio(config, owner, spec['source'], spec['audio_index'], origin, stop - origin,
                  decoded_path, channels=channels, preserve_layout=layout)
    decoded, rate = sf.read(decoded_path, dtype='float32', always_2d=True)
    evidence = resolve_channel_layout(decoded.shape[1], source=stream, wav=wav_format(decoded_path))
    _require(rate == RATE and origin + len(decoded) / RATE >= end - 1 / RATE, 'xt_source_geometry', 'Incomplete window')
    bundle = first_pass_parent.run(config, owner, folder=directory, original_source_wav=decoded_path,
        source_layout_evidence=evidence, origin=origin, frames=len(decoded), scope=None)
    parent = (bundle or {}).get('parent') or {}
    _require(parent.get('producer') == 'resident' and parent.get('device') == 'cuda'
             and parent.get('admission') == 'movie-lease', 'xt_separation_not_cuda', 'Resident CUDA separation required')
    if cached is not None:
        return cached
    left, right = round((begin - origin) * RATE), round((end - origin) * RATE)
    _require(0 <= left < right <= len(decoded), 'xt_central_frames_missing', 'No shortened central stem may be cached')
    payload = {'version': XT_VERSION, 'window': window, 'start': begin, 'end': end,
        'source_fingerprint': spec['source_fingerprint'], 'engine_fingerprint': spec['engine_fingerprint'],
        'analysis_origin': origin, 'analysis_end': stop, 'parent': parent, 'first_pass_key': bundle.get('key')}
    for name, source_name in (('vocals', 'raw_vocals'), ('background', 'background'), ('processing', 'processing')):
        source = bundle[source_name]
        _require(file_hash(source['wav']) == source['sha256'], 'xt_stem_changed', source_name)
        samples, sr = sf.read(source['wav'], dtype='float32', always_2d=True)
        _require(sr == RATE and samples.shape == (len(decoded), 2) and np.isfinite(samples).all(),
                 'xt_stem_geometry', source_name)
        path = directory / (name + '-central.wav')
        sf.write(path, samples[left:right], RATE, subtype='FLOAT')
        payload[name] = {'path': str(path), 'sha256': file_hash(path), 'frames': right - left, 'sample_rate': RATE}
    float_feed = directory / 'vocals-central-16k-float.wav'
    owner.run([config['ffmpeg'], '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
        '-i', payload['vocals']['path'], '-ar', '16000', '-ac', '1', '-c:a', 'pcm_f32le', str(float_feed)])
    feed_samples, feed_rate = sf.read(float_feed, dtype='float32', always_2d=True)
    _require(feed_rate == 16000 and feed_samples.shape[1] == 1, 'xt_feed_geometry', 'Need mono16k float input')
    pcm_bytes, conditioning = _condition_feed(feed_samples[:, 0])
    _require(len(pcm_bytes) == len(feed_samples) * 2, 'xt_feed_geometry', 'Conditioning changed PCM frame count')
    pcm = directory / 'vocals-central-16k.pcm'
    pcm.write_bytes(pcm_bytes)
    payload['feed_conditioning'] = {**conditioning, 'input_audio': str(float_feed),
        'input_sha256': file_hash(float_feed), 'output_audio': str(pcm), 'output_sha256': file_hash(pcm)}
    payload['feed_pcm'] = {'path': str(pcm), 'sha256': file_hash(pcm), 'sample_rate': 16000,
                           'frames': pcm.stat().st_size // 2}
    payload['vad_regions'], payload['vad_receipt'] = _energy_regions(pcm.read_bytes(), begin)
    payload['vad_receipt']['feed_normalization_version'] = FEED_NORMALIZATION
    payload['vad_receipt']['feed_gain_db'] = conditioning['gain_db']
    payload['vad_receipt']['disclosure'] += '; conditioned feed energy differs from earlier unconditioned energy runs'
    payload['vad_regions'] = _clip(payload['vad_regions'], begin, end)
    payload['vad_basis'] = VAD_BASIS
    _check(store, owner)
    store.xt_stem_put(job_id, revision, window, payload)
    return payload


def _refine_speech_span(store, job_id, revision, outcome, spec):
    """Bind timing once to all prepared conditioned-energy windows before journaling."""
    source = _union(outcome['source_spans'])
    segment = float(spec['segment_seconds'])
    indexes = sorted({index for a, b in source
                      for index in range(math.floor(a / segment), math.ceil(b / segment))})
    energy, evidence = [], []
    for index in indexes:
        stem = store.xt_stem(job_id, revision, index)
        _require(stem is not None and stem.get('window') == index
                 and stem.get('source_fingerprint') == spec['source_fingerprint']
                 and stem.get('engine_fingerprint') == spec['engine_fingerprint']
                 and stem.get('vad_basis') == VAD_BASIS
                 and stem.get('feed_conditioning', {}).get('version') == FEED_NORMALIZATION,
                 'xt_speech_span_evidence_missing', 'Prepared conditioned-energy window missing or changed: ' + str(index))
        energy.extend(_clip(stem['vad_regions'], stem['start'], stem['end']))
        evidence.append({'window': index, 'start': stem['start'], 'end': stem['end'],
                         'feed_sha256': stem['feed_pcm']['sha256']})
    speech = _union([part for a, b in source for part in _clip(energy, a, b)])
    outcome.update(controller_span=[outcome['film_start'], outcome['film_end']], controller_source_spans=source,
        source_spans=speech, onset=speech[0][0] if speech else outcome['film_start'],
        speech_refinement={'version': 'xt-conditioned-speech-span-v1', 'vad_basis': VAD_BASIS,
            'feed_normalization_version': FEED_NORMALIZATION, 'windows': evidence,
            'speech_seconds': _seconds(speech), 'status': 'speech_intersection' if speech else 'no_detected_speech_in_span'})


def _journal_events(store, job_id, revision, session, state, directory, spec, owner):
    """Only this claimed owner writes; the websocket reader never touches Store."""
    final = state.setdefault('final_outcomes', {})
    segment_seconds = float(spec['segment_seconds'])
    for event in session.take_events():
        _check(store, owner)
        uid = str(event.get('seg_uid', '@session'))
        observed = dict(event)
        if 'status' in observed:
            observed['upstream_status'] = observed.pop('status')
            observed['status'] = 'observed'
        store.xt_journal_put(job_id, revision, session.session_uid, uid, observed)
        if event.get('event') == 'source_language_detected':
            state['language'] = {'language': event.get('language_code'), 'raw': event.get('language_raw'),
                'basis': event.get('language_source'), 'seg_uid': uid, 'session_uid': session.session_uid,
                'subtitles_used': False}
        if uid == '@session' or uid in final or event.get('status') not in TERMINAL:
            continue
        outcome = {key: value for key, value in event.items() if key != 'event'}
        outcome['event'] = 'terminal_outcome'
        state['language'] = {'language': outcome.get('language_code'), 'raw': outcome.get('language_raw'),
            'basis': outcome.get('language_source'), 'seg_uid': uid, 'session_uid': session.session_uid,
            'subtitles_used': False}
        if outcome.get('film_start') is None:
            outcome.update(status='pad_output_discarded', error='No mapped movie audio; padding is never credited')
        else:
            owner_index = math.floor(float(outcome['film_start']) / segment_seconds)
            owned_chunk = store.chunk(job_id, owner_index)
            outcome['owner_chunk'] = owner_index
            if owned_chunk and owned_chunk.get('state') in ('ready', 'approved'):
                outcome.update(status='owner_chunk_already_published', error='Late controller outcome cannot replace ready audio')
            elif outcome.get('mapping_error'):
                outcome.update(status='empty_output', error=outcome['mapping_error'])
        if outcome['status'] in ('ok', 'source_english'):
            _refine_speech_span(store, job_id, revision, outcome, spec)
        if outcome['status'] == 'ok':
            frame = session.audio_frame(uid)
            _require(len(frame) == outcome['tts_bytes'] and hashlib.sha256(frame).hexdigest() == outcome['tts_sha256'],
                     'xt_frame_changed', 'Verified upstream frame changed before persistence')
            name = hashlib.sha256((session.session_uid + ':' + uid).encode()).hexdigest()[:24]
            path = directory / (name + '-upstream.wav')
            _require(not path.exists(), 'xt_pcm_exists', 'Never overwrite upstream full PCM')
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(outcome['tts_sample_rate'])
                output.writeframes(frame)
            outcome.update(tts_wav=str(path), tts_wav_sha256=file_hash(path),
                           frames=len(frame) // 2, seconds=len(frame) / (2 * outcome['tts_sample_rate']))
        store.xt_journal_put(job_id, revision, session.session_uid, uid, outcome)
        final[uid] = outcome
        session.release_audio(uid)
        if outcome.get('film_end') is not None:
            state['terminal_film'] = max(float(state.get('terminal_film', outcome['film_end'])), float(outcome['film_end']))


def process_xt_chunk(config, store, job_id, index, owner, services, xt_services, xt_state):
    import numpy as np
    import soundfile as sf
    started = time.monotonic()
    _check(store, owner)
    job = store.job(job_id)
    spec = job['payload']
    _require(spec['speech_engine'] == 'xtranslator' and fingerprint(spec['source']) == spec['source_fingerprint']
             and config.engine_fingerprint() == spec['engine_fingerprint'], 'xt_job_identity', 'Source/engine changed')
    revision, segment_seconds = int(spec.get('revision', 1)), float(spec['segment_seconds'])
    identity = (job_id, revision, spec['engine_fingerprint'])
    _require(xt_state.get('identity') in (None, identity), 'xt_runtime_identity', 'Root must close the previous runtime')
    xt_state.update(identity=identity, job_id=job_id)
    existing_session = xt_state.get('session')
    if existing_session is not None:
        existing_session.owner = owner
        existing_session.check()  # do not start more feeder work after an already observed session loss
    chunk = store.chunk(job_id, index)['payload']
    begin, end = float(chunk['start']), min(float(chunk['end']), float(spec['duration']))
    directory = Path(config.cache) / job_id / ('chunk-%06d' % index) / ('xt-claim-%s-%s' % (store.claim_token, uuid.uuid4().hex))
    directory.mkdir(parents=True, exist_ok=False)
    result = {'generic_version': VERSION, 'xt_version': XT_VERSION, 'speech_engine': 'xtranslator', 'engine': 'xtranslator',
        'job_id': job_id, 'revision': revision, 'chunk': index, 'start': begin, 'end': end, 'origin': begin,
        'source_fingerprint': spec['source_fingerprint'], 'source_audio_index': spec['audio_index'],
        'engine_fingerprint': spec['engine_fingerprint'], 'claim_token': int(store.claim_token),
        'mode': 'generic-English-candidate', 'generic_takes': [], 'generic_failures': [], 'generic_original_intervals': [],
        'issues': [], 'accuracy': 'unmeasured', 'human_listening_acceptance': False, 'subtitle_role': 'not-consumed'}
    data = getattr(config, 'data', config)
    configured_language = str(data.get('xt_source_language', 'auto')).lower()
    _require(configured_language == 'auto' or configured_language in SUPPORTED_LANGUAGES,
             'xt_source_language_invalid', 'Configured language is not supported by upstream LMT')
    result['configured_source_language'] = configured_language
    extend_max = int(data.get('xt_extend_max', 2))
    lead = float(data.get('xt_lead_seconds', 60.))
    barrier_seconds = float(data.get('xt_barrier_timeout_seconds', 120.))
    _require(0 <= extend_max <= 8 and math.isfinite(lead) and lead >= segment_seconds
             and math.isfinite(barrier_seconds) and barrier_seconds >= 1.,
             'xt_stream_bounds_invalid', 'Need bounded lookahead, >=one chunk lead and a finite barrier timeout')
    barrier_seconds = min(120., barrier_seconds)
    windows = {}
    def window(number):
        if number not in windows:
            windows[number] = _stem_window(config, store, job_id, revision, number, owner, services, spec)
        return windows[number]
    try:
        current = window(index)
        result['generic_vad_regions'] = current['vad_regions']
        result['generic_vad_detector'] = current['vad_receipt']
        result['xt_feed_conditioning'] = current['feed_conditioning']
        result['separation'] = {'parent': current['parent'], 'bed_basis': 'demucs-background-full-chunk', 'bed_gain': 1.}
        result['vad_basis'] = VAD_BASIS
        result['coverage_denominator_disclosure'] = current['vad_receipt']['disclosure']
        language = xt_state.get('language')
        original_intervals = []
        processing = None
        frontier = None
        session = xt_state.get('session')
        if session is not None:
            session.owner = owner
            if session.failed:
                session.check()  # typed fault; root alone closes/reclaims within the persisted reset budget
        if session is None and not current['vad_regions']:
            kind = 'language_pending_no_speech'
            uid = xt_state.setdefault('logical_uid', uuid.uuid4().hex)
            if xt_state.get('activated_uid') != uid:
                store.xt_journal_put(job_id, revision, uid, '@session', {'event': 'session_open', 'kind': kind})
                xt_state['activated_uid'] = uid
            result.update(xt_session_uid=uid, xt_session_kind=kind, source_language='pending',
                          detected_language=language, mode='no-dialogue',
                          backend_identity={'upstream_started': False}, speaker_count=0, speaker_ids=[])
            source_file = current['background']
            bed, _ = sf.read(source_file['path'], dtype='float32', always_2d=True)
            vocals, _ = sf.read(current['vocals']['path'], dtype='float32', always_2d=True)
            lines = []
            context_stop = end
        else:
            xt_services.set_demucs_memory(services.last_window_memory)
            if session is None:
                session = XTSession(config, owner)
                uid = session.open(xt_services, job_id, revision, configured_language)
                xt_state.update(session=session, final_outcomes={}, terminal_film=begin, next_window=index)
                store.xt_journal_put(job_id, revision, uid, '@session', {'event': 'session_open', 'kind': 'upstream',
                    'language': language, 'speaker_state_reset': int(xt_state.get('reset_count', 0))})
                xt_state['activated_uid'] = uid
            else:
                endpoints = xt_services.ensure(owner)
                _require(endpoints.get('generation') is not None, 'xt_generation_missing', 'Service generation tag is required')
                if endpoints['generation'] != session.endpoints.get('generation'):
                    raise recoverable_session('generation_changed', 'Service generation changed; coordinator must close and reset session')
            # A genuine seek reuses the upstream controller and speaker buckets,
            # but separates source epochs with unmapped silence and a real drain.
            if xt_state.get('last_chunk') is not None and index != xt_state['last_chunk'] + 1:
                session.pad(float(config['xt_seek_pad_seconds']))
                session.drain(barrier_seconds)
                _journal_events(store, job_id, revision, session, xt_state, directory, spec, owner)
                session.feed_epoch += 1
                xt_state['next_window'] = index
            acknowledged = None
            for number in range(index, index + extend_max + 1):
                if number * segment_seconds >= float(spec['duration']):
                    break
                row = window(number)
                if not session.fed_range(row['start'], row['end']):
                    progress = max(float(xt_state.get('terminal_film', begin)), session.frontier_film() or begin)
                    if row['end'] > progress + lead + 1e-6:
                        acknowledged = session.drain(barrier_seconds)
                        _journal_events(store, job_id, revision, session, xt_state, directory, spec, owner)
                        progress = max(float(xt_state.get('terminal_film', begin)), session.frontier_film() or begin)
                        if row['end'] > progress + lead + 1e-6:
                            break
                    session.feed(row['start'], Path(row['feed_pcm']['path']).read_bytes())
                if row['end'] >= float(spec['duration']) - 1 / RATE:
                    session.pad(float(config['xt_seek_pad_seconds']))
                acknowledged = session.drain(barrier_seconds)
                _journal_events(store, job_id, revision, session, xt_state, directory, spec, owner)
                if (session.frontier_film() or -1) >= end - 1e-6 and not session.open_uids():
                    break
            frontier = session.frontier_film()
            if frontier is None or frontier < end - 1e-6:
                start = max(begin, frontier if frontier is not None else begin)
                failure = {'event': 'controller_timeout', 'status': 'controller_commit_timeout',
                           'source_spans': [[start, end]], 'film_start': start, 'film_end': end,
                           'owner_chunk': index, 'complete_credit': 0, 'extend_max': extend_max}
                store.xt_journal_put(job_id, revision, session.session_uid, 'timeout:%s' % index, failure)
                result['generic_failures'].append({'reason': failure['status'], **failure})
                xt_state['terminal_film'] = max(float(xt_state.get('terminal_film', begin)), end)
            result['xt_readiness'] = {'frontier_film': frontier, 'barrier': acknowledged,
                                      'open_seg_uids': session.open_uids(), 'source_fed_once': session.fed_range(begin, end)}
            language = xt_state.get('language')
            result.update(xt_session_uid=session.session_uid, source_language=(language or {}).get('language', 'pending'), detected_language=language,
                          backend_identity=xt_services.health(), feed_epoch=session.feed_epoch,
                          xt_server_session_id=session.server_session_id)
            final = xt_state['final_outcomes']
            lines, speakers = [], set()
            current_outcomes = []
            for uid, outcome in final.items():
                if outcome.get('speaker_id') is not None and outcome['speaker_id'] != '':
                    speakers.add(str(outcome['speaker_id']))
                if outcome.get('film_start') is not None and min(end, outcome['film_end']) > max(begin, outcome['film_start']):
                    current_outcomes.append(outcome)
                    if (outcome.get('speech_refinement') or {}).get('status') == 'no_detected_speech_in_span':
                        result['generic_failures'].append({'reason': 'no_detected_speech_in_span', 'seg_uid': uid,
                            'controller_span': outcome['controller_span'], 'source_spans': [],
                            'onset': outcome['onset'], 'complete_credit': 0,
                            'scope': 'No conditioned energy intersects this controller span; no speech source credit'})
                if outcome['status'] == 'source_english':
                    original_intervals.extend(outcome.get('source_spans', []))
                    continue
                if outcome.get('owner_chunk') != index:
                    continue
                if outcome['status'] != 'ok':
                    result['generic_failures'].append({'reason': outcome['status'], 'seg_uid': uid,
                        'source_spans': outcome.get('source_spans', []), 'detail': outcome.get('error'), 'complete_credit': 0})
                    continue
                ident = hashlib.sha256(json.dumps([session.session_uid, uid], separators=(',', ':')).encode()).hexdigest()[:24]
                lines.append({'id': ident, 'onset': outcome['onset'], 'start': outcome['film_start'],
                    'end': outcome['film_end'], 'source_spans': outcome['source_spans'], 'text': outcome['translated_text'],
                    'audio': outcome['tts_wav'], 'voice': {'engine': 'xvoice', 'speaker_id': outcome.get('speaker_id'),
                                                         'automatic_reference': True},
                    'ownership': {'owned': True, 'placement_floor': begin, 'reason': 'controller-source-start', 'seam_rescue': False},
                    'synthesis': {'audio': outcome['tts_wav'], 'sha256': outcome['tts_wav_sha256'],
                        'seconds': outcome['seconds'], 'sample_rate': outcome['tts_sample_rate'], 'frames': outcome['frames'],
                        'truncated': False, 'upstream_pcm_sha256': outcome['tts_sha256']},
                    'take_metadata': {'origin_session_uid': session.session_uid, 'seg_uid': uid,
                        **{key: outcome.get(key) for key in ('commit_basis', 'language_context',
                                                            'language_observation_id', 'reference_refresh')},
                        'feed_epoch': outcome.get('feed_epoch'), 'speaker_id': outcome.get('speaker_id'),
                        'controller_span': outcome['controller_span'],
                        'controller_source_spans': outcome['controller_source_spans'],
                        'speech_refinement': outcome['speech_refinement'],
                        'speech_engine': 'xtranslator', 'engine': 'xtranslator', 'xt_version': XT_VERSION}})
            original_intervals = _union(original_intervals)
            result['xt_language_outcomes'] = [{key: value for key, value in row.items()
                if key in ('seg_uid', 'status', 'language_code', 'language_raw', 'language_source', 'source_spans',
                           'commit_basis', 'language_context', 'language_observation_id', 'reference_refresh',
                           'controller_span', 'controller_source_spans', 'onset', 'speech_refinement', 'owner_chunk')}
                for row in current_outcomes]
            if (current_outcomes and not result['generic_failures']
                    and all(row['status'] == 'source_english' for row in current_outcomes)):
                result['mode'] = 'original-English'
            result.update(speaker_count=len(speakers), speaker_ids=sorted(speakers))
            context_index = max([index] + [math.floor((line['end'] - 1e-6) / segment_seconds) for line in lines])
            raw_parts, bed_parts = [], []
            for number in range(index, context_index + 1):
                row = window(number)
                raw_parts.append(sf.read(row['vocals']['path'], dtype='float32', always_2d=True)[0])
                bed_parts.append(sf.read(row['background']['path'], dtype='float32', always_2d=True)[0])
            vocals, bed = np.concatenate(raw_parts), np.concatenate(bed_parts)
            # The processing stem is restored only for native-Qwen English spans.
            processing = sf.read(current['processing']['path'], dtype='float32', always_2d=True)[0]
            result['xt_original_processing'] = dict(current['processing'])
            context_stop = begin + len(vocals) / RATE
            xt_state['last_chunk'] = index

        stored, prior = [], []
        for row in store.utterances(job_id, -1e18, 1e18):
            take = row['payload']
            if int(take.get('revision', -1)) != revision:
                continue
            _require(take.get('job_id') == job_id and take.get('generic_version') == VERSION
                     and take.get('origin_session_uid') and take['source_fingerprint'] == spec['source_fingerprint']
                     and file_hash(take['audio']) == take['sha256'], 'xt_carry_invalid', 'Committed take identity/bytes changed')
            stored.append(take)
            if int(take['owner_chunk']) < index and take.get('playable'):
                prior.append(take)
        ready_successors = [float(row['payload']['start']) for row in store.chunks(job_id)
                            if int(row['idx']) > index and row['state'] in ('ready', 'approved')]
        successor = min(ready_successors) if ready_successors else None
        # Published original-English decisions survive a session reset just as
        # published full takes do. Only ready same-revision receipts contribute.
        for row in store.chunks(job_id):
            recorded = row['payload']
            if row['state'] not in ('ready', 'approved') or int(recorded.get('revision', -1)) != revision:
                continue
            _check(store, owner)
            # job_id belongs to the Store row; publish_generic omits it from
            # payload kwargs. Follow the same unique-receipt contract as the
            # generic predecessor reader, retaining the full saved decisions.
            receipt = json.loads(Path(recorded['receipt_path']).read_text(encoding='utf-8-sig'))
            _require(row.get('job_id') == job_id and receipt.get('job_id') == job_id
                     and receipt.get('source_fingerprint') == spec['source_fingerprint']
                     and receipt.get('engine_fingerprint') == spec['engine_fingerprint']
                     and int(receipt.get('chunk', -1)) == int(row['idx'])
                     and all(receipt.get(key) == recorded.get(key) for key in
                             ('chunk', 'revision', 'generic_version', 'start', 'end', 'audio', 'sha256',
                              'source_fingerprint', 'engine_fingerprint',
                              'speech_engine', 'engine', 'xt_version', 'xt_session_uid',
                              'xt_language_outcomes', 'generic_original_intervals'))
                     and file_hash(recorded['audio']) == recorded['sha256'],
                     'xt_original_receipt_identity', 'Published source-English receipt or output changed')
            for outcome in receipt.get('xt_language_outcomes', []):
                if outcome.get('status') == 'source_english':
                    original_intervals.extend(outcome.get('source_spans', []))
        original_intervals = _union(original_intervals)
        if original_intervals and processing is None:
            processing = sf.read(current['processing']['path'], dtype='float32', always_2d=True)[0]
            result['xt_original_processing'] = dict(current['processing'])
        if result['mode'] == 'original-English' and any(take['end'] > begin and take['start'] < end for take in prior):
            result['mode'] = 'generic-English-candidate'
        result.update(speaker_state_reset_count=int(xt_state.get('reset_count', 0)),
                      xt_detected_language=language, xt_speaker_count=result.get('speaker_count', 0),
                      xt_reset_count=int(xt_state.get('reset_count', 0)), xt_receipt_path=str(directory / 'receipt.json'),
                      analysis_end=context_stop)
        store.xt_journal_put(job_id, revision, result['xt_session_uid'], '@session', {
            'event': 'session_state', 'language': language, 'source_language': result['source_language'],
            'speaker_ids': result.get('speaker_ids', []), 'reset_count': result['speaker_state_reset_count'],
            'chunk': index, 'feed_epoch': result.get('feed_epoch'), 'upstream_started': 'session' in xt_state})
        result['_render_context'] = {'directory': str(directory), 'spec': spec, 'started': started, 'stop': context_stop,
            'prior_good': prior, 'prior_sources': _union([p for take in prior for p in take['source_spans']]),
            'all_stored': stored, 'successor_boundary': successor,
            'successor_limit': successor - GAP_SECONDS if successor is not None else None,
            'placement_limit': ((frontier if frontier is not None else begin) if session is not None
                                and (frontier is None or frontier < float(spec['duration']) - 1 / RATE) else None),
            'processing': processing if original_intervals else None,
            'bed_basis': 'demucs-background-with-exact-source-English-intervals' if original_intervals else 'demucs-background-full-chunk'}
        return render_and_publish(config, store, job_id, index, owner, begin=begin, end=end, origin=begin,
            bed=bed, raw_vocals=vocals, vad_regions=current['vad_regions'], lines=sorted(lines, key=lambda line: line['onset']),
            result=result, original_intervals=original_intervals)
    except BaseException as error:
        result.pop('_render_context', None)
        atomic_json(directory / 'failure.json', {'xt_version': XT_VERSION, 'job_id': job_id, 'chunk': index,
            'error': str(error), 'error_type': type(error).__name__, 'fault': fault_record(error),
            'partial': result, 'ready_published': False})
        raise
