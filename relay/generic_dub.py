"""Serial generic English dubbing: independent speech detection, whole takes, fenced publication.

GPU inference is delegated to registered lease-owned services. This coordinator
module performs file/PCM work only. Coverage is published source-time coverage;
translation accuracy and listening quality are unmeasured.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from .media import RATE, atomic_json, extract_audio, file_hash, fingerprint, resolve_channel_layout, wav_format
from .movie_faults import MovieConfigurationError, MovieContentFallback, fault_record
from .process import Cancelled

VERSION = 'generic-dub-v1'
GAP_SECONDS = .05


def _require(condition, code, detail):
    if not condition:
        raise MovieConfigurationError(code, detail)


def _span(start, end):
    start, end = float(start), float(end)
    _require(math.isfinite(start) and math.isfinite(end) and start < end,
             'generic_interval_invalid', 'Invalid source or placement interval')
    return [start, end]


def _union(intervals):
    result = []
    for a, b in sorted(_span(*item) for item in intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def _clip(intervals, begin, end):
    return _union([[max(begin, a), min(end, b)] for a, b in intervals if min(end, b) > max(begin, a)])


def _subtract(intervals, covered):
    result = []
    walls = _union(covered)
    for a, b in _union(intervals):
        cursor = a
        for left, right in walls:
            if right <= cursor or left >= b:
                continue
            if left > cursor:
                result.append([cursor, min(left, b)])
            cursor = max(cursor, right)
            if cursor >= b:
                break
        if cursor < b:
            result.append([cursor, b])
    return result


def _seconds(intervals):
    return sum(b - a for a, b in _union(intervals))


def _overlaps(a, b, walls):
    return any(min(b, right) > max(a, left) for left, right in walls)


def _check(store, owner):
    owner.check()
    store.check_claim()


def _full_publication(take, chunks):
    if not take.get('playable') or take.get('overflow_past_end_seconds', 0) > 0:
        return False
    spans = []
    needed = take.get('required_chunk_indices') or []
    if not needed:
        return False
    for index in needed:
        chunk = chunks.get(index)
        if not chunk or chunk.get('state') not in ('ready', 'approved'):
            return False
        payload = chunk['payload']
        if payload.get('generic_version') != VERSION or int(payload.get('revision', -1)) != int(take['revision']):
            return False
        matches = [row for row in payload.get('generic_takes') or []
                   if row.get('id') == take['id'] and row.get('sha256') == take['sha256']]
        if len(matches) != 1:
            return False
        row = matches[0]
        if any(row.get(key) != take.get(key) for key in ('start', 'end', 'source_spans', 'required_chunk_indices')):
            return False
        spans.append(row['mixed_interval'])
    return not _subtract([[take['start'], take['end']]], spans)


def _coverage(store, job_id, takes, index, prospective):
    chunks = {int(row['idx']): row for row in store.chunks(job_id)}
    chunks[index] = {'state': 'ready', 'payload': prospective}
    detected = _union([interval for row in chunks.values()
                       if row.get('state') in ('ready', 'approved')
                       and row['payload'].get('generic_version') == VERSION
                       and int(row['payload'].get('revision', -1)) == int(prospective['revision'])
                       for interval in row['payload'].get('generic_vad_regions') or []])
    full = [take for take in takes if _full_publication(take, chunks)]
    covered = _union([part for take in full for a, b in take['source_spans'] for part in _clip(detected, a, b)])
    local = _clip(detected, prospective['start'], prospective['end'])
    local_covered = _clip(covered, prospective['start'], prospective['end'])
    return {'version': VERSION, 'scope': 'ready-chunk snapshot at this atomic publication; later carry may complete older takes',
            'job_detected_seconds': _seconds(detected), 'job_published_full_source_seconds': _seconds(covered),
            'job_coverage_percent': 100 * _seconds(covered) / _seconds(detected) if detected else None,
            'chunk_detected_seconds': _seconds(local), 'chunk_published_full_source_seconds': _seconds(local_covered),
            'chunk_coverage_percent': 100 * _seconds(local_covered) / _seconds(local) if local else None,
            'published_full_take_ids': [take['id'] for take in full], 'accuracy': 'unmeasured',
            'heard': 'not inferred from publication; player ranges are separate'}


def process_generic_chunk(config, store, job_id, index, owner, services):
    """Process one claimed chunk; only publish_generic may publish takes or ready state."""
    import numpy as np
    import soundfile as sf
    from . import first_pass_parent
    from . import generic_policy

    started = time.monotonic()
    data = getattr(config, 'data', config)
    _require(callable(getattr(store, 'check_claim', None)) and callable(getattr(store, 'publish_generic', None)),
             'generic_claim_required', 'Generic processing requires the claimed store publication API')
    _check(store, owner)
    job = store.job(job_id)
    if not job or job['state'] == 'cancelled':
        raise Cancelled('Generic job cancelled or missing')
    spec = job['payload']
    _require(spec.get('speech_engine', config['speech_engine']) == 'generic', 'generic_engine_mismatch', 'Job is not generic')
    _require(fingerprint(spec['source']) == spec['source_fingerprint'] and
             config.engine_fingerprint() == spec['engine_fingerprint'],
             'generic_identity_changed', 'Source or engine changed; create a new job')
    context = float(data.get('generic_context_seconds', 6.0))
    max_speed = float(data.get('generic_max_speed', 1.35))
    lag_target = float(data.get('generic_max_lag_seconds', 2.0))
    _require(math.isfinite(context) and context >= 6 and math.isfinite(max_speed) and 1 <= max_speed <= 1.35
             and math.isfinite(lag_target) and lag_target >= 0,
             'generic_config_invalid', 'Generic context/speed/lag configuration is invalid')
    movie_end = float(spec['duration'])
    segment_seconds = float(spec['segment_seconds'])
    first_index = int(spec.get('generic_active_index', spec['generic_start_index']))
    previous_chunks = {int(row['idx']): row for row in store.chunks(job_id)}
    _require(all(i in previous_chunks and previous_chunks[i]['state'] in ('ready', 'approved')
             for i in range(first_index, index)), 'generic_serial_order', 'Earlier active-session chunks must publish first')
    chunk = store.chunk(job_id, index)['payload']
    begin, end = _span(chunk['start'], min(chunk['end'], movie_end))
    successors = [float(row['payload']['start']) for number, row in previous_chunks.items()
                  if number > index and row['state'] in ('ready', 'approved')]
    successor_boundary = min(successors) if successors else None
    successor_limit = successor_boundary - GAP_SECONDS if successor_boundary is not None else None
    origin, stop = max(0., begin - context), min(movie_end, end + context)
    revision = int(spec.get('revision', 1))
    token = int(store.claim_token)
    directory = Path(config.cache) / job_id / ('chunk-%06d' % index) / ('generic-claim-%d-%s' % (token, uuid.uuid4().hex))
    directory.mkdir(parents=True, exist_ok=False)
    result = {'generic_version': VERSION, 'job_id': job_id, 'revision': revision, 'chunk': index,
              'engine_fingerprint': spec['engine_fingerprint'],
              'generic_policy_sha256': file_hash(Path(generic_policy.__file__)),
              'start': begin, 'end': end, 'origin': origin, 'analysis_end': stop,
              'source_fingerprint': spec['source_fingerprint'], 'source_audio_index': spec['audio_index'],
              'claim_token': token, 'mode': 'generic-English-candidate',
              'generic_vad_regions': [], 'generic_takes': [], 'generic_original_intervals': [],
              'generic_failures': [], 'issues': [], 'accuracy': 'unmeasured',
              'human_listening_acceptance': False, 'subtitle_role': 'diagnostic-only',
              'lag_target_seconds': lag_target, 'lag_target_is_guarantee': False}
    try:
        stream = next((s for s in spec['media']['streams'] if s.get('index') == spec['audio_index']), None)
        _require(stream is not None and stream.get('codec_type', 'audio') == 'audio',
                 'generic_source_stream_missing', 'Selected source audio stream is missing')
        channels = int(stream.get('channels', 0))
        layout = stream.get('channel_layout') if channels == 6 else {1: 'mono', 2: 'stereo'}.get(channels)
        _require(layout in ('mono', 'stereo', '5.1', '5.1(side)'), 'generic_source_layout', 'Unsupported source layout')
        original_path = directory / 'original-layout.wav'
        extract_audio(config, owner, spec['source'], spec['audio_index'], origin, stop - origin,
                      original_path, channels=channels, preserve_layout=layout)
        decoded, rate = sf.read(original_path, dtype='float32', always_2d=True)
        _require(rate == RATE and np.isfinite(decoded).all(), 'generic_source_pcm_invalid', 'Invalid decoded PCM')
        layout_evidence = resolve_channel_layout(decoded.shape[1], source=stream, wav=wav_format(original_path))
        actual_stop = origin + len(decoded) / rate
        _require(actual_stop + 1 / rate >= end, 'generic_source_short', 'Decoded source does not cover the chunk')
        stop = min(stop, actual_stop)
        _check(store, owner)
        bundle = first_pass_parent.run(config, owner, folder=directory, original_source_wav=original_path,
            source_layout_evidence=layout_evidence, origin=origin, frames=len(decoded), scope=None)
        _require(bundle is not None, 'generic_separation_unconfigured', 'Generic requires registered Demucs first pass')
        producer = bundle.get('parent') or {}
        _require(producer.get('producer') == 'resident' and producer.get('device') == 'cuda'
                 and producer.get('admission') == 'movie-lease',
                 'generic_separation_not_cuda', 'Demucs must use the resident CUDA movie-lease producer')
        stems = {}
        for name in ('processing', 'raw_vocals', 'background', 'removal_stem'):
            entry = bundle[name]
            _require(file_hash(entry['wav']) == entry['sha256'], 'generic_stem_changed', name)
            samples, sample_rate = sf.read(entry['wav'], dtype='float32', always_2d=True)
            _require(sample_rate == RATE and samples.shape == (len(decoded), 2) and np.isfinite(samples).all(),
                     'generic_stem_geometry', name)
            stems[name] = samples
        result['separation'] = {'first_pass_key': bundle.get('key'), 'scope': 'full-analysis-window',
            'parent': dict(producer),
            'vocals': dict(bundle['raw_vocals']), 'background': dict(bundle['background']),
            'removal': dict(bundle['removal_stem']), 'bed_basis': 'demucs-background-full-chunk',
            'bed_gain': 1.0, 'disclosure': 'Vocals stem is removed everywhere, including undetected speech and any singing, screams or breaths assigned to vocals'}
        del decoded
        client = services.language(owner)
        vad = client.generic_vad(Path(bundle['raw_vocals']['wav']))
        result['generic_vad_detector'] = dict(client.generic_vad_receipt)
        detected = _union([_span(origin + row['start'], origin + row['end']) for row in vad])
        _require(all(origin <= a < b <= actual_stop + 1 / RATE for a, b in detected),
                 'generic_vad_geometry', 'VAD regions exceed analysed vocals')
        detected = _clip(detected, origin, stop)
        result['generic_vad_regions'] = _clip(detected, begin, end)
        result['generic_vad_analysis_regions'] = detected
        result['vad_scope'] = 'independent Silero VAD on separated vocals; supporting VAD CPU, main inference CUDA'

        prior, all_stored = [], []
        for row in store.utterances(job_id, -1e18, 1e18):
            take = row['payload']
            if int(take.get('revision', -1)) != revision:
                continue
            _require(take.get('generic_version') == VERSION and take.get('job_id') == job_id,
                     'generic_ledger_mismatch', 'Current-revision take has wrong version or job identity')
            _require(int(take['owner_chunk']) != index, 'generic_nonserial_take', 'This chunk already owns immutable takes')
            _require(take['source_fingerprint'] == spec['source_fingerprint'], 'generic_take_source_mismatch', take['id'])
            _require(file_hash(take['audio']) == take['sha256'], 'generic_take_changed', take['id'])
            all_stored.append(take)
            if int(take['owner_chunk']) < index:
                prior.append(take)
        prior_good = [take for take in prior if take.get('playable')]
        prior_sources = _union([part for take in prior_good for part in take['source_spans']])
        predecessor_ready, predecessor_takes = False, []
        predecessor = previous_chunks.get(index - 1)
        if predecessor and predecessor.get('state') in ('ready', 'approved'):
            payload = predecessor['payload']
            if payload.get('generic_version') == VERSION and int(payload.get('revision', -1)) == revision:
                predecessor_receipt = json.loads(Path(payload['receipt_path']).read_text(encoding='utf-8-sig'))
                _require(predecessor_receipt.get('job_id') == job_id
                         and predecessor_receipt.get('source_fingerprint') == spec['source_fingerprint']
                         and int(predecessor_receipt.get('chunk', -1)) == index - 1
                         and all(predecessor_receipt.get(key) == payload.get(key)
                                 for key in ('revision', 'generic_version', 'start', 'end', 'audio', 'sha256'))
                         and file_hash(payload['audio']) == payload['sha256'],
                         'generic_predecessor_changed', 'Published predecessor receipt or output changed')
                predecessor_ready = True
                predecessor_takes = predecessor_receipt.get('generic_takes') or []

        def ownership(onset, spans):
            return generic_policy.owns(onset, begin, end, prior_sources, predecessor_takes,
                source_spans=spans, predecessor_ready=predecessor_ready)

        failures, translations, translated_spans = result['generic_failures'], [], []
        for number, (a, b) in enumerate(detected):
            _check(store, owner)
            # Translate every independently detected region, including context;
            # only source starts owned by this chunk may create a new take.
            region_path = directory / ('vad-%04d.wav' % number)
            left, right = round((a - origin) * RATE), round((b - origin) * RATE)
            sf.write(region_path, stems['raw_vocals'][left:right], RATE, subtype='FLOAT')
            try:
                segments, info = client.speech(region_path, language=config['generic_source_language'] or None,
                                              task='translate', vad=False, beam_size=5)
            except MovieContentFallback as error:
                failures.append({'reason': 'no_translation', 'source_spans': [[a, b]], 'detail': str(error)})
                continue
            for ordinal, segment in enumerate(segments):
                sa, sb = _span(a + segment['start'], a + segment['end'])
                source_spans = _clip(detected, max(a, sa), min(b, sb)) if min(b, sb) > max(a, sa) else []
                text = str(segment.get('text') or '').strip()
                onset = generic_policy.line_onset({'start': sa, 'end': sb}, {'start': a, 'end': b}, ordinal)
                owned = ownership(onset, source_spans)
                record = {'region': number, 'segment': ordinal, 'start': sa, 'end': sb, 'text': text,
                          'source_spans': source_spans, 'whisper': info, 'onset': onset,
                          'ownership': owned, 'owned': owned['owned']}
                translations.append(record)
                if not source_spans:
                    failures.append({'reason': 'outside_detected_speech', 'source_spans': [],
                                     'segment_start': sa, 'segment_end': sb, 'text': text})
                    continue
                if not text:
                    failures.append({'reason': 'empty_translation', 'source_spans': source_spans})
                    continue
                translated_spans.extend(source_spans)
                if not owned['owned'] and onset < end:
                    missing = _subtract(_clip(source_spans, begin, end), prior_sources)
                    if missing:
                        failures.append({'reason': 'seam_unowned', 'source_spans': missing,
                                         'observed_source_spans': source_spans, 'onset': onset, 'ownership': owned})
        for part in _subtract(detected, translated_spans + prior_sources):
            failures.append({'reason': 'no_translation', 'source_spans': [part]})
        result['generic_translations'] = translations
        voice = {'engine': 'generic-piper', 'path': str(config.resolve(config['generic_voice_model'])),
                 'speaker_id': int(config['generic_speaker_id'])}
        synthesis_lines, seam_translations = [], []
        for line_number, line in enumerate(translations):
            if not line['owned'] or not line['source_spans'] or not line['text']:
                continue
            rest = _subtract(line['source_spans'], prior_sources)
            if rest == _union(line['source_spans']):
                synthesis_lines.append(line)
                continue
            if _seconds(rest) < .4:
                failures.append({'reason': 'source_already_owned', 'source_spans': rest,
                                 'scope': 'unowned seam remainder below 0.4 seconds'})
                continue
            # Exactly one extra translation on this original line's uncovered
            # source hull. Its output goes directly to synthesis, never back
            # through this seam-recovery loop.
            _check(store, owner)
            a, b = rest[0][0], rest[-1][1]
            region_path = directory / ('seam-%04d.wav' % line_number)
            left, right = round((a - origin) * RATE), round((b - origin) * RATE)
            sf.write(region_path, stems['raw_vocals'][left:right], RATE, subtype='FLOAT')
            try:
                segments, info = client.speech(region_path, language=config['generic_source_language'] or None,
                                              task='translate', vad=False, beam_size=5)
            except MovieContentFallback as error:
                failures.append({'reason': 'no_translation', 'source_spans': rest,
                                 'scope': 'one seam-recovery call', 'detail': str(error)})
                continue
            recovered = []
            for ordinal, segment in enumerate(segments):
                sa, sb = _span(a + segment['start'], a + segment['end'])
                spans = _clip(rest, max(a, sa), min(b, sb)) if min(b, sb) > max(a, sa) else []
                text = str(segment.get('text') or '').strip()
                onset = generic_policy.line_onset({'start': sa, 'end': sb}, {'start': a, 'end': b}, ordinal)
                owned = ownership(onset, spans)
                extra = {'region': line['region'], 'segment': ordinal, 'start': sa, 'end': sb,
                         'text': text, 'source_spans': spans, 'whisper': info, 'onset': onset,
                         'ownership': owned, 'owned': owned['owned'],
                         'seam_recovery': {'original_segment': line_number, 'rest': rest,
                                           'input_hull': [a, b], 'recursive': False}}
                seam_translations.append(extra)
                if not spans:
                    failures.append({'reason': 'outside_detected_speech', 'source_spans': [],
                                     'segment_start': sa, 'segment_end': sb, 'scope': 'seam recovery'})
                elif not owned['owned']:
                    missing = _subtract(_clip(spans, begin, end), prior_sources) if onset < end else []
                    if missing:
                        failures.append({'reason': 'seam_unowned', 'source_spans': missing,
                                         'observed_source_spans': spans, 'onset': onset, 'ownership': owned,
                                         'scope': 'seam-recovery output not owned'})
                elif not text:
                    failures.append({'reason': 'empty_translation', 'source_spans': spans, 'scope': 'seam recovery'})
                elif extra['owned']:
                    recovered.extend(spans)
                    synthesis_lines.append(extra)
            for part in _subtract(rest, recovered):
                failures.append({'reason': 'no_translation', 'source_spans': [part],
                                 'scope': 'no owned translation from the single seam-recovery call'})
        translations.extend(seam_translations)

        def synthesized_lines():
            identities = set()
            for line in synthesis_lines:
                ident = hashlib.sha256(json.dumps([VERSION, job_id, revision, line['start'], line['end'], line['text']],
                                                 separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()[:24]
                if ident in identities:
                    continue
                identities.add(ident)
                _check(store, owner)
                raw_path = directory / (ident + '-piper.wav')
                try:
                    tts = client.generic_synthesize(line['text'], voice, raw_path)
                except MovieContentFallback as error:
                    failures.append({'reason': 'tts_error', 'source_spans': line['source_spans'],
                                     'id': ident, 'detail': str(error)})
                    continue
                _require(file_hash(tts['audio']) == tts['sha256'] and not tts.get('truncated'),
                         'generic_tts_integrity', 'TTS returned changed/truncated PCM')
                _require(Path(tts['audio']).resolve() == raw_path.resolve() and tts.get('device') == 'cuda'
                         and tts.get('speed') == 1,
                         'generic_tts_contract', 'TTS must return untrimmed CUDA output in the request-owned path')
                yield {**line, 'id': ident, 'audio': tts['audio'], 'voice': voice, 'synthesis': tts}
        result['_render_context'] = {'directory': str(directory), 'spec': spec, 'started': started, 'stop': stop,
            'prior_good': prior_good, 'prior_sources': prior_sources, 'all_stored': all_stored,
            'successor_boundary': successor_boundary, 'successor_limit': successor_limit}
        return render_and_publish(config, store, job_id, index, owner, begin=begin, end=end, origin=origin,
            bed=stems['background'], raw_vocals=stems['raw_vocals'], vad_regions=result['generic_vad_regions'],
            lines=synthesized_lines(), result=result)
    except BaseException as error:
        # Unique private evidence is never ready/publication; existing published
        # files and takes are untouched. Coordinator handles fatal/resource state.
        atomic_json(directory / 'failure.json', {'generic_version': VERSION, 'job_id': job_id, 'chunk': index,
            'claim_token': token, 'error': str(error), 'error_type': type(error).__name__,
            'fault': fault_record(error), 'partial': result, 'ready_published': False})
        raise


def render_and_publish(config, store, job_id, index, owner, *, begin, end, origin,
                       bed, raw_vocals, vad_regions, lines, result, original_intervals=()):
    """Shared whole-PCM renderer. ``lines`` yields already synthesized full takes.

    The generic caller yields lazily, preserving its original synthesis/fit order.
    X-Translator supplies completed upstream session outcomes, never Piper calls.
    Private context is removed before any receipt is written.
    """
    import numpy as np
    import soundfile as sf
    from . import generic_policy
    context = result.pop('_render_context')
    directory = Path(context['directory'])
    spec = context['spec']
    started, stop = context['started'], context['stop']
    revision, movie_end = int(spec.get('revision', 1)), float(spec['duration'])
    segment_seconds = float(spec['segment_seconds'])
    max_speed = float(getattr(config, 'data', config).get('generic_max_speed', 1.35))
    lag_target = float(getattr(config, 'data', config).get('generic_max_lag_seconds', 2.0))
    prior_good, prior_sources, all_stored = context['prior_good'], context['prior_sources'], context['all_stored']
    successor_boundary, successor_limit = context['successor_boundary'], context['successor_limit']
    failures = result['generic_failures']
    original_intervals = _union(original_intervals)
    placement_limit = context.get('placement_limit')
    # Published full PCM is immutable; never silently trim it to restore speech.
    for take in prior_good:
        _require(not any(min(take['end'], b) > max(take['start'], a)
                         for a, b in original_intervals), 'xt_carry_original_conflict',
                 'Published generated carry intersects native source-English speech')
    def after_original(start, frames):
        for a, b in original_intervals:
            if min(start + frames / RATE, b) > max(start, a):
                start = b + GAP_SECONDS
        return math.ceil(start * RATE - 1e-7) / RATE
    stems = {'raw_vocals': raw_vocals, 'background': bed}
    result['generic_vad_regions'] = vad_regions
    def leveled_pcm(path, spans, suffix):
        # Feed conditioning is analysis-only. Match active speech on both sides
        # against these unboosted stems, not the gain-adjusted model input.
        pcm, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        _require(sample_rate == RATE and pcm.shape[1] == 1 and len(pcm) > 0 and np.isfinite(pcm).all(),
                 'generic_level_pcm_invalid', 'Level matching needs the final whole mono tempo PCM')
        gain, metadata = generic_policy.level_gain(stems['raw_vocals'], spans, pcm[:, 0], RATE,
            result['generic_vad_regions'], origin=origin)
        _require(math.isfinite(gain) and 0 < gain <= 2., 'generic_level_gain_invalid', 'Invalid source-relative gain')
        output = directory / suffix
        _require(not output.exists(), 'generic_level_path_exists', 'Never overwrite a leveled take')
        _check(store, owner)
        sf.write(output, (pcm * gain).astype(np.float32), RATE, subtype='FLOAT')
        return output, {**metadata, 'gain': gain, 'applied_to_pcm': True,
                        'input_audio': str(path), 'input_sha256': file_hash(path),
                        'frames': len(pcm), 'sample_rate': RATE}

    candidates, identities = [], set()
    for line in lines:
        ident = line['id']
        if ident in identities:
            continue
        identities.add(ident)
        _check(store, owner)
        tts, voice = line['synthesis'], line['voice']
        take_path = directory / (ident + '-full.wav')
        _require(file_hash(tts['audio']) == tts['sha256'] and not tts.get('truncated'),
                 'generic_tts_integrity', 'Synthesized full PCM changed or was truncated')
        raw_info = sf.info(tts['audio'])
        raw_seconds = raw_info.frames / raw_info.samplerate
        _require(raw_seconds > 0 and raw_info.channels == 1, 'generic_tts_geometry', 'TTS must return whole mono PCM')
        target = line['end'] - line['onset']
        speed = min(max_speed, generic_policy.fit_speed(raw_seconds, target))
        owner.run([config['ffmpeg'], '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
            '-i', str(tts['audio']), '-af', 'atempo=%.9f' % speed, '-ar', str(RATE), '-ac', '1',
            '-c:a', 'pcm_f32le', str(take_path)])
        info = sf.info(take_path)
        _require(info.samplerate == RATE and info.channels == 1 and info.frames > 0,
                 'generic_take_geometry', 'Invalid whole take after bounded speed conversion')
        try:
            leveled_path, level = leveled_pcm(take_path, line['source_spans'], ident + '-leveled.wav')
        except ValueError as error:
            if str(error) not in ('level_reference_unavailable', 'generic_level_take_silent_or_invalid'):
                raise
            reason = 'tts_silent_or_invalid' if str(error) == 'generic_level_take_silent_or_invalid' else str(error)
            failures.append({'reason': reason, 'detail': str(error), 'id': ident,
                'source_spans': line['source_spans'], 'tempo_audio': str(take_path),
                'tempo_sha256': file_hash(take_path), 'synthesis': tts, 'take_admitted': False})
            continue
        candidates.append({'generic_version': VERSION, 'id': ident, 'owner_chunk': index, 'revision': revision,
            'job_id': job_id, 'source_fingerprint': spec['source_fingerprint'],
            'source_start': line['start'], 'source_end': line['end'], 'source_spans': line['source_spans'],
            'onset': line['onset'], 'ownership': line['ownership'], 'target_duration': target,
            'duration_gap': target - info.frames / RATE,
            'text': line['text'], 'audio': str(leveled_path), 'sha256': file_hash(leveled_path),
            'level_match': level,
            'seconds': info.frames / RATE, 'frames': info.frames, 'sample_rate': RATE,
            'raw_seconds': raw_seconds, 'speed': speed, 'truncated': False, 'voice': voice,
            'synthesis': tts, 'playable': True, 'status': 'pending_publication', **line.get('take_metadata', {})})

    candidates.sort(key=lambda take: (take['onset'], take['source_end'], take['id']))
    # A late predecessor may not inject carry into an already-ready successor.
    # Failed lines leave bed-only gaps. Native English intervals are hard walls.
    for _ in range(len(candidates) + 1):
        active = [take for take in candidates if take['playable']]
        cursor = max([begin - GAP_SECONDS] + [take['end'] for take in prior_good])
        changed = False
        for take in active:
            placement_floor = take['ownership'].get('placement_floor')
            start = max(begin if placement_floor is None else placement_floor, take['onset'], cursor + GAP_SECONDS)
            start = after_original(start, take['frames'])
            finish = (round(start * RATE) + take['frames']) / RATE
            limits = [v for v in (successor_limit, placement_limit) if v is not None]
            fit_limit = min(limits) if limits else None
            if fit_limit is not None and finish > fit_limit and take['speed'] < max_speed:
                # One bounded tempo adjustment of the existing full PCM, not
                # a new TTS attempt and never an overwrite of a published file.
                fitted = directory / (take['id'] + '-full-maxspeed.wav')
                _check(store, owner)
                owner.run([config['ffmpeg'], '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                    '-i', str(take['synthesis']['audio']), '-af', 'atempo=%.9f' % max_speed,
                    '-ar', str(RATE), '-ac', '1', '-c:a', 'pcm_f32le', str(fitted)])
                fitted_info = sf.info(fitted)
                _require(fitted_info.samplerate == RATE and fitted_info.channels == 1 and fitted_info.frames > 0,
                         'generic_take_geometry', 'Invalid max-speed whole PCM')
                try:
                    leveled_path, level = leveled_pcm(fitted, take['source_spans'], take['id'] + '-maxspeed-leveled.wav')
                except ValueError as error:
                    if str(error) not in ('level_reference_unavailable', 'generic_level_take_silent_or_invalid'):
                        raise
                    take['level_failure'] = ('tts_silent_or_invalid'
                        if str(error) == 'generic_level_take_silent_or_invalid' else str(error))
                    take['level_failure_detail'] = str(error)
                else:
                    take.update(audio=str(leveled_path), sha256=file_hash(leveled_path), speed=max_speed,
                                frames=fitted_info.frames, seconds=fitted_info.frames / RATE, level_match=level)
                    start = max(begin if placement_floor is None else placement_floor, take['onset'], cursor + GAP_SECONDS)
                    start = after_original(start, take['frames'])
                    finish = (round(start * RATE) + take['frames']) / RATE
            take.update(start=start, end=finish, lag_seconds=max(0., start - take['onset']),
                        start_lag=max(0., start - take['onset']), duration_gap=take['target_duration'] - take['seconds'],
                        overflow_past_end_seconds=max(0., finish - movie_end))
            take['required_chunk_indices'] = list(range(math.floor(start / segment_seconds),
                                                        math.ceil(finish / segment_seconds)))
            if take.get('level_failure'):
                take.update(playable=False, status=take['level_failure'], complete=False)
                failures.append({'reason': take['level_failure'], 'detail': take['level_failure_detail'],
                                 'id': take['id'], 'source_spans': take['source_spans'], 'scope': 'late-fit leveling failed'})
                changed = True
            elif successor_limit is not None and finish > successor_limit:
                take.update(playable=False, status='successor_already_published', complete=False)
                failures.append({'reason': 'successor_already_published', 'id': take['id'],
                                 'source_spans': take['source_spans'], 'successor_start': successor_boundary,
                                 'latest_end_with_gap': successor_limit})
                changed = True
            elif placement_limit is not None and finish > placement_limit:
                take.update(playable=False, status='controller_frontier_placement_limit', complete=False)
                failures.append({'reason': 'controller_frontier_placement_limit', 'id': take['id'],
                    'source_spans': take['source_spans'], 'commit_frontier': placement_limit})
                changed = True
            else:
                cursor = finish
        if not changed:
            break
    else:
        raise MovieConfigurationError('generic_schedule_unstable', 'Successor fallback schedule did not converge')
    active = [take for take in candidates if take['playable']]
    for take in active:
        if take['overflow_past_end_seconds'] > 0:
            take.update(status='overflow_past_end', complete=False)
            failures.append({'reason': 'overflow_past_end', 'id': take['id'], 'source_spans': take['source_spans'],
                'overflow_past_end_seconds': take['overflow_past_end_seconds'], 'complete_credit': 0,
                'in_film_placed_seconds': max(0., min(movie_end, take['end']) - min(movie_end, take['start']))})
    carried = [take for take in prior_good if min(take['end'], end) > max(take['start'], begin)]
    result['generic_bed_only_failure_intervals'] = _subtract(result['generic_vad_regions'],
        prior_sources + [part for take in active for part in take['source_spans']] + original_intervals)
    mixed_takes = sorted(carried + [t for t in active if t['end'] > begin and t['start'] < end],
                         key=lambda take: (take['start'], take['id']))
    for left, right in zip(mixed_takes, mixed_takes[1:]):
        _require(left['end'] <= right['start'] + 1 / RATE,
                 'generic_english_collision', 'Stored/full English takes overlap')
    for take in mixed_takes:
        _require(not any(min(take['end'], b) > max(take['start'], a) for a, b in original_intervals),
                 'xt_generated_original_collision', 'Generated English intersects restored original speech')
    first, last = round((begin - origin) * RATE), round((end - origin) * RATE)
    bed = stems['background'][first:last]
    generated = np.zeros_like(bed)
    for take in mixed_takes:
        _check(store, owner)
        _require(file_hash(take['audio']) == take['sha256'], 'generic_take_changed', take['id'])
        pcm, sample_rate = sf.read(take['audio'], dtype='float32', always_2d=True)
        _require(sample_rate == RATE and pcm.shape == (take['frames'], 1) and np.isfinite(pcm).all(),
                 'generic_take_pcm_invalid', take['id'])
        lo, hi = max(begin, take['start']), min(end, take['end'])
        destination = round(lo * RATE) - round(begin * RATE)
        source = round(lo * RATE) - round(take['start'] * RATE)
        frames = round(hi * RATE) - round(lo * RATE)
        _require(frames > 0 and source >= 0 and destination >= 0 and source + frames <= len(pcm)
                 and destination + frames <= len(generated), 'generic_carry_geometry', take['id'])
        generated[destination:destination + frames] += pcm[source:source + frames]
        result['generic_takes'].append({**take, 'mixed_interval': [lo, hi], 'publication_status': 'fragment'})
    passthrough = result.get('mode') == 'original-English'
    result['generic_original_intervals'] = _clip(original_intervals, begin, end)
    if passthrough:
        result['translation_not_required'] = True
    if original_intervals:
        result['generic_original_interval_scope'] = 'exact native-Qwen source-English segment spans clipped to chunk'
    _require(np.isfinite(bed).all(), 'generic_bed_nonfinite', 'Background PCM contains nonfinite samples')
    bed_peak = float(np.max(np.abs(bed))) if bed.size else 0.
    bed_overrange = np.abs(bed) > 1.
    bed_overrange_samples = int(np.count_nonzero(bed_overrange))
    if bed_overrange_samples:
        bed = bed.copy()
        bed[bed_overrange] = np.clip(bed[bed_overrange], -1., 1.)
    allowed = np.ones_like(generated)
    positive, negative = generated > 0, generated < 0
    allowed[positive] = np.clip((1. - bed[positive]) / generated[positive], 0., 1.)
    allowed[negative] = np.clip((-1. - bed[negative]) / generated[negative], 0., 1.)
    # Shared per-frame gain preserves equal-channel English; the background
    # stays at unity even where English headroom is limited.
    english_limit = np.min(allowed, axis=1)
    mixed = bed + generated * english_limit[:, None]
    if result['generic_original_intervals']:
        processing = context.get('processing')
        _require(processing is not None and processing.shape == mixed.shape and np.isfinite(processing).all(),
                 'xt_original_processing_geometry', 'Exact source-English processing PCM is unavailable')
        for a, b in result['generic_original_intervals']:
            lo, hi = round((a - begin) * RATE), round((b - begin) * RATE)
            mixed[lo:hi] = processing[lo:hi]
    result['mix'] = {'policy': 'generic-background-unity-english-only-headroom-v1',
        'bed_basis': context.get('bed_basis', 'demucs-background-full-chunk'), 'bed_gain': 1.0,
        'bed_peak': bed_peak, 'bed_overrange_samples': bed_overrange_samples,
        'bed_clipping_scope': 'only samples outside [-1,1]; no global scale',
        'english_limited_frames': int(np.count_nonzero(english_limit < 1.)),
        'minimum_english_limiter_gain': float(english_limit.min()) if len(english_limit) else 1.,
        'original_voice_restored': bool(result['generic_original_intervals']), 'accuracy': 'unmeasured'}
    if passthrough:
        result['mix']['source_English_passthrough'] = True
    result['generic_failed_takes'] = [take for take in candidates
                                      if not take['playable'] or take.get('overflow_past_end_seconds', 0) > 0]
    for failure in failures:
        failure['complete_credit'] = 0
        failure['failure_id'] = hashlib.sha256(json.dumps([job_id, revision, index, failure['reason'],
            failure.get('source_spans'), failure.get('id'), failure.get('take_id')],
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    result['generic_lag'] = {'target_seconds': lag_target,
        'max_lag_seconds': max([0.] + [take['lag_seconds'] for take in candidates]),
        'over_target_take_ids': [take['id'] for take in candidates if take['lag_seconds'] > lag_target],
        'target_is_guarantee': False}
    # Subtitles never select, reject or rewrite a translated line.
    try:
        if result.get('speech_engine') == 'xtranslator':
            raise LookupError('X-Translator does not consume subtitles')
        from .embedded_subtitles import chunk_captions
        subtitles, subtitle_source = chunk_captions(config, owner, spec, origin, stop, directory)
        spans = _clip([[float(row['start']), float(row['end'])] for row in subtitles], begin, end)
        overlap = _union([p for a, b in spans for p in _clip(result['generic_vad_regions'], a, b)])
        result['generic_subtitle_comparison'] = {'status': 'diagnostic', 'source': subtitle_source,
            'subtitle_intervals': spans, 'subtitle_seconds': _seconds(spans),
            'vad_overlap_seconds': _seconds(overlap), 'used_for_line_selection': False}
    except Exception as error:
        _check(store, owner)
        if fault_record(error):
            raise
        result['generic_subtitle_comparison'] = {'status': 'unavailable', 'error': str(error),
                                                 'used_for_line_selection': False}
    _check(store, owner)
    output = directory / 'english.wav'
    sf.write(output, mixed, RATE, subtype='PCM_16')
    result.update(audio=str(output), sha256=file_hash(output), receipt_path=str(directory / 'receipt.json'),
                  elapsed_seconds=time.monotonic() - started, quality={'accuracy': 'unmeasured',
                  'human_listening_acceptance': False, 'scope': 'generic machine construction only'})
    if result.get('speech_engine') == 'xtranslator':
        result['quality']['scope'] = 'X-Translator rendering construction only; listening and translation accuracy unmeasured'
    result['generic_coverage'] = _coverage(store, job_id, all_stored + candidates, index, result)
    if result.get('mode') == 'original-English':
        result['generic_coverage'].update(job_coverage_percent=None, chunk_coverage_percent=None,
            scope='Source-English passthrough (language evidence recorded); translation credit is not applicable')
    _check(store, owner)
    atomic_json(directory / 'receipt.json', result)
    _check(store, owner)
    store.publish_generic(job_id, index,
        [(take['id'], take['start'], take['end'], take) for take in candidates],
        **{key: value for key, value in result.items() if key != 'job_id'})
    return result
