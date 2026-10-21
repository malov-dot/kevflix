from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import re
import threading
from collections import OrderedDict, deque
from pathlib import Path

from . import diart_turns
from .dialogue import is_no_speech
from .language import backend_fault, model_device, prepare_model_runtime, torch_model_device
from .movie_faults import MovieConfigurationError, MovieContentFallback


EMBEDDING_POLICY = 'community1-native-mono-f32-16khz-v3'
_SPACE_IDENTITIES = OrderedDict()
_SPACE_IDENTITIES_LOCK = threading.Lock()
_SPACE_IDENTITY_LIMIT = 16


def _model_root(config):
    path = Path(config['diarization_model'])
    root = path if path.is_dir() else path.parent
    if not (root / 'config.yaml').is_file() or not (root / 'embedding' / 'pytorch_model.bin').is_file():
        raise MovieConfigurationError('speaker_assets_missing',
                                      'Local Community-1 config and embedding assets are required')
    return root


def embedding_space_identity(config):
    """One content identity for centroids, utterances and saved references.

    Model assets are immutable for an owned worker's lifetime. A bounded cache
    avoids re-reading weights inside every bank/DB transaction; file replacement
    or stat changes invalidate its entry. The first read hashes actual content
    and rejects a concurrent change. Same-stat external edits while a worker is
    alive are outside the immutable-model contract (replace/reload the worker).
    The optional external ONNX checkpoint is deliberately not this space.
    """
    from .analysis_cache import file_identity
    root = _model_root(config)
    try:
        model = (root / 'embedding' / 'pytorch_model.bin').resolve()
        def signature():
            info = model.stat()
            return (str(model), info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                    info.st_dev, info.st_ino)
        before = signature()
        with _SPACE_IDENTITIES_LOCK:
            cached = _SPACE_IDENTITIES.get(before)
            if cached is not None:
                _SPACE_IDENTITIES.move_to_end(before)
                return cached
        identity = {'policy': EMBEDDING_POLICY,
                    'embedding': file_identity(model)}
        if signature() != before:
            raise MovieConfigurationError('speaker_model_changed',
                                          'Speaker model changed while its identity was being read')
    except OSError as error:
        raise MovieConfigurationError('speaker_assets_missing', str(error)) from error
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    result = 'community1-native:' + hashlib.sha256(encoded).hexdigest()
    with _SPACE_IDENTITIES_LOCK:
        _SPACE_IDENTITIES[before] = result
        _SPACE_IDENTITIES.move_to_end(before)
        while len(_SPACE_IDENTITIES) > _SPACE_IDENTITY_LIMIT:
            _SPACE_IDENTITIES.popitem(last=False)
    return result


def _model_identity(config):
    from .analysis_cache import file_identity
    root = _model_root(config)
    names = ('config.yaml', 'embedding/pytorch_model.bin', 'segmentation/pytorch_model.bin',
             'plda/plda.npz', 'plda/xvec_transform.npz')
    try:
        return {name: file_identity(root / name) for name in names}
    except OSError as error:
        raise MovieConfigurationError('speaker_assets_missing', str(error)) from error


def _cache_policy_identity():
    """Raw observations expire on implementation/runtime changes, not actor IDs.

    Computed once per analyzer. Package metadata reads do not import the model
    libraries. An environment missing version metadata gets analyzer-local
    reuse only rather than claiming compatibility with an unknown runtime.
    """
    from importlib.metadata import PackageNotFoundError, version
    libraries = {}
    for name in ('pyannote.audio', 'pyannote.core', 'pyannote.pipeline', 'torch',
                 'torchaudio', 'numpy', 'scipy', 'scikit-learn'):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    try:
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError as error:
        raise MovieConfigurationError('speaker_cache_source_unavailable', str(error)) from error
    identity = {'version': 1, 'speakers_source_sha256': source_hash, 'libraries': libraries,
                'persistent_reuse': all(value is not None for value in libraries.values())}
    if not identity['persistent_reuse']:
        import uuid
        identity['analyzer_nonce'] = uuid.uuid4().hex
    return identity


def cosine(a, b):
    if len(a) != len(b) or len(a) == 0:
        return -1.0
    aa = sum(x * x for x in a)
    bb = sum(x * x for x in b)
    return sum(x * y for x, y in zip(a, b)) / max(1e-12, math.sqrt(aa * bb))


def continuing_known_speaker(sample,ranked,anchors):
    if not ranked or ranked[0][0]<.55 or (len(ranked)>1 and ranked[0][0]-ranked[1][0]<.15):return None
    label='KNOWN::'+ranked[0][1]
    supporting=[s for s in anchors if s.get('known_speaker')==label and s['end']<=sample['start']
                and sample['start']-s['start']<=8]
    if len(supporting)<2:return None
    nearest=max(s['end'] for s in supporting)
    if sample['start']-nearest>.75:return None
    if any(s.get('known_speaker') not in (None,label) and s['end']>nearest and s['start']<sample['start'] for s in anchors):return None
    return label


def speaker_units(segments,turns,captions,origin=0,*,whisper_no_speech=False):
    """Use continuous single-speaker sentences for embeddings, not tiny ASR fragments.

    This changes identity-analysis windows only. Recognized words remain intact
    in the dialogue plan. Different captions, raw speakers, and explicitly
    multi-speaker captions cannot be joined by this rule.
    """
    def shared(a,b):return max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
    units=[]
    for index,segment in enumerate(segments):
        absolute={**segment,'start':segment['start']+origin,'end':segment['end']+origin}
        totals={}
        for turn in turns:totals[turn['speaker']]=totals.get(turn['speaker'],0)+shared(absolute,turn)
        ranked=sorted(totals.items(),key=lambda x:x[1],reverse=True)
        raw=ranked[0][0] if ranked and ranked[0][1]>=.5*max(.001,segment['end']-segment['start']) else None
        hits=sorted(((shared(absolute,c),i) for i,c in enumerate(captions)),reverse=True)
        caption=hits[0][1] if hits and hits[0][0]>0 else None
        multiple=caption is not None and len(re.findall(r'(?:^|\s)-\s+',captions[caption]['text']))>1
        key=(raw,caption)
        merge=bool(units and raw is not None and caption is not None and not multiple and
                   (not is_no_speech(segment,True) and not is_no_speech(units[-1],True) if whisper_no_speech else
                    segment.get('no_speech_probability',0)<=.6 and
                    units[-1].get('no_speech_probability',0)<=.6) and
                   units[-1]['unit_key']==key and not units[-1]['multiple_speakers'] and
                   -.1<=segment['start']-units[-1]['end']<=.55 and segment['end']-units[-1]['start']<=8)
        if merge:
            units[-1]['end']=max(units[-1]['end'],segment['end'])
            units[-1]['text']+=' '+segment['text']
            units[-1]['words'].extend(segment.get('words') or [dict(segment,probability=0)])
            units[-1]['source_segments'].append(index);units[-1]['speaker_unit']=True
            units[-1]['no_speech_probability']=max(units[-1].get('no_speech_probability',0),
                                                  segment.get('no_speech_probability',0))
            if whisper_no_speech:
                # Both parts passed the two-part rule; keep the unit's (max no_speech,
                # max log_probability) pair passing it too.
                units[-1]['log_probability']=max(units[-1].get('log_probability',float('-inf')),
                                                 segment.get('log_probability',float('-inf')))
        else:
            units.append({**segment,'words':list(segment.get('words') or [dict(segment,probability=0)]),
                          'unit_key':key,'multiple_speakers':multiple,'source_segments':[index]})
    return units


def diarize(config, wav, offset=0.0, pipeline=None, *, device='cuda'):
    prepare_model_runtime(device)
    try:
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline
    except Exception as error:
        raise backend_fault('diarization_runtime', error, device) from error
    path = config["diarization_model"]
    if not Path(path).exists():
        raise MovieConfigurationError('speaker_assets_missing', 'Local pyannote Community-1 assets are missing')
    owns_pipeline = pipeline is None
    try:
        if owns_pipeline:
            target = torch_model_device(device)
            pipeline = Pipeline.from_pretrained(path)
            if pipeline is None:
                raise MovieConfigurationError('speaker_model_load_failed', 'The local speaker model could not be opened')
            pipeline.to(target)
        # Pyannote's Audio/cropping contract uses a host waveform. Its registered
        # segmentation/embedding inferences move batches onto pipeline.device.
        # Do not move the resident model back to CPU on successive windows.
        data, rate = sf.read(str(wav), dtype="float32", always_2d=True)
        audio = {"waveform": torch.from_numpy(data.mean(axis=1)).unsqueeze(0),
                 "sample_rate": rate}
        output = pipeline(audio)
        annotation = output.speaker_diarization
        exclusive = output.exclusive_speaker_diarization
        embeddings = output.speaker_embeddings
        labels = annotation.labels()
        if embeddings is None or len(embeddings) != len(labels):
            raise MovieConfigurationError('speaker_embeddings_missing',
                                          'Diarization did not return aligned persistent-identity embeddings')
        vectors = {}
        for i, label in enumerate(labels):
            vector = [float(x) for x in embeddings[i]]
            # Community-1 may have a turn with insufficient clean embedding
            # support. Keep the turn, but no fabricated/non-finite identity.
            vectors[str(label)] = vector if all(math.isfinite(x) for x in vector) else []
        turns = [{"start": float(t.start) + offset, "end": float(t.end) + offset,
                  "speaker": str(s)} for t, _, s in annotation.itertracks(yield_label=True)]
        aligned = [{"start": float(t.start) + offset, "end": float(t.end) + offset,
                    "speaker": str(s)} for t, _, s in exclusive.itertracks(yield_label=True)]
        return turns, aligned, vectors
    except Exception as error:
        raise backend_fault('diarization', error, device) from error
    finally:
        if owns_pipeline:
            del pipeline
            gc.collect()


class SpeakerAnalyzer:
    """Resident Community-1 models, constructed only in the lease-owned child."""
    def __init__(self, config, *, cinematic=False, device='cuda', cache=None):
        self.config = config
        self.cache = cache
        backend, index = model_device(device)
        self.device = ('cuda:' + str(index)) if backend == 'cuda' else 'cpu'
        self.embedding_space = embedding_space_identity(config)
        self.model_identity = _model_identity(config)
        self.cache_policy_identity = _cache_policy_identity()
        settings = getattr(config, 'data', config)
        policy = settings.get('speaker_clustering_policy', 'community1-stock')
        if policy not in ('community1-stock', 'legacy-agglomerative'):
            raise MovieConfigurationError('speaker_clustering_policy_invalid', str(policy))
        try:
            target = torch_model_device(self.device)
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained(config['diarization_model'])
            if self.pipeline is None:
                raise MovieConfigurationError('speaker_model_load_failed', 'The local speaker model could not be opened')
            self.pipeline.to(target)
            self.pipeline._embedding.to(target)
            self.pipeline.segmentation_batch_size = 4
            self.pipeline.embedding_batch_size = 4
            self.pipeline.embedding_exclude_overlap = True
            if policy == 'legacy-agglomerative':
                from functools import partial
                from pyannote.audio.pipelines.clustering import AgglomerativeClustering
                cluster=AgglomerativeClustering(metric='cosine',constrained_assignment=True)
                cluster.instantiate({'method':'average','threshold':.5,'min_cluster_size':1})
                cluster.filter_embeddings=partial(cluster.filter_embeddings,min_active_ratio=.05)
                self.pipeline.clustering=cluster
                self.pipeline.klustering = 'AgglomerativeClustering'
                self.pipeline._expects_num_speakers = cluster.expects_num_clusters
        except Exception as error:
            raise backend_fault('speaker_model_load', error, self.device) from error
        # Centroids exported by Community-1 are in its native embedding space.
        # Loading an independent ONNX encoder for only reference/refinement
        # vectors made equal-length but incompatible vectors comparable in v1.
        self.reference_embedding = None
        self._policy = {
            'version': 3, 'requested_device': self.device,
            'clustering_policy': policy,
            'clustering_class': type(self.pipeline.clustering).__name__,
            'cinematic_requested': bool(cinematic),
            'embedding_policy': EMBEDDING_POLICY,
            'embedding_space': self.embedding_space,
            'model_identity': self.model_identity,
            'external_onnx_configured': bool(settings.get('speaker_onnx_model')),
            'external_onnx_used': False,
            'embedding_exclude_overlap': True,
            'segmentation_batch_size': 4, 'embedding_batch_size': 4}
        # Opt-in (actor_beta.diart_turns): film-global speaker ids from the Diart pre-pass artifact. Absent = no key, so the
        # default policy (and every cache key derived from it) is byte-identical to before.
        self.diart_decisions = deque(maxlen=64)
        self._diart_reasons_logged = set()
        diart_setting = diart_turns.setting(config)
        if diart_setting is not None:
            self._policy['diart_turns'] = {'setting': diart_setting, **diart_turns.policy()}

    def policy_metadata(self):
        # Callers may serialize this; they may not mutate cache-key state.
        return json.loads(json.dumps({**self._policy, 'cache_policy_identity': self.cache_policy_identity}))

    def device_report(self):
        return {'requested': self.device, 'pipeline': str(self.pipeline.device),
                'embedding': str(self.pipeline._embedding.device), 'cpu_fallback': False,
                'supporting_cpu_work': ['waveform-io', 'resampling', 'clustering-bookkeeping']}

    def _diart_note(self, used, reason, offset, sha256=None):
        """Record why a window used (or did not use) the Diart artifact; a fallback reason is logged once."""
        self.diart_decisions.append({'offset': float(offset), 'used': bool(used), 'reason': reason,
                                     'artifact_sha256': sha256})
        if not used and reason not in self._diart_reasons_logged:
            self._diart_reasons_logged.add(reason)
            logging.getLogger(__name__).warning('Diart turns not used (%s); offline pyannote analysis instead', reason)

    def _diart_analyze(self, wav, offset, source_fingerprint):
        """(turns, aligned, vectors) from the Diart artifact, or None after recording why pyannote must be used."""
        try:
            artifact = diart_turns.open_for(self.config, source_fingerprint)
            import soundfile as sf
            data, rate = sf.read(str(wav), dtype='float32', always_2d=True)
            if rate != diart_turns.SAMPLE_RATE:
                raise diart_turns.DiartTurnsUnavailable('window_rate_unsupported', str(rate))
            mono = data.mean(axis=1)
            start = float(offset)
            end = start + len(mono) / rate
            artifact.require_cover(start, end)
        except diart_turns.DiartTurnsUnavailable as error:
            self._diart_note(False, error.reason, offset)
            return None
        def compute():
            window = artifact.window(start, end)
            return [window['turns'], window['aligned'],
                    diart_turns.label_vectors(window, mono, start, self.embed)]
        if self.cache is None:
            result = compute()
        else:
            from .analysis_cache import file_identity
            result = self.cache.get_or_compute(
                diart_turns.NAMESPACE,
                {'artifact_sha256': artifact.sha256, 'audio': file_identity(wav), 'offset': round(start, 6),
                 'embedding_space': self.embedding_space, 'policy': self._policy,
                 'cache_policy': self.cache_policy_identity}, compute)
        self._diart_note(True, 'diart', offset, artifact.sha256)
        turns, aligned, vectors = result
        return list(turns), list(aligned), dict(vectors)

    def analyze(self, wav, offset=0.0, *, source_fingerprint=None):
        if 'diart_turns' in self._policy:
            result = self._diart_analyze(wav, offset, source_fingerprint)
            if result is not None:
                return result
        if self.cache is None:
            return diarize(self.config, wav, offset, pipeline=self.pipeline, device=self.device)
        from .analysis_cache import file_identity
        # Raw local-time analysis has no mutable actor identities or seek offset.
        # Offset is applied only to a detached copy after reading the cache.
        result = self.cache.get_or_compute(
            'community1-diarization-v3',
            {'audio': file_identity(wav), 'policy': self._policy,
             'cache_policy': self.cache_policy_identity},
            lambda: list(diarize(self.config, wav, 0, pipeline=self.pipeline, device=self.device)))
        turns, aligned, vectors = result
        shift = lambda rows: [{**row, 'start': row['start'] + offset, 'end': row['end'] + offset}
                              for row in rows]
        return shift(turns), shift(aligned), vectors

    def embed(self,waveform):
        import numpy as np
        import torch
        samples = np.ascontiguousarray(waveform, dtype=np.float32)
        if samples.ndim != 1 or not 4000 <= len(samples) <= 16000 * 120 or not np.isfinite(samples).all():
            raise MovieContentFallback('speaker_reference_invalid', 'Speaker embedding requires bounded finite mono 16 kHz speech')
        if np.max(np.abs(samples)) < 1e-5:
            raise MovieContentFallback('speaker_reference_silent', 'Speaker reference contains no usable signal')
        def compute():
            try:
                with torch.inference_mode():
                    tensor = torch.from_numpy(samples)[None, None, :].to(self.device)
                    result = self.pipeline._embedding(tensor)[0]
            except Exception as error:
                raise backend_fault('speaker_embedding', error, self.device) from error
            values = np.asarray(result, dtype=np.float32)
            if values.ndim != 1 or not np.isfinite(values).all() or np.linalg.norm(values) < 1e-8:
                raise MovieContentFallback('speaker_embedding_invalid', 'Speaker embedding returned insufficient voice evidence')
            return values.tolist()
        if self.cache is None:
            return compute()
        from .analysis_cache import array_identity
        return self.cache.get_or_compute(
            'community1-embedding-v3',
            {'audio': array_identity(samples, sample_rate=16000),
             'embedding_space': self.embedding_space, 'policy': self._policy,
             'cache_policy': self.cache_policy_identity}, compute)

    def refine_utterances(self, waveform, rate, segments, vectors, offset=0, speech_regions=(), known_speakers=None, *,
                          whisper_no_speech=False):
        """Use clear complete utterances to correct mixed window centroids.

        Returns candidates and confidence evidence, never an asserted cast list.
        Ambiguous or sub-half-second evidence stays explicitly unresolved.
        """
        import numpy as np
        from scipy.signal import resample_poly
        data=resample_poly(np.asarray(waveform,dtype=np.float32),16000,rate) if rate!=16000 else waveform
        samples=[]
        rejected=[]
        for segment in segments:
            if is_no_speech(segment,whisper_no_speech):continue
            tokens=segment.get('words') or [segment]
            phrases=[]
            for token in tokens:
                if not phrases or (phrases[-1] and (token['start']-phrases[-1][-1]['end']>(.8 if segment.get('speaker_unit') else .45) or token['end']-phrases[-1][0]['start']>8)):
                    phrases.append([])
                phrases[-1].append(token)
                if not segment.get('speaker_unit') and token['text'].strip().endswith(('.', '!', '?')):
                    phrases.append([])
            for phrase in phrases:
                if not phrase:continue
                begin,end=phrase[0]['start'],phrase[-1]['end']
                if end-begin<.25:continue
                support=[t for t in speech_regions if min(t['end'],end+offset)>max(t['start'],begin+offset)]
                if support:
                    begin=max(begin,min(t['start'] for t in support)-offset-.04)
                    end=min(end,max(t['end'] for t in support)-offset+.04)
                # Keep consonant attacks/decays around very short words. Padding
                # a cropped half-word with zeros degraded these actor embeddings.
                embed_begin=max(0,begin-.10);embed_end=min(len(data)/16000,end+.10)
                clip=np.asarray(data[round(embed_begin*16000):round(embed_end*16000)],dtype=np.float32)
                if len(clip)<4000:continue
                if len(clip)<8000:clip=np.pad(clip,(0,8000-len(clip)))
                try:
                    vector=self.embed(clip)
                except MovieContentFallback as error:
                    rejected.append({'start':phrase[0]['start']+offset,'end':phrase[-1]['end']+offset,
                                     'resolved':False,'reason_code':error.fault_code,
                                     'embedding_space':self.embedding_space})
                    continue
                vector=list(map(float,vector))
                if not all(math.isfinite(x) for x in vector) or sum(x*x for x in vector)<1e-8:continue
                samples.append({'start':phrase[0]['start']+offset,'end':phrase[-1]['end']+offset,'vector':vector,
                    'speech_start':begin+offset,'speech_end':end+offset})
        valid={k:v for k,v in vectors.items() if all(math.isfinite(x) for x in v) and sum(x*x for x in v)>1e-8}
        known_speakers=known_speakers or {}
        for sample in samples:
            ranked=sorted(((max(cosine(sample['vector'],v) for v in examples),ident)
                           for ident,examples in known_speakers.items() if examples),reverse=True)
            sample['known_scores']=ranked[:3]
            if ranked and ranked[0][0]>=.65 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.10):
                sample['known_speaker']='KNOWN::'+ranked[0][1]
                sample['known_scores']=ranked[:3]
        independent_anchors=[s for s in samples if s.get('known_speaker')]
        for sample in samples:
            if sample.get('known_speaker'):continue
            label=continuing_known_speaker(sample,sample['known_scores'],independent_anchors)
            if label:sample.update(known_speaker=label,known_basis='saved-reference-and-continuing-turn')
        prototypes={}
        anchors={}
        for sample in samples:
            scores=sorted(((cosine(sample['vector'],v),k) for k,v in valid.items()),reverse=True)
            if scores and scores[0][0]>=.6 and (len(scores)<2 or scores[0][0]-scores[1][0]>=.1):
                prototypes.setdefault(scores[0][1],[]).append(sample['vector'])
                anchors.setdefault(scores[0][1],[]).append(sample)
        refined={k:np.mean(v,axis=0).tolist() for k,v in prototypes.items()}
        templates={**valid,**refined}
        for ident,examples in known_speakers.items():
            if examples:templates['KNOWN::'+ident]=examples[0]
        exemplars={k:prototypes.get(k) or [v] for k,v in valid.items()}
        def ranking(vector):
            return sorted(((max(cosine(vector,x) for x in exemplars[k]),k) for k in exemplars),reverse=True)
        # New speakers may never become a window centroid when a short scene
        # contains rapid alternation. Admit a new provisional prototype only
        # with corroborating independent utterances, not a guessed actor count.
        unmatched=[s for s in samples if (ranking(s['vector']) or [(-1,None)])[0][0]<.45]
        if len(unmatched)>=2:
            from scipy.cluster.hierarchy import linkage,fcluster
            clusters=fcluster(linkage([s['vector'] for s in unmatched],method='average',metric='cosine'),.58,criterion='distance')
            for cluster in sorted(set(clusters)):
                members=[s for s,c in zip(unmatched,clusters) if c==cluster]
                if len(members)<2:continue
                known_labels={s['known_speaker'] for s in members if s.get('known_speaker')}
                if len(known_labels)==1:
                    known_label=next(iter(known_labels))
                    anchors_known=[s for s in members if s.get('known_speaker')==known_label]
                    if len(anchors_known)>=2:
                        for member in members:
                            if member.get('known_speaker'):continue
                            ranked=sorted(((max(cosine(member['vector'],v) for v in examples),ident)
                                           for ident,examples in known_speakers.items() if examples),reverse=True)
                            if ranked and 'KNOWN::'+ranked[0][1]==known_label and ranked[0][0]>=.4 and (
                                len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.1):
                                member.update(known_speaker=known_label,known_scores=ranked[:3],
                                              known_basis='reference-confirmed-utterance-cluster')
                        if all(s.get('known_speaker') for s in members):continue
                ident='UTTERANCE_%02d'%cluster
                templates[ident]=np.mean([s['vector'] for s in members],axis=0).tolist()
                exemplars[ident]=[s['vector'] for s in members]
                anchors[ident]=members
        turns,evidence=[],list(rejected)
        for sample in samples:
            if sample.get('known_speaker'):
                ident=sample['known_speaker']
                evidence.append({'start':sample['start'],'end':sample['end'],'candidates':sample['known_scores'],
                                 'resolved':True,'basis':sample.get('known_basis','saved-actor-reference'),'speaker':ident,
                                 'speech_start':sample['speech_start'],'speech_end':sample['speech_end']})
                turns.append({'start':sample['start'],'end':sample['end'],'speaker':ident,
                              'speech_start':sample['speech_start'],'speech_end':sample['speech_end']})
                continue
            scores=ranking(sample['vector'])
            best=scores[0] if scores else (-1,None)
            margin=best[0]-(scores[1][0] if len(scores)>1 else -1)
            resolved=best[0]>=.45 and margin>=.08
            nearby=any(min(abs(sample['end']-a['start']),abs(sample['start']-a['end']))<=.65
                       for a in anchors.get(best[1],[]) if a is not sample)
            contextual=not resolved and best[0]>=.4 and margin>=.15 and nearby and len(anchors.get(best[1],[]))>=2
            resolved=resolved or contextual
            evidence.append({'start':sample['start'],'end':sample['end'],'candidates':scores[:3],
                             'resolved':resolved,'margin':margin,'context_supported':contextual,
                             'reason_code':None if resolved else ('speaker_match_ambiguous' if best[0]>=.45 else 'speaker_match_below_threshold'),
                             'embedding_space':self.embedding_space,
                             'speech_start':sample['speech_start'],'speech_end':sample['speech_end']})
            if resolved:turns.append({'start':sample['start'],'end':sample['end'],'speaker':best[1],
                                     'speech_start':sample['speech_start'],'speech_end':sample['speech_end']})
        return turns,templates,evidence


class SpeakerBank:
    def __init__(self, store, job_id, config, voice_pool=None):
        self.store, self.job_id, self.config = store, job_id, config
        self.pool = voice_pool
        self.known = store.speakers(job_id)
        self.embedding_space = embedding_space_identity(config)

    def bind(self, vectors):
        mapping, problems, used = {}, [], set()
        self.last_bind_evidence = {}
        for label, vector in vectors.items():
            if not vector or not all(math.isfinite(x) for x in vector) or sum(x*x for x in vector) < 1e-8:
                mapping[label] = None
                self.last_bind_evidence[label] = {'reason': 'embedding_invalid', 'best': None, 'margin': None,
                                                   'candidates': [], 'decision': None}
                problems.append({"kind": "speaker_embedding_invalid", "local_speaker": label})
                continue
            if label.startswith('KNOWN::'):
                ident=label[len('KNOWN::'):]
                existing=self.known.get(ident,{})
                evidence=existing.get('reference_vectors',[])
                reference_score = (max(cosine(vector,v) for v in evidence)
                                   if existing.get('reference_vectors_space') == self.embedding_space and evidence else None)
                if (existing.get('reference_vectors_space') != self.embedding_space or
                        not evidence or reference_score<.99):
                    mapping[label]=None
                    self.last_bind_evidence[label] = {'reason': 'reference_mismatch', 'best': reference_score,
                        'margin': None, 'candidates': [] if reference_score is None else [(reference_score, ident)],
                        'decision': None, 'reference_similarity_threshold': .99}
                    problems.append({'kind':'speaker_reference_mismatch','local_speaker':label})
                    continue
                self.last_bind_evidence[label] = {'reason': 'known_ok', 'best': reference_score, 'margin': None,
                    'candidates': [(reference_score, ident)], 'decision': ident, 'reference_similarity_threshold': .99}
                mapping[label]=ident;used.add(ident)
                continue
            ranked = sorted(((cosine(vector, value["embedding"]), key)
                             for key, value in self.known.items() if value.get("embedding") and
                             value.get('embedding_space') == self.embedding_space),
                            reverse=True)
            best = ranked[0] if ranked else (-1, None)
            margin = best[0] - (ranked[1][0] if len(ranked) > 1 else -1)
            observed = {'best': best[0] if ranked else None, 'margin': margin if ranked else None,
                        'candidates': ranked[:3], 'similarity_threshold': self.config['speaker_similarity'],
                        'margin_threshold': self.config['speaker_margin']}
            if best[0] >= self.config["speaker_similarity"]:
                if best[1] in used or margin < self.config["speaker_margin"]:
                    mapping[label] = None
                    self.last_bind_evidence[label] = {**observed, 'reason': 'ambiguous', 'decision': None,
                        'actor_already_used': best[1] in used, 'margin_below_threshold': margin < self.config['speaker_margin']}
                    problems.append({"kind": "speaker_ambiguous", "local_speaker": label,
                                     "candidates": ranked[:3]})
                    continue
                ident = best[1]
                self.last_bind_evidence[label] = {**observed, 'reason': 'bound_existing', 'decision': ident}
                old = self.known[ident]
                old["embedding"] = [(x * 4 + y) / 5 for x, y in zip(old["embedding"], vector)]
            else:
                incompatible = [ident for ident, value in self.known.items()
                                if value.get('embedding_space') != self.embedding_space]
                if incompatible:
                    # A new canonical vector cannot prove that it differs from
                    # an old encoder's stored centroid. Keep the old identities
                    # and references, then re-embed them via reference_vectors.
                    mapping[label] = None
                    self.last_bind_evidence[label] = {**observed, 'reason': 'migration_required', 'decision': None}
                    problems.append({'kind':'speaker_embedding_migration_required',
                                     'local_speaker':label, 'speakers':incompatible,
                                     'detail':'Saved identities need canonical source-reference embeddings before comparison'})
                    continue
                ident = f"speaker-{len(self.known) + 1:03d}"
                if self.pool is not None and len(self.known) >= len(self.pool):
                    mapping[label] = None
                    self.last_bind_evidence[label] = {**observed, 'reason': 'voice_pool_exhausted', 'decision': None}
                    problems.append({"kind": "voice_pool_exhausted", "local_speaker": label})
                    continue
                self.last_bind_evidence[label] = {**observed, 'reason': 'new_identity', 'decision': ident}
                voice = self.pool[len(self.known)] if self.pool is not None else {'engine':'actor','speaker':ident}
                self.known[ident] = {"embedding": vector, "voice": voice,
                                     "label": ident, 'embedding_space':self.embedding_space}
            used.add(ident)
            mapping[label] = ident
            self.store.put_speaker(self.job_id, ident, self.known[ident])
        return mapping, problems

    def reference_vectors(self,encoder):
        """Retain whole-reference and individual solo-turn voice exemplars."""
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        from .media import file_hash
        result={}
        for ident,value in self.known.items():
            voice=value.get('voice',{});path=voice.get('reference')
            if not path:continue
            try:
                digest=file_hash(path)
            except OSError as error:
                raise MovieConfigurationError('speaker_reference_unavailable', str(error)) from error
            if digest!=voice.get('reference_sha256'):
                raise MovieConfigurationError('speaker_reference_changed', 'Saved actor reference changed')
            changed = False
            if (value.get('reference_vectors_sha256')!=digest or value.get('reference_vectors_version')!=3 or
                    value.get('reference_vectors_space')!=self.embedding_space or not value.get('reference_vectors')):
                wave,rate=sf.read(path,dtype='float32',always_2d=True)
                mono=resample_poly(wave.mean(axis=1),16000,rate).astype('float32')
                examples=[list(map(float,encoder(mono)))]
                intervals=voice.get('source_intervals',[])
                lengths=[round((end-begin)*16000) for begin,end in intervals]
                expected=sum(lengths)+1920*max(0,len(lengths)-1)
                if len(lengths)>1 and abs(expected-len(mono))<=len(lengths)*2:
                    offset=0
                    for count in lengths:
                        if count>=4000:examples.append(list(map(float,encoder(mono[offset:offset+count]))))
                        offset+=count+1920
                if len(mono)>=40000:
                    # Different phonetic portions of the same clean actor clip
                    # provide evidence for short returning lines. No audio or
                    # voice reference is modified.
                    starts=list(range(0,len(mono)-25600+1,12800))
                    if starts[-1]!=len(mono)-25600:starts.append(len(mono)-25600)
                    for at in starts:examples.append(list(map(float,encoder(mono[at:at+25600]))))
                value.update(reference_vectors=examples,reference_vectors_sha256=digest,reference_vectors_version=3,
                             reference_vectors_space=self.embedding_space)
                changed = True
            if value.get('embedding_space') != self.embedding_space:
                # Even a partial previous transaction may already have current
                # reference vectors but an old centroid. Repair from those
                # validated vectors without an encoder call. Keep ID/voice/PCM.
                value.update(embedding=np.mean(value['reference_vectors'],axis=0).tolist(),
                             embedding_space=self.embedding_space)
                changed = True
            if changed:
                self.store.put_speaker(self.job_id,ident,value)
            result[ident]=value['reference_vectors']
        return result

    def voice(self, ident):
        value = self.known.get(ident)
        if value is None:
            raise ValueError("Resolve the speaker before synthesizing this line")
        return value["voice"]

    def attach_reference(self, ident, path, transcript='', *, source_intervals=()):
        """Freeze the first accepted local reference; never use a stock fallback."""
        from .media import file_hash
        value = self.known[ident]
        if value['voice'].get('reference'):
            return value['voice']
        if value['voice']['engine'] != 'actor':
            raise ValueError('A stock voice identity cannot become an actor clone implicitly')
        value['voice'].update(reference=str(Path(path).resolve()), reference_sha256=file_hash(path),
                              reference_text=transcript, source_intervals=list(source_intervals))
        self.store.put_speaker(self.job_id,ident,value)
        return value['voice']


def cue_speaker(cue, turns, mapping, minimum):
    duration = max(.001, cue["end"] - cue["start"])
    overlap = {}
    for turn in turns:
        value = max(0, min(cue["end"], turn["end"]) - max(cue["start"], turn["start"]))
        overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0) + value
    ranked = sorted(overlap.items(), key=lambda pair: pair[1], reverse=True)
    if not ranked or ranked[0][1] / duration < minimum:
        return None, "insufficient speaker overlap"
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * .55:
        return None, "line covers more than one speaker; split or assign it"
    speaker = mapping.get(ranked[0][0])
    return speaker, None if speaker else "speaker identity is ambiguous"
