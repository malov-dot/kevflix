"""Cached actor speech with guarded GPU ownership and natural timing."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .actor_voice import ActorVoice
from .actor_client import ActorClient
from .media import atomic_json,file_hash
from .resource_budget import ResourceBudget


class ActorRenderer:
    def __init__(self,config,owner):
        self.config,self.owner=config,owner
        self.engine=None
        self.capacity=0
        self.request_capacity=0
        self.budget=ResourceBudget.for_actor(config)
        self.manifest=Path(config['actor_runtime_manifest'])
        self.model=json.loads(self.manifest.read_text(encoding='utf-8-sig'))['modelSha256']

    def prepare_references(self,voices):
        import soundfile as sf
        durations=[sf.info(v['reference']).duration for v in voices]
        if any(d>self.config['actor_reference_seconds'] for d in durations):
            raise ValueError('Actor reference exceeds the configured limit')
        self.request_capacity=min(float(self.config['actor_reference_seconds']),
                                  max((math.ceil(d*10)/10+.01 for d in durations),default=0))

    def render(self,text,voice,directory,window,*,seed=42,reference_only=False,context_key=None):
        import soundfile as sf
        self.owner.check()
        reference=voice.get('reference')
        if not reference or file_hash(reference)!=voice.get('reference_sha256'):
            raise ValueError('Actor reference is missing or changed; resolve it before dubbing')
        identity=[text,voice,self.model,file_hash(self.manifest),'native-duration-v1',seed,reference_only,context_key]
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
        path,receipt=folder/(key+'.wav'),folder/(key+'.json')
        if path.is_file() and receipt.is_file():
            saved=json.loads(receipt.read_text(encoding='utf-8'))
            if file_hash(path)==saved['sha256']:
                return path,{**saved,'timing_overrun':saved['seconds']>window,'window_seconds':window,'cache_hit':True}
        try:
            if isinstance(self.engine,ActorClient) and self.engine.child.poll() is not None:self.close()
            duration=sf.info(reference).duration
            if duration>self.config['actor_reference_seconds']:
                raise ValueError('Actor reference exceeds the configured limit')
            needed=max(self.request_capacity,min(float(self.config['actor_reference_seconds']),math.ceil(duration*10)/10+.01))
            if self.engine is not None and needed>self.capacity:
                self.close()
            if self.engine is None:
                if self.config['actor_backend']=='cuda':
                    self.engine=ActorClient(self.manifest,needed,self.budget,self.owner,folder,self.config['python'])
                else:
                    self.engine=ActorVoice(self.manifest,backend='cpu',threads=2,max_reference_seconds=needed)
                self.capacity=needed
            if self.config['actor_backend']=='cuda':
                response=self.engine.synthesize_file(text,reference,'' if reference_only else voice.get('reference_text',''),path,seed)
                result=response['receipt'];seconds=response['seconds']
            else:
                audio,result=self.engine.synthesize(text,reference,'' if reference_only else voice.get('reference_text',''),
                    cancelled=self.owner.cancelled.is_set,seed=seed)
                temporary=path.with_suffix('.tmp.wav')
                sf.write(temporary,audio,48000,subtype='FLOAT');temporary.replace(path)
                seconds=len(audio)/48000
        except BaseException:
            self.close()
            raise
        self.owner.check()
        result.update(text=text,voice=voice,seconds=seconds,speed=1.0,sha256=file_hash(path),
                      timing_overrun=seconds>window,window_seconds=window,cache_hit=False,seed=seed,reference_only=reference_only)
        atomic_json(receipt,result)
        return path,result

    def close(self):
        if self.engine:
            self.engine.close()
            self.engine=None
            self.capacity=0
