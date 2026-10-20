"""Bounded CPU overlap candidates using ClearerVoice's MossFormer2.

Upstream: modelscope/ClearerVoice-Studio (Apache-2.0), pinned separately.
The two outputs are unassigned vocal candidates, not verified actor identities.
This module never replaces the full-band music/effects bed.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys,time
from .media import file_hash

MODEL_SHA256='00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a'


def source_level(stems,mixture):
    """One common gain preserves the loud/quiet relationship between heads."""
    import numpy as np
    stems=np.asarray(stems,dtype=np.float32);mixture=np.asarray(mixture,dtype=np.float32)
    if stems.ndim!=2 or stems.shape!=(2,len(mixture)) or not np.isfinite(stems).all() or not np.isfinite(mixture).all():
        raise ValueError('Invalid overlap separator geometry or samples')
    combined=stems.sum(axis=0)
    power=float(np.dot(combined,combined))
    if power<1e-12:
        if np.max(np.abs(mixture),initial=0)<1e-6:return np.zeros_like(stems),{'gain':0,'residual_ratio':0}
        raise ValueError('Overlap model returned silence for audible input')
    gain=float(np.dot(combined,mixture)/power)
    if not 0<gain<100:raise ValueError('Overlap model source-level correction is invalid')
    result=stems*gain
    residual=float(np.sqrt(np.mean((mixture-result.sum(axis=0))**2))/max(1e-8,np.sqrt(np.mean(mixture**2))))
    return result,{'gain':gain,'residual_ratio':residual,
                  'output_rms':[float(np.sqrt(np.mean(x*x))) for x in result]}


class OverlapSeparator:
    def __init__(self,checkpoint,code_root,*,dependency_root=None,threads=4):
        import torch
        if not 1<=threads<=8:raise ValueError('Invalid CPU overlap thread count')
        if file_hash(checkpoint)!=MODEL_SHA256:raise ValueError('Overlap model identity changed')
        if dependency_root:sys.path.insert(0,str(Path(dependency_root).resolve()))
        sys.path.insert(0,str(Path(code_root).resolve()/'clearvoice/clearvoice/models'))
        from mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        torch.set_num_threads(threads)
        started=time.perf_counter()
        self.model=MossFormer2_SS_16K(SimpleNamespace(encoder_embedding_dim=512,mossformer_sequence_dim=512,
            num_mossformer_layer=24,encoder_kernel_size=16,num_spks=2)).model
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if 'model' in weights:weights=weights['model']
        self.model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True)
        self.model.eval().to('cpu');self.threads=threads;self.load_seconds=time.perf_counter()-started

    def separate(self,mixture,rate=16000,*,cancelled=lambda:False):
        import numpy as np
        import torch
        from .process import Cancelled
        samples=np.asarray(mixture,dtype=np.float32)
        if rate!=16000 or samples.ndim!=1 or not 8000<=len(samples)<=96000 or not np.isfinite(samples).all():
            raise ValueError('Overlap analysis requires .5 to 6 seconds of finite 16 kHz mono audio')
        if cancelled():raise Cancelled('Overlap processing cancelled')
        started=time.perf_counter()
        with torch.inference_mode():values=self.model(torch.from_numpy(samples)[None,:])
        if cancelled():raise Cancelled('Overlap processing cancelled')
        stems,levels=source_level(np.stack([v[0].numpy() for v in values]),samples)
        return stems,{'engine':'ClearerVoice MossFormer2_SS_16K','device':'cpu','threads':self.threads,
                      'seconds':time.perf_counter()-started,'input_seconds':len(samples)/rate,
                      'speaker_identity_verified':False,**levels}
