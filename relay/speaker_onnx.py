"""Explicit-device WeSpeaker embeddings with pinned model identity.

Uses the WeSpeaker ResNet34-LM feature contract documented by
wenet-e2e/wespeaker and wespeakerruntime (Apache-2.0).
This optional encoder is not interchangeable with Community-1's native
centroids. SpeakerAnalyzer deliberately uses its native encoder for every
identity comparison instead of substituting this independent model.
"""
from __future__ import annotations
from .media import file_hash
from .language import backend_fault, model_device, prepare_model_runtime
from .movie_faults import MovieConfigurationError, MovieContentFallback

MODEL_SHA256='7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068'


class SpeakerEmbedding:
    def __init__(self,path,threads=2,*,device='cuda'):
        backend, index = model_device(device)
        self.device = ('cuda:' + str(index)) if backend == 'cuda' else 'cpu'
        self.embedding_space = 'wespeaker-resnet34-lm-onnx:' + MODEL_SHA256
        if not 1<=threads<=4:
            raise MovieConfigurationError('speaker_embedding_threads_invalid', 'Invalid embedding CPU helper thread count')
        try:
            prepare_model_runtime(self.device)
            import onnxruntime as ort
            if file_hash(path)!=MODEL_SHA256:
                raise MovieConfigurationError('speaker_embedding_identity_changed', 'Speaker embedding model identity changed')
            options=ort.SessionOptions()
            options.intra_op_num_threads=threads
            options.inter_op_num_threads=1
            if backend == 'cuda':
                if 'CUDAExecutionProvider' not in ort.get_available_providers():
                    raise MovieConfigurationError('onnx_cuda_unavailable',
                                                  'ONNX Runtime CUDA provider is unavailable; CPU inference was not substituted')
                # Avoid both graph-node CPU fallback and ORT's execution-error
                # fallback. A missing CUDA dependency must be a visible fault.
                options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
                providers=[('CUDAExecutionProvider', {'device_id':index})]
            else:
                providers=['CPUExecutionProvider']
            self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
            self.session.disable_fallback()
            active = self.session.get_providers()
            expected = 'CUDAExecutionProvider' if backend == 'cuda' else 'CPUExecutionProvider'
            if not active or active[0] != expected:
                raise MovieConfigurationError('speaker_embedding_device_mismatch',
                                              'ONNX Runtime did not activate the requested provider')
        except Exception as error:
            raise backend_fault('speaker_onnx_load', error, self.device) from error

    def device_report(self):
        return {'requested':self.device, 'providers':self.session.get_providers(),
                'cpu_fallback':False, 'embedding_space':self.embedding_space,
                'supporting_cpu_work':['resampling','filterbank-features']}

    def embed(self,waveform,sample_rate=16000):
        import numpy as np
        import torch
        import torchaudio.compliance.kaldi as kaldi
        from scipy.signal import resample_poly
        if not isinstance(sample_rate,int) or not 8000<=sample_rate<=192000:
            raise MovieContentFallback('speaker_reference_rate_invalid', 'Invalid speaker reference sample rate')
        samples=np.asarray(waveform,dtype=np.float32)
        if samples.ndim!=1 or not .25<=len(samples)/sample_rate<=30 or not np.isfinite(samples).all():
            raise MovieContentFallback('speaker_reference_invalid', 'Speaker embedding needs bounded finite mono speech')
        if np.max(np.abs(samples))<1e-5:
            raise MovieContentFallback('speaker_reference_silent', 'Speaker embedding reference is silent')
        if sample_rate!=16000:samples=resample_poly(samples,16000,sample_rate).astype('float32')
        features=kaldi.fbank(torch.from_numpy(samples)[None,:]*32768,num_mel_bins=80,
            frame_length=25,frame_shift=10,dither=0,sample_frequency=16000,window_type='hamming',use_energy=False).numpy()
        features-=features.mean(axis=0,keepdims=True)
        try:
            result=self.session.run(['embs'],{'feats':features[None,:,:]})[0][0]
        except Exception as error:
            raise backend_fault('speaker_onnx_embedding', error, self.device) from error
        if result.shape!=(256,) or not np.isfinite(result).all() or np.linalg.norm(result)<1e-8:
            raise MovieContentFallback('speaker_embedding_invalid', 'Speaker embedding result is invalid')
        return result
