"""Owned continuous CPU encoder for incrementally supplied actor PCM."""
from __future__ import annotations
import os,subprocess
from pathlib import Path


class LiveMux:
    def __init__(self,ffmpeg,source,folder,origin,end,channels,owner,*,rate=48000,
                 channel_layout=None,output_channels=None):
        if rate!=48000 or channels not in (1,2,6) or not 0<=origin<end:
            raise ValueError('Invalid live mux geometry')
        if channels==6 and channel_layout not in ('5.1','5.1(side)'):
            raise ValueError('Six-channel PCM requires its actual source channel layout')
        output_channels=channels if output_channels is None else output_channels
        if output_channels not in (1,2,6) or output_channels>channels:
            raise ValueError('Choose an explicit supported output channel count')
        self.channel_layout=channel_layout if channels==6 else ('mono' if channels==1 else 'stereo')
        self.output_channels=output_channels
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        if (self.folder/'index.m3u8').exists():raise ValueError('Use a fresh mux directory')
        self.owner=owner;self.frames=0;self.expected=round(end*rate)-round(origin*rate)
        self.channels=channels;self.closed=False
        self.log=(self.folder/'mux.log').open('wb')
        args=[str(ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-ss',str(origin),'-i',str(source),
              '-f','f32le','-ar',str(rate),'-ac',str(channels)]
        if channels==6:args+=['-channel_layout',channel_layout]
        args+=['-i','pipe:0','-t',str(end-origin),'-map','0:v:0','-map','1:a:0','-c:v','libx264',
               '-preset','veryfast','-crf','20','-threads','2','-filter_threads','1',
               '-vf',"scale=w='min(3840,iw)':h='min(2160,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
               '-force_key_frames','expr:gte(t,n_forced*2)','-c:a','aac',
               '-b:a',{1:'128k',2:'256k',6:'384k'}[output_channels],'-ac',str(output_channels),
               '-f','hls','-hls_time','2','-hls_list_size','0','-hls_playlist_type','event',
               '-hls_flags','temp_file','-hls_segment_filename',str(self.folder/'segment-%03d.ts'),
               '-y',str(self.folder/'index.m3u8')]
        flags={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name=='nt' else {'start_new_session':True}
        self.child=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log,**flags)
        owner.register(self.child)

    def write(self,samples):
        import numpy as np
        self.owner.check();samples=np.asarray(samples,dtype=np.float32)
        if self.closed or samples.ndim!=2 or samples.shape[1]!=self.channels or not np.isfinite(samples).all():
            raise ValueError('Invalid streaming PCM block')
        if self.frames+len(samples)>self.expected:raise ValueError('Streaming PCM exceeded its timeline')
        self.child.stdin.write(samples.astype('<f4',copy=False).tobytes());self.child.stdin.flush()
        self.frames+=len(samples)

    def finish(self):
        if self.frames!=self.expected:raise ValueError('Streaming PCM is incomplete')
        self.child.stdin.close()
        code=self.child.wait(timeout=60)
        self.owner.unregister(self.child);self.log.close();self.closed=True
        if code:raise RuntimeError('Live mux failed; inspect its private log')
        return {'frames':self.frames,'exit_code':code,'input_channel_layout':self.channel_layout,
                'output_channels':self.output_channels,'lossy_audio_codec':'aac'}

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.owner.terminate(self.child)
        finally:
            self.owner.unregister(self.child)
            self.child.stdin.close();self.log.close()
