"""TV capability probe: the silent test clips the TV plays to show what it can really decode (4K, HEVC, AV1, 120 fps).

"Probably supported" from MediaSource.isTypeSupported says nothing about 4K: Samsung publishes that MSE (what hls.js uses) is limited to
FHD on several TV groups. The TV's Settings page has a "Test this TV" row: it asks the PC to make these clips (CPU only, at low priority,
about 3 s each, no audio track, a few MB to 15 MB each, kept in <state>/tv-probe), plays each one muted through MediaSource, and sends what
happened (size reached, time advanced, decoded and dropped frames, errors) to the PC's log as one ``tv_diagnostic`` line.

Nothing here runs on its own: clips are made only when the TV asks, nothing touches the GPU (so neither Sem nor the dubbing is affected), and
no clip is ever part of a movie stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

SPEC_VERSION = 'tv-probe-clips-v1'
SECONDS = 3
NAME = re.compile(r'[a-z0-9_]{1,40}')
# name, label, MSE codec string, encoder family, size, frame rate, bitrate, level; levels: H.264 5.1 = 4K30, HEVC 5.1 = 4K60, 5.2 = 4K120.
CLIPS = (
    {'name': 'h264_1080p60', 'label': 'H.264 1080p 60 fps (control)', 'codec': 'avc1.64002a', 'family': 'h264', 'width': 1920,
     'height': 1080, 'fps': 60, 'bitrate': '12M', 'level': '4.2'},
    {'name': 'h264_2160p30', 'label': 'H.264 4K 30 fps', 'codec': 'avc1.640033', 'family': 'h264', 'width': 3840,
     'height': 2160, 'fps': 30, 'bitrate': '30M', 'level': '5.1'},
    {'name': 'hevc_1080p60', 'label': 'HEVC 1080p 60 fps', 'codec': 'hvc1.1.6.L123.B0', 'family': 'hevc', 'width': 1920,
     'height': 1080, 'fps': 60, 'bitrate': '12M', 'level': '4.1'},
    {'name': 'hevc_2160p24', 'label': 'HEVC 4K 24 fps', 'codec': 'hvc1.1.6.L153.B0', 'family': 'hevc', 'width': 3840,
     'height': 2160, 'fps': 24, 'bitrate': '25M', 'level': '5.1'},
    {'name': 'hevc_2160p60', 'label': 'HEVC 4K 60 fps', 'codec': 'hvc1.1.6.L153.B0', 'family': 'hevc', 'width': 3840,
     'height': 2160, 'fps': 60, 'bitrate': '35M', 'level': '5.1'},
    {'name': 'hevc_2160p120', 'label': 'HEVC 4K 120 fps', 'codec': 'hvc1.1.6.L156.B0', 'family': 'hevc', 'width': 3840,
     'height': 2160, 'fps': 120, 'bitrate': '40M', 'level': '5.2'},
    {'name': 'hevc10_2160p24', 'label': 'HEVC Main10 4K 24 fps', 'codec': 'hvc1.2.4.L153.B0', 'family': 'hevc10', 'width': 3840,
     'height': 2160, 'fps': 24, 'bitrate': '25M', 'level': '5.1'},
    {'name': 'av1_2160p60', 'label': 'AV1 4K 60 fps', 'codec': 'av01.0.13M.08', 'family': 'av1', 'width': 3840,
     'height': 2160, 'fps': 60, 'bitrate': '35M', 'level': '5.1'},
)
EXPECTED_CODEC = {'h264': 'h264', 'hevc': 'hevc', 'hevc10': 'hevc', 'av1': 'av1'}
LOW_PRIORITY = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)} \
    if os.name == 'nt' else {}


def spec(name):
    return next((c for c in CLIPS if c['name'] == name), None)


def ffmpeg_args(ffmpeg, clip, output, seconds=None):
    """The ffmpeg command for one clip: a synthetic moving test pattern, no audio, fragmented MP4 (init + media in one file, ready to
    append to a SourceBuffer), a key frame every second. CPU encoders only."""
    seconds = SECONDS if seconds is None else seconds
    fps, family = int(clip['fps']), clip['family']
    video = ['-f', 'lavfi', '-i', 'testsrc2=size=%dx%d:rate=%d:duration=%d' % (clip['width'], clip['height'], fps, seconds), '-an',
             '-t', str(seconds)]
    rate, buffer = clip['bitrate'], '%dM' % (2 * int(clip['bitrate'].rstrip('M')))
    if family == 'h264':
        encode = ['-c:v', 'libx264', '-preset', 'ultrafast', '-profile:v', 'high', '-level', clip['level'], '-pix_fmt', 'yuv420p',
                  '-b:v', rate, '-maxrate', rate, '-bufsize', buffer, '-g', str(fps), '-keyint_min', str(fps), '-sc_threshold', '0']
    elif family in ('hevc', 'hevc10'):
        params = 'level-idc=%s:high-tier=0:keyint=%d:min-keyint=%d:scenecut=0:log-level=error:vbv-maxrate=%d:vbv-bufsize=%d' % (
            clip['level'], fps, fps, int(rate.rstrip('M')) * 1000, 2 * int(rate.rstrip('M')) * 1000)
        encode = ['-c:v', 'libx265', '-preset', 'ultrafast', '-tag:v', 'hvc1', '-pix_fmt', 'yuv420p10le' if family == 'hevc10' else 'yuv420p',
                  '-b:v', rate, '-x265-params', params]
    elif family == 'av1':
        encode = ['-c:v', 'libsvtav1', '-preset', '10', '-pix_fmt', 'yuv420p', '-b:v', rate, '-g', str(fps),
                  '-svtav1-params', 'tile-columns=2:tile-rows=1']
    else:
        raise ValueError('unknown clip family %r' % (family,))
    return [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', *video, *encode,
            '-f', 'mp4', '-movflags', '+frag_keyframe+empty_moov+default_base_moof', '-y', str(output)]


class ProbeClips:
    """Makes and serves the clips; ``prepare()`` starts one background worker, ``status()`` is what the TV polls."""

    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.thread = None
        self.state = 'idle'
        self.clips = {}
        self.error = None
        self.updated = time.time()

    # ---- folder, manifest ------------------------------------------------------------------------------------------------
    def folder(self):
        folder = Path(self.config.state) / 'tv-probe'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _manifest(self):
        try:
            data = json.loads((self.folder() / 'manifest.json').read_text(encoding='utf-8'))
            return data if data.get('spec') == SPEC_VERSION else {}
        except (OSError, ValueError):
            return {}

    def _row(self, clip, state='pending', error=None, size=0):
        return {'name': clip['name'], 'label': clip['label'], 'codec': clip['codec'], 'width': clip['width'], 'height': clip['height'],
                'fps': clip['fps'], 'state': state, 'bytes': size, 'error': error}

    def _ready_on_disk(self, clip, manifest):
        path = self.folder() / (clip['name'] + '.mp4')
        saved = (manifest.get('clips') or {}).get(clip['name'])
        try:
            return bool(saved and path.is_file() and path.stat().st_size == saved.get('bytes') and saved.get('ok')
                        and saved.get('spec') == clip_signature(clip))
        except OSError:
            return False

    # ---- API ---------------------------------------------------------------------------------------------------------------
    def status(self):
        with self.lock:
            if self.state == 'working' or self.clips:
                rows = [dict(self.clips.get(c['name']) or self._row(c)) for c in CLIPS]
                return self._status(rows)
        manifest = self._manifest()
        rows = []
        for clip in CLIPS:
            if self._ready_on_disk(clip, manifest):
                rows.append(self._row(clip, 'ready', size=manifest['clips'][clip['name']]['bytes']))
            else:
                rows.append(self._row(clip))
        state = 'ready' if all(r['state'] == 'ready' for r in rows) else 'idle'
        return self._status(rows, state)

    def _status(self, rows, state=None):
        done = sum(1 for r in rows if r['state'] in ('ready', 'failed'))
        return {'state': state or self.state, 'clips': rows, 'done': done, 'total': len(rows), 'seconds': SECONDS,
                'folder_note': 'clips are made on the PC CPU and kept in the app state folder (tv-probe)', 'error': self.error}

    def prepare(self):
        """Start making the missing clips (once; a second call while working just reports). Returns status()."""
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return self._status([dict(self.clips.get(c['name']) or self._row(c)) for c in CLIPS])
            manifest = self._manifest()
            self.clips = {c['name']: (self._row(c, 'ready', size=manifest['clips'][c['name']]['bytes'])
                                      if self._ready_on_disk(c, manifest) else self._row(c)) for c in CLIPS}
            self.error = None
            if all(r['state'] == 'ready' for r in self.clips.values()):
                self.state = 'ready'
                return self._status([dict(r) for r in self.clips.values()])
            self.state = 'working'
            self.thread = threading.Thread(target=self._make, name='tv-probe-clips', daemon=True)
            self.thread.start()
            return self._status([dict(self.clips.get(c['name'])) for c in CLIPS])

    def path(self, name):
        """The clip file when ``name`` is one of the known clips and it was made and checked, else None. Never builds a path from input."""
        clip = spec(name) if isinstance(name, str) and NAME.fullmatch(name) else None
        if clip is None or not self._ready_on_disk(clip, self._manifest()):
            return None
        return self.folder() / (clip['name'] + '.mp4')

    # ---- the worker --------------------------------------------------------------------------------------------------------
    def _make(self):
        ffmpeg = self.config['ffmpeg']
        manifest = self._manifest()
        saved = dict(manifest.get('clips') or {})
        try:
            for clip in CLIPS:
                with self.lock:
                    if self.clips[clip['name']]['state'] == 'ready':
                        continue
                row = self._make_one(ffmpeg, clip)
                with self.lock:
                    self.clips[clip['name']] = row
                    self.updated = time.time()
                if row['state'] == 'ready':
                    saved[clip['name']] = {'ok': True, 'bytes': row['bytes'], 'spec': clip_signature(clip)}
                    self._save(saved)
            with self.lock:
                self.state = 'ready' if all(r['state'] == 'ready' for r in self.clips.values()) else 'partial'
        except Exception as error:  # noqa: BLE001 - reported to the TV; never raised into the server
            with self.lock:
                self.state, self.error = 'error', '%s: %s' % (type(error).__name__, str(error)[:200])

    def _save(self, clips):
        folder = self.folder()
        temporary = folder / 'manifest.json.partial'
        temporary.write_text(json.dumps({'spec': SPEC_VERSION, 'clips': clips}), encoding='utf-8')
        os.replace(temporary, folder / 'manifest.json')

    def _make_one(self, ffmpeg, clip):
        folder = self.folder()
        final, temporary = folder / (clip['name'] + '.mp4'), folder / (clip['name'] + '.mp4.partial')
        try:
            done = subprocess.run(ffmpeg_args(ffmpeg, clip, temporary), capture_output=True, timeout=600, **LOW_PRIORITY)
            if done.returncode != 0:
                return self._row(clip, 'failed', 'encoder failed: ' + done.stderr.decode('utf-8', 'replace')[-160:].strip())
            problem = check_clip(self.config['ffprobe'], temporary, clip)
            if problem:
                return self._row(clip, 'failed', problem)
            os.replace(temporary, final)
            return self._row(clip, 'ready', size=final.stat().st_size)
        except (OSError, subprocess.SubprocessError) as error:
            return self._row(clip, 'failed', '%s: %s' % (type(error).__name__, str(error)[:160]))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def clip_signature(clip):
    return hashlib.sha256(json.dumps([SPEC_VERSION, SECONDS, clip], sort_keys=True).encode()).hexdigest()[:16]


def check_clip(ffprobe, path, clip):
    """None when the file is what the clip promises (codec, size, frame rate, about SECONDS long, NO audio stream), else the problem."""
    try:
        out = subprocess.run([str(ffprobe), '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], capture_output=True,
                             timeout=60, **LOW_PRIORITY)
        data = json.loads(out.stdout or b'{}')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return 'ffprobe failed: %s' % type(error).__name__
    streams = data.get('streams') or []
    if any(s.get('codec_type') == 'audio' for s in streams):
        return 'the clip has an audio track'
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    if video is None:
        return 'no video stream'
    if video.get('codec_name') != EXPECTED_CODEC[clip['family']]:
        return 'codec is %s' % video.get('codec_name')
    if (video.get('width'), video.get('height')) != (clip['width'], clip['height']):
        return 'size is %sx%s' % (video.get('width'), video.get('height'))
    try:
        numerator, denominator = str(video.get('r_frame_rate', '0/1')).split('/')
        rate = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        rate = 0.0
    if abs(rate - clip['fps']) > 0.5:
        return 'frame rate is %.2f' % rate
    duration = float((data.get('format') or {}).get('duration') or 0)
    if not SECONDS - 0.6 <= duration <= SECONDS + 1.0:
        return 'duration is %.1f s' % duration
    return None
