"""Opt-in Demucs music/effects bed for an existing actor window (MIX ONLY).

Imported only by demucs_bed_child.py in an owned child process; never by the
PC server. remix_window() never regenerates speech, references or ASR. It
changes only the background component inside the removal envelope the
original mixer already applied:

    new_final = existing_final + allowed_env * (demucs_bed - (S(original) - S(speech)))

S is the ffmpeg -ac 2 convention the worker used to make existing_final from
the mono/stereo/5.1 actor mix (to_stereo), so the placed voice it already
contains keeps its gain and timing mathematically; the result is re-quantized
to PCM16, so changed frames are not claimed bitexact. allowed_env is the
mixer's own removal envelope times its protection gain (zero in protected
cores, ramping over the 10 ms transitions outside them, exactly as the mixer
edited). Frames where allowed_env == 0 (outside every removal envelope, inside
protected cores, outside the render window, or inside a carried take that
could not be proven) are copied as exact PCM16 values. Unprovable inputs and
clipping are refused, never guessed or limited.
"""
import ctypes
import hashlib
import json
import math
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .cinematic import replacement_envelope
from .original_audio_policy import original_protection_masks

VERSION = 'demucs-bed-option-v2'
CONTEXT_SECONDS = .5
# Removal padding DialogueLedger.entry and the actor mixer apply to new takes.
LEAD_SECONDS, TAIL_SECONDS = .12, .25
EDGE_FADE_SECONDS = .04  # replacement_envelope default
PROTECTION_TRANSITION_SECONDS = .01  # protect_dialogue_mix default
TIMEOUT_SECONDS = 120
MAX_WORKING_SET = 8 * 1024**3
MIN_AVAILABLE = 12 * 1024**3
BELOW_NORMAL = 0x4000
RATE = 48000
_MODEL = {}
_NETWORK = {'installed': False, 'block': False, 'attempts': []}


class Refused(Exception):
    """Geometry, hash, provenance or clipping cannot be established; fatal for this window."""


class ResourceAbort(Exception):
    """Sem not idle, Sem restarted, stale status, RAM or time bound: yield, retry later."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _audit(event, args):
    if _NETWORK['block'] and event in ('socket.connect', 'socket.connect_ex', 'socket.sendto', 'socket.getaddrinfo'):
        _NETWORK['attempts'].append(event)
        raise RuntimeError('network disabled during Demucs bed separation')


def _block_network():
    if not _NETWORK['installed']:
        sys.addaudithook(_audit)
        _NETWORK['installed'] = True
    _NETWORK['block'] = True


class _MemoryStatus(ctypes.Structure):
    _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


class _ProcessCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def _kernel32():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessCounters), wintypes.DWORD]
    return kernel


def available_ram():
    status = _MemoryStatus(); status.dwLength = ctypes.sizeof(status)
    if not _kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), 'GlobalMemoryStatusEx failed')
    return status.ullAvailPhys


def working_set():
    kernel = _kernel32()
    counters = _ProcessCounters(); counters.cb = ctypes.sizeof(counters)
    if not kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
    return counters.WorkingSetSize


def sem_idle_pid(status_file):
    from datetime import datetime, timezone
    state = json.loads(Path(status_file).read_text(encoding='utf-8-sig'))
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(state['at'].replace('Z', '+00:00'))).total_seconds()
    voice = state.get('voice') or {}
    if age > 15 or age < -2 or not state.get('serverReady') or state.get('busy') or state.get('codexVoiceActive') \
            or voice.get('active') or voice.get('closing'):
        raise ResourceAbort('Sem state is not fresh idle')
    return state['pid']


class Guard:
    """Child-process-only 200 ms Sem-idle / RAM / wall-time guard.

    Demucs cannot be interrupted, so a violation writes a ResourceYield abort
    file and hard-exits THIS child; the parent maps that to waiting_resources.
    """

    EXIT_CODE = 75

    def __init__(self, status_file, abort_path, start, timeout=TIMEOUT_SECONDS):
        self.status_file, self.abort_path = status_file, Path(abort_path)
        self.start, self.timeout = start, timeout
        self.done, self.ready = threading.Event(), threading.Event()
        self.telemetry = {'guard_checks': 0, 'peak_working_set': 0, 'min_ram_available': None,
                          'priority': 'below_normal', 'interval_seconds': .2, 'timeout_seconds': timeout}
        self.pid = None

    def check(self):
        if time.monotonic() - self.start > self.timeout:
            raise ResourceAbort('%d second bound' % self.timeout)
        if sem_idle_pid(self.status_file) != self.pid:
            raise ResourceAbort('Sem PID changed')
        rss, avail = working_set(), available_ram()
        t = self.telemetry
        t['guard_checks'] += 1
        t['peak_working_set'] = max(t['peak_working_set'], rss)
        t['min_ram_available'] = min(avail, t['min_ram_available'] or avail)
        if rss > MAX_WORKING_SET or avail < MIN_AVAILABLE:
            raise ResourceAbort('RAM reserve bound')

    def _run(self):
        try:
            while True:
                self.check()
                self.ready.set()
                if self.done.wait(.2):
                    return
        except Exception as error:
            kind = 'ResourceYield' if isinstance(error, (ResourceAbort, OSError, ValueError, KeyError)) else 'error'
            write_json(self.abort_path, {'kind': kind, 'error': str(error), 'telemetry': self.telemetry})
            os._exit(self.EXIT_CODE)

    def __enter__(self):
        kernel = _kernel32()
        handle = kernel.GetCurrentProcess()
        self.previous_priority = kernel.GetPriorityClass(handle)
        if not kernel.SetPriorityClass(handle, BELOW_NORMAL):
            raise OSError(ctypes.get_last_error(), 'SetPriorityClass failed')
        try:
            self.pid = sem_idle_pid(self.status_file)
        except (OSError, ValueError, KeyError) as error:
            raise ResourceAbort('Sem status unavailable: %s' % error)
        self.telemetry['sem_pid'] = self.pid
        threading.Thread(target=self._run, daemon=True, name='demucs-bed-guard').start()
        if not self.ready.wait(5):
            raise ResourceAbort('guard did not become ready')
        return self

    def __exit__(self, *exc):
        self.done.set()
        if self.previous_priority:
            kernel = _kernel32()
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), self.previous_priority)
        return False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def itu_stereo(x):
    """ITU 5.1 -> stereo without LFE: .414 L + .293 C + .293 SL (ffmpeg default -ac 2)."""
    import numpy as np
    return np.stack((x[:, 0] + 2**-.5 * (x[:, 2] + x[:, 4]), x[:, 1] + 2**-.5 * (x[:, 2] + x[:, 5])), 1) / (1 + 2**.5)


LAYOUT_CHANNELS = {'mono': 1, 'stereo': 2, '5.1': 6, '5.1(side)': 6}


def to_stereo(x, layout):
    """The ffmpeg -ac 2 convention the worker used for english.wav (verified:
    evidence/demucs-bed-option/ffmpeg-channel-convention.json): mono -> [m/sqrt2, m/sqrt2],
    stereo identity, 5.1/5.1(side) ITU without LFE. Explicit known layouts only."""
    import numpy as np
    if LAYOUT_CHANNELS.get(layout) != x.shape[1]:
        raise Refused('layout %r does not match %d channels' % (layout, x.shape[1]))
    if layout == 'mono':
        return np.repeat(x * 2**-.5, 2, axis=1)
    if layout == 'stereo':
        return x
    return itu_stereo(x)


def model_pins(model_dir, expected):
    """Existing htdemucs yaml + exactly one safetensors under an explicit absolute root, hash-pinned."""
    import yaml
    root = Path(model_dir)
    if not root.is_absolute() or not root.is_dir():
        raise Refused('model_dir must be an existing absolute directory')
    manifest = root / 'htdemucs.yaml'
    bag = yaml.safe_load(manifest.read_text())
    paths = [root / (str(name) + '.safetensors') for name in bag.get('models') or []]
    if len(paths) != 1 or not all(p.is_file() for p in [manifest, *paths]):
        raise Refused('expected exactly one existing htdemucs safetensors model')
    pins = {p.name: sha256(p) for p in [manifest, *paths]}
    if not expected or pins != dict(expected):
        raise Refused('Demucs model files do not match the configured pins')
    return bag, manifest, paths, pins


def load_model(model_dir, expected):
    key = (str(model_dir), json.dumps(expected, sort_keys=True))
    if _MODEL.get('key') == key:
        return _MODEL['model'], {'cached': True, 'load_seconds': 0.0, 'pins': _MODEL['pins'], 'model_dir': str(model_dir)}
    bag, manifest, paths, pins = model_pins(model_dir, expected)
    import torch
    from demucs.hf import load_safetensors_model
    from demucs.apply import BagOfModels
    started = time.monotonic()
    model = BagOfModels([load_safetensors_model(p) for p in paths], bag.get('weights'), bag.get('segment')).to('cpu').eval()
    if model.samplerate != 44100 or model.audio_channels != 2 or 'vocals' not in model.sources:
        raise Refused('unexpected htdemucs geometry')
    if torch.cuda.is_initialized():
        raise Refused('CUDA initialized in the Demucs child')
    _MODEL.update(key=key, model=model, pins=pins)
    return model, {'cached': False, 'load_seconds': time.monotonic() - started, 'pins': pins, 'model_dir': str(model_dir)}


def separate_bed(model, stereo48):
    """Sum of non-vocal Demucs stems at 48 kHz for one stereo span (mean/std normalized)."""
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from demucs.apply import apply_model
    wav = torch.from_numpy(np.ascontiguousarray(resample_poly(stereo48, 147, 160).T)).float()
    reference = wav.mean(0); mean = reference.mean(); std = reference.std() + 1e-8
    with torch.inference_mode():
        stems = apply_model(model, ((wav - mean) / std)[None], device='cpu', shifts=0, split=True,
                            overlap=.25, progress=False, num_workers=0)[0] * std + mean
    vocals = list(model.sources).index('vocals')
    bed = stems[[i for i in range(len(stems)) if i != vocals]].sum(0).T.numpy()
    bed48 = resample_poly(bed, 160, 147)[:len(stereo48)]
    if bed48.shape != stereo48.shape or not np.isfinite(bed48).all():
        raise Refused('Demucs bed geometry or values invalid')
    return bed48.astype(np.float64)


def merged_spans(mask, rate, context):
    import numpy as np
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    pad, spans = round(context * rate), []
    for left, right in zip(edges[::2], edges[1::2]):
        left, right = max(0, int(left) - pad), min(len(mask), int(right) + pad)
        if spans and left <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], right)
        else:
            spans.append([left, right])
    return spans


def _span(value, what):
    try:
        begin, end = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise Refused('invalid %s span' % what)
    if not (math.isfinite(begin) and math.isfinite(end) and end > begin):
        raise Refused('invalid %s span' % what)
    return [begin, end]


def _same(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-6
    except (TypeError, ValueError):
        return False


def placement_cue(placement, report, lines):
    """Cue ID of one post-fallback placement: explicit stamp, else the unique span match.

    report['utterances'] is recorded before beta fallbacks filter lines while
    report['placements'] is reassigned after them, so the two lists are never
    zipped. Legacy placements carry no ID: a new take's placement keeps its
    cue's source span; a reused canonical line's placement carries the take's
    own source span.
    """
    for key in ('cue', 'cue_id', 'id'):
        if placement.get(key):
            return str(placement[key]), 'explicit-' + key
    begin, end = placement.get('source_start'), placement.get('source_end')
    matches = {c['id'] for c in report.get('cues') or []
               if isinstance(c, dict) and c.get('id') in lines and _same(c.get('start'), begin) and _same(c.get('end'), end)}
    if not matches:
        matches = {cue for cue, line in lines.items() if line.get('canonical_id')
                   and _same(line.get('source_start'), begin) and _same(line.get('source_end'), end)}
    if len(matches) != 1:
        raise Refused('placement %.3f-%.3f does not map to exactly one cue' % (float(begin), float(end)))
    return matches.pop(), 'source-span'


def removal_intervals(report, carried=None, carried_exclusions=None):
    """Removal envelope intervals the original mixer used, with provenance.

    Only actual mixed placements count: each is mapped to its cue, beta
    fallback IDs are skipped. New takes: cue source span -.12/+.25. Reused
    canonical lines: their own replacement span. Carried takes: only canonical
    replacement spans proven by the caller; known but unproven carried takes
    become exclusions (left exactly as existing PCM). A carried ID with
    neither is refused rather than guessed.
    """
    excluded = set(report.get('beta_original_fallback') or {}) | set(report.get('beta_runtime_fallback') or {})
    lines = {}
    for line in report.get('utterances') or []:
        if line.get('cue') in lines:
            raise Refused('duplicate utterance cue %s' % line.get('cue'))
        lines[line.get('cue')] = line
    used, skipped, seen = [], sorted(c for c in lines if c in excluded), set()
    for placement in report.get('placements') or []:
        cue, basis = placement_cue(placement, report, lines)
        if cue in seen:
            raise Refused('two placements map to cue %s' % cue)
        seen.add(cue)
        if cue in excluded:
            if cue not in skipped:
                skipped.append(cue)
            continue
        line = lines.get(cue)
        if line is None:
            raise Refused('placement cue %s has no utterance' % cue)
        if line.get('canonical_id'):
            interval = _span((line.get('replacement_start'), line.get('replacement_end')), 'reused canonical')
            used.append({'cue': cue, 'basis': 'reused-canonical/' + basis, 'interval': interval}); continue
        begin, end = _span((placement.get('source_start'), placement.get('source_end')), 'placement source')
        used.append({'cue': cue, 'basis': 'placement-source/' + basis, 'interval': [begin - LEAD_SECONDS, end + TAIL_SECONDS]})
    carried, carried_exclusions = dict(carried or {}), dict(carried_exclusions or {})
    proven, unproven = {}, {}
    for cue in report.get('carried_utterances') or []:
        if cue in carried:
            proven[cue] = {**carried[cue], 'interval': _span(carried[cue]['interval'], 'carried canonical')}
        elif cue in carried_exclusions:
            unproven[cue] = {**carried_exclusions[cue], 'span': _span(carried_exclusions[cue]['span'], 'carried exclusion')}
        else:
            raise Refused('carried utterance %s has no canonical envelope or bounded exclusion' % cue)
    return used, skipped, proven, unproven


def remix_window(config, original_wav, speech_wav, existing_final_wav, report, output, owner=None,
                 *, carried=None, carried_exclusions=None, separate=None, guard=None, deadline_start=None):
    """Write `output` (PCM16, geometry of existing_final_wav) and return a receipt dict.

    config: recipe mapping (model_dir, model_pins, sem_status_file, threads,
    timeout_seconds, context_seconds, abort_path). separate/guard are
    injection points for model-free checks; production passes neither.
    Raises Refused (fatal) or ResourceAbort (yield).
    """
    import numpy as np
    import soundfile as sf
    started = time.monotonic()
    config = dict(config or {})
    context = float(config.get('context_seconds', CONTEXT_SECONDS))
    threads = int(config.get('threads', 4))
    output = Path(output)
    inputs = {'original': Path(original_wav), 'speech': Path(speech_wav), 'existing_final': Path(existing_final_wav)}
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    if report.get('sha256') != input_hashes['existing_final']:
        raise Refused('existing final does not match report sha256')
    stem_receipt = inputs['speech'].with_suffix('.json')
    try:
        stem_sha = json.loads(stem_receipt.read_text(encoding='utf-8-sig')).get('sha256')
    except (OSError, ValueError, AttributeError):
        stem_sha = None
    if stem_sha != input_hashes['speech']:
        raise Refused('speech stem is not the one recorded in %s' % stem_receipt.name)
    layout = report.get('channel_layout')
    if layout not in LAYOUT_CHANNELS or (LAYOUT_CHANNELS[layout] == 6 and
                                         (report.get('channel_layout_evidence') or {}).get('lfe_index', 3) != 3):
        raise Refused('unsupported channel layout %r' % layout)
    original, rate = sf.read(str(inputs['original']), dtype='float64', always_2d=True)
    speech, speech_rate = sf.read(str(inputs['speech']), dtype='float64', always_2d=True)
    final_info = sf.info(str(inputs['existing_final']))
    existing, final_rate = sf.read(str(inputs['existing_final']), dtype='int16', always_2d=True)
    origin, render_start, render_end = float(report['origin']), float(report['render_start']), float(report['render_end'])
    crop_left = round((render_start - origin) * rate)  # the worker's ffmpeg -ss (start-origin)
    crop_frames = round(render_end * rate) - round(render_start * rate)  # coordinator.audio_chunk frames
    if not (rate == speech_rate == final_rate == RATE and original.shape == speech.shape
            and original.shape[1] == LAYOUT_CHANNELS[layout]
            and final_info.subtype == 'PCM_16' and existing.shape == (crop_frames, 2)
            and 0 <= crop_left and crop_left + crop_frames <= len(original)
            and np.isfinite(original).all() and np.isfinite(speech).all()):
        raise Refused('input geometry mismatch')
    used, skipped, proven, unproven = removal_intervals(report, carried, carried_exclusions)
    protection = report.get('original_audio_protection') or {}
    protected_intervals = [_span(x, 'protected') for x in (protection.get('applied_protected_intervals') or [])]
    protected_intervals += [s for s in (_span(x, 'fallback') for x in (report.get('beta_fallback_intervals') or []))
                            if s not in protected_intervals]
    frames = len(original)
    intervals = [u['interval'] for u in used] + [c['interval'] for c in proven.values()]
    envelope = replacement_envelope(frames, rate, origin, intervals).astype(np.float64)
    protected, allowed_gain = original_protection_masks(frames, rate, origin, protected_intervals,
                                                        transition_seconds=PROTECTION_TRANSITION_SECONDS)
    allowed = envelope * allowed_gain
    allowed[protected] = 0
    exclusion = replacement_envelope(frames, rate, origin, [c['span'] for c in unproven.values()]) > 0
    if exclusion.any():
        # Widen by the edge fade so an envelope fade touching the excluded take stays exact too.
        for left, right in merged_spans(exclusion, rate, EDGE_FADE_SECONDS):
            exclusion[left:right] = True
    allowed[exclusion] = 0
    window = np.zeros(frames, dtype=bool); window[crop_left:crop_left + crop_frames] = True
    allowed[~window] = 0
    crop = slice(crop_left, crop_left + crop_frames)
    removal_in_window = envelope[crop] > 0
    partial = []
    if unproven:
        partial.append({'reason': 'carried take not proven; existing PCM kept (residual original speech may remain)',
                        'cues': sorted(unproven), 'frames': int(np.count_nonzero(exclusion[crop] & removal_in_window))})
    receipt = {'version': VERSION, 'state': 'unchanged',
               'formula': 'existing + allowed_env*(demucs_bed - (S(original)-S(speech))), S = ffmpeg -ac 2 convention',
               'inputs': {k: str(v) for k, v in inputs.items()}, 'input_sha256': input_hashes,
               'speech_receipt': str(stem_receipt), 'speech_receipt_sha256': stem_sha,
               'geometry': {'rate': rate, 'source_frames': frames, 'source_channels': LAYOUT_CHANNELS[layout], 'channel_layout': layout,
                            'origin': origin, 'render_start': render_start, 'render_end': render_end,
                            'crop_left': crop_left, 'output_frames': crop_frames, 'output_channels': 2, 'subtype': 'PCM_16'},
               'envelope_cues': used, 'excluded_fallback_cues': skipped, 'carried_proven': proven,
               'carried_unproven': unproven, 'partial_bed_coverage': partial,
               'protected_intervals': protected_intervals, 'lead_seconds': LEAD_SECONDS, 'tail_seconds': TAIL_SECONDS,
               'edge_fade_seconds': EDGE_FADE_SECONDS, 'protection_transition_seconds': PROTECTION_TRANSITION_SECONDS,
               'masks': {'removal_frames': int(np.count_nonzero(removal_in_window)),
                         'allowed_frames': int(np.count_nonzero(allowed)),
                         'protected_frames_in_window': int(np.count_nonzero(protected[crop])),
                         'excluded_carry_frames_in_window': int(np.count_nonzero(exclusion[crop]))},
               'helper_sha256': sha256(__file__),
               'voice_component': 'Existing placed voice retained via existing+delta (gain/timing preserved mathematically); '
                                  'changed frames are re-quantized PCM16, not claimed bitexact. No TTS/reference/ASR change.'}
    if not np.any(allowed):
        return receipt  # caller keeps the existing final; nothing is written or claimed cleaned
    spans = merged_spans(allowed > 0, rate, context)
    receipt['separation_spans'] = [[origin + a / rate, origin + b / rate] for a, b in spans]
    original_itu, speech_itu = to_stereo(original, layout), to_stereo(speech, layout)
    delta = np.zeros((frames, 2), dtype=np.float64)
    timings, telemetry, model_info = [], {}, {'injected': True}
    if separate is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        previous_threads = torch.get_num_threads()
        _block_network()
        try:
            with Guard(config['sem_status_file'], config['abort_path'], deadline_start or started,
                       float(config.get('timeout_seconds', TIMEOUT_SECONDS))) as active:
                torch.set_num_threads(threads)
                if torch.cuda.is_initialized():
                    raise Refused('CUDA initialized before separation')
                model, model_info = load_model(config['model_dir'], config.get('model_pins'))
                for left, right in spans:
                    t = time.monotonic()
                    bed = separate_bed(model, original_itu[left:right])
                    timings.append({'seconds': time.monotonic() - t, 'audio_seconds': (right - left) / rate})
                    delta[left:right] = allowed[left:right, None] * (bed - (original_itu[left:right] - speech_itu[left:right]))
                if torch.cuda.is_initialized():
                    raise Refused('CUDA initialized during separation')
                telemetry = dict(active.telemetry)
        finally:
            _NETWORK['block'] = False
            torch.set_num_threads(previous_threads)
        receipt['cuda_initialized'] = False
    else:
        context_manager = guard if guard is not None else _NullGuard()
        with context_manager:
            for left, right in spans:
                if owner is not None and hasattr(owner, 'check'):
                    owner.check()
                t = time.monotonic()
                bed = np.asarray(separate(original_itu[left:right]), dtype=np.float64)
                if bed.shape != (right - left, 2) or not np.isfinite(bed).all():
                    raise Refused('injected bed geometry or values invalid')
                timings.append({'seconds': time.monotonic() - t, 'audio_seconds': (right - left) / rate})
                delta[left:right] = allowed[left:right, None] * (bed - (original_itu[left:right] - speech_itu[left:right]))
    changed = allowed[crop] > 0
    result = existing.copy()
    candidate = np.rint(existing[changed].astype(np.float64) + delta[crop][changed] * 32768.)
    peak = float(np.abs(candidate).max() / 32768.) if candidate.size else 0.
    if candidate.size and (candidate.max() > 32767 or candidate.min() < -32768):
        raise Refused('remix peak %.4f exceeds full scale; no limiter change allowed' % peak)
    result[changed] = candidate.astype(np.int16)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial.wav')
    sf.write(str(temporary), result, rate, subtype='PCM_16')
    written, _ = sf.read(str(temporary), dtype='int16', always_2d=True)
    exact_outside = bool(np.array_equal(written[~changed], existing[~changed]))
    protected_exact = bool(np.array_equal(written[protected[crop]], existing[protected[crop]]))
    if not (np.array_equal(written, result) and exact_outside and protected_exact):
        temporary.unlink(missing_ok=True)
        raise Refused('PCM16 write-back is not exact outside the allowed envelope')
    if sha256(inputs['existing_final']) != input_hashes['existing_final']:
        temporary.unlink(missing_ok=True)
        raise Refused('existing final changed during remix')
    try:
        os.replace(temporary, output)
    except PermissionError as error:  # a previous output still open for playback
        temporary.unlink(missing_ok=True)
        raise ResourceAbort('output busy: %s' % error)
    receipt.update(state='remixed', output=str(output), output_sha256=sha256(output),
                   output_peak=peak, existing_peak=float(np.abs(existing).max() / 32768.),
                   changed_frames=int(np.count_nonzero(changed)),
                   modified_sample_frames=int(np.count_nonzero(np.any(result != existing, axis=1))),
                   exact_outside_allowed=exact_outside, protected_exact=protected_exact,
                   model={'name': 'htdemucs existing safetensors', 'device': 'cpu', 'threads': threads,
                          'precision': 'fp32', 'shifts': 0, 'overlap': .25, 'normalization': 'mean/std (demucs.api)',
                          'bed': 'sum of non-vocal stems', **{k: v for k, v in model_info.items()}},
                   separation=timings, separation_seconds=sum(t['seconds'] for t in timings),
                   separated_audio_seconds=sum(t['audio_seconds'] for t in timings),
                   network_attempts=list(_NETWORK['attempts']), telemetry=telemetry,
                   seconds=time.monotonic() - started,
                   scope='Mix-only machine construction; no listening/ASR. Owner listening pending.')
    return receipt


class _NullGuard:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
