"""Stdlib-only cache/trust contract for the one-pass Demucs bundle (shared by server parent and model child).

The key binds everything that changes model input or output: source file + decoded PCM, source
layout, processing transform, window frames, explicit scope mode with canonical frames and
context frames, model pins, params including actual threads/precision, rates, and the code of
first_pass.py / first_pass_child.py / this module. The absolute time origin is not keyed (content
key) but every bundle is validated against the caller's EXPECTED request, origin included, so a
stale-but-valid bundle for another origin/scope/context/mode/pins/source is refused.
"""
import hashlib
import json
from pathlib import Path

KEY_VERSION = 'demucs-first-pass-key-v1'
RATE, MODEL_RATE = 48000, 44100
LAYOUT_CHANNELS = {'mono': 1, 'stereo': 2, '5.1': 6, '5.1(side)': 6}
TRANSFORMS = {'mono': 'ffmpeg-ac2:mono*sqrt(1/2)->L,R', 'stereo': 'ffmpeg-ac2:identity',
              '5.1': 'ffmpeg-ac2:itu(L+.7071C+.7071BL)/(1+sqrt2),no-lfe',
              '5.1(side)': 'ffmpeg-ac2:itu(L+.7071C+.7071SL)/(1+sqrt2),no-lfe'}
# The accepted demucs_bed settings (release 0435). 'threads' is replaced by the actual value.
PARAMS = {'model': 'htdemucs', 'precision': 'fp32', 'device': 'cpu', 'threads': 4, 'shifts': 0, 'overlap': .25,
          'split': True, 'normalization': 'demucs.api mean/std (unbiased) of channel-mean, stems*std+mean',
          'resample': 'scipy.signal.resample_poly 147/160 then 160/147, cropped to exact frames',
          'background': 'sum of non-vocal stems'}
# All code on the producer path: adapter, child, this contract, the server parent and the reused 0435 guard helper.
CODE_FILES = ('first_pass.py', 'first_pass_child.py', 'first_pass_key.py', 'first_pass_parent.py', 'demucs_bed.py',
              'first_pass_resident.py')
# movie-trial-v1 overlay (W10): the Demucs device is a KEYED param ('cpu' default = the accepted value, byte-identical
# params for cpu callers); 'cuda' bundles are produced only by the resident trial worker (first_pass_resident.py) and
# are never reusable by a cpu request or vice versa.
DEVICES = ('cpu', 'cuda')
LIBRARIES = ('torch', 'demucs', 'numpy', 'scipy', 'soundfile')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def code_hashes(directory):
    return {name: sha256_file(Path(directory) / name) for name in CODE_FILES}


def library_versions():
    """Installed versions read from package metadata (no torch import), so the stdlib parent can check them."""
    from importlib import metadata
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def canonical_scope(scope, frames, rate=RATE):
    """Explicit mode. 'full': one whole-window input (no scope, or scope with a reason to separate everything).
    'scoped': canonical sorted window-relative core frames (possibly EMPTY = no inference) + context frames."""
    if scope is None or scope.get('reason'):
        return {'mode': 'full', 'frames': [[0, int(frames)]], 'context_frames': 0}
    cores = sorted([int(a), int(b)] for a, b in scope.get('frames') or [])
    if any(not 0 <= a < b <= frames for a, b in cores):
        raise ValueError('scope frames outside the window')
    context = float(scope['context_seconds'])
    if not context >= 0:
        raise ValueError('scope context must be non-negative')
    return {'mode': 'scoped', 'frames': cores, 'context_frames': int(round(context * rate))}


def key_inputs(*, source_sha256, original_pcm_sha256, source_layout, frames, model_pins, threads, code, scope, libraries,
               device='cpu'):
    if source_layout not in TRANSFORMS:
        raise ValueError('unknown source layout %r' % source_layout)
    if device not in DEVICES:
        raise ValueError('unknown Demucs device %r' % (device,))
    return {'version': KEY_VERSION, 'source_sha256': source_sha256, 'original_pcm_sha256': original_pcm_sha256,
            'source_layout': source_layout, 'frames': int(frames), 'rates': [RATE, MODEL_RATE],
            'transform': TRANSFORMS[source_layout], 'model_pins': dict(sorted(dict(model_pins).items())),
            'params': {**PARAMS, 'threads': int(threads), 'device': device}, 'code': dict(sorted(dict(code).items())), 'scope': scope,
            'libraries': dict(sorted(dict(libraries).items()))}


def cache_key(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


EXPECTED_FIELDS = ('source_sha256', 'source_layout', 'frames', 'transform', 'model_pins', 'params', 'code', 'scope', 'rates',
                   'libraries', 'version')


def expected_request(*, original_source_wav, source_layout, frames, model_pins, threads, code_directory, scope, origin,
                     device='cpu'):
    """What the CALLER expects, computed without decoding audio (PCM sha is child-attested from the bound file)."""
    inputs = key_inputs(source_sha256=sha256_file(original_source_wav), original_pcm_sha256=None,
                        source_layout=source_layout, frames=frames, model_pins=model_pins, threads=threads,
                        code=code_hashes(code_directory), scope=canonical_scope(scope, frames),
                        libraries=library_versions(), device=device)
    return {'inputs': inputs, 'origin': float(origin), 'original_source_wav': str(original_source_wav)}


def bundle_problems(bundle, expected):
    """Every mismatch between a bundle and the caller's expected request; empty list means trusted."""
    problems = []
    inputs = bundle.get('key_inputs') or {}
    for field in EXPECTED_FIELDS:
        if inputs.get(field) != expected['inputs'][field]:
            problems.append('key input %s differs' % field)
    if not isinstance(inputs.get('original_pcm_sha256'), str):
        problems.append('original PCM identity missing')
    if (bundle.get('model') or {}).get('actual_threads') != (inputs.get('params') or {}).get('threads'):
        problems.append('actual model threads differ from the keyed threads')
    if bundle.get('key') != cache_key(inputs):
        problems.append('key does not match its recorded inputs')
    if bundle.get('origin') != expected['origin']:
        problems.append('origin differs')
    source = bundle.get('original_source') or {}
    if source.get('wav') != expected['original_source_wav'] or source.get('sha256') != expected['inputs']['source_sha256']:
        problems.append('original source lineage differs')
    return problems
