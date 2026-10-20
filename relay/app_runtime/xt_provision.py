"""Install the isolated WSL X-Translator stack without loading any models.

The CLI takes an explicit profile. It never starts a server, modifies Sem, or
selects a new engine on behalf of the settings UI. Dependency and asset receipts
describe installation, not inference or playback acceptance.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import venv

ASSETS = (
    ('Qwen/Qwen3-ASR-1.7B', '7278e1e70fe206f11671096ffdd38061171dd6e5', 'qwen-asr'),
    ('NiuTrans/LMT-60-8B', 'c77b9bf3b01db78174e7be3ec1d4927e3a0454f7', 'lmt'),
    ('mradermacher/LMT-60-8B-GGUF', '5db5f369e614a63c90c4b3100585ef15a0fe53d0', 'lmt-q8'),
    ('XRXRX/X-Voice', '7f24fe778ddf7a47e25d87e5d5153599c1d4d5c2', 'xvoice'),
    ('charactr/vocos-mel-24khz', '0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21', 'vocos'),
)
ESPEAK_COMMIT = 'b0b605c8a80f76c4c19e18033c6780c3cc4afc5b'
CAMPLUS_REVISION = 'v2.0.2'
MT_Q8_FILE = 'LMT-60-8B.Q8_0.gguf'
MT_Q8_SHA256 = 'f7671cd4fca54b20398754d89de00e340cdf422b6ffe6976f63e4ee522150ead'
MT_Q8_BYTES = 8709518528
NLTK_REVISION = '550b6625bcef1f2abff2ff770a5a0d272c9c6b2a'
CMUDICT_SHA256 = 'd07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6'
CMUDICT_BYTES = 896069


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value, *, acl_from=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.install-tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    if acl_from is not None and os.name == 'nt':
        from relay.settings import preserve_private_acl
        preserve_private_acl(acl_from, temporary)
    os.replace(temporary, path)


def linux_path(path):
    path = Path(path).resolve()
    if len(path.drive) != 2 or path.drive[1] != ':':
        raise ValueError('X-Translator installation requires a local drive path')
    return '/mnt/' + path.drive[0].lower() + '/' + path.as_posix()[3:]


def _run(args, *, cwd=None, env=None, timeout=7200):
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True, timeout=timeout)


def _environment(root, name):
    directory = root / 'envs' / name
    if not (directory / 'bin/python').is_file():
        venv.EnvBuilder(with_pip=True).create(directory)
    python = directory / 'bin/python'
    _run([python, '-m', 'pip', 'install', '--disable-pip-version-check',
          'pip==25.3', 'wheel', 'setuptools<81'])
    return python


def _assets_main(request):
    # This entry runs in the small download environment, never imports torch.
    from huggingface_hub import snapshot_download
    model_root = Path(request['model_root'])
    records = []
    for repo, revision, folder in ASSETS:
        target = model_root / folder
        sidecars = None
        patterns = ['*.json', '*.safetensors', '*.txt', '*.model', '*.bin', '*.yaml', '*.jinja', 'README.md', 'LICENSE*']
        if folder == 'lmt':
            # Original tokenizer/config/chat template, no new BF16 weight download.
            # Existing full-precision assets are preserved on disk.
            listing_url = 'https://huggingface.co/api/models/' + repo + '/revision/' + revision + '?blobs=true'
            with urllib.request.urlopen(listing_url, timeout=60) as response:
                listing = json.load(response)
            if listing.get('sha') != revision:
                raise RuntimeError('Original LMT sidecar revision differs from the pinned commit')
            weight_extensions = ('.safetensors', '.bin', '.pt', '.pth', '.gguf')
            sidecars = [entry for entry in listing['siblings']
                        if not entry['rfilename'].lower().endswith(weight_extensions)]
            patterns = [entry['rfilename'] for entry in sidecars]
            if 'chat_template.jinja' not in patterns or 'generation_config.json' not in patterns:
                raise RuntimeError('Pinned original LMT template/config sidecars are incomplete')
        elif folder == 'lmt-q8':
            patterns = [MT_Q8_FILE]
        if folder == 'xvoice':
            patterns = ['README.md', 'config.json', 'XVoice_Base_Stage2/*',
                        'SpeedPredictor/*', 'XVoice_Base_Stage1/vocab.txt']
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), allow_patterns=patterns)
        sidecar_identity = []
        for entry in sidecars or ():
            name = entry['rfilename']
            path = (target / name).resolve()
            if not path.is_relative_to(target.resolve()) or not path.is_file():
                raise RuntimeError('Invalid original LMT sidecar path: ' + name)
            data = path.read_bytes()
            blob = hashlib.sha1(('blob %d\0' % len(data)).encode() + data).hexdigest()
            lfs = entry.get('lfs')
            expected_hash = lfs.get('sha256') if lfs else entry.get('blobId')
            actual_hash = hashlib.sha256(data).hexdigest() if lfs else blob
            if len(data) != entry.get('size') or not expected_hash or actual_hash != expected_hash:
                raise RuntimeError('Original LMT sidecar upstream identity mismatch: ' + name)
            sidecar_identity.append({'path': name, 'bytes': len(data), 'git_blob_sha1': blob,
                'sha256': hashlib.sha256(data).hexdigest(), 'upstream': entry})
        if folder == 'lmt-q8':
            quant = target / MT_Q8_FILE
            if quant.stat().st_size != MT_Q8_BYTES or digest(quant) != MT_Q8_SHA256:
                raise RuntimeError('Pinned Q8 translation model size/hash mismatch')
        record = {'repo': repo, 'revision': revision, 'folder': str(target)}
        if sidecars is not None:
            record['upstream_sidecars'] = sidecar_identity
        records.append(record)
    camp = model_root / 'campplus'
    camp.mkdir(parents=True, exist_ok=True)
    repo = 'iic/speech_campplus_sv_zh-cn_16k-common'
    url = 'https://modelscope.cn/api/v1/models/' + repo + '/repo/files?Revision=' + CAMPLUS_REVISION + '&Recursive=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        listing = json.load(response)
    if not listing.get('Success'):
        raise RuntimeError('Could not resolve pinned CAM++ model files')
    wanted = {'campplus_cn_common.bin', 'configuration.json', 'config.yaml', 'README.md'}
    found = set()
    from urllib.parse import urlencode
    for entry in listing['Data']['Files']:
        name = entry['Path']
        if name not in wanted:
            continue
        target = camp / name
        expected = entry.get('Sha256')
        if not expected or len(expected) != 64:
            raise RuntimeError('CAM++ upstream file hash missing: ' + name)
        if not target.is_file() or digest(target) != expected:
            query = urlencode({'Revision': CAMPLUS_REVISION, 'FilePath': name})
            with urllib.request.urlopen('https://modelscope.cn/api/v1/models/' + repo + '/repo?' + query,
                                        timeout=300) as response, target.with_suffix(target.suffix + '.part').open('wb') as out:
                while block := response.read(4 << 20):
                    out.write(block)
            temporary = target.with_suffix(target.suffix + '.part')
            if digest(temporary) != expected:
                raise RuntimeError('CAM++ asset digest mismatch: ' + name)
            os.replace(temporary, target)
        found.add(name)
    if not wanted <= found:
        raise RuntimeError('CAM++ pinned revision is incomplete')
    records.append({'repo': repo, 'revision': CAMPLUS_REVISION, 'folder': str(camp)})
    # g2pk imports the NLTK CMU dictionary even for an English-only X-Voice load.
    # Provision the original corpus here; the offline runtime never downloads it.
    nltk_root = model_root / 'nltk'
    corpus = nltk_root / 'corpora/cmudict.zip'
    corpus.parent.mkdir(parents=True, exist_ok=True)
    if not corpus.is_file() or corpus.stat().st_size != CMUDICT_BYTES or digest(corpus) != CMUDICT_SHA256:
        url = ('https://raw.githubusercontent.com/nltk/nltk_data/' + NLTK_REVISION
               + '/packages/corpora/cmudict.zip')
        temporary = corpus.with_suffix('.zip.part')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
            while block := response.read(1 << 20):
                out.write(block)
        if temporary.stat().st_size != CMUDICT_BYTES or digest(temporary) != CMUDICT_SHA256:
            raise RuntimeError('Pinned NLTK CMU dictionary size/hash mismatch')
        os.replace(temporary, corpus)
    records.append({'repo': 'nltk/nltk_data', 'revision': NLTK_REVISION, 'folder': str(nltk_root)})
    for record in records:
        directory = Path(record['folder'])
        record['files'] = [{'path': str(p.relative_to(directory)), 'bytes': p.stat().st_size,
                            'sha256': digest(p)} for p in sorted(directory.rglob('*'))
                           if p.is_file() and '.cache' not in p.relative_to(directory).parts]
        if record['repo'] == 'mradermacher/LMT-60-8B-GGUF':
            record.update(format='gguf-q8_0', file=MT_Q8_FILE, sha256=MT_Q8_SHA256,
                          size=MT_Q8_BYTES, tokenizer_dir=str(model_root / 'lmt'),
                          config_dir=str(model_root / 'lmt'))
    atomic_json(request['manifest'], {'created': time.time(), 'models': records,
                'model_loading_performed': False, 'xvoice_weight_license': 'CC-BY-NC-4.0'})


def _linux_main(request):
    # Provisioning is the explicit download phase; runtime services set their
    # own offline environment after these assets exist.
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'MODELSCOPE_OFFLINE', 'PIP_REQUIRE_VIRTUALENV'):
        os.environ.pop(key, None)
    os.environ.setdefault('CMAKE_BUILD_PARALLEL_LEVEL', '4')
    os.environ.setdefault('MAX_JOBS', '4')
    root = Path(request.get('runtime_root') or Path.home() / 'kevflix-xt').resolve()
    if root == Path('/') or not root.is_absolute():
        raise ValueError('Choose a dedicated X-Translator runtime directory')
    marker = root / '.kevflix-xt-runtime.json'
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError('Refusing to install over a directory not owned by KevFLIX X-Translator')
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        atomic_json(marker, {'owner': 'kevflix-xtranslator', 'created': time.time()})
    elif json.loads(marker.read_text()).get('owner') != 'kevflix-xtranslator':
        raise ValueError('X-Translator runtime ownership marker is invalid')
    vendor = Path(request['vendor']).resolve()
    for required in ('translator/backend/session.py', 'xvoice/pyproject.toml', 'linux_supervisor.py',
                     'runtime-requirements.txt', 'runtime-constraints.txt'):
        if not (vendor / required).is_file():
            raise RuntimeError('Vendored component is missing: ' + required)
    apt = ['apt-get'] if os.geteuid() == 0 else ['sudo', '-n', 'apt-get']
    _run(apt + ['update'])
    _run(apt + ['install', '-y', 'python3-venv', 'python3-dev', 'build-essential', 'cmake',
                'pkg-config', 'git', 'ffmpeg', 'libsndfile1', 'libsox-dev', 'sox',
                'autoconf', 'automake', 'libtool'])
    downloader = _environment(root, 'download')
    _run([downloader, '-m', 'pip', 'install', 'huggingface_hub==0.36.0'])
    # ASR and MT share compatible vLLM dependencies, but run as separately owned services.
    asr = _environment(root, 'asr')
    _run([asr, '-m', 'pip', 'install', 'qwen-asr[vllm]==0.0.6',
          'fastapi', 'uvicorn', 'python-multipart', 'soundfile'])
    # X-Voice and the speaker-aware controller share a second isolated environment.
    backend = _environment(root, 'backend')
    _run([backend, '-m', 'pip', 'install', 'torch==2.8.0', 'torchaudio==2.8.0',
          '--index-url', 'https://download.pytorch.org/whl/cu128'])
    # Runtime imports the pinned source through PYTHONPATH. Installing the full
    # upstream project metadata would pull in unused training/demo/eval stacks.
    # Copy only the audited dependency inputs; never build inside the app tree.
    build_input = root / 'build-inputs' / str(time.time_ns())
    build_input.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in ('runtime-requirements.txt', 'runtime-constraints.txt'):
        shutil.copy2(vendor / name, build_input / name)
        copied.append({'path': name, 'sha256': digest(build_input / name)})
    atomic_json(build_input / 'source-inputs.json', copied)
    _run([backend, '-m', 'pip', 'install', '-c', build_input / 'runtime-constraints.txt',
          '-r', build_input / 'runtime-requirements.txt'])
    freezes, metadata_checks = {}, {}
    for name, python in (('asr-and-mt', asr), ('backend-and-xvoice', backend), ('download', downloader)):
        freezes[name] = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True).splitlines()
        checked = subprocess.run([str(python), '-m', 'pip', 'check'], capture_output=True, text=True)
        metadata_checks[name] = {'exit_code': checked.returncode, 'output': checked.stdout + checked.stderr,
                                 'scope': 'installed dependency metadata only; no model or app imports'}
        if checked.returncode:
            atomic_json(root / 'dependency-metadata-failure.json', metadata_checks)
            raise RuntimeError('Installed dependency metadata is inconsistent: ' + name)
    # Phonemizer build is confined to this prefix; never run upstream's global cleanup/tests.
    source, build, prefix = root / 'sources/espeak-ng', root / 'build/espeak-ng', root / 'espeak'
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        _run(['git', 'clone', '--no-checkout', 'https://github.com/espeak-ng/espeak-ng.git', source])
    _run(['git', '-C', source, 'fetch', 'origin', ESPEAK_COMMIT])
    _run(['git', '-C', source, 'checkout', '--detach', ESPEAK_COMMIT])
    _run(['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
          '-DCMAKE_INSTALL_PREFIX=' + str(prefix), '-DCMAKE_INSTALL_LIBDIR=lib',
          '-DBUILD_SHARED_LIBS=ON', '-DENABLE_TESTS=OFF'])
    _run(['cmake', '--build', build, '-j4'])
    _run(['cmake', '--install', build])
    manifest = Path(request['manifest'])
    asset_request = {'model_root': str(root / 'models'), 'manifest': str(manifest)}
    encoded = base64.b64encode(json.dumps(asset_request).encode()).decode()
    _run([downloader, Path(__file__).resolve(), '--asset-request', encoded])
    models = root / 'models'
    settings = {
        'xt_python': str(backend), 'xt_tts_python': str(backend),
        'xt_asr_python': str(asr), 'xt_mt_python': str(asr),
        'xt_asr_model': str(models / 'qwen-asr'), 'xt_mt_model': str(models / 'lmt-q8' / MT_Q8_FILE),
        'xt_mt_tokenizer': str(models / 'lmt'), 'xt_mt_format': 'gguf-q8_0',
        'xt_mt_dtype': 'half', 'xt_mt_gguf_overhead_mib': 1536,
        'xt_xvoice_ckpt': str(models / 'xvoice/XVoice_Base_Stage2/model_70000.safetensors'),
        'xt_rate_ckpt': str(models / 'xvoice/SpeedPredictor/model_28000.safetensors'),
        'xt_vocab': str(models / 'xvoice/XVoice_Base_Stage1/vocab.txt'),
        'xt_vocos_dir': str(models / 'vocos'), 'xt_campplus_model': str(models / 'campplus'),
        'xt_nltk_data': str(models / 'nltk'),
        'xt_espeak_library': str(prefix / 'lib/libespeak-ng.so'),
        'xt_espeak_data': str(prefix / 'share/espeak-ng-data'),
    }
    atomic_json(request['receipt'], {'installed': True, 'scope': 'dependency and asset installation only',
                'created': time.time(), 'settings': settings, 'packages': freezes,
                'dependency_metadata_checks': metadata_checks, 'dependency_inputs': copied,
                'espeak_commit': ESPEAK_COMMIT, 'models_manifest_sha256': digest(manifest),
                'model_loading_performed': False, 'playback_tested': False})


def provision(config_file, *, distro='Ubuntu-24.04', runtime_root=None, assets_only=False):
    if distro != 'Ubuntu-24.04':
        raise ValueError('Provision only the Ubuntu-24.04 runtime; other distributions are untouched')
    config_file = Path(config_file).resolve()
    original = config_file.read_bytes()
    config = json.loads(original.decode('utf-8-sig'))
    app = Path(__file__).resolve().parents[2]
    vendor = app / 'relay/vendor/xtranslator'
    evidence = config_file.parent / ('xtranslator-assets-%d' % time.time_ns() if assets_only else 'xtranslator-install')
    evidence.mkdir(parents=True, exist_ok=True)
    manifest, receipt = evidence / 'models.json', evidence / 'runtime.json'
    request = {'vendor': linux_path(vendor), 'runtime_root': runtime_root,
               'manifest': linux_path(manifest), 'receipt': linux_path(receipt)}
    encoded = base64.b64encode(json.dumps(request).encode()).decode()
    flags = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)} if os.name == 'nt' else {}
    with (evidence / 'install.log').open('a', encoding='utf-8') as log:
        if assets_only:
            prior = Path(config.get('xt_runtime_receipt', ''))
            installed = json.loads(prior.read_text(encoding='utf-8-sig'))
            if not installed.get('installed'):
                raise ValueError('Assets-only update requires an installed runtime receipt')
            from pathlib import PurePosixPath
            mt_python = PurePosixPath(installed['settings']['xt_mt_python'])
            existing_root = mt_python.parents[3]
            if (not mt_python.is_absolute() or mt_python.parts[-4:] != ('envs', 'asr', 'bin', 'python')
                    or runtime_root is not None and PurePosixPath(runtime_root) != existing_root):
                raise ValueError('Assets-only update must use the existing isolated runtime')
            asset_request = {'model_root': str(existing_root / 'models'), 'manifest': linux_path(manifest)}
            encoded_assets = base64.b64encode(json.dumps(asset_request).encode()).decode()
            subprocess.run(['wsl.exe', '-d', distro, '--exec', str(existing_root / 'envs/download/bin/python'),
                            linux_path(__file__), '--asset-request', encoded_assets], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=14400, **flags)
            installed['settings'].update(xt_mt_model=str(existing_root / 'models/lmt-q8' / MT_Q8_FILE),
                xt_mt_tokenizer=str(existing_root / 'models/lmt'), xt_mt_format='gguf-q8_0',
                xt_mt_dtype='half', xt_mt_gguf_overhead_mib=1536,
                xt_nltk_data=str(existing_root / 'models/nltk'))
            installed.update(created=time.time(), scope='model asset update only; existing dependencies retained',
                prior_runtime_receipt=str(prior), models_manifest_sha256=digest(manifest),
                model_loading_performed=False, playback_tested=False)
            atomic_json(receipt, installed)
        else:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'websocket-client==1.9.0'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600, **flags)
            subprocess.run(['wsl.exe', '-d', distro, '--exec', 'python3', linux_path(__file__),
                            '--linux-request', encoded], stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=14400, **flags)
    installed = json.loads(receipt.read_text(encoding='utf-8'))
    if config_file.read_bytes() != original:
        raise RuntimeError('Profile changed during installation; installed paths preserved in runtime.json')
    backup = evidence / ('profile-before-%d.json' % time.time_ns())
    backup.write_bytes(original)
    if os.name == 'nt':
        from relay.settings import preserve_private_acl
        preserve_private_acl(config_file, backup)
    config.update(installed['settings'], xt_vendor_dir=str(vendor), xt_wsl_distro=distro,
                  xt_model_manifest=str(manifest), xt_runtime_receipt=str(receipt))
    atomic_json(config_file, config, acl_from=config_file)
    return installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--distro', default='Ubuntu-24.04')
    parser.add_argument('--runtime-root')
    parser.add_argument('--assets-only', action='store_true', help='Update model assets using an existing installed runtime')
    parser.add_argument('--linux-request')
    parser.add_argument('--asset-request')
    args = parser.parse_args()
    if args.asset_request:
        _assets_main(json.loads(base64.b64decode(args.asset_request)))
    elif args.linux_request:
        _linux_main(json.loads(base64.b64decode(args.linux_request)))
    elif args.config:
        provision(args.config, distro=args.distro, runtime_root=args.runtime_root, assets_only=args.assets_only)
    else:
        parser.error('--config is required; no implicit live profile is modified')


if __name__ == '__main__':
    main()
