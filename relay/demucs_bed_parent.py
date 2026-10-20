"""PC-server side of the opt-in Demucs mix bed (actor_beta.demucs_mix_bed).

Stdlib only: never imports demucs_bed, torch or the resource guard, never
exits the process, never installs audit hooks or changes priority. It derives
the recipe and canonical carried envelopes, starts the owned child with the
configured project interpreter (python -I demucs_bed_child.py request.json),
validates the child receipt and output hash/geometry, and only then moves the
ready pointer to the new file. Owner cancel kills the child only (Cancelled);
child resource aborts become ResourceYield (waiting_resources); refusals and
invalid outputs are fatal ValueErrors. With the flag off nothing changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from pathlib import Path

from .process import Cancelled
from .resource_budget import ResourceYield

VERSION = 'demucs-bed-parent-v1'
HERE = Path(__file__).resolve().parent
CHILD = HERE / 'demucs_bed_child.py'
HELPER = HERE / 'demucs_bed.py'
CHILD_TIMEOUT_SECONDS = 120
LAUNCH_ALLOWANCE_SECONDS = 60
EXIT_REFUSED, EXIT_RESOURCE = 2, 75
RATE = 48000


def options(config):
    data = getattr(config, 'data', config)
    beta = data.get('actor_beta') if hasattr(data, 'get') else None
    if not isinstance(beta, dict) or beta.get('enabled') is not True or beta.get('demucs_mix_bed') is not True:
        return None
    return beta


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(temporary, path)


def _finite(*values):
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def canonical_carry(store, job_id, revision, carried_ids):
    """Proven canonical removal spans for carried IDs, bounded exclusions, and absent IDs."""
    proven, exclusions, absent = {}, {}, []
    if not carried_ids:
        return proven, exclusions, absent
    rows = {row['payload'].get('id'): row['payload'] for row in store.utterances(job_id, -1e18, 1e18)
            if isinstance(row.get('payload'), dict)}
    for cue in carried_ids:
        item = rows.get(cue)
        if item is None:
            absent.append(cue); continue
        span_keys = ('start', 'end', 'replacement_start', 'replacement_end', 'source_start', 'source_end')
        bounded = all(k in item for k in span_keys) and _finite(*(item[k] for k in span_keys))
        problem = None
        if item.get('ledger_version') != 1 or item.get('revision') != revision:
            problem = 'ledger version or revision differs'
        elif not bounded:
            problem = 'canonical spans missing or non-finite'
        elif not Path(str(item.get('audio', ''))).is_file() or sha256(item['audio']) != item.get('sha256'):
            problem = 'canonical take audio missing or hash changed'
        if problem is None:
            proven[cue] = {'interval': [float(item['replacement_start']), float(item['replacement_end'])],
                           'basis': 'store-dialogue-ledger-v1', 'audio': str(item['audio']), 'audio_sha256': item['sha256']}
        elif bounded:
            exclusions[cue] = {'span': [min(float(item[k]) for k in ('start', 'replacement_start', 'source_start')),
                                        max(float(item[k]) for k in ('end', 'replacement_end', 'source_end'))],
                               'reason': problem}
        else:
            absent.append(cue)
    return proven, exclusions, absent


def _wav_geometry(path):
    with wave.open(str(path), 'rb') as handle:
        return handle.getnchannels(), handle.getframerate(), handle.getsampwidth(), handle.getnframes()


def _skip(result, state, reason, **extra):
    result['demucs_mix_bed'] = {'version': VERSION, 'state': state, 'reason': reason,
                                'audio_unchanged': True, **extra}
    return result


def after_ready(config, result, original, directory, ready, owner, *, store, job_id, revision):
    """Worker hook after english.wav exists. Returns result (ready pointer moved only on success)."""
    beta = options(config)
    if beta is None:
        return result
    if result.get('state') != 'candidate' or result.get('mode', 'actor-English-candidate') != 'actor-English-candidate':
        return _skip(result, 'not_applicable', 'window is not a dubbed actor candidate')
    if not result.get('audio') or Path(result['audio']) != Path(ready) or not Path(ready).is_file():
        return _skip(result, 'not_applicable', 'no ready english.wav')
    excluded = set(result.get('beta_original_fallback') or {}) | set(result.get('beta_runtime_fallback') or {})
    dubbed = [u for u in result.get('utterances') or [] if u.get('cue') not in excluded]
    carried_ids = list(result.get('carried_utterances') or [])
    if not dubbed and not carried_ids:
        return _skip(result, 'not_applicable', 'no dubbed replacement envelope (zero-dub window)')
    proven, exclusions, absent = canonical_carry(store, job_id, revision, carried_ids)
    if absent:
        return _skip(result, 'skipped_unproven_carry',
                     'carried utterances without canonical ledger rows; existing final kept', carried_absent=absent)
    python = config['python']
    if not python or not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError('actor_beta.demucs_mix_bed requires an explicit absolute project python')
    model_dir = Path(config['separation_model_dir'])
    pins = beta.get('demucs_model_pins')
    if not model_dir.is_absolute() or not isinstance(pins, dict) or not pins:
        raise ValueError('actor_beta.demucs_mix_bed requires an absolute separation_model_dir and demucs_model_pins')
    if not config['sem_status_file']:
        raise ValueError('actor_beta.demucs_mix_bed requires sem_status_file for the resource guard')
    directory = Path(directory)
    timeout = float(beta.get('demucs_timeout_seconds', CHILD_TIMEOUT_SECONDS))
    threads = int(beta.get('demucs_threads', 4))
    paths = {name: directory / ('demucs-bed-' + name) for name in
             ('request.json', 'report.json', 'receipt.json', 'abort.json')}
    output = directory / ('english-demucs-bed-r%d.wav' % int(revision))
    for stale in (paths['receipt.json'], paths['abort.json']):
        stale.unlink(missing_ok=True)
    base_sha = result['sha256']
    _write_json(paths['report.json'], result)
    request = {'version': VERSION, 'job': job_id, 'revision': revision,
               'recipe': {'model_dir': str(model_dir), 'model_pins': pins, 'sem_status_file': str(config['sem_status_file']),
                          'threads': threads, 'timeout_seconds': timeout,
                          'context_seconds': float(beta.get('demucs_context_seconds', .5))},
               'original': str(original), 'speech': str(directory / 'speech-estimate.wav'),
               'existing_final': str(ready), 'report': str(paths['report.json']), 'output': str(output),
               'receipt': str(paths['receipt.json']), 'abort': str(paths['abort.json']),
               'carried': proven, 'carried_exclusions': exclusions}
    _write_json(paths['request.json'], request)
    sources = {str(p): sha256(p) for p in (Path(__file__).resolve(), CHILD, HELPER)}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    try:
        owner.run([python, '-I', CHILD, paths['request.json']], timeout=timeout + LAUNCH_ALLOWANCE_SECONDS,
                  env=env, cwd=directory)
    except Cancelled:
        raise
    except TimeoutError as error:
        raise ResourceYield('Demucs bed child exceeded its bound: %s' % error)
    except RuntimeError as error:
        if paths['abort.json'].is_file():
            abort = json.loads(paths['abort.json'].read_text(encoding='utf-8'))
            if abort.get('kind') == 'ResourceYield':
                raise ResourceYield('Demucs bed yielded: %s' % abort.get('error'))
        if paths['receipt.json'].is_file():
            refused = json.loads(paths['receipt.json'].read_text(encoding='utf-8'))
            if refused.get('state') == 'refused':
                raise ValueError('Demucs bed refused: %s' % refused.get('reason'))
        raise
    receipt = json.loads(paths['receipt.json'].read_text(encoding='utf-8'))
    if sha256(ready) != base_sha or (receipt.get('input_sha256') or {}).get('existing_final') != base_sha:
        raise ValueError('Demucs bed receipt is not bound to the ready english.wav')
    if receipt.get('helper_sha256') != sources[str(HELPER)]:
        raise ValueError('Demucs bed helper changed during the window')
    summary = {'version': VERSION, 'state': receipt.get('state'), 'receipt': str(paths['receipt.json']),
               'receipt_sha256': sha256(paths['receipt.json']), 'request_sha256': sha256(paths['request.json']),
               'sources': sources, 'base_audio': str(ready), 'base_sha256': base_sha,
               'partial_bed_coverage': receipt.get('partial_bed_coverage') or [],
               'carried_proven': sorted(proven), 'carried_unproven': sorted(exclusions),
               'masks': receipt.get('masks'), 'human_listening_acceptance': False}
    if receipt.get('state') == 'unchanged':
        result['demucs_mix_bed'] = {**summary, 'audio_unchanged': True,
                                    'reason': 'no allowed frames after protection/carry masks'}
        return result
    if receipt.get('state') != 'remixed' or Path(receipt.get('output', '')) != output or not output.is_file():
        raise ValueError('Demucs bed child returned no published output')
    digest = sha256(output)
    frames = round(float(result['render_end']) * RATE) - round(float(result['render_start']) * RATE)
    if digest != receipt.get('output_sha256') or _wav_geometry(output) != (2, RATE, 2, frames):
        raise ValueError('Demucs bed output hash or geometry is invalid')
    summary.update(audio_unchanged=False, output=str(output), output_sha256=digest,
                   changed_frames=receipt.get('changed_frames'), output_peak=receipt.get('output_peak'),
                   separation_seconds=receipt.get('separation_seconds'), model=receipt.get('model'))
    result['demucs_mix_bed'] = summary
    result.update(audio=str(output), sha256=digest)
    return result
