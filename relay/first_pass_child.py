"""PROTOTYPE owned model child: python -I <relay>/first_pass_child.py request.json (not authorized to run yet).

Same isolation contract as the accepted demucs_bed_child (release 0435): only this process
imports torch/Demucs, blocks the network, lowers its own priority and may hard-exit from the
reused relay.demucs_bed.Guard. Exit 0 bundle written, 2 refused, 75 resource yield.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
HERE = Path(__file__).resolve().parent
EXIT_REFUSED, EXIT_RESOURCE = 2, 75


def main(argv):
    request = json.loads(Path(argv[1]).read_text(encoding='utf-8'))
    if HERE.name != 'relay':
        return EXIT_REFUSED
    if request.get('admission') == 'movie-lease' or request.get('device', 'cpu') != 'cpu':
        # W10: movie-lease admission is served only by the resident worker inside the lease Job (first_pass_resident.py);
        # this per-window child keeps the original Sem-status guard and CPU placement.
        return EXIT_REFUSED
    sys.path.insert(0, str(HERE.parent))
    from relay import demucs_bed, first_pass  # demucs_bed only for its accepted Guard/write_json/network block
    abort = Path(request['abort'])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(int(request['threads']))
    started = time.monotonic()
    try:
        import numpy as np
        import soundfile as sf
        original, rate = sf.read(request['original_source_wav'], dtype='float32', always_2d=True)
        if rate != first_pass.RATE:
            raise first_pass.Refused('original source WAV is not 48 kHz')
        demucs_bed._block_network()
        with demucs_bed.Guard(request['sem_status_file'], abort, started, float(request['timeout_seconds'])) as guard:
            infer = first_pass.DemucsInfer(request['model_dir'], request['model_pins'], request['threads']).load()
            bundle = first_pass.write_bundle(request['folder'], original_source_wav=request['original_source_wav'],
                                             source_layout_evidence=request['source_layout_evidence'], original=original,
                                             origin=request['origin'], scope=request.get('scope'), infer=infer,
                                             sources=infer.sources, model_pins=request['model_pins'],
                                             threads=request['threads'], model_info=infer.info, code_directory=HERE)
            bundle['telemetry'] = dict(guard.telemetry)
        bundle['child_seconds'] = time.monotonic() - started
        demucs_bed.write_json(request['receipt'], bundle)
        return 0
    except (demucs_bed.ResourceAbort, MemoryError) as error:
        demucs_bed.write_json(abort, {'kind': 'ResourceYield', 'error': '%s: %s' % (type(error).__name__, error)})
        return EXIT_RESOURCE
    except (first_pass.Refused, demucs_bed.Refused) as error:
        demucs_bed.write_json(request['receipt'], {'version': first_pass.VERSION, 'state': 'refused', 'reason': str(error)})
        return EXIT_REFUSED


if __name__ == '__main__':
    sys.exit(main(sys.argv))
