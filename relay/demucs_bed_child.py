"""Owned Demucs bed child: python -I <relay>/demucs_bed_child.py <request.json>.

Only this process loads torch/Demucs, installs the network audit hook, lowers
its own priority and may hard-exit from its resource guard. Exit codes:
0 receipt written (remixed or unchanged), 2 refused (fatal for the window),
75 resource yield (retry later), anything else an unexpected failure.
"""
import json
import os
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
HERE = Path(__file__).resolve().parent
EXIT_REFUSED, EXIT_RESOURCE = 2, 75


def main(argv):
    if len(argv) != 2:
        print('usage: demucs_bed_child.py request.json', file=sys.stderr)
        return EXIT_REFUSED
    request = json.loads(Path(argv[1]).read_text(encoding='utf-8'))
    receipt_path, abort_path = Path(request['receipt']), Path(request['abort'])
    # The installed runtime's own relay package; -I leaves the script directory off sys.path.
    if HERE.name != 'relay':
        print('demucs_bed_child.py must be installed inside a relay package', file=sys.stderr)
        return EXIT_REFUSED
    sys.path.insert(0, str(HERE.parent))
    from relay import demucs_bed
    if Path(demucs_bed.__file__).resolve().parent != HERE:
        print('relay package resolved outside this runtime', file=sys.stderr)
        return EXIT_REFUSED
    recipe = dict(request['recipe'], abort_path=str(abort_path))
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(int(recipe.get('threads', 4)))  # torch is imported later, in remix_window
    report = json.loads(Path(request['report']).read_text(encoding='utf-8'))
    try:
        receipt = demucs_bed.remix_window(recipe, request['original'], request['speech'], request['existing_final'],
                                          report, request['output'], None, carried=request.get('carried'),
                                          carried_exclusions=request.get('carried_exclusions'))
    except (demucs_bed.ResourceAbort, MemoryError) as error:
        demucs_bed.write_json(abort_path, {'kind': 'ResourceYield', 'error': '%s: %s' % (type(error).__name__, error)})
        return EXIT_RESOURCE
    except RuntimeError as error:
        if 'DefaultCPUAllocator' not in str(error) and 'not enough memory' not in str(error):
            raise
        demucs_bed.write_json(abort_path, {'kind': 'ResourceYield', 'error': 'torch allocation: %s' % error})
        return EXIT_RESOURCE
    except demucs_bed.Refused as error:
        demucs_bed.write_json(receipt_path, {'version': demucs_bed.VERSION, 'state': 'refused', 'reason': str(error)})
        return EXIT_REFUSED
    demucs_bed.write_json(receipt_path, receipt)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
