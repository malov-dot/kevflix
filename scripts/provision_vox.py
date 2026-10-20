"""Provision the Vox actor engine for this PC (explicit step; Setup-Worker.ps1 -VoxModelSource/-VoxRuntimeSource or
KevFLIX Setup /vox-model /vox-runtime). See relay/app_runtime/vox_provision.py.

  python scripts/provision_vox.py --model-source <folder or voxcpm2-actor-q4_k.gguf> --runtime-source <folder>
                                  [--reference-in-place]
  python scripts/provision_vox.py                     # keep/verify an already configured Vox install

Every sha256 is verified. The files are copied into the user profile (or referenced in place), then
actor_runtime_manifest and the Vox pins are written into the user's settings without changing anything else.
A missing or mismatching artifact is an explicit error (exit 1); there is no piper fallback.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relay.app_runtime import vox_provision  # noqa: E402
from relay.app_runtime.paths import Profile  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-source')
    parser.add_argument('--runtime-source')
    parser.add_argument('--reference-in-place', action='store_true')
    args = parser.parse_args(argv)
    try:
        from relay.settings import preserve_private_acl
    except Exception:  # noqa: BLE001
        preserve_private_acl = None
    try:
        record = vox_provision.provision(Profile(install_root=ROOT), model_source=args.model_source,
                                         runtime_source=args.runtime_source,
                                         reference_in_place=args.reference_in_place, acl_from=preserve_private_acl,
                                         install_root=ROOT)
    except vox_provision.ProvisionError as error:
        print(json.dumps({'ok': False, 'error': str(error)}, indent=1))
        return 1
    print(json.dumps({'ok': True, **{k: record.get(k) for k in ('mode', 'manifest', 'model', 'runtime')}}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
