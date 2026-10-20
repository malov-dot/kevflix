"""Windows CUDA dependency search for owned model workers.

Calling this prepares DLL discovery only. It imports no model/tensor library and
does not allocate a CUDA context. The caller must already be in its owned GPU
worker; the HTTP server does not call this helper.
"""
from __future__ import annotations

import os
from pathlib import Path
import sysconfig
import threading

_lock = threading.Lock()
_handles = []
_registered = set()
_COMPONENTS = ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc", "cufft", "curand",
               "cusolver", "cusparse", "nvjitlink")


def configure_cuda_dll_search():
    """Expose DLLs installed in this interpreter's environment, keeping handles.

    Torch Windows wheels carry their CUDA dependencies in torch/lib; separate
    NVIDIA wheels use nvidia/<component>/bin. Neither system configuration nor
    another process's environment is modified. Missing directories are ignored;
    the model backend reports any required dependency failure explicitly.
    """
    if os.name != "nt":
        return []
    with _lock:
        candidates = []
        for name in ("purelib", "platlib"):
            base = sysconfig.get_path(name)
            if not base:
                continue
            site = Path(base)
            candidates.append(site / "torch" / "lib")
            candidates.extend(site / "nvidia" / component / "bin" for component in _COMPONENTS)
        for directory in candidates:
            if not directory.is_dir():
                continue
            full = str(directory.resolve())
            identity = os.path.normcase(full)
            if identity in _registered:
                continue
            # Handles must stay alive for the worker lifetime. A failed DLL
            # registration propagates to the backend's configuration boundary.
            handle = os.add_dll_directory(full)
            _handles.append(handle)
            _registered.add(identity)
            current = os.environ.get("PATH", "")
            entries = {os.path.normcase(p) for p in current.split(os.pathsep) if p}
            if identity not in entries:
                os.environ["PATH"] = full + (os.pathsep + current if current else "")
        return sorted(_registered)

