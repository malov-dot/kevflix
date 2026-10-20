from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import re


class Cancelled(Exception):
    pass


class ProcessOwner:
    def __init__(self):
        self.cancelled = threading.Event()
        self._children = set()
        self._lock = threading.Lock()

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled("Operation cancelled")

    @staticmethod
    def terminate(child):
        if child.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=20, check=False)
        else:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)

    def cancel(self):
        self.cancelled.set()
        with self._lock:
            children = list(self._children)
        for child in children:
            self.terminate(child)

    def register(self,child):
        """Register an interactive owned worker for the same cancellation tree."""
        with self._lock:self._children.add(child)
        if self.cancelled.is_set():
            self.terminate(child)
            self.check()

    def unregister(self,child):
        with self._lock:self._children.discard(child)

    def run(self, argv, *, timeout=1800, input_data=None, env=None, cwd=None):
        self.check()
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        child = subprocess.Popen([str(x) for x in argv], stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd, **kwargs)
        with self._lock:
            self._children.add(child)
        started = time.monotonic()
        first = True
        try:
            while True:
                self.check()
                try:
                    out, err = child.communicate(input=input_data if first else None, timeout=.5)
                    break
                except subprocess.TimeoutExpired:
                    first = False
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("Owned process exceeded its allowed duration")
            self.check()
            if child.returncode:
                detail = err.decode("utf-8", "replace")[-4000:]
                detail = re.sub(r"(?i)([?&](?:token|x-plex-token|signature|api_key)=)[^&\s]+",
                                r"\1[redacted]", detail)
                raise RuntimeError(f"{os.path.basename(str(argv[0]))} exited {child.returncode}: "
                                   + detail)
            return out
        except BaseException:
            self.terminate(child)
            raise
        finally:
            with self._lock:
                self._children.discard(child)


def choose_device(config):
    """Never unload Sem models. GPU use is conditional on explicit configuration."""
    if config["device"] == "cpu":
        return "cpu"
    try:
        from pathlib import Path
        p = Path(config["sem_status_file"])
        if not p.is_file() or time.time() - p.stat().st_mtime > 20:
            return "cpu"
        state = json.loads(p.read_text(encoding="utf-8-sig"))
        voice = state.get("voice", {})
        if state.get("busy") or state.get("codingBusy") or voice.get("active") or voice.get("closing"):
            return "cpu"
        resources = state.get("resources", {})
        if resources.get("activeJobKind") or resources.get("preparing"):
            return "cpu"
        flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, timeout=5, check=True, **flags)
        free = int(result.stdout.decode().strip().splitlines()[0])
        return "cuda" if free >= int(config["gpu_min_free_mib"]) else "cpu"
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return "cpu"
