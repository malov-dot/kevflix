"""Background Windows-startup entry point. Starts only this application's worker."""
import logging
import logging.handlers
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
venv = ROOT / ".venv/Scripts/pythonw.exe"
if os.name == "nt" and venv.is_file() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    subprocess.Popen([str(venv), str(Path(__file__).resolve())], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    raise SystemExit(0)
for name, value in {"PYTHONUTF8":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                    "PYANNOTE_METRICS_ENABLED":"0","HF_HUB_DISABLE_TELEMETRY":"1","DO_NOT_TRACK":"1"}.items():
    os.environ[name] = value
os.environ["PATH"] = str(ROOT / "tools/ffmpeg-9.0.2-full_build-shared/bin") + os.pathsep + str(ROOT / ".venv/Lib/site-packages/torch/lib") + os.pathsep + os.environ.get("PATH","")
logs = ROOT / "state"
logs.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, handlers=[
    logging.handlers.RotatingFileHandler(logs / "worker.log", maxBytes=2*1024*1024, backupCount=3)],
    format="%(asctime)s %(levelname)s %(message)s")
from relay.desktop_updates import apply_pending
try:
    if apply_pending():
        raise SystemExit(0)
except Exception:
    logging.exception("Desktop update deferred; keeping the installed worker")
sys.argv = [sys.argv[0], "--config", str(ROOT / "relay-config.json")]
from relay.server import main
main()
