@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Worker.ps1 first.
  exit /b 1
)
if not exist "relay-config.json" (
  echo Create and configure relay-config.json first.
  exit /b 1
)
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYANNOTE_METRICS_ENABLED=0
set HF_HUB_DISABLE_TELEMETRY=1
set DO_NOT_TRACK=1
".venv\Scripts\python.exe" -m relay.server --config relay-config.json --setup-only
