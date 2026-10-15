@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Setup-Worker.ps1 during an approved setup window first.
  exit /b 1
)
if not exist "relay-config.json" (
  echo Create and edit relay-config.json as described in README.md.
  exit /b 1
)
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYANNOTE_METRICS_ENABLED=0
set HF_HUB_DISABLE_TELEMETRY=1
set DO_NOT_TRACK=1
set "PATH=%~dp0tools\ffmpeg-9.0.2-full_build-shared\bin;%~dp0.venv\Lib\site-packages\torch\lib;%PATH%"
".venv\Scripts\python.exe" -m relay.server --config relay-config.json
