@echo off
rem Start KevFLIX in setup-only mode (settings, Plex linking and TV pairing; no media or model work).
cd /d "%~dp0"
set "PY=runtime\python\python.exe"
if not exist "%PY%" set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo KevFLIX is not installed in this folder. Run KevFLIX Setup.
  exit /b 1
)
"%PY%" -I -B scripts\run_worker.py --setup-only
exit /b %errorlevel%
