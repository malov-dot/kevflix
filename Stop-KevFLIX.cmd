@echo off
rem Stop the running KevFLIX app: only the registered, validated KevFLIX instance is asked to stop.
cd /d "%~dp0"
set "PY=runtime\python\python.exe"
if not exist "%PY%" set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo KevFLIX is not installed in this folder.
  exit /b 1
)
"%PY%" -I -B scripts\stop_worker.py
exit /b %errorlevel%
