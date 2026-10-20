@echo off
rem Start KevFLIX in this window (the normal background start is Windows startup / KevFLIX PC).
rem The guard verifies the installed files, uses your profile settings and runs the app until Stop-KevFLIX.cmd.
cd /d "%~dp0"
set "PY=runtime\python\python.exe"
if not exist "%PY%" set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo KevFLIX is not installed in this folder. Run KevFLIX Setup.
  exit /b 1
)
"%PY%" -I -B scripts\run_worker.py
exit /b %errorlevel%
