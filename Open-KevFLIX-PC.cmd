@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  if exist "runtime\python\pythonw.exe" (
    start "" "runtime\python\pythonw.exe" scripts\open_pc.py
    exit /b 0
  )
  echo Run Setup-Worker.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\open_pc.py
if errorlevel 1 pause
