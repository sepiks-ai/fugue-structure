@echo off
rem Double-click to open the Fugue Map on this computer.
rem Starts a small web server in this folder and opens your browser.
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY (python -c "import sys" >nul 2>nul && set "PY=python")
if not defined PY if exist "C:\msys64\ucrt64\bin\python.exe" set "PY=C:\msys64\ucrt64\bin\python.exe"
if not defined PY (
  echo Python was not found. Install it from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation.
  pause
  exit /b 1
)

echo Fugue Map running at http://localhost:8000/web/
echo Close this window to stop it.
start "" /b cmd /c "ping -n 3 127.0.0.1 >nul && start "" http://localhost:8000/web/"
"%PY%" -m http.server 8000
