@echo off
setlocal
cd /d "%~dp0"
title Eye Tracker - Gaze Dot

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Install 64-bit Python 3.11 from https://www.python.org/downloads/
        echo Enable the Python launcher during installation, then run this file again.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import hashlib,pathlib,sys; p=pathlib.Path('.venv/.requirements-hash'); h=hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(); sys.exit(0 if p.exists() and p.read_text().strip()==h else 1)" >nul 2>&1
if errorlevel 1 goto install
".venv\Scripts\python.exe" -c "import cv2,mediapipe,numpy,PySide6; assert PySide6.__version__ == '6.8.3'" >nul 2>&1
if errorlevel 1 goto install
goto run

:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import hashlib,pathlib; pathlib.Path('.venv/.requirements-hash').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())"
if errorlevel 1 goto failed

:run
".venv\Scripts\python.exe" eye_tracker.py
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Eye Tracker could not start. Read the error above before closing this window.
pause
exit /b 1
