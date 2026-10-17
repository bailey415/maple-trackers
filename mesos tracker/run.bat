@echo off
cd /d "%~dp0"
title Mesos Tracker
echo Checking libraries ...
python -m pip install --quiet --disable-pip-version-check mss opencv-python numpy pywin32
if errorlevel 1 (
    echo Library install FAILED - see the error above.
    pause
    exit /b 1
)
echo.
python meso_tracker.py
pause
