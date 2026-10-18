@echo off
cd /d "%~dp0"
title EXP Tracker
where python >nul 2>nul
if errorlevel 1 goto nopython
python -c "import sys" >nul 2>nul
if errorlevel 1 goto nopython
echo Checking libraries ...
python -m pip install --quiet --disable-pip-version-check mss opencv-python numpy pywin32
if errorlevel 1 (
    echo Library install FAILED - see the error above.
    pause
    exit /b 1
)
echo.
if exist "..\maple tracker\maple_tracker.py" (
    echo Opening the combined Maple Tracker - EXP and mesos on one page ...
    python "..\maple tracker\maple_tracker.py"
) else (
    python exp_tracker.py
)
pause
exit /b 0

:nopython
echo.
echo Python is not installed on this PC - that's fine!
echo You don't need it: download the ready-to-run version instead.
echo Opening the download now: unzip it and double-click the .exe inside.
echo.
echo   https://github.com/bailey415/maple-trackers/releases/latest/download/MapleTrackers.zip
echo.
start "" "https://github.com/bailey415/maple-trackers/releases/latest/download/MapleTrackers.zip"
pause
