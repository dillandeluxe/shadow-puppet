@echo off
setlocal
cd /d "%~dp0"
title Hand-Gesture Shadow Puppets AI

echo ==================================================
echo     Hand-Gesture Shadow Puppets AI Launcher
echo ==================================================

if exist .venv\Scripts\python.exe (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    echo [OK] Python virtual environment detected.
) else (
    set "PYTHON_CMD=python"
    echo [*] Checking Python installation...
)

echo [*] Launching application...
echo   - Press [H] in app for Gesture Cheat Sheet
echo   - Press [S] to toggle Shadow Theater Mode
echo   - Press [L] to toggle Landmark Mesh
echo   - Press [C] to flip camera
echo   - Press [Q] or [ESC] to exit
echo ==================================================

"%PYTHON_CMD%" shadow_puppets.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
