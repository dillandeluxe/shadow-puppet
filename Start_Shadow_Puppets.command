#!/usr/bin/env bash
# ==============================================================================
# One-Click Automatic Launcher for Hand-Gesture Shadow Puppets AI
# Double-click this file in macOS Finder or run from terminal: ./Start_Shadow_Puppets.command
# ==============================================================================

# Resolve the absolute directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "    Hand-Gesture Shadow Puppets AI Launcher       "
echo "=================================================="

# Check Python environment
if [ ! -f ".venv/bin/python" ]; then
    echo "[1/2] Setting up virtual environment (.venv)..."
    python3 -m venv .venv
    echo "[2/2] Installing dependencies from requirements.txt..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
else
    echo "✓ Python virtual environment detected."
fi

echo "✓ Launching application..."
echo "  - Press [H] in app for Gesture Cheat Sheet"
echo "  - Press [S] to toggle Shadow Theater Mode"
echo "  - Press [Q] or [ESC] to exit"
echo "=================================================="

.venv/bin/python shadow_puppets.py "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Application exited with code $EXIT_CODE."
    echo "Press [Enter] to close this window..."
    read -r
fi
