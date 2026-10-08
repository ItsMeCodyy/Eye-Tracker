#!/bin/sh
set -e
cd "$(dirname "$0")"

# iCloud flags synced files as hidden, so Qt sees an empty plugin folder; .nosync opts the venv out.
if [ -d ".venv" ] && [ ! -L ".venv" ]; then
    mv .venv .venv.nosync
    ln -s .venv.nosync .venv
fi

if [ ! -x ".venv/bin/python" ]; then
    python3 -m venv .venv.nosync
    ln -sfn .venv.nosync .venv
fi

chflags -R nohidden .venv.nosync

if ! .venv/bin/python -c 'import cv2, mediapipe as mp, numpy, PySide6, AppKit; assert hasattr(mp, "solutions") and PySide6.__version__ == "6.8.3"' >/dev/null 2>&1; then
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/python eye_tracker.py