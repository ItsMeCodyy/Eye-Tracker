#!/bin/sh
set -e
cd "$(dirname "$0")"

# .nosync keeps iCloud from hiding or evicting Qt's native plugins.
if [ -d ".venv" ] && [ ! -L ".venv" ] && [ ! -d ".venv.nosync" ]; then
    mv .venv .venv.nosync
fi

if [ ! -x ".venv.nosync/bin/python" ]; then
    python3 -m venv .venv.nosync
fi
if [ ! -e ".venv" ]; then
    ln -s .venv.nosync .venv
fi
chflags -R nohidden .venv.nosync

requirements_hash=$(shasum -a 256 requirements.txt | cut -d ' ' -f 1)
installed_hash=$(cat .venv.nosync/.requirements-hash 2>/dev/null || true)
if [ "$requirements_hash" != "$installed_hash" ] || ! .venv.nosync/bin/python -c 'import cv2, mediapipe, numpy, PySide6, AppKit; assert PySide6.__version__ == "6.8.3"' >/dev/null 2>&1; then
    .venv.nosync/bin/python -m pip install --upgrade pip
    .venv.nosync/bin/python -m pip install -r requirements.txt
    printf '%s\n' "$requirements_hash" > .venv.nosync/.requirements-hash
    chflags -R nohidden .venv.nosync
fi

exec .venv.nosync/bin/python eye_tracker.py
