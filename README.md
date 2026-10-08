# Gaze Dot for macOS

A small webcam-based gaze estimator that draws a soft, click-through blob directly over your display, above every app and Space, including full-screen video. It is a calibrated approximation, not a substitute for dedicated eye-tracking hardware.

## Run

Double-click **Run Gaze Dot.command** in Finder. On its first run, it creates a local Python environment and installs the dependencies.

The environment lives in `.venv.nosync` (with `.venv` pointing to it) because iCloud-synced folders such as Documents mark the files as hidden, and Qt then cannot find its macOS plugin.

To run it from Terminal instead, from this folder:

```sh
python3 -m venv .venv.nosync
ln -s .venv.nosync .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python eye_tracker.py
```

Choose **Scan, calibrate and start**. The control window minimizes and the screen guides you through four steps, taking about a minute with no key presses:

1. **Eye scan.** Look at the centre dot and hold still. It measures your eyelid opening, iris size, distance, lighting and head position, and flags problems such as low light or a tilted head.
2. **Calibration.** Look at 13 dots until each moves on.
3. **Head motion.** Keep looking at one dot while slowly moving your head left, right, up and down. This teaches the model to cancel head movement, which is the biggest cause of drift, so you can move more naturally afterwards.
4. **Verification.** Look at 5 more dots the model has not seen. The error it reports is measured on these, so it reflects real tracking rather than the calibration fit.

The blob then follows your gaze, and a report stays on screen for a few seconds with the verified error in pixels and degrees, the model it chose, and the scan results. The model is chosen automatically: it compares combinations of eye features (iris position, eyelid opening, the face model's eye-gaze signals) and head-pose features by how well each predicts dots it was not trained on. Blinks are ignored, and the blob hides when your face is lost. Click the app in the Dock to bring the controls back, then **Stop tracking** turns off the camera and removes the blob.

For best accuracy, face the camera in even light (no bright window behind you), and recalibrate if you change seats. The first launch downloads MediaPipe's face model (about 4 MB) into `models.nosync` and checks it against a pinned hash.

The first camera use may trigger a macOS privacy prompt. If access was denied, enable camera access for the Python or Terminal app under **System Settings > Privacy & Security > Camera**, then restart the tracker.

The blob appears on the primary display and is visible to screen recordings. Webcam gaze is an estimate: expect errors of a few tenths of a degree up to a couple of degrees depending on your face, lighting and camera, which is why the blob is deliberately large. Large head movements beyond what you did in step 3 still reduce accuracy. This prototype processes camera frames locally and does not save video.