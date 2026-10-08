import sys
import time
from collections import deque, namedtuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from face_scan import FaceAnalyzer, ensure_model
from gaze_model import (
    MEDIAN_WINDOW,
    CalibrationData,
    OneEuroFilter,
    analyze_scan,
    evaluate_frames,
    pixels_to_degrees,
    select_model,
)
from screen_overlay import ScreenOverlay, display_width_mm

Step = namedtuple("Step", "kind target settle_ms capture_ms")


class GazeWorker(QThread):
    measurement_ready = Signal(object)
    status_changed = Signal(str)

    def run(self):
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self.status_changed.emit(
                "Camera unavailable. Check macOS camera privacy permissions."
            )
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        analyzer = FaceAnalyzer()
        self.status_changed.emit("Camera on. Center your face in the frame.")
        missing_face_frames = 0
        try:
            while not self.isInterruptionRequested():
                success, frame = camera.read()
                if not success:
                    self.status_changed.emit("Could not read from the camera.")
                    break

                measurement = analyzer.analyze(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(time.monotonic() * 1000)
                )
                if measurement is not None:
                    self.measurement_ready.emit(measurement)
                    missing_face_frames = 0
                else:
                    missing_face_frames += 1
                    if missing_face_frames == 1:
                        self.status_changed.emit("Looking for your face...")
        finally:
            analyzer.close()
            camera.release()


class EyeTracker(QWidget):
    STATIC_DOTS = (
        (0.06, 0.06), (0.5, 0.06), (0.94, 0.06),
        (0.72, 0.28), (0.28, 0.28),
        (0.06, 0.5), (0.5, 0.5), (0.94, 0.5),
        (0.72, 0.72), (0.28, 0.72),
        (0.06, 0.94), (0.5, 0.94), (0.94, 0.94),
    )
    MOTION_DOTS = ((0.5, 0.5), (0.75, 0.3))
    VALIDATION_DOTS = ((0.2, 0.62), (0.4, 0.16), (0.62, 0.84), (0.85, 0.4), (0.5, 0.38))
    TITLES = {
        "scan": "Step 1 of 4 - Scanning your eyes: look at the dot and hold still",
        "static": "Step 2 of 4 - Calibrating: look at the dot",
        "motion": "Step 3 of 4 - Keep looking at the dot and slowly move your head around",
        "validate": "Step 4 of 4 - Checking accuracy: look at the dot",
    }
    MIN_SAMPLES = {"scan": 20, "static": 12, "motion": 40, "validate": 12}
    READY_MS = 3000
    BLINK_LIMIT = 0.5
    VERDICTS = ((0.06, "excellent"), (0.10, "good"), (0.16, "fair"), (float("inf"), "poor, recalibrate"))

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Dot")
        self.setFixedWidth(360)
        self.status = QLabel("Ready when you are.")
        self.status.setWordWrap(True)
        self.calibrate_button = QPushButton("Scan, calibrate and start")
        self.calibrate_button.clicked.connect(self.start_calibration)
        self.stop_button = QPushButton("Stop tracking")
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gaze Dot"))
        layout.addWidget(self.status)
        layout.addWidget(self.calibrate_button)
        layout.addWidget(self.stop_button)

        screen = QApplication.primaryScreen()
        self.screen_size = np.array(
            [screen.geometry().width(), screen.geometry().height()], dtype=float
        )
        self.screen_mm = display_width_mm()
        self.overlay = ScreenOverlay(screen)
        self.worker = None
        self.model = None
        self.phase = "idle"
        self.step_timer = QTimer(self)
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.advance)
        self.presence_timer = QTimer(self)
        self.presence_timer.setInterval(250)
        self.presence_timer.timeout.connect(self.check_presence)
        self.recent_features = deque(maxlen=MEDIAN_WINDOW)
        self.filter = OneEuroFilter()
        self.last_seen = 0.0
        self.reset_session()

    def reset_session(self):
        self.scan = None
        self.model = None
        self.steps = [Step("scan", (0.5, 0.5), 600, 3500)]
        self.steps += [Step("static", dot, 800, 1200) for dot in self.STATIC_DOTS]
        self.steps += [Step("motion", dot, 800, 4000) for dot in self.MOTION_DOTS]
        self.steps += [Step("validate", dot, 800, 1200) for dot in self.VALIDATION_DOTS]
        self.step_index = 0
        self.buffer = []
        self.train = CalibrationData()
        self.validation_frames = []

    def start_calibration(self):
        self.step_timer.stop()
        self.reset_session()
        self.recent_features.clear()
        self.calibrate_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status.setText("Scanning and calibrating. Allow camera access if macOS asks.")

        if self.worker is None:
            self.worker = GazeWorker(self)
            self.worker.measurement_ready.connect(self.on_measurement)
            self.worker.status_changed.connect(self.on_worker_status)
            self.worker.start()

        self.phase = "ready"
        self.overlay.show_calibration(
            None,
            False,
            "Face scan and calibration: about a minute\n"
            "Sit as you will while watching. Look at each dot until it moves.",
        )
        self.showMinimized()
        self.step_timer.start(self.READY_MS)

    def current_step(self):
        return self.steps[self.step_index]

    def step_message(self, note=None):
        step = self.current_step()
        same_kind = [s for s in self.steps if s.kind == step.kind]
        position = same_kind.index(step) + 1
        title = self.TITLES[step.kind]
        if len(same_kind) > 1 and step.kind != "motion":
            title += f" ({position}/{len(same_kind)})"
        return title + (f"\n{note}" if note else "")

    def advance(self):
        if self.phase == "ready":
            self.begin_step()
        elif self.phase == "settle":
            self.phase = "capture"
            self.buffer = []
            step = self.current_step()
            self.overlay.show_calibration(step.target, True, self.step_message())
            self.step_timer.start(step.capture_ms)
        elif self.phase == "capture":
            self.finish_step()

    def begin_step(self, note=None):
        step = self.current_step()
        self.phase = "settle"
        self.overlay.show_calibration(step.target, False, self.step_message(note))
        self.step_timer.start(step.settle_ms)

    def finish_step(self):
        step = self.current_step()
        if len(self.buffer) < self.MIN_SAMPLES[step.kind]:
            self.begin_step("Can't see your eyes clearly. Face the screen in even light.")
            return

        features = np.array([m.features for m in self.buffer])
        if step.kind == "scan":
            self.scan = analyze_scan(self.buffer)
        elif step.kind == "static":
            self.train.add_static(self.step_index, step.target, features)
        elif step.kind == "motion":
            self.train.add_motion(self.step_index, step.target, features)
        else:
            self.validation_frames.append(features)

        self.step_index += 1
        if self.step_index == len(self.steps):
            self.finish_calibration()
            return
        if self.current_step().kind == "validate" and self.model is None:
            self.model, _ = select_model(self.train, self.screen_size)
        self.begin_step()

    def finish_calibration(self):
        errors = evaluate_frames(
            self.model, self.validation_frames, self.VALIDATION_DOTS, self.screen_size
        )
        typical, worst = float(np.median(errors)), float(errors.max())

        # Refit with the verification dots too, now that they have been used to measure accuracy.
        final = CalibrationData()
        final.extend(self.train)
        for offset, (frames, dot) in enumerate(zip(self.validation_frames, self.VALIDATION_DOTS)):
            final.add_static(1000 + offset, dot, frames)
        self.model, _ = select_model(final, self.screen_size)

        verdict = next(word for limit, word in self.VERDICTS if typical / self.screen_size[1] < limit)
        accuracy = f"Tracking: {verdict}. Typical error {typical:.0f}px, worst {worst:.0f}px"
        distance_mm = self.scan.distance_cm * 10
        if 200 < distance_mm < 1200:
            degrees = pixels_to_degrees(typical, self.screen_size[0], self.screen_mm, distance_mm)
            accuracy += f" (about {degrees:.1f} degrees)"
        report = "\n".join([accuracy, f"Model: {self.model.description}", *self.scan.lines])

        self.filter = OneEuroFilter()
        self.recent_features.clear()
        self.phase = "tracking"
        self.presence_timer.start()
        self.overlay.show_tracking(report)
        QTimer.singleShot(9000, self.overlay.clear_message)
        self.calibrate_button.setEnabled(True)
        self.status.setText(report)

    def on_worker_status(self, message):
        if message.startswith(("Camera unavailable", "Could not read")):
            self.stop_tracking()
        self.status.setText(message)

    def is_blink(self, measurement):
        if measurement.blink > self.BLINK_LIMIT:
            return True
        return self.scan is not None and measurement.eye_open.mean() < 0.55 * self.scan.eye_open

    def on_measurement(self, measurement):
        self.last_seen = time.monotonic()
        if self.is_blink(measurement):
            return
        if self.phase == "capture":
            self.buffer.append(measurement)
        elif self.phase == "tracking":
            self.recent_features.append(measurement.features)
            fraction = self.model.predict(np.median(self.recent_features, axis=0))
            x, y = self.filter(fraction * self.screen_size, time.monotonic())
            self.overlay.set_blob(x, y)

    def check_presence(self):
        if self.phase == "tracking" and time.monotonic() - self.last_seen > 0.6:
            self.overlay.hide_blob()
            self.filter = OneEuroFilter()
            self.recent_features.clear()

    def shutdown(self):
        self.phase = "idle"
        self.step_timer.stop()
        self.presence_timer.stop()
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()
            self.worker = None
        self.model = None
        self.overlay.hide()

    def stop_tracking(self):
        self.shutdown()
        self.stop_button.setEnabled(False)
        self.calibrate_button.setEnabled(True)
        self.status.setText("Tracking stopped.")
        self.showNormal()
        self.raise_()

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


def main():
    try:
        ensure_model()
    except OSError as error:
        print(f"Could not get the face model ({error}). Check your internet connection and try again.")
        return 1

    app = QApplication(sys.argv)
    window = EyeTracker()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())