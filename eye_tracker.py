"""Local webcam gaze estimation with per-eye diagnostics and calibrated overlays."""
import csv
import json
import math
import sys
import time
from collections import deque, namedtuple
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QStandardPaths, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from face_scan import FaceAnalyzer
from gaze_model import (
    FEATURE_NAMES, MEDIAN_WINDOW, CalibrationData, GazeModel, OneEuroFilter,
    ScanReport, analyze_scan, evaluate_frames, pixels_to_degrees, select_model,
)
from screen_overlay import ScreenOverlay, display_width_mm

Step = namedtuple("Step", "kind target settle_ms capture_ms")


def open_camera(index):
    """Use native capture first, then OpenCV's automatic backend as a fallback."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else (
        cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    )
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened() and backend != cv2.CAP_ANY:
        camera.release()
        camera = cv2.VideoCapture(index)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def preview_image(frame, measurement, draw_mesh):
    """Render only the diagnostic preview; inference always uses the original frame."""
    canvas = frame.copy()
    if measurement is not None and draw_mesh:
        for contour, ring in zip(measurement.eye_contours, measurement.iris_rings):
            cv2.polylines(canvas, [np.rint(contour).astype(np.int32)], True, (130, 245, 70), 1, cv2.LINE_AA)
            cv2.polylines(canvas, [np.rint(ring).astype(np.int32)], True, (255, 195, 65), 1, cv2.LINE_AA)
            for point in contour:
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 2, (130, 245, 70), -1, cv2.LINE_AA)
    canvas = cv2.flip(canvas, 1)
    canvas = cv2.resize(canvas, (480, 270), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, 480, 270, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class GazeWorker(QThread):
    measurement_ready = Signal(object)
    preview_ready = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.preview_enabled = True
        self.draw_mesh = True

    def run(self):
        camera = None
        analyzer = None
        try:
            self.status_changed.emit("Loading the face model. First launch may download it.")
            analyzer = FaceAnalyzer()
            if self.isInterruptionRequested():
                return
            camera = open_camera(self.camera_index)
            if not camera.isOpened():
                permission = "System Settings > Privacy & Security > Camera" if sys.platform == "darwin" else (
                    "Settings > Privacy & security > Camera; enable desktop app access" if sys.platform == "win32" else
                    "your camera permissions and whether another app is using it"
                )
                raise RuntimeError(f"Camera unavailable. Check {permission}, or choose another camera index.")
            self.status_changed.emit("Camera on. Center your face in the frame.")
            missing = False
            last_preview = 0.0
            while not self.isInterruptionRequested():
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read from the camera. Reconnect it and try again.")
                now = time.monotonic()
                measurement = analyzer.analyze(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(now * 1000))
                if measurement is not None:
                    self.measurement_ready.emit(measurement)
                    if missing:
                        self.status_changed.emit("Face found.")
                    missing = False
                elif not missing:
                    self.status_changed.emit("Looking for your face...")
                    missing = True
                if self.preview_enabled and now - last_preview >= 1 / 12:
                    self.preview_ready.emit(preview_image(frame, measurement, self.draw_mesh))
                    last_preview = now
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if analyzer is not None:
                analyzer.close()
            if camera is not None:
                camera.release()


class ModelWorker(QThread):
    fitted = Signal(object, float)
    failed = Signal(str)

    def __init__(self, data, screen_size, parent=None):
        super().__init__(parent)
        self.data = data
        self.screen_size = screen_size.copy()

    def run(self):
        try:
            model, error = select_model(self.data, self.screen_size, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.fitted.emit(model, error)
        except InterruptedError:
            pass
        except Exception as error:
            self.failed.emit(f"Model fitting failed: {error}")


class EyeTracker(QWidget):
    # The original, faster 13-point workflow remains available.
    STATIC_DOTS = (
        (0.06, 0.06), (0.5, 0.06), (0.94, 0.06), (0.72, 0.28), (0.28, 0.28),
        (0.06, 0.5), (0.5, 0.5), (0.94, 0.5), (0.72, 0.72), (0.28, 0.72),
        (0.06, 0.94), (0.5, 0.94), (0.94, 0.94),
    )
    # Serpentine order covers all four edges without unnecessary long jumps.
    PRECISION_DOTS = tuple(
        (x, y) for row, y in enumerate((0.04, 0.25, 0.5, 0.75, 0.96))
        for x in ((0.04, 0.25, 0.5, 0.75, 0.96) if row % 2 == 0 else (0.96, 0.75, 0.5, 0.25, 0.04))
    )
    MOTION_DOTS = ((0.5, 0.5), (0.75, 0.3), (0.2, 0.75))
    VALIDATION_DOTS = (
        (0.12, 0.12), (0.5, 0.16), (0.88, 0.12),
        (0.15, 0.5), (0.5, 0.38), (0.85, 0.5),
        (0.12, 0.88), (0.5, 0.84), (0.88, 0.88),
    )
    TITLES = {
        "scan": "Step 1 of 4 - Scanning your eyes: look at the dot and hold still",
        "static": "Step 2 of 4 - Calibrating: look at the dot",
        "motion": "Step 3 of 4 - Keep looking at the dot and slowly move your head around",
        "validate": "Step 4 of 4 - Checking accuracy: look at the dot",
    }
    MIN_SAMPLES = {"scan": 20, "static": 12, "motion": 40, "validate": 12}
    READY_MS = 3000
    BLINK_LIMIT = 0.5
    QUALITY_LIMIT = 0.22
    VERDICTS = ((0.06, "excellent"), (0.10, "good"), (0.16, "fair"), (float("inf"), "poor; recalibrate"))
    FILTERS = {"Balanced": (0.8, 0.004), "Responsive": (1.4, 0.008), "Smooth": (0.5, 0.0015)}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracker | Gaze Dot")
        self.setMinimumWidth(560)
        self.worker = None
        self.fit_worker = None
        self.model = None
        self.phase = "idle"
        self._closing = False
        self._loaded_profile = None
        self._last_diagnostic = 0.0
        self._last_preview = None
        self.last_seen = 0.0
        self.last_usable = 0.0
        self.capture_started = 0.0
        self.step_retries = 0
        self.csv_file = None
        self.csv_writer = None
        self.record_started = 0.0
        self.timestamps = deque(maxlen=60)
        self.recent_features = deque(maxlen=MEDIAN_WINDOW)
        self.filter = OneEuroFilter()
        self.metrics = {}
        self._build_ui()
        self.overlay = ScreenOverlay(QApplication.primaryScreen())
        self._set_screen()
        self.step_timer = QTimer(self)
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.advance)
        self.presence_timer = QTimer(self)
        self.presence_timer.setInterval(100)
        self.presence_timer.timeout.connect(self.check_presence)
        self.presence_timer.start()
        self.report_timer = QTimer(self)
        self.report_timer.setSingleShot(True)
        self.report_timer.timeout.connect(self.overlay.clear_message)
        self.reset_session()
        self.screen_picker.currentIndexChanged.connect(self._set_screen)
        self.radius_slider.valueChanged.connect(self.overlay.set_radius)
        self.smoothing.currentTextChanged.connect(self.reset_filter)
        self.preview_toggle.toggled.connect(self._preview_changed)
        self.mesh_toggle.toggled.connect(self._preview_changed)
        QApplication.instance().screenRemoved.connect(self._screen_removed)
        QApplication.instance().screenAdded.connect(self._screen_added)
        self.shortcuts = []
        for key, callback in (("Escape", self.stop_tracking), ("Space", self.toggle_pause), ("Ctrl+R", self.start_calibration)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { background: #101722; color: #dfeaf4; font-size: 13px; }
            QLabel#title { font-size: 28px; font-weight: 700; color: #f2f8ff; }
            QLabel#subtitle { color: #8eacc3; }
            QLabel#preview { background: #080e17; border: 1px solid #2b3c50; border-radius: 10px; }
            QPushButton { background: #213247; border: 1px solid #3b5269; border-radius: 7px; padding: 10px; }
            QPushButton:hover { background: #304761; }
            QPushButton#primary { background: #176b60; border-color: #2daf98; font-weight: 600; }
            QPushButton:disabled { color: #64788e; background: #182331; border-color: #243447; }
            QComboBox, QSpinBox { background: #1a2838; border: 1px solid #3b5269; border-radius: 5px; padding: 5px; }
            QProgressBar { border: 1px solid #2b3c50; border-radius: 5px; text-align: center; min-height: 20px; }
            QProgressBar::chunk { background: #238f80; }
            QSlider::groove:horizontal { background: #2c4157; height: 5px; }
            QSlider::handle:horizontal { background: #60d8bd; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(10)
        # Keep start/stop controls reachable even on a 768-pixel laptop display.
        content = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)
        title = QLabel("EYE TRACKER")
        title.setObjectName("title")
        subtitle = QLabel("Gaze Dot  /  binocular geometry  /  local processing")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.preview = QLabel("Camera preview appears when you start calibration.")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedSize(480, 270)
        layout.addWidget(self.preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        preview_row = QHBoxLayout()
        self.preview_toggle = QCheckBox("Camera preview")
        self.preview_toggle.setChecked(True)
        self.mesh_toggle = QCheckBox("Eye contours + iris rings")
        self.mesh_toggle.setChecked(True)
        preview_row.addWidget(self.preview_toggle)
        preview_row.addWidget(self.mesh_toggle)
        layout.addLayout(preview_row)
        self.diagnostics = QLabel("Left / right eye: --   |   Image quality: --   |   FPS: --")
        self.diagnostics.setWordWrap(True)
        layout.addWidget(self.diagnostics)
        form = QFormLayout()
        self.screen_picker = QComboBox()
        for screen in QApplication.screens():
            geometry = screen.geometry()
            self.screen_picker.addItem(f"{screen.name()}  ({geometry.width()} x {geometry.height()})", screen)
        self.screen_picker.setCurrentIndex(QApplication.screens().index(QApplication.primaryScreen()))
        self.camera_picker = QSpinBox()
        self.camera_picker.setRange(0, 10)
        self.camera_picker.setToolTip("0 is normally your built-in webcam; try 1 for a USB camera.")
        self.mode_picker = QComboBox()
        self.mode_picker.addItems(["Precision - 25 points + 9 verification", "Quick - original 13 points + 9 verification"])
        self.smoothing = QComboBox()
        self.smoothing.addItems(list(self.FILTERS))
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(16, 180)
        self.radius_slider.setValue(90)
        form.addRow("Display", self.screen_picker)
        form.addRow("Camera index", self.camera_picker)
        form.addRow("Calibration", self.mode_picker)
        form.addRow("Smoothing", self.smoothing)
        form.addRow("Gaze blob size", self.radius_slider)
        layout.addLayout(form)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = QLabel("Ready when you are. Use even lighting and keep both eyes visible.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout = root
        self.calibrate_button = QPushButton("Scan, calibrate and start")
        self.calibrate_button.setObjectName("primary")
        self.calibrate_button.clicked.connect(self.start_calibration)
        layout.addWidget(self.calibrate_button)
        row = QHBoxLayout()
        self.pause_button = QPushButton("Pause overlay")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("Stop tracking")
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)
        row.addWidget(self.pause_button)
        row.addWidget(self.stop_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.save_button = QPushButton("Save calibration")
        self.save_button.clicked.connect(self.save_calibration)
        self.save_button.setEnabled(False)
        self.load_button = QPushButton("Load calibration")
        self.load_button.clicked.connect(self.load_calibration)
        self.record_button = QPushButton("Record gaze CSV")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        for button in (self.save_button, self.load_button, self.record_button):
            row.addWidget(button)
        layout.addLayout(row)
        hint = QLabel("Esc: stop  |  Space: pause  |  Ctrl+R: recalibrate (controls focused)\nCamera frames stay in memory. CSV recording starts only when you choose it.")
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.resize(560, min(900, max(500, QApplication.primaryScreen().availableGeometry().height() - 80)))

    def _set_screen(self, *_):
        screen = self.screen_picker.currentData() or QApplication.primaryScreen()
        previous = getattr(self, "selected_screen", None)
        if previous is not screen:
            if previous is not None:
                try:
                    previous.geometryChanged.disconnect(self._geometry_changed)
                except RuntimeError:
                    pass
            screen.geometryChanged.connect(self._geometry_changed)
        self.selected_screen = screen
        self.screen_size = np.array([screen.geometry().width(), screen.geometry().height()], dtype=float)
        self.screen_mm = display_width_mm(screen)
        self.overlay.setGeometry(screen.geometry())

    def _geometry_changed(self, *_):
        if self.phase != "idle":
            self.stop_tracking()
            self.status.setText("Display geometry changed. Calibrate again for the new display size.")
        self._set_screen()

    def _screen_removed(self, screen):
        if screen is self.selected_screen:
            self.stop_tracking()
        for index in range(self.screen_picker.count() - 1, -1, -1):
            if self.screen_picker.itemData(index) is screen:
                self.screen_picker.removeItem(index)
        self._set_screen()

    def _screen_added(self, screen):
        geometry = screen.geometry()
        self.screen_picker.addItem(f"{screen.name()}  ({geometry.width()} x {geometry.height()})", screen)

    def reset_filter(self, *_):
        cutoff, beta = self.FILTERS[self.smoothing.currentText()]
        self.filter = OneEuroFilter(min_cutoff=cutoff, beta=beta)
        self.recent_features.clear()

    def reset_session(self):
        self.scan = None
        self.model = None
        dots = self.PRECISION_DOTS if self.mode_picker.currentIndex() == 0 else self.STATIC_DOTS
        self.steps = [Step("scan", (0.5, 0.5), 600, 3500)]
        self.steps += [Step("static", dot, 800, 1400) for dot in dots]
        self.steps += [Step("motion", dot, 800, 4500) for dot in self.MOTION_DOTS]
        self.steps += [Step("validate", dot, 800, 1400) for dot in self.VALIDATION_DOTS]
        self.step_index = 0
        self.step_retries = 0
        self.buffer = []
        self.train = CalibrationData()
        self.validation_frames = []
        self.metrics = {}
        self.progress.setValue(0)

    def _set_busy(self, busy):
        for control in (self.screen_picker, self.camera_picker, self.mode_picker, self.load_button):
            control.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def _start_camera(self):
        if self.worker is not None:
            return
        self.worker = GazeWorker(self.camera_picker.value(), self)
        self.worker.measurement_ready.connect(self.on_measurement)
        self.worker.preview_ready.connect(self.on_preview)
        self.worker.status_changed.connect(self.on_worker_status)
        self.worker.failed.connect(self.on_failure)
        self.worker.finished.connect(self._worker_finished)
        self._preview_changed()
        self.worker.start()

    def start_calibration(self):
        if self.fit_worker is not None or (self.worker is not None and self.worker.isInterruptionRequested()):
            return
        self._stop_recording()
        self.step_timer.stop()
        self.report_timer.stop()
        self._loaded_profile = None
        self.reset_session()
        self.reset_filter()
        self.last_seen = self.last_usable = 0.0
        self.calibrate_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self._set_busy(True)
        self._set_screen()
        self.phase = "waiting"
        self.status.setText("Starting the camera. Allow camera access if your system asks.")
        self.overlay.show_calibration(None, False, "Preparing your camera and eye scan...")
        self._start_camera()
        # Camera/model startup does not consume the first scan's capture time.
        if self.worker is not None and self.worker.isRunning() and self.timestamps:
            self.status.setText("Waiting for a clear view of both eyes...")

    def current_step(self):
        return self.steps[self.step_index]

    def step_message(self, note=None):
        step = self.current_step()
        same_kind = [s for s in self.steps if s.kind == step.kind]
        title = self.TITLES[step.kind]
        if len(same_kind) > 1:
            title += f" ({same_kind.index(step) + 1}/{len(same_kind)})"
        return title + (f"\n{note}" if note else "")

    def advance(self):
        if self.phase == "ready":
            self.begin_step()
        elif self.phase == "settle":
            self.phase = "capture"
            self.buffer = []
            self.capture_started = time.monotonic()
            step = self.current_step()
            self.overlay.set_capture_progress(0)
            self.overlay.show_calibration(step.target, True, self.step_message())
            self.step_timer.start(step.capture_ms)
        elif self.phase == "capture":
            self.finish_step()

    def begin_step(self, note=None):
        step = self.current_step()
        self.phase = "settle"
        message = self.step_message(note)
        self.status.setText(message)
        self.overlay.show_calibration(step.target, False, message)
        self.step_timer.start(step.settle_ms)

    def finish_step(self):
        step = self.current_step()
        enough = len(self.buffer) >= self.MIN_SAMPLES[step.kind]
        features = np.array([m.features for m in self.buffer]) if enough else None
        stable = enough and (step.kind == "motion" or self.train.stable(features))
        if not stable:
            self.step_retries += 1
            if self.step_retries > 3:
                self.on_failure("Calibration stopped after repeated unclear captures. Improve lighting, keep both eyes visible, and try again.")
                return
            note = "Keep looking at the dot; hold your eyes steady." if enough else "Can't see both eyes clearly. Face the screen in even light."
            self.begin_step(note)
            return
        self.step_retries = 0
        if step.kind == "scan":
            self.scan = analyze_scan(self.buffer)
        elif step.kind == "static":
            self.train.add_static(self.step_index, step.target, features)
        elif step.kind == "motion":
            self.train.add_motion(self.step_index, step.target, features)
        else:
            self.validation_frames.append(features)
        self.step_index += 1
        self.progress.setValue(round(100 * self.step_index / len(self.steps)))
        if self.step_index == len(self.steps):
            self.finish_calibration()
        elif self.current_step().kind == "validate" and self.model is None:
            self.phase = "fitting"
            self.status.setText("Comparing eye features and head compensation on held-out calibration groups...")
            self.overlay.show_calibration(None, False, "Choosing the model that best predicts unseen calibration dots...")
            self.fit_worker = ModelWorker(self.train, self.screen_size, self)
            self.fit_worker.fitted.connect(self._model_fitted)
            self.fit_worker.failed.connect(self.on_failure)
            self.fit_worker.finished.connect(self._fit_finished)
            self.fit_worker.start()
        else:
            self.begin_step()

    def _model_fitted(self, model, error):
        if self.phase != "fitting":
            return
        self.model = model
        self.metrics["cross_validated_error"] = error
        self.begin_step()

    def _fit_finished(self):
        worker = self.fit_worker
        self.fit_worker = None
        if worker is not None:
            worker.deleteLater()
        self._finish_shutdown()

    def finish_calibration(self):
        errors = evaluate_frames(self.model, self.validation_frames, self.VALIDATION_DOTS, self.screen_size)
        typical, worst = float(np.median(errors)), float(errors.max())
        # Keep the evaluated model: verification points NEVER enter its training data.
        verdict = next(word for limit, word in self.VERDICTS if typical / self.screen_size[1] < limit)
        self.metrics.update(typical_error=typical, worst_error=worst, per_dot_errors=errors.tolist(),
                            validation_targets=[list(dot) for dot in self.VALIDATION_DOTS], verdict=verdict)
        accuracy = f"Tracking: {verdict}. Typical error {typical:.0f}, worst {worst:.0f} logical pixels."
        distance_mm = self.scan.distance_cm * 10
        if self.screen_mm > 0 and 200 < distance_mm < 1200:
            degrees = pixels_to_degrees(typical, self.screen_size[0], self.screen_mm, distance_mm)
            accuracy += f" About {degrees:.1f} degrees (estimated geometry)."
        report = "\n".join([accuracy, f"Model: {self.model.description}", *self.scan.lines])
        self._activate_tracking(report)
        self.progress.setValue(100)

    def _activate_tracking(self, report):
        self.reset_filter()
        self.phase = "tracking"
        self.overlay.show_tracking(report)
        self.report_timer.start(9000)
        self.calibrate_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause overlay")
        self.save_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.status.setText(report)

    def on_worker_status(self, message):
        if self.phase in ("waiting", "ready"):
            self.status.setText(message)

    def on_failure(self, message):
        self.stop_tracking()
        self.status.setText(message)

    def is_blink(self, measurement):
        if max(measurement.blink, float(np.max(measurement.eye_blinks))) > self.BLINK_LIMIT:
            return True
        if self.scan is None:
            return measurement.eye_open.min() < 0.045
        baseline = self.scan.eye_open_by_eye
        if np.all(baseline > 0):
            return bool((measurement.eye_open < 0.5 * baseline).any())
        return measurement.eye_open.mean() < 0.55 * self.scan.eye_open

    def on_measurement(self, measurement):
        if self.phase == "idle":
            return
        now = time.monotonic()
        self.last_seen = now
        self.timestamps.append(now)
        usable = not self.is_blink(measurement) and measurement.quality >= self.QUALITY_LIMIT
        if now - self._last_diagnostic > 0.25:
            fps = (len(self.timestamps) - 1) / max(0.001, self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0
            state = "BLINK / OCCLUDED" if self.is_blink(measurement) else "OPEN"
            self.diagnostics.setText(
                f"Left / right opening: {measurement.eye_open[0]:.2f} / {measurement.eye_open[1]:.2f}  |  {state}\n"
                f"Image quality: {measurement.quality:.0%} (heuristic)  |  Iris: {measurement.iris_px:.0f}px  |  {fps:.0f} FPS\n"
                f"Head yaw / pitch / roll: {measurement.features[6]:.0f} / {measurement.features[7]:.0f} / {measurement.features[8]:.0f} degrees"
            )
            self._last_diagnostic = now
        if not usable:
            return
        if now - self.last_usable > 0.3:
            self.reset_filter()
        self.last_usable = now
        if self.phase == "waiting":
            if self._loaded_profile is not None:
                self._activate_tracking("Saved calibration loaded. Recalibrate if your seat, camera, or lighting changed.\nSaved verification results describe the earlier session.")
                self._loaded_profile = None
            else:
                self.phase = "ready"
                seconds = "about 2 minutes" if self.mode_picker.currentIndex() == 0 else "about 1 minute"
                message = f"Face scan and calibration: {seconds}\nLook at each dot until it moves. Keep your controls available to stop."
                self.overlay.show_calibration(None, False, message)
                self.status.setText(message)
                self.step_timer.start(self.READY_MS)
        elif self.phase == "capture":
            self.buffer.append(measurement)
        elif self.phase == "tracking" and self.model is not None:
            self.recent_features.append(measurement.features)
            fraction = self.model.predict(np.median(self.recent_features, axis=0))
            x, y = self.filter(fraction * self.screen_size, now)
            self.overlay.set_blob(x, y, measurement.quality)
            if self.csv_writer is not None:
                try:
                    geometry = self.selected_screen.geometry()
                    self.csv_writer.writerow([f"{now - self.record_started:.4f}", f"{x:.2f}", f"{y:.2f}",
                                              f"{x + geometry.x():.2f}", f"{y + geometry.y():.2f}",
                                              f"{x / self.screen_size[0]:.6f}", f"{y / self.screen_size[1]:.6f}",
                                              f"{measurement.quality:.3f}"])
                except OSError as error:
                    self._stop_recording()
                    self.status.setText(f"Gaze recording stopped: {error}")

    def check_presence(self):
        now = time.monotonic()
        if self.phase == "capture":
            self.overlay.set_capture_progress((now - self.capture_started) * 1000 / self.current_step().capture_ms)
        if self.phase == "tracking" and now - self.last_usable > 0.3:
            self.overlay.hide_blob()
            self.reset_filter()
        if self.phase in ("tracking", "paused") and now - self.last_seen > 0.6:
            self.diagnostics.setText("Face lost. Bring both eyes back into view; tracking resumes automatically.")

    def on_preview(self, image):
        if self.phase != "idle" and self.preview_toggle.isChecked() and self.isVisible() and not self.isMinimized():
            self._last_preview = image
            self.preview.setPixmap(QPixmap.fromImage(image))

    def _preview_changed(self, *_):
        self.preview.setVisible(self.preview_toggle.isChecked())
        if self.worker is not None:
            self.worker.preview_enabled = self.preview_toggle.isChecked() and not self.isMinimized()
            self.worker.draw_mesh = self.mesh_toggle.isChecked()

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "preview_toggle"):
            self._preview_changed()

    def toggle_pause(self):
        if self.phase == "tracking":
            self.phase = "paused"
            self.report_timer.stop()
            self.overlay.hide()
            self.overlay.hide_blob()
            self.pause_button.setText("Resume overlay")
            self.status.setText("Overlay paused. Camera diagnostics continue; gaze recording pauses too.")
        elif self.phase == "paused":
            self.reset_filter()
            self.phase = "tracking"
            self.overlay.show_tracking()
            self.pause_button.setText("Pause overlay")
            self.status.setText("Tracking resumed.")

    def _profile_path(self):
        directory = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / "calibration.json")

    def save_calibration(self):
        if self.model is None or self.scan is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", self._profile_path(), "Calibration JSON (*.json)")
        if not path:
            return
        payload = {"schema": 1, "model": self.model.to_dict(), "screen_size": self.screen_size.tolist(),
                   "screen_name": self.selected_screen.name(), "camera_index": self.camera_picker.value(),
                   "metrics": self.metrics, "scan": {"lines": self.scan.lines, "warnings": self.scan.warnings,
                   "eye_open": self.scan.eye_open, "distance_cm": self.scan.distance_cm,
                   "eye_open_by_eye": self.scan.eye_open_by_eye.tolist()}}
        try:
            temporary = Path(path + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)
            self.status.setText(f"Calibration saved to {path}. Use it only with the same camera, seat, and display.")
        except (OSError, ValueError) as error:
            self.status.setText(f"Could not save calibration: {error}")

    def load_calibration(self):
        if self.worker is not None or self.fit_worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", self._profile_path(), "Calibration JSON (*.json)")
        if not path:
            return
        try:
            if Path(path).stat().st_size > 1_000_000:
                raise ValueError("Calibration file is too large")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self._set_screen()
            if payload.get("schema") != 1 or payload["screen_size"] != self.screen_size.tolist():
                raise ValueError("Display size changed; please recalibrate")
            if payload["screen_name"] != self.selected_screen.name() or payload["camera_index"] != self.camera_picker.value():
                raise ValueError("Choose the display and camera used for this calibration")
            model = GazeModel.from_dict(payload["model"])
            scan = payload["scan"]
            opening = np.asarray(scan["eye_open_by_eye"], dtype=float)
            distance = float(scan["distance_cm"])
            average = float(scan["eye_open"])
            if (opening.shape != (2,) or not np.isfinite(opening).all() or (opening <= 0).any()
                    or not math.isfinite(distance) or not math.isfinite(average) or average <= 0):
                raise ValueError("Invalid eye scan in calibration file")
            self.scan = ScanReport(list(map(str, scan["lines"])), list(map(str, scan["warnings"])), average, distance, opening)
            self.model = model
            self.metrics = payload.get("metrics", {})
            self._loaded_profile = payload
            self.phase = "waiting"
            self.last_seen = self.last_usable = 0.0
            self._set_busy(True)
            self.calibrate_button.setEnabled(False)
            self.status.setText("Loading saved calibration. Waiting for both eyes...")
            self.overlay.show_calibration(None, False, "Loading saved calibration: look toward the camera...")
            self._start_camera()
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            self.status.setText(f"Could not load calibration: {error}")

    def toggle_recording(self):
        if self.csv_file is not None:
            self._stop_recording()
            self.status.setText("Gaze recording saved.")
            return
        if self.phase not in ("tracking", "paused"):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Record gaze samples", "gaze-session.csv", "Gaze CSV (*.csv)")
        if not path:
            return
        try:
            self.csv_file = open(path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["elapsed_seconds", "display_x", "display_y", "desktop_x", "desktop_y",
                                      "gaze_fraction_x", "gaze_fraction_y", "image_quality_heuristic"])
            self.record_started = time.monotonic()
            self.record_button.setText("Stop recording CSV")
            self.status.setText(f"Recording gaze coordinates to {path}. Camera video is not recorded.")
        except OSError as error:
            self._stop_recording()
            self.status.setText(f"Could not start recording: {error}")

    def _stop_recording(self):
        file = self.csv_file
        self.csv_file = self.csv_writer = None
        if file is not None:
            try:
                file.close()
            except OSError:
                pass
        self.record_button.setText("Record gaze CSV")

    def shutdown(self):
        self.phase = "idle"
        self.step_timer.stop()
        self.report_timer.stop()
        self._stop_recording()
        self._loaded_profile = None
        self.overlay.hide()
        self.overlay.hide_blob()
        self.model = None
        self.reset_filter()
        # Release workers asynchronously so stopping never freezes the control window.
        for worker in (self.worker, self.fit_worker):
            if worker is not None:
                worker.requestInterruption()
        self._finish_shutdown()

    def _worker_finished(self):
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self.timestamps.clear()
        self._finish_shutdown()

    def _finish_shutdown(self):
        if self.phase == "idle" and self.worker is None and self.fit_worker is None:
            self._set_busy(False)
            self.calibrate_button.setEnabled(True)
            if self._closing:
                self.overlay.close()
                QTimer.singleShot(0, self.close)

    def stop_tracking(self):
        self.shutdown()
        self.stop_button.setEnabled(False)
        self.calibrate_button.setEnabled(self.worker is None and self.fit_worker is None)
        self.pause_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.status.setText("Tracking stopped." if self.worker is None else "Stopping camera...")
        self.showNormal()
        self.raise_()

    def closeEvent(self, event):
        self._closing = True
        self.shutdown()
        if self.worker is not None or self.fit_worker is not None:
            event.ignore()
            self.status.setText("Releasing the camera and shutting down...")
        else:
            self.presence_timer.stop()
            self.overlay.close()
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Eye Tracker")
    app.setOrganizationName("ItsMeCodyy")
    window = EyeTracker()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
