"""Camera-free regression tests for geometry, model selection, and app state."""
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from eye_tracker import EyeTracker, open_camera, preview_image
from face_scan import EYES, EYE_CONTOURS, _eye_geometry, _head_pose, ensure_model
from gaze_model import (
    FEATURE_NAMES, CalibrationData, GazeModel, Measurement, OneEuroFilter,
    ScanReport, evaluate_frames, robust_center, select_model,
)
from screen_overlay import _pin_windows


def measurement(**changes):
    features = np.zeros(len(FEATURE_NAMES))
    features[11] = -50
    return replace(Measurement(features, 0, np.array([0.2, 0.2]), 16, 120, 0.3,
                               timestamp=time.monotonic()), **changes)


def calibration():
    data = CalibrationData()
    for i, (x, y) in enumerate((x, y) for y in (0.04, 0.25, 0.5, 0.75, 0.96)
                               for x in (0.04, 0.25, 0.5, 0.75, 0.96)):
        features = np.zeros(len(FEATURE_NAMES))
        features[0:2] = (np.array([x, y]) - 0.5) * 0.35
        features[11] = -50
        features[12:14] = features[22:24] = features[:2]
        data.add_static(i, (x, y), [features] * 20)
    return data


class GeometryTests(unittest.TestCase):
    def eye_points(self, angle=0, scale=1):
        eye, contour = EYES[0], EYE_CONTOURS[0]
        positions = {}
        for i, index in enumerate(contour[0]):
            positions[index] = np.array([100 + i * 10, 100 - 12 * np.sin(i * np.pi / 8)])
        for i, index in enumerate(contour[1]):
            positions[index] = np.array([100 + i * 10, 100 + 12 * np.sin(i * np.pi / 8)])
        positions[eye[4]] = np.array([146, 102])
        for index, offset in zip(eye[5], ((10, 0), (0, -10), (-10, 0), (0, 10))):
            positions[index] = positions[eye[4]] + offset
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = [SimpleNamespace(x=0, y=0, z=0) for _ in range(478)]
        for index, point in positions.items():
            transformed = rotation @ (point * scale) + np.array([200, 200])
            points[index] = SimpleNamespace(x=transformed[0] / 1000, y=transformed[1] / 1000, z=0)
        return points

    def test_full_contour_geometry_is_roll_and_scale_invariant(self):
        original = _eye_geometry(self.eye_points(), 1000, 1000, EYES[0], EYE_CONTOURS[0])
        moved = _eye_geometry(self.eye_points(0.23, 1.4), 1000, 1000, EYES[0], EYE_CONTOURS[0])
        self.assertEqual(len(original["contour"]), 16)
        self.assertEqual(len(original["ring"]), 4)
        for name in ("iris_x", "iris_y", "iris_lid_y", "lid_open", "iris_ratio", "inner_open", "outer_open"):
            self.assertAlmostEqual(original[name], moved[name], places=9)

    def test_degenerate_eye_is_rejected(self):
        points = [SimpleNamespace(x=0.5, y=0.5, z=0) for _ in range(478)]
        self.assertIsNone(_eye_geometry(points, 100, 100, EYES[0]))

    def test_head_pose_preserves_translation_and_removes_scale(self):
        matrix = np.eye(4)
        matrix[:3, :3] *= 1.4
        matrix[:3, 3] = (2, 3, -50)
        np.testing.assert_allclose(_head_pose(matrix), [0, 0, 0, 2, 3, -50])

    def test_model_download_failure_cleans_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "face.task")
            with patch("face_scan.MODEL_PATH", path), patch("face_scan.urllib.request.urlopen", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    ensure_model()
            self.assertFalse(Path(path + ".part").exists())


class ModelTests(unittest.TestCase):
    def test_robust_center_ignores_landmark_outlier(self):
        samples = np.zeros((20, len(FEATURE_NAMES)))
        samples[-1, :2] = 10
        center, retained = robust_center(samples)
        self.assertEqual(retained, 19)
        np.testing.assert_allclose(center[:2], 0)

    def test_empty_or_nonfinite_samples_are_rejected(self):
        for samples in ([], [[float("nan")] * len(FEATURE_NAMES)]):
            with self.assertRaises(ValueError):
                robust_center(samples)

    def test_motion_chunks_keep_total_weight_without_empty_rows(self):
        data = CalibrationData()
        data.add_motion(1, (0.5, 0.5), np.zeros((3, len(FEATURE_NAMES))))
        self.assertEqual(len(data.features), 3)
        self.assertAlmostEqual(sum(data.weights), data.MOTION_WEIGHT)
        self.assertTrue(np.isfinite(data.arrays()[0]).all())

    def test_saccade_capture_is_rejected(self):
        samples = np.zeros((20, len(FEATURE_NAMES)))
        samples[:10, :2] = -0.2
        samples[10:, :2] = 0.2
        self.assertFalse(CalibrationData().stable(samples))

    def test_cross_validation_predicts_unseen_edges(self):
        model, error = select_model(calibration(), np.array([1920, 1080]))
        self.assertLess(error, 60)
        for target in ((0.09, 0.09), (0.91, 0.91), (0.08, 0.88)):
            features = np.zeros(len(FEATURE_NAMES))
            features[:2] = (np.array(target) - 0.5) * 0.35
            features[11] = -50
            features[12:14] = features[22:24] = features[:2]
            self.assertLess(np.linalg.norm((model.predict(features) - target) * [1920, 1080]), 60)

    def test_selection_can_be_cancelled(self):
        with self.assertRaises(InterruptedError):
            select_model(calibration(), [1920, 1080], lambda: True)

    def test_numeric_profile_roundtrip_and_shape_validation(self):
        features, targets, weights, _ = calibration().arrays()
        model = GazeModel.fit(features, targets, weights, (0, 1), 2, 0.01)
        payload = json.loads(json.dumps(model.to_dict()))
        loaded = GazeModel.from_dict(payload)
        np.testing.assert_allclose(model.predict_many(features), loaded.predict_many(features))
        payload["scale"] = [0, 1]
        with self.assertRaises(ValueError):
            GazeModel.from_dict(payload)
        payload = model.to_dict()
        payload["weights"] = [[float("nan"), 0]]
        with self.assertRaises(ValueError):
            GazeModel.from_dict(payload)

    def test_filter_follows_saccade_without_overshooting(self):
        smooth = OneEuroFilter()
        smooth([0, 0], 0)
        values = [smooth([1000, 500], (i + 1) / 30).copy() for i in range(30)]
        self.assertGreater(values[5][0], 900)
        self.assertTrue(all(0 <= value[0] <= 1000 for value in values))
        self.assertTrue(np.all(np.diff([value[0] for value in values]) >= 0))

    def test_verification_requires_live_median_window(self):
        features, targets, weights, _ = calibration().arrays()
        model = GazeModel.fit(features, targets, weights, (0, 1), 1, 0.01)
        with self.assertRaises(ValueError):
            evaluate_frames(model, [features[:2]], [targets[0]], [1920, 1080])


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = EyeTracker()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_precision_and_original_calibration_paths(self):
        self.assertEqual(len([s for s in self.window.steps if s.kind == "static"]), 25)
        self.window.mode_picker.setCurrentIndex(1)
        self.window.reset_session()
        self.assertEqual(len([s for s in self.window.steps if s.kind == "static"]), 13)
        self.assertEqual(len(self.window.VALIDATION_DOTS), 9)
        self.assertFalse(set(self.window.PRECISION_DOTS) & set(self.window.VALIDATION_DOTS))

    def test_one_eye_blink_is_rejected_even_when_mean_is_low(self):
        sample = measurement(blink=0.35, eye_blinks=np.array([0.7, 0]))
        self.assertTrue(self.window.is_blink(sample))

    def test_per_eye_scan_baseline_detects_partial_occlusion(self):
        self.window.scan = ScanReport([], [], 0.2, 50, np.array([0.2, 0.2]))
        self.assertTrue(self.window.is_blink(measurement(eye_open=np.array([0.06, 0.25]))))
        self.assertFalse(self.window.is_blink(measurement()))

    def test_low_quality_frames_do_not_enter_calibration(self):
        self.window.phase = "capture"
        self.window.on_measurement(measurement(quality=0.1))
        self.assertEqual(self.window.buffer, [])
        self.window.on_measurement(measurement())
        self.assertEqual(len(self.window.buffer), 1)

    def test_camera_startup_waits_for_a_clear_eye_measurement(self):
        with patch.object(self.window, "_start_camera"):
            self.window.start_calibration()
        self.assertEqual(self.window.phase, "waiting")
        self.assertFalse(self.window.step_timer.isActive())
        self.window.on_measurement(measurement())
        self.assertEqual(self.window.phase, "ready")
        self.assertTrue(self.window.step_timer.isActive())

    def test_pause_resume_and_face_loss_hide_overlay(self):
        self.window.phase = "tracking"
        self.window.overlay.set_blob(100, 100)
        self.window.toggle_pause()
        self.assertEqual(self.window.phase, "paused")
        self.assertFalse(self.window.overlay.isVisible())
        with patch("screen_overlay.pin_above_everything"):
            self.window.toggle_pause()
        self.assertEqual(self.window.phase, "tracking")
        self.window.last_usable = 0
        self.window.check_presence()
        self.assertIsNone(self.window.overlay.blob)

    def test_validation_does_not_train_or_replace_evaluated_model(self):
        data = calibration()
        features, targets, weights, _ = data.arrays()
        model = GazeModel.fit(features, targets, weights, (0, 1), 1, 0.01)
        self.window.model = model
        self.window.scan = ScanReport([], [], 0.2, 50, np.array([0.2, 0.2]))
        self.window.validation_frames = []
        for target in self.window.VALIDATION_DOTS:
            row = np.zeros(len(FEATURE_NAMES))
            row[:2] = (np.array(target) - 0.5) * 0.35
            self.window.validation_frames.append(np.tile(row, (20, 1)))
        with patch("screen_overlay.pin_above_everything"):
            self.window.finish_calibration()
        self.assertIs(self.window.model, model)
        self.assertEqual(len(self.window.metrics["per_dot_errors"]), 9)
        self.assertEqual(self.window.phase, "tracking")

    def test_recording_saves_coordinates_and_stops_cleanly(self):
        features, targets, weights, _ = calibration().arrays()
        self.window.model = GazeModel.fit(features, targets, weights, (0, 1), 1, 0.01)
        self.window.phase = "tracking"
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "session.csv")
            with patch("eye_tracker.QFileDialog.getSaveFileName", return_value=(path, "")):
                self.window.toggle_recording()
            self.window.on_measurement(measurement())
            self.window.toggle_recording()
            content = Path(path).read_text()
            self.assertIn("image_quality_heuristic", content)
            self.assertEqual(len(content.splitlines()), 2)
            self.assertIsNone(self.window.csv_file)

    def test_preview_owns_its_image_memory(self):
        frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
        image = preview_image(frame, None, False)
        frame[:] = 0
        self.assertEqual(image.size().width(), 480)
        self.assertEqual(image.pixelColor(0, 0).red(), 127)

    def test_native_camera_failure_falls_back_and_releases_first_handle(self):
        first = SimpleNamespace(isOpened=lambda: False, release=lambda: None)
        second = SimpleNamespace(isOpened=lambda: True, release=lambda: None, set=lambda *_: None)
        with patch("eye_tracker.sys.platform", "win32"), patch("eye_tracker.cv2.VideoCapture", side_effect=[first, second]) as capture, patch.object(first, "release") as release:
            self.assertIs(open_camera(1), second)
            release.assert_called_once()
            self.assertEqual(capture.call_count, 2)

    def test_windows_native_overlay_is_topmost_clickthrough_and_nonactivating(self):
        import ctypes
        get_style = Mock(return_value=0)
        set_style = Mock(return_value=0)
        position = Mock(return_value=True)
        user32 = SimpleNamespace(GetWindowLongPtrW=get_style, SetWindowLongPtrW=set_style,
                                 GetWindowLongW=get_style, SetWindowLongW=set_style, SetWindowPos=position)
        with patch("screen_overlay.ctypes.WinDLL", return_value=user32, create=True):
            _pin_windows(SimpleNamespace(winId=lambda: 123))
        get_style.assert_called_once_with(123, -20)
        self.assertEqual(set_style.call_args.args[2] & 0x08000020, 0x08000020)
        self.assertEqual(position.call_args.args[1].value, ctypes.c_void_p(-1).value)
        self.assertTrue(position.call_args.args[-1] & 0x10)
        self.assertEqual(position.argtypes[0], ctypes.wintypes.HWND)

    def test_calibration_save_load_preserves_model_and_waits_for_camera(self):
        features, targets, weights, _ = calibration().arrays()
        model = GazeModel.fit(features, targets, weights, (0, 1), 1, 0.01)
        self.window.model = model
        self.window.scan = ScanReport(["Scan good"], [], 0.2, 50, np.array([0.2, 0.2]))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "profile.json")
            with patch.object(self.window, "_profile_path", return_value=path), patch(
                "eye_tracker.QFileDialog.getSaveFileName", return_value=(path, "")
            ):
                self.window.save_calibration()
            self.assertTrue(Path(path).exists())
            self.window.model = None
            with patch.object(self.window, "_profile_path", return_value=path), patch(
                "eye_tracker.QFileDialog.getOpenFileName", return_value=(path, "")
            ), patch.object(self.window, "_start_camera"):
                self.window.load_calibration()
            self.assertEqual(self.window.phase, "waiting")
            np.testing.assert_allclose(self.window.model.predict_many(features), model.predict_many(features))
            self.window.on_measurement(measurement())
            self.assertEqual(self.window.phase, "tracking")
            self.assertIn("earlier session", self.window.status.text())

    def test_repeated_capture_failures_stop_instead_of_retrying_forever(self):
        self.window.phase = "capture"
        for _ in range(4):
            self.window.finish_step()
        self.assertEqual(self.window.phase, "idle")
        self.assertFalse(self.window.step_timer.isActive())
        self.assertIn("repeated unclear captures", self.window.status.text())


if __name__ == "__main__":
    unittest.main()
