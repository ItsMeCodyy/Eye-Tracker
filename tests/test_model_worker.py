"""Run real background fitting in a subprocess to catch native library crashes."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class ModelWorkerTests(unittest.TestCase):
    def test_background_calibration_fits_without_native_stack_overflow(self):
        # Main-thread NumPy tests cannot catch the small default QThread stack
        # that caused SIGBUS in OpenBLAS on macOS. Exercise the actual worker.
        code = textwrap.dedent("""
            import numpy as np
            from PySide6.QtCore import QCoreApplication, QTimer
            from eye_tracker import ModelWorker
            from gaze_model import CalibrationData, FEATURE_NAMES

            app = QCoreApplication([])
            data = CalibrationData()
            for group, (x, y) in enumerate(
                (x, y) for y in (.04, .25, .5, .75, .96)
                       for x in (.04, .25, .5, .75, .96)
            ):
                features = np.zeros(len(FEATURE_NAMES))
                features[:2] = (np.array([x, y]) - .5) * .35
                features[11] = -50
                features[12:14] = features[22:24] = features[:2]
                data.add_static(group, (x, y), [features] * 20)

            worker = ModelWorker(data, np.array([1920., 1080.]))
            results, errors = [], []
            worker.fitted.connect(lambda model, error: results.append((model, error)))
            worker.failed.connect(errors.append)
            worker.finished.connect(app.quit)
            QTimer.singleShot(45000, worker.requestInterruption)
            worker.start()
            app.exec()
            assert worker.wait(5000), "Worker did not shut down"
            assert not errors, errors
            assert len(results) == 1, "Worker did not deliver its fitted model"
            model, error = results[0]
            assert np.isfinite(error) and error >= 0, error
            assert np.isfinite(model.weights).all()
            features, targets, _, _ = data.arrays()
            np.testing.assert_allclose(
                model.predict_many(features), targets, atol=.02, rtol=0
            )
            print("Background model fitting and result delivery passed.")
        """)
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Background model fitting and result delivery passed.", result.stdout)


if __name__ == "__main__":
    unittest.main()
