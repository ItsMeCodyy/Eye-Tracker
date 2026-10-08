import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

FEATURE_NAMES = (
    "iris_x",  # iris offset along the eye-corner line, in eye widths
    "iris_y",  # iris offset across the eye-corner line, in eye widths
    "gaze_h",  # learned horizontal gaze from the eye blendshapes
    "gaze_v",  # learned vertical gaze from the eye blendshapes
    "lid_open",  # eyelid opening / eye width
    "iris_lid_y",  # iris position inside the eyelid opening, 0 upper lid .. 1 lower lid
    "yaw",  # head angles in degrees
    "pitch",
    "roll",
    "head_x",  # head position relative to the camera in cm
    "head_y",
    "head_z",
)
EYE_FEATURE_COUNT = 6
# Smallest believable spread of each feature; stops noise in near-constant ones being amplified.
FEATURE_FLOOR = np.array([0.02, 0.02, 0.05, 0.05, 0.02, 0.03, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0])

EYE_SETS = {
    "iris": ("iris_x", "iris_y"),
    "blendshape gaze": ("gaze_h", "gaze_v"),
    "iris + eyelids": ("iris_x", "iris_y", "iris_lid_y", "lid_open"),
    "iris + blendshape": ("iris_x", "iris_y", "gaze_h", "gaze_v"),
    "full eye scan": ("iris_x", "iris_y", "gaze_h", "gaze_v", "iris_lid_y", "lid_open"),
}
HEAD_SETS = {
    "no head": (),
    "head angles": ("yaw", "pitch"),
    "full head pose": ("yaw", "pitch", "roll", "head_x", "head_y", "head_z"),
}
ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0)
CLIP_MARGIN = 0.2
TIE_TOLERANCE = 1.03
MEDIAN_WINDOW = 5  # live tracking takes the median of this many recent frames


@dataclass
class Measurement:
    features: np.ndarray
    blink: float  # mean eyeBlink blendshape, 0 open .. 1 closed
    eye_open: np.ndarray  # eyelid opening / eye width, left and right
    iris_px: float
    luma: float  # mean face brightness, 0..255
    face_width: float  # fraction of the frame width


def robust_center(samples):
    """Average of the frames that agree on the eye features, ignoring saccades and glitches."""
    samples = np.asarray(samples, dtype=float)
    eye = slice(0, EYE_FEATURE_COUNT)
    center = np.median(samples[:, eye], axis=0)
    spread = np.maximum(
        1.4826 * np.median(np.abs(samples[:, eye] - center), axis=0),
        0.25 * FEATURE_FLOOR[eye],
    )
    keep = (np.abs(samples[:, eye] - center) / spread).max(axis=1) <= 3.5
    if keep.sum() < max(3, len(samples) // 2):
        keep[:] = True
    return samples[keep].mean(axis=0), int(keep.sum())


class CalibrationData:
    """Denoised calibration rows. Averaging first keeps the fit from shrinking toward the centre."""

    MOTION_CHUNKS = 8
    MOTION_WEIGHT = 3.0  # total weight of one head-motion dot, in static-dot units

    def __init__(self):
        self.features, self.targets, self.weights, self.groups = [], [], [], []

    def _append(self, features, target, weight, group):
        self.features.append(features)
        self.targets.append(target)
        self.weights.append(weight)
        self.groups.append(group)

    def add_static(self, group, target, samples):
        center, _ = robust_center(samples)
        self._append(center, target, 1.0, group)

    def add_motion(self, group, target, samples):
        """Fixed gaze while the head moves: teaches the model to cancel head motion."""
        for chunk in np.array_split(np.asarray(samples, dtype=float), self.MOTION_CHUNKS):
            self._append(
                np.median(chunk, axis=0), target, self.MOTION_WEIGHT / self.MOTION_CHUNKS, group
            )

    def extend(self, other):
        for row in zip(other.features, other.targets, other.weights, other.groups):
            self._append(*row)

    def arrays(self):
        return (
            np.array(self.features),
            np.array(self.targets),
            np.array(self.weights),
            np.array(self.groups),
        )


def _design(z, degree):
    parts = [np.ones((len(z), 1)), z]
    if degree == 2:  # curvature on the two primary eye features
        parts.append(np.column_stack([z[:, 0] ** 2, z[:, 1] ** 2, z[:, 0] * z[:, 1]]))
    return np.hstack(parts)


@dataclass
class GazeModel:
    columns: tuple
    degree: int
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    low: np.ndarray
    high: np.ndarray
    description: str = field(default="")

    @classmethod
    def fit(cls, features, targets, sample_weights, columns, degree, alpha, description=""):
        columns = tuple(columns)
        x = np.asarray(features, dtype=float)[:, columns]
        y = np.asarray(targets, dtype=float)
        w = np.asarray(sample_weights, dtype=float)
        floor = FEATURE_FLOOR[list(columns)]
        mean = np.average(x, axis=0, weights=w)
        spread = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
        scale = np.maximum(spread, floor)

        design = _design((x - mean) / scale, degree)
        penalty = alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        weighted = design * w[:, None]
        weights = np.linalg.solve(
            weighted.T @ design / w.sum() + penalty, weighted.T @ y / w.sum()
        )

        low, high = x.min(axis=0), x.max(axis=0)
        margin = np.maximum(CLIP_MARGIN * (high - low), 0.75 * floor)
        return cls(columns, degree, mean, scale, weights, low - margin, high + margin, description)

    def predict_many(self, features):
        """Map feature rows to screen positions as fractions of width and height."""
        x = np.clip(np.asarray(features, dtype=float)[:, self.columns], self.low, self.high)
        return np.clip(_design((x - self.mean) / self.scale, self.degree) @ self.weights, 0.0, 1.0)

    def predict(self, features):
        return self.predict_many(np.asarray(features, dtype=float)[None])[0]


def _group_errors(model, features, targets, groups, screen_size):
    errors = []
    for group in np.unique(groups):
        rows = groups == group
        distance = np.linalg.norm(
            (model.predict_many(features[rows]) - targets[rows]) * screen_size, axis=1
        )
        errors.append(distance.mean())
    return np.array(errors)


def _leave_one_group_out(features, targets, weights, groups, columns, degree, alpha, screen_size):
    errors = []
    for group in np.unique(groups):
        held = groups == group
        model = GazeModel.fit(
            features[~held], targets[~held], weights[~held], columns, degree, alpha
        )
        errors.append(_group_errors(model, features[held], targets[held], groups[held], screen_size)[0])
    return float(np.mean(errors))


def select_model(data, screen_size):
    """Pick the eye features, head features and smoothness that predict unseen dots best.

    Returns (model, cross-validated error in pixels). Among configurations within a few
    percent of the best, the simplest one wins.
    """
    features, targets, weights, groups = data.arrays()
    candidates = []
    for (eye_name, eye), (head_name, head), alpha, degree in product(
        EYE_SETS.items(), HEAD_SETS.items(), ALPHAS, (1, 2)
    ):
        columns = tuple(FEATURE_NAMES.index(name) for name in eye + head)
        error = _leave_one_group_out(
            features, targets, weights, groups, columns, degree, alpha, screen_size
        )
        complexity = len(columns) + (3 if degree == 2 else 0)
        label = f"{eye_name} + {head_name}, {'curved' if degree == 2 else 'linear'}, ridge {alpha}"
        candidates.append((error, complexity, -alpha, columns, degree, alpha, label))

    best_error = min(c[0] for c in candidates)
    pool = [c for c in candidates if c[0] <= best_error * TIE_TOLERANCE]
    error, _, _, columns, degree, alpha, label = min(pool, key=lambda c: c[1:3])
    model = GazeModel.fit(features, targets, weights, columns, degree, alpha, label)
    return model, error


def evaluate_frames(model, frames_by_dot, targets, screen_size):
    """Pixel error per dot when scored like live tracking: median of recent frames, then predict."""
    errors = []
    for frames, target in zip(frames_by_dot, targets):
        frames = np.asarray(frames, dtype=float)
        medians = np.array(
            [
                np.median(frames[i - MEDIAN_WINDOW + 1 : i + 1], axis=0)
                for i in range(MEDIAN_WINDOW - 1, len(frames))
            ]
        )
        distance = np.linalg.norm((model.predict_many(medians) - target) * screen_size, axis=1)
        errors.append(distance.mean())
    return np.array(errors)


def pixels_to_degrees(pixels, screen_px, screen_mm, distance_mm):
    return math.degrees(math.atan2(pixels * screen_mm / screen_px, distance_mm))


@dataclass
class ScanReport:
    lines: list
    warnings: list
    eye_open: float
    distance_cm: float


def analyze_scan(measurements):
    """Summarise the eyes, lighting, distance and head position seen while looking straight ahead."""
    features = np.array([m.features for m in measurements])
    eye_open = np.mean([m.eye_open for m in measurements], axis=0)
    iris_px = float(np.mean([m.iris_px for m in measurements]))
    luma = float(np.mean([m.luma for m in measurements]))
    face_width = float(np.mean([m.face_width for m in measurements]))
    yaw, pitch, roll = (float(np.mean(features[:, FEATURE_NAMES.index(n)])) for n in ("yaw", "pitch", "roll"))
    unsteady = max(features[:, FEATURE_NAMES.index(n)].std() for n in ("yaw", "pitch"))
    distance_cm = float(abs(np.mean(features[:, FEATURE_NAMES.index("head_z")])))

    warnings = []
    if face_width < 0.15:
        warnings.append("Move closer to the screen")
    if face_width > 0.6:
        warnings.append("Move back a little")
    if luma < 70:
        warnings.append("Too dark: add light on your face")
    if luma > 200:
        warnings.append("Face is overexposed: reduce glare")
    if abs(roll) > 12:
        warnings.append("Head is tilted")
    if abs(yaw) > 20 or abs(pitch) > 20:
        warnings.append("Face the screen straight on")
    if unsteady > 3:
        warnings.append("Hold still during the scan")
    if eye_open.mean() < 0.15:
        warnings.append("Eyes are narrow, which lowers accuracy")
    if abs(eye_open[0] - eye_open[1]) > 0.3 * eye_open.mean():
        warnings.append("Eyelids open unevenly: check tilt, glare or glasses")

    lines = [
        f"Eyelids open {eye_open[0]:.2f} / {eye_open[1]:.2f} \u00b7 iris {iris_px:.0f}px \u00b7 about {distance_cm:.0f} cm away",
        "; ".join(warnings) if warnings else "Lighting, distance and head position look good",
    ]
    return ScanReport(lines, warnings, float(eye_open.mean()), distance_cm)


class OneEuroFilter:
    """Smooths hard while the point is still and follows quickly when it moves."""

    def __init__(self, min_cutoff=0.15, beta=0.0002, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._time = None
        self._value = None
        self._derivative = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff) / dt)

    def __call__(self, value, now):
        value = np.asarray(value, dtype=float)
        if self._time is None:
            self._time, self._value, self._derivative = now, value, np.zeros_like(value)
            return value

        dt = max(now - self._time, 1e-3)
        raw_derivative = (value - self._value) / dt
        a = self._alpha(self.derivative_cutoff, dt)
        self._derivative = a * raw_derivative + (1 - a) * self._derivative
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self._derivative)
        a = self._alpha(cutoff, dt)
        self._value = a * value + (1 - a) * self._value
        self._time = now
        return self._value
