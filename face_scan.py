import hashlib
import math
import os
import time
import urllib.request

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gaze_model import DETAIL_NAMES, Measurement

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
# The .nosync suffix keeps iCloud from evicting or hiding the file.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.nosync", "face_landmarker.task")

# (corner, corner, upper lid, lower lid, iris centre, iris ring) in FaceMesh numbering.
EYES = (
    (33, 133, 159, 145, 468, (469, 470, 471, 472)),
    (362, 263, 386, 374, 473, (474, 475, 476, 477)),
)
# Full eyelid contours, including both corners and the inner/outer lid arcs.
# FaceMesh's first eye is the person's right eye, not their left.
EYE_CONTOURS = (
    ((33, 246, 161, 160, 159, 158, 157, 173, 133),
     (33, 7, 163, 144, 145, 153, 154, 155, 133)),
    ((362, 398, 384, 385, 386, 387, 388, 466, 263),
     (362, 382, 381, 380, 374, 373, 390, 249, 263)),
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model():
    """Download the official face landmarker once, and check it against the pinned hash."""
    if os.path.exists(MODEL_PATH) and _sha256(MODEL_PATH) == MODEL_SHA256:
        return MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    partial = MODEL_PATH + ".part"
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=20) as response, open(partial, "wb") as file:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                file.write(block)
        if _sha256(partial) != MODEL_SHA256:
            raise OSError("Downloaded face model failed its integrity check")
        os.replace(partial, MODEL_PATH)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return MODEL_PATH


def _eye_geometry(points, width, height, eye, contour=None):
    corner_a, corner_b, upper, lower, iris, ring = eye

    def pixel(index):
        return np.array([points[index].x * width, points[index].y * height])

    left, right = sorted((pixel(corner_a), pixel(corner_b)), key=lambda p: p[0])
    span = right - left
    eye_width = np.linalg.norm(span)
    if eye_width < 1e-3:
        return None

    along = span / eye_width
    across = np.array([-along[1], along[0]])
    upper_lid, lower_lid, center = pixel(upper), pixel(lower), pixel(iris)
    offset = center - (left + right) / 2
    opening = np.dot(lower_lid - upper_lid, across)
    iris_diameter = (
        np.linalg.norm(pixel(ring[0]) - pixel(ring[2]))
        + np.linalg.norm(pixel(ring[1]) - pixel(ring[3]))
    ) / 2
    result = {
        "iris_x": np.dot(offset, along) / eye_width,
        "iris_y": np.dot(offset, across) / eye_width,
        "lid_open": opening / eye_width,
        "iris_lid_y": np.dot(center - upper_lid, across) / opening if opening > 1e-3 else 0.5,
        "iris_px": iris_diameter,
    }
    if contour is not None:
        upper_arc, lower_arc = (np.array([pixel(i) for i in arc]) for arc in contour)
        openings = (lower_arc - upper_arc) @ across / eye_width
        result.update({
            "lid_open": float(np.median(openings[2:7])),
            "inner_open": float(np.mean(openings[5:7] if corner_a == 33 else openings[2:4])),
            "outer_open": float(np.mean(openings[2:4] if corner_a == 33 else openings[5:7])),
            "upper_curve": float(np.mean((upper_arc[2:7] - left) @ across) / eye_width),
            "lower_curve": float(np.mean((lower_arc[2:7] - left) @ across) / eye_width),
            "contour": np.vstack((upper_arc, lower_arc[-2:0:-1])),
        })
    else:
        result.update(inner_open=result["lid_open"], outer_open=result["lid_open"],
                      upper_curve=0.0, lower_curve=0.0, contour=np.array([left, upper_lid, right, lower_lid]))
    result["iris_ratio"] = iris_diameter / eye_width
    result["iris_depth"] = (points[iris].z - (points[corner_a].z + points[corner_b].z) / 2) * width / eye_width
    result["ring"] = np.array([pixel(i) for i in ring])
    return result


def _head_pose(matrix):
    """Head angles in degrees and head position in cm from the metric face transform."""
    matrix = np.asarray(matrix, dtype=float)
    rotation = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return (*map(math.degrees, (yaw, pitch, roll)), *matrix[:3, 3])


def _measure(result, rgb):
    points = result.face_landmarks[0]
    if len(points) < 478 or not result.face_blendshapes or not result.facial_transformation_matrixes:
        return None
    height, width = rgb.shape[:2]
    geometry = [_eye_geometry(points, width, height, eye, contour)
                for eye, contour in zip(EYES, EYE_CONTOURS)]
    if any(g is None for g in geometry):
        return None

    scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
    gaze_h = (
        scores["eyeLookInLeft"] + scores["eyeLookOutRight"]
        - scores["eyeLookOutLeft"] - scores["eyeLookInRight"]
    ) / 2
    gaze_v = (
        scores["eyeLookUpLeft"] + scores["eyeLookUpRight"]
        - scores["eyeLookDownLeft"] - scores["eyeLookDownRight"]
    ) / 2

    def mean(key):
        return float(np.mean([g[key] for g in geometry]))

    features = np.array(
        [
            mean("iris_x"), mean("iris_y"), gaze_h, gaze_v, mean("lid_open"), mean("iris_lid_y"),
            *_head_pose(result.facial_transformation_matrixes[0]),
            *[g[name] for g in reversed(geometry) for name in DETAIL_NAMES],
            geometry[1]["iris_x"] - geometry[0]["iris_x"],
            geometry[1]["iris_y"] - geometry[0]["iris_y"],
        ]
    )
    if not np.isfinite(features).all():
        return None

    xs = np.array([p.x for p in points]) * width
    ys = np.array([p.y for p in points]) * height
    x0, x1 = int(max(xs.min(), 0)), int(min(xs.max(), width))
    y0, y1 = int(max(ys.min(), 0)), int(min(ys.max(), height))
    face = rgb[y0:y1:4, x0:x1:4].astype(float)
    luma = float((face @ [0.299, 0.587, 0.114]).mean()) if face.size else 0.0
    # A transparent heuristic for usable image geometry, rather than an ML confidence claim.
    iris_quality = np.clip((mean("iris_px") - 3) / 9, 0, 1)
    light_quality = min(1.0, luma / 65, max(0.0, (255 - luma) / 35))
    opening_quality = np.clip(min(g["lid_open"] for g in geometry) / 0.12, 0, 1)
    in_frame = all(np.all((g["contour"][:, 0] >= 0) & (g["contour"][:, 0] < width)
                         & (g["contour"][:, 1] >= 0) & (g["contour"][:, 1] < height))
                   for g in geometry)
    quality = float(min(iris_quality, light_quality, opening_quality)) if in_frame else 0.0

    return Measurement(
        features=features,
        blink=(scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2,
        eye_open=np.array([g["lid_open"] for g in reversed(geometry)]),
        iris_px=mean("iris_px"),
        luma=luma,
        face_width=(x1 - x0) / width,
        quality=quality,
        eye_blinks=np.array([scores["eyeBlinkLeft"], scores["eyeBlinkRight"]]),
        eye_contours=tuple(g["contour"] for g in reversed(geometry)),
        iris_rings=tuple(g["ring"] for g in reversed(geometry)),
        timestamp=time.monotonic(),
        frame_size=(width, height),
    )


class FaceAnalyzer:
    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def analyze(self, rgb, timestamp_ms):
        """Return a Measurement for the face in an RGB frame, or None when there is none."""
        self._last_ms = max(timestamp_ms, self._last_ms + 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, self._last_ms)
        return _measure(result, rgb) if result.face_landmarks else None

    def close(self):
        self._landmarker.close()
