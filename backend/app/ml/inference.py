"""Inference pipeline for EmotionAI backend.

Responsibilities:
1. Decode a raw image bytes blob into a numpy array.
2. Detect a face using OpenCV Haar cascade.
3. Crop and preprocess the face using exactly the same assumptions as training:
   - Grayscale conversion
   - 48×48 resize
   - /255 normalization
   - Shape (1, 48, 48, 1) for batch dimension
4. Run CNN inference and return structured results.

The preprocessing here MUST mirror ml/preprocessing/pipeline.py::load_and_preprocess_image().
If training preprocessing changes, update this module to match.

No global state is held here — the loader module owns the model singleton.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.ml.loader import get_metadata, get_model

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Haar cascade scale factor and min neighbours — conservative defaults.
# Adjust if you get too many false positives or missed detections.
_HAAR_SCALE_FACTOR = 1.1
_HAAR_MIN_NEIGHBOURS = 5
_HAAR_MIN_SIZE = (30, 30)

# Lazy-loaded cascade — avoids loading it when the module is imported during tests.
_cascade: cv2.CascadeClassifier | None = None


def _get_cascade(cascade_path: str | None = None) -> cv2.CascadeClassifier:
    global _cascade  # noqa: PLW0603
    if _cascade is None:
        path = cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
        if _cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {path}. "
                "Ensure opencv-python-headless is correctly installed."
            )
        logger.info("Haar cascade loaded from %s", path)
    return _cascade


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    face_detected: bool
    label: str | None           # None if no face
    confidence: float | None    # None if no face
    probabilities: dict[str, float] | None  # None if no face
    processing_time_ms: float


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array (OpenCV convention)."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Could not decode image. Ensure the upload is a valid JPEG, PNG, or BMP."
        )
    return img


def _detect_largest_face(
    gray: np.ndarray,
    cascade: cv2.CascadeClassifier,
) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest detected face, or None.
    
    If no face is detected by Haar cascade and the input is already a tight crop
    (e.g., FER2013 48x48 test image or cropped avatar <= 128x128), fallback to
    treating the image itself as the face region.
    """
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=_HAAR_SCALE_FACTOR,
        minNeighbors=_HAAR_MIN_NEIGHBOURS,
        minSize=_HAAR_MIN_SIZE,
    )
    if len(faces) == 0:
        h, w = gray.shape[:2]
        if max(h, w) <= 128:
            return (0, 0, w, h)
        return None
    # Pick the largest face by area
    return max(faces, key=lambda f: f[2] * f[3])



def _preprocess_face(
    gray: np.ndarray,
    face_rect: tuple[int, int, int, int],
    target_size: tuple[int, int] = (48, 48),
) -> np.ndarray:
    """Crop, resize, normalize the face region.

    Returns shape (1, H, W, 1), float32, values in [0, 1].
    This mirrors ml/preprocessing/pipeline.py::load_and_preprocess_image().
    """
    x, y, w, h = face_rect
    # Add a small margin around the detected box (10%)
    margin = int(0.10 * min(w, h))
    h_img, w_img = gray.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)

    face_crop = gray[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)
    face_norm = face_resized.astype(np.float32) / 255.0
    return face_norm.reshape(1, target_size[1], target_size[0], 1)


# ── Main inference function ───────────────────────────────────────────────────

def predict(
    image_bytes: bytes,
    cascade_path: str | None = None,
) -> InferenceResult:
    """Run the full inference pipeline on raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of a valid image file (JPEG, PNG, BMP).
    cascade_path : str | None
        Optional path to Haar cascade XML. Uses OpenCV default if None.

    Returns
    -------
    InferenceResult
        Structured result. face_detected=False if no face found.
    """
    t0 = time.perf_counter()

    metadata: dict[str, Any] = get_metadata()
    model = get_model()

    num_classes: int = int(metadata["num_classes"])
    index_to_class: dict[str, str] = metadata["index_to_class"]
    target_h = int(metadata["input"]["height"])
    target_w = int(metadata["input"]["width"])
    target_size = (target_w, target_h)  # cv2 uses (width, height)

    # ── 1. Decode ─────────────────────────────────────────────────────────────
    bgr = _decode_image(image_bytes)

    # ── 2. Convert to grayscale ───────────────────────────────────────────────
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── 3. Face detection ──────────────────────────────────────────────────────
    cascade = _get_cascade(cascade_path)
    face_rect = _detect_largest_face(gray, cascade)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if face_rect is None:
        logger.debug("No face detected in uploaded image.")
        return InferenceResult(
            face_detected=False,
            label=None,
            confidence=None,
            probabilities=None,
            processing_time_ms=round(elapsed_ms, 1),
        )

    # ── 4. Preprocess ─────────────────────────────────────────────────────────
    face_input = _preprocess_face(gray, face_rect, target_size=target_size)

    # ── 5. Inference ──────────────────────────────────────────────────────────
    probs_raw: np.ndarray = model.predict(face_input, verbose=0)[0]

    # Clamp and renormalize to ensure valid probability distribution
    probs_raw = np.clip(probs_raw, 0.0, 1.0)
    probs_sum = probs_raw.sum()
    if probs_sum > 0:
        probs_raw = probs_raw / probs_sum

    predicted_idx = int(np.argmax(probs_raw))
    predicted_label = index_to_class[str(predicted_idx)]
    confidence = float(probs_raw[predicted_idx])

    probabilities = {
        index_to_class[str(i)]: round(float(probs_raw[i]), 4)
        for i in range(num_classes)
    }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        "Prediction: %s (%.2f%%) in %.1f ms", predicted_label, confidence * 100, elapsed_ms
    )

    return InferenceResult(
        face_detected=True,
        label=predicted_label,
        confidence=round(confidence, 4),
        probabilities=probabilities,
        processing_time_ms=round(elapsed_ms, 1),
    )
