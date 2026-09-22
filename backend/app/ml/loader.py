"""Singleton model loader for EmotionAI backend.

The model and metadata are loaded once during application startup and
cached in module-level state. All inference requests share the same
loaded model — no per-request model loading overhead.

Design decisions:
- TensorFlow is imported lazily so the module can be imported without TF
  installed (e.g., in test environments that mock the loader).
- load_model() raises RuntimeError with a clear, actionable message if the
  model file is missing — no silent fallback.
- validate_model_shape() checks that the loaded model's input shape matches
  the metadata to catch mismatched model/metadata pairs early.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_lock = Lock()
_model: Any = None          # keras.Model once loaded
_metadata: dict[str, Any] = {}
_is_loaded: bool = False


def load_model(model_path: Path, metadata_path: Path) -> None:
    """Load model and metadata into module-level cache (thread-safe, idempotent)."""
    global _model, _metadata, _is_loaded  # noqa: PLW0603

    with _lock:
        if _is_loaded:
            return

        # ── Validate paths ────────────────────────────────────────────────────
        if not model_path.exists():
            raise RuntimeError(
                f"Model file not found: {model_path}\n"
                "Train the model first: python -m ml.training.train\n"
                "Then copy models/emotion_cnn.keras to the backend's model directory."
            )
        if not metadata_path.exists():
            raise RuntimeError(
                f"Metadata file not found: {metadata_path}\n"
                "Ensure models/metadata.json was written by the training script."
            )

        # ── Load metadata first ───────────────────────────────────────────────
        with metadata_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        required_meta_keys = ["num_classes", "index_to_class", "input"]
        missing = [k for k in required_meta_keys if k not in meta]
        if missing:
            raise RuntimeError(f"metadata.json is missing required keys: {missing}")

        # ── Load Keras model ──────────────────────────────────────────────────
        import tensorflow as tf  # noqa: PLC0415

        logger.info("Loading model from %s ...", model_path)
        model = tf.keras.models.load_model(str(model_path))
        logger.info("Model loaded: %s", model.name)

        # ── Shape validation ──────────────────────────────────────────────────
        expected_h = int(meta["input"]["height"])
        expected_w = int(meta["input"]["width"])
        expected_c = int(meta["input"]["channels"])
        expected_classes = int(meta["num_classes"])

        input_shape = tuple(model.input_shape[1:])   # (H, W, C)
        output_units = model.output_shape[-1]

        if input_shape != (expected_h, expected_w, expected_c):
            raise RuntimeError(
                f"Model input shape {input_shape} does not match metadata "
                f"({expected_h}, {expected_w}, {expected_c})"
            )
        if output_units != expected_classes:
            raise RuntimeError(
                f"Model output units {output_units} does not match metadata "
                f"num_classes={expected_classes}"
            )

        _model = model
        _metadata = meta
        _is_loaded = True
        logger.info(
            "Model ready | classes=%d | input=%s",
            expected_classes, input_shape,
        )


def get_model() -> Any:
    """Return the loaded Keras model. Raises if not yet loaded."""
    if not _is_loaded or _model is None:
        raise RuntimeError(
            "Model has not been loaded. Call load_model() during app startup."
        )
    return _model


def get_metadata() -> dict[str, Any]:
    """Return the loaded metadata dict. Raises if not yet loaded."""
    if not _is_loaded:
        raise RuntimeError("Model has not been loaded.")
    return _metadata


def is_loaded() -> bool:
    """Return True if the model has been successfully loaded."""
    return _is_loaded
