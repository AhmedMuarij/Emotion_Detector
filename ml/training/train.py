"""Training script for EmotionAI CNN.

Usage (from repo root with venv active):
    python -m ml.training.train

Prerequisites:
    1. Run `python -m ml.preprocessing.pipeline` first to generate:
       - data/processed/X_train.npy, y_train.npy
       - data/processed/X_val.npy,   y_val.npy
       - data/processed/X_test.npy,  y_test.npy
    2. TensorFlow installed: pip install tensorflow>=2.15 (or tensorflow-cpu)

Outputs (written to models/):
    - emotion_cnn.keras           — saved model (best val_accuracy checkpoint)
    - metadata.json               — runtime metadata for backend

Training report written to:
    - data/reports/training_history.json

Design notes:
    - Class weights compensate for Happy/Surprise imbalance (~2.3:1 ratio).
    - ImageDataGenerator provides mild augmentation (h-flip, rotation, zoom).
    - ReduceLROnPlateau + EarlyStopping prevent overfitting.
    - ModelCheckpoint saves best val_accuracy checkpoint, not last epoch.
    - Test-set evaluation runs only once at the end (never touches training loop).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Repo root relative to this file: ml/training/train.py → parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "data" / "reports"
MODELS_DIR = REPO_ROOT / "models"
SHARED_LABELS = REPO_ROOT / "shared" / "labels.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_labels() -> dict[str, Any]:
    with SHARED_LABELS.open(encoding="utf-8") as f:
        return json.load(f)


def _load_arrays(
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed .npy arrays. Raises FileNotFoundError if missing."""
    needed = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [n for n in needed if not (processed_dir / f"{n}.npy").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing arrays in {processed_dir}: {missing}\n"
            "Run: python -m ml.preprocessing.pipeline"
        )
    return tuple(np.load(processed_dir / f"{n}.npy") for n in needed)  # type: ignore[return-value]


def _compute_class_weights(y_train: np.ndarray, num_classes: int) -> dict[int, float]:
    """Balanced class weights using sklearn's formula."""
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train,
    )
    return {i: float(w) for i, w in enumerate(weights)}


def _build_augmentation_generator():
    """Mild augmentation: h-flip, small rotation/zoom. No shear (distorts faces)."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode="nearest",
    )


def _save_metadata(
    labels: dict[str, Any],
    history_metrics: dict[str, Any],
    test_metrics: dict[str, float],
    training_params: dict[str, Any],
    model_path: Path,
) -> Path:
    """Write models/metadata.json with measured (not fabricated) metrics."""
    metadata = {
        "model_name": "emotion_cnn",
        "model_version": labels.get("model_version", "0.1.0"),
        "model_file": model_path.name,
        "framework": "tensorflow.keras",
        "task": "facial_expression_recognition",
        "disclaimer": (
            "Classifies visible facial expressions. "
            "Does not determine internal emotional state."
        ),
        "input": {
            "height": labels["image"]["height"],
            "width": labels["image"]["width"],
            "channels": labels["image"]["channels"],
            "color_mode": labels["image"]["color_mode"],
            "normalization": labels["image"]["normalization"],
            "dtype": "float32",
        },
        "num_classes": labels["num_classes"],
        "class_to_index": labels["class_to_index"],
        "index_to_class": labels["index_to_class"],
        "training": {
            "dataset": "FER2013",
            "excluded_classes": list(labels.get("excluded_source_classes", [])),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            **training_params,
        },
        "evaluation": {
            "test_accuracy": test_metrics.get("test_accuracy"),
            "test_loss": test_metrics.get("test_loss"),
            "best_val_accuracy": history_metrics.get("best_val_accuracy"),
            "best_val_loss": history_metrics.get("best_val_loss"),
            "epochs_trained": history_metrics.get("epochs_trained"),
            "note": "Metrics measured on held-out test set. Do not fabricate values.",
        },
    }
    meta_path = MODELS_DIR / "metadata.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata → %s", meta_path)
    return meta_path


# ── Main training function ────────────────────────────────────────────────────

def train(
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the EmotionAI CNN and export the best model.

    Returns a summary dict with paths and final metrics.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    # ── Late TF import (avoid slow startup if only importing train module) ────
    import tensorflow as tf  # noqa: PLC0415

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    logger.info("TensorFlow %s | GPU: %s", tf.__version__,
                tf.config.list_physical_devices("GPU") or "none (CPU mode)")

    # ── Load data ─────────────────────────────────────────────────────────────
    labels = _load_labels()
    num_classes = int(labels["num_classes"])

    X_train, y_train, X_val, y_val, X_test, y_test = _load_arrays(PROCESSED_DIR)
    logger.info(
        "Loaded arrays | train=%s val=%s test=%s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = _compute_class_weights(y_train, num_classes)
    logger.info("Class weights: %s", class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    from ml.training.model import build_emotion_cnn  # noqa: PLC0415

    model = build_emotion_cnn(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "emotion_cnn.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Augmentation generator ────────────────────────────────────────────────
    datagen = _build_augmentation_generator()
    datagen.fit(X_train)

    train_gen = datagen.flow(
        X_train, y_train, batch_size=batch_size, seed=random_state
    )
    steps_per_epoch = len(X_train) // batch_size

    # ── Training ──────────────────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info("Training complete in %.1f seconds", elapsed)

    # ── History report ────────────────────────────────────────────────────────
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch = int(np.argmax(hist_dict["val_accuracy"]))
    history_metrics = {
        "best_val_accuracy": hist_dict["val_accuracy"][best_epoch],
        "best_val_loss": hist_dict["val_loss"][best_epoch],
        "epochs_trained": len(hist_dict["val_accuracy"]),
        "best_epoch": best_epoch + 1,
        "training_seconds": round(elapsed, 1),
    }
    history_path = REPORTS_DIR / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump({"history": hist_dict, "summary": history_metrics}, f, indent=2)
    logger.info("Training history → %s", history_path)

    # ── Load best checkpoint for test evaluation ──────────────────────────────
    best_model = tf.keras.models.load_model(str(checkpoint_path))
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    test_metrics = {"test_accuracy": float(test_acc), "test_loss": float(test_loss)}
    logger.info("Test accuracy: %.4f | Test loss: %.4f", test_acc, test_loss)

    # ── Save metadata ─────────────────────────────────────────────────────────
    training_params = {
        "epochs_max": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "early_stopping_patience": patience,
        "random_state": random_state,
        "augmentation": "h-flip, rotation±10°, zoom±10%, shift±5%",
        "class_weight_strategy": "balanced",
    }
    meta_path = _save_metadata(
        labels=labels,
        history_metrics=history_metrics,
        test_metrics=test_metrics,
        training_params=training_params,
        model_path=checkpoint_path,
    )

    summary = {
        "model_path": str(checkpoint_path),
        "metadata_path": str(meta_path),
        "history_path": str(history_path),
        **test_metrics,
        **history_metrics,
    }
    logger.info("Summary: %s", summary)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EmotionAI CNN on preprocessed FER2013 arrays."
    )
    parser.add_argument("--epochs", type=int, default=60,
                        help="Max training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial Adam learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10,
                        help="EarlyStopping patience (default: 10)")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
