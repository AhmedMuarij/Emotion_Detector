"""Post-training evaluation for EmotionAI CNN.

Usage (from repo root with venv active):
    python -m ml.evaluation.evaluate

Prerequisites:
    - models/emotion_cnn.keras must exist (run train.py first)
    - data/processed/X_test.npy and y_test.npy must exist (run pipeline.py first)

Outputs (written to data/reports/):
    - confusion_matrix.png
    - training_curves.png       (if training_history.json exists)
    - classification_report.txt
    - evaluation_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "data" / "reports"
MODELS_DIR = REPO_ROOT / "models"
SHARED_LABELS = REPO_ROOT / "shared" / "labels.json"


def _load_labels() -> dict[str, Any]:
    with SHARED_LABELS.open(encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Render and save a labelled confusion matrix using Matplotlib only."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Normalized frequency")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Expression", fontsize=12)
    ax.set_ylabel("True Expression", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized)\nEmotionAI CNN — Test Set", fontsize=13)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > thresh else "black"
            raw = cm[i, j]
            ax.text(
                j, i,
                f"{cm_norm[i,j]:.2f}\n({raw})",
                ha="center", va="center",
                fontsize=9, color=color,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix → %s", output_path)


def plot_training_curves(
    history_path: Path,
    output_path: Path,
) -> None:
    """Plot loss + accuracy curves from saved training_history.json."""
    import matplotlib.pyplot as plt

    if not history_path.exists():
        logger.warning("training_history.json not found, skipping curve plot.")
        return

    with history_path.open(encoding="utf-8") as f:
        data = json.load(f)
    hist = data["history"]

    epochs = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs, hist["loss"], "b-o", ms=4, label="Train loss")
    ax1.plot(epochs, hist["val_loss"], "r-o", ms=4, label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Sparse Categorical Cross-Entropy")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, hist["accuracy"], "b-o", ms=4, label="Train accuracy")
    ax2.plot(epochs, hist["val_accuracy"], "r-o", ms=4, label="Val accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("EmotionAI CNN — Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves → %s", output_path)


def evaluate(
    model_path: Path | None = None,
    processed_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Load best model and evaluate on the held-out test set."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    import tensorflow as tf  # noqa: PLC0415
    from sklearn.metrics import classification_report  # noqa: PLC0415

    model_path = model_path or MODELS_DIR / "emotion_cnn.keras"
    p_dir = processed_dir or PROCESSED_DIR
    r_dir = reports_dir or REPORTS_DIR
    r_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: python -m ml.training.train"
        )

    labels = _load_labels()
    num_classes = int(labels["num_classes"])
    class_names: list[str] = [
        labels["index_to_class"][str(i)] for i in range(num_classes)
    ]

    # Load test arrays
    X_test = np.load(p_dir / "X_test.npy")
    y_test = np.load(p_dir / "y_test.npy")
    logger.info("Test set: X%s y%s", X_test.shape, y_test.shape)

    # Load model
    logger.info("Loading model: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded successfully")

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    logger.info("Test accuracy: %.4f | Test loss: %.4f", test_acc, test_loss)

    # Predictions
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Classification report
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, digits=4
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT — TEST SET")
    print("=" * 60)
    print(report_str)

    report_path = r_dir / "classification_report.txt"
    report_path.write_text(report_str, encoding="utf-8")
    logger.info("Classification report → %s", report_path)

    # Confusion matrix plot
    cm_path = r_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path)

    # Training curves (if history exists)
    history_path = r_dir / "training_history.json"
    curves_path = r_dir / "training_curves.png"
    plot_training_curves(history_path, curves_path)

    # Per-class accuracy breakdown
    per_class: dict[str, float] = {}
    for idx, name in enumerate(class_names):
        mask = y_test == idx
        if mask.sum() > 0:
            per_class[name] = float((y_pred[mask] == idx).mean())

    summary: dict[str, Any] = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "per_class_accuracy": per_class,
        "num_test_samples": int(len(y_test)),
        "model_path": str(model_path),
        "confusion_matrix_plot": str(cm_path),
        "training_curves_plot": str(curves_path) if history_path.exists() else None,
        "classification_report_path": str(report_path),
        "note": (
            "These metrics reflect dataset test-set performance. "
            "Real-world webcam accuracy will differ due to lighting, angle, and domain shift."
        ),
    }

    summary_path = r_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evaluation summary → %s", summary_path)

    return summary


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
