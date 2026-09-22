"""Shared path helpers and label config loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# EmotionAI repo root: ml/preprocessing/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_LABELS_PATH = REPO_ROOT / "shared" / "labels.json"
DEFAULT_TRAIN_DIR = REPO_ROOT / "train"
DEFAULT_TEST_DIR = REPO_ROOT / "test"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "data" / "reports"


def load_labels_config(path: Path | None = None) -> dict[str, Any]:
    """Load the single source of truth for class mapping and image assumptions."""
    config_path = path or SHARED_LABELS_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Labels config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        data = json.load(f)
    required = ["class_to_index", "index_to_class", "num_classes", "image"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"labels.json missing keys: {missing}")
    if int(data["num_classes"]) != len(data["class_to_index"]):
        raise ValueError("num_classes does not match class_to_index length")
    return data


def ensure_output_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
