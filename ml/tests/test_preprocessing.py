"""Unit tests for Phase 1 preprocessing (no fabricated metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml.preprocessing.config import REPO_ROOT, load_labels_config
from ml.preprocessing.pipeline import (
    build_manifests,
    load_and_preprocess_image,
)
from ml.preprocessing.validate import validate_folder_dataset, validate_image_file


@pytest.fixture(scope="module")
def labels():
    return load_labels_config()


def test_labels_config_five_classes(labels):
    assert labels["num_classes"] == 5
    assert labels["index_to_class"] == {
        "0": "Angry",
        "1": "Happy",
        "2": "Sad",
        "3": "Surprise",
        "4": "Neutral",
    }
    assert set(labels["class_to_index"].keys()) == {
        "Angry",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    }


def test_train_test_dirs_exist():
    assert (REPO_ROOT / "train").is_dir()
    assert (REPO_ROOT / "test").is_dir()


def test_sample_image_is_48x48_grayscale(labels):
    sample = next((REPO_ROOT / "train" / "happy").glob("*.jpg"))
    reason = validate_image_file(
        sample,
        (labels["image"]["width"], labels["image"]["height"]),
    )
    assert reason is None
    arr = load_and_preprocess_image(sample, labels)
    assert arr.shape == (48, 48, 1)
    assert arr.dtype == np.float32
    assert 0.0 <= float(arr.min()) <= float(arr.max()) <= 1.0


def _make_mini_fer_tree(tmp_path: Path, labels: dict, train_n: int = 20, test_n: int = 10):
    """Tiny synthetic FER-style folders for fast unit tests."""
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    for split_root, n in ((train_root, train_n), (test_root, test_n)):
        for folder in labels["source_folder_names"]:
            d = split_root / folder
            d.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(
                    np.full((48, 48), i % 255, dtype=np.uint8), mode="L"
                ).save(d / f"{folder}_{i}.png")
        for excluded in labels["excluded_source_classes"]:
            d = split_root / excluded
            d.mkdir(parents=True)
            Image.fromarray(np.zeros((48, 48), dtype=np.uint8), mode="L").save(
                d / "x.png"
            )
    return train_root, test_root


def test_validate_folder_excludes_disgust_fear(labels, tmp_path):
    train_root, _ = _make_mini_fer_tree(tmp_path, labels, train_n=3, test_n=1)
    report = validate_folder_dataset(train_root, "train", labels)
    assert "Disgust" not in report.class_counts
    assert "Fear" not in report.class_counts
    assert "disgust" in report.excluded_class_counts
    assert "fear" in report.excluded_class_counts
    assert set(report.class_counts.keys()) == {
        "Angry",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    }
    assert report.valid_images == 15  # 5 classes * 3 images


def test_build_manifests_no_leakage_and_stratified(labels, tmp_path):
    train_root, test_root = _make_mini_fer_tree(tmp_path, labels)

    manifest, removed = build_manifests(
        train_dir=train_root,
        test_dir=test_root,
        labels=labels,
        val_ratio=0.2,
        random_state=0,
    )
    assert removed == []
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert manifest["path"].is_unique
    assert set(manifest["label"]) == set(labels["class_to_index"].keys())
    for _, row in manifest.iterrows():
        assert labels["class_to_index"][row["label"]] == row["class_index"]


@pytest.mark.integration
def test_real_train_folder_has_five_kept_classes(labels):
    """Optional: light check against real data (folder names only, no full image scan)."""
    train = REPO_ROOT / "train"
    if not train.is_dir():
        pytest.skip("train/ not present")
    kept = set(labels["source_folder_names"].keys())
    present = {p.name.lower() for p in train.iterdir() if p.is_dir()}
    assert kept.issubset(present)
    assert {"disgust", "fear"}.issubset(present)

def test_shared_labels_json_is_valid_json():
    path = REPO_ROOT / "shared" / "labels.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["num_classes"] == 5
