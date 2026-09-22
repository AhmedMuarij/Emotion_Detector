"""Build stratified manifests and preprocess FER2013 images for training."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from ml.preprocessing.config import (
    DEFAULT_TEST_DIR,
    DEFAULT_TRAIN_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
    load_labels_config,
)
from ml.preprocessing.validate import (
    SUPPORTED_EXTENSIONS,
    validate_folder_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class ManifestRow:
    path: str
    label: str
    class_index: int
    split: str


def _collect_rows(
    root: Path,
    split_name: str,
    labels: dict[str, Any],
) -> tuple[list[ManifestRow], list[dict[str, str]]]:
    source_map = {k.lower(): v for k, v in labels["source_folder_names"].items()}
    excluded = {c.lower() for c in labels.get("excluded_source_classes", [])}
    class_to_index = labels["class_to_index"]

    rows: list[ManifestRow] = []
    removed: list[dict[str, str]] = []

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        folder_key = class_dir.name.lower()
        if folder_key in excluded:
            continue
        if folder_key not in source_map:
            logger.warning("Skipping unknown class folder: %s", class_dir)
            continue

        label = source_map[folder_key]
        class_index = int(class_to_index[label])
        files = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in files:
            if path.stat().st_size <= 0:
                removed.append(
                    {"path": str(path), "reason": "empty_file", "split": split_name}
                )
                continue
            rows.append(
                ManifestRow(
                    path=str(path.resolve()),
                    label=label,
                    class_index=class_index,
                    split=split_name,
                )
            )
    return rows, removed


def build_manifests(
    train_dir: Path,
    test_dir: Path,
    labels: dict[str, Any],
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """
    Create train/val/test manifests from folder datasets.

    - Train folder is stratified into train + validation (no leakage into test).
    - Existing test folder remains the held-out test set.
    """
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio must be between 0 and 0.5")

    train_rows, removed = _collect_rows(train_dir, "train_source", labels)
    test_rows, removed_test = _collect_rows(test_dir, "test", labels)
    removed.extend(removed_test)

    if not train_rows:
        raise RuntimeError(f"No valid training images found under {train_dir}")
    if not test_rows:
        raise RuntimeError(f"No valid test images found under {test_dir}")

    train_df = pd.DataFrame([asdict(r) for r in train_rows])
    test_df = pd.DataFrame([asdict(r) for r in test_rows])

    train_part, val_part = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_df["class_index"],
    )
    train_part = train_part.copy()
    val_part = val_part.copy()
    train_part["split"] = "train"
    val_part["split"] = "val"
    test_df = test_df.copy()
    test_df["split"] = "test"

    # Leakage guard: identical absolute paths must not appear across splits.
    all_parts = pd.concat([train_part, val_part, test_df], ignore_index=True)
    dupes = all_parts["path"].duplicated(keep=False)
    if dupes.any():
        leaked = all_parts.loc[dupes, ["path", "split"]].sort_values("path")
        raise RuntimeError(
            "Potential train/val/test leakage detected (duplicate paths):\n"
            + leaked.head(20).to_string(index=False)
        )

    return all_parts, removed


def load_and_preprocess_image(path: Path | str, labels: dict[str, Any]) -> np.ndarray:
    """Match training/inference assumptions: grayscale, resize, /255 float32."""
    img_cfg = labels["image"]
    target = (int(img_cfg["width"]), int(img_cfg["height"]))
    with Image.open(path) as im:
        im = im.convert("L")
        if im.size != target:
            im = im.resize(target, Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    if img_cfg.get("normalization") == "divide_by_255":
        arr = arr / 255.0
    else:
        raise ValueError(f"Unsupported normalization: {img_cfg.get('normalization')}")
    return arr.reshape(target[1], target[0], 1)


def materialize_arrays(
    manifest: pd.DataFrame,
    labels: dict[str, Any],
    splits: Iterable[str] = ("train", "val", "test"),
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, str]]]:
    """Load images referenced by the manifest into X/y arrays per split."""
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    load_failures: list[dict[str, str]] = []
    for split in splits:
        subset = manifest[manifest["split"] == split]
        if subset.empty:
            raise RuntimeError(f"No rows for split={split}")
        images: list[np.ndarray] = []
        labels_idx: list[int] = []
        for path, class_index in zip(
            subset["path"].tolist(), subset["class_index"].tolist(), strict=True
        ):
            try:
                images.append(load_and_preprocess_image(path, labels))
                labels_idx.append(int(class_index))
            except Exception as exc:  # noqa: BLE001 - record and continue
                load_failures.append(
                    {"path": str(path), "reason": f"load_error:{exc}", "split": split}
                )
                logger.warning("Failed to load %s (%s)", path, exc)
        if not images:
            raise RuntimeError(f"All images failed to load for split={split}")
        x = np.stack(images, axis=0)
        y = np.asarray(labels_idx, dtype=np.int64)
        outputs[split] = (x, y)
        logger.info("Materialized %s: X%s y%s", split, x.shape, y.shape)
    return outputs, load_failures


def save_processed_artifacts(
    manifest: pd.DataFrame,
    removed: list[dict[str, str]],
    labels: dict[str, Any],
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    write_arrays: bool,
) -> dict[str, Any]:
    ensure_output_dirs()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    manifest_path = PROCESSED_DIR / "manifest.csv"
    removed_path = REPORTS_DIR / "removed_records.json"
    summary_path = REPORTS_DIR / f"preprocess_summary_{timestamp}.json"
    latest_summary = REPORTS_DIR / "preprocess_summary_latest.json"
    labels_copy = PROCESSED_DIR / "labels.json"

    manifest.to_csv(manifest_path, index=False)
    with removed_path.open("w", encoding="utf-8") as f:
        json.dump(removed, f, indent=2)
    with labels_copy.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    array_info: dict[str, Any] = {}
    if write_arrays and arrays is not None:
        for split, (x, y) in arrays.items():
            x_path = PROCESSED_DIR / f"X_{split}.npy"
            y_path = PROCESSED_DIR / f"y_{split}.npy"
            np.save(x_path, x)
            np.save(y_path, y)
            array_info[split] = {
                "X_path": str(x_path),
                "y_path": str(y_path),
                "X_shape": list(x.shape),
                "y_shape": list(y.shape),
                "class_distribution": {
                    labels["index_to_class"][str(i)]: int((y == i).sum())
                    for i in range(labels["num_classes"])
                },
            }

    summary = {
        "created_at_utc": timestamp,
        "manifest_path": str(manifest_path),
        "removed_records_path": str(removed_path),
        "removed_count": len(removed),
        "counts_by_split": manifest.groupby("split").size().to_dict(),
        "counts_by_split_and_label": (
            manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
        ),
        "labels_config": labels,
        "arrays": array_info,
        "write_arrays": write_arrays,
    }
    for path in (summary_path, latest_summary):
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def run_preprocessing(
    train_dir: Path = DEFAULT_TRAIN_DIR,
    test_dir: Path = DEFAULT_TEST_DIR,
    val_ratio: float = 0.15,
    random_state: int = 42,
    write_arrays: bool = True,
    validate_only: bool = False,
    deep_validate: bool = False,
    sample_per_class: int | None = 100,
) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    labels = load_labels_config()
    ensure_output_dirs()

    if validate_only:
        sample = None if deep_validate else sample_per_class
        train_report = validate_folder_dataset(
            train_dir,
            "train",
            labels,
            deep=deep_validate,
            sample_per_class=sample,
            random_state=random_state,
        )
        test_report = validate_folder_dataset(
            test_dir,
            "test",
            labels,
            deep=deep_validate,
            sample_per_class=sample,
            random_state=random_state,
        )
        validation_path = REPORTS_DIR / "dataset_validation.json"
        with validation_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"train": train_report.to_dict(), "test": test_report.to_dict()},
                f,
                indent=2,
            )
        logger.info(
            "Validation complete | train_valid=%s test_valid=%s removed=%s sampled=%s",
            train_report.valid_images,
            test_report.valid_images,
            train_report.removed_images + test_report.removed_images,
            train_report.sampled_checked + test_report.sampled_checked,
        )
        return {
            "validate_only": True,
            "validation_report": str(validation_path),
            "train": train_report.to_dict(),
            "test": test_report.to_dict(),
        }

    manifest, removed = build_manifests(
        train_dir=train_dir,
        test_dir=test_dir,
        labels=labels,
        val_ratio=val_ratio,
        random_state=random_state,
    )
    arrays = None
    if write_arrays:
        arrays, load_failures = materialize_arrays(manifest, labels)
        removed.extend(load_failures)
        if load_failures:
            failed_paths = {item["path"] for item in load_failures}
            manifest = manifest[~manifest["path"].isin(failed_paths)].reset_index(
                drop=True
            )
    summary = save_processed_artifacts(
        manifest=manifest,
        removed=removed,
        labels=labels,
        arrays=arrays,
        write_arrays=write_arrays,
    )
    logger.info(
        "Preprocessing complete | removed=%s | splits=%s",
        len(removed),
        summary["counts_by_split"],
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and preprocess FER2013 folder dataset for EmotionAI."
    )
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write manifests/reports without materializing .npy arrays.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate dataset integrity; do not write processed arrays.",
    )
    parser.add_argument(
        "--deep-validate",
        action="store_true",
        help="Check every image (and fully decode pixels). Slower.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=100,
        help="Images opened per class during --validate-only (ignored with --deep-validate).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        write_arrays=not args.manifest_only,
        validate_only=args.validate_only,
        deep_validate=args.deep_validate,
        sample_per_class=args.sample_per_class,
    )


if __name__ == "__main__":
    main()
