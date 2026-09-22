"""Validate FER2013 folder layout and image integrity."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ml.preprocessing.config import load_labels_config

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


@dataclass
class ValidationIssue:
    path: str
    reason: str


@dataclass
class SplitValidationReport:
    split_name: str
    root: str
    class_counts: dict[str, int] = field(default_factory=dict)
    excluded_class_counts: dict[str, int] = field(default_factory=dict)
    valid_images: int = 0
    removed_images: int = 0
    sampled_checked: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "root": self.root,
            "class_counts": self.class_counts,
            "excluded_class_counts": self.excluded_class_counts,
            "valid_images": self.valid_images,
            "removed_images": self.removed_images,
            "sampled_checked": self.sampled_checked,
            "issues": [{"path": i.path, "reason": i.reason} for i in self.issues],
        }


def _expected_size(labels: dict[str, Any]) -> tuple[int, int]:
    img = labels["image"]
    return int(img["width"]), int(img["height"])


def validate_image_file(
    path: Path,
    expected_size: tuple[int, int],
    expect_grayscale: bool = True,
    deep: bool = False,
) -> str | None:
    """Return an error reason string, or None if the image is valid.

    By default, only header metadata is checked (fast). Set deep=True to fully
    decode pixels when hunting corrupted files.
    """
    try:
        with Image.open(path) as im:
            if im.size != expected_size:
                return f"unexpected_size:{im.size}"
            if expect_grayscale and im.mode not in {"L", "LA", "RGB", "RGBA", "P"}:
                return f"unsupported_mode:{im.mode}"
            if deep:
                im.load()
    except UnidentifiedImageError:
        return "unidentified_image"
    except OSError as exc:
        return f"os_error:{exc}"
    return None


def list_image_files(class_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_folder_dataset(
    root: Path,
    split_name: str,
    labels_config: dict[str, Any] | None = None,
    max_issues: int = 500,
    deep: bool = False,
    sample_per_class: int | None = 100,
    random_state: int = 42,
) -> SplitValidationReport:
    """
    Validate a FER2013-style class-folder dataset.

    Keeps only classes listed in shared labels config.
    By default, counts all files and opens a stratified sample per class
    (fast). Set sample_per_class=None to open every image.
    """
    labels = labels_config or load_labels_config()
    source_map: dict[str, str] = {
        k.lower(): v for k, v in labels["source_folder_names"].items()
    }
    excluded = {c.lower() for c in labels.get("excluded_source_classes", [])}
    expected_size = _expected_size(labels)
    rng = random.Random(random_state)

    report = SplitValidationReport(split_name=split_name, root=str(root.resolve()))
    if not root.exists():
        report.issues.append(ValidationIssue(str(root), "missing_directory"))
        return report

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        folder_key = class_dir.name.lower()
        files = list_image_files(class_dir)

        if folder_key in excluded:
            report.excluded_class_counts[folder_key] = len(files)
            continue

        if folder_key not in source_map:
            report.issues.append(
                ValidationIssue(str(class_dir), f"unknown_class_folder:{folder_key}")
            )
            continue

        label_name = source_map[folder_key]
        if sample_per_class is None or sample_per_class >= len(files):
            to_check = files
        else:
            to_check = rng.sample(files, sample_per_class)

        failed_paths: set[str] = set()
        for path in to_check:
            reason = validate_image_file(path, expected_size, deep=deep)
            report.sampled_checked += 1
            if reason is not None:
                failed_paths.add(str(path))
                report.removed_images += 1
                if len(report.issues) < max_issues:
                    report.issues.append(ValidationIssue(str(path), reason))
                logger.warning("Removed %s (%s)", path, reason)

        # Count all files as valid except those that failed the sample check.
        # Full corrupt detection requires sample_per_class=None.
        kept = len(files) - len(failed_paths)
        report.class_counts[label_name] = kept
        report.valid_images += kept

    return report
