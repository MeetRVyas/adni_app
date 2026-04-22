"""
quality_checks.py
=================
Data quality validation for the BDA pipeline.

Checks performed:
  1. File existence  — image_path actually exists on disk
  2. File readability — image can be opened (not corrupt)
  3. Schema conformance — all required columns present and non-null
  4. Label consistency — label ↔ label_name mapping is bijective
  5. Split coverage — train/val/test all present for every fold
  6. Class balance warning — alerts when any class < 1% of total

Usage
-----
    from data_pipeline.validation.quality_checks import run_quality_checks

    issues = run_quality_checks(df, disease="adni", max_workers=8)
    if issues["errors"]:
        raise RuntimeError("Quality check failed")
"""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

from data_pipeline.registry.schema import CORE_FIELDS, VALID_SPLITS


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def check_file_existence(df: pd.DataFrame) -> List[str]:
    """Return list of image_path values that do not exist on disk."""
    missing = df["image_path"][~df["image_path"].apply(os.path.exists)]
    return missing.tolist()


def _try_open(path: str) -> Optional[str]:
    """Return path if corrupt, None if OK."""
    try:
        with Image.open(path) as img:
            img.verify()
        return None
    except Exception as e:
        return f"{path}: {e}"


def check_corrupt_images(
    df: pd.DataFrame,
    max_workers: int = 4,
    sample_size: Optional[int] = None,
) -> List[str]:
    """
    Try to open each image; return list of corrupt file paths.

    Parameters
    ----------
    sample_size : int | None
        If set, only check a random sample (faster for large datasets).
    """
    paths = df["image_path"].tolist()
    if sample_size and len(paths) > sample_size:
        import random
        paths = random.sample(paths, sample_size)

    corrupt = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_try_open, paths):
            if result is not None:
                corrupt.append(result)

    return corrupt


def check_schema_conformance(df: pd.DataFrame) -> List[str]:
    """Check that all required columns are present and non-null."""
    errors = []
    for field in CORE_FIELDS:
        if field.name not in df.columns:
            errors.append(f"Missing column: {field.name}")
        elif not field.nullable and df[field.name].isna().any():
            n = df[field.name].isna().sum()
            errors.append(f"Null values in non-nullable column '{field.name}': {n} rows")
    return errors


def check_label_consistency(df: pd.DataFrame) -> List[str]:
    """
    Ensure label ↔ label_name mapping is bijective:
      • Each label index maps to exactly one name
      • Each name maps to exactly one index
    """
    errors = []
    pairs = df[["label", "label_name"]].drop_duplicates()

    # label → multiple names?
    dupes_label = pairs.groupby("label")["label_name"].nunique()
    for lbl, n in dupes_label[dupes_label > 1].items():
        names = pairs.loc[pairs["label"] == lbl, "label_name"].unique().tolist()
        errors.append(f"Label {lbl} maps to multiple names: {names}")

    # name → multiple labels?
    dupes_name = pairs.groupby("label_name")["label"].nunique()
    for name, n in dupes_name[dupes_name > 1].items():
        labels = pairs.loc[pairs["label_name"] == name, "label"].unique().tolist()
        errors.append(f"Label name '{name}' maps to multiple indices: {labels}")

    return errors


def check_split_coverage(df: pd.DataFrame) -> List[str]:
    """Warn if train/val/test are all present for all folds."""
    warnings = []
    splits_present = set(df["split"].unique())
    missing_splits = VALID_SPLITS - splits_present
    if missing_splits:
        warnings.append(f"Missing splits: {missing_splits}")

    # Check fold coverage within train split
    train_df = df[df["split"] == "train"]
    if not train_df.empty and "fold" in train_df.columns:
        folds = sorted(train_df["fold"].unique())
        if folds and folds[0] >= 0:
            expected = set(range(max(folds) + 1))
            actual   = set(folds)
            if expected != actual:
                warnings.append(f"Fold indices incomplete: expected {expected}, got {actual}")

    return warnings


def check_class_balance(df: pd.DataFrame, min_pct: float = 1.0) -> List[str]:
    """Warn if any class is below min_pct percent of the training set."""
    warnings = []
    train_df = df[df["split"] == "train"]
    if train_df.empty:
        return warnings

    counts = train_df["label"].value_counts()
    total  = counts.sum()
    for label_idx, cnt in counts.items():
        pct = 100 * cnt / total
        if pct < min_pct:
            name = train_df.loc[train_df["label"] == label_idx, "label_name"].iloc[0]
            warnings.append(
                f"Class '{name}' (label={label_idx}) is severely underrepresented: "
                f"{cnt} samples ({pct:.2f}%)"
            )
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_checks(
    df: pd.DataFrame,
    disease: str = "unknown",
    check_files: bool = True,
    check_corrupt: bool = False,
    max_workers: int = 4,
    corrupt_sample_size: Optional[int] = 500,
) -> Dict[str, List[str]]:
    """
    Run all quality checks and return a report dict.

    Returns
    -------
    {
        "errors":   [...],  # must fix before proceeding
        "warnings": [...],  # informational
    }
    """
    print(f"\n[quality_checks] Running checks for '{disease}' ({len(df)} rows) …")
    errors   = []
    warnings = []

    # 1. Schema
    schema_errors = check_schema_conformance(df)
    errors.extend(schema_errors)

    # 2. Label consistency
    label_errors = check_label_consistency(df)
    errors.extend(label_errors)

    # 3. Split coverage
    split_warns = check_split_coverage(df)
    warnings.extend(split_warns)

    # 4. Class balance
    balance_warns = check_class_balance(df)
    warnings.extend(balance_warns)

    # 5. File existence
    if check_files:
        missing = check_file_existence(df)
        if missing:
            errors.append(f"{len(missing)} missing image files (first 5: {missing[:5]})")

    # 6. Corrupt images (opt-in, slow)
    if check_corrupt:
        # Only check existing files
        existing_df = df[df["image_path"].apply(os.path.exists)]
        corrupt = check_corrupt_images(existing_df, max_workers, corrupt_sample_size)
        if corrupt:
            warnings.append(f"{len(corrupt)} potentially corrupt images (first 3: {corrupt[:3]})")

    # Summary
    print(f"  Errors   : {len(errors)}")
    print(f"  Warnings : {len(warnings)}")
    for e in errors:
        print(f"  [ERROR]   {e}")
    for w in warnings:
        print(f"  [WARN]    {w}")

    return {"errors": errors, "warnings": warnings}


def assert_quality(
    df: pd.DataFrame,
    disease: str = "unknown",
    **kwargs,
):
    """Run quality checks and raise RuntimeError if any errors found."""
    report = run_quality_checks(df, disease, **kwargs)
    if report["errors"]:
        raise RuntimeError(
            f"Quality check failed for '{disease}':\n"
            + "\n".join(report["errors"])
        )
    print(f"[quality_checks] ✓ All checks passed for '{disease}'")
