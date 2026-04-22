"""
split_validator.py
==================
Validates that train/val/test splits are patient-safe:
  • No patient_id appears in more than one split
  • No patient_id appears in more than one fold (for CV folds)
  • Class distribution is approximately equal across folds (stratification check)

This is critical for:
  - NIH ChestX-ray14: patients have multiple images (frontal + lateral)
  - ISIC 2024: isic_id per patient
  - ADNI: same subject scanned at multiple timepoints

Usage
-----
    from data_pipeline.validation.split_validator import SplitValidator

    validator = SplitValidator(df, disease="chestxray14")
    validator.validate()                    # raises on critical errors
    report = validator.report()             # returns summary dict
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SplitValidator:
    """
    Validates patient-level data leakage across splits and folds.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame conforming to the unified schema.
    disease : str
        Disease tag for logging.
    """

    def __init__(self, df: pd.DataFrame, disease: str = "unknown"):
        self.df      = df.copy()
        self.disease = disease

    # ── Core checks ──────────────────────────────────────────────────────────

    def check_split_leakage(self) -> List[str]:
        """
        Return list of patient_ids that appear in more than one split
        (train/val/test).  An empty list means no leakage.
        """
        # Ignore synthetic patient IDs when all are synthetic (MedMNIST)
        if self._all_synthetic():
            return []   # No real patient IDs → leakage not meaningful

        pid_splits = (
            self.df.groupby("patient_id")["split"]
            .apply(lambda s: set(s.unique()))
        )
        leaking = pid_splits[pid_splits.apply(len) > 1]
        return leaking.index.tolist()

    def check_fold_leakage(self) -> List[str]:
        """
        Within the train split, check that no patient_id appears in more
        than one fold.  This catches improperly done k-fold splitting.
        """
        if self._all_synthetic():
            return []

        train_df = self.df[self.df["split"] == "train"]
        if train_df.empty or "fold" not in train_df.columns:
            return []

        pid_folds = (
            train_df.groupby("patient_id")["fold"]
            .apply(lambda s: set(s.unique()))
        )
        leaking = pid_folds[pid_folds.apply(len) > 1]
        return leaking.index.tolist()

    def check_stratification(
        self, tolerance: float = 0.10
    ) -> List[str]:
        """
        Check that class distribution is roughly equal across folds.

        Parameters
        ----------
        tolerance : float
            Maximum allowed deviation from the global class fraction (0–1).
        """
        warnings = []
        train_df = self.df[(self.df["split"] == "train") & (self.df["fold"] >= 0)]
        if train_df.empty:
            return warnings

        global_dist = train_df["label"].value_counts(normalize=True)
        folds       = sorted(train_df["fold"].unique())

        for fold_idx in folds:
            fold_df   = train_df[train_df["fold"] == fold_idx]
            fold_dist = fold_df["label"].value_counts(normalize=True)

            for label, global_frac in global_dist.items():
                fold_frac = fold_dist.get(label, 0.0)
                deviation = abs(fold_frac - global_frac)
                if deviation > tolerance:
                    warnings.append(
                        f"Fold {fold_idx}, label {label}: "
                        f"global={global_frac:.3f}, fold={fold_frac:.3f}, "
                        f"deviation={deviation:.3f} > {tolerance}"
                    )

        return warnings

    def check_test_set_integrity(self) -> List[str]:
        """Ensure test split exists and is non-empty."""
        errors = []
        test_df = self.df[self.df["split"] == "test"]
        if test_df.empty:
            errors.append("Test split is empty or missing")
        elif len(test_df) < 10:
            errors.append(f"Test split has only {len(test_df)} samples (suspiciously small)")
        return errors

    # ── Orchestrator ─────────────────────────────────────────────────────────

    def validate(self, strict: bool = True):
        """
        Run all checks.  Raises RuntimeError on critical failures.

        Parameters
        ----------
        strict : bool
            If True, patient leakage is a hard error.
            If False, it is only reported as a warning.
        """
        report = self.report()

        critical = report["split_leakage"] + report["fold_leakage"] + report["test_errors"]
        if critical and strict:
            raise RuntimeError(
                f"[SplitValidator] Critical failures for '{self.disease}':\n"
                + "\n".join(f"  • {e}" for e in critical)
            )

        print(f"[SplitValidator] '{self.disease}' validation complete.")
        if critical:
            for msg in critical:
                print(f"  [ERROR] {msg}")
        for msg in report["stratification_warnings"]:
            print(f"  [WARN]  {msg}")
        if not critical:
            print("  ✓ No patient leakage detected")

    def report(self) -> Dict[str, List[str]]:
        """Return a dict with all check results."""
        return {
            "split_leakage":           self.check_split_leakage(),
            "fold_leakage":            self.check_fold_leakage(),
            "stratification_warnings": self.check_stratification(),
            "test_errors":             self.check_test_set_integrity(),
        }

    # ── Summary table ─────────────────────────────────────────────────────────

    def patient_split_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame showing how patients are distributed
        across splits — useful for spot-checking.
        """
        if self._all_synthetic():
            return pd.DataFrame({"note": ["Synthetic patient IDs — no real patients"]})

        summary = (
            self.df.groupby("split")["patient_id"]
            .nunique()
            .reset_index()
            .rename(columns={"patient_id": "unique_patients"})
        )
        total_images = self.df.groupby("split").size().reset_index(name="total_images")
        return summary.merge(total_images, on="split")

    def fold_class_distribution(self) -> pd.DataFrame:
        """
        Return a DataFrame with class counts per fold.
        Useful for verifying stratification quality.
        """
        train_df = self.df[(self.df["split"] == "train") & (self.df["fold"] >= 0)]
        if train_df.empty:
            return pd.DataFrame()

        pivot = (
            train_df.groupby(["fold", "label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        return pivot

    # ── Internal ─────────────────────────────────────────────────────────────

    def _all_synthetic(self) -> bool:
        """Return True if all patient_ids look synthetic."""
        if "patient_id" not in self.df.columns:
            return True
        return self.df["patient_id"].str.startswith("synthetic_").all()
