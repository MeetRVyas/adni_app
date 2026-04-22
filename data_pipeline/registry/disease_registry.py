"""
disease_registry.py
===================
Cross-disease Dask-powered registry queries.

All Parquet files in registry/ are opened lazily. Queries that touch only
one disease never load the others.  The registry acts as the single source
of truth for:
  • which images exist and where they live
  • train/val/test assignment
  • fold membership
  • per-class counts and weights

Public API
----------
    reg = DiseaseRegistry("registry/")
    df  = reg.query(disease="adni", split="train", fold=0)
    w   = reg.class_weights("adni")
    reg.summary()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    import warnings
    warnings.warn(
        "Dask not installed — DiseaseRegistry will use pandas (slower for large datasets). "
        "Install with: pip install dask",
        stacklevel=2,
    )


class DiseaseRegistry:
    """
    Lazy, cross-disease metadata registry backed by Parquet files.

    Parameters
    ----------
    registry_dir : str | Path
        Directory containing <disease>.parquet files.
    use_dask : bool
        Force Dask (True) or pandas (False).  Defaults to auto-detect.
    """

    def __init__(self, registry_dir: str = "registry/", use_dask: Optional[bool] = None):
        self.registry_dir = Path(registry_dir)
        self._use_dask = DASK_AVAILABLE if use_dask is None else use_dask
        self._cache: Dict[str, Union[pd.DataFrame, "dd.DataFrame"]] = {}

    # ── Discovery ─────────────────────────────────────────────────────────────

    @property
    def available_diseases(self) -> List[str]:
        return [p.stem for p in self.registry_dir.glob("*.parquet")]

    def _load(self, disease: str):
        if disease in self._cache:
            return self._cache[disease]
        path = self.registry_dir / f"{disease}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No registry file for disease '{disease}'. "
                f"Run build_metadata.py first. Expected: {path}"
            )
        if self._use_dask:
            df = dd.read_parquet(str(path))
        else:
            df = pd.read_parquet(str(path))
        self._cache[disease] = df
        return df

    # ── Core query ────────────────────────────────────────────────────────────

    def query(
        self,
        disease: str,
        split: Optional[str] = None,
        fold: Optional[int] = None,
        label: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return a pandas DataFrame of metadata rows matching the filters.

        Parameters
        ----------
        disease : str
            Disease tag, e.g. 'adni'.
        split : str | None
            'train' | 'val' | 'test'.  None = all splits.
        fold : int | None
            Fold index (0-based). None = all folds.
            Note: for datasets with fixed splits (MedMNIST), fold=-1.
        label : int | None
            Filter to a single class index.
        """
        df = self._load(disease)

        if self._use_dask:
            if split is not None:
                df = df[df["split"] == split]
            if fold is not None:
                df = df[df["fold"] == fold]
            if label is not None:
                df = df[df["label"] == label]
            return df.compute()
        else:
            mask = pd.Series([True] * len(df), index=df.index)
            if split is not None:
                mask &= df["split"] == split
            if fold is not None:
                mask &= df["fold"] == fold
            if label is not None:
                mask &= df["label"] == label
            return df[mask].reset_index(drop=True)

    def query_all_diseases(
        self,
        split: Optional[str] = None,
        fold: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Combine all available disease registries into one DataFrame.
        Useful for cross-disease analysis.
        """
        parts = []
        for disease in self.available_diseases:
            try:
                parts.append(self.query(disease, split=split, fold=fold))
            except Exception as e:
                print(f"[DiseaseRegistry] Skipping {disease}: {e}")
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    # ── Statistics ────────────────────────────────────────────────────────────

    def class_weights(self, disease: str, split: str = "train") -> np.ndarray:
        """
        Compute inverse-frequency class weights for the training set.

        Returns a numpy array of shape (num_classes,), aligned by label index.
        For multi-label datasets, uses primary label counts only.
        """
        df = self.query(disease, split=split)
        counts = df["label"].value_counts().sort_index()
        n_classes = counts.index.max() + 1
        total = counts.sum()

        weights = np.zeros(n_classes, dtype=np.float32)
        for label_idx, cnt in counts.items():
            weights[label_idx] = total / (n_classes * cnt)

        return weights

    def class_names(self, disease: str) -> List[str]:
        """Return ordered class name list for a disease."""
        df = self._load(disease)
        if self._use_dask:
            sub = df[["label", "label_name"]].drop_duplicates().compute()
        else:
            sub = df[["label", "label_name"]].drop_duplicates()
        sub = sub.sort_values("label")
        return sub["label_name"].tolist()

    def num_classes(self, disease: str) -> int:
        return len(self.class_names(disease))

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, disease: Optional[str] = None) -> pd.DataFrame:
        """
        Print a per-class summary table for one or all diseases.
        Returns a DataFrame suitable for logging.
        """
        diseases = [disease] if disease else self.available_diseases
        rows = []
        for d in diseases:
            try:
                df = self._load(d)
                if self._use_dask:
                    df = df.compute()
                for split in ("train", "val", "test"):
                    split_df = df[df["split"] == split]
                    if split_df.empty:
                        continue
                    counts = split_df["label"].value_counts().sort_index()
                    for label_idx, cnt in counts.items():
                        rows.append({
                            "disease":    d,
                            "split":      split,
                            "label":      label_idx,
                            "label_name": split_df.loc[
                                split_df["label"] == label_idx, "label_name"
                            ].iloc[0],
                            "count":      cnt,
                            "pct":        f"{100 * cnt / len(split_df):.1f}%",
                        })
            except Exception as e:
                print(f"[DiseaseRegistry] summary error for {d}: {e}")

        out = pd.DataFrame(rows)
        if not out.empty:
            print(out.to_string(index=False))
        return out

    def cross_disease_summary(self) -> pd.DataFrame:
        """Aggregate sample counts across all diseases."""
        rows = []
        for d in self.available_diseases:
            try:
                df = self._load(d)
                if self._use_dask:
                    df = df.compute()
                rows.append({
                    "disease":    d,
                    "total":      len(df),
                    "train":      (df["split"] == "train").sum(),
                    "val":        (df["split"] == "val").sum(),
                    "test":       (df["split"] == "test").sum(),
                    "num_classes": df["label"].nunique(),
                })
            except Exception as e:
                print(f"[DiseaseRegistry] {d}: {e}")
        return pd.DataFrame(rows)

    # ── Invalidate cache ──────────────────────────────────────────────────────

    def reload(self, disease: Optional[str] = None):
        """Force re-read of Parquet file(s) from disk."""
        if disease:
            self._cache.pop(disease, None)
        else:
            self._cache.clear()
