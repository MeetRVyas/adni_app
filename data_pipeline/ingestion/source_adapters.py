"""
source_adapters.py
==================
Per-source download/access adapters.  Each adapter is responsible for:
  1. Authenticating (if required)
  2. Downloading / streaming raw files to a local staging directory
  3. Returning the local path for the next pipeline stage

Adapters
--------
  KaggleAdapter       — any Kaggle dataset (API key required)
  NIHChestXrayAdapter — NIH ChestX-ray14 via Kaggle mirror
  ISIC2024Adapter     — ISIC 2024 SLICE-3D via Kaggle
  HuggingFaceAdapter  — see huggingface_adapter.py (separate file)
  LocalAdapter        — data already on disk (no download needed)

Usage
-----
    from data_pipeline.ingestion.source_adapters import NIHChestXrayAdapter

    adapter = NIHChestXrayAdapter(staging_dir="/data/staging/chestxray14")
    local_path = adapter.fetch()   # returns path to downloaded/staged data
"""

from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseAdapter(ABC):
    """Abstract base for all source adapters."""

    def __init__(self, staging_dir: str):
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self) -> Path:
        """Download / stage data and return local directory path."""
        ...

    def is_complete(self, marker_file: str = ".download_complete") -> bool:
        """Return True if a previous successful download exists."""
        return (self.staging_dir / marker_file).exists()

    def mark_complete(self, marker_file: str = ".download_complete"):
        (self.staging_dir / marker_file).touch()


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle adapter
# ─────────────────────────────────────────────────────────────────────────────

class KaggleAdapter(BaseAdapter):
    """
    Download any Kaggle dataset via the Kaggle API.

    Requirements:
        pip install kaggle
        ~/.kaggle/kaggle.json  with {"username": "...", "key": "..."}
    """

    def __init__(
        self,
        dataset_slug: str,
        staging_dir: str,
        unzip: bool = True,
    ):
        super().__init__(staging_dir)
        self.dataset_slug = dataset_slug
        self.unzip        = unzip

    def fetch(self) -> Path:
        if self.is_complete():
            print(f"[KaggleAdapter] Already staged: {self.staging_dir}")
            return self.staging_dir

        try:
            import kaggle  # noqa: F401
        except ImportError:
            raise ImportError(
                "kaggle package not installed. Run: pip install kaggle\n"
                "Also set up ~/.kaggle/kaggle.json with your API credentials."
            )

        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", self.dataset_slug,
            "-p", str(self.staging_dir),
        ]
        if self.unzip:
            cmd.append("--unzip")

        print(f"[KaggleAdapter] Downloading {self.dataset_slug} …")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed:\n{result.stderr}"
            )
        print(f"[KaggleAdapter] Done → {self.staging_dir}")
        self.mark_complete()
        return self.staging_dir


# ─────────────────────────────────────────────────────────────────────────────
# NIH ChestX-ray14
# ─────────────────────────────────────────────────────────────────────────────

class NIHChestXrayAdapter(KaggleAdapter):
    """
    NIH ChestX-ray14 — fully open access, no approval needed.

    Kaggle slug: nih-chest-xrays/data
    After download, images are in: staging_dir/images/
    Metadata CSV: staging_dir/Data_Entry_2017.csv
    """

    DEFAULT_SLUG = "nih-chest-xrays/data"

    def __init__(self, staging_dir: str = "staging/chestxray14"):
        super().__init__(
            dataset_slug=self.DEFAULT_SLUG,
            staging_dir=staging_dir,
            unzip=True,
        )

    @property
    def images_dir(self) -> Path:
        return self.staging_dir / "images"

    @property
    def csv_path(self) -> Path:
        return self.staging_dir / "Data_Entry_2017.csv"

    def validate(self) -> bool:
        """Check that the key files exist after download."""
        ok = self.images_dir.exists() and self.csv_path.exists()
        if not ok:
            print(f"[NIHChestXrayAdapter] Validation FAILED — check {self.staging_dir}")
        return ok


# ─────────────────────────────────────────────────────────────────────────────
# ISIC 2024
# ─────────────────────────────────────────────────────────────────────────────

class ISIC2024Adapter(KaggleAdapter):
    """
    ISIC 2024 SLICE-3D Challenge Dataset.

    License: CC-BY-NC (academic) or Permissive variant.
    Kaggle competition: isic-2024-challenge
    Images: train-image/image/*.jpg
    Metadata: train-metadata.csv
    """

    DEFAULT_SLUG = "isic-2024-challenge"

    def __init__(self, staging_dir: str = "staging/isic2024"):
        super().__init__(
            dataset_slug=self.DEFAULT_SLUG,
            staging_dir=staging_dir,
            unzip=True,
        )

    def fetch(self) -> Path:
        """Use competition download instead of dataset download."""
        if self.is_complete():
            print(f"[ISIC2024Adapter] Already staged: {self.staging_dir}")
            return self.staging_dir

        try:
            import kaggle  # noqa: F401
        except ImportError:
            raise ImportError("pip install kaggle required")

        cmd = [
            sys.executable, "-m", "kaggle", "competitions", "download",
            "-c", self.DEFAULT_SLUG,
            "-p", str(self.staging_dir),
            "--unzip",
        ]
        print(f"[ISIC2024Adapter] Downloading ISIC 2024 …")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ISIC download failed:\n{result.stderr}")
        self.mark_complete()
        return self.staging_dir

    @property
    def images_dir(self) -> Path:
        return self.staging_dir / "train-image" / "image"

    @property
    def csv_path(self) -> Path:
        return self.staging_dir / "train-metadata.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Local adapter (data already on disk)
# ─────────────────────────────────────────────────────────────────────────────

class LocalAdapter(BaseAdapter):
    """
    No-op adapter for data already present on disk.

    Parameters
    ----------
    data_dir : str
        Path to the existing data directory.
    staging_dir : str | None
        If None, staging_dir = data_dir (no copy performed).
    """

    def __init__(self, data_dir: str, staging_dir: Optional[str] = None):
        effective_staging = staging_dir or data_dir
        super().__init__(effective_staging)
        self.data_dir = Path(data_dir)

    def fetch(self) -> Path:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"[LocalAdapter] Data directory not found: {self.data_dir}"
            )
        print(f"[LocalAdapter] Using local data at: {self.data_dir}")
        return self.data_dir


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_adapter(disease: str, staging_dir: str, **kwargs) -> BaseAdapter:
    """
    Factory function.  Returns the appropriate adapter for a disease tag.

    Supported disease tags:
        adni         → LocalAdapter (data already present)
        chestxray14  → NIHChestXrayAdapter
        isic2024     → ISIC2024Adapter
        <any other>  → KaggleAdapter (requires dataset_slug kwarg)
    """
    disease = disease.lower()

    if disease == "adni":
        data_dir = kwargs.get("data_dir", "OriginalDataset")
        return LocalAdapter(data_dir=data_dir, staging_dir=staging_dir)

    elif disease == "chestxray14":
        return NIHChestXrayAdapter(staging_dir=staging_dir)

    elif disease == "isic2024":
        return ISIC2024Adapter(staging_dir=staging_dir)

    else:
        slug = kwargs.get("dataset_slug")
        if slug is None:
            raise ValueError(
                f"Unknown disease '{disease}'. "
                "Provide dataset_slug=<kaggle_slug> for custom Kaggle datasets."
            )
        return KaggleAdapter(
            dataset_slug=slug,
            staging_dir=staging_dir,
            unzip=kwargs.get("unzip", True),
        )
