"""
webdataset_loader.py
====================
Drop-in DataLoader replacement for the existing ML pipeline.

This module provides get_dataloader() — the single function that
Cross_Validator calls instead of constructing a DataLoader directly.

Design goals:
  • Fully backward compatible with the existing module/ code
  • Falls back to the original FullDataset/DataLoader if shards are absent
  • Optionally streams from WebDataset .tar shards for large datasets

Public API
----------
    from data_pipeline.loaders.webdataset_loader import get_dataloader

    # Replaces:
    #   DataLoader(Subset(full_dataset, train_idx), ...)
    # With:
    train_loader = get_dataloader(
        disease="adni",
        fold=0,
        split="train",
        img_size=224,
        batch_size=32,
        num_workers=4,
        registry_dir="registry/",
        shards_dir="shards/",
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Registry-backed Dataset (no shards required)
# ─────────────────────────────────────────────────────────────────────────────

class RegistryDataset(Dataset):
    """
    PyTorch Dataset backed by a metadata DataFrame.

    This is the fallback when WebDataset shards are not available.
    It reads images directly from disk using image_path from the registry.
    """

    def __init__(
        self,
        df,                          # pandas DataFrame, registry rows
        transform=None,
    ):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = row["image_path"]
        label = int(row["label"])

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

    @property
    def targets(self) -> List[int]:
        return self.df["label"].tolist()

    @property
    def classes(self) -> List[str]:
        pairs = self.df[["label", "label_name"]].drop_duplicates().sort_values("label")
        return pairs["label_name"].tolist()

    @property
    def samples(self) -> List[Tuple[str, int]]:
        return list(zip(self.df["image_path"].tolist(), self.df["label"].tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# WebDataset-backed loader (when shards exist)
# ─────────────────────────────────────────────────────────────────────────────

def _build_webdataset_loader(
    shard_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader from WebDataset .tar shards."""
    try:
        import webdataset as wds
    except ImportError:
        raise ImportError(
            "webdataset not installed. Run: pip install webdataset\n"
            "Or let the loader fall back to RegistryDataset (remove shards_dir arg)."
        )

    import json

    shard_pattern = os.path.join(shard_dir, "shard-{000000..999999}.tar")
    tar_files = sorted(Path(shard_dir).glob("shard-*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No shard files in {shard_dir}")

    shard_urls = [str(p) for p in tar_files]

    def decode_sample(sample):
        img_bytes = sample.get("jpg") or sample.get("png")
        meta_bytes = sample.get("json", b"{}")

        img  = Image.open(__import__("io").BytesIO(img_bytes)).convert("RGB")
        meta = json.loads(meta_bytes)

        if transform:
            img = transform(img)

        label = int(meta.get("label", 0))
        return img, label

    dataset = (
        wds.WebDataset(shard_urls, shardshuffle=shuffle)
        .decode("pil")
        .map(decode_sample)
        .batched(batch_size, partial=True)
    )

    return DataLoader(
        dataset,
        batch_size=None,         # batching done by WebDataset
        num_workers=num_workers,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloader(
    disease: str,
    fold: int,
    split: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    registry_dir: str = "registry/",
    shards_dir: Optional[str] = "shards/",
    augment: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    stats: Optional[dict] = None,
) -> DataLoader:
    """
    Drop-in replacement for the DataLoader construction in Cross_Validator.

    Priority:
      1. WebDataset shards (if shards_dir and shard files exist)
      2. RegistryDataset  (reads from image_path in Parquet)

    Parameters
    ----------
    disease : str
        Disease tag, e.g. 'adni'.
    fold : int
        Fold index (0-based). -1 for datasets with fixed splits.
    split : str
        'train' | 'val' | 'test'.
    img_size : int
        Image resize target.
    batch_size : int
    num_workers : int
    registry_dir : str
        Directory containing <disease>.parquet.
    shards_dir : str | None
        Directory containing shards/<disease>/<partition>/*.tar.
        Set to None to always use RegistryDataset.
    augment : bool
        If True, applies training augmentations (only for split='train').
    stats : dict | None
        Normalisation stats {'mean': [...], 'std': [...]}.
        If None, loads from registry/stats/<disease>_stats.json,
        falling back to ImageNet.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    # ── Resolve normalisation stats ──────────────────────────────────────────
    if stats is None:
        from data_pipeline.preprocessing.normalization import load_stats
        stats = load_stats(disease, cache_dir=os.path.join(registry_dir, "stats"))

    # ── Build transform ──────────────────────────────────────────────────────
    do_augment = augment and (split == "train")
    transform  = _build_transform(img_size, stats, augment=do_augment)

    # ── Try shard-based loader ───────────────────────────────────────────────
    if shards_dir:
        partition  = _shard_partition_name(split, fold)
        shard_path = Path(shards_dir) / disease / partition

        if shard_path.exists() and any(shard_path.glob("shard-*.tar")):
            try:
                return _build_webdataset_loader(
                    str(shard_path), transform, batch_size, num_workers,
                    shuffle=(split == "train"),
                )
            except Exception as e:
                print(f"[webdataset_loader] Shard loader failed ({e}), falling back to RegistryDataset")

    # ── Fall back: RegistryDataset ───────────────────────────────────────────
    from data_pipeline.registry.disease_registry import DiseaseRegistry

    reg = DiseaseRegistry(registry_dir, use_dask=False)
    df  = _select_rows(reg, disease, split, fold)

    dataset = RegistryDataset(df, transform=transform)

    shuffle = (split == "train")
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        drop_last=(split == "train"),
    )
    print(
        f"[webdataset_loader] RegistryDataset: disease={disease}, "
        f"split={split}, fold={fold}, n={len(dataset)}"
    )
    return loader


def get_train_val_test_loaders(
    disease: str,
    fold: int,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    registry_dir: str = "registry/",
    shards_dir: Optional[str] = "shards/",
    augment_train: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience wrapper that returns all three loaders in one call.
    """
    common = dict(
        disease=disease, img_size=img_size, batch_size=batch_size,
        num_workers=num_workers, registry_dir=registry_dir, shards_dir=shards_dir,
    )
    train_loader = get_dataloader(fold=fold, split="train",  augment=augment_train, **common)
    val_loader   = get_dataloader(fold=fold, split="val",    augment=False,        **common)
    test_loader  = get_dataloader(fold=fold, split="test",   augment=False,        **common)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _build_transform(img_size: int, stats: dict, augment: bool = False):
    aug_transforms = []
    if augment:
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]

    return transforms.Compose([
        *aug_transforms,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])


def _shard_partition_name(split: str, fold: int) -> str:
    if split == "test":
        return "test"
    if split == "val" and fold < 0:
        return "val"                   # MedMNIST fixed val
    return f"{split}-fold{fold}"


def _select_rows(reg, disease: str, split: str, fold: int):
    """
    Select the right rows from the registry for a given split and fold.

    Training rows: all samples in split='train' with fold != this fold.
    Val rows: all samples in split='train' with fold == this fold,
              OR samples in split='val' (MedMNIST-style).
    Test rows: split='test'.
    """
    if split == "test":
        return reg.query(disease, split="test")

    if split == "val":
        # Try fixed val split first (MedMNIST)
        fixed_val = reg.query(disease, split="val")
        if not fixed_val.empty:
            return fixed_val
        # Otherwise use k-fold val partition
        return reg.query(disease, split="train", fold=fold)

    if split == "train":
        # All training rows NOT in this fold
        df = reg.query(disease, split="train")
        return df[df["fold"] != fold].reset_index(drop=True)

    raise ValueError(f"Unknown split: {split}")
