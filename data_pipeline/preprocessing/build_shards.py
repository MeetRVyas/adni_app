"""
build_shards.py
===============
Converts a registry Parquet file into WebDataset-format .tar shards.

Each shard contains a fixed number of samples. Within each shard,
each sample is represented as:
    <key>.jpg     – the image (re-encoded at target_size × target_size)
    <key>.json    – metadata (label, label_name, patient_id, disease, fold, split, …)

WebDataset shards enable:
  • Streaming-friendly loading (no random-access required)
  • Large-dataset training without loading everything into RAM
  • Easy integration with the existing DataLoader-based pipeline

Shard layout
------------
    shards/
        <disease>/
            train-fold0/
                shard-000000.tar
                shard-000001.tar
                ...
            val-fold0/
                shard-000000.tar
            test/
                shard-000000.tar

Usage
-----
    python -m data_pipeline.preprocessing.build_shards \\
        --disease adni \\
        --registry_dir registry/ \\
        --shards_dir shards/ \\
        --target_size 224 \\
        --samples_per_shard 1000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def build_shards(
    df: pd.DataFrame,
    output_dir: str,
    target_size: int = 224,
    samples_per_shard: int = 1000,
    quality: int = 90,
    shuffle: bool = True,
    seed: int = 42,
) -> List[str]:
    """
    Write WebDataset .tar shards from a metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered metadata (one split × one fold).
    output_dir : str
        Directory where .tar files are written.
    target_size : int
        Images are resized to (target_size, target_size) JPEG.
    samples_per_shard : int
        Number of samples per tar file.
    quality : int
        JPEG compression quality (1-95).
    shuffle : bool
        Whether to shuffle samples before sharding.

    Returns
    -------
    List of paths to the created .tar files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = df.to_dict(orient="records")
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(rows)

    shard_idx  = 0
    shard_rows = []
    tar_paths  = []

    def flush_shard(rows_batch: list, idx: int):
        tar_path = output_dir / f"shard-{idx:06d}.tar"
        with tarfile.open(str(tar_path), "w") as tar:
            for i, row in enumerate(rows_batch):
                key = f"{idx:06d}_{i:06d}"
                # Image
                img_bytes = _resize_to_jpeg(row["image_path"], target_size, quality)
                _add_bytes_to_tar(tar, f"{key}.jpg", img_bytes)
                # Metadata
                meta = {k: v for k, v in row.items() if k != "image_path"}
                meta_bytes = json.dumps(meta, default=str).encode("utf-8")
                _add_bytes_to_tar(tar, f"{key}.json", meta_bytes)

        tar_paths.append(str(tar_path))
        return str(tar_path)

    for row in rows:
        shard_rows.append(row)
        if len(shard_rows) >= samples_per_shard:
            flush_shard(shard_rows, shard_idx)
            shard_idx  += 1
            shard_rows  = []

    if shard_rows:
        flush_shard(shard_rows, shard_idx)

    print(f"[build_shards] {len(rows)} samples → {len(tar_paths)} shards in {output_dir}")
    return tar_paths


def build_all_shards(
    disease: str,
    registry_dir: str = "registry/",
    shards_dir: str = "shards/",
    target_size: int = 224,
    samples_per_shard: int = 1000,
    nfolds: Optional[int] = None,
) -> dict:
    """
    Build shards for all splits and folds of a disease.

    Returns
    -------
    dict mapping partition_name → list of .tar paths.
    """
    from data_pipeline.registry.build_metadata import load_parquet

    df = load_parquet(registry_dir, disease)
    print(f"[build_shards] Loaded {len(df)} rows for '{disease}'")

    created = {}
    disease_shards_dir = Path(shards_dir) / disease

    # Test set (not fold-specific)
    test_df = df[df["split"] == "test"]
    if not test_df.empty:
        test_out = disease_shards_dir / "test"
        paths = build_shards(test_df, str(test_out), target_size, samples_per_shard, shuffle=False)
        created["test"] = paths

    # Train/val per fold
    folds = sorted(df[df["split"] == "train"]["fold"].unique())
    if nfolds:
        folds = [f for f in folds if f < nfolds]

    for fold_idx in folds:
        if fold_idx < 0:
            continue

        # Training data for this fold = everything NOT in this fold
        train_df = df[(df["split"] == "train") & (df["fold"] != fold_idx)]
        val_df   = df[(df["split"] == "train") & (df["fold"] == fold_idx)]

        if not train_df.empty:
            train_out = disease_shards_dir / f"train-fold{fold_idx}"
            created[f"train-fold{fold_idx}"] = build_shards(
                train_df, str(train_out), target_size, samples_per_shard
            )

        if not val_df.empty:
            val_out = disease_shards_dir / f"val-fold{fold_idx}"
            created[f"val-fold{fold_idx}"] = build_shards(
                val_df, str(val_out), target_size, samples_per_shard, shuffle=False
            )

    # MedMNIST-style: fixed val split
    fixed_val = df[df["split"] == "val"]
    if not fixed_val.empty:
        val_out = disease_shards_dir / "val"
        created["val"] = build_shards(
            fixed_val, str(val_out), target_size, samples_per_shard, shuffle=False
        )

    return created


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize_to_jpeg(image_path: str, size: int, quality: int) -> bytes:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def _add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Build WebDataset shards from registry Parquet")
    p.add_argument("--disease",           required=True)
    p.add_argument("--registry_dir",      default="registry/")
    p.add_argument("--shards_dir",        default="shards/")
    p.add_argument("--target_size",       type=int, default=224)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--nfolds",            type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_all_shards(
        disease=args.disease,
        registry_dir=args.registry_dir,
        shards_dir=args.shards_dir,
        target_size=args.target_size,
        samples_per_shard=args.samples_per_shard,
        nfolds=args.nfolds,
    )
