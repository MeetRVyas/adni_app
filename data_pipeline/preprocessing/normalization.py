"""
normalization.py
================
Computes and stores per-disease image normalisation statistics
(channel-wise mean and std) from the training split.

These statistics are used at inference time via the standard
torchvision.transforms.Normalize transform.

Usage
-----
    from data_pipeline.preprocessing.normalization import compute_stats, load_stats

    # Compute and cache
    stats = compute_stats(
        image_paths=train_df["image_path"].tolist(),
        disease="adni",
        cache_dir="registry/stats",
    )
    print(stats)
    # {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    # Load from cache
    stats = load_stats("adni", cache_dir="registry/stats")
    transform = get_normalize_transform(stats)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ImageNet defaults — used as fallback when stats file not found
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def compute_stats(
    image_paths: List[str],
    disease: str,
    cache_dir: str = "registry/stats",
    target_size: int = 224,
    max_samples: Optional[int] = 2000,
    num_workers: int = 4,
) -> Dict[str, List[float]]:
    """
    Compute per-channel mean and std over the training set.

    Parameters
    ----------
    image_paths : list of str
        Paths to training images (from registry).
    disease : str
        Disease tag — used as the cache file key.
    cache_dir : str
        Directory where JSON stats files are stored.
    target_size : int
        All images are resized to (target_size, target_size) before
        computing stats (matches the actual training input size).
    max_samples : int | None
        Cap the number of images for efficiency. None = use all.
    num_workers : int
        Parallel workers for image loading.

    Returns
    -------
    dict with keys 'mean' and 'std', each a list of 3 floats.
    """
    cache_path = Path(cache_dir) / f"{disease}_stats.json"
    if cache_path.exists():
        print(f"[normalization] Loading cached stats for '{disease}': {cache_path}")
        return _load_json(cache_path)

    print(f"[normalization] Computing stats for '{disease}' ({len(image_paths)} images) …")

    paths = image_paths
    if max_samples and len(paths) > max_samples:
        import random
        random.seed(42)
        paths = random.sample(paths, max_samples)

    # Welford online algorithm — numerically stable, single pass
    n        = 0
    mean     = np.zeros(3, dtype=np.float64)
    M2       = np.zeros(3, dtype=np.float64)

    for path in paths:
        try:
            arr = _load_rgb_array(path, target_size)   # (H, W, 3) float [0,1]
        except Exception as e:
            print(f"  [WARN] Skipping {path}: {e}")
            continue

        pixels = arr.reshape(-1, 3)    # (H*W, 3)
        for px in pixels:
            n += 1
            delta = px - mean
            mean += delta / n
            delta2 = px - mean
            M2 += delta * delta2

    if n < 2:
        print("[normalization] Not enough valid images — using ImageNet defaults")
        stats = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    else:
        variance = M2 / (n - 1)
        std      = np.sqrt(variance)
        stats = {
            "mean": mean.tolist(),
            "std":  std.tolist(),
        }

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  mean = {[f'{v:.4f}' for v in stats['mean']]}")
    print(f"  std  = {[f'{v:.4f}' for v in stats['std']]}")
    print(f"  Saved → {cache_path}")
    return stats


def load_stats(
    disease: str,
    cache_dir: str = "registry/stats",
    fallback_to_imagenet: bool = True,
) -> Dict[str, List[float]]:
    """
    Load precomputed stats from cache.

    Falls back to ImageNet defaults if no cached file is found
    and fallback_to_imagenet=True.
    """
    cache_path = Path(cache_dir) / f"{disease}_stats.json"
    if cache_path.exists():
        return _load_json(cache_path)

    if fallback_to_imagenet:
        print(
            f"[normalization] No stats found for '{disease}'. "
            "Using ImageNet defaults. Run compute_stats() first."
        )
        return {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}

    raise FileNotFoundError(
        f"Stats file not found: {cache_path}. "
        "Run compute_stats() first."
    )


def get_normalize_transform(stats: Dict[str, List[float]]):
    """
    Return a torchvision.transforms.Normalize object from a stats dict.
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision required: pip install torchvision")

    return transforms.Normalize(mean=stats["mean"], std=stats["std"])


def get_transform_pipeline(
    stats: Dict[str, List[float]],
    img_size: int = 224,
    augment: bool = False,
):
    """
    Return a complete torchvision transform pipeline.

    Parameters
    ----------
    augment : bool
        If True, includes random flips and colour jitter (for training).
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision required")

    base = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ]

    if augment:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        pipeline = aug + base
    else:
        pipeline = base

    return transforms.Compose(pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _load_rgb_array(path: str, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr   # (H, W, 3)


def _load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)
