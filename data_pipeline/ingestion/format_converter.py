"""
format_converter.py
===================
Converts raw medical imaging formats into the canonical PNG representation
expected by the rest of the pipeline.

Supported input formats:
  • DICOM  (.dcm)         → PNG via pydicom + windowing
  • NIfTI  (.nii, .nii.gz) → per-slice PNGs via nibabel
  • NPZ    (MedMNIST)    → per-split PNG files
  • JPEG / PNG           → copy / lossless re-save

All output images are saved as 8-bit RGB PNGs.

Usage
-----
python -m data_pipeline.ingestion.format_converter \\
    --input_dir /path/to/raw \\
    --output_dir /path/to/pngs \\
    --format dicom

python -m data_pipeline.ingestion.format_converter \\
    --input_dir /path/to/npz_files \\
    --output_dir /path/to/pngs \\
    --format npz \\
    --dataset_name retinamnist
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Core converter functions
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly normalise any float/int array to uint8 [0, 255]."""
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert a 2-D grayscale array to (H, W, 3) RGB."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    return arr


def convert_dicom(
    dcm_path: str,
    out_path: str,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
) -> str:
    """
    Convert a DICOM file to PNG.

    Applies standard windowing (W/L) if provided; otherwise clips to
    the dataset's own min/max.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom required: pip install pydicom")

    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Window / level
    if window_center is not None and window_width is not None:
        lo = window_center - window_width / 2
        hi = window_center + window_width / 2
        arr = np.clip(arr, lo, hi)

    arr = _normalise_to_uint8(arr)
    img = Image.fromarray(_to_rgb(arr))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path


def convert_nifti(
    nii_path: str,
    out_dir: str,
    axis: int = 2,
) -> list[str]:
    """
    Convert a NIfTI volume to per-slice PNGs.

    Parameters
    ----------
    nii_path : path to .nii or .nii.gz
    out_dir  : directory where slices are saved
    axis     : axis along which to slice (0=sagittal, 1=coronal, 2=axial)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    img  = nib.load(nii_path)
    data = img.get_fdata()

    # Ensure 3D
    if data.ndim == 4:
        data = data[..., 0]   # take first volume of 4-D series

    os.makedirs(out_dir, exist_ok=True)
    stem = Path(nii_path).stem.replace(".nii", "")
    saved = []

    n_slices = data.shape[axis]
    for i in range(n_slices):
        if axis == 0:
            sl = data[i, :, :]
        elif axis == 1:
            sl = data[:, i, :]
        else:
            sl = data[:, :, i]

        sl_u8  = _normalise_to_uint8(sl)
        out_path = os.path.join(out_dir, f"{stem}_slice{i:04d}.png")
        img_pil = Image.fromarray(_to_rgb(sl_u8))
        img_pil.save(out_path, format="PNG")
        saved.append(out_path)

    return saved


def convert_npz_medmnist(
    npz_path: str,
    out_dir: str,
    dataset_name: str = "medmnist",
    target_size: Tuple[int, int] = (224, 224),
) -> dict:
    """
    Convert a MedMNIST NPZ file into per-split, per-class PNG files.

    NPZ keys expected:
        train_images, train_labels,
        val_images,   val_labels,
        test_images,  test_labels

    Returns a dict mapping split → list of saved paths.
    """
    data   = np.load(npz_path)
    splits = {
        "train": ("train_images", "train_labels"),
        "val":   ("val_images",   "val_labels"),
        "test":  ("test_images",  "test_labels"),
    }
    saved = {}

    for split, (img_key, lbl_key) in splits.items():
        if img_key not in data:
            continue
        images = data[img_key]   # (N, H, W) or (N, H, W, C)
        labels = data[lbl_key]   # (N, 1) or (N,)
        labels = labels.flatten()

        split_paths = []
        for idx, (img, lbl) in enumerate(zip(images, labels)):
            # Ensure uint8 RGB
            if img.dtype != np.uint8:
                img = _normalise_to_uint8(img)
            img = _to_rgb(img)

            pil_img = Image.fromarray(img.astype(np.uint8))
            # Upscale to model input size with bicubic interpolation
            if pil_img.size != target_size:
                pil_img = pil_img.resize(target_size, Image.BICUBIC)

            class_dir = Path(out_dir) / split / str(int(lbl))
            class_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(class_dir / f"img_{idx:08d}.png")
            pil_img.save(out_path, format="PNG")
            split_paths.append(out_path)

        saved[split] = split_paths
        print(f"[format_converter] {dataset_name}/{split}: {len(split_paths)} images saved")

    return saved


def convert_directory_dicom(
    input_dir: str,
    output_dir: str,
    recursive: bool = True,
) -> list[str]:
    """
    Batch convert all DICOM files in a directory tree.
    Mirrors the directory structure in output_dir.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    pattern    = "**/*.dcm" if recursive else "*.dcm"
    dcm_files  = list(input_dir.glob(pattern))
    saved      = []

    print(f"[format_converter] Converting {len(dcm_files)} DICOM files...")
    for dcm_path in dcm_files:
        rel  = dcm_path.relative_to(input_dir)
        out  = output_dir / rel.with_suffix(".png")
        try:
            convert_dicom(str(dcm_path), str(out))
            saved.append(str(out))
        except Exception as e:
            print(f"  [WARN] {dcm_path.name}: {e}")

    print(f"[format_converter] Done. {len(saved)}/{len(dcm_files)} converted.")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Convert medical images to PNG")
    p.add_argument("--input_dir",    required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--format",       choices=["dicom", "nifti", "npz", "copy"],
                   default="dicom")
    p.add_argument("--dataset_name", default="medmnist",
                   help="Dataset name (for NPZ MedMNIST conversion)")
    p.add_argument("--target_size",  type=int, default=224,
                   help="Output image size (square, for NPZ upscaling)")
    p.add_argument("--nifti_axis",   type=int, default=2,
                   help="Slicing axis for NIfTI (0=sag, 1=cor, 2=ax)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.format == "dicom":
        convert_directory_dicom(args.input_dir, args.output_dir)

    elif args.format == "nifti":
        inp = Path(args.input_dir)
        for nii in list(inp.glob("**/*.nii")) + list(inp.glob("**/*.nii.gz")):
            out_sub = Path(args.output_dir) / nii.stem.replace(".nii", "")
            convert_nifti(str(nii), str(out_sub), axis=args.nifti_axis)

    elif args.format == "npz":
        for npz in Path(args.input_dir).glob("*.npz"):
            out_sub = Path(args.output_dir) / npz.stem
            convert_npz_medmnist(
                str(npz), str(out_sub),
                dataset_name=args.dataset_name,
                target_size=(args.target_size, args.target_size),
            )

    elif args.format == "copy":
        # Simply copy JPEG/PNG images to output_dir preserving structure
        from shutil import copy2
        for p in Path(args.input_dir).rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel = p.relative_to(args.input_dir)
                out = Path(args.output_dir) / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                copy2(str(p), str(out))
