"""
build_metadata.py
=================
Converts a raw ImageFolder-style directory (or a CSV-annotated dataset)
into the unified Parquet metadata file stored in registry/<disease>.parquet.

Supports:
  • ImageFolder layout  (ADNI, any folder-per-class arrangement)
  • CSV-annotated layout (NIH ChestX-ray14, ISIC 2024)
  • NPZ layout          (MedMNIST — handled by format_converter first)

Usage
-----
python -m data_pipeline.registry.build_metadata \\
    --disease adni \\
    --source_dir /path/to/OriginalDataset \\
    --registry_dir registry/ \\
    --source local \\
    --nfolds 5

python -m data_pipeline.registry.build_metadata \\
    --disease chestxray14 \\
    --source_dir /path/to/images \\
    --csv /path/to/Data_Entry_2017.csv \\
    --registry_dir registry/ \\
    --source nihcc \\
    --nfolds 5

python -m data_pipeline.registry.build_metadata \\
    --disease isic2024 \\
    --source_dir /path/to/isic/train-image/image \\
    --csv /path/to/train-metadata.csv \\
    --registry_dir registry/ \\
    --source isic \\
    --nfolds 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# Make package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.registry.schema import (
    EXTENDED_DEFAULTS,
    FULL_SCHEMA,
    VALID_SPLITS,
    validate_row,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(path: str, chunk: int = 65_536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _image_size(path: str):
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _assign_folds(df: pd.DataFrame, nfolds: int, label_col: str = "label") -> pd.DataFrame:
    """
    Assign stratified k-fold indices.  If patient_id is available and
    non-synthetic, fold assignment is patient-level (no leakage).
    """
    df = df.copy()
    df["fold"] = -1

    # Detect synthetic patient IDs
    has_real_patients = (
        "patient_id" in df.columns
        and df["patient_id"].notna().all()
        and not df["patient_id"].str.startswith("synthetic_").all()
    )

    if has_real_patients:
        # Patient-level stratified split
        patient_df = (
            df.groupby("patient_id")[label_col]
            .agg(lambda s: s.mode().iloc[0])  # majority label per patient
            .reset_index()
        )
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        patient_df["fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(
            skf.split(patient_df["patient_id"], patient_df[label_col])
        ):
            patient_df.loc[val_idx, "fold"] = fold_idx

        pid_to_fold = dict(zip(patient_df["patient_id"], patient_df["fold"]))
        df["fold"] = df["patient_id"].map(pid_to_fold).fillna(-1).astype(int)
    else:
        # Sample-level stratified split
        skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        labels = df[label_col].values
        for fold_idx, (_, val_idx) in enumerate(skf.split(labels, labels)):
            df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    return df


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing optional columns with schema defaults."""
    for col, default in EXTENDED_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    # Ensure all schema columns exist
    for field in FULL_SCHEMA:
        if field.name not in df.columns:
            df[field.name] = None
    return df[list(FULL_SCHEMA.names)]


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific builders
# ─────────────────────────────────────────────────────────────────────────────

def build_from_imagefolder(
    source_dir: str,
    disease: str,
    source: str,
    nfolds: int,
    compute_checksums: bool = False,
    compute_sizes: bool = True,
) -> pd.DataFrame:
    """
    Builds metadata from an ImageFolder-style directory.

    Expected layout:
        source_dir/
            class_A/
                img1.jpg
            class_B/
                img2.jpg
    """
    source_dir = Path(source_dir)
    classes = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No subdirectories found in {source_dir}")

    label_map: Dict[str, int] = {cls: i for i, cls in enumerate(classes)}
    rows: List[dict] = []

    for cls in classes:
        class_dir = source_dir / cls
        img_files = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        ]
        for idx, img_path in enumerate(img_files):
            w, h = _image_size(str(img_path)) if compute_sizes else (None, None)
            row = {
                "image_path":    str(img_path.resolve()),
                "label":         label_map[cls],
                "label_name":    cls,
                "split":         "train",        # will be revised by fold assignment
                "disease":       disease,
                "fold":          -1,
                "patient_id":    f"synthetic_{disease}_{cls}_{idx:06d}",
                "source":        source,
                "image_width":   w,
                "image_height":  h,
                "original_format": img_path.suffix.upper().lstrip("."),
            }
            if compute_checksums:
                row["checksum"] = _sha256(str(img_path))
            rows.append(row)

    df = pd.DataFrame(rows)
    df = _assign_folds(df, nfolds)
    df = _finalise(df)
    print(f"[build_metadata] ImageFolder: {len(df)} samples, {len(classes)} classes")
    return df


def build_from_nih_chestxray14(
    source_dir: str,
    csv_path: str,
    disease: str,
    source: str,
    nfolds: int,
    compute_sizes: bool = True,
) -> pd.DataFrame:
    """
    Builds metadata from NIH ChestX-ray14.

    CSV layout (Data_Entry_2017.csv):
        Image Index | Finding Labels | Follow-up # | Patient ID | ...
    Finding Labels is pipe-separated, e.g. "Atelectasis|Effusion"
    """
    source_dir = Path(source_dir)
    meta = pd.read_csv(csv_path)
    meta.columns = [c.strip() for c in meta.columns]

    ALL_LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
        "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
        "No Finding",
    ]
    label_to_idx = {l: i for i, l in enumerate(ALL_LABELS)}

    rows: List[dict] = []
    for _, r in meta.iterrows():
        fname = str(r["Image Index"]).strip()
        img_path = source_dir / fname
        if not img_path.exists():
            continue

        labels_raw = [l.strip() for l in str(r["Finding Labels"]).split("|")]
        primary_label = labels_raw[0] if labels_raw else "No Finding"
        primary_idx   = label_to_idx.get(primary_label, len(ALL_LABELS) - 1)
        is_multi      = len(labels_raw) > 1

        w, h = _image_size(str(img_path)) if compute_sizes else (None, None)
        row = {
            "image_path":        str(img_path.resolve()),
            "label":             primary_idx,
            "label_name":        primary_label,
            "split":             "train",
            "disease":           disease,
            "fold":              -1,
            "patient_id":        str(r.get("Patient ID", f"synthetic_{fname}")),
            "source":            source,
            "image_width":       w,
            "image_height":      h,
            "is_multilabel":     is_multi,
            "additional_labels": json.dumps(labels_raw),
            "original_format":   "PNG",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _assign_folds(df, nfolds, label_col="label")
    df = _finalise(df)
    print(f"[build_metadata] NIH ChestX-ray14: {len(df)} samples")
    return df


def build_from_isic2024(
    source_dir: str,
    csv_path: str,
    disease: str,
    source: str,
    nfolds: int,
    compute_sizes: bool = True,
) -> pd.DataFrame:
    """
    Builds metadata from ISIC 2024 SLICE-3D.

    CSV layout (train-metadata.csv):
        isic_id | target | patient_id | ...
    target: 0 = benign, 1 = malignant
    """
    source_dir = Path(source_dir)
    meta = pd.read_csv(csv_path)
    label_names = {0: "benign", 1: "malignant"}
    rows: List[dict] = []

    for _, r in meta.iterrows():
        isic_id = str(r["isic_id"]).strip()
        # ISIC images may be stored as <isic_id>.jpg
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = source_dir / f"{isic_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        target = int(r["target"])
        w, h = _image_size(str(img_path)) if compute_sizes else (None, None)
        row = {
            "image_path":    str(img_path.resolve()),
            "label":         target,
            "label_name":    label_names.get(target, str(target)),
            "split":         "train",
            "disease":       disease,
            "fold":          -1,
            "patient_id":    isic_id,   # isic_id is the patient proxy
            "source":        source,
            "image_width":   w,
            "image_height":  h,
            "original_format": img_path.suffix.upper().lstrip("."),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _assign_folds(df, nfolds)
    df = _finalise(df)
    print(f"[build_metadata] ISIC 2024: {len(df)} samples")
    return df


def build_from_medmnist_pngs(
    source_dir: str,
    disease: str,
    source: str,
    nfolds: int,
    compute_sizes: bool = True,
) -> pd.DataFrame:
    """
    Builds metadata from MedMNIST PNGs already extracted by format_converter.

    Expected layout after conversion:
        source_dir/
            train/
                0/img_000000.png
                1/img_000001.png
            val/
                0/...
            test/
                0/...
    """
    source_dir = Path(source_dir)
    rows: List[dict] = []

    for split in ("train", "val", "test"):
        split_dir = source_dir / split
        if not split_dir.exists():
            continue
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        for cls in classes:
            label_idx = int(cls)
            for idx, img_path in enumerate((split_dir / cls).iterdir()):
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                w, h = _image_size(str(img_path)) if compute_sizes else (None, None)
                row = {
                    "image_path":    str(img_path.resolve()),
                    "label":         label_idx,
                    "label_name":    cls,
                    "split":         split,
                    "disease":       disease,
                    "fold":          0 if split == "val" else (-1 if split == "test" else -2),
                    "patient_id":    f"synthetic_{disease}_{split}_{idx:08d}",
                    "source":        source,
                    "image_width":   w,
                    "image_height":  h,
                    "original_format": "PNG",
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    df = _finalise(df)
    print(f"[build_metadata] MedMNIST ({disease}): {len(df)} samples")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Parquet I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, registry_dir: str, disease: str) -> Path:
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    out_path = registry_dir / f"{disease}.parquet"
    table = pa.Table.from_pandas(df, schema=FULL_SCHEMA, safe=False)
    pq.write_table(table, out_path, compression="snappy")
    print(f"[build_metadata] Saved → {out_path}  ({len(df)} rows)")
    return out_path


def load_parquet(registry_dir: str, disease: str) -> pd.DataFrame:
    path = Path(registry_dir) / f"{disease}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    return pq.read_table(path).to_pandas()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Build disease.parquet metadata file")
    p.add_argument("--disease",      required=True,  help="Disease tag, e.g. adni")
    p.add_argument("--source_dir",   required=True,  help="Root image directory")
    p.add_argument("--registry_dir", default="registry/", help="Output Parquet directory")
    p.add_argument("--source",       default="local", help="Source tag")
    p.add_argument("--nfolds",       type=int, default=5)
    p.add_argument("--csv",          default=None,   help="Optional CSV annotation file")
    p.add_argument("--layout",
                   choices=["imagefolder", "chestxray14", "isic2024", "medmnist_png"],
                   default="imagefolder",
                   help="Dataset layout type")
    p.add_argument("--no_sizes",     action="store_true",
                   help="Skip reading image dimensions (faster)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compute_sizes = not args.no_sizes

    if args.layout == "imagefolder":
        df = build_from_imagefolder(
            args.source_dir, args.disease, args.source,
            args.nfolds, compute_sizes=compute_sizes,
        )
    elif args.layout == "chestxray14":
        if not args.csv:
            raise ValueError("--csv required for chestxray14 layout")
        df = build_from_nih_chestxray14(
            args.source_dir, args.csv, args.disease, args.source,
            args.nfolds, compute_sizes=compute_sizes,
        )
    elif args.layout == "isic2024":
        if not args.csv:
            raise ValueError("--csv required for isic2024 layout")
        df = build_from_isic2024(
            args.source_dir, args.csv, args.disease, args.source,
            args.nfolds, compute_sizes=compute_sizes,
        )
    elif args.layout == "medmnist_png":
        df = build_from_medmnist_pngs(
            args.source_dir, args.disease, args.source,
            args.nfolds, compute_sizes=compute_sizes,
        )
    else:
        raise ValueError(f"Unknown layout: {args.layout}")

    save_parquet(df, args.registry_dir, args.disease)
