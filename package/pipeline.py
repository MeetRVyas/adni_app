"""
pipeline.py
===========
Top-level orchestrator for the BDA data pipeline.

This is the single entry point to run the full ETL:
    Data Download → Format Conversion → Metadata Registry → Validation → Sharding

Usage
-----
    # Full pipeline for ADNI (existing data, already in ImageFolder format)
    python -m data_pipeline.pipeline \\
        --disease adni \\
        --source_dir OriginalDataset/ \\
        --layout imagefolder \\
        --source local \\
        --nfolds 5

    # Full pipeline for NIH ChestX-ray14
    python -m data_pipeline.pipeline \\
        --disease chestxray14 \\
        --source_dir staging/chestxray14/images \\
        --csv staging/chestxray14/Data_Entry_2017.csv \\
        --layout chestxray14 \\
        --source nihcc \\
        --nfolds 5

    # Full pipeline for ISIC 2024
    python -m data_pipeline.pipeline \\
        --disease isic2024 \\
        --source_dir staging/isic2024/train-image/image \\
        --csv staging/isic2024/train-metadata.csv \\
        --layout isic2024 \\
        --source isic \\
        --nfolds 5

    # MedMNIST via HuggingFace (downloads automatically)
    python -m data_pipeline.pipeline \\
        --disease retinamnist \\
        --layout medmnist_hf \\
        --hf_subset retinamnist \\
        --source huggingface \\
        --nfolds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def run_pipeline(
    disease: str,
    source_dir: Optional[str] = None,
    layout: str = "imagefolder",
    source: str = "local",
    nfolds: int = 5,
    registry_dir: str = "registry/",
    shards_dir: str = "shards/",
    target_size: int = 224,
    samples_per_shard: int = 1000,
    csv: Optional[str] = None,
    hf_dataset: str = "albertvillanova/medmnist",
    hf_subset: Optional[str] = None,
    staging_dir: Optional[str] = None,
    compute_stats: bool = True,
    build_shards: bool = True,
    validate: bool = True,
    skip_if_exists: bool = True,
):
    """
    Run the full ETL pipeline for one disease dataset.

    Steps:
      1. Ingest / download raw data (if needed)
      2. Convert format (DICOM → PNG, NPZ → PNG, etc.)
      3. Build metadata Parquet file
      4. Run quality checks + split validation
      5. Compute normalisation statistics
      6. Build WebDataset shards (optional)

    Parameters
    ----------
    disease : str
        Disease tag, e.g. 'adni', 'chestxray14', 'isic2024', 'retinamnist'.
    source_dir : str | None
        Path to raw images. Required for all layouts except 'medmnist_hf'.
    layout : str
        One of: 'imagefolder', 'chestxray14', 'isic2024',
                'medmnist_png', 'medmnist_hf'.
    """
    print(f"\n{'='*70}")
    print(f"BDA PIPELINE — Disease: {disease.upper()}")
    print(f"  layout={layout}  source={source}  nfolds={nfolds}")
    print(f"{'='*70}\n")

    parquet_path = Path(registry_dir) / f"{disease}.parquet"

    # ── Step 1: Skip if already processed ────────────────────────────────────
    if skip_if_exists and parquet_path.exists():
        print(f"[pipeline] Registry already exists: {parquet_path}. Skipping ETL.")
        print("[pipeline] (use skip_if_exists=False to force re-build)")
        return {"parquet": str(parquet_path)}

    # ── Step 2: Ingest / Download ─────────────────────────────────────────────
    if layout == "medmnist_hf":
        staging = staging_dir or f"staging/{disease}"
        from data_pipeline.ingestion.huggingface_adapter import HuggingFaceAdapter
        adapter = HuggingFaceAdapter(
            hf_dataset=hf_dataset,
            subset=hf_subset or disease,
            output_dir=staging,
            target_size=target_size,
        )
        source_dir = str(adapter.fetch())
        layout     = "medmnist_png"          # After HF download, use png layout

    elif layout == "chestxray14" and source_dir is None:
        staging = staging_dir or "staging/chestxray14"
        from data_pipeline.ingestion.source_adapters import NIHChestXrayAdapter
        adapter    = NIHChestXrayAdapter(staging_dir=staging)
        adapter.fetch()
        source_dir = str(adapter.images_dir)
        csv        = csv or str(adapter.csv_path)

    elif layout == "isic2024" and source_dir is None:
        staging = staging_dir or "staging/isic2024"
        from data_pipeline.ingestion.source_adapters import ISIC2024Adapter
        adapter    = ISIC2024Adapter(staging_dir=staging)
        adapter.fetch()
        source_dir = str(adapter.images_dir)
        csv        = csv or str(adapter.csv_path)

    # ── Step 3: Build metadata Parquet ────────────────────────────────────────
    print("[pipeline] Step 3: Building metadata registry …")
    from data_pipeline.registry.build_metadata import (
        build_from_imagefolder,
        build_from_nih_chestxray14,
        build_from_isic2024,
        build_from_medmnist_pngs,
        save_parquet,
    )

    if layout == "imagefolder":
        df = build_from_imagefolder(source_dir, disease, source, nfolds)

    elif layout == "chestxray14":
        if not csv:
            raise ValueError("--csv required for chestxray14 layout")
        df = build_from_nih_chestxray14(source_dir, csv, disease, source, nfolds)

    elif layout == "isic2024":
        if not csv:
            raise ValueError("--csv required for isic2024 layout")
        df = build_from_isic2024(source_dir, csv, disease, source, nfolds)

    elif layout == "medmnist_png":
        df = build_from_medmnist_pngs(source_dir, disease, source, nfolds)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    save_parquet(df, registry_dir, disease)

    # ── Step 4: Validate ──────────────────────────────────────────────────────
    if validate:
        print("[pipeline] Step 4: Validation …")
        from data_pipeline.validation.quality_checks import run_quality_checks
        from data_pipeline.validation.split_validator import SplitValidator

        report = run_quality_checks(df, disease=disease, check_files=True)
        if report["errors"]:
            print(f"[pipeline] ⚠ Quality errors found: {report['errors']}")

        sv = SplitValidator(df, disease=disease)
        sv.validate(strict=False)     # warnings, not hard failure for legacy data

    # ── Step 5: Normalisation stats ───────────────────────────────────────────
    if compute_stats:
        print("[pipeline] Step 5: Computing normalisation statistics …")
        from data_pipeline.preprocessing.normalization import (
            compute_stats as _compute_stats,
        )
        train_df     = df[df["split"] == "train"]
        stats_dir    = str(Path(registry_dir) / "stats")
        _compute_stats(
            image_paths=train_df["image_path"].tolist(),
            disease=disease,
            cache_dir=stats_dir,
            target_size=target_size,
        )

    # ── Step 6: Build shards ──────────────────────────────────────────────────
    if build_shards:
        print("[pipeline] Step 6: Building WebDataset shards …")
        from data_pipeline.preprocessing.build_shards import build_all_shards
        build_all_shards(
            disease=disease,
            registry_dir=registry_dir,
            shards_dir=shards_dir,
            target_size=target_size,
            samples_per_shard=samples_per_shard,
            nfolds=nfolds,
        )

    print(f"\n[pipeline] ✓ Pipeline complete for '{disease}'")
    print(f"  Registry : {parquet_path}")
    return {"parquet": str(parquet_path)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="BDA Multi-Disease Data Pipeline")
    p.add_argument("--disease",           required=True)
    p.add_argument("--source_dir",        default=None)
    p.add_argument("--layout",
                   choices=["imagefolder", "chestxray14", "isic2024",
                             "medmnist_png", "medmnist_hf"],
                   default="imagefolder")
    p.add_argument("--source",            default="local")
    p.add_argument("--nfolds",            type=int, default=5)
    p.add_argument("--registry_dir",      default="registry/")
    p.add_argument("--shards_dir",        default="shards/")
    p.add_argument("--target_size",       type=int, default=224)
    p.add_argument("--samples_per_shard", type=int, default=1000)
    p.add_argument("--csv",               default=None)
    p.add_argument("--hf_dataset",        default="albertvillanova/medmnist")
    p.add_argument("--hf_subset",         default=None)
    p.add_argument("--staging_dir",       default=None)
    p.add_argument("--no_shards",         action="store_true")
    p.add_argument("--no_stats",          action="store_true")
    p.add_argument("--no_validate",       action="store_true")
    p.add_argument("--force",             action="store_true",
                   help="Re-run even if parquet already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        disease=args.disease,
        source_dir=args.source_dir,
        layout=args.layout,
        source=args.source,
        nfolds=args.nfolds,
        registry_dir=args.registry_dir,
        shards_dir=args.shards_dir,
        target_size=args.target_size,
        samples_per_shard=args.samples_per_shard,
        csv=args.csv,
        hf_dataset=args.hf_dataset,
        hf_subset=args.hf_subset,
        staging_dir=args.staging_dir,
        compute_stats=not args.no_stats,
        build_shards=not args.no_shards,
        validate=not args.no_validate,
        skip_if_exists=not args.force,
    )
