# BDA Data Pipeline — Multi-Disease Medical Imaging

## Overview

The `data_pipeline/` directory implements the Big Data Architecture (BDA) layer
described in `BDA_Architecture_Document.docx`. It sits **entirely between** raw
data sources and the existing ML pipeline. **No existing files in `module/` are
modified.**

```
RAW SOURCES  →  [data_pipeline/]  →  DataLoader  →  module/ (UNCHANGED)  →  Results
```

---

## Directory Structure

```
data_pipeline/
├── __init__.py
├── pipeline.py                  # Top-level orchestrator (run this first)
├── requirements_bda.txt         # Additional dependencies
│
├── ingestion/
│   ├── format_converter.py      # DICOM/NIfTI/NPZ → PNG
│   ├── source_adapters.py       # Kaggle/NIH/ISIC/Local download
│   └── huggingface_adapter.py   # HuggingFace MedMNIST streaming
│
├── registry/
│   ├── schema.py                # Unified column definitions (PyArrow schema)
│   ├── build_metadata.py        # Raw folder → disease.parquet
│   └── disease_registry.py      # Cross-disease Dask queries
│
├── validation/
│   ├── quality_checks.py        # Missing files, corrupt images, schema
│   └── split_validator.py       # Patient-level leakage detection
│
├── preprocessing/
│   ├── normalization.py         # Per-disease mean/std computation
│   └── build_shards.py          # Parquet → WebDataset .tar shards
│
├── loaders/
│   └── webdataset_loader.py     # get_dataloader() drop-in replacement
│
└── tracking/
    └── mlflow_tracker.py        # Replaces master_results.csv
```

Generated directories (created by the pipeline):
```
registry/           # Parquet files (one per disease)
  adni.parquet
  chestxray14.parquet
  stats/            # Normalisation JSON files
    adni_stats.json

shards/             # WebDataset .tar files
  adni/
    train-fold0/shard-000000.tar
    val-fold0/shard-000000.tar
    test/shard-000000.tar

mlruns/             # MLflow experiment store
staging/            # Downloaded raw data
```

---

## Installation

```bash
pip install -r data_pipeline/requirements_bda.txt
```

Minimum required (everything else is optional):
```bash
pip install pyarrow pandas numpy Pillow scikit-learn
```

Optional extras:
```bash
pip install dask[dataframe]   # Cross-disease Dask queries (large scale)
pip install mlflow            # Experiment tracking UI
pip install webdataset        # Streaming tar-shard DataLoader
pip install kaggle            # Kaggle dataset downloads
pip install datasets          # HuggingFace MedMNIST download
pip install pydicom           # DICOM conversion
pip install nibabel           # NIfTI/brain MRI conversion
```

---

## Where Does the Data Come From?

### Dataset 1 — ADNI (your existing data)
**You already have this.** It is the `OriginalDataset/` folder used by the
current pipeline. No download needed.

```
OriginalDataset/
    MildDemented/     *.jpg
    ModerateDemented/ *.jpg
    NonDemented/      *.jpg
    VeryMildDemented/ *.jpg
```

### Dataset 2 — NIH ChestX-ray14
- **License:** Fully open access. No approval required.
- **Download:** Kaggle mirror (requires free Kaggle account + API key)
- **Kaggle slug:** `nih-chest-xrays/data`
- **Official NIH page:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Citation:** Wang et al., CVPR 2017, ChestX-ray8
- ~112k chest X-ray images, 14 pathology labels (multi-label)

Setup Kaggle API:
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/kaggle.json
# Get it from: https://www.kaggle.com/settings → API → Create New Token
```

### Dataset 3 — ISIC 2024 SLICE-3D
- **License:** CC-BY-NC (academic use)
- **Download:** Kaggle competition
- **Kaggle slug:** `isic-2024-challenge`
- ~400k dermoscopy images, binary classification (benign/malignant)

### Dataset 4 — MedMNIST (via HuggingFace)
- **License:** CC-BY-4.0 (individual subsets may vary)
- **HuggingFace repo:** `albertvillanova/medmnist`
- **No registration required**
- Available subsets: `pathmnist`, `dermamnist`, `octmnist`, `pneumoniamnist`,
  `retinamnist`, `breastmnist`, `bloodmnist`, `tissuemnist`,
  `organamnist`, `organcmnist`, `organsmnist`

---

## How to Run

### Phase 1 — ADNI (start here, validates the full pipeline)

```bash
# Step 1: Build registry Parquet
python -m data_pipeline.pipeline \
    --disease adni \
    --source_dir OriginalDataset/ \
    --layout imagefolder \
    --source local \
    --nfolds 5

# Output: registry/adni.parquet, registry/stats/adni_stats.json
# Optional shards: shards/adni/train-fold0/shard-000000.tar ...
```

### Phase 2 — NIH ChestX-ray14

```bash
# Option A: Auto-download via Kaggle
python -m data_pipeline.pipeline \
    --disease chestxray14 \
    --layout chestxray14 \
    --source nihcc \
    --nfolds 5
# (downloads to staging/chestxray14/ automatically)

# Option B: Data already downloaded
python -m data_pipeline.pipeline \
    --disease chestxray14 \
    --source_dir /path/to/chestxray14/images \
    --csv /path/to/Data_Entry_2017.csv \
    --layout chestxray14 \
    --source nihcc \
    --nfolds 5
```

### Phase 3 — ISIC 2024

```bash
python -m data_pipeline.pipeline \
    --disease isic2024 \
    --source_dir /path/to/isic2024/train-image/image \
    --csv /path/to/train-metadata.csv \
    --layout isic2024 \
    --source isic \
    --nfolds 5
```

### Phase 4 — MedMNIST (HuggingFace, auto-download)

```bash
# RetinaMNIST (ordinal, same structure as ADNI)
python -m data_pipeline.pipeline \
    --disease retinamnist \
    --layout medmnist_hf \
    --hf_subset retinamnist \
    --source huggingface \
    --nfolds 5

# PneumoniaMNIST
python -m data_pipeline.pipeline \
    --disease pneumoniamnist \
    --layout medmnist_hf \
    --hf_subset pneumoniamnist \
    --source huggingface \
    --nfolds 5
```

### Skip shards (faster, for small datasets)

```bash
python -m data_pipeline.pipeline \
    --disease adni \
    --source_dir OriginalDataset/ \
    --layout imagefolder \
    --no_shards
```

---

## Integrating with Existing Cross_Validator

### Option A — Minimal change (use RegistryDataset instead of FullDataset)

In `module/cross_validation.py`, replace the DataLoader construction:

```python
# BEFORE (existing code)
from torch.utils.data import DataLoader, Subset
train_loader = DataLoader(Subset(full_dataset, train_idx), ...)
val_loader   = DataLoader(Subset(full_dataset, val_idx), ...)
test_loader  = DataLoader(Subset(full_dataset, test_idx), ...)

# AFTER (BDA integration)
from data_pipeline.loaders.webdataset_loader import get_dataloader

train_loader = get_dataloader(disease="adni", fold=fold, split="train",
                               img_size=224, batch_size=32)
val_loader   = get_dataloader(disease="adni", fold=fold, split="val",
                               img_size=224, batch_size=32)
test_loader  = get_dataloader(disease="adni", fold=fold, split="test",
                               img_size=224, batch_size=32)
```

### Option B — MLflow tracking (replaces CSV append)

At the end of `_run_classifier()` in `cross_validation.py`:

```python
# BEFORE
pd.DataFrame(results_data).to_csv(self.master_file, mode='a', ...)

# AFTER
from data_pipeline.tracking.mlflow_tracker import log_cross_validation_result

log_cross_validation_result(
    model_name=model_name,
    classifier_type=classifier_type,
    disease=disease,
    fold_metrics=fold_metrics,
    final_metrics={"test_recall": final_recall, "test_accuracy": final_accuracy},
    config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
    model_path=checkpoint_path,
)
```

Then query results:
```bash
mlflow ui --backend-store-uri mlruns/
# → http://localhost:5000
```

---

## Cross-Disease Queries (DiseaseRegistry)

```python
from data_pipeline.registry.disease_registry import DiseaseRegistry

reg = DiseaseRegistry("registry/")

# List available diseases
print(reg.available_diseases)  # ['adni', 'chestxray14', 'retinamnist', ...]

# Summary table
reg.summary("adni")

# Cross-disease sample counts
print(reg.cross_disease_summary())

# Class weights for a disease
weights = reg.class_weights("adni")  # np.array([0.5, 0.714, 1.786, 25.0])

# Query specific split + fold
df = reg.query("adni", split="train", fold=0)
```

---

## Validation

```python
from data_pipeline.validation.quality_checks import run_quality_checks
from data_pipeline.validation.split_validator import SplitValidator
import pandas as pd, pyarrow.parquet as pq

df = pq.read_table("registry/adni.parquet").to_pandas()

# Quality checks
report = run_quality_checks(df, disease="adni")

# Split/fold validation
sv = SplitValidator(df, disease="adni")
sv.validate()
print(sv.patient_split_summary())
print(sv.fold_class_distribution())
```

---

## Architecture Design Principles

1. **Additive only** — no files in `module/` are modified
2. **ETL is one-time** — run the pipeline once per dataset; training reads from registry
3. **Fallback chain** — WebDataset shards → RegistryDataset → original FullDataset
4. **Patient-safe splits** — automatic patient-level stratified k-fold
5. **Cross-disease ready** — unified Parquet schema across all datasets
6. **No Hadoop** — Dask on a single multi-core machine for datasets up to ~1M images

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: data_pipeline` | Run from the project root directory |
| `FileNotFoundError: registry/adni.parquet` | Run `python -m data_pipeline.pipeline --disease adni ...` first |
| `kaggle: command not found` | `pip install kaggle` then set up `~/.kaggle/kaggle.json` |
| `ImportError: mlflow` | `pip install mlflow` (optional — CSV fallback is automatic) |
| `ImportError: dask` | `pip install dask[dataframe]` (optional — pandas fallback is automatic) |
| ISIC download asks for competition acceptance | Accept competition rules at kaggle.com/c/isic-2024-challenge |
| Out of memory building shards | Reduce `--samples_per_shard` or `--target_size` |
