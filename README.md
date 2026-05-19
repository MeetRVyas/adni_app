# NeuroScan — Alzheimer's MRI Classifier

> Swin Transformer · Progressive Fine-Tuning · SAM · Focal Loss · EigenCAM · Big Data Architecture Pipeline

A production-grade medical imaging pipeline that classifies brain MRI slices into four Alzheimer's stages, explains every prediction with GradCAM, and plugs into a multi-disease Big Data Architecture layer — all served behind FastAPI.

**Deployed app classifies ADNI (Alzheimer's) only.** The BDA pipeline and multi-disease support are fully implemented — run `train_swin.py --disease chestxray14` to extend to any supported dataset.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Architecture](#architecture)
- [Model — Swin Transformer](#model--swin-transformer)
- [Progressive Fine-Tuning](#progressive-fine-tuning)
- [SAM Optimizer](#sam-optimizer)
- [Focal Loss](#focal-loss)
- [EigenCAM Explainability](#eigencam-explainability)
- [BDA Pipeline](#bda-pipeline)
- [Datasets](#datasets)
- [API Endpoints](#api-endpoints)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)

---

## What This Does

| Capability | Details |
|---|---|
| **Classification** | 4-class Alzheimer's staging: NonDemented · VeryMild · Mild · Moderate |
| **Model** | `swin_base_patch4_window7_224` pretrained on ImageNet-22k |
| **Training** | 3-phase progressive unfreezing + discriminative LRs + SAM (Phase 3) |
| **Loss** | Custom Focal Loss + inverse-frequency class weights |
| **Explainability** | Native GradCAM (no external library) + spatial stats + clinical notes |
| **Data layer** | Unified Parquet registry across ADNI, ChestX-ray14, ISIC2024, MedMNIST |
| **Serving** | FastAPI — batch predict, aggregate stats, per-image explanations |
| **Tracking** | MLflow + CSV fallback |

---

## Architecture

```
RAW SOURCES          BDA PIPELINE              ML PIPELINE           SERVING
─────────────        ────────────────          ───────────           ───────
DICOM / NIfTI  ───▶  ingestion/          ───▶  DataLoader    ───▶  FastAPI
NPZ / JPEG           format_converter          (WebDataset          /predict
                     registry/*.parquet         or Parquet           /explain
                     shards/*.tar               fallback)            /predict/summary
                     normalization
                     validation
                     mlflow_tracker

         ↑ Entirely additive — zero modifications to existing module/ code ↑
```

### Serving Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Classify batch · modes: `max_probability`, `avg_probability`, `per_image` |
| `POST` | `/predict/summary` | Class distribution, confidence histogram, mean/std/max probs |
| `POST` | `/explain` | GradCAM overlay + spatial stats + clinical note (single image) |
| `POST` | `/explain/stats` | Spatial stats only — lightweight, no images returned |
| `GET` | `/classes` | List class names |
| `GET` | `/health` | Model load status + device |
| `GET` | `/` | NeuroScan web UI |

---

## Model — Swin Transformer

`swin_base_patch4_window7_224.ms_in22k_ft_in1k` from `timm`. ~86M parameters.

Swin uses **shifted-window self-attention** — unlike ViT, attention is computed within local windows (7×7) that shift between layers to enable cross-window connections. This gives linear complexity w.r.t. image size and strong spatial locality, which matters for MRI where pathology is spatially structured.

The model produces hierarchical feature maps at 4 scales (H/4, H/8, H/16, H/32), making later stages ideal targets for GradCAM.

**Class-weighted logit biasing at inference:**

```python
# Bias logits by log(class_weight) before softmax
# Equivalent to: softmax(x + log(w)) = w·exp(x) / Σ w·exp(x)
log_weights = torch.log(class_weights_tensor + 1e-10)
logits = logits + log_weights.view(1, -1)
probs  = F.softmax(logits, dim=1)
```

This improves recall on rare classes (especially `ModerateDemented`, ~1% of ADNI) without retraining.

---

## Progressive Fine-Tuning

Training proceeds in three phases, gradually unfreezing the backbone:

```
Phase 1 (5 epochs)    │ Classifier head only    │ LR = 1e-3  │ No SAM │ Patience 5
Phase 2 (10 epochs)   │ Top 50% layers          │ LR = 1e-4  │ No SAM │ Patience 10
Phase 3 (remaining)   │ All layers, disc. LRs   │ LR = 1e-4  │ SAM ✓  │ Patience 20
```

### Discriminative Learning Rates

Each Swin stage gets a different learning rate. Earlier layers encode general ImageNet features — large updates would destroy them.

```
┌─────────────────────────────────────┬───────────┬──────────────┬────────────┐
│ Layer Group                         │ Swin Stage│ LR Mult      │ Eff. LR    │
├─────────────────────────────────────┼───────────┼──────────────┼────────────┤
│ patch_embed + absolute_pos_embed    │ Stem      │ × 1/100      │ 1e-6       │
│ layers[0]   (H/4  tokens,  ~8M p)  │ Stage 1   │ × 1/100      │ 1e-6       │
│ layers[1]   (H/8  tokens, ~17M p)  │ Stage 2   │ × 1/10       │ 1e-5       │
│ layers[2]   (H/16 tokens, ~34M p)  │ Stage 3   │ × 1/3        │ 3.3e-5     │
│ layers[3]   (H/32 tokens, ~34M p)  │ Stage 4   │ × 1.0        │ 1e-4       │
│ norm                                │ Final LN  │ × 1.0        │ 1e-4       │
│ head        (4-class linear)        │ Classifier│ × 10         │ 1e-3       │
└─────────────────────────────────────┴───────────┴──────────────┴────────────┘
```

The head gets 10× because it starts from random initialization. Patch embedding gets 1/100× because it already encodes useful low-level features from ImageNet-22k pretraining.

---

## SAM Optimizer (Used in phase 3 only)
- **SAM**: Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021. [[paper]](https://arxiv.org/abs/2010.01412)

Standard SGD/Adam minimizes `L(θ)`. SAM minimizes `max_{‖ε‖≤ρ} L(θ + ε)` — it finds a perturbation that maximises loss locally, then descends from that adversarial point. The result is convergence to **flat minima** with lower curvature, which generalize better under distribution shift.

```
Standard:  θ → gradient at θ → update θ          (1 forward + 1 backward)
SAM:       θ → gradient at θ
             → θ̃ = θ + ρ · g/‖g‖                 (perturb to adversarial point)
             → gradient at θ̃
             → restore θ, apply gradient from θ̃   (2 forward + 2 backward)
```

```python
# package/optimizer.py — key logic
def first_step(self, zero_grad=False):
    grad_norm = self._grad_norm()                     # global l2 norm
    scale = group["rho"] / (grad_norm + 1e-12)
    for p in group["params"]:
        self.state[p]["old_p"] = p.data.clone()       # save θ
        p.add_(p.grad * scale)                        # θ̃ = θ + ρ·g/‖g‖

def second_step(self, zero_grad=False):
    for p in group["params"]:
        p.data = self.state[p]["old_p"]               # restore θ
    self.base_optimizer.step()                        # update using ∇L(θ̃)
```

SAM is skipped in Phases 1 and 2 — early gradients are noisy and the double pass doubles wall time when the head hasn't converged yet.

---

## Focal Loss
- **Focal Loss**: Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017. [[paper]](https://arxiv.org/abs/1708.02002)

ADNI has severe class imbalance — `ModerateDemented` is roughly 1% of samples. Two mechanisms address this:

**1. Focal Loss** (Lin et al., 2017) down-weights easy, well-classified examples so the model focuses on hard/rare samples:

```
FL(pt) = α · (1 − pt)^γ · CE(pt)
```

When `pt = 0.9` (model is confident and correct), `(1−0.9)^2 = 0.01` — the loss contribution is almost zero. When `pt = 0.1` (confused), `(1−0.1)^2 = 0.81` — near full gradient.

```python
# package/loss.py
def forward(self, inputs, targets):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    p_t     = torch.exp(-ce_loss)                  # prob of true class
    focal   = self.alpha * (1 - p_t)**self.gamma * ce_loss

    # Per-sample inverse-frequency class weight
    weight_per_sample = self.weights[targets]
    focal = focal * weight_per_sample
    return focal.mean()
```

**2. Inverse-frequency class weights** from the BDA registry applied both inside the loss and to bias logits at inference (see above).

| γ | Effect on p=0.9 example | Effect on p=0.1 example |
|---|---|---|
| 0 | Standard CE (no modulation) | Standard CE |
| 1 | 10× reduction | 1.1× reduction |
| 2 | 100× reduction | 1.2× reduction |
| 5 | 100,000× reduction | 1.6× reduction |

Default: `α=1.0, γ=2.0`.

---

## EigenCAM Explainability

No `pytorch-grad-cam` dependency. Fully native hook-based GradCAM in `package/explainability.py`.

**Target layer:** `model.layers[-1].blocks[-1].norm2` — the final norm in Swin's last stage. This is the last point where spatial token structure is preserved before global pooling.

**Swin-specific reshape:** Swin outputs tokens as `(B, N, C)` sequences, not `(B, C, H, W)` feature maps. A reshape transform is applied before computing spatial statistics:

```python
def _reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)    # (B, C, H, W)

# GradCAM computation
weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # channel importance
cam     = torch.sum(weights * fmaps, dim=1)             # weighted sum
cam     = F.relu(cam)
cam     = cam / (cam.max() + 1e-7)                      # [0, 1]
```

### Spatial Statistics (returned by `/explain`)

| Statistic | Description |
|---|---|
| `peak` | Maximum activation value |
| `mean` | Mean activation across the CAM |
| `peak_mean_ratio` | Focus indicator — high = concentrated hotspot |
| `entropy` | Shannon entropy — low = focused, high = diffuse |
| `focus_score` | Fraction of total activation in top-10% pixels |
| `centroid_x/y` | Normalised centre of mass of top-10% activations |
| `spread_x/y` | Spatial spread of top activations |
| `quadrant_weights` | Activation fraction per image quadrant (TL/TR/BL/BR) |
| `num_hotspots` | BFS-counted connected components on 32×32 binary map |

### Region Heuristic

`_estimate_region()` classifies CAM into anatomical-sounding regions (`superior`, `inferior`, `central`, `left temporal`, `right temporal`, `diffuse`) based on the centroid of top-10% activations. Used for the natural-language clinical note only — not a true anatomical atlas alignment.

---

## BDA Pipeline

The Big Data Architecture layer lives in `data_pipeline/` and is **entirely additive** — zero modifications to existing `module/` code.

```
data_pipeline/
├── pipeline.py                 # Top-level orchestrator (run this)
├── ingestion/
│   ├── format_converter.py     # DICOM / NIfTI / NPZ → PNG
│   ├── source_adapters.py      # Kaggle / NIH / ISIC / Local
│   └── huggingface_adapter.py  # HuggingFace MedMNIST streaming
├── registry/
│   ├── schema.py               # Unified PyArrow schema
│   ├── build_metadata.py       # Raw folder → disease.parquet
│   └── disease_registry.py     # Cross-disease Dask queries
├── validation/
│   ├── quality_checks.py       # Missing files, corrupt images, schema
│   └── split_validator.py      # Patient-level leakage detection
├── preprocessing/
│   ├── normalization.py        # Welford online mean/std
│   └── build_shards.py         # Parquet → WebDataset .tar shards
├── loaders/
│   └── webdataset_loader.py    # get_dataloader() drop-in replacement
└── tracking/
    └── mlflow_tracker.py       # Replaces master_results.csv
```

### Pipeline Steps

```
1. Ingest      Source adapters download/stage raw data. Marker file prevents re-download.
       ↓
2. Convert     DICOM (pydicom + windowing) · NIfTI (nibabel, per-slice) · NPZ → 8-bit RGB PNG
       ↓
3. Registry    build_metadata.py → registry/<disease>.parquet (unified PyArrow schema)
               Patient-level stratified k-fold assignment prevents data leakage.
       ↓
4. Validate    Schema conformance · file existence · label consistency
               SplitValidator: no patient_id appears in more than one split
       ↓
5. Stats       Welford online algorithm → registry/stats/<disease>_stats.json
               Falls back to ImageNet defaults if not enough valid images.
       ↓
6. Shards      Parquet → shards/<disease>/train-fold{k}/shard-{n}.tar
               Each sample: <key>.jpg + <key>.json. Streaming-friendly for large datasets.
```

### Unified Parquet Schema

```
Core (required for all diseases):
  image_path      string    Absolute path — single source of truth
  label           int32     0-based class index
  label_name      string    Human-readable class name
  split           string    train | val | test
  fold            int32     K-fold index; -1 for fixed-split datasets
  patient_id      string    Used for leakage-free splitting
  disease         string    Dataset tag (adni, chestxray14, ...)
  source          string    kaggle | huggingface | local | nihcc | isic
  image_width     int32     Pixel width
  image_height    int32     Pixel height

Extended (optional, per-disease):
  is_multilabel   bool      Multi-label flag (ChestX-ray14)
  additional_labels string  JSON list of extra label strings
  age             float32   Subject age if available
  sex             string    M | F | U
  original_format string    DICOM | NIfTI | NPZ | PNG | JPEG
  checksum        string    SHA-256 hex (opt-in)
```

### DataLoader Fallback Chain

```
get_dataloader() tries in order:
  1. WebDataset shards   (shards/<disease>/<partition>/shard-*.tar)
  2. RegistryDataset     (reads image_path from Parquet directly)
  3. Original FullDataset (legacy ImageFolder — when BDA registry absent)
```

---

## Datasets

| Dataset | Disease | Images | Classes | License | Source |
|---|---|---|---|---|---|
| ADNI | Alzheimer's MRI | ~6,400 | 4 | Restricted | Local (you provide) |
| NIH ChestX-ray14 | 14 chest pathologies | 112,120 | 14 | Open | `nih-chest-xrays/data` (Kaggle) |
| ISIC 2024 SLICE-3D | Skin lesion (benign/malignant) | ~400,000 | 2 | CC-BY-NC | `isic-2024-challenge` (Kaggle) |
| MedMNIST (11 subsets) | Multi-organ / multi-modality | varies | 2–11 | CC-BY-4.0 | `albertvillanova/medmnist` (HF) |

ChestX-ray14 is multi-label — primary finding goes in `label`, all findings in `additional_labels` as a JSON list. ADNI and ISIC have real patient IDs, so `SplitValidator` enforces no patient appears across splits.

---

## Quickstart

### Install

```bash
# Required
pip install pyarrow pandas numpy Pillow scikit-learn timm fastapi uvicorn torch torchvision

# Optional
pip install webdataset mlflow dask[dataframe] kaggle datasets pydicom nibabel
```

### Build registry and train

```bash
# Step 1: build BDA registry (one-time ETL)
python -m data_pipeline.pipeline \
    --disease adni \
    --source_dir OriginalDataset/ \
    --layout imagefolder \
    --nfolds 5

# Step 2: train
python train_swin.py --disease adni

# Skip shards for small datasets (faster ETL)
python -m data_pipeline.pipeline --disease adni \
    --source_dir OriginalDataset/ --no_shards

# Force legacy mode (no BDA, original FullDataset)
python train_swin.py --no_bda
```

### Other datasets

```bash
# NIH ChestX-ray14 (auto-download via Kaggle API)
python -m data_pipeline.pipeline \
    --disease chestxray14 --layout chestxray14 --source nihcc --nfolds 5

# ISIC 2024 (data already downloaded)
python -m data_pipeline.pipeline \
    --disease isic2024 \
    --source_dir /path/to/isic2024/train-image/image \
    --csv /path/to/train-metadata.csv \
    --layout isic2024 --source isic --nfolds 5

# MedMNIST via HuggingFace (no registration needed)
python -m data_pipeline.pipeline \
    --disease retinamnist \
    --layout medmnist_hf \
    --hf_subset retinamnist \
    --source huggingface --nfolds 5
```

### Serve

```bash
export HF_MODEL_REPO=your-user/your-model-repo
export HF_TOKEN=hf_...

uvicorn main:app --host 0.0.0.0 --port 7860
```

### Integrate BDA into existing training code

```python
# Replace DataLoader construction in cross_validation.py
from data_pipeline.loaders.webdataset_loader import get_dataloader

# BEFORE
train_loader = DataLoader(Subset(full_dataset, train_idx), ...)
val_loader   = DataLoader(Subset(full_dataset, val_idx), ...)

# AFTER
train_loader = get_dataloader(disease="adni", fold=fold, split="train",
                               img_size=224, batch_size=32)
val_loader   = get_dataloader(disease="adni", fold=fold, split="val",
                               img_size=224, batch_size=32)
```

### Query the registry

```python
from data_pipeline.registry.disease_registry import DiseaseRegistry

reg = DiseaseRegistry("registry/")
print(reg.available_diseases)           # ['adni', 'retinamnist', ...]
print(reg.cross_disease_summary())      # total/train/val/test per disease
weights = reg.class_weights("adni")     # inverse-frequency weights as np.array
df = reg.query("adni", split="train", fold=0)
```

### MLflow

```bash
mlflow ui --backend-store-uri mlruns/
# → http://localhost:5000
```

```python
from data_pipeline.tracking.mlflow_tracker import MLflowTracker
best = MLflowTracker.get_best_run("adni_cross_validation", metric="test_recall")
```

---

## Project Structure

```
├── main.py                         # FastAPI app + model loading + inference
├── train_swin.py                   # Training script (BDA + legacy modes)
├── index.html                      # NeuroScan web UI
│
├── package/
│   ├── config.py                   # All hyperparameters and paths
│   ├── model.py                    # ProgressiveClassifier (3-phase training)
│   ├── loss.py                     # FocalLoss with class weights
│   ├── optimizer.py                # SAM (Sharpness-Aware Minimization)
│   ├── layer_groups.py             # Swin discriminative LR groups
│   ├── explainability.py           # Native GradCAM + region heuristic + templates
│   ├── visualization.py            # cam_statistics() + batch_summary()
│   ├── utils.py                    # FullDataset, Logger, ensure_pipeline_ready()
│   └── pipeline.py                 # run_pipeline() entry point (legacy compat)
│
├── data_pipeline/
│   ├── pipeline.py                 # BDA orchestrator CLI
│   ├── ingestion/
│   │   ├── format_converter.py     # DICOM / NIfTI / NPZ → PNG
│   │   ├── source_adapters.py      # Download adapters (Kaggle, Local)
│   │   └── huggingface_adapter.py  # HuggingFace streaming
│   ├── registry/
│   │   ├── schema.py               # PyArrow schema + validation
│   │   ├── build_metadata.py       # Raw data → Parquet (4 layout types)
│   │   └── disease_registry.py     # DiseaseRegistry (Dask + pandas)
│   ├── validation/
│   │   ├── quality_checks.py       # 6 data quality checks
│   │   └── split_validator.py      # Patient leakage detection
│   ├── preprocessing/
│   │   ├── normalization.py        # Welford online stats
│   │   └── build_shards.py         # Parquet → WebDataset .tar
│   ├── loaders/
│   │   └── webdataset_loader.py    # get_dataloader() with fallback chain
│   └── tracking/
│       └── mlflow_tracker.py       # MLflow wrapper + CSV fallback
│
├── registry/                       # Generated: disease.parquet files
├── shards/                         # Generated: WebDataset .tar shards
├── mlruns/                         # Generated: MLflow experiment store
├── staging/                        # Generated: downloaded raw data
└── output/
    └── saved_models/
        ├── swin_model_weights.pth
        ├── swin_class_names.txt
        └── class_weights.npy
```
---

> **Clinical disclaimer:** NeuroScan outputs are not diagnostic conclusions. All predictions require clinical correlation and review by a qualified specialist.
