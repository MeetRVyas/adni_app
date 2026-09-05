"""
FastAPI backend for the NeuroScan Alzheimer MRI Classifier.

This is the plain-routing deployment target: a single FastAPI app serving
`index.html` plus a small JSON API (`/predict`, `/predict/summary`, `/explain`, `/explain/stats`).
"""

import io
import os
import zipfile
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from package.config import (
    CLASS_NAMES_PATH,
    CLASS_WEIGHTS_PATH,
    DEVICE,
    IMG_SIZE,
    MODEL_NAME,
    ROOT,
    SAVE_DIR,
    WEIGHTS_PATH,
)
from package.explainability import explain_image
from package.visualization import batch_summary, cam_statistics

# Fallback class order only used if swin_class_names.txt is missing; keep in
# sync with the order the checkpoint was actually trained on.
_DEFAULT_CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


class PredictMode(str, Enum):
    """Aggregation strategy for /predict, matching index.html's mode radio values."""
    MAX_PROBABILITY = "max_probability"
    AVG_PROBABILITY = "avg_probability"
    PER_IMAGE = "per_image"


# ─── Model state ─────────────────────────────────────────────────────────────
_model: torch.nn.Module | None = None
_class_names: List[str] = []
_transform: transforms.Compose | None = None
_class_weights_tensor: torch.Tensor | None = None


def load_model() -> None:
    """Download weights (if needed) and load the model, class names, and transform."""
    global _model, _class_names, _transform, _class_weights_tensor

    if not WEIGHTS_PATH.exists():
        print("[INFO] Downloading weights from Hugging Face Hub...")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        repo_id = os.environ["HF_MODEL_REPO"]
        token = os.environ.get("HF_TOKEN")

        for filename in ["swin_progressive_best.pth", "swin_class_names.txt", "class_weights.npy"]:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(SAVE_DIR),
                token=token,
            )

        print("[INFO] Weights downloaded from HF Hub.")

    _class_names = (
        CLASS_NAMES_PATH.read_text().strip().splitlines()
        if CLASS_NAMES_PATH.exists()
        else _DEFAULT_CLASS_NAMES
    )

    if CLASS_WEIGHTS_PATH.exists():
        class_weights = np.load(CLASS_WEIGHTS_PATH)
    else:
        print("[WARN] class_weights.npy not found — using uniform weights.")
        class_weights = np.ones(len(_class_names))
    _class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    model = timm.create_model(model_name=MODEL_NAME, pretrained=False, num_classes=len(_class_names))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
    _model = model.eval().to(DEVICE)

    _transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[INFO] Model loaded from {WEIGHTS_PATH.name}  |  classes={_class_names}  |  device={DEVICE}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Alzheimer MRI Classifier", lifespan=lifespan)


# ─── Inference helpers ───────────────────────────────────────────────────────
def _get_predictions(logits: torch.Tensor) -> np.ndarray:
    """Apply the (log) class-weight prior to the logits, then softmax."""
    log_weights = torch.log(_class_weights_tensor + 1e-10)
    logits = logits + log_weights.view(1, -1)
    return F.softmax(logits, dim=1).cpu().numpy()


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0)


def _predict_image(pil_img: Image.Image) -> dict:
    with torch.inference_mode():
        logits = _model(_pil_to_tensor(pil_img).to(DEVICE))
        if isinstance(logits, tuple):
            logits = logits[0]
    probs = _get_predictions(logits)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": _class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(_class_names, probs)},
    }


def _aggregate_max_prob(results: List[dict]) -> dict:
    agg = {c: 0.0 for c in _class_names}
    for r in results:
        for c, p in r["probabilities"].items():
            if p > agg[c]:
                agg[c] = p
    pred_class = max(agg, key=agg.get)
    return {
        "mode": "max_probability",
        "predicted_class": pred_class,
        "max_probabilities": agg,
        "num_images": len(results),
    }


def _aggregate_mean_prob(results: List[dict]) -> dict:
    agg = {c: 0.0 for c in _class_names}
    for r in results:
        for c, p in r["probabilities"].items():
            agg[c] += p
    agg = {c: v / len(results) for c, v in agg.items()}
    pred_class = max(agg, key=agg.get)
    return {
        "mode": "mean_probability",
        "predicted_class": pred_class,
        "mean_probabilities": agg,
        "num_images": len(results),
    }


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")


async def _collect_images(files: List[UploadFile]) -> List[Tuple[Image.Image, str]]:
    """Read uploads into (PIL image, filename) pairs, transparently expanding ZIPs.

    A malformed ZIP is a hard error (400). A single unreadable *entry* inside
    an otherwise-valid ZIP is skipped rather than aborting the whole batch —
    one corrupt file shouldn't sink an upload of a hundred good ones.
    """
    pairs: List[Tuple[Image.Image, str]] = []
    for uf in files:
        raw = await uf.read()
        if uf.filename and uf.filename.lower().endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
            except zipfile.BadZipFile as exc:
                raise HTTPException(400, f"Cannot open ZIP '{uf.filename}': {exc}") from exc
            with zf:
                for name in zf.namelist():
                    if not name.lower().endswith(_IMAGE_EXTENSIONS):
                        continue
                    try:
                        pil = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
                    except Exception:
                        continue  # skip unreadable entries instead of failing the batch
                    pairs.append((pil, name))
        else:
            try:
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                pairs.append((pil, uf.filename or "unknown"))
            except Exception as exc:
                raise HTTPException(400, f"Cannot open file: {uf.filename}") from exc
    return pairs


def _predict_all(pairs: List[Tuple[Image.Image, str]]) -> List[dict]:
    results = []
    for pil, fname in pairs:
        res = _predict_image(pil)
        res["filename"] = fname
        results.append(res)
    return results


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/classes")
def classes() -> Dict[str, List[str]]:
    return {"classes": _class_names}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    mode: PredictMode = Form(PredictMode.AVG_PROBABILITY),
):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    pairs = await _collect_images(files)
    if not pairs:
        raise HTTPException(400, "No valid images were found in the uploaded files.")

    results = _predict_all(pairs)

    if mode is PredictMode.PER_IMAGE:
        return JSONResponse({"mode": "per_image", "results": results})

    agg = (
        _aggregate_max_prob(results)
        if mode is PredictMode.MAX_PROBABILITY
        else _aggregate_mean_prob(results)
    )
    agg["per_image"] = results
    return JSONResponse(agg)


@app.post("/predict/summary")
async def predict_summary(files: List[UploadFile] = File(...)):
    """Batch-level statistics: class distribution, mean/std/max probabilities, confidence histogram."""
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    pairs = await _collect_images(files)
    if not pairs:
        raise HTTPException(400, "No valid images were found.")

    results = _predict_all(pairs)
    summary = batch_summary(results)
    summary["per_image"] = results
    return JSONResponse(summary)


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """Run GradCAM + natural-language explanation on a single image."""
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot open file: {file.filename}") from exc

    pred = _predict_image(pil)
    expl = explain_image(
        pil_img=pil,
        model=_model,
        transform=_transform,
        class_names=_class_names,
        predicted_class=pred["predicted_class"],
        confidence=pred["confidence"],
        device=DEVICE,
    )
    stats = cam_statistics(expl["grayscale_cam"])

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": pred["predicted_class"],
        "confidence": pred["confidence"],
        "probabilities": pred["probabilities"],
        "original_b64": expl["original_b64"],
        "overlay_b64": expl["overlay_b64"],
        "text": expl["text"],
        "region": expl["region"],
        "cam_stats": stats,
    })


@app.post("/explain/stats")
async def explain_stats(file: UploadFile = File(...)):
    """Same as /explain but skips the overlay images — for lightweight scanning of many files."""
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot open file: {file.filename}") from exc

    pred = _predict_image(pil)
    expl = explain_image(
        pil_img=pil,
        model=_model,
        transform=_transform,
        class_names=_class_names,
        predicted_class=pred["predicted_class"],
        confidence=pred["confidence"],
        device=DEVICE,
    )
    stats = cam_statistics(expl["grayscale_cam"])

    return JSONResponse({
        "filename": file.filename,
        "predicted_class": pred["predicted_class"],
        "confidence": pred["confidence"],
        "probabilities": pred["probabilities"],
        "region": expl["region"],
        "cam_stats": stats,
    })


@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = ROOT / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return FileResponse(str(html_path))