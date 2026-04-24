import io, sys, zipfile, base64
from pathlib import Path
from typing import List
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from contextlib import asynccontextmanager
import spaces
from huggingface_hub import snapshot_download

from package.explainability import explain_image
from package.visualization import cam_statistics, batch_summary
from package.config import (
    MODEL_NAME, ROOT, SAVE_DIR, WEIGHTS_PATH,
    CLASS_NAMES_PATH, CLASS_WEIGHTS_PATH,
    IMG_SIZE, DEVICE
)


# ─── Startup ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="Alzheimer MRI Classifier", lifespan=lifespan)

_model = None
_class_names: List[str] = []
_transform = None
_class_weights_tensor = None


def load_model():
    global _model, _class_names, _transform, _class_weights_tensor

    if not WEIGHTS_PATH.exists():
        print("[INFO] Downloading weights from Hugging Face Hub...")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=os.environ["HF_MODEL_REPO"],
            repo_type="model",
            local_dir=str(SAVE_DIR),
            token=os.environ.get("HF_TOKEN"),
        )
        print("[INFO] Weights downloaded.")

    _class_names = (
        CLASS_NAMES_PATH.read_text().strip().splitlines()
        if CLASS_NAMES_PATH.exists()
        else ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    )

    if CLASS_WEIGHTS_PATH.exists():
        class_weights = np.load(CLASS_WEIGHTS_PATH)
    else:
        print("[WARN] class_weights.npy not found — using uniform weights.")
        class_weights = np.ones(len(_class_names))
    _class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    model = timm.create_model(model_name=MODEL_NAME, pretrained=False, num_classes=len(_class_names))

    checkpoint = WEIGHTS_PATH
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    _model = model.eval().to(DEVICE)

    _transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[INFO] Model loaded from {checkpoint.name}  |  classes={_class_names}  |  device={DEVICE}")


def _get_predictions(logits: torch.Tensor) -> np.ndarray:
    log_weights = torch.log(_class_weights_tensor + 1e-10)
    logits      = logits + log_weights.view(1, -1)
    return F.softmax(logits, dim=1).cpu().numpy()


@spaces.GPU
def _predict_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        logits = _model(img_tensor.to(DEVICE))
        if isinstance(logits, tuple):
            logits = logits[0]
    return _get_predictions(logits)


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0)


def _predict_image(pil_img: Image.Image) -> dict:
    probs    = _predict_tensor(_pil_to_tensor(pil_img))[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": _class_names[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "probabilities":   {c: float(p) for c, p in zip(_class_names, probs)},
    }


def _aggregate_max_prob(results: List[dict]) -> dict:
    if not results:
        raise HTTPException(400, "No valid images found in uploaded files.")
    agg = {c: 0.0 for c in _class_names}
    for r in results:
        for c, p in r["probabilities"].items():
            if p > agg[c]:
                agg[c] = p
    pred_class = max(agg, key=agg.get)
    return {
        "mode":              "max_probability",
        "predicted_class":   pred_class,
        "max_probabilities": agg,
        "num_images":        len(results),
    }


def _aggregate_mean_prob(results: List[dict]) -> dict:
    if not results:
        raise HTTPException(400, "No valid images found in uploaded files.")
    agg = {c: 0.0 for c in _class_names}
    for r in results:
        for c, p in r["probabilities"].items():
            agg[c] += p
    agg = {c: v / len(results) for c, v in agg.items()}
    pred_class = max(agg, key=agg.get)
    return {
        "mode":               "mean_probability",
        "predicted_class":    pred_class,
        "mean_probabilities": agg,
        "num_images":         len(results),
    }


def _read_images_from_uploads(files: List[UploadFile]) -> List[tuple[Image.Image, str]]:
    """Yield (PIL image, filename) pairs from a list of uploads, expanding ZIPs."""
    out = []
    for uf in files:
        raw = uf.read() if not hasattr(uf, '_body') else uf._body  # sync fallback
        raise RuntimeError("Use the async version below")
    return out


async def _collect_images(files: List[UploadFile]) -> List[tuple]:
    """Async: return list of (pil_image, filename) from uploads + ZIPs."""
    pairs = []
    for uf in files:
        raw = await uf.read()
        if uf.filename and uf.filename.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for name in z.namelist():
                        if name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                            pil = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                            pairs.append((pil, name))
            except Exception as exc:
                raise HTTPException(400, f"Cannot open ZIP '{uf.filename}': {exc}")
        else:
            try:
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                pairs.append((pil, uf.filename or "unknown"))
            except Exception:
                raise HTTPException(400, f"Cannot open file: {uf.filename}")
    return pairs


# Routes
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.get("/classes")
def classes():
    return {"classes": _class_names}


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    mode: str = Form("avg_probability"),
):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    pairs = await _collect_images(files)
    if not pairs:
        raise HTTPException(400, "No valid images were found in the uploaded files.")

    results = []
    for pil, fname in pairs:
        res = _predict_image(pil)
        res["filename"] = fname
        results.append(res)

    if mode == "per_image":
        return JSONResponse({"mode": "per_image", "results": results})
    elif mode == "max_probability":
        agg = _aggregate_max_prob(results)
    else:
        agg = _aggregate_mean_prob(results)

    agg["per_image"] = results
    return JSONResponse(agg)


@app.post("/predict/summary")
async def predict_summary(files: List[UploadFile] = File(...)):
    """
    Run prediction on all images and return batch-level statistics:
    class distribution, mean/std/max probabilities, confidence histogram.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    pairs = await _collect_images(files)
    if not pairs:
        raise HTTPException(400, "No valid images were found.")

    results = []
    for pil, fname in pairs:
        res = _predict_image(pil)
        res["filename"] = fname
        results.append(res)

    summary = batch_summary(results)
    summary["per_image"] = results
    return JSONResponse(summary)


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """
    Run GradCAM + natural-language explanation on a single image.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, f"Cannot open file: {file.filename}")

    pred = _predict_image(pil)
    expl = explain_image(
        pil_img         = pil,
        model           = _model,
        transform       = _transform,
        class_names     = _class_names,
        predicted_class = pred["predicted_class"],
        confidence      = pred["confidence"],
        device          = DEVICE,
    )

    stats = cam_statistics(expl["grayscale_cam"])

    return JSONResponse({
        "filename":        file.filename,
        "predicted_class": pred["predicted_class"],
        "confidence":      pred["confidence"],
        "probabilities":   pred["probabilities"],
        "original_b64":    expl["original_b64"],
        "overlay_b64":     expl["overlay_b64"],
        "text":            expl["text"],
        "region":          expl["region"],
        "cam_stats":       stats,
    })


@app.post("/explain/stats")
async def explain_stats(file: UploadFile = File(...)):
    """
    Return GradCAM spatial statistics without the overlay images —
    useful for lightweight analytics dashboards scanning many files.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, f"Cannot open file: {file.filename}")

    pred = _predict_image(pil)
    expl = explain_image(
        pil_img         = pil,
        model           = _model,
        transform       = _transform,
        class_names     = _class_names,
        predicted_class = pred["predicted_class"],
        confidence      = pred["confidence"],
        device          = DEVICE,
    )
    stats = cam_statistics(expl["grayscale_cam"])

    return JSONResponse({
        "filename":        file.filename,
        "predicted_class": pred["predicted_class"],
        "confidence":      pred["confidence"],
        "probabilities":   pred["probabilities"],
        "region":          expl["region"],
        "cam_stats":       stats,
    })


@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = ROOT / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return FileResponse(str(html_path))