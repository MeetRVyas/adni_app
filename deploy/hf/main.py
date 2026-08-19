# `spaces` must be imported before anything that touches torch/CUDA — it
# monkey-patches torch's CUDA entrypoints so `.to("cuda")` calls at module
# scope succeed even though no physical GPU is attached to this process.
# (No-op outside Hugging Face's ZeroGPU hardware, so this is safe locally too.)
import spaces

import io
import os
import zipfile
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from huggingface_hub import hf_hub_download
from gradio import Server

from package.explainability import explain_image
from package.visualization import cam_statistics, batch_summary
from package.config import (
    MODEL_NAME, ROOT, SAVE_DIR, WEIGHTS_PATH,
    CLASS_NAMES_PATH, CLASS_WEIGHTS_PATH,
    IMG_SIZE, DEVICE,
)


# ─── App ───────────────────────────────────────────────────────────────────
# gradio.Server is a FastAPI app with Gradio's queue/ZeroGPU engine wired in,
# so every route below is a normal FastAPI route — same signatures, same
# request/response shapes as the original main.py.
app = Server(title="Alzheimer MRI Classifier")

_model = None
_class_names: List[str] = []
_transform = None
_class_weights_tensor = None


def load_model():
    global _model, _class_names, _transform, _class_weights_tensor

    if not WEIGHTS_PATH.exists():
        print("[INFO] Downloading weights from Hugging Face Hub...")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        repo_id = os.environ["HF_MODEL_REPO"]
        token   = os.environ.get("HF_TOKEN")

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


# ZeroGPU wants models loaded — and moved to "cuda" — eagerly at module
# scope, not lazily on first request. `DEVICE` above already resolves to
# "cuda" under ZeroGPU (torch.cuda.is_available() is monkey-patched to
# report True), so this is the standard idiom, not a lazy-load pattern.
load_model()


# ─── Inference helpers ───────────────────────────────────────────────────────
def _get_predictions(logits: torch.Tensor) -> np.ndarray:
    log_weights = torch.log(_class_weights_tensor + 1e-10)
    logits      = logits + log_weights.view(1, -1)
    return F.softmax(logits, dim=1).cpu().numpy()


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0)


def _predict_duration(pil_images: List[Image.Image]) -> int:
    # Dynamic ZeroGPU duration: scales with batch size instead of a fixed
    # worst-case ceiling, so small requests aren't penalised in the quota
    # pre-check or the node-level queue.
    return min(120, max(15, 5 + 2 * len(pil_images)))


@spaces.GPU(duration=_predict_duration)
def _run_batch_predict(pil_images: List[Image.Image]) -> List[dict]:
    """
    Runs the whole batch through the model in a single GPU entry (one fork,
    one CUDA re-attach) rather than one entry per image.
    """
    tensors = torch.cat([_pil_to_tensor(p) for p in pil_images], dim=0)
    with torch.inference_mode():
        logits = _model(tensors.to(DEVICE))
        if isinstance(logits, tuple):
            logits = logits[0]
    probs_batch = _get_predictions(logits)

    results = []
    for probs in probs_batch:
        pred_idx = int(np.argmax(probs))
        results.append({
            "predicted_class": _class_names[pred_idx],
            "confidence":      float(probs[pred_idx]),
            "probabilities":   {c: float(p) for c, p in zip(_class_names, probs)},
        })
    return results


@spaces.GPU(duration=30)
def _run_explain(pil_img: Image.Image) -> dict:
    """
    One GPU entry covering both the classification forward pass and the
    GradCAM forward+backward pass, so /explain only forks once.
    """
    tensor = _pil_to_tensor(pil_img)
    with torch.inference_mode():
        logits = _model(tensor.to(DEVICE))
        if isinstance(logits, tuple):
            logits = logits[0]
    probs    = _get_predictions(logits)[0]
    pred_idx = int(np.argmax(probs))
    pred = {
        "predicted_class": _class_names[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "probabilities":   {c: float(p) for c, p in zip(_class_names, probs)},
    }

    # NativeGradCAM needs grad enabled for its backward hook; inference_mode
    # above has already exited by this point so this is unaffected.
    expl = explain_image(
        pil_img         = pil_img,
        model           = _model,
        transform       = _transform,
        class_names     = _class_names,
        predicted_class = pred["predicted_class"],
        confidence      = pred["confidence"],
        device          = DEVICE,
    )
    return {"pred": pred, "expl": expl}


async def _collect_images(files: List[UploadFile]) -> List[Tuple[Image.Image, str]]:
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


# ─── Routes — identical paths, request shapes, and response shapes ──────────
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

    pil_images = [p for p, _ in pairs]
    fnames     = [f for _, f in pairs]
    batch      = _run_batch_predict(pil_images)

    results = []
    for res, fname in zip(batch, fnames):
        res = dict(res)
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

    pil_images = [p for p, _ in pairs]
    fnames     = [f for _, f in pairs]
    batch      = _run_batch_predict(pil_images)

    results = []
    for res, fname in zip(batch, fnames):
        res = dict(res)
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

    out   = _run_explain(pil)
    pred  = out["pred"]
    expl  = out["expl"]
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

    out   = _run_explain(pil)
    pred  = out["pred"]
    expl  = out["expl"]
    stats = cam_statistics(expl["grayscale_cam"])

    return JSONResponse({
        "filename":        file.filename,
        "predicted_class": pred["predicted_class"],
        "confidence":      pred["confidence"],
        "probabilities":   pred["probabilities"],
        "region":          expl["region"],
        "cam_stats":       stats,
    })


# Routes
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = ROOT / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return FileResponse(str(html_path))


# Unconditional (not gated behind `if __name__ == "__main__"`) so this also
# works under `gradio app.py` hot-reload, matching standard Gradio app.py
# convention — and it's what Hugging Face Spaces' Gradio SDK runs directly.
app.launch()