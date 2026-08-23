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
from gradio.data_classes import FileData

from package.explainability import explain_image
from package.visualization import cam_statistics, batch_summary
from package.ood_guard import load_ood_guard, check_images, is_enabled as ood_guard_enabled
from package.config import (
    MODEL_NAME, ROOT, SAVE_DIR, WEIGHTS_PATH,
    CLASS_NAMES_PATH, CLASS_WEIGHTS_PATH,
    IMG_SIZE, DEVICE,
)

print("TORCH:", torch.__version__)
print("TORCH CUDA:", torch.version.cuda)
print("CUDA AVAILABLE:", torch.cuda.is_available())
print("CUDA DEVICE COUNT:", torch.cuda.device_count())


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

# Same eager-load idiom for the OOD guard. Unlike load_model(), a missing
# calibration artifact doesn't crash the app — it disables the gate and
# predictions run without it (see package/ood_guard.py's load_ood_guard).
load_ood_guard()


# ─── Inference helpers ───────────────────────────────────────────────────────
def _get_predictions(logits: torch.Tensor) -> np.ndarray:
    log_weights = torch.log(_class_weights_tensor + 1e-10)
    logits      = logits + log_weights.view(1, -1)
    return F.softmax(logits, dim=1).cpu().numpy()


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0)


def _ood_check_duration(pil_images: List[Image.Image]) -> int:
    return min(60, max(10, 5 + len(pil_images)))


def _ood_check_duration_gated(pil_images: List[Image.Image], *args, **kwargs) -> int:
    return min(60, max(10, 5 + len(pil_images)))


def _run_ood_check(pil_images: List[Image.Image]) -> List[dict]:
    """
    One GPU entry for the whole batch's CLIP forward pass, mirroring
    _run_batch_predict's batching below — runs BEFORE the Swin classifier so
    out-of-distribution inputs (non-MRI images) never reach it. No-op (every
    image passes) if the OOD guard didn't load; see package/ood_guard.py.
    """
    return check_images(pil_images)


def _predict_duration(pil_images: List[Image.Image]) -> int:
    # Dynamic ZeroGPU duration: scales with batch size instead of a fixed
    # worst-case ceiling, so small requests aren't penalised in the quota
    # pre-check or the node-level queue.
    return min(120, max(15, 5 + 2 * len(pil_images)))


# @spaces.GPU(duration=_predict_duration)
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


# @spaces.GPU(duration=30)
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


def _ood_rejected_result(fname: str, ood: dict) -> dict:
    return {
        "filename":        fname,
        "predicted_class": None,
        "confidence":      None,
        "probabilities":   None,
        "ood_rejected":    True,
        "ood_reason":      ood["reason"],
        "ood_scores":      ood["scores"],
    }


# @spaces.GPU(duration=_ood_check_duration_gated)
def _run_gated_batch_predict(pil_images: List[Image.Image], fnames: List[str]) -> List[dict]:
    """
    Runs the OOD gate first, then the Swin classifier only on the images
    that pass it. High-confidence OOD images get an ood_rejected result
    instead of a forced (and potentially confidently wrong) classification;
    images that only trip the softer "warn" tier are still classified, with
    an ood_warning attached alongside the prediction.
    """
    ood = _run_ood_check(pil_images)
    
    keep_idx = [i for i, o in enumerate(ood) if not o["is_rejected"]]
    classified = {}
    if keep_idx:
        batch = _run_batch_predict([pil_images[i] for i in keep_idx])
        classified = dict(zip(keep_idx, batch))

    results = []
    for i, fname in enumerate(fnames):
        o = ood[i]
        if o["is_rejected"]:
            results.append(_ood_rejected_result(fname, o))
        else:
            res = dict(classified[i])
            res["filename"]     = fname
            res["ood_rejected"] = False
            res["ood_warning"]  = {"reason": o["reason"], "scores": o["scores"]} if o["verdict"] == "warn" else None
            results.append(res)
    return results


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


@app.api(name="predict")
def predict(
    files: List[FileData],
    mode: str = "avg_probability",
):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    pil_images: List[Image.Image] = []
    fnames: List[str] = []

    for file_data in files:
        path = file_data.path if hasattr(file_data, "path") else file_data.get("path")
        if not path:
            raise HTTPException(400, "Uploaded file has no local path.")
        try:
            pil_images.append(Image.open(path).convert("RGB"))
            fnames.append(os.path.basename(path))
        except Exception as exc:
            raise HTTPException(400, f"Cannot open file: {path}: {exc}")

    if not pil_images:
        raise HTTPException(400, "No valid images were found in the uploaded files.")

    results = _run_gated_batch_predict(pil_images, fnames)

    if mode == "per_image":
        return {"mode": "per_image", "results": results}

    classified = [r for r in results if not r["ood_rejected"]]
    if not classified:
        return {
            "mode": mode,
            "predicted_class": None,
            "num_images": len(results),
            "ood_rejected_count": len(results),
            "per_image": results,
            "note": "Every uploaded image was rejected by the OOD guardrail — none appear to be brain MRIs.",
        }

    agg = (_aggregate_max_prob(classified)
           if mode == "max_probability"
           else _aggregate_mean_prob(classified))
    agg["num_images"] = len(results)
    agg["ood_rejected_count"] = len(results) - len(classified)
    agg["per_image"] = results
    return agg


@app.api(name="predict_summary")
def predict_summary(files: List[FileData]):
    """
    Run prediction on all images and return batch-level statistics:
    class distribution, mean/std/max probabilities, confidence histogram.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    pil_images = []
    fnames = []
    for file_data in files:
        path = file_data.path if hasattr(file_data, "path") else file_data.get("path")
        if not path:
            raise HTTPException(400, "Uploaded file has no local path.")
        try:
            pil_images.append(Image.open(path).convert("RGB"))
            fnames.append(os.path.basename(path))
        except Exception as exc:
            raise HTTPException(400, f"Cannot open file: {path}: {exc}")
    if not pil_images:
        raise HTTPException(400, "No valid images were found.")
    results    = _run_gated_batch_predict(pil_images, fnames)

    classified = [r for r in results if not r["ood_rejected"]]
    summary = batch_summary(classified)   # {} if every image was OOD-rejected
    summary["num_images"]         = len(results)
    summary["ood_rejected_count"] = len(results) - len(classified)
    summary["per_image"]          = results
    return JSONResponse(summary)


@app.api(name="explain")
def explain(file: FileData):
    """
    Run GradCAM + natural-language explanation on a single image.
    High-confidence OOD images are rejected before GradCAM ever runs, since
    generating a clinical-sounding explanation for a non-MRI input would be
    actively misleading.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    path = file.path if hasattr(file, "path") else file.get("path")
    filename = os.path.basename(path) if path else "unknown"
    if not path:
        raise HTTPException(400, "Uploaded file has no local path.")
    try:
        pil = Image.open(path).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot open file: {filename}: {exc}")

    ood = _run_ood_check([pil])[0]
    if ood["is_rejected"]:
        raise HTTPException(422, detail={
            "filename":     filename,
            "ood_rejected": True,
            "ood_reason":   ood["reason"],
            "ood_scores":   ood["scores"],
        })

    out   = _run_explain(pil)
    pred  = out["pred"]
    expl  = out["expl"]
    stats = cam_statistics(expl["grayscale_cam"])

    return JSONResponse({
        "filename":        filename,
        "predicted_class": pred["predicted_class"],
        "confidence":      pred["confidence"],
        "probabilities":   pred["probabilities"],
        "original_b64":    expl["original_b64"],
        "overlay_b64":     expl["overlay_b64"],
        "text":            expl["text"],
        "region":          expl["region"],
        "cam_stats":       stats,
        "ood_warning":     {"reason": ood["reason"], "scores": ood["scores"]} if ood["verdict"] == "warn" else None,
    })


@app.api(name="explain_stats")
def explain_stats(file: FileData):
    """
    Return GradCAM spatial statistics without the overlay images —
    useful for lightweight analytics dashboards scanning many files.
    """
    if _model is None:
        raise HTTPException(503, "Model not loaded.")

    path = file.path if hasattr(file, "path") else file.get("path")
    filename = os.path.basename(path) if path else "unknown"
    if not path:
        raise HTTPException(400, "Uploaded file has no local path.")
    try:
        pil = Image.open(path).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot open file: {filename}: {exc}")

    ood = _run_ood_check([pil])[0]
    if ood["is_rejected"]:
        raise HTTPException(422, detail={
            "filename":     filename,
            "ood_rejected": True,
            "ood_reason":   ood["reason"],
            "ood_scores":   ood["scores"],
        })

    out   = _run_explain(pil)
    pred  = out["pred"]
    expl  = out["expl"]
    stats = cam_statistics(expl["grayscale_cam"])

    return JSONResponse({
        "filename":        filename,
        "predicted_class": pred["predicted_class"],
        "confidence":      pred["confidence"],
        "probabilities":   pred["probabilities"],
        "region":          expl["region"],
        "cam_stats":       stats,
        "ood_warning":     {"reason": ood["reason"], "scores": ood["scores"]} if ood["verdict"] == "warn" else None,
    })


@app.api(name="ood_check")
def ood_check(file: FileData):
    """
    Run just the OOD guardrail on a single image, without the Swin
    classifier — useful for testing/monitoring the gate on its own.
    """
    path = file.path if hasattr(file, "path") else file.get("path")
    filename = os.path.basename(path) if path else "unknown"
    if not path:
        raise HTTPException(400, "Uploaded file has no local path.")
    try:
        pil = Image.open(path).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Cannot open file: {filename}: {exc}")

    ood = _run_ood_check([pil])[0]
    return {"filename": filename, **ood}


# Routes
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "model_loaded":     _model is not None,
        "ood_guard_loaded": ood_guard_enabled(),
        "device":           DEVICE,
    }


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