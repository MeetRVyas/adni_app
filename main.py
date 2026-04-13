import io, sys, zipfile
from pathlib import Path
from typing import List
import os

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
import spaces
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME          = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
CLASSIFIER_TYPE     = "progressive"
SAVE_DIR            = ROOT / "saved_models"
WEIGHTS_PATH        = SAVE_DIR / f"swin_{CLASSIFIER_TYPE}_best.pth"
BEST_FOLD_PATH      = SAVE_DIR / f"swin_{CLASSIFIER_TYPE}_best_fold.pth"
CLASS_NAMES_PATH    = SAVE_DIR / "swin_class_names.txt"
CLASS_WEIGHTS_PATH  = SAVE_DIR / "class_weights.npy"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE            = 224

# ─── Load model once at startup ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app : FastAPI) :
    load_model()
    yield

app = FastAPI(title="Alzheimer MRI Classifier", lifespan = lifespan)

_model = None
_class_names: List[str] = []
_transform = None
_class_weights_tensor = None


def load_model():
    global _model, _class_names, _transform, _class_weights_tensor


def load_model():
    global _model, _class_names, _transform, _class_weights_tensor

    # Download weights from HF Hub if not present locally
    if not WEIGHTS_PATH.exists():
        print("[INFO] Downloading weights from Hugging Face Hub...")
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=os.environ["HF_MODEL_REPO"],   # set as Space secret
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

    model = timm.create_model(model_name = MODEL_NAME, pretrained = False, num_classes = len(_class_names))

    checkpoint = WEIGHTS_PATH if WEIGHTS_PATH.exists() else BEST_FOLD_PATH
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    _model = model.eval().to(DEVICE)

    _transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"[INFO] Model loaded from {checkpoint.name}  |  classes={_class_names}  |  device={DEVICE}")


# ─── Inference helpers ───────────────────────────────────────────────────────
def _get_predictions(logits: torch.Tensor, class_weights_tensor=None) -> np.ndarray:
    if class_weights_tensor is not None:
        log_weights = torch.log(class_weights_tensor + 1e-10)
        logits = logits + log_weights.view(1, -1)
    probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()


@spaces.GPU
def _predict_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        logits = _model(img_tensor.to(DEVICE))
        if isinstance(logits, tuple):
            logits = logits[0]
    return _get_predictions(logits, _class_weights_tensor)


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0)


def _predict_image(pil_img: Image.Image) -> dict:
    tensor = _pil_to_tensor(pil_img)
    probs  = _predict_tensor(tensor)[0]          # shape (n_classes,)
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": _class_names[pred_idx],
        "confidence":      float(probs[pred_idx]),
        "probabilities":   {c: float(p) for c, p in zip(_class_names, probs)},
    }


def _aggregate_max_prob(results: List[dict]) -> dict:
    """For each class, take the max probability across all images."""
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
    """For each class, take the mean probability across all images."""
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


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.get("/classes")
def classes():
    return {"classes": _class_names}


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    mode: str = Form("avg_probability"),  # "avg_probability" | "max_probability" | "per_image"
):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Run train_swin.py first.")

    results = []
    for uf in files:
        raw = await uf.read()

        # ── ZIP: unpack and predict each image inside ──
        if uf.filename and uf.filename.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    for name in z.namelist():
                        if name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                            pil = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                            res = _predict_image(pil)
                            res["filename"] = name
                            results.append(res)
            except Exception as exc:
                raise HTTPException(400, f"Cannot open ZIP '{uf.filename}': {exc}")
            continue  # ← skip the image-open block below

        # ── Single image ──
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(400, f"Cannot open file: {uf.filename}")

        res = _predict_image(pil)
        res["filename"] = uf.filename
        results.append(res)

    if not results:
        raise HTTPException(400, "No valid images were found in the uploaded files.")

    if mode == "per_image":
        return JSONResponse({"mode": "per_image", "results": results})
    elif mode == "max_probability":
        agg = _aggregate_max_prob(results)
    else:
        agg = _aggregate_mean_prob(results)

    agg["per_image"] = results
    return JSONResponse(agg)


# ─── Serve the single-page UI ─────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = ROOT / "index.html"   # ← always relative to this file, not CWD
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return FileResponse(str(html_path))