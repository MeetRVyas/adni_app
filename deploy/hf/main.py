"""Standalone Gradio Blocks app for the NeuroScan Alzheimer MRI Classifier, built for Hugging Face's ZeroGPU."""

# `spaces` must be imported before anything that touches torch/CUDA — it
# monkey-patches torch's CUDA entrypoints so `.to("cuda")` calls at module
# scope succeed even though no physical GPU is attached to this process.
import spaces

import base64
import io
import os
import re
import zipfile
from html import escape as _esc
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from package.config import (
    CLASS_NAMES_PATH,
    CLASS_WEIGHTS_PATH,
    DEVICE,
    IMG_SIZE,
    MODEL_NAME,
    SAVE_DIR,
    WEIGHTS_PATH,
)
from package.explainability import explain_image
from package.ood_guard import check_images
from package.ood_guard import is_enabled as ood_guard_enabled
from package.ood_guard import load_ood_guard
from package.visualization import batch_summary, cam_statistics

print("TORCH:", torch.__version__)
print("TORCH CUDA:", torch.version.cuda)
print("CUDA AVAILABLE:", torch.cuda.is_available())
print("CUDA DEVICE COUNT:", torch.cuda.device_count())

_DEFAULT_CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif")

RESULTS_COLUMNS = ["Image", "Class", "Confidence", "Status"]
STATS_COLUMNS = ["Class", "Mean", "Std", "Max"]

_MODE_LABELS = {
    "Mean probability": "avg_probability",
    "Max probability": "max_probability",
    "Per image": "per_image",
}


# ─── Model state ─────────────────────────────────────────────────────────────
_model = None
_class_names: List[str] = []
_transform: Optional[transforms.Compose] = None
_class_weights_tensor: Optional[torch.Tensor] = None


def load_model() -> None:
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


# ZeroGPU wants models loaded — and moved to "cuda" — eagerly at module scope, not lazily on first request.
# `DEVICE` resolves to "cuda" under ZeroGPU.
load_model()
load_ood_guard()


# ─── GPU-decorated pipeline (unchanged from the previous gradio_app.py) ──────
def _get_predictions(logits: torch.Tensor) -> np.ndarray:
    log_weights = torch.log(_class_weights_tensor + 1e-10)
    logits = logits + log_weights.view(1, -1)
    return F.softmax(logits, dim=1).cpu().numpy()


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _transform(pil_img.convert("RGB")).unsqueeze(0)


def _ood_check_duration(pil_images: List[Image.Image]) -> int:
    return min(60, max(10, 5 + len(pil_images)))


def _ood_check_duration_gated(pil_images: List[Image.Image], *args, **kwargs) -> int:
    return min(60, max(10, 5 + len(pil_images)))


@spaces.GPU(duration=_ood_check_duration)
def _run_ood_check(pil_images: List[Image.Image]) -> List[dict]:
    """One GPU entry for the whole batch's CLIP forward pass, mirroring
    _run_batch_predict's batching below — runs BEFORE the Swin classifier so
    out-of-distribution inputs (non-MRI images) never reach it. No-op (every
    image passes) if the OOD guard didn't load; see package/ood_guard.py."""
    return check_images(pil_images)


# Intentionally NOT @spaces.GPU-decorated: only ever called from inside
# _run_gated_batch_predict, which is decorated below. One real GPU entry
# point per pipeline is simpler to reason about than nested GPU calls.
def _run_batch_predict(pil_images: List[Image.Image]) -> List[dict]:
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
            "confidence": float(probs[pred_idx]),
            "probabilities": {c: float(p) for c, p in zip(_class_names, probs)},
        })
    return results


@spaces.GPU(duration=30)
def _run_explain(pil_img: Image.Image) -> dict:
    """Single GPU entry for the Explain tab: the OOD gate, the classification
    forward pass, and the GradCAM forward+backward pass all run in one fork."""
    ood = check_images([pil_img])[0]
    if ood["is_rejected"]:
        return {"ood_rejected": True, "ood": ood}

    tensor = _pil_to_tensor(pil_img)
    with torch.inference_mode():
        logits = _model(tensor.to(DEVICE))
        if isinstance(logits, tuple):
            logits = logits[0]
    probs = _get_predictions(logits)[0]
    pred_idx = int(np.argmax(probs))
    pred = {
        "predicted_class": _class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(_class_names, probs)},
    }

    # NativeGradCAM needs grad enabled for its backward hook; inference_mode
    # above has already exited by this point so this is unaffected.
    expl = explain_image(
        pil_img=pil_img,
        model=_model,
        transform=_transform,
        class_names=_class_names,
        predicted_class=pred["predicted_class"],
        confidence=pred["confidence"],
        device=DEVICE,
    )
    return {"ood_rejected": False, "ood": ood, "pred": pred, "expl": expl}


def _ood_rejected_result(fname: str, ood: dict) -> dict:
    return {
        "filename": fname,
        "predicted_class": None,
        "confidence": None,
        "probabilities": None,
        "ood_rejected": True,
        "ood_reason": ood["reason"],
        "ood_scores": ood["scores"],
    }


@spaces.GPU(duration=_ood_check_duration_gated)
def _run_gated_batch_predict(pil_images: List[Image.Image], fnames: List[str]) -> List[dict]:
    """Runs the OOD gate first, then the Swin classifier only on images that
    pass it. High-confidence OOD images get an ood_rejected result instead
    of a forced classification; soft "warn"-tier images are still classified,
    with an ood_warning attached alongside the prediction."""
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
            res["filename"] = fname
            res["ood_rejected"] = False
            res["ood_warning"] = {"reason": o["reason"], "scores": o["scores"]} if o["verdict"] == "warn" else None
            results.append(res)
    return results


@spaces.GPU(duration=15)
def _gpu_smoke_test_inner() -> dict:
    """Minimal, isolated ZeroGPU check — confirms a real CUDA device gets
    attached inside a fresh worker fork. If this fails, the problem is with
    the Space's ZeroGPU allocation/hardware/quota, not the model pipeline."""
    t = torch.tensor([1.0]).cuda()
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0),
        "tensor_ok": bool((t + 1).item() == 2.0),
    }


# ─── Plain CPU helpers (aggregation, uploads, formatting) ────────────────────
def _aggregate_max(results: List[dict]) -> dict:
    agg = {c: 0.0 for c in _class_names}
    for r in results:
        for c, p in r["probabilities"].items():
            if p > agg[c]:
                agg[c] = p
    return {"predicted_class": max(agg, key=agg.get), "probabilities": agg}


def _aggregate_mean(results: List[dict]) -> dict:
    agg = {c: 0.0 for c in _class_names}
    for r in results:
        for c, p in r["probabilities"].items():
            agg[c] += p
    agg = {c: v / len(results) for c, v in agg.items()}
    return {"predicted_class": max(agg, key=agg.get), "probabilities": agg}


def _pairs_from_paths(paths: Optional[List[str]]) -> Dict[str, Image.Image]:
    """Rebuild {filename: PIL image} from gr.File's current path list,
    transparently expanding ZIPs. A duplicate filename (e.g. across two
    ZIPs) overwrites the earlier one — a known, documented simplification
    of keying purely by name rather than by upload batch."""
    files: Dict[str, Image.Image] = {}
    if not paths:
        return files
    for p in paths:
        base = os.path.basename(p)
        if base.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(p) as zf:
                    for name in zf.namelist():
                        if not name.lower().endswith(_IMAGE_EXTENSIONS):
                            continue
                        try:
                            files[name] = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
                        except Exception:
                            continue  # skip unreadable entries, don't fail the whole zip
            except zipfile.BadZipFile:
                continue
        else:
            try:
                files[base] = Image.open(p).convert("RGB")
            except Exception:
                continue
    return files


def _b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def _package_explanation(out: dict) -> dict:
    pred, expl, ood = out["pred"], out["expl"], out["ood"]
    stats = cam_statistics(expl["grayscale_cam"])
    return {
        "original": _b64_to_pil(expl["original_b64"]),
        "overlay": _b64_to_pil(expl["overlay_b64"]),
        "predicted_class": pred["predicted_class"],
        "confidence": pred["confidence"],
        "probabilities": pred["probabilities"],
        "text": expl.get("text"),
        "region": expl.get("region"),
        "cam_stats": stats,
        "ood_warning": ood if ood.get("verdict") == "warn" else None,
    }


# ─── Risk palette (mirrors index.html's RISK_MAP / risk-chip colors) ─────────
_RISK_MAP = {
    "NonDemented": {"level": "safe", "label": "Safe"},
    "VeryMildDemented": {"level": "low", "label": "Very Mild"},
    "MildDemented": {"level": "mid", "label": "Mild"},
    "ModerateDemented": {"level": "high", "label": "Moderate"},
}
_RISK_HEX = {"safe": "#4caf7d", "low": "#c9a84c", "mid": "#c97c3a", "high": "#c0392b"}
_RISK_EMOJI = {"safe": "🟢", "low": "🟡", "mid": "🟠", "high": "🔴"}


def _risk_level(cls: str) -> str:
    return _RISK_MAP.get(cls, {}).get("level", "low")


def _risk_color(cls: str) -> str:
    return _RISK_HEX.get(_risk_level(cls), "#a0a0a0")


def _risk_chip_html(cls: str) -> str:
    return f'<span class="risk-chip risk-{_risk_level(cls)}">{_esc(cls)}</span>'


def _short_label(cls: str) -> str:
    """'VeryMildDemented' -> 'Very Mild', 'NonDemented' -> 'Non', etc."""
    out = (cls or "").replace("Demented", "")
    out = re.sub(r"([a-z])([A-Z])", r"\1 \2", out)
    return out or (cls or "—")


def _class_color_map() -> Dict[str, str]:
    return {c: _risk_color(c) for c in _class_names}


# ─── HTML builders (ported from index.html's JS render functions) ───────────
def _render_results_html(mode: str, classified: List[dict], total: int) -> str:
    if not classified:
        return '<p class="nsc-muted">Every uploaded image was flagged as a non-MRI input — nothing to score.</p>'

    if mode == "per_image":
        return (
            f'<div class="nsc-results-title">Per-image results</div>'
            f'<div class="nsc-results-meta">{len(classified)} of {total} image(s) — see the table below</div>'
        )

    agg = _aggregate_max(classified) if mode == "max_probability" else _aggregate_mean(classified)
    label = "max probability" if mode == "max_probability" else "mean probability"
    sorted_probs = sorted(agg["probabilities"].items(), key=lambda kv: -kv[1])

    bars = "".join(
        f'<div class="bar-row">'
        f'<span class="bar-label">{_risk_chip_html(c)}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{p*100:.2f}%;background:{_risk_color(c)}"></div></div>'
        f'<span class="bar-val">{p*100:.1f}%</span>'
        f'</div>'
        for c, p in sorted_probs
    )
    title = f'{_esc(agg["predicted_class"])} {_risk_chip_html(agg["predicted_class"])}'
    meta = f'{len(classified)} of {total} image(s) · {label}'
    return (
        f'<div class="nsc-results-title">{title}</div>'
        f'<div class="nsc-results-meta">{meta}</div>'
        f'<div class="agg-bars">{bars}</div>'
    )


def _results_dataframe(results: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name in sorted(results.keys()):
        r = results[name]
        if r["ood_rejected"]:
            rows.append({
                "Image": name, "Class": "— rejected —", "Confidence": "—",
                "Status": f"Non-MRI ({r.get('ood_reason') or 'unspecified'})",
            })
        else:
            emoji = _RISK_EMOJI.get(_risk_level(r["predicted_class"]), "⚪")
            rows.append({
                "Image": name,
                "Class": f'{emoji} {r["predicted_class"]}',
                "Confidence": f'{r["confidence"]*100:.1f}%',
                "Status": "Warning" if r.get("ood_warning") else "OK",
            })
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS) if rows else pd.DataFrame(columns=RESULTS_COLUMNS)


def _build_scatter(results: Dict[str, dict]):
    classified = [(n, r) for n, r in sorted(results.items()) if not r["ood_rejected"]]
    if not classified:
        return gr.ScatterPlot(visible=False)
    df = pd.DataFrame({
        "Index": list(range(len(classified))),
        "Confidence": [r["confidence"] * 100 for _, r in classified],
        "Class": [r["predicted_class"] for _, r in classified],
        "Filename": [n for n, _ in classified],
    })
    return gr.ScatterPlot(
        value=df, x="Index", y="Confidence", color="Class",
        color_map=_class_color_map(), tooltip=["Filename", "Class", "Confidence"],
        title="Per-image confidence", y_title="Confidence (%)",
        height=260, show_label=False, visible=True,
    )


def _cam_stats_html(cam_stats: Optional[dict], region: Optional[str]) -> str:
    if not cam_stats:
        return ""
    mean = f'{cam_stats["mean"]*100:.1f}%' if cam_stats.get("mean") is not None else "—"
    peak = f'{cam_stats["max"]*100:.1f}%' if cam_stats.get("max") is not None else "—"
    std = f'{cam_stats["std"]*100:.1f}%' if cam_stats.get("std") is not None else "—"

    if cam_stats.get("coverage_pct") is not None:
        coverage = f'{cam_stats["coverage_pct"]:.1f}%'
    elif cam_stats.get("coverage") is not None:
        coverage = f'{cam_stats["coverage"]*100:.1f}%'
    else:
        coverage = "—"

    cx = f'{cam_stats["centroid_x"]:.2f}' if cam_stats.get("centroid_x") is not None else "—"
    cy = f'{cam_stats["centroid_y"]:.2f}' if cam_stats.get("centroid_y") is not None else "—"
    entropy = cam_stats.get("entropy")
    region_html = f'<span class="region-tag">{_esc(region)}</span>' if region else ""
    entropy_html = (
        f'<div class="cam-stat-cell"><div class="cam-stat-value">{entropy:.3f}</div>'
        f'<div class="cam-stat-label">Entropy</div></div>'
    ) if entropy is not None else ""

    return f"""<div class="cam-stats-panel">
  <div class="cam-stats-header"><span>GradCAM Spatial Statistics</span>{region_html}</div>
  <div class="cam-stats-grid">
    <div class="cam-stat-cell"><div class="cam-stat-value">{mean}</div><div class="cam-stat-label">Mean activation</div></div>
    <div class="cam-stat-cell"><div class="cam-stat-value">{peak}</div><div class="cam-stat-label">Peak activation</div></div>
    <div class="cam-stat-cell"><div class="cam-stat-value">{std}</div><div class="cam-stat-label">Std deviation</div></div>
    <div class="cam-stat-cell"><div class="cam-stat-value">{coverage}</div><div class="cam-stat-label">Coverage</div></div>
    <div class="cam-stat-cell"><div class="cam-stat-value">{cx}</div><div class="cam-stat-label">Centroid X</div></div>
    <div class="cam-stat-cell"><div class="cam-stat-value">{cy}</div><div class="cam-stat-label">Centroid Y</div></div>
    {entropy_html}
  </div>
  {_minimap_html(cam_stats)}
</div>"""


def _heat_color(t: float) -> str:
    """0–1 -> blue -> green -> yellow -> orange -> red, matching index.html's camHeatColor gradient."""
    stops = [
        (0.0, (49, 130, 189)), (0.25, (65, 171, 93)), (0.5, (255, 255, 51)),
        (0.75, (253, 141, 60)), (1.0, (165, 15, 21)),
    ]
    lo, hi = stops[0], stops[-1]
    for i in range(len(stops) - 1):
        if stops[i][0] <= t <= stops[i + 1][0]:
            lo, hi = stops[i], stops[i + 1]
            break
    span = (hi[0] - lo[0]) or 1.0
    f = (t - lo[0]) / span
    r = round(lo[1][0] + (hi[1][0] - lo[1][0]) * f)
    g = round(lo[1][1] + (hi[1][1] - lo[1][1]) * f)
    b = round(lo[1][2] + (hi[1][2] - lo[1][2]) * f)
    return f"rgb({r},{g},{b})"


def _minimap_html(cam_stats: dict) -> str:
    raw = cam_stats.get("raw") or cam_stats.get("grid")
    if raw is None:
        return ""
    flat = np.asarray(raw, dtype=float).flatten()
    if flat.size == 0:
        return ""
    lo, hi = float(flat.min()), float(flat.max())
    span = (hi - lo) or 1.0
    cells = "".join(
        f'<div class="cam-minimap-cell" style="background:{_heat_color((v - lo) / span)};'
        f'opacity:{0.4 + ((v - lo) / span) * 0.6:.2f}"></div>'
        for v in flat
    )
    return (
        '<div class="cam-minimap-wrap"><div class="cam-minimap-label">Activation map (downsampled)</div>'
        f'<div class="cam-minimap">{cells}</div></div>'
    )


def _render_explanation_html(expl: dict) -> str:
    header = (
        f'<div class="nsc-results-title">{_esc(expl["predicted_class"])} {_risk_chip_html(expl["predicted_class"])}</div>'
        f'<div class="nsc-results-meta">{expl["confidence"]*100:.1f}% confidence</div>'
    )
    warn_html = ""
    if expl.get("ood_warning"):
        reason = expl["ood_warning"].get("reason", "")
        warn_html = f'<p class="nsc-warn">⚠️ Borderline non-MRI score — {_esc(reason)}</p>'
    text_html = f'<p class="nsc-quote">{_esc(expl["text"])}</p>' if expl.get("text") else ""
    return header + warn_html + text_html + _cam_stats_html(expl.get("cam_stats"), expl.get("region"))


def _render_kpi_html(summary: dict, total: int, rejected: int) -> str:
    dominant = summary.get("dominant_class", "—")
    mean_conf = summary.get("mean_confidence")
    mean_conf_str = f"{mean_conf*100:.1f}%" if mean_conf is not None else "—"
    n_classes = len(summary.get("class_distribution", {}))
    note = f'<p class="nsc-meta">{rejected} image(s) excluded as non-MRI input.</p>' if rejected else ""
    return f"""<div class="summary-kpis">
  <div class="summary-kpi"><div class="summary-kpi-value">{total}</div><div class="summary-kpi-label">Images analysed</div></div>
  <div class="summary-kpi"><div class="summary-kpi-value" style="color:{_risk_color(dominant)}">{_esc(_short_label(dominant))}</div><div class="summary-kpi-label">Dominant class</div></div>
  <div class="summary-kpi"><div class="summary-kpi-value">{mean_conf_str}</div><div class="summary-kpi-label">Mean confidence</div></div>
  <div class="summary-kpi"><div class="summary-kpi-value">{n_classes}</div><div class="summary-kpi-label">Classes detected</div></div>
</div>{note}"""


def _class_distribution_plot(summary: dict):
    dist = summary.get("class_distribution", {})
    if not dist:
        return gr.BarPlot(visible=False)
    df = pd.DataFrame({"Class": list(dist.keys()), "Count": list(dist.values())})
    return gr.BarPlot(
        value=df, x="Class", y="Count", color="Class", color_map=_class_color_map(),
        title="Class distribution", height=260, show_label=False, visible=True,
    )


def _confidence_histogram_plot(classified: List[dict]):
    if not classified:
        return gr.BarPlot(visible=False)
    confidences = [r["confidence"] for r in classified]
    bins = np.linspace(0, 1, 11)
    counts, edges = np.histogram(confidences, bins=bins)
    labels = [f"{int(edges[i]*100)}–{int(edges[i+1]*100)}%" for i in range(len(counts))]
    df = pd.DataFrame({"Confidence": labels, "Count": counts})
    return gr.BarPlot(
        value=df, x="Confidence", y="Count",
        title="Confidence histogram", height=260, show_label=False, visible=True,
    )


def _prob_stats_dataframe(summary: dict) -> pd.DataFrame:
    mean_p = summary.get("mean_probabilities", {})
    std_p = summary.get("std_probabilities", {})
    max_p = summary.get("max_probabilities", {})
    classes = sorted(mean_p.keys(), key=lambda c: -mean_p.get(c, 0))
    rows = [{
        "Class": c,
        "Mean": f'{mean_p.get(c, 0)*100:.1f}%',
        "Std": f'{std_p[c]*100:.1f}%' if c in std_p else "—",
        "Max": f'{max_p[c]*100:.1f}%' if c in max_p else "—",
    } for c in classes]
    return pd.DataFrame(rows, columns=STATS_COLUMNS) if rows else pd.DataFrame(columns=STATS_COLUMNS)


# ─── Blocks callbacks ─────────────────────────────────────────────────────────
def on_files_changed(paths):
    files = _pairs_from_paths(paths)
    n = len(files)
    summary = "No images loaded yet." if n == 0 else f"**{n}** image{'s' if n != 1 else ''} loaded and ready to run."
    df = pd.DataFrame({"Filename": sorted(files.keys())})
    return files, summary, df, gr.update(interactive=n > 0)


def on_run(mode_label, files, results):
    if not files:
        raise gr.Error("Upload at least one image first.")

    results = dict(results)
    to_predict = {name: img for name, img in files.items() if name not in results}
    if to_predict:
        names = list(to_predict.keys())
        imgs = [to_predict[n] for n in names]
        for r in _run_gated_batch_predict(imgs, names):  # ZeroGPU call, from a real click handler
            results[r["filename"]] = r

    mode = _MODE_LABELS[mode_label]
    classified = [r for r in results.values() if not r["ood_rejected"]]
    rejected = len(results) - len(classified)
    status = (
        f"⚠️ {rejected} of {len(results)} image(s) were flagged as non-MRI input and excluded from scoring."
        if rejected else ""
    )

    names_sorted = sorted(r["filename"] for r in classified)
    return (
        results,
        status,
        _render_results_html(mode, classified, len(results)),
        _results_dataframe(results),
        _build_scatter(results),
        gr.update(choices=names_sorted, value=(names_sorted[0] if names_sorted else None)),
    )


def on_auto_explain(enabled, results, files, explanations):
    if not enabled:
        return explanations, gr.update()
    explanations = dict(explanations)
    todo = [n for n, r in results.items() if not r["ood_rejected"] and n not in explanations]
    for name in todo:
        img = files.get(name)
        if img is None:
            continue
        out = _run_explain(img)  # ZeroGPU call, sequential by design (see index.html's runAutoExplain)
        if not out["ood_rejected"]:
            explanations[name] = _package_explanation(out)
    return explanations, (f"Auto-explained {len(todo)} image(s)." if todo else "")


def on_generate_explanation(name, files, explanations):
    if not name:
        raise gr.Error("Select an image from the dropdown first.")
    if name in explanations:
        expl = explanations[name]
        return expl["original"], expl["overlay"], _render_explanation_html(expl), explanations, ""

    img = files.get(name)
    if img is None:
        raise gr.Error("That image is no longer loaded — re-upload it and run again.")

    out = _run_explain(img)  # ZeroGPU call
    if out["ood_rejected"]:
        raise gr.Error(f"Rejected as a non-MRI input: {out['ood']['reason']}")

    expl = _package_explanation(out)
    explanations = dict(explanations)
    explanations[name] = expl
    return expl["original"], expl["overlay"], _render_explanation_html(expl), explanations, ""


def on_explain_dropdown_change(name, explanations):
    if not name or name not in explanations:
        return None, None, "", ""
    expl = explanations[name]
    return expl["original"], expl["overlay"], _render_explanation_html(expl), ""


def on_batch_summary(results):
    classified = [r for r in results.values() if not r["ood_rejected"]]
    if not classified:
        raise gr.Error("No classified images yet — run prediction first.")
    summary = batch_summary(classified)  # CPU-only, no GPU needed
    return (
        _render_kpi_html(summary, len(results), len(results) - len(classified)),
        _class_distribution_plot(summary),
        _confidence_histogram_plot(classified),
        _prob_stats_dataframe(summary),
    )


def on_gpu_smoke_test():
    try:
        return _gpu_smoke_test_inner()
    except Exception as exc:
        return {"error": repr(exc)}


def on_clear_all():
    return (
        None,                                                        # file_input
        {}, {}, {},                                                  # state_files, state_results, state_explanations
        "No images loaded yet.",                                     # files_summary_md
        pd.DataFrame({"Filename": []}),                              # files_table
        gr.update(interactive=False),                                # run_btn
        '<p class="nsc-muted">Run a prediction to see results here.</p>',  # results_html
        pd.DataFrame(columns=RESULTS_COLUMNS),                       # results_df
        gr.ScatterPlot(visible=False),                               # scatter_plot
        gr.update(choices=[], value=None),                           # explain_dropdown
        "",                                                            # explain_html
        None, None,                                                  # explain_original, explain_overlay
        "",                                                            # explain_status_md
        "",                                                            # status_md
    )


# ─── Styling: same tokens/fonts/risk palette as index.html ───────────────────
HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
"""

# Toggled via document.body.classList.toggle('dark') — Gradio's own dark-mode
# hook. The BASE (non-suffixed) theme values below are our dark palette, so
# the app defaults to looking like index.html's default dark theme; the
# `_dark`-suffixed values are our light palette, so toggling gives the light
# theme. The naming is inverted from Gradio's point of view, but it's the
# same "dark by default, one button to flip" behavior index.html has.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("DM Mono"), "ui-monospace", "monospace"],
    font_mono=[gr.themes.GoogleFont("DM Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#0f0f0f",
    body_text_color="#f0f0f0",
    body_text_color_subdued="#a0a0a0",
    background_fill_primary="#1c1c1c",
    background_fill_secondary="#252525",
    border_color_primary="#2e2e2e",
    border_color_accent="#3a3a3a",
    block_background_fill="#1c1c1c",
    block_border_color="#2e2e2e",
    block_label_background_fill="#1c1c1c",
    block_label_text_color="#a0a0a0",
    block_title_text_color="#f0f0f0",
    input_background_fill="#1c1c1c",
    input_border_color="#2e2e2e",
    panel_background_fill="#1c1c1c",
    button_primary_background_fill="#f0f0f0",
    button_primary_text_color="#0f0f0f",
    button_secondary_background_fill="#252525",
    button_secondary_text_color="#f0f0f0",
    body_background_fill_dark="#f5f2ed",
    body_text_color_dark="#2c1f14",
    body_text_color_subdued_dark="#7a6a5a",
    background_fill_primary_dark="#ece8e1",
    background_fill_secondary_dark="#e2ddd6",
    border_color_primary_dark="#d4cdc4",
    border_color_accent_dark="#c4bcb2",
    block_background_fill_dark="#ece8e1",
    block_border_color_dark="#d4cdc4",
    block_label_background_fill_dark="#ece8e1",
    block_label_text_color_dark="#7a6a5a",
    block_title_text_color_dark="#2c1f14",
    input_background_fill_dark="#ece8e1",
    input_border_color_dark="#d4cdc4",
    panel_background_fill_dark="#ece8e1",
    button_primary_background_fill_dark="#2c1f14",
    button_primary_text_color_dark="#f5f2ed",
    button_secondary_background_fill_dark="#e2ddd6",
    button_secondary_text_color_dark="#2c1f14",
)

CSS = """
.gradio-container, .gradio-container * { font-family: 'DM Mono', ui-monospace, monospace; }

.nsc-topbar { align-items: center; }
.nsc-logo { display: flex; align-items: baseline; gap: 10px; padding: 4px 0; }
.nsc-logo-name { font-family: 'Instrument Serif', serif; font-size: 20px; }
.nsc-logo-tag { font-size: 11px; opacity: 0.6; letter-spacing: 0.08em; text-transform: uppercase; }

.nsc-muted { opacity: 0.55; font-size: 13px; }
.nsc-meta  { opacity: 0.6; font-size: 12px; margin-top: 2px; }
.nsc-warn  { color: #c97c3a; font-size: 12px; }
.nsc-quote {
  font-style: italic; opacity: 0.85; font-size: 12px; line-height: 1.6;
  background: rgba(128,128,128,0.08); border-radius: 6px; padding: 10px 14px; margin: 8px 0;
}
.nsc-results-title { font-family: 'Instrument Serif', serif; font-size: 20px; }
.nsc-results-meta  { font-size: 11px; opacity: 0.55; margin-bottom: 10px; }

/* Risk chips — same palette as index.html */
.risk-chip {
  display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 10px;
  font-family: 'DM Mono', monospace; border: 1px solid transparent; white-space: nowrap;
}
.risk-safe { color: #4caf7d; background: rgba(76,175,125,.13); border-color: #4caf7d; }
.risk-low  { color: #c9a84c; background: rgba(201,168,76,.13); border-color: #c9a84c; }
.risk-mid  { color: #c97c3a; background: rgba(201,124,58,.13); border-color: #c97c3a; }
.risk-high { color: #c0392b; background: rgba(192,57,43,.13);  border-color: #c0392b; }

/* Aggregate probability bars */
.agg-bars { display: flex; flex-direction: column; gap: 10px; margin-top: 8px; }
.bar-row { display: grid; grid-template-columns: 150px 1fr 52px; align-items: center; gap: 12px; }
.bar-track { height: 4px; background: rgba(128,128,128,0.25); border-radius: 2px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 2px; }
.bar-val   { font-size: 11px; opacity: 0.6; text-align: right; }

/* CAM spatial-statistics panel */
.cam-stats-panel { border: 1px solid rgba(128,128,128,0.25); border-radius: 8px; overflow: hidden; margin-top: 10px; }
.cam-stats-header {
  padding: 8px 14px; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.6;
  border-bottom: 1px solid rgba(128,128,128,0.25); display: flex; align-items: center; justify-content: space-between;
}
.region-tag {
  font-size: 10px; padding: 2px 8px; border-radius: 3px; background: rgba(128,128,128,0.15);
  text-transform: none; letter-spacing: 0.02em;
}
.cam-stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); }
.cam-stat-cell {
  padding: 12px 14px; border-right: 1px solid rgba(128,128,128,0.15); border-bottom: 1px solid rgba(128,128,128,0.15);
}
.cam-stat-value { font-family: 'Instrument Serif', serif; font-size: 17px; }
.cam-stat-label { font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.55; }
.cam-minimap-wrap { padding: 10px 14px; border-top: 1px solid rgba(128,128,128,0.15); }
.cam-minimap-label { font-size: 9px; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.55; margin-bottom: 6px; }
.cam-minimap { display: grid; grid-template-columns: repeat(8, 1fr); gap: 2px; border-radius: 4px; overflow: hidden; }
.cam-minimap-cell { aspect-ratio: 1; border-radius: 2px; }

/* Batch-summary KPI row */
.summary-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: rgba(128,128,128,0.2); border-radius: 8px; overflow: hidden; margin-bottom: 6px; }
.summary-kpi { background: var(--block-background-fill); padding: 14px 16px; }
.summary-kpi-value { font-family: 'Instrument Serif', serif; font-size: 22px; }
.summary-kpi-label { font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.55; }
"""

PAGE_LOAD_JS = "() => { document.body.classList.remove('dark'); }"
THEME_TOGGLE_JS = "() => { document.body.classList.toggle('dark'); }"


# ─── Blocks UI ────────────────────────────────────────────────────────────────
# `theme=`/`css=`/`js=`/`head=` are passed on the Blocks() constructor rather
# than to .launch() — Gradio 6 moved them to launch(), but the constructor
# still accepts them (with a deprecation warning) via a backward-compat
# shim, and it's the only spelling that works on Gradio 4/5. Since the exact
# Gradio version pinned by the Space isn't known ahead of time, this is the
# more portable choice.
with gr.Blocks(
    theme=THEME, css=CSS, js=PAGE_LOAD_JS, head=HEAD_HTML,
    title="NeuroScan — Alzheimer MRI Classifier",
) as demo:
    state_files = gr.State({})         # filename -> PIL.Image
    state_results = gr.State({})       # filename -> prediction result dict
    state_explanations = gr.State({})  # filename -> explanation dict

    with gr.Row(elem_classes=["nsc-topbar"]):
        gr.HTML(
            '<div class="nsc-logo"><span class="nsc-logo-name">NeuroScan</span>'
            '<span class="nsc-logo-tag">Alzheimer MRI · ZeroGPU</span></div>'
        )
        theme_btn = gr.Button("◑ Theme", size="sm", scale=0)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Upload")
            file_input = gr.File(
                label="Drop MRI scans or a ZIP",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".tif", ".tiff", ".zip"],
                type="filepath",
            )
            files_summary_md = gr.Markdown("No images loaded yet.")
            files_table = gr.Dataframe(
                headers=["Filename"], datatype=["str"], value=pd.DataFrame({"Filename": []}),
                interactive=False, wrap=True, show_label=False,
            )
            clear_btn = gr.Button("Clear all", size="sm")

            gr.Markdown("#### Run")
            mode_radio = gr.Radio(list(_MODE_LABELS.keys()), value="Mean probability", label="Aggregation mode")
            auto_explain_cb = gr.Checkbox(label="Auto-explain every classified image after Run", value=False)
            run_btn = gr.Button("Run", variant="primary", interactive=False)
            status_md = gr.Markdown("")

        with gr.Column(scale=2):
            gr.Markdown("#### Results")
            results_html = gr.HTML('<p class="nsc-muted">Run a prediction to see results here.</p>')
            results_df = gr.Dataframe(
                headers=RESULTS_COLUMNS, datatype=["str"] * 4, value=pd.DataFrame(columns=RESULTS_COLUMNS),
                interactive=False, wrap=True, show_label=False,
            )
            scatter_plot = gr.ScatterPlot(visible=False, show_label=False)

    with gr.Accordion("Explain · GradCAM", open=False):
        with gr.Row():
            explain_dropdown = gr.Dropdown(choices=[], label="Image", interactive=True)
            explain_btn = gr.Button("Generate explanation")
        explain_status_md = gr.Markdown("")
        with gr.Row():
            explain_original = gr.Image(label="Original MRI", interactive=False, height=280)
            explain_overlay = gr.Image(label="EigenCAM overlay", interactive=False, height=280)
        explain_html = gr.HTML("")

    with gr.Accordion("Batch summary", open=False):
        summary_btn = gr.Button("Compute batch summary")
        summary_kpi_html = gr.HTML("")
        with gr.Row():
            dist_plot = gr.BarPlot(visible=False, show_label=False)
            hist_plot = gr.BarPlot(visible=False, show_label=False)
        summary_stats_df = gr.Dataframe(
            headers=STATS_COLUMNS, datatype=["str"] * 4, value=pd.DataFrame(columns=STATS_COLUMNS),
            interactive=False, show_label=False,
        )

    with gr.Accordion("Diagnostics", open=False):
        gr.Markdown(f"OOD guard loaded: **{ood_guard_enabled()}**  ·  Device: **{DEVICE}**")
        gr.Markdown(
            "Use this to confirm ZeroGPU actually attaches a GPU to this Space "
            "(the thing that didn't work when this app was routed through custom "
            "FastAPI-style endpoints instead of real Gradio events)."
        )
        smoke_btn = gr.Button("Run GPU smoke test")
        smoke_json = gr.JSON(label="Result")

    # ── Wiring ──────────────────────────────────────────────────────────────
    file_input.change(
        on_files_changed, inputs=[file_input],
        outputs=[state_files, files_summary_md, files_table, run_btn],
    )

    clear_btn.click(
        on_clear_all, inputs=[],
        outputs=[
            file_input, state_files, state_results, state_explanations,
            files_summary_md, files_table, run_btn,
            results_html, results_df, scatter_plot, explain_dropdown,
            explain_html, explain_original, explain_overlay, explain_status_md,
            status_md,
        ],
    )

    run_btn.click(
        on_run, inputs=[mode_radio, state_files, state_results],
        outputs=[state_results, status_md, results_html, results_df, scatter_plot, explain_dropdown],
    ).then(
        on_auto_explain, inputs=[auto_explain_cb, state_results, state_files, state_explanations],
        outputs=[state_explanations, explain_status_md],
    )

    explain_btn.click(
        on_generate_explanation, inputs=[explain_dropdown, state_files, state_explanations],
        outputs=[explain_original, explain_overlay, explain_html, state_explanations, explain_status_md],
    )

    explain_dropdown.change(
        on_explain_dropdown_change, inputs=[explain_dropdown, state_explanations],
        outputs=[explain_original, explain_overlay, explain_html, explain_status_md],
    )

    summary_btn.click(
        on_batch_summary, inputs=[state_results],
        outputs=[summary_kpi_html, dist_plot, hist_plot, summary_stats_df],
    )

    smoke_btn.click(on_gpu_smoke_test, inputs=[], outputs=[smoke_json])

    theme_btn.click(fn=None, js=THEME_TOGGLE_JS)
    demo.load(fn=None, js=PAGE_LOAD_JS)


# Unconditional (not gated behind `if __name__ == "__main__"`) so this also
# works under `gradio app.py` hot-reload, matching standard Gradio app.py
# convention — and it's what Hugging Face Spaces' Gradio SDK runs directly.
demo.launch()