from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

from package.config import SAVE_DIR, DEVICE

"""
Out-of-distribution guardrail, runs BEFORE the Swin classifier.

Two signals, both built on frozen general-CLIP embeddings (CLIP is only ever used here as a fixed feature extractor / zero-shot sanity check):
  1. Distance-based: Euclidean + Mahalanobis (shared pooled covariance) +
     relative Mahalanobis distance (RMD) from a query embedding to the
     nearest per-class training centroid.
  2. Zero-shot: CLIP's own text tower scores the image against a fixed
     "brain MRI" vs. distractor-modality/non-medical prompt set.
"""


_CLIP_IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_IMAGE_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])


_ARTIFACT_FILENAME   = "ood_calibration.npz"
OOD_CALIBRATION_PATH = SAVE_DIR / _ARTIFACT_FILENAME

_REQUIRED_KEYS = [
    "clip_model_id", "class_names", "centroids", "pooled_precision",
    "global_mean", "global_precision",
    "euclid_p95", "euclid_p99", "mahal_p95", "mahal_p99", "rmd_p95", "rmd_p99",
    "zero_shot_prompts", "zero_shot_prompt_is_positive",
    "zero_shot_p_mri_p05", "zero_shot_p_mri_p10",
]


# ── CLIP preprocessing ────────────────────────────────────────────────────
# Mirrors load_grayscale_as_rgb_tensor() + clip_preprocess() from the
# embedding-extraction notebook exactly, so a query embedding lands in the
# same space as the centroids it's compared against.


def _grayscale_replicate(pil_img: Image.Image) -> torch.Tensor:
    """PIL -> (3, H, W) tensor, forced through grayscale then replicated."""
    gray   = pil_img.convert("L")
    arr    = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor.repeat(3, 1, 1)


def _clip_preprocess(tensor_3hw: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Resize shortest side -> center crop -> normalize (standard CLIP preprocessing)."""
    c, h, w = tensor_3hw.shape
    scale = image_size / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = F.interpolate(
        tensor_3hw.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0, 1)

    top  = max(0, (new_h - image_size) // 2)
    left = max(0, (new_w - image_size) // 2)
    cropped = resized[:, top: top + image_size, left: left + image_size]

    pad_h, pad_w = image_size - cropped.shape[1], image_size - cropped.shape[2]
    if pad_h > 0 or pad_w > 0:
        cropped = F.pad(cropped, (0, max(pad_w, 0), 0, max(pad_h, 0)))

    return (cropped - _CLIP_IMAGE_MEAN[:, None, None]) / _CLIP_IMAGE_STD[:, None, None]


def _pil_to_clip_tensor(pil_img: Image.Image) -> torch.Tensor:
    return _clip_preprocess(_grayscale_replicate(pil_img)).unsqueeze(0)


def _extract_features(output, preferred_attrs=("image_embeds", "text_embeds", "pooler_output")):
    """Version-robust unwrapping of HF CLIP feature outputs across transformers releases.
    Ported from the embedding-extraction notebook -- get_image_features/get_text_features
    return a plain tensor on some versions and a wrapped output object on others."""
    if torch.is_tensor(output):
        return output
    for attr in preferred_attrs:
        val = getattr(output, attr, None)
        if torch.is_tensor(val):
            return val
    raise TypeError(f"Unrecognized feature output type: {type(output)}")


# ── Artifact + model loading ──────────────────────────────────────────────
_clip_model:         Optional[torch.nn.Module] = None
_clip_tokenizer                                = None
_calib:               Optional[dict]           = None
_prompt_text_embeds: Optional[torch.Tensor]    = None  # (P, D), L2-normalized
_prompt_is_positive: Optional[torch.Tensor]    = None  # (P,) bool
_guard_enabled = False


def _validate_calibration(calib: dict) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in calib]
    if missing:
        raise ValueError(f"calibration artifact missing keys: {missing}")

    K, D = calib["centroids"].shape
    if calib["pooled_precision"].shape != (D, D):
        raise ValueError(f"pooled_precision shape {calib['pooled_precision'].shape} != ({D}, {D})")
    if calib["global_precision"].shape != (D, D):
        raise ValueError(f"global_precision shape {calib['global_precision'].shape} != ({D}, {D})")
    if calib["global_mean"].shape != (D,):
        raise ValueError(f"global_mean shape {calib['global_mean'].shape} != ({D},)")
    if len(calib["class_names"]) != K:
        raise ValueError("class_names length must match centroids' row count")
    if len(calib["zero_shot_prompts"]) != len(calib["zero_shot_prompt_is_positive"]):
        raise ValueError("zero_shot_prompts and zero_shot_prompt_is_positive length mismatch")
    if not bool(np.any(calib["zero_shot_prompt_is_positive"])):
        raise ValueError("zero_shot_prompt_is_positive has no positive ('brain MRI') prompts")


def load_ood_guard() -> None:
    """
    Downloads (if needed) the calibration artifact from the HF Hub repo used
    for the Swin weights -- or OOD_HF_REPO, if that env var is set to a
    different repo -- then loads the general-CLIP backbone named inside the
    artifact and pre-encodes its zero-shot prompt set once.

    Calibration is trained offline; this function only ever *consumes* it.
    A missing/broken artifact disables the guard (logged, not raised) rather
    than crashing the app -- predictions still work, just without the gate,
    until 'ood_calibration.npz' exists in the HF repo.
    """
    global _clip_model, _clip_tokenizer, _calib, _prompt_text_embeds, _prompt_is_positive, _guard_enabled

    try:
        if not OOD_CALIBRATION_PATH.exists():
            print("[INFO] Downloading OOD calibration artifact from Hugging Face Hub...")
            SAVE_DIR.mkdir(parents=True, exist_ok=True)

            repo_id = os.environ.get("OOD_HF_REPO") or os.environ["HF_MODEL_REPO"]
            token   = os.environ.get("HF_TOKEN")

            hf_hub_download(
                repo_id=repo_id,
                filename=_ARTIFACT_FILENAME,
                local_dir=str(SAVE_DIR),
                token=token,
            )
            print("[INFO] OOD calibration artifact downloaded from HF Hub.")

        calib = dict(np.load(OOD_CALIBRATION_PATH, allow_pickle=False))
        _validate_calibration(calib)

        # transformers is imported lazily here, matching hf_clip_backend()'s
        # lazy import in the embedding-extraction notebook.
        from transformers import CLIPModel, CLIPProcessor

        clip_model_id = str(calib["clip_model_id"])
        model     = CLIPModel.from_pretrained(clip_model_id).eval().to(DEVICE)
        processor = CLIPProcessor.from_pretrained(clip_model_id)

        prompts     = [str(p) for p in calib["zero_shot_prompts"]]
        is_positive = calib["zero_shot_prompt_is_positive"].astype(bool)

        with torch.inference_mode():
            tokens        = processor.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
            text_features = _extract_features(model.get_text_features(**tokens))
            text_features = F.normalize(text_features, dim=-1)

        _clip_model         = model
        _clip_tokenizer     = processor.tokenizer
        _calib              = calib
        _prompt_text_embeds = text_features
        _prompt_is_positive = torch.from_numpy(is_positive).to(DEVICE)
        _guard_enabled      = True

        print(f"[INFO] OOD guard ready  |  backbone={clip_model_id}  |  "
              f"classes={list(calib['class_names'])}  |  prompts={len(prompts)}  |  device={DEVICE}")

    except Exception as exc:
        print(f"[WARN] OOD guard disabled -- calibration unavailable ({exc!r}). "
              f"Predictions will run without the OOD gate until '{_ARTIFACT_FILENAME}' "
              f"is published to the HF repo.")
        _guard_enabled = False


def is_enabled() -> bool:
    return _guard_enabled


# ── Distance-based signal ─────────────────────────────────────────────────
def _threshold_for_class(arr: np.ndarray, class_idx: int) -> float:
    arr = np.asarray(arr)
    return float(arr) if arr.ndim == 0 else float(arr[class_idx])


def _nearest_centroid_distances(x: np.ndarray) -> dict:
    """Euclidean, Mahalanobis (pooled covariance), and relative Mahalanobis
    distance (RMD = class Mahalanobis - background Mahalanobis) from a single
    query embedding to its nearest training centroid, each metric finding its
    own nearest class independently."""
    centroids = _calib["centroids"]                                # (K, D)
    diffs     = centroids - x[None, :]                             # (K, D)

    euclid = np.linalg.norm(diffs, axis=1)                         # (K,)
    mahal_sq = np.einsum("kd,de,ke->k", diffs, _calib["pooled_precision"], diffs)
    mahal    = np.sqrt(np.clip(mahal_sq, 0.0, None))                # (K,)

    e_idx = int(np.argmin(euclid))
    m_idx = int(np.argmin(mahal))

    global_diff   = x - _calib["global_mean"]
    background_md = float(np.sqrt(max(0.0, global_diff @ _calib["global_precision"] @ global_diff)))
    rmd = float(mahal[m_idx]) - background_md

    return {
        "euclidean":             float(euclid[e_idx]),
        "euclidean_class":       str(_calib["class_names"][e_idx]),
        "euclidean_p99":         _threshold_for_class(_calib["euclid_p99"], e_idx),
        "mahalanobis":           float(mahal[m_idx]),
        "mahalanobis_class":     str(_calib["class_names"][m_idx]),
        "mahalanobis_p99":       _threshold_for_class(_calib["mahal_p99"], m_idx),
        "relative_mahalanobis":  rmd,
        "rmd_p95":               _threshold_for_class(_calib["rmd_p95"], m_idx),
        "rmd_p99":               _threshold_for_class(_calib["rmd_p99"], m_idx),
    }


# ── Zero-shot signal ───────────────────────────────────────────────────────
def _zero_shot_scores(image_features_1d: torch.Tensor) -> tuple[float, int]:
    """Standard CLIP zero-shot protocol: L2-normalize, scale by the model's
    learned logit_scale, softmax over ALL prompts. Returns (p_mri, top_prompt_idx)."""
    img    = F.normalize(image_features_1d.unsqueeze(0), dim=-1)      # (1, D)
    scale  = _clip_model.logit_scale.exp()
    logits = scale * img @ _prompt_text_embeds.T                     # (1, P)
    probs  = F.softmax(logits, dim=-1)[0]                             # (P,)

    p_mri   = float(probs[_prompt_is_positive].sum())
    top_idx = int(torch.argmax(probs))
    return p_mri, top_idx


# ── Free, model-free heuristic ────────────────────────────────────────────
def _color_divergence(pil_img: Image.Image) -> float:
    """MRI scans in this dataset are stored near-perfectly grayscale even
    when saved as RGB. Large channel-mean divergence is a strong, free tell
    that the input isn't a scan at all -- no model call needed."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    channel_means = arr.mean(axis=0)
    return float(channel_means.max() - channel_means.min())


# ── Verdict ────────────────────────────────────────────────────────────────
def _build_verdict(dist: dict, p_mri: float, top_prompt: str, top_is_positive: bool,
                    color_div: float) -> dict:
    zshot_p05 = float(_calib["zero_shot_p_mri_p05"])
    zshot_p10 = float(_calib["zero_shot_p_mri_p10"])

    rmd_reject = dist["relative_mahalanobis"] > dist["rmd_p99"]
    rmd_warn   = dist["rmd_p95"] < dist["relative_mahalanobis"] <= dist["rmd_p99"]
    mahal_warn = dist["mahalanobis"] > dist["mahalanobis_p99"]
    euclid_warn = dist["euclidean"] > dist["euclidean_p99"]

    zshot_reject = p_mri < zshot_p05
    zshot_warn   = zshot_p05 <= p_mri < zshot_p10

    color_p99 = _calib.get("color_divergence_p99")
    color_reject = bool(color_p99 is not None and color_div > float(color_p99))

    reject_votes = sum([rmd_reject, zshot_reject, color_reject])
    warn_votes   = sum([rmd_warn, mahal_warn, euclid_warn, zshot_warn])

    if reject_votes >= 2:
        verdict = "reject"
    elif reject_votes == 1 or warn_votes >= 1:
        verdict = "warn"
    else:
        verdict = "pass"

    reason = None
    if verdict != "pass":
        parts = []
        if rmd_reject or rmd_warn:
            parts.append(
                f"embedding distance from the nearest training cluster (relative Mahalanobis "
                f"{dist['relative_mahalanobis']:.2f}) is past the "
                f"{'99th' if rmd_reject else '95th'}-percentile threshold seen on training scans "
                f"({dist['rmd_p99' if rmd_reject else 'rmd_p95']:.2f})"
            )
        if (zshot_reject or zshot_warn) and not top_is_positive:
            parts.append(f'zero-shot check matched this image most closely to "{top_prompt}" rather than a brain MRI (p_mri={p_mri:.3f})')
        if color_reject:
            parts.append("the image is not grayscale, unlike the training MRIs")
        prefix = "Rejected: " if verdict == "reject" else "Warning: "
        reason = prefix + "; ".join(parts) + "." if parts else prefix + "borderline OOD signal."

    return {
        "verdict":     verdict,
        "is_rejected": verdict == "reject",
        "reason":      reason,
        "scores": {
            "euclidean":            round(dist["euclidean"], 4),
            "euclidean_class":      dist["euclidean_class"],
            "mahalanobis":          round(dist["mahalanobis"], 4),
            "mahalanobis_class":    dist["mahalanobis_class"],
            "relative_mahalanobis": round(dist["relative_mahalanobis"], 4),
            "zero_shot_p_mri":      round(p_mri, 4),
            "zero_shot_top_prompt": top_prompt,
            "color_divergence":     round(color_div, 4),
        },
    }


# ── Public entry point ─────────────────────────────────────────────────────
def check_images(pil_images: List[Image.Image]) -> List[dict]:
    """
    Runs the OOD gate on a batch of images (one CLIP forward pass for the
    whole batch, mirroring _run_batch_predict's batching in main.py) and
    returns one verdict dict per image, same order as the input:

        {
            "verdict":     "pass" | "warn" | "reject",
            "is_rejected": bool,
            "reason":      str | None,
            "scores":      dict | None,
        }

    Fails open: if the guard didn't load (see load_ood_guard), every image
    passes through unchanged so the classifier keeps working regardless.
    """
    if not _guard_enabled:
        return [{"verdict": "pass", "is_rejected": False, "reason": None, "scores": None}
                for _ in pil_images]

    # The whole body (not just the CLIP forward pass) needs to stay inside
    # inference_mode: _prompt_text_embeds was itself produced under
    # inference_mode in load_ood_guard(), and CLIP's logit_scale is a
    # learnable (requires_grad) parameter, so multiplying it against an
    # inference-mode tensor outside this context raises at runtime.
    with torch.inference_mode():
        tensors = torch.cat([_pil_to_clip_tensor(p) for p in pil_images], dim=0).to(DEVICE)
        image_features = _extract_features(_clip_model.get_image_features(pixel_values=tensors))

        feats_np = image_features.cpu().numpy()
        results = []
        for i, pil_img in enumerate(pil_images):
            dist               = _nearest_centroid_distances(feats_np[i])
            p_mri, top_idx     = _zero_shot_scores(image_features[i])
            top_prompt         = str(_calib["zero_shot_prompts"][top_idx])
            top_is_positive    = bool(_calib["zero_shot_prompt_is_positive"][top_idx])
            color_div          = _color_divergence(pil_img)
            results.append(_build_verdict(dist, p_mri, top_prompt, top_is_positive, color_div))
    return results