"""
package/explainability.py
─────────────────────────
Unified explainability entry point for the Alzheimer MRI classifier.

Public API
----------
    explain_image(pil_img, model, class_names, predicted_class, confidence,
                  method='eigengradcam')
    → dict with keys: original_b64, overlay_b64, text, region, method

Supported methods (swap via the METHOD constant in main.py):
    eigengradcam  – EigenGradCAM   (default, recommended for Swin)
    gradcampp     – GradCAM++
    gradcam       – vanilla GradCAM
    scorecam      – ScoreCAM       (slow, no-gradient)
    rise          – RISE            (slowest, model-agnostic)
    attention     – Attention Rollout (Swin-native)
"""

from __future__ import annotations

import io
import base64
import math
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# pytorch-grad-cam
from pytorch_grad_cam import (
    EigenGradCAM,
    GradCAMPlusPlus,
    GradCAM,
    ScoreCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 224

# Confidence tiers
_TIER_HIGH   = 0.80
_TIER_MEDIUM = 0.55

# RISE defaults (kept low for interactive speed)
_RISE_MASKS  = 500
_RISE_MASK_S = 8      # mask grid size
_RISE_P1     = 0.5    # probability of a mask cell being 1


# ─── Public entry point ───────────────────────────────────────────────────────

def explain_image(
    pil_img: Image.Image,
    model: torch.nn.Module,
    transform,
    class_names: list[str],
    predicted_class: str,
    confidence: float,
    method: str = "eigengradcam",
    device: str = "cpu",
) -> dict:
    """
    Run visual explanation + text explanation on a single PIL image.

    Returns
    -------
    {
        "original_b64":  str,   # base64 PNG of the 224×224 original
        "overlay_b64":   str,   # base64 PNG of heatmap overlay (original size)
        "text":          str,   # natural-language interpretation
        "region":        str,   # rough anatomical region label
        "method":        str,   # which visual method was used
    }
    """
    orig_w, orig_h = pil_img.size
    img_rgb = pil_img.convert("RGB")
    img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE))

    # Float [0,1] array for overlay rendering
    rgb_float = np.array(img_resized, dtype=np.float32) / 255.0

    # Input tensor
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    # ── Visual explanation ────────────────────────────────────────────────────
    grayscale_cam = _dispatch_visual(
        method=method,
        model=model,
        input_tensor=input_tensor,
        img_resized=img_resized,
        rgb_float=rgb_float,
        device=device,
    )  # shape (H, W), values in [0, 1]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # uint8 (H,W,3)
    overlay_pil = Image.fromarray(overlay).resize((orig_w, orig_h), Image.LANCZOS)

    # ── Region estimation ─────────────────────────────────────────────────────
    region = _estimate_region(grayscale_cam)

    # ── Text explanation ──────────────────────────────────────────────────────
    text = _generate_text(predicted_class, confidence, region)

    # ── Encode images ─────────────────────────────────────────────────────────
    original_b64 = _pil_to_b64(img_resized)
    overlay_b64  = _pil_to_b64(overlay_pil)

    return {
        "original_b64": original_b64,
        "overlay_b64":  overlay_b64,
        "text":         text,
        "region":       region,
        "method":       method,
    }


# ─── Visual method dispatcher ─────────────────────────────────────────────────

def _dispatch_visual(
    method: str,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    img_resized: Image.Image,
    rgb_float: np.ndarray,
    device: str,
) -> np.ndarray:
    """Return a (H, W) grayscale activation map in [0, 1]."""

    method = method.lower().strip()

    if method == "eigengradcam":
        return _run_eigengradcam(model, input_tensor)
    elif method == "gradcampp":
        return _run_gradcam_pp(model, input_tensor)
    elif method == "gradcam":
        return _run_gradcam(model, input_tensor)
    elif method == "scorecam":
        return _run_scorecam(model, input_tensor)
    elif method == "rise":
        return _run_rise(model, input_tensor, device)
    elif method == "attention":
        return _run_attention_rollout(model, input_tensor, device)
    else:
        raise ValueError(
            f"Unknown explainability method: '{method}'. "
            "Choose from: eigengradcam, gradcampp, gradcam, scorecam, rise, attention"
        )


# ─── Target layer helper ─────────────────────────────────────────────────────

def _swin_target_layer(model: torch.nn.Module) -> list:
    """Return the target layer list for Swin's last stage."""
    return [model.layers[-1].blocks[-1].norm2]


# ─── Individual visual methods ────────────────────────────────────────────────

def _run_eigengradcam(model, input_tensor) -> np.ndarray:
    """
    EigenGradCAM — projects activations onto the principal eigenvector.
    Smooth, blob-shaped, minimal stripe artefacts on Swin. ~100 ms.
    """
    cam = EigenGradCAM(model=model, target_layers=_swin_target_layer(model))
    result = cam(input_tensor=input_tensor)
    return result[0]


def _run_gradcam_pp(model, input_tensor) -> np.ndarray:
    """
    GradCAM++ — weights positive gradients more carefully than vanilla.
    Handles multiple activations of the same class; slight stripe risk on Swin.
    """
    cam = GradCAMPlusPlus(model=model, target_layers=_swin_target_layer(model))
    result = cam(input_tensor=input_tensor)
    return result[0]


def _run_gradcam(model, input_tensor) -> np.ndarray:
    """
    Vanilla GradCAM — baseline. Included for direct comparison.
    Most artifact-prone on Swin due to patch-based activations.
    """
    cam = GradCAM(model=model, target_layers=_swin_target_layer(model))
    result = cam(input_tensor=input_tensor)
    return result[0]


def _run_scorecam(model, input_tensor) -> np.ndarray:
    """
    ScoreCAM — perturbation-based, no gradients required.
    Smoother than gradient methods; significantly slower (~2–5 s per image).
    """
    cam = ScoreCAM(model=model, target_layers=_swin_target_layer(model))
    result = cam(input_tensor=input_tensor)
    return result[0]


def _run_rise(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    n_masks: int = _RISE_MASKS,
    mask_size: int = _RISE_MASK_S,
    p1: float = _RISE_P1,
) -> np.ndarray:
    """
    RISE — Random Input Sampling for Explanation.
    Fully model-agnostic; generates random binary masks, measures score delta.
    Slowest method (~5–15 s for 500 masks) but produces clean blob-shaped maps.

    Parameters
    ----------
    n_masks   : number of random masks (lower = faster but noisier)
    mask_size : coarse grid size (upsampled to IMG_SIZE)
    p1        : probability of a mask cell being transparent (unmasked)
    """
    model.eval()
    H = W = IMG_SIZE

    # Upsample grid masks to image size
    sal_map = np.zeros((H, W), dtype=np.float32)
    count   = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        # Baseline prediction (unmasked)
        logits_base = model(input_tensor)
        if isinstance(logits_base, tuple):
            logits_base = logits_base[0]
        probs_base = F.softmax(logits_base, dim=1)
        pred_idx   = int(probs_base.argmax(dim=1).item())

        for _ in range(n_masks):
            # Random coarse mask, bilinear upsample
            mask_small = (np.random.rand(mask_size, mask_size) < p1).astype(np.float32)
            mask_up    = np.array(
                Image.fromarray(mask_small).resize((W, H), Image.BILINEAR)
            )
            mask_up = np.clip(mask_up, 0, 1)

            # Apply mask to input tensor
            mask_t = torch.tensor(mask_up, dtype=torch.float32, device=device)
            masked = input_tensor * mask_t.unsqueeze(0).unsqueeze(0)

            logits = model(masked)
            if isinstance(logits, tuple):
                logits = logits[0]
            score = float(F.softmax(logits, dim=1)[0, pred_idx].item())

            sal_map += score * mask_up
            count   += mask_up

    # Normalise
    sal_map = sal_map / (count + 1e-7)
    sal_min, sal_max = sal_map.min(), sal_map.max()
    if sal_max > sal_min:
        sal_map = (sal_map - sal_min) / (sal_max - sal_min)

    return sal_map


def _run_attention_rollout(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str,
) -> np.ndarray:
    """
    Attention Rollout for Swin Transformer.

    Extracts self-attention weights from every SwinTransformerBlock in the
    last stage, averages across heads, rolls attention through blocks
    (product of attention matrices), then upsamples to IMG_SIZE×IMG_SIZE.

    Notes
    -----
    - Swin uses shifted-window attention; each block attends within its own
      local window, so rollout is an approximation across windows.
    - The result can be noisy but is architecturally the most faithful to
      what the transformer actually "looked at."
    """
    model.eval()
    attentions = []

    def _hook(module, inp, out):
        # out is the attn_output; we need the raw attn_weights.
        # Swin's WindowAttention returns (x, attn_weights) when output_attentions=True,
        # but timm's default forward returns only x.  We instead tap the
        # stored `attn` buffer that some timm versions expose, or fall back
        # to using the softmax(QK^T/sqrt(d)) computed inline via a monkey-patch.
        # Simplest robust approach: store the *input* to the softmax via another hook.
        pass

    # Robust fallback: hook into every WindowAttention's softmax input.
    # timm SwinTransformerBlock → attn (WindowAttention) → attn_weights stored after softmax.
    # We hook `WindowAttention.forward` and capture attn_weights via a closure.

    hooks = []
    captured = []

    last_stage = model.layers[-1]

    for block in last_stage.blocks:
        win_attn = block.attn

        def make_hook(wa):
            original_forward = wa.forward

            def patched_forward(x, mask=None):
                # Replicate enough of WindowAttention.forward to get attn_weights
                B_, N, C = x.shape
                qkv = wa.qkv(x).reshape(B_, N, 3, wa.num_heads, C // wa.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q = q * wa.scale
                attn = q @ k.transpose(-2, -1)

                # Relative position bias
                if hasattr(wa, 'relative_position_bias_table'):
                    rp_idx  = wa.relative_position_index.view(-1)
                    rel_pos = wa.relative_position_bias_table[rp_idx].view(
                        wa.window_size[0] * wa.window_size[1],
                        wa.window_size[0] * wa.window_size[1],
                        -1,
                    ).permute(2, 0, 1).contiguous()
                    attn = attn + rel_pos.unsqueeze(0)

                if mask is not None:
                    nW = mask.shape[0]
                    attn = attn.view(B_ // nW, nW, wa.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, wa.num_heads, N, N)

                attn_weights = attn.softmax(dim=-1)
                captured.append(attn_weights.detach().cpu())

                # Continue original forward using the computed weights
                attn_drop = wa.attn_drop(attn_weights)
                x_out = (attn_drop @ v).transpose(1, 2).reshape(B_, N, C)
                x_out = wa.proj(x_out)
                x_out = wa.proj_drop(x_out)
                return x_out

            return patched_forward

        original = win_attn.forward
        win_attn.forward = make_hook(win_attn)
        hooks.append((win_attn, original))

    with torch.no_grad():
        _ = model(input_tensor.to(device))

    # Restore originals
    for wa, orig in hooks:
        wa.forward = orig

    if not captured:
        # Fallback to EigenGradCAM if we couldn't capture attention
        return _run_eigengradcam(model, input_tensor)

    # Average attention weights across all captured blocks and heads
    # Each captured tensor: (num_windows * B, num_heads, N_tokens, N_tokens)
    # Average over heads → (num_windows, N, N), then average window token attention
    rollout = None
    for attn_w in captured:
        # Mean over heads
        attn_mean = attn_w.mean(dim=1)   # (nW, N, N)
        # Add residual connection and re-normalise (standard rollout trick)
        I = torch.eye(attn_mean.shape[-1], device=attn_mean.device).unsqueeze(0)
        attn_mean = 0.5 * attn_mean + 0.5 * I
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)

        if rollout is None:
            rollout = attn_mean.mean(dim=0)   # (N, N) averaged across windows
        else:
            rollout = rollout @ attn_mean.mean(dim=0)

    # Take the mean attention from [CLS] equivalent: mean over all token-to-token
    sal = rollout.mean(dim=0).numpy()   # (N,)

    # Reshape to 2D grid
    n = sal.shape[0]
    side = int(math.isqrt(n))
    if side * side != n:
        # Non-square window; just use a 1D array reshaped to nearest square
        side = int(math.ceil(math.sqrt(n)))
        sal = np.pad(sal, (0, side * side - n))
    sal_2d = sal.reshape(side, side)

    # Upsample to IMG_SIZE × IMG_SIZE
    sal_pil = Image.fromarray((sal_2d * 255).clip(0, 255).astype(np.uint8))
    sal_up  = np.array(sal_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32)

    # Normalise to [0, 1]
    sal_min, sal_max = sal_up.min(), sal_up.max()
    if sal_max > sal_min:
        sal_up = (sal_up - sal_min) / (sal_max - sal_min)

    return sal_up


# ─── Region estimation ────────────────────────────────────────────────────────

def _estimate_region(grayscale_cam: np.ndarray) -> str:
    """
    Map the peak activation on the grayscale CAM to a rough spatial label.

    Strategy
    --------
    1. Compute peak-to-mean ratio; if low (< 1.8), call it "diffuse".
    2. Otherwise, find the centroid of the top-10% activation pixels.
    3. Map the centroid (normalised [0,1] row, col) to a quadrant label.
    """
    H, W = grayscale_cam.shape
    flat  = grayscale_cam.flatten()
    mean  = float(flat.mean())
    peak  = float(flat.max())

    if mean < 1e-6 or (peak / (mean + 1e-6)) < 1.8:
        return "diffuse"

    # Centroid of top-10% pixels
    threshold = np.percentile(flat, 90)
    ys, xs    = np.where(grayscale_cam >= threshold)
    cy = float(ys.mean()) / H   # 0 = top, 1 = bottom
    cx = float(xs.mean()) / W   # 0 = left, 1 = right

    # Vertical split
    if cy < 0.4:
        vert = "superior"
    elif cy > 0.6:
        vert = "inferior"
    else:
        vert = "central"

    # Horizontal split
    if cx < 0.38:
        horiz = "left temporal"
    elif cx > 0.62:
        horiz = "right temporal"
    else:
        horiz = None

    if horiz:
        # Both axes informative → combine
        if vert == "central":
            return horiz
        return f"{vert} {horiz}"
    return vert


# ─── Text explanation ─────────────────────────────────────────────────────────

# Template matrix: TEMPLATES[class_name][tier]
# tier: "high" | "medium" | "low"

_TEMPLATES: dict[str, dict[str, str]] = {
    "NonDemented": {
        "high": (
            "The model found no significant indicators of dementia-related structural change, "
            "with high confidence ({pct}%). Activation was concentrated in the {region} region, "
            "which aligns with normal imaging variation. "
            "Routine follow-up remains advisable given the progressive nature of age-related changes. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "medium": (
            "The model found no strong indicators of dementia-related change at moderate confidence ({pct}%). "
            "Activation in the {region} region did not correspond to typical atrophy patterns. "
            "While reassuring, the intermediate confidence warrants careful clinical review before conclusions are drawn. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "low": (
            "The model found no strong indicators of dementia-related change, though confidence is low ({pct}%). "
            "The scan may contain ambiguous features or imaging artefacts in the {region} region. "
            "A manual review is strongly advised before drawing conclusions. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
    },
    "VeryMildDemented": {
        "high": (
            "The model identified subtle imaging patterns consistent with very mild cognitive impairment "
            "at high confidence ({pct}%). Activation concentrated in the {region} region may reflect "
            "early microstructural changes not yet clinically apparent. "
            "Longitudinal monitoring and a baseline neuropsychological assessment are recommended. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "medium": (
            "The model detected patterns potentially consistent with very mild cognitive impairment ({pct}% confidence). "
            "Activation in the {region} region shows subtle deviation from healthy baselines. "
            "These findings should be interpreted cautiously and correlated with cognitive screening results. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "low": (
            "The model suggested possible very mild cognitive impairment, though confidence is low ({pct}%). "
            "Activation in the {region} region is ambiguous and may reflect normal anatomical variation. "
            "Additional imaging and clinical assessment are recommended before any clinical action. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
    },
    "MildDemented": {
        "high": (
            "The model identified imaging patterns associated with mild cognitive impairment "
            "at high confidence ({pct}%). Activation was concentrated in the {region} region, "
            "consistent with early cortical and hippocampal changes typical of mild Alzheimer's disease. "
            "Further neuropsychological assessment and specialist review are warranted. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "medium": (
            "The model detected patterns suggestive of mild cognitive impairment at moderate confidence ({pct}%). "
            "Activation in the {region} region may reflect early atrophic change. "
            "These findings should be correlated with clinical history and formal cognitive evaluation. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "low": (
            "The model found some indicators of mild cognitive impairment, but confidence is low ({pct}%). "
            "The {region} region shows possible irregularities that may or may not be pathological. "
            "Independent clinical assessment and repeat imaging should be considered. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
    },
    "ModerateDemented": {
        "high": (
            "The model identified imaging patterns strongly associated with moderate dementia "
            "at high confidence ({pct}%). Activation in the {region} region indicates substantial "
            "cortical involvement consistent with moderate-stage Alzheimer's disease. "
            "Urgent specialist referral and comprehensive care planning are recommended. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "medium": (
            "The model detected patterns consistent with moderate dementia at moderate confidence ({pct}%). "
            "Activation in the {region} region suggests notable structural change. "
            "Clinical correlation with cognitive and functional assessments is essential. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
        "low": (
            "The model found possible indicators of moderate dementia, though confidence is low ({pct}%). "
            "Activation in the {region} region is present but does not decisively localise pathology. "
            "A full clinical and neuroimaging workup is strongly advised before any conclusions. "
            "This output is not a diagnostic conclusion. Clinical correlation is required."
        ),
    },
}

# Fallback template for unknown class names
_FALLBACK_TEMPLATE = (
    "The model returned a prediction of '{cls}' at {pct}% confidence. "
    "Activation was concentrated in the {region} region. "
    "Please consult a qualified specialist for interpretation. "
    "This output is not a diagnostic conclusion. Clinical correlation is required."
)


def _generate_text(predicted_class: str, confidence: float, region: str) -> str:
    """
    Produce a natural-language explanation using the template matrix.

    Parameters
    ----------
    predicted_class : one of the four class names (or unknown)
    confidence      : float in [0, 1]
    region          : region label from _estimate_region()
    """
    pct = f"{confidence * 100:.0f}"

    if confidence >= _TIER_HIGH:
        tier = "high"
    elif confidence >= _TIER_MEDIUM:
        tier = "medium"
    else:
        tier = "low"

    templates = _TEMPLATES.get(predicted_class)
    if templates:
        template = templates[tier]
        return template.format(pct=pct, region=region)
    else:
        return _FALLBACK_TEMPLATE.format(cls=predicted_class, pct=pct, region=region)


# ─── Utility ─────────────────────────────────────────────────────────────────

def _pil_to_b64(pil_img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")