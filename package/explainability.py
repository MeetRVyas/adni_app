from __future__ import annotations

import io
import base64
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from package.config import IMG_SIZE

_TIER_HIGH    = 0.80
_TIER_MEDIUM  = 0.55
_GRADCAM_ALPHA = 0.4   # heatmap blend strength in overlay


def _reshape_transform_swin(tensor, height: int = 7, width: int = 7) -> torch.Tensor:
    """
    Reshape Swin's (B, N, C) token tensor → (B, C, H, W) for spatial pooling.
    Mirrors reshape_transform_swin from the reference exactly.
    """
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class NativeGradCAM:
    """
    Hook-based GradCAM — no pytorch-grad-cam dependency.
      - forward hook  → saves activations
      - backward hook → saves gradients
      - reshape_transform applied to both before pooling
      - weights = mean(grads, dim=(2,3)), cam = sum(weights * fmaps, dim=1)
      - ReLU + min-max normalisation
      - remove_hooks() called in a finally block by the caller
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        reshape_transform=None,
    ):
        self.model             = model.eval()
        self.reshape_transform = reshape_transform
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self.hooks = []

        self.hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, inp, output):
        self.activations = (
            output.clone() if isinstance(output, torch.Tensor) else output
        )

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = (
            grad_output[0].clone()
            if isinstance(grad_output[0], torch.Tensor)
            else grad_output[0]
        )

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Forward + backward pass → (H, W) GradCAM heatmap in [0, 1]."""
        self.model.zero_grad()
        self.model.eval()

        with torch.enable_grad():
            inp    = input_tensor.clone().detach().requires_grad_(True)
            output = self.model(inp)
            pred_index = output.argmax(dim=1)
            score  = output[:, pred_index]
            score.backward()

        grads = self.gradients
        fmaps = self.activations

        if self.reshape_transform is not None:
            grads = self.reshape_transform(grads)
            fmaps = self.reshape_transform(fmaps)

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam     = torch.sum(weights * fmaps, dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam - cam.min()
        cam     = cam / (cam.max() + 1e-7)

        return cam.detach().cpu().numpy()[0, 0]   # (H, W) float32 in [0, 1]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def _swin_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Last norm layer in Swin's final stage — matches the reference heuristic."""
    return model.layers[-1].blocks[-1].norm2


def _gradcam_overlay(
    heatmap: np.ndarray,
    original_img_np: np.ndarray,
    alpha: float = _GRADCAM_ALPHA,
) -> np.ndarray:
    """
    Resize heatmap, apply COLORMAP_JET, blend with addWeighted.
    Mirrors generate_gradcam_plot from the reference exactly.
    """
    h, w = original_img_np.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img = original_img_np
    if img.max() <= 1.0:
        img = np.uint8(255 * img)
    else:
        img = np.uint8(img)

    return cv2.addWeighted(img, 1, heatmap_colored, alpha, 0)


def explain_image(
    pil_img: Image.Image,
    model: torch.nn.Module,
    transform,
    class_names: list[str],
    predicted_class: str,
    confidence: float,
    device: str = "cpu",
) -> dict:
    """Run GradCAM + text explanation on a single PIL image."""
    orig_w, orig_h = pil_img.size
    img_resized    = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    original_np    = np.array(img_resized)           # uint8 (H, W, 3)
    input_tensor   = transform(img_resized).unsqueeze(0).to(device)

    # ── GradCAM ───────────────────────────────────────────────────────────────
    cam_engine = NativeGradCAM(
        model             = model,
        target_layer      = _swin_target_layer(model),
        reshape_transform = _reshape_transform_swin,
    )
    try:
        grayscale_cam = cam_engine(input_tensor)
    finally:
        cam_engine.remove_hooks()

    # ── Overlay ───────────────────────────────────────────────────────────────
    overlay_np  = _gradcam_overlay(grayscale_cam, original_np)
    overlay_pil = Image.fromarray(overlay_np).resize((orig_w, orig_h), Image.LANCZOS)

    # ── Region + text ─────────────────────────────────────────────────────────
    region = _estimate_region(grayscale_cam)
    text   = _generate_text(predicted_class, confidence, region)

    # ── Encode ────────────────────────────────────────────────────────────────
    return {
        "original_b64":  _pil_to_b64(img_resized),
        "overlay_b64":   _pil_to_b64(overlay_pil),
        "text":          text,
        "region":        region,
        "grayscale_cam": grayscale_cam,
    }


def _estimate_region(grayscale_cam: np.ndarray) -> str:
    H, W  = grayscale_cam.shape
    flat  = grayscale_cam.flatten()
    mean  = float(flat.mean())
    peak  = float(flat.max())

    if mean < 1e-6 or (peak / (mean + 1e-6)) < 1.8:
        return "diffuse"

    threshold = np.percentile(flat, 90)
    ys, xs    = np.where(grayscale_cam >= threshold)
    cy = float(ys.mean()) / H
    cx = float(xs.mean()) / W

    if cy < 0.4:
        vert = "superior"
    elif cy > 0.6:
        vert = "inferior"
    else:
        vert = "central"

    if cx < 0.38:
        horiz = "left temporal"
    elif cx > 0.62:
        horiz = "right temporal"
    else:
        horiz = None

    if horiz:
        if vert == "central":
            return horiz
        return f"{vert} {horiz}"
    return vert


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

_FALLBACK_TEMPLATE = (
    "The model returned a prediction of '{cls}' at {pct}% confidence. "
    "Activation was concentrated in the {region} region. "
    "Please consult a qualified specialist for interpretation. "
    "This output is not a diagnostic conclusion. Clinical correlation is required."
)


def _generate_text(predicted_class: str, confidence: float, region: str) -> str:
    pct = f"{confidence * 100:.0f}"

    if confidence >= _TIER_HIGH:
        tier = "high"
    elif confidence >= _TIER_MEDIUM:
        tier = "medium"
    else:
        tier = "low"

    templates = _TEMPLATES.get(predicted_class)
    if templates:
        return templates[tier].format(pct=pct, region=region)
    return _FALLBACK_TEMPLATE.format(cls=predicted_class, pct=pct, region=region)


def _pil_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")