from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image



def cam_statistics(grayscale_cam: np.ndarray) -> dict:
    """Derive rich spatial statistics from a (H, W) GradCAM activation map."""
    H, W = grayscale_cam.shape
    flat = grayscale_cam.flatten()
    peak = float(flat.max())
    mean = float(flat.mean())

    # ── Shannon entropy ───────────────────────────────────────────────────────
    p       = flat / (flat.sum() + 1e-9)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))

    # ── Focus score ───────────────────────────────────────────────────────────
    threshold   = np.percentile(flat, 90)
    top_mask    = grayscale_cam >= threshold
    focus_score = float(grayscale_cam[top_mask].sum() / (grayscale_cam.sum() + 1e-9))

    # ── Centroid & spread of top-10% pixels ───────────────────────────────────
    ys, xs = np.where(top_mask)
    if len(ys) > 0:
        cy = float(ys.mean()) / H
        cx = float(xs.mean()) / W
        sy = float(ys.std())  / H
        sx = float(xs.std())  / W
    else:
        cy = cx = 0.5
        sy = sx = 0.0

    # ── Quadrant weights ──────────────────────────────────────────────────────
    half_h, half_w = H // 2, W // 2
    quads = {
        "top_left":     float(grayscale_cam[:half_h, :half_w].sum()),
        "top_right":    float(grayscale_cam[:half_h, half_w:].sum()),
        "bottom_left":  float(grayscale_cam[half_h:, :half_w].sum()),
        "bottom_right": float(grayscale_cam[half_h:, half_w:].sum()),
    }
    total = sum(quads.values()) + 1e-9
    quads = {k: round(v / total, 4) for k, v in quads.items()}

    # ── Hotspot count via BFS on a downsampled binary map ────────────────────
    small = np.array(
        Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize(
            (32, 32), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    binary       = (small >= 0.5).astype(np.int8)
    num_hotspots = _count_blobs(binary)

    return {
        "peak":             round(peak, 4),
        "mean":             round(mean, 4),
        "peak_mean_ratio":  round(peak / (mean + 1e-9), 2),
        "entropy":          round(entropy, 4),
        "focus_score":      round(focus_score, 4),
        "centroid_x":       round(cx, 4),
        "centroid_y":       round(cy, 4),
        "spread_x":         round(sx, 4),
        "spread_y":         round(sy, 4),
        "quadrant_weights": quads,
        "num_hotspots":     num_hotspots,
    }


def _count_blobs(binary: np.ndarray) -> int:
    """BFS 4-connected component count on a 2D binary array. No scipy needed."""
    visited = np.zeros_like(binary, dtype=bool)
    H, W    = binary.shape
    count   = 0

    for r in range(H):
        for c in range(W):
            if binary[r, c] == 1 and not visited[r, c]:
                count += 1
                queue = [(r, c)]
                while queue:
                    cr, cc = queue.pop()
                    if cr < 0 or cr >= H or cc < 0 or cc >= W:
                        continue
                    if visited[cr, cc] or binary[cr, cc] != 1:
                        continue
                    visited[cr, cc] = True
                    queue.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
    return count


# ─── Batch summary statistics ─────────────────────────────────────────────────

def batch_summary(results: List[dict]) -> dict:
    """
    Compute aggregate statistics over a list of per-image prediction dicts
    (as returned by /predict with mode='per_image').
    """
    if not results:
        return {}

    class_names = list(results[0]["probabilities"].keys())

    class_dist: dict[str, int]         = {c: 0   for c in class_names}
    all_probs:  dict[str, list[float]] = {c: []  for c in class_names}
    confidences: list[float]           = []

    for r in results:
        pred = r.get("predicted_class", "")
        if pred in class_dist:
            class_dist[pred] += 1
        confidences.append(r.get("confidence", 0.0))
        for c in class_names:
            all_probs[c].append(r["probabilities"].get(c, 0.0))

    mean_probs = {c: float(np.mean(v)) for c, v in all_probs.items()}
    std_probs  = {c: float(np.std(v))  for c, v in all_probs.items()}
    max_probs  = {c: float(np.max(v))  for c, v in all_probs.items()}
    dominant   = max(mean_probs, key=mean_probs.get)

    hist = [0] * 10
    for conf in confidences:
        hist[min(int(conf * 10), 9)] += 1

    return {
        "num_images":         len(results),
        "class_distribution": class_dist,
        "mean_probabilities": {k: round(v, 4) for k, v in mean_probs.items()},
        "std_probabilities":  {k: round(v, 4) for k, v in std_probs.items()},
        "max_probabilities":  {k: round(v, 4) for k, v in max_probs.items()},
        "dominant_class":     dominant,
        "mean_confidence":    round(float(np.mean(confidences)), 4),
        "confidence_hist":    hist,
    }