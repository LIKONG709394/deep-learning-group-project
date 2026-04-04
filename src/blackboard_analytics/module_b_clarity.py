"""
Module B: handwriting clarity (Laplacian variance + stroke-width consistency) and heatmap visualization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:
    plt = None  # type: ignore
    cm = None  # type: ignore


def laplacian_variance(gray: np.ndarray) -> float:
    """Global blur score: variance of Laplacian response (higher usually sharper)."""
    if gray is None or gray.size == 0:
        return 0.0
    g = gray
    if len(g.shape) == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap.var())


def _binarize_for_strokes(gray: np.ndarray) -> np.ndarray:
    """Binary image with text as foreground 255."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray) if len(gray.shape) == 2 else clahe.apply(cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    return bw


def _stroke_widths_per_component(binary_inv: np.ndarray) -> Tuple[List[float], float]:
    """
    Per connected component: mean stroke radius via distance transform.
    Returns per-component means and variance across components.
    """
    dist = cv2.distanceTransform((binary_inv > 0).astype(np.uint8), cv2.DIST_L2, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_inv > 0).astype(np.uint8))
    widths: List[float] = []
    h, w = binary_inv.shape[:2]
    min_area = max(20, int(0.0001 * h * w))
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = labels == i
        vals = dist[mask]
        if vals.size == 0:
            continue
        m = float(np.mean(vals))
        if m > 0.5:
            widths.append(m)
    if len(widths) < 2:
        var_between = float(np.var(widths)) if widths else 0.0
        return widths, var_between
    var_between = float(np.var(widths))
    return widths, var_between


def _score_to_clarity(
    lap_var: float,
    stroke_var: float,
    lap_clear: float,
    lap_messy: float,
    stroke_messy_min: float,
) -> Tuple[str, float, str]:
    """
    Combined 0–100 score and three-level label.
    High lap_var and low stroke_var -> higher score.
    """
    if lap_var >= lap_clear:
        lap_s = 1.0
    elif lap_var <= lap_messy:
        lap_s = 0.0
    else:
        lap_s = (lap_var - lap_messy) / (lap_clear - lap_messy + 1e-6)

    stroke_penalty = min(1.0, stroke_var / max(stroke_messy_min, 1e-6))
    stroke_s = max(0.0, 1.0 - stroke_penalty)

    total = 100.0 * (0.55 * lap_s + 0.45 * stroke_s)
    total = float(max(0.0, min(100.0, total)))

    if total >= 70:
        label = "clear"
        suggestion = "Good contrast and stroke consistency; expected readable from the back of the room."
    elif total >= 45:
        label = "fair"
        suggestion = "Consider stronger contrast or slower writing; avoid very thin strokes and rapid erase-write cycles."
    else:
        label = "poor"
        suggestion = "Use a thicker marker, increase black/white contrast, and enlarge or zone key terms for readability."

    return label, total, suggestion


def evaluate_handwriting_clarity(
    image: np.ndarray,
    *,
    laplacian_clear_min: float = 120.0,
    laplacian_messy_max: float = 40.0,
    stroke_variance_messy_min: float = 8.0,
) -> Dict[str, Any]:
    """
    Input: blackboard image (BGR or grayscale numpy).

    Output:
        clarity, score, suggestion, laplacian_variance, stroke_width_variance, details
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_v = laplacian_variance(gray)
    bw = _binarize_for_strokes(gray)
    stroke_widths, stroke_var = _stroke_widths_per_component(bw)

    label, score, suggestion = _score_to_clarity(
        lap_v,
        stroke_var,
        laplacian_clear_min,
        laplacian_messy_max,
        stroke_variance_messy_min,
    )

    return {
        "clarity": label,
        "score": round(score, 2),
        "suggestion": suggestion,
        "laplacian_variance": lap_v,
        "stroke_width_variance": stroke_var,
        "details": {"num_stroke_components": len(stroke_widths)},
    }


def clarity_heatmap_data(
    gray: np.ndarray,
    grid_rows: int = 24,
    grid_cols: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split image into a grid; Laplacian variance per cell. Returns (heatmap, grid_shape)."""
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    cell_h = max(1, h // grid_rows)
    cell_w = max(1, w // grid_cols)
    heat = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    for i in range(grid_rows):
        for j in range(grid_cols):
            y0, y1 = i * cell_h, min(h, (i + 1) * cell_h)
            x0, x1 = j * cell_w, min(w, (j + 1) * cell_w)
            patch = gray[y0:y1, x0:x1]
            heat[i, j] = laplacian_variance(patch) if patch.size else 0.0
    return heat, np.array([grid_rows, grid_cols])


def plot_clarity_heatmap(
    image_bgr: np.ndarray,
    *,
    grid_rows: int = 24,
    grid_cols: int = 32,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[Any]:
    """
    Matplotlib: original + clarity heatmap (local Laplacian variance, normalized).
    Returns figure or None if matplotlib missing.
    """
    if plt is None or cm is None:
        logger.warning("matplotlib not installed; skipping heatmap.")
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    heat, _ = clarity_heatmap_data(gray, grid_rows=grid_rows, grid_cols=grid_cols)
    heat_norm = heat.copy()
    if heat_norm.max() > heat_norm.min():
        heat_norm = (heat_norm - heat_norm.min()) / (heat_norm.max() - heat_norm.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    im = axes[1].imshow(heat_norm, cmap="inferno", interpolation="nearest")
    axes[1].set_title("Clarity heatmap (local Laplacian variance, normalized)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def run_module_b(
    image: np.ndarray,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    cfg = (config or {}).get("clarity", {})
    result = {"clarity_result": None, "error": None}
    try:
        result["clarity_result"] = evaluate_handwriting_clarity(
            image,
            laplacian_clear_min=float(cfg.get("laplacian_clear_min", 120.0)),
            laplacian_messy_max=float(cfg.get("laplacian_messy_max", 40.0)),
            stroke_variance_messy_min=float(cfg.get("stroke_variance_messy_min", 8.0)),
        )
    except Exception as e:
        result["error"] = str(e)
        logger.exception("run_module_b")
    return result
