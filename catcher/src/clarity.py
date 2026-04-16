"""
clarity.py — Handwriting clarity scoring for Catcher

Method:
1. Measure image sharpness via Laplacian variance
2. Estimate stroke-width consistency from connected components
3. Combine both into a 0–100 score
4. Return label + suggestion for report generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def evaluate_clarity(
    image_bgr: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Public entry point used by pipeline.py.

    Returns:
    {
        "score": 73.4,
        "label": "clear" | "fair" | "poor",
        "suggestion": "...",
        "laplacian": 182.2,
        "stroke_width_variance": 1.41,
        "num_stroke_components": 27,
    }
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to evaluate_clarity().")

    clarity_cfg = config.get("clarity") or {}
    lap_clear = float(clarity_cfg.get("laplacian_clear_min", 300.0))
    lap_messy = float(clarity_cfg.get("laplacian_messy_max", 80.0))
    stroke_messy = float(clarity_cfg.get("stroke_variance_messy_min", 2.5))

    gray = _to_gray(image_bgr)
    lap_var = laplacian_variance(gray)

    ink_mask = binarize_for_strokes(gray)
    stroke_widths, stroke_var = stroke_widths_per_component(ink_mask)

    label, score, suggestion = score_to_clarity(
        lap_var=lap_var,
        stroke_var=stroke_var,
        lap_clear=lap_clear,
        lap_messy=lap_messy,
        stroke_messy_min=stroke_messy,
    )

    return {
        "score": round(score, 2),
        "label": label,
        "suggestion": suggestion,
        "laplacian": round(lap_var, 2),
        "stroke_width_variance": round(stroke_var, 4),
        "num_stroke_components": len(stroke_widths),
    }


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr.copy()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def laplacian_variance(gray: np.ndarray) -> float:
    """
    Higher = sharper / clearer image.
    """
    if gray is None or gray.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def binarize_for_strokes(gray: np.ndarray) -> np.ndarray:
    """
    Produce inverse binary ink mask:
    white strokes on black background.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    binary_inv = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    return binary_inv


def stroke_widths_per_component(binary_inv: np.ndarray) -> Tuple[List[float], float]:
    """
    Approximate stroke width using distance transform on connected components.
    Returns:
      (list_of_mean_half-widths, variance_between_components)
    """
    if binary_inv is None or binary_inv.size == 0:
        return [], 0.0

    fg = (binary_inv > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

    h, w = fg.shape[:2]
    min_area = max(20, int(0.0001 * h * w))

    widths: List[float] = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        mask = labels == i
        vals = dist[mask]
        if vals.size == 0:
            continue

        mean_half_width = float(np.mean(vals))
        if mean_half_width > 0.5:
            widths.append(mean_half_width)

    if len(widths) < 2:
        return widths, 0.0

    variance_between = float(np.var(widths))
    return widths, variance_between


def score_to_clarity(
    lap_var: float,
    stroke_var: float,
    lap_clear: float,
    lap_messy: float,
    stroke_messy_min: float,
) -> Tuple[str, float, str]:
    """
    Convert metrics into:
    - label: clear / fair / poor
    - score: 0..100
    - suggestion
    """
    # Sharpness score, 0..1
    if lap_var >= lap_clear:
        sharpness01 = 1.0
    elif lap_var <= lap_messy:
        sharpness01 = 0.0
    else:
        sharpness01 = (lap_var - lap_messy) / max(lap_clear - lap_messy, 1e-6)

    # Lower stroke variance is better
    stroke_penalty = min(1.0, stroke_var / max(stroke_messy_min, 1e-6))
    stroke_consistency01 = max(0.0, 1.0 - stroke_penalty)

    # Weighted fusion
    total = 100.0 * (0.55 * sharpness01 + 0.45 * stroke_consistency01)
    total = float(max(0.0, min(100.0, total)))

    if total >= 70:
        label = "clear"
        suggestion = (
            "Good contrast and stroke consistency — likely readable from the back of the room."
        )
    elif total >= 45:
        label = "fair"
        suggestion = (
            "Moderate readability — consider stronger contrast, slower writing, or thicker strokes."
        )
    else:
        label = "poor"
        suggestion = (
            "Low readability — use a thicker marker/chalk, increase contrast, and enlarge key terms."
        )

    return label, total, suggestion