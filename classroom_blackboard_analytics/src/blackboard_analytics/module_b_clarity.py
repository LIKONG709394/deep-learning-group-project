# Guess how readable the board photo is: sharpness (Laplacian) plus whether stroke
# thickness jumps around a lot between ink blobs. Combined into 0-100 and clear/fair/poor.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def laplacian_variance(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    g = gray
    if len(g.shape) == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap.var())


def _binarize_for_strokes(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray) if len(gray.shape) == 2 else clahe.apply(cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    return bw


def _stroke_widths_per_component(binary_inv: np.ndarray) -> Tuple[List[float], float]:
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
    # Map Laplacian variance to 0..1 (higher variance = sharper photo).
    if lap_var >= lap_clear:
        sharpness_01 = 1.0
    elif lap_var <= lap_messy:
        sharpness_01 = 0.0
    else:
        sharpness_01 = (lap_var - lap_messy) / (lap_clear - lap_messy + 1e-6)

    # Penalize uneven stroke width across connected components.
    stroke_penalty = min(1.0, stroke_var / max(stroke_messy_min, 1e-6))
    stroke_consistency_01 = max(0.0, 1.0 - stroke_penalty)

    total = 100.0 * (0.55 * sharpness_01 + 0.45 * stroke_consistency_01)
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
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_metric = laplacian_variance(gray)
    ink_mask = _binarize_for_strokes(gray)
    stroke_widths, stroke_width_variance = _stroke_widths_per_component(ink_mask)

    label, score, suggestion = _score_to_clarity(
        blur_metric,
        stroke_width_variance,
        laplacian_clear_min,
        laplacian_messy_max,
        stroke_variance_messy_min,
    )

    return {
        "clarity": label,
        "score": round(score, 2),
        "suggestion": suggestion,
        "laplacian_variance": blur_metric,
        "stroke_width_variance": stroke_width_variance,
        "details": {"num_stroke_components": len(stroke_widths)},
    }


def run_module_b(
    image: np.ndarray,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    thresholds = (config or {}).get("clarity", {})
    result = {"clarity_result": None, "error": None}
    try:
        result["clarity_result"] = evaluate_handwriting_clarity(
            image,
            laplacian_clear_min=float(thresholds.get("laplacian_clear_min", 120.0)),
            laplacian_messy_max=float(thresholds.get("laplacian_messy_max", 40.0)),
            stroke_variance_messy_min=float(thresholds.get("stroke_variance_messy_min", 8.0)),
        )
    except Exception as e:
        result["error"] = str(e)
        logger.exception("run_module_b")
    return result
