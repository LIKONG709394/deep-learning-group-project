# Optional EasyOCR / PaddleOCR backends for slide-like (printed) board regions.

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EASY_READERS: dict[tuple[tuple[str, ...], bool], Any] = {}
_PADDLE_INSTANCES: dict[tuple[str, bool], Any] = {}
_EASY_LOCK = threading.Lock()
_PADDLE_LOCK = threading.Lock()


def normalize_ocr_engine_name(raw: Any) -> str:
    s = str(raw or "trocr").strip().lower()
    if s in ("trocr", "transformers", "hf", "huggingface"):
        return "trocr"
    if s in ("easyocr", "easy"):
        return "easyocr"
    if s in ("paddleocr", "paddle"):
        return "paddleocr"
    logger.warning("Unknown trocr.ocr_engine %r; using trocr", raw)
    return "trocr"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_ocr_use_gpu(trocr_device: Optional[str]) -> bool:
    if trocr_device == "cuda":
        return _cuda_available()
    if trocr_device == "cpu":
        return False
    return _cuda_available()


def _cluster_detections_to_lines(
    entries: List[Tuple[float, float, float, str]],
    *,
    y_gap_ratio: float = 0.55,
) -> List[str]:
    """Group (ymin, ymax, xmin, text) into reading-order lines."""
    cleaned = [(a, b, c, (d or "").strip()) for a, b, c, d in entries if (d or "").strip()]
    if not cleaned:
        return []
    heights = [max(1.0, ymax - ymin) for ymin, ymax, _, _ in cleaned]
    med_h = float(np.median(heights)) if heights else 12.0
    y_thresh = max(5.0, med_h * y_gap_ratio)
    sorted_e = sorted(cleaned, key=lambda t: ((t[0] + t[1]) / 2.0, t[2]))
    lines_blocks: List[List[Tuple[float, str]]] = []
    current: List[Tuple[float, float, float, str]] = []
    for e in sorted_e:
        ymin, ymax, xmin, text = e
        cy = (ymin + ymax) / 2.0
        if not current:
            current = [e]
            continue
        prev_cy = sum((a[0] + a[1]) / 2.0 for a in current) / len(current)
        if abs(cy - prev_cy) <= y_thresh:
            current.append(e)
        else:
            current.sort(key=lambda x: x[2])
            lines_blocks.append([(x[2], x[3]) for x in current])
            current = [e]
    if current:
        current.sort(key=lambda x: x[2])
        lines_blocks.append([(x[2], x[3]) for x in current])
    return [" ".join(t for _, t in block).strip() for block in lines_blocks if block]


def _easyocr_detections_to_entries(result: Sequence[Any]) -> List[Tuple[float, float, float, str]]:
    out: List[Tuple[float, float, float, str]] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        bbox, text = item[0], item[1]
        conf = float(item[2]) if len(item) > 2 else 1.0
        if conf < 0.15:
            continue
        try:
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
        except (TypeError, IndexError, ValueError):
            continue
        if not xs or not ys:
            continue
        out.append((min(ys), max(ys), min(xs), str(text).strip()))
    return out


def _paddle_detections_to_entries(ocr_result: Any) -> List[Tuple[float, float, float, str]]:
    out: List[Tuple[float, float, float, str]] = []
    if not ocr_result or ocr_result[0] is None:
        return out
    for line in ocr_result[0]:
        if not line or len(line) < 2:
            continue
        box, tx = line[0], line[1]
        if not isinstance(tx, (list, tuple)) or len(tx) < 1:
            continue
        text = str(tx[0] or "").strip()
        conf = float(tx[1]) if len(tx) > 1 else 1.0
        if conf < 0.15 or not text:
            continue
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
        except (TypeError, IndexError, ValueError):
            continue
        if not xs or not ys:
            continue
        out.append((min(ys), max(ys), min(xs), text))
    return out


def recognize_lines_easyocr(
    image_bgr: np.ndarray,
    *,
    languages: Sequence[str],
    use_gpu: bool,
) -> List[str]:
    try:
        import easyocr
    except ImportError as e:
        raise RuntimeError(
            "easyocr is not installed. Run: pip install easyocr"
        ) from e

    langs = tuple(str(x).strip() for x in languages if str(x).strip()) or ("en",)
    key = (langs, use_gpu)
    with _EASY_LOCK:
        reader = _EASY_READERS.get(key)
        if reader is None:
            reader = easyocr.Reader(list(langs), gpu=use_gpu)
            _EASY_READERS[key] = reader
    if image_bgr is None or image_bgr.size == 0:
        return []
    result = reader.readtext(image_bgr)
    entries = _easyocr_detections_to_entries(result)
    return _cluster_detections_to_lines(entries)


def recognize_lines_paddleocr(
    image_bgr: np.ndarray,
    *,
    lang: str,
    use_gpu: bool,
) -> List[str]:
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise RuntimeError(
            "paddleocr is not installed. Run: pip install paddlepaddle paddleocr"
        ) from e

    lang = str(lang or "en").strip() or "en"
    key = (lang, use_gpu)
    with _PADDLE_LOCK:
        ocr = _PADDLE_INSTANCES.get(key)
        if ocr is None:
            kwargs: dict[str, Any] = {
                "use_angle_cls": True,
                "lang": lang,
                "show_log": False,
            }
            try:
                ocr = PaddleOCR(**kwargs, use_gpu=use_gpu)
            except TypeError:
                try:
                    ocr = PaddleOCR(**kwargs)
                except Exception as e2:
                    raise RuntimeError(f"Failed to init PaddleOCR: {e2}") from e2
            _PADDLE_INSTANCES[key] = ocr
    if image_bgr is None or image_bgr.size == 0:
        return []
    try:
        res = ocr.ocr(image_bgr, cls=True)
    except Exception as e:
        logger.warning("PaddleOCR.ocr failed: %s", e)
        return []
    entries = _paddle_detections_to_entries(res)
    return _cluster_detections_to_lines(entries)
