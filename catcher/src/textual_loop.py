"""
textual_loop.py — Consumer side of Catcher
Reads candidate keyframes produced by capture_loop.py,
runs OCR on each ROI, deduplicates, and collects board lines.

Producer-consumer handshake:
    capture_loop appends dicts to keyframe_results (shared list)
    textual_loop polls that list and processes unread entries
    stop_event signals that capture is done
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Strip, lowercase, collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_substring_of_existing(line: str, existing: List[str], min_len: int = 8) -> bool:
    """Return True if line is already covered by an existing longer line."""
    norm = _normalise_text(line)
    if len(norm) < min_len:
        return False
    for ex in existing:
        norm_ex = _normalise_text(ex)
        if norm in norm_ex and norm != norm_ex:
            return True
    return False


def _dedupe_lines(new_lines: List[str], existing: List[str], min_len: int = 8) -> List[str]:
    """
    Filter new_lines against existing:
    - skip exact duplicates (case-insensitive)
    - skip lines already subsumed by a longer existing line
    """
    seen = {_normalise_text(e) for e in existing}
    result = []
    for line in new_lines:
        norm = _normalise_text(line)
        if not norm:
            continue
        if norm in seen:
            continue
        if _is_substring_of_existing(line, existing, min_len):
            continue
        seen.add(norm)
        result.append(line.strip())
    return result


# ---------------------------------------------------------------------------
# Visual similarity gate (skip near-identical frames)
# ---------------------------------------------------------------------------

def _frame_hash(roi_bgr: np.ndarray, size: int = 16) -> np.ndarray:
    """Tiny perceptual hash — resize to size×size, return uint8 array."""
    import cv2
    small = cv2.resize(roi_bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.flatten()


def _similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Normalised dot product similarity between two frame hashes."""
    a = h1.astype(np.float32)
    b = h2.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Clarity scoring (used to pick best OCR result per frame)
# ---------------------------------------------------------------------------

def _laplacian_variance(gray: np.ndarray) -> float:
    import cv2
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _clarity_label(lap_var: float) -> str:
    if lap_var >= 300.0:
        return "clear"
    if lap_var >= 80.0:
        return "fair"
    return "poor"


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def run_textual_loop(
    keyframe_results: List[dict],
    config: Dict[str, Any],
    stop_event: threading.Event,
    board_lines: List[str],
) -> None:
    """
    Consumer loop.

    Polls keyframe_results for unprocessed entries from capture_loop.
    For each entry, runs OCR and merges unique lines into board_lines.

    Exits when stop_event is set AND all entries have been processed.
    """
    import cv2

    video_cfg = _get_video_cfg(config)
    min_len   = int(video_cfg.get("merge_substring_min_len", 8))
    threshold = float(video_cfg.get("dedupe_similarity_threshold", 0.985))
    min_interval = float(video_cfg.get("dedupe_min_interval_sec", 8.0))

    processed_count = 0
    prev_hash: Optional[np.ndarray] = None
    prev_timestamp: float = -999.0

    while True:
        # Are there new entries to process?
        if processed_count < len(keyframe_results):
            entry = keyframe_results[processed_count]

            roi_bgr     = entry.get("roi_bgr")
            timestamp   = float(entry.get("timestamp_sec", 0.0))
            frame_index = int(entry.get("frame_index", 0))

            if roi_bgr is not None and roi_bgr.size > 0:
                # --- Visual dedup gate ---
                h = _frame_hash(roi_bgr)
                skip = False
                if prev_hash is not None:
                    sim = _similarity(h, prev_hash)
                    dt  = abs(timestamp - prev_timestamp)
                    if sim >= threshold and dt < min_interval:
                        skip = True
                        logger.debug(
                            "Frame %d (t=%.1fs) skipped — similarity %.4f",
                            frame_index, timestamp, sim
                        )

                if not skip:
                    # --- Run OCR ---
                    ocr_lines, ocr_meta = _run_ocr(roi_bgr, config)

                    # --- Clarity score for this frame ---
                    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                    lap  = _laplacian_variance(gray)
                    clarity_result = {
                        "score":      round(lap, 2),
                        "label":      _clarity_label(lap),
                        "laplacian":  round(lap, 2),
                        "suggestion": _clarity_suggestion(_clarity_label(lap)),
                    }

                    # --- Merge unique lines into global board_lines ---
                    fresh = _dedupe_lines(ocr_lines, board_lines, min_len=min_len)
                    board_lines.extend(fresh)

                    # --- Write results back into the shared entry ---
                    entry["ocr_lines"]      = ocr_lines
                    entry["ocr_meta"]       = ocr_meta
                    entry["clarity_result"] = clarity_result
                    entry["new_lines"]      = fresh

                    prev_hash      = h
                    prev_timestamp = timestamp

                    logger.info(
                        "Frame %d (t=%.1fs) | OCR lines: %d | new: %d | clarity: %s (%.0f)",
                        frame_index, timestamp,
                        len(ocr_lines), len(fresh),
                        clarity_result["label"], lap
                    )
                else:
                    # Mark as processed even when skipped
                    entry["ocr_lines"]      = []
                    entry["ocr_meta"]       = {"skipped": "visual_dedup"}
                    entry["clarity_result"] = {}
                    entry["new_lines"]      = []

            else:
                logger.warning("Frame %d has no ROI data — skipping OCR.", frame_index)
                entry["ocr_lines"]      = []
                entry["ocr_meta"]       = {"skipped": "empty_roi"}
                entry["clarity_result"] = {}
                entry["new_lines"]      = []

            entry["processed"] = True
            processed_count += 1

        else:
            # Nothing new yet
            if stop_event.is_set() and processed_count >= len(keyframe_results):
                # Capture is done and we've caught up — exit
                break
            time.sleep(0.05)   # short sleep before polling again


# ---------------------------------------------------------------------------
# OCR dispatcher — routes to the engine set in config
# ---------------------------------------------------------------------------

def _run_ocr(
    roi_bgr: np.ndarray,
    config: Dict[str, Any],
) -> tuple[List[str], Dict[str, Any]]:
    """
    Dispatch OCR to the configured engine.
    Returns (text_lines, meta_dict).
    Falls back gracefully if an engine fails.
    """
    trocr_cfg  = config.get("trocr") or {}
    engine     = str(trocr_cfg.get("ocr_engine", "easyocr")).strip().lower()

    try:
        if engine == "easyocr":
            return _ocr_easyocr(roi_bgr, config)
        elif engine == "paddleocr":
            return _ocr_paddleocr(roi_bgr, config)
        elif engine == "trocr":
            return _ocr_trocr(roi_bgr, config)
        else:
            logger.warning("Unknown OCR engine '%s' — falling back to easyocr.", engine)
            return _ocr_easyocr(roi_bgr, config)
    except Exception as e:
        logger.error("OCR engine '%s' failed: %s — trying easyocr fallback.", engine, e)
        try:
            return _ocr_easyocr(roi_bgr, config)
        except Exception as e2:
            logger.error("EasyOCR fallback also failed: %s", e2)
            return [], {"error": str(e2)}


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

_EASYOCR_READERS: Dict[str, Any] = {}
_EASYOCR_LOCK = threading.Lock()


def _ocr_easyocr(
    roi_bgr: np.ndarray,
    config: Dict[str, Any],
) -> tuple[List[str], Dict[str, Any]]:
    import easyocr

    trocr_cfg = config.get("trocr") or {}
    raw_langs = trocr_cfg.get("easyocr_languages") or ["en"]
    langs     = [str(x).strip() for x in raw_langs if str(x).strip()]
    key       = tuple(langs)

    with _EASYOCR_LOCK:
        if key not in _EASYOCR_READERS:
            _EASYOCR_READERS[key] = easyocr.Reader(langs, gpu=_use_gpu(config))
        reader = _EASYOCR_READERS[key]

    results = reader.readtext(roi_bgr)

    lines = []
    for item in results:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        conf = float(item[2]) if len(item) > 2 else 1.0
        if conf < 0.15 or not text:
            continue
        lines.append(text)

    return lines, {"engine": "easyocr", "count": len(lines)}


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

_PADDLE_INSTANCES: Dict[str, Any] = {}
_PADDLE_LOCK = threading.Lock()


def _ocr_paddleocr(
    roi_bgr: np.ndarray,
    config: Dict[str, Any],
) -> tuple[List[str], Dict[str, Any]]:
    from paddleocr import PaddleOCR

    trocr_cfg = config.get("trocr") or {}
    lang      = str(trocr_cfg.get("paddleocr_lang", "en")).strip() or "en"
    use_gpu   = _use_gpu(config)
    key       = (lang, use_gpu)

    with _PADDLE_LOCK:
        if key not in _PADDLE_INSTANCES:
            _PADDLE_INSTANCES[key] = PaddleOCR(
                lang=lang,
                device="gpu:0" if use_gpu else "cpu",
                use_text_line_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        ocr = _PADDLE_INSTANCES[key]

    result = ocr.predict(roi_bgr)
    lines  = []

    for item in (result or []):
        if not item or len(item) < 2:
            continue
        bbox, tc = item[0], item[1]
        if not isinstance(tc, (list, tuple)) or len(tc) < 1:
            continue
        text = str(tc[0]).strip()
        conf = float(tc[1]) if len(tc) > 1 else 1.0
        if conf < 0.15 or not text:
            continue
        lines.append(text)

    return lines, {"engine": "paddleocr", "count": len(lines)}


# ---------------------------------------------------------------------------
# TrOCR
# ---------------------------------------------------------------------------

_TROCR_PIPELINES: Dict[str, Any] = {}
_TROCR_LOCK = threading.Lock()


def _ocr_trocr(
    roi_bgr: np.ndarray,
    config: Dict[str, Any],
) -> tuple[List[str], Dict[str, Any]]:
    import cv2
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch

    trocr_cfg  = config.get("trocr") or {}
    model_name = str(
        trocr_cfg.get("default_model", "microsoft/trocr-base-handwritten")
    ).strip()

    with _TROCR_LOCK:
        if model_name not in _TROCR_PIPELINES:
            processor = TrOCRProcessor.from_pretrained(model_name)
            model     = VisionEncoderDecoderModel.from_pretrained(model_name)
            device    = "cuda" if _use_gpu(config) else "cpu"
            model     = model.to(device)
            _TROCR_PIPELINES[model_name] = (processor, model, device)
        processor, model, device = _TROCR_PIPELINES[model_name]

    # Convert BGR → PIL RGB
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    lines = [t.strip() for t in text.splitlines() if t.strip()]
    return lines, {"engine": "trocr", "count": len(lines), "model": model_name}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_video_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("video") or {}


def _use_gpu(config: Dict[str, Any]) -> bool:
    import torch
    device = str((config.get("trocr") or {}).get("device", "auto")).strip().lower()
    if device == "cuda":
        return True
    if device == "cpu":
        return False
    return torch.cuda.is_available()


def _clarity_suggestion(label: str) -> str:
    if label == "clear":
        return "Good contrast and stroke consistency — readable from the back of the room."
    if label == "fair":
        return "Consider stronger contrast or slower writing; avoid very thin strokes."
    return "Use a thicker marker, increase contrast, and enlarge key terms."