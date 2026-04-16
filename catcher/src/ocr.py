"""
ocr.py — Shared OCR utilities for Catcher

Responsibilities:
- expose a single run_ocr_on_frame(...) entry point
- wrap EasyOCR / PaddleOCR / TrOCR
- normalise and deduplicate board lines
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

_EASYOCR_READERS: Dict[tuple, Any] = {}
_PADDLE_INSTANCES: Dict[tuple, Any] = {}
_TROCR_MODELS: Dict[str, Any] = {}

_EASY_LOCK = threading.Lock()
_PADDLE_LOCK = threading.Lock()
_TROCR_LOCK = threading.Lock()


def run_ocr_on_frame(frame_bgr: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run OCR on one frame or ROI image.

    Returns:
    {
        "texts": [...],
        "roi": {...} | None,
        "engine": "easyocr" | "paddleocr" | "trocr",
        "error": None | str,
    }
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "texts": [],
            "roi": None,
            "engine": None,
            "error": "Empty frame passed to OCR.",
        }

    trocr_cfg = config.get("trocr") or {}
    engine = str(trocr_cfg.get("ocr_engine", "easyocr")).strip().lower()

    try:
        if engine == "easyocr":
            texts = _ocr_easyocr(frame_bgr, config)
        elif engine == "paddleocr":
            texts = _ocr_paddleocr(frame_bgr, config)
        elif engine == "trocr":
            texts = _ocr_trocr(frame_bgr, config)
        else:
            texts = _ocr_easyocr(frame_bgr, config)
            engine = "easyocr"

        texts = normalise_board_lines(texts)
        return {
            "texts": texts,
            "roi": None,
            "engine": engine,
            "error": None,
        }

    except Exception as e:
        return {
            "texts": [],
            "roi": None,
            "engine": engine,
            "error": str(e),
        }


def normalise_board_lines(lines: List[str], min_substring_len: int = 8) -> List[str]:
    """
    Clean OCR output into readable unique board lines.

    Steps:
    - strip whitespace
    - collapse repeated spaces
    - remove obvious garbage-only lines
    - deduplicate exact matches
    - drop lines fully contained inside longer lines
    """
    cleaned: List[str] = []

    for line in lines:
        s = str(line).strip()
        if not s:
            continue

        s = re.sub(r"\s+", " ", s)
        s = s.replace("•", "-").replace("·", "-")

        if _is_noise_line(s):
            continue

        cleaned.append(s)

    cleaned = _dedupe_exact(cleaned)
    cleaned = _drop_subsumed(cleaned, min_len=min_substring_len)
    return cleaned


def _ocr_easyocr(frame_bgr: np.ndarray, config: Dict[str, Any]) -> List[str]:
    import easyocr

    trocr_cfg = config.get("trocr") or {}
    langs_raw = trocr_cfg.get("easyocr_languages") or ["en"]
    langs = [str(x).strip() for x in langs_raw if str(x).strip()]
    if not langs:
        langs = ["en"]

    gpu = _use_gpu(config)
    key = (tuple(langs), gpu)

    with _EASY_LOCK:
        if key not in _EASYOCR_READERS:
            _EASYOCR_READERS[key] = easyocr.Reader(langs, gpu=gpu)
        reader = _EASYOCR_READERS[key]

    results = reader.readtext(frame_bgr)
    texts: List[str] = []

    for item in results:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        conf = float(item[2]) if len(item) > 2 else 1.0
        if conf >= 0.15 and text:
            texts.append(text)

    return texts


def _ocr_paddleocr(frame_bgr: np.ndarray, config: Dict[str, Any]) -> List[str]:
    from paddleocr import PaddleOCR

    trocr_cfg = config.get("trocr") or {}
    lang = str(trocr_cfg.get("paddleocr_lang", "en")).strip() or "en"
    gpu = _use_gpu(config)
    key = (lang, gpu)

    with _PADDLE_LOCK:
        if key not in _PADDLE_INSTANCES:
            _PADDLE_INSTANCES[key] = PaddleOCR(
                lang=lang,
                device="gpu:0" if gpu else "cpu",
                use_text_line_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        ocr = _PADDLE_INSTANCES[key]

    result = ocr.predict(frame_bgr)
    texts: List[str] = []

    for item in (result or []):
        if not item or len(item) < 2:
            continue
        tc = item[1]
        if not isinstance(tc, (list, tuple)) or len(tc) < 1:
            continue
        text = str(tc[0]).strip()
        conf = float(tc[1]) if len(tc) > 1 else 1.0
        if conf >= 0.15 and text:
            texts.append(text)

    return texts


def _ocr_trocr(frame_bgr: np.ndarray, config: Dict[str, Any]) -> List[str]:
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    trocr_cfg = config.get("trocr") or {}
    model_name = str(
        trocr_cfg.get("default_model", "microsoft/trocr-base-handwritten")
    ).strip()

    with _TROCR_LOCK:
        if model_name not in _TROCR_MODELS:
            processor = TrOCRProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            device = "cuda" if _use_gpu(config) else "cpu"
            model = model.to(device)
            model.eval()
            _TROCR_MODELS[model_name] = (processor, model, device)
        processor, model, device = _TROCR_MODELS[model_name]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=128)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if not text:
        return []

    parts = [x.strip() for x in text.splitlines() if x.strip()]
    return parts if parts else [text]


def _use_gpu(config: Dict[str, Any]) -> bool:
    import torch

    trocr_cfg = config.get("trocr") or {}
    device_pref = str(trocr_cfg.get("device", "auto")).strip().lower()

    if device_pref == "cpu":
        return False
    if device_pref == "cuda":
        return True
    return torch.cuda.is_available()


def _dedupe_exact(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _drop_subsumed(lines: List[str], min_len: int = 8) -> List[str]:
    """
    Drop a line if it is fully contained inside another longer line.
    """
    kept: List[str] = []
    lowered = [(line, line.casefold()) for line in lines]

    for i, (line_i, low_i) in enumerate(lowered):
        if len(low_i) < min_len:
            kept.append(line_i)
            continue

        is_subsumed = False
        for j, (_, low_j) in enumerate(lowered):
            if i == j:
                continue
            if len(low_j) <= len(low_i):
                continue
            if low_i in low_j:
                is_subsumed = True
                break

        if not is_subsumed:
            kept.append(line_i)

    return kept


def _is_noise_line(text: str) -> bool:
    """
    Conservative noise filter:
    remove lines that are almost entirely symbols or too short to matter.
    """
    s = text.strip()
    if not s:
        return True

    if len(s) == 1 and not s.isalnum():
        return True

    alnum = sum(ch.isalnum() for ch in s)
    alpha = sum(ch.isalpha() for ch in s)
    digit = sum(ch.isdigit() for ch in s)

    if alnum == 0:
        return True

    if len(s) < 2:
        return True

    # Likely OCR garbage: very long symbols, almost no letters/digits
    if len(s) >= 4 and alnum / max(len(s), 1) < 0.25:
        return True

    # Drop isolated numeric junk like "7" or "12" unless embedded in richer text
    if digit > 0 and alpha == 0 and len(s) <= 2:
        return True

    return False