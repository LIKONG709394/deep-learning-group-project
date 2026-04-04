"""
Module A: blackboard ROI (YOLOv8) + handwritten OCR (microsoft/trocr-base-handwritten).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union  # Any for PIL return

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    torch = None  # type: ignore
    TrOCRProcessor = None  # type: ignore
    VisionEncoderDecoderModel = None  # type: ignore


@dataclass
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, w: int, h: int) -> "ROIBox":
        return ROIBox(
            max(0, self.x1),
            max(0, self.y1),
            min(w, self.x2),
            min(h, self.y2),
        )

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def preprocess_image(
    image_bgr: np.ndarray,
    *,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    binary_block_size: int = 31,
    binary_c: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV: grayscale -> CLAHE -> adaptive threshold.

    Returns:
        gray_enhanced: CLAHE grayscale (for YOLO / viz)
        binary_inv: binary (text white 255, background 0) for line segmentation
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")

    if len(image_bgr.shape) == 2:
        gray = image_bgr.copy()
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray_enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        gray_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(3, binary_block_size | 1),
        binary_c,
    )
    return gray_enhanced, binary


def _largest_contour_roi(
    binary_inv: np.ndarray,
    min_area_ratio: float = 0.05,
) -> Optional[ROIBox]:
    """Heuristic: largest contour bounding box as blackboard candidate (no YOLO weights)."""
    h, w = binary_inv.shape[:2]
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    areas = [(cv2.contourArea(c), c) for c in contours]
    areas.sort(key=lambda x: x[0], reverse=True)
    min_area = h * w * min_area_ratio
    for area, cnt in areas:
        if area < min_area:
            break
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < w * 0.2 or ch < h * 0.15:
            continue
        return ROIBox(x, y, x + cw, y + ch).clip(w, h)
    return None


def detect_blackboard_roi(
    image_bgr: np.ndarray,
    *,
    yolo_weights_path: Optional[Union[str, Path]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    blackboard_class_id: int = 0,
) -> Tuple[ROIBox, str]:
    """
    YOLOv8 blackboard detection; on failure fall back to contour heuristic or full frame.

    Returns:
        (roi, method) where method is 'yolo' | 'heuristic' | 'full_frame'
    """
    h, w = image_bgr.shape[:2]

    if yolo_weights_path and Path(yolo_weights_path).is_file():
        if YOLO is None:
            logger.warning("ultralytics not installed; skipping YOLO.")
        else:
            try:
                model = YOLO(str(yolo_weights_path))
                results = model.predict(
                    source=image_bgr,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )
                if results and results[0].boxes is not None and len(results[0].boxes):
                    boxes = results[0].boxes
                    best = None
                    best_area = 0
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else 0
                        if cls_id != blackboard_class_id:
                            continue
                        xyxy = boxes.xyxy[i].cpu().numpy().ravel()
                        x1, y1, x2, y2 = map(int, xyxy)
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        if area > best_area:
                            best_area = area
                            best = ROIBox(x1, y1, x2, y2).clip(w, h)
                    if best is not None and best_area > 0:
                        return best, "yolo"
            except Exception as e:
                logger.warning("YOLO inference failed, using heuristic: %s", e)

    try:
        _, binary_inv = preprocess_image(image_bgr)
        roi = _largest_contour_roi(binary_inv)
        if roi is not None:
            return roi, "heuristic"
    except Exception as e:
        logger.warning("Heuristic ROI failed: %s", e)

    return ROIBox(0, 0, w, h), "full_frame"


def crop_roi(image: np.ndarray, roi: ROIBox) -> np.ndarray:
    x1, y1, x2, y2 = roi.as_tuple()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI")
    return image[y1:y2, x1:x2].copy()


def segment_text_lines(
    binary_inv_roi: np.ndarray,
    *,
    min_line_height: int = 8,
    min_gap: int = 5,
    pad_y: int = 4,
) -> List[Tuple[int, int]]:
    """
    Horizontal projection line segmentation.
    Returns list of (y_start, y_end) with padding, relative to ROI.
    """
    h, w = binary_inv_roi.shape[:2]
    proj = (binary_inv_roi > 0).astype(np.float32).sum(axis=1)
    threshold = max(1.0, 0.02 * w)
    in_line = False
    start = 0
    lines: List[Tuple[int, int]] = []
    for y in range(h):
        if proj[y] >= threshold:
            if not in_line:
                start = y
                in_line = True
        else:
            if in_line and y - start >= min_line_height:
                y0 = max(0, start - pad_y)
                y1 = min(h, y + pad_y)
                lines.append((y0, y1))
            in_line = False
    if in_line and h - start >= min_line_height:
        y0 = max(0, start - pad_y)
        y1 = h
        lines.append((y0, y1))

    merged: List[Tuple[int, int]] = []
    for y0, y1 in lines:
        if not merged:
            merged.append((y0, y1))
            continue
        py0, py1 = merged[-1]
        if y0 - py1 < min_gap:
            merged[-1] = (py0, y1)
        else:
            merged.append((y0, y1))
    return merged


def _pil_line_from_gray(gray_line: np.ndarray, target_h: int = 384) -> Any:
    from PIL import Image

    g = gray_line
    if g.size == 0:
        raise ValueError("Empty line image")
    h, w = g.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Invalid line dimensions")
    scale = target_h / float(h)
    new_w = max(1, int(w * scale))
    resized = cv2.resize(g, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(resized).convert("RGB")


class TrOCRHandwritingEngine:
    """Lazy-load TrOCR to avoid GPU/RAM use on import."""

    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten") -> None:
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None or torch is None:
            raise RuntimeError("Install torch and transformers")
        try:
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            self._processor = None
            self._model = None
            raise RuntimeError(f"Failed to load TrOCR: {e}") from e

    def decode_line(self, gray_line: np.ndarray) -> str:
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None and self._device is not None
        pil = _pil_line_from_gray(gray_line)
        pixel_values = self._processor(images=pil, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            gen_ids = self._model.generate(pixel_values)
        text = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return (text or "").strip()


def recognize_blackboard_handwriting(
    image_bgr: np.ndarray,
    *,
    yolo_weights_path: Optional[Union[str, Path]] = None,
    trocr_model_name: str = "microsoft/trocr-base-handwritten",
    conf: float = 0.25,
    iou: float = 0.45,
    blackboard_class_id: int = 0,
    engine: Optional[TrOCRHandwritingEngine] = None,
) -> List[str]:
    """
    Full pipeline: preprocess -> ROI -> line split -> TrOCR per line.

    Returns:
        List of recognized strings (one per line, stripped; may be empty on failure)
    """
    texts: List[str] = []
    try:
        roi, method = detect_blackboard_roi(
            image_bgr,
            yolo_weights_path=yolo_weights_path,
            conf=conf,
            iou=iou,
            blackboard_class_id=blackboard_class_id,
        )
        logger.info("Blackboard ROI method: %s", method)
        roi_img = crop_roi(image_bgr, roi)
        gray_enhanced, binary_inv = preprocess_image(roi_img)
        line_spans = segment_text_lines(binary_inv)
        if not line_spans:
            line_spans = [(0, gray_enhanced.shape[0])]

        ocr_engine = engine or TrOCRHandwritingEngine(trocr_model_name)
        for y0, y1 in line_spans:
            line_gray = gray_enhanced[y0:y1, :]
            if line_gray.size == 0:
                continue
            if (line_gray > 0).sum() < 50:
                continue
            try:
                t = ocr_engine.decode_line(line_gray)
                if t:
                    texts.append(t)
            except Exception as e:
                logger.warning("Line OCR failed (skipped): %s", e)
    except Exception as e:
        logger.error("Blackboard OCR pipeline failed: %s", e)
        raise
    return texts


def run_module_a(
    image_bgr: np.ndarray,
    config: Optional[dict] = None,
) -> dict:
    """
    Module A output for pipeline wiring.

    Input:
        image_bgr: BGR uint8 numpy
    Output:
        dict: texts, roi, roi_method, error (optional)
    """
    cfg = config or {}
    ycfg = cfg.get("yolo", {})
    tcfg = cfg.get("trocr", {})
    out: dict = {"texts": [], "roi": None, "roi_method": None, "error": None}
    try:
        roi, method = detect_blackboard_roi(
            image_bgr,
            yolo_weights_path=ycfg.get("weights_path"),
            conf=float(ycfg.get("conf", 0.25)),
            iou=float(ycfg.get("iou", 0.45)),
            blackboard_class_id=int(ycfg.get("blackboard_class_id", 0)),
        )
        out["roi"] = roi.as_tuple()
        out["roi_method"] = method
        out["texts"] = recognize_blackboard_handwriting(
            image_bgr,
            yolo_weights_path=ycfg.get("weights_path"),
            trocr_model_name=str(tcfg.get("model_name", "microsoft/trocr-base-handwritten")),
            conf=float(ycfg.get("conf", 0.25)),
            iou=float(ycfg.get("iou", 0.45)),
            blackboard_class_id=int(ycfg.get("blackboard_class_id", 0)),
        )
    except Exception as e:
        out["error"] = str(e)
        logger.exception("run_module_a")
    return out
