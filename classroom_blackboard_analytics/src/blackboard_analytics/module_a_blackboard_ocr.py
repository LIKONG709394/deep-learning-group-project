# Read handwriting from a classroom photo.
# Rough flow: locate the board (trained detector if you have weights, else "biggest ink blob",
# else the whole photo), split into horizontal text bands, run TrOCR on each band.

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np

from blackboard_analytics.model_cache import (
    ensure_project_model_cache_dirs,
    has_hf_repo_cache,
)
from blackboard_analytics.module_a_alt_ocr import (
    normalize_ocr_engine_name,
    recognize_lines_easyocr,
    recognize_lines_paddleocr,
    resolve_ocr_use_gpu,
)

logger = logging.getLogger(__name__)
ensure_project_model_cache_dirs()

TROCR_DEFAULT = "microsoft/trocr-base-handwritten"
TROCR_PRINTED = "microsoft/trocr-base-printed"

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


_TROCR_BUNDLES: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
_TROCR_LOCK = threading.Lock()

_YOLO_WORLD_MODELS: dict[tuple[str, tuple[str, ...]], Any] = {}
_YOLO_WORLD_LOCK = threading.Lock()


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


def coerce_roi_box(
    roi: Union["ROIBox", Tuple[int, int, int, int], List[int]],
    *,
    image_shape: Optional[Tuple[int, ...]] = None,
) -> "ROIBox":
    if isinstance(roi, ROIBox):
        box = roi
    else:
        if len(roi) != 4:
            raise ValueError("ROI tuple/list must contain four integers")
        box = ROIBox(*map(int, roi))
    if image_shape is None:
        return box
    h, w = image_shape[:2]
    return box.clip(w, h)


def preprocess_image(
    image_bgr: np.ndarray,
    *,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    binary_block_size: int = 31,
    binary_c: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    # CLAHE gray + adaptive INV (fg white) for line masks
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


def parse_trocr_device_option(raw: Any) -> Optional[str]:
    """None = auto (CUDA then CPU on failure). 'cpu' / 'cuda' = fixed."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s == "auto":
        return None
    if s in ("cpu", "torch.cpu"):
        return "cpu"
    if s in ("cuda", "gpu", "torch.cuda"):
        return "cuda"
    logger.warning("Unknown trocr.device %r; using auto", raw)
    return None


def _normalize_yolo_world_prompts(raw: Any, *, english_only: bool) -> List[str]:
    """Build YOLO-World class strings; optional ASCII-only (English/Latin labels)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        return []
    out: List[str] = []
    for p in items:
        s = str(p).strip()
        if not s:
            continue
        if english_only and not s.isascii():
            logger.warning("Skipping non-ASCII YOLO-World prompt (english_only): %r", s[:80])
            continue
        out.append(s)
    return out


def _get_yolo_world_model(model_name: str, prompts: List[str]) -> Any:
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    if not prompts:
        raise ValueError("YOLO-World requires at least one text class")
    key = (str(model_name), tuple(prompts))
    with _YOLO_WORLD_LOCK:
        cached = _YOLO_WORLD_MODELS.get(key)
        if cached is not None:
            return cached
        model = YOLO(str(model_name))
        model.set_classes(prompts)
        _YOLO_WORLD_MODELS[key] = model
        return model


def _largest_yolo_world_box(
    image_bgr: np.ndarray,
    model_name: str,
    prompts: List[str],
    conf: float,
    iou: float,
    frame_w: int,
    frame_h: int,
) -> Optional[ROIBox]:
    """Open-vocabulary detections: keep the largest box across all English text classes."""
    if YOLO is None or not prompts:
        return None
    try:
        model = _get_yolo_world_model(model_name, prompts)
    except Exception as e:
        logger.warning("YOLO-World load failed: %s", e)
        return None
    try:
        results = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        logger.warning("YOLO-World predict failed: %s", e)
        return None
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    best: Optional[ROIBox] = None
    best_area = 0
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[idx].cpu().numpy().ravel())
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = ROIBox(x1, y1, x2, y2).clip(frame_w, frame_h)
    return best if best_area > 0 else None


def _largest_yolo_box_for_class(
    image_bgr: np.ndarray,
    weights_file: Union[str, Path],
    conf: float,
    iou: float,
    wanted_class_id: int,
    frame_w: int,
    frame_h: int,
) -> Optional[ROIBox]:
    """Pick the biggest detection whose class id matches the blackboard class."""
    if YOLO is None:
        return None
    model = YOLO(str(weights_file))
    results = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    biggest: Optional[ROIBox] = None
    biggest_area = 0
    for idx in range(len(boxes)):
        cls_id = int(boxes.cls[idx].item()) if boxes.cls is not None else 0
        if cls_id != wanted_class_id:
            continue
        x1, y1, x2, y2 = map(int, boxes.xyxy[idx].cpu().numpy().ravel())
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > biggest_area:
            biggest_area = area
            biggest = ROIBox(x1, y1, x2, y2).clip(frame_w, frame_h)
    return biggest if biggest_area > 0 else None


def detect_blackboard_roi(
    image_bgr: np.ndarray,
    *,
    yolo_weights_path: Optional[Union[str, Path]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    blackboard_class_id: int = 0,
    yolo_world: Optional[dict] = None,
) -> Tuple[ROIBox, str]:
    frame_h, frame_w = image_bgr.shape[:2]

    yw = yolo_world if isinstance(yolo_world, dict) else {}
    if yw.get("enabled"):
        prompts = _normalize_yolo_world_prompts(
            yw.get("text_classes") or yw.get("classes"),
            english_only=bool(yw.get("english_only_prompts", True)),
        )
        model_name = str(yw.get("model", "yolov8s-worldv2.pt"))
        w_conf = float(yw.get("conf", conf))
        w_iou = float(yw.get("iou", iou))
        if prompts:
            try:
                world_roi = _largest_yolo_world_box(
                    image_bgr,
                    model_name,
                    prompts,
                    w_conf,
                    w_iou,
                    frame_w,
                    frame_h,
                )
                if world_roi is not None:
                    return world_roi, "yolo_world"
            except Exception as e:
                logger.warning("YOLO-World ROI failed, falling back: %s", e)
        else:
            logger.warning("YOLO-World enabled but no valid English text_classes; falling back.")

    weights_ok = yolo_weights_path and Path(yolo_weights_path).is_file()
    if weights_ok:
        if YOLO is None:
            logger.warning("ultralytics not installed; skipping YOLO.")
        else:
            try:
                yolo_roi = _largest_yolo_box_for_class(
                    image_bgr,
                    Path(yolo_weights_path),
                    conf,
                    iou,
                    blackboard_class_id,
                    frame_w,
                    frame_h,
                )
                if yolo_roi is not None:
                    return yolo_roi, "yolo"
            except Exception as e:
                logger.warning("YOLO inference failed, using heuristic: %s", e)

    try:
        _, ink_mask = preprocess_image(image_bgr)
        contour_roi = _largest_contour_roi(ink_mask)
        if contour_roi is not None:
            return contour_roi, "heuristic"
    except Exception as e:
        logger.warning("Heuristic ROI failed: %s", e)

    whole_shot = ROIBox(0, 0, frame_w, frame_h)
    return whole_shot, "full_frame"


def crop_roi(image: np.ndarray, roi: ROIBox) -> np.ndarray:
    x1, y1, x2, y2 = roi.as_tuple()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI")
    return image[y1:y2, x1:x2].copy()


def prepare_ocr_inputs(
    image_bgr: np.ndarray,
    roi: Union[ROIBox, Tuple[int, int, int, int], List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    roi_box = coerce_roi_box(roi, image_shape=image_bgr.shape)
    board_crop = crop_roi(image_bgr, roi_box)
    gray_enhanced, ink_for_lines = preprocess_image(board_crop)
    return board_crop, gray_enhanced, ink_for_lines


def segment_text_lines(
    binary_inv_roi: np.ndarray,
    *,
    min_line_height: int = 8,
    min_gap: int = 5,
    pad_y: int = 4,
) -> List[Tuple[int, int]]:
    # row sums → line bands (y0,y1) in ROI coords
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
    # Models load on first use so importing this file stays light.

    def __init__(self, model_name: str = TROCR_DEFAULT, device: Optional[str] = None) -> None:
        self.model_name = model_name
        # device: None = auto (CUDA if possible, else CPU; OOM on CUDA -> CPU)
        self._device_pref = device
        self._processor = None
        self._model = None
        self._device = None

    def _try_order(self) -> List[str]:
        pref = self._device_pref
        if pref == "cpu":
            return ["cpu"]
        if pref == "cuda":
            if torch is not None and torch.cuda.is_available():
                return ["cuda"]
            raise RuntimeError("trocr.device=cuda but CUDA is not available")
        if torch is not None and torch.cuda.is_available():
            return ["cuda", "cpu"]
        return ["cpu"]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if TrOCRProcessor is None or VisionEncoderDecoderModel is None or torch is None:
            raise RuntimeError("Install torch and transformers")

        order = self._try_order()
        last_err: Optional[BaseException] = None
        for dev_name in order:
            with _TROCR_LOCK:
                cached = _TROCR_BUNDLES.get((self.model_name, dev_name))
            if cached is not None:
                self._processor, self._model, self._device = cached
                return
            try:
                local_only = has_hf_repo_cache(self.model_name)
                processor = TrOCRProcessor.from_pretrained(
                    self.model_name,
                    local_files_only=local_only,
                )
                model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_name,
                    local_files_only=local_only,
                )
                device_obj = torch.device(dev_name)
                model.to(device_obj)
                model.eval()
                with _TROCR_LOCK:
                    _TROCR_BUNDLES[(self.model_name, dev_name)] = (processor, model, device_obj)
                self._processor = processor
                self._model = model
                self._device = device_obj
                if dev_name == "cpu" and "cuda" in order:
                    logger.info("TrOCR %s running on CPU (GPU unavailable or OOM during load).", self.model_name)
                return
            except Exception as e:
                last_err = e
                if dev_name == "cuda" and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if self._device_pref is None and dev_name == "cuda":
                    logger.warning(
                        "TrOCR CUDA load failed for %s (%s); trying CPU.",
                        self.model_name,
                        e,
                    )
                    continue
                if self._device_pref == "cuda":
                    raise RuntimeError(f"Failed to load TrOCR: {e}") from e
                raise RuntimeError(f"Failed to load TrOCR: {e}") from e

        raise RuntimeError(f"Failed to load TrOCR: {last_err}") from last_err

    def decode_line(self, gray_line: np.ndarray) -> str:
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None and self._device is not None
        pil = _pil_line_from_gray(gray_line)
        pixel_values = self._processor(images=pil, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            # Avoid transformers warning about default max_length=21; allow longer board lines.
            gen_ids = self._model.generate(pixel_values, max_new_tokens=128)
        text = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return (text or "").strip()


def recognize_blackboard_handwriting(
    image_bgr: np.ndarray,
    *,
    yolo_weights_path: Optional[Union[str, Path]] = None,
    yolo_world: Optional[dict] = None,
    trocr_model_name: str = TROCR_DEFAULT,
    trocr_device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    blackboard_class_id: int = 0,
    engine: Optional[TrOCRHandwritingEngine] = None,
    board_region: Optional[Union[ROIBox, Tuple[int, int, int, int], List[int]]] = None,
    ocr_engine: str = "trocr",
    easyocr_languages: Optional[List[str]] = None,
    paddleocr_lang: str = "en",
) -> List[str]:
    recognized_lines: List[str] = []
    try:
        if board_region is None:
            board_region, how_found = detect_blackboard_roi(
                image_bgr,
                yolo_weights_path=yolo_weights_path,
                conf=conf,
                iou=iou,
                blackboard_class_id=blackboard_class_id,
                yolo_world=yolo_world,
            )
        else:
            board_region = coerce_roi_box(board_region, image_shape=image_bgr.shape)
            how_found = "provided"
        logger.info("Blackboard ROI method: %s", how_found)

        assert board_region is not None
        board_crop = crop_roi(image_bgr, board_region)
        recognized_lines = recognize_text_lines_in_image(
            board_crop,
            trocr_model_name=trocr_model_name,
            trocr_device=trocr_device,
            engine=engine,
            ocr_engine=ocr_engine,
            easyocr_languages=easyocr_languages,
            paddleocr_lang=paddleocr_lang,
        )
    except Exception as e:
        logger.error("Blackboard OCR pipeline failed: %s", e)
        raise
    return recognized_lines


def recognize_text_lines_in_image(
    image_bgr: np.ndarray,
    *,
    trocr_model_name: str = TROCR_DEFAULT,
    trocr_device: Optional[str] = None,
    engine: Optional[TrOCRHandwritingEngine] = None,
    ocr_engine: str = "trocr",
    easyocr_languages: Optional[List[str]] = None,
    paddleocr_lang: str = "en",
) -> List[str]:
    backend = normalize_ocr_engine_name(ocr_engine)
    use_gpu = resolve_ocr_use_gpu(trocr_device)
    if backend == "easyocr":
        langs = easyocr_languages if easyocr_languages else ["en"]
        return recognize_lines_easyocr(image_bgr, languages=langs, use_gpu=use_gpu)
    if backend == "paddleocr":
        return recognize_lines_paddleocr(
            image_bgr,
            lang=str(paddleocr_lang or "en"),
            use_gpu=use_gpu,
        )

    recognized_lines: List[str] = []
    gray_enhanced, ink_for_lines = preprocess_image(image_bgr)
    line_spans = segment_text_lines(ink_for_lines)
    if not line_spans:
        line_spans = [(0, gray_enhanced.shape[0])]

    reader = engine or TrOCRHandwritingEngine(trocr_model_name, device=trocr_device)
    for row_top, row_bottom in line_spans:
        line_gray = gray_enhanced[row_top:row_bottom, :]
        if line_gray.size == 0:
            continue
        if (line_gray > 0).sum() < 50:
            continue
        try:
            line_text = reader.decode_line(line_gray)
            if line_text:
                recognized_lines.append(line_text)
        except Exception as e:
            logger.warning("Line OCR failed (skipped): %s", e)
    return recognized_lines


def run_module_a(
    image_bgr: np.ndarray,
    config: Optional[dict] = None,
    *,
    roi_override: Optional[Union[ROIBox, Tuple[int, int, int, int], List[int]]] = None,
    roi_method_override: Optional[str] = None,
    engine_override: Optional[TrOCRHandwritingEngine] = None,
) -> dict:
    cfg = config or {}
    yolo_opts = cfg.get("yolo", {})
    yolo_world_opts = cfg.get("yolo_world") if isinstance(cfg.get("yolo_world"), dict) else {}
    trocr_opts = cfg.get("trocr", {})

    weights = yolo_opts.get("weights_path")
    det_conf = float(yolo_opts.get("conf", 0.25))
    det_iou = float(yolo_opts.get("iou", 0.45))
    board_class = int(yolo_opts.get("blackboard_class_id", 0))
    handwriting_model = str(trocr_opts.get("model_name", TROCR_DEFAULT))
    trocr_dev = parse_trocr_device_option(trocr_opts.get("device", "auto"))
    ocr_engine = normalize_ocr_engine_name(trocr_opts.get("ocr_engine", "trocr"))
    raw_langs = trocr_opts.get("easyocr_languages")
    if isinstance(raw_langs, str):
        easy_langs: Optional[List[str]] = [raw_langs.strip()] if raw_langs.strip() else None
    elif isinstance(raw_langs, list):
        easy_langs = [str(x).strip() for x in raw_langs if str(x).strip()]
        if not easy_langs:
            easy_langs = None
    else:
        easy_langs = None
    paddle_lang = str(trocr_opts.get("paddleocr_lang", "en") or "en")

    out: dict = {"texts": [], "roi": None, "roi_method": None, "error": None}
    try:
        if roi_override is None:
            roi, method = detect_blackboard_roi(
                image_bgr,
                yolo_weights_path=weights,
                conf=det_conf,
                iou=det_iou,
                blackboard_class_id=board_class,
                yolo_world=yolo_world_opts,
            )
        else:
            roi = coerce_roi_box(roi_override, image_shape=image_bgr.shape)
            method = roi_method_override or "provided"
        out["roi"] = roi.as_tuple()
        out["roi_method"] = method
        trocr_engine = (
            engine_override if ocr_engine == "trocr" and engine_override is not None else None
        )
        out["texts"] = recognize_blackboard_handwriting(
            image_bgr,
            yolo_weights_path=weights,
            yolo_world=yolo_world_opts,
            trocr_model_name=handwriting_model,
            trocr_device=trocr_dev,
            conf=det_conf,
            iou=det_iou,
            blackboard_class_id=board_class,
            engine=trocr_engine,
            board_region=roi,
            ocr_engine=ocr_engine,
            easyocr_languages=easy_langs,
            paddleocr_lang=paddle_lang,
        )
    except Exception as e:
        out["error"] = str(e)
        logger.exception("run_module_a")
    return out
