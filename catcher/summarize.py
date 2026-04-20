"""
Multi-Engine Text Summarization & OCR
=====================================

This module serves as the intelligence layer for visual text extraction within 
the Teaching Analysis System. It combines multiple OCR engines (EasyOCR, 
PaddleOCR, and TrOCR) to accurately transcribe board content, while applying 
NLP-based filtering to refine and deduplicate the extracted information.

Key Features:
    * Hybrid OCR Strategy: Dynamically utilizes TrOCR for handwriting and 
      PaddleOCR/EasyOCR for printed text or mathematical symbols.
    * Intelligence Filtering: Uses perplexity scoring (GPT-2) to prune 
      OCR noise and low-confidence transcriptions.
    * Geometric Text Analysis: Implements line segmentation and height-based 
      filtering to preserve the spatial hierarchy of board content.
    * LLM Integration: Interfaces with DeepSeek for semantic synthesis and 
      relevance classification of teaching materials.
    * Thread-Safe Model Loading: Manages high-memory-demand models (Transformers, 
      Paddle) via global caching and resource locks.

Dependencies:
    * transformers: For TrOCR (Vision-Encoder-Decoder) and perplexity models.
    * paddleocr & easyocr: For robust multi-engine text recognition.
    * neattext: For cleaning and normalizing extracted instructional text.
    * torch: For managing model weights and GPU-accelerated inference.

Author: [Lai Tsz Yeung/Group J]
Date: 2026
License: MIT
"""
import os
from datetime import time
import json
from pathlib import Path
import threading
import logging
from typing import List, Dict, Any, Tuple, Optional, Set, Union

import cv2
import neattext.functions as nfx
import numpy as np
import paddle
import torch
from requests import request
from transformers import AutoModelForCausalLM, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
from paddleocr import PaddleOCR

from capture import ROIBox, coerce_roi_box, crop_roi, preprocess_image

# Assuming these are imported from your custom modules
# from m.capture import ROIBox, crop_roi
# from m.preprocess import preprocess_image, segment_text_lines
# from m.trocr import TrOCRHandwritingEngine

# --- Configuration Constants ---
DODEDUPE = True  # Fixed missing '='
MATH_SYMBOLS: Set[str] = set("=+-*/??÷^()[]{}<>?????????????·.,:;%")
FILTER_NOISITY_TEXT = True

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"

ALLOWED_HEIGHT_TWEAK = 0.55
PERPLEX_SCORE_MAX = 800
ALLOWED_RELEVANCE = {"high", "medium", "low"}  # Added missing constant

# --- Global Locks and Caches ---
_PERPLEXITY_LOCK = threading.Lock()
_PERPLEXITY_CACHE = {}

_EASY_READERS = {}
_EASY_LOCK = threading.Lock()

_PADDLE_INSTANCES = {}
_PADDLE_LOCK = threading.Lock()

_TROCR_INSTANCES = {}
_TROCR_LOCK = threading.Lock()

TROCR_DEFAULT = "microsoft/trocr-base-handwritten"
TROCR_PRINTED = "microsoft/trocr-base-printed"

# --- Utility Functions ---

def _get_device() -> str:
    if paddle.device.is_compiled_with_cuda() and paddle.device.get_device().startswith("gpu"):
        return "gpu"
    return "cpu"

def _dedupe_subsumed_lines(lines: List[str], *, min_len: int = 1) -> List[str]:
    items = [(s.strip(), s.strip().casefold()) for s in lines if s and s.strip()]
    if len(items) < 2:
        return [text for text, _ in items]
    
    drop: set[int] = set()
    for i, (_, item_cf) in enumerate(items):
        if len(item_cf) < min_len:
            continue
        for j, (_, other_cf) in enumerate(items):
            if i == j or len(other_cf) <= len(item_cf):
                continue
            if item_cf in other_cf:
                drop.add(i)
                break
                
    return [items[i][0] for i in range(len(items)) if i not in drop]


def _get_perplexity_model(device: str):
    key = "distilgpt2"
    with _PERPLEXITY_LOCK:
        cached = _PERPLEXITY_CACHE.get(key)
        if cached is not None:
            return cached
            
        tokenizer = AutoTokenizer.from_pretrained(key)
        model = AutoModelForCausalLM.from_pretrained(key).to(device)
        model.eval()

        _PERPLEXITY_CACHE[key] = (model, tokenizer)
        return model, tokenizer


def get_perplexity_input(text: str, device: str) -> Tuple[Optional[Any], Optional[Dict]]:
    model, tokenizer = _get_perplexity_model(device)
    if not text.strip():
        return None, None
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return model, inputs


def _calculate_perplexity(text: str, device: str) -> float:
    model, inputs = get_perplexity_input(text, device)
    if model is None or inputs is None:
        return float("inf")
        
    with torch.no_grad():
        # DO NOT pass labels here; this bypasses the buggy ForCausalLMLoss in transformers
        outputs = model(**inputs)
    
    # 1. Get the raw prediction scores (logits)
    logits = outputs.logits
    
    # 2. Shift the logits and labels so that token < n predicts token n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    
    # 3. Manually calculate the cross entropy loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return float(torch.exp(loss).item())


def _meaningful(lines: List[str]) -> bool:
    joined = " ".join((line or "").strip() for line in lines).strip()
    if not joined:
        return True
        
    alpha_count = sum(ch.isalpha() for ch in joined)
    digit_count = sum(ch.isdigit() for ch in joined)
    useful_count = sum(ch.isalnum() for ch in joined)
    
    if alpha_count >= 8:
        return True
    if useful_count == 0:
        return False
    return digit_count < alpha_count and useful_count > 12


def _fallback_relevance_from_score(score: float) -> str:
    """Helper for _normalize_analysis to guess relevance based on score."""
    if score > 75: return "high"
    if score > 40: return "medium"
    return "low"


def _normalize_analysis(raw: dict, enabled: bool, model: str) -> dict:
    overall = str(raw.get("overall_relevance") or "").strip().lower()
    score_raw = raw.get("score")
    
    try:
        score = max(0.0, min(100.0, float(score_raw)))
    except (TypeError, ValueError):
        score = 0.0

    if overall not in ALLOWED_RELEVANCE:
        overall = _fallback_relevance_from_score(score)

    reason = str(raw.get("reason") or "").strip() or "No reason returned by DeepSeek."
    evidence_raw = raw.get("evidence")
    evidence = []
    
    if isinstance(evidence_raw, list):
        evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]
    elif isinstance(evidence_raw, str) and evidence_raw.strip():
        evidence = [evidence_raw.strip()]
        
    if len(evidence) > 5:
        evidence = evidence[:5]
    if not evidence:
        evidence = [reason]

    return {
        "enabled": enabled,
        "model": model,
        "overall_relevance": overall,
        "score": round(score, 2),
        "reason": reason,
        "evidence": evidence,
        "error": None,
    }


def _normalize_ocr_engine_name(raw: str) -> str:
    s = str(raw or "trocr").strip().lower()
    if s in ("trocr", "transformers", "hf", "huggingface"):
        return "trocr"
    if s in ("easyocr", "easy"):
        return "easyocr"
    if s in ("paddleocr", "paddle"):
        return "paddleocr"
    return "trocr"


def _getEasyLangs(raw_langs) -> Optional[List[str]]:
    if isinstance(raw_langs, str):
        return [raw_langs.strip()] if raw_langs.strip() else None
    if isinstance(raw_langs, list):
        easy_langs = [str(x).strip() for x in raw_langs if str(x).strip()]
        return None if not easy_langs else easy_langs
    return None


# --- OCR Implementations ---

def recognize_lines_other_ocr(frame_bgr, engine, trocr_model_name, device):
    """Requires preprocess_image and segment_text_lines to be defined in scope."""
    recognized_lines = []
    gray_enhanced, ink_for_lines = preprocess_image(frame_bgr)
    line_spans = segment_text_lines(ink_for_lines)
    
    if not line_spans:
        line_spans = [(0, gray_enhanced.shape[0])]

    reader = engine or TrOCRHandwritingEngine(trocr_model_name, device)
    for row_top, row_bottom in line_spans:
        line_gray = gray_enhanced[row_top:row_bottom, :]
        if line_gray.size == 0 or (line_gray > 0).sum() < 50:
            continue
            
        line_text = reader.decode_line(line_gray)
        if line_text:
            recognized_lines.append(line_text)
            
    return recognized_lines


def _recognize_lines_paddleocr(frame_bgr, languages, device: str):
    lang = str(languages or "en").strip() or "en"
    key = (lang, device == "cuda")
    
    with _PADDLE_LOCK:
        ocr = _PADDLE_INSTANCES.get(key)
        if ocr is None:
            ocr = PaddleOCR(
                lang=lang,
                device="gpu:0" if device == "cuda" else "cpu",                    
                use_textline_orientation=True,    
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            _PADDLE_INSTANCES[key] = ocr
    return ocr.predict(frame_bgr)


def _paddle_detections_to_entries(result):
    out = []
    if not result or not result[0]:
        return out
        
    result = result[0]
    for item in result:
        if not item or len(item) < 2: 
            continue
            
        bbox, tc = item[0], item[1]
        if not isinstance(tc, (list, tuple)) or len(tc) < 1:
            continue    
            
        text = str(tc[0] or "").strip()
        conf = float(tc[1]) if len(tc) > 1 else 1.0
        
        if conf < 0.15 or not text: 
            continue
            
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        out.append((min(ys), max(ys), min(xs), max(xs), str(text).strip()))
        
    return out


def _recognize_lines_easyocr(frame_bgr, languages, device: str):
    langs = tuple(str(x).strip() for x in languages if str(x).strip()) or ("en",)
    key = (langs, device == "cuda")
    
    with _EASY_LOCK:
        reader = _EASY_READERS.get(key)
        if reader is None: 
            reader = easyocr.Reader(list(langs), gpu=(device == "cuda"))
            _EASY_READERS[key] = reader
            
    return reader.readtext(frame_bgr)


def _easyocr_detections_to_entities(result):
    out = []
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
        out.append((min(ys), max(ys), min(xs), max(xs), str(text).strip()))
        
    return out


# --- Bounding Box / Layout Logistics ---

def arrange_bboxes(bboxes, axis1, axis2):
    if axis1 == 0:
        return sorted(bboxes, key=lambda t: ((t[0] + t[1]) / 2.0, t[2]))
    return sorted(bboxes, key=lambda t: (t[2], (t[0] + t[1]) / 2.0))


def merge_lines(bboxes, allowed_height_tweak):    
    lines = []
    current = []
    for box in bboxes:
        ymin, ymax, xmin, xmax, text = box
        cy = (ymin + ymax) / 2.0
        if not current:
            current = [box]
            continue
            
        prev_cy = sum((a[0] + a[1]) / 2.0 for a in current) / len(current)
        if abs(cy - prev_cy) <= allowed_height_tweak:
            current.append(box)
        else:
            current.sort(key=lambda x: x[2])
            lines.append(current)
            current = [box]
            
    if current:
        current.sort(key=lambda x: x[2])
        lines.append(current)    
        
    textlines = [" ".join(t for _, _, _, _, t in block).strip() for block in lines if block]
    out_boxes = []
    
    for block in lines:
        max_x, max_y = 0, 0
        min_x, min_y = 10000, 10000
        ts = ""
        for y, my, x, mx, t in block:
            if y < min_y: min_y = y
            if x < min_x: min_x = x
            if my > max_y: max_y = my
            if mx > max_x: max_x = mx
            ts += t + " "
        out_boxes.append((min_y, max_y, min_x, max_x, ts.strip()))
        
    return textlines, out_boxes


# --- Main Pipeline Functions ---
idd = 0
def extract(visual_pause: dict, config: dict) -> dict:
    """
    Extracts text from visual pause frame regions using configured OCR engines.
    """
    device = _get_device()
    engines: List[str] = config.get("engines", ["easyocr", "paddleocr"] if device == "cpu" else ["paddleocr", "easyocr"])
    
    # Safely obtain timestamp 
    timestamp = visual_pause.get("timestamp", 0.0)
    areas = visual_pause.get("areas", [])
    
    textual_pause_dict = {}
    best_model = ""
    
    
    for engine in engines:
        textual_pause_dict[engine] = {"textlines": [], "ocr_rois": [], "score": 0, "useful_count": 0}
        abs_ocr_rois = []
        
        for area in areas:
            # Assuming `crop_roi` exists in scope
            try:
                cropped = crop_roi(visual_pause.get("frame_bgr", None), area["roi"])
            except Exception as e:
                logging.error(f"Failed to crop ROI: {e}")
                continue

            ocr_rois = []
            if engine == "easyocr":
                try:
                    easy_langs = config.get("easy_langs", ["en"])
                    ocr_result = _recognize_lines_easyocr(cropped, easy_langs, device)
                    ocr_rois = _easyocr_detections_to_entities(ocr_result)
                except Exception as e:
                    logging.warning(f"EasyOCR failed: {e}")
                    ocr_rois = []
                    
            elif engine == "paddleocr":
                try:
                    paddle_lang = config.get("paddle_lang", "en")
                    ocr_result = _recognize_lines_paddleocr(cropped, paddle_lang, device)
                    ocr_rois = _paddle_detections_to_entries(ocr_result)
                except Exception as e:
                    logging.warning(f"PaddleOCR failed: {e}")
                    ocr_rois = []
            
            # Fix: Extract correctly from a list of rois
            for roi in ocr_rois:
                yns, ymx, xns, xmxm, text = roi
                # Absolute offset translation
                abs_ocr_rois.append((yns + area["roi"].y1, ymx + area["roi"].y1, xns + area["roi"].x1, xmxm + area["roi"].x1, text))
                
        arranged_ocr_rois = arrange_bboxes(abs_ocr_rois, 0, 1) 
        textlines, merged_rois = merge_lines(arranged_ocr_rois, ALLOWED_HEIGHT_TWEAK)            
        
        textual_pause_dict[engine]["textlines"].extend(textlines)
        textual_pause_dict[engine]["ocr_rois"].extend(merged_rois)                        
        
        paragraph = " ".join(textual_pause_dict[engine]["textlines"]).strip()        
        textual_pause_dict[engine]["score"] = _calculate_perplexity(paragraph, device)
        textual_pause_dict[engine]["useful_count"] = sum(ch.isalnum() for ch in paragraph)            
        
        # Evaluate best model
        if best_model == "": 
            best_model = engine
        elif textual_pause_dict[engine]["score"] < textual_pause_dict[best_model]["score"]: 
            best_model = engine

        # Early exit thresholds
        if textual_pause_dict[engine]["score"] < PERPLEX_SCORE_MAX: 
            break
        if textual_pause_dict[engine]["useful_count"] > 6: 
            break

    out_dict = dict(visual_pause)
    if best_model:
        out_dict["altextlines"] = textual_pause_dict[best_model]["textlines"]
        out_dict["textlines"] = _dedupe_subsumed_lines(textual_pause_dict[best_model]["textlines"])
        out_dict["ocr_rois"] = textual_pause_dict[best_model]["ocr_rois"]
        out_dict["paragraph"] = " ".join(out_dict["textlines"])  # Fixed empty string join
    else:
        out_dict["altextlines"] = []
        out_dict["textlines"] = []
        out_dict["ocr_rois"] = []
        out_dict["paragraph"] = ""
    
        
    """
    global idd
    print("save_annotated_image")
    save_annotated_image(out_dict["frame_bgr"], out_dict["ocr_rois"], str(idd)+".jpg")
    idd+=1
    """
    visual_pause["frame_bgr"] = None

    return out_dict    


def match(visual_pause: dict, tokenized_segments: List[dict], config: dict) -> dict:
    """
    Cleans OCR output and ranks relevance based on alignment with transcribed speech segments.
    """
    texts = []
    blocks = []
    original_textlines = visual_pause.get("textlines", [])
    original_rois = visual_pause.get("ocr_rois", [])
    
    
    for i, text in enumerate(original_textlines):
            
        clean_val = text
        clean_val = nfx.remove_urls(clean_val)
        clean_val = nfx.remove_emails(clean_val)
        clean_val = nfx.remove_emojis(clean_val)
        clean_val = nfx.remove_numbers(clean_val)
        clean_val = nfx.remove_puncts(clean_val)
        clean_val = nfx.remove_multiple_spaces(clean_val)
        clean_val = clean_val.strip()
        texts.append(clean_val)
        # Fix: ensure index match
        if i < len(original_rois):
            blocks.append(original_rois[i])

    visual_pause["textlines"] = texts
    visual_pause["ocr_rois"] = blocks 

    paragraph = " ".join(visual_pause["textlines"]).strip()        
    
    # Extract tokens for matching (using standard tokenizer from perplexity model)
    device = _get_device()
    _, tokenizer = _get_perplexity_model(device)
    # Using tokenizer to generate a list of strings instead of PyTorch tensors for direct string matching
    tokens = tokenizer.tokenize(paragraph) if paragraph else []
    
    spoken_near = 0
    after = 0
    pause_time = visual_pause.get("timestamp", 0.0)
    
    for segment in tokenized_segments:
        if segment.get("start_sec", 0.0) > pause_time:
            after += 1
        if after > 10:
            break
            
        seg_tokens = segment.get("tokens", [])
        for token_p in tokens:
            if token_p in seg_tokens:
                spoken_near += 1
                break  # match found for this token
                
    # Normalize score
    spoken_near_score = (spoken_near / len(tokens)) if tokens else 0.0
    
    out = dict(visual_pause)
    out["tokens"] = tokens
    out["spoken_near"] = spoken_near_score
    return out
        
        
def correct(visual_pauses: List[dict], delete_indices: List[int], config: dict) -> List[dict]:
    """
    Clears out OCR textlines mapped to specified indices (e.g. after DeepSeek filter validation).
    """
    idx = 0
    for pause in visual_pauses:
        textlines = pause.get("textlines", [])
        for i, line in enumerate(textlines):
            if idx in delete_indices:
                pause["textlines"][i] = ""
            idx += 1
            
    return visual_pauses

def save_annotated_image(img_bgr, boxes, output_path):
    out = img_bgr.copy()
    for box in boxes:
        max_y, min_y, max_x, min_x, text = box
        x1, y1, x2, y2 = int(min_x), int(min_y), int(max_x), int(max_y)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, text, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, out)
    return output_path

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

def _prepare_ocr_inputs(
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
    # row sums ??? line bands (y0,y1) in ROI coords
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
                return
            except Exception as e:
                last_err = e
                if dev_name == "cuda" and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if self._device_pref is None and dev_name == "cuda":
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