from datetime import time
import json
from pathlib import Path
import threading
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

import cv2
import neattext.functions as nfx
import torch
from requests import request
from transformers import AutoModelForCausalLM, AutoTokenizer
import easyocr
from paddleocr import PaddleOCR

from capture2 import crop_roi

# Assuming these are imported from your custom modules
# from m.capture import ROIBox, crop_roi
# from m.preprocess import preprocess_image, segment_text_lines
# from m.trocr import TrOCRHandwritingEngine

# --- Configuration Constants ---
DODEDUPE = True  # Fixed missing '='
MATH_SYMBOLS: Set[str] = set("=+-*/×÷^()[]{}<>∫∑√πΔ·.,:;%")
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


# --- Utility Functions ---

def _get_device() -> str:
    """Returns the optimal device for PyTorch/Models."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dedupe_subsumed_lines(lines: List[str], *, min_len: int = 8) -> List[str]:
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


def get_preplexity_input(text: str, device: str) -> Tuple[Optional[Any], Optional[Dict]]:
    model, tokenizer = _get_perplexity_model(device)
    if not text.strip():
        return None, None
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return model, inputs


def _calculate_perplexity(text: str, device: str) -> float:
    print("_calculate_perplexity")
    model, inputs = get_preplexity_input(text, device)
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
    print("calculate_perplexity_")
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
        out_dict["textlines"] = textual_pause_dict[best_model]["textlines"]
        out_dict["ocr_rois"] = textual_pause_dict[best_model]["ocr_rois"]
        out_dict["paragraph"] = " ".join(out_dict["textlines"])  # Fixed empty string join
    else:
        out_dict["textlines"] = []
        out_dict["ocr_rois"] = []
        out_dict["paragraph"] = ""
        
    global idd
    print("save_annotated_image")
    save_annotated_image(out_dict["frame_bgr"], out_dict["ocr_rois"], str(idd)+".jpg")
    idd+=1

    return out_dict    


def match(visual_pause: dict, tokenized_segments: List[dict], config: dict) -> dict:
    """
    Cleans OCR output and ranks relevance based on alignment with transcribed speech segments.
    """
    texts = []
    blocks = []
    original_textlines = visual_pause.get("textlines", [])
    original_rois = visual_pause.get("ocr_rois", [])
    
    filtered = _dedupe_subsumed_lines(original_textlines)
    
    for i, text in enumerate(original_textlines):
        if text not in filtered: 
            continue
            
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