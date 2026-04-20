"""
Visual Teaching Activity Capture
================================

This module handles the visual processing component of the Teaching Analysis 
System. It utilizes YOLO-based object detection to identify teaching surfaces 
(blackboards/whiteboards) and monitors frame-by-frame changes to detect 
meaningful pauses in writing or instruction.

Key Features:
    * Thread-safe YOLO model management and caching for ROI detection.
    * Adaptive change detection using Laplacian variance and frame subtraction.
    * ROI (Region of Interest) tracking and stabilization to handle camera jitter.
    * Automated "pause" detection to trigger OCR and visual content analysis.
    * Efficient image preprocessing and grayscale conversion for downstream OCR.

Dependencies:
    * ultralytics (YOLOv8): For real-time object detection and segmentation.
    * opencv-python (cv2): For video stream handling and image transformations.
    * torch: For GPU-accelerated model inference and tensor operations.
    * numpy: For numerical matrix operations and frame delta analysis.

Author: [Lai Tsz Yeung/Group J]
Date: 2026
License: MIT
"""
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import cuda
import torch.nn.functional as F  # <-- Fix is here
from ultralytics import YOLO

CLARFY_SIZE = (160, 90)
CHANGE_RATIO = 0.01
AREA_REDUCE_R = 0.1
_LAP_KERNEL = None

_YOLO_LOCK = threading.Lock()
_YOLO_MODELS = {}

def load_yolo_model(weight_path: Union[str, Path], device: str) -> YOLO:
    """
    Loads and caches a YOLO model in a thread-safe manner.

    Args:
        weight_path: Path to the .pt model weights.
        device: Device to load the model onto ('cuda' or 'cpu').

    Returns:
        YOLO: The cached Ultralytics YOLO model instance.
    """
    model_key = (str(weight_path), device)
    
    with _YOLO_LOCK:
        cached = _YOLO_MODELS.get(model_key)
        if cached is not None:
            return cached
        
        try:
            # Initialize model and move to the target device immediately
            model = YOLO(str(weight_path)).to(device)
            _YOLO_MODELS[model_key] = model
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {weight_path}: {e}")

def _get_textarea_yolo(frame_bgr, weight_path, conf, iou, device):    
    model = load_yolo_model(weight_path, device)
    results = model.predict(source=frame_bgr, conf=conf, iou=iou, verbose=False)
    return results    

def _get_boxes(results, frame_w, frame_h):
    res = []
    boxes = results[0].boxes
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[idx].cpu().numpy().ravel())
        res.append(ROIBox(x1, y1, x2, y2).clip(frame_w, frame_h))
    return res
    
def get_largest_box(boxes):
    best_area = 0
    best_box = None
    for box in boxes:
        x1, y1, x2, y2 = box.as_tuple()
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < best_area: continue
        best_area = area
        best_box = box
    return best_box

def _gettimestamp(capture, frame_index, fps):
    ts_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if ts_msec > 0: return ts_msec / 1000.0
    if fps > 0: return float(frame_index / fps)
    return 0.0

def _getfps(capture):
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0: fps = 25.0
    return fps

def detect_single_bgr_image(image_path: str, config, on_pause:Callable) -> np.ndarray:
    weights_path = config["weights_path"]
    iou = config["iou"]    
    conf = config["conf"]   
    device = _get_device()
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    
    yolo_prediction = _get_textarea_yolo(frame_bgr, weights_path, conf, iou, device)        
    frame_h, frame_w = frame_bgr.shape[:2]
    captured_text_areas = _get_boxes(yolo_prediction, frame_w, frame_h)
    timestamp = round(_gettimestamp(capture, frame_bgr, 1), 3)   
    visual_pause = {"timestamp":timestamp,"frame_bgr":frame_bgr,"areas":[]}
    for area in captured_text_areas:
        cropped = crop_roi(frame_bgr, area)
        clarity = _evaluate_handwriting_clarity(cropped, device) 
        visual_pause["areas"].append({"clarity":clarity,"roi":area})                
    on_pause(visual_pause)
    return frame_bgr

def opencv_extract_frame_dicts(video_path, stride=5):
    capture = cv2.VideoCapture(video_path)
    fps = _getfps(capture)
    frame_index = -1
    frame_dicts = []
    logging.info(f"start read video{video_path}")
    while True:        
        ok, frame = capture.read()
        if not ok: break
        else: frame_index += 1
        if stride!=0 and frame_index%stride !=0: continue
        frame_dicts.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(_gettimestamp(capture, frame_index, fps), 3),
                "frame_bgr": frame,
            }
        )
    capture.release()
    return frame_dicts

def get_abs_trait(frame_bgr, size):
    _, binary = preprocess_image(frame_bgr)
    resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
    return (resized > 32).astype(np.uint8) * 255

def get_rel_trait(previous_trait, current_trait):
    if previous_trait is None:
        return 1.0
    diff = cv2.absdiff(previous_trait, current_trait)
    return float(np.mean(diff) / 255.0)

def _get_lap_kernel(device):
    global _LAP_KERNEL
    if _LAP_KERNEL is None:
        _LAP_KERNEL = torch.tensor(
            [[0., 1., 0.],
             [1.,-4., 1.],
             [0., 1., 0.]], device=device
        ).unsqueeze(0).unsqueeze(0)   # (1,1,3,3)
    return _LAP_KERNEL

def _laplacian_variance(gray: np.ndarray, device) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    g = gray if len(gray.shape) == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    t = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0).to(_get_device())
    lap = F.conv2d(t, _get_lap_kernel(device), padding=1)
    return float(lap.var().item())

def _evaluate_handwriting_clarity(
    image: np.ndarray, device,
    laplacian_clear_min: float = 120.0,
    laplacian_messy_max: float = 40.0,
    stroke_variance_messy_min: float = 8.0,
):
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_metric = _laplacian_variance(gray, device)
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

def _binarize_for_strokes(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray) if len(gray.shape) == 2 else clahe.apply(cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    return bw

def _stroke_widths_per_component(binary_inv: np.ndarray):
    dist = cv2.distanceTransform((binary_inv > 0).astype(np.uint8), cv2.DIST_L2, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_inv > 0).astype(np.uint8))
    widths = []
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
):
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

def preprocess_image(
        frame_bgr, *,
        clahe_clip=2.0, clahe_grid=(8, 8),
        binary_block_size=31, binary_c=5,
        brightness_boost=1.0,
):
    if len(frame_bgr.shape) == 2: gray = frame_bgr.copy()
    else: gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    if brightness_boost != 1.0:
        gray = np.clip(gray.astype(np.float32) * brightness_boost, 0, 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray_enhanced = clahe.apply(gray)
    
    # Denoise after CLAHE only when boosting ??? boost amplifies compression noise
    if brightness_boost != 1.0:
        gray_enhanced = cv2.fastNlMeansDenoising(gray_enhanced, h=10)
    
    is_dark_bg = np.mean(gray) < 128
    thresh_type = cv2.THRESH_BINARY if is_dark_bg else cv2.THRESH_BINARY_INV
    
    binary = cv2.adaptiveThreshold(
        gray_enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        max(3, binary_block_size | 1),
        binary_c,
    )
    return gray_enhanced, binary

def _get_device():
    return "cuda" if cuda.is_available() else "cpu"

def _pack_pause(start_timestamp, last_frame, last_detected, device, on_pause):
    visual_pause = {"timestamp":start_timestamp,"frame_bgr":last_frame,"areas":[]}
    for area in last_detected:
        cropped = crop_roi(last_frame, area)
        clarity = _evaluate_handwriting_clarity(cropped, device) 
        visual_pause["areas"].append({"clarity":clarity,"roi":area})   
    on_pause(visual_pause)
    


def capture(video_path: str, config:dict, on_pause:Callable):
    """
    in:
    video_path: str, abspath
    config: dict{
        stride
        weights_path
        siou
        conf    
    }

    out:
    {
        "timestamp":timestamp,
        "frame_bgr":np.array,
        "areas":[
            "clarity":{
                "clarity": label,
                "score": round(score, 2),
                "suggestion": suggestion,
                "laplacian_variance": blur_metric,
                "stroke_width_variance": stroke_width_variance,
                "details": {"num_stroke_components": len(stroke_widths)},
            },
            "roi": ROIBox{
                x1, y1, x2, y2
            }
        ]}
    """
    stride = config["stride"]
    weights_path = config["weights_path"]
    iou = config["iou"]    
    conf = config["conf"]   
    device = _get_device()
    capture = cv2.VideoCapture(video_path)
    fps:float = _getfps(capture)
    last_frame_area: float = None
    frame_index: int = 0
    last_trait: float = 0.0
    last_detected: list = []
    last_detected_always: list = []
    last_frame: np.array = None
    start_timestamp: float = 0
    while True:        
        ok, frame_bgr = capture.read()
        if not ok: break
        timestamp = round(_gettimestamp(capture, frame_index, fps), 3)        
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_index += 1
        try:
            frame_trait = np.sum(get_abs_trait(frame_bgr, CLARFY_SIZE))
        except ZeroDivisionError:
            continue
        if frame_index !=1:
            if abs(last_trait - frame_trait)/last_trait < CHANGE_RATIO:
                last_trait = frame_trait
                continue
            if frame_index %stride != 0 :
                continue
        yolo_prediction = _get_textarea_yolo(frame_bgr, weights_path, conf, iou, device)        
        captured_text_areas = _get_boxes(yolo_prediction, frame_w, frame_h)
        last_detected_always = captured_text_areas
        borderROI = ROIBox(10000, 10000, 0, 0)
        if len(captured_text_areas) == 0: continue
        for box in captured_text_areas:
            x1, y1, x2, y2 = box.as_tuple()
            if x1 < borderROI.x1: borderROI.x1 = x1
            if y1 < borderROI.y1: borderROI.y1 = y1
            if x2 > borderROI.x2: borderROI.x2 = x2
            if y2 > borderROI.y2: borderROI.y2 = y2
        current_area = borderROI.area()
        renewed = False
        if last_frame_area != None:
            diff = current_area - last_frame_area
            if diff < -1*last_frame_area*AREA_REDUCE_R: 
                renewed = True
        last_frame_area = current_area
        if renewed:
            _pack_pause(start_timestamp, last_frame, last_detected, device, on_pause)
            start_timestamp = timestamp
        last_frame = frame_bgr
        last_detected = captured_text_areas
    _pack_pause(start_timestamp, last_frame, last_detected_always, device, on_pause)
    capture.release()    
        

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

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)
    
    def area(self):
        return (self.x2-self.x1)*(self.y2-self.y1)
    
def crop_roi(image: np.ndarray, roi: ROIBox) -> np.ndarray:
    x1, y1, x2, y2 = roi.as_tuple()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI")
    return image[y1:y2, x1:x2].copy()
    
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

