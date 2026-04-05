# Video helpers for the classroom pipeline.
# We keep this separate from pipeline.py so the still-image path stays simple.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from blackboard_analytics.module_a_blackboard_ocr import ROIBox, crop_roi, detect_blackboard_roi, preprocess_image
from blackboard_analytics.module_b_clarity import evaluate_handwriting_clarity

logger = logging.getLogger(__name__)


def _frame_timestamp_sec(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    ts_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if ts_msec > 0:
        return ts_msec / 1000.0
    if fps > 0:
        return float(frame_index / fps)
    return 0.0


def _sample_video_frames(
    video_path: str,
    *,
    sample_every_sec: float,
) -> List[Dict[str, Any]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    frame_step = max(1, int(round(max(0.25, float(sample_every_sec)) * fps)))

    sampled: List[Dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                sampled.append(
                    {
                        "timestamp_sec": round(_frame_timestamp_sec(capture, frame_index, fps), 3),
                        "frame_bgr": frame,
                    }
                )
            frame_index += 1
    finally:
        capture.release()

    if not sampled:
        raise RuntimeError(f"No frames sampled from video: {video_path}")
    return sampled


def _board_signature(board_crop: np.ndarray, *, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    # Scene-cut heuristics are too coarse for incremental board writing, so compare
    # a normalized ink mask of the detected board region instead.
    _, ink_mask = preprocess_image(board_crop)
    resized = cv2.resize(ink_mask, size, interpolation=cv2.INTER_AREA)
    return (resized > 32).astype(np.uint8) * 255


def _change_ratio(previous_signature: Optional[np.ndarray], current_signature: np.ndarray) -> float:
    if previous_signature is None:
        return 1.0
    diff = cv2.absdiff(previous_signature, current_signature)
    return float(np.mean(diff) / 255.0)


def _roi_area_ratio(roi: ROIBox, frame_shape: tuple[int, int, int]) -> float:
    h, w = frame_shape[:2]
    frame_area = max(1, h * w)
    return float(max(0, roi.x2 - roi.x1) * max(0, roi.y2 - roi.y1) / frame_area)


def extract_blackboard_keyframes(
    video_path: str,
    config: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    cfg = config or {}
    video_cfg = cfg.get("video", {})
    yolo_cfg = cfg.get("yolo", {})
    clarity_cfg = cfg.get("clarity", {})

    sampled_frames = _sample_video_frames(
        video_path,
        sample_every_sec=float(video_cfg.get("sample_every_sec", 2.0)),
    )

    min_change_ratio = float(video_cfg.get("min_change_ratio", 0.02))
    min_keyframe_score = float(video_cfg.get("min_keyframe_score", 45.0))
    max_keyframes = max(1, int(video_cfg.get("max_keyframes", 8)))
    min_roi_area_ratio = float(video_cfg.get("min_roi_area_ratio", 0.35))

    kept: List[Dict[str, Any]] = []
    best_fallback: Optional[Dict[str, Any]] = None
    previous_signature: Optional[np.ndarray] = None

    for sampled in sampled_frames:
        frame_bgr = sampled["frame_bgr"]
        roi, roi_method = detect_blackboard_roi(
            frame_bgr,
            yolo_weights_path=yolo_cfg.get("weights_path"),
            conf=float(yolo_cfg.get("conf", 0.25)),
            iou=float(yolo_cfg.get("iou", 0.45)),
            blackboard_class_id=int(yolo_cfg.get("blackboard_class_id", 0)),
        )
        if _roi_area_ratio(roi, frame_bgr.shape) < min_roi_area_ratio:
            frame_h, frame_w = frame_bgr.shape[:2]
            roi = ROIBox(0, 0, frame_w, frame_h)
            roi_method = "full_frame_video_fallback"
        board_crop = crop_roi(frame_bgr, roi)
        clarity = evaluate_handwriting_clarity(
            board_crop,
            laplacian_clear_min=float(clarity_cfg.get("laplacian_clear_min", 120.0)),
            laplacian_messy_max=float(clarity_cfg.get("laplacian_messy_max", 40.0)),
            stroke_variance_messy_min=float(clarity_cfg.get("stroke_variance_messy_min", 8.0)),
        )
        signature = _board_signature(board_crop)
        change_ratio = _change_ratio(previous_signature, signature)

        candidate = {
            "timestamp_sec": sampled["timestamp_sec"],
            "frame_bgr": frame_bgr,
            "roi": roi.as_tuple(),
            "roi_method": roi_method,
            "clarity_result": clarity,
            "change_ratio": round(change_ratio, 4),
        }

        if best_fallback is None or clarity.get("score", 0.0) > best_fallback["clarity_result"].get("score", 0.0):
            best_fallback = candidate

        if float(clarity.get("score", 0.0)) < min_keyframe_score:
            continue

        if previous_signature is None or change_ratio >= min_change_ratio:
            kept.append(candidate)
            previous_signature = signature
            if len(kept) >= max_keyframes:
                break

    if kept:
        return kept
    if best_fallback is not None:
        logger.info("No clear-changing keyframe found; falling back to the clearest sampled frame.")
        return [best_fallback]
    raise RuntimeError(f"No usable keyframe extracted from video: {video_path}")
