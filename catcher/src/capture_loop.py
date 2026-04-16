"""
capture_loop.py — Producer side of Catcher
Reads video frames, detects candidate board pauses, and extracts audio.

This version is intentionally clean and marker-friendly:
- one file with the core producer logic
- simple visual change gating
- ffmpeg audio extraction
- optional YOLO hook can be added later
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def extract_audio_ffmpeg(
    video_path: str,
    wav_out: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Extract mono 16k WAV audio from a video using ffmpeg.
    """
    if wav_out is None:
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        Path(tmp_name).unlink(missing_ok=True)
        out_path = Path(tmp_name)
    else:
        out_path = Path(wav_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found on PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e}") from e

    return str(out_path.resolve())


def _get_video_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("video") or {}


def _frame_timestamp_sec(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if pos_msec > 0:
        return round(pos_msec / 1000.0, 3)
    if fps > 0:
        return round(frame_index / fps, 3)
    return 0.0


def _normalised_gray(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def _change_ratio(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> float:
    if prev_gray is None:
        return 1.0
    diff = cv2.absdiff(prev_gray, gray)
    return float(np.mean(diff) / 255.0)


def _laplacian_score(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _text_presence_ratio(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float(np.count_nonzero(bw))
    total = float(bw.size) if bw.size else 1.0
    return ink / total


def _mock_board_roi(frame_bgr: np.ndarray) -> Dict[str, int]:
    """
    Conservative fallback ROI:
    use a centered region covering most of the frame.
    Replace this later with YOLO if needed.
    """
    h, w = frame_bgr.shape[:2]
    x1 = int(w * 0.08)
    y1 = int(h * 0.10)
    x2 = int(w * 0.92)
    y2 = int(h * 0.90)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _crop_roi(frame_bgr: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    return frame_bgr[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]].copy()


def run_capture_loop(
    video_path: str,
    config: Dict[str, Any],
    stop_event: threading.Event,
    keyframe_results: List[dict],
) -> None:
    """
    Producer loop.

    Appends dict objects into keyframe_results, which textual_loop.py will consume.
    Each item looks like:
    {
        "timestamp_sec": 12.345,
        "frame_index": 123,
        "roi": {...},
        "frame_bgr": np.ndarray,
        "roi_bgr": np.ndarray,
        "change_ratio": 0.11,
        "laplacian": 215.2,
        "text_presence_ratio": 0.006,
        "processed": False,
    }
    """
    video_cfg = _get_video_cfg(config)

    max_keyframes = int(video_cfg.get("max_keyframes", 20))
    min_change_ratio = float(video_cfg.get("min_change_ratio", 0.08))
    min_keyframe_score = float(video_cfg.get("min_keyframe_score", 40.0))
    text_presence_enabled = bool(video_cfg.get("text_presence_enabled", True))
    text_presence_min_ink_ratio = float(video_cfg.get("text_presence_min_ink_ratio", 0.002))
    fast_mode = bool(video_cfg.get("fast_mode", False))

    stride = 5 if not fast_mode else 10

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    prev_gray: Optional[np.ndarray] = None
    frame_index = -1

    try:
        while not stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % stride != 0:
                continue

            gray = _normalised_gray(frame_bgr)
            change_ratio = _change_ratio(prev_gray, gray)
            prev_gray = gray

            if change_ratio < min_change_ratio:
                continue

            lap_score = _laplacian_score(frame_bgr)
            if lap_score < min_keyframe_score:
                continue

            ink_ratio = _text_presence_ratio(frame_bgr)
            if text_presence_enabled and ink_ratio < text_presence_min_ink_ratio:
                continue

            roi = _mock_board_roi(frame_bgr)
            roi_bgr = _crop_roi(frame_bgr, roi)
            timestamp_sec = _frame_timestamp_sec(cap, frame_index, fps)

            keyframe_results.append({
                "timestamp_sec": timestamp_sec,
                "frame_index": frame_index,
                "roi": roi,
                "frame_bgr": frame_bgr.copy(),
                "roi_bgr": roi_bgr,
                "change_ratio": round(change_ratio, 4),
                "laplacian": round(lap_score, 2),
                "text_presence_ratio": round(ink_ratio, 6),
                "processed": False,
            })

            if len(keyframe_results) >= max_keyframes:
                break

    finally:
        cap.release()
        stop_event.set()