# Video helpers for the classroom pipeline.
# We keep this separate from pipeline.py so the still-image path stays simple.

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from blackboard_analytics.module_a_blackboard_ocr import ROIBox, crop_roi, detect_blackboard_roi, preprocess_image
from blackboard_analytics.module_b_clarity import evaluate_handwriting_clarity

logger = logging.getLogger(__name__)


def _find_ffmpeg_executable() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for key in ("FFMPEG_PATH", "BLACKBOARD_FFMPEG"):
        raw = os.environ.get(key, "").strip().strip('"')
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            return str(p)
        exe = p / "ffmpeg.exe"
        if exe.is_file():
            return str(exe)
    if sys.platform != "win32":
        return None
    for pf in (
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ):
        if not pf:
            continue
        for rel in (
            Path(pf) / "ffmpeg" / "bin" / "ffmpeg.exe",
            Path(pf) / "ffmpeg" / "ffmpeg.exe",
        ):
            if rel.is_file():
                return str(rel)
    la = os.environ.get("LOCALAPPDATA", "")
    if la:
        winget_link = Path(la) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
        if winget_link.is_file():
            return str(winget_link)
    prof = os.environ.get("USERPROFILE", "")
    if prof:
        scoop = Path(prof) / "scoop" / "shims" / "ffmpeg.exe"
        if scoop.is_file():
            return str(scoop)
    return None


def _ensure_ffmpeg_on_path() -> bool:
    if shutil.which("ffmpeg"):
        return True
    exe = _find_ffmpeg_executable()
    if not exe:
        return False
    folder = str(Path(exe).resolve().parent)
    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    logger.info("Prepended ffmpeg directory to process PATH: %s", folder)
    return shutil.which("ffmpeg") is not None

_KF_PNG = re.compile(r"^kf_(\d+)\.png$", re.IGNORECASE)


def _find_ffprobe_executable() -> Optional[str]:
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    parent = Path(ffmpeg).resolve().parent
    for name in ("ffprobe.exe", "ffprobe"):
        cand = parent / name
        if cand.is_file():
            return str(cand)
    return None


def _ffprobe_keyframe_packet_times(video_path: str, ffprobe: str) -> List[float]:
    """Fast path: packet-level keyframe flags (avoids scanning every frame on long files)."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time,dts_time,flags",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"ffprobe failed: {e}") from e
    times: List[float] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        flags = parts[-1]
        if "K" not in flags:
            continue
        t_val: Optional[float] = None
        for cell in parts[:-1]:
            if not cell or cell == "N/A":
                continue
            try:
                t_val = float(cell)
                break
            except ValueError:
                continue
        if t_val is not None:
            times.append(t_val)
    return times


def _ffprobe_iframe_times(video_path: str, ffprobe: str) -> List[float]:
    """Per-frame pict_type == I (slower; used if packet keyframe list is empty)."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pict_type",
        "-of",
        "csv=p=0",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"ffprobe failed: {e}") from e
    times: List[float] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pict = parts[-1]
        if pict != "I":
            continue
        t_val: Optional[float] = None
        for cell in parts[:-1]:
            if not cell or cell == "N/A":
                continue
            try:
                t_val = float(cell)
                break
            except ValueError:
                continue
        if t_val is not None:
            times.append(t_val)
    return times


def _ffprobe_keyframe_times(video_path: str, ffprobe: str) -> List[float]:
    times = _ffprobe_keyframe_packet_times(video_path, ffprobe)
    if times:
        return times
    return _ffprobe_iframe_times(video_path, ffprobe)


def _extract_iframe_frames_ffmpeg(
    video_path: str,
    *,
    max_iframes: int,
) -> List[Dict[str, Any]]:
    if not _ensure_ffmpeg_on_path():
        raise RuntimeError("ffmpeg not found on PATH (needed for I-frame extraction)")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    ffprobe = _find_ffprobe_executable()
    if not ffprobe:
        raise RuntimeError("ffprobe not found (install ffmpeg bundle with ffprobe)")

    iframe_times_full = sorted(_ffprobe_keyframe_times(video_path, ffprobe))
    if not iframe_times_full:
        raise RuntimeError("ffprobe found no keyframes (or no timestamps); cannot use iframe mode")

    n_decode = min(len(iframe_times_full), max(1, int(max_iframes)))
    iframe_times = iframe_times_full[:n_decode]

    with tempfile.TemporaryDirectory(prefix="blackboard_iframes_") as tmp:
        out_pattern = str(Path(tmp) / "kf_%06d.png")
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-skip_frame",
            "nokey",
            "-i",
            video_path,
            "-vsync",
            "0",
            "-frames:v",
            str(n_decode),
            out_pattern,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()
            raise RuntimeError(f"ffmpeg I-frame extract failed: {err or e}") from e

        paths: List[Tuple[int, Path]] = []
        for p in Path(tmp).iterdir():
            if not p.is_file():
                continue
            m = _KF_PNG.match(p.name)
            if m:
                paths.append((int(m.group(1)), p))
        paths.sort(key=lambda x: x[0])

        if not paths:
            raise RuntimeError("ffmpeg produced no keyframe PNGs")

        n = min(len(paths), len(iframe_times))
        if n < len(paths) or n < len(iframe_times):
            logger.warning(
                "I-frame count mismatch: expected=%s pngs=%s; pairing first %s",
                len(iframe_times),
                len(paths),
                n,
            )

        out: List[Dict[str, Any]] = []
        for i in range(n):
            _, png_path = paths[i]
            frame_bgr = cv2.imread(str(png_path))
            if frame_bgr is None:
                logger.warning("Could not read decoded keyframe: %s", png_path)
                continue
            out.append(
                {
                    "timestamp_sec": round(float(iframe_times[i]), 3),
                    "frame_bgr": frame_bgr,
                }
            )

        if not out:
            raise RuntimeError("No readable I-frame images after ffmpeg decode")

        logger.info(
            "FFmpeg I-frame extract: decoded %s keyframes (probe I-count=%s, cap=%s)",
            len(out),
            len(iframe_times_full),
            max_iframes,
        )
        return out


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


def _yolo_world_or_weights_configured(yolo_cfg: dict, yolo_world_cfg: dict) -> bool:
    w = yolo_cfg.get("weights_path")
    weights_ok = bool(w and Path(str(w)).is_file())
    yw = yolo_world_cfg if isinstance(yolo_world_cfg, dict) else {}
    return bool(yw.get("enabled")) or weights_ok


def _yolo_monitor_scan_video(
    video_path: str,
    *,
    yolo_cfg: dict,
    yolo_world_cfg: dict,
    stride_sec: float,
    max_pool: int,
    min_roi_area_ratio: float,
) -> List[Dict[str, Any]]:
    """
    Sequential video pass: every stride_sec run YOLO (World or custom weights) once,
    keep frames where the board ROI is large enough. No FFmpeg keyframe extraction.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    frame_step = max(1, int(round(max(0.1, float(stride_sec)) * fps)))

    yolo_world_d = yolo_world_cfg if isinstance(yolo_world_cfg, dict) else {}
    conf = float(yolo_cfg.get("conf", 0.25))
    iou = float(yolo_cfg.get("iou", 0.45))
    board_class = int(yolo_cfg.get("blackboard_class_id", 0))
    weights_path = yolo_cfg.get("weights_path")

    out: List[Dict[str, Any]] = []
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            ts = round(_frame_timestamp_sec(capture, frame_index, fps), 3)
            roi, roi_method = detect_blackboard_roi(
                frame_bgr,
                yolo_weights_path=weights_path,
                conf=conf,
                iou=iou,
                blackboard_class_id=board_class,
                yolo_world=yolo_world_d,
            )
            if _roi_area_ratio(roi, frame_bgr.shape) < min_roi_area_ratio:
                frame_index += 1
                continue

            out.append(
                {
                    "timestamp_sec": ts,
                    "frame_bgr": frame_bgr,
                    "roi": roi.as_tuple(),
                    "roi_method": roi_method,
                }
            )
            if len(out) >= max(1, int(max_pool)):
                break
            frame_index += 1
    finally:
        capture.release()

    if not out:
        raise RuntimeError(
            "YOLO monitor found no frames with a sufficiently large board ROI; "
            "check yolo_world.text_classes / lighting, or lower video.min_roi_area_ratio."
        )
    logger.info("YOLO monitor: kept %s frames (stride_sec=%s, max_pool=%s)", len(out), stride_sec, max_pool)
    return out


def extract_blackboard_keyframes(
    video_path: str,
    config: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    cfg = config or {}
    video_cfg = cfg.get("video", {})
    yolo_cfg = cfg.get("yolo", {})
    yolo_world_cfg = cfg.get("yolo_world") if isinstance(cfg.get("yolo_world"), dict) else {}
    clarity_cfg = cfg.get("clarity", {})

    keyframe_source = str(video_cfg.get("keyframe_source", "yolo_monitor") or "yolo_monitor").lower().strip()
    sampled_frames: List[Dict[str, Any]]
    min_roi_area_ratio = float(video_cfg.get("min_roi_area_ratio", 0.35))

    if keyframe_source in ("yolo_monitor", "yolo", "monitor", "continuous"):
        if not _yolo_world_or_weights_configured(yolo_cfg, yolo_world_cfg):
            logger.warning(
                "video.keyframe_source=yolo_monitor but neither yolo_world.enabled nor yolo.weights_path; "
                "ROI will fall back to heuristics/full frame."
            )
        try:
            sampled_frames = _yolo_monitor_scan_video(
                video_path,
                yolo_cfg=yolo_cfg,
                yolo_world_cfg=yolo_world_cfg,
                stride_sec=float(video_cfg.get("yolo_monitor_stride_sec", 0.5)),
                max_pool=max(8, int(video_cfg.get("yolo_monitor_max_pool", 120))),
                min_roi_area_ratio=min_roi_area_ratio,
            )
        except Exception as e:
            logger.warning("YOLO monitor failed (%s); falling back to time-based sampling.", e)
            sampled_frames = _sample_video_frames(
                video_path,
                sample_every_sec=float(video_cfg.get("sample_every_sec", 2.0)),
            )
    elif keyframe_source in ("iframe", "i_frame", "if", "i"):
        try:
            sampled_frames = _extract_iframe_frames_ffmpeg(
                video_path,
                max_iframes=max(1, int(video_cfg.get("iframe_max_decode", 80))),
            )
        except Exception as e:
            logger.warning("FFmpeg I-frame mode failed (%s); falling back to time-based sampling.", e)
            sampled_frames = _sample_video_frames(
                video_path,
                sample_every_sec=float(video_cfg.get("sample_every_sec", 2.0)),
            )
    elif keyframe_source in ("sampled", "interval", "time", "uniform"):
        sampled_frames = _sample_video_frames(
            video_path,
            sample_every_sec=float(video_cfg.get("sample_every_sec", 2.0)),
        )
    else:
        logger.warning("Unknown video.keyframe_source %r; trying yolo_monitor then sampled.", keyframe_source)
        try:
            sampled_frames = _yolo_monitor_scan_video(
                video_path,
                yolo_cfg=yolo_cfg,
                yolo_world_cfg=yolo_world_cfg,
                stride_sec=float(video_cfg.get("yolo_monitor_stride_sec", 0.5)),
                max_pool=max(8, int(video_cfg.get("yolo_monitor_max_pool", 120))),
                min_roi_area_ratio=min_roi_area_ratio,
            )
        except Exception as e:
            logger.warning("YOLO monitor failed (%s); falling back to time-based sampling.", e)
            sampled_frames = _sample_video_frames(
                video_path,
                sample_every_sec=float(video_cfg.get("sample_every_sec", 2.0)),
            )

    min_change_ratio = float(video_cfg.get("min_change_ratio", 0.02))
    min_keyframe_score = float(video_cfg.get("min_keyframe_score", 45.0))
    max_keyframes = max(1, int(video_cfg.get("max_keyframes", 8)))

    kept: List[Dict[str, Any]] = []
    best_fallback: Optional[Dict[str, Any]] = None
    previous_signature: Optional[np.ndarray] = None

    for sampled in sampled_frames:
        frame_bgr = sampled["frame_bgr"]
        pre_roi = sampled.get("roi")
        pre_method = sampled.get("roi_method")
        if pre_roi is not None and len(pre_roi) == 4:
            roi = ROIBox(int(pre_roi[0]), int(pre_roi[1]), int(pre_roi[2]), int(pre_roi[3]))
            roi_method = str(pre_method or "precomputed_roi")
        else:
            roi, roi_method = detect_blackboard_roi(
                frame_bgr,
                yolo_weights_path=yolo_cfg.get("weights_path"),
                conf=float(yolo_cfg.get("conf", 0.25)),
                iou=float(yolo_cfg.get("iou", 0.45)),
                blackboard_class_id=int(yolo_cfg.get("blackboard_class_id", 0)),
                yolo_world=yolo_world_cfg,
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
