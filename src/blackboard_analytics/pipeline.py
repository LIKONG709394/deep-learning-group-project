"""
End-to-end pipeline: video frame + audio -> modules A/B/C/D -> PDF.

Full video workflows need ffmpeg to extract frames and audio; this package provides
`run_from_frame_and_audio` (single BGR frame + audio file) and optional `extract_audio_ffmpeg`.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from blackboard_analytics.module_a_blackboard_ocr import run_module_a
from blackboard_analytics.module_b_clarity import run_module_b
from blackboard_analytics.module_c_whisper import run_module_c
from blackboard_analytics.module_d_semantic import run_module_d
from blackboard_analytics.module_e_report import run_module_e

logger = logging.getLogger(__name__)


def load_bgr_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img


def extract_audio_ffmpeg(video_path: str, wav_out: str, overwrite: bool = True) -> str:
    """Extract mono 16 kHz WAV from video using ffmpeg (ffmpeg must be on PATH)."""
    out = Path(wav_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found; install ffmpeg and add it to PATH") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extract failed: {e}") from e
    return str(out.resolve())


def run_from_frame_and_audio(
    frame_bgr: np.ndarray,
    audio_path: str,
    config: Optional[dict] = None,
    pdf_output: str = "output/teaching_feedback.pdf",
) -> Dict[str, Any]:
    """
    Pipeline:
      A(frame) -> board texts
      B(frame) -> clarity
      C(audio) -> speech text
      D(join(board), speech) -> alignment
      E -> PDF

    Returns:
        Aggregated dict including per-module outputs and errors.
    """
    cfg = config or {}
    errors: Dict[str, str] = {}

    a = run_module_a(frame_bgr, cfg)
    if a.get("error"):
        errors["module_a"] = a["error"]

    b = run_module_b(frame_bgr, cfg)
    if b.get("error"):
        errors["module_b"] = b["error"]

    c = run_module_c(audio_path, cfg)
    if c.get("error"):
        errors["module_c"] = c["error"]

    board_full = "\n".join(a.get("texts") or [])
    d = run_module_d(board_full, c.get("speech_text") or "", cfg)
    if d.get("error"):
        errors["module_d"] = d["error"]

    clarity_dict = (b.get("clarity_result") or {}) if b else {}
    align_dict = (d.get("alignment") or None) if d else None

    payload = {
        "board_lines": a.get("texts") or [],
        "clarity": clarity_dict,
        "alignment": align_dict,
        "speech_text": c.get("speech_text") or "",
        "module_errors": errors or None,
    }
    e = run_module_e(pdf_output, payload)
    if e.get("error"):
        errors["module_e"] = e["error"]

    return {
        "board_texts": a.get("texts"),
        "board_roi": a.get("roi"),
        "roi_method": a.get("roi_method"),
        "clarity": clarity_dict,
        "speech_text": c.get("speech_text"),
        "alignment": align_dict,
        "pdf_path": e.get("pdf_path"),
        "errors": errors,
    }


def run_from_image_and_audio_files(
    image_path: str,
    audio_path: str,
    config: Optional[dict] = None,
    pdf_output: str = "output/teaching_feedback.pdf",
) -> Dict[str, Any]:
    frame = load_bgr_image(image_path)
    return run_from_frame_and_audio(frame, audio_path, config=config, pdf_output=pdf_output)
