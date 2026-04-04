# One still frame of the board + one audio clip -> structured feedback + PDF.
# Order matches how a person would look at a lesson: what is written, how legible it is,
# what was said, whether talk matches the board, then a printable summary.

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
    # mono 16kHz wav; needs ffmpeg on PATH
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
    settings = config or {}
    # Anything that went wrong in a step; we keep going so the report still has partial info.
    problems: Dict[str, str] = {}

    # Step 1 - What text appears on the board? (find the board area, then read lines.)
    board_reading = run_module_a(frame_bgr, settings)
    if board_reading.get("error"):
        problems["module_a"] = board_reading["error"]

    # Step 2 - How easy is that handwriting to read from a photo? (blur + stroke consistency.)
    handwriting_clarity = run_module_b(frame_bgr, settings)
    if handwriting_clarity.get("error"):
        problems["module_b"] = handwriting_clarity["error"]

    # Step 3 - What does the teacher actually say on the recording?
    spoken_transcript = run_module_c(audio_path, settings)
    if spoken_transcript.get("error"):
        problems["module_c"] = spoken_transcript["error"]

    lines_on_board = board_reading.get("texts") or []
    board_as_paragraph = "\n".join(lines_on_board)
    speech_text = spoken_transcript.get("speech_text") or ""

    # Step 4 - Does the lesson audio line up with what is written? (meaning + shared terms.)
    lesson_alignment = run_module_d(board_as_paragraph, speech_text, settings)
    if lesson_alignment.get("error"):
        problems["module_d"] = lesson_alignment["error"]

    clarity_numbers = handwriting_clarity.get("clarity_result") or {}
    alignment_summary = lesson_alignment.get("alignment")

    # Step 5 - Turn the above into one PDF someone can save or print.
    bundle_for_pdf = {
        "board_lines": lines_on_board,
        "clarity": clarity_numbers,
        "alignment": alignment_summary,
        "speech_text": speech_text,
        "module_errors": problems or None,
    }
    pdf_bundle = run_module_e(pdf_output, bundle_for_pdf)
    if pdf_bundle.get("error"):
        problems["module_e"] = pdf_bundle["error"]

    return {
        "board_texts": board_reading.get("texts"),
        "board_roi": board_reading.get("roi"),
        "roi_method": board_reading.get("roi_method"),
        "clarity": clarity_numbers,
        "speech_text": spoken_transcript.get("speech_text"),
        "alignment": alignment_summary,
        "pdf_path": pdf_bundle.get("pdf_path"),
        "errors": problems,
    }


def run_from_image_and_audio_files(
    image_path: str,
    audio_path: str,
    config: Optional[dict] = None,
    pdf_output: str = "output/teaching_feedback.pdf",
) -> Dict[str, Any]:
    picture = load_bgr_image(image_path)
    return run_from_frame_and_audio(picture, audio_path, config=config, pdf_output=pdf_output)
