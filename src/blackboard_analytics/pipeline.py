# One still frame of the board + one audio clip -> structured feedback + PDF.
# Order matches how a person would look at a lesson: what is written, how legible it is,
# what was said, whether talk matches the board, then a printable summary.

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from blackboard_analytics.module_a_blackboard_ocr import TROCR_PRINTED, recognize_text_lines_in_image, run_module_a
from blackboard_analytics.module_b_clarity import run_module_b
from blackboard_analytics.module_c_whisper import run_module_c
from blackboard_analytics.module_d_semantic import run_module_d
from blackboard_analytics.module_e_report import run_module_e
from blackboard_analytics.module_video_keyframes import extract_blackboard_keyframes

logger = logging.getLogger(__name__)


def _merge_unique_lines(existing: list[str], fresh: list[str]) -> list[str]:
    merged = list(existing)
    seen = {line.strip().casefold() for line in existing if line and line.strip()}
    for line in fresh:
        cleaned = (line or "").strip()
        if not cleaned:
            continue
        token = cleaned.casefold()
        if token in seen:
            continue
        seen.add(token)
        merged.append(cleaned)
    return merged


def _texts_look_low_value(lines: list[str]) -> bool:
    joined = " ".join((line or "").strip() for line in lines).strip()
    if not joined:
        return True
    alpha_count = sum(ch.isalpha() for ch in joined)
    digit_count = sum(ch.isdigit() for ch in joined)
    useful_count = sum(ch.isalnum() for ch in joined)
    if alpha_count >= 8:
        return False
    if useful_count == 0:
        return True
    return digit_count >= alpha_count and useful_count <= 12


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
        "speech_segments": spoken_transcript.get("speech_segments") or [],
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


def run_from_video_file(
    video_path: str,
    config: Optional[dict] = None,
    pdf_output: str = "output/teaching_feedback.pdf",
) -> Dict[str, Any]:
    settings = config or {}
    problems: Dict[str, str] = {}
    trocr_opts = settings.get("trocr", {})
    printed_fallback_model = str(trocr_opts.get("printed_model_name", TROCR_PRINTED))
    use_video_printed_fallback = bool(trocr_opts.get("video_enable_printed_fallback", True))

    with tempfile.TemporaryDirectory(prefix="blackboard_video_") as tmp_dir:
        extracted_audio = extract_audio_ffmpeg(video_path, str(Path(tmp_dir) / "audio.wav"))

        try:
            keyframes = extract_blackboard_keyframes(video_path, settings)
        except Exception as e:
            logger.exception("extract_blackboard_keyframes")
            raise RuntimeError(f"Video keyframe extraction failed: {e}") from e

        aggregated_board_lines: list[str] = []
        best_clarity: Dict[str, Any] = {}
        best_roi = None
        best_roi_method = None
        keyframe_results = []

        for item in keyframes:
            frame_bgr = item["frame_bgr"]
            board_reading = run_module_a(frame_bgr, settings)
            if board_reading.get("error") and "module_a" not in problems:
                problems["module_a"] = board_reading["error"]

            texts = board_reading.get("texts") or []
            if use_video_printed_fallback and _texts_look_low_value(texts):
                try:
                    fallback_texts = recognize_text_lines_in_image(
                        frame_bgr,
                        trocr_model_name=printed_fallback_model,
                    )
                    if fallback_texts and not _texts_look_low_value(fallback_texts):
                        texts = fallback_texts
                except Exception as e:
                    logger.warning("Printed OCR fallback failed on video keyframe: %s", e)
            aggregated_board_lines = _merge_unique_lines(aggregated_board_lines, texts)

            clarity_result = item.get("clarity_result") or {}
            if clarity_result and clarity_result.get("score", 0.0) >= best_clarity.get("score", 0.0):
                best_clarity = clarity_result
                best_roi = board_reading.get("roi") or item.get("roi")
                best_roi_method = board_reading.get("roi_method") or item.get("roi_method")

            keyframe_results.append(
                {
                    "timestamp_sec": item.get("timestamp_sec"),
                    "change_ratio": item.get("change_ratio"),
                    "clarity": clarity_result,
                    "roi": board_reading.get("roi") or item.get("roi"),
                    "roi_method": board_reading.get("roi_method") or item.get("roi_method"),
                    "board_texts": texts,
                }
            )

        spoken_transcript = run_module_c(extracted_audio, settings)
        if spoken_transcript.get("error"):
            problems["module_c"] = spoken_transcript["error"]

        board_as_paragraph = "\n".join(aggregated_board_lines)
        speech_text = spoken_transcript.get("speech_text") or ""

        lesson_alignment = run_module_d(board_as_paragraph, speech_text, settings)
        if lesson_alignment.get("error"):
            problems["module_d"] = lesson_alignment["error"]

        alignment_summary = lesson_alignment.get("alignment")
        bundle_for_pdf = {
            "board_lines": aggregated_board_lines,
            "clarity": best_clarity,
            "alignment": alignment_summary,
            "speech_text": speech_text,
            "module_errors": problems or None,
        }
        pdf_bundle = run_module_e(pdf_output, bundle_for_pdf)
        if pdf_bundle.get("error"):
            problems["module_e"] = pdf_bundle["error"]

    return {
        "input_mode": "video",
        "board_texts": aggregated_board_lines,
        "board_roi": best_roi,
        "roi_method": best_roi_method,
        "clarity": best_clarity,
        "speech_text": spoken_transcript.get("speech_text"),
        "speech_segments": spoken_transcript.get("speech_segments") or [],
        "alignment": alignment_summary,
        "pdf_path": pdf_bundle.get("pdf_path"),
        "errors": problems,
        "video_keyframes": keyframe_results,
    }
