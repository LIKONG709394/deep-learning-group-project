from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess

import cv2

from src.alignment import evaluate_alignment
from src.clarity import evaluate_clarity
from src.ocr import run_ocr_on_frame
from src.report_pdf import build_teaching_feedback_pdf
from src.whisper_asr import transcribe_audio


def startpipeline(
    videopath: Optional[str] = None,
    configpath: Optional[str] = None,
    pdfpath: Optional[str] = None,
    imgpath: Optional[str] = None,
    audiopath: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main public entry point.

    Modes:
    - image + audio
    - video

    Returns a structured dict ready for CLI JSON output.
    """
    config = load_config(configpath)
    output_pdf = str(pdfpath or "output/teaching_feedback.pdf")

    if videopath:
        if imgpath or audiopath:
            raise ValueError("Use either video mode or image+audio mode, not both.")
        return run_from_video_file(videopath, config, output_pdf)

    if imgpath and audiopath:
        return run_from_image_and_audio_files(imgpath, audiopath, config, output_pdf)

    raise ValueError("Provide either videopath, or both imgpath and audiopath.")


def run_from_image_and_audio_files(
    imgpath: str,
    audiopath: str,
    config: Dict[str, Any],
    pdfpath: str,
) -> Dict[str, Any]:
    module_errors: Dict[str, str] = {}

    image = load_bgr_image(imgpath)

    board_lines: List[str] = []
    clarity_result: Dict[str, Any] = {}
    asr_result: Dict[str, Any] = {"speech_text": "", "segments": []}
    alignment_result: Dict[str, Any] = {}

    try:
        ocr_result = run_ocr_on_frame(image, config)
        board_lines = ocr_result.get("texts", []) or []
        if ocr_result.get("error"):
            module_errors["ocr"] = str(ocr_result["error"])
    except Exception as e:
        module_errors["ocr"] = str(e)

    try:
        clarity_result = evaluate_clarity(image, config)
    except Exception as e:
        module_errors["clarity"] = str(e)
        clarity_result = {
            "score": 0.0,
            "label": "unknown",
            "suggestion": "Clarity scoring failed.",
        }

    try:
        asr_result = transcribe_audio(audiopath, config)
    except Exception as e:
        module_errors["whisper_asr"] = str(e)
        asr_result = {"speech_text": "", "segments": []}

    try:
        alignment_result = evaluate_alignment(
            board_lines=board_lines,
            speech_text=asr_result.get("speech_text", ""),
            config=config,
        )
    except Exception as e:
        module_errors["alignment"] = str(e)
        alignment_result = {
            "semantic_similarity": 0.0,
            "keyword_overlap_rate": 0.0,
            "verdict": "unknown",
            "matched_topics": [],
            "board_only_topics": [],
            "speech_only_topics": [],
            "board_text": " ".join(board_lines),
            "speech_text": asr_result.get("speech_text", ""),
        }

    try:
        final_pdf = build_teaching_feedback_pdf(
            output_path=pdfpath,
            board_lines=board_lines,
            clarity=clarity_result,
            alignment=alignment_result,
            speech_text=asr_result.get("speech_text", ""),
            module_errors=module_errors or None,
        )
    except Exception as e:
        module_errors["report_pdf"] = str(e)
        final_pdf = None

    return {
        "mode": "image_audio",
        "board_lines": board_lines,
        "clarity": clarity_result,
        "speech_text": asr_result.get("speech_text", ""),
        "segments": asr_result.get("segments", []),
        "alignment": alignment_result,
        "pdf_path": final_pdf,
        "module_errors": module_errors,
    }


def run_from_video_file(videopath: str, config: dict, pdfpath: str):
    temp_audio = None
    temp_frame = None

    try:
        frame = extract_representative_frame(videopath)
        temp_frame = save_temp_frame(frame)
        temp_audio = extract_audio_for_whisper(videopath)

        result = run_from_image_and_audio_files(
            imgpath=temp_frame,
            audiopath=temp_audio,
            config=config,
            pdfpath=pdfpath,
        )
        result["mode"] = "video"
        result["video_path"] = videopath
        return result

    finally:
        for p in [temp_audio, temp_frame]:
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except PermissionError:
                    pass


def load_config(configpath: Optional[str]) -> Dict[str, Any]:
    default_config: Dict[str, Any] = {
        "trocr": {
            "ocr_engine": "easyocr",
            "easyocr_languages": ["en"],
            "paddleocr_lang": "en",
        },
        "clarity": {
            "laplacian_clear_min": 120.0,
            "laplacian_messy_max": 40.0,
            "stroke_variance_messy_min": 8.0,
        },
        "whisper": {
            "model_size": "base",
            "language": "en",
            "task": "transcribe",
            "initial_prompt": "English classroom lecture with clear educational vocabulary.",
            "enable_silence_segmentation": False,
            "silence_threshold": 0.0004,
            "silence_duration_sec": 1.0,
            "min_segment_sec": 2.0,
            "max_segment_sec": 20.0,
            "analysis_window_sec": 0.1,
            "skip_energy_threshold": 0.00012,
        },
        "sbert": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
        "semantic": {
            "high_match_min": 0.72,
            "partial_min": 0.45,
            "keyword_overlap_high": 0.35,
            "keyword_overlap_partial": 0.20,
        },
    }

    if not configpath:
        return default_config

    path = Path(configpath)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {configpath}")

    if path.suffix.lower() == ".json":
        user_cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("PyYAML is required for YAML config files.") from e
        user_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return deep_merge(default_config, user_cfg)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_bgr_image(imagepath: str):
    image = cv2.imread(str(imagepath))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {imagepath}")
    return image


def extract_representative_frame(videopath: str):
    cap = cv2.VideoCapture(str(videopath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {videopath}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_index = max(0, frame_count // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read representative frame from video.")

    return frame


def save_temp_frame(frame) -> str:
    fd, temp_name = tempfile.mkstemp(suffix=".png")
    Path(temp_name).unlink(missing_ok=True)
    ok = cv2.imwrite(temp_name, frame)
    if not ok:
        raise RuntimeError("Failed to save temporary frame image.")
    return temp_name


def extract_audio_for_whisper(video_path: str) -> str:
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    out_path = Path(temp_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return str(out_path)
    except Exception:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

def try_delete(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass