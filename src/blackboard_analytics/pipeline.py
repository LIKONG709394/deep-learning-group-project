# One still frame of the board + one audio clip -> structured feedback + PDF.
# Video input reuses the same OCR / Whisper / semantic modules, with optional
# debug artifacts to help inspect keyframe selection and OCR inputs.

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from blackboard_analytics.module_a_alt_ocr import normalize_ocr_engine_name
from blackboard_analytics.module_a_blackboard_ocr import (
    TROCR_DEFAULT,
    TROCR_PRINTED,
    coerce_roi_box,
    parse_trocr_device_option,
    prepare_ocr_inputs,
    recognize_text_lines_in_image,
    run_module_a,
    segment_text_lines,
)
from blackboard_analytics.module_b_clarity import run_module_b
from blackboard_analytics.module_c_whisper import run_module_c
from blackboard_analytics.module_d_deepseek import run_deepseek_filter_board_lines, run_module_d_deepseek
from blackboard_analytics.module_d_semantic import run_module_d
from blackboard_analytics.module_e_report import run_module_e
from blackboard_analytics.module_video_keyframes import (
    build_content_signature,
    content_signature_similarity,
    extract_blackboard_keyframes,
)

logger = logging.getLogger(__name__)
_QUOTE_LIKE_RE = re.compile(r"[`'\"“”‘’]")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")
_WS_RE = re.compile(r"\s+")


def _recognize_line_image_kwargs(
    settings: dict,
    *,
    trocr_model_name: str,
    ocr_engine_override: Optional[str] = None,
) -> dict[str, Any]:
    t = settings.get("trocr") if isinstance(settings.get("trocr"), dict) else {}
    dev = parse_trocr_device_option(t.get("device", "auto"))
    eng = ocr_engine_override or normalize_ocr_engine_name(t.get("ocr_engine", "trocr"))
    raw_langs = t.get("easyocr_languages")
    if isinstance(raw_langs, str) and raw_langs.strip():
        easy_langs: List[str] = [raw_langs.strip()]
    elif isinstance(raw_langs, list):
        easy_langs = [str(x).strip() for x in raw_langs if str(x).strip()]
        if not easy_langs:
            easy_langs = ["en"]
    else:
        easy_langs = ["en"]
    paddle_lang = str(t.get("paddleocr_lang", "en") or "en")
    return {
        "ocr_engine": eng,
        "trocr_model_name": trocr_model_name,
        "trocr_device": dev,
        "easyocr_languages": easy_langs,
        "paddleocr_lang": paddle_lang,
    }


# When fast_mode is on, still keep some text harvest + more keyframes so PPT/printed slides are not skipped.
_FAST_VIDEO_PRESET: dict[str, Any] = {
    "max_keyframes": 14,
    "yolo_monitor_stride_sec": 0.75,
    "yolo_monitor_max_pool": 96,
    "text_harvest_every_sec": 2.5,
    "text_harvest_max_scans": 100,
    "text_harvest_full_frame_printed": True,
    "min_keyframe_score": 40.0,
}


def _apply_fast_video_settings(settings: dict) -> dict:
    base: dict[str, Any] = dict(settings)
    video_cfg = dict(base.get("video") or {})
    env_fast = os.environ.get("BLACKBOARD_VIDEO_FAST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not bool(video_cfg.get("fast_mode")) and not env_fast:
        base["video"] = video_cfg
        return base

    merged = {**video_cfg, **_FAST_VIDEO_PRESET}
    merged["fast_mode"] = True
    base["video"] = merged
    if bool(merged.get("fast_whisper_tiny")):
        whisper_cfg = dict(base.get("whisper") or {})
        whisper_cfg["model_size"] = "tiny"
        base["whisper"] = whisper_cfg
        logger.info("video.fast_whisper_tiny enabled: using Whisper tiny")
    return base


def _merge_unique_lines(existing: list[str], fresh: list[str]) -> list[str]:
    merged = list(existing)
    seen = [_canonicalize_line_for_dedupe(line) for line in existing if line and line.strip()]
    for line in fresh:
        cleaned = (line or "").strip()
        if not cleaned:
            continue
        token = _canonicalize_line_for_dedupe(cleaned)
        if not token:
            continue
        if any(_lines_near_duplicate(token, prior) for prior in seen):
            continue
        seen.append(token)
        merged.append(cleaned)
    return merged


def _canonicalize_line_for_dedupe(text: str) -> str:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return ""
    cleaned = _QUOTE_LIKE_RE.sub("", cleaned)
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return _WS_RE.sub(" ", cleaned).strip()


def _lines_near_duplicate(left: str, right: str, *, min_ratio: float = 0.94) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) >= 8 and shorter in longer:
        return True
    if len(shorter) < 10:
        return False
    return SequenceMatcher(None, left, right).ratio() >= min_ratio


def _filter_noise_board_lines(
    lines: list[str],
    *,
    min_chars: int,
    min_letters: int,
) -> list[str]:
    filtered: list[str] = []
    for line in lines:
        text = (line or "").strip()
        if len(text) < max(1, min_chars):
            continue
        if sum(ch.isalpha() for ch in text) < max(1, min_letters):
            continue
        filtered.append(text)
    return filtered


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


def _apply_optional_deepseek_line_filter(
    lines: list[str],
    speech_text: str,
    settings: dict,
) -> tuple[list[str], Optional[dict[str, Any]]]:
    """When deepseek.filter_board_lines is true, ask the API which OCR lines to keep."""
    out_lines = list(lines)
    meta = run_deepseek_filter_board_lines(out_lines, speech_text, settings)
    if not meta.get("enabled"):
        return out_lines, None
    kept = meta.get("kept_lines")
    if isinstance(kept, list):
        return [str(x) for x in kept if str(x).strip()], meta
    return out_lines, meta


def _dedupe_subsumed_lines(lines: list[str], *, min_len: int = 8) -> list[str]:
    items = [
        (s.strip(), _canonicalize_line_for_dedupe(s))
        for s in lines
        if s and s.strip() and _canonicalize_line_for_dedupe(s)
    ]
    if len(items) < 2:
        return [text for text, _ in items]
    drop: set[int] = set()
    for i, (_, item_cf) in enumerate(items):
        if len(item_cf) < min_len:
            continue
        for j, (_, other_cf) in enumerate(items):
            if i == j or len(other_cf) <= len(item_cf):
                continue
            if item_cf in other_cf or _lines_near_duplicate(item_cf, other_cf):
                drop.add(i)
                break
    return [items[i][0] for i in range(len(items)) if i not in drop]


def _normalize_board_lines(lines: list[str], *, min_substring_len: int) -> list[str]:
    unique = _merge_unique_lines([], lines)
    return _dedupe_subsumed_lines(unique, min_len=max(4, min_substring_len))


def _measure_text_presence(
    ocr_mask: np.ndarray,
    *,
    min_ink_ratio: float,
    min_line_spans: int,
    min_components: int,
    max_largest_component_ratio: float,
    max_single_span_height_ratio: float,
) -> dict[str, Any]:
    if ocr_mask is None or ocr_mask.size == 0:
        return {
            "likely_text": False,
            "reason": "empty_mask",
            "ink_ratio": 0.0,
            "line_span_count": 0,
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "max_line_span_height_ratio": 0.0,
        }

    binary = (ocr_mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    area = float(max(1, h * w))
    ink_ratio = float(binary.sum() / area)
    line_spans = segment_text_lines((binary * 255).astype(np.uint8))
    line_span_count = len(line_spans)
    max_line_span_height_ratio = 0.0
    if line_spans:
        max_line_span_height_ratio = max((y1 - y0) / float(max(1, h)) for y0, y1 in line_spans)

    component_count = 0
    largest_component_ratio = 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        component_count = int(component_areas.size)
        largest_component_ratio = float(np.max(component_areas) / area)

    likely_text = False
    reason = "no_text_like_pattern"
    if ink_ratio < max(0.0, min_ink_ratio):
        reason = "too_little_ink"
    elif line_span_count >= max(1, min_line_spans):
        likely_text = True
        reason = "multiple_line_spans"
    elif (
        line_span_count == 1
        and component_count >= max(1, min_components)
        and largest_component_ratio <= max(0.0, max_largest_component_ratio)
        and max_line_span_height_ratio <= max(0.0, max_single_span_height_ratio)
    ):
        likely_text = True
        reason = "single_text_like_span"
    elif (
        line_span_count == 0
        and component_count >= max(6, min_components * 2)
        and largest_component_ratio <= max(0.0, max_largest_component_ratio * 0.65)
    ):
        likely_text = True
        reason = "fragmented_text_components"
    elif largest_component_ratio > max(0.0, max_largest_component_ratio):
        reason = "dominant_non_text_blob"
    elif component_count < max(1, min_components):
        reason = "too_few_components"
    elif max_line_span_height_ratio > max(0.0, max_single_span_height_ratio):
        reason = "single_tall_blob"

    return {
        "likely_text": likely_text,
        "reason": reason,
        "ink_ratio": round(ink_ratio, 5),
        "line_span_count": line_span_count,
        "component_count": component_count,
        "largest_component_ratio": round(largest_component_ratio, 5),
        "max_line_span_height_ratio": round(max_line_span_height_ratio, 5),
    }


def _downscale_max_width(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr
    scale = max_width / float(w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _diagnose_video_ocr(
    keyframe_results: List[dict],
    *,
    harvest_ran: bool,
    harvest_full_frame: bool,
    line_count_before_substring_dedupe: int,
    line_count_after_substring_dedupe: int,
) -> dict[str, Any]:
    if not keyframe_results:
        return {
            "primary_issue_guess": "no_keyframes",
            "summary_en": "No keyframes were produced.",
            "summary_zh": "没有成功抽取到关键帧。",
            "stats": {},
        }

    roi_counts: dict[str, int] = {}
    printed_full_frame_count = 0
    total_lines = 0
    for item in keyframe_results:
        roi_method = str(item.get("roi_method") or "unknown")
        roi_counts[roi_method] = roi_counts.get(roi_method, 0) + 1
        if item.get("ocr_source") == "printed_full_frame":
            printed_full_frame_count += 1
        total_lines += len(item.get("board_texts") or [])

    keyframe_count = len(keyframe_results)
    avg_lines = total_lines / float(max(1, keyframe_count))
    fullish_count = (
        roi_counts.get("full_frame", 0)
        + roi_counts.get("full_frame_video_fallback", 0)
        + roi_counts.get("heuristic", 0)
    )

    if fullish_count >= keyframe_count * 0.45:
        primary_issue = "localization"
        summary_en = "Many frames fell back to full-frame or heuristic ROI; board localization is likely the main problem."
        summary_zh = "很多关键帧落到了全帧或启发式 ROI，主要问题更像是黑板定位不准。"
    elif printed_full_frame_count >= keyframe_count * 0.35:
        primary_issue = "mixed_fullframe_ocr"
        summary_en = "Printed full-frame OCR fallback fired often, so slide/UI text may be mixed into board OCR."
        summary_zh = "整帧 printed OCR 回退触发较多，可能把投影或界面文字混进了板书 OCR。"
    elif harvest_ran and harvest_full_frame and line_count_before_substring_dedupe > max(
        80, line_count_after_substring_dedupe * 2
    ):
        primary_issue = "harvest_full_frame_noise"
        summary_en = "Timeline harvest likely added noisy full-frame text. Tighten or disable full-frame harvest OCR."
        summary_zh = "时间线补采样可能引入了较多整帧噪声文本，建议收紧或关闭整帧 harvest OCR。"
    elif avg_lines < 0.75:
        primary_issue = "recognition_or_sparse_board"
        summary_en = "ROI seems reasonable, but OCR returns few lines; recognition quality or sparse content may be the bottleneck."
        summary_zh = "ROI 看起来还算正常，但 OCR 行数偏少，瓶颈更像是识别质量或画面内容过少。"
    else:
        primary_issue = "healthy_or_merge_noise"
        summary_en = "The video OCR path looks broadly healthy; remaining issues are likely OCR noise or merge artifacts."
        summary_zh = "视频 OCR 主链路整体正常，剩余问题更像是 OCR 噪声或多帧合并伪影。"

    return {
        "primary_issue_guess": primary_issue,
        "summary_en": summary_en,
        "summary_zh": summary_zh,
        "stats": {
            "keyframe_count": keyframe_count,
            "roi_method_counts": roi_counts,
            "keyframes_with_printed_fullframe_fallback": printed_full_frame_count,
            "avg_ocr_lines_per_keyframe": round(avg_lines, 4),
            "text_harvest_ran": harvest_ran,
            "text_harvest_full_frame_printed": harvest_full_frame,
            "lines_before_substring_dedupe": line_count_before_substring_dedupe,
            "lines_after_substring_dedupe": line_count_after_substring_dedupe,
        },
    }


def _video_text_harvest_pass(
    video_path: str,
    settings: dict,
    *,
    every_sec: float,
    max_scans: int,
    full_frame_printed: bool,
    full_frame_max_width: int,
    printed_model: str,
    trocr_device: Optional[str] = None,
) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    frame_step = max(1, int(round(max(0.2, float(every_sec)) * fps)))

    merged: list[str] = []
    frame_index = 0
    scans = 0
    try:
        while scans < max(1, int(max_scans)):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            try:
                board_reading = run_module_a(frame_bgr, settings)
                if board_reading.get("error"):
                    logger.debug("Harvest run_module_a error: %s", board_reading.get("error"))
                merged = _merge_unique_lines(merged, board_reading.get("texts") or [])
                if full_frame_printed:
                    full_frame = _downscale_max_width(frame_bgr, full_frame_max_width)
                    try:
                        full_lines = recognize_text_lines_in_image(
                            full_frame,
                            **_recognize_line_image_kwargs(settings, trocr_model_name=printed_model),
                        )
                        merged = _merge_unique_lines(merged, full_lines)
                    except Exception as e:
                        logger.debug("Harvest full-frame printed OCR skipped: %s", e)
            except Exception as e:
                logger.warning("Harvest frame failed: %s", e)

            scans += 1
            frame_index += 1
    finally:
        cap.release()

    logger.info("Video text harvest: %s unique lines from %s sampled frames", len(merged), scans)
    return merged


def _resolve_video_debug_dir(
    video_path: str,
    pdf_output: str,
    settings: dict,
) -> Optional[Path]:
    video_cfg = settings.get("video", {})
    if not bool(video_cfg.get("debug_enabled", True)):
        return None
    debug_dir = Path(pdf_output).resolve().parent / f"{Path(video_path).stem}_video_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _write_debug_image(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save debug image: {path}")
    return str(path.resolve())


def _save_video_debug_assets(
    *,
    debug_dir: Optional[Path],
    keyframe_idx: int,
    frame_bgr: np.ndarray,
    roi_tuple: tuple[int, int, int, int],
    ocr_input_bgr: np.ndarray,
    ocr_mask: np.ndarray,
) -> Dict[str, str]:
    if debug_dir is None:
        return {}
    prefix = f"frame_{keyframe_idx:03d}"
    overlay = frame_bgr.copy()
    x1, y1, x2, y2 = map(int, roi_tuple)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
    return {
        "full_frame": _write_debug_image(debug_dir / f"{prefix}_full.png", frame_bgr),
        "roi_overlay": _write_debug_image(debug_dir / f"{prefix}_roi_overlay.png", overlay),
        "ocr_input": _write_debug_image(debug_dir / f"{prefix}_ocr_input.png", ocr_input_bgr),
        "ocr_mask": _write_debug_image(debug_dir / f"{prefix}_ocr_mask.png", ocr_mask),
    }


def _find_reusable_ocr_result(
    cache: list[dict[str, Any]],
    *,
    signature: np.ndarray,
    timestamp_sec: float,
    similarity_threshold: float,
    min_interval_sec: float,
) -> Optional[dict[str, Any]]:
    best_match: Optional[dict[str, Any]] = None
    for entry in reversed(cache):
        time_gap = abs(float(timestamp_sec) - float(entry.get("timestamp_sec") or 0.0))
        if min_interval_sec > 0 and time_gap > min_interval_sec:
            continue
        similarity = content_signature_similarity(signature, entry.get("signature"))
        if similarity < similarity_threshold:
            continue
        if best_match is None or similarity > float(best_match["similarity"]):
            best_match = {
                "frame_index": entry.get("frame_index"),
                "timestamp_sec": entry.get("timestamp_sec"),
                "texts": list(entry.get("texts") or []),
                "ocr_source": entry.get("ocr_source"),
                "similarity": round(similarity, 4),
            }
    return best_match


def _append_ocr_cache(
    cache: list[dict[str, Any]],
    *,
    signature: np.ndarray,
    frame_index: Optional[int],
    timestamp_sec: float,
    texts: list[str],
    ocr_source: str,
    cache_size: int,
) -> None:
    cache.append(
        {
            "signature": signature.copy(),
            "frame_index": frame_index,
            "timestamp_sec": float(timestamp_sec),
            "texts": list(texts),
            "ocr_source": ocr_source,
        }
    )
    overflow = len(cache) - max(1, cache_size)
    if overflow > 0:
        del cache[:overflow]


def load_bgr_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def extract_audio_ffmpeg(video_path: str, wav_out: str, overwrite: bool = True) -> str:
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
    problems: Dict[str, str] = {}

    board_reading = run_module_a(frame_bgr, settings)
    if board_reading.get("error"):
        problems["module_a"] = board_reading["error"]

    handwriting_clarity = run_module_b(frame_bgr, settings)
    if handwriting_clarity.get("error"):
        problems["module_b"] = handwriting_clarity["error"]

    spoken_transcript = run_module_c(audio_path, settings)
    if spoken_transcript.get("error"):
        problems["module_c"] = spoken_transcript["error"]

    lines_on_board = list(board_reading.get("texts") or [])
    speech_text = spoken_transcript.get("speech_text") or ""
    lines_on_board, deepseek_line_filter = _apply_optional_deepseek_line_filter(
        lines_on_board,
        speech_text,
        settings,
    )
    lines_on_board = _normalize_board_lines(
        lines_on_board,
        min_substring_len=max(
            4,
            int(
                ((settings.get("video") if isinstance(settings.get("video"), dict) else {}) or {}).get(
                    "merge_substring_min_len",
                    8,
                )
            ),
        ),
    )
    board_as_paragraph = "\n".join(lines_on_board)

    lesson_alignment = run_module_d(board_as_paragraph, speech_text, settings)
    if lesson_alignment.get("error"):
        problems["module_d"] = lesson_alignment["error"]
    deepseek_alignment = run_module_d_deepseek(board_as_paragraph, speech_text, settings)
    if deepseek_alignment.get("enabled") and deepseek_alignment.get("error"):
        problems["module_d_deepseek"] = str(deepseek_alignment["error"])

    clarity_numbers = handwriting_clarity.get("clarity_result") or {}
    alignment_summary = lesson_alignment.get("alignment")

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

    if deepseek_line_filter is not None and deepseek_line_filter.get("error"):
        problems["module_deepseek_line_filter"] = str(deepseek_line_filter["error"])

    out: Dict[str, Any] = {
        "board_texts": lines_on_board,
        "board_roi": board_reading.get("roi"),
        "roi_method": board_reading.get("roi_method"),
        "clarity": clarity_numbers,
        "speech_text": speech_text,
        "speech_segments": spoken_transcript.get("speech_segments") or [],
        "alignment": alignment_summary,
        "deepseek_alignment": deepseek_alignment,
        "pdf_path": pdf_bundle.get("pdf_path"),
        "errors": problems,
    }
    if deepseek_line_filter is not None:
        out["deepseek_board_line_filter"] = deepseek_line_filter
    return out


def run_from_image_and_audio_files(
    image_path: str,
    audio_path: str,
    config: Optional[dict] = None,
    pdf_output: str = "output/teaching_feedback.pdf",
) -> Dict[str, Any]:
    frame_bgr = load_bgr_image(image_path)
    return run_from_frame_and_audio(frame_bgr, audio_path, config=config, pdf_output=pdf_output)


def run_from_video_file(
    video_path: str,
    config: Optional[dict] = None,
    pdf_output: str = "output/teaching_feedback.pdf",
) -> Dict[str, Any]:
    settings = _apply_fast_video_settings(config or {})
    problems: Dict[str, str] = {}
    trocr_opts = settings.get("trocr", {}) if isinstance(settings.get("trocr"), dict) else {}
    video_cfg = settings.get("video") if isinstance(settings.get("video"), dict) else {}
    printed_fallback_model = str(trocr_opts.get("printed_model_name", TROCR_PRINTED))
    handwriting_model = str(trocr_opts.get("model_name", TROCR_DEFAULT))
    video_prefer_printed = bool(trocr_opts.get("video_prefer_printed_model", False))
    use_video_printed_fallback = bool(trocr_opts.get("video_enable_printed_fallback", True))
    use_video_hw_fallback = bool(trocr_opts.get("video_enable_handwriting_fallback", True))
    trocr_device = parse_trocr_device_option(trocr_opts.get("device", "auto"))
    debug_dir = _resolve_video_debug_dir(video_path, pdf_output, settings)
    deepseek_line_filter: Optional[dict[str, Any]] = None
    dedupe_enabled = bool(video_cfg.get("dedupe_enabled", True))
    dedupe_similarity_threshold = max(
        0.0,
        min(1.0, float(video_cfg.get("dedupe_similarity_threshold", 0.985))),
    )
    dedupe_min_interval_sec = max(0.0, float(video_cfg.get("dedupe_min_interval_sec", 8.0)))
    dedupe_cache_size = max(1, int(video_cfg.get("dedupe_cache_size", 12)))
    text_presence_enabled = bool(video_cfg.get("text_presence_enabled", True))
    text_presence_min_ink_ratio = max(0.0, float(video_cfg.get("text_presence_min_ink_ratio", 0.002)))
    text_presence_min_line_spans = max(1, int(video_cfg.get("text_presence_min_line_spans", 2)))
    text_presence_min_components = max(1, int(video_cfg.get("text_presence_min_components", 8)))
    text_presence_max_component_ratio = max(
        0.0,
        min(1.0, float(video_cfg.get("text_presence_max_component_ratio", 0.14))),
    )
    text_presence_max_single_span_height_ratio = max(
        0.0,
        min(1.0, float(video_cfg.get("text_presence_max_single_span_height_ratio", 0.42))),
    )
    ocr_result_cache: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="blackboard_video_") as tmp_dir:
        extracted_audio = extract_audio_ffmpeg(video_path, str(Path(tmp_dir) / "audio.wav"))

        try:
            keyframes = extract_blackboard_keyframes(video_path, settings)
        except Exception as e:
            logger.exception("extract_blackboard_keyframes")
            raise RuntimeError(f"Video keyframe extraction failed: {e}") from e

        aggregated_board_lines: list[str] = []
        best_board_lines: list[str] = []
        best_clarity: Dict[str, Any] = {}
        best_roi = None
        best_roi_method = None
        keyframe_results: list[dict[str, Any]] = []
        debug_metadata: list[dict[str, Any]] = []
        harvest_ran = False
        harvest_full_frame_cfg = False

        for keyframe_idx, item in enumerate(keyframes, start=1):
            frame_bgr = item["frame_bgr"]
            frame_index = item.get("frame_index")
            frame_timestamp_sec = float(item.get("timestamp_sec") or 0.0)
            precomputed_roi_box = coerce_roi_box(
                item.get("roi") or (0, 0, frame_bgr.shape[1], frame_bgr.shape[0]),
                image_shape=frame_bgr.shape,
            )
            ocr_input_bgr, _, ocr_mask = prepare_ocr_inputs(frame_bgr, precomputed_roi_box)
            content_signature = build_content_signature(ocr_mask)
            text_presence = _measure_text_presence(
                ocr_mask,
                min_ink_ratio=text_presence_min_ink_ratio,
                min_line_spans=text_presence_min_line_spans,
                min_components=text_presence_min_components,
                max_largest_component_ratio=text_presence_max_component_ratio,
                max_single_span_height_ratio=text_presence_max_single_span_height_ratio,
            )
            dedupe_match = (
                _find_reusable_ocr_result(
                    ocr_result_cache,
                    signature=content_signature,
                    timestamp_sec=frame_timestamp_sec,
                    similarity_threshold=dedupe_similarity_threshold,
                    min_interval_sec=dedupe_min_interval_sec,
                )
                if dedupe_enabled
                else None
            )
            skip_for_text_presence = text_presence_enabled and not bool(text_presence.get("likely_text"))
            if video_prefer_printed:
                trocr_video = dict(trocr_opts)
                trocr_video["model_name"] = printed_fallback_model
                module_a_settings = {**settings, "trocr": trocr_video}
            else:
                module_a_settings = settings
            if skip_for_text_presence:
                board_reading = {
                    "texts": [],
                    "roi": precomputed_roi_box.as_tuple(),
                    "roi_method": str(item.get("roi_method") or "precomputed_roi"),
                    "error": None,
                }
                texts = []
                ocr_source = f"skipped_{text_presence.get('reason') or 'no_text_like_content'}"
            elif dedupe_match is not None:
                board_reading = {
                    "texts": list(dedupe_match.get("texts") or []),
                    "roi": precomputed_roi_box.as_tuple(),
                    "roi_method": str(item.get("roi_method") or "precomputed_roi"),
                    "error": None,
                }
                texts = list(board_reading["texts"])
                ocr_source = str(dedupe_match.get("ocr_source") or "reused_ocr")
            else:
                board_reading = run_module_a(
                    frame_bgr,
                    module_a_settings,
                    roi_override=item.get("roi"),
                    roi_method_override=str(item.get("roi_method") or "keyframe_selected"),
                )
                if board_reading.get("error") and "module_a" not in problems:
                    problems["module_a"] = board_reading["error"]

                texts = list(board_reading.get("texts") or [])
                ocr_source = "none"
                if video_prefer_printed:
                    primary_eng = normalize_ocr_engine_name(trocr_video.get("ocr_engine", "trocr"))
                    if texts and not _texts_look_low_value(texts):
                        ocr_source = {
                            "easyocr": "roi_easyocr",
                            "paddleocr": "roi_paddleocr",
                        }.get(primary_eng, "roi_printed")
                    elif texts:
                        ocr_source = {
                            "easyocr": "roi_easyocr_weak",
                            "paddleocr": "roi_paddleocr_weak",
                        }.get(primary_eng, "roi_printed_weak")
                    if use_video_hw_fallback and _texts_look_low_value(texts):
                        try:
                            hw_texts = recognize_text_lines_in_image(
                                frame_bgr,
                                **_recognize_line_image_kwargs(
                                    settings,
                                    trocr_model_name=handwriting_model,
                                    ocr_engine_override="trocr",
                                ),
                            )
                            if hw_texts and not _texts_look_low_value(hw_texts):
                                texts = hw_texts
                                ocr_source = "handwriting_full_frame"
                        except Exception as e:
                            logger.warning("Handwriting OCR fallback failed on video keyframe: %s", e)
                elif use_video_printed_fallback and _texts_look_low_value(texts):
                    try:
                        fallback_texts = recognize_text_lines_in_image(
                            frame_bgr,
                            **_recognize_line_image_kwargs(
                                settings,
                                trocr_model_name=printed_fallback_model,
                                ocr_engine_override="trocr",
                            ),
                        )
                        if fallback_texts and not _texts_look_low_value(fallback_texts):
                            texts = fallback_texts
                            ocr_source = "printed_full_frame"
                    except Exception as e:
                        logger.warning("Printed OCR fallback failed on video keyframe: %s", e)
                if ocr_source == "none":
                    if texts and not _texts_look_low_value(texts):
                        ocr_source = "roi_handwriting"
                    elif texts:
                        ocr_source = "roi_handwriting_weak"
                if texts and not _texts_look_low_value(texts):
                    _append_ocr_cache(
                        ocr_result_cache,
                        signature=content_signature,
                        frame_index=frame_index,
                        timestamp_sec=frame_timestamp_sec,
                        texts=texts,
                        ocr_source=ocr_source,
                        cache_size=dedupe_cache_size,
                    )

            aggregated_board_lines = _merge_unique_lines(aggregated_board_lines, texts)

            clarity_result = item.get("clarity_result") or {}
            if clarity_result and clarity_result.get("score", 0.0) >= best_clarity.get("score", 0.0):
                best_clarity = clarity_result
                best_board_lines = list(texts)
                best_roi = board_reading.get("roi") or item.get("roi")
                best_roi_method = board_reading.get("roi_method") or item.get("roi_method")

            roi_box = coerce_roi_box(
                board_reading.get("roi") or precomputed_roi_box.as_tuple(),
                image_shape=frame_bgr.shape,
            )
            debug_paths = _save_video_debug_assets(
                debug_dir=debug_dir,
                keyframe_idx=keyframe_idx,
                frame_bgr=frame_bgr,
                roi_tuple=roi_box.as_tuple(),
                ocr_input_bgr=ocr_input_bgr,
                ocr_mask=ocr_mask,
            )

            keyframe_result = {
                "frame_index": item.get("frame_index"),
                "timestamp_sec": item.get("timestamp_sec"),
                "change_ratio": item.get("change_ratio"),
                "clarity": clarity_result,
                "roi": roi_box.as_tuple(),
                "roi_method": board_reading.get("roi_method") or item.get("roi_method"),
                "board_texts": texts,
                "ocr_source": ocr_source,
                "text_presence_enabled": text_presence_enabled,
                "text_presence_likely_text": text_presence.get("likely_text"),
                "text_presence_reason": text_presence.get("reason"),
                "text_presence_ink_ratio": text_presence.get("ink_ratio"),
                "text_presence_line_spans": text_presence.get("line_span_count"),
                "text_presence_component_count": text_presence.get("component_count"),
                "text_presence_largest_component_ratio": text_presence.get("largest_component_ratio"),
                "dedupe_enabled": dedupe_enabled,
                "dedupe_is_duplicate": dedupe_match is not None,
                "ocr_skipped": skip_for_text_presence or dedupe_match is not None,
                "ocr_skip_reason": text_presence.get("reason") if skip_for_text_presence else (
                    "reused_previous_ocr" if dedupe_match is not None else None
                ),
                "dedupe_similarity": dedupe_match.get("similarity") if dedupe_match is not None else None,
                "reused_from_frame_index": dedupe_match.get("frame_index") if dedupe_match is not None else None,
                "reused_from_timestamp_sec": dedupe_match.get("timestamp_sec") if dedupe_match is not None else None,
                "reused_from_ocr_source": dedupe_match.get("ocr_source") if dedupe_match is not None else None,
                "debug_paths": debug_paths or None,
            }
            keyframe_results.append(keyframe_result)
            debug_metadata.append(keyframe_result)

        harvest_every = float(video_cfg.get("text_harvest_every_sec", 0) or 0)
        if harvest_every > 0:
            try:
                extra_lines = _video_text_harvest_pass(
                    video_path,
                    settings,
                    every_sec=harvest_every,
                    max_scans=max(1, int(video_cfg.get("text_harvest_max_scans", 240))),
                    full_frame_printed=bool(video_cfg.get("text_harvest_full_frame_printed", True)),
                    full_frame_max_width=max(480, int(video_cfg.get("full_frame_ocr_max_width", 1280))),
                    printed_model=printed_fallback_model,
                    trocr_device=trocr_device,
                )
                aggregated_board_lines = _merge_unique_lines(aggregated_board_lines, extra_lines)
                harvest_ran = True
                harvest_full_frame_cfg = bool(video_cfg.get("text_harvest_full_frame_printed", True))
            except Exception as e:
                logger.warning("Video text harvest pass failed: %s", e)

        line_count_before_substring_dedupe = len(aggregated_board_lines)
        merged_deduped = _normalize_board_lines(
            aggregated_board_lines,
            min_substring_len=max(4, int(video_cfg.get("merge_substring_min_len", 8))),
        )
        line_count_after_substring_dedupe = len(merged_deduped)

        merge_mode = str(video_cfg.get("board_text_mode", "merged") or "merged").lower().strip()
        if merge_mode in {"best", "best_clarity", "single", "single_frame"}:
            board_lines_primary = list(best_board_lines) if best_board_lines else list(merged_deduped)
        else:
            board_lines_primary = merged_deduped if merged_deduped else list(best_board_lines)

        if bool(video_cfg.get("filter_noise_board_lines", False)):
            board_lines_primary = _filter_noise_board_lines(
                board_lines_primary,
                min_chars=max(4, int(video_cfg.get("noise_line_min_chars", 14))),
                min_letters=max(2, int(video_cfg.get("noise_line_min_letters", 5))),
            )

        spoken_transcript = run_module_c(extracted_audio, settings)
        if spoken_transcript.get("error"):
            problems["module_c"] = spoken_transcript["error"]

        speech_text = spoken_transcript.get("speech_text") or ""
        board_lines_primary, deepseek_line_filter = _apply_optional_deepseek_line_filter(
            board_lines_primary,
            speech_text,
            settings,
        )
        board_lines_primary = _normalize_board_lines(
            board_lines_primary,
            min_substring_len=max(4, int(video_cfg.get("merge_substring_min_len", 8))),
        )
        board_as_paragraph = "\n".join(board_lines_primary)

        lesson_alignment = run_module_d(board_as_paragraph, speech_text, settings)
        if lesson_alignment.get("error"):
            problems["module_d"] = lesson_alignment["error"]
        alignment_summary = lesson_alignment.get("alignment")
        deepseek_alignment = run_module_d_deepseek(board_as_paragraph, speech_text, settings)
        if deepseek_alignment.get("enabled") and deepseek_alignment.get("error"):
            problems["module_d_deepseek"] = str(deepseek_alignment["error"])

        bundle_for_pdf = {
            "board_lines": board_lines_primary,
            "clarity": best_clarity,
            "alignment": alignment_summary,
            "speech_text": speech_text,
            "module_errors": problems or None,
        }
        pdf_bundle = run_module_e(pdf_output, bundle_for_pdf)
        if pdf_bundle.get("error"):
            problems["module_e"] = pdf_bundle["error"]

        if deepseek_line_filter is not None and deepseek_line_filter.get("error"):
            problems["module_deepseek_line_filter"] = str(deepseek_line_filter["error"])

    metadata_path = None
    if debug_dir is not None:
        metadata_path = debug_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "video_path": str(Path(video_path).resolve()),
                    "pdf_output": str(Path(pdf_output).resolve()),
                    "debug_dir": str(debug_dir.resolve()),
                    "video_fast_mode": bool(video_cfg.get("fast_mode")),
                    "keyframes": debug_metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    result: Dict[str, Any] = {
        "input_mode": "video",
        "video_fast_mode": bool(video_cfg.get("fast_mode")),
        "board_texts": board_lines_primary,
        "board_roi": best_roi,
        "roi_method": best_roi_method,
        "clarity": best_clarity,
        "speech_text": spoken_transcript.get("speech_text"),
        "speech_segments": spoken_transcript.get("speech_segments") or [],
        "alignment": alignment_summary,
        "deepseek_alignment": deepseek_alignment,
        "pdf_path": pdf_bundle.get("pdf_path"),
        "errors": problems,
        "video_keyframes": keyframe_results,
        "video_debug_dir": str(debug_dir.resolve()) if debug_dir is not None else None,
        "video_debug_metadata": str(metadata_path.resolve()) if metadata_path is not None else None,
    }
    if bool(video_cfg.get("ocr_diagnostics", True)):
        result["ocr_diagnostics"] = _diagnose_video_ocr(
            keyframe_results,
            harvest_ran=harvest_ran,
            harvest_full_frame=harvest_full_frame_cfg,
            line_count_before_substring_dedupe=line_count_before_substring_dedupe,
            line_count_after_substring_dedupe=line_count_after_substring_dedupe,
        )
    if deepseek_line_filter is not None:
        result["deepseek_board_line_filter"] = deepseek_line_filter
    return result
