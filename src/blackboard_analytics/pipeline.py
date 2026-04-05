# One still frame of the board + one audio clip -> structured feedback + PDF.
# Order matches how a person would look at a lesson: what is written, how legible it is,
# what was said, whether talk matches the board, then a printable summary.

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from blackboard_analytics.module_a_blackboard_ocr import (
    TROCR_DEFAULT,
    TROCR_PRINTED,
    TrOCRHandwritingEngine,
    coerce_roi_box,
    prepare_ocr_inputs,
    parse_trocr_device_option,
    recognize_text_lines_in_image,
    run_module_a,
)
from blackboard_analytics.module_b_clarity import run_module_b
from blackboard_analytics.module_c_whisper import run_module_c
from blackboard_analytics.module_d_semantic import run_module_d
from blackboard_analytics.module_e_report import run_module_e
from blackboard_analytics.module_video_keyframes import extract_blackboard_keyframes

logger = logging.getLogger(__name__)

# When video.fast_mode (or env BLACKBOARD_VIDEO_FAST=1): fewer TrOCR frames, no text_harvest second pass.
_FAST_VIDEO_PRESET: dict[str, Any] = {
    "max_keyframes": 6,
    "yolo_monitor_stride_sec": 1.2,
    "yolo_monitor_max_pool": 48,
    "text_harvest_every_sec": 0,
    "text_harvest_max_scans": 0,
    "text_harvest_full_frame_printed": False,
    "min_keyframe_score": 40.0,
}


def _apply_fast_video_settings(settings: dict) -> dict:
    base: dict[str, Any] = dict(settings)
    vc = dict(base.get("video") or {})
    env_fast = os.environ.get("BLACKBOARD_VIDEO_FAST", "").strip().lower() in ("1", "true", "yes", "on")
    if not bool(vc.get("fast_mode")) and not env_fast:
        base["video"] = vc
        return base
    merged = {**vc, **_FAST_VIDEO_PRESET}
    merged["fast_mode"] = True
    base["video"] = merged
    if bool(merged.get("fast_whisper_tiny")):
        wh = dict(base.get("whisper") or {})
        wh["model_size"] = "tiny"
        base["whisper"] = wh
        logger.info("video.fast_whisper_tiny: Whisper model_size=tiny for faster transcription")
    logger.info(
        "video.fast_mode on: max_keyframes=%s, stride_sec=%s, text_harvest off",
        merged.get("max_keyframes"),
        merged.get("yolo_monitor_stride_sec"),
    )
    return base


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


def _filter_noise_board_lines(
    lines: list[str],
    *,
    min_chars: int,
    min_letters: int,
) -> list[str]:
    """Drop very short or mostly non-letter lines (UI crumbs, OCR specks)."""
    out: list[str] = []
    for line in lines:
        s = (line or "").strip()
        if len(s) < max(1, min_chars):
            continue
        letters = sum(c.isalpha() for c in s)
        if letters < max(1, min_letters):
            continue
        out.append(s)
    return out


def _diagnose_video_ocr(
    keyframe_results: List[dict],
    *,
    harvest_ran: bool,
    harvest_full_frame: bool,
    line_count_before_substring_dedupe: int,
    line_count_after_substring_dedupe: int,
) -> dict[str, Any]:
    """
    Heuristic: tell whether failures are more likely localization (wrong ROI) vs recognition vs harvest noise.
    """
    n = len(keyframe_results)
    if n == 0:
        return {
            "primary_issue_guess": "no_keyframes",
            "summary_zh": "沒有取得任何關鍵幀，請檢查影片與 keyframe 設定。",
            "summary_en": "No keyframes were produced.",
            "stats": {},
        }

    roi_counts: dict[str, int] = {}
    printed_ff = 0
    total_lines = 0
    for k in keyframe_results:
        rm = str(k.get("roi_method") or "unknown")
        roi_counts[rm] = roi_counts.get(rm, 0) + 1
        if k.get("ocr_source") == "printed_full_frame":
            printed_ff += 1
        total_lines += len(k.get("board_texts") or [])

    avg_lines = total_lines / float(n)
    fullish = (
        roi_counts.get("full_frame", 0)
        + roi_counts.get("full_frame_video_fallback", 0)
        + roi_counts.get("heuristic", 0)
    )

    primary = "uncertain"
    zh = ""
    en = ""

    if fullish >= n * 0.45:
        primary = "localization"
        zh = (
            "超過約四成五的影格使用「整張圖」或輪廓後備 ROI，代表板書區域偵測（YOLO-World / 自訓權重）"
            "經常沒框到真正的黑板；此時 OCR 讀到多是無關畫面，問題主因在「偵測/定位」而非讀字模型本身。"
        )
        en = (
            "Many frames use full-frame or heuristic ROI: the board detector likely missed the real board; "
            "OCR is reading the wrong region—primarily a localization/detection issue."
        )
    elif printed_ff >= n * 0.35:
        primary = "mixed_fullframe_ocr"
        zh = (
            "多數影格走了「整張印刷體 OCR 後備」（手寫 ROI 結果被判太差時）。"
            "這會把瀏覽器、字幕、按鈕等畫面字也讀進來，看起來像一堆不相干文字；請關閉或限縮該後備，或改善 ROI。"
        )
        en = (
            "Printed full-frame OCR fallback fired on many frames, which pulls UI/chrome text into results—"
            "tighten ROI or disable/reduce this fallback."
        )
    elif harvest_ran and harvest_full_frame and line_count_before_substring_dedupe > 80:
        primary = "harvest_full_frame_noise"
        zh = (
            "已啟用全片文字採樣且含「整張印刷 OCR」，合併後行數很多時，雜訊常來自投影片/螢幕 UI，而非板書手寫。"
            "可將 text_harvest_full_frame_printed 設為 false 或加大採樣間隔做對照。"
        )
        en = "Text harvest with full-frame printed OCR often adds slide/UI lines; try turning off full-frame harvest OCR."
    elif roi_counts.get("yolo_world", 0) > n * 0.5 and avg_lines < 0.65:
        primary = "recognition_or_sparse_board"
        zh = (
            "偵測多為 yolo_world，但每幀 OCR 行數極少：可能是 TrOCR 讀手寫失敗、字太糊，或板上確實幾乎沒字。"
            "若畫面上明明有字，主因較像「辨識」；若畫面本來就空，屬正常。"
        )
        en = (
            "YOLO-World ROI looks active but few OCR lines per frame: likely weak handwriting recognition or an empty board."
        )
    else:
        primary = "recognition_or_merge_artifacts"
        zh = (
            "ROI 來源大致正常時，若內容仍怪或殘留短句，多半是 TrOCR 幻覺/錯讀，或多幀合併帶入零星雜訊。"
            "可試 filter_noise_board_lines、關閉全片採樣對照，或改用手寫微調模型。"
        )
        en = "With reasonable ROI, garbled output is usually OCR errors or merge noise; try line filters or better OCR/finetuning."

    return {
        "primary_issue_guess": primary,
        "summary_zh": zh,
        "summary_en": en,
        "stats": {
            "keyframe_count": n,
            "roi_method_counts": roi_counts,
            "keyframes_with_printed_fullframe_fallback": printed_ff,
            "avg_ocr_lines_per_keyframe": round(avg_lines, 4),
            "text_harvest_ran": harvest_ran,
            "text_harvest_full_frame_printed": harvest_full_frame,
            "lines_before_substring_dedupe": line_count_before_substring_dedupe,
            "lines_after_substring_dedupe": line_count_after_substring_dedupe,
        },
        "how_to_interpret": {
            "localization": "偵測問題：框錯區域 → 先修 YOLO 提示詞/權重或 min_roi_area_ratio。",
            "recognition": "辨識問題：框對了但讀錯 → 換/微調 TrOCR 或改前處理。",
            "harvest_full_frame_noise": "全片採樣帶入 UI → 關閉 text_harvest_full_frame_printed 或縮短採樣。",
        },
    }


def _dedupe_subsumed_lines(lines: list[str], *, min_len: int = 8) -> list[str]:
    """
    Drop lines that are strict substrings of another longer line (case-insensitive),
    to collapse OCR fragments that repeat across frames. Short lines (< min_len) are kept.
    """
    items = [(s.strip(), s.strip().casefold()) for s in lines if s and s.strip()]
    if len(items) < 2:
        return [s for s, _ in items]
    drop: set[int] = set()
    for i, (s, cf) in enumerate(items):
        if len(cf) < min_len:
            continue
        for j, (_, cf2) in enumerate(items):
            if i == j or len(cf2) <= len(cf):
                continue
            if cf in cf2:
                drop.add(i)
                break
    return [items[i][0] for i in range(len(items)) if i not in drop]


def _downscale_max_width(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image_bgr is None or image_bgr.size == 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    if w <= max_width:
        return image_bgr
    scale = max_width / float(w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


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
    """
    Second pass over the whole timeline: board ROI OCR (via run_module_a) on a fixed stride,
    optional printed OCR on a downscaled full frame for slides / on-screen text.
    """
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
                    logger.debug("Harvest run_module_a: %s", board_reading.get("error"))
                texts = board_reading.get("texts") or []
                merged = _merge_unique_lines(merged, texts)
                if full_frame_printed:
                    small = _downscale_max_width(frame_bgr, full_frame_max_width)
                    try:
                        full_lines = recognize_text_lines_in_image(
                            small,
                            trocr_model_name=printed_model,
                            trocr_device=trocr_device,
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
    handwritten_model = str(trocr_opts.get("model_name", TROCR_DEFAULT))
    printed_fallback_model = str(trocr_opts.get("printed_model_name", TROCR_PRINTED))
    use_video_printed_fallback = bool(trocr_opts.get("video_enable_printed_fallback", True))
    debug_dir = _resolve_video_debug_dir(video_path, pdf_output, settings)
    handwritten_engine = TrOCRHandwritingEngine(handwritten_model)
    printed_engine = TrOCRHandwritingEngine(printed_fallback_model) if use_video_printed_fallback else None
    settings = _apply_fast_video_settings(config or {})
    problems: Dict[str, str] = {}
    trocr_opts = settings.get("trocr", {})
    printed_fallback_model = str(trocr_opts.get("printed_model_name", TROCR_PRINTED))
    use_video_printed_fallback = bool(trocr_opts.get("video_enable_printed_fallback", True))
    trocr_device = parse_trocr_device_option(trocr_opts.get("device", "auto"))

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
        debug_metadata: list[dict[str, Any]] = []

        for keyframe_idx, item in enumerate(keyframes, start=1):
            frame_bgr = item["frame_bgr"]
            board_reading = run_module_a(
                frame_bgr,
                settings,
                roi_override=item.get("roi"),
                roi_method_override=str(item.get("roi_method") or "keyframe_selected"),
                engine_override=handwritten_engine,
            )
            if board_reading.get("error") and "module_a" not in problems:
                problems["module_a"] = board_reading["error"]

            roi_tuple = tuple((board_reading.get("roi") or item.get("roi") or (0, 0, frame_bgr.shape[1], frame_bgr.shape[0])))  # type: ignore[arg-type]
            roi_box = coerce_roi_box(roi_tuple, image_shape=frame_bgr.shape)
            ocr_input_bgr, _, ocr_mask = prepare_ocr_inputs(frame_bgr, roi_box)

            texts = board_reading.get("texts") or []
            fallback_used = False
            if use_video_printed_fallback and _texts_look_low_value(texts):
                try:
                    fallback_texts = recognize_text_lines_in_image(
                        ocr_input_bgr,
                        trocr_model_name=printed_fallback_model,
                        engine=printed_engine,
                    )
                    if fallback_texts and not _texts_look_low_value(fallback_texts):
                        texts = fallback_texts
                        fallback_used = True
                except Exception as e:
                    logger.warning("Printed OCR fallback failed on video keyframe: %s", e)
        best_board_lines: list[str] = []
        best_roi = None
        best_roi_method = None
        keyframe_results = []
        harvest_ran = False
        harvest_full_frame_cfg = False

        for item in keyframes:
            frame_bgr = item["frame_bgr"]
            board_reading = run_module_a(frame_bgr, settings)
            if board_reading.get("error") and "module_a" not in problems:
                problems["module_a"] = board_reading["error"]

            texts = list(board_reading.get("texts") or [])
            ocr_source = "none"
            if use_video_printed_fallback and _texts_look_low_value(texts):
                try:
                    fallback_texts = recognize_text_lines_in_image(
                        frame_bgr,
                        trocr_model_name=printed_fallback_model,
                        trocr_device=trocr_device,
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
            aggregated_board_lines = _merge_unique_lines(aggregated_board_lines, texts)

            clarity_result = item.get("clarity_result") or {}
            if clarity_result and clarity_result.get("score", 0.0) >= best_clarity.get("score", 0.0):
                best_clarity = clarity_result
                best_roi = board_reading.get("roi") or item.get("roi")
                best_roi_method = board_reading.get("roi_method") or item.get("roi_method")

            debug_paths = _save_video_debug_assets(
                debug_dir=debug_dir,
                keyframe_idx=keyframe_idx,
                frame_bgr=frame_bgr,
                roi_tuple=roi_box.as_tuple(),
                ocr_input_bgr=ocr_input_bgr,
                ocr_mask=ocr_mask,
            )
            keyframe_results.append(
                {
                    "frame_index": item.get("frame_index"),
                    "timestamp_sec": item.get("timestamp_sec"),
                    "change_ratio": item.get("change_ratio"),
                    "clarity": clarity_result,
                    "roi": roi_box.as_tuple(),
                    "roi_method": board_reading.get("roi_method") or item.get("roi_method"),
                    "board_texts": texts,
                    "printed_fallback_used": fallback_used,
                    "debug_paths": debug_paths or None,
                }
            )
            debug_metadata.append(keyframe_results[-1])
                best_board_lines = list(texts)
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
                    "ocr_source": ocr_source,
                }
            )

        video_cfg = settings.get("video") if isinstance(settings.get("video"), dict) else {}
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

        spoken_transcript = run_module_c(extracted_audio, settings)
        if spoken_transcript.get("error"):
            problems["module_c"] = spoken_transcript["error"]

        board_as_paragraph = "\n".join(aggregated_board_lines)
        merge_mode = str(video_cfg.get("board_text_mode", "merged") or "merged").lower().strip()
        min_sub = max(4, int(video_cfg.get("merge_substring_min_len", 8)))
        merged_deduped = _dedupe_subsumed_lines(aggregated_board_lines, min_len=min_sub)
        line_count_after_substring_dedupe = len(merged_deduped)

        # merged = time-ordered union across keyframes + substring collapse (closer to "full board").
        # best_clarity = single sharpest frame only (less noise from multi-frame, misses content).
        if merge_mode in ("best", "best_clarity", "single", "single_frame"):
            board_lines_primary = list(best_board_lines) if best_board_lines else list(merged_deduped)
        else:
            board_lines_primary = merged_deduped if merged_deduped else (list(best_board_lines) if best_board_lines else [])

        if bool(video_cfg.get("filter_noise_board_lines", False)):
            board_lines_primary = _filter_noise_board_lines(
                board_lines_primary,
                min_chars=max(4, int(video_cfg.get("noise_line_min_chars", 14))),
                min_letters=max(2, int(video_cfg.get("noise_line_min_letters", 5))),
            )

        ocr_diagnostics: Optional[dict[str, Any]] = None
        if bool(video_cfg.get("ocr_diagnostics", True)):
            ocr_diagnostics = _diagnose_video_ocr(
                keyframe_results,
                harvest_ran=harvest_ran,
                harvest_full_frame=harvest_full_frame_cfg,
                line_count_before_substring_dedupe=line_count_before_substring_dedupe,
                line_count_after_substring_dedupe=line_count_after_substring_dedupe,
            )

        board_as_paragraph = "\n".join(board_lines_primary)
        speech_text = spoken_transcript.get("speech_text") or ""

        lesson_alignment = run_module_d(board_as_paragraph, speech_text, settings)
        if lesson_alignment.get("error"):
            problems["module_d"] = lesson_alignment["error"]

        alignment_summary = lesson_alignment.get("alignment")
        bundle_for_pdf = {
            "board_lines": aggregated_board_lines,
            "board_lines": board_lines_primary,
            "clarity": best_clarity,
            "alignment": alignment_summary,
            "speech_text": speech_text,
            "module_errors": problems or None,
        }
        pdf_bundle = run_module_e(pdf_output, bundle_for_pdf)
        if pdf_bundle.get("error"):
            problems["module_e"] = pdf_bundle["error"]

    metadata_path = None
    if debug_dir is not None:
        metadata_path = debug_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "video_path": str(Path(video_path).resolve()),
                    "pdf_output": str(Path(pdf_output).resolve()),
                    "debug_dir": str(debug_dir.resolve()),
                    "keyframes": debug_metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

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
        "video_debug_dir": str(debug_dir.resolve()) if debug_dir is not None else None,
        "video_debug_metadata": str(metadata_path.resolve()) if metadata_path is not None else None,
    }
        out_video: dict[str, Any] = {
            "input_mode": "video",
            "video_fast_mode": bool((settings.get("video") or {}).get("fast_mode")),
            "board_texts": board_lines_primary,
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
        if ocr_diagnostics is not None:
            out_video["ocr_diagnostics"] = ocr_diagnostics
        return out_video
