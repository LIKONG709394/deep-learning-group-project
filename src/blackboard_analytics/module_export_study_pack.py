from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional


_QUOTE_RE = re.compile(r"[`'\"“”‘’]")
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _canonicalize(text: str) -> str:
    s = _normalize_text(text).lower()
    if not s:
        return ""
    s = _QUOTE_RE.sub("", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _near_duplicate(left: str, right: str, *, min_ratio: float = 0.94) -> bool:
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


def _looks_low_value(text: str) -> bool:
    cleaned = _normalize_text(text)
    if not cleaned:
        return True
    alpha_count = sum(ch.isalpha() for ch in cleaned)
    alnum_count = sum(ch.isalnum() for ch in cleaned)
    if alnum_count == 0:
        return True
    if alpha_count == 0 and alnum_count < 4:
        return True
    if len(cleaned) <= 2:
        return True
    return False


def _trim_line(text: str, *, max_chars: int) -> str:
    s = _normalize_text(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def _clean_unique_lines(
    lines: list[str],
    *,
    max_items: int,
    max_chars_per_line: int,
) -> list[str]:
    cleaned_lines: list[str] = []
    seen: list[str] = []
    for raw in lines:
        text = _trim_line(raw or "", max_chars=max_chars_per_line)
        token = _canonicalize(text)
        if not token or _looks_low_value(text):
            continue
        if any(_near_duplicate(token, prior) for prior in seen):
            continue
        seen.append(token)
        cleaned_lines.append(text)
        if len(cleaned_lines) >= max_items:
            break
    return cleaned_lines


def _clean_transcript_from_segments(
    speech_segments: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    parts: list[str] = []
    seen: list[str] = []
    total_chars = 0
    for seg in speech_segments:
        text = _normalize_text(str(seg.get("text") or ""))
        token = _canonicalize(text)
        if not token or _looks_low_value(text):
            continue
        if seen and _near_duplicate(token, seen[-1], min_ratio=0.96):
            continue
        if any(_near_duplicate(token, prior, min_ratio=0.97) for prior in seen[-6:]):
            continue
        parts.append(text)
        seen.append(token)
        total_chars += len(text) + 1
        if total_chars >= max_chars:
            break
    joined = " ".join(parts)
    return joined[:max_chars].strip()


def _clean_transcript(
    speech_text: str,
    speech_segments: Optional[list[dict[str, Any]]],
    *,
    max_chars: int,
) -> str:
    if speech_segments:
        text = _clean_transcript_from_segments(speech_segments, max_chars=max_chars)
        if text:
            return text
    return _normalize_text(speech_text)[:max_chars].strip()


def _format_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    mm, ss = divmod(total, 60)
    hh, mm = divmod(mm, 60)
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _build_timeline_lines(
    *,
    speech_segments: Optional[list[dict[str, Any]]],
    video_keyframes: Optional[list[dict[str, Any]]],
    max_items: int,
    min_gap_sec: float,
    max_chars_per_item: int,
) -> list[str]:
    events: list[tuple[float, str, str]] = []
    if video_keyframes:
        for item in video_keyframes:
            lines = _clean_unique_lines(
                list(item.get("board_texts") or []),
                max_items=2,
                max_chars_per_line=80,
            )
            if not lines:
                continue
            ts = float(item.get("timestamp_sec") or 0.0)
            text = f"[{_format_ts(ts)}] Board: {' | '.join(lines[:2])}"
            events.append((ts, text, _canonicalize(text)))
    if speech_segments:
        for seg in speech_segments:
            text = _normalize_text(str(seg.get("text") or ""))
            if _looks_low_value(text) or len(text) < 24:
                continue
            ts = float(seg.get("start") or seg.get("start_sec") or 0.0)
            rendered = f"[{_format_ts(ts)}] Speech: {_trim_line(text, max_chars=max_chars_per_item)}"
            events.append((ts, rendered, _canonicalize(rendered)))

    events.sort(key=lambda x: x[0])
    kept: list[str] = []
    kept_tokens: list[str] = []
    last_ts: Optional[float] = None
    for ts, rendered, token in events:
        if last_ts is not None and abs(ts - last_ts) < max(0.0, min_gap_sec):
            if any(_near_duplicate(token, prior, min_ratio=0.95) for prior in kept_tokens[-3:]):
                continue
        if any(_near_duplicate(token, prior, min_ratio=0.96) for prior in kept_tokens):
            continue
        kept.append(rendered)
        kept_tokens.append(token)
        last_ts = ts
        if len(kept) >= max_items:
            break
    return kept


def _default_prompt_hint() -> str:
    return (
        "Optional note for your AI: The following material was automatically extracted from class "
        "video/audio. OCR and speech recognition may contain errors, and the teacher may use English "
        "together with other languages. You can use these notes to help summarize the lesson topic, "
        "main concepts, examples, and anything that may need verification."
    )


def build_study_pack(
    *,
    board_lines: list[str],
    speech_text: str,
    speech_segments: Optional[list[dict[str, Any]]] = None,
    video_keyframes: Optional[list[dict[str, Any]]] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = dict((config or {}).get("study_pack") or {})
    enabled = bool(cfg.get("enabled", True))
    title = str(cfg.get("title", "Class Content Export") or "Class Content Export")
    max_board_lines = max(1, int(cfg.get("max_board_lines", 40)))
    max_line_chars = max(40, int(cfg.get("max_line_chars", 220)))
    max_speech_chars = max(400, int(cfg.get("max_speech_chars", 12000)))
    include_timeline = bool(cfg.get("include_timeline", True))
    max_timeline_items = max(0, int(cfg.get("max_timeline_items", 10)))
    timeline_min_gap_sec = max(0.0, float(cfg.get("timeline_min_gap_sec", 20.0)))
    include_ai_prompt_hint = bool(cfg.get("include_ai_prompt_hint", True))

    base = {
        "enabled": enabled,
        "title": title,
        "board_lines_clean": [],
        "board_text_clean": "",
        "speech_text_clean": "",
        "timeline_items": [],
        "timeline_text": "",
        "ai_prompt_hint": _default_prompt_hint() if include_ai_prompt_hint else "",
        "full_export_text": "",
        "error": None,
    }
    if not enabled:
        return base

    try:
        board_lines_clean = _clean_unique_lines(
            list(board_lines or []),
            max_items=max_board_lines,
            max_chars_per_line=max_line_chars,
        )
        speech_text_clean = _clean_transcript(
            speech_text or "",
            speech_segments or [],
            max_chars=max_speech_chars,
        )
        timeline_items = (
            _build_timeline_lines(
                speech_segments=speech_segments or [],
                video_keyframes=video_keyframes or [],
                max_items=max_timeline_items,
                min_gap_sec=timeline_min_gap_sec,
                max_chars_per_item=max_line_chars,
            )
            if include_timeline and max_timeline_items > 0
            else []
        )

        board_text_clean = "\n".join(f"- {line}" for line in board_lines_clean) if board_lines_clean else "(no reliable board or slide text extracted)"
        speech_block = speech_text_clean or "(no reliable lecture transcript extracted)"
        timeline_text = "\n".join(timeline_items)

        parts = [
            title,
            "",
            "Board / Slide Text",
            board_text_clean,
            "",
            "Lecture Transcript",
            speech_block,
        ]
        if timeline_text:
            parts.extend(["", "Timeline / Key Moments", timeline_text])
        if include_ai_prompt_hint and base["ai_prompt_hint"]:
            parts.extend(["", "Optional prompt hint for your AI", base["ai_prompt_hint"]])

        base.update(
            {
                "board_lines_clean": board_lines_clean,
                "board_text_clean": board_text_clean,
                "speech_text_clean": speech_block,
                "timeline_items": timeline_items,
                "timeline_text": timeline_text,
                "full_export_text": "\n".join(parts).strip(),
            }
        )
        return base
    except Exception as exc:
        base["error"] = str(exc)
        return base
