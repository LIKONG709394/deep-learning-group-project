# Supplementary LLM-based alignment check for board text vs speech text.
# This does not replace the traditional SBERT + keyword module; it adds an
# optional DeepSeek judgment layer with a stable JSON-shaped output.

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
ALLOWED_RELEVANCE = {
    "highly_relevant",
    "partially_relevant",
    "weakly_relevant",
    "off_topic",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _PROJECT_ROOT / ".env"


def _empty_result(
    *,
    enabled: bool,
    model: str,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "model": model,
        "overall_relevance": None,
        "score": None,
        "reason": None,
        "evidence": [],
        "error": error_message,
    }


def _normalize_endpoint(base_url: str) -> str:
    cleaned = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def _strip_optional_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _read_project_dotenv() -> Dict[str, str]:
    if not _DOTENV_PATH.is_file():
        return {}

    values: Dict[str, str] = {}
    try:
        for raw_line in _DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = _strip_optional_quotes(value)
    except Exception as e:
        logger.warning("Failed to read .env file at %s: %s", _DOTENV_PATH, e)
        return {}
    return values


def _resolve_api_key(env_name: str) -> str:
    dotenv_values = _read_project_dotenv()
    dotenv_value = dotenv_values.get(env_name, "").strip()
    if dotenv_value:
        return dotenv_value
    return os.environ.get(env_name, "").strip()


def _build_messages(board_text: str, speech_text: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an evaluator for classroom teaching alignment. "
        "You compare board_text and speech_text, then judge whether the speech stays focused on the board content "
        "or is clearly off-topic. "
        "Return JSON only. Do not include markdown, prose outside JSON, or extra keys."
    )
    user_prompt = (
        "Task:\n"
        "1. Compare the two inputs: board_text and speech_text.\n"
        "2. Judge whether the speech is centered on the board content.\n"
        "3. Judge whether the speech is clearly off-topic.\n"
        "4. Be conservative when information is missing, empty, or too short. Do not guess details.\n\n"
        "Output requirements:\n"
        '- Return exactly one JSON object with these fields only:\n'
        '  "overall_relevance": one of ["highly_relevant", "partially_relevant", "weakly_relevant", "off_topic"]\n'
        '  "score": a number from 0 to 100\n'
        '  "reason": a short explanation in one or two sentences\n'
        '  "evidence": an array of 2 to 5 short strings\n'
        "- If the texts are empty or insufficient, give a cautious low-confidence style judgment rather than making things up.\n\n"
        f"board_text:\n{board_text.strip() or '[EMPTY]'}\n\n"
        f"speech_text:\n{speech_text.strip() or '[EMPTY]'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_message_content(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("DeepSeek response missing message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise RuntimeError("DeepSeek response missing text content")


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Empty DeepSeek content")
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in DeepSeek content")
    snippet = cleaned[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek content JSON is not an object")
    return parsed


def _fallback_relevance_from_score(score: float) -> str:
    if score >= 80:
        return "highly_relevant"
    if score >= 60:
        return "partially_relevant"
    if score >= 35:
        return "weakly_relevant"
    return "off_topic"


def _normalize_analysis(raw: Dict[str, Any], *, enabled: bool, model: str) -> Dict[str, Any]:
    overall = str(raw.get("overall_relevance") or "").strip().lower()
    score_raw = raw.get("score")
    try:
        score = max(0.0, min(100.0, float(score_raw)))
    except (TypeError, ValueError):
        score = 0.0

    if overall not in ALLOWED_RELEVANCE:
        overall = _fallback_relevance_from_score(score)

    reason = str(raw.get("reason") or "").strip() or "No reason returned by DeepSeek."
    evidence_raw = raw.get("evidence")
    evidence: List[str] = []
    if isinstance(evidence_raw, list):
        evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]
    elif isinstance(evidence_raw, str) and evidence_raw.strip():
        evidence = [evidence_raw.strip()]
    if len(evidence) > 5:
        evidence = evidence[:5]
    if not evidence:
        evidence = [reason]

    return {
        "enabled": enabled,
        "model": model,
        "overall_relevance": overall,
        "score": round(score, 2),
        "reason": reason,
        "evidence": evidence,
        "error": None,
    }


def _call_deepseek_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout_sec: float,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {detail[:500]}") from e
    except error.URLError as e:
        raise RuntimeError(f"DeepSeek request failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"DeepSeek request failed: {e}") from e

    try:
        response_json = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"DeepSeek returned invalid JSON response: {raw[:500]}") from e
    if not isinstance(response_json, dict):
        raise RuntimeError("DeepSeek response root is not a JSON object")
    return response_json


def _strip_to_ascii_only(s: str) -> str:
    """Replace non-ASCII characters (CJK, Cyrillic, accented Latin, etc.) with spaces."""
    if not s:
        return ""
    out = "".join(c if ord(c) < 128 else " " for c in s)
    return re.sub(r"\s+", " ", out).strip()


def _drop_singleton_letter_tokens(s: str) -> str:
    """Remove whitespace-separated tokens whose only letters are a single A–Z/a–z (except a, I). Keeps e.g. 3D, x=5."""
    if not s:
        return ""
    parts = s.split()
    keep: List[str] = []
    for p in parts:
        letters = re.sub(r"[^A-Za-z]", "", p)
        digits = re.sub(r"[^0-9]", "", p)
        if len(letters) == 1 and letters not in ("a", "I") and not digits:
            continue
        if len(letters) == 0 and len(digits) == 0:
            if len(p) <= 1:
                continue
        keep.append(p)
    return " ".join(keep).strip()


def _refine_board_line_after_filter(line: str, cfg: dict) -> str:
    """Post-process kept lines: ASCII-only strip and orphan single-letter token removal."""
    if not bool(cfg.get("filter_refine_post_process", True)):
        return line.strip()
    s = line.strip()
    if not s:
        return ""
    if bool(cfg.get("filter_refine_ascii_only", True)):
        s = _strip_to_ascii_only(s)
    if bool(cfg.get("filter_refine_drop_singleton_letters", True)):
        s = _drop_singleton_letter_tokens(s)
    return re.sub(r"\s+", " ", s).strip()


def _refine_kept_lines_after_filter(kept: List[str], cfg: dict) -> List[str]:
    out: List[str] = []
    for line in kept:
        r = _refine_board_line_after_filter(line, cfg)
        if r:
            out.append(r)
    return out


def build_filter_board_lines_messages(
    lines: List[str],
    speech_text: str,
) -> List[Dict[str, str]]:
    system_prompt = (
        "You clean noisy OCR lines from a classroom whiteboard, projector slide, or similar. "
        "Remove: UI chrome, button labels, stray single words that are not educational content, "
        "watermarks, timestamps, window titles, duplicate near-empty fragments, and obvious OCR garbage. "
        "Keep: instructional text, headings, bullet content, formulas words, and phrases that belong to the lesson. "
        "Prefer lines that are primarily English/Latin script; drop obvious non-Latin script noise when it is not lesson content. "
        "speech_text is optional context only (what the teacher said); use it to prefer lines that match the topic, "
        "but do not invent lines. Return JSON only."
    )
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
    user_prompt = (
        "Task:\n"
        "1. Review each numbered OCR line below.\n"
        "2. Decide which line indices to KEEP (educational / slide content).\n"
        "3. When uncertain, prefer KEEP over DROP.\n\n"
        "Output requirements:\n"
        '- Return exactly one JSON object with these fields only:\n'
        '  "kept_indices": array of integers — 0-based indices referring ONLY to the numbered list below\n'
        '  "reason": one short sentence explaining the main noise you removed\n'
        "Preserve the order of indices as they appear in the list (ascending).\n\n"
        f"speech_text (context, may be empty):\n{(speech_text or '').strip() or '[EMPTY]'}\n\n"
        f"numbered OCR lines:\n{numbered}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



def run_deepseek_filter_board_lines(lines, speech_text, config, env):
    """Use DeepSeek to drop meaningless OCR lines. On failure, returns original lines."""
    out_base = {
        "enabled": False,
        "model": config.deepseek_model,
        "kept_lines": list(lines),
        "dropped_lines": [],
        "reason": None,
        "error": None,
    }
    out_base["enabled"] = True    
    endpoint = _normalize_endpoint(DEFAULT_BASE_URL)
    messages = _build_filter_board_lines_messages(lines, speech_text)
    try:
        response_json = _call_deepseek_chat_completion(
            endpoint=endpoint,
            api_key=env.deepseek,
            model=config.deepseek_model,
            timeout_sec=config.deepseek_timeout_sec,
            messages=messages,
        )
        content = _extract_message_content(response_json)
        parsed = _extract_json_object(content)
    except Exception as e:
        logger.exception("run_deepseek_filter_board_lines")
        out_base["error"] = str(e)
        return out_base

    indices_raw = parsed.get("kept_indices")
    reason = str(parsed.get("reason") or "").strip() or None
    out_base["reason"] = reason

    if not isinstance(indices_raw, list):
        out_base["error"] = (
            "DeepSeek filter response missing kept_indices array; keeping all lines unchanged"
        )
        return out_base

    if len(indices_raw) == 0:
        kept = []
        dropped = list(work_lines)
    else:
        kept_set: set[int] = set()
        for x in indices_raw:
            try:
                idx = int(x)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(work_lines):
                kept_set.add(idx)

        kept_indices_sorted = sorted(kept_set)
        kept = [work_lines[i] for i in kept_indices_sorted]
        dropped = [work_lines[i] for i in range(len(work_lines)) if i not in kept_set]
        if not kept_set and work_lines:
            logger.warning("DeepSeek filter: no valid kept_indices; keeping all lines")
            kept = list(work_lines)
            dropped = []
            out_base["error"] = "No valid kept_indices in model response; kept all lines"

    if truncated:
        kept.extend(cleaned[max_lines:])

    kept = _refine_kept_lines_after_filter(kept, cfg)

    out_base["kept_lines"] = kept
    out_base["dropped_lines"] = dropped
    out_base["error"] = None
    return out_base


def run_module_d_deepseek(
    board_text: str,
    speech_text: str,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    cfg = (config or {}).get("deepseek", {})
    enabled = bool(cfg.get("enabled", False))
    model = str(cfg.get("model", DEFAULT_MODEL))
    if not enabled:
        return _empty_result(enabled=False, model=model, error_message=None)

    api_key_env = str(cfg.get("api_key_env", DEFAULT_API_KEY_ENV) or DEFAULT_API_KEY_ENV)
    api_key = _resolve_api_key(api_key_env)
    if not api_key:
        return _empty_result(
            enabled=True,
            model=model,
            error_message=(
                f"Missing DeepSeek API key: expected `{api_key_env}` in project .env "
                f"({ _DOTENV_PATH }) or process environment"
            ),
        )

    endpoint = _normalize_endpoint(str(cfg.get("base_url", DEFAULT_BASE_URL)))
    temperature = float(cfg.get("temperature", 0.1))
    timeout_sec = float(cfg.get("timeout_sec", 30))
    messages = _build_messages(board_text, speech_text)

    try:
        response_json = _call_deepseek_chat_completion(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout_sec=timeout_sec,
            messages=messages,
        )
        content = _extract_message_content(response_json)
        parsed = _extract_json_object(content)
        return _normalize_analysis(parsed, enabled=True, model=model)
    except Exception as e:
        logger.exception("run_module_d_deepseek")
        return _empty_result(enabled=True, model=model, error_message=str(e))
