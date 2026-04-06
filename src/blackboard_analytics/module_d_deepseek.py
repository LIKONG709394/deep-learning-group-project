# Supplementary LLM-based alignment check for board text vs speech text.
# This does not replace the traditional SBERT + keyword module; it adds an
# optional DeepSeek judgment layer with a stable JSON-shaped output.

from __future__ import annotations

import json
import logging
import os
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
