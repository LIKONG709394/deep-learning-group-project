# Compare "what is on the board" with "what was said": sentence vectors for paraphrase-style
# similarity, plus a simple token overlap score (English words, single Chinese chars, digits).

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, Optional, Set

import numpy as np

from blackboard_analytics.model_cache import (
    ensure_project_model_cache_dirs,
    has_hf_repo_cache,
)

logger = logging.getLogger(__name__)
ensure_project_model_cache_dirs()

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

DEFAULT_SBERT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_SBERT_MODELS: dict[str, Any] = {}
_SBERT_LOCK = threading.Lock()

# Verdict rules in plain language:
# - highly_aligned: meaning is close AND a fair share of words/characters match.
# - partially_related: meaning is somewhat close OR at least some words overlap.
# - content_mismatch: neither of the above.
KEYWORD_OVERLAP_FOR_PARTIAL = 0.2


def tokenize_mixed(text: str) -> Set[str]:
    if not text or not text.strip():
        return set()
    words = re.findall(r"[a-zA-Z]+", text.lower())
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    digits = re.findall(r"\d+", text)
    return set(words) | set(chars) | set(digits)


def keyword_overlap_rate(a: str, b: str) -> float:
    sa, sb = tokenize_mixed(a), tokenize_mixed(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0




def judge_alignment(
    semantic_sim: float,
    keyword_overlap: float,
    *,
    high_sim: float = 0.72,
    partial_sim: float = 0.45,
    keyword_high: float = 0.35,
) -> str:
    strong_meaning = semantic_sim >= high_sim
    enough_shared_words = keyword_overlap >= keyword_high
    if strong_meaning and enough_shared_words:
        return "highly_aligned"

    meaning_close_enough = semantic_sim >= partial_sim
    some_shared_words = keyword_overlap >= KEYWORD_OVERLAP_FOR_PARTIAL
    if meaning_close_enough or some_shared_words:
        return "partially_related"

    return "content_mismatch"



def compare_board_and_speech(
    board_text: str,
    speech_text: str,
    *,
    model_name: str = DEFAULT_SBERT,
    high_sim: float = 0.72,
    partial_sim: float = 0.45,
    keyword_high: float = 0.35,
    aligner: Optional[SemanticAligner] = None,
) -> Dict[str, Any]:
    sbert = aligner or SemanticAligner(model_name)
    try:
        cosine_similarity = sbert.similarity(board_text, speech_text)
    except Exception:
        logger.exception("SBERT similarity")
        raise
    token_overlap = keyword_overlap_rate(board_text, speech_text)
    verdict = judge_alignment(
        cosine_similarity,
        token_overlap,
        high_sim=high_sim,
        partial_sim=partial_sim,
        keyword_high=keyword_high,
    )
    return {
        "semantic_similarity": round(cosine_similarity, 4),
        "keyword_overlap_rate": round(token_overlap, 4),
        "verdict": verdict,
    }


def run_module_d(
    board_text: str,
    speech_text: str,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    merged = config or {}
    verdict_rules = merged.get("semantic", {})
    encoder_opts = merged.get("sbert", {})
    out: Dict[str, Any] = {"alignment": None, "error": None}
    try:
        out["alignment"] = compare_board_and_speech(
            board_text,
            speech_text,
            model_name=str(encoder_opts.get("model_name", DEFAULT_SBERT)),
            high_sim=float(verdict_rules.get("high_match_min", 0.72)),
            partial_sim=float(verdict_rules.get("partial_min", 0.45)),
            keyword_high=float(verdict_rules.get("keyword_overlap_high", 0.35)),
        )
    except Exception as e:
        out["error"] = str(e)
        logger.exception("run_module_d")
    return out
