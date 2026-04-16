"""
alignment.py — Compare board OCR text with spoken transcript

Method:
- semantic similarity with Sentence-BERT
- keyword overlap as a lexical safety check
- rule-based verdict mapping
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None


SBERT_MODELS: Dict[str, Any] = {}
SBERT_LOCK = threading.Lock()

DEFAULT_SBERT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def evaluate_alignment(
    board_lines: List[str],
    speech_text: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Public API.

    Returns:
    {
        "semantic_similarity": 0.81,
        "keyword_overlap_rate": 0.42,
        "verdict": "highly_aligned",
        "matched_topics": [...],
        "board_only_topics": [...],
        "speech_only_topics": [...],
        "board_text": "...",
        "speech_text": "...",
    }
    """
    board_text = " ".join(str(x).strip() for x in board_lines if str(x).strip()).strip()
    speech_text = str(speech_text or "").strip()

    semantic_cfg = config.get("semantic") or {}
    sbert_cfg = config.get("sbert") or {}

    model_name = str(sbert_cfg.get("model_name", DEFAULT_SBERT)).strip() or DEFAULT_SBERT
    high_sim = float(semantic_cfg.get("high_match_min", 0.72))
    partial_sim = float(semantic_cfg.get("partial_min", 0.45))
    keyword_high = float(semantic_cfg.get("keyword_overlap_high", 0.35))
    keyword_partial = float(semantic_cfg.get("keyword_overlap_partial", 0.20))

    semantic_similarity = compute_semantic_similarity(board_text, speech_text, model_name)
    keyword_overlap = keyword_overlap_rate(board_text, speech_text)

    verdict = judge_alignment(
        semantic_similarity=semantic_similarity,
        keyword_overlap=keyword_overlap,
        high_sim=high_sim,
        partial_sim=partial_sim,
        keyword_high=keyword_high,
        keyword_partial=keyword_partial,
    )

    board_tokens = tokenize_mixed(board_text)
    speech_tokens = tokenize_mixed(speech_text)

    matched = sorted(board_tokens & speech_tokens)
    board_only = sorted(board_tokens - speech_tokens)
    speech_only = sorted(speech_tokens - board_tokens)

    return {
        "semantic_similarity": round(semantic_similarity, 4),
        "keyword_overlap_rate": round(keyword_overlap, 4),
        "verdict": verdict,
        "matched_topics": matched[:30],
        "board_only_topics": board_only[:30],
        "speech_only_topics": speech_only[:30],
        "board_text": board_text,
        "speech_text": speech_text,
    }


class SemanticAligner:
    def __init__(self, model_name: str = DEFAULT_SBERT) -> None:
        self.model_name = model_name
        self.model = None

    def ensure(self) -> None:
        if self.model is not None:
            return

        if SentenceTransformer is None:
            raise RuntimeError("Install sentence-transformers first.")

        with SBERT_LOCK:
            cached = SBERT_MODELS.get(self.model_name)
            if cached is not None:
                self.model = cached
                return

            model = SentenceTransformer(self.model_name)
            SBERT_MODELS[self.model_name] = model
            self.model = model

    def similarity(self, text_a: str, text_b: str) -> float:
        self.ensure()
        assert self.model is not None

        a = (text_a or "").strip()
        b = (text_b or "").strip()

        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        emb = self.model.encode(
            [a, b],
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        if util is not None:
            cos = util.cos_sim(emb[0], emb[1]).item()
        else:
            v1 = emb[0].detach().cpu().numpy()
            v2 = emb[1].detach().cpu().numpy()
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            cos = float(np.dot(v1, v2) / denom)

        return float(max(0.0, min(1.0, cos)))


def compute_semantic_similarity(text_a: str, text_b: str, model_name: str) -> float:
    aligner = SemanticAligner(model_name)
    return aligner.similarity(text_a, text_b)


def tokenize_mixed(text: str) -> Set[str]:
    """
    Basic mixed-script tokeniser:
    - English words
    - CJK chars
    - digits
    """
    if not text or not text.strip():
        return set()

    words = re.findall(r"[A-Za-z]+", text.lower())
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    digits = re.findall(r"\d+", text)

    return set(words) | set(chars) | set(digits)


def keyword_overlap_rate(a: str, b: str) -> float:
    sa = tokenize_mixed(a)
    sb = tokenize_mixed(b)

    if not sa and not sb:
        return 1.0

    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


def judge_alignment(
    semantic_similarity: float,
    keyword_overlap: float,
    high_sim: float = 0.72,
    partial_sim: float = 0.45,
    keyword_high: float = 0.35,
    keyword_partial: float = 0.20,
) -> str:
    """
    Rule-based verdict layer.
    """
    strong_meaning = semantic_similarity >= high_sim
    enough_shared_words = keyword_overlap >= keyword_high

    if strong_meaning and enough_shared_words:
        return "highly_aligned"

    meaning_close_enough = semantic_similarity >= partial_sim
    some_shared_words = keyword_overlap >= keyword_partial

    if meaning_close_enough or some_shared_words:
        return "partially_related"

    return "content_mismatch"