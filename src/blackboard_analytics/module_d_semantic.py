"""
Module D: SBERT semantic similarity + mixed-language keyword overlap + alignment verdict.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


def tokenize_mixed(text: str) -> Set[str]:
    """
    Mixed Chinese/English: lowercase Latin words + single CJK characters + digit tokens (no jieba).
    """
    if not text or not text.strip():
        return set()
    words = re.findall(r"[a-zA-Z]+", text.lower())
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    digits = re.findall(r"\d+", text)
    return set(words) | set(chars) | set(digits)


def keyword_overlap_rate(a: str, b: str) -> float:
    """Jaccard |A∩B|/|A∪B|; returns 1.0 if both empty."""
    sa, sb = tokenize_mixed(a), tokenize_mixed(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0


class SemanticAligner:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise RuntimeError("Install sentence-transformers")
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SBERT: {e}") from e

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity in [0, 1] (typical for L2-normalized embeddings)."""
        self._ensure()
        assert self._model is not None
        t1 = (text_a or "").strip() or " "
        t2 = (text_b or "").strip() or " "
        emb = self._model.encode([t1, t2], convert_to_tensor=True, show_progress_bar=False)
        if util is not None:
            cos = util.cos_sim(emb[0], emb[1]).item()
        else:
            v1 = emb[0].detach().cpu().numpy()
            v2 = emb[1].detach().cpu().numpy()
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        return float(max(0.0, min(1.0, cos)))


def judge_alignment(
    semantic_sim: float,
    keyword_overlap: float,
    *,
    high_sim: float = 0.72,
    partial_sim: float = 0.45,
    keyword_high: float = 0.35,
) -> str:
    """Verdict: highly_aligned | partially_related | content_mismatch."""
    if semantic_sim >= high_sim and keyword_overlap >= keyword_high:
        return "highly_aligned"
    if semantic_sim >= partial_sim or keyword_overlap >= 0.2:
        return "partially_related"
    return "content_mismatch"


def compare_board_and_speech(
    board_text: str,
    speech_text: str,
    *,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    high_sim: float = 0.72,
    partial_sim: float = 0.45,
    keyword_high: float = 0.35,
    aligner: Optional[SemanticAligner] = None,
) -> Dict[str, Any]:
    """Returns semantic_similarity, keyword_overlap_rate, verdict (English keys)."""
    eng = aligner or SemanticAligner(model_name)
    try:
        sim = eng.similarity(board_text, speech_text)
    except Exception as e:
        logger.exception("SBERT similarity")
        raise
    overlap = keyword_overlap_rate(board_text, speech_text)
    verdict = judge_alignment(sim, overlap, high_sim=high_sim, partial_sim=partial_sim, keyword_high=keyword_high)
    return {
        "semantic_similarity": round(sim, 4),
        "keyword_overlap_rate": round(overlap, 4),
        "verdict": verdict,
    }


def run_module_d(
    board_text: str,
    speech_text: str,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    cfg = (config or {}).get("semantic", {})
    scfg = (config or {}).get("sbert", {})
    out: Dict[str, Any] = {"alignment": None, "error": None}
    try:
        out["alignment"] = compare_board_and_speech(
            board_text,
            speech_text,
            model_name=str(scfg.get("model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")),
            high_sim=float(cfg.get("high_match_min", 0.72)),
            partial_sim=float(cfg.get("partial_min", 0.45)),
            keyword_high=float(cfg.get("keyword_overlap_high", 0.35)),
        )
    except Exception as e:
        out["error"] = str(e)
        logger.exception("run_module_d")
    return out
