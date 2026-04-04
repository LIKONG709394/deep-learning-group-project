# Optional helper if you already have board text and only want: audio -> transcript -> match.
# The main app uses pipeline.run_from_frame_and_audio instead.

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from blackboard_analytics.module_c_whisper import transcribe_audio
from blackboard_analytics.module_d_semantic import DEFAULT_SBERT, compare_board_and_speech

logger = logging.getLogger(__name__)


def analyze_speech_vs_board(
    audio_path: str,
    board_text: str,
    *,
    whisper_model_size: str = "base",
    sbert_model_name: str = DEFAULT_SBERT,
    whisper_language: Optional[str] = None,
    high_sim: float = 0.72,
    partial_sim: float = 0.45,
    keyword_high: float = 0.35,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "speech_text": "",
        "semantic_similarity": None,
        "keyword_overlap_rate": None,
        "verdict": None,
        "error": None,
    }
    try:
        transcript = transcribe_audio(
            audio_path,
            model_size=whisper_model_size,
            language=whisper_language,
        )
        out["speech_text"] = transcript
        match = compare_board_and_speech(
            board_text,
            transcript,
            model_name=sbert_model_name,
            high_sim=high_sim,
            partial_sim=partial_sim,
            keyword_high=keyword_high,
        )
        out["semantic_similarity"] = match["semantic_similarity"]
        out["keyword_overlap_rate"] = match["keyword_overlap_rate"]
        out["verdict"] = match["verdict"]
    except Exception as e:
        out["error"] = str(e)
        logger.exception("analyze_speech_vs_board")
    return out
