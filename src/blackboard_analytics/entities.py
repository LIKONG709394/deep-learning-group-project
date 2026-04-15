from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, util

from default import *

import threading

from support import *

import numpy as np

from support import has_hf_repo_cache

@dataclass
class ROIBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, w: int, h: int) -> "ROIBox":
        return ROIBox(
            max(0, self.x1),
            max(0, self.y1),
            min(w, self.x2),
            min(h, self.y2),
        )

    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)
    
_SBERT_LOCK = threading.Lock()   
_SBERT_MODELS = {} 
class SemanticAligner:
    def __init__(self, model_name: str = DEFAULT_SBERT) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            raise RuntimeError("Install sentence-transformers")
        with _SBERT_LOCK:
            cached = _SBERT_MODELS.get(self.model_name)
            if cached is not None:
                self._model = cached
                return
            try:
                model = SentenceTransformer(
                    self.model_name,
                    local_files_only=has_hf_repo_cache(self.model_name),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load SBERT: {e}") from e
            self._model = model
            _SBERT_MODELS[self.model_name] = model

    def similarity(self, text_a: str, text_b: str) -> float:
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
