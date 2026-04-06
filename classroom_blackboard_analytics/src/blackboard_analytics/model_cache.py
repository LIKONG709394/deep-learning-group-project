from __future__ import annotations

import os
from pathlib import Path

# Hugging Face `transformers` (TrOCR + sentence-transformers deps) may probe TensorFlow/Keras.
# Keras 3 breaks the TF stub unless `tf-keras` is installed; this app is PyTorch-only.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_model_cache_root() -> Path:
    return get_project_root() / ".model_cache"


def get_hf_home() -> Path:
    return get_model_cache_root() / "huggingface"


def get_hf_hub_cache_dir() -> Path:
    return get_hf_home() / "hub"


def get_transformers_cache_dir() -> Path:
    return get_hf_home() / "transformers"


def get_sentence_transformers_cache_dir() -> Path:
    return get_model_cache_root() / "sentence_transformers"


def get_whisper_cache_dir() -> Path:
    return get_model_cache_root() / "whisper"


def get_torch_home() -> Path:
    return get_model_cache_root() / "torch"


def ensure_project_model_cache_dirs() -> Path:
    root = get_model_cache_root()
    get_hf_hub_cache_dir().mkdir(parents=True, exist_ok=True)
    get_transformers_cache_dir().mkdir(parents=True, exist_ok=True)
    get_sentence_transformers_cache_dir().mkdir(parents=True, exist_ok=True)
    get_whisper_cache_dir().mkdir(parents=True, exist_ok=True)
    get_torch_home().mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(get_hf_home())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(get_hf_hub_cache_dir())
    # HF_HOME + hub cache are enough; TRANSFORMERS_CACHE is deprecated in transformers v5.
    os.environ["TORCH_HOME"] = str(get_torch_home())
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(get_sentence_transformers_cache_dir())
    return root


def has_hf_repo_cache(model_name: str) -> bool:
    repo_dir = get_hf_hub_cache_dir() / f"models--{model_name.replace('/', '--')}"
    return repo_dir.is_dir() and (repo_dir / "snapshots").is_dir()
