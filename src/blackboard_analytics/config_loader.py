# Load optional YAML settings for the pipeline (CLI and web use the same defaults).

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def default_config_path() -> Path:
    # .../src/blackboard_analytics/config_loader.py -> project root
    return Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_pipeline_config(path: Optional[Path | str] = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else default_config_path()
    if not p.is_file():
        return {}
    import yaml

    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
