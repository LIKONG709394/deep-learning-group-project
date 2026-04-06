# Quick check: venv, YAML config, critical imports (run from project root).

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    os.chdir(ROOT)
    ok = True
    print("ROOT:", ROOT)
    print("Python:", sys.executable)

    try:
        import yaml
    except ImportError:
        print("FAIL: PyYAML not installed")
        ok = False
    else:
        cfg_path = ROOT / "config" / "default.yaml"
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            print("config/default.yaml:", "loaded" if isinstance(data, dict) else "invalid")
            if isinstance(data, dict):
                print("  video.fast_mode:", (data.get("video") or {}).get("fast_mode"))
        else:
            print("WARN: missing", cfg_path)

    for mod in ("uvicorn", "fastapi", "cv2", "torch", "ultralytics", "whisper"):
        try:
            __import__(mod)
            print(f"import {mod}: ok")
        except Exception as e:
            print(f"import {mod}: FAIL ({e})")
            ok = False

    try:
        from blackboard_analytics.config_loader import load_pipeline_config

        c = load_pipeline_config()
        print("load_pipeline_config keys:", list(c.keys())[:8], "...")
    except Exception as e:
        print("FAIL blackboard_analytics.config_loader:", e)
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
