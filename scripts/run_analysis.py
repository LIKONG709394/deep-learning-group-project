"""
CLI: python run_analysis.py --image frame.jpg --audio clip.mp3 --pdf out.pdf
(Audio must be MP3; ffmpeg required for MP3.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

from blackboard_analytics.pipeline import run_from_image_and_audio_files


def main() -> None:
    p = argparse.ArgumentParser(description="Classroom blackboard analytics (single frame + audio)")
    p.add_argument("--image", required=True, help="Blackboard frame image path (OpenCV-readable)")
    p.add_argument("--audio", required=True, help="Aligned audio segment (.mp3 only; ffmpeg required)")
    p.add_argument("--pdf", default="output/teaching_feedback.pdf", help="Output PDF path")
    p.add_argument("--config", default=None, help="Optional YAML config path")
    args = p.parse_args()

    if Path(args.audio).suffix.lower() != ".mp3":
        print("Error: audio must be MP3 (.mp3).", file=sys.stderr)
        sys.exit(2)

    cfg = None
    if args.config and Path(args.config).is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    result = run_from_image_and_audio_files(args.image, args.audio, config=cfg, pdf_output=args.pdf)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
