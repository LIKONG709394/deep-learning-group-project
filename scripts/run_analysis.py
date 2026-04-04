# Run the teaching pipeline from the terminal: still image + MP3 in, JSON on stdout.

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Blackboard frame image path (OpenCV-readable)")
    parser.add_argument("--audio", required=True, help="Aligned audio segment (.mp3 only; ffmpeg required)")
    parser.add_argument("--pdf", default="output/teaching_feedback.pdf", help="Output PDF path")
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    args = parser.parse_args()

    # We only accept MP3 here so the support story stays simple (one decoder path on the server).
    if Path(args.audio).suffix.lower() != ".mp3":
        print("Error: audio must be MP3 (.mp3).", file=sys.stderr)
        sys.exit(2)

    optional_settings = None
    if args.config and Path(args.config).is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            optional_settings = yaml.safe_load(f)

    outcome = run_from_image_and_audio_files(
        args.image,
        args.audio,
        config=optional_settings,
        pdf_output=args.pdf,
    )
    print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
