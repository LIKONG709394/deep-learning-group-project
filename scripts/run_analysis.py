# Run the teaching pipeline from the terminal: either still image + MP3, or a video file.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from blackboard_analytics.pipeline import run_from_image_and_audio_files, run_from_video_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Blackboard frame image path (OpenCV-readable)")
    parser.add_argument("--audio", default=None, help="Aligned audio segment (.mp3 only; ffmpeg required)")
    parser.add_argument("--video", default=None, help="Video path; audio is extracted with ffmpeg")
    parser.add_argument("--pdf", default="output/teaching_feedback.pdf", help="Output PDF path")
    parser.add_argument("--config", default=None, help="Optional YAML config path")
    args = parser.parse_args()

    if args.video:
        if args.image or args.audio:
            print("Error: use either --video or --image + --audio.", file=sys.stderr)
            sys.exit(2)
    else:
        if not args.image or not args.audio:
            print("Error: provide --image + --audio, or provide --video.", file=sys.stderr)
            sys.exit(2)
        if Path(args.audio).suffix.lower() != ".mp3":
            print("Error: audio must be MP3 (.mp3).", file=sys.stderr)
            sys.exit(2)

    optional_settings = None
    if args.config and Path(args.config).is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            optional_settings = yaml.safe_load(f)

    if args.video:
        outcome = run_from_video_file(
            args.video,
            config=optional_settings,
            pdf_output=args.pdf,
        )
    else:
        outcome = run_from_image_and_audio_files(
            args.image,
            args.audio,
            config=optional_settings,
            pdf_output=args.pdf,
        )
    print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
