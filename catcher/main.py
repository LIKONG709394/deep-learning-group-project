"""
Catcher — Blackboard Teaching Feedback Pipeline
Author: Lai Tsz Yeung (Thomas), Tam Chun To (Joe)   
Course: INT4097 Deep Learning for Computer Vision and Education 2025/26 S2

Entry point. Run with:
    python main.py --video lecture.mp4 --pdf output/feedback.pdf
    python main.py --image board.jpg --audio lecture.mp3 --pdf output/feedback.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.pipeline import startpipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catcher",
        description="Run the Catcher teaching-feedback pipeline on image+audio or video input.",
    )

    parser.add_argument(
        "--image",
        default=None,
        help="Blackboard frame image path (OpenCV-readable).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Aligned audio path for image mode (.mp3 recommended).",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Video path; audio is extracted automatically.",
    )
    parser.add_argument(
        "--pdf",
        default="output/teaching_feedback.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    has_video = bool(args.video)
    has_image = bool(args.image)
    has_audio = bool(args.audio)

    if has_video and (has_image or has_audio):
        raise SystemExit("Error: use either --video or --image + --audio.")

    if not has_video and not (has_image and has_audio):
        raise SystemExit("Error: provide --image + --audio, or provide --video.")

    if has_image and not Path(args.image).exists():
        raise SystemExit(f"Error: image file not found: {args.image}")

    if has_audio and not Path(args.audio).exists():
        raise SystemExit(f"Error: audio file not found: {args.audio}")

    if has_video and not Path(args.video).exists():
        raise SystemExit(f"Error: video file not found: {args.video}")

    if has_audio and Path(args.audio).suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        raise SystemExit("Error: unsupported audio type. Use .mp3, .wav, .m4a, .flac, or .ogg.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)

        outcome = startpipeline(
            videopath=args.video,
            configpath=args.config,
            pdfpath=args.pdf,
            imgpath=args.image,
            audiopath=args.audio,
        )

        if args.pretty:
            print(json.dumps(outcome, ensure_ascii=False, indent=2, default=str))
        else:
            print(json.dumps(outcome, ensure_ascii=False, default=str))

    except KeyboardInterrupt:
        print(json.dumps({
            "ok": False,
            "error": "Interrupted by user."
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e)
        }, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()