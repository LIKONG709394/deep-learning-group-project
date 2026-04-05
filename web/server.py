# Small HTTP front-end: browser can send a photo + MP3, or a video file.
# The current UI still posts photo + MP3; video support is exposed at the backend API layer.

from __future__ import annotations

import copy
import logging
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import yaml

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from blackboard_analytics.pipeline import run_from_frame_and_audio, run_from_video_file
from blackboard_analytics.config_loader import default_config_path, load_pipeline_config  # noqa: E402
from blackboard_analytics.pipeline import run_from_frame_and_audio, run_from_video_file  # noqa: E402

logger = logging.getLogger(__name__)

# Web UI previously passed config=None, so default.yaml (e.g. whisper.language) was ignored.
_PIPELINE_CONFIG = load_pipeline_config()
if not _PIPELINE_CONFIG:
    logger.warning("Pipeline YAML missing or empty; using built-in defaults only (%s)", default_config_path())

# After an hour we forget the session and delete the temp folder so old uploads don’t pile up.
PDF_TTL_SEC = 3600
_lock = threading.Lock()
_pending: Dict[str, Dict[str, Any]] = {}
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"


def _expire_old_uploads() -> None:
    now = time.time()
    with _lock:
        too_old = [
            session_id
            for session_id, meta in _pending.items()
            if now - meta.get("created", 0) > PDF_TTL_SEC
        ]
        for session_id in too_old:
            meta = _pending.pop(session_id, None)
            folder = meta.get("workdir") if meta else None
            if folder:
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                except Exception:
                    pass


def decode_uploaded_photo(file_bytes: bytes) -> np.ndarray:
    raw = np.frombuffer(file_bytes, dtype=np.uint8)
    picture = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if picture is None:
        raise ValueError("Could not open this file as an image (try JPG, PNG, or WebP).")
    return picture


def load_default_config() -> dict:
    if not DEFAULT_CONFIG_PATH.is_file():
        logger.warning("Default config file not found: %s", DEFAULT_CONFIG_PATH)
        return {}
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return copy.deepcopy(data)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze")
async def api_analyze(
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
) -> JSONResponse:
    _expire_old_uploads()
    loaded_config = load_default_config()

    using_video = video is not None
    using_image_audio = image is not None or audio is not None
    if using_video and using_image_audio:
        raise HTTPException(400, "Send either video, or image + audio, not both")
    if not using_video and not (image is not None and audio is not None):
        raise HTTPException(400, "Send either video, or image + audio")

    # Each request gets its own folder so two people analyzing at once don’t overwrite each other.
    scratch_dir = ROOT / "web_uploads" / uuid.uuid4().hex
    scratch_dir.mkdir(parents=True, exist_ok=True)
    path_to_pdf = scratch_dir / "report.pdf"

    try:
        if using_video and video is not None:
            video_bytes = await video.read()
            if len(video_bytes) < 32:
                raise HTTPException(400, "Video too small or empty")
            suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
            path_to_video = scratch_dir / f"video{suffix}"
            path_to_video.write_bytes(video_bytes)
            analysis = run_from_video_file(
                str(path_to_video),
                config=loaded_config,
                config=_PIPELINE_CONFIG,
                pdf_output=str(path_to_pdf),
            )
        else:
            assert image is not None and audio is not None
            photo_bytes = await image.read()
            sound_bytes = await audio.read()
            if len(photo_bytes) < 32:
                raise HTTPException(400, "Image too small or empty")
            if len(sound_bytes) < 32:
                raise HTTPException(400, "Audio too small or empty")

            audio_filename = audio.filename or ""
            if Path(audio_filename).suffix.lower() != ".mp3":
                raise HTTPException(400, "Audio must be MP3 (.mp3 extension)")

            suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"
            path_to_photo = scratch_dir / f"frame{suffix}"
            path_to_mp3 = scratch_dir / "audio.mp3"

            path_to_photo.write_bytes(photo_bytes)
            path_to_mp3.write_bytes(sound_bytes)

            try:
                blackboard_frame = decode_uploaded_photo(photo_bytes)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e

            analysis = run_from_frame_and_audio(
                blackboard_frame,
                str(path_to_mp3),
                config=loaded_config,
                pdf_output=str(path_to_pdf),
            )
    except HTTPException:
        raise
                config=_PIPELINE_CONFIG,
                pdf_output=str(path_to_pdf),
            )
    except Exception as e:
        logger.exception("pipeline")
        raise HTTPException(500, f"Analysis failed: {e}") from e

    download_id = uuid.uuid4().hex
    with _lock:
        _pending[download_id] = {
            "created": time.time(),
            "pdf_path": str(path_to_pdf) if path_to_pdf.is_file() else None,
            "workdir": str(scratch_dir),
        }

    return JSONResponse(
        content=jsonable_encoder({"session_id": download_id, "result": analysis})
    )


@app.get("/api/report/{session_id}")
async def api_report(session_id: str) -> FileResponse:
    _expire_old_uploads()
    with _lock:
        remembered = _pending.get(session_id)
    if not remembered or not remembered.get("pdf_path"):
        raise HTTPException(404, "Report not found or expired; run analyze again")
    pdf_on_disk = Path(remembered["pdf_path"])
    if not pdf_on_disk.is_file():
        raise HTTPException(404, "PDF file missing")
    return FileResponse(
        path=str(pdf_on_disk),
        filename="teaching_feedback.pdf",
        media_type="application/pdf",
    )


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/diagnostics")
async def api_diagnostics() -> Dict[str, Any]:
    """Quick environment check for debugging (no model load)."""

    def _try(mod: str) -> str:
        try:
            __import__(mod)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    video = _PIPELINE_CONFIG.get("video") if isinstance(_PIPELINE_CONFIG.get("video"), dict) else {}
    return {
        "ok": True,
        "python_executable": sys.executable,
        "default_config_path": str(default_config_path()),
        "config_yaml_loaded": bool(_PIPELINE_CONFIG),
        "video_fast_mode": bool(video.get("fast_mode")),
        "ocr_diagnostics_enabled": bool(video.get("ocr_diagnostics", True)),
        "optional_imports": {
            "uvicorn": _try("uvicorn"),
            "ultralytics": _try("ultralytics"),
            "whisper": _try("whisper"),
            "sentence_transformers": _try("sentence_transformers"),
            "transformers": _try("transformers"),
            "torch": _try("torch"),
        },
        "hint": "After /api/analyze (video), inspect result.ocr_diagnostics for localization vs OCR hints.",
    }


_static = WEB_DIR / "static"
if _static.is_dir():
    app.mount("/", StaticFiles(directory=str(_static), html=True), name="ui")
