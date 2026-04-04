"""
Blackboard analytics HTTP API: upload image + MP3 -> pipeline -> JSON + PDF download.

Run from classroom_blackboard_analytics:
  uvicorn web.server:app --host 0.0.0.0 --port 8766
  uvicorn web.server:app --host 127.0.0.1 --port 8766
"""

from __future__ import annotations

import logging
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

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from blackboard_analytics.pipeline import run_from_frame_and_audio

logger = logging.getLogger(__name__)

SESSION_TTL_SEC = 3600
_session_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}


def _cleanup_sessions() -> None:
    now = time.time()
    with _session_lock:
        dead = [k for k, v in _sessions.items() if now - v.get("created", 0) > SESSION_TTL_SEC]
        for k in dead:
            meta = _sessions.pop(k, None)
            if meta and meta.get("workdir"):
                import shutil

                try:
                    shutil.rmtree(meta["workdir"], ignore_errors=True)
                except Exception:
                    pass


def _bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image; use JPG/PNG/WebP")
    return img


app = FastAPI(title="Blackboard analytics", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze")
async def api_analyze(
    image: UploadFile = File(..., description="Blackboard frame (JPG/PNG)"),
    audio: UploadFile = File(..., description="MP3 only (ffmpeg on PATH)"),
) -> JSONResponse:
    _cleanup_sessions()
    img_bytes = await image.read()
    aud_bytes = await audio.read()
    if len(img_bytes) < 32:
        raise HTTPException(400, "Image too small or empty")
    if len(aud_bytes) < 32:
        raise HTTPException(400, "Audio too small or empty")

    aud_name = audio.filename or ""
    if Path(aud_name).suffix.lower() != ".mp3":
        raise HTTPException(400, "Audio must be MP3 (.mp3 extension)")

    workdir = ROOT / "web_uploads" / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    img_suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"
    img_path = workdir / f"frame{img_suffix}"
    aud_path = workdir / "audio.mp3"
    pdf_path = workdir / "report.pdf"

    img_path.write_bytes(img_bytes)
    aud_path.write_bytes(aud_bytes)

    try:
        frame = _bytes_to_bgr(img_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    try:
        result = run_from_frame_and_audio(
            frame,
            str(aud_path),
            config=None,
            pdf_output=str(pdf_path),
        )
    except Exception as e:
        logger.exception("pipeline")
        raise HTTPException(500, f"Analysis failed: {e}") from e

    sid = uuid.uuid4().hex
    with _session_lock:
        _sessions[sid] = {
            "created": time.time(),
            "pdf_path": str(pdf_path) if pdf_path.is_file() else None,
            "workdir": str(workdir),
        }

    payload = {"session_id": sid, "result": result}
    return JSONResponse(content=jsonable_encoder(payload))


@app.get("/api/report/{session_id}")
async def api_report(session_id: str) -> FileResponse:
    _cleanup_sessions()
    with _session_lock:
        meta = _sessions.get(session_id)
    if not meta or not meta.get("pdf_path"):
        raise HTTPException(404, "Report not found or expired; run analyze again")
    p = Path(meta["pdf_path"])
    if not p.is_file():
        raise HTTPException(404, "PDF file missing")
    return FileResponse(
        path=str(p),
        filename="teaching_feedback.pdf",
        media_type="application/pdf",
    )


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "service": "blackboard-analytics"}


static_dir = WEB_DIR / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
