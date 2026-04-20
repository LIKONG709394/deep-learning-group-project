"""
Classroom Teaching Analysis Pipeline
====================================

This module serves as the central orchestrator for the Teaching Analysis System. 
It integrates computer vision (YOLO), automated speech recognition (Whisper), 
handwriting/text OCR (TrOCR/EasyOCR), and LLM-based analysis (DeepSeek) into 
a unified processing workflow.

The pipeline supports both a RESTful API (via FastAPI) and a CLI interface to 
process instructional videos and generate comprehensive teaching reports in 
PDF, Word, and JSON formats.

Key Features:
    * Asynchronous task management with thread-safe status tracking.
    * End-to-end processing: Video -> Audio/Frames -> Text -> Feedback.
    * Integration of YOLOv8 for blackboard/whiteboard detection.
    * Semantic alignment between spoken content and board-written text.
    * Modular error handling to ensure partial results are saved on failure.

Dependencies:
    * FastAPI & Uvicorn: For API hosting and request handling.
    * capture2, audio2, summarize2, report2: Internal processing modules.
    * torch & ultralytics: For deep learning model execution.

Author: [Lai Tsz Yeung/Group J]
Date: 2026
License: MIT
"""
import argparse
import logging
import threading
import time
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse

# Import the refined modules
import capture
import audio
import summarize
import report

app = FastAPI(
    title="Classroom Teaching Analysis API",
    description="Analyzes instructional video/audio/images using YOLO, Whisper, OCR, and DeepSeek.",
    version="2.0.0"
)

tasks_lock = threading.Lock()
tasks = []
        

# --- Configuration Mocks for the Pipeline ---
class PipelineConfig:
    def __init__(self):
        # Audio & Whisper Settings
        self.model_size = "base"
        self.task = "transcribe"
        self.language = "en"
        self.pause_threshold_sec = 1.5
        
        # Capture Settings
        self.stride = 50
        self.weights_path = "C:/Users/user/projects/deep-learning/dcatcher/text_blackboard_detector_best.pt" # Replace with your actual YOLO weights path
        self.iou = 0.45
        self.conf = 0.25
        
        # Summarize Settings
        self.ocr_config = {
            "engines": ["easyocr","paddleocr"], # Use easyocr as default to avoid paddleocr heavy setup
            "easy_langs": ["en"]
        }

class PipelineEnv:
    def __init__(self):
        self.device = summarize._get_device()
        self.deepseek = os.getenv("DEEPSEEK_API_KEY", "sk-6a304dfb56ec43958e10ed366b8b961f")

def receiver_analysis_pipeline(
    video_path: Optional[str] = None, 
    image_path: Optional[str] = None, 
    audio_path: Optional[str] = None,
    output_dir: str = "./output"
) -> int:
    config = PipelineConfig()
    env = PipelineEnv()
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "status": "success",
        "transcription": "",
        "segments":[],
        "board_lines": [],        
        "alignment_verdict": None,
        "deepseek_alignment_verdict": None,
        "visual_pauses": [],
        "all_board_lines": []        
    }

    with tasks_lock:
        tasks.append(results)

    result_id = len(tasks)-1

    thread = threading.Thread(target=run_analysis_pipeline, args=(
        result_id, config, env, 
        video_path, image_path, audio_path, output_dir,))
    thread.start()

    return result_id
    

# --- Core Pipeline Logic ---
def run_analysis_pipeline(
    result_id:  int,
    config,
    env,
    video_path: Optional[str] = None, 
    image_path: Optional[str] = None, 
    audio_path: Optional[str] = None,
    output_dir: str = "./output"
) -> Dict[str, Any]:  

    logging.info("\n[Pipeline] Starting Analysis Workflow...")

    # 1. PROCESS AUDIO (Extract & Transcribe)
    segments = []
    full_speech = ""
    target_audio = audio_path

    if video_path and not audio_path:
        logging.info(f"[Pipeline] Extracting audio from {video_path}...")
        target_audio = audio.extract_audio_ffmpeg(video_path, output_dir)

    if target_audio:
        logging.info(f"[Pipeline] Transcribing audio with Whisper ({config.model_size})...")
        full_speech, raw_segments = audio.extract_pause_from_audio(target_audio, config, env)
        segments = audio.pause_audio(raw_segments, config)
        with tasks_lock:
            tasks[result_id]["transcription"] = full_speech
            tasks[result_id]["segments"] = segments
        logging.info(f"[Pipeline] Found {len(segments)} spoken segments.")

    # 2. PROCESS VISUALS (Capture & OCR)
    working_ocr_workers = 0
    worked_ocr_workers_lock = threading.Lock()
    worked_ocr_workers = []
    
    
    def on_pause_callback(visual_pause: dict):
        nonlocal working_ocr_workers
        """Callback triggered by capture2 when a stable ROI is found."""
        logging.info(f"  -> Captured stable board at {int(visual_pause['timestamp']//60):02d}:{int(visual_pause['timestamp']%60):02d}s")
        def fuc(visual_pause, worked_ocr_workers_lock):
            extracted_vp = summarize.extract(visual_pause, config.ocr_config)
            # Match text timeline with audio segments
            matched_vp = summarize.match(extracted_vp, segments, {})
            with tasks_lock:
                tasks[result_id]["visual_pauses"].append(matched_vp)
                tasks[result_id]["all_board_lines"].extend(matched_vp.get("textlines", []))
            with worked_ocr_workers_lock:
                worked_ocr_workers.append("Done")
        thread = threading.Thread(target=fuc, args=(visual_pause, worked_ocr_workers_lock,))
        working_ocr_workers +=1
        thread.start()
        # Extract text via OCR        

    if video_path:
        logging.info(f"[Pipeline] Analyzing video frames for board text...")
        capture.capture(
            video_path, 
            {"stride": config.stride, "weights_path": config.weights_path, "iou": config.iou, "conf": config.conf}, 
            on_pause_callback
        )
    elif image_path:
        logging.info(f"[Pipeline] Processing single image...")
        frame_bgr = capture.detect_single_bgr_image(image_path, config, )        
        # Mocking a visual pause for a single image
        mock_vp = {"timestamp": 0.0, "frame_bgr": frame_bgr, "areas": []}
        on_pause_callback(mock_vp)

    while True:
        time.sleep(0.001)
        if working_ocr_workers == len(worked_ocr_workers): break

    # Clean up duplicate lines across the whole session    
    deduped_lines = summarize._dedupe_subsumed_lines(tasks[result_id]["all_board_lines"])
    with tasks_lock:
        tasks[result_id]["board_lines"] = deduped_lines

    for visual_pause in tasks[result_id]["visual_pauses"]:
        res = []
        for textline in visual_pause["textlines"]:
            if textline.strip().strip().casefold() in res: continue
            res.append(textline.strip().strip().casefold())        
        visual_pause["textlines"] = res

    # 3. REPORT & ALIGNMENT
    if deduped_lines or full_speech:
        logging.info("[Pipeline] Evaluating semantic alignment between board and speech...")
        board_text_combined = " ".join(deduped_lines)
        
        # Semantic evaluation via SBERT
        alignment = report.compare_board_and_speech(board_text_combined, full_speech)
        with tasks_lock:
            tasks[result_id]["alignment_verdict"] = alignment

        alignment = report.deepseek_alignment_evaluate(board_text_combined, full_speech, config, env)
        with tasks_lock:
            tasks[result_id]["deepseek_alignment_verdict"] = alignment
        

        logging.info("[Pipeline] Generating PDF Report...")
        
        with tasks_lock:
            tasks[result_id]["status"] = "succeed"

    logging.info("[Pipeline] Workflow Complete!\n")
    return tasks[result_id]


# ==========================================
# POST API ENDPOINT (FastAPI)
# ==========================================

@app.post("/analyze")
async def analyze_endpoint(
    video: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    if not video and not (image and audio):
        raise HTTPException(status_code=400, detail="Provide a video, OR both image and audio.")

    temp_dir = tempfile.mkdtemp()
    
    try:
        video_path, image_path, audio_path = None, None, None

        if video:
            video_path = os.path.join(temp_dir, video.filename)
            with open(video_path, "wb") as f: shutil.copyfileobj(video.file, f)
        if image:
            image_path = os.path.join(temp_dir, image.filename)
            with open(image_path, "wb") as f: shutil.copyfileobj(image.file, f)
        if audio:
            audio_path = os.path.join(temp_dir, audio.filename)
            with open(audio_path, "wb") as f: shutil.copyfileobj(audio.file, f)

        result_id = receiver_analysis_pipeline(video_path, image_path, audio_path, output_dir=temp_dir)
        
        # If you wanted to return the PDF directly, you could use FileResponse(results["report_pdf"])
        # For now, returning the JSON metrics.
        return JSONResponse(content={
            "result_id": result_id
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{result_id}")
async def get_result(result_id: int):
    update = None
    with tasks_lock:
        if result_id < 0 or result_id >= len(tasks):
            raise HTTPException(status_code=404, detail="Result ID not found.")

        update = dict(tasks[result_id])
    
    filter_visual_pauses = {}
    for pause in update["visual_pauses"]:
        print(pause["spoken_near"])
        filter_visual_pauses[str(pause["timestamp"])+"_"+(str(pause["spoken_near"]))] = pause["textlines"]

    packed_segments = {}
    for seg in update["segments"]:
        packed_segments[str(seg["start_sec"])] = seg["text"]

    return JSONResponse(content={
        "status": update["status"],
        "transcription": packed_segments,
        "board_lines": update["board_lines"],        
        "alignment_verdict": update["alignment_verdict"],
        "deepseek_alignment_verdict": update["deepseek_alignment_verdict"],       
        "visual_pauses": filter_visual_pauses,
    })

pdf_lock = threading.Lock()
word_lock = threading.Lock()

@app.get("/download/{result_id}")
async def download_report(result_id: int, fmt: str = "pdf"):
    with tasks_lock:
        if result_id < 0 or result_id >= len(tasks):
            raise HTTPException(status_code=404, detail="Result ID not found.")
        task = dict(tasks[result_id])

    if task.get("status") != "succeed":
        raise HTTPException(status_code=409, detail="Report is not ready yet.")

    payload = {
        "board_lines": task.get("board_lines", []),
        "speech_text": task.get("transcription", ""),
        "alignment": task.get("alignment_verdict"),
        "deepseek_alignment_verdict": task.get("deepseek_alignment_verdict"),
        "clarity": task["visual_pauses"][-1]["areas"][0]["clarity"]
            if task.get("visual_pauses") and task["visual_pauses"][-1].get("areas")
            else None,
    }

    fmt = (fmt or "pdf").strip().lower()

    if fmt == "pdf":
        with pdf_lock:
            output_path = os.path.join("./output", f"teaching_feedback_{time.time()}_{result_id}.pdf")
            result = report.tmp_save_pdf(output_path, payload)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        file_path = result.get("pdf_path")
        media_type = "application/pdf"
        download_name = f"teaching_feedback_{result_id}.pdf"

    elif fmt == "docx":
        with word_lock:
            output_path = os.path.join("./output", f"teaching_feedback_{time.time()}_{result_id}.docx")
            result = report.tmp_save_word(output_path, payload)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        file_path = result.get("word_path")
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        download_name = f"teaching_feedback_{result_id}.docx"

    elif fmt == "json":
        # Ensure the output directory exists
        os.makedirs("./output", exist_ok=True)
        
        # Consistent naming convention using result_id and timestamp
        output_path = os.path.join("./output", f"teaching_feedback_{time.time()}_{result_id}.json")
        
        # Assuming report.tmp_save_json follows the same signature as tmp_save_word
        result = report.tmp_save_json(output_path, payload)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        file_path = result.get("json_path")
        media_type = "application/json"
        download_name = f"teaching_feedback_{result_id}.json"

    else:
        raise HTTPException(status_code=400, detail="fmt must be 'pdf' or 'docx'.")

    if not file_path:
        raise HTTPException(status_code=404, detail=f"{fmt.upper()} report not available.")

    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise HTTPException(status_code=404, detail=f"Report file not found on disk: {file_path}")

    return FileResponse(
        path=str(path_obj),
        media_type=media_type,
        filename=download_name,
    )


# ==========================================
# CLI ENTRY POINT (Argparse)
# ==========================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Classroom Teaching Analysis Pipeline")
    
    parser.add_argument("--api", action="store_true", help="Start the FastAPI web server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server (default: 8000).")
    parser.add_argument("--video", default=None, help="Path to video file.")
    parser.add_argument("--image", default=None, help="Path to blackboard image.")
    parser.add_argument("--audio", default=None, help="Path to speech audio file.")
    parser.add_argument("--output", default="./output", help="Directory to save PDF reports.")

    args = parser.parse_args()

    # Route 1: Run Web Server
    if args.api:
        print(f"Starting API server on port {args.port}...")
        uvicorn.run("pipeline:app", host="0.0.0.0", port=args.port, reload=True)
        return

    # Route 2: Run CLI
    if not args.video and not (args.image and args.audio):
        print("Error: Provide either --video OR both --image and --audio.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    try:
        start = time.perf_counter()
        results_id = receiver_analysis_pipeline(
            video_path=args.video,
            image_path=args.image,
            audio_path=args.audio,
            output_dir=args.output
        )
        print("\n--- Summary ---")
        update = None
        while True:
            time.sleep(2)
            with tasks_lock:
                update = tasks[results_id]
            print(update["all_board_lines"])
            print(update["status"])
            if update["status"] == "succeed": break
        import json
        # Remove PDF path from stdout json for cleaner printing
        if "report_pdf" in update:
            print(f"Report saved to: {update.pop('report_pdf')}")
            
        #print(json.dumps(update, indent=2))
        end = time.perf_counter()
        elapsed = end - start
        print(f"Elapsed time: {elapsed:.4f} seconds")


        
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()