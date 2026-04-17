import argparse
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
import capture2
import audio2
import summarize2
import report2

app = FastAPI(
    title="Classroom Teaching Analysis API",
    description="Analyzes instructional video/audio/images using YOLO, Whisper, OCR, and DeepSeek.",
    version="2.0.0"
)

# --- Configuration Mocks for the Pipeline ---
class PipelineConfig:
    def __init__(self):
        # Audio & Whisper Settings
        self.model_size = "base"
        self.task = "transcribe"
        self.language = "en"
        self.pause_threshold_sec = 1.5
        
        # Capture Settings
        self.stride = 5
        self.weights_path = "best.pt" # Replace with your actual YOLO weights path
        self.iou = 0.45
        self.conf = 0.25
        
        # Summarize Settings
        self.ocr_config = {
            "engines": ["easyocr"], # Use easyocr as default to avoid paddleocr heavy setup
            "easy_langs": ["en"]
        }

class PipelineEnv:
    def __init__(self):
        self.device = summarize2._get_device()
        self.deepseek = os.getenv("DEEPSEEK_API_KEY", "")

# --- Core Pipeline Logic ---
def run_analysis_pipeline(
    video_path: Optional[str] = None, 
    image_path: Optional[str] = None, 
    audio_path: Optional[str] = None,
    output_dir: str = "./output"
) -> Dict[str, Any]:
    
    config = PipelineConfig()
    env = PipelineEnv()
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "status": "success",
        "transcription": "",
        "board_lines": [],
        "alignment_verdict": None,
        "report_pdf": None
    }

    print("\n[Pipeline] Starting Analysis Workflow...")

    # 1. PROCESS AUDIO (Extract & Transcribe)
    segments = []
    full_speech = ""
    target_audio = audio_path

    if video_path and not audio_path:
        print(f"[Pipeline] Extracting audio from {video_path}...")
        target_audio = audio2.extract_audio_ffmpeg(video_path, output_dir)

    if target_audio:
        print(f"[Pipeline] Transcribing audio with Whisper ({config.model_size})...")
        full_speech, raw_segments = audio2.extract_pause_from_audio(target_audio, config, env)
        segments = audio2.pause_audio(raw_segments, config)
        results["transcription"] = full_speech
        print(f"[Pipeline] Found {len(segments)} spoken segments.")

    # 2. PROCESS VISUALS (Capture & OCR)
    visual_pauses = []
    all_board_lines = []
    
    def on_pause_callback(visual_pause: dict):
        """Callback triggered by capture2 when a stable ROI is found."""
        print(f"  -> Captured stable board at {visual_pause['timestamp']//60}:{visual_pause['timestamp']%60}s")
        # Extract text via OCR
        extracted_vp = summarize2.extract(visual_pause, config.ocr_config)
        # Match text timeline with audio segments
        matched_vp = summarize2.match(extracted_vp, segments, {})
        visual_pauses.append(matched_vp)
        all_board_lines.extend(matched_vp.get("textlines", []))

    def on_pause_callback(visual_pause: dict):
        print(f"  -> Captured stable board at {visual_pause['timestamp']//60}:{visual_pause['timestamp']%60}s")
        # Extract text via OCR
        extracted_vp = summarize2.extract(visual_pause, config)
        # Match text timeline with audio segments
        matched_vp = summarize2.match(extracted_vp, segments, {})
        visual_pauses.append(matched_vp)
        all_board_lines.extend(matched_vp.get("textlines", []))

    if video_path:
        print(f"[Pipeline] Analyzing video frames for board text...")
        capture2.capture(
            video_path, 
            {"stride": config.stride, "weights_path": config.weights_path, "iou": config.iou, "conf": config.conf}, 
            on_pause_callback
        )
    elif image_path:
        print(f"[Pipeline] Processing single image...")
        frame_bgr = capture2.detect_single_bgr_image(image_path, config, )        
        # Mocking a visual pause for a single image
        mock_vp = {"timestamp": 0.0, "frame_bgr": frame_bgr, "areas": []}
        on_pause_callback(mock_vp)

    # Clean up duplicate lines across the whole session
    deduped_lines = summarize2._dedupe_subsumed_lines(all_board_lines)
    results["board_lines"] = deduped_lines

    # 3. REPORT & ALIGNMENT
    if deduped_lines or full_speech:
        print("[Pipeline] Evaluating semantic alignment between board and speech...")
        board_text_combined = " ".join(deduped_lines)
        
        # Semantic evaluation via SBERT
        alignment = report2.compare_board_and_speech(board_text_combined, full_speech)
        results["alignment_verdict"] = alignment

        print("[Pipeline] Generating PDF Report...")
        pdf_path = os.path.join(output_dir, "teaching_feedback.pdf")
        report2.tmp_save_pdf(pdf_path, {
            "board_lines": deduped_lines,
            "speech_text": full_speech,
            "alignment": alignment,
            "clarity": visual_pauses[-1]["areas"][0]["clarity"] if visual_pauses and visual_pauses[-1]["areas"] else None
        })
        results["report_pdf"] = pdf_path

    print("[Pipeline] Workflow Complete!\n")
    return results


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

        results = run_analysis_pipeline(video_path, image_path, audio_path, output_dir=temp_dir)
        
        # If you wanted to return the PDF directly, you could use FileResponse(results["report_pdf"])
        # For now, returning the JSON metrics.
        return JSONResponse(content={
            "transcription": results["transcription"],
            "board_lines": results["board_lines"],
            "alignment": results["alignment_verdict"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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
        results = run_analysis_pipeline(
            video_path=args.video,
            image_path=args.image,
            audio_path=args.audio,
            output_dir=args.output
        )
        print("\n--- Summary ---")
        import json
        # Remove PDF path from stdout json for cleaner printing
        if "report_pdf" in results:
            print(f"Report saved to: {results.pop('report_pdf')}")
            
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()