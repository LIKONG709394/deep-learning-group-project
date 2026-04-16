# Catcher — Blackboard Teaching Feedback Pipeline

**Author:** Lai Tsz Yeung (Thomas), Tam Chun To (Joe)   
**Course:** INT4097 Deep Learning for Computer Vision and Education 2025/26 S2  

---

## Overview

Catcher processes a classroom blackboard video (or a still image + audio clip) and
produces a structured teaching feedback PDF report covering:

- **Board Text** — OCR extraction via EasyOCR / PaddleOCR
- **Handwriting Clarity** — scored using Laplacian variance and stroke width analysis
- **Speech Transcript** — transcribed with OpenAI Whisper
- **Board–Speech Alignment** — semantic similarity via SBERT cosine scoring
- **PDF Report** — generated with ReportLab

---

## Architecture

- `Catcher uses a **single producer-consumer pipeline** with two threads, replacing`
- `the fragmented multi-pass loop design found in conventional implementations.`
- `This improves readability, reduces redundant video I/O, and makes each stage`
- `independently testable.`
- ``
- `Video File`
- `│`
- `┌───▼────────────┐`
- `│ Capture Loop │ ← producer thread: YOLO ROI detection → visualPause queue`
- `└───┬────────────┘`
- `│ visualPause queue`
- `┌───▼────────────┐`
- `│ Textual Loop │ ← consumer thread: OCR per pause → dedup → normalise`
- `└───┬────────────┘`
- `│`
- `┌───▼────────────┐`
- `│ Whisper ASR │ ← speech segments + token extraction`
- `│ SBERT Align │ ← cosine similarity: board text ↔ speech`
- `│ PDF Report │ ← ReportLab output`
- `└────────────────┘`


---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+, ffmpeg on PATH, and a CUDA GPU (CPU fallback supported).

### 2. Set API key (optional — for DeepSeek line filtering)

```bash
export DEEPSEEK_API_KEY=your_key_here
```

### 3. Run

```bash
# Video mode (audio extracted automatically)
python main.py --video path/to/lecture.mp4 --pdf output/feedback.pdf

# Still image + audio mode
python main.py --image board.jpg --audio lecture.mp3 --pdf output/feedback.pdf
```

---

## Project Structure

- `catcher/`
- `├── src/`
- `│ ├── pipeline.py ← main pipeline orchestrator`
- `│ ├── capture_loop.py ← producer thread (video → visualPause queue)`
- `│ ├── textual_loop.py ← consumer thread (OCR per pause)`
- `│ ├── ocr.py ← EasyOCR / PaddleOCR wrappers`
- `│ ├── clarity.py ← Laplacian + stroke width clarity scorer`
- `│ ├── whisper_asr.py ← Whisper transcription`
- `│ ├── alignment.py ← SBERT semantic alignment`
- `│ ├── deepseek_filter.py ← DeepSeek OCR line filter (optional)`
- `│ └── report.py ← PDF report generator`
- `├── web/ ← Web UI (see web/README.md)`
- `├── config/`
- `│ └── default.yaml ← pipeline configuration`
- `├── tests/`
- `│ └── test_pipeline.py ← smoke tests`
- `├── output/ ← generated PDFs (gitignored)`
- `├── main.py ← CLI entry point`
- `└── requirements.txt`

---

## Configuration

Edit `config/default.yaml` to tune:
- OCR engine (`easyocr` / `paddleocr` / `trocr`)
- Whisper model size (`tiny` / `base` / `small`)
- Clarity thresholds
- DeepSeek filter toggle

---

## Web UI

The `web/` folder contains the interactive dashboard for this pipeline.  
See [`web/README.md`](web/README.md) for details.

---

## Design Decisions

| Decision | Reason |
|---|---|
| Producer-consumer over multi-pass loops | Eliminates 7+ redundant `VideoCapture` open/read/close cycles |
| Per-module files | Each stage independently testable and replaceable |
| YAML config | Tunable without code changes — evaluators can run without editing source |
| Single `main.py` entry | One command runs everything — meets submission requirement |