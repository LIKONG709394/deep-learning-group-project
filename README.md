# Classroom blackboard analytics

Python pipeline and small web UI for:

- **Module A**: Blackboard ROI (YOLOv8 optional, heuristic fallback) + handwritten OCR (TrOCR)
- **Module B**: Clarity score (Laplacian variance + stroke-width variance) + optional heatmap
- **Module C**: Speech-to-text (OpenAI Whisper); **MP3 requires ffmpeg**
- **Module D**: Board vs speech semantic similarity (multilingual SBERT) + keyword overlap + Deepseek Analyse
- **Module E**: PDF report

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on `PATH` for MP3 (e.g. `winget install Gyan.FFmpeg`)
- GPU optional (CPU works, slower)

## Install

```bash
cd classroom_blackboard_analytics
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## CLI

```bash
python scripts/run_analysis.py --image frame.jpg --audio clip.mp3 --pdf output/report.pdf --config config/default.yaml
```

## Web UI

```bash
python scripts/run_web.py
```

Open `http://127.0.0.1:8766`. Bind `0.0.0.0` by default for LAN access; use `--local` for localhost only. The web UI always uses built-in defaults; pass `--config` to the CLI if you need a custom YAML.

### Public HTTPS (Cloudflare quick tunnel)

```bash
# Install cloudflared first, then:
python scripts/run_web_public_tunnel.py
```

