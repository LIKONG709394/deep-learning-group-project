# deep-learning-group-project

This repository contains **classroom blackboard analytics**: OCR, speech alignment, optional DeepSeek filtering, and a small web UI.

All application code and documentation live under [`classroom_blackboard_analytics/`](classroom_blackboard_analytics/README.md).

## Implementation map (main features)

- **Hugging Face / PyTorch**: `model_cache.py` sets `USE_TF=0`, `USE_TORCH=1` so `transformers` stays on PyTorch and avoids `tf-keras` issues. Cache dirs use `HF_HOME` / `HUGGINGFACE_HUB_CACHE` (not deprecated `TRANSFORMERS_CACHE`). `requirements.txt` pins `torch>=2.1`, `torchvision>=0.16`, plus EasyOCR and Paddle stack.
- **Video / OCR**: `config/default.yaml` — `video.fast_mode`, `trocr.ocr_engine` (e.g. EasyOCR for slides), printed preference and TrOCR handwriting fallback; `pipeline.py` and `module_video_keyframes.py` wire keyframes, harvest, and `ocr_source` labels. `module_a_alt_ocr.py` implements EasyOCR/Paddle line clustering.
- **DeepSeek**: `module_d_deepseek.py` — alignment plus `run_deepseek_filter_board_lines` after Whisper; `pipeline.py` exposes `deepseek_board_line_filter`; web UI shows the line-filter block (`app.js` / `index.html` cache bust `v=4`).
- **Python 3.9**: `web/server.py` uses `Optional[UploadFile]` for multipart fields (not `X | None` syntax).
