# Installation

## 1. Install PyTorch first (GPU)

**30-series (RTX 3060/3070/3080/3090) — CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**50-series (RTX 5070/5080/5090) — CUDA 12.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**CPU only (no GPU):**
```bash
pip install torch torchvision torchaudio
```

## 2. Install everything else
```bash
pip install -r requirements.txt
```

## 3. Install ffmpeg
- **Windows:** `winget install --id Gyan.FFmpeg`
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

## 4. Verify
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```