import logging
from typing import Any, Dict, List, Optional
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from paddle.version import cuda
import cv2
import numpy as np
from default import *
BLACKBOARD_VIDEO_FAST = 1
TROCR_DEFAULT = "microsoft/trocr-base-handwritten"
TROCR_PRINTED = "microsoft/trocr-base-printed"
_KF_PNG = re.compile(r"^kf_(\d+)\.png$", re.IGNORECASE)
class ENV:
    def __init__(self):
        self.device = getDevice()
        self.ffprobe = _find_ffprobe_executable()
        self.deepseek = getDeepseekAPIKey()
        
def getDeepseekAPIKey():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key == "": raise Exception(
        "DEEPSEEK_API_KEY is missed from env paths")
    return key

def getDevice():
    print("Cuda version:", cuda.get_version()
          if cuda.is_available() else "CUDA not available")
    return "cuda" if cuda.is_available() else "cpu"

def _find_ffmpeg_executable() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for key in ("FFMPEG_PATH", "BLACKBOARD_FFMPEG"):
        raw = os.environ.get(key, "").strip().strip('"')
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            return str(p)
        exe = p / "ffmpeg.exe"
        if exe.is_file():
            return str(exe)
    if sys.platform != "win32":
        return None
    for pf in (
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ):
        if not pf:
            continue
        for rel in (
            Path(pf) / "ffmpeg" / "bin" / "ffmpeg.exe",
            Path(pf) / "ffmpeg" / "ffmpeg.exe",
        ):
            if rel.is_file():
                return str(rel)
    la = os.environ.get("LOCALAPPDATA", "")
    if la:
        winget_link = Path(la) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
        if winget_link.is_file():
            return str(winget_link)
    prof = os.environ.get("USERPROFILE", "")
    if prof:
        scoop = Path(prof) / "scoop" / "shims" / "ffmpeg.exe"
        if scoop.is_file():
            return str(scoop)
    return None
def _ensure_ffmpeg_on_path() -> bool:
    if shutil.which("ffmpeg"):
        return True
    exe = _find_ffmpeg_executable()
    if not exe:
        return False
    folder = str(Path(exe).resolve().parent)
    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    logger.info("Prepended ffmpeg directory to process PATH: %s", folder)
    return shutil.which("ffmpeg") is not None
def _find_ffprobe_executable() -> Optional[str]:
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    parent = Path(ffmpeg).resolve().parent
    for name in ("ffprobe.exe", "ffprobe"):
        cand = parent / name
        if cand.is_file():
            return str(cand)
    return None