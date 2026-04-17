import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict, Callable

import cv2
import numpy as np
import whisper

# Note: make sure this module exists in your project structure
from summarize2 import get_preplexity_input

SILENCE_THERSHOLD = 0.0004
SILENCE_DURATION_SEC = 1.0
MIN_SEG_SEC = 2.0
MAX_SEG_SEC = 20.0
ANALYSIS_WINDOW_SEC = 0.1
SKIP_ENERGY_THERSHOLD = 0.00012

KF_PNG = re.compile(r"^kf_(\d+)\.png$", re.IGNORECASE)

# --- INTERNAL PATH & FFMPEG HELPERS ---

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
    return shutil.which("ffmpeg") is not None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_model_cache_root() -> Path:
    return _get_project_root() / ".model_cache"


def _get_hf_home() -> Path: 
    return _get_model_cache_root() / "huggingface"


def _get_hf_hub_cache_dir() -> Path:
    return _get_hf_home() / "hub"


def _get_transformers_cache_dir() -> Path:
    return _get_hf_home() / "transformers"


def _get_sentence_transformers_cache_dir() -> Path:
    return _get_model_cache_root() / "sentence_transformers"


def _get_whisper_cache_dir() -> Path:
    return _get_model_cache_root() / "whisper"


def _get_torch_home() -> Path:
    return _get_model_cache_root() / "torch"


def _has_hf_repo_cache(model_name: str) -> bool:
    repo_dir = _get_hf_hub_cache_dir() / f"models--{model_name.replace('/', '--')}"
    return repo_dir.is_dir() and (repo_dir / "snapshots").is_dir()


_WHISPER_LOCK = threading.Lock()
_WHISPER_MODELS = {}

def load_whisper_model(model_size: str) -> Any:
    """
    Loads and caches an OpenAI Whisper model.

    Args:
        model_size (str): Size of the model (e.g. 'base', 'small', 'medium').

    Returns:
        Any: The instantiated Whisper model.
    """
    if whisper is None:
        raise RuntimeError("Install openai-whisper")
    with _WHISPER_LOCK:
        cached = _WHISPER_MODELS.get(model_size)
        if cached is not None:
            return cached
        try:
            model = whisper.load_model(model_size, download_root=str(_get_whisper_cache_dir()))
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e
        _WHISPER_MODELS[model_size] = model
        return model

# Support for internal references bridging the same logic
_load_whisper_model = load_whisper_model


# --- AUDIO LOADING & PROCESSING HELPERS ---

def _resample_to_16k(audio: np.ndarray, fr: int) -> np.ndarray:
    if fr == 16000 or len(audio) == 0:
        return audio.astype(np.float32)
    try:
        from scipy import signal
        num = max(1, int(len(audio) * 16000 / fr))
        return signal.resample(audio, num).astype(np.float32)
    except ImportError as e:
        raise RuntimeError(
            f"Audio is {fr} Hz; install scipy to resample to 16000 Hz, or use ffmpeg via Whisper"
        ) from e


def _load_wav_stdlib_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sw != 2:
        raise ValueError("Not 16-bit PCM")
    data = np.frombuffer(raw, dtype="<i2").copy()
    if channels == 2:
        data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
    audio = (data.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    return _resample_to_16k(audio, fr)


def _load_wav_scipy_16k_mono(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    fr, data = wavfile.read(str(path))
    audio = np.asarray(data)
    return _resample_to_16k(audio, int(fr))


def _load_soundfile_16k_mono(path: Path) -> np.ndarray:
    import soundfile as sf
    data, fr = sf.read(str(path), always_2d=False, dtype="float32")
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = np.clip(data.astype(np.float32), -1.0, 1.0)
    return _resample_to_16k(audio, int(fr))


def _get_audio_loaders(path: Path) -> Optional[List[Callable]]:
    print(f"Opening audio {path}")
    format = path.suffix.lower()
    print(f"Detected audio format {format}")
    if format in (".wav", ".wave"):
        return [_load_wav_stdlib_16k_mono, _load_wav_scipy_16k_mono]
    elif format in (".flac", ".ogg", ".oga"):
        return [_load_soundfile_16k_mono]
    return None


def _load_audio(path: Path, loaders: List[Callable]) -> Optional[np.ndarray]:    
    if not loaders:
        return None
    for loader in loaders:
        try: 
            return loader(path)
        except Exception: 
            pass
    return None


def _segment_audio_by_silence(
    audio: np.ndarray,
    *,
    sample_rate: int,
    silence_threshold: float,
    silence_duration_sec: float,
    min_segment_sec: float,
    max_segment_sec: float,
    analysis_window_sec: float,
) -> List[Tuple[np.ndarray, float, float]]:
    if audio is None or len(audio) == 0 or sample_rate <= 0:
        return []

    audio = np.asarray(audio, dtype=np.float32)
    window_sec = max(0.01, float(analysis_window_sec))
    frame_samples = max(1, int(sample_rate * window_sec))
    silence_frames_needed = max(1, int(float(silence_duration_sec) / window_sec))
    min_segment_sec = max(0.0, float(min_segment_sec))
    max_segment_sec = max(min_segment_sec if min_segment_sec > 0 else 0.1, float(max_segment_sec))

    segments: List[Tuple[np.ndarray, float, float]] = []
    buffered_frames: List[np.ndarray] = []
    buffered_start = 0
    buffered_duration = 0.0
    silent_frame_count = 0
    has_speech = False

    idx = 0
    while idx < len(audio):
        frame = audio[idx : idx + frame_samples]
        if len(frame) == 0:
            break

        if not buffered_frames:
            buffered_start = idx
        buffered_frames.append(frame)
        buffered_duration += len(frame) / sample_rate

        if _rms_energy(frame) >= float(silence_threshold):
            silent_frame_count = 0
            has_speech = True
        else:
            silent_frame_count += 1

        should_flush = False
        if has_speech and silent_frame_count >= silence_frames_needed and buffered_duration >= min_segment_sec:
            should_flush = True
        if buffered_duration >= max_segment_sec:
            should_flush = True

        if should_flush and buffered_frames:
            segment_audio = np.concatenate(buffered_frames)
            start_sec = buffered_start / sample_rate
            duration_sec = len(segment_audio) / sample_rate
            segments.append((segment_audio, float(start_sec), float(duration_sec)))
            buffered_frames = []
            buffered_duration = 0.0
            silent_frame_count = 0
            has_speech = False

        if (not has_speech) and buffered_duration > 5.0:
            buffered_frames = []
            buffered_duration = 0.0
            silent_frame_count = 0

        idx += frame_samples

    if buffered_frames:
        segment_audio = np.concatenate(buffered_frames)
        start_sec = buffered_start / sample_rate
        duration_sec = len(segment_audio) / sample_rate
        segments.append((segment_audio, float(start_sec), float(duration_sec)))

    return segments


def _rms_energy(audio: np.ndarray) -> float:
    audio32 = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio32 * audio32)))


def _transcribe_with_model(model: Any, audio_input: np.ndarray, config: Any, env: Any) -> Dict[str, Any]:
    kwargs = {
        "task": getattr(config, 'task', 'transcribe'),
        "language": getattr(config, 'language', 'en'),
        "initial_prompt": getattr(config, 'initial_prompt', None),
        "fp16": getattr(env, 'device', 'cpu') == "cuda"
    }
    return model.transcribe(audio_input, **kwargs)


def _speech_text_from_segments(segments: List[Dict[str, Any]]) -> str:
    return " ".join(seg["text"] for seg in segments if str(seg.get("text") or "").strip()).strip()


def _ffmpeg_extract_pause_from_segments(model: Any, audio: np.ndarray, config: Any, env: Any) -> Tuple[str, List[Dict[str, Any]]]:
    sample_rate = 16000
    raw_segments = _segment_audio_by_silence(
        audio,
        sample_rate=sample_rate,
        silence_threshold=SILENCE_THERSHOLD,
        silence_duration_sec=SILENCE_DURATION_SEC,
        min_segment_sec=MIN_SEG_SEC,
        max_segment_sec=MAX_SEG_SEC,
        analysis_window_sec=ANALYSIS_WINDOW_SEC,
    )    
    
    speech_segments = []
    for segment_audio, start_sec, duration_sec in raw_segments:
        if _rms_energy(segment_audio) < float(SKIP_ENERGY_THERSHOLD):
            continue
        result = _transcribe_with_model(model, segment_audio, config, env)
        text = str(result.get("text") or "").strip()
        if not text: 
            continue
        end_sec = start_sec + duration_sec
        speech_segments.append(
            {
                "start_sec": round(float(start_sec), 3),
                "end_sec": round(float(end_sec), 3),
                "duration_sec": round(float(duration_sec), 3),
                "text": text,
            }
        )
    return _speech_text_from_segments(speech_segments), speech_segments


def _fallback_relevance_from_score(score: float) -> str:
    if score >= 80:
        return "highly_relevant"
    if score >= 60:
        return "partially_relevant"
    if score >= 35:
        return "weakly_relevant"
    return "off_topic"


# --- PUBLIC FUNCTIONS ---

def merge_unique_lines(existing: List[str], fresh: List[str]) -> List[str]:
    """
    Merges a list of fresh text lines into an existing list securely, avoiding case-insensitive duplicates.

    Args:
        existing (List[str]): Base list of lines.
        fresh (List[str]): New lines to merge.

    Returns:
        List[str]: The strictly unique merged list of text lines.
    """
    merged = list(existing)
    seen = {line.strip().casefold() for line in existing if line and line.strip()}
    for line in fresh:
        cleaned = (line or "").strip()
        if not cleaned:
            continue
        token = cleaned.casefold()
        if token in seen:
            continue
        seen.add(token)
        merged.append(cleaned)
    return merged


def extract_audio_ffmpeg(video_path: str, wav_out: str, overwrite: bool = True) -> str:
    """
    Extracts the audio track from a video file utilizing FFmpeg and downsamples it to 16kHz PCM.

    Args:
        video_path (str): The absolute path to the video.
        wav_out (str): Intended output directory or prefix (handled dynamically internally).
        overwrite (bool): Whether to overwrite if an output already exists.

    Returns:
        str: Absolute path to the temporarily generated or saved WAV file.
    """
    _ensure_ffmpeg_on_path()
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp_name)
    cmd = [
        "ffmpeg","-y",
        "-i", video_path,
        "-vn",
        "-acodec","pcm_s16le",
        "-ar","16000",
        "-ac","1",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found; install ffmpeg and add it to PATH") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extract failed: {e}") from e
    return str(tmp_path.resolve())


def extract_pause_from_audio(audio_path: str, config: Any, env: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Reads audio from disk, segments it based on silences, and transcribes segments using Whisper.

    Args:
        audio_path (str): Filepath to the audio file.
        config (Any): Configuration object with model parameters (like `.model_size`).
        env (Any): Environment object tracking device specifications (e.g. `.device`).

    Returns:
        Tuple[str, List[Dict]]: A tuple containing the full concatenated text, and the segmented data dictionary.
    """
    whisper_model = load_whisper_model(getattr(config, 'model_size', 'base'))
    audio_loaders = _get_audio_loaders(Path(audio_path))
    loaded_audio = _load_audio(Path(audio_path), audio_loaders)
    
    if loaded_audio is None:
        raise RuntimeError(f"Failed to load audio file: {audio_path}")
        
    return _ffmpeg_extract_pause_from_segments(
        whisper_model, loaded_audio, config, env
    )


def segment_text_lines(
    binary_inv_roi: np.ndarray,
    *,
    min_line_height: int = 8,
    min_gap: int = 5,
    pad_y: int = 4,
) -> List[Tuple[int, int]]:
    """
    Finds y-axis projection boundaries representing individual lines of text from an image region.

    Args:
        binary_inv_roi (np.ndarray): Inverted binary OpenCV image array.
        min_line_height (int): Minimum pixel height to register as a valid line.
        min_gap (int): Pixel gap threshold to distinguish between different lines.
        pad_y (int): Padding to append around discovered line bounds.

    Returns:
        List[Tuple[int, int]]: A list of (y0, y1) tuples representing line boundaries.
    """
    h, w = binary_inv_roi.shape[:2]
    proj = (binary_inv_roi > 0).astype(np.float32).sum(axis=1)
    threshold = max(1.0, 0.02 * w)
    
    in_line = False
    start = 0
    lines = []
    
    for y in range(h):
        if proj[y] >= threshold:
            if not in_line:
                start = y
                in_line = True
        else:
            if in_line and y - start >= min_line_height:
                y0 = max(0, start - pad_y)
                y1 = min(h, y + pad_y)
                lines.append((y0, y1))
            in_line = False
            
    if in_line and h - start >= min_line_height:
        y0 = max(0, start - pad_y)
        y1 = h
        lines.append((y0, y1))
        
    merged = []
    for y0, y1 in lines:
        if not merged:
            merged.append((y0, y1))
            continue
        py0, py1 = merged[-1]
        if y0 - py1 < min_gap:
            merged[-1] = (py0, y1)
        else:
            merged.append((y0, y1))
            
    return merged


def pause_audio(segments: List[Dict[str, Any]], config: Any) -> List[Dict[str, Any]]:
    """
    Filters and refines pre-extracted audio segments (Placeholder for user's downstream logic).

    Args:
        segments (List[Dict]): Segments list containing start/end and text data.
        config (Any): Configuration handling parameters.

    Returns:
        List[Dict]: The updated segments list.
    """
    # Logic goes here based on downstream needs
    return segments


if __name__ == "__main__":
    # --- Wrapped Floating Execution Logic ---
    # Moved here so it doesn't accidentally execute upon importing the module
    
    class MockVideo:
        def __init__(self):
            self.video_path_abs = "sample.mp4"
            self.segments = []
            
    class MockConfig:
        model_size = "base"
        task = "transcribe"
        language = "en"
        initial_prompt = ""
        
    class MockEnv:
        device = "cpu"
        
    video = MockVideo()
    config = MockConfig()
    env = MockEnv()
    
    if os.path.exists(video.video_path_abs):
        print("[Demo] Extracting audio...")
        audio_path = extract_audio_ffmpeg(video.video_path_abs, "./audio/wav")      
        print(f"[Demo] Extracting pauses from {audio_path}...")
        _, video.segments = extract_pause_from_audio(audio_path, config, env)
        print("[Demo] Done.")