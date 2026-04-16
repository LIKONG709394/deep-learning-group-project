"""
whisper_asr.py — Whisper speech transcription for Catcher

Responsibilities:
- load and cache Whisper models
- decode audio to 16k mono float32
- transcribe either full audio or silence-split chunks
- return transcript text + timestamped segments
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import whisper
except ImportError:
    whisper = None


WHISPER_MODELS: Dict[str, Any] = {}
WHISPER_LOCK = threading.Lock()

FFMPEG_HINT = (
    "ffmpeg not found or failed to run. Install ffmpeg and add it to PATH, "
    "or set FFMPEG_PATH / BLACKBOARD_FFMPEG to ffmpeg.exe."
)


def transcribe_audio(audio_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main public API.

    Returns:
    {
        "speech_text": str,
        "segments": [
            {
                "startsec": 0.0,
                "endsec": 3.24,
                "durationsec": 3.24,
                "text": "..."
            }
        ],
        "model_size": "base",
        "language": "en",
    }
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    whisper_cfg = config.get("whisper") or {}
    model_size = str(whisper_cfg.get("model_size", "base")).strip() or "base"
    language = whisper_cfg.get("language", None)
    language = None if language in (None, "", "auto") else str(language).strip()
    task = str(whisper_cfg.get("task", "transcribe")).strip() or "transcribe"
    initial_prompt = whisper_cfg.get("initial_prompt", None)
    if isinstance(initial_prompt, str):
        initial_prompt = initial_prompt.strip() or None

    enable_seg = bool(whisper_cfg.get("enable_silence_segmentation", False))
    silence_threshold = float(whisper_cfg.get("silence_threshold", 0.0004))
    silence_duration_sec = float(whisper_cfg.get("silence_duration_sec", 1.0))
    min_segment_sec = float(whisper_cfg.get("min_segment_sec", 2.0))
    max_segment_sec = float(whisper_cfg.get("max_segment_sec", 20.0))
    analysis_window_sec = float(whisper_cfg.get("analysis_window_sec", 0.1))
    skip_energy_threshold = float(whisper_cfg.get("skip_energy_threshold", 0.00012))

    model = load_whisper_model(model_size)

    audio = load_audio_16k_mono(path)

    if enable_seg:
        speech_text, segments = transcribe_by_segments(
            model=model,
            audio=audio,
            task=task,
            language=language,
            initial_prompt=initial_prompt,
            silence_threshold=silence_threshold,
            silence_duration_sec=silence_duration_sec,
            min_segment_sec=min_segment_sec,
            max_segment_sec=max_segment_sec,
            analysis_window_sec=analysis_window_sec,
            skip_energy_threshold=skip_energy_threshold,
        )
    else:
        speech_text, segments = transcribe_full_audio(
            model=model,
            audio=audio,
            task=task,
            language=language,
            initial_prompt=initial_prompt,
        )

    return {
        "speech_text": speech_text,
        "segments": segments,
        "model_size": model_size,
        "language": language,
    }


def load_whisper_model(model_size: str) -> Any:
    if whisper is None:
        raise RuntimeError("Install openai-whisper first.")

    key = str(model_size).strip() or "base"

    with WHISPER_LOCK:
        if key in WHISPER_MODELS:
            return WHISPER_MODELS[key]

        cache_dir = Path(".modelcache") / "whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model = whisper.load_model(key, download_root=str(cache_dir))
        WHISPER_MODELS[key] = model
        return model


def transcribe_full_audio(
    model: Any,
    audio: np.ndarray,
    task: str,
    language: Optional[str],
    initial_prompt: Optional[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    result = _transcribe_with_model(
        model=model,
        audio_input=audio,
        task=task,
        language=language,
        initial_prompt=initial_prompt,
    )

    speech_text = str(result.get("text") or "").strip()
    segments = segments_from_whisper(result.get("segments") or [])
    return speech_text, segments


def transcribe_by_segments(
    model: Any,
    audio: np.ndarray,
    task: str,
    language: Optional[str],
    initial_prompt: Optional[str],
    silence_threshold: float,
    silence_duration_sec: float,
    min_segment_sec: float,
    max_segment_sec: float,
    analysis_window_sec: float,
    skip_energy_threshold: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    samplerate = 16000

    raw_segments = segment_audio_by_silence(
        audio=audio,
        samplerate=samplerate,
        silence_threshold=silence_threshold,
        silence_duration_sec=silence_duration_sec,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
        analysis_window_sec=analysis_window_sec,
    )

    output_segments: List[Dict[str, Any]] = []

    for chunk_audio, startsec, durationsec in raw_segments:
        if rms_energy(chunk_audio) < skip_energy_threshold:
            continue

        result = _transcribe_with_model(
            model=model,
            audio_input=chunk_audio,
            task=task,
            language=language,
            initial_prompt=initial_prompt,
        )

        text = str(result.get("text") or "").strip()
        if not text:
            continue

        endsec = startsec + durationsec
        output_segments.append(
            {
                "startsec": round(float(startsec), 3),
                "endsec": round(float(endsec), 3),
                "durationsec": round(float(durationsec), 3),
                "text": text,
            }
        )

    speech_text = " ".join(seg["text"] for seg in output_segments if seg.get("text"))
    return speech_text.strip(), output_segments


def _transcribe_with_model(
    model: Any,
    audio_input: np.ndarray,
    task: str,
    language: Optional[str],
    initial_prompt: Optional[str],
) -> Dict[str, Any]:
    kwargs = {
        "task": task,
        "fp16": False,
    }
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    return model.transcribe(audio_input, **kwargs)


def segments_from_whisper(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    for raw in raw_segments:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue

        startsec = float(raw.get("start") or 0.0)
        endsec = float(raw.get("end") or startsec)
        endsec = max(startsec, endsec)

        segments.append(
            {
                "startsec": round(startsec, 3),
                "endsec": round(endsec, 3),
                "durationsec": round(endsec - startsec, 3),
                "text": text,
            }
        )

    return segments


def load_audio_16k_mono(path: Path) -> np.ndarray:
    arr = load_audio_without_ffmpeg(path)
    if arr is not None:
        return arr

    return load_audio_via_ffmpeg(path)


def load_audio_without_ffmpeg(path: Path) -> Optional[np.ndarray]:
    ext = path.suffix.lower()

    if ext in {".wav", ".wave"}:
        for loader in (load_wav_stdlib_16k_mono, load_wav_scipy_16k_mono):
            try:
                return loader(path)
            except Exception:
                pass

    if ext in {".flac", ".ogg", ".oga"}:
        try:
            return load_soundfile_16k_mono(path)
        except Exception:
            pass

    return None


def load_audio_via_ffmpeg(path: Path) -> np.ndarray:
    ffmpeg = find_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError(FFMPEG_HINT)

    fd, tmpname = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmppath = Path(tmpname)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(tmppath),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return load_wav_stdlib_16k_mono(tmppath)
    except FileNotFoundError as e:
        raise RuntimeError(FFMPEG_HINT) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg decode failed: {e}") from e
    finally:
        try:
            tmppath.unlink(missing_ok=True)
        except Exception:
            pass


def find_ffmpeg_executable() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found

    for key in ("FFMPEG_PATH", "BLACKBOARD_FFMPEG"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        p = Path(raw)
        if p.is_file():
            return str(p)
        exe = p / "ffmpeg.exe"
        if exe.is_file():
            return str(exe)

    return None


def as_float32_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio)

    if data.ndim == 2:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        maxv = float(np.iinfo(data.dtype).max)
        out = data.astype(np.float32) / maxv
        return np.clip(out, -1.0, 1.0)

    out = data.astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.5:
        out = out / peak

    return np.clip(out, -1.0, 1.0)


def resample_to_16k(audio: np.ndarray, fr: int) -> np.ndarray:
    if fr == 16000 or len(audio) == 0:
        return audio.astype(np.float32)

    try:
        from scipy import signal
    except ImportError as e:
        raise RuntimeError(
            f"Audio is {fr} Hz. Install scipy to resample to 16000 Hz."
        ) from e

    num = max(1, int(len(audio) * 16000 / fr))
    return signal.resample(audio, num).astype(np.float32)


def load_wav_stdlib_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported in stdlib loader.")

    data = np.frombuffer(raw, dtype=np.int16).copy()

    if channels == 2:
        data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)

    audio = data.astype(np.float32) / 32768.0
    audio = np.clip(audio, -1.0, 1.0)
    return resample_to_16k(audio, framerate)


def load_wav_scipy_16k_mono(path: Path) -> np.ndarray:
    from scipy.io import wavfile

    fr, data = wavfile.read(str(path))
    audio = as_float32_mono(np.asarray(data))
    return resample_to_16k(audio, int(fr))


def load_soundfile_16k_mono(path: Path) -> np.ndarray:
    import soundfile as sf

    data, fr = sf.read(str(path), always_2d=False, dtype="float32")
    audio = as_float32_mono(np.asarray(data))
    return resample_to_16k(audio, int(fr))


def rms_energy(audio: np.ndarray) -> float:
    audio32 = np.asarray(audio, dtype=np.float32)
    if audio32.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio32 * audio32)))


def segment_audio_by_silence(
    audio: np.ndarray,
    samplerate: int,
    silence_threshold: float,
    silence_duration_sec: float,
    min_segment_sec: float,
    max_segment_sec: float,
    analysis_window_sec: float,
) -> List[Tuple[np.ndarray, float, float]]:
    if audio is None or len(audio) == 0 or samplerate <= 0:
        return []

    audio = np.asarray(audio, dtype=np.float32)

    window_sec = max(0.01, float(analysis_window_sec))
    frame_samples = max(1, int(samplerate * window_sec))
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
        buffered_duration += len(frame) / samplerate

        if rms_energy(frame) > silence_threshold:
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
            chunk = np.concatenate(buffered_frames)
            startsec = buffered_start / samplerate
            durationsec = len(chunk) / samplerate
            segments.append((chunk, float(startsec), float(durationsec)))

            buffered_frames = []
            buffered_duration = 0.0
            silent_frame_count = 0
            has_speech = False

        if not has_speech and buffered_duration > 5.0:
            buffered_frames = []
            buffered_duration = 0.0
            silent_frame_count = 0

        idx += frame_samples

    if buffered_frames:
        chunk = np.concatenate(buffered_frames)
        startsec = buffered_start / samplerate
        durationsec = len(chunk) / samplerate
        segments.append((chunk, float(startsec), float(durationsec)))

    return segments