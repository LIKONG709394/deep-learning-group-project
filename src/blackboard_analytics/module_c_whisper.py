# Turn the MP3 (or decodable wav) into text with Whisper. We load wav-like files into
# numpy first when we can, so Windows users see fewer "ffmpeg not found" surprises.

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from blackboard_analytics.model_cache import ensure_project_model_cache_dirs, get_whisper_cache_dir

logger = logging.getLogger(__name__)
ensure_project_model_cache_dirs()

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import whisper
except ImportError:
    whisper = None  # type: ignore

_FFMPEG_HINT = (
    "ffmpeg not found or failed to run. Whisper needs ffmpeg to decode MP3.\n"
    "Fix: (1) Install ffmpeg (e.g. winget install Gyan.FFmpeg), then fully restart the terminal / IDE / web server; "
    "or (2) set environment variable FFMPEG_PATH to the full path of ffmpeg.exe, e.g. "
    "C:\\\\Program Files\\\\ffmpeg\\\\bin\\\\ffmpeg.exe, then restart the app."
)

_WHISPER_MODELS: Dict[str, Any] = {}
_WHISPER_LOCK = threading.Lock()


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


def _ffmpeg_available() -> bool:
    return _ensure_ffmpeg_on_path()


def _as_float32_mono(audio: "np.ndarray") -> "np.ndarray":
    data = np.asarray(audio)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        maxv = float(np.iinfo(data.dtype).max)
        out = (data.astype(np.float32) / maxv).clip(-1.0, 1.0)
    else:
        out = data.astype(np.float32)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.5:
            out = (out / peak).clip(-1.0, 1.0)
        else:
            out = np.clip(out, -1.0, 1.0)
    return out


def _resample_to_16k(audio: "np.ndarray", fr: int) -> "np.ndarray":
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


def _load_wav_stdlib_16k_mono(path: Path) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy required")
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


def _load_wav_scipy_16k_mono(path: Path) -> "np.ndarray":
    from scipy.io import wavfile

    fr, data = wavfile.read(str(path))
    audio = _as_float32_mono(np.asarray(data))
    return _resample_to_16k(audio, int(fr))


def _load_soundfile_16k_mono(path: Path) -> "np.ndarray":
    import soundfile as sf

    data, fr = sf.read(str(path), always_2d=False, dtype="float32")
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = np.clip(data.astype(np.float32), -1.0, 1.0)
    return _resample_to_16k(audio, int(fr))


def _try_loaders(path: Path, loaders: List[Callable[[Path], "np.ndarray"]]) -> Optional["np.ndarray"]:
    for fn in loaders:
        try:
            return fn(path)
        except Exception as e:
            logger.debug("%s: %s", fn.__name__, e)
    return None


def _load_audio_as_numpy_no_ffmpeg(path: Path) -> Optional["np.ndarray"]:
    if np is None:
        return None
    ext = path.suffix.lower()
    loaders: List[Callable[[Path], np.ndarray]] = []

    if ext in (".wav", ".wave"):
        loaders.append(_load_wav_stdlib_16k_mono)
        loaders.append(_load_wav_scipy_16k_mono)
        try:
            import soundfile  # noqa: F401

            loaders.append(_load_soundfile_16k_mono)
        except ImportError:
            pass
    elif ext in (".flac", ".ogg", ".oga"):
        try:
            import soundfile  # noqa: F401

            loaders.append(_load_soundfile_16k_mono)
        except ImportError:
            pass

    if not loaders:
        return None
    return _try_loaders(path, loaders)


def _load_audio_ffmpeg_16k_mono(path: Path) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy required")
    if not _ensure_ffmpeg_on_path():
        raise RuntimeError(_FFMPEG_HINT)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(_FFMPEG_HINT)

    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp_name)
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
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _load_wav_stdlib_16k_mono(tmp_path)
    except FileNotFoundError as e:
        raise RuntimeError(_FFMPEG_HINT) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg decode failed: {e}") from e
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _load_audio_as_numpy(path: Path) -> "np.ndarray":
    arr = _load_audio_as_numpy_no_ffmpeg(path)
    if arr is not None:
        logger.info("Loaded audio as numpy array without ffmpeg")
        return arr
    return _load_audio_ffmpeg_16k_mono(path)


def _prepare_audio_input(path: Path) -> Union[str, Any]:
    arr = _load_audio_as_numpy_no_ffmpeg(path)
    if arr is not None:
        logger.info("Loaded audio as numpy array (no ffmpeg)")
        return arr
    if not _ensure_ffmpeg_on_path():
        raise RuntimeError(_FFMPEG_HINT)
    return str(path)


def _rms_energy(audio: "np.ndarray") -> float:
    if np is None or audio is None or len(audio) == 0:
        return 0.0
    audio32 = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio32 * audio32)))


def _segment_audio_by_energy(
    audio: "np.ndarray",
    *,
    sample_rate: int,
    silence_threshold: float,
    silence_duration_sec: float,
    min_segment_sec: float,
    max_segment_sec: float,
    analysis_window_sec: float,
) -> List[Tuple["np.ndarray", float, float]]:
    if np is None or audio is None or len(audio) == 0 or sample_rate <= 0:
        return []

    audio = np.asarray(audio, dtype=np.float32)
    window_sec = max(0.01, float(analysis_window_sec))
    frame_samples = max(1, int(sample_rate * window_sec))
    silence_frames_needed = max(1, int(float(silence_duration_sec) / window_sec))
    min_segment_sec = max(0.0, float(min_segment_sec))
    max_segment_sec = max(min_segment_sec if min_segment_sec > 0 else 0.1, float(max_segment_sec))

    segments: List[Tuple["np.ndarray", float, float]] = []
    buffered_frames: List["np.ndarray"] = []
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

        # Avoid accumulating long pure-silence buffers at the start of a file.
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


def _fixed_size_segments(
    audio: "np.ndarray",
    *,
    sample_rate: int,
    max_segment_sec: float,
) -> List[Tuple["np.ndarray", float, float]]:
    if np is None or audio is None or len(audio) == 0 or sample_rate <= 0:
        return []

    step = max(1, int(sample_rate * max(0.1, float(max_segment_sec))))
    segments: List[Tuple["np.ndarray", float, float]] = []
    start = 0
    while start < len(audio):
        chunk = audio[start : start + step]
        duration_sec = len(chunk) / sample_rate
        segments.append((chunk, float(start / sample_rate), float(duration_sec)))
        start += step
    return segments


def _load_whisper_model(model_size: str) -> Any:
    if whisper is None:
        raise RuntimeError("Install openai-whisper")
    with _WHISPER_LOCK:
        cached = _WHISPER_MODELS.get(model_size)
        if cached is not None:
            return cached
        try:
            model = whisper.load_model(model_size, download_root=str(get_whisper_cache_dir()))
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e
        _WHISPER_MODELS[model_size] = model
        return model


def _transcribe_with_model(
    model: Any,
    audio_input: Union[str, Any],
    *,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"task": task}
    if language:
        kwargs["language"] = language

    try:
        return model.transcribe(audio_input, **kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(_FFMPEG_HINT) from e
    except OSError as e:
        win = getattr(e, "winerror", None)
        if win == 2 or getattr(e, "errno", None) == 2:
            raise RuntimeError(_FFMPEG_HINT) from e
        logger.exception("Whisper transcribe")
        raise RuntimeError(f"Transcription failed: {e}") from e
    except Exception as e:
        logger.exception("Whisper transcribe")
        raise RuntimeError(f"Transcription failed: {e}") from e


def _simplify_whisper_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    simplified: List[Dict[str, Any]] = []
    for raw in raw_segments or []:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        start_sec = float(raw.get("start") or 0.0)
        end_sec = float(raw.get("end") or start_sec)
        end_sec = max(start_sec, end_sec)
        simplified.append(
            {
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "text": text,
            }
        )
    return simplified


def _speech_text_from_segments(segments: List[Dict[str, Any]]) -> str:
    return " ".join(seg["text"] for seg in segments if str(seg.get("text") or "").strip()).strip()


def transcribe_audio(
    audio_path: str,
    *,
    model_size: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe",
) -> str:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    model = _load_whisper_model(model_size)
    audio_input = _prepare_audio_input(path)
    result = _transcribe_with_model(model, audio_input, language=language, task=task)
    text = (result.get("text") or "").strip()
    return text


def transcribe_audio_with_segments(
    audio_path: str,
    *,
    model_size: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe",
    enable_silence_segmentation: bool = True,
    silence_threshold: float = 0.0004,
    silence_duration_sec: float = 1.0,
    min_segment_sec: float = 2.0,
    max_segment_sec: float = 20.0,
    analysis_window_sec: float = 0.1,
    skip_energy_threshold: float = 0.00012,
) -> Dict[str, Any]:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    model = _load_whisper_model(model_size)

    if not enable_silence_segmentation:
        audio_input = _prepare_audio_input(path)
        result = _transcribe_with_model(model, audio_input, language=language, task=task)
        speech_text = (result.get("text") or "").strip()
        speech_segments = _simplify_whisper_segments(result.get("segments") or [])
        return {
            "speech_text": speech_text,
            "speech_segments": speech_segments,
        }

    audio = _load_audio_as_numpy(path)
    sample_rate = 16000
    raw_segments = _segment_audio_by_energy(
        audio,
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        silence_duration_sec=silence_duration_sec,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
        analysis_window_sec=analysis_window_sec,
    )
    if not raw_segments:
        raw_segments = _fixed_size_segments(audio, sample_rate=sample_rate, max_segment_sec=max_segment_sec)

    speech_segments: List[Dict[str, Any]] = []
    for segment_audio, start_sec, duration_sec in raw_segments:
        if _rms_energy(segment_audio) < float(skip_energy_threshold):
            continue
        result = _transcribe_with_model(model, segment_audio, language=language, task=task)
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

    return {
        "speech_text": _speech_text_from_segments(speech_segments),
        "speech_segments": speech_segments,
    }


def run_module_c(
    audio_path: str,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    whisper_opts = (config or {}).get("whisper", {})
    model_size = str(whisper_opts.get("model_size", "base"))
    out: Dict[str, Any] = {"speech_text": "", "speech_segments": [], "error": None}
    try:
        result = transcribe_audio_with_segments(
            audio_path,
            model_size=model_size,
            language=whisper_opts.get("language"),
            task=str(whisper_opts.get("task", "transcribe")),
            enable_silence_segmentation=bool(whisper_opts.get("enable_silence_segmentation", True)),
            silence_threshold=float(whisper_opts.get("silence_threshold", 0.0004)),
            silence_duration_sec=float(whisper_opts.get("silence_duration_sec", 1.0)),
            min_segment_sec=float(whisper_opts.get("min_segment_sec", 2.0)),
            max_segment_sec=float(whisper_opts.get("max_segment_sec", 20.0)),
            analysis_window_sec=float(whisper_opts.get("analysis_window_sec", 0.1)),
            skip_energy_threshold=float(whisper_opts.get("skip_energy_threshold", 0.00012)),
        )
        out["speech_text"] = result.get("speech_text", "")
        out["speech_segments"] = result.get("speech_segments", [])
    except Exception as e:
        out["error"] = str(e)
        logger.exception("run_module_c")
    return out
