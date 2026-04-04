# Turn the MP3 (or decodable wav) into text with Whisper. We load wav-like files into
# numpy first when we can, so Windows users see fewer "ffmpeg not found" surprises.

from __future__ import annotations

import logging
import os
import shutil
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

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


def transcribe_audio(
    audio_path: str,
    *,
    model_size: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe",
) -> str:
    if whisper is None:
        raise RuntimeError("Install openai-whisper")
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    kwargs: Dict[str, Any] = {"task": task}
    if language:
        kwargs["language"] = language

    audio_input: Union[str, Any] = str(path)
    arr = _load_audio_as_numpy_no_ffmpeg(path)
    if arr is not None:
        audio_input = arr
        logger.info("Loaded audio as numpy array (no ffmpeg)")
    else:
        if not _ensure_ffmpeg_on_path():
            raise RuntimeError(_FFMPEG_HINT)

    try:
        result = model.transcribe(audio_input, **kwargs)
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

    text = (result.get("text") or "").strip()
    return text


def run_module_c(
    audio_path: str,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    whisper_opts = (config or {}).get("whisper", {})
    model_size = str(whisper_opts.get("model_size", "base"))
    out: Dict[str, Any] = {"speech_text": "", "error": None}
    try:
        out["speech_text"] = transcribe_audio(audio_path, model_size=model_size)
    except Exception as e:
        out["error"] = str(e)
        logger.exception("run_module_c")
    return out
