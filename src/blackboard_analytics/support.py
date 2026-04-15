from datetime import datetime
from typing import List, Tuple

from paddle.version import cuda
from pathlib import Path
import subprocess
import cv2
from ultralytics import YOLO
import threading
from blackboard_analytics.module_a_blackboard_ocr import TrOCRHandwritingEngine
from blackboard_analytics.module_d_semantic import SemanticAligner
from entities import *
import shutil, os
import tempfile
import numpy as np
from  default import *
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

def has_hf_repo_cache(model_name: str) -> bool:
    repo_dir = get_hf_hub_cache_dir() / f"models--{model_name.replace('/', '--')}"
    return repo_dir.is_dir() and (repo_dir / "snapshots").is_dir()

def _normalize_endpoint(base_url: str) -> str:
    cleaned = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    return f"{cleaned}/chat/completions"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_model_cache_root() -> Path:
    return get_project_root() / ".model_cache"


def get_hf_home() -> Path:
    return get_model_cache_root() / "huggingface"


def get_hf_hub_cache_dir() -> Path:
    return get_hf_home() / "hub"


def get_transformers_cache_dir() -> Path:
    return get_hf_home() / "transformers"


def get_sentence_transformers_cache_dir() -> Path:
    return get_model_cache_root() / "sentence_transformers"


def get_whisper_cache_dir() -> Path:
    return get_model_cache_root() / "whisper"


def get_torch_home() -> Path:
    return get_model_cache_root() / "torch"

def normalize_ocr_engine_name(raw: Any) -> str:
    s = str(raw or "trocr").strip().lower()
    if s in ("trocr", "transformers", "hf", "huggingface"):
        return "trocr"
    if s in ("easyocr", "easy"):
        return "easyocr"
    if s in ("paddleocr", "paddle"):
        return "paddleocr"
    return "trocr"

def getEasyLangs(raw_langs):
    if isinstance(raw_langs, str):
        return [raw_langs.strip()] if raw_langs.strip() else None
    if isinstance(raw_langs, list):
        easy_langs = [str(x).strip() for x in raw_langs if str(x).strip()]
        return None if not easy_langs else easy_langs
    return None

def makeSiblingDir(root, name_root):
    dir = Path(root).resolve().parent / f"{Path(name_root).stem}_video_debug"
    dir.mkdir(parents=True, exist_ok=True)
    return dir

def extract_audio_ffmpeg(video_path: str, wav_out: str, overwrite: bool = True) -> str:
    out = Path(wav_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg","-y",
        "-i", video_path,
        "-vn",
        "-acodec","pcm_s16le",
        "-ar","16000",
        "-ac","1",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found; install ffmpeg and add it to PATH") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extract failed: {e}") from e
    return str(out.resolve())

def getfps(capture):
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0: fps = 25.0
    return fps

def gettimestamp(capture, frame_index, fps):
    ts_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if ts_msec > 0: return ts_msec / 1000.0
    if fps > 0: return float(frame_index / fps)
    return 0.0

def opencv_extract_frame_dicts(video_path, stride=5):
    capture = cv2.VideoCapture(video_path)
    fps = getfps(capture)
    frame_index = -1
    frame_dicts = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        else: frame_index += 1
        if stride!=0 and frame_index%stride !=0: continue
        frame_dicts.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(gettimestamp(capture, frame_index, fps), 3),
                "frame_bgr": frame,
            }
        )        
    capture.release()
    return frame_dicts

def indicate_classes(raw, en_only):
    if raw is None: return []

    if isinstance(raw, str): items = [raw]
    elif isinstance(raw, (list, tuple)): items = [str(x) for x in raw]
    else: return []
    
    res = []
    for p in items:
        s = str(p).strip()
        if not s: continue
        if en_only and not s.isascii(): continue
        res.append(s)
    return res

_YOLO_WORLD_LOCK = threading.lock()
_YOLO_LOADED_CLASSES = {}
def _get_yolo_world_model(model_name, prompts):
    key = (str(model_name), tuple(prompts))    
    with _YOLO_WORLD_LOCK:
        cached = _YOLO_LOADED_CLASSES.get(key)
        if cached is not None:
            return cached
        model = YOLO(str(model_name))
        model.set_classes(prompts)
        _YOLO_LOADED_CLASSES[key] = model
        return model

def get_textarea_yolo_world(image_bgr, model_name, conf, iou, textclassess, en_only):
    prompts = indicate_classes(textclassess, en_only)
    model = _get_yolo_world_model(model_name, prompts)
    results = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)
    return results

def get_textarea_yolo(image_bgr, weight_path, conf, iou):
    model = YOLO(str(weight_path))
    results = model.predict(source=image_bgr, conf=conf, iou=iou, verbose=False)
    return results    

def get_boxes(results, frame_w, frame_h):
    res = []
    boxes = results[0].boxes
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[idx].cpu().numpy().ravel())
        res.append(ROIBox(x1, y1, x2, y2).clip(frame_w, frame_h))
    return res
    
def get_largest_box(boxes):
    best_area = 0
    best_box = None
    for box in boxes:
        x1, y1, x2, y2 = boxes.as_tuple()
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < best_area: continue
        best_area = area
        best_box = box
    return best_box

def crop_roi(image: np.ndarray, roi: ROIBox) -> np.ndarray:
    x1, y1, x2, y2 = roi.as_tuple()
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI")
    return image[y1:y2, x1:x2].copy()

def preprocess_image(
        image_bgr, *,
        clahe_clip=2.0, clahe_grid=(8, 8), 
        binary_block_size=31, binary_c=5,
):
    if len(image_bgr.shape) == 2: gray = image_bgr.copy()
    else: gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray_enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(3, binary_block_size | 1),
        binary_c,
    )
    return binary


def getAbsTrait(image_bgr, size):
    binary = preprocess_image(image_bgr)
    resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
    return (resized > 32).astype(np.uint8) * 255

def getRelTrait(previous_trait, current_trait):
    if previous_trait is None:
        return 1.0
    diff = cv2.absdiff(previous_trait, current_trait)
    return float(np.mean(diff) / 255.0)

_EASY_READERS = {}
_PADDLE_INSTANCES = {}
_EASY_LOCK = threading.Lock()
_PADDLE_LOCK = threading.Lock()
import easyocr
def recognize_lines_easyocr(image_bgr,*,languages, env):
    langs = tuple(str(x).strip() for x in languages if str(x).strip()) or ("en",)
    key = (langs, env.device=="cuda")
    with _EASY_LOCK:
        reader = _EASY_READERS.get(key)
        if reader is None: 
            reader = easyocr.Reader(list(langs), env.device=="cuda")
            _EASY_READERS[key] = reader
    return reader.readtext(image_bgr)

    #entries = _easyocr_detections_to_entries(result)
    #return _cluster_detections_to_lines(entries)
from paddleocr import PaddleOCR
def recognize_lines_paddleocr(image_bgr,*,languages, env):
    lang = str(languages or "en").strip() or "en"
    key = (lang, env.device=="cuda")
    with _PADDLE_LOCK:
        ocr = _PADDLE_INSTANCES.get(key)
        if ocr is None:
            kwargs = {
                "use_angle_cls": True,
                "lang": lang,
                "show_log": False,
            }
            ocr = PaddleOCR(**kwargs, use_gpu=bool(env.device=="cuda"))
            _PADDLE_INSTANCES[key] = ocr
    return ocr.predict(image_bgr, cls=True)

def easyocr_detections_to_entries(result):
    out = []
    for item in result:
        if not item or len(item) < 2:
            continue
        bbox, text = item[0], item[1]
        conf = float(item[2]) if len(item) > 2 else 1.0
        if conf < 0.15:
            continue
        try:
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
        except (TypeError, IndexError, ValueError):
            continue
        if not xs or not ys:
            continue
        out.append((min(ys), max(ys), min(xs), str(text).strip()))
    return out

def paddle_detections_to_entries(result):
    out = []
    result = result[0]
    for item in result:
        if not item or len(item) < 2: continue
        bbox, tc = item[0], item[1]
        if not isinstance(tc, (list, tuple)) or len(tc) < 1:continue    
        text = str(tc[0] or "").strip()
        conf = float(tc) if len(tc) > 1 else 1.0
        if conf < 0.15 or not text: continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        out.append((min(ys), max(ys), min(xs), max(xs), text))
    return out

def arrange_bboxes(bboxes, axis1, axis2):
    if axis1 == 0:
        return sorted(bboxes, key=lambda t: ((t[0] + t[1]) / 2.0, t[2]))
    return sorted(bboxes, key=lambda t: (t[2], (t[0] + t[1]) / 2.0))

def merge_lines(bboxes, allowed_height_tweak):    
    lines = []
    current = []
    for box in bboxes:
        ymin, ymax, xmin, xmax, text = box
        cy = (ymin + ymax) / 2.0
        if not current:
            current = [box]
            continue
        prev_cy = sum((a[0] + a[1]) / 2.0 for a in current) / len(current)
        if abs(cy - prev_cy) <= allowed_height_tweak:
            current.append(box)
        else:
            current.sort(key=lambda x: x[2])
            lines.append(current)
            current = [box]
    if current:
        current.sort(key=lambda x: x[2])
        lines.append(current)    
    return [" ".join(t for _, _, _, _, t in block).strip() for block in lines if block], lines 

def segment_text_lines(
    binary_inv_roi: np.ndarray,
    *,
    min_line_height: int = 8,
    min_gap: int = 5,
    pad_y: int = 4,
) -> List[Tuple[int, int]]:
    # row sums → line bands (y0,y1) in ROI coords
    h, w = binary_inv_roi.shape[:2]
    proj = (binary_inv_roi > 0).astype(np.float32).sum(axis=1)
    threshold = max(1.0, 0.02 * w)
    in_line = False
    start = 0
    lines: List[Tuple[int, int]] = []
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

def recognize_lines_other_ocr(image_bgr,engine, trocr_model_name, device):
    recognized_lines = []
    gray_enhanced, ink_for_lines = preprocess_image(image_bgr)
    line_spans = segment_text_lines(ink_for_lines)
    if not line_spans:
        line_spans = [(0, gray_enhanced.shape[0])]

    reader = engine or TrOCRHandwritingEngine(trocr_model_name, device)
    for row_top, row_bottom in line_spans:
        line_gray = gray_enhanced[row_top:row_bottom, :]
        if line_gray.size == 0:
            continue
        if (line_gray > 0).sum() < 50:
            continue
        line_text = reader.decode_line(line_gray)
        if line_text:
            recognized_lines.append(line_text)
    return recognized_lines

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
def initTextScoreModel():
    MODEL_NAME = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer
def calculate_perplexity(text: str, tokenizer, model) -> float:
    if not text.strip(): return float("inf")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity

def meaningful(lines: list[str]) -> bool:
    joined = " ".join((line or "").strip() for line in lines).strip()
    if not joined:
        return True
    alpha_count = sum(ch.isalpha() for ch in joined)
    digit_count = sum(ch.isdigit() for ch in joined)
    useful_count = sum(ch.isalnum() for ch in joined)
    if alpha_count >= 8:
        return True
    if useful_count == 0:
        return False
    return digit_count < alpha_count and useful_count > 12

def merge_unique_lines(existing: list[str], fresh: list[str]) -> list[str]:
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

def save_annotated_image(img_bgr, boxes, output_path):
    out = img_bgr.copy()
    for _box in boxes:#label box
        for box in _box["ocr_rois"]:
            max_y, min_y, max_x, min_x, text = box
            x1, y1, x2, y2 = int(min_x), int(min_y), int(max_x), int(max_y)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, text, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, out)
    return output_path

def dedupe_subsumed_lines(lines: list[str], *, min_len: int = 8) -> list[str]:
    items = [(s.strip(), s.strip().casefold()) for s in lines if s and s.strip()]
    if len(items) < 2:
        return [text for text, _ in items]
    drop: set[int] = set()
    for i, (_, item_cf) in enumerate(items):
        if len(item_cf) < min_len:
            continue
        for j, (_, other_cf) in enumerate(items):
            if i == j or len(other_cf) <= len(item_cf):
                continue
            if item_cf in other_cf:
                drop.add(i)
                break
    return [items[i][0] for i in range(len(items)) if i not in drop]

def filter_noise_board_lines(lines, min_chars, min_letters):
    filtered: list[str] = []
    for line in lines:
        text = (line or "").strip()
        if len(text) < max(1, min_chars):
            continue
        if sum(ch.isalpha() or ch in MATH_SYMBOLS for ch in text) < max(1, min_letters):
            continue
        filtered.append(text)
    return filtered

import wave
import whisper

_WHISPER_LOCK = threading.Lock()
_WHISPER_MODELS = {}
def load_whisper_model(model_size):
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
    audio =     (np.asarray(data))
    return _resample_to_16k(audio, int(fr))


def _load_soundfile_16k_mono(path: Path) -> "np.ndarray":
    import soundfile as sf

    data, fr = sf.read(str(path), always_2d=False, dtype="float32")
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio = np.clip(data.astype(np.float32), -1.0, 1.0)
    return _resample_to_16k(audio, int(fr))

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


def get_audio_loaders(path:Path):
    format = path.suffix.lower()
    if format in ("wav", "wave"):
        return [_load_wav_stdlib_16k_mono, _load_wav_scipy_16k_mono]
    elif format in (".flac", ".ogg", ".oga"):
        return [_load_soundfile_16k_mono]
    return None

def load_audio(path:Path, loaders:List[function]):    
    for loader in loaders:
        try: return loader(path)
        except: pass
    return None

def _segment_audio_by_silence(
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


def _transcribe_with_model(model,audio_input,config,env):
    kwargs = {
        "task": config.task,
        "language": config.language,
        "initial_prompt": config.initial_prompt,
        "fp16": env.device == "cuda"
    }
    return model.transcribe(audio_input, **kwargs)
    
def ffmpeg_extract_pause_dicts(raw_segments):
    pause_dicts = []
    for raw in raw_segments or []:
        text = str(raw.get("text") or "").strip()
        if not text: continue
        start_sec = float(raw.get("start") or 0.0)
        end_sec = float(raw.get("end") or start_sec)
        end_sec = max(start_sec, end_sec)
        pause_dicts.append(
            {
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "text": text,
            }
        )
    return pause_dicts

def _rms_energy(audio: "np.ndarray") -> float:
    audio32 = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(audio32 * audio32)))

def _speech_text_from_segments(segments):
    return " ".join(seg["text"] for seg in segments if str(seg.get("text") or "").strip()).strip()

def ffmpeg_extract_pause_from_audio(model, audio, config, env):
    result = _transcribe_with_model(model, audio, config, env)
    speech_text = (result.get("text") or "").strip()
    speech_segments = ffmpeg_extract_pause_dicts(result.get("segments") or [])
    return speech_text, speech_segments

def ffmpeg_extract_pause_from_segments(model, audio, config, env):
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
        if not text: continue
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

import json, request
def _call_deepseek_chat_completion(
        url, api_key, model, temperature, timeout_sec, messages,
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except:
        pass
    return None

import error
def _call_deepseek_chat_completion(
        url, api_key, model, timeout_sec, messages
):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout_sec)) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {detail[:500]}") from e
    except error.URLError as e:
        raise RuntimeError(f"DeepSeek request failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"DeepSeek request failed: {e}") from e
    try:
        response_json = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"DeepSeek returned invalid JSON response: {raw[:500]}") from e
    if not isinstance(response_json, dict):
        raise RuntimeError("DeepSeek response root is not a JSON object")
    return response_json

def _extract_message_content(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("DeepSeek response missing message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise RuntimeError("DeepSeek response missing text content")


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Empty DeepSeek content")
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in DeepSeek content")
    snippet = cleaned[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek content JSON is not an object")
    return parsed

def build_filter_board_lines_messages(lines, speech_text):
    system_prompt = (
        "You clean noisy OCR lines from a classroom whiteboard, projector slide, or similar. "
        "Remove: UI chrome, button labels, stray single words that are not educational content, "
        "watermarks, timestamps, window titles, duplicate near-empty fragments, and obvious OCR garbage. "
        "Keep: instructional text, headings, bullet content, formulas words, and phrases that belong to the lesson. "
        "Prefer lines that are primarily English/Latin script; drop obvious non-Latin script noise when it is not lesson content. "
        "speech_text is optional context only (what the teacher said); use it to prefer lines that match the topic, "
        "but do not invent lines. Return JSON only."
    )
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
    user_prompt = (
        "Task:\n"
        "1. Review each numbered OCR line below.\n"
        "2. Decide which line indices to KEEP (educational / slide content).\n"
        "3. When uncertain, prefer KEEP over DROP.\n\n"
        "Output requirements:\n"
        '- Return exactly one JSON object with these fields only:\n'
        '  "kept_indices": array of integers — 0-based indices referring ONLY to the numbered list below\n'
        '  "reason": one short sentence explaining the main noise you removed\n'
        "Preserve the order of indices as they appear in the list (ascending).\n\n"
        f"speech_text (context, may be empty):\n{(speech_text or '').strip() or '[EMPTY]'}\n\n"
        f"numbered OCR lines:\n{numbered}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]



def deepseek_filter_lines(messages, config, env):
    url = _normalize_endpoint(DEFAULT_BASE_URL)    
    try:
        response_json = _call_deepseek_chat_completion(
            url = url,
            api_key=env.deepseek,
            model=config.deepseek_model,
            timeout_sec=config.deepseek_timeout_sec,
            messages=messages,
        )
        content = _extract_message_content(response_json)
        return _extract_json_object(content)        
    except Exception as e:
        return messages
    
import re
    
def tokenize_mixed(text: str) -> Set[str]:
    if not text or not text.strip():
        return set()
    words = re.findall(r"[a-zA-Z]+", text.lower())
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    digits = re.findall(r"\d+", text)
    return set(words) | set(chars) | set(digits)
    
def keyword_overlap_rate(a: str, b: str) -> float:
    sa, sb = tokenize_mixed(a), tokenize_mixed(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / union) if union else 0.0
    
def getAligmentModel(model_name):
    return SemanticAligner(model_name)

def judge_alignment(
    semantic_sim: float,
    keyword_overlap: float,
    *,
    high_sim: float = 0.72,
    partial_sim: float = 0.45,
    keyword_high: float = 0.35,
) -> str:
    strong_meaning = semantic_sim >= high_sim
    enough_shared_words = keyword_overlap >= keyword_high
    if strong_meaning and enough_shared_words:
        return "highly_aligned"

    meaning_close_enough = semantic_sim >= partial_sim
    some_shared_words = keyword_overlap >= KEYWORD_OVERLAP_FOR_PARTIAL
    if meaning_close_enough or some_shared_words:
        return "partially_related"

    return "content_mismatch"

def build_alignment_messages(board_text: str, speech_text: str):
    system_prompt = (
        "You are an evaluator for classroom teaching alignment. "
        "You compare board_text and speech_text, then judge whether the speech stays focused on the board content "
        "or is clearly off-topic. "
        "Return JSON only. Do not include markdown, prose outside JSON, or extra keys."
    )
    user_prompt = (
        "Task:\n"
        "1. Compare the two inputs: board_text and speech_text.\n"
        "2. Judge whether the speech is centered on the board content.\n"
        "3. Judge whether the speech is clearly off-topic.\n"
        "4. Be conservative when information is missing, empty, or too short. Do not guess details.\n\n"
        "Output requirements:\n"
        '- Return exactly one JSON object with these fields only:\n'
        '  "overall_relevance": one of ["highly_relevant", "partially_relevant", "weakly_relevant", "off_topic"]\n'
        '  "score": a number from 0 to 100\n'
        '  "reason": a short explanation in one or two sentences\n'
        '  "evidence": an array of 2 to 5 short strings\n'
        "- If the texts are empty or insufficient, give a cautious low-confidence style judgment rather than making things up.\n\n"
        f"board_text:\n{board_text.strip() or '[EMPTY]'}\n\n"
        f"speech_text:\n{speech_text.strip() or '[EMPTY]'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def evaluate_handwriting_clarity(
    image: np.ndarray,
    *,
    laplacian_clear_min: float = 120.0,
    laplacian_messy_max: float = 40.0,
    stroke_variance_messy_min: float = 8.0,
) -> Dict[str, Any]:
    if image is None or image.size == 0:
        raise ValueError("Empty image")

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_metric = laplacian_variance(gray)
    ink_mask = _binarize_for_strokes(gray)
    stroke_widths, stroke_width_variance = _stroke_widths_per_component(ink_mask)

    label, score, suggestion = _score_to_clarity(
        blur_metric,
        stroke_width_variance,
        laplacian_clear_min,
        laplacian_messy_max,
        stroke_variance_messy_min,
    )

    return {
        "clarity": label,
        "score": round(score, 2),
        "suggestion": suggestion,
        "laplacian_variance": blur_metric,
        "stroke_width_variance": stroke_width_variance,
        "details": {"num_stroke_components": len(stroke_widths)},
    }

def extract_pause_from_audio(audio_path, config, env):
    whisper_model = load_whisper_model(config.model_size)
    audio_loaders = get_audio_loaders(audio_path)
    loaded_audio = load_audio(audio_path, audio_loaders)
    return ffmpeg_extract_pause_from_segments(
        whisper_model, loaded_audio, config, env
    )

def load_bgr_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image

def _register_cjk_font() -> str:
    if canvas is None:
        return "Helvetica"
    candidates = [
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for p in candidates:
        if not p.is_file():
            continue
        name = "CJKFont"
        try:
            pdfmetrics.registerFont(TTFont(name, str(p)))
            return name
        except Exception as e:
            logger.debug("Font register failed %s: %s", p, e)
    return "Helvetica"


def build_teaching_feedback_pdf(
    output_path, *, board_lines, clarity,
    alignment, speech_text, module_errors=None,
) -> str:
    font = _register_cjk_font()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    y = h - 50
    line_h = 16

    def draw_line(text: str, indent: int = 0) -> None:
        nonlocal y
        if y < 60:
            c.showPage()
            y = h - 50
        c.setFont(font, 11)
        safe = text.encode("latin-1", "replace").decode("latin-1") if font == "Helvetica" else text
        try:
            c.drawString(50 + indent, y, safe[:120])
        except Exception:
            c.drawString(50 + indent, y, safe.encode("ascii", "replace").decode("ascii")[:120])
        y -= line_h

    draw_line("Classroom blackboard analytics - teaching feedback report")
    y -= 8
    draw_line(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    y -= 8

    # Section 1: recognized lines from the board image
    draw_line("1. Board OCR")
    if board_lines:
        for t in board_lines:
            draw_line(f"- {t}", indent=10)
    else:
        draw_line("(no text recognized)", indent=10)
    y -= 8

    # Section 2: legibility score and short advice
    draw_line("2. Handwriting clarity")
    if clarity:
        draw_line(f"Level: {clarity.get('clarity', '')}  Score: {clarity.get('score', '')}", indent=10)
        draw_line(f"Suggestion: {clarity.get('suggestion', '')}", indent=10)
        draw_line(
            f"Laplacian variance: {clarity.get('laplacian_variance', '')}  "
            f"Stroke width variance (across components): {clarity.get('stroke_width_variance', '')}",
            indent=10,
        )
    y -= 8

    # Section 3: raw transcript
    draw_line("3. Speech summary (Whisper)")
    st = speech_text or "(missing or transcription failed)"
    for chunk in range(0, len(st), 90):
        draw_line(st[chunk : chunk + 90], indent=10)
    y -= 8

    # Section 4: do board and speech agree?
    draw_line("4. Board vs speech alignment")
    if alignment:
        draw_line(f"Semantic similarity: {alignment.get('semantic_similarity')}", indent=10)
        draw_line(f"Keyword overlap (Jaccard): {alignment.get('keyword_overlap_rate')}", indent=10)
        draw_line(f"Verdict: {alignment.get('verdict')}", indent=10)
    else:
        draw_line("(skipped or unavailable)", indent=10)

    if module_errors:
        y -= 8
        draw_line("5. Steps that reported an error")
        for step_name, message in module_errors.items():
            if message:
                draw_line(f"{step_name}: {message}", indent=10)

    c.save()
    return str(path.resolve())


def process_end(output_path, payload):
    lines = payload.get("board_lines") or []
    clarity_block = payload.get("clarity") or {}
    align_block = payload.get("alignment")
    spoken = payload.get("speech_text") or ""
    failures = payload.get("module_errors")

    out = {"pdf_path": None, "error": None}
    out["pdf_path"] = build_teaching_feedback_pdf(
        output_path,
        board_lines=lines,
        clarity=clarity_block,
        alignment=align_block,
        speech_text=spoken,
        module_errors=failures,
    )
    return out

def laplacian_variance(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    g = gray
    if len(g.shape) == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap.var())


def _binarize_for_strokes(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray) if len(gray.shape) == 2 else clahe.apply(cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    return bw

def _stroke_widths_per_component(binary_inv: np.ndarray) -> Tuple[List[float], float]:
    dist = cv2.distanceTransform((binary_inv > 0).astype(np.uint8), cv2.DIST_L2, 5)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_inv > 0).astype(np.uint8))
    widths: List[float] = []
    h, w = binary_inv.shape[:2]
    min_area = max(20, int(0.0001 * h * w))
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = labels == i
        vals = dist[mask]
        if vals.size == 0:
            continue
        m = float(np.mean(vals))
        if m > 0.5:
            widths.append(m)
    if len(widths) < 2:
        var_between = float(np.var(widths)) if widths else 0.0
        return widths, var_between
    var_between = float(np.var(widths))
    return widths, var_between

def _score_to_clarity(
    lap_var: float,
    stroke_var: float,
    lap_clear: float,
    lap_messy: float,
    stroke_messy_min: float,
) -> Tuple[str, float, str]:
    # Map Laplacian variance to 0..1 (higher variance = sharper photo).
    if lap_var >= lap_clear:
        sharpness_01 = 1.0
    elif lap_var <= lap_messy:
        sharpness_01 = 0.0
    else:
        sharpness_01 = (lap_var - lap_messy) / (lap_clear - lap_messy + 1e-6)

    # Penalize uneven stroke width across connected components.
    stroke_penalty = min(1.0, stroke_var / max(stroke_messy_min, 1e-6))
    stroke_consistency_01 = max(0.0, 1.0 - stroke_penalty)

    total = 100.0 * (0.55 * sharpness_01 + 0.45 * stroke_consistency_01)
    total = float(max(0.0, min(100.0, total)))

    if total >= 70:
        label = "clear"
        suggestion = "Good contrast and stroke consistency; expected readable from the back of the room."
    elif total >= 45:
        label = "fair"
        suggestion = "Consider stronger contrast or slower writing; avoid very thin strokes and rapid erase-write cycles."
    else:
        label = "poor"
        suggestion = "Use a thicker marker, increase black/white contrast, and enlarge or zone key terms for readability."

    return label, total, suggestion

