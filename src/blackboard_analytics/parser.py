import json, os, sys
from pathlib import Path
import yaml
from blackboard_analytics.support import makeSiblingDir, mkBroDir
from blackboard_analytics.venv import *
from default import *
from support import *
class Parser:
    def __init__(self, config_path, device):
        self.datetime = None
        self.init_update(config_path)

    def load_config_dict(self, config_path, device):
        self.config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
        if config_path is not None: self.config_path = Path(config_path)
        with open(self.config_path, encoding="utf-8") as f:
            self.config_dict = yaml.safe_load(f)        
    
    def set_video_conf(self):
        env_fast = os.environ.get("BLACKBOARD_VIDEO_FAST", "").strip().lower() in {"1","true","yes","on"}
        self.video_conf = dict(self.config_dict)        
        video_cfg = dict(self.video_conf.get("video") or {})
        if not bool(video_cfg.get("fast_mode")) and not env_fast:
             self.video_conf["video"] = video_cfg             
             self.video_conf = self.video_conf
        else:
            merged = {**video_cfg, **FAST_VIDEO_PRESET}
            merged["fast_mode"] = True
            self.video_conf["video"] = merged
            if bool(merged.get("fast_whisper_tiny")):
                whisper_cfg = dict(self.video_conf.get("whisper") or {})
                whisper_cfg["model_size"] = "tiny"
                self.video_conf["whisper"] = whisper_cfg
        self.keyframe_source = str(self.video_conf.get("keyframe_source", "yolo_monitor") or "yolo_monitor").lower().strip()
        self.min_roi_area_ratio = float(self.video_conf.get("min_roi_area_ratio", MIN_ROI_AREA_RATIO))
        self.min_change_ratio = float(self.video_conf.get("min_change_ratio", MIN_CHANGE_RATIO))
        self.min_keyframe_score = float(self.video_conf.get("min_keyframe_score", MIN_KEYFRAME_SCORE))
        self.max_keyframes = max(1, int(self.video_conf.get("max_keyframes", MAX_KEYFRAMES)))
        self.min_roi_area_ratio = float(self.video_conf.get("min_roi_area_ratio", MIN_ROI_AREA_RATIO))
        self.merge_substring_min_len = int(self.video_conf.get("merge_substring_min_len", MERGE_SUBSTRING_MIN_LEN))
        self.noise_line_min_chars = int(self.video_conf.get("noise_line_min_chars", MERGE_SUBSTRING_MIN_LEN))
        self.noise_line_min_letters = int(self.video_conf.get("noise_line_min_letters", NOISE_LINE_MIN_LETTERS))
    def set_orc_conf(self, device):
        self.trocr_device = device
        self.trocr_conf = self.config_dict.get("trocr") or {}
        self.video_conf = self.config_dict.get("video") or {}
        self.trocr_printed = self.trocr_conf.get("printed_model") or TROCR_PRINTED
        self.trocr_handwriting = self.trocr_conf.get("default_model") or TROCR_DEFAULT
        self.trocr_tPrintedf = self.video_conf.get("prefer_printed_model", USEPRINTOCR)
        self.trocr_printed_fb = self.trocr_conf.get("printed_model_fallback", PRINTOCR_FALLBACK)
        self.trocr_handwriting_fb = self.trocr_conf.get("handwriting_model_fallback", HWOCT_FALLBACK)     
        self.trocr_enginee = normalize_ocr_engine_name(self.trocr_conf.get("ocr_engine", HWOCT_FALLBACK))   
        self.easy_langs = getEasyLangs(self.trocr_conf.get("easyocr_languages"))
        self.paddle_lang = str(self.trocr_conf.get("paddleocr_lang", "en") or "en")


    def set_yolo_conf(self):
        self.yolo_conf = self.config_dict.get("yolo") 
        self.yolo_conf = {} if not isinstance(self.config_dict.get("yolo"), dict)
        self.yolo_d = self.yolo_conf if isinstance(self.yolo_conf, dict) else {}
        self.conf = float(self.yolo_conf.get("conf", 0.25))
        self.iou = float(self.yolo_conf.get("iou", 0.45))
        self.board_class = int(self.yolo_conf.get("blackboard_class_id", 0))
        self.weights_path = self.yolo_conf.get("weights_path")
        self.textclasses = self.yolo_conf.get("text_classes")
        self.en_only = self.yolo_conf.get("english_only_prompts")
        self.yolo_model_name = self.yolo_conf.get("model")
        

    def set_clarity_conf(self):
        self.clarity_conf = self.config_dict.get("clarity")         
        self.laplacian_clear_min = self.clarity_conf.get("laplacian_clear_min", LAPLACIAN_CLEAR_MIN)
        self.laplacian_messy_max = self.clarity_conf.get("laplacian_messy_max", LAPLACIAN_MESSY_MAX)
        self.stroke_variance_messy_min = self.clarity_conf.get("stroke_variance_messy_min", STROKE_VARIANCE_MESSY_MIN)

    def set_whisper_conf(self):
        self.whisper_conf = (self.config_dict or {}).get("whisper", {})
        self.model_size = str(self.whisper_conf.get("model_size", WHISPER_MODEL_SIZE))
        self.lang_raw = self.whisper_conf.get("language")
        self.language = None if self.lang_raw is None else str(self.lang_raw).strip() or None
        self.task = str(self.whisper_conf.get("task", WHISPER_TASK)).strip() or WHISPER_TASK
        self.ip_raw = self.whisper_conf.get("initial_prompt")
        self.initial_prompt = None if self.ip_raw is None else str(self.ip_raw).strip() or None
        self.fp16_opt = self.whisper_conf.get("fp16")
        self.fp16 = self.fp16_opt if isinstance(self.fp16_opt, bool) else None

    def set_deepseek_conf(self):
        self.deepseek_conf = (self.config_dict or {}).get("deepseek")
        self.deepseek_conf = self.deepseek_conf if isinstance(self.deepseek_conf, dict) else {}
        self.deepseek_model = self.deepseek_conf.get("model", DEFAULT_MODEL)
        self.deepseek_timeout_sec = self.deepseek_conf.get(
            "filter_timeout_sec", self.deepseek_conf.get("timeout_sec", FILTER_TIMEOUT_SEC))

    def set_debug_conf(self):
        debug_cfg = self.config_dict.get("debug",{})
        self.debug = bool(debug_cfg.get("enabled", DEBUG))

    def set_relevant_score(self):
        self.verdict_rules = self.config_dict.get("semantic", {})
        self.encoder_opts = self.config_dict.get("sbert", {})
        self.alignment_model_name=str(self.encoder_opts.get("model_name", DEFAULT_SBERT)),
        self.high_sim=float(self.verdict_rules.get("high_match_min", 0.72)),
        self.partial_sim=float(self.verdict_rules.get("partial_min", 0.45)),
        self.keyword_high=float(self.verdict_rules.get("keyword_overlap_high", 0.35)),


    def init_update(self, config_path,device):
        datetime = os.path.getmtime(config_path)
        if datetime == self.datetime: return            
        self.datetime = os.path.getmtime(config_path)
        self.load_config_dict(config_path)
        self.set_video_conf()
        self.set_orc_conf(device)
        self.set_yolo_conf()
    
    