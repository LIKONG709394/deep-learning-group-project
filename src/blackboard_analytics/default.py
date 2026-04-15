FAST_VIDEO_PRESET = {
    "max_keyframes": 14,
    "yolo_monitor_stride_sec": 0.75,
    "yolo_monitor_max_pool": 96,
    "text_harvest_every_sec": 2.5,
    "text_harvest_max_scans": 100,
    "text_harvest_full_frame_printed": True,
    "min_keyframe_score": 40.0,
}
USEPRINTOCR = False
PRINTOCR_FALLBACK = False
HWOCT_FALLBACK = False
DEBUG = False
SIBILING = True
DIRSUFFIX = "_debug"
YOLOMODEL = "yolov8s-worldv2.pt"
_YOLO_WORLD_LOCK = None
MIN_ROI_AREA_RATIO = 0.35
MIN_CHANGE_RATIO = 0.02
MIN_KEYFRAME_SCORE = 45.0
MAX_KEYFRAMES = 8
MIN_ROI_AREA_RATIO = 0.35
LAPLACIAN_CLEAR_MIN = 120.0
LAPLACIAN_MESSY_MAX = 40.0
STROKE_VARIANCE_MESSY_MIN = 8.0
CLARFY_SIZE = (160, 90)
TROCR_ENGINE = "trocr"
ALLOWED_HEIGHT_TWEAK = 0.55
PERPLEX_SCORE_MAX = 800
DODEDUPE = True
MERGE_SUBSTRING_MIN_LEN = 8
MATH_SYMBOLS = set("=+-*/×÷^()[]{}<>∫∑√πΔ·.,:;%")
FILTER_NOISITY_TEXT = True
NOISE_LINE_MIN_CHARS = 14
NOISE_LINE_MIN_LETTERS = 5
WHISPER_MODEL_SIZE = "base"
WHISPER_TASK = "transcribe"
ENABLE_SILENCE_SEG = True,
SILENCE_THERSHOLD = 0.0004,
SILENCE_DURATION_SEC = 1.0,
MIN_SEG_SEC = 2.0,
MAX_SEG_SEC = 20.0,
ANALYSIS_WINDOW_SEC = 0.1,
SKIP_ENERGY_THERSHOLD = 0.00012,
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
FILTER_TIMEOUT_SEC = 45
ALLOWED_RELEVANCE = {
    "highly_relevant",
    "partially_relevant",
    "weakly_relevant",
    "off_topic",
}
DEFAULT_SBERT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KEYWORD_OVERLAP_FOR_PARTIAL = 0.2
HIGH_SIM = 0.72
PARTIAL_SIM = 0.45,
KEYWORD_HIGH = 0.35,