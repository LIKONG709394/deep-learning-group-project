from blackboard_analytics.module_e_report import run_module_e
from support import *
from support_ff import *
from parser import *
from default import *
import threading

def analyze_board(_box, frame_bgr, enginee, config, env):
    box = _box["box"]
    cropped = crop_roi(frame_bgr, box)
    lines = []
    arranged_iny_ocr_rois = []
    ocr_rois = []
    OCR = False
    if enginee["name"] == "easyocr":
        OCR = True
        ocr_result = recognize_lines_easyocr(cropped, config.easy_langs, env)
        ocr_rois = easyocr_detections_to_entries(ocr_result)
    if enginee["name"] == "paddleocr":
        OCR = True
        ocr_result = recognize_lines_paddleocr(cropped, config.easy_langs, env)
        ocr_rois = paddle_detections_to_entries(ocr_result)
    if OCR:
        arranged_ocr_rois = arrange_bboxes(ocr_rois, 0, 1) 
        lines, arranged_iny_ocr_rois = merge_lines(arranged_ocr_rois, ALLOWED_HEIGHT_TWEAK)
    else:
        lines = recognize_lines_other_ocr(cropped, config.trocr_enginee, env)
    frame_lines+=lines
    frame_arranged_iny_ocr_rois+=arranged_iny_ocr_rois
    _box["ocr"][enginee["name"]] = ocr_rois                
    return _box

def analyze_frame(video_path,frame_dict,config,env,latest_cropped_img_trait,the_highest_clarity):
    frame_h, frame_w = frame_dict["image_bgr"].shape[:2]
    frame_bgr = frame_dict["image_bgr"]
    boxes = []
    yolo_world_prediction = get_textarea_yolo_world(
        frame_bgr,            
        config.yolo_model_name,
        config.conf, config.iou, 
        config.textclasses,config.eng_only
    )
    if config.debug: 
        yolo_world_prediction.save(
            f"yw_{Path(video_path).name}_{frame_dict["frame_index"]}.jpg")
    boxes+=get_boxes(yolo_world_prediction, frame_w, frame_h)
    YW_largest = get_largest_box(boxes)
    yolo_prediction = get_textarea_yolo(
        frame_bgr,            
        config.weights_path,
        config.conf, config.iou
    )
    if config.debug: 
        yolo_prediction.save(
            f"yo_{Path(video_path).name}_{frame_dict["frame_index"]}.jpg")
    boxes+=get_boxes(yolo_prediction, frame_w, frame_h)
    YO_largest = get_largest_box(boxes)        
    orc_enginees = [{"name":"easyocr"},{"name":"easyocr"},{"name":"default"}]        
    perplex_model, perplex_tokenizer = initTextScoreModel()
    sel_lines = []
    sel_arranged_iny_ocr_rois = []
    sel_score = 1000
    sel_char = 0 
    sel_ocr = ""
    traited_boxes = []
    highest_clarity = 0        
    for box in enumerate(boxes):                
        absTrait = getAbsTrait(cropped, CLARFY_SIZE)
        relTrait = getRelTrait(latest_cropped_img_trait, absTrait)
        cropped = crop_roi(frame_bgr, box)            
        clarity = evaluate_handwriting_clarity(
            cropped,
            laplacian_clear_min=config.laplacian_clear_min,
            laplacian_messy_max=config.laplacian_messy_max,
            stroke_variance_messy_min=config.stroke_variance_messy_min
        )
        if clarity>highest_clarity: highest_clarity = clarity
        traited_boxes.append({"box":box,"trait":relTrait,"ocr_rois":{}})        
    for enginee in orc_enginees:        
        frame_lines = []
        frame_arranged_iny_ocr_rois = []
        for _box in enumerate(traited_boxes):                
            _box = analyze_board(_box,frame_bgr, enginee, config, env)        
        frame_paragraph = " ".join(
            (line or "").strip() for line in frame_lines).strip()
        enginee["score"] = calculate_perplexity(
            frame_paragraph, perplex_tokenizer, perplex_model)
        enginee["useful_count"] = sum(ch.isalnum() for ch in frame_paragraph)            
        if enginee["score"] < sel_score or enginee["useful_count"] > sel_char:
            sel_lines = frame_lines
            sel_arranged_iny_ocr_rois = frame_arranged_iny_ocr_rois  
            sel_ocr = enginee["name"]
        if enginee["score"] < PERPLEX_SCORE_MAX:break
        if enginee["useful_count"]>12:break
    for box in enumerate(traited_boxes): 
        traited_boxes["ocr_rois"]=traited_boxes["ocr_rois"][sel_ocr]
    if highest_clarity > the_highest_clarity: the_highest_clarity = highest_clarity
    frame_dict["traited_boxes"] = traited_boxes
    frame_dict["highest_clarity"] = highest_clarity            
    frame_dict["lines"] = sel_lines
    frame_dict["paragraph"] = " ".join(
        (line or "").strip() for line in sel_lines).strip()
    frame_dict["arranged_iny_ocr_rois"] = sel_arranged_iny_ocr_rois
    frame_dict["ocr"] = sel_ocr
    aggregated_board_lines = merge_unique_lines(aggregated_board_lines, sel_lines)
    if config.debug: 
        save_annotated_image(
            frame_dict["frame_bgr"], frame_dict["traited_boxes"],
            f"orc_{Path(video_path).name}_{frame_dict["frame_index"]}.jpg")            
    return latest_cropped_img_trait,the_highest_clarity

def summarize(aggregated_board_lines, entire_speech, config, env):
    if DODEDUPE:
        board_lines_primary = dedupe_subsumed_lines(
            aggregated_board_lines,
            min_len=max(4, int(config.merge_substring_min_len)))
    if FILTER_NOISITY_TEXT:
        board_lines_primary = filter_noise_board_lines(
            board_lines_primary,
            min_chars=max(4, config.noise_line_min_char),
            min_letters=max(2, config.noise_line_min_letter),
        )
    messages = build_filter_board_lines_messages(
            board_lines_primary, entire_speech),
    deepseek_filted_lines = deepseek_filter_lines(
        messages, config, env
    )
    return board_lines_primary, deepseek_filted_lines

def critises(config, env, deepseek_filted_lines, entire_speech):
    sbert = getAligmentModel(config.alignment_model_name)
    similarity = sbert.similarity(deepseek_filted_lines, entire_speech)
    token_overlap = keyword_overlap_rate(deepseek_filted_lines, entire_speech)
    verdict = judge_alignment(
        similarity,
        token_overlap,
        high_sim=HIGH_SIM,
        partial_sim=PARTIAL_SIM,
        keyword_high=KEYWORD_HIGH,
    )
    lesson_alignment = {
        "semantic_similarity": round(similarity, 4),
        "keyword_overlap_rate": round(token_overlap, 4),
        "verdict": verdict,
    }
    messages = build_alignment_messages(deepseek_filted_lines, entire_speech)
    deepseek_alignment = deepseek_filter_lines(
        messages, config, env
    )
    return similarity, token_overlap, verdict, lesson_alignment, deepseek_alignment

def output(
        board_lines_primary, the_highest_clarity, alignment_summary, entire_speech,
        pdf_path, debug_dir, video_path, config, segments_speech, deepseek_alignment):
    bundle_for_pdf = {
        "board_lines": board_lines_primary,
        "clarity": the_highest_clarity,
        "alignment": alignment_summary,
        "speech_text": entire_speech
    }
    pdf_bundle = run_module_e(pdf_path, bundle_for_pdf)
    metadata_path = None
    if debug_dir is not None:
        metadata_path = debug_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "video_path": str(Path(video_path).resolve()),
                    "pdf_output": str(Path(pdf_path).resolve()),
                    "debug_dir": str(debug_dir.resolve()),
                    "video_fast_mode": bool(config.fast),
                    "keyframes": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    result: Dict[str, Any] = {
        "input_mode": "video",
        "video_fast_mode": bool(config.fast),
        "board_texts": board_lines_primary,
        "board_roi": [],
        "roi_method": [],
        "clarity": the_highest_clarity,
        "speech_text": entire_speech,
        "speech_segments": segments_speech,
        "alignment": alignment_summary,
        "deepseek_alignment": deepseek_alignment,
        "pdf_path": pdf_bundle.get("pdf_path"),
        "video_keyframes": [],
        "video_debug_dir": str(
            debug_dir.resolve()) if debug_dir is not None else None,
        "video_debug_metadata": str(
            metadata_path.resolve()) if metadata_path is not None else None,
    }
    return result

def run_from_video_file(video_path, config, env, pdf_path, debug_dir):
    if config.debug: debug_dir = makeSiblingDir(pdf_path, video_path) 
    
    audio_path = extract_audio_ffmpeg(video_path, "./audio/wav")        
    frame_dicts_fm_ffmpeg = extract_iframe_frames_ffmpeg(video_path)
    
    latest_cropped_img_trait = None
    aggregated_board_lines = []
    the_highest_clarity = 0
    for stride in [5, 0]:
        frame_dicts_fm_opencv = opencv_extract_frame_dicts(video_path,stride)
        for frame_dict in frame_dicts_fm_opencv: 
            latest_cropped_img_trait,the_highest_clarity = analyze_frame(
                video_path, frame_dict, config, env, latest_cropped_img_trait)
    board_lines_primary = aggregated_board_lines

    entire_speech, segments_speech = extract_pause_from_audio(
        audio_path, config, env
    )

    board_lines_primary, deepseek_filted_lines = summarize(board_lines_primary, entire_speech)

    similarity, token_overlap, verdict, lesson_alignment, deepseek_alignment = critises(
        config, env, deepseek_filted_lines, entire_speech)
    
    alignment_summary = lesson_alignment.get("alignment") 
    
    result = output(
        board_lines_primary, the_highest_clarity, alignment_summary, entire_speech,
        pdf_path, debug_dir, video_path, config, segments_speech, deepseek_alignment)
    
    return result
    
    
    
def run_from_image_and_audio_files(image_path, audio_path, config, env, pdf_path="output/teaching_feedback.pdf"):
    if config.debug: debug_dir = makeSiblingDir(pdf_path, image_path) 
    
    latest_cropped_img_trait = None
    aggregated_board_lines = []
    the_highest_clarity = 0

    frame_dict = {
        "frame_index": 0,
        "timestamp_sec": 0,
        "frame_bgr": load_bgr_image(image_path),
    }

    latest_cropped_img_trait,the_highest_clarity = analyze_frame(
        image_path, frame_dict, config, env, latest_cropped_img_trait)
    board_lines_primary = aggregated_board_lines

    entire_speech, segments_speech = extract_pause_from_audio(
        audio_path, config, env
    )

    board_lines_primary, deepseek_filted_lines = summarize(board_lines_primary, entire_speech)

    similarity, token_overlap, verdict, lesson_alignment, deepseek_alignment = critises(
        config, env, deepseek_filted_lines, entire_speech)
    
    alignment_summary = lesson_alignment.get("alignment") 
    
    result = output(
        board_lines_primary, the_highest_clarity, alignment_summary, entire_speech,
        pdf_path, debug_dir, image_path, config, segments_speech, deepseek_alignment)
        
    return result
  
        

def start_pipeline(video_path, config_path, pdf_path, img_path, audio_path):
    env = ENV()
    config = Parser(config_path, env.device)
    if video_path is not None:
        return run_from_video_file(video_path, pdf_path, config, env)
    if img_path is not None and audio_path is not None:
        return run_from_image_and_audio_files(
            img_path, audio_path, config, env, pdf_path)

        
