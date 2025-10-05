# app.py
# Flask 網頁應用程式 - AI 控制貪食蛇遊戲

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import threading
import time

app = Flask(__name__)

# 全域變數
MODEL_PATH = 'model_unquant.tflite'
LABELS_PATH = 'labels.txt'
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.5

# 模型和攝影機
interpreter = None
input_details = None
output_details = None
class_names = []
camera = None
latest_predictions = [0.0, 0.0, 0.0, 0.0]
latest_direction = None
camera_lock = threading.Lock()

def load_model():
    """載入 TFLite 模型"""
    global interpreter, input_details, output_details, class_names
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print(f'✓ 模型載入成功！類別數：{len(class_names)}')
        return True
    except Exception as e:
        print(f'✗ 模型載入失敗：{e}')
        return False

def init_camera(camera_id=0):
    """初始化攝影機"""
    global camera
    
    with camera_lock:
        if camera is not None:
            camera.release()
        
        camera = cv2.VideoCapture(camera_id)
        if camera.isOpened():
            print(f'✓ 攝影機 {camera_id} 已開啟')
            return True
        else:
            print(f'✗ 無法開啟攝影機 {camera_id}')
            return False

def predict_frame(frame):
    """對單一幀進行預測"""
    global latest_predictions, latest_direction
    
    if interpreter is None:
        return None, None
    
    try:
        # 轉換為 PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # 調整大小並正規化
        image = ImageOps.fit(pil_image, IMG_SIZE, Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)
        normalized = (image_array / 127.5) - 1.0
        data = np.expand_dims(normalized, axis=0)
        
        # 執行推論
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # 更新最新預測
        latest_predictions = prediction.tolist()
        
        # 取得最高信心度的類別
        index = int(np.argmax(prediction))
        confidence = float(prediction[index])
        label = class_names[index] if index < len(class_names) else None
        
        # 映射到方向
        if confidence >= CONF_THRESHOLD and label:
            direction = map_label_to_direction(label)
            if direction:
                latest_direction = direction
                return label, confidence
        
        return label, confidence
        
    except Exception as e:
        print(f'預測錯誤：{e}')
        return None, None

def map_label_to_direction(label):
    """將標籤映射到方向"""
    if not label:
        return None
    
    name = label.lower()
    if 'up' in name:
        return 'up'
    elif 'left' in name:
        return 'left'
    elif 'right' in name:
        return 'right'
    elif 'down' in name or 'bottom' in name:
        return 'down'
    return None

def generate_frames():
    """生成視訊串流"""
    global camera
    
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                time.sleep(0.1)
                continue
            
            success, frame = camera.read()
            if not success:
                time.sleep(0.1)
                continue
        
        # 執行預測
        label, confidence = predict_frame(frame)
        
        # 在幀上繪製預測結果
        if label and confidence:
            text = f'{label}: {confidence:.2f}'
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 翻轉影像（鏡像效果）
        frame = cv2.flip(frame, 1)
        
        # 編碼為 JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """視訊串流端點"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def get_prediction():
    """取得最新預測結果"""
    return jsonify({
        'predictions': latest_predictions,
        'direction': latest_direction
    })

@app.route('/start_camera/<int:camera_id>')
def start_camera(camera_id):
    """啟動指定攝影機"""
    success = init_camera(camera_id)
    return jsonify({'success': success})

if __name__ == '__main__':
    print('=' * 60)
    print('AI 控制貪食蛇 - Flask 網頁版')
    print('=' * 60)
    print()
    
    # 載入模型
    if not load_model():
        print('模型載入失敗，請檢查檔案是否存在')
        exit(1)
    
    # 初始化攝影機
    init_camera(0)
    
    print()
    print('伺服器啟動中...')
    print('請開啟瀏覽器並前往：http://127.0.0.1:8080')
    print('按 Ctrl+C 停止伺服器')
    print('=' * 60)
    print()
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
