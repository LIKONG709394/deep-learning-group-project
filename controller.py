# controller.py
# 使用者提供的推論程式已整合成可重用函式，同時支援單張圖片測試或啟動攝影機即時推論

from time import sleep
import argparse
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import sys

# 與遊戲控制整合
from controls import up, down, left, right

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

MODEL_PATH = 'keras_model.h5'  # 注意是小寫
LABELS_PATH = 'labels.txt'
IMG_SIZE = (224, 224)

# 延遲載入模型（僅在需要時載入），避免啟動時就因模型問題而失敗
model = None
class_names = []

def load_model_safe():
    """安全載入模型，處理不同 TF/Keras 版本的兼容性問題"""
    global model, class_names
    
    if not os.path.exists(MODEL_PATH):
        print(f'警告：找不到模型檔案 {MODEL_PATH}，將無法進行預測')
        return False
    
    if not os.path.exists(LABELS_PATH):
        print(f'警告：找不到標籤檔案 {LABELS_PATH}')
        return False
    
    try:
        # 嘗試使用 TF 2.x / Keras 3.x 的方式
        import tensorflow as tf
        
        # 自訂 layer 來處理不相容參數（不使用 decorator，直接傳入 custom_objects）
        class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)
        
        # 嘗試多種載入方式以處理不同 Keras 版本
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH, 
                compile=False,
                custom_objects={'DepthwiseConv2D': CompatDepthwiseConv2D}
            )
        except Exception as e1:
            # 如果上面失敗，嘗試使用 legacy 方式
            print(f'第一次載入嘗試失敗，嘗試備用方法...')
            try:
                from tensorflow import keras
                model = keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                    custom_objects={'DepthwiseConv2D': CompatDepthwiseConv2D}
                )
            except Exception as e2:
                raise Exception(f'所有載入方法都失敗: {e1}, {e2}')
        
        # 載入標籤
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print(f'✓ 模型載入成功！類別數：{len(class_names)}')
        return True
        
    except Exception as e:
        print(f'✗ 模型載入失敗：{e}')
        print('提示：此模型可能需要特定版本的 TensorFlow/Keras')
        print('建議方案：')
        print('  1. 在 Teachable Machine 重新匯出時選擇 "TensorFlow Lite" 格式')
        print('  2. 或使用 Python 3.10 + tensorflow==2.11.0 重新建立虛擬環境')
        return False

# Create the array of the right shape to feed into the keras model
_data_placeholder = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Predict from a PIL Image (implements the user's snippet)
def predict_from_pil(pil_image):
    if model is None:
        print('模型尚未載入')
        return None, 0.0
    
    # resizing the image to be at least 224x224 and then cropping from the center
    size = IMG_SIZE
    image = ImageOps.fit(pil_image.convert('RGB'), size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image).astype(np.float32)

    # Normalize the image
    normalized_image_array = (image_array / 127.5) - 1.0

    # Load the image into the array
    _data_placeholder[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(_data_placeholder, verbose=0)
    index = int(np.argmax(prediction))
    class_name = class_names[index] if index < len(class_names) else None
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score

# Map label text to direction and call control functions
def apply_control_from_label(label):
    if not label:
        return
    name = label.lower()
    if 'up' in name:
        up()
    elif 'down' in name or 'bottom' in name:  # 支援 down 或 bottom
        down()
    elif 'left' in name:
        left()
    elif 'right' in name:
        right()

# Predict from image file path (CLI helper)
def predict_from_path(image_path):
    if not os.path.exists(image_path):
        print(f'找不到圖片檔案：{image_path}')
        return None, 0.0
    
    image = Image.open(image_path).convert('RGB')
    class_name, confidence = predict_from_pil(image)
    # Print prediction and confidence score
    print('Class:', class_name if class_name is not None else 'None')
    print('Confidence Score:', confidence)
    return class_name, confidence

# Camera loop: read frames, predict, and apply control
def camera_loop(camera_id=0, interval=0.12, conf_threshold=0.5):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print('無法開啟攝影機')
        return
    
    print(f'攝影機已啟動，按 Ctrl+C 結束')
    print(f'信心閾值：{conf_threshold}，預測間隔：{interval}秒')
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                sleep(interval)
                continue
            # OpenCV BGR -> PIL RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            label, conf = predict_from_pil(pil)
            print(f'label={label} conf={conf:.3f}', end='')
            if conf >= conf_threshold and label:
                apply_control_from_label(label)
                print(f' -> 已套用控制')
            else:
                print()
            sleep(interval)
    except KeyboardInterrupt:
        print('\n已停止攝影機')
    finally:
        cap.release()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Keras 影像分類控制器')
    p.add_argument('--image', '-i', help='單次預測：指定圖片路徑')
    p.add_argument('--camera', '-c', action='store_true', help='啟動攝影機即時預測並控制遊戲')
    p.add_argument('--threshold', '-t', type=float, default=0.5, help='信心閾值（預設 0.5）')
    args = p.parse_args()

    # 載入模型
    if not load_model_safe():
        print('因模型載入失敗而退出')
        sys.exit(1)

    if args.image:
        predict_from_path(args.image)
    elif args.camera:
        camera_loop(conf_threshold=args.threshold)
    else:
        print('未指定動作。使用 --image <路徑> 或 --camera')
        p.print_help()
