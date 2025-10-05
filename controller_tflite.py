# controller_tflite.py
# 使用 TensorFlow Lite 格式的模型進行推論（相容性更好）

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

MODEL_PATH = 'model_unquant.tflite'  # TensorFlow Lite 模型（未量化版本）
LABELS_PATH = 'labels.txt'
IMG_SIZE = (224, 224)

# 延遲載入模型
interpreter = None
input_details = None
output_details = None
class_names = []

def load_model_safe():
    """載入 TensorFlow Lite 模型"""
    global interpreter, input_details, output_details, class_names
    
    if not os.path.exists(MODEL_PATH):
        print(f'警告：找不到模型檔案 {MODEL_PATH}')
        print('請從 Teachable Machine 匯出 TensorFlow Lite 格式並將 model.tflite 放到專案根目錄')
        return False
    
    if not os.path.exists(LABELS_PATH):
        print(f'警告：找不到標籤檔案 {LABELS_PATH}')
        return False
    
    try:
        import tensorflow as tf
        
        # 載入 TFLite 模型
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        # 取得輸入輸出資訊
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 載入標籤
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print(f'✓ TFLite 模型載入成功！')
        print(f'  輸入形狀：{input_details[0]["shape"]}')
        print(f'  類別數：{len(class_names)}')
        return True
        
    except Exception as e:
        print(f'✗ 模型載入失敗：{e}')
        return False

def predict_from_pil(pil_image):
    """使用 TFLite 進行預測"""
    if interpreter is None:
        print('模型尚未載入')
        return None, 0.0
    
    # 調整圖片大小
    size = IMG_SIZE
    image = ImageOps.fit(pil_image.convert('RGB'), size, Image.Resampling.LANCZOS)
    
    # 轉為 numpy array 並正規化
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0
    
    # 擴展維度以符合模型輸入
    data = np.expand_dims(normalized_image_array, axis=0)
    
    # 執行推論
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    
    # 取得輸出
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    
    index = int(np.argmax(prediction))
    class_name = class_names[index] if index < len(class_names) else None
    confidence_score = float(prediction[index])
    
    return class_name, confidence_score

def apply_control_from_label(label):
    """根據標籤呼叫對應的控制函式"""
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

def predict_from_path(image_path):
    """從圖片檔案進行預測"""
    if not os.path.exists(image_path):
        print(f'找不到圖片檔案：{image_path}')
        return None, 0.0
    
    image = Image.open(image_path).convert('RGB')
    class_name, confidence = predict_from_pil(image)
    print('Class:', class_name if class_name is not None else 'None')
    print('Confidence Score:', confidence)
    return class_name, confidence

# Camera loop: read frames, predict, and apply control
def camera_loop(camera_id=0, interval=0.12, conf_threshold=0.5, scan=False, allow_external=False):
    """攝影機即時預測迴圈

    預設行為：強制使用筆電內建攝影機（camera_id 0）。
    - 無論使用者傳入哪個 camera_id，若未啟用 allow_external，皆會改為使用 0。
    - 若筆電內建鏡頭無法開啟，程式會停止並提示，而不會自動掃描或選用其他裝置（例如手機鏡頭）。
    若你確實要使用外接或手機鏡頭，請加上 CLI 選項 --allow-external（同時可加 --scan 或 --camera-id）。
    """
    # 若未允許外接裝置，強制使用內建鏡頭 id 0
    if not allow_external:
        enforced_camera_id = 0
        if camera_id != enforced_camera_id:
            print(f'警告：未允許外接攝影機，忽略傳入 camera_id {camera_id}，改以內建攝影機 ID {enforced_camera_id}')
        camera_id = enforced_camera_id
        scan = False  # 關閉掃描，避免選到其他裝置

    cap = None

    # 只嘗試使用指定的 camera_id（可能是強制後的內建鏡頭）
    print(f'嘗試以指定攝影機 ID {camera_id} 開啟...')
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'✗ 無法開啟攝影機 ID {camera_id}（內建鏡頭）')
        print('提示：')
        print('  1. 確認已授予攝影機權限')
        print('  2. 確認沒有其他應用程式正在使用攝影機')
        print('  3. 若你確定要使用外接攝影機或手機鏡頭，請加上 --allow-external 選項')
        return

    print(f'✓ 使用攝影機 ID {camera_id}（僅限內建鏡頭）')
    print(f'✓ 攝影機已啟動，按 Ctrl+C 結束')
    print(f'  信心閾值：{conf_threshold}')
    print(f'  預測間隔：{interval}秒')
    print()

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                sleep(interval)
                continue

            frame_count += 1
            # OpenCV BGR -> PIL RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            label, conf = predict_from_pil(pil)

            # 每 5 幀才顯示一次（減少終端輸出）
            if frame_count % 5 == 0:
                print(f'[{frame_count:04d}] {label:15s} {conf:.3f}', end='')
                if conf >= conf_threshold and label:
                    apply_control_from_label(label)
                    print(f' ✓ 已套用')
                else:
                    print()

            sleep(interval)

    except KeyboardInterrupt:
        print('\n✓ 已停止攝影機')
    finally:
        cap.release()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='TensorFlow Lite 影像分類控制器')
    p.add_argument('--image', '-i', help='單次預測：指定圖片路徑')
    p.add_argument('--camera', '-c', action='store_true', help='啟動攝影機即時預測並控制遊戲')
    p.add_argument('--camera-id', type=int, default=0, help='攝影機 ID（預設 0，通常 MacBook 內建是 0）')
    p.add_argument('--threshold', '-t', type=float, default=0.5, help='信心閾值（預設 0.5）')
    p.add_argument('--scan', action='store_true', help='若指定，則自動掃描所有攝影機以尋找可用裝置（預設關閉）')
    p.add_argument('--allow-external', action='store_true', help='若指定，允許使用外接或手機攝影機（預設不允許）')
    args = p.parse_args()

    # 載入模型
    if not load_model_safe():
        print('\n請按照以下步驟重新匯出模型：')
        print('1. 到 Teachable Machine (https://teachablemachine.withgoogle.com/)')
        print('2. 開啟你的專案')
        print('3. 點選 "Export Model"')
        print('4. 選擇 "TensorFlow" -> "TensorFlow Lite"')
        print('5. 下載後解壓縮，將 model.tflite 和 labels.txt 複製到此資料夾')
        sys.exit(1)

    if args.image:
        predict_from_path(args.image)
    elif args.camera:
        # 強制僅使用筆電內建鏡頭，除非使用 --allow-external
        camera_loop(camera_id=args.camera_id, conf_threshold=args.threshold, scan=args.scan, allow_external=args.allow_external)
    else:
        print('未指定動作。使用 --image <路徑> 或 --camera')
        p.print_help()
