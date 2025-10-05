# test_model.py
# 測試模型預測是否正常運作

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import cv2

print('=' * 60)
print('TFLite 模型測試工具')
print('=' * 60)
print()

# 載入模型
print('步驟 1：載入模型...')
try:
    interpreter = tf.lite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('✓ 模型載入成功')
    print(f'  輸入形狀：{input_details[0]["shape"]}')
    print(f'  輸入類型：{input_details[0]["dtype"]}')
    print(f'  輸出形狀：{output_details[0]["shape"]}')
except Exception as e:
    print(f'✗ 模型載入失敗：{e}')
    exit(1)

# 載入標籤
print()
print('步驟 2：載入標籤...')
with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print(f'✓ 載入 {len(class_names)} 個類別：')
for i, name in enumerate(class_names):
    print(f'  [{i}] {name}')

# 開啟攝影機
print()
print('步驟 3：開啟攝影機...')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('✗ 無法開啟攝影機')
    exit(1)
print('✓ 攝影機已開啟')

print()
print('=' * 60)
print('開始即時預測（按 Q 或 Ctrl+C 結束）')
print('=' * 60)
print()

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # 轉換並預處理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # 調整大小
        image = ImageOps.fit(pil_image, (224, 224), Image.Resampling.LANCZOS)
        
        # 轉為 numpy 並正規化
        image_array = np.asarray(image).astype(np.float32)
        normalized = (image_array / 127.5) - 1.0
        
        # 擴展維度
        data = np.expand_dims(normalized, axis=0)
        
        # 執行預測
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # 取得結果
        index = int(np.argmax(prediction))
        confidence = float(prediction[index])
        label = class_names[index] if index < len(class_names) else 'Unknown'
        
        # 顯示結果（每 10 幀顯示一次）
        if frame_count % 10 == 0:
            print(f'幀 {frame_count:04d}:')
            print(f'  預測類別：{label}')
            print(f'  信心度：{confidence:.3f}')
            print(f'  所有機率：')
            for i, prob in enumerate(prediction):
                print(f'    [{i}] {class_names[i]:20s} {prob:.3f} {"█" * int(prob * 50)}')
            print()
        
        # 在畫面上顯示結果
        cv2.putText(frame, f'{label}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 顯示視窗
        cv2.imshow('Model Test - Press Q to quit', frame)
        
        # 按 Q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print('\n已停止')
finally:
    cap.release()
    cv2.destroyAllWindows()

print()
print('測試完成！')
