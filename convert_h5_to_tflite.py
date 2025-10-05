# convert_h5_to_tflite.py
# 嘗試將 keras_model.h5 轉換為 model.tflite
# 注意：由於模型載入問題，此腳本可能失敗

import sys
import os

print('=' * 60)
print('Keras .h5 → TensorFlow Lite 轉換工具')
print('=' * 60)
print()

# 檢查檔案
if not os.path.exists('keras_model.h5'):
    print('✗ 找不到 keras_model.h5')
    sys.exit(1)

print('✓ 找到 keras_model.h5')
print()

try:
    import tensorflow as tf
    print(f'✓ TensorFlow 版本：{tf.__version__}')
    print()
    
    # 定義相容的 layer
    class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(*args, **kwargs)
    
    print('步驟 1/3：載入 Keras 模型...')
    try:
        model = tf.keras.models.load_model(
            'keras_model.h5',
            compile=False,
            custom_objects={'DepthwiseConv2D': CompatDepthwiseConv2D}
        )
        print('✓ 模型載入成功')
    except Exception as e:
        print(f'✗ 模型載入失敗：{e}')
        print()
        print('很遺憾，此 .h5 模型與目前環境不相容。')
        print('建議：到 Teachable Machine 重新匯出 TensorFlow Lite 格式')
        sys.exit(1)
    
    print()
    print('步驟 2/3：轉換為 TFLite 格式...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print('✓ 轉換成功')
    
    print()
    print('步驟 3/3：儲存 model.tflite...')
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    file_size = os.path.getsize('model.tflite') / 1024 / 1024
    print(f'✓ 已儲存 model.tflite（{file_size:.2f} MB）')
    print()
    print('=' * 60)
    print('轉換完成！現在可以執行：')
    print('  python3 controller_tflite.py --camera')
    print('=' * 60)
    
except Exception as e:
    print(f'✗ 發生錯誤：{e}')
    print()
    print('建議：到 Teachable Machine 重新匯出 TensorFlow Lite 格式')
    sys.exit(1)
