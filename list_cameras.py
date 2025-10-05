#!/usr/bin/env python3
# list_cameras.py
# 列出所有可用的攝影機裝置

import cv2

print('=' * 60)
print('掃描可用的攝影機裝置')
print('=' * 60)
print()

found_cameras = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f'✓ 攝影機 ID {i}：')
            print(f'    解析度：{width} x {height}')
            
            # 嘗試取得裝置名稱（不是所有系統都支援）
            try:
                backend = cap.getBackendName()
                print(f'    後端：{backend}')
            except:
                pass
            
            found_cameras.append(i)
            print()
        cap.release()

print('=' * 60)
if found_cameras:
    print(f'找到 {len(found_cameras)} 個攝影機：{found_cameras}')
    print()
    print('使用方式：')
    print(f'  python3 controller_tflite.py --camera --camera-id {found_cameras[0]}')
    if len(found_cameras) > 1:
        print()
        print('若要使用其他攝影機，請嘗試：')
        for cam_id in found_cameras[1:]:
            print(f'  python3 controller_tflite.py --camera --camera-id {cam_id}')
else:
    print('✗ 沒有找到可用的攝影機')
    print('請檢查：')
    print('  1. 攝影機權限是否已授予')
    print('  2. 沒有其他應用程式正在使用攝影機')
print('=' * 60)
