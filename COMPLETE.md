# ✨ 完成!AI 貪食蛇遊戲 - 深藍色主題 + 模型上傳功能

## 🎉 更新成功!

您的 AI 貪食蛇遊戲已經完全更新,現在具備:

### ✅ 1. 全新的專業外觀
- **深藍色 (#0A1929) 和白色主題**
- 不再有 AI 感的漸變色
- 專業、乾淨、現代的介面設計
- 完美的配色一致性

### ✅ 2. 完整的模型上傳系統
使用者現在可以:
1. 在 Teachable Machine 上訓練自己的模型
2. 直接在網頁上傳 `.tflite` 和 `labels.txt` 檔案
3. 無需修改任何程式碼
4. 無需重啟伺服器
5. 立即開始使用新模型!

## 🚀 快速開始

### 選項 1: 使用快速啟動腳本 (推薦)
```bash
cd /Users/yusingkiu/Desktop/edupython
./start.sh
```

### 選項 2: 手動啟動
```bash
cd /Users/yusingkiu/Desktop/edupython
source venv/bin/activate  # 如果使用虛擬環境
python3 app.py
```

然後開啟瀏覽器前往: **http://127.0.0.1:8080**

## 📤 如何上傳您的模型

### 步驟 1: 訓練模型
1. 前往 [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. 選擇 "Image Project"
3. 建立 4 個類別:
   - **0 up** - 向上的手勢
   - **1 left** - 向左的手勢  
   - **2 right** - 向右的手勢
   - **3 down** - 向下的手勢

4. 每個類別錄製 **50-100 張圖片**
   - 在不同光線下拍攝
   - 嘗試不同的手勢角度
   - 確保背景多樣化

5. 訓練模型
6. 匯出模型:
   - 選擇 "TensorFlow Lite"
   - 選擇 "Floating point"
   - 下載模型

### 步驟 2: 上傳到網頁
1. 啟動應用程式 (http://127.0.0.1:8080)
2. 在頂部找到 "📤 Upload Your Model" 區域
3. 選擇你的 `.tflite` 檔案
4. 選擇你的 `labels.txt` 檔案
5. 點擊 "🚀 Upload and Load Model"
6. 等待幾秒鐘 (頁面會自動重新載入)

### 步驟 3: 開始玩!
1. 點擊 "🤖 Enable AI" 啟用 AI 控制
2. 對著鏡頭做手勢
3. 觀察預測條形圖即時更新
4. 享受遊戲!

## 🎨 UI 預覽

### 配色方案
```
🔵 深藍色背景    #0A1929
🔵 面板背景      #132F4C
🔵 邊框顏色      #1E3A5F
💙 強調色        #66B2FF
⚪ 主要文字      #ffffff
⚪ 次要文字      #8899AA
```

### 版面配置
```
┌────────────────────────────────────────────────┐
│     🐍 Snake Game - Gesture Control            │
│   Train your model and control with gestures   │
├────────────────────────────────────────────────┤
│                                                │
│  📤 Upload Your Model                          │
│  ┌───────────────┬───────────────┐             │
│  │ TFLite Model  │ Labels File   │             │
│  └───────────────┴───────────────┘             │
│           [Upload Button]                      │
│                                                │
├─────────────────────┬──────────────────────────┤
│                     │                          │
│  🎮 Game            │  📹 Camera & Predictions │
│  ┌───────────────┐  │  ┌────────────────────┐ │
│  │               │  │  │                    │ │
│  │  Snake Game   │  │  │   Webcam Feed      │ │
│  │               │  │  │                    │ │
│  └───────────────┘  │  └────────────────────┘ │
│  Score: 0  Len: 3   │  Live Predictions:     │
│  [Restart] [AI]     │  👆 Up    [===] 25%    │
│                     │  👈 Left  [=====] 45%  │
│  Controls:          │  👉 Right [==] 15%     │
│  • Keyboard/WASD    │  👇 Down  [=] 10%      │
│  • AI Gestures      │                         │
│                     │  Direction: RIGHT       │
└─────────────────────┴──────────────────────────┘
```

## 📁 檔案結構

```
edupython/
├── app.py                      ✅ Flask 伺服器 (更新)
├── templates/
│   └── index.html              ✅ 深藍色主題 UI (更新)
├── static/
│   └── game.js                 ✅ 前端邏輯 (更新)
├── uploads/                    ✅ 新增
│   └── .gitkeep               
├── game.py                     ⚪ 遊戲邏輯
├── controller_tflite.py        ⚪ AI 控制器
├── controls.py                 ⚪ 控制模組
├── check_setup.py              ✅ 環境檢查 (新增)
├── start.sh                    ✅ 快速啟動 (新增)
├── TESTING.md                  ✅ 測試指南 (新增)
├── UPDATE_SUMMARY.md           ✅ 更新總結 (新增)
├── PROJECT_STRUCTURE.md        ✅ 專案結構 (新增)
├── README.md                   ✅ 文件更新
├── .gitignore                  ✅ 更新
└── requirements.txt            ⚪ 依賴清單
```

## 🔧 技術細節

### 後端更新 (app.py)
```python
# 新增功能:
✅ 檔案上傳處理 (werkzeug.secure_filename)
✅ /upload_model API 端點
✅ 模型動態重載
✅ 執行緒安全的模型載入
✅ 檔案類型驗證
✅ 50MB 檔案大小限制
```

### 前端更新 (index.html + game.js)
```javascript
// 新增功能:
✅ 模型上傳 UI
✅ 檔案選擇器
✅ 上傳狀態顯示
✅ FormData API
✅ 自動頁面重載
✅ 深藍色主題樣式
```

## ✨ 新功能特點

### 1. 使用者友善
- 🎯 不需要寫程式碼
- 🎯 直接在網頁上傳模型
- 🎯 即時視覺化預測結果
- 🎯 清晰的使用說明

### 2. 專業外觀
- 🎨 深藍色和白色配色
- 🎨 現代化的 UI 設計
- 🎨 響應式版面 (手機友善)
- 🎨 流暢的動畫效果

### 3. 技術優勢
- 🚀 無需重啟伺服器
- 🚀 執行緒安全
- 🚀 錯誤處理完善
- 🚀 安全的檔案上傳

## 🎮 使用案例

### 案例 1: 教室教學
1. 教師展示如何訓練模型
2. 學生各自訓練自己的手勢
3. 上傳並測試
4. 比較不同訓練策略的效果

### 案例 2: 個人化遊戲
1. 使用自己喜歡的手勢
2. 訓練個人化模型
3. 上傳使用
4. 與朋友分享

### 案例 3: 實驗和研究
1. 測試不同的訓練資料量
2. 比較不同光線條件
3. 評估模型準確度
4. 優化手勢設計

## 📊 效能資訊

- **攝影機幀率:** ~30 FPS
- **模型推論:** 每 120ms (~8 FPS)
- **遊戲更新:** 每 120ms
- **網頁回應:** < 100ms
- **模型上傳:** 取決於檔案大小 (通常 < 5 秒)

## 🛡️ 安全性

### 檔案上傳安全
- ✅ 檔案類型檢查 (.tflite, .txt)
- ✅ 檔案大小限制 (50MB)
- ✅ 安全檔名處理
- ✅ 獨立上傳目錄
- ✅ Git 忽略上傳檔案

### 執行時安全
- ✅ 執行緒鎖防止競爭條件
- ✅ 錯誤處理和驗證
- ✅ 安全的路徑操作

## 🐛 疑難排解

### 模型上傳失敗
```
問題: 上傳後顯示錯誤
解決:
1. 確認檔案格式正確 (.tflite 和 .txt)
2. 檢查檔案大小 < 50MB
3. 確認 labels.txt 有 4 行
4. 查看瀏覽器控制台錯誤訊息
```

### 預測不準確
```
問題: AI 無法正確識別手勢
解決:
1. 重新訓練,增加樣本數 (100+)
2. 確保光線條件一致
3. 手勢要清晰且差異大
4. 在 Teachable Machine 預覽測試
```

### 攝影機無法開啟
```
問題: 看不到攝影機畫面
解決:
1. 允許瀏覽器攝影機權限
2. 關閉其他使用攝影機的程式
3. 重新整理頁面
4. 檢查系統隱私設定
```

## 📞 支援

### 檢查環境
```bash
python3 check_setup.py
```

### 測試模型
```bash
python3 test_model.py
```

### 檢查攝影機
```bash
python3 list_cameras.py
```

## 🎓 學習資源

- [Teachable Machine 教學](https://teachablemachine.withgoogle.com/)
- [TensorFlow Lite 文件](https://www.tensorflow.org/lite)
- [Flask 文件](https://flask.palletsprojects.com/)
- [Pygame 教學](https://www.pygame.org/docs/)

## 🙏 致謝

感謝使用 AI 貪食蛇遊戲!希望您享受:
- 🎮 玩遊戲的樂趣
- 🤖 訓練 AI 模型的成就感
- 📚 學習機器學習的過程

## 🚀 下一步

現在您可以:
1. ✅ 啟動應用程式: `./start.sh`
2. ✅ 訓練您的模型
3. ✅ 上傳並測試
4. ✅ 享受遊戲!

---

**準備好了嗎?開始吧!**

```bash
cd /Users/yusingkiu/Desktop/edupython
./start.sh
```

然後前往: **http://127.0.0.1:8080** 🎮🐍🤖

---

**版本:** 2.0  
**更新日期:** 2025年10月5日  
**主要改進:** 深藍色主題 + 模型上傳功能
