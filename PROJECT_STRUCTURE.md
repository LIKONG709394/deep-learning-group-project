# 📂 專案結構

```
edupython/
│
├── 🌐 Web 應用程式
│   ├── app.py                      # Flask 伺服器 (主程式)
│   ├── templates/
│   │   └── index.html              # 網頁 UI (深藍色主題)
│   └── static/
│       └── game.js                 # 前端遊戲邏輯
│
├── 🎮 遊戲核心
│   ├── game.py                     # Pygame 貪食蛇遊戲
│   └── controls.py                 # 控制邏輯模組
│
├── 🤖 AI 控制器
│   ├── controller_tflite.py        # TFLite 手勢辨識
│   ├── controller.py               # Keras 版本控制器
│   └── test_model.py               # 模型測試工具
│
├── 📤 模型檔案
│   ├── model_unquant.tflite        # 預設 TFLite 模型
│   ├── labels.txt                  # 類別標籤 (up/left/right/down)
│   └── uploads/                    # 使用者上傳的模型
│       ├── .gitkeep                # Git 目錄追蹤
│       ├── uploaded_model.tflite   # (上傳後生成)
│       └── uploaded_labels.txt     # (上傳後生成)
│
├── 🔧 工具和測試
│   ├── check_setup.py              # 環境檢查腳本
│   ├── list_cameras.py             # 攝影機偵測工具
│   ├── start.sh                    # 快速啟動腳本
│   └── convert_h5_to_tflite.py     # H5 轉 TFLite 工具
│
├── 📚 文件
│   ├── README.md                   # 專案說明文件
│   ├── TESTING.md                  # 測試指南
│   ├── UPDATE_SUMMARY.md           # 更新總結
│   └── PROJECT_STRUCTURE.md        # 本檔案
│
├── ⚙️ 設定檔
│   ├── requirements.txt            # Python 依賴套件
│   └── .gitignore                  # Git 忽略規則
│
└── 🐍 Python 環境
    └── venv/                       # 虛擬環境 (自動建立)

```

## 📋 檔案說明

### 核心應用程式檔案

| 檔案 | 用途 | 重要性 |
|------|------|--------|
| `app.py` | Flask 網頁伺服器,處理路由和模型上傳 | ⭐⭐⭐⭐⭐ |
| `game.py` | Pygame 貪食蛇遊戲邏輯 | ⭐⭐⭐⭐⭐ |
| `controller_tflite.py` | 使用 TFLite 模型進行即時手勢辨識 | ⭐⭐⭐⭐ |
| `controls.py` | 遊戲控制抽象層 | ⭐⭐⭐ |

### Web 介面檔案

| 檔案 | 用途 | 技術 |
|------|------|------|
| `templates/index.html` | 網頁 UI,深藍色主題設計 | HTML5 + CSS3 |
| `static/game.js` | 前端遊戲邏輯和 API 通訊 | JavaScript (ES6+) |

### 工具檔案

| 檔案 | 用途 | 使用時機 |
|------|------|----------|
| `check_setup.py` | 檢查環境和依賴 | 初次設定或除錯 |
| `list_cameras.py` | 列出可用攝影機 | 攝影機問題排查 |
| `test_model.py` | 測試模型預測 | 驗證模型準確度 |
| `start.sh` | 自動化啟動腳本 | 快速啟動應用程式 |

### 模型檔案

| 檔案 | 用途 | 可更換 |
|------|------|--------|
| `model_unquant.tflite` | 預設的 TFLite 模型 | ✅ 是 |
| `labels.txt` | 類別標籤檔案 | ✅ 是 |
| `uploads/` | 使用者上傳的模型儲存位置 | 🔄 動態 |

## 🔄 資料流程

### 模型上傳流程
```
使用者選擇檔案
    ↓
index.html (uploadFiles())
    ↓
POST /upload_model
    ↓
app.py (驗證 + 儲存)
    ↓
load_model() (重新載入)
    ↓
頁面重新整理
    ↓
使用新模型
```

### 遊戲控制流程
```
攝影機擷取畫面
    ↓
predict_frame() (TFLite 推論)
    ↓
更新 latest_predictions
    ↓
/prediction API
    ↓
game.js (輪詢)
    ↓
setDirection()
    ↓
蛇移動
```

## 📦 依賴關係

```
app.py
├── Flask (網頁伺服器)
├── OpenCV (攝影機和影像處理)
├── TensorFlow Lite (模型推論)
├── PIL (影像預處理)
└── NumPy (數值運算)

game.py
└── Pygame (遊戲引擎)

controller_tflite.py
├── OpenCV
├── TensorFlow Lite
├── PIL
└── controls.py
```

## 🎯 使用場景

### 1. 網頁版 (推薦)
```bash
./start.sh
# 或
python3 app.py
```
- ✅ 最簡單的使用方式
- ✅ 可視覺化預測結果
- ✅ 可直接上傳模型

### 2. 獨立遊戲 (鍵盤控制)
```bash
python3 game.py
```
- 純鍵盤控制
- 不需要模型

### 3. 命令列 AI 模式
```bash
# Terminal 1
python3 game.py

# Terminal 2
python3 controller_tflite.py --camera
```
- 進階使用
- 可自訂參數

## 🔐 安全和隱私

### Git 忽略規則
```gitignore
# .gitignore
uploads/*.tflite      # 不提交使用者模型
uploads/*.txt         # 不提交使用者標籤
!uploads/.gitkeep     # 但保留目錄結構
```

### 檔案上傳限制
- 檔案大小: 50MB
- 允許類型: `.tflite`, `.txt`
- 檔名清理: `secure_filename()`

## 📊 技術堆疊

| 層級 | 技術 | 版本 |
|------|------|------|
| 後端框架 | Flask | 3.1+ |
| 前端 | HTML5 + CSS3 + JavaScript | ES6+ |
| 遊戲引擎 | Pygame | 2.6+ |
| AI 推論 | TensorFlow Lite | 2.20+ |
| 影像處理 | OpenCV | 4.0+ |
| HTTP 伺服器 | Werkzeug | (Flask 內建) |

## 🎨 UI 設計規範

### 配色方案 (深藍色主題)
```css
--bg-dark: #0A1929      /* 主背景 */
--bg-panel: #132F4C     /* 面板背景 */
--border: #1E3A5F       /* 邊框 */
--accent: #66B2FF       /* 強調色 */
--text-primary: #ffffff /* 主要文字 */
--text-secondary: #8899AA /* 次要文字 */
```

### 元件結構
```
Container
├── Header
├── Upload Section
│   └── Upload Grid (2 columns)
└── Game Container (2 columns)
    ├── Game Panel
    └── Camera Panel
```

## 🚀 效能考量

- **攝影機 FPS:** ~30 fps
- **模型推論:** 每 120ms (8.3 fps)
- **遊戲更新:** 每 120ms
- **預測輪詢:** 每 120ms
- **執行緒:** 2 (主執行緒 + 攝影機執行緒)

## 📝 維護建議

1. **定期更新依賴:**
   ```bash
   pip list --outdated
   pip install --upgrade -r requirements.txt
   ```

2. **清理上傳檔案:**
   ```bash
   rm uploads/*.tflite uploads/*.txt
   ```

3. **重建虛擬環境:**
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

**版本:** 2.0  
**最後更新:** 2025年10月5日  
**作者:** AI Snake Game Team
