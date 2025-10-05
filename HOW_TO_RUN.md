# 🚀 如何運行程式 - 快速指南

## 方法 1: 使用快速啟動腳本 (最簡單) ⭐

```bash
cd /Users/yusingkiu/Desktop/edupython
./start.sh
```

這個腳本會自動:
- ✅ 檢查虛擬環境
- ✅ 安裝依賴套件
- ✅ 驗證環境設定
- ✅ 啟動 Flask 伺服器

---

## 方法 2: 手動啟動 (逐步執行)

### 步驟 1: 進入專案目錄
```bash
cd /Users/yusingkiu/Desktop/edupython
```

### 步驟 2: 啟動虛擬環境 (如果有的話)
```bash
source venv/bin/activate
```

### 步驟 3: 運行主程式
```bash
python3 app.py
```

### 步驟 4: 開啟瀏覽器
在瀏覽器中前往:
```
http://127.0.0.1:8080
```

---

## 方法 3: 先檢查環境再運行

### 檢查環境是否正確設定
```bash
cd /Users/yusingkiu/Desktop/edupython
python3 check_setup.py
```

### 如果檢查通過,運行程式
```bash
python3 app.py
```

---

## 🎮 運行後你會看到:

### 終端機輸出:
```
============================================================
AI 控制貪食蛇 - Flask 網頁版
============================================================

✓ 模型載入成功!類別數:4
✓ 攝影機 0 已開啟

伺服器啟動中...
請開啟瀏覽器並前往:http://127.0.0.1:8080
按 Ctrl+C 停止伺服器
============================================================

 * Serving Flask app 'app'
 * Running on http://0.0.0.0:8080
```

### 瀏覽器畫面:
- 🎨 深藍色和白色主題介面
- 🎮 左側: 貪食蛇遊戲
- 📹 右側: 攝影機畫面和預測結果
- 📤 頂部: 模型上傳區域

---

## ⌨️ 如何停止程式

在運行 `python3 app.py` 的終端機中按:
```
Ctrl + C
```

---

## 🎯 完整的使用流程

### 1️⃣ 啟動程式
```bash
cd /Users/yusingkiu/Desktop/edupython
python3 app.py
```

### 2️⃣ 開啟瀏覽器
```
http://127.0.0.1:8080
```

### 3️⃣ 上傳你的模型 (可選)
- 點擊 "Upload Your Model" 區域
- 選擇你的 `.tflite` 檔案
- 選擇你的 `labels.txt` 檔案
- 點擊 "Upload and Load Model"

### 4️⃣ 啟用 AI 控制
- 點擊 "🤖 Enable AI" 按鈕
- 攝影機會開始工作
- 對著鏡頭做手勢 (up, down, left, right)

### 5️⃣ 開始玩!
- 手勢控制蛇的移動
- 或使用鍵盤 (方向鍵/WASD)
- 吃食物得分

---

## 🐛 常見問題

### 問題 1: `python3: command not found`
**解決方案:**
```bash
python app.py  # 試試看用 python 而不是 python3
```

### 問題 2: `ModuleNotFoundError: No module named 'flask'`
**解決方案:**
```bash
pip install -r requirements.txt
# 或
pip3 install -r requirements.txt
```

### 問題 3: `Address already in use`
**解決方案:**
端口 8080 已被佔用,殺掉舊進程:
```bash
lsof -ti:8080 | xargs kill -9
```
然後重新運行程式。

### 問題 4: 攝影機無法開啟
**解決方案:**
1. 檢查系統隱私設定 (允許 Terminal 使用攝影機)
2. 關閉其他使用攝影機的程式
3. 重新啟動應用程式

---

## 📋 系統需求

- ✅ Python 3.10 或更高版本
- ✅ macOS / Linux / Windows
- ✅ 攝影機 (內建或外接)
- ✅ 現代瀏覽器 (Chrome, Firefox, Safari, Edge)
- ✅ 網路連線 (訓練模型時需要)

---

## 🎓 訓練你自己的模型

### 步驟:
1. 前往 https://teachablemachine.withgoogle.com/
2. 選擇 "Image Project"
3. 建立 4 個類別:
   - `0 up` - 向上手勢
   - `1 left` - 向左手勢
   - `2 right` - 向右手勢
   - `3 down` - 向下手勢
4. 每個類別拍 50-100 張照片
5. 訓練模型
6. 匯出 "TensorFlow Lite" (Floating point)
7. 下載模型和標籤檔案
8. 在網頁上上傳!

---

## 🎉 就這麼簡單!

現在你可以:
1. ✅ 運行程式: `python3 app.py`
2. ✅ 開啟瀏覽器: `http://127.0.0.1:8080`
3. ✅ 上傳模型
4. ✅ 開始玩!

**享受你的 AI 貪食蛇遊戲! 🐍🎮🤖**
