# 🚀 快速測試指南

## 啟動網頁應用程式

1. **啟動 Flask 伺服器:**
   ```bash
   cd /Users/yusingkiu/Desktop/edupython
   source venv/bin/activate  # 如果你使用虛擬環境
   python3 app.py
   ```

2. **開啟瀏覽器:**
   - 前往: http://127.0.0.1:8080
   - 你應該會看到深藍色和白色的介面

3. **測試功能:**
   - ✅ 左側應該顯示貪食蛇遊戲
   - ✅ 右側應該顯示攝影機畫面
   - ✅ 可以看到即時的預測條形圖
   - ✅ 上方有上傳模型的區域

## 上傳自己的模型

1. **訓練模型 (Teachable Machine):**
   - 前往: https://teachablemachine.withgoogle.com/
   - 選擇 "Image Project"
   - 建立 4 個類別:
     - `0 up` - 向上的手勢
     - `1 left` - 向左的手勢
     - `2 right` - 向右的手勢
     - `3 down` - 向下的手勢
   - 每個類別錄製 50-100 張圖片
   - 訓練完成後選擇 "Export Model"
   - 選擇 "TensorFlow Lite" → "Floating point"
   - 下載模型和標籤檔

2. **上傳到網頁:**
   - 在網頁上找到 "📤 Upload Your Model" 區域
   - 點擊 "TensorFlow Lite Model" 選擇你的 `.tflite` 檔案
   - 點擊 "Labels File" 選擇你的 `labels.txt` 檔案
   - 點擊 "🚀 Upload and Load Model" 按鈕
   - 等待幾秒鐘,頁面會自動重新載入

3. **開始玩!**
   - 點擊 "🤖 Enable AI" 按鈕啟用 AI 控制
   - 對著鏡頭做手勢來控制貪食蛇
   - 你應該會看到預測條形圖即時更新

## 鍵盤控制

如果不想用 AI,也可以用鍵盤控制:
- 方向鍵 (↑ ↓ ← →) 或 WASD
- 按 R 重新開始遊戲

## 疑難排解

### 模型上傳失敗
- 確認檔案格式正確 (.tflite 和 .txt)
- 檢查檔案大小不超過 50MB
- 查看瀏覽器控制台的錯誤訊息

### 攝影機無法開啟
- 檢查瀏覽器權限 (允許使用攝影機)
- 確認其他應用程式沒有佔用攝影機
- 嘗試重新整理頁面

### 預測不準確
- 重新訓練模型,增加更多樣本
- 確保訓練時的光線條件與使用時相似
- 調整信心度閾值 (在 `app.py` 中的 `CONF_THRESHOLD`)

## 檔案位置

- 上傳的模型會儲存在: `uploads/uploaded_model.tflite`
- 上傳的標籤會儲存在: `uploads/uploaded_labels.txt`
- 這些檔案會在上傳新模型時被覆蓋
