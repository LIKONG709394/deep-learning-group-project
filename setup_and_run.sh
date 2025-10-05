#!/bin/bash

# 🚀 AI Snake Kid - 完整啟動指南
# 這個腳本會自動設定並啟動程式

echo "════════════════════════════════════════════════════════════"
echo "🐍 AI Snake Kid - 自動設定和啟動"
echo "════════════════════════════════════════════════════════════"
echo ""

# 進入專案目錄
cd /Users/yusingkiu/Desktop/aisnakekid

# 步驟 1: 檢查 Python
echo "📋 步驟 1/5: 檢查 Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 未安裝!"
    echo "請先安裝 Python 3: https://www.python.org/downloads/"
    exit 1
fi
echo "✅ Python 3 已安裝: $(python3 --version)"
echo ""

# 步驟 2: 建立虛擬環境
echo "📋 步驟 2/5: 設定虛擬環境..."
if [ ! -d "venv" ]; then
    echo "建立虛擬環境中..."
    python3 -m venv venv
    echo "✅ 虛擬環境已建立"
else
    echo "✅ 虛擬環境已存在"
fi
echo ""

# 步驟 3: 啟動虛擬環境
echo "📋 步驟 3/5: 啟動虛擬環境..."
source venv/bin/activate
echo "✅ 虛擬環境已啟動"
echo ""

# 步驟 4: 安裝依賴
echo "📋 步驟 4/5: 安裝依賴套件..."
echo "這可能需要幾分鐘時間..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✅ 依賴套件已安裝"
echo ""

# 步驟 5: 執行環境檢查
echo "📋 步驟 5/5: 執行環境檢查..."
python3 check_setup.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 環境檢查失敗!"
    echo "請檢查上面的錯誤訊息"
    exit 1
fi

# 啟動程式
echo ""
echo "════════════════════════════════════════════════════════════"
echo "🚀 啟動 AI Snake Kid..."
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📱 請開啟瀏覽器並前往:"
echo ""
echo "   👉 http://127.0.0.1:8080"
echo ""
echo "⌨️  按 Ctrl+C 停止伺服器"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

python3 app.py
