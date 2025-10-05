#!/bin/bash

# 快速啟動 AI Snake Kid

echo "🐍 AI Snake Kid - Quick Start"
echo "=============================="
echo ""

# 檢查是否在正確的目錄
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found."
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# 啟動虛擬環境
echo "📦 Activating virtual environment..."
source venv/bin/activate

# 檢查依賴
echo "🔍 Checking dependencies..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not installed."
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

# 執行環境檢查
echo ""
echo "🔍 Running environment check..."
python3 check_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Environment check failed!"
    echo "Please fix the issues above and try again."
    exit 1
fi

# 啟動應用程式
echo ""
echo "🚀 Starting Flask server..."
echo "=============================="
echo ""
echo "The app will be available at:"
echo "👉 http://127.0.0.1:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
