#!/usr/bin/env python3
"""
檢查專案檔案和相依性
"""
import os
import sys

def check_file(path, description):
    """檢查檔案是否存在"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_import(module_name):
    """檢查 Python 套件是否已安裝"""
    try:
        __import__(module_name)
        print(f"✓ {module_name} is installed")
        return True
    except ImportError:
        print(f"✗ {module_name} is NOT installed")
        return False

def main():
    print("=" * 60)
    print("AI Snake Kid - Environment Check")
    print("=" * 60)
    print()
    
    all_good = True
    
    # 檢查主要檔案
    print("📄 Checking project files...")
    files = [
        ("app.py", "Flask web server"),
        ("game.py", "Snake game"),
        ("controller_tflite.py", "TFLite controller"),
        ("controls.py", "Control module"),
        ("templates/index.html", "Web UI template"),
        ("static/game.js", "Frontend JavaScript"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
    ]
    
    for path, desc in files:
        if not check_file(path, desc):
            all_good = False
    
    print()
    
    # 檢查模型檔案 (可選)
    print("🤖 Checking model files (optional)...")
    check_file("model_unquant.tflite", "Default TFLite model")
    check_file("labels.txt", "Default labels file")
    print("  (These are optional - you can upload your own via web UI)")
    print()
    
    # 檢查 Python 套件
    print("📦 Checking Python packages...")
    packages = [
        "flask",
        "cv2",
        "numpy",
        "PIL",
        "tensorflow",
        "pygame"
    ]
    
    for pkg in packages:
        if not check_import(pkg):
            all_good = False
    
    print()
    print("=" * 60)
    
    if all_good:
        print("✅ All checks passed! You're ready to go!")
        print()
        print("To start the web app:")
        print("  python3 app.py")
        print()
        print("Then open: http://127.0.0.1:8080")
    else:
        print("❌ Some checks failed. Please:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. Make sure all files are in place")
        print("  3. Run this check again")
    
    print("=" * 60)
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
