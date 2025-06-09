#!/usr/bin/env python3
"""
滑板車偵測專案狀態檢查工具
檢查專案檔案完整性和環境配置
"""

import os
import sys
from pathlib import Path

def check_project_status():
    """檢查專案狀態"""
    print("🛴 滑板車偵測專案狀態檢查")
    print("=" * 50)
    
    # 檢查關鍵檔案
    critical_files = {
        "requirements.txt": "依賴套件清單",
        "README.md": "專案說明文件",
        "GUIDE.md": "使用指南",
        "models/scooter_yolov8n4/weights/best.pt": "最佳模型檔案",
        "src/detection/camera_detection_en.py": "攝影機偵測腳本",
        "src/detection/detect_scooter_en.py": "單張圖片偵測腳本",
        "src/detection/batch_detect_en.py": "批次偵測腳本",
        "camera_detection.bat": "攝影機偵測快速啟動",
        "scooter.bat": "專案工具快速啟動"
    }
    
    print("\n📁 關鍵檔案檢查:")
    missing_files = []
    for file_path, description in critical_files.items():
        if Path(file_path).exists():
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} (缺失)")
            missing_files.append(file_path)
    
    # 檢查目錄結構
    directories = {
        "data/images/positive": "正樣本圖片",
        "data/images/negative": "負樣本圖片",
        "data/annotations": "標註檔案",
        "data/yolo_dataset": "YOLO 資料集",
        "models": "模型檔案",
        "results": "偵測結果",
        "src/detection": "偵測模組",
        "src/training": "訓練模組",
        "notebooks": "教學筆記本"
    }
    
    print("\n📂 目錄結構檢查:")
    for dir_path, description in directories.items():
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                count = len(list(path.glob("*")))
                print(f"✅ {description}: {dir_path} ({count} 個項目)")
            else:
                print(f"⚠️  {description}: {dir_path} (不是目錄)")
        else:
            print(f"❌ {description}: {dir_path} (不存在)")
    
    # 檢查 Python 環境
    print("\n🐍 Python 環境檢查:")
    print(f"Python 版本: {sys.version}")
    
    # 檢查關鍵套件
    packages = ["torch", "ultralytics", "cv2", "matplotlib", "numpy"]
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安裝")
        except ImportError:
            print(f"❌ {package} 未安裝")
    
    # 總結
    print("\n📊 專案狀態總結:")
    if missing_files:
        print(f"⚠️  發現 {len(missing_files)} 個缺失檔案")
        print("建議重新下載或恢復缺失的檔案")
    else:
        print("✅ 所有關鍵檔案完整")
    
    print("\n🚀 快速開始:")
    print("1. 攝影機偵測: camera_detection.bat")
    print("2. 查看說明: 閱讀 README.md")
    print("3. 詳細指南: 閱讀 GUIDE.md")
    print("4. 完整教學: notebooks/滑板車偵測完整流程.ipynb")

if __name__ == "__main__":
    check_project_status()
