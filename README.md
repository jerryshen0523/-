# 滑板車偵測系統

這是一個基於深度學習的滑板車物件偵測專案，使用 Python 和 YOLOv8 模型來識別圖片中的滑板車。

## 專案概述

本專案旨在開發一個能夠自動偵測圖片中滑板車的智慧系統，適用於元智大學校園環境。

## 功能特色

- 🛴 高精度滑板車物件偵測 (mAP: 83.6%, Precision: 89.5%)
- 📸 支援多種圖片格式 (JPG, PNG, BMP, TIFF)
- 🎥 即時攝影機偵測功能
- 📊 批次圖片處理與結果分析
- 🏷️ 內建圖片標註工具
- 🤖 YOLOv8 模型訓練流程

## 專案結構

```
滑板車偵測系統/
├── data/                    # 資料集目錄
│   ├── images/             # 原始圖片
│   │   ├── positive/       # 正樣本 (包含滑板車)
│   │   └── negative/       # 負樣本 (不包含滑板車)
│   ├── annotations/        # 原始標註文件 (JSON 格式)
│   ├── labels/             # 標註文件 (YOLO 格式)
│   └── yolo_dataset/       # 預處理後的 YOLO 資料集
├── models/                  # 訓練好的模型
│   ├── scooter_yolov8n4/   # 最佳模型 (mAP: 83.6%)
│   └── scooter_yolov8n_final.pt  # 最終模型檔案
├── src/                    # 源代碼
│   ├── data_preprocessing/ # 資料預處理
│   ├── annotation_tool/    # 標註工具
│   ├── training/          # 模型訓練
│   ├── detection/         # 偵測推理
│   └── utils/             # 工具函數
├── notebooks/              # Jupyter 筆記本教學
├── results/               # 偵測結果輸出
├── camera_detection.bat   # 攝影機偵測快速啟動
├── scooter.bat           # 專案工具快速啟動
├── requirements.txt      # 依賴套件
└── yolov8n.pt           # 預訓練模型
```

## 安裝與設定

### 環境需求
- Python 3.8+
- CUDA 11.0+ (可選，用於 GPU 加速)
- 4GB+ RAM
- 攝影機設備 (用於即時偵測)

### 安裝步驟

1. **建立虛擬環境**：
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. **安裝依賴套件**：
```bash
pip install -r requirements.txt
```

3. **驗證安裝**：
```bash
python -c "import torch; print(torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLOv8 ready')"
```

## 使用方法

### 🚀 快速開始

#### 即時攝影機偵測 (推薦)
最簡單的方式是直接執行批次檔：
```bash
camera_detection.bat
```

或手動執行：
```bash
python src/detection/camera_detection_en.py --model models/scooter_yolov8n4/weights/best.pt --conf 0.25
```

#### 單張圖片偵測
```bash
python src/detection/detect_scooter_en.py --model models/scooter_yolov8n4/weights/best.pt --image path/to/image.jpg
```

#### 批次圖片偵測
```bash
python src/detection/batch_detect_en.py --model models/scooter_yolov8n4/weights/best.pt --input-dir path/to/images --output-dir results/batch_detection
```

### 🎮 攝影機偵測控制說明
- **按 'q'**: 退出程式
- **按 'r'**: 開始/停止錄製影片
- **按 '+'**: 增加信心度閾值 (+0.05)
- **按 '-'**: 降低信心度閾值 (-0.05)
- **按 's'**: 截圖儲存

### 📊 進階功能

#### 資料標註 (用於新資料)
```bash
python src/annotation_tool/annotate.py
```

#### 資料預處理
```bash
python src/data_preprocessing/prepare_dataset.py
```

#### 模型訓練 (用於改進模型)
```bash
python src/training/train_yolo.py --data data/yolo_dataset/data.yaml --epochs 100
```

## 🏆 模型性能

### 訓練結果
- **模型架構**: YOLOv8n (輕量級版本)
- **輸入尺寸**: 640×640 像素
- **訓練資料**: 200+ 標註樣本
- **訓練輪數**: 50+ epochs

### 評估指標
- **mAP50**: 83.6% (IoU=0.5 時的平均精度)
- **Precision**: 89.5% (精確率)
- **Recall**: 78.2% (召回率)
- **F1-Score**: 83.5%

### 檢測能力
✅ 校園內停放的滑板車  
✅ 行駛中的滑板車  
✅ 不同光照條件下的滑板車  
✅ 多角度、多距離的滑板車  
✅ 部分遮擋的滑板車  

## 📁 輸出結果

### 批次偵測結果
- **圖片標註**: `results/batch_detection/detected_*.jpg`
- **偵測報告**: `results/batch_detection/batch_detection_results.json`
- **統計圖表**: `results/batch_detection/batch_detection_summary.png`

### 即時偵測結果
- **錄製影片**: `results/recording_YYYYMMDD_HHMMSS.mp4`
- **截圖**: `results/screenshot_YYYYMMDD_HHMMSS.jpg`

## 🛠️ 技術棧

- **Python 3.8+**: 主要開發語言
- **PyTorch**: 深度學習框架
- **Ultralytics YOLOv8**: 物件偵測模型
- **OpenCV**: 圖像處理與攝影機介面
- **Matplotlib**: 結果可視化
- **JSON**: 標註資料格式

## 📋 常見問題

**Q: 攝影機無法開啟**  
A: 檢查攝影機連接，確認其他程式未佔用攝影機

**Q: 偵測結果不準確**  
A: 調整信心度閾值 (按 +/- 鍵)，或在光線充足環境下使用

**Q: 程式運行緩慢**  
A: 確認是否使用 GPU 加速，或降低輸入解析度

**Q: 缺少模型檔案**  
A: 確認 `models/scooter_yolov8n4/weights/best.pt` 檔案存在

## 📚 學習資源

- **完整教學**: 查看 `notebooks/滑板車偵測完整流程.ipynb`
- **使用指南**: 閱讀 `GUIDE.md`
- **範例結果**: 參考 `results/` 目錄下的輸出
- **專案狀態**: 運行 `python check_project.py` 檢查系統狀態
- **清理報告**: 查看 `PROJECT_CLEANUP_REPORT.md` 了解專案清理詳情


## 📄 授權

本專案僅供學術研究用途。
