# 滑板車偵測系統

這是一個基於深度學習的滑板車物件偵測專案，使用 Python 和 YOLO 模型來識別圖片中的滑板車。

## 專案概述

本專案旨在開發一個能夠自動偵測圖片中滑板車的智慧系統，適用於元智大學校園環境。

## 功能特色

- 🛴 滑板車物件偵測
- 📸 支援多種圖片格式
- 🏷️ 圖片標註工具
- 🤖 YOLO 模型訓練
- 📊 偵測結果評估

## 專案結構

```
滑板車偵測系統/
├── data/                    # 資料集目錄
│   ├── images/             # 原始圖片
│   │   ├── positive/       # 正樣本 (包含滑板車)
│   │   └── negative/       # 負樣本 (不包含滑板車)
│   ├── labels/             # 標註文件 (YOLO 格式)
│   └── annotations/        # 原始標註文件
├── models/                  # 訓練好的模型
├── src/                    # 源代碼
│   ├── data_preprocessing/ # 資料預處理
│   ├── annotation_tool/    # 標註工具
│   ├── training/          # 模型訓練
│   └── detection/         # 偵測推理
├── notebooks/              # Jupyter 筆記本
├── results/               # 結果輸出
└── requirements.txt       # 依賴套件
```

## 安裝與設定

1. 建立虛擬環境：
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. 安裝依賴套件：
```bash
pip install -r requirements.txt
```

3. 準備資料集：
   - 將包含滑板車的圖片放入 `data/images/positive/`
   - 將不包含滑板車的圖片放入 `data/images/negative/`

## 使用方法

### 1. 資料標註
```bash
python src/annotation_tool/annotate.py
```

### 2. 資料預處理
```bash
python src/data_preprocessing/prepare_dataset.py
```

### 3. 模型訓練
```bash
python src/training/train_yolo.py
```

### 4. 滑板車偵測

#### 單張圖片偵測
```bash
python src/detection/detect_scooter.py --model models/scooter_yolov8n4/weights/best.pt --image path/to/image.jpg
```

#### 即時攝影機偵測
```bash
python src/detection/camera_detection.py --model models/scooter_yolov8n4/weights/best.pt
```

或直接執行批次檔：
```bash
camera_detection.bat
```

#### 攝影機偵測控制說明
- **按 'q'**: 退出程式
- **按 'r'**: 開始/停止錄製影片
- **按 '+'**: 增加信心度閾值
- **按 '-'**: 降低信心度閾值  
- **按 's'**: 截圖儲存

## 模型資訊

- **模型架構**: YOLOv8
- **輸入尺寸**: 640x640
- **類別數量**: 1 (滑板車)
- **標註格式**: YOLO txt 格式

## 評估指標

- **mAP (mean Average Precision)**: 模型整體性能
- **Precision**: 精確率
- **Recall**: 召回率
- **F1-Score**: F1 分數

## 範例結果

模型能夠準確識別各種角度和環境下的滑板車，包括：
- 校園內停放的滑板車
- 行駛中的滑板車
- 不同光照條件下的滑板車

## 技術棧

- **Python 3.8+**
- **PyTorch**: 深度學習框架
- **Ultralytics**: YOLO 實現
- **OpenCV**: 圖像處理
- **PIL**: 圖像操作
- **Matplotlib**: 結果可視化
- **LabelImg**: 標註工具

## 貢獻

歡迎提交問題和改進建議！

## 授權

本專案僅供學術用途。
