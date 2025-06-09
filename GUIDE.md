# 滑板車偵測系統使用指南

## 🚀 快速開始

本指南將幫助您快速上手滑板車偵測系統。系統已完成訓練，可直接用於偵測任務。

### 1. 環境設定

```bash
# 建立虛擬環境 (推薦)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安裝依賴套件
pip install -r requirements.txt
```

### 2. 驗證安裝

```bash
# 檢查 PyTorch 安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 檢查 YOLOv8 安裝
python -c "from ultralytics import YOLO; print('YOLOv8 ready')"

# 檢查 OpenCV 安裝
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### 3. 立即開始偵測

#### 方法一：使用批次檔 (最簡單)
```bash
# 攝影機即時偵測
camera_detection.bat

# 其他功能
scooter.bat
```

#### 方法二：直接執行 Python 腳本
```bash
# 攝影機即時偵測
python src/detection/camera_detection_en.py --model models/scooter_yolov8n4/weights/best.pt --conf 0.25
```

## 🎯 主要功能使用

### 即時攝影機偵測

這是最常用的功能，可以即時偵測攝影機畫面中的滑板車。

```bash
# 基本用法
python src/detection/camera_detection_en.py --model models/scooter_yolov8n4/weights/best.pt

# 自訂信心度閾值
python src/detection/camera_detection_en.py --model models/scooter_yolov8n4/weights/best.pt --conf 0.3

# 自動錄影
python src/detection/camera_detection_en.py --model models/scooter_yolov8n4/weights/best.pt --save-video
```

**操作提示：**
- 按 `q` 退出程式
- 按 `r` 開始/停止錄影
- 按 `+` 增加信心度閾值
- 按 `-` 降低信心度閾值
- 按 `s` 拍攝截圖

### 單張圖片偵測

適用於測試特定圖片的偵測效果。

```bash
# 偵測單張圖片
python src/detection/detect_scooter_en.py --model models/scooter_yolov8n4/weights/best.pt --image path/to/image.jpg --conf 0.25
```

結果會儲存在 `results/` 目錄中。

### 批次圖片偵測

適用於處理大量圖片並生成統計報告。

```bash
# 批次偵測
python src/detection/batch_detect_en.py --model models/scooter_yolov8n4/weights/best.pt --input-dir path/to/images --output-dir results/batch_detection --conf 0.1
```

輸出包括：
- 標註後的圖片
- JSON 格式的偵測結果
- 統計圖表和摘要報告

## 🔧 進階設定

### 調整偵測參數

```bash
# 降低信心度閾值 (偵測更多但可能有誤報)
--conf 0.1

# 提高信心度閾值 (偵測更準確但可能遺漏)
--conf 0.5

# 指定攝影機 ID
--camera 0  # 預設攝影機
--camera 1  # 第二個攝影機
```

### 使用不同模型

```bash
# 使用最終模型檔案
--model models/scooter_yolov8n_final.pt

# 使用最佳模型 (推薦)
--model models/scooter_yolov8n4/weights/best.pt
```

## 🏗️ 開發與訓練 (進階用戶)

如果您需要添加新的訓練資料或改進模型：

### 1. 準備新資料

將圖片按以下結構放置：

```
data/
├── images/
│   ├── positive/     # 包含滑板車的圖片
│   └── negative/     # 不包含滑板車的圖片
```

### 2. 標註新圖片

```bash
python src/annotation_tool/annotate.py
```

使用內建標註工具為新圖片標記滑板車位置。

### 3. 資料預處理

```bash
python src/data_preprocessing/prepare_dataset.py
```

將標註資料轉換為 YOLO 格式。

### 4. 重新訓練模型

```bash
python src/training/train_yolo.py --data data/yolo_dataset/data.yaml --epochs 100 --batch 16
```

## 📊 結果解讀

### 信心度分數

- **0.0-0.3**: 低信心度，可能為誤報
- **0.3-0.7**: 中等信心度，需要人工確認
- **0.7-1.0**: 高信心度，很可能是真實的滑板車

### 偵測框顏色

- **紅色框**: 偵測到的滑板車
- **數字標籤**: 信心度分數 (0-1)

## 🚨 故障排除

### 常見問題及解決方案

**攝影機無法開啟**
```bash
# 檢查可用攝影機
python -c "import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(3)]"

# 嘗試不同攝影機 ID
python src/detection/camera_detection_en.py --camera 1
```

**模型檔案不存在**
```bash
# 檢查模型檔案
ls models/scooter_yolov8n4/weights/best.pt

# 使用備用模型
python src/detection/camera_detection_en.py --model models/scooter_yolov8n_final.pt
```

**記憶體不足**
```bash
# 降低批次大小 (僅限訓練時)
python src/training/train_yolo.py --batch 8

# 關閉其他程式釋放記憶體
```

**偵測效果不理想**
- 確保光線充足
- 調整攝影機角度和距離
- 嘗試不同的信心度閾值
- 增加訓練資料重新訓練

## 📖 深入學習

### 完整教學

詳細的步驟說明和原理解釋請參考：
```
notebooks/滑板車偵測完整流程.ipynb
```

### 技術文件

- **YOLOv8 官方文檔**: https://docs.ultralytics.com/
- **OpenCV 教學**: https://opencv.org/
- **PyTorch 指南**: https://pytorch.org/tutorials/

### 專案架構

理解專案各模組的功能：
- `src/annotation_tool/`: 圖片標註工具
- `src/data_preprocessing/`: 資料預處理模組
- `src/training/`: 模型訓練模組
- `src/detection/`: 偵測推理模組
- `src/utils/`: 工具函數

## 💡 最佳實踐

1. **使用虛擬環境**避免套件衝突
2. **定期備份模型**避免意外遺失
3. **記錄實驗參數**便於重現結果
4. **驗證偵測結果**確保準確性
5. **適當調整閾值**平衡精度和召回率
