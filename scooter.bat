@echo off
echo 滑板車偵測專案啟動器
echo ========================

if "%1"=="" goto help

if "%1"=="install" goto install
if "%1"=="annotate" goto annotate
if "%1"=="preprocess" goto preprocess
if "%1"=="train" goto train
if "%1"=="detect" goto detect
if "%1"=="camera" goto camera
if "%1"=="batch" goto batch
if "%1"=="notebook" goto notebook
goto help

:install
echo 安裝依賴套件...
pip install -r requirements.txt
goto end

:annotate
echo 啟動標註工具...
python src/annotation_tool/annotate.py
goto end

:preprocess
echo 執行資料預處理...
python src/data_preprocessing/prepare_dataset.py
goto end

:train
echo 開始訓練模型...
python src/training/train_yolo.py --data data/yolo_dataset/data.yaml --epochs 50 %2 %3 %4
goto end

:detect
echo 執行單張圖片偵測...
python src/detection/detect_scooter_en.py --model models/scooter_yolov8n4/weights/best.pt %2 %3 %4 %5
goto end

:camera
echo 啟動攝影機即時偵測...
python src/detection/camera_detection_en.py --model models/scooter_yolov8n4/weights/best.pt %2 %3 %4
goto end

:batch
echo 執行批次偵測...
python src/detection/batch_detect_en.py --model models/scooter_yolov8n4/weights/best.pt %2 %3 %4 %5
goto end

:notebook
echo 開啟 Jupyter 筆記本...
jupyter notebook notebooks/滑板車偵測完整流程.ipynb
goto end

:help
echo 使用方法:
echo   %0 install        - 安裝依賴套件
echo   %0 annotate       - 啟動標註工具
echo   %0 preprocess     - 執行資料預處理
echo   %0 train          - 訓練模型
echo   %0 detect         - 執行單張圖片偵測
echo   %0 camera         - 攝影機即時偵測
echo   %0 batch          - 批次圖片偵測
echo   %0 notebook       - 開啟 Jupyter 筆記本
echo.
echo 範例:
echo   %0 install
echo   %0 camera --conf 0.3
echo   %0 detect --image test.jpg
echo   %0 batch --input-dir images --output-dir results
goto end

:end
