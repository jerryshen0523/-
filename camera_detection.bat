@echo off
echo Starting real-time scooter detection...
echo.
echo Please make sure your camera is connected and working properly
echo.
pause

cd /d r:\test
set KMP_DUPLICATE_LIB_OK=TRUE

python src/detection/camera_detection_en.py --model models/scooter_yolov8n4/weights/best.pt --conf 0.25

pause
