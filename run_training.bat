@echo off
echo ========================================================
echo Installing Ultralytics & PyTorch dependencies...
echo ========================================================
pip install ultralytics
echo ========================================================
echo Launching GPU Training...
echo ========================================================
python train_yolo.py
pause
