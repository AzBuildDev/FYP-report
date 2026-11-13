@echo off
chcp 65001 >nul
setlocal

call "%~dp0..\环境配置\配置本地环境.bat"
if errorlevel 1 exit /b 1

set "HF_HOME=%PROJECT_ROOT%\运行环境\缓存\huggingface"
set "TORCH_HOME=%PROJECT_ROOT%\运行环境\缓存\torch"
set "HF_LEROBOT_CALIBRATION=%PROJECT_ROOT%\标定备份"
set "TMP=%PROJECT_ROOT%\运行环境\临时文件"
set "TEMP=%PROJECT_ROOT%\运行环境\临时文件"
set "PYTHONPATH=%LEROBOT_ROOT%\src;%PYTHONPATH%"

cd /d "%PROJECT_ROOT%"

python "%PROJECT_ROOT%\脚本\历史工具\旧方案视觉对齐.py" ^
  --mode=calibrate ^
  --model="%PROJECT_ROOT%\旧视觉方案\模型\overhead_yolo_seg.pt" ^
  --config="%PROJECT_ROOT%\旧视觉方案\配置\桌面相机安全平面标定.json" ^
  --target-class=red_cube ^
  --camera-index=%OVERHEAD_CAMERA% ^
  --follower-port=%FOLLOWER_PORT% ^
  --leader-port=%LEADER_PORT% ^
  %*

if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

endlocal
