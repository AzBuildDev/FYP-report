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
  --mode=align ^
  --model="%PROJECT_ROOT%\旧视觉方案\模型\overhead_yolo_seg.pt" ^
  --gripper-model="%PROJECT_ROOT%\旧视觉方案\模型\gripper_yolo_seg.pt" ^
  --config="%PROJECT_ROOT%\旧视觉方案\配置\桌面相机安全平面标定.json" ^
  --target-class=red_cube ^
  --camera-index=%OVERHEAD_CAMERA% ^
  --gripper-camera-index=%GRIPPER_CAMERA% ^
  --follower-port=%FOLLOWER_PORT% ^
  --leader-port=%LEADER_PORT% ^
  --enable-manual-correction ^
  --manual-teleop-fps=60 ^
  --enable-gripper-final-align ^
  --alignment-pose-mode=full_pose ^
  --gripper-servo-mode=verify_only ^
  --no-auto-learn-overhead-after-gripper ^
  --gripper-ibvs-fresh-start ^
  --gripper-ibvs-probe-deg=1.2 ^
  --gripper-debug-ibvs-probe-deg=1.2 ^
  --gripper-debug-ibvs-max-step-deg=0.5 ^
  --gripper-debug-ibvs-min-step-deg=0.2 ^
  --gripper-debug-ibvs-restore-threshold-deg=0.8 ^
  --gripper-debug-ibvs-restore-timeout-s=8.0 ^
  --gripper-debug-ibvs-restore-resend-interval-s=0.05 ^
  --gripper-debug-ibvs-min-actual-controlled-deg=0.25 ^
  --gripper-debug-ibvs-min-pixel-motion-px=3.0 ^
  --gripper-debug-ibvs-max-jitter-px=6.0 ^
  --gripper-debug-ibvs-uncontrolled-warn-deg=0.5 ^
  --gripper-debug-ibvs-max-uncontrolled-drift-deg=1.0 ^
  --gripper-reference-mode=recorded_robust ^
  --gripper-reference-min-count=1 ^
  --gripper-reference-min-radius-px=16 ^
  --gripper-center-tolerance-px=25 ^
  --joint-error-threshold=0.8 ^
  --safe-position-threshold=5.0 ^
  --safe-lift-position-threshold=5.0 ^
  --full-safe-position-threshold=5.0 ^
  --safe-lift-continue-threshold=8.0 ^
  --full-safe-continue-threshold=8.0 ^
  --safe-wait-timeout-s=4.0 ^
  --safe-progress-extra-timeout-s=2.0 ^
  --safe-retry-count=0 ^
  --local-snap-px=2.0 ^
  --gripper-replay-snap-px=2.0 ^
  --closed-loop-iters=3 ^
  --max-joint-step=1.5 ^
  --joint-speed-deg-s=18 ^
  --gripper-refine-joint-speed-deg-s=35 ^
  --xy-hover-wait-timeout-s=1.0 ^
  --gripper-refine-wait-timeout-s=1.0 ^
  --gripper-refine-stable-epsilon-deg=0.06 ^
  --gripper-refine-stable-samples=8 ^
  --gripper-max-step-deg=2.3 ^
  --gripper-min-step-deg=0.75 ^
  --gripper-correction-max-deg-per-px=0.35 ^
  --gripper-motion-local-joint-radius-deg=5.0 ^
  --gripper-motion-local-overhead-radius-px=70 ^
  --gripper-refine-continue-threshold=0.8 ^
  --no-gripper-mask-area-guard ^
  --no-gripper-mask-shape-guard ^
  --gripper-mask-shape-profile="%PROJECT_ROOT%\旧视觉方案\配置\夹爪目标轮廓配置.json" ^
  --gripper-max-area-growth=0.30 ^
  --gripper-mask-area-tolerance=0.28 ^
  --gripper-mask-axis-tolerance=0.35 ^
  --gripper-mask-circularity-drop=0.12 ^
  --gripper-mask-solidity-drop=0.06 ^
  --gripper-max-cumulative-lift-drift-deg=999.0 ^
  --gripper-cumulative-lift-margin-deg=0.45 ^
  --gripper-height-guard-action=stop ^
  --gripper-hover-input-margin-deg=0.5 ^
  --gripper-motion-max-px-per-deg=35 ^
  --gripper-motion-extra-joint-max-deg=0.85 ^
  --gripper-motion-extra-joint-ratio=0.80 ^
  --disable-csv-orientation ^
  %*

if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

endlocal
