@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

call "%SCRIPT_DIR%run_yolo_guided_align_correct.bat" ^
  --gripper-servo-mode=ibvs ^
  --gripper-center-tolerance-px=16 ^
  --gripper-final-iters=8 ^
  --gripper-ibvs-probe-deg=0.8 ^
  --gripper-ibvs-gain=0.45 ^
  --gripper-ibvs-max-probes=4 ^
  --gripper-ibvs-max-worse-px=10 ^
  --gripper-max-step-deg=1.0 ^
  --gripper-min-step-deg=0.30 ^
  --gripper-observation-samples=3 ^
  --gripper-observation-interval-s=0.025 ^
  %*

if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

endlocal
