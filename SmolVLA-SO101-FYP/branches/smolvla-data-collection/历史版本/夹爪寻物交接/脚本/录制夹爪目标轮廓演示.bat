@echo off
setlocal

set "PROJECT_ROOT=F:\FYP\Project\so101_mac_bringup"
set "LEROBOT_ROOT=%PROJECT_ROOT%\lerobot"

where conda >nul 2>nul
if "%ERRORLEVEL%"=="0" (
  call conda activate lerobot
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
  call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate lerobot
) else if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
  call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate lerobot
) else if exist "%USERPROFILE%\miniforge3\condabin\conda.bat" (
  call "%USERPROFILE%\miniforge3\condabin\conda.bat" activate lerobot
) else (
  echo ERROR: conda was not found. Run this from Anaconda Prompt or update this bat with your conda.bat path.
  exit /b 1
)

if not "%ERRORLEVEL%"=="0" (
  echo ERROR: failed to activate conda environment lerobot.
  exit /b %ERRORLEVEL%
)

set "HF_HOME=%PROJECT_ROOT%\.cache\huggingface"
set "TORCH_HOME=%PROJECT_ROOT%\.cache\torch"
set "HF_LEROBOT_CALIBRATION=%PROJECT_ROOT%\calibration_backup"
set "TMP=%PROJECT_ROOT%\.tmp"
set "TEMP=%PROJECT_ROOT%\.tmp"
set "PYTHONPATH=%PROJECT_ROOT%\vision\scripts;%LEROBOT_ROOT%\src;%PYTHONPATH%"

if not exist "%TMP%" mkdir "%TMP%"
if not exist "%PROJECT_ROOT%\vision\outputs\gripper_mask_shape_demos" mkdir "%PROJECT_ROOT%\vision\outputs\gripper_mask_shape_demos"

cd /d "%PROJECT_ROOT%"

python "%PROJECT_ROOT%\vision\scripts\record_gripper_servo_demo.py" ^
  --follower-port=COM11 ^
  --leader-port=COM12 ^
  --follower-id=my_awesome_follower_arm ^
  --leader-id=my_awesome_leader_arm ^
  --gripper-camera-index=0 ^
  --target-class=red_cube ^
  --teleop-fps=60 ^
  --record-fps=15 ^
  --session-prefix=gripper_mask_shape_demo ^
  --output-root="%PROJECT_ROOT%\vision\outputs\gripper_mask_shape_demos" ^
  %*

if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

endlocal
