@echo off
setlocal

set "PROJECT_ROOT=F:\FYP\Project\so101_mac_bringup"
set "LEROBOT_ROOT=%PROJECT_ROOT%\lerobot"
set "HF_HOME=%PROJECT_ROOT%\.cache\huggingface"
set "TORCH_HOME=%PROJECT_ROOT%\.cache\torch"
set "HF_LEROBOT_CALIBRATION=%PROJECT_ROOT%\calibration_backup"
set "TMP=%PROJECT_ROOT%\.tmp"
set "TEMP=%PROJECT_ROOT%\.tmp"

cd /d "%LEROBOT_ROOT%"

python "%PROJECT_ROOT%\scripts\记录遥操作电机轨迹.py" ^
  --follower_port=COM11 ^
  --leader_port=COM12 ^
  --follower_id=my_awesome_follower_arm ^
  --leader_id=my_awesome_leader_arm ^
  --output_dir="%PROJECT_ROOT%\inference_traces" ^
  --trial_id=teleop_home_demo ^
  --fps=20 ^
  --duration_s=30 ^
  --max_relative_target=10

endlocal
