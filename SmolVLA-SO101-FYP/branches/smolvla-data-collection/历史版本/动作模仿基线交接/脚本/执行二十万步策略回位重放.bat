@echo off
setlocal

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"

set "PROJECT_ROOT=F:\FYP\Project\so101_mac_bringup"
set "LEROBOT_ROOT=%PROJECT_ROOT%\lerobot"
set "TRACE_ROOT=%PROJECT_ROOT%\inference_traces"
set "HF_HOME=%PROJECT_ROOT%\.cache\huggingface"
set "TORCH_HOME=%PROJECT_ROOT%\.cache\torch"
set "HF_LEROBOT_CALIBRATION=%PROJECT_ROOT%\calibration_backup"
set "TMP=%PROJECT_ROOT%\.tmp"
set "TEMP=%PROJECT_ROOT%\.tmp"

if not exist "%TMP%" mkdir "%TMP%"
if not exist "%TRACE_ROOT%" mkdir "%TRACE_ROOT%"

call conda activate lerobot
cd /d "%LEROBOT_ROOT%"

python -m lerobot.scripts.server.robot_client ^
  --robot.type=so101_follower ^
  --robot.port=COM11 ^
  --robot.id=my_awesome_follower_arm ^
  --robot.max_relative_target=5 ^
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" ^
  --server_address=127.0.0.1:8080 ^
  --policy_type=act ^
  --pretrained_name_or_path=F:/FYP/Project/so101_mac_bringup/lerobot/outputs/train/act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4/checkpoints/200000/pretrained_model ^
  --policy_device=cuda ^
  --actions_per_chunk=10 ^
  --fps=10 ^
  --chunk_size_threshold=0.5 ^
  --aggregate_fn_name=latest_only ^
  --task="Pick up the object and place it in the target area" ^
  --trace_enabled=true ^
  --trace_output_dir=F:/FYP/Project/so101_mac_bringup/inference_traces ^
  --trace_trial_id=200k_auto_replay_home_%TS% ^
  --post_release_home_enabled=true ^
  --post_release_home_auto_trigger_enabled=true ^
  --post_release_home_manual_trigger_enabled=false ^
  --post_release_home_closed_threshold=25 ^
  --post_release_home_release_gripper_min=45 ^
  --post_release_home_release_gripper_max=60 ^
  --post_release_home_min_open_observations=10 ^
  --post_release_home_release_dwell_s=3 ^
  --post_release_home_min_elapsed_s=20 ^
  --post_release_home_target_pan_min=35 ^
  --post_release_home_target_pan_max=65 ^
  --post_release_home_target_lift_min=25 ^
  --post_release_home_target_lift_max=55 ^
  --post_release_home_target_elbow_min=-90 ^
  --post_release_home_target_elbow_max=-50 ^
  --post_release_home_target_wrist_min=75 ^
  --post_release_home_target_wrist_max=115 ^
  --post_release_home_prelift_enabled=false ^
  --post_release_home_prelift_delta=-65 ^
  --post_release_home_prelift_steps=80 ^
  --post_release_home_sleep_s=0.08 ^
  --post_release_home_replay_trace_path=F:/FYP/Project/so101_mac_bringup/inference_traces/回位演示重放与观测偏移数据.csv ^
  --post_release_home_replay_start_s=0 ^
  --post_release_home_replay_end_s=5.2 ^
  --post_release_home_replay_steps_per_waypoint=1 ^
  --post_release_home_replay_bridge_steps=35 ^
  --post_release_home_replay_bridge_sleep_s=0.06 ^
  --post_release_home_replay_sleep_s=0.05
