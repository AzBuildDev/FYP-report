@echo off
setlocal

call conda activate lerobot
cd /d F:\FYP\Project\so101_mac_bringup\lerobot

set "HF_HOME=F:\FYP\Project\so101_mac_bringup\.cache\huggingface"
set "TORCH_HOME=F:\FYP\Project\so101_mac_bringup\.cache\torch"
set "HF_LEROBOT_CALIBRATION=F:\FYP\Project\so101_mac_bringup\calibration_backup"
set "TMP=F:\FYP\Project\so101_mac_bringup\.tmp"
set "TEMP=F:\FYP\Project\so101_mac_bringup\.tmp"

python -m lerobot.scripts.server.policy_server --host=127.0.0.1 --port=8080 --fps=10 --inference_latency=0.1 --obs_queue_timeout=1
