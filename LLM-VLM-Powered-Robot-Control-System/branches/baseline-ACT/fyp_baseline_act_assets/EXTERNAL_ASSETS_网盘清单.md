# External Assets Not Included in Git

This branch intentionally does not include datasets, trained model checkpoints, or large runtime logs. These files should be shared separately through cloud storage.

## Must Upload to Cloud Storage

### Formal Dataset

- Local path:
  - `F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep_v3`
- Archive found locally:
  - `F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep_v3.zip`
- Approximate size:
  - dataset folder: about 1.5 GB
  - zip archive: about 593 MB
- Purpose:
  - 150-episode dual-camera SO101 pick-and-place dataset.
  - Task text: `Pick up the object and place it in the target area`.

### ACT Training Outputs and Checkpoints

- Local path:
  - `F:\FYP\Project\so101_mac_bringup\lerobot\outputs\train`
- Most important run:
  - `F:\FYP\Project\so101_mac_bringup\lerobot\outputs\train\act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4`
- Priority checkpoint for inference:
  - `...\checkpoints\200000\pretrained_model`
- Other checkpoints tested:
  - `100000`
  - `120000`
  - `150000`
  - `170000`
  - `250000`
- Approximate local size:
  - `lerobot\outputs`: about 16 GB
- Purpose:
  - ACT policy checkpoints for SO101 pick-and-place inference.
  - 200k is the current preferred baseline because it releases the gripper more reliably than 170k.

## Optional Uploads

### Raw Inference Traces and Logs

- Local paths:
  - `F:\FYP\Project\so101_mac_bringup\inference_traces`
  - `F:\FYP\Project\so101_mac_bringup\logs`
  - `F:\FYP\Project\so101_mac_bringup\lerobot\logs`
- Note:
  - The current replay trajectory CSV used by the baseline script is included in this branch under `fyp_baseline_act_assets`.
  - Full raw logs are not required to run the baseline, but can be useful for debugging and thesis evidence.

### Vision Experiments

- Local path:
  - `F:\FYP\Project\so101_mac_bringup\vision`
- Approximate size:
  - about 282 MB
- Note:
  - This folder is outside the LeRobot Git repository and is not included in the baseline branch unless copied manually.

### Dataset Safety Backups and Cache

- Local paths:
  - `F:\FYP\Project\so101_mac_bringup\dataset_safety_backups`
  - `F:\FYP\Project\so101_mac_bringup\.cache`
- Note:
  - These are not required for reproducing the current baseline.

## Files Included in Git Instead

The branch includes code changes and a small handoff package:

- `src/lerobot/scripts/server/robot_client.py`
- `src/lerobot/scripts/server/configs.py`
- `src/lerobot/constants.py`
- `fyp_baseline_act_assets/PROJECT_HANDOFF_当前进度与后续指南.md`
- `fyp_baseline_act_assets/README_Windows_训练推理清单.md`
- `fyp_baseline_act_assets/run_policy_server_8080.bat`
- `fyp_baseline_act_assets/run_robot_200k_auto_replay_home.bat`
- `fyp_baseline_act_assets/run_record_teleop_home_demo.bat`
- `fyp_baseline_act_assets/record_teleop_motor_trace.py`
- `fyp_baseline_act_assets/teleop_home_demo_1783936689_replay_obs_plus10.csv`

