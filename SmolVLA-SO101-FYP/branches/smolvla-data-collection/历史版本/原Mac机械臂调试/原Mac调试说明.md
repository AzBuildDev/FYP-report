# SO101 Mac Bring-up Working Directory

This folder is the local working copy we will use for the SO-ARM101 / SO-101 bring-up on macOS.

## Active Code

- Code root: `./lerobot`
- Active conda env: `lerobot`
- Verified import path: `./lerobot/src/lerobot/__init__.py`
- Safety patch: `./lerobot/src/lerobot/robots/utils.py`

The safety patch allows `--robot.max_relative_target=2` style integer values to work correctly, so first teleoperation tests can be run with conservative relative motion limiting.

## Detected Ports

- Follower arm: `/dev/tty.usbmodem5C821082421`
- Leader arm: `/dev/tty.usbmodem5C821083141`

Verified again on 2026-06-11:

- `/dev/tty.usbmodem5C821082421` is still present.
- `/dev/tty.usbmodem5C821083141` is still present.
- `torch`, `lerobot`, `pyserial`, and `cv2` import successfully inside the `lerobot` conda environment.
- `lerobot-teleoperate --help` exposes `so101_follower`, `so101_leader`, and `--robot.max_relative_target`.
- Current camera check fails because macOS has not granted camera access to the terminal/app yet.

## Calibration Backup

The current EEPROM calibration values were read from the motors and backed up here:

- `./calibration_backup/robots/so101_follower/my_awesome_follower_arm.json`
- `./calibration_backup/teleoperators/so101_leader/my_awesome_leader_arm.json`

The active LeRobot calibration files are also installed under:

- `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_awesome_follower_arm.json`
- `~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_awesome_leader_arm.json`

Verified again on 2026-06-11:

- Active follower calibration parses correctly and contains 6 joints.
- Active leader calibration parses correctly and contains 6 joints.
- Backup follower calibration parses correctly and contains 6 joints.
- Backup leader calibration parses correctly and contains 6 joints.

Do not rerun `lerobot-setup-motors`, flash firmware, or overwrite calibration unless there is a clear hardware problem.

## Teleoperation Status

Verified on 2026-06-11:

- Conservative leader-to-follower teleoperation works.
- `--robot.max_relative_target=2` is very slow when leader and follower poses differ.
- `--robot.max_relative_target=5`, `--fps=10`, and `--teleop_time_s=30` successfully tracks the leader arm.
- `--robot.max_relative_target=10` and `--fps=20` works better for the current arm and is the preferred teleoperation setting for now.
- One motor appears slower than the others; keep monitoring it, but it does not block the next bring-up step.
- The gripper open/close mapping has been recalibrated and is ready for another teleoperation check.
- The follower gripper motor can report overload if held hard at the physical closed limit. Use the fast setting below, but avoid pressing the gripper against the limit for a long time.

## Camera Permission

Before running camera-based teleoperation or data collection, grant camera permission to the terminal/app used to run LeRobot:

1. Open macOS System Settings.
2. Go to Privacy & Security -> Camera.
3. Enable camera access for the terminal or Codex/IDE app.
4. Restart the terminal/app.
5. Recheck with:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd "/Users/az/Documents/01_学习工作/FYP/so101_mac_bringup/lerobot"

python -m lerobot.find_cameras opencv --record-time-s 2
```

The camera checker saves frames under `outputs/captured_images` by default. If images are saved there, the camera is visible to LeRobot.

Verified on 2026-06-12:

- OpenCV detects camera indices `0`, `1`, and `2`.
- `index_or_path: 2` is the Mac front-facing camera.
- The robot kit camera is `index_or_path: 0`.

Verified on 2026-06-13 after connecting the overhead camera:

- macOS sees `icspring camera`, `1080P USB Camera`, the MacBook camera, and iPhone camera.
- LeRobot/OpenCV detects camera indices `0`, `1`, `2`, and `3`.
- `index_or_path: 0` is the overhead/table camera.
- `index_or_path: 1` is the robot front/gripper camera.
- `index_or_path: 2` is the MacBook camera.
- `index_or_path: 3` is not a useful robot view for dataset collection.
- If `python -m lerobot.find_cameras opencv --record-time-s 2` reports read failures for `1` or `2`, test explicit `640x480@30`; the direct probe successfully read all four indices at that profile.
- Current macOS/OpenCV behavior: robot cameras `0` and `1` cannot be read at the same time on the current USB/camera setup. Whichever of `0` or `1` opens first can read frames; the second fails. Pairs `0+2`, `1+2`, `0+3`, and `1+3` can read simultaneously. For two robot-view cameras, try moving one USB camera to a different USB controller/adapter before recording dual-camera datasets.
- A one-camera overhead smoke recording succeeded on 2026-06-13 at `./datasets/so101_overhead_smoke_20260613_111814`, producing both `data/chunk-000/episode_000000.parquet` and `videos/chunk-000/observation.images.overhead/episode_000000.mp4`.

Verified on 2026-06-14 after plugging the two robot cameras into separate USB hubs/docks:

- LeRobot/OpenCV still detects camera indices `0`, `1`, `2`, and `3`.
- Explicit concurrent OpenCV read of robot cameras `0` and `1` at `640x480@30` succeeded.
- Both cameras returned `60/60` frames in the same process.
- `index_or_path: 0` is the overhead/table camera.
- `index_or_path: 1` is the robot front/gripper camera.
- A side-by-side verification image was saved at `./outputs/camera_dual_probe_20260614/camera_0_1_side_by_side.jpg`.
- The previous limitation that cameras `0` and `1` could not be read simultaneously appears resolved with the separate hub/dock setup.

## Camera Preview

LeRobot is mostly command-line driven. It does not provide a polished end-user GUI for the whole workflow, but it can show camera streams through its data viewer when `--display_data=true` is used.

After the camera is mounted, connected, and detected, test live preview with:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd "/Users/az/Documents/01_学习工作/FYP/so101_mac_bringup/lerobot"

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5C821082421 \
  --robot.id=my_awesome_follower_arm \
  --robot.max_relative_target=10 \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5C821083141 \
  --teleop.id=my_awesome_leader_arm \
  --fps=20 \
  --teleop_time_s=30 \
  --display_data=true
```

If either camera opens the wrong view, use frames saved under `outputs/captured_images` to remap the indices.

## Preferred Teleoperation Test

Use this as the current default teleoperation command:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd "/Users/az/Documents/01_学习工作/FYP/so101_mac_bringup/lerobot"

lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5C821082421 \
  --robot.id=my_awesome_follower_arm \
  --robot.max_relative_target=10 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5C821083141 \
  --teleop.id=my_awesome_leader_arm \
  --fps=20 \
  --teleop_time_s=30
```

## First Local Recording Test

After teleoperation and camera preview are stable, record a tiny local dataset before any large collection.
This command records 2 short episodes locally and does not upload to Hugging Face:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd "/Users/az/Documents/01_学习工作/FYP/so101_mac_bringup/lerobot"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5C821082421 \
  --robot.id=my_awesome_follower_arm \
  --robot.max_relative_target=10 \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5C821083141 \
  --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=az/so101_pick_test_20260612 \
  --dataset.root="/Users/az/Documents/01_学习工作/FYP/so101_mac_bringup/datasets" \
  --dataset.single_task="Pick up the object and place it in the target area" \
  --dataset.fps=20 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=2 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false
```

If this succeeds, inspect the saved dataset under `./datasets/az/so101_pick_test_20260612` before increasing episode count.
