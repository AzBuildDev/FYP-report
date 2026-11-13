# SO101 Windows 训练推理清单

这份文档面向当前这台 Windows 电脑上的 SO101 项目使用流程。

当前正式数据集是：

```text
F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep_v3
```

当前计划使用的正式训练输出文件夹是：

```text
act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4
```

目标链路：

1. 主臂控制从臂
2. 双摄像头录制数据
3. 训练 ACT 模型
4. 用训练好的模型做真实机械臂推理

## 0. 当前确认的硬件映射

- 从臂 follower: `COM11`
- 主臂 leader: `COM12`
- `overhead` 摄像头: `index_or_path: 2`
- `front` 摄像头: `index_or_path: 0`

## 1. 每次开新终端先执行

### 1.1 分行版

```bat
conda activate lerobot
```

```bat
cd /d F:\FYP\Project\so101_mac_bringup\lerobot
```

```bat
set HF_HOME=F:\FYP\Project\so101_mac_bringup\.cache\huggingface
```

```bat
set TORCH_HOME=F:\FYP\Project\so101_mac_bringup\.cache\torch
```

```bat
set HF_LEROBOT_CALIBRATION=F:\FYP\Project\so101_mac_bringup\calibration_backup
```

```bat
set TMP=F:\FYP\Project\so101_mac_bringup\.tmp
```

```bat
set TEMP=F:\FYP\Project\so101_mac_bringup\.tmp
```

如果当前项目代码还没装进环境，再执行一次：

```bat
pip install -e .
```

### 1.2 通用一行版

```bat
conda activate lerobot && cd /d F:\FYP\Project\so101_mac_bringup\lerobot && set "HF_HOME=F:\FYP\Project\so101_mac_bringup\.cache\huggingface" && set "TORCH_HOME=F:\FYP\Project\so101_mac_bringup\.cache\torch" && set "HF_LEROBOT_CALIBRATION=F:\FYP\Project\so101_mac_bringup\calibration_backup" && set "TMP=F:\FYP\Project\so101_mac_bringup\.tmp" && set "TEMP=F:\FYP\Project\so101_mac_bringup\.tmp"
```

## 2. 确认 CUDA 和显卡可用

```bat
python -c "import torch; print('torch =', torch.__version__); print('torch.version.cuda =', torch.version.cuda); print('cuda_available =', torch.cuda.is_available()); print('gpu_name =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

如果输出里有：

- `cuda_available = True`
- `gpu_name = NVIDIA GeForce ...`

就说明后面的训练和推理都可以用 GPU。

## 3. 遥操作测试

这是当前已经验证过可以正常工作的主从臂加双摄像头命令：

```bat
lerobot-teleoperate --robot.type=so101_follower --robot.port=COM11 --robot.id=my_awesome_follower_arm --robot.max_relative_target=10 --robot.cameras="{ overhead: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM12 --teleop.id=my_awesome_leader_arm --fps=20 --teleop_time_s=30 --display_data=true
```

如果提示：

```text
Press ENTER to use provided calibration file associated with the id ...
```

直接按 `Enter`，不要输入 `c`。

### 3.1 遥操作一行启动版

```bat
conda activate lerobot && cd /d F:\FYP\Project\so101_mac_bringup\lerobot && set "HF_HOME=F:\FYP\Project\so101_mac_bringup\.cache\huggingface" && set "TORCH_HOME=F:\FYP\Project\so101_mac_bringup\.cache\torch" && set "HF_LEROBOT_CALIBRATION=F:\FYP\Project\so101_mac_bringup\calibration_backup" && set "TMP=F:\FYP\Project\so101_mac_bringup\.tmp" && set "TEMP=F:\FYP\Project\so101_mac_bringup\.tmp" && lerobot-teleoperate --robot.type=so101_follower --robot.port=COM11 --robot.id=my_awesome_follower_arm --robot.max_relative_target=10 --robot.cameras="{ overhead: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM12 --teleop.id=my_awesome_leader_arm --fps=20 --teleop_time_s=30 --display_data=true
```

## 4. 录制数据

当前正式录制数据集：

```text
F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep_v3
```

当前本机检查结果：

- `150` 个 episode
- `81447` 帧
- `300` 个视频，也就是 `front` 和 `overhead` 两路摄像头各 `150` 个

### 4.1 从零录制 v3 150 条

只有当你明确要重新开始一套新的 `v3` 数据时才用这条。如果同名目录已经存在，不要直接运行。

```bat
lerobot-record --robot.type=so101_follower --robot.port=COM11 --robot.id=my_awesome_follower_arm --robot.max_relative_target=10 --robot.cameras="{ overhead: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM12 --teleop.id=my_awesome_leader_arm --dataset.repo_id=local/so101_pick_place_dualcam_150ep_v3 --dataset.root=F:/FYP/Project/so101_mac_bringup/datasets/so101_pick_place_dualcam_150ep_v3 --dataset.single_task="Pick up the object and place it in the target area" --dataset.fps=20 --dataset.episode_time_s=30 --dataset.reset_time_s=5 --dataset.num_episodes=150 --dataset.manual_episode_control=true --dataset.push_to_hub=false --display_data=true --play_sounds=false
```

### 4.2 续录规则

`Ready for episode X` 是本次命令会话里的编号，不一定是整个数据集的真实编号。

真正要看的是日志里的：

```text
Recording episode Y
```

例如日志显示 `Recording episode 70`，保存后下一条真实编号就是 `71`。

如果要续录，必须加：

```text
--resume=true
```

并且 `--dataset.num_episodes` 填“还差多少条”，不是填总数。

### 4.3 录制时怎么操作

这条命令只需要启动一次，不需要每个 episode 都重新输入。

录制过程中最常用的输入是：

- `y` 然后回车：保留当前 episode
- `n` 然后回车：丢弃当前 episode 并重录
- `q` 然后回车：结束整次录制会话

如果停在：

```text
Ready for episode ...
```

这个界面，按 `Ctrl+C` 暂停通常是安全的，因为下一条还没开始录。

## 5. 训练

训练不需要连接机械臂和摄像头，只需要数据集在本地，环境和 CUDA 正常即可。

### 5.1 当前正式 250k 训练命令

这条是在另一台电脑上训练 `v3` 数据集时使用的命令。那台电脑项目路径是 `D:/FYP/so101_mac_bringup`。

```bat
lerobot-train --policy.type=act --policy.device=cuda --policy.dim_model=64 --policy.n_action_steps=20 --policy.chunk_size=20 --policy.push_to_hub=false --dataset.repo_id=local/so101_pick_place_dualcam_150ep_v3 --dataset.root=D:/FYP/so101_mac_bringup/datasets/so101_pick_place_dualcam_150ep_v3 --batch_size=8 --steps=250000 --eval_freq=0 --save_freq=10000 --save_checkpoint=true --log_freq=20 --num_workers=4 --wandb.enable=false --output_dir=outputs/train/act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4
```

训练完成后，模型输出文件夹应该是：

```text
D:\FYP\so101_mac_bringup\lerobot\outputs\train\act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4
```

### 5.2 如果要在当前电脑训练同一套 v3

当前电脑项目路径是 `F:/FYP/Project/so101_mac_bringup`，所以命令里的 `--dataset.root` 要换成 F 盘路径：

```bat
lerobot-train --policy.type=act --policy.device=cuda --policy.dim_model=64 --policy.n_action_steps=20 --policy.chunk_size=20 --policy.push_to_hub=false --dataset.repo_id=local/so101_pick_place_dualcam_150ep_v3 --dataset.root=F:/FYP/Project/so101_mac_bringup/datasets/so101_pick_place_dualcam_150ep_v3 --batch_size=8 --steps=250000 --eval_freq=0 --save_freq=10000 --save_checkpoint=true --log_freq=20 --num_workers=4 --wandb.enable=false --output_dir=outputs/train/act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4
```

如果当前电脑训练速度或显存压力比较大，可以把 `--batch_size=8 --num_workers=4` 改成：

```text
--batch_size=2 --num_workers=0
```

### 5.3 step 和 checkpoint

- `250000` 指的是 `250000 steps`，不是 `250000 个 episode`
- 当前 `v3` 数据集有 `150` 个 episode，约 `81447` 帧
- `batch_size=8` 时，`1 epoch` 约等于 `81447 / 8 = 10181` steps
- `250000 steps` 约等于 `24.5` epochs
- 因为 `--save_freq=10000`，所以会每 `10000` steps 保存一个 checkpoint

训练完成后，优先测试：

```text
checkpoints/250000/pretrained_model
```

如果效果不如中间模型，也可以回测 `200000`、`150000`、`100000` 等 checkpoint。

## 6. 从另一台电脑拿回训练好的模型

另一台电脑训练完后，把这个完整文件夹复制到当前电脑：

```text
D:\FYP\so101_mac_bringup\lerobot\outputs\train\act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4
```

放到当前电脑这里：

```text
F:\FYP\Project\so101_mac_bringup\lerobot\outputs\train\act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4
```

复制完成后，最终模型路径应该类似：

```text
F:\FYP\Project\so101_mac_bringup\lerobot\outputs\train\act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4\checkpoints\250000\pretrained_model
```

## 7. 推理

推理需要两个终端。

### 7.1 终端 1：启动 policy server

```bat
python -m lerobot.scripts.server.policy_server --host=127.0.0.1 --port=8080 --fps=10 --inference_latency=0.1 --obs_queue_timeout=1
```

一行启动版：

```bat
conda activate lerobot && cd /d F:\FYP\Project\so101_mac_bringup\lerobot && set "HF_HOME=F:\FYP\Project\so101_mac_bringup\.cache\huggingface" && set "TORCH_HOME=F:\FYP\Project\so101_mac_bringup\.cache\torch" && set "HF_LEROBOT_CALIBRATION=F:\FYP\Project\so101_mac_bringup\calibration_backup" && set "TMP=F:\FYP\Project\so101_mac_bringup\.tmp" && set "TEMP=F:\FYP\Project\so101_mac_bringup\.tmp" && python -m lerobot.scripts.server.policy_server --host=127.0.0.1 --port=8080 --fps=10 --inference_latency=0.1 --obs_queue_timeout=1
```

### 7.2 终端 2：用 250k 模型推理

这条命令要求你已经把另一台电脑训练好的模型文件夹复制到当前电脑。

```bat
python -m lerobot.scripts.server.robot_client --robot.type=so101_follower --robot.port=COM11 --robot.id=my_awesome_follower_arm --robot.max_relative_target=5 --robot.cameras="{ overhead: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --server_address=127.0.0.1:8080 --policy_type=act --pretrained_name_or_path=F:/FYP/Project/so101_mac_bringup/lerobot/outputs/train/act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4/checkpoints/250000/pretrained_model --policy_device=cuda --actions_per_chunk=10 --fps=10 --chunk_size_threshold=0.5 --aggregate_fn_name=latest_only --task="Pick up the object and place it in the target area"
```

一行启动版：

```bat
conda activate lerobot && cd /d F:\FYP\Project\so101_mac_bringup\lerobot && set "HF_HOME=F:\FYP\Project\so101_mac_bringup\.cache\huggingface" && set "TORCH_HOME=F:\FYP\Project\so101_mac_bringup\.cache\torch" && set "HF_LEROBOT_CALIBRATION=F:\FYP\Project\so101_mac_bringup\calibration_backup" && set "TMP=F:\FYP\Project\so101_mac_bringup\.tmp" && set "TEMP=F:\FYP\Project\so101_mac_bringup\.tmp" && python -m lerobot.scripts.server.robot_client --robot.type=so101_follower --robot.port=COM11 --robot.id=my_awesome_follower_arm --robot.max_relative_target=5 --robot.cameras="{ overhead: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --server_address=127.0.0.1:8080 --policy_type=act --pretrained_name_or_path=F:/FYP/Project/so101_mac_bringup/lerobot/outputs/train/act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4/checkpoints/250000/pretrained_model --policy_device=cuda --actions_per_chunk=10 --fps=10 --chunk_size_threshold=0.5 --aggregate_fn_name=latest_only --task="Pick up the object and place it in the target area"
```

如果要换成其他 checkpoint，只改这一段：

```text
--pretrained_name_or_path=F:/FYP/Project/so101_mac_bringup/lerobot/outputs/train/act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4/checkpoints/250000/pretrained_model
```

例如测试 `200000`：

```text
--pretrained_name_or_path=F:/FYP/Project/so101_mac_bringup/lerobot/outputs/train/act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4/checkpoints/200000/pretrained_model
```

## 8. 数据和模型会存到哪里

- 当前正式数据集：`F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep_v3`
- 旧版 150 条数据集：`F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep`
- 训练输出：`F:\FYP\Project\so101_mac_bringup\lerobot\outputs\train\...`
- 标定文件：`F:\FYP\Project\so101_mac_bringup\calibration_backup`
- 被移走或删除前备份的数据：`F:\FYP\Project\so101_mac_bringup\dataset_safety_backups`

## 9. 为了避免把 C 盘弄乱

每次执行第 1 节里的初始化命令后，下面这些缓存默认都会去项目目录：

- Hugging Face 缓存：`F:\FYP\Project\so101_mac_bringup\.cache\huggingface`
- PyTorch 缓存：`F:\FYP\Project\so101_mac_bringup\.cache\torch`
- 临时文件：`F:\FYP\Project\so101_mac_bringup\.tmp`

## 10. 当前状态总结

当前已经确认：

- 主从臂遥操作正常
- 双摄像头命令正常
- CUDA 可用
- 当前正式数据集 `so101_pick_place_dualcam_150ep_v3` 已录满 `150` 条
- 当前正式训练计划是 `250000 steps`
- 另一台电脑训练输出文件夹名应为 `act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4`

后续最常用流程：

1. 从另一台电脑复制 `act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4` 到当前电脑的 `lerobot\outputs\train`
2. 开第一个终端运行 `policy_server`
3. 开第二个终端运行 `robot_client`
4. 优先测试 `250000` checkpoint，再按效果回测 `200000`、`150000`、`100000`
