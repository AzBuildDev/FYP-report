# SO101 夹爪寻找物体阶段交接说明

日期：2026-07-29

这个目录是当前 SO101 “夹爪寻找物体 / 双相机视觉对齐”阶段的 GitHub 轻量交接包。完整本地项目根目录是：

```text
F:\FYP\Project\so101_mac_bringup
```

实际连接 GitHub remote 的仓库不是项目根目录，而是里面的 LeRobot 子仓库：

```text
F:\FYP\Project\so101_mac_bringup\lerobot
```

本次使用的 GitHub remote 是：

```text
fyp = https://github.com/Azusagawa0409/LLM-VLM-Powered-Robot-Control-System.git
```

当前分支名是：

```text
gripper-object-search-complete
```

## 这个分支包含什么

这个分支采用“代码 + 可复现实验配置 + 启动脚本 + 测试 + 说明文档”的轻量上传方式。大型数据集、训练 checkpoint、YOLO 权重、运行输出和备份目录没有放进 GitHub，避免超过普通 Git 仓库限制，也避免朋友 clone 时被大文件拖住。

本次包含的快照：

- `vision_scripts/yolo_guided_align.py`：当前最新的双相机 YOLO 对齐主脚本，包含 `N/G/D` 主臂交接和局部 ACT 录制入口。
- `vision_scripts/gripper_visual_servo.py`：夹爪视觉伺服相关的数学 helper。
- `vision_tests/test_yolo_guided_align_regression.py`：视觉对齐主逻辑的回归测试。
- `vision_tests/test_gripper_visual_servo.py`：夹爪视觉伺服 helper 的测试。
- `vision_config/overhead_to_safeplane_homography.json`：当前标定、历史点、紧凑参数模型和 coarse bias 状态。
- `vision_config/gripper_mask_shape_profile.json`：当前夹爪 mask shape profile。
- `run_scripts/*.bat`：当前推荐启动脚本和之前一些关键实验脚本。
- `SO101_STAGE3_RETEACH_HANDOFF_20260722.md`：中文详细交接文档，包含最近暂停前状态。

## 当前推荐启动方式

在真实 SO101 机器的 Anaconda Prompt 里运行：

```bat
conda activate lerobot
F:
cd F:\FYP\Project\so101_mac_bringup
run_yolo_guided_align_learned_tune.bat
```

视觉对齐窗口里的主要操作流程：

```text
A = 自动视觉对齐
N = 从臂张开夹爪，把主臂同步到从臂姿态，并锁住主臂 torque
G = 释放主臂锁定，从同步姿态进入主从 teleoperation
D = 开始/停止局部 ACT 抓取录制
Q = 停止/退出
```

## 已完成的工作

当前成功基线备份是：

```text
backups\good_autonomous_coarse_feedback_20260723_194248
```

已经完成并且软件层面可用的部分：

- 顶部相机 YOLO 目标检测，用于桌面物体的粗定位。
- 通过 `pose_model`、replay points、homography fallback 和 `full_pose` target，把目标像素位置转换成从臂粗定位姿态。
- 夹爪相机末端精定位：用目标中心和 recorded robust reference 的像素误差来判断夹爪是否对准。
- learned full-pose correction：把夹爪相机像素误差映射成五个主体关节的补偿量，而不是只调 pan/lift。
- `comp_reverse_actual`：如果一次视觉移动让目标更远，就按实际测到的关节位移反向补偿，而不是简单按命令反向。
- 紧凑的 `gripper_learned_param_model`：不再无限追加新 calibration point，而是更新像素误差到五关节补偿的参数。
- 紧凑的 `coarse_pose_bias_model`：Stage 3 微调成功后，会反哺 Stage 2 粗定位，让下一次粗定位提前补一点偏差。
- `N` 交接入口：从臂夹爪会先自动张开，主臂会移动到从臂当前姿态，并保持 torque 锁住。
- `G` 交接释放：释放主臂 torque hold，并从同步姿态进入主从 teleoperation。
- `D` 局部 ACT 录制入口：在同一个对齐脚本进程里录制，不另开 `lerobot-record` 抢占 COM11/COM12 或摄像头。

## 还没有实机测试的内容

最新的局部 ACT 抓取录制代码还没有在真实 SO101 硬件上完整验证。

具体还没测：

- 按 `D` 后，是否能在 teleoperation 正常运行时平滑开始局部 ACT 录制。
- 再按一次 `D` 后，是否能稳定保存 episode，并且不影响机器人控制。
- 保存出来的局部 ACT episode 是否有正确的时间节奏和双相机画面质量。
- 主臂夹爪行为是否需要纳入同步。当前默认不同步主臂夹爪，因为物理主臂握把可能不需要或不适合程序控制。
- 第一批局部抓取数据是否足够训练一个 local-grasp ACT policy。

暂停前做过的软件验证：

```bat
F:\Anaconda\envs\lerobot\python.exe -m py_compile vision\scripts\yolo_guided_align.py
F:\Anaconda\envs\lerobot\python.exe -m unittest vision.tests.test_yolo_guided_align_regression
```

结果：

```text
24 tests OK, 2 skipped
无硬件 LeRobotDataset 写入 smoke test 已通过：用假图像和假关节可以 add_frame 并 save_episode。
```

## 接下来还要做什么

下一阶段建议先做这些工作：

- 在真实硬件上测试完整流程：`A -> N -> G -> D -> 手动完成局部抓取 -> D`。
- 检查生成的数据集：`datasets\so101_local_grasp_act_handoff_test`。
- 确认新数据集里的 `observation.state`、`action`、`observation.images.overhead`、`observation.images.front` 和之前双相机 ACT 数据格式一致。
- 在不同物体位置上录制几组“视觉对齐后局部抓取”的 episode。
- 用这些局部抓取片段训练一个 local-grasp ACT policy。
- 写一个简单的数据集回放或 QA 脚本，检查新录制 episode 的双相机画面、action/state 和时间戳。
- 给紧凑学习参数加回滚功能，避免某次错误自动更新污染模型。
- 决定主臂夹爪是否真的需要同步；如果不需要，就保持现在的“只同步主体五关节，夹爪由从臂/ACT 处理”的策略。

## 之前失败或效果有限的尝试

这些内容可以作为论文里的失败路线和工程分析证据：

- 单独使用 pan/lift 做 IBVS 不够稳定，因为 SO101 的关节运动存在耦合、回差和实际执行误差。
- 只靠 encoder rollback 无法稳定恢复同一个视觉状态，即使编码器数值看起来接近，夹爪相机画面也可能没回到同一个位置。
- 早期直接 replay 保存点不够准，后来加入 outer correction 后才明显改善。
- mask area、mask shape 和 height guard 可以作为安全检查，但作为主要对齐逻辑太容易提前 STOP 或误判。
- 每次成功微调都追加新空间点会导致数据越来越大，也有污染 calibration set 的风险，所以后来改成 compact online parameter update。
- 早期完整 ACT pick-and-place policy 对不同物体位置不够鲁棒，所以现在改成“视觉先自动对齐，ACT 只负责局部抓取”的路线。

## GitHub 没有包含的外部资产

下面这些内容有意没有上传到 GitHub。如果朋友需要复现训练、推理或完整实验环境，需要你用网盘或移动硬盘单独发给他们。

大型 ACT 数据集：

```text
F:\FYP\Project\so101_mac_bringup\datasets
F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep_v3
F:\FYP\Project\so101_mac_bringup\datasets\so101_pick_place_dualcam_150ep_v3.zip
F:\FYP\Project\so101_mac_bringup\datasets\so101_local_grasp_act_handoff_test
```

训练输出和 ACT checkpoint：

```text
F:\FYP\Project\so101_mac_bringup\lerobot\outputs\train
F:\FYP\Project\so101_mac_bringup\lerobot\outputs\train\act_so101_pick_place_dualcam_150ep_v3_250000_5080_bs8_w4
```

上面这个训练 run 里比较重要的 checkpoint 文件夹：

```text
100000
120000
150000
170000
200000
250000
```

当前推荐推理 checkpoint：

```text
200000\pretrained_model
```

原因：`200000` checkpoint 比 `170000` 更容易自动松开夹爪；`170000` 抓取更顺，但经常到目标区后不松手；`250000` 容易抽搐；`120000` 效果较弱。

YOLO 模型权重：

```text
F:\FYP\Project\so101_mac_bringup\vision\models\overhead_yolo_seg.pt
F:\FYP\Project\so101_mac_bringup\vision\models\gripper_yolo_seg.pt
```

备份和生成输出目录：

```text
F:\FYP\Project\so101_mac_bringup\backups
F:\FYP\Project\so101_mac_bringup\dataset_safety_backups
F:\FYP\Project\so101_mac_bringup\calibration_backup
F:\FYP\Project\so101_mac_bringup\vision\outputs
F:\FYP\Project\so101_mac_bringup\lerobot\logs
F:\FYP\Project\so101_mac_bringup\lerobot\outputs
```

## 如何还原这份快照

这个目录里的内容是适合 GitHub 上传的轻量快照。真正运行项目时，仍然建议保持原本的项目根目录结构：

```text
F:\FYP\Project\so101_mac_bringup
```

如果要在另一台机器上还原，可以把这个目录里的脚本、配置和测试复制回对应位置，然后再单独提供上面列出的数据集、YOLO 权重和 ACT checkpoint。
