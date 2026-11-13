# LLM-VLM-Powered-Robot-Control-System

基于 SO101 与 LeRobot 的轻量级语言-视觉-动作机器人控制系统。

本项目当前的研究路线不是“只训练一个 ACT 模型去重复固定动作”，而是构建一个更接近 VLA 的闭环控制流程：

```text
用户语言指令
-> 理解任务
-> 在相机画面中找到目标
-> 判断当前任务阶段
-> 生成下一步子目标
-> 输出机器人动作
-> 再次观察
-> 持续循环直到任务完成
```

当前工程基线：

- 硬件：SO101 小型机械臂
- 软件框架：LeRobot
- 当前控制基线：ACT imitation learning
- 研究方向：显式子目标引导的 VLA 式控制，并与直接行为克隆进行比较

## 1. 仓库用途

这个仓库目前保存的是一份可运行的 LeRobot 代码快照，并包含我们针对 SO101 本地 bring-up 做过的关键改动，例如：

- SO101 follower 的软件安全限位
- 电机总线读写重试与日志调整
- 录制流程的手动保留/重录控制
- 一些相机与配置读取兼容性修正

如果你是第一次接手这个项目，可以把它理解为：

- `src/lerobot`：主代码
- `src/lerobot/robots/so101_follower`：SO101 机械臂相关实现
- `src/lerobot/record.py`：数据采集入口
- `src/lerobot/scripts/train.py`：训练入口
- `tests/`：测试与测试产物

## 2. 环境准备

建议使用 `conda` 创建 Python 3.10 环境：

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

安装依赖并以可编辑模式安装本仓库：

```bash
pip install -e .
```

如果你在 macOS 上工作，建议额外参考：

- `requirements-macos.txt`

如果你在 Ubuntu 上工作，参考：

- `requirements-ubuntu.txt`

## 3. 首次检查

进入仓库后，先确认导入正常：

```bash
python -c "import lerobot; print('lerobot import ok')"
```

再确认 LeRobot 命令可用：

```bash
lerobot-teleoperate --help
lerobot-record --help
```

## 4. 相机检查

在开始遥操作或数据采集前，先检查相机。

```bash
python -m lerobot.find_cameras opencv --record-time-s 2
```

如果是 macOS，需要先给终端或 IDE 打开摄像头权限。

我们当前本地使用过的典型相机分配是：

- `0`：俯视/桌面相机
- `1`：机械臂前视/夹爪相机

但不同电脑上编号可能不同，所以请先实际检查后再改命令。

## 5. 遥操作

先完成机械臂连接、串口识别和标定文件准备，再做遥操作测试。

一个当前可参考的 SO101 遥操作命令如下：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem_FOLLOWER \
  --robot.id=my_awesome_follower_arm \
  --robot.max_relative_target=10 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem_LEADER \
  --teleop.id=my_awesome_leader_arm \
  --fps=20 \
  --teleop_time_s=30
```

说明：

- 把 `/dev/tty.usbmodem_FOLLOWER` 和 `/dev/tty.usbmodem_LEADER` 替换成你自己的串口
- `--robot.max_relative_target=10` 是目前较稳妥的设置
- 如果刚开始测试，建议先短时间运行，确认每个关节方向都正常

如果需要双相机可视化预览，可以这样跑：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem_FOLLOWER \
  --robot.id=my_awesome_follower_arm \
  --robot.max_relative_target=10 \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem_LEADER \
  --teleop.id=my_awesome_leader_arm \
  --fps=20 \
  --teleop_time_s=30 \
  --display_data=true
```

## 6. 数据采集

在遥操作稳定后，再开始采集演示数据。

一个本地数据采集样例：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem_FOLLOWER \
  --robot.id=my_awesome_follower_arm \
  --robot.max_relative_target=10 \
  --robot.cameras="{ overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem_LEADER \
  --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=az/so101_pick_test \
  --dataset.root=./data \
  --dataset.single_task="Pick up the object and place it in the target area" \
  --dataset.fps=20 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=2 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false
```

建议先录一个很小的数据集做 smoke test，再扩大采集量。

当前代码还支持手动控制每个 episode 是否保留；如果你在配置里启用了 `manual_episode_control`，就可以在录制后选择：

- 保留当前 episode
- 丢弃并重录
- 直接停止录制

## 7. 训练 ACT 基线

当前 practical baseline 是 ACT imitation learning。

训练前先确认：

- 数据集路径正确
- 图像和动作都已经正确保存
- 任务描述和数据采集任务一致

一个常见训练入口是：

```bash
python -m lerobot.scripts.train --help
```

如果你已经准备好配置，可以基于 LeRobot 的训练脚本启动 ACT 训练。建议一开始先用少量 episode 做通路验证，再扩充数据量和训练步数。

## 8. 推理与评估

训练完成后，可以把策略部署到 SO101 上做真实机器人测试。建议记录以下内容：

- 成功率
- 失败类型
- 每次任务所处阶段
- 是否生成了合理的子目标
- 最终动作是否与当前视觉状态匹配

这部分对 FYP 很关键，因为我们的目标不只是“机械臂能动”，而是证明：

- 语言会影响目标选择
- 视觉会影响场景定位与阶段判断
- 子目标会影响下一步动作
- 系统能在闭环中持续重规划

## 9. 目前建议的协作流程

其他同学接手时，建议按这个顺序：

1. 克隆仓库并配置 `conda` 环境
2. `pip install -e .`
3. 检查相机是否可读
4. 检查 follower 与 leader 串口
5. 做一次短时遥操作测试
6. 录制一个最小数据集
7. 跑通一次 ACT 训练
8. 做一次真实机器人推理验证
9. 再继续做 VLA / 子目标层扩展

## 10. 安全说明

操作真实机械臂前请注意：

- 首次运行时降低速度，不要直接大范围运动
- 确认每个关节方向和映射正常
- 避免夹爪长时间顶在物理极限位置
- 遇到异常抖动、过载、串口失联时立即停止

当前仓库已经加入了一部分软件安全保护，但这不能替代人工看护。

## 11. 当前项目定位

这个仓库现阶段更像：

- 一个可运行的 SO101 + LeRobot 研究基线
- 一个用于演示采集、ACT 训练、真实臂验证的工程基础
- 一个准备继续往“语言 + 视觉 + 子目标 + 动作”闭环系统扩展的起点

后续如果继续完善，优先建议补充：

- 更清晰的数据集说明
- 训练配置样例
- 真实机器人评估脚本
- 子目标状态机实现
- 面向 FYP 的实验记录规范
