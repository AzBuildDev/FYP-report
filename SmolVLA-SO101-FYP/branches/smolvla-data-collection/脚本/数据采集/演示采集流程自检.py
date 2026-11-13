"""用模拟机械臂验证开始、停止、保存及视频重读；不访问真实串口和相机。"""
import tempfile
import queue
import threading
import time
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tkinter as tk
if "--no-console" in sys.argv:
    sys.stdout = sys.stderr = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "程序模块"))
from 录制任务演示 import RecordingWindow, ROOT
from serial.tools import list_ports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower import so101_follower
from lerobot.teleoperators.so101_leader import so101_leader
from 检查采集批次 import inspect_collection

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


class SimulatedBus:
    is_connected = False
    motors = {k.removesuffix(".pos"): None for k in NAMES}
    torque = 0
    fault = 0
    goal_writes = 0

    def read(self, key, motor, **kwargs):
        return self.torque if key == "Torque_Enable" else self.fault

    def write(self, key, motor, value, **kwargs):
        self.torque = value


    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque):
        assert not disable_torque
        self.is_connected = False

    def disable_torque(self):
        self.torque = 0

    def sync_write(self, *args):
        self.goal_writes += 1

    def sync_read(self, key, **kwargs):
        return {k.removesuffix(".pos"): self.torque if key == "Torque_Enable" else 1 for k in NAMES}


class SimulatedArm:
    is_calibrated = True
    name = "so101_follower"
    action_features = {k: float for k in NAMES}
    instances = []

    def __init__(self, config):
        self.bus, self.n = SimulatedBus(), 0
        self.instances.append(self)

    def configure(self):
        self.bus.torque = 1

    def get_observation(self):
        self.n += 1
        return {k: self.n / 10 for k in NAMES}

    get_action = get_observation

    def send_action(self, action):
        return action


def main():
    ports = [SimpleNamespace(vid=0x1A86, pid=0x55D3, serial_number=s, device=p) for s, p in [("5C82108314", "SIM_LEADER"), ("5C82108242", "SIM_FOLLOWER")]]
    with patch.object(so101_follower, "SO101Follower", SimulatedArm), patch.object(so101_leader, "SO101Leader", SimulatedArm), patch.object(list_ports, "comports", return_value=ports):
        # 故障必须在配置扭矩/位置目标前拦截，所有串口仍要释放。
        probe = RecordingWindow.__new__(RecordingWindow)
        probe.dataset, probe.attempt, probe.closing, probe.sealed = None, None, False, False
        probe.events = queue.Queue()
        probe.path = ROOT / "运行环境/临时文件/仅模拟故障检查"
        with patch.object(SimulatedBus, "fault", 32):
            probe.record_episode("软件故障测试")
        assert all(arm.bus.goal_writes == 0 and not arm.bus.is_connected for arm in SimulatedArm.instances)
        assert not probe.events.get_nowait()[1][0]
        SimulatedArm.instances.clear()
        window = tk.Tk()
        window.withdraw()
        app = RecordingWindow(window)
        app.path = Path(tempfile.mkdtemp(prefix="recording_ui_check_", dir=ROOT / "运行环境/临时文件")) / "dataset"
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[50:300, 50:300] = 130
        stop = threading.Event()

        def feed():
            while not stop.wait(.02):
                app.camera_message("frame", (time.monotonic(), image))

        threading.Thread(target=feed, daemon=True).start()
        state = {"stage": 0, "failure": None, "saved": False}

        def monitor():
            try:
                size = app.dataset.episode_buffer.get("size", 0) if app.dataset is not None else 0
                if size >= 60:
                    app.arm_stop.set()
                if app.pending and not app.busy:
                    if state["stage"] == 0:
                        state["stage"] = 1
                        app.finish_episode(False)
                    elif state["stage"] == 2:
                        state["stage"] = 3
                        app.finish_episode(True)
                    elif state["stage"] == 4:
                        state["stage"] = 5
                        def encoding_failure():
                            # 模拟上游保存中途已经改变缓冲结构的编码异常。
                            app.dataset.episode_buffer.pop("size", None)
                            raise RuntimeError("模拟视频编码失败")
                        app.dataset.save_episode = encoding_failure
                        app.finish_episode(True)
                if not app.busy and not app.pending:
                    if state["stage"] == 1:
                        records = list((app.path / "采集尝试").glob("*/尝试状态.json"))
                        assert len(records) == 1
                        row = json.loads(records[0].read_text(encoding="utf-8"))
                        assert row["status"] == "rejected" and row["archive_complete"]
                        assert len(list((records[0].parent / "原始帧").glob("*.png"))) >= 60
                        state["stage"] = 2
                        app.start_episode()
                    elif state["stage"] == 3:
                        assert app.dataset.num_episodes == 1
                        assert (app.path / "第001段逐帧时间.json").is_file()
                        result = inspect_collection(app.path)
                        assert result["status"] == "passed", result["errors"]
                        state["stage"] = 4
                        app.start_episode()
                    elif state["stage"] == 5 and app.sealed:
                        records = [json.loads(p.read_text(encoding="utf-8")) for p in (app.path / "采集尝试").glob("*/尝试状态.json")]
                        assert sorted(r["status"] for r in records) == ["rejected", "save_failed", "saved"]
                        assert len({r["attempt_id"] for r in records}) == 3
                        assert all(r["archive_complete"] and r["frames"] >= 60 for r in records)
                        assert all(not arm.bus.is_connected for arm in SimulatedArm.instances)
                        assert str(app.start_button["state"]) == "disabled", repr(app.start_button["state"])
                        assert app.path.joinpath("批次封存状态.json").is_file()
                        state["saved"] = True
                        stop.set()
                        app.close()
                        return
            except Exception as exc:
                state["failure"] = repr(exc)
                stop.set()
                app.pending = False
                app.close()
                return
            window.after(50, monitor)

        def timeout():
            state["failure"] = app.status.get()
            stop.set()
            app.pending = False
            app.close()

        window.after(300, app.start_episode)
        window.after(700, monitor)
        window.after(90000, timeout)
        window.mainloop()
        assert state["saved"] and not state["failure"], state
        assert app.dataset.num_episodes == 1
        report = {"status": "passed", "scope": "simulated_arms_real_video_encoding", "robot_commands_sent": 0,
                  "real_demonstrations_collected": 0, "checks": ["fault_before_enable_blocks_goals", "failed_take_retained", "retry_unique_id", "actual_video_roundtrip",
                  "action_state_parquet_match", "injected_save_failure_preserved", "sealed_batch_blocks_retry",
                  "all_ports_closed", "pythonw_streams_writable"], "dataset_root": str(app.path)}
        (ROOT / "运行记录/采集流程自检最新结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print("采集流程自检通过：失败归档、重录、视频保存重读、编码故障保留与封存均正常。", flush=True)
        print(f"合成测试数据：{app.path}", flush=True)



if __name__ == "__main__":
    main()
