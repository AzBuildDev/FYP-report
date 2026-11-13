"""用模拟机械臂验证开始、停止、保存及视频重读；不访问真实串口和相机。"""
import tempfile
import threading
import time
import json
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tkinter as tk
if "--no-console" in sys.argv:
    sys.stdout = sys.stderr = None
from 录制任务演示 import RecordingWindow, ROOT, CAMERAS
from serial.tools import list_ports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower import so101_follower
from lerobot.teleoperators.so101_leader import so101_leader
from 检查演示数据 import inspect_dataset
from 红蓝抓取任务 import TASKS

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


class SimulatedBus:
    is_connected = False

    def connect(self):
        self.is_connected = True

    def disconnect(self, disable_torque):
        assert disable_torque
        self.is_connected = False

    def disable_torque(self):
        pass

    def sync_write(self, *args):
        pass

    def sync_read(self, key, **kwargs):
        return {k.removesuffix(".pos"): 1 for k in NAMES}


class SimulatedArm:
    is_calibrated = True
    name = "so101_follower"
    action_features = {k: float for k in NAMES}
    instances = []

    def __init__(self, config):
        self.bus, self.n = SimulatedBus(), 0
        self.instances.append(self)

    def configure(self):
        pass

    def get_observation(self):
        self.n += 1
        return {k: self.n / 10 for k in NAMES}

    get_action = get_observation

    def send_action(self, action):
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-cameras", action="store_true", help="真实双摄配合模拟机械臂，结果仅放在临时文件中")
    parser.add_argument("--no-console", action="store_true")
    parser.add_argument("--color-switch", action="store_true", help="模拟 RC/BC 逐段切换、快捷键及丢弃重录，不连接真实硬件")
    args = parser.parse_args()
    if args.color_switch and args.live_cameras:
        parser.error("选色自检只使用模拟相机")
    ports = [SimpleNamespace(vid=0x1A86, pid=0x55D3, serial_number=s, device=p) for s, p in [("5C82108314", "SIM_LEADER"), ("5C82108242", "SIM_FOLLOWER")]]
    with patch.object(so101_follower, "SO101Follower", SimulatedArm), patch.object(so101_leader, "SO101Leader", SimulatedArm), patch.object(list_ports, "comports", return_value=ports):
        window = tk.Tk()
        window.withdraw()
        app = RecordingWindow(window)
        assert not app.auto_switch.get()
        app.auto_switch.set(args.color_switch)
        app.path = Path(tempfile.mkdtemp(prefix="recording_ui_check_", dir=ROOT / "运行环境/临时文件")) / "dataset"
        attempts = [("RC", True), ("BC", False), ("BC", True), ("RC", True)] if args.color_switch else [("RC", True)]
        images = {key: np.zeros((spec["height"], spec["width"], 3), dtype=np.uint8)
                  for key, spec in CAMERAS.items()}
        for image in images.values():
            image[50:300, 50:300] = 130
        stop = threading.Event()

        def feed():
            while not stop.wait(.02):
                stamp = time.monotonic()
                for key, image in images.items():
                    app.camera_message(key, "frame", (stamp, image))

        if args.live_cameras:
            app.start_camera()
        else:
            threading.Thread(target=feed, daemon=True).start()
        state = {"saved": False, "failure": None, "attempt": 0, "finishing": False}
        target = sum(save for _, save in attempts)

        def press(key, button):
            event = SimpleNamespace(keysym=key)
            app.shortcut_press(event, button)
            app.shortcut_release(event)

        def begin_attempt():
            assert str(app.object_combo["state"]) == "readonly"
            obj, _ = attempts[state["attempt"]]
            assert app.object_id.get() == obj
            assert app.task.get() == TASKS[obj]
            # 即使显示变量被外部误改，开始录制也必须以本段物体生成任务。
            app.task.set("stale task must not be recorded")
            press("F5", app.start_button)
            assert app.busy and app.task.get() == TASKS[obj]
            assert str(app.object_combo["state"]) == "disabled"
            assert app.save_button.instate(("disabled",))
            assert app.discard_button.instate(("disabled",))

        def monitor():
            if app.dataset is not None and app.dataset.episode_buffer.get("size", 0) >= 60:
                press("F6", app.stop_button)
            if app.pending and not app.busy and not state["finishing"]:
                assert str(app.object_combo["state"]) == "disabled"
                assert app.start_button.instate(("disabled",))
                current_obj, _ = attempts[state["attempt"]]
                expected_next = ("BC" if current_obj == "RC" else "RC") if app.auto_switch.get() else current_obj
                assert app.recorded_object == current_obj
                assert app.object_id.get() == expected_next
                assert app.task.get() == TASKS[expected_next]
                state["finishing"] = True
                save = attempts[state["attempt"]][1]
                press("F7" if save else "F8", app.save_button if save else app.discard_button)
            elif state["finishing"] and not app.busy and not app.pending:
                assert app.dataset.num_episodes == sum(save for _, save in attempts[:state["attempt"] + 1])
                current_obj, saved = attempts[state["attempt"]]
                expected_next = ("BC" if current_obj == "RC" else "RC") if saved and app.auto_switch.get() else current_obj
                assert app.object_id.get() == expected_next
                assert app.task.get() == TASKS[expected_next]
                state["attempt"] += 1
                state["finishing"] = False
                if state["attempt"] < len(attempts):
                    begin_attempt()
                    window.after(50, monitor)
                    return
                state["saved"] = True
                assert app.dataset.num_episodes == target
                assert (app.path / "第001段逐帧时间.json").is_file()
                backup = [json.loads(x) for x in app.journal_path.read_text(encoding="utf-8").splitlines()]
                assert len([x for x in backup if x["type"] == "frame"]) == len(app.last_timing)
                assert all(not arm.bus.is_connected for arm in SimulatedArm.instances)
                stop.set()
                app.close()
                return
            window.after(50, monitor)

        def timeout():
            state["failure"] = app.status.get()
            stop.set()
            app.pending = False
            app.close()

        window.after(5000 if args.live_cameras else 300, begin_attempt)
        window.after(700, monitor)
        window.after(120000 if args.color_switch else 60000, timeout)
        window.mainloop()
        assert state["saved"] and not state["failure"], state
        result, _, _ = inspect_dataset(app.path)
        assert result["status"] == "passed", result["errors"]
        assert result["video_keys"] == ["observation.images.front", "observation.images.wrist"]
        assert result["fps"] == 20
        if args.color_switch:
            from 管理训练数据来源 import dataset_source
            _, checked_info, tables = inspect_dataset(app.path)
            texts = {t["task_index"]: t["task"] for t in result["tasks"]}
            expected_objects = [obj for obj, save in attempts if save]
            for ep, obj in enumerate(expected_objects):
                assert {texts[i] for i in tables[ep]["task_index"]} == {TASKS[obj]}
                journal = app.path / "逐帧动作备份" / f"第{ep + 1:03d}段动作备份.jsonl"
                header = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
                assert header["task"] == TASKS[obj] and header["target_object_id"] == obj
                assert header["episode"] == ep and header["collection_label"] is None
            episodes = [json.loads(line) for line in (app.path / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            assert [ep["tasks"] for ep in episodes] == [[TASKS[obj]] for obj in expected_objects]
            assert "collection_labels" not in result
            source = dataset_source(app.path, checked_info, tables, result["tasks"])
            assert len(source["episodes"]) == target
        info = json.loads((app.path / "meta/info.json").read_text(encoding="utf-8"))
        for spec in CAMERAS.values():
            assert info["features"][spec["feature"]]["shape"] == [spec["height"], spec["width"], 3]
        (ROOT / "运行记录" / ("红蓝逐段切换录制自检.json" if args.color_switch else "高清真实相机模拟录制自检.json" if args.live_cameras else "高清模拟录制自检.json")).write_text(
            json.dumps({**result, "scope": "live_cameras_simulated_arms" if args.live_cameras else "simulated_cameras_and_arms",
                        "camera_specs": CAMERAS, "robot_commands_sent": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"采集流程自检通过：{result['total_frames']} 帧，开始/停止/保存/视频重读/串口清理正常。", flush=True)
        print(f"合成测试数据：{app.path}", flush=True)
        if "--no-console" in sys.argv:
            (ROOT / "运行记录/后台窗口保存自检.json").write_text(json.dumps({"status": "passed", "mode": "pythonw/no console", "frames": result["total_frames"], "dataset_root": str(app.path), "action_journal_verified": True}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
