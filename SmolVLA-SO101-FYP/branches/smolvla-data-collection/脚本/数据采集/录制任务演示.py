"""单相机任务演示入口；操作者开始、停止并选择保存每段。"""
import argparse
import json
import os
import queue
import shutil
import hashlib
import threading
import time
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "脚本/程序模块"))
(ROOT / "运行记录").mkdir(exist_ok=True)
# pythonw 没有控制台，第三方进度条和异常输出仍需要可写入的流。
if sys.stdout is None or sys.stderr is None:
    _window_log = (ROOT / "运行记录/演示录制窗口.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _window_log
    if sys.stderr is None:
        sys.stderr = _window_log
os.environ["HF_LEROBOT_CALIBRATION"] = str(ROOT / "标定备份")
os.environ["HF_HOME"] = str(ROOT / "运行环境/缓存/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
from 相机测试窗口 import named_capture
from 检查控制输入 import validate_frame
from 检查与释放电机 import MotorHealthGuard, release_bus
from 管理采集尝试 import CollectionAttempt, JOINTS, joint_values, write_json


class RecordingWindow:
    def __init__(self, window):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.window = tk, window
        self.events = queue.Queue()
        self.lock = threading.Lock()
        self.frame = None
        self.frame_time = 0
        self.valid_frames = 0
        self.camera_stop = threading.Event()
        self.arm_stop = threading.Event()
        self.camera_thread = self.arm_thread = None
        self.dataset = None
        self.attempt = None
        self.sealed = False
        self.episode_labels = {}
        self.busy = self.pending = self.closing = False
        self.path = ROOT.parent / "训练数据采集/机械臂控制项目/任务演示" / ("单相机演示_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        window.title("录制任务演示 · 单相机")
        window.geometry("1020x850")
        window.protocol("WM_DELETE_WINDOW", self.close)
        ttk.Label(window, text="录制任务演示", font=("Microsoft YaHei UI", 20)).pack(anchor="w", padx=20, pady=12)
        ttk.Label(window, text="开启桌面相机 → 摆好两臂和物体 → 开始一段 → 完成后停止 → 保存或重录").pack(anchor="w", padx=20)
        row = ttk.Frame(window)
        row.pack(fill="x", padx=20, pady=10)
        ttk.Label(row, text="任务指令：").pack(side="left")
        self.task = tk.StringVar(value="Pick up the red cube and place it in the target area.")
        self.entry = ttk.Entry(row, textvariable=self.task, width=75)
        self.entry.pack(side="left", fill="x", expand=True)
        ttk.Label(window, text="默认任务：抓起红色方块，放到固定目标区域。可修改上面的任务指令；同一批数据保持一致。").pack(anchor="w", padx=20)
        ttk.Label(window, text="桌面设备：1080P USB Camera；录制 640×480 / 20 帧每秒，正常速度同步全部六个关节。").pack(anchor="w", padx=20, pady=5)
        labels = ttk.Frame(window)
        labels.pack(fill="x", padx=20, pady=5)
        self.split = tk.StringVar(value="训练")
        self.kind = tk.StringVar(value="普通演示")
        self.scene = tk.StringVar(value="中央，正向，固定目标区")
        ttk.Label(labels, text="本批用途：").pack(side="left")
        self.split_box = ttk.Combobox(labels, textvariable=self.split, values=["训练", "独立验证"], state="readonly", width=10)
        self.split_box.pack(side="left")
        ttk.Label(labels, text="本段类型：").pack(side="left", padx=8)
        self.kind_box = ttk.Combobox(labels, textvariable=self.kind, values=["普通演示", "纠偏演示"], state="readonly", width=10)
        self.kind_box.pack(side="left")
        ttk.Label(labels, text="场景：").pack(side="left", padx=8)
        self.scene_entry = ttk.Entry(labels, textvariable=self.scene, width=35)
        self.scene_entry.pack(side="left", fill="x", expand=True)
        controls = ttk.Frame(window)
        controls.pack(fill="x", padx=20, pady=8)
        self.camera_button = ttk.Button(controls, text="开启桌面相机", command=self.start_camera)
        self.camera_button.pack(side="left")
        self.start_button = ttk.Button(controls, text="开始一段（最长 60 秒）", command=self.start_episode)
        self.start_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls, text="停止本段 / 空格", command=self.arm_stop.set)
        self.stop_button.pack(side="left")
        self.save_button = ttk.Button(controls, text="保存成功演示", command=lambda: self.finish_episode(True))
        self.save_button.pack(side="left", padx=8)
        self.discard_button = ttk.Button(controls, text="归档失败并重录", command=lambda: self.finish_episode(False))
        self.discard_button.pack(side="left")
        self.status = tk.StringVar(value="未开启相机和机械臂。先阅读同目录采集指引；开始前将两臂摆成相近姿态；停止后释放扭矩，请托好机械臂。")
        ttk.Label(window, textvariable=self.status, wraplength=880).pack(anchor="w", padx=20, pady=8)
        self.preview = tk.Label(window, bg="#20252d", fg="white", text="等待开启桌面相机")
        self.preview.pack(fill="both", expand=True, padx=20, pady=8)
        self.counter = tk.StringVar(value="已保存 0 段；先试录检查，再按指引采集 30 段训练和 10 段独立验证。")
        ttk.Label(window, textvariable=self.counter).pack(anchor="w", padx=20)
        ttk.Label(window, text=f"本批数据目录：{self.path}", wraplength=880).pack(anchor="w", padx=20, pady=8)
        window.bind("<space>", self.stop_key)
        window.bind("<Escape>", lambda _: self.arm_stop.set())
        self.refresh_controls()
        window.after(60, self.poll)

    def stop_key(self, event):
        if event.widget != self.entry:
            self.arm_stop.set()
            return "break"

    def refresh_controls(self):
        self.start_button.configure(state="normal" if not self.busy and not self.pending and not self.closing and not self.sealed else "disabled")
        self.stop_button.configure(state="normal" if self.busy else "disabled")
        for button in (self.save_button, self.discard_button):
            button.configure(state="normal" if self.pending and not self.busy else "disabled")
        self.split_box.configure(state="disabled" if self.dataset is not None or self.busy else "readonly")
        self.kind_box.configure(state="disabled" if self.busy or self.pending else "readonly")
        self.scene_entry.configure(state="disabled" if self.busy or self.pending else "normal")
        self.entry.configure(state="disabled" if self.busy or self.pending or self.dataset is not None else "normal")

    def camera_message(self, kind, value):
        if kind == "frame":
            stamp, frame = value
            with self.lock:
                self.frame, self.frame_time = frame, stamp
                self.valid_frames += 1 if frame.std() > 1 else 0
        elif kind == "error":
            self.events.put(("camera_error", value))

    def start_camera(self):
        if self.camera_thread and self.camera_thread.is_alive():
            return
        self.camera_stop.clear()
        def capture():
            try:
                named_capture(1, "MJPG", 25, self.camera_message, self.camera_stop, "1080P USB Camera", None)
            except Exception as exc:
                self.events.put(("camera_error", str(exc)))
        self.camera_thread = threading.Thread(target=capture, daemon=True)
        self.camera_thread.start()
        self.camera_button.configure(state="disabled")
        self.status.set("正在开启桌面相机，请确认画面包含机械臂、方块和目标区域。")

    def start_episode(self):
        if self.busy or self.pending or self.closing or self.sealed:
            return
        with self.lock:
            ready = self.frame is not None and self.valid_frames >= 5 and time.monotonic() - self.frame_time < .2 and self.frame.std() > 1
        if not ready:
            self.status.set("请先开启桌面相机，确认正常画面后再开始。")
            return
        task = self.task.get().strip()
        if not task:
            self.status.set("请填写本次任务指令。")
            return
        if not self.scene.get().strip():
            self.status.set("请填写本段场景，例如：偏左、旋转 45 度、固定目标区。")
            return
        self.episode_labels = {"scene": self.scene.get().strip(), "kind": "recovery" if self.kind.get() == "纠偏演示" else "ordinary",
                               "split": "held_out" if self.split.get() == "独立验证" else "train"}
        if shutil.disk_usage(ROOT.parent).free < 2 * 1024**3:
            self.status.set("磁盘剩余不足 2 GB，请先腾出空间；未启动机械臂。")
            return
        self.busy = True
        self.arm_stop.clear()
        self.refresh_controls()
        self.arm_thread = threading.Thread(target=self.record_episode, args=(task,), daemon=True)
        self.arm_thread.start()

    def record_episode(self, task):
        import numpy as np
        from serial.tools import list_ports
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
        from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
        arms, count, timing, failure, journal = [], 0, [], None, None
        self.attempt = None
        try:
            ports = {p.serial_number: p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x1A86, 0x55D3)}
            if "5C82108314" not in ports or "5C82108242" not in ports:
                raise RuntimeError("未检测到两臂 USB，请确认 USB 和电源已接好。")
            leader = SO101Leader(SO101LeaderConfig(port=ports["5C82108314"], id="my_awesome_leader_arm"))
            follower = SO101Follower(SO101FollowerConfig(port=ports["5C82108242"], id="my_awesome_follower_arm", max_relative_target=None))
            arms = [follower, leader]
            leader.bus.connect()
            follower.bus.connect()
            if not leader.is_calibrated or not follower.is_calibrated:
                raise RuntimeError("当前标定不匹配，已停止；请先核对标定。")
            names = list(follower.action_features)
            if names != JOINTS:
                raise RuntimeError("关节顺序与模型固定接口不一致")
            guards = [MotorHealthGuard(arm.bus) for arm in arms]
            for guard in guards:
                guard.check(force=True)
            if self.dataset is None:
                names = list(follower.action_features)
                features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": names}, "action": {"dtype": "float32", "shape": (6,), "names": names}, "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}}
                self.dataset = LeRobotDataset.create("local/" + self.path.name, fps=20, features=features, root=self.path, robot_type=follower.name, use_videos=True, image_writer_threads=2, video_backend="pyav")
                write_json(self.path / "采集配置.json", {
                    "schema_version": 2, "task": task, "camera": "1080P USB Camera", "resolution": [640, 480],
                    "control_fps": 60, "record_fps": 20, "joint_names": JOINTS,
                    "joint_units": "LeRobot SO101 normalized position", "leader_serial": "5C82108314",
                    "follower_serial": "5C82108242", "split": self.episode_labels["split"],
                    "session_id": self.path.name, "push_to_hub": False,
                    "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            self.attempt = CollectionAttempt(self.path, self.dataset.num_episodes, dict(self.episode_labels), task)
            self.journal_path = self.attempt.journal
            journal = self.journal_path.open("x", encoding="utf-8", buffering=1)
            journal.write(json.dumps({"type": "header", "task": task, "fps": 20, "episode": self.dataset.num_episodes,
                                      "joint_names": JOINTS, "labels": self.episode_labels}) + "\n")
            # 倒计时在启用从臂扭矩前；取消不会启动同步。
            for i in range(3, 0, -1):
                self.events.put(("status", f"{i} 秒后启用同步和录制，请准备移动主臂。"))
                if self.arm_stop.wait(1):
                    return
            initial_state = joint_values(follower.get_observation())
            initial_action = joint_values(leader.get_action())
            if max(abs(a-b) for a,b in zip(initial_state, initial_action)) > 30:
                raise RuntimeError("两臂起始姿态相差超过 30 标定单位；请托好两臂、摆成相近姿态")
            for guard in guards:
                guard.check(force=True)
            with self.lock:
                validate_frame(self.frame_time, self.frame, time.monotonic(), max_age=.2)
            if self.arm_stop.is_set():
                return
            leader.configure()
            follower.bus.disable_torque()
            follower.bus.sync_write("Goal_Position", follower.bus.sync_read("Present_Position"))
            follower.configure()
            torque = follower.bus.sync_read("Torque_Enable", normalize=False)
            if len(torque) != 6 or any(v != 1 for v in torque.values()):
                raise RuntimeError("六个电机没有全部启用，已停止。")
            start, next_frame = time.monotonic(), 0.0
            names = list(follower.action_features)
            while not self.arm_stop.is_set() and time.monotonic() - start < 60:
                tick = time.monotonic()
                with self.lock:
                    stamp, image = self.frame_time, self.frame
                validate_frame(stamp, image, time.monotonic(), max_age=.2)
                if image.shape != (480, 640, 3):
                    raise RuntimeError("相机采集分辨率变化，停止录制")
                image = image.copy()
                for guard in guards:
                    guard.check()
                observation = follower.get_observation()
                state_time = time.monotonic()
                state_values = joint_values(observation)
                action = leader.get_action()
                joint_values(action)
                validate_frame(stamp, image, time.monotonic(), max_age=.2)
                if self.arm_stop.is_set():
                    break
                sent = follower.send_action(action)
                action_time = time.monotonic()
                action_values = joint_values(sent)
                elapsed = tick - start
                if elapsed >= next_frame:
                    if elapsed - next_frame > .1:
                        raise RuntimeError("采集延迟过大，本段不能保存，请重录。")
                    self.dataset.add_frame({"observation.state": np.array(state_values, dtype=np.float32), "action": np.array(action_values, dtype=np.float32), "observation.images.front": image[:, :, ::-1].copy()}, task=task)
                    image_path = self.dataset._get_image_file_path(self.dataset.num_episodes, "observation.images.front", count)
                    journal.write(json.dumps({"type": "frame", "frame_index": count, "timestamp": count / 20, "state": state_values, "action": action_values, "image_path": str(image_path.relative_to(self.path)), "elapsed_s": elapsed, "camera_age_s": action_time - stamp, "camera_elapsed_s": stamp-start, "state_elapsed_s": state_time-start, "action_elapsed_s": action_time-start}) + "\n")
                    timing.append({"elapsed_s": elapsed, "camera_age_s": action_time - stamp, "camera_elapsed_s": stamp-start, "state_elapsed_s": state_time-start, "action_elapsed_s": action_time-start})
                    count += 1
                    next_frame = count / 20
                    if count % 20 == 0:
                        self.events.put(("status", f"正在录制：{count / 20:.1f} 秒，{count} 帧。完成后点击停止或按空格。"))
                self.arm_stop.wait(max(0, 1 / 60 - (time.monotonic() - tick)))
        except Exception as exc:
            failure = str(exc)
            traceback.print_exc()
        finally:
            if journal is not None:
                try:
                    journal.flush()
                    os.fsync(journal.fileno())
                except Exception as exc:
                    failure = (failure or "") + f"；动作备份写入异常：{exc}"
                finally:
                    journal.close()
            cleanup_errors = []
            for arm in arms:
                cleanup_errors.extend(release_bus(arm.bus))
            if cleanup_errors:
                failure = (failure or "") + "；停扭矩未完全确认：" + "；".join(cleanup_errors)
                self.seal(failure)
            self.last_timing = timing
            if failure or count < 50 or self.closing:
                reason = failure or ("关闭窗口，中断采集" if self.closing else "本段不足 2.5 秒，请录制完整任务")
                try:
                    if self.dataset is not None and self.attempt is not None:
                        self.attempt.archive_and_reset(self.dataset, "failed" if failure else "interrupted", reason)
                except Exception as exc:
                    reason += "；归档未完成：" + str(exc)
                    self.seal(reason)
                self.events.put(("finished", (False, reason + "；已保留可用原始文件。")))
            else:
                try:
                    self.attempt.set_status("pending", frames=count, cleanup_errors=cleanup_errors)
                except Exception as exc:
                    self.seal(str(exc))
                    self.events.put(("finished", (False, "采集状态写入失败，保留原始帧并封存：" + str(exc))))
                    return
                self.events.put(("finished", (True, f"本段 {count} 帧；已逐电机确认停扭矩。任务最终成功可保存；空抓或未放好请归档重录。")))

    def seal(self, reason):
        self.sealed = True
        if self.path.exists():
            try:
                write_json(self.path / "批次封存状态.json", {"reason": reason, "next_step": "关闭窗口，检查原始文件和设备；重新打开会创建新批次，不覆盖本批次"})
            except OSError:
                # 磁盘已满时也必须在内存中封存并允许关闭，不能继续重试写入。
                traceback.print_exc()


    def finish_episode(self, save):
        if not self.pending or self.busy:
            return
        self.busy = True
        self.refresh_controls()
        self.status.set("正在归档原始数据并编码视频…" if save else "正在保留失败原始数据，完成后可重录…")
        def finish():
            try:
                ep = self.dataset.num_episodes
                self.attempt.archive(self.dataset)
                if save:
                    self.dataset.save_episode()
                    from 检查演示数据 import inspect_dataset
                    result, _, _ = inspect_dataset(self.path)
                    if result["status"] != "passed":
                        raise RuntimeError("保存后完整性检查未通过：" + "；".join(result["errors"]))
                    write_json(self.path / f"第{ep+1:03d}段逐帧时间.json", self.last_timing)
                    self.attempt.set_status("saved", human_success=True, quality_check="passed")
                else:
                    self.attempt.set_status("rejected", reason="操作者选择归档重录，未纳入训练", human_success=False)
                    self.dataset.clear_episode_buffer()
                self.events.put(("saved", None))
            except Exception as exc:
                traceback.print_exc()
                try:
                    self.attempt.set_status("save_failed", reason=str(exc), human_success=False)
                except OSError:
                    traceback.print_exc()
                self.seal(str(exc))
                self.events.put(("save_failed", str(exc)))
        self.arm_thread = threading.Thread(target=finish, daemon=True)
        self.arm_thread.start()

    def poll(self):
        from PIL import Image, ImageTk
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == "status":
                self.status.set(value)
            elif kind == "camera_error":
                self.status.set(value)
                self.arm_stop.set()
                self.camera_button.configure(state="normal")
            elif kind == "finished":
                self.pending, message = value
                self.busy = False
                self.status.set(message)
            elif kind == "saved":
                self.pending = self.busy = False
                self.status.set("本段已处理。重新摆好物体和两臂后，再点击开始一段。")
                self.counter.set(f"已保存 {self.dataset.num_episodes} 段；数据仅保存在本机。")
            elif kind == "save_failed":
                self.status.set("保存失败，已保留本段原始文件，请先检查：" + value)
                self.pending = False
                self.busy = False
        self.refresh_controls()
        with self.lock:
            frame, stamp = self.frame, self.frame_time
        if frame is not None and time.monotonic() - stamp < 1:
            image = Image.fromarray(frame[:, :, ::-1])
            image.thumbnail((860, 480))
            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo, text="")
        if self.closing and not (self.arm_thread and self.arm_thread.is_alive()):
            if self.dataset is not None:
                if self.pending and not self.sealed and self.dataset.episode_buffer and self.dataset.episode_buffer.get("size", 0):
                    try:
                        self.attempt.archive_and_reset(self.dataset, "interrupted", "关闭未保存窗口")
                    except Exception as exc:
                        self.seal(str(exc))
                self.dataset.stop_image_writer()
            if self.camera_thread and self.camera_thread.is_alive():
                self.window.after(60, self.poll)
                return
            self.window.destroy()
            return
        self.window.after(60, self.poll)

    def close(self):
        if self.closing:
            return
        if self.pending and not self.busy:
            self.finish_episode(False)
        self.closing = True
        self.arm_stop.set()
        self.camera_stop.set()
        self.status.set("正在停止机械臂、释放相机并关闭窗口…")
        self.refresh_controls()


if __name__ == "__main__":
    import tkinter as tk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    window = tk.Tk()
    app = RecordingWindow(window)
    if args.smoke_test:
        window.after(300, app.close)
    window.mainloop()
