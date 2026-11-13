"""双相机任务演示入口；操作者开始、停止并选择保存每段。"""
import argparse
import json
import os
import queue
import threading
import time
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
from 双摄采集配置 import CAMERAS, RECORD_FPS
from 红蓝抓取任务 import TASKS


class RecordingWindow:
    def __init__(self, window):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.window = tk, window
        self.events = queue.Queue()
        self.lock = threading.Lock()
        self.frames = {key: {"image": None, "time": 0.0, "valid": 0} for key in CAMERAS}
        self.camera_stop = threading.Event()
        self.arm_stop = threading.Event()
        self.camera_threads = {}
        self.arm_thread = None
        self.dataset = None
        self.recorded_object = None
        self.busy = self.pending = self.closing = False
        self.path = ROOT.parent / "训练数据采集/机械臂控制项目/任务演示" / ("双相机演示_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        window.title("录制任务演示 · 双相机")
        window.geometry("1280x900")
        window.protocol("WM_DELETE_WINDOW", self.close)
        ttk.Label(window, text="录制任务演示", font=("Microsoft YaHei UI", 20)).pack(anchor="w", padx=20, pady=12)
        ttk.Label(window, text="开启双路相机 → 确认桌面和夹爪画面 → 摆好两臂和物体 → 开始一段 → 完成后停止 → 保存或重录").pack(anchor="w", padx=20)
        row = ttk.Frame(window)
        row.pack(fill="x", padx=20, pady=10)
        ttk.Label(row, text="任务指令：").pack(side="left")
        self.task = tk.StringVar(value=TASKS["RC"])
        self.entry = ttk.Entry(row, textvariable=self.task, width=75, state="readonly")
        self.entry.pack(side="left", fill="x", expand=True)
        options = ttk.Frame(window)
        options.pack(fill="x", padx=20, pady=4)
        self.object_id = tk.StringVar(value="RC")
        ttk.Label(options, text="本段抓取物体：").pack(side="left")
        self.object_combo = ttk.Combobox(options, textvariable=self.object_id,
                                        values=tuple(TASKS), width=10, state="readonly")
        self.object_combo.pack(side="left", padx=(0, 12))
        self.object_combo.bind("<<ComboboxSelected>>", self.update_task)
        self.auto_switch = tk.BooleanVar(value=False)
        self.auto_switch_check = ttk.Checkbutton(options, text="录完一段自动切换 RC/BC", variable=self.auto_switch)
        self.auto_switch_check.pack(side="left", padx=(0, 12))
        ttk.Label(options, text="RC＝红方块；BC＝蓝方块。勾选才切换；丢弃后恢复本段目标。").pack(side="left")
        specs_text = "；".join(f"{spec['label']} {spec['width']}×{spec['height']}" for spec in CAMERAS.values())
        ttk.Label(window, text=f"{specs_text}。两路统一保存 {RECORD_FPS} FPS，同步全部六个关节。").pack(anchor="w", padx=20, pady=5)
        controls = ttk.Frame(window)
        controls.pack(fill="x", padx=20, pady=8)
        self.camera_button = ttk.Button(controls, text="开启双路相机", command=self.start_camera)
        self.camera_button.pack(side="left")
        self.start_button = ttk.Button(controls, text="开始一段 [F5]（最长 30 秒）", command=self.start_episode)
        self.start_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls, text="停止本段 [F6 / 空格]", command=self.arm_stop.set)
        self.stop_button.pack(side="left")
        self.save_button = ttk.Button(controls, text="保存成功演示 [F7]", command=lambda: self.finish_episode(True))
        self.save_button.pack(side="left", padx=8)
        self.discard_button = ttk.Button(controls, text="丢弃本段并重录 [F8]", command=lambda: self.finish_episode(False))
        self.discard_button.pack(side="left")
        self.status = tk.StringVar(value="未开启相机和机械臂。开始前将两臂摆成相近姿态；停止后释放扭矩，请托好机械臂。")
        ttk.Label(window, textvariable=self.status, wraplength=880).pack(anchor="w", padx=20, pady=8)
        previews = ttk.Frame(window)
        previews.pack(fill="both", expand=True, padx=14, pady=8)
        self.previews = {}
        for column, key in enumerate(CAMERAS):
            panel = ttk.LabelFrame(previews, text=CAMERAS[key]["label"], padding=6)
            panel.grid(row=0, column=column, sticky="nsew", padx=6)
            previews.columnconfigure(column, weight=1, uniform="camera")
            label = tk.Label(panel, bg="#20252d", fg="white", text="等待开启相机")
            label.pack(fill="both", expand=True)
            self.previews[key] = {"label": label, "photo": None}
        previews.rowconfigure(0, weight=1)
        self.counter = tk.StringVar(value="已保存 0 段；建议先录 3 段并检查，再逐步积累。")
        ttk.Label(window, textvariable=self.counter).pack(anchor="w", padx=20)
        ttk.Label(window, text=f"本批数据目录：{self.path}", wraplength=880).pack(anchor="w", padx=20, pady=8)
        window.bind("<space>", self.stop_key)
        window.bind("<Escape>", lambda _: self.arm_stop.set())
        self.shortcut_keys_down = set()
        for key, button in (("F5", self.start_button), ("F6", self.stop_button),
                            ("F7", self.save_button), ("F8", self.discard_button)):
            window.bind(f"<KeyPress-{key}>", lambda event, b=button: self.shortcut_press(event, b))
            window.bind(f"<KeyRelease-{key}>", self.shortcut_release)
        self.refresh_controls()
        window.after(60, self.poll)

    def stop_key(self, event):
        if event.widget not in (self.entry, self.object_combo):
            self.arm_stop.set()
            return "break"

    def shortcut_press(self, event, button):
        # 长按只触发一次；复用按钮的启用状态，防止保存/编码时开始下一段。
        if event.keysym not in self.shortcut_keys_down:
            self.shortcut_keys_down.add(event.keysym)
            if not self.closing and button.instate(("!disabled",)):
                button.invoke()
        return "break"

    def shortcut_release(self, event):
        self.shortcut_keys_down.discard(event.keysym)
        return "break"

    def refresh_controls(self):
        self.start_button.configure(state="normal" if not self.busy and not self.pending and not self.closing else "disabled")
        self.stop_button.configure(state="normal" if self.busy else "disabled")
        for button in (self.save_button, self.discard_button):
            button.configure(state="normal" if self.pending and not self.busy else "disabled")
        locked = self.busy or self.pending or self.closing
        self.entry.configure(state="readonly")
        self.object_combo.configure(state="disabled" if locked else "readonly")
        self.auto_switch_check.configure(state="disabled" if locked else "normal")

    def update_task(self, event=None):
        self.task.set(TASKS[self.object_id.get()])
        self.refresh_controls()

    def switch_target_object(self):
        self.object_id.set("BC" if self.object_id.get() == "RC" else "RC")
        self.update_task()

    def camera_message(self, camera, kind, value):
        if kind == "frame":
            stamp, frame = value
            with self.lock:
                state = self.frames[camera]
                state["image"], state["time"] = frame, stamp
                state["valid"] += 1 if frame[::8, ::8].std() > 1 else 0
        elif kind == "error":
            self.events.put(("camera_error", f"{CAMERAS[camera]['label']}：{value}"))

    def start_camera(self):
        if any(thread.is_alive() for thread in self.camera_threads.values()):
            return
        self.camera_stop.clear()
        def launch(key):
            if self.closing or self.camera_stop.is_set():
                return
            spec = CAMERAS[key]
            def capture():
                try:
                    named_capture(spec["index"], spec["fourcc"], spec["input_fps"],
                                  lambda kind, value: self.camera_message(key, kind, value),
                                  self.camera_stop, spec["device"], None,
                                  width=spec["width"], height=spec["height"])
                except Exception as exc:
                    self.events.put(("camera_error", f"{spec['label']}：{exc}"))
            thread = threading.Thread(target=capture, daemon=True, name=spec["label"])
            self.camera_threads[key] = thread
            thread.start()
        launch("front")
        self.window.after(1200, lambda: launch("wrist"))
        self.camera_button.configure(state="disabled")
        self.status.set("正在依次开启桌面和夹爪相机；两路都有实时画面后才能开始。")

    def start_episode(self):
        if self.busy or self.pending or self.closing:
            return
        if self.object_id.get() not in TASKS:
            self.status.set("请选择本段抓取物体：RC 或 BC。")
            return
        with self.lock:
            now = time.monotonic()
            ready = all(state["image"] is not None and state["valid"] >= 5 and
                        now - state["time"] < .25 and state["image"].std() > 1
                        for state in self.frames.values())
        if not ready:
            self.status.set("请先开启双路相机，确认桌面和夹爪画面都正常后再开始。")
            return
        # 开始时从目标生成并快照指令；后续切换不会改写已录段的任务。
        task = TASKS[self.object_id.get()]
        self.task.set(task)
        self.recorded_object = self.object_id.get()
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
            if self.dataset is None:
                names = list(follower.action_features)
                features = {
                    "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
                    "action": {"dtype": "float32", "shape": (6,), "names": names},
                    **{spec["feature"]: {"dtype": "video", "shape": (spec["height"], spec["width"], 3),
                                           "names": ["height", "width", "channels"]}
                       for spec in CAMERAS.values()},
                }
                self.dataset = LeRobotDataset.create("local/" + self.path.name, fps=RECORD_FPS, features=features,
                                                     root=self.path, robot_type=follower.name, use_videos=True,
                                                     image_writer_threads=4, video_backend="pyav")
                (self.path / "采集配置.json").write_text(json.dumps({
                    "task": task,
                    "task_scope": "per_episode",
                    "collection_label_schema": None,
                    "cameras": {key: {"feature": spec["feature"], "device": spec["device"],
                                       "capture_fps": spec["input_fps"], "format": spec["fourcc"],
                                       "width": spec["width"], "height": spec["height"],
                                       "record_fps": RECORD_FPS}
                                for key, spec in CAMERAS.items()},
                    "control_fps": 60, "record_fps": RECORD_FPS,
                    "joint_units": "LeRobot SO101 normalized position",
                    "leader_serial": "5C82108314", "follower_serial": "5C82108242", "push_to_hub": False,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            backup_dir = self.path / "逐帧动作备份"
            backup_dir.mkdir(exist_ok=True)
            self.journal_path = backup_dir / f"第{self.dataset.num_episodes+1:03d}段动作备份.jsonl"
            journal = self.journal_path.open("w", encoding="utf-8", buffering=1)
            journal.write(json.dumps({"type": "header", "task": task, "fps": RECORD_FPS,
                                      "collection_label": None,
                                      "target_object_id": next(obj for obj, text in TASKS.items() if text == task),
                                      "episode": self.dataset.num_episodes,
                                      "video_keys": [spec["feature"] for spec in CAMERAS.values()]}) + "\n")
            leader.configure()
            follower.bus.disable_torque()
            follower.bus.sync_write("Goal_Position", follower.bus.sync_read("Present_Position"))
            follower.configure()
            torque = follower.bus.sync_read("Torque_Enable", normalize=False)
            if len(torque) != 6 or any(v != 1 for v in torque.values()):
                raise RuntimeError("六个电机没有全部启用，已停止。")
            for i in range(3, 0, -1):
                self.events.put(("status", f"{i} 秒后开始同步和录制，请准备移动主臂。"))
                if self.arm_stop.wait(1):
                    return
            start, next_frame = time.monotonic(), 0.0
            names = list(follower.action_features)
            while not self.arm_stop.is_set() and time.monotonic() - start < 30:
                tick = time.monotonic()
                with self.lock:
                    snapshots = {key: (state["time"], state["image"]) for key, state in self.frames.items()}
                for key, (stamp, image) in snapshots.items():
                    if image is None or tick - stamp > .25:
                        raise RuntimeError(f"{CAMERAS[key]['label']}画面中断，本段不能保存，请检查相机后重录。")
                    if image.mean() < 3 and image.std() < 1:
                        raise RuntimeError(f"{CAMERAS[key]['label']}返回黑帧，本段不能保存，请检查相机后重录。")
                skew = max(stamp for stamp, _ in snapshots.values()) - min(stamp for stamp, _ in snapshots.values())
                if skew > .15:
                    raise RuntimeError(f"双路相机不同步（相差 {skew:.3f} 秒），本段不能保存。")
                observation = follower.get_observation()
                action = leader.get_action()
                sent = follower.send_action(action)
                elapsed = tick - start
                if elapsed >= next_frame:
                    if elapsed - next_frame > .1:
                        raise RuntimeError("采集延迟过大，本段不能保存，请重录。")
                    state_values = [observation[k] for k in names]
                    action_values = [sent[k] for k in names]
                    frame = {"observation.state": np.array(state_values, dtype=np.float32),
                             "action": np.array(action_values, dtype=np.float32)}
                    image_paths, camera_ages = {}, {}
                    for key, spec in CAMERAS.items():
                        stamp, image = snapshots[key]
                        frame[spec["feature"]] = image[:, :, ::-1].copy()
                        image_path = self.dataset._get_image_file_path(self.dataset.num_episodes, spec["feature"], count)
                        image_paths[key] = str(image_path.relative_to(self.path))
                        camera_ages[key] = tick - stamp
                    self.dataset.add_frame(frame, task=task)
                    journal.write(json.dumps({"type": "frame", "frame_index": count,
                                              "timestamp": count / RECORD_FPS, "state": state_values,
                                              "action": action_values, "image_paths": image_paths,
                                              "elapsed_s": elapsed, "camera_age_s": camera_ages,
                                              "camera_skew_s": skew}) + "\n")
                    timing.append({"elapsed_s": elapsed, "camera_age_s": camera_ages, "camera_skew_s": skew})
                    count += 1
                    next_frame = count / RECORD_FPS
                    if count % RECORD_FPS == 0:
                        self.events.put(("status", f"双摄正在录制：{count / RECORD_FPS:.1f} 秒，{count} 帧。完成后点击停止或按空格。"))
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
            for arm in arms:
                if arm.bus.is_connected:
                    try:
                        arm.bus.disconnect(disable_torque=True)
                    except Exception as exc:
                        failure = (failure or "") + f"；关闭机械臂异常：{exc}"
                        if getattr(arm.bus.port_handler, "ser", None):
                            arm.bus.port_handler.closePort()
            if self.dataset is not None and count:
                self.last_timing = timing
            if failure or count < 50:
                if self.dataset is not None:
                    self.dataset._wait_image_writer()
                    self.dataset.clear_episode_buffer()
                self.events.put(("finished", (False, failure or "本段不足 2.5 秒，已丢弃；请录制完整任务。")))
            else:
                self.events.put(("finished", (True, f"本段 {count} 帧，机械臂已停止并释放扭矩。成功完成任务就保存，否则丢弃重录。")))

    def finish_episode(self, save):
        if not self.pending or self.busy:
            return
        self.busy = True
        self.refresh_controls()
        self.status.set("正在保存和编码视频…" if save else "正在丢弃本段…")
        def finish():
            try:
                ep = self.dataset.num_episodes
                if save:
                    self.dataset.save_episode()
                    (self.path / f"第{ep+1:03d}段逐帧时间.json").write_text(json.dumps(self.last_timing), encoding="utf-8")
                else:
                    self.dataset._wait_image_writer()
                    self.dataset.clear_episode_buffer()
                self.events.put(("saved", (save, self.recorded_object)))
            except Exception as exc:
                # 保留原始图片和数据，禁止继续添加或覆盖失败的本段。
                traceback.print_exc()
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
                self.camera_stop.set()
                self.camera_button.configure(state="normal")
            elif kind == "finished":
                self.pending, message = value
                self.busy = False
                if self.pending:
                    previous = self.recorded_object
                    if self.auto_switch.get():
                        self.switch_target_object()
                        self.status.set(f"{message} 下一段已预选 {self.object_id.get()}；若本段要丢弃，按 F8 会恢复 {previous}。")
                    else:
                        self.status.set(f"{message} 自动切换未开启；下一段仍为 {self.object_id.get()}。")
                else:
                    self.recorded_object = None
                    self.status.set(message)
            elif kind == "saved":
                saved, recorded_object = value
                self.pending = self.busy = False
                if saved:
                    self.status.set(f"{recorded_object} 已保存；下一段目标为 {self.object_id.get()}。重新摆好物体和两臂后再开始。")
                else:
                    self.object_id.set(recorded_object)
                    self.update_task()
                    self.status.set(f"本段已丢弃；已恢复 {recorded_object}，请重新摆好物体和两臂后重录。")
                self.recorded_object = None
                self.counter.set(f"已保存 {self.dataset.num_episodes} 段；数据仅保存在本机。")
            elif kind == "save_failed":
                self.status.set("保存失败，已保留本段原始文件，请先检查：" + value)
                self.pending = False
                self.busy = True
        self.refresh_controls()
        with self.lock:
            snapshots = {key: (state["image"], state["time"]) for key, state in self.frames.items()}
        for key, (frame, stamp) in snapshots.items():
            if frame is not None and time.monotonic() - stamp < 1:
                image = Image.fromarray(frame[:, :, ::-1])
                image.thumbnail((540, 405))
                preview = self.previews[key]
                preview["photo"] = ImageTk.PhotoImage(image)
                preview["label"].configure(image=preview["photo"], text="")
        if self.closing and not (self.arm_thread and self.arm_thread.is_alive()):
            if self.dataset is not None:
                if self.pending and self.dataset.episode_buffer and self.dataset.episode_buffer.get("size", 0):
                    self.dataset._wait_image_writer()
                    self.dataset.clear_episode_buffer()
                self.dataset.stop_image_writer()
            if any(thread.is_alive() for thread in self.camera_threads.values()):
                self.window.after(60, self.poll)
                return
            self.window.destroy()
            return
        self.window.after(60, self.poll)

    def close(self):
        if self.closing:
            return
        if self.pending:
            from tkinter import messagebox
            if not messagebox.askyesno("尚未保存", "关闭会丢弃尚未保存的本段，是否关闭？"):
                return
        self.closing = True
        self.arm_stop.set()
        self.camera_stop.set()
        self.status.set("正在停止机械臂、释放相机并关闭窗口…")
        self.refresh_controls()


if __name__ == "__main__":
    import tkinter as tk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--target-object", choices=tuple(TASKS), default="RC",
                        help="新批次的初始抓取目标，RC 为红方块，BC 为蓝方块")
    args = parser.parse_args()
    window = tk.Tk()
    app = RecordingWindow(window)
    app.object_id.set(args.target_object)
    app.update_task()
    if args.smoke_test:
        window.after(300, app.close)
    window.mainloop()
