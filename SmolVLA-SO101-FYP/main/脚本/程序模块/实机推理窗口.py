"""选模型后点击启动，子进程执行真实推理；窗口负责状态、停止和退出等待。"""
import argparse
import json
from io import BytesIO
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import zipfile
from tkinter import messagebox, ttk
from datetime import datetime

from 选择推理模型 import (ROOT, ModelSelectionWindow, validate_selection,
                         recent_selection, remember_selection)
from 推理运行设置 import task_inference_settings
from 推理过程记录 import PREFIX, JOINT_LABELS, format_progress, export_run_report
from 红蓝抓取任务 import TASKS
from 语音指挥控件 import VoiceControls


class InferenceWindow(ModelSelectionWindow):
    def __init__(self, window, root=ROOT, model=None, dataset=None, processor=None, process_factory=None,
                 target_object="RC"):
        restored = recent_selection(root) if model is None else None
        if restored is not None:
            model = restored.model
            dataset = dataset or restored.dataset
            processor = processor or restored.processor
        super().__init__(window, root, model, dataset, processor)
        window.title("模型推理 · 控制从臂")
        window.geometry(f"{min(1440, window.winfo_screenwidth() - 60)}x{min(960, window.winfo_screenheight() - 80)}")
        window.minsize(1000, 760)
        self.content.pack_forget()
        self.content.configure(padding=12)
        self.content.grid(row=0, column=0, sticky="nsew")
        window.columnconfigure(0, weight=3, minsize=560)
        window.columnconfigure(1, weight=2, minsize=360)
        window.rowconfigure(0, weight=1)
        # 模型选择表单留在左上，缩短说明行以适应左右分栏。
        for widget in self.content.winfo_children():
            if widget.winfo_class() == "TLabel":
                widget.configure(wraplength=540)
                if int(widget.grid_info()["row"]) >= 2:
                    widget.grid_configure(pady=(5, 2))
        self.process_factory = process_factory or subprocess.Popen
        self.process = self.reader = self.temporary = None
        self.closing = self.stop_requested = False
        self.cleanup_error = False
        self.failure_text = ""
        self.last_start_error = ""
        self.preview_photo = None
        self.wrist_preview_photo = None
        self.preview_frames = None
        self.action_log_path = None
        self.latest_telemetry = None
        self.messages = queue.Queue()
        self.status = tk.StringVar(value="核对具体检查点后，可说话自动启动，或点击“启动推理”。")
        self.seconds = tk.StringVar(value="30")
        self.unlimited = tk.BooleanVar(value=True)
        self.notice.configure(text="点击启动或说出有效语音目标后，加载模型、开启双摄并控制从臂。相邻目标每步最多变化 0.5，另有限位、跟随误差与卡住停机保护；可随时停止。")
        self.confirm_button.configure(text="启动推理")
        self.cancel_button.configure(text="关闭", command=self.close)
        controls = ttk.Frame(self.content)
        controls_row = self.selection_buttons_row + 1
        task_controls = ttk.Frame(self.content)
        task_controls.grid(row=controls_row, column=0, columnspan=2, sticky="ew", pady=5)
        self.target_object = tk.StringVar(value=target_object)
        self.active_target = target_object
        self.latest_target_request = None
        self.current_target_label = tk.StringVar()
        ttk.Label(task_controls, textvariable=self.current_target_label, font=("Microsoft YaHei", 11, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.switch_button = ttk.Button(task_controls, command=self.toggle_target)
        self.switch_button.grid(row=0, column=1, sticky="w", padx=3)
        self.task_status = tk.StringVar(value="点击按钮直接切换，并清空旧动作。")
        ttk.Label(task_controls, textvariable=self.task_status, wraplength=500).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.show_target()
        controls.grid(row=controls_row + 1, column=0, sticky="ew", pady=8)
        ttk.Label(controls, text="本次运行时长（秒）").pack(side="left")
        self.duration_entry = ttk.Entry(controls, textvariable=self.seconds, width=9)
        self.duration_entry.pack(side="left", padx=8)
        self.unlimited_check = ttk.Checkbutton(controls, text="持续运行，直到点击停止", variable=self.unlimited,
                                              command=self.update_duration_entry)
        self.unlimited_check.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls, text="停止推理", command=self.stop, state="disabled")
        self.stop_button.pack(side="right")
        status_row = controls_row + 2
        ttk.Label(self.content, textvariable=self.status, wraplength=540, font=("Microsoft YaHei", 11)).grid(row=status_row, column=0, columnspan=2, sticky="w", pady=5)
        self.camera_panel = ttk.Frame(window, padding=(0, 12, 12, 12))
        self.camera_panel.grid(row=0, column=1, sticky="nsew")
        self.camera_panel.columnconfigure(0, weight=1)
        self.camera_panel.rowconfigure(1, weight=1, uniform="camera")
        self.camera_panel.rowconfigure(2, weight=1, uniform="camera")
        self.camera_status = tk.StringVar(value="上：桌面相机　|　下：夹爪相机。启动后显示实时画面。")
        ttk.Label(self.camera_panel, textvariable=self.camera_status, wraplength=360).grid(row=0, column=0, sticky="w", pady=(0, 6))
        preview_labels = []
        for row, title in ((1, "桌面相机 · front"), (2, "夹爪相机 · wrist")):
            pane = ttk.LabelFrame(self.camera_panel, text=title, padding=4)
            pane.grid(row=row, column=0, sticky="nsew", pady=(0 if row == 1 else 8, 0))
            pane.grid_propagate(False)
            pane.columnconfigure(0, weight=1)
            pane.rowconfigure(0, weight=1)
            label = ttk.Label(pane, text="等待开启摄像头……", anchor="center")
            label.grid(row=0, column=0, sticky="nsew")
            label.bind("<Configure>", self.render_preview)
            preview_labels.append(label)
        self.preview_label, self.wrist_preview_label = preview_labels
        details_row = status_row + 1
        details = ttk.Frame(self.content)
        details.grid(row=details_row, column=0, columnspan=2, sticky="ew")
        self.live_summary = tk.StringVar(value="等待运行：将显示推理耗时、动作步数、反馈、预测、发送目标与故障码。")
        ttk.Label(details, textvariable=self.live_summary, wraplength=540).pack(fill="x", pady=3)
        self.telemetry_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(details, text="显示关节状态", variable=self.telemetry_visible,
                        command=self.toggle_telemetry).pack(anchor="w")
        output_row = details_row + 1
        self.detail_panes = ttk.Panedwindow(self.content, orient="horizontal")
        self.detail_panes.grid(row=output_row, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        self.telemetry_panel = ttk.LabelFrame(self.detail_panes, text="关节状态", padding=4)
        self.telemetry = ttk.Treeview(self.telemetry_panel, columns=("joint", "observed", "predicted", "sent", "limited", "fault"),
                                     show="headings", height=6)
        for key, label, width in (("joint", "关节", 66), ("observed", "实测", 60), ("predicted", "预测", 60),
                                  ("sent", "发送", 60), ("limited", "限幅", 42), ("fault", "故障码", 52)):
            self.telemetry.heading(key, text=label)
            self.telemetry.column(key, width=width, stretch=True, anchor="center")
        for i, name in enumerate(JOINT_LABELS):
            self.telemetry.insert("", "end", iid=str(i), values=(name, "—", "—", "—", "—", "—"))
        self.telemetry.pack(fill="both", expand=True)
        ttk.Label(self.telemetry_panel, text="归一化位置，非角度；实测为发送前反馈，发送值是指令。", wraplength=330).pack(anchor="w", pady=3)
        output_page = ttk.LabelFrame(self.detail_panes, text="运行日志", padding=4)
        self.detail_panes.add(output_page, weight=2)
        self.output = tk.Text(output_page, width=1, height=6, state="disabled", wrap="word")
        scroll = ttk.Scrollbar(output_page, command=self.output.yview)
        scroll.pack(side="right", fill="y")
        self.output.configure(yscrollcommand=scroll.set)
        self.output.pack(fill="both", expand=True)
        log_bar = ttk.Frame(self.content)
        log_bar.grid(row=output_row + 1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(log_bar, text="复制本次日志", command=self.copy_log).pack(side="left")
        ttk.Button(log_bar, text="导出日志与画面", command=self.export_log).pack(side="left", padx=6)
        self.log_location = tk.StringVar(value="日志自动保存在项目的“运行记录”目录，无需手动截屏。")
        ttk.Label(self.content, textvariable=self.log_location, wraplength=540).grid(row=output_row + 2, column=0, columnspan=2, sticky="w")
        self.voice = VoiceControls(self, output_row + 3)
        self.content.rowconfigure(output_row, weight=1)
        self.run_controls = [widget for widget in self.content.winfo_children()
                             if widget.winfo_class() in ("TEntry", "TCombobox", "TButton")]
        self.run_controls += [self.duration_entry, self.unlimited_check, self.confirm_button]
        self.update_duration_entry()
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.bind("<Escape>", lambda _: self.stop() if self.process else self.close())
        self.poll_id = window.after(100, self.poll)

    def update_duration_entry(self):
        self.duration_entry.configure(state="disabled" if self.unlimited.get() or self.process is not None else "normal")

    def toggle_telemetry(self):
        if self.telemetry_visible.get():
            if str(self.telemetry_panel) not in self.detail_panes.panes():
                self.detail_panes.insert(0, self.telemetry_panel, weight=1)
        else:
            if str(self.telemetry_panel) in self.detail_panes.panes():
                self.detail_panes.forget(self.telemetry_panel)

    def confirm(self, *, source="button", transcript=None):
        if self.process is not None:
            return False
        self.last_start_error = ""
        try:
            settings = task_inference_settings(0 if self.unlimited.get() else float(self.seconds.get()))
            chosen = validate_selection(self.model.get(), self.dataset.get(), self.processor.get(), self.root)
            if self.target_object.get() not in TASKS:
                raise ValueError("请选择 RC 抓红或 BC 抓蓝")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.last_start_error = str(exc)
            self.status.set("无法启动推理：" + str(exc))
            if source != "voice":
                messagebox.showerror("无法启动推理", str(exc), parent=self.window)
            return False
        self.voice.invalidate()
        temporary_root = self.root / "运行环境/临时文件"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="推理窗口_", dir=temporary_root)
        self.stop_path = Path(self.temporary.name) / "请求停止"
        (Path(self.temporary.name) / "窗口心跳").touch()
        logs = self.root / "运行记录"
        logs.mkdir(parents=True, exist_ok=True)
        self.output_path = logs / f"推理窗口_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
        command = [sys.executable, "-u", str(self.root / "脚本/程序模块/实机抓取测试.py"),
                   "--model-dir", str(chosen.model), "--dataset-root", str(chosen.dataset),
                   "--processor-dir", str(chosen.processor), "--task-inference", "--window-start",
                   "--seconds", str(settings.seconds if settings.seconds is not None else 0), "--stop-file", str(self.stop_path),
                   "--preview-directory", self.temporary.name, "--target-object", self.target_object.get()]
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        try:
            self.process = self.process_factory(command, cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                                encoding="utf-8", errors="replace", bufsize=1,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.temporary.cleanup()
            self.temporary = None
            self.last_start_error = str(exc)
            self.status.set("无法启动推理：" + str(exc))
            if source != "voice":
                messagebox.showerror("无法启动推理", str(exc), parent=self.window)
            return False
        try:
            remember_selection(chosen, self.root)
        except OSError:
            # 保存偏好失败不改变本轮已校验的模型和启动流程。
            pass
        self.stop_requested = False
        self.cleanup_error = False
        self.failure_text = ""
        self.preview_photo = None
        self.wrist_preview_photo = None
        self.preview_frames = None
        self.action_log_path = None
        self.latest_telemetry = None
        self.live_summary.set("正在准备，尚未发送模型动作。")
        for i, name in enumerate(JOINT_LABELS):
            self.telemetry.item(str(i), values=(name, "—", "—", "—", "—", "—"))
        self.log_location.set(f"窗口输出：{self.output_path}")
        self.preview_label.configure(image="", text="正在加载模型，随后开启双摄……")
        self.wrist_preview_label.configure(image="", text="正在加载模型，随后开启双摄……")
        self.camera_status.set("上：桌面相机　|　下：夹爪相机。等待实时画面。")
        self.status.set("正在加载所选模型，请等待；准备完成后机械臂会自动运动。")
        for control in self.run_controls:
            control.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.switch_button.configure(state="normal")
        self.active_target = self.target_object.get()
        self.latest_target_request = None
        self.show_target()
        self.task_status.set("点击按钮可直接切换目标，并清空旧动作。")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        start_request = {"source": source, "transcript": transcript,
                         "target": self.target_object.get(), "model_path": str(chosen.model)}

        def read_output():
            try:
                with self.output_path.open("w", encoding="utf-8", buffering=1) as log:
                    log.write("启动请求：" + json.dumps(start_request, ensure_ascii=False) + "\n")
                    for line in self.process.stdout:
                        log.write(line)
                        self.messages.put(line)
            except (OSError, ValueError) as exc:
                self.messages.put(f"窗口读取运行输出失败：{exc}\n")
            finally:
                self.process.stdout.close()

        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()
        return True

    def show_target(self):
        target = self.active_target if self.process is not None else self.target_object.get()
        self.current_target_label.set("当前目标：" + ("RC · 红方块" if target == "RC" else "BC · 蓝方块"))
        other = "BC · 蓝方块" if self.target_object.get() == "RC" else "RC · 红方块"
        self.switch_button.configure(text="切换为 " + other)

    def toggle_target(self):
        if self.process is not None and self.stop_requested:
            return
        previous = self.target_object.get()
        self.target_object.set("BC" if previous == "RC" else "RC")
        if not self.request_target():
            self.target_object.set(previous)
            self.show_target()

    def request_target(self, *, source="button", transcript=None):
        self.voice.invalidate()
        if self.process is None:
            self.show_target()
            self.task_status.set("启动后执行当前目标；运行中也可直接切换。")
            return True
        if self.stop_requested or self.process.poll() is not None:
            return False
        directory = Path(self.temporary.name)
        request = {"id": time.time_ns(), "target": self.target_object.get(),
                   "source": source, "transcript": transcript}
        try:
            temp = directory / "目标请求.tmp"
            temp.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            temp.replace(directory / "目标请求.json")
        except OSError as exc:
            messagebox.showerror("目标请求失败", str(exc), parent=self.window)
            return False
        self.latest_target_request = request["id"]
        self.show_target()
        self.task_status.set("正在切换为 " + request["target"] + "，等待控制端清空旧动作。")
        return True

    def stop(self):
        self.voice.invalidate()
        if self.process is not None:
            self.stop_path.touch()
            self.stop_requested = True
            self.switch_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.status.set("已请求停止，正在等待当前操作结束并释放机械臂……")

    def report_text(self):
        if self.action_log_path and self.action_log_path.is_file():
            return export_run_report(self.action_log_path, getattr(self, "output_path", None))
        path = getattr(self, "output_path", None)
        if path and path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
        return self.output.get("1.0", "end").strip() or "尚无运行日志。"

    def copy_log(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(self.report_text())
        self.log_location.set("本次完整日志已复制，可直接粘贴到聊天。")

    def export_log(self):
        source = self.action_log_path or getattr(self, "output_path", None)
        if source is None:
            self.log_location.set("尚未开始运行，没有可导出的日志。")
            return
        try:
            output = source.with_name(source.stem + "_完整报告.zip")
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("运行报告.txt", self.report_text())
                if self.action_log_path and self.action_log_path.is_file():
                    bundle.write(self.action_log_path, self.action_log_path.name)
                    frames = self.action_log_path.with_suffix("")
                    if frames.is_dir():
                        for path in sorted(frames.iterdir()):
                            if path.is_file() and path.suffix in (".jpg", ".jsonl"):
                                bundle.write(path, "模型输入画面/" + path.name)
            self.log_location.set(f"完整报告已导出：{output}")
        except OSError as exc:
            self.log_location.set(f"日志导出失败：{exc}")

    def update_telemetry(self, event):
        self.latest_telemetry = event
        self.live_summary.set(f"运行 {event['t']:.1f}s | 推理 {event['inferences']} 次，最近 {event['inference_s']:.3f}s | "
                              f"已发送 {event['sent_commands']} 步 | 本步 {event['limited_joints']} 轴限幅")
        faults = list((event.get("fault_status") or {}).values())
        for i, name in enumerate(JOINT_LABELS):
            values = (name, f"{event['observed_state'][i]:.2f}", f"{event['model_target'][i]:.2f}",
                      f"{event['sent_target'][i]:.2f}", "是" if event['limited'][i] else "否",
                      faults[i] if len(faults) == 6 else "—")
            self.telemetry.item(str(i), values=values)

    def close(self):
        self.voice.close()
        if self.process is not None:
            self.closing = True
            self.stop()
        else:
            self.window.after_cancel(self.poll_id)
            self.window.destroy()

    def render_preview(self, event=None):
        if self.preview_frames is None:
            return
        from PIL import Image, ImageTk, ImageOps
        photos = []
        for frame, label in zip(self.preview_frames, (self.preview_label, self.wrist_preview_label)):
            width, height = label.winfo_width(), label.winfo_height()
            size = (width if width > 1 else 360, height if height > 1 else 270)
            fitted = ImageOps.contain(frame, size, Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(fitted, master=self.window)
            label.configure(image=photo, text="")
            photos.append(photo)
        self.preview_photo, self.wrist_preview_photo = photos

    def refresh_preview(self):
        if self.process is None or self.temporary is None:
            return
        from PIL import Image
        directory = Path(self.temporary.name)
        (directory / "窗口心跳").touch()
        path = directory / "当前预览.jpg"
        try:
            data = path.read_bytes()
            with Image.open(BytesIO(data)) as source:
                frame = source.convert("RGB")
            # 传输仍沿用 front + wrist 横向拼图，只在显示时拆开，不改模型输入。
            middle = frame.width // 2
            self.preview_frames = (frame.crop((0, 0, middle, frame.height)),
                                   frame.crop((middle, 0, frame.width, frame.height)))
            self.window.update_idletasks()
            self.render_preview()
            (directory / "预览已显示").touch()
            age = time.time() - path.stat().st_mtime
            self.camera_status.set("上：桌面相机　|　下：夹爪相机　|　" +
                                   ("实时画面" if age < 1 else "画面更新已中断"))
        except (OSError, ValueError):
            # 加载阶段尚无文件；预览未成功显示时控制进程不会进入运动。
            pass

    def poll(self):
        self.voice.poll()
        try:
            self.refresh_preview()
        except OSError as exc:
            self.camera_status.set(f"画面通信失败：{exc}")
            self.stop()
        lines = []
        while True:
            try:
                lines.append(self.messages.get_nowait())
            except queue.Empty:
                break
        if lines:
            display_lines = []
            for line in lines:
                if line.startswith(PREFIX):
                    try:
                        event = json.loads(line[len(PREFIX):])
                        self.update_telemetry(event)
                        display_lines.append(format_progress(event))
                    except (ValueError, KeyError, TypeError, IndexError):
                        display_lines.append(line)
                else:
                    display_lines.append(line)
            self.output.configure(state="normal")
            self.output.insert("end", "".join(display_lines))
            self.output.see("end")
            self.output.configure(state="disabled")
            for line in lines:
                if line.startswith("目标状态："):
                    try:
                        task_event = json.loads(line.removeprefix("目标状态："))
                        self.active_target = task_event['current']
                        self.show_target()
                        if (task_event.get('request') or {}).get('id') == self.latest_target_request:
                            self.task_status.set(task_event['message'])
                    except (ValueError, KeyError):
                        pass
                if line.startswith("运行日志："):
                    candidate = Path(line.removeprefix("运行日志：").strip()).resolve()
                    if candidate.is_relative_to((self.root / "运行记录").resolve()):
                        self.action_log_path = candidate
                        self.log_location.set(f"逐步日志：{candidate}（可复制或导出）")
                if line.startswith("资源清理异常："):
                    self.cleanup_error = True
                if "Overload error!" in line:
                    self.failure_text = "电机报告过载，已停止。请托住机械臂并切断从臂电源、检查夹爪与机械阻挡；不要直接重试。"
                elif "夹爪原始位置异常：" in line and not self.failure_text:
                    self.failure_text = line[line.index("夹爪原始位置异常："):].strip()
                elif "夹爪起点超出演示范围：" in line and not self.failure_text:
                    self.failure_text = line[line.index("夹爪起点超出演示范围："):].strip()
                elif "实机保持验证尚未通过" in line and not self.failure_text:
                    self.failure_text = "夹爪坐标已核对，须先完成独立保持验证；本次未启动模型控制。"
                elif "夹爪负坐标保持验证失败" in line and not self.failure_text:
                    self.failure_text = "夹爪保持验证失败，推理已禁用。请保持断电，先解决硬件坐标与标定不一致。"
                if not self.stop_requested:
                    if line.startswith("【"):
                        self.status.set(line.strip())
                    elif line.startswith("开始闭环控制"):
                        self.status.set("正在推理并控制从臂运动，可点击“停止推理”。")
        if self.process is not None and self.process.poll() is not None and not self.reader.is_alive() and self.messages.empty():
            code = self.process.returncode
            self.process = self.reader = None
            self.temporary.cleanup()
            self.temporary = None
            for control in self.run_controls:
                control.configure(state="normal")
            self.switch_button.configure(state="normal")
            self.show_target()
            self.update_duration_entry()
            self.stop_button.configure(state="disabled")
            self.camera_status.set("上：桌面相机　|　下：夹爪相机　|　运行已结束，显示最后一帧")
            if self.failure_text:
                self.status.set(self.failure_text + (" 停扭矩未确认，请勿仅凭程序退出判断已断电。" if self.cleanup_error else ""))
            elif self.cleanup_error:
                self.status.set("停止时资源清理异常，请托住从臂并切断电源，详情见“运行日志”。")
            elif code != 0:
                self.status.set(f"推理中止，请查看“运行日志”。日志已保存：{self.output_path}")
            else:
                self.status.set("推理已停止。" if self.stop_requested else "本次推理已结束。")
            if self.closing:
                self.window.destroy()
                return
        self.poll_id = self.window.after(100, self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--processor-dir", type=Path)
    parser.add_argument("--target-object", choices=tuple(TASKS), default="RC")
    args = parser.parse_args()
    window = tk.Tk()
    InferenceWindow(window, model=args.model_dir, dataset=args.dataset_root, processor=args.processor_dir,
                    target_object=args.target_object)
    window.mainloop()


if __name__ == "__main__":
    main()
