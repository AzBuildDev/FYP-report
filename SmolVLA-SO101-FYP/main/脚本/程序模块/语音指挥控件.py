"""窗口内显式录音，独立CPU语音进程，不自行启动机械臂。"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from 语音实机桥接 import PREFIX
from 云端语音识别 import (load_config, load_key, save_config, DEFAULT_URL,
                           DEFAULT_MODEL, DEFAULT_SEMANTIC_MODEL)


class VoiceControls:
    def __init__(self, owner, row):
        self.owner = owner
        self.process = None
        self.ready = self.busy = False
        self.messages = queue.Queue()
        self.epoch = 0
        self.request_id = None
        self.deadline = None
        self.last_worker_error = None
        self.root = getattr(owner, 'root', Path(__file__).resolve().parents[2])
        try:
            config = load_config(self.root)
        except ValueError:
            config = dict(backend='dashscope', base_url=DEFAULT_URL, model=DEFAULT_MODEL)
        pane = ttk.LabelFrame(owner.content, text="语音指挥 · 演示目标区", padding=5)
        pane.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        bar = ttk.Frame(pane)
        bar.pack(fill="x")
        self.backend = tk.StringVar(value='百炼API' if config['backend'] == 'dashscope' else '本地Whisper')
        self.backend_selector = ttk.Combobox(bar, textvariable=self.backend,
                                             values=('百炼API', '本地Whisper'), width=12, state='readonly')
        self.backend_selector.pack(side='left', padx=(0, 4))
        self.backend_selector.bind('<<ComboboxSelected>>', self.change_backend)
        self.prepare_button = ttk.Button(bar, text="准备语音", command=self.prepare)
        self.prepare_button.pack(side="left")
        self.record_button = ttk.Button(bar, text="说话并执行（5秒）", command=self.record, state="disabled")
        self.record_button.pack(side="left", padx=4)
        tools = ttk.Frame(pane)
        tools.pack(fill='x', pady=(3, 0))
        self.settings_button = ttk.Button(tools, text='API设置', command=self.configure_api)
        self.settings_button.pack(side='left')
        ttk.Button(tools, text="关闭语音", command=self.close).pack(side="right")
        self.language = tk.StringVar(value="zh")
        ttk.Combobox(bar, textvariable=self.language, values=("zh", "en"), width=4, state="readonly").pack(side="left")
        self.status = tk.StringVar(value="点击后才录音；有效目标会自动启动推理，运行中会切换目标。百炼API会上传本次音频。")
        ttk.Label(pane, textvariable=self.status, wraplength=520).pack(fill="x")

    def change_backend(self, event=None):
        self.close()
        self.status.set('已选择'+self.backend.get()+'，请点击准备语音。')

    def configure_api(self):
        if self.busy:
            self.status.set('请等本次语音处理结束后再设置 API。')
            return
        try:
            config = load_config(self.root)
        except ValueError:
            config = dict(base_url=DEFAULT_URL, model=DEFAULT_MODEL,
                          semantic_model=DEFAULT_SEMANTIC_MODEL)
        dialog = tk.Toplevel(self.owner.content)
        dialog.title('百炼语音 API 设置')
        dialog.transient(self.owner.content.winfo_toplevel())
        form = ttk.Frame(dialog, padding=12)
        form.pack(fill='both', expand=True)
        ttk.Label(form, text='仅录音后上传音频到百炼；API 按服务商规则计费。\n密钥由 Windows 当前用户加密保存，不写入日志或源码。').grid(row=0, column=0, columnspan=2, sticky='w')
        address = tk.StringVar(value=config['base_url'])
        model = tk.StringVar(value=config['model'])
        semantic_model = tk.StringVar(value=config.get('semantic_model', DEFAULT_SEMANTIC_MODEL))
        key = tk.StringVar()
        for row, label, value, masked in [(1, 'Base URL', address, False),
                                           (2, 'ASR 模型', model, False),
                                           (3, '语义模型', semantic_model, False),
                                           (4, 'API Key', key, True)]:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky='w', pady=5)
            ttk.Entry(form, textvariable=value, show='*' if masked else '', width=60).grid(row=row, column=1, sticky='ew', pady=5)
        ttk.Label(form, text='语义模型只输出 RC/BC/停止/拒绝，不生成关节动作。\n密钥留空保留已保存值；DASHSCOPE_API_KEY 环境变量优先。').grid(row=5, column=0, columnspan=2, sticky='w')
        def save():
            try:
                save_config(self.root, 'dashscope', address.get(), model.get(), key.get(), semantic_model.get())
            except (ValueError, RuntimeError, OSError) as exc:
                messagebox.showerror('无法保存 API 设置', str(exc), parent=dialog)
                return
            key.set('')
            self.close()
            self.backend.set('百炼API')
            self.status.set('API 设置已保存；语音会先转写，再做受限语义理解。点击准备语音。')
            dialog.destroy()
        ttk.Button(form, text='保存', command=save).grid(row=6, column=1, sticky='e', pady=(8, 0))

    def invalidate(self):
        self.epoch += 1

    def prepare(self):
        if self.process is not None:
            return
        self.last_worker_error = None
        backend = 'dashscope' if self.backend.get() == '百炼API' else 'local'
        if backend == 'dashscope':
            try:
                load_key(self.root)
            except (ValueError, RuntimeError, OSError) as exc:
                self.status.set(str(exc))
                return
        command = [sys.executable, "-u", str(self.root / "脚本/程序模块/语音实机桥接.py"), "--serve", '--backend', backend]
        worker_env = os.environ.copy()
        worker_env['PYTHONIOENCODING'] = 'utf-8'
        worker_env['PYTHONUTF8'] = '1'
        try:
            self.process = subprocess.Popen(command, cwd=self.root, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                                            env=worker_env,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.status.set("语音启动失败：" + str(exc))
            return
        self.deadline = time.monotonic() + (15 if backend == 'dashscope' else 180)
        self.prepare_button.configure(state="disabled")
        self.status.set("正在准备"+self.backend.get()+"，请等待……")
        process = self.process

        def reader():
            try:
                for line in process.stdout:
                    if line.startswith(PREFIX):
                        try:
                            event = json.loads(line[len(PREFIX):])
                            event["process"] = process
                            self.messages.put(event)
                        except ValueError:
                            pass
            finally:
                process.stdout.close()
                self.messages.put({"status": "exited", "process": process})
        threading.Thread(target=reader, daemon=True).start()

    def record(self):
        if not self.ready or self.busy or self.owner.closing or (self.owner.stop_requested and self.owner.process is not None):
            return
        self.request_id = f"{self.epoch}:{time.time_ns()}"
        self.request_epoch = self.epoch
        self.request_process = self.owner.process
        request = {"kind": "record", "id": self.request_id, "language": self.language.get()}
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.status.set("语音请求失败：" + str(exc))
            return
        self.busy = True
        self.deadline = time.monotonic() + 90
        self.record_button.configure(state="disabled")
        self.status.set("正在准备录音，请等待“开始录音”提示。")

    def poll(self):
        while True:
            try:
                event = self.messages.get_nowait()
            except queue.Empty:
                break
            status = event.get("status")
            if event.get("process", self.process) is not self.process:
                continue
            if status == "exited":
                if event.get("process") is self.process:
                    reason = self.last_worker_error
                    self.close()
                    self.status.set("语音不可用："+reason if reason else "语音进程已退出，可重新准备；机械臂状态不受此提示影响。")
                continue
            if status == "ready":
                self.ready = True
                self.deadline = None
                self.record_button.configure(state="normal")
                note = ("语音进程已就绪；百炼接口尚未验证，点击说话并执行。"
                        if self.backend.get() == '百炼API' else "语音已就绪，点击说话并执行。")
                self.status.set(note + "日志：" + event.get("log", ""))
            elif status in ("loading", "recording", "processing"):
                self.status.set(event.get("message", status))
            elif status in ("target", "stop", "rejected", "error"):
                self.busy = False
                self.deadline = None
                self.record_button.configure(state="normal" if self.ready else "disabled")
                if status == "error":
                    self.last_worker_error = event.get("message", "未知错误")
                    self.status.set("语音不可用：" + event.get("message", ""))
                    continue
                if event.get("id") != self.request_id or getattr(self, "request_epoch", None) != self.epoch or getattr(self, "request_process", None) is not self.owner.process or self.owner.closing or (self.owner.stop_requested and self.owner.process is not None):
                    self.status.set("语音结果已过期，不执行；请重新说话。")
                    continue
                if status == "rejected":
                    self.status.set("未执行：" + event.get("transcript", "") + "；" + event.get("message", "识别失败"))
                elif status == "stop":
                    self.owner.stop()
                    self.status.set("识别到停止，已请求停止。语音有延迟，紧急情况请用停止按钮。")
                else:
                    self.owner.target_object.set(event["target"])
                    if self.owner.process is not None:
                        submitted = self.owner.request_target(source="voice", transcript=event["transcript"])
                        self.status.set(f"识别：{event['transcript']} → {event['target']}，" +
                                        ("已提交切换请求。" if submitted else "切换请求未提交，请检查运行状态。"))
                    else:
                        if hasattr(self.owner, "show_target"):
                            self.owner.show_target()
                        started = self.owner.confirm(source="voice", transcript=event["transcript"])
                        self.status.set(f"识别：{event['transcript']} → {event['target']}，" +
                                        ("已自动启动推理，正在加载模型。" if started else
                                         "未能启动：" + self.owner.last_start_error))
        if self.deadline is not None and time.monotonic() > self.deadline:
            preparing = not self.ready
            self.close()
            self.status.set("语音启动未收到就绪消息，请检查语音进程通信后重新准备。" if preparing else
                            "语音识别超时，已丢弃结果；可以重新准备语音。")

    def close(self):
        self.invalidate()
        process, self.process = self.process, None
        self.ready = self.busy = False
        self.deadline = None
        if process is not None:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                # 仅终止本控件创建的语音进程树；不操作机械臂进程。
                if sys.platform == "win32":
                    subprocess.Popen(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.terminate()
        self.prepare_button.configure(state="normal")
        self.record_button.configure(state="disabled")
        self.status.set("语音已关闭，不再录音；机械臂停止请使用停止推理按钮。")
