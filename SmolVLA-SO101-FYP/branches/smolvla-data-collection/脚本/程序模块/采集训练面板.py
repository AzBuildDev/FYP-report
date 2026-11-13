"""在窗口中选择本地演示、检查训练计划、微调及离线评估；实机入口由操作者另行启动。"""
import json
import os
import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from 管理采集训练任务 import ProjectTasks, SoftwareJob

ROOT = Path(__file__).resolve().parents[2]
KINDS = {"check": "数据检查", "train_plan": "训练计划", "train": "模型训练",
         "eval_plan": "评估计划", "eval": "离线评估"}


class TrainingPanel:
    def __init__(self, window, project=None, job_factory=SoftwareJob):
        self.window = window
        self.project = project or ProjectTasks(ROOT)
        self.job_factory, self.job = job_factory, None
        self.closing, self.finished = False, False
        window.title("SmolVLA 采集与训练")
        window.geometry("1050x850")
        self.status = tk.StringVar(value="选择批次和完整演示段，先检查计划。")
        self.steps, self.batch_size, self.name = tk.StringVar(value="3000"), tk.StringVar(value="4"), tk.StringVar(value="红色方块微调")
        self.samples, self.strict = tk.StringVar(value="3"), tk.BooleanVar(value=True)
        self.train_batch, self.eval_batch, self.model = (tk.StringVar() for _ in range(3))
        self.train_hint, self.eval_hint, self.model_hint = (tk.StringVar() for _ in range(3))
        self.buttons = []
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="采集与训练", font=("Microsoft YaHei", 20, "bold")).pack(anchor="w")
        ttk.Label(frame, text="整段选择 → 检查计划 → 训练新模型 → 留出数据评估。动作误差不代表抓取成功率。").pack(anchor="w", pady=6)
        operations = ttk.Frame(frame)
        operations.pack(fill="x", pady=4)
        for label, action in (("打开演示录制", lambda: self.open_operator("录制任务演示.py")),
                              ("打开实机结果记录", lambda: self.open_operator("实机结果记录窗口.py")),
                              ("刷新本地批次与模型", self.refresh)):
            button = ttk.Button(operations, text=label, command=action)
            button.pack(side="left", padx=(0, 8))
            self.buttons.append(button)
        columns = ttk.Frame(frame)
        columns.pack(fill="both", expand=True, pady=6)
        columns.columnconfigure((0, 1), weight=1)
        columns.rowconfigure(0, weight=1)
        training = ttk.LabelFrame(columns, text="训练数据", padding=10)
        evaluation = ttk.LabelFrame(columns, text="离线评估数据", padding=10)
        training.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        evaluation.grid(row=0, column=1, sticky="nsew")
        self.train_combo, self.train_list = self.batch_widgets(training, self.train_batch, self.train_hint, "train")
        self.eval_combo, self.eval_list = self.batch_widgets(evaluation, self.eval_batch, self.eval_hint, "eval")
        for label, variable in (("训练步数", self.steps), ("每批数量（1～16）", self.batch_size), ("实验名称（中文用途）", self.name)):
            row = ttk.Frame(training)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=23).pack(side="left")
            ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        self.action_buttons(training, (("检查所选批次", "check"), ("检查训练计划", "train_plan"), ("开始训练新模型", "train")))
        ttk.Label(evaluation, text="评估模型").pack(anchor="w")
        self.model_combo = ttk.Combobox(evaluation, textvariable=self.model, state="readonly")
        self.model_combo.pack(fill="x", pady=3)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _: self.update_model_hint())
        ttk.Label(evaluation, textvariable=self.model_hint, wraplength=450).pack(anchor="w", pady=3)
        row = ttk.Frame(evaluation)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="每段抽样动作块").pack(side="left")
        ttk.Entry(row, textvariable=self.samples, width=8).pack(side="left", padx=8)
        ttk.Checkbutton(evaluation, text="强制留出检查（拒绝训练内容重叠或未知来源）", variable=self.strict).pack(anchor="w", pady=4)
        self.action_buttons(evaluation, (("检查评估计划", "eval_plan"), ("开始离线评估", "eval")))
        ttk.Label(frame, textvariable=self.status, wraplength=990).pack(anchor="w", pady=6)
        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        self.stop_button = ttk.Button(bar, text="停止当前软件任务", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="打开任务记录目录", command=self.open_logs).pack(side="left")
        self.output = tk.Text(frame, height=10, wrap="word", state="disabled")
        self.output.pack(fill="both", expand=True, pady=6)
        ttk.Label(frame, text="模型另存至本项目“模型”；报告与中断输出保留在“运行记录/采集训练面板”。录制窗口由你自己开始、停止和保存。").pack(anchor="w")
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        window.after(100, self.poll)

    def batch_widgets(self, parent, variable, hint, side):
        combo = ttk.Combobox(parent, textvariable=variable, state="readonly")
        combo.pack(fill="x", pady=3)
        combo.bind("<<ComboboxSelected>>", lambda _: self.select_batch(side))
        ttk.Label(parent, textvariable=hint, wraplength=450).pack(anchor="w", pady=3)
        ttk.Label(parent, text="选择完整演示段（按 Ctrl 或 Shift 多选）").pack(anchor="w")
        listing = tk.Listbox(parent, selectmode="extended", exportselection=False, height=5)
        listing.pack(fill="x", pady=3)
        return combo, listing

    def action_buttons(self, parent, entries):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=8)
        for label, kind in entries:
            button = ttk.Button(row, text=label, command=lambda value=kind: self.start(value))
            button.pack(side="left", padx=(0, 4))
            self.buttons.append(button)

    def refresh(self):
        self.batches = {item.label: item for item in self.project.batches()}
        values = tuple(self.batches)
        ready = next((item.label for item in self.batches.values() if item.ready and item.label.startswith("任务演示/")),
                     next((item.label for item in self.batches.values() if item.ready), ""))
        for combo, variable, side in ((self.train_combo, self.train_batch, "train"), (self.eval_combo, self.eval_batch, "eval")):
            combo["values"] = values
            if variable.get() not in self.batches:
                variable.set(ready)
            self.select_batch(side)
        self.models = {path.name: path for path in self.project.models()}
        self.model_combo["values"] = tuple(self.models)
        if self.model.get() not in self.models:
            self.model.set("整段留出实验模型" if "整段留出实验模型" in self.models else next(iter(self.models), ""))
        self.update_model_hint()

    def select_batch(self, side):
        variable, listing, hint = (self.train_batch, self.train_list, self.train_hint) if side == "train" else (self.eval_batch, self.eval_list, self.eval_hint)
        batch = self.batches.get(variable.get())
        listing.delete(0, "end")
        hint.set(batch.description if batch else "暂无完整的演示批次，请先录制并保存。")
        if batch and batch.ready:
            for ep, length in batch.episodes:
                listing.insert("end", f"第 {ep + 1} 段（内部编号 {ep}） · {length} 帧 / {length / 20:.2f} 秒")
            count = len(batch.episodes)
            if side == "train":
                listing.selection_set(0, max(0, count - 2))
            elif count > 1:
                listing.selection_set(count - 1)
        # 只有一段时评估默认不选，避免默认将训练段当作留出。

    def update_model_hint(self):
        path = self.models.get(self.model.get())
        try:
            source = json.loads((path / "训练数据来源.json").read_text(encoding="utf-8"))["training_data"]
            selected = "、".join(str(row["episode"] + 1) for row in source["episodes"])
            self.model_hint.set(f"来源记录：第 {selected} 段训练。检查评估计划时核对模型内容和数据重叠。")
        except (OSError, ValueError, KeyError, TypeError):
            self.model_hint.set("无可读训练来源；强制留出检查会拒绝未知来源模型。")

    def make_spec(self, kind):
        side = "eval" if kind.startswith("eval") else "train"
        variable, listing = (self.eval_batch, self.eval_list) if side == "eval" else (self.train_batch, self.train_list)
        batch = self.batches.get(variable.get())
        if not batch or not batch.ready:
            raise ValueError("请选择元数据完整的已保存批次；未就绪批次不能启动任务。")
        episodes = tuple(batch.episodes[index][0] for index in listing.curselection())
        options = {}
        if kind.startswith("train"):
            options.update(steps=int(self.steps.get()), batch_size=int(self.batch_size.get()), name=self.name.get())
        elif kind.startswith("eval"):
            options.update(model=self.models.get(self.model.get()), samples=int(self.samples.get()), require_held_out=self.strict.get())
        return self.project.build(kind, batch.path, episodes=episodes, **options)

    def start(self, kind):
        if self.job and self.job.thread.is_alive():
            return
        try:
            spec = self.make_spec(kind)
            self.job = self.job_factory(self.project, spec).start()
            self.finished = False
            for button in self.buttons:
                button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status.set(f"{KINDS[kind]}进行中；任务目录：{spec.directory.name}")
            self.append(f"\n开始{KINDS[kind]}：{spec.dataset.name}，内部段号 {spec.episodes}\n")
        except Exception as exc:
            messagebox.showerror("无法启动软件任务", str(exc), parent=self.window)

    def append(self, value):
        self.output.configure(state="normal")
        self.output.insert("end", value)
        if int(self.output.index("end-1c").split(".")[0]) > 600:
            self.output.delete("1.0", "200.0")
        self.output.see("end")
        self.output.configure(state="disabled")

    def poll(self):
        if self.job:
            for _ in range(100):
                try:
                    self.append(self.job.messages.get_nowait())
                except queue.Empty:
                    break
            if not self.job.thread.is_alive() and not self.finished:
                self.finished = True
                kind = self.job.spec.kind
                if self.job.status == "succeeded":
                    details = {"train_plan": "计划检查通过，尚未训练或保存模型。", "eval_plan": "计划检查通过，尚未运行模型推理。",
                               "train": "训练完成，已核对新模型与训练来源。", "eval": "动作评估完成；不代表实机任务成功。", "check": "视频、动作和状态检查通过。"}
                    self.status.set(details[kind] + f" 报告：{self.job.spec.report}")
                    if kind == "train":
                        self.refresh()
                        self.model.set(self.job.spec.model.name)
                        self.update_model_hint()
                elif self.job.status == "cancelled":
                    self.status.set("软件任务已中断，原始数据和已有输出保留；需要重新启动一次新任务。")
                else:
                    self.status.set(f"{KINDS[kind]}未通过：{self.job.error}；详情见任务记录。")
                for button in self.buttons:
                    button.configure(state="normal")
                self.stop_button.configure(state="disabled")
        if self.closing and (not self.job or not self.job.thread.is_alive()):
            self.window.destroy()
            return
        self.window.after(100, self.poll)

    def stop(self):
        if self.job and self.job.cancel():
            self.status.set("正在停止本面板的软件任务；已有日志和输出保留。")

    def open_operator(self, script):
        try:
            self.project.launch_operator(script)
            self.status.set("已打开操作窗口，请在该窗口自行操作。")
        except Exception as exc:
            messagebox.showerror("无法打开窗口", str(exc), parent=self.window)

    def open_logs(self):
        path = self.job.spec.directory if self.job and self.job.spec.directory.exists() else self.project.logs
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def close(self):
        self.closing = True
        self.stop()
        self.status.set("正在关闭面板；独立打开的演示录制窗口请自行停止和关闭。")
        # 只终止自己的软件任务；不能强制结束正在操作机械臂的录制窗口。


if __name__ == "__main__":
    window = tk.Tk()
    TrainingPanel(window)
    window.mainloop()
