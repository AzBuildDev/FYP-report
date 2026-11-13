"""人工记录实机抓取结果，关联日志并导出汇总；此窗口不连接或启动机械臂。"""
import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from 管理实机评测记录 import EvaluationStore, OUTCOMES

ROOT = Path(__file__).resolve().parents[2]


class ResultsWindow:
    def __init__(self, window, store):
        self.window, self.store, self.runs = window, store, {}
        window.title("实机抓取结果记录")
        window.geometry("1100x740")
        ttk.Label(window, text="仅整理日志和人工结果，不连接机械臂。请选择已经结束的测试，再按实际观察标记。", padding=10).pack(anchor="w")
        actions = ttk.Frame(window)
        actions.pack(fill="x", padx=10)
        ttk.Button(actions, text="刷新测试列表", command=self.refresh).pack(side="left")
        ttk.Button(actions, text="导出评测汇总", command=self.export).pack(side="left", padx=10)
        self.table = ttk.Treeview(window, columns=("name", "source", "commands", "outcome"), show="headings", height=9)
        for key, label, width in (("name", "运行日志", 450), ("source", "日志来源", 180), ("commands", "动作次数", 90), ("outcome", "人工标记", 170)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width)
        self.table.pack(fill="x", padx=10, pady=10)
        self.table.bind("<<TreeviewSelect>>", self.select)
        self.detail = tk.StringVar()
        ttk.Label(window, textvariable=self.detail, wraplength=1050, justify="left").pack(anchor="w", padx=10)
        form = ttk.Frame(window, padding=10)
        form.pack(fill="x")
        self.scene = tk.StringVar(value="演示起始位置")
        self.reviewer = tk.StringVar(value="本人")
        self.outcome = tk.StringVar()
        self.observed = tk.BooleanVar(value=False)
        ttk.Label(form, text="场景 / 方块位置").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.scene, width=40).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(form, text="记录者").grid(row=0, column=2, sticky="w")
        ttk.Entry(form, textvariable=self.reviewer, width=18).grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(form, text="实际结果").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Combobox(form, textvariable=self.outcome, values=OUTCOMES, state="readonly", width=37).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Checkbutton(form, text="本人观察了这次真实硬件测试", variable=self.observed).grid(row=1, column=2, columnspan=2, sticky="w")
        ttk.Label(form, text="备注（位置、失败原因、人工介入等）").grid(row=2, column=0, columnspan=4, sticky="w")
        self.notes = tk.Text(form, height=3, width=120)
        self.notes.grid(row=3, column=0, columnspan=4, sticky="ew", pady=5)
        ttk.Button(form, text="保存 / 修正人工结果", command=self.save).grid(row=4, column=0, columnspan=4, sticky="w")
        self.summary_text = tk.Text(window, height=9, wrap="word")
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.refresh()

    def refresh(self):
        try:
            runs, excluded, orphaned = self.store.scan()
            self.table.delete(*self.table.get_children())
            self.runs = {}
            for run in runs:
                key = run["log_sha256"]
                self.runs[key] = run
                source = {"real_hardware": "实机程序日志", "legacy_unverified": "旧日志，需本人确认", "simulation": "模拟，不能计入实机"}.get(run["run_kind"], "非实机来源")
                self.table.insert("", "end", iid=key, values=(run["filename"], source, run["sent_commands"],
                                                             run["annotation"]["outcome"] if run["annotation"] else "待标记"))
            report = self.store.summary()
            lines = ["已开始且发送动作的尝试才计入成功率；人工中断、程序异常保留在分母。待标记时不显示成功率。"]
            for group in report["groups"]:
                rate = f"{group['success_rate']:.1%}" if group["success_rate"] is not None else "尚不能计算"
                model = group["checkpoint"][:18]
                lines.append(f"{group['scene']} | 模型 {model} | {group['mode']}：尝试 {group['attempts']}，成功 {group['successes']}，待标记 {group['pending_attempts']}，成功率 {rate}；零动作 / 未开始 {group['non_attempts']}")
            if not report["groups"]:
                lines.append("尚无可汇总的已确认实机尝试。")
            lines.append(f"未确认旧日志 {report['unverified_legacy_runs']}；排除模拟日志 {report['excluded_simulation_runs']}。")
            if excluded:
                lines.append("未列入可标记列表：" + "；".join(f"{item['filename']}：{item['reason']}" for item in excluded))
            if orphaned:
                lines.append("标记已失去匹配日志（内容改变或文件缺失），不计入汇总：" + "、".join(orphaned))
            self.summary_text.configure(state="normal")
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", "\n".join(lines))
            self.summary_text.configure(state="disabled")
            self.detail.set("请选择一条完整日志。没有完整结束记录的测试不会开放标记。")
            self.observed.set(False)
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc), parent=self.window)

    def select(self, event=None):
        selection = self.table.selection()
        if not selection:
            return
        run = self.runs[selection[0]]
        annotation = run["annotation"]
        self.scene.set(annotation["scene"] if annotation else run["scene"])
        self.outcome.set(annotation["outcome"] if annotation else "")
        self.observed.set(False)
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", annotation["notes"] if annotation else "")
        hz = f"{run['active_control_hz']:.2f} Hz" if run["active_control_hz"] is not None else "未记录"
        self.detail.set(f"任务：{run['task']}\n模型：{run['model_path'] or '旧日志未记录模型身份'}；停止原因：{run['stop_reason']}；已保存修订 {run['revision_count']} 次。\n活动发送频率：{hz}；推理次数：{run['inferences']}；错过节拍：{run['missed_ticks']}。程序异常：{'；'.join(map(str, run['errors'])) or '日志未记录异常'}")

    def save(self):
        selection = self.table.selection()
        if not selection:
            messagebox.showinfo("选择测试", "请先选择一条已经结束的测试。", parent=self.window)
            return
        try:
            self.store.record(self.runs[selection[0]]["path"], self.outcome.get(), self.scene.get(), self.reviewer.get(),
                              self.notes.get("1.0", "end").strip(), self.observed.get())
            self.refresh()
            messagebox.showinfo("已保存", "已保存人工结果。修正会保留之前的标记，不重复计为新尝试。", parent=self.window)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self.window)

    def export(self):
        try:
            paths = self.store.export()
            messagebox.showinfo("汇总已保存", "\n".join(map(str, paths)), parent=self.window)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc), parent=self.window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true", help="只导出汇总，不打开窗口")
    args = parser.parse_args()
    store = EvaluationStore(ROOT / "运行记录")
    if args.summary_only:
        paths = store.export()
        print(json.dumps({"files": list(map(str, paths)), "summary": store.summary()}, ensure_ascii=False, indent=2))
        return
    window = tk.Tk()
    ResultsWindow(window, store)
    window.mainloop()


if __name__ == "__main__":
    main()
