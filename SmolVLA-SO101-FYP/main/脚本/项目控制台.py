"""统一中文项目入口；只有点击具体操作才启动任务，打开控制台不连接硬件。"""
import argparse
import json
import os
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "脚本"

def load_entries(script_root=SCRIPT_ROOT):
    script_root = Path(script_root).resolve()
    entries = json.loads((script_root / "入口清单.json").read_text(encoding="utf-8-sig"))["entries"]
    seen = set()
    for entry in entries:
        path = (script_root / entry["path"]).resolve()
        if (entry["id"] in seen or not path.is_relative_to(script_root) or not path.is_file()
                or path.suffix != ".bat" or not entry["title"] or not entry["group"]):
            raise ValueError("入口清单包含重复、越界或缺失的操作")
        seen.add(entry["id"])
    return entries

class ProjectConsole:
    def __init__(self, window, entries=None, factory=subprocess.Popen):
        self.window, self.factory = window, factory
        self.entries = load_entries() if entries is None else entries
        self.by_id = {entry["id"]: entry for entry in self.entries}
        self.buttons, self.processes = {}, {}
        window.title("SmolVLA 毕业设计 · 项目控制台")
        window.geometry("950x690")
        window.minsize(860, 620)
        outer = ttk.Frame(window, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="机械臂项目控制台", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="先采集演示，再训练与评估；常用操作集中在这里。打开本窗口不会启动相机或机械臂。",
                  wraplength=880).pack(anchor="w", pady=(8, 14))
        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        for group in dict.fromkeys(entry["group"] for entry in self.entries):
            tab = ttk.Frame(notebook, padding=12)
            notebook.add(tab, text=group)
            canvas = tk.Canvas(tab, highlightthickness=0)
            scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            rows = ttk.Frame(canvas)
            inner = canvas.create_window((0, 0), window=rows, anchor="nw")
            rows.bind("<Configure>", lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.bind("<Configure>", lambda event, c=canvas, i=inner: c.itemconfigure(i, width=event.width))
            for entry in (item for item in self.entries if item["group"] == group):
                row = ttk.Frame(rows, padding=(0, 6))
                row.pack(fill="x")
                button = ttk.Button(row, text=entry["title"], width=24,
                                    command=lambda identifier=entry["id"]: self.launch(identifier))
                button.pack(side="left", padx=(0, 16))
                self.buttons[entry["id"]] = button
                ttk.Label(row, text=entry["description"], wraplength=600).pack(side="left", fill="x", expand=True)
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(12, 6))
        for title, path in (("使用说明", SCRIPT_ROOT / "使用说明.md"), ("当前进展", ROOT / "文档/项目进展/当前进展.md"),
                            ("运行记录", ROOT / "运行记录"), ("训练数据", ROOT.parent / "训练数据采集/机械臂控制项目")):
            ttk.Button(bottom, text=title, command=lambda p=path: os.startfile(p)).pack(side="left", padx=(0, 8))
        self.status = tk.StringVar(value="就绪。请打开需要的操作；模型推理需选择模型后点击“启动推理”。")
        ttk.Label(outer, textvariable=self.status, wraplength=890).pack(anchor="w", pady=(8, 0))

    def launch(self, identifier):
        if identifier not in self.by_id:
            raise ValueError("未知操作，不启动任何任务")
        entry = self.by_id[identifier]
        path = (SCRIPT_ROOT / entry["path"]).resolve()
        if not path.is_relative_to(SCRIPT_ROOT.resolve()) or not path.is_file():
            raise ValueError("操作入口路径无效")
        previous = self.processes.get(identifier)
        if previous is not None and previous.poll() is None:
            self.status.set("此操作窗口仍在运行，请先使用已有窗口。")
            return previous
        try:
            process = self.factory('cmd.exe /d /c call "' + str(path) + '"', cwd=ROOT,
                                   creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
            self.processes[identifier] = process
            self.status.set("已打开：" + entry["title"] + "。" +
                            ("请选择模型并点击“启动推理”。" if entry.get("start_in_window") else
                             "实机动作须在操作终端按回车开始。" if entry.get("motion") else "请在该操作窗口继续。"))
            return process
        except OSError as exc:
            messagebox.showerror("无法打开操作", str(exc), parent=self.window)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--software-test", action="store_true", help="隐藏窗口测试，不启动真实子任务")
    args = parser.parse_args()
    entries = load_entries()
    if args.list:
        print(json.dumps({"status": "passed", "entry_count": len(entries), "entries": entries,
                          "hardware_started": False}, ensure_ascii=False, indent=2))
        return
    window = tk.Tk()
    if args.software_test:
        window.withdraw()
        calls = []
        class FakeProcess:
            def poll(self):
                return None
        panel = ProjectConsole(window, entries, factory=lambda *a, **k: calls.append((a, k)) or FakeProcess())
        window.update()
        assert not calls, "打开入口不得自动启动子任务"
        panel.buttons["离线实验对比"].invoke()
        window.update()
        assert len(calls) == 1 and "离线实验" in calls[0][0][0]
        panel.buttons["离线实验对比"].invoke()
        assert len(calls) == 1
        try:
            panel.launch("不存在的入口")
            raise AssertionError("未知操作不能启动")
        except ValueError:
            pass
        panel.buttons["启动模型推理"].invoke()
        assert len(calls) == 2 and "启动推理" in panel.status.get() and "回车" not in panel.status.get()
        window.destroy()
        print(json.dumps({"status": "passed", "checks": 5, "entry_count": len(entries),
                          "hardware_started": False, "robot_commands_sent": 0}, ensure_ascii=False))
        return
    ProjectConsole(window, entries)
    window.mainloop()

if __name__ == "__main__":
    main()
