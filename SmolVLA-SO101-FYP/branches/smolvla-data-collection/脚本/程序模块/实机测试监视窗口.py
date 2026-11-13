"""独立进程中的主线程创建 Tk 窗口；只交换预览、文字和停止请求，不访问机械臂。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from io import BytesIO


def replace_preview(source, target):
    # Windows 上短暂读取预览可能阻止替换；只在预览线程有限重试。
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01)


def run_monitor(root, get_frame, camera_stop, user_stop, get_text, *, hidden=False, test_stop_after=0):
    import cv2
    temp_root = (root / "运行环境/临时文件").resolve()
    if not temp_root.is_relative_to(root.resolve()):
        raise ValueError("预览临时目录超出项目")
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="实机监视预览_", dir=temp_root) as temporary:
        directory = Path(temporary)
        command = [sys.executable, str(Path(__file__).resolve()), "--directory", str(directory)]
        if hidden:
            command.append("--hidden")
        if test_stop_after:
            command.extend(["--software-test-stop-after", str(test_stop_after)])
        with (directory / "窗口输出.log").open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while not camera_stop.is_set():
                    if (directory / "请求停止").exists():
                        user_stop.set()
                        return
                    if process.poll() is not None:
                        raise RuntimeError("监视窗口异常退出：" + (directory / "窗口输出.log").read_text(encoding="utf-8", errors="replace")[-800:])
                    _, image = get_frame()
                    if image is not None:
                        ok, encoded = cv2.imencode(".jpg", image)
                        if ok:
                            (directory / "预览暂存.jpg").write_bytes(encoded.tobytes())
                            replace_preview(directory / "预览暂存.jpg", directory / "当前预览.jpg")
                    (directory / "状态暂存.json").write_text(json.dumps({"text": get_text()}, ensure_ascii=False), encoding="utf-8")
                    replace_preview(directory / "状态暂存.json", directory / "当前状态.json")
                    camera_stop.wait(0.1)
            finally:
                (directory / "控制已结束").touch()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()  # 只关闭本函数创建的预览进程，硬件由控制主线程释放。
                    process.wait(timeout=2)


def window_main(directory, hidden=False, test_stop_after=0):
    import tkinter as tk
    from PIL import Image, ImageTk
    window = tk.Tk()
    window.title("实机抓取测试监视")
    if hidden:
        window.withdraw()
    label_img = tk.Label(window)
    label_img.pack()
    label_txt = tk.Label(window, font=("Consolas", 10), justify="left", anchor="w")
    label_txt.pack(fill="x")

    def stop():
        (directory / "请求停止").touch()
        window.destroy()

    tk.Button(window, text="停止测试", command=stop).pack(fill="x")
    window.protocol("WM_DELETE_WINDOW", stop)

    def refresh():
        if (directory / "控制已结束").exists():
            window.destroy()
            return
        try:
            with Image.open(BytesIO((directory / "当前预览.jpg").read_bytes())) as source:
                photo = ImageTk.PhotoImage(source.copy())
            label_img.configure(image=photo)
            label_img.image = photo
            label_txt.configure(text=json.loads((directory / "当前状态.json").read_text(encoding="utf-8"))["text"])
        except (OSError, ValueError):
            pass
        window.after(100, refresh)

    refresh()
    if test_stop_after:
        window.after(int(test_stop_after * 1000), stop)
    window.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--software-test-stop-after", type=float, default=0)
    options = parser.parse_args()
    window_main(options.directory, options.hidden, options.software_test_stop_after)
