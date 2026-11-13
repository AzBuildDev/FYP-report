"""User-controlled local camera preview; no robot connection or commands."""
import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import subprocess
import threading
import time
import traceback
from pathlib import Path
from 双摄采集配置 import CAMERAS, DEVICE_PROFILES

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "运行记录/相机测试窗口.log"
DEVICE_INDICES = {"PC Camera": 0, "1080P USB Camera": 1, "icspring camera": 2}


def named_capture(index, fourcc, fps, send, stop, device_name=None, output_fps=15,
                  *, width=640, height=480, log_path=None):
    """Negotiate the supported format before opening the named DirectShow device."""
    import imageio_ffmpeg
    import numpy as np
    names = {value: key for key, value in DEVICE_INDICES.items()}
    if index not in names:
        send("error", "按设备名采集支持编号 0、1、2；其他编号请选择 DirectShow。")
        return
    fmt = (["-vcodec", "mjpeg"] if fourcc == "MJPG" else ["-pixel_format", "yuyv422"])
    selected_device = device_name or names[index]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning", "-f", "dshow", "-rtbufsize", "32M", *fmt, "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", f"video={selected_device}", "-an", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    driver_log = Path(log_path) if log_path is not None else LOG
    with driver_log.open("ab") as log:
        log.write(f"\nSTART {selected_device}: {fourcc}, {width}x{height}, {fps} FPS\n".encode("utf-8"))
        log.flush()
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as exc:
            send("error", f"{selected_device} 启动采集进程失败：{exc}")
            return
        def read_frames():
            last_sent = 0.0
            frame_bytes = width * height * 3
            try:
                while not stop.is_set():
                    data = bytearray()
                    started = time.monotonic()
                    if log_path is not None:
                        send("read_started", {"monotonic": started})
                    while len(data) < frame_bytes:
                        part = process.stdout.read(frame_bytes - len(data))
                        if not part:
                            send("reader_end", {"reason": "stdout_eof", "received_bytes": len(data),
                                                "expected_bytes": frame_bytes, "returncode": process.poll(),
                                                "stop_requested": stop.is_set()})
                            return
                        data.extend(part)
                    now = time.monotonic()
                    if log_path is not None:
                        send("read_complete", {"duration_s": now-started, "bytes": len(data)})
                    if output_fps is None or now - last_sent >= 1 / output_fps:
                        send("frame", (now, np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()))
                        last_sent = now
            except Exception as exc:
                send("error", f"{selected_device} 读取图像异常：{type(exc).__name__}: {exc}")
        reader = threading.Thread(target=read_frames, daemon=True)
        reader.start()
        try:
            send("opened", {"fps_reported": fps, "pid": process.pid, "device": selected_device,
                            "driver_log": str(driver_log)})
            while not stop.wait(.05):
                if process.poll() is not None:
                    send("process_exit", {"pid": process.pid, "returncode": process.returncode})
                    send("error", f"{selected_device} 采集进程退出，返回码 {process.returncode}；驱动日志：{driver_log}")
                    break
                if not reader.is_alive():
                    send("error", f"{selected_device} 图像读取线程已结束；驱动日志：{driver_log}")
                    break
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            reader.join(timeout=.3)
            process.stdout.close()
            send("closed", {"pid": process.pid, "returncode": process.poll(),
                            "stop_requested": stop.is_set(), "reader_alive": reader.is_alive()})


def capture_worker(index, backend, fourcc, fps, messages, stop, device_name=None):
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
    import cv2

    def send(kind, payload):
        if kind not in ("frame", "opened", "error"):
            return
        try:
            messages.put_nowait((kind, payload))
        except queue.Full:
            pass

    cap = None
    try:
        spec = DEVICE_PROFILES.get(device_name, {})
        width, height = spec.get("width", 640), spec.get("height", 480)
        if backend == "按设备名":
            named_capture(index, fourcc, fps, send, stop, device_name, width=width, height=height)
            return
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if backend == "DirectShow" else cv2.CAP_MSMF)
        if not cap.isOpened():
            send("error", "无法打开相机。请检查编号，或关闭其他占用相机的程序。")
            return
        if fourcc != "自动":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        send("opened", {"fps_reported": cap.get(cv2.CAP_PROP_FPS)})
        failures = 0
        last_sent = 0.0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures == 10:
                    send("error", "相机已打开，但无法取得画面。可停止后换接口或采集方式重试。")
                time.sleep(0.05)
                continue
            failures = 0
            now = time.monotonic()
            if now - last_sent >= 1 / 15:
                send("frame", (now, frame))
                last_sent = now
    except Exception:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        send("error", "采集异常，详情已保存到本地日志。")
    finally:
        if cap is not None:
            cap.release()
        send("closed", None)


class CameraTestWindow:
    def __init__(self, window):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.window = tk, ttk, window
        self.ctx = mp.get_context("spawn")
        self.cameras = []
        self.closing = False
        self.stopping = False
        self.pending_start = None
        self.last_report = 0
        self.window.title("双相机测试 — 自己控制开启与停止")
        self.window.geometry("1160x700")
        self.window.minsize(900, 580)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        ttk.Label(window, text="相机测试", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w", padx=22, pady=(16, 4))
        ttk.Label(window, text="桌面：1080P USB Camera；夹爪：icspring camera。请核对两侧“设备”选择；PC Camera 是笔记本内置相机。", font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=22)
        toolbar = ttk.Frame(window)
        toolbar.pack(fill="x", padx=22, pady=14)
        self.order = tk.StringVar(value="桌面先开")
        ttk.Label(toolbar, text="双路启动顺序").pack(side="left", padx=(0, 8))
        ttk.Combobox(toolbar, textvariable=self.order, values=["桌面先开", "夹爪先开"], state="readonly", width=12).pack(side="left")
        ttk.Button(toolbar, text="开启两路", command=self.start_both).pack(side="left", padx=12)
        ttk.Button(toolbar, text="停止全部", command=self.stop_all).pack(side="left")
        ttk.Button(toolbar, text="打开检测记录", command=lambda: os.startfile(str(ROOT / "运行记录"))).pack(side="right")
        panels = ttk.Frame(window)
        panels.pack(fill="both", expand=True, padx=16)
        for number, (name, index, fourcc, fps) in enumerate([
            ("桌面相机 · 1080P USB Camera", int(os.environ.get("OVERHEAD_CAMERA", "1")), CAMERAS["front"]["fourcc"], CAMERAS["front"]["input_fps"]),
            ("夹爪相机 · icspring camera", int(os.environ.get("GRIPPER_CAMERA", "2")), CAMERAS["wrist"]["fourcc"], CAMERAS["wrist"]["input_fps"]),
        ]):
            panel = ttk.LabelFrame(panels, text=name, padding=10)
            panel.grid(row=0, column=number, sticky="nsew", padx=6)
            panels.columnconfigure(number, weight=1, uniform="camera")
            state = {"name": name, "device": tk.StringVar(value=name.split(" · ", 1)[1]), "index": tk.StringVar(value=str(index)), "backend": tk.StringVar(value="按设备名"), "fourcc": tk.StringVar(value=fourcc), "fps": tk.StringVar(value=str(fps)), "process": None, "photo": None, "last_frame": None, "frame_count": 0, "valid_count": 0, "image_std": 0, "last_valid": 0, "last_received": 0, "started": 0, "error": None}
            settings = ttk.Frame(panel)
            settings.pack(fill="x")
            ttk.Label(settings, text="设备").pack(side="left")
            controls = []
            control = ttk.Combobox(settings, textvariable=state["device"], values=list(DEVICE_INDICES), width=21, state="readonly")
            control.pack(side="left", padx=(4, 8))
            controls.append(control)
            for key, values, width in [("backend", ["按设备名", "DirectShow", "MSMF"], 11), ("fourcc", ["MJPG", "YUY2", "自动"], 6), ("fps", ["5", "15", "20", "25", "30"], 4)]:
                control = ttk.Combobox(settings, textvariable=state[key], values=values, width=width, state="readonly")
                control.pack(side="left", padx=3)
                controls.append(control)
            state["controls"] = controls
            buttons = ttk.Frame(panel)
            buttons.pack(fill="x", pady=10)
            ttk.Button(buttons, text="开启此相机", command=lambda s=state: self.start(s)).pack(side="left")
            ttk.Button(buttons, text="停止", command=lambda s=state: self.stop(s)).pack(side="left", padx=6)
            ttk.Button(buttons, text="保存当前画面", command=lambda s=state: self.save_frame(s)).pack(side="left")
            state["status"] = tk.StringVar(value="未开启")
            ttk.Label(panel, text="当前采集分辨率：桌面 640×480，夹爪 640×480；预览显示频率独立于录制。").pack(anchor="w", pady=(0, 4))
            ttk.Label(panel, textvariable=state["status"], wraplength=500).pack(anchor="w", pady=(0, 8))
            state["preview"] = tk.Label(panel, text="等待开启相机", background="#20252d", foreground="#dce3ed", font=("Microsoft YaHei UI", 13))
            state["preview"].pack(fill="both", expand=True)
            self.cameras.append(state)
        panels.rowconfigure(0, weight=1)
        self.summary = tk.StringVar(value="准备就绪。换 USB 接口后，如果画面不对应，可修改相机编号再开启。")
        ttk.Label(window, textvariable=self.summary, wraplength=1100, font=("Microsoft YaHei UI", 10)).pack(fill="x", padx=22, pady=16)
        window.after(80, self.poll)

    def start(self, state):
        if self.closing or self.stopping or state["process"] is not None:
            return
        try:
            device_name = state["device"].get()
            index, fps = DEVICE_INDICES[device_name], int(state["fps"].get())
            state["index"].set(str(index))
            if not 0 <= index <= 9 or not 1 <= fps <= 60:
                raise ValueError
        except ValueError:
            state["status"].set("请填写有效的编号（0–9）和帧率（1–60）。")
            return
        if any(other is not state and other["process"] is not None and other["device"].get() == device_name for other in self.cameras):
            state["status"].set("另一侧已经开启这台设备，请选择不同的相机。")
            return
        state.update(messages=self.ctx.Queue(maxsize=3), stop=self.ctx.Event(), frame_count=0, valid_count=0, image_std=0, last_valid=0, last_received=0, started=time.monotonic(), error=None, last_frame=None)
        process = self.ctx.Process(target=capture_worker, args=(index, state["backend"].get(), state["fourcc"].get(), fps, state["messages"], state["stop"], device_name), daemon=True)
        state["process"] = process
        state["status"].set("正在开启…")
        state["preview"].configure(image="", text="正在等待画面…")
        for control in state["controls"]:
            control.configure(state="disabled")
        process.start()

    def start_both(self):
        if self.closing or self.stopping:
            return
        if self.pending_start is not None:
            self.window.after_cancel(self.pending_start)
        order = list(self.cameras)
        if self.order.get() == "夹爪先开":
            order.reverse()
        self.start(order[0])
        # Give the first device time to initialize without blocking the window.
        def start_second():
            self.pending_start = None
            self.start(order[1])
        self.pending_start = self.window.after(1200, start_second)

    def stop(self, state):
        if self.pending_start is not None:
            self.window.after_cancel(self.pending_start)
            self.pending_start = None
        process = state["process"]
        if process is None or state.get("stop_deadline") is not None:
            return
        state["stop"].set()
        state["stop_deadline"] = time.monotonic() + 3
        state["status"].set("正在停止并释放相机…")

    def stop_all(self):
        self.stopping = True
        for state in self.cameras:
            self.stop(state)

    def finish(self, state):
        state["process"].join(timeout=0)
        state["process"].close()
        state["messages"].close()
        state["process"] = None
        state["stop_deadline"] = None
        state["status"].set(state["error"] or "已停止，相机已释放")
        state["preview"].configure(image="", text="已停止")
        state["photo"] = None
        for control in state["controls"]:
            control.configure(state="readonly")
        self.write_report()

    def paint(self, state, frame):
        from PIL import Image, ImageTk
        state["last_frame"] = frame
        state["image_std"] = float(frame.std())
        state["image_mean"] = float(frame.mean())
        image = Image.fromarray(frame[:, :, ::-1])
        image.thumbnail((max(320, state["preview"].winfo_width()), max(240, state["preview"].winfo_height())))
        state["photo"] = ImageTk.PhotoImage(image)
        state["preview"].configure(image=state["photo"], text="")

    def poll(self):
        now = time.monotonic()
        for state in self.cameras:
            process = state["process"]
            if process is None:
                continue
            # Do not read a pipe while terminating a worker that might be
            # publishing a frame. Closing that queue discards pending previews.
            if state.get("stop_deadline"):
                if now > state["stop_deadline"] and process.is_alive():
                    process.terminate()
                if not process.is_alive():
                    self.finish(state)
                continue
            latest = None
            try:
                while True:
                    kind, payload = state["messages"].get_nowait()
                    if kind == "frame":
                        state["frame_count"] += 1
                        state["last_received"] = now
                        latest = payload[1]
                        state["error"] = None
                    elif kind == "error":
                        state["error"] = payload
                        state["status"].set(payload)
                    elif kind == "opened":
                        state["reported_fps"] = payload["fps_reported"]
            except queue.Empty:
                pass
            if latest is not None and not state.get("stop_deadline"):
                self.paint(state, latest)
                h, w = latest.shape[:2]
                if state["image_std"] <= 1 and state["image_mean"] < 3:
                    state["status"].set(f"收到 {state['frame_count']} 帧，但画面全黑 · {w}×{h}，请停止后更换采集方式。")
                else:
                    state["valid_count"] += 1
                    state["last_valid"] = now
                    state["status"].set(f"正在预览 · {w}×{h} · 收到 {state['frame_count']} 帧")
            if state.get("stop_deadline") and now > state["stop_deadline"] and process.is_alive():
                process.terminate()
            if not process.is_alive():
                self.finish(state)
            elif not state.get("stop_deadline") and now - (state["last_received"] or state["started"]) > 8:
                state["status"].set(state["error"] or "暂未收到有效画面，可点击停止后重试。")
        active = [s for s in self.cameras if s["process"] is not None]
        fresh = [s for s in active if s["valid_count"] >= 5 and s["last_valid"] and now - s["last_valid"] < 2 and not s.get("stop_deadline")]
        if len(fresh) == 2:
            self.summary.set("两路正在同时出画面。停止后会保存本次检测记录。")
            if not getattr(self, "both_seen", False):
                self.both_seen = True
                self.write_report()
        elif active:
            self.summary.set(f"已开启 {len(active)} 路，当前有效画面 {len(fresh)} 路。可自行停止、更改设置后重试。")
        else:
            self.stopping = False
            self.summary.set("相机均已停止。单路测试请只开启一侧；可以按设备名选择相机，再测试两路同时开启。")
        if active and now - self.last_report > 2:
            self.write_report()
            self.last_report = now
        if self.closing and not active:
            self.window.destroy()
            return
        self.window.after(80, self.poll)

    def write_report(self):
        report = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "both_preview_observed_in_session": getattr(self, "both_seen", False), "cameras": [{"name": s["name"], "selected_device": s["device"].get(), "index": s["index"].get(), "backend": s["backend"].get(), "format": s["fourcc"].get(), "requested_fps": s["fps"].get(), "preview_frames_received": s["frame_count"], "nonblack_previews": s["valid_count"], "last_image_std": s["image_std"], "error": s["error"]} for s in self.cameras]}
        (ROOT / "运行记录/相机窗口检查.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_frame(self, state):
        if state["last_frame"] is None:
            state["status"].set("先开启相机，收到画面后再保存。")
            return
        import cv2
        folder = ROOT / "运行记录/相机画面"
        folder.mkdir(exist_ok=True)
        path = folder / f"manual_camera_{state['index'].get()}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        if cv2.imwrite(str(path), state["last_frame"]):
            self.summary.set(f"画面已保存：{path.name}（点击“打开检测记录”查看）")

    def close(self):
        self.closing = True
        self.stop_all()


def main():
    import tkinter as tk
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    window = tk.Tk()
    app = CameraTestWindow(window)
    if args.smoke_test:
        import numpy as np
        window.update()
        app.paint(app.cameras[0], np.zeros((480, 640, 3), dtype=np.uint8))
        window.after(300, app.close)
    try:
        window.mainloop()
    finally:
        for state in app.cameras:
            process = state["process"]
            if process is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                process.close()
                state["messages"].close()


if __name__ == "__main__":
    mp.freeze_support()
    try:
        main()
    except Exception:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        raise
