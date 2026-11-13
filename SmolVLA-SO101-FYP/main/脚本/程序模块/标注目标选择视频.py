"""双摄视频逐帧标注；只读已有视频，不打开摄像头、不连接机械臂。"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import queue
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import av
from PIL import Image, ImageTk

from 目标选择标注数据 import AnnotationStore, DatasetCatalog, MARKERS, derive_intervals

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT.parent / "AutoDL_新200段_4090D_20261102/训练数据"


def annotation_status(entry):
    return "已标注" if entry.get("markers") and entry.get("status") != "not_applicable" else "未标注"


class ExactVideoReader:
    """按PTS核对帧号，禁止把不精确的关键帧seek结果冒充所选帧。"""
    def __init__(self, catalog):
        self.catalog, self.episode = catalog, None
        self.containers, self.cache = {}, OrderedDict()

    def read(self, ep, index):
        meta = self.catalog.episodes[ep]
        if not 0 <= index < meta["length"]:
            raise ValueError("帧号超出范围")
        if self.episode != ep:
            self.close()
            self.episode = ep
        cache_key = (ep, index)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        images = []
        for key in self.catalog.keys:
            if key not in self.containers:
                container = av.open(str(meta["videos"][key]))
                container.streams.video[0].thread_count = 1
                self.containers[key] = container
            container = self.containers[key]
            stream = container.streams.video[0]
            origin = stream.start_time or 0
            target = origin + int(index / self.catalog.fps / stream.time_base)
            container.seek(target, stream=stream, backward=True, any_frame=False)
            selected = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise RuntimeError("视频缺少PTS，无法保证帧号准确")
                actual = round(float((frame.pts - origin) * stream.time_base) * self.catalog.fps)
                if actual == index:
                    selected = frame.to_image().convert("RGB")
                    break
                if actual > index:
                    break
            if selected is None:
                raise RuntimeError(f"视频{ep:06d}的{key}无法读取精确帧{index}")
            images.append(selected)
        self.cache[cache_key] = images
        while len(self.cache) > 32:
            self.cache.popitem(last=False)
        return images

    def close(self):
        for container in self.containers.values():
            container.close()
        self.containers.clear()
        self.cache.clear()


class FrameWorker:
    def __init__(self, catalog):
        self.requests, self.results = queue.Queue(maxsize=1), queue.Queue()
        self.end = threading.Event()
        self.reader = ExactVideoReader(catalog)
        self.thread = threading.Thread(target=self.run, daemon=True, name="精确视频解码")
        self.thread.start()

    def request(self, item):
        try:
            self.requests.get_nowait()
        except queue.Empty:
            pass
        self.requests.put_nowait(item)

    def run(self):
        try:
            while not self.end.is_set():
                try:
                    token, ep, index = self.requests.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    images = self.reader.read(ep, index)
                    self.results.put((token, ep, index, images, None))
                except Exception as exc:
                    self.results.put((token, ep, index, None, str(exc)))
        finally:
            self.reader.close()

    def close(self):
        self.end.set()
        self.thread.join(timeout=3)


class AnnotationWindow:
    def __init__(self, window, dataset=DEFAULT_DATASET, annotation=None):
        self.window = window
        self.catalog = self.store = self.worker = None
        self.current_ep, self.frame, self.displayed_frame = None, 0, None
        self.request_id, self.playing, self.updating_scale = 0, False, False
        self.play_generation = 0
        self.rendered_images, self.photos, self.dirty = None, [], False
        self.closed = False
        self.loader = ThreadPoolExecutor(max_workers=1)
        window.title("目标选择视频标注 · 单标记版 · 自动保存")
        window.geometry(f"{min(1460, window.winfo_screenwidth()-60)}x{min(990, window.winfo_screenheight()-70)}")
        window.minsize(1080, 790)
        self.status = tk.StringVar(value="正在读取数据目录……")
        self.title = tk.StringVar(value="尚未加载视频")
        self.path_text = tk.StringVar()
        self.frame_text = tk.StringVar()
        self.frame_entry = tk.StringVar(value="0")
        self.interval_text = tk.StringVar(value="每段只标1处，按1后自动保存，可直接导出。")
        self.filter_status = tk.StringVar(value="全部")
        self.speed = tk.StringVar(value="1×")
        outer = ttk.Frame(window, padding=10)
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text="目标选择视频标注", font=("Microsoft YaHei UI", 17, "bold")).pack(side="left")
        self.folder_button = ttk.Button(top, text="切换数据集…", command=self.choose_dataset)
        self.folder_button.pack(side="right")
        ttk.Button(top, text="输入数据集路径…", command=self.enter_dataset).pack(side="right", padx=6)
        ttk.Label(outer, textvariable=self.path_text, wraplength=1380).pack(anchor="w", pady=(4, 8))
        panes = ttk.Panedwindow(outer, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left = ttk.Frame(panes, width=205)
        panes.add(left, weight=1)
        self.progress = ttk.Label(left, text="正在加载……")
        self.progress.pack(anchor="w", pady=(0, 4))
        combo = ttk.Combobox(left, textvariable=self.filter_status, values=("全部", "未标注", "已标注"), state="readonly", width=19)
        combo.pack(fill="x", pady=(0, 5))
        combo.bind("<<ComboboxSelected>>", lambda _: self.refresh_list())
        listing = ttk.Frame(left)
        listing.pack(fill="both", expand=True)
        self.episode_list = tk.Listbox(listing, width=25, exportselection=False)
        self.episode_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(listing, command=self.episode_list.yview)
        scroll.pack(side="right", fill="y")
        self.episode_list.configure(yscrollcommand=scroll.set)
        self.episode_list.bind("<<ListboxSelect>>", self.select_from_list)
        ttk.Label(left, text="视频号从000000开始；\n“第1段”只表示列表顺序，\n不自动套用原录制JSON编号。", wraplength=200).pack(anchor="w", pady=7)
        right = ttk.Frame(panes, padding=(10, 0, 0, 0))
        panes.add(right, weight=5)
        ttk.Label(right, textvariable=self.title, font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        cameras = ttk.Frame(right)
        cameras.pack(fill="both", expand=True, pady=(5, 3))
        cameras.rowconfigure(0, weight=1)
        self.image_labels = []
        for column, label in enumerate(("桌面相机 front（主要依据）", "夹爪相机 wrist（辅助确认接触）")):
            cameras.columnconfigure(column, weight=1, uniform="camera")
            box = ttk.LabelFrame(cameras, text=label, padding=3)
            box.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5, 0))
            box.grid_propagate(False)
            box.rowconfigure(0, weight=1)
            box.columnconfigure(0, weight=1)
            image_label = tk.Label(box, bg="#20242b", fg="white", text="加载视频中……")
            image_label.grid(row=0, column=0, sticky="nsew")
            image_label.bind("<Configure>", lambda _: self.render_images())
            image_label.bind("<Button-1>", lambda _: self.window.focus_set())
            self.image_labels.append(image_label)
        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=3)
        for text, command in (("上一段 [PgUp]", lambda: self.switch_episode(-1)),
                              ("下一段 [PgDn]", lambda: self.switch_episode(1)),
                              ("前一帧 [←]", lambda: self.step(-1)),
                              ("后一帧 [→]", lambda: self.step(1))):
            ttk.Button(controls, text=text, command=command).pack(side="left", padx=(0, 4))
        self.play_button = ttk.Button(controls, text="播放 [空格]", command=self.toggle_play)
        self.play_button.pack(side="left", padx=4)
        ttk.Combobox(controls, textvariable=self.speed, values=("0.25×", "0.5×", "1×", "2×"), state="readonly", width=6).pack(side="left")
        ttk.Entry(controls, textvariable=self.frame_entry, width=6).pack(side="left", padx=(10, 3))
        ttk.Button(controls, text="跳到帧", command=self.jump_entry).pack(side="left")
        ttk.Label(right, textvariable=self.frame_text).pack(anchor="w")
        self.scale = ttk.Scale(right, from_=0, to=1, command=self.slider_changed)
        self.scale.pack(fill="x")
        self.scale.bind("<ButtonPress-1>", lambda _: self.pause())
        self.timeline = tk.Canvas(right, height=29, highlightthickness=0, background="#f1f4f8")
        self.timeline.pack(fill="x")
        self.timeline.bind("<Configure>", lambda _: self.draw_timeline())
        marker_bar = ttk.Frame(right)
        marker_bar.pack(fill="x", pady=4)
        self.marker_buttons = []
        for number, (key, (label, _)) in enumerate(MARKERS.items(), 1):
            button = ttk.Button(marker_bar, text=f"标记{label} [{number}]", command=lambda k=key: self.add_marker(k))
            button.pack(side="left", padx=(0, 6))
            self.marker_buttons.append(button)
        ttk.Button(marker_bar, text="删除选中标记 [Del]", command=self.delete_selected).pack(side="left")
        marker_area = ttk.Frame(right)
        marker_area.pack(fill="x")
        self.markers = ttk.Treeview(marker_area, columns=("label", "frame", "time"), show="headings", height=1, selectmode="browse")
        for name, text, width in (("label", "标记（双击跳转）", 210), ("frame", "帧号（从0开始）", 130), ("time", "视频时间", 120)):
            self.markers.heading(name, text=text)
            self.markers.column(name, width=width, stretch=True)
        self.markers.pack(fill="x")
        self.markers.bind("<Double-1>", lambda _: self.jump_marker())
        ttk.Label(right, textvariable=self.interval_text, wraplength=1100).pack(anchor="w", pady=(4, 2))
        help_box = ttk.LabelFrame(right, text="每段只标1处：首次接触目标 [1]；改错可覆盖或删除", padding=6)
        help_box.pack(fill="x", pady=(5, 0))
        for label, explanation in MARKERS.values():
            ttk.Label(help_box, text=label + "：" + explanation, wraplength=1060).pack(anchor="w")
        ttk.Label(help_box, text="按1立即保存，修改也自动保存；不需要核对勾选或逐段确认。看不清时可先跳到下一段。", wraplength=1060).pack(anchor="w", pady=(3, 0))
        ttk.Label(help_box, text="用途：增加接触前输入的抽样机会，仍学习完整后续动作；不是错抓惩罚。标完不会自动启动训练。", wraplength=1060).pack(anchor="w", pady=(3, 0))
        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Button(bottom, text="立即保存 [Ctrl+S]", command=self.save).pack(side="left")
        ttk.Button(bottom, text="导出已标注数据…", command=self.export).pack(side="left", padx=7)
        ttk.Button(bottom, text="打开标注所在文件夹", command=self.open_output).pack(side="left")
        ttk.Label(outer, textvariable=self.status, wraplength=1400).pack(anchor="w", pady=(5, 0))
        self.visible_ids = []
        for key, callback in (("<Left>", lambda: self.step(-1)), ("<Right>", lambda: self.step(1)),
                              ("<Prior>", lambda: self.switch_episode(-1)), ("<Next>", lambda: self.switch_episode(1)),
                              ("<space>", self.toggle_play), ("<Delete>", self.delete_selected),
                              ("<Control-s>", self.save)):
            window.bind(key, lambda event, cb=callback: self.shortcut(event, cb))
        for number, key in enumerate(MARKERS, 1):
            window.bind(str(number), lambda event, k=key: self.shortcut(event, lambda: self.add_marker(k)))
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = window.after(30, self.poll)
        self.load_dataset(dataset, annotation)

    def shortcut(self, event, callback):
        if event.widget.winfo_class() in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"):
            return None
        callback()
        return "break"

    def load_dataset(self, dataset, annotation=None):
        if getattr(self, "load_future", None) is not None:
            return
        if self.dirty:
            messagebox.showerror("尚未保存", "请先成功保存当前标注，再切换目录。", parent=self.window)
            return
        self.pause()
        if self.store:
            self.save()
            if self.dirty:
                return
        self.folder_button.configure(state="disabled")
        self.status.set("正在读取数据集及视频索引……")
        self.load_future = self.loader.submit(DatasetCatalog, dataset)
        self.pending_annotation = annotation

    def finish_load(self):
        future = self.load_future
        self.load_future = None
        new_store = None
        try:
            catalog = future.result()
            path = Path(self.pending_annotation) if self.pending_annotation else ROOT / "运行记录/目标选择时间标注" / f"首次接触_{catalog.fingerprint[:12]}.json"
            if self.store and path.resolve() == self.store.path:
                # 当前数据集重复选择时复用已有写锁，避免把自己当作第二个窗口。
                self.status.set("当前已打开该数据集，标注已保存。")
                self.folder_button.configure(state="normal")
                return
            legacy = ROOT / "运行记录/目标选择时间标注" / f"目标选择_{catalog.fingerprint[:12]}.json"
            new_store = AnnotationStore(catalog, path, legacy_path=legacy if not self.pending_annotation else None)
            if self.store:
                self.store.close()
            if self.worker:
                self.worker.close()
            self.catalog, self.store = catalog, new_store
            self.worker = FrameWorker(catalog)
            self.path_text.set(f"数据：{catalog.root}\n标注自动保存：{path}")
            cursor = new_store.data.get("cursor", {})
            ep = cursor.get("episode", catalog.ids[0])
            self.current_ep = None
            self.refresh_list()
            self.show_episode(ep if ep in catalog.episodes else catalog.ids[0], cursor.get("frame", 0))
        except Exception as exc:
            if new_store and new_store is not self.store:
                new_store.close()
            self.status.set("加载失败：" + str(exc))
            messagebox.showerror("无法打开数据", str(exc), parent=self.window)
        self.folder_button.configure(state="normal")

    def choose_dataset(self):
        selected = filedialog.askdirectory(title="切换数据集（可选择根目录或相机视频文件夹）", initialdir=str(self.catalog.root if self.catalog else DEFAULT_DATASET), parent=self.window)
        if selected:
            self.load_dataset(selected)

    def enter_dataset(self):
        from tkinter import simpledialog
        selected = simpledialog.askstring("输入数据集路径", "粘贴数据集根目录或相机视频文件夹路径：", parent=self.window)
        if selected and selected.strip():
            self.load_dataset(selected)

    def refresh_list(self):
        if not self.catalog:
            return
        selected_filter = self.filter_status.get()
        self.visible_ids = []
        self.episode_list.delete(0, "end")
        marked = 0
        for position, ep in enumerate(self.catalog.ids, 1):
            entry = self.store.entry(ep)
            status = annotation_status(entry)
            marked += status == "已标注"
            if selected_filter != "全部" and status != selected_filter:
                continue
            self.visible_ids.append(ep)
            self.episode_list.insert("end", f"{ep:06d}  {entry['target']}  {status} ({len(entry['markers'])}/1)")
        self.progress.configure(text=f"共{len(self.catalog.ids)}段｜已标注{marked}\n未标注{len(self.catalog.ids)-marked}｜标记后自动保存")
        if self.current_ep in self.visible_ids:
            row = self.visible_ids.index(self.current_ep)
            self.episode_list.selection_clear(0, "end")
            self.episode_list.selection_set(row)
            self.episode_list.see(row)

    def select_from_list(self, _=None):
        rows = self.episode_list.curselection()
        if rows and rows[0] < len(self.visible_ids):
            ep = self.visible_ids[rows[0]]
            if ep != self.current_ep:
                self.show_episode(ep)

    def switch_episode(self, offset):
        if self.current_ep is None or not self.catalog:
            return
        ids = self.visible_ids if self.current_ep in self.visible_ids else self.catalog.ids
        position = ids.index(self.current_ep) + offset
        if 0 <= position < len(ids):
            self.show_episode(ids[position])

    def show_episode(self, ep, frame=0):
        if self.dirty:
            self.status.set("保存尚未成功，请先保存后再切换视频。")
            return
        self.pause()
        self.current_ep = ep
        meta = self.catalog.episodes[ep]
        ordinal = self.catalog.ids.index(ep) + 1
        self.title.set(f"第{ordinal}/{len(self.catalog.ids)}段｜视频号 {ep:06d}｜标签 {meta['target']}（{'抓红' if meta['target']=='RC' else '抓蓝'}）")
        self.scale.configure(to=meta["length"] - 1)
        self.refresh_markers()
        self.refresh_list()
        self.seek(min(max(0, int(frame)), meta["length"] - 1))

    def seek(self, frame):
        if self.current_ep is None or not self.worker:
            return
        frame = min(max(int(frame), 0), self.catalog.episodes[self.current_ep]["length"] - 1)
        self.frame = frame
        self.request_id += 1
        self.displayed_frame = None
        for button in self.marker_buttons:
            button.configure(state="disabled")
        self.updating_scale = True
        self.scale.set(frame)
        self.updating_scale = False
        self.frame_entry.set(str(frame))
        self.frame_text.set(f"正在读取帧 {frame}……（画面更新完成后才允许打标记）")
        self.worker.request((self.request_id, self.current_ep, frame))

    def slider_changed(self, value):
        if not self.updating_scale:
            self.pause()
            target = int(round(float(value)))
            if target != self.frame or self.displayed_frame is None:
                self.seek(target)

    def step(self, delta):
        self.pause()
        self.seek(self.frame + delta)

    def jump_entry(self):
        self.pause()
        try:
            value = int(self.frame_entry.get())
            if not 0 <= value < self.catalog.episodes[self.current_ep]["length"]:
                raise ValueError("帧号超出当前视频范围")
            self.seek(value)
        except (ValueError, AttributeError, KeyError) as exc:
            self.status.set("请输入有效的整数帧号：" + str(exc))

    def pause(self):
        self.playing = False
        self.play_generation += 1
        if hasattr(self, "play_button"):
            self.play_button.configure(text="播放 [空格]")

    def toggle_play(self):
        if self.current_ep is None:
            return
        if self.playing:
            self.pause()
            return
        self.playing = True
        self.play_button.configure(text="暂停 [空格]")
        if self.displayed_frame is not None:
            generation = self.play_generation
            self.window.after(1, lambda: self.play_next(generation))

    def play_next(self, generation):
        if generation != self.play_generation or not self.playing or self.current_ep is None:
            return
        if self.frame + 1 >= self.catalog.episodes[self.current_ep]["length"]:
            self.pause()
            return
        self.seek(self.frame + 1)

    def render_images(self):
        if not self.rendered_images:
            return
        photos = []
        for label, original in zip(self.image_labels, self.rendered_images):
            image = original.copy()
            image.thumbnail((max(30, label.winfo_width()-4), max(30, label.winfo_height()-4)), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            photos.append(photo)
            label.configure(image=photo, text="")
        self.photos = photos

    def refresh_markers(self):
        if self.current_ep is None:
            return
        selected = self.markers.selection()
        self.markers.delete(*self.markers.get_children())
        entry = self.store.entry(self.current_ep)
        for key, (label, _) in MARKERS.items():
            frame = entry["markers"].get(key)
            self.markers.insert("", "end", iid=key, values=(label, "未标" if frame is None else frame,
                                      "—" if frame is None else f"{frame/self.catalog.fps:.3f}秒"))
        if selected and selected[0] in MARKERS:
            self.markers.selection_set(selected[0])
        try:
            intervals = derive_intervals(entry["markers"], entry["length"])
            bounds = intervals["pre_contact_window_starts_inclusive"]
            available = f"帧{bounds[0]}～{bounds[1]}" if bounds is not None else "无（接触在帧0或视频不足50帧；不补造动作）"
            self.interval_text.set(f"{annotation_status(entry)}｜首次接触：帧{intervals['first_contact_frame']}；接触前完整50步窗口的输入帧：{available}。后续动作可跨接触点。")
        except ValueError as exc:
            self.interval_text.set(f"{annotation_status(entry)}｜{exc}")
        self.draw_timeline()

    def draw_timeline(self):
        self.timeline.delete("all")
        if not self.catalog or self.current_ep is None:
            return
        width, height = max(20, self.timeline.winfo_width()), 29
        length = self.catalog.episodes[self.current_ep]["length"]
        colors = ("#216e46", "#b56a00", "#b52d42")
        for number, (key, color) in enumerate(zip(MARKERS, colors), 1):
            frame = self.store.entry(self.current_ep)["markers"].get(key)
            if frame is None:
                continue
            x = 10 + frame / max(1, length-1) * (width-20)
            self.timeline.create_line(x, 0, x, height, fill=color, width=2)
            text = self.timeline.create_text(min(max(x+28, 50), width-50), 7+(number % 2)*12, text=f"{number}:帧{frame}", fill=color)
            self.timeline.tag_bind(text, "<Button-1>", lambda _, f=frame: (self.pause(), self.seek(f)))

    def edit(self, callback):
        try:
            callback()
            self.dirty = False
            self.status.set("已自动保存：" + str(self.store.path))
        except Exception as exc:
            self.dirty = True
            self.status.set("未能完成保存／操作：" + str(exc))
            messagebox.showerror("操作未完成", str(exc), parent=self.window)
        self.refresh_markers()
        self.refresh_list()

    def add_marker(self, key):
        self.pause()
        if self.displayed_frame != self.frame or self.current_ep is None:
            self.status.set("画面正在加载，请等当前帧显示后再打标记。")
            return
        frame = self.displayed_frame
        self.edit(lambda: self.store.set_marker(self.current_ep, key, frame))
        self.markers.selection_set(key)

    def delete_selected(self):
        self.pause()
        selected = self.markers.selection()
        if self.current_ep is not None and selected:
            self.edit(lambda: self.store.delete_marker(self.current_ep, selected[0]))
        else:
            self.status.set("先在标记表选中要删除的一行；只删除标记，不删除视频。")

    def jump_marker(self):
        selected = self.markers.selection()
        if self.current_ep is not None and selected:
            frame = self.store.entry(self.current_ep)["markers"].get(selected[0])
            if frame is not None:
                self.pause()
                self.seek(frame)

    def save(self):
        if self.store:
            def persist():
                if self.current_ep is not None:
                    self.store.data["cursor"] = {"episode": self.current_ep, "frame": self.frame}
                self.store.save()
            self.edit(persist)

    def export(self):
        self.pause()
        if not self.store:
            return
        selected = filedialog.asksaveasfilename(title="导出已标注数据（未标注段不导出）", parent=self.window,
                    initialdir=str(self.store.path.parent), initialfile=self.store.path.stem + "_训练导出.json",
                    defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if selected:
            try:
                count = self.store.export(selected)
                self.status.set(f"已导出{count}段：{selected}。尚未启动或接入训练。")
            except Exception as exc:
                messagebox.showerror("无法导出", str(exc), parent=self.window)

    def open_output(self):
        if self.store:
            import os
            os.startfile(self.store.path.parent)

    def poll(self):
        if self.closed:
            return
        if getattr(self, "load_future", None) is not None and self.load_future.done():
            self.finish_load()
        if self.worker:
            while True:
                try:
                    token, ep, frame, images, error = self.worker.results.get_nowait()
                except queue.Empty:
                    break
                if token != self.request_id or ep != self.current_ep:
                    continue
                if error:
                    self.pause()
                    self.status.set("读取失败：" + error)
                    continue
                self.displayed_frame = frame
                self.rendered_images = images
                self.render_images()
                length = self.catalog.episodes[ep]["length"]
                self.frame_text.set(f"当前帧 {frame} / {length-1}（从0开始）｜{frame/self.catalog.fps:.3f}秒｜{self.catalog.fps:g} FPS｜按1标首次接触，共1处，可覆盖、删除")
                for button in self.marker_buttons:
                    button.configure(state="normal")
                if self.playing:
                    delay = max(1, int(1000 / self.catalog.fps / float(self.speed.get().rstrip("×"))))
                    generation = self.play_generation
                    self.window.after(delay, lambda g=generation: self.play_next(g))
        self.poll_id = self.window.after(25, self.poll)

    def close(self):
        self.pause()
        if self.store:
            self.save()
            if self.dirty:
                return
            self.store.close()
        self.closed = True
        self.window.after_cancel(self.poll_id)
        if self.worker:
            self.worker.close()
        self.loader.shutdown(wait=False, cancel_futures=True)
        self.window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--annotations", type=Path, help="可选：数据集目录之外的标注工作JSON")
    args = parser.parse_args()
    window = tk.Tk()
    AnnotationWindow(window, args.dataset_root, args.annotations)
    window.mainloop()


if __name__ == "__main__":
    main()
