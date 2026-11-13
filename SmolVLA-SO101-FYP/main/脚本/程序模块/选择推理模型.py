"""选择本机 SmolVLA 检查点和演示；仅在内存中修正处理器路径，不改训练文件。"""
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = ("config.json", "model.safetensors", "dataset_stats.json")
PROCESSOR_FILES = ("config.json", "processor_config.json", "preprocessor_config.json",
                   "tokenizer_config.json", "tokenizer.json")
CAMERA_KEYS = {"observation.images.front", "observation.images.wrist"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_models(root=ROOT):
    """包括模型包内的多层检查点；同名模型按完整相对目录区分。"""
    found = set()
    for directory in (Path(root) / "模型", Path(root) / "models"):
        if directory.is_dir():
            for config in directory.rglob("config.json"):
                path = config.parent.resolve()
                if path.is_relative_to(directory.resolve()) and all((path / name).is_file() for name in MODEL_FILES):
                    found.add(path)
    return sorted(found)


def processor_path(model_dir, override=None, root=ROOT):
    model_dir = Path(model_dir).resolve()
    config = read_json(model_dir / "config.json")
    if override:
        candidates = [Path(override).expanduser().resolve()]
    else:
        configured = Path(config.get("vlm_model_name", ""))
        candidates = [model_dir / "视觉语言处理器", model_dir.parent / "视觉语言处理器"]
        if str(configured) not in ("", "."):
            candidates.append(configured if configured.is_absolute() else model_dir / configured)
        candidates.append(Path(root) / "模型/视觉语言处理器")
    for path in candidates:
        if all((path / name).is_file() for name in PROCESSOR_FILES):
            return path.resolve()
    raise ValueError("找不到完整的本机视觉语言处理器，请选择包含 tokenizer.json、processor_config.json 等文件的目录。")


def infer_dataset(model_dir, root=ROOT):
    record = Path(model_dir) / "training_provenance.json"
    if not record.is_file():
        return None
    recorded = read_json(record).get("training_data", {}).get("dataset_root", "")
    if not recorded:
        return None
    # 朋友电脑的盘符只用于识别批次名，不复制、创建或改写该路径。
    name = recorded.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    candidates = [Path(root).parent / "训练数据采集/机械臂控制项目" / section / name
                  for section in ("任务演示", "历史演示")]
    # AutoDL 导出的记录指向 Linux 项目；本机部署包保留同名项目及数据目录。
    recorded_parts = recorded.replace("\\", "/").rstrip("/").split("/")
    if len(recorded_parts) >= 2:
        candidates.append(Path(root).parent / recorded_parts[-2] / name)
    candidates.append(Path(recorded))
    return next((path.resolve() for path in candidates if (path / "meta/info.json").is_file()), None)


@dataclass(frozen=True)
class InferenceSelection:
    model: Path
    dataset: Path
    processor: Path
    experiment: Path | None = None
    checkpoint_name: str = ""


@dataclass(frozen=True)
class ModelOption:
    """界面中的一级实验目录和二级 checkpoint。"""
    experiment: Path
    checkpoint: Path
    checkpoint_name: str


def discover_model_options(root=ROOT):
    """按实验目录分组模型；旧项目的单目录模型也作为单 checkpoint 兼容。"""
    checkpoints = discover_models(root)
    by_parent = {}
    for checkpoint in checkpoints:
        by_parent.setdefault(checkpoint.parent, []).append(checkpoint)
    options = []
    for checkpoint in checkpoints:
        siblings = by_parent[checkpoint.parent]
        # 同一目录下有多个完整模型时，把父目录作为实验目录；这是新训练包和旧的
        # “3000步模型/10000步模型”目录都能自然适配的分组方式。
        experiment = checkpoint.parent if len(siblings) > 1 else checkpoint
        options.append(ModelOption(experiment, checkpoint, checkpoint.name))
    return sorted(options, key=lambda item: (str(item.experiment).casefold(), str(item.checkpoint).casefold()))


def model_option_for_path(model_dir, root=ROOT):
    model = Path(model_dir).expanduser().resolve()
    for option in discover_model_options(root):
        if option.checkpoint.resolve() == model:
            return option
    return ModelOption(model, model, model.name)


def validate_selection(model_dir, dataset_root=None, processor_dir=None, root=ROOT):
    if not model_dir or not str(model_dir).strip():
        raise ValueError("尚未选择模型。请先在上方选择模型实验目录，再选择具体 Checkpoint；不能只选模型总目录。")
    model = Path(model_dir).expanduser().resolve()
    missing = [name for name in MODEL_FILES if not (model / name).is_file()]
    if missing:
        raise ValueError("模型目录缺少：" + "、".join(missing) + "；请选择具体检查点，不要选择外层模型包。")
    dataset = Path(dataset_root).expanduser().resolve() if dataset_root else infer_dataset(model, root)
    if dataset is None or not (dataset / "meta/info.json").is_file():
        raise ValueError("找不到对应演示，请选择包含 meta、data、videos 的数据集目录。")
    if not any(dataset.glob("data/chunk-*/*.parquet")):
        raise ValueError("所选数据集没有关节数据，无法计算安全活动范围。")
    config, info = read_json(model / "config.json"), read_json(dataset / "meta/info.json")
    inputs = config.get("input_features", {})
    image_keys = {key for key, spec in inputs.items() if spec.get("type") == "VISUAL"}
    features = info.get("features", {})
    video_keys = {key for key in features if key.startswith("observation.images.")}
    if config.get("type") != "smolvla" or image_keys != CAMERA_KEYS or video_keys != CAMERA_KEYS:
        raise ValueError("当前实机入口需要 SmolVLA 桌面 front、夹爪 wrist 双摄模型及对应双摄演示。")
    if inputs.get("observation.state", {}).get("shape") != [6] or config.get("output_features", {}).get("action", {}).get("shape") != [6]:
        raise ValueError("模型必须输入六关节状态并输出六关节动作。")
    names = features.get("action", {}).get("names", [])
    if info.get("fps") != 20 or len(names) != 6 or names != features.get("observation.state", {}).get("names"):
        raise ValueError("演示必须为 20 FPS，且动作与状态的六关节顺序一致。")
    record = model / "training_provenance.json"
    if record.is_file():
        training = read_json(record).get("training_data", {})
        if training.get("joint_names", names) != names or training.get("state_names", names) != names:
            raise ValueError("所选演示与模型训练记录的关节顺序不同。")
    option = model_option_for_path(model, root)
    return InferenceSelection(model, dataset, processor_path(model, processor_dir, root),
                              option.experiment, option.checkpoint_name)


def recent_selection(root=ROOT):
    """恢复上次使用的具体检查点；失效时留空，不猜另一份权重。"""
    root = Path(root)
    saved = root / '运行环境/私密配置/推理选择.json'
    try:
        if saved.is_file():
            record = read_json(saved)
        else:
            logs = list((root / '运行记录').glob('实机抓取测试_*.jsonl'))
            if not logs:
                return None
            latest = max(logs, key=lambda path: path.stat().st_mtime_ns)
            record = None
            with latest.open(encoding='utf-8') as log:
                for index, line in enumerate(log):
                    if index >= 20:
                        break
                    event = json.loads(line)
                    if event.get('type') == 'model_identity':
                        record = event
                        break
            if record is None:
                return None
        return validate_selection(record['model_path'], record['dataset_path'],
                                  record['processor_path'], root)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def remember_selection(chosen, root=ROOT):
    path = Path(root) / '运行环境/私密配置/推理选择.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(model_path=str(chosen.model), dataset_path=str(chosen.dataset),
                  processor_path=str(chosen.processor))
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_local_policy(model_dir, processor_dir=None, device="cuda"):
    """强制离线、严格加载；训练包中的 config.json（包括原盘符）保持原样。"""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    model = Path(model_dir).resolve()
    processor = processor_path(model, processor_dir)
    config = PreTrainedConfig.from_pretrained(model, local_files_only=True)
    config.vlm_model_name = str(processor)
    config.device = device
    return SmolVLAPolicy.from_pretrained(model, config=config, local_files_only=True, strict=True).to(device).eval()


class ModelSelectionWindow:
    def __init__(self, window, root=ROOT, model=None, dataset=None, processor=None):
        import tkinter as tk
        from tkinter import ttk

        self.window, self.root, self.result = window, Path(root), None
        window.title("选择推理模型与对应演示")
        window.geometry("1080x650")
        self.model, self.dataset, self.processor = (tk.StringVar(value=str(value) if value else "")
                                                   for value in (model, dataset, processor))
        self.experiment = tk.StringVar()
        self.checkpoint = tk.StringVar()
        self._syncing_selection = False
        self.options = discover_model_options(root)
        self.options_by_experiment = {}
        for option in self.options:
            self.options_by_experiment.setdefault(option.experiment.resolve(), []).append(option)
        frame = ttk.Frame(window, padding=18)
        self.content = frame
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="选择本次使用的模型", font=("Microsoft YaHei", 17, "bold")).grid(row=0, column=0, sticky="w", pady=6)
        ttk.Label(frame, text="演示数据用于计算活动范围。选择模型后会按训练记录寻找本机批次，请核对。", wraplength=1000).grid(row=1, column=0, sticky="w")
        self.combo = None
        self.experiment_combo = ttk.Combobox(frame, textvariable=self.experiment,
                                             values=[str(path) for path in self.options_by_experiment], state="readonly")
        self.checkpoint_combo = ttk.Combobox(frame, textvariable=self.checkpoint, state="readonly")
        self.model_entry = ttk.Entry(frame, textvariable=self.model)
        fields = (("模型实验目录（先选训练实验）", self.experiment_combo, None),
                  ("Checkpoint（再选 best_val、last 或训练步数）", self.checkpoint_combo, None),
                  ("具体模型目录（最终传给推理的路径）", self.model_entry, self.model),
                  ("对应演示目录（包含 meta、data、videos）", ttk.Entry(frame, textvariable=self.dataset), self.dataset),
                  ("视觉语言处理器目录（留空自动查找）", ttk.Entry(frame, textvariable=self.processor), self.processor))
        for index, (label, entry, variable) in enumerate(fields):
            label_row = 2 + index * 2
            ttk.Label(frame, text=label).grid(row=label_row, column=0, sticky="w", pady=(10, 2))
            entry.grid(row=label_row + 1, column=0, sticky="ew", padx=(0, 8))
            if variable is not None:
                ttk.Button(frame, text="浏览文件夹…", command=lambda value=variable: self.browse(value)).grid(row=label_row + 1, column=1)
        self.combo = self.model_entry
        self.experiment_combo.bind("<<ComboboxSelected>>", lambda _: self.experiment_changed())
        self.checkpoint_combo.bind("<<ComboboxSelected>>", lambda _: self.checkpoint_changed())
        self.model_entry.bind("<FocusOut>", lambda _: self.model_changed())
        self.selection_notice_row = 12
        self.selection_buttons_row = 13
        self.last_model = None
        if model:
            self.model_changed()
        if dataset:
            self.dataset.set(str(dataset))
        if processor:
            self.processor.set(str(processor))
        self.notice = ttk.Label(frame, text="继续后先加载和检查；实机运动仍需在终端按回车确认。关闭此窗口即可取消。", wraplength=1000)
        self.notice.grid(row=self.selection_notice_row, column=0, sticky="w", pady=16)
        buttons = ttk.Frame(frame)
        buttons.grid(row=self.selection_buttons_row, column=0, sticky="e")
        self.cancel_button = ttk.Button(buttons, text="取消", command=window.destroy)
        self.cancel_button.pack(side="left", padx=10)
        self.confirm_button = ttk.Button(buttons, text="使用所选模型继续", command=self.confirm)
        self.confirm_button.pack(side="left")
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        window.bind("<Escape>", lambda _: window.destroy())

    def experiment_changed(self):
        if self._syncing_selection:
            return
        path = Path(self.experiment.get()).expanduser().resolve() if self.experiment.get() else None
        options = self.options_by_experiment.get(path, [])
        self.checkpoint_combo.configure(values=[option.checkpoint_name for option in options])
        if not options:
            self.checkpoint.set("")
            return
        preferred = next((option for option in options if option.checkpoint_name == "best_val"), None)
        preferred = preferred or next((option for option in options if option.checkpoint_name == "last"), None)
        preferred = preferred or sorted(options, key=lambda option: option.checkpoint_name)[-1]
        self.checkpoint.set(preferred.checkpoint_name)
        self.checkpoint_changed()

    def checkpoint_changed(self):
        if self._syncing_selection:
            return
        path = Path(self.experiment.get()).expanduser().resolve() if self.experiment.get() else None
        option = next((item for item in self.options_by_experiment.get(path, [])
                       if item.checkpoint_name == self.checkpoint.get()), None)
        if option is None:
            return
        self._syncing_selection = True
        try:
            self.model.set(str(option.checkpoint))
        finally:
            self._syncing_selection = False
        self.model_changed()

    def model_changed(self):
        if self._syncing_selection:
            return
        if self.last_model == self.model.get():
            return
        self.last_model = self.model.get()
        if not self.model.get():
            return
        option = model_option_for_path(self.model.get(), self.root)
        self._syncing_selection = True
        try:
            self.experiment.set(str(option.experiment))
            options = self.options_by_experiment.get(option.experiment.resolve(), [option])
            self.checkpoint_combo.configure(values=[item.checkpoint_name for item in options])
            self.checkpoint.set(option.checkpoint_name)
        finally:
            self._syncing_selection = False
        try:
            inferred = infer_dataset(self.model.get(), self.root)
        except (OSError, ValueError, TypeError):
            inferred = None
        self.dataset.set(str(inferred) if inferred else "")
        self.processor.set("")

    def browse(self, variable):
        from tkinter import filedialog
        selected = filedialog.askdirectory(parent=self.window, title="选择文件夹", initialdir=str(self.root), mustexist=True)
        if selected:
            variable.set(selected)
            if variable is self.model:
                self.model_changed()
            elif variable is self.experiment:
                self.experiment_changed()

    def confirm(self):
        from tkinter import messagebox
        try:
            self.result = validate_selection(self.model.get(), self.dataset.get(), self.processor.get(), self.root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("请检查所选文件", str(exc), parent=self.window)
            return
        self.window.destroy()


def choose_model(root=ROOT, model=None, dataset=None, processor=None):
    import tkinter as tk
    window = tk.Tk()
    selection = ModelSelectionWindow(window, root, model, dataset, processor)
    window.mainloop()
    return selection.result


def add_selection_arguments(parser):
    parser.add_argument("--model-dir", type=Path, help="本机模型目录；不指定时打开选择窗口")
    parser.add_argument("--dataset-root", type=Path, help="对应演示目录；可按模型训练记录寻找本机批次")
    parser.add_argument("--processor-dir", type=Path, help="可选：本机视觉语言处理器目录")


def selection_from_args(args, root=ROOT):
    if args.model_dir:
        return validate_selection(args.model_dir, args.dataset_root, args.processor_dir, root)
    return choose_model(root, dataset=args.dataset_root, processor=args.processor_dir)
