"""发现本项目批次和模型、构建软件任务并异步跟踪；不包含机械臂控制任务。"""
import json
import math
import os
import queue
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from 管理训练数据来源 import file_sha256, load_training_source


@dataclass(frozen=True)
class Batch:
    path: Path
    label: str
    episodes: tuple
    ready: bool
    description: str


@dataclass(frozen=True)
class JobSpec:
    kind: str
    argv: tuple
    directory: Path
    report: Path
    dataset: Path
    episodes: tuple
    model: Path | None = None
    require_held_out: bool = False
    steps: int = 0
    batch_size: int = 0


class ProjectTasks:
    SOFTWARE = {"check": "检查演示数据.py", "train_plan": "微调预训练模型.py", "train": "微调预训练模型.py",
                "eval_plan": "评估微调模型.py", "eval": "评估微调模型.py"}
    OPERATOR_WINDOWS = {"录制任务演示.py", "实机结果记录窗口.py"}

    def __init__(self, root, interpreter=None):
        self.root = Path(root).resolve()
        self.data = self.root.parent / "训练数据采集/机械臂控制项目"
        self.python = Path(interpreter) if interpreter else self.root / "运行环境/虚拟环境/Scripts/python.exe"
        self.logs = self.root / "运行记录/采集训练面板"
        self.operator_processes = {}

    def batches(self):
        result = []
        for section in ("任务演示", "历史演示"):
            directory = self.data / section
            if not directory.exists():
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_dir() or not path.resolve().is_relative_to(self.data.resolve()):
                    continue
                label = f"{section}/{path.name}"
                try:
                    if (path / "批次封存状态.json").exists():
                        raise ValueError("保存或设备清理异常后封存，需人工检查")
                    info = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
                    episodes = [json.loads(line) for line in (path / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                    if (not episodes or [row["episode_index"] for row in episodes] != list(range(info["total_episodes"])) or
                        any(type(row["length"]) is not int or row["length"] < 1 for row in episodes) or
                        sum(row["length"] for row in episodes) != info["total_frames"]):
                        raise ValueError("段数或帧数元数据不一致")
                    names = info["features"]["action"]["names"]
                    if info["fps"] != 20 or len(names) != 6 or names != info["features"]["observation.state"]["names"]:
                        raise ValueError("需要 20 FPS、六关节且动作/状态顺序一致")
                    result.append(Batch(path.resolve(), label, tuple((row["episode_index"], row["length"]) for row in episodes), True,
                                        f"{len(episodes)} 段 / {info['total_frames']} 帧；仅元数据可读，需运行数据检查"))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    result.append(Batch(path.resolve(), label, (), False, f"未就绪：{exc}"))
        return result

    def models(self):
        directory = self.root / "模型"
        if not directory.exists():
            return []
        return [path.resolve() for path in sorted(directory.iterdir())
                if path.is_dir() and path.resolve().is_relative_to(directory.resolve()) and
                all((path / filename).is_file() for filename in ("config.json", "model.safetensors", "dataset_stats.json"))]

    def build(self, kind, dataset, episodes=(), steps=3000, batch_size=4, name="红色方块微调", model=None, samples=3, require_held_out=True):
        if kind not in self.SOFTWARE:
            raise ValueError("面板只允许数据检查、训练计划、训练和离线评估任务")
        dataset = Path(dataset).resolve()
        batch = next((item for item in self.batches() if item.path == dataset), None)
        if batch is None or not batch.ready:
            raise ValueError("请选择本项目已保存且元数据完整的演示批次")
        if any(type(ep) is not int for ep in episodes):
            raise ValueError("演示段号必须为整数")
        selected = tuple(sorted(episodes))
        lengths = dict(batch.episodes)
        if kind != "check" and (not selected or len(selected) != len(set(selected)) or any(ep not in lengths for ep in selected)):
            raise ValueError("至少选择一段完整演示，不能重复或引用不存在的段")
        if kind != "check" and any(lengths[ep] < 50 for ep in selected):
            raise ValueError("所选演示每段至少需要 50 帧")
        if not self.python.is_file():
            raise ValueError("项目 Python 环境缺失，请先安装本地环境")
        run_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}"
        directory = self.logs / ("任务_" + run_id)
        report = directory / "任务结果.json"
        args = [str(self.python), "-u", str(self.root / "脚本/程序模块" / self.SOFTWARE[kind]), "--dataset-root", str(dataset),
                "--output-report", str(report)]
        selected_model = None
        if kind.startswith("train"):
            config_path = dataset / "采集配置.json"
            if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")).get("split") == "held_out":
                raise ValueError("独立验证批次不能用于训练；请选训练批次")
            if type(steps) is not int or steps < 1 or type(batch_size) is not int or not 1 <= batch_size <= 16:
                raise ValueError("训练步数须为正整数，批量须在 1 至 16 之间")
            name = name.strip()
            if not name or len(name) > 40 or not re.search(r"[\u4e00-\u9fff]", name) or re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
                raise ValueError("实验名称需有中文用途说明，最多 40 字，不含路径符号")
            selected_model = self.root / "模型" / f"{name}_{run_id}"
            args.extend(["--episodes", *map(str, selected), "--steps", str(steps), "--batch-size", str(batch_size),
                         "--output-dir", str(selected_model)])
            if kind == "train_plan":
                args.append("--dry-run")
        elif kind.startswith("eval"):
            selected_model = Path(model).resolve() if model is not None else None
            if selected_model not in self.models():
                raise ValueError("请选择本项目完整的微调模型目录")
            if type(samples) is not int or samples < 1:
                raise ValueError("每段抽样数量须为正整数")
            args.extend(["--episodes", *map(str, selected), "--model-dir", str(selected_model), "--samples-per-episode", str(samples)])
            if require_held_out:
                args.append("--require-held-out")
            if kind == "eval_plan":
                args.append("--check-only")
        return JobSpec(kind, tuple(args), directory, report, dataset, selected, selected_model, bool(require_held_out),
                       steps if kind.startswith("train") else 0, batch_size if kind.startswith("train") else 0)

    def environment(self):
        env = os.environ.copy()
        temporary = self.root / "运行环境/临时文件"
        temporary.mkdir(parents=True, exist_ok=True)
        env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONPATH=str(self.root / "上游框架/src"),
                   HF_HOME=str(self.root / "运行环境/缓存/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", HF_LEROBOT_CALIBRATION=str(self.root / "标定备份"),
                   TEMP=str(temporary), TMP=str(temporary))
        return env

    def launch_operator(self, script, factory=subprocess.Popen):
        if script not in self.OPERATOR_WINDOWS:
            raise ValueError("面板只能手动打开演示录制或结果记录窗口")
        previous = self.operator_processes.get(script)
        if previous and previous.poll() is None:
            return previous
        self.logs.mkdir(parents=True, exist_ok=True)
        path = self.logs / f"窗口日志_{Path(script).stem}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
        with path.open("w", encoding="utf-8") as stream:
            process = factory([str(self.python), str(self.root / ("脚本/数据采集" if script == "录制任务演示.py" else "脚本/程序模块") / script)], cwd=self.root, env=self.environment(),
                              stdin=subprocess.DEVNULL, stdout=stream, stderr=stream, shell=False,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.operator_processes[script] = process
        return process

    def validate_result(self, spec):
        result = json.loads(spec.report.read_text(encoding="utf-8"))
        if result.get("status") != "passed" or result.get("robot_commands_sent") != 0:
            raise ValueError("任务缺少通过的零硬件命令结果报告")
        if spec.kind == "check":
            if Path(result["dataset_root"]).resolve() != spec.dataset:
                raise ValueError("检查报告对应的批次不一致")
            return result
        source = result.get("training_data") if spec.kind.startswith("train") else result.get("evaluation_data")
        if (not source or Path(source["dataset_root"]).resolve() != spec.dataset or
            tuple(row["episode"] for row in source["episodes"]) != spec.episodes):
            raise ValueError("结果报告中的批次或所选段不一致")
        if spec.kind.startswith("train"):
            if (Path(result["output_dir"]).resolve() != spec.model or result.get("steps") != spec.steps or
                result.get("batch_size") != spec.batch_size):
                raise ValueError("训练输出目录或参数与选择不一致")
        if spec.kind == "train_plan":
            if result.get("scope") != "training_plan_only" or result.get("checkpoint_saved") is not False:
                raise ValueError("训练计划不能冒充已完成训练")
        elif spec.kind == "train":
            if result.get("checkpoint_saved") is not True or Path(result["output_dir"]).resolve() != spec.model:
                raise ValueError("训练未保存对应检查点")
            if load_training_source(spec.model) != source:
                raise ValueError("模型来源记录与训练报告不一致")
            provenance = json.loads((spec.model / "训练数据来源.json").read_text(encoding="utf-8"))
            if (provenance.get("steps") != spec.steps or provenance.get("batch_size") != spec.batch_size or
                result.get("checkpoint_sha256") != provenance["checkpoint_sha256"]):
                raise ValueError("检查点来源中的训练参数或内容指纹与报告不一致")
        else:
            if Path(result["model_dir"]).resolve() != spec.model:
                raise ValueError("评估模型与选择不一致")
            if result.get("checkpoint_sha256") != file_sha256(spec.model / "model.safetensors"):
                raise ValueError("评估报告与当前模型内容不一致")
            assessment = result.get("data_assessment", result)
            if spec.require_held_out and assessment.get("held_out_validation") is not True:
                raise ValueError("任务未通过强制留出检查")
            if spec.kind == "eval_plan":
                if result.get("scope") != "evaluation_plan_only" or result.get("model_inference_executed") is not False:
                    raise ValueError("评估计划不能冒充已完成推理")
            elif (result.get("scope") == "evaluation_plan_only" or
                  not math.isfinite(result["normalized_mae"]) or result["normalized_mae"] < 0 or result["samples"] < 1):
                raise ValueError("评估指标无效")
        return result


class SoftwareJob:
    def __init__(self, project, spec, factory=subprocess.Popen):
        self.project, self.spec, self.factory = project, spec, factory
        self.messages = queue.Queue()
        self.cancel_requested = threading.Event()
        self.process = None
        self.status, self.error, self.result = "starting", None, None
        self.thread = threading.Thread(target=self._run, name="采集训练软件任务", daemon=False)

    @property
    def running(self):
        return self.status in ("starting", "running", "verifying")

    def start(self):
        self.thread.start()
        return self

    def cancel(self):
        if self.process is None:
            if self.running:
                self.cancel_requested.set()
                return True
            return False
        if self.process.poll() is not None:
            return False
        self.cancel_requested.set()
        try:
            self.process.terminate()
        except OSError:
            if self.process.poll() is None:
                raise
        return True

    def _run(self):
        created = False
        try:
            self.spec.directory.mkdir(parents=True, exist_ok=False)
            created = True
            log = self.spec.directory / "控制台输出.log"
            if Path(self.spec.argv[2]).resolve() != (self.project.root / "脚本/程序模块" / self.project.SOFTWARE[self.spec.kind]).resolve():
                raise ValueError("任务不在软件脚本白名单中")
            if self.cancel_requested.is_set():
                self.status = "cancelled"
                return
            with log.open("w", encoding="utf-8") as stream:
                self.process = self.factory(list(self.spec.argv), cwd=self.project.root, env=self.project.environment(),
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding="utf-8", errors="replace", shell=False,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.status = "running"
                if self.cancel_requested.is_set():
                    self.cancel()
                for line in self.process.stdout:
                    stream.write(line)
                    stream.flush()
                    self.messages.put(line)
                code = self.process.wait()
            if self.cancel_requested.is_set():
                self.status = "cancelled"
            elif code != 0:
                raise ValueError(f"软件任务退出码 {code}，请查看控制台输出")
            else:
                self.status = "verifying"
                self.result = self.project.validate_result(self.spec)
                self.status = "succeeded"
        except Exception as exc:
            self.status, self.error = "failed", str(exc)
            self.messages.put(f"\n任务未通过：{exc}\n")
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired) as cleanup:
                    self.error += f"；软件进程清理未完成：{cleanup}"
        finally:
            if self.process and self.process.stdout:
                try:
                    self.process.stdout.close()
                except OSError:
                    pass
            state = {"kind": self.spec.kind, "status": self.status, "error": self.error,
                     "argv": self.spec.argv, "pid": self.process.pid if self.process else None,
                     "report": str(self.spec.report), "updated_at": datetime.now().isoformat(timespec="seconds"),
                     "scope": "software_job_only", "robot_commands_sent": 0}
            if created:
                try:
                    (self.spec.directory / "任务状态.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                except OSError as exc:
                    self.status, self.error = "failed", f"无法保存任务状态：{exc}"
                    self.messages.put(self.error + "\n")
