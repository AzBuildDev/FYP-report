"""真实语音模型→RC/BC→真实SmolVLA→模拟动作接收器；不连接串口或相机。"""
import argparse
import json
from pathlib import Path
import tempfile
import threading
import time
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "上游框架/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--model", type=Path, default=ROOT / "模型/新200段_阶段采样_20261103/step_007500")
    parser.add_argument("--replay", type=Path, default=ROOT / "运行记录/实机抓取测试_20261103_190205_813970")
    args = parser.parse_args()
    import psutil
    for process in psutil.process_iter(["name", "cmdline"]):
        if "python" in (process.info["name"] or "").lower() and any(
                Path(arg).name == "实机抓取测试.py" for arg in process.info["cmdline"] or []):
            raise RuntimeError("发现实机控制进程，拒绝运行离线GPU验收")
    import torch
    torch.set_num_threads(2)
    from 本地语音模型 import LocalASR, LocalClassifier
    from 语音实机桥接 import process_request
    asr = LocalASR(ROOT / "模型/语音指令/whisper-tiny", "cpu")
    classifier = LocalClassifier(ROOT / "模型/语音指令/Qwen2.5-1.5B-Instruct", "cpu", timeout=45)
    start = time.monotonic()
    speech = process_request({"audio": str(args.audio), "language": args.language}, asr, classifier)
    print(json.dumps({"speech": speech}, ensure_ascii=False), flush=True)
    report = {"scope": "real_models_replayed_images_simulated_motor", "audio_source": str(args.audio),
              "hardware_connected": False, "robot_commands_sent": 0, "task_success_evaluated": False,
              "speech": speech, "speech_seconds": time.monotonic() - start}
    if speech["status"] == "target":
        from 选择推理模型 import validate_selection, load_local_policy
        from 实机抓取测试 import make_predict
        from 红蓝抓取任务 import TASKS
        from 运行中目标切换 import TaskSwitch
        from 异步动作控制 import run_async_control
        chosen = validate_selection(str(args.model), "", "", ROOT)
        policy = load_local_policy(chosen.model, chosen.processor)
        sample = json.loads((args.replay / "模型输入索引.jsonl").read_text(encoding="utf-8").splitlines()[0])
        current = np.asarray(sample["state"], dtype=np.float32)
        images = {key: np.asarray(Image.open(args.replay / name).convert("RGB"))[:, :, ::-1].copy()
                  for key, name in sample["images"].items()}
        stop, events, commands, workers = threading.Event(), [], [], []
        with tempfile.TemporaryDirectory() as directory:
            initial = "RC" if speech["target"] == "BC" else "BC"
            switch = TaskSwitch(TASKS[initial], directory, events.append)
            predict = make_predict(policy, task=switch)
            predict(current, images)  # 独立预热，不发送动作。
            request = {"id": 1, "target": speech["target"], "source": "offline_voice", "transcript": speech["transcript"]}
            (Path(directory) / "目标请求.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

            def receive(target):
                nonlocal current
                current = target.copy()
                commands.append(target.tolist())
                return target.tolist()

            try:
                stats = run_async_control(predict, lambda: current.copy(),
                                          lambda: (time.monotonic() - .005, images), receive, stop,
                                          current - 10, current + 10, duration=3, workers=workers,
                                          task_switch=switch, on_event=events.append)
            finally:
                for worker in workers:
                    worker.thread.join(timeout=10)
                    if worker.thread.is_alive():
                        raise RuntimeError("离线推理线程未退出")
            if not commands or switch.current != speech["target"]:
                raise RuntimeError("语音目标未进入模拟动作接收器")
            report.update(model=str(chosen.model), task=switch.current, control=stats,
                          simulated_commands=len(commands), first_simulated_target=commands[0],
                          switch_events=[event for event in events if event["type"].startswith("task_switch")])
    report["status"] = "passed" if speech["status"] == "target" else "rejected"
    path = ROOT / "运行记录" / f"语音动作离线验收_{time.time_ns()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(path), "speech": speech,
                      "simulated_commands": report.get("simulated_commands", 0), "robot_commands_sent": 0}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
