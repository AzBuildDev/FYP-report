"""用本机模型和已录制画面验证预热、异步交接及取消；不打开串口或相机。"""
import json
import threading
import time
from datetime import datetime

import av
import numpy as np
import pyarrow.parquet as pq
import torch

from 实机抓取测试 import ROOT, MODEL_DIR, DATASET_ROOT, CONTROL_HZ, MAX_STEP_DELTA, CHUNK_EXECUTE, make_predict
from 预热模型与检查时效 import warm_model
from 异步动作控制 import run_async_control
from 检查控制输入 import bounds_including_start
from 管理训练数据来源 import file_sha256
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("需要本机 CUDA GPU")
    report = {"status": "failed", "created_at": datetime.now().isoformat(timespec="seconds"),
              "scope": "offline_model_startup", "hardware_tested": False,
              "robot_commands_sent": 0, "task_success_evaluated": False,
              "model": str(MODEL_DIR.relative_to(ROOT)),
              "checkpoint_sha256": file_sha256(MODEL_DIR / "model.safetensors"),
              "config_sha256": file_sha256(MODEL_DIR / "config.json"),
              "control_hz": CONTROL_HZ, "max_step_delta": MAX_STEP_DELTA,
              "chunk_execute": CHUNK_EXECUTE,
              "software_sha256": {name: file_sha256(ROOT / "脚本/程序模块" / name) for name in
                                  ("预热模型与检查时效.py", "实机抓取测试.py", "异步动作控制.py", "模型启动离线验证.py",
                                   "检查与释放电机.py", "实机测试监视窗口.py")},
              "notes": "画面来自录制视频，模拟关节即时跟随目标；不验证实时相机、真实扭矩、接触或任务成功。预热预算只检查启动推理，运行时仍检查过期。"}
    path = ROOT / "运行记录" / f"模型启动离线验证_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    workers, async_workers = [], []
    try:
        torch.manual_seed(42)
        policy = SmolVLAPolicy.from_pretrained(MODEL_DIR, local_files_only=True).to("cuda").eval()
        info = json.loads((DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))
        table = pq.read_table(next(iter(sorted(DATASET_ROOT.glob("data/chunk-*/*.parquet")))))
        state = np.asarray(table["observation.state"][0].as_py(), dtype=np.float32)
        ep = int(table["episode_index"][0].as_py())
        video_key = next(k for k, value in info["features"].items() if value["dtype"] == "video")
        video = DATASET_ROOT / info["video_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=video_key)
        with av.open(str(video)) as container:
            image = next(container.decode(video=0)).to_ndarray(format="bgr24")
        predict = make_predict(policy, next(iter(policy.config.image_features)))
        report["warmup"] = warm_model(predict, state, image, threading.Event(),
                                     chunk_size=policy.config.chunk_size, hz=CONTROL_HZ, workers=workers)
        if report["warmup"]["status"] != "ready":
            raise RuntimeError("启动预热未通过")
        policy.reset()
        current, targets = state.copy(), []
        lo, hi = bounds_including_start(current, current - 8, current + 8)

        def send(target):
            nonlocal current
            if np.any(target < lo) or np.any(target > hi) or np.max(np.abs(target - current)) > MAX_STEP_DELTA + 1e-4:
                raise RuntimeError("模拟目标超出限制")
            current = target.copy()
            targets.append(True)

        # 新一次推理使用新快照；预热输出没有传入调度器。
        control = run_async_control(predict, lambda: current.copy(), lambda: (time.monotonic(), image.copy()),
                                    send, threading.Event(), lo, hi, duration=4, hz=CONTROL_HZ,
                                    max_delta=MAX_STEP_DELTA, execute_limit=CHUNK_EXECUTE, workers=async_workers)
        report["simulated_control"] = control
        if control["inferences"] < 2 or control["sent_commands"] < 20 or control["stop_reason"] != "time_limit":
            raise RuntimeError("预热后的异步模拟交接未通过")
        for worker in async_workers:
            worker.thread.join(5)
            if worker.thread.is_alive():
                raise RuntimeError("模拟推理尚未结束")

        stop, entered = threading.Event(), threading.Event()
        requested = []
        def cancel_predict(s, im):
            entered.set()
            return predict(s, im)

        def cancel():
            if entered.wait(5):
                time.sleep(0.05)
                requested.append(time.monotonic())
                stop.set()

        canceller = threading.Thread(target=cancel, daemon=True)
        canceller.start()
        cancelled = warm_model(cancel_predict, state, image, stop,
                               chunk_size=policy.config.chunk_size, hz=CONTROL_HZ, workers=workers)
        returned = time.monotonic()
        canceller.join(5)
        if not requested or cancelled["status"] != "cancelled" or returned - requested[0] >= 0.2:
            raise RuntimeError("GPU 预热取消响应未通过")
        report["warmup_cancel_response_s"] = returned - requested[0]
        report["cancel_prediction_reused"] = cancelled["prediction_reused"]
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        for worker in workers:
            worker.join(5)
        for worker in async_workers:
            worker.thread.join(5)
        if any(worker.is_alive() for worker in workers) or any(worker.thread.is_alive() for worker in async_workers):
            report["status"] = "failed"
            report["error"] = "后台 GPU 推理未退出"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"离线启动验证：{report['status']}；报告：{path}", flush=True)
    if report["status"] != "passed":
        raise RuntimeError(report["error"])
    pointer = {"status": "passed", "report": str(path), "report_sha256": file_sha256(path)}
    (ROOT / "运行记录/模型启动离线验证最新结果.json").write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
