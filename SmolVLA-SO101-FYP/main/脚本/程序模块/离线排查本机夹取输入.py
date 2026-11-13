"""对照已保存实机输入与演示，检查模型动作；不连接相机或机械臂。"""
import argparse
import json
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from 选择推理模型 import load_local_policy
from 实机抓取测试 import make_predict
from 管理训练数据来源 import file_sha256

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "模型/五位置稳定训练_20261019/batch64/best_val"
DATA = ROOT.parent / "训练数据采集/机械臂控制项目/任务演示/双相机演示_20261019_100106_674366"
RUN = ROOT / "运行记录/实机抓取测试_20261024_015653_004460"
OUT = ROOT / "运行记录/本机夹取输入离线对照_20261024"


def main(model=MODEL, output=OUT, run=RUN, record_index=-1):
    model, output, run = Path(model).resolve(), Path(output).resolve(), Path(run).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (run / "模型输入索引.jsonl").read_text(encoding="utf-8").splitlines()]
    record = records[record_index]
    info = json.loads((DATA / "meta/info.json").read_text(encoding="utf-8"))
    scale = np.asarray(json.loads((model / "dataset_stats.json").read_text(encoding="utf-8"))["observation.state"]["std"])
    tables = [pq.read_table(p, columns=["observation.state", "action", "episode_index"]).to_pydict()
              for p in sorted(DATA.glob("data/chunk-*/*.parquet"))]
    live_state = np.asarray(record["state"], dtype=np.float32)
    candidates = []
    for table in tables:
        states = np.asarray(table["observation.state"], dtype=np.float32)
        distances = np.linalg.norm((states - live_state) / scale, axis=1)
        index = int(distances.argmin())
        candidates.append((float(distances[index]), int(table["episode_index"][0]), index, table))
    distance, episode, index, table = min(candidates, key=lambda item: item[0])
    live_images = {key: cv2.imdecode(np.frombuffer((run / filename).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                   for key, filename in record["images"].items()}
    demo_images = {}
    for key in live_images:
        video = DATA / info["video_path"].format(episode_chunk=episode // info["chunks_size"],
                                                episode_index=episode, video_key=key)
        with av.open(str(video)) as container:
            demo_images[key] = next(frame.to_ndarray(format="bgr24") for i, frame in enumerate(container.decode(video=0))
                                    if i == index)
    canvas = Image.new("RGB", (1280, 1010), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "Saved live input (top) / nearest demonstration joint pose (bottom)", fill="black")
    for row, images in enumerate((live_images, demo_images)):
        for col, image in enumerate(images.values()):
            canvas.paste(Image.fromarray(image[:, :, ::-1]), (col * 640, 25 + row * 490))
    canvas.save(output / "实时与相近演示画面对照.jpg")
    print(f"Nearest demo: episode={episode}, frame={index}, normalized distance={distance:.3f}", flush=True)
    policy = load_local_policy(model)
    predict = make_predict(policy)
    cases = (("saved_live", live_state, live_images),
             ("nearest_demo", np.asarray(table["observation.state"][index], dtype=np.float32), demo_images),
             ("same_live_state_demo_images", live_state, demo_images))
    result = dict(model=str(model), run=str(run), record_index=record_index, input_record=record,
                  checkpoint_sha256=file_sha256(model / "model.safetensors"),
                  hardware_connected=False, motor_commands_sent=0, nearest_episode=episode,
                  nearest_frame=index, normalized_state_distance=distance, cases=[])
    for label, state, images in cases:
        predictions = []
        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            policy.reset()
            started = time.perf_counter()
            actions = predict(state, images)
            predictions.append(dict(seed=seed, seconds=time.perf_counter() - started,
                                    finite=bool(np.isfinite(actions).all()), actions=actions.tolist(),
                                    samples={str(i): actions[i].tolist() for i in (0, 10, 17, 25, 49)},
                                    gripper_min=float(actions[:, 5].min()), gripper_max=float(actions[:, 5].max())))
        result["cases"].append(dict(label=label, state=state.tolist(), predictions=predictions))
        print(label, "prediction complete", flush=True)
    result["nearest_demo_actual_future"] = {
        str(offset): table["action"][min(index + offset, len(table["action"]) - 1)]
        for offset in (0, 10, 17, 25, 49)}
    (output / "离线结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output / "离线结果.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--record-index", type=int, default=-1)
    args = parser.parse_args()
    main(args.model_dir, args.output_dir, args.run_dir, args.record_index)
