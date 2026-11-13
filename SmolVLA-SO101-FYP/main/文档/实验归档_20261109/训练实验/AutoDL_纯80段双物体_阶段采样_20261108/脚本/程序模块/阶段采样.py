"""RC/BC阶段采样：仅改变观察帧抽样，真实50步动作保持完整。"""
import json
from pathlib import Path
import numpy as np
from 目标选择标注数据 import DatasetCatalog, derive_intervals, PROTOCOL

SAMPLING_VERSION = "stage_balanced_v2_no_review"


def load_annotations(dataset_root, annotation_path, split, chunk=50):
    catalog = DatasetCatalog(dataset_root)
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8-sig"))
    expected = {"schema_version": 2, "protocol": PROTOCOL,
                "kind": "reviewed_first_contact_stages", "episode_index_base": 0,
                "frame_index_base": 0, "interval_endpoints": "inclusive", "chunk_size": chunk,
                "fps": catalog.fps, "dataset_fingerprint": catalog.fingerprint}
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError("标注协议、编号基准、帧率或数据指纹不匹配")
    rows = data.get("episodes", [])
    if data.get("count") != len(rows) or len(rows) != len(catalog.ids):
        raise ValueError("要求整批完整标注，不能将缺失段默默排除")
    annotations = {}
    for row in rows:
        ep = row.get("episode_index")
        if type(ep) is not int or ep in annotations or ep not in catalog.episodes:
            raise ValueError("段号重复、无效或不存在")
        meta = catalog.episodes[ep]
        if row.get("status") == "not_applicable" or any(row.get(k) != meta[k] for k in ("task", "target", "length")):
            raise ValueError(f"段{ep}任务、长度或标记状态不匹配")
        intervals = derive_intervals(row.get("markers", {}), meta["length"], chunk)
        if intervals != row.get("intervals") or intervals["pre_contact_window_starts_inclusive"] is None:
            raise ValueError(f"段{ep}区间不一致或没有完整的标记前窗口")
        if row.get("video_sha256") != catalog.video_hashes(ep):
            raise ValueError(f"段{ep}视频内容已变化")
        annotations[ep] = row
    train, val = split["train"], split["validation"]
    if (not train or not val or any(type(ep) is not int for ep in train + val)
            or len(set(train)) != len(train) or len(set(val)) != len(val)
            or set(train) & set(val) or set(train) | set(val) != set(catalog.ids)):
        raise ValueError("训练/验证必须互斥、无重复、完整覆盖数据集")
    counts = {name: {target: sum(annotations[ep]["target"] == target for ep in ids)
                     for target in ("RC", "BC")}
              for name, ids in (("train", train), ("validation", val))}
    if any(c["RC"] != c["BC"] or c["RC"] == 0 for c in counts.values()):
        raise ValueError("训练和验证均须红蓝均衡")
    return annotations, {"sampling_version": SAMPLING_VERSION, "dataset_fingerprint": catalog.fingerprint,
                         "episodes": len(catalog.ids), "train_episodes": len(train), "validation_episodes": len(val),
                         "train_target_counts": counts["train"], "validation_target_counts": counts["validation"],
                         "stage_fraction": {"pre_marker": 0.5, "full_trajectory": 0.5},
                         "actual_pre_marker_fraction_may_exceed_half": True,
                         "chunk_size": chunk, "action_window_truncated": False,
                         "negative_instruction_supervision": False, "confirmation_required": False}


class StageSampler:
    def __init__(self, tables, annotations, batch_size, seed=42, chunk=50, mode="stage"):
        if type(batch_size) is not int or batch_size < 4 or batch_size % 4:
            raise ValueError("batch必须是4的正整数倍")
        if mode not in ("stage", "full"):
            raise ValueError("采样方式仅支持stage/full")
        self.batch_size, self.seed, self.chunk, self.mode = batch_size, seed, chunk, mode
        self.tables = {int(t["episode_index"][0]): t for t in tables}
        if len(self.tables) != len(tables) or not tables:
            raise ValueError("训练段号重复或为空")
        self.annotations = annotations
        self.groups = {0: [], 1: []}
        for ep, table in self.tables.items():
            row = annotations[ep]
            task = int(table["task_index"][0])
            if task not in (0, 1) or set(table["task_index"]) != {task} or row["target"] != ("RC" if task == 0 else "BC"):
                raise ValueError("段内RC/BC不一致或任务编号不符合0红1蓝")
            if len(table["action"]) != row["length"] or set(table["episode_index"]) != {ep}:
                raise ValueError("动作长度或段号不符")
            intervals = derive_intervals(row["markers"], row["length"], chunk)
            if intervals["pre_contact_window_starts_inclusive"] is None:
                raise ValueError("标记前没有完整窗口，不允许静默退化为全段采样")
            self.groups[task].append(ep)
        if not all(self.groups.values()):
            raise ValueError("训练集缺少一种颜色")

    def plan(self, step):
        if type(step) is not int or step < 1:
            raise ValueError("更新步必须从1开始")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
        records = []
        stages = ("pre_marker", "full_trajectory") if self.mode == "stage" else ("full_trajectory",)
        for stage in stages:
            for task in (0, 1):
                group = self.groups[task]
                for _ in range(self.batch_size // (2 * len(stages))):
                    ep = group[int(rng.integers(len(group)))]
                    row = self.annotations[ep]
                    maximum = row["length"] - self.chunk
                    if stage == "pre_marker":
                        maximum = min(maximum, row["markers"]["first_contact"] - 1)
                    records.append({"episode": ep, "frame": int(rng.integers(maximum + 1)),
                                    "task_index": task, "stage": stage})
        return [records[int(i)] for i in rng.permutation(len(records))]

    def records(self, step):
        return [(self.tables[r["episode"]], r["frame"]) for r in self.plan(step)]

    def statistics(self, step):
        from collections import Counter
        plan = self.plan(step)
        return {"draw_counts": dict(Counter(f"{r['stage']}_{r['task_index']}" for r in plan)),
                "actual_inputs_before_marker": sum(r["frame"] < self.annotations[r["episode"]]["markers"]["first_contact"] for r in plan),
                "windows_crossing_marker": sum(r["frame"] < self.annotations[r["episode"]]["markers"]["first_contact"] <= r["frame"] + self.chunk - 1 for r in plan)}
