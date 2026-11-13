"""目标选择时间标注的数据层；视频/Parquet只读，标注单独保存。"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

MARKERS = {
    "first_contact": ("首次接触目标", "夹爪第一次明确碰到本段RC/BC目标方块的那一帧；不要求尚未合爪，不等夹紧或提起。"),
}
PROTOCOL = "first_contact_stage_sampling_v1"
TASK_CODES = {
    "Pick up the red cube and place it in the target area.": "RC",
    "Pick up the blue cube and place it in the target area.": "BC",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class DatasetCatalog:
    def __init__(self, root):
        import pyarrow.parquet as pq
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8-sig"))
        self.fps = float(self.info["fps"])
        if not 0 < self.fps <= 240:
            raise ValueError("视频帧率无效")
        self.keys = [k for k, v in self.info["features"].items() if v["dtype"] == "video"]
        if set(self.keys) != {"observation.images.front", "observation.images.wrist"}:
            raise ValueError("本标注器要求桌面front与夹爪wrist双摄数据")
        self.keys = ["observation.images.front", "observation.images.wrist"]
        tasks = {r["task_index"]: r["task"] for r in self._jsonl("meta/tasks.jsonl")}
        fingerprint = hashlib.sha256()
        for name in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
            fingerprint.update(name.encode())
            fingerprint.update((self.root / name).read_bytes())
        self.episodes = {}
        for meta in self._jsonl("meta/episodes.jsonl"):
            ep = int(meta["episode_index"])
            if ep < 0 or ep in self.episodes:
                raise ValueError("段号无效或重复")
            relative = self.info["data_path"].format(
                episode_chunk=ep // self.info["chunks_size"], episode_index=ep)
            path = self.safe_path(relative)
            table = pq.read_table(path, columns=["frame_index", "task_index"]).to_pydict()
            length = int(meta["length"])
            if length < 1 or table["frame_index"] != list(range(length)):
                raise ValueError(f"视频{ep:06d}的逐帧编号与元数据不一致")
            task_set = {tasks[t] for t in table["task_index"]}
            if len(task_set) != 1 or next(iter(task_set)) not in TASK_CODES:
                raise ValueError(f"视频{ep:06d}不是单一规范RC/BC任务")
            task = next(iter(task_set))
            videos = {key: self.safe_path(self.info["video_path"].format(
                episode_chunk=ep // self.info["chunks_size"], episode_index=ep, video_key=key))
                for key in self.keys}
            if any(not p.is_file() for p in videos.values()):
                raise ValueError(f"视频{ep:06d}缺少相机文件")
            fingerprint.update(relative.encode())
            fingerprint.update(bytes.fromhex(sha256_file(path)))
            self.episodes[ep] = {"episode_index": ep, "length": length, "task": task,
                                 "target": TASK_CODES[task], "videos": videos}
        if len(self.episodes) != self.info["total_episodes"]:
            raise ValueError("演示段数与info.json不一致")
        self.ids = sorted(self.episodes)
        self.fingerprint = fingerprint.hexdigest()

    def _jsonl(self, relative):
        return [json.loads(line) for line in (self.root / relative).read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    def safe_path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("数据路径超出所选目录")
        return path

    def video_hashes(self, ep):
        return {key: sha256_file(path) for key, path in self.episodes[ep]["videos"].items()}


def derive_intervals(markers, length, chunk=50):
    """只划分接触前输入帧；未来动作窗口不在接触点截断。空区间用None。"""
    if set(markers) != set(MARKERS):
        raise ValueError("每段只需1个标记：首次接触目标")
    c = markers["first_contact"]
    if type(length) is not int or length < 1 or type(c) is not int or not 0 <= c < length:
        raise ValueError("标记帧超出本段范围")
    if type(chunk) is not int or chunk < 1:
        raise ValueError("动作窗口长度必须为正整数")
    last = min(c - 1, length - chunk)
    return {"first_contact_frame": c,
            "pre_contact_observation_range_inclusive": [0, c - 1] if c > 0 else None,
            "pre_contact_window_starts_inclusive": [0, last] if last >= 0 else None,
            "full_window_starts_inclusive": [0, length - chunk] if length >= chunk else None,
            "chunk_size": chunk, "future_actions_cross_contact_allowed": True}


def decision_mask_for_window(entry, observation_frame, chunk=50):
    """旧负例掩码接口停用，防止把单接触点误当成目标决策监督。"""
    raise ValueError("单接触点仅供阶段采样，不能生成错误指令对照损失的决策掩码")


class AnnotationStore:
    def __init__(self, catalog, path, legacy_path=None):
        self.catalog, self.path = catalog, Path(path).resolve()
        if self.path.is_relative_to(catalog.root):
            raise ValueError("请把标注保存在数据集之外，避免改变已核验的训练数据目录")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a+b")
        try:
            if self.lock.seek(0, 2) == 0:
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError("这批标注已在另一个窗口打开，请使用已有窗口")
        try:
            source_path = self.path if self.path.exists() else (Path(legacy_path) if legacy_path and Path(legacy_path).is_file() else None)
            self.data = json.loads(source_path.read_text(encoding="utf-8-sig")) if source_path else {
                "schema_version": 2, "kind": "first_contact_stage_annotations", "protocol": PROTOCOL, "created_at": now(),
                "dataset_root": str(catalog.root), "dataset_fingerprint": catalog.fingerprint,
                "fps": catalog.fps, "episode_index_base": 0, "frame_index_base": 0,
                "interval_endpoints": "inclusive", "episodes": {}, "cursor": {}}
            if self.data.get("schema_version") not in (1, 2) or self.data.get("dataset_fingerprint") != catalog.fingerprint:
                raise ValueError("标注版本或数据指纹不匹配，拒绝混用")
            migrated = self.data["schema_version"] == 1
            if migrated:
                # 永久保留旧文件快照；只继承同义的首次接触，旧审核一律撤销。
                backup = self.path.with_name(self.path.stem + ".三标记旧版.json")
                if not backup.exists():
                    atomic_json(backup, self.data)
                self.data = {**self.data, "schema_version": 2, "kind": "first_contact_stage_annotations",
                             "protocol": PROTOCOL, "migrated_from": str(source_path),
                             "legacy_backup": str(backup), "migrated_at": now()}
                for entry in self.data["episodes"].values():
                    legacy = dict(entry)
                    markers = entry.get("markers", {})
                    entry.update(markers={"first_contact": markers["first_contact"]} if "first_contact" in markers else {},
                                 status="draft", scene_confirmed=False, legacy_v1=legacy)
                    for key in ("intervals", "reviewed_at", "video_sha256", "reason"):
                        entry.pop(key, None)
            if self.data.get("protocol") != PROTOCOL:
                raise ValueError("标注协议不匹配，拒绝把旧决策区间当作阶段采样数据")
            for ep, entry in self.data["episodes"].items():
                if (int(ep) not in catalog.episodes or entry.get("task") != catalog.episodes[int(ep)]["task"]
                        or entry.get("length") != catalog.episodes[int(ep)]["length"]):
                    raise ValueError("标注段号或任务与本批数据不匹配")
                markers = entry.get("markers", {})
                if markers:
                    derive_intervals(markers, entry["length"])
                if entry.get("status") == "reviewed":
                    derive_intervals(markers, entry["length"])
            if migrated:
                self.save()
        except Exception:
            self.close()
            raise

    def entry(self, ep):
        meta = self.catalog.episodes[ep]
        return self.data["episodes"].setdefault(str(ep), {
            "episode_index": ep, "task": meta["task"], "target": meta["target"],
            "length": meta["length"], "markers": {}, "status": "draft", "scene_confirmed": False})

    def save(self):
        self.data["updated_at"] = now()
        self.data["dataset_root"] = str(self.catalog.root)
        if self.path.exists():
            previous = json.loads(self.path.read_text(encoding="utf-8-sig"))
            atomic_json(self.path.with_suffix(".备份.json"), previous)
        atomic_json(self.path, self.data)

    def set_marker(self, ep, key, frame):
        if key not in MARKERS or type(frame) is not int or not 0 <= frame < self.catalog.episodes[ep]["length"]:
            raise ValueError("标记或帧号无效")
        entry = self.entry(ep)
        entry["markers"][key] = frame
        entry.update(status="draft", scene_confirmed=False)
        self.save()

    def delete_marker(self, ep, key):
        entry = self.entry(ep)
        entry["markers"].pop(key, None)
        entry.update(status="draft", scene_confirmed=False)
        self.save()

    def review(self, ep, scene_confirmed):
        if not scene_confirmed:
            raise ValueError("请先核对并勾选：两块均在，且实际抓取与标签一致")
        entry = self.entry(ep)
        intervals = derive_intervals(entry["markers"], entry["length"])
        hashes = self.catalog.video_hashes(ep)
        entry.update(status="reviewed", scene_confirmed=True, reviewed_at=now(),
                     intervals=intervals, video_sha256=hashes)
        self.save()

    def skip(self, ep, reason):
        if not reason.strip():
            raise ValueError("请填写原因，例如首次接触被遮挡、开录已在接触、未碰到目标或实际抓错颜色")
        self.entry(ep).update(status="not_applicable", scene_confirmed=False, reason=reason.strip())
        self.save()

    def export(self, path):
        path = Path(path).resolve()
        protected = {self.path, self.path.with_suffix(".lock"), self.path.with_suffix(".备份.json"),
                     self.path.with_name(self.path.stem + ".三标记旧版.json")}
        protected.update(Path(self.data[key]).resolve() for key in ("migrated_from", "legacy_backup") if self.data.get(key))
        if path.is_relative_to(self.catalog.root) or path in protected:
            raise ValueError("导出文件须在训练数据之外，并与标注工作文件不同")
        rows = []
        for ep in self.catalog.ids:
            entry = self.entry(ep)
            if entry["status"] != "reviewed":
                continue
            if not entry.get("scene_confirmed") or entry.get("video_sha256") != self.catalog.video_hashes(ep):
                raise ValueError(f"视频{ep:06d}已改变或未确认，请重新核对")
            intervals = derive_intervals(entry["markers"], entry["length"])
            rows.append({**entry, "intervals": intervals,
                         "marker_seconds": {k: v / self.catalog.fps for k, v in entry["markers"].items()}})
        if not rows:
            raise ValueError("尚无已确认的段；草稿和不适用段不会进入训练导出")
        atomic_json(path, {"schema_version": 2, "kind": "reviewed_first_contact_stages", "protocol": PROTOCOL,
                          "created_at": now(), "dataset_fingerprint": self.catalog.fingerprint,
                          "dataset_root": str(self.catalog.root), "fps": self.catalog.fps,
                          "episode_index_base": 0, "frame_index_base": 0, "interval_endpoints": "inclusive",
                          "chunk_size": 50, "count": len(rows), "episodes": rows,
                          "purpose": "increase_sampling_of_pre_contact_observations_keep_full_future_actions",
                          "negative_instruction_supervision": False,
                          "training_integration_complete": False,
                          "training_started": False, "physical_success_evaluated": False})
        return len(rows)

    def close(self):
        if getattr(self, "lock", None) and not self.lock.closed:
            self.lock.close()
