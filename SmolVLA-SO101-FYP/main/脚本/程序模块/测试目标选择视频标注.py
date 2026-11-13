"""使用临时双摄数据验证标注、精确seek、窗口交互和源数据只读。"""
import json
from pathlib import Path
import tempfile
import time
import tkinter as tk

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from 标注目标选择视频 import AnnotationWindow, ExactVideoReader
from 目标选择标注数据 import AnnotationStore, DatasetCatalog, MARKERS, TASK_CODES, decision_mask_for_window, derive_intervals, sha256_file


def expect_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("应拒绝无效操作")


def fixture(root):
    (root / "meta").mkdir(parents=True)
    info = {"fps": 20, "chunks_size": 1000, "total_episodes": 2,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {k: {"dtype": "video"} for k in ("observation.images.front", "observation.images.wrist")}}
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    tasks = list(TASK_CODES)
    (root / "meta/tasks.jsonl").write_text("\n".join(json.dumps({"task_index": i, "task": task}) for i, task in enumerate(tasks)), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": ep, "length": 60, "tasks": [tasks[ep]]}) for ep in range(2)), encoding="utf-8")
    for ep in range(2):
        data = root / f"data/chunk-000/episode_{ep:06d}.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame_index": list(range(60)), "task_index": [ep] * 60}), data)
        for key in info["features"]:
            path = root / f"videos/chunk-000/{key}/episode_{ep:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), "w") as output:
                stream = output.add_stream("mpeg4", rate=20)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for frame in range(60):
                    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
                    rgb[:, :, ep] = frame * 4
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)


def wait_window(window, predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate():
        window.update()
        if time.monotonic() >= deadline:
            raise AssertionError("窗口操作超时")
        time.sleep(0.01)
    window.update()


def main():
    with tempfile.TemporaryDirectory(prefix="标注软件测试_") as temp:
        base = Path(temp)
        root = base / "dataset"
        fixture(root)
        originals = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}
        catalog = DatasetCatalog(root)
        path = base / "annotations.json"
        store = AnnotationStore(catalog, path)
        expect_error(lambda: AnnotationStore(catalog, path))
        expect_error(lambda: store.export(base / "empty.json"))
        assert list(MARKERS) == ["first_contact"]
        store.set_marker(0, "first_contact", 20)
        assert store.entry(0)["status"] == "annotated"
        intervals = derive_intervals(store.entry(0)["markers"], 60)
        assert intervals["pre_contact_observation_range_inclusive"] == [0, 19]
        assert intervals["pre_contact_window_starts_inclusive"] == [0, 10]
        assert intervals["full_window_starts_inclusive"] == [0, 10]
        assert intervals["future_actions_cross_contact_allowed"] is True
        expect_error(lambda: decision_mask_for_window(store.entry(0), 9))
        assert store.export(base / "export.json") == 1
        exported = json.loads((base / "export.json").read_text(encoding="utf-8"))
        assert exported["schema_version"] == 2 and not exported["negative_instruction_supervision"]
        store.delete_marker(0, "first_contact")
        assert store.entry(0)["status"] == "draft"
        expect_error(lambda: store.export(base / "draft_export.json"))
        store.entry(1).update(status="not_applicable", reason="旧版跳过记录")
        store.save()
        store.close()
        store = AnnotationStore(catalog, path)
        assert store.entry(1)["status"] == "not_applicable"
        assert store.entry(0)["markers"] == {}
        store.close()
        for markers in ({}, {"first_contact": 60}, {"first_contact": -1}, {"first_contact": True},
                        {"first_contact": 20, "ready_start": 0}):
            expect_error(lambda m=markers: derive_intervals(m, 60))
        assert derive_intervals({"first_contact": 0}, 60)["pre_contact_window_starts_inclusive"] is None
        assert derive_intervals({"first_contact": 59}, 60)["pre_contact_window_starts_inclusive"] == [0, 10]
        assert derive_intervals({"first_contact": 30}, 40)["full_window_starts_inclusive"] is None
        assert derive_intervals({"first_contact": 49}, 50)["pre_contact_window_starts_inclusive"] == [0, 0]

        # 旧三标记逐字备份；接触点继承但撤销审核，不把原决策区间用于新训练。
        legacy = json.loads(path.read_text(encoding="utf-8"))
        legacy.update(schema_version=1, kind="target_selection_time_annotations")
        legacy.pop("protocol")
        legacy["episodes"]["0"].update(markers={"ready_start": 0, "approach_start": 10, "first_contact": 20},
                                       status="reviewed", scene_confirmed=True, intervals={"old": True})
        legacy["episodes"]["1"].update(reason="开录时已在接近", status="not_applicable")
        legacy_path = base / "legacy.json"
        legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        legacy_hash = sha256_file(legacy_path)
        migrated_path = base / "contact.json"
        migrated = AnnotationStore(catalog, migrated_path, legacy_path=legacy_path)
        assert sha256_file(legacy_path) == legacy_hash
        assert migrated.entry(0)["markers"] == {"first_contact": 20}
        assert migrated.entry(0)["status"] == "draft" and "intervals" not in migrated.entry(0)
        assert migrated.entry(1)["status"] == "draft" and "reason" not in migrated.entry(1)
        assert json.loads(Path(migrated.data["legacy_backup"]).read_text(encoding="utf-8")) == legacy
        assert migrated.export(base / "unreviewed_migration.json") == 1
        migrated.set_marker(0, "first_contact", 20)
        expect_error(lambda: migrated.export(legacy_path))
        expect_error(lambda: migrated.export(Path(migrated.data["legacy_backup"])))
        assert migrated.export(base / "migrated_export.json") == 1
        migrated.close()
        migrated = AnnotationStore(catalog, migrated_path)
        assert migrated.entry(0)["status"] == "annotated"
        migrated.close()

        # 兼容旧版已确认及已打标草稿，不再依赖scene_confirmed。
        legacy_store = AnnotationStore(catalog, base / "old_status.json")
        legacy_store.set_marker(0, "first_contact", 20)
        legacy_store.entry(0).update(status="reviewed", scene_confirmed=True)
        legacy_store.set_marker(1, "first_contact", 22)
        legacy_store.entry(1).update(status="draft", scene_confirmed=False)
        assert legacy_store.export(base / "old_status_export.json") == 2
        legacy_store.close()
        assert DatasetCatalog.resolve_root(root / "videos/chunk-000/observation.images.front") == root.resolve()
        assert DatasetCatalog.resolve_root(root / "videos/chunk-000/observation.images.front/episode_000000.mp4") == root.resolve()
        expect_error(lambda: DatasetCatalog.resolve_root(base))

        # 随机seek必须等于逐帧顺序解码的实际图像，特别检查末帧和倒退。
        expected = {}
        with av.open(str(catalog.episodes[0]["videos"][catalog.keys[0]])) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in (0, 1, 17, 59):
                    expected[index] = np.asarray(frame.to_image().convert("RGB"))
        reader = ExactVideoReader(catalog)
        for index in (59, 0, 17, 1):
            assert np.array_equal(np.asarray(reader.read(0, index)[0]), expected[index])
        reader.close()

        window = tk.Tk()
        window.withdraw()
        callback_errors = []
        window.report_callback_exception = lambda *args: callback_errors.append(args)
        panel = AnnotationWindow(window, root, base / "ui_annotations.json")
        wait_window(window, lambda: panel.displayed_frame == 0)
        assert len(panel.marker_buttons) == 1 and "单标记版" in window.title()
        panel.seek(20)
        wait_window(window, lambda: panel.displayed_frame == 20)
        panel.marker_buttons[0].invoke()
        panel.seek(21)
        # 仍在读取时不允许把旧画面标成新帧。
        panel.add_marker("first_contact")
        assert panel.store.entry(0)["markers"]["first_contact"] == 20
        wait_window(window, lambda: panel.displayed_frame == 21)
        panel.marker_buttons[0].invoke()
        assert panel.store.entry(0)["markers"] == {"first_contact": 21}
        assert not hasattr(panel, "confirm_scene") and not hasattr(panel, "review")
        assert panel.store.entry(0)["status"] == "annotated"
        assert panel.store.export(base / "ui_direct_export.json") == 1
        panel.switch_episode(1)
        wait_window(window, lambda: panel.current_ep == 1 and panel.displayed_frame == 0)
        panel.switch_episode(-1)
        wait_window(window, lambda: panel.current_ep == 0 and panel.displayed_frame == 0)
        assert len(panel.store.entry(0)["markers"]) == 1
        panel.scale.set(17)
        wait_window(window, lambda: panel.displayed_frame == 17)
        panel.step(-1)
        wait_window(window, lambda: panel.displayed_frame == 16)
        panel.seek(40)
        panel.seek(4)
        wait_window(window, lambda: panel.displayed_frame == 4)
        panel.markers.selection_set("first_contact")
        panel.delete_selected()
        assert panel.store.entry(0)["status"] == "draft" and len(panel.store.entry(0)["markers"]) == 0
        panel.toggle_play()
        wait_window(window, lambda: panel.frame > 4)
        panel.pause()

        # 相同数据集重选、切换到另一个数据集、切回时标记与游标均保持。
        panel.seek(21)
        wait_window(window, lambda: panel.displayed_frame == 21)
        panel.marker_buttons[0].invoke()
        panel.load_dataset(root, base / "ui_annotations.json")
        wait_window(window, lambda: panel.load_future is None)
        assert panel.store.entry(0)["markers"] == {"first_contact": 21}
        other = base / "other_dataset"
        fixture(other)
        info_path = other / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["dataset_name"] = "second_dataset"
        info_path.write_text(json.dumps(info), encoding="utf-8")
        other_subfolder = other / "videos/chunk-000/observation.images.front"
        panel.load_dataset(other_subfolder, base / "other_annotations.json")
        wait_window(window, lambda: panel.catalog.root == other.resolve() and panel.displayed_frame == 0)
        assert panel.store.entry(0)["markers"] == {}
        panel.seek(30)
        wait_window(window, lambda: panel.displayed_frame == 30)
        panel.marker_buttons[0].invoke()
        panel.load_dataset(root, base / "ui_annotations.json")
        wait_window(window, lambda: panel.catalog.root == root.resolve() and panel.displayed_frame == 21)
        assert panel.store.entry(0)["markers"] == {"first_contact": 21}
        assert json.loads((base / "other_annotations.json").read_text(encoding="utf-8"))["episodes"]["0"]["markers"] == {"first_contact": 30}
        panel.close()
        assert not callback_errors, callback_errors
        assert originals == {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}

        # 画面文件改变后，已审核标注必须拒绝导出。
        store = AnnotationStore(catalog, base / "integrity.json")
        store.set_marker(0, "first_contact", 20)
        with catalog.episodes[0]["videos"][catalog.keys[0]].open("ab") as stream:
            stream.write(b"changed")
        expect_error(lambda: store.export(base / "changed_export.json"))
        store.close()
        print(json.dumps({"status": "passed", "checks": ["精确非顺序seek与顺序解码一致", "单接触点及0/末帧/短片边界", "完整50步窗口可跨接触点", "旧负例掩码接口拒绝使用", "三标记备份迁移", "单窗口写锁", "标记后直接导出无需确认", "兼容旧已确认及已标记草稿", "自动保存重开", "窗口拖动/逐帧/前后段/播放/加删标记", "读取期间禁止标旧画面", "从相机文件夹识别数据集", "重复选同数据集不自锁", "切换两数据集后标记/游标分别恢复", "视频变更拒绝导出", "源文件内容不变"], "robot_commands_sent": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
