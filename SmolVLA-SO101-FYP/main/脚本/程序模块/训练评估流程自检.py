"""检查整段划分、训练统计、复制泄漏和模型来源绑定；临时素材只用于软件自检。"""
import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from 管理训练数据来源 import assess_sources, dataset_source, file_sha256, load_training_source, select_tables, training_stats, verify_checkpoint_stats

ROOT = Path(__file__).resolve().parents[2]


def table(ep, value=1, frames=50):
    return {"episode_index": [ep] * frames, "timestamp": [i / 20 for i in range(frames)],
            "action": [[value] * 6 for _ in range(frames)],
            "observation.state": [[value] * 6 for _ in range(frames)], "task_index": [0] * frames}


class DataFlowTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "运行环境/临时文件"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="训练来源自检_", dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.info = {"fps": 20, "chunks_size": 1000, "video_path": "videos/{video_key}/episode_{episode_index}.mp4",
                     "features": {"action": {"names": list(range(6)), "dtype": "float32"},
                                  "observation.state": {"names": list(range(6)), "dtype": "float32"},
                                  "front": {"dtype": "video"}}}
        self.tasks = [{"task_index": 0, "task": "软件测试素材"}]
        for ep in range(3):
            path = self.root / f"videos/front/episode_{ep}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"仅哈希测试占位，不是真实视频 {ep}".encode("utf-8"))
        self.tables = [table(ep, ep + 1) for ep in range(3)]

    def source(self, selected):
        return dataset_source(self.root, self.info, select_tables(self.tables, selected), self.tasks)

    def test_whole_episode_selection_and_statistics_exclude_validation(self):
        selected = select_tables([table(0, 1), table(1, 3), table(2, 1000)], [0, 1])
        self.assertEqual([t["episode_index"][0] for t in selected], [0, 1])
        self.assertEqual(float(training_stats(selected)["action"]["mean"][0]), 2)
        self.assertEqual(float(training_stats(selected)["action"]["std"][0]), 1)

    def test_invalid_selection_and_short_episode_rejected(self):
        for selection in ([], [0, 0], [8]):
            with self.assertRaises(ValueError):
                select_tables(self.tables, selection)
        with self.assertRaisesRegex(ValueError, "短段"):
            select_tables([table(0, frames=49)])

    def test_disjoint_episode_is_held_out_with_same_batch_caveat(self):
        assessment = assess_sources(self.source([0, 1]), self.source([2]))
        self.assertTrue(assessment["held_out_validation"])
        self.assertTrue(assessment["same_recording_batch"])
        self.assertEqual(assessment["overlapping_episodes"], [])

    def test_overlap_and_unknown_provenance_are_not_held_out(self):
        self.assertFalse(assess_sources(self.source([0, 1]), self.source([1, 2]))["held_out_validation"])
        self.assertFalse(assess_sources(None, self.source([2]))["held_out_validation"])

    def test_copied_reindexed_actions_detected_even_with_different_video(self):
        copied = copy.deepcopy(self.tables[0])
        copied["episode_index"] = [2] * 50
        evaluation = dataset_source(self.root, self.info, [copied], self.tasks)
        evaluation["dataset_root"] = "另一个目录"
        self.assertEqual(assess_sources(self.source([0]), evaluation)["overlapping_episodes"], [2])

    def test_video_copy_detected_when_actions_differ(self):
        evaluation = self.source([2])
        evaluation["episodes"][0]["video_sha256"] = self.source([0])["episodes"][0]["video_sha256"]
        self.assertFalse(assess_sources(self.source([0]), evaluation)["held_out_validation"])

    def test_schema_mismatch_and_path_escape_rejected(self):
        evaluation = self.source([2])
        evaluation["fps"] = 30
        with self.assertRaisesRegex(ValueError, "fps"):
            assess_sources(self.source([0]), evaluation)
        unsafe = copy.deepcopy(self.info)
        unsafe["video_path"] = "../outside.mp4"
        with self.assertRaisesRegex(ValueError, "超出"):
            dataset_source(self.root, unsafe, [self.tables[0]], self.tasks)

    def test_unsupported_fps_and_joint_order_rejected_before_training(self):
        info = copy.deepcopy(self.info)
        info["fps"] = 30
        with self.assertRaisesRegex(ValueError, "20 FPS"):
            dataset_source(self.root, info, [self.tables[0]], self.tasks)
        info["fps"] = 20
        info["features"]["observation.state"]["names"] = list(reversed(range(6)))
        with self.assertRaisesRegex(ValueError, "关节顺序"):
            dataset_source(self.root, info, [self.tables[0]], self.tasks)

    def test_checkpoint_binding_rejects_changed_weights(self):
        checkpoint = self.root / "model.safetensors"
        checkpoint.write_bytes(b"software-test-only")
        record = {"schema_version": 1, "status": "completed", "training_data": self.source([0, 1]),
                  "checkpoint_sha256": file_sha256(checkpoint)}
        (self.root / "训练数据来源.json").write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(load_training_source(self.root), record["training_data"])
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "不匹配"):
            load_training_source(self.root)

    def test_checkpoint_normalization_matches_only_training_statistics(self):
        from safetensors.numpy import save_file
        stats = training_stats(select_tables(self.tables, [0, 1]))
        tensors = {f"{prefix}.buffer_{feature.replace('.', '_')}.{name}": stats[feature][name]
                   for prefix, feature in (("normalize_inputs", "observation.state"),
                                           ("normalize_targets", "action"), ("unnormalize_outputs", "action"))
                   for name in ("mean", "std")}
        save_file(tensors, str(self.root / "model.safetensors"))
        self.assertTrue(verify_checkpoint_stats(self.root, stats))
        with self.assertRaisesRegex(ValueError, "归一化"):
            verify_checkpoint_stats(self.root, training_stats(self.tables))

    def test_video_random_reads_match_sequential_decode_after_cache_eviction(self):
        import av
        import numpy as np
        from 微调预训练模型 import EpisodeVideoFrames, VideoFrameCache
        cache = VideoFrameCache(max_bytes=3 * 160 * 120 * 3)
        for width, height in ((320, 180), (160, 120)):
            path = self.root / f"随机读取_{width}x{height}.mkv"
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("ffv1", rate=20)
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                for index in range(80):
                    image = np.full((height, width, 3), index * 3, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(str(path)) as container:
                reference = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            frames = EpisodeVideoFrames(path, len(reference), 20, cache)
            for index in (0, 17, 3, 79, -1, 41, 17):
                np.testing.assert_array_equal(frames[index], reference[index])
                self.assertLessEqual(cache.bytes, cache.max_bytes)
            with self.assertRaises(IndexError):
                frames[len(reference)]


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(DataFlowTests))
    report = {"status": "passed" if result.wasSuccessful() and result.testsRun else "failed",
              "created_at": datetime.now().isoformat(timespec="seconds"), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "hardware_tested": False, "robot_commands_sent": 0, "scope": "data_pipeline_software_tests"}
    (ROOT / "运行记录/训练评估流程自检.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)
