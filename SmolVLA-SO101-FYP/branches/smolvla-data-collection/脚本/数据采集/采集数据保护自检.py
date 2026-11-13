"""采集原始数据保留与训练用途约束自检；不连接设备。"""
import json
import queue
import threading
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "程序模块"))
from 管理采集尝试 import CollectionAttempt, JOINTS, joint_values, write_json
from 管理训练数据来源 import validate_training_assignment

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.attempt = CollectionAttempt(self.root, 0, {"scene":"中央", "kind":"ordinary", "split":"train"}, "pick")
        self.image = self.root / "images/frame_000000.png"
        self.image.parent.mkdir()
        from PIL import Image
        Image.new("RGB", (16,16), (80,120,160)).save(self.image)
        self.attempt.journal.write_text(json.dumps({"type":"frame","frame_index":0,"image_path":"images/frame_000000.png"})+"\n", encoding="utf-8")
        self.reset = []
        self.dataset = SimpleNamespace(episode_buffer={"size":1}, _wait_image_writer=lambda: None,
                                       clear_episode_buffer=lambda: self.reset.append(True))

    def test_rejected_take_retains_image_and_journal_before_reset(self):
        self.attempt.archive_and_reset(self.dataset, "rejected", "操作者重录")
        self.assertEqual(self.reset, [True])
        self.assertEqual((self.attempt.path/"原始帧/frame_000000.png").read_bytes(), self.image.read_bytes())
        self.assertTrue(self.attempt.metadata["archive_complete"])
        self.assertTrue(self.attempt.journal.exists())

    def test_retry_same_episode_has_independent_identity(self):
        other = CollectionAttempt(self.root,0,{}, "pick")
        self.assertNotEqual(other.path, self.attempt.path)
        self.assertTrue(self.attempt.journal.exists())

    def test_missing_image_blocks_reset(self):
        self.image.unlink()
        with self.assertRaises(RuntimeError):
            self.attempt.archive_and_reset(self.dataset, "failed", "缺帧")
        self.assertEqual(self.reset, [])

    def test_row_mismatch_blocks_reset(self):
        self.dataset.episode_buffer["size"] = 2
        with self.assertRaises(RuntimeError):
            self.attempt.archive_and_reset(self.dataset, "failed", "缺动作")
        self.assertTrue(self.image.exists())
        self.assertEqual(self.reset, [])

    def test_copy_failure_preserves_source_and_blocks_reset(self):
        with patch("管理采集尝试.shutil.copy2", side_effect=OSError("模拟磁盘写入失败")):
            with self.assertRaises(OSError):
                self.attempt.archive_and_reset(self.dataset, "failed", "磁盘故障")
        self.assertTrue(self.image.exists())
        self.assertEqual(self.reset, [])
        self.assertFalse(self.attempt.metadata["archive_complete"])

    def test_path_escape_blocks_reset(self):
        self.attempt.journal.write_text(json.dumps({"type":"frame", "frame_index":0,"image_path":"../outside.png"})+"\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.attempt.archive(self.dataset)
        self.assertEqual(self.reset, [])

    def test_archives_cannot_be_overwritten(self):
        self.attempt.archive(self.dataset)
        with self.assertRaises(RuntimeError):
            self.attempt.archive(self.dataset)

    def test_invalid_joint_positions_rejected_before_send(self):
        for bad in (float("nan"), float("inf"), 101, -101):
            with self.subTest(value=bad):
                with self.assertRaises(RuntimeError):
                    joint_values({name: bad if i == 0 else 10 for i,name in enumerate(JOINTS)})
        with self.assertRaises(RuntimeError):
            joint_values({name: -1 if name == "gripper.pos" else 10 for name in JOINTS})

    def test_full_disk_metadata_failure_still_seals_and_reports(self):
        from 录制任务演示 import RecordingWindow
        app = RecordingWindow.__new__(RecordingWindow)
        app.pending, app.busy, app.sealed = True, False, False
        app.dataset = SimpleNamespace(num_episodes=0)
        app.attempt = SimpleNamespace(archive=lambda _: (_ for _ in ()).throw(OSError("磁盘已满")),
                                      set_status=lambda *a, **k: (_ for _ in ()).throw(OSError("状态也无法写入")))
        app.events = queue.Queue()
        app.path = self.root
        app.refresh_controls = lambda: None
        app.status = SimpleNamespace(set=lambda _: None)
        with patch("录制任务演示.write_json", side_effect=OSError("封存状态也无法写入")), contextlib.redirect_stderr(io.StringIO()):
            app.finish_episode(True)
            app.arm_thread.join(5)
        self.assertFalse(app.arm_thread.is_alive())
        self.assertTrue(app.sealed)
        self.assertEqual(app.events.get_nowait()[0], "save_failed")

    def test_held_out_and_sealed_batches_cannot_train(self):
        validate_training_assignment(self.root) # historical unlabeled batches remain compatible
        write_json(self.root/"采集配置.json", {"split":"held_out"})
        with self.assertRaises(ValueError):
            validate_training_assignment(self.root)
        write_json(self.root/"采集配置.json", {"split":"train"})
        write_json(self.root/"批次封存状态.json", {"reason":"保存失败"})
        with self.assertRaises(ValueError):
            validate_training_assignment(self.root)


if __name__ == "__main__":
    unittest.main()
