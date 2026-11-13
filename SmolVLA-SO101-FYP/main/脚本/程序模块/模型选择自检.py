"""用临时中文目录检查模型迁移、选择、取消及严格加载；不打开硬件。"""
import argparse
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import 选择推理模型 as selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        directory = selection.ROOT / "运行环境/临时文件"
        directory.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(prefix="模型选择 中文 空格_", dir=directory)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "机械臂项目"
        self.model = self.root / "模型/朋友的模型包/3000步模型"
        self.model.mkdir(parents=True)
        self.config = {"type": "smolvla", "vlm_model_name": "D:/朋友电脑/视觉语言处理器",
                       "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                                          **{key: {"type": "VISUAL", "shape": [3, 480, 640]} for key in selection.CAMERA_KEYS}},
                       "output_features": {"action": {"shape": [6]}}}
        self.write(self.model / "config.json", self.config)
        self.write(self.model / "dataset_stats.json", {})
        (self.model / "model.safetensors").write_bytes(b"test-fixture-only")
        self.processor = self.model.parent / "视觉语言处理器"
        self.processor.mkdir()
        for name in selection.PROCESSOR_FILES:
            self.write(self.processor / name, {})
        self.dataset = self.root.parent / "训练数据采集/机械臂控制项目/任务演示/双相机演示 测试批次"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "data/chunk-000").mkdir(parents=True)
        (self.dataset / "data/chunk-000/episode_000000.parquet").touch()
        names = [f"joint{i}.pos" for i in range(6)]
        self.info = {"fps": 20, "features": {"action": {"names": names}, "observation.state": {"names": names},
                                               **{key: {} for key in selection.CAMERA_KEYS}}}
        self.write(self.dataset / "meta/info.json", self.info)
        self.write(self.model / "training_provenance.json", {"training_data": {
            "dataset_root": "D:\\其他机器\\" + self.dataset.name, "joint_names": names, "state_names": names}})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_infers_chinese_batch_without_using_friends_drive(self):
        chosen = selection.validate_selection(self.model, root=self.root)
        self.assertEqual(chosen.dataset, self.dataset.resolve())
        self.assertEqual(chosen.processor, self.processor.resolve())

    def test_remembers_valid_selection_and_rejects_stale_saved_model(self):
        chosen = selection.validate_selection(self.model, root=self.root)
        selection.remember_selection(chosen, self.root)
        self.assertEqual(selection.recent_selection(self.root), chosen)
        (self.model / 'model.safetensors').unlink()
        self.assertIsNone(selection.recent_selection(self.root))

    def test_restores_last_real_run_checkpoint_without_starting_it(self):
        logs = self.root / '运行记录'
        logs.mkdir()
        identity = dict(type='model_identity', model_path=str(self.model),
                        dataset_path=str(self.dataset), processor_path=str(self.processor))
        log = logs / '实机抓取测试_模拟.jsonl'
        log.write_text(json.dumps(identity, ensure_ascii=False)+'\n', encoding='utf-8')
        self.assertEqual(selection.recent_selection(self.root).model, self.model.resolve())
        identity['model_path'] = str(self.model.parent)
        log.write_text(json.dumps(identity, ensure_ascii=False)+'\n', encoding='utf-8')
        self.assertIsNone(selection.recent_selection(self.root))

    def test_nested_discovery_excludes_processor_and_incomplete_model(self):
        (self.model.parent / "10000步模型").mkdir()
        self.write(self.model.parent / "10000步模型/config.json", {})
        self.assertEqual(selection.discover_models(self.root), [self.model.resolve()])

    def test_missing_or_outer_model_is_rejected(self):
        for path in ("", self.model.parent):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "具体"):
                selection.validate_selection(path, self.dataset, root=self.root)

    def test_no_silent_fallback_to_old_dataset(self):
        (self.model / "training_provenance.json").unlink()
        with self.assertRaisesRegex(ValueError, "对应演示"):
            selection.validate_selection(self.model, root=self.root)

    def test_explicit_external_dataset_and_processor(self):
        chosen = selection.validate_selection(self.model, self.dataset, self.processor, self.root)
        self.assertEqual(chosen.model, self.model.resolve())
        self.assertEqual(chosen.dataset, self.dataset.resolve())
        with self.assertRaisesRegex(ValueError, "处理器"):
            selection.validate_selection(self.model, self.dataset, self.model, self.root)

    def test_wrong_cameras_fps_and_joints_are_rejected(self):
        for change in ("camera", "fps", "joints", "provenance"):
            info = json.loads(json.dumps(self.info))
            if change == "camera":
                del info["features"]["observation.images.wrist"]
            elif change == "fps":
                info["fps"] = 30
            elif change == "joints":
                info["features"]["action"]["names"] = ["wrong"] * 6
            else:
                info["features"]["action"]["names"].reverse()
                info["features"]["observation.state"]["names"].reverse()
            self.write(self.dataset / "meta/info.json", info)
            with self.subTest(change=change), self.assertRaises(ValueError):
                selection.validate_selection(self.model, root=self.root)

    def test_missing_joint_data_rejected(self):
        next(self.dataset.glob("data/chunk-*/*.parquet")).unlink()
        with self.assertRaisesRegex(ValueError, "关节数据"):
            selection.validate_selection(self.model, root=self.root)

    def test_cli_bypasses_window_and_keeps_spaces(self):
        parser = argparse.ArgumentParser()
        selection.add_selection_arguments(parser)
        args = parser.parse_args(["--model-dir", str(self.model)])
        with patch.object(selection, "choose_model") as dialog:
            self.assertEqual(selection.selection_from_args(args, self.root).dataset, self.dataset)
            dialog.assert_not_called()

    def test_noargs_cancel_returns_none(self):
        parser = argparse.ArgumentParser()
        selection.add_selection_arguments(parser)
        with patch.object(selection, "choose_model", return_value=None):
            self.assertIsNone(selection.selection_from_args(parser.parse_args([]), self.root))

    def test_loader_changes_only_memory_and_keeps_strict_offline_load(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        original = (self.model / "config.json").read_bytes()
        cfg = SimpleNamespace(vlm_model_name="D:/朋友电脑/视觉语言处理器", device="cpu")
        with patch.object(PreTrainedConfig, "from_pretrained", return_value=cfg) as config_loader, \
             patch.object(SmolVLAPolicy, "from_pretrained") as loader:
            selection.load_local_policy(self.model)
            config_loader.assert_called_once_with(self.model.resolve(), local_files_only=True)
            loader.assert_called_once_with(self.model.resolve(), config=cfg, local_files_only=True, strict=True)
            self.assertEqual(cfg.vlm_model_name, str(self.processor.resolve()))
            self.assertEqual(cfg.device, "cuda")
        self.assertEqual((self.model / "config.json").read_bytes(), original)
        self.assertFalse((self.model / "config.json.原件").exists())

    def test_real_entry_cancel_never_enters_hardware_preparation(self):
        import 实机抓取测试 as grasp
        with patch.object(grasp, "choose_model", return_value=None), patch.object(grasp, "run_test") as run:
            grasp.main()
            run.assert_not_called()

    def test_window_selection_autofill_confirm_and_cancel(self):
        window = tk.Tk()
        window.withdraw()
        panel = selection.ModelSelectionWindow(window, self.root)
        panel.model.set(str(self.model))
        panel.model_changed()
        self.assertEqual(panel.dataset.get(), str(self.dataset.resolve()))
        panel.confirm()
        self.assertEqual(panel.result.model, self.model.resolve())
        window = tk.Tk()
        window.withdraw()
        panel = selection.ModelSelectionWindow(window, self.root)
        window.destroy()
        self.assertIsNone(panel.result)


if __name__ == "__main__":
    unittest.main()
