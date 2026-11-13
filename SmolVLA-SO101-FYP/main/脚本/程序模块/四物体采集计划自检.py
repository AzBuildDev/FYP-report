"""验证四物体配额、规范文字、布局划分及缺失/错误标签拦截；不连接硬件。"""
import copy
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from 四物体采集计划 import (ROOT, collection_plan, task_text, plan_row, episode_label,
                        read_labels, write_label, label_path, require_training_split, export_plan)
from 管理训练数据来源 import assess_sources


class CollectionTests(unittest.TestCase):
    def test_budget_and_per_position_balance(self):
        rows = collection_plan()
        self.assertEqual(len(rows), 800)
        self.assertEqual(len({r["plan_id"] for r in rows}), 800)
        self.assertEqual(Counter(r["stage"] for r in rows), {"I-A": 200, "I-B": 200, "II": 400})
        counts = Counter((r["stage"], r["object_id"], r["destination"], r["position"], r["split"]) for r in rows)
        self.assertTrue(all(n == (8 if k[-1] == "train" else 2) for k, n in counts.items()))
        self.assertEqual(Counter(r["split"] for r in rows), {"train": 640, "dev": 160})

    def test_layout_groups_across_stages(self):
        groups = defaultdict(list)
        for row in collection_plan():
            if row["stage"] != "I-A":
                groups[row["scene_group_id"]].append(row)
        self.assertEqual(len(groups), 50)
        for rows in groups.values():
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({r["split"] for r in rows}), 1)
            self.assertTrue(all(r["placement"] == rows[0]["placement"] for r in rows))
            self.assertEqual(set(rows[0]["placement"].values()), {"RC", "BC", "GCy", "RCy", "empty"})
        self.assertEqual(list(groups["R09-K1"][0]["placement"].values()), ["BC", "GCy", "RC", "RCy", "empty"])

    def test_canonical_tasks_and_explicit_destination(self):
        self.assertEqual(task_text("BC"), "Pick up the blue cube and place it in the target area.")
        self.assertEqual(task_text("RCy", "B"), "Pick up the red cylinder and place it in zone B.")
        self.assertEqual(plan_row("I-A", "BC", 1, "P3", 1), plan_row("I-A", "BC", 1, "P3", 5))
        self.assertEqual(len({task_text(obj, dest) for obj in ("RC", "BC", "GCy", "RCy")
                              for dest in ("target_area", "A", "B")}), 12)
        for args in (("II", "RC", 1), ("I-B", "RC", 0), ("I-A", "red", 1)):
            with self.assertRaises(ValueError):
                plan_row(*args)

    def test_saved_labels_match_actual_text_and_cannot_be_silently_lost(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "运行环境/临时文件") as temp:
            root = Path(temp)
            (root / "采集配置.json").write_text(json.dumps({"collection_label_schema": 1}), encoding="utf-8")
            row = plan_row("I-B", "BC", 9)
            label = episode_label(row, root.name, 0, "test", "test", "test-config")
            tables = [{"episode_index": [0, 0], "task_index": [0, 0]}]
            tasks = [{"task_index": 0, "task": row["task_text"]}]
            with self.assertRaises(ValueError):
                read_labels(root, tables, tasks)
            write_label(root, label)
            with self.assertRaises(FileExistsError):
                write_label(root, label)
            self.assertEqual(read_labels(root, tables, tasks)[0], label)
            with self.assertRaises(ValueError):
                read_labels(root, tables, [{"task_index": 0, "task": task_text("RC")}])
            label["split"] = "train"
            label_path(root, 0).write_text(json.dumps(label), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_labels(root, tables, tasks)
            plan = root / "plan.csv"
            export_plan(plan)
            with self.assertRaises(FileExistsError):
                export_plan(plan)

    def test_dev_is_excluded_and_cross_stage_layout_leakage_detected(self):
        row = plan_row("I-B", "BC", 1)
        ep = {"episode": 0, "action_state_sha256": "train", "video_sha256": {"front": "train"},
              "collection_label": {**row, "layout_config": "same-table-v1"}}
        training = {"dataset_root": "train", "fps": 20, "joint_names": [1], "state_names": [1], "episodes": [ep]}
        require_training_split(training)
        evaluation = copy.deepcopy(training)
        evaluation["dataset_root"] = "new-batch"
        evaluation["episodes"][0].update(action_state_sha256="different", video_sha256={"front": "different"})
        evaluation["episodes"][0]["collection_label"].update(plan_row("II", "RCy", 1, destination="B"))
        result = assess_sources(training, evaluation)
        self.assertFalse(result["held_out_validation"])
        self.assertEqual(result["overlapping_scene_group_episodes"], [0])
        evaluation["episodes"][0]["collection_label"].update(plan_row("II", "RCy", 9, destination="B"))
        self.assertTrue(assess_sources(training, evaluation)["held_out_validation"])
        with self.assertRaises(ValueError):
            require_training_split(evaluation)


if __name__ == "__main__":
    unittest.main()
