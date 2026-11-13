"""运行所有控制模拟测试，生成结构化结果；中文模块显式加载，避免默认发现遗漏。"""
import importlib
import json
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = ["控制输入自检", "异步控制自检", "模型预热自检", "电机故障退出自检", "监视窗口自检", "实机入口模拟自检"]


def main():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name))
                               for name in MODULES)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    report = {"status": "passed" if result.wasSuccessful() and result.testsRun else "failed",
              "created_at": datetime.now().isoformat(timespec="seconds"), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "modules": MODULES, "hardware_tested": False, "robot_commands_sent": 0}
    (ROOT / "运行记录/控制流程完整自检.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
