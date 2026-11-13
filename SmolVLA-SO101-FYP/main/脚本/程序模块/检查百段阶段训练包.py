"""控制台中的安全检查入口，只核验训练包，不启动训练。"""
import os
from pathlib import Path
import subprocess
import sys


def main():
    package = Path(__file__).resolve().parents[2].parent / "AutoDL_纯100段_阶段采样_20261107"
    if not (package / "准备训练.py").is_file():
        raise FileNotFoundError("尚未生成百段训练包：" + str(package))
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run([sys.executable, str(package / "准备训练.py")], cwd=package, env=environment, check=True)
    subprocess.run([sys.executable, str(package / "启动训练.py"), "--arm", "B"], cwd=package, env=environment, check=True)
    print("仅完成检查与计划展示，未启动训练。")


if __name__ == "__main__":
    main()
