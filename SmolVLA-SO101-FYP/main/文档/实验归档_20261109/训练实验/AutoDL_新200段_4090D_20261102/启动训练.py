"""新200段单卡训练/恢复入口，所有路径基于解压目录。"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--output', type=Path, default=ROOT/'输出/新200段_batch16_30K')
    a = p.parse_args()
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               PYTHONPATH=str(ROOT/'上游框架/src'), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, '-u', str(ROOT/'脚本/程序模块/新200段训练.py'),
               '--dataset-root', str(ROOT/'训练数据'), '--data-check', str(ROOT/'数据检查.json'),
               '--output-dir', str(a.output.resolve()), '--steps', str(a.steps),
               '--batch-size', '16', '--warmup', '500', '--save-every', '5000',
               '--validate-every', '1000', '--validation-samples', '10']
    if a.resume:
        command += ['--resume', str(a.output.resolve()/'last')]
    if a.stop_after is not None:
        command += ['--stop-after', str(a.stop_after)]
    subprocess.run(command, cwd=ROOT, env=env, check=True)

if __name__ == '__main__':
    main()
