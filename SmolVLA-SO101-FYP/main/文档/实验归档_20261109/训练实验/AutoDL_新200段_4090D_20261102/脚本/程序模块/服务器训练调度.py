"""AutoDL部署调度：等待安装，真实GPU试跑与恢复成功后启动正式30K训练。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

def status(state, **values):
    record = {'state': state, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S%z'), **values}
    target = ROOT/'服务器部署状态.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(record, ensure_ascii=False), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--installer-pid', type=int, required=True)
    a = p.parse_args()
    os.chdir(ROOT)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUTF8='1',
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(ROOT/'上游框架/src'))
    logs = ROOT/'日志'
    logs.mkdir(exist_ok=True)
    def run(command, logfile):
        with (logs/logfile).open('a', encoding='utf-8', buffering=1) as f:
            subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    status('waiting_for_dependencies', installer_pid=a.installer_pid)
    while Path(f'/proc/{a.installer_pid}').exists():
        time.sleep(3)
    status('checking_environment')
    run([sys.executable, '-c', "import torch; from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.__version__.split('+')[0]=='2.7.1'; assert torch.cuda.is_bf16_supported()"], '环境验证.log')
    # 每段1个窗口做保存/恢复检查；正式训练仍每段10个验证窗口。
    smoke_output = ROOT/'输出/服务器试跑'
    smoke = [sys.executable, '-u', str(ROOT/'脚本/程序模块/新200段训练.py'),
             '--dataset-root', str(ROOT/'训练数据'), '--data-check', str(ROOT/'数据检查.json'),
             '--output-dir', str(smoke_output), '--steps', '30000', '--batch-size', '16',
             '--warmup', '500', '--save-every', '5000', '--validate-every', '1000',
             '--validation-samples', '1']
    status('gpu_smoke_training')
    run(smoke+['--stop-after', '2'], 'GPU试跑.log')
    status('gpu_resume_test')
    run(smoke+['--resume', str(smoke_output/'last'), '--stop-after', '3'], 'GPU恢复试跑.log')
    summary = json.loads((smoke_output/'last/checkpoint_summary.json').read_text(encoding='utf-8'))
    if summary['step'] != 3:
        raise RuntimeError('恢复试跑没有推进至第3步')
    status('formal_training', smoke_checkpoint_step=3, steps=30000, batch_size=16)
    run([sys.executable, '-u', str(ROOT/'启动训练.py')], '正式训练.log')
    status('completed', steps=30000)

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
