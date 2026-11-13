"""训练入口的系统内存维护层；不改模型、采样、损失或检查点格式。"""
import ctypes
from datetime import datetime
import functools
import gc
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'输出/新200段_batch32_持续训练'
LOG = ROOT/'日志/内存维护.jsonl'

def rss_mib():
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmRSS:'))) / 1024

def report(event, **values):
    record = {'event': event, 'time': datetime.now().isoformat(), **values}
    with LOG.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')

import torch

original_step = torch.optim.AdamW.step
trim = ctypes.CDLL(None).malloc_trim
trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
updates = 0

@functools.wraps(original_step)
def maintained_step(self, *args, **kwargs):
    global updates
    result = original_step(self, *args, **kwargs)
    updates += 1
    if updates % 50 == 0:
        before = rss_mib()
        collected = gc.collect()
        trim(0)
        after = rss_mib()
        report('memory_maintenance', updates_since_resume=updates, rss_before_mib=before,
               rss_after_mib=after, gc_collected=collected)
        # 保存后由父进程重启继续；与用户主动PAUSE区分，保留完整优化器和随机状态。
        if after > 32768 and not (OUTPUT/'PAUSE').exists():
            (OUTPUT/'MEMORY_RESTART.json').write_text(json.dumps({'rss_mib':after,
                'time':datetime.now().isoformat()}), encoding='utf-8')
            (OUTPUT/'PAUSE').touch()
            report('requested_checkpoint_restart', rss_mib=after)
    return result

torch.optim.AdamW.step = maintained_step
trainer = ROOT/'脚本/加速模块/新200段训练.py'
sys.path.insert(0, str(trainer.parent))
report('start', rss_mib=rss_mib(), maintenance_every_updates=50)
sys.argv = [str(trainer), *sys.argv[1:]]
runpy.run_path(str(trainer), run_name='__main__')
