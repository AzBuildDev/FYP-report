"""batch32、单线程视频解码和双层预读；首次从绑定的batch16检查点迁移。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--resume',action='store_true')
p.add_argument('--stop-after',type=int)
a=p.parse_args()
output=ROOT/'输出/新200段_batch32_持续训练'
env=os.environ.copy()
env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
           PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
           MALLOC_ARENA_MAX='2', MALLOC_MMAP_THRESHOLD_='131072', MALLOC_TRIM_THRESHOLD_='131072')
command=[sys.executable,'-u',str(ROOT/'内存稳定运行.py'),
         '--dataset-root',str(ROOT/'训练数据'),'--data-check',str(ROOT/'数据检查.json'),
         '--output-dir',str(output),'--steps','30000','--batch-size','32','--warmup','500',
         '--save-every','5000','--validate-every','1000','--validation-samples','10','--continuous']
if a.resume:
    command += ['--resume',str(output/'last')]
else:
    parent=json.loads((ROOT/'加速训练迁移.json').read_text(encoding='utf-8'))
    command += ['--migrate-from',parent['checkpoint_path']]
if a.stop_after is not None:
    command += ['--stop-after',str(a.stop_after)]
subprocess.run(command,cwd=ROOT,env=env,check=True)
