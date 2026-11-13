"""逐进程实测batch，保留1.5GiB显存余量；不保留试跑模型。"""
from pathlib import Path
from datetime import datetime
import json,math,os,subprocess,sys,time

ROOT=Path('/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107')
PYTHON=str(ROOT/'.venv/bin/python')
LOG=ROOT/'日志'
LOG.mkdir(exist_ok=True)
env=os.environ.copy()
env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
    PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072')
results=[]
def save():
    (LOG/'显存上限实测.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
def test(batch):
    output=ROOT/'验收输出'/f'显存_batch{batch}'
    if output.exists():raise RuntimeError('不覆盖已有实测：'+str(output))
    cmd=[PYTHON,'-u',str(ROOT/'脚本/程序模块/百段阶段训练.py'),
        '--dataset-root',str(ROOT/'训练数据'),'--data-check',str(ROOT/'数据检查.json'),
        '--annotations',str(ROOT/'标注/阶段标注_100段.json'),'--output-dir',str(output),
        '--batch-size',str(batch),'--steps','30000','--warmup','500','--sampling-mode','stage','--benchmark-only']
    path=LOG/f'显存_batch{batch}.log'
    print('实测batch='+str(batch),flush=True)
    started=time.monotonic()
    with path.open('w',encoding='utf-8') as log:
        code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    row={'batch_size':batch,'exit_code':code,'wall_s':time.monotonic()-started}
    report=output/'显存吞吐实测.json'
    if code==0 and report.exists():
        row.update(json.loads(report.read_text(encoding='utf-8')))
        row['headroom_mib']=row['gpu_total_mib']-row['peak_gpu_reserved_mib']
        row['safe']=row['headroom_mib']>=1536
    else:
        text=path.read_text(encoding='utf-8',errors='replace')
        row.update(safe=False,oom=('out of memory' in text.lower()))
        if not row['oom']:
            results.append(row);save()
            print(text[-5000:],flush=True)
            raise RuntimeError('非显存错误，停止探测')
    results.append(row);save()
    print(json.dumps(row,ensure_ascii=False),flush=True)
    return row['safe']

low,high=0,None
candidate=128
if test(candidate):
    low=candidate
    while True:
        candidate=low+max(32,((low//4)//4)*4)
        if test(candidate):low=candidate
        else:high=candidate;break
else:
    high=candidate
    while low==0:
        candidate=(candidate//2//4)*4
        if candidate<4:raise RuntimeError('没有安全batch')
        if test(candidate):low=candidate
        else:high=candidate
while high-low>4:
    mid=((low+high)//2//4)*4
    if test(mid):low=mid
    else:high=mid
selected=next(r for r in results if r['batch_size']==low)
summary={'status':'passed','selected_batch_size':low,'next_rejected_batch_size':high,
    'reserve_mib':1536,'selection':'4的倍数中、4次真实前向反向及优化器更新后保留1.5GiB余量的最大batch',
    'benchmark':selected,'formal_training_started':False,'time':datetime.now().isoformat()}
(LOG/'最大稳定batch.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
