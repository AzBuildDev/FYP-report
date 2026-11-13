"""新GPU上实际执行4次完整更新，搜索保留2GiB显存余量的最大batch；不保存训练模型。"""
from pathlib import Path
from datetime import datetime
import json,os,subprocess,time
ROOT=Path('/root/autodl-tmp/AutoDL_纯80段双物体_阶段采样_20261108')
env=os.environ.copy()
env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
           PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072')
BASE=ROOT/'显存实测';BASE.mkdir(exist_ok=True)
REPORT=BASE/'批次搜索结果.json'
results=[]
def save():
    temp=REPORT.with_suffix('.tmp')
    temp.write_text(json.dumps({'state':'benchmarking','updated_at':datetime.now().isoformat(),
                              'reserve_mib':2048,'results':results},ensure_ascii=False,indent=2))
    temp.replace(REPORT)
def bench(batch):
    output=BASE/f'batch_{batch:06d}'
    if output.exists():raise ValueError('保留已有显存实测目录，拒绝覆盖：'+str(output))
    args=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'脚本/程序模块/百段阶段训练.py'),
       '--dataset-root',str(ROOT/'训练数据'),'--data-check',str(ROOT/'数据检查.json'),
       '--annotations',str(ROOT/'标注/阶段标注_80段.json'),'--output-dir',str(output),
       '--steps','6858','--warmup','115','--batch-size',str(batch),'--save-every','1145',
       '--validate-every','229','--validation-samples','10','--sampling-mode','stage','--benchmark-only']
    logfile=BASE/f'batch_{batch:06d}.log'
    print(json.dumps({'event':'benchmark_start','batch':batch,'time':datetime.now().isoformat()},ensure_ascii=False),flush=True)
    with logfile.open('w') as log:
        process=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    resultfile=output/'显存吞吐实测.json'
    if resultfile.exists() and process.returncode==0:
        row=json.loads(resultfile.read_text());row['batch']=batch
        row['headroom_mib']=row['gpu_total_mib']-row['peak_gpu_reserved_mib']
        row['safe']=row['headroom_mib']>=2048
    else:
        tail=logfile.read_text(errors='replace')[-7000:]
        oom=any(s in tail.lower() for s in ('out of memory','cuda error: memory allocation','cudaerroroutofmemory'))
        row={'batch':batch,'exit_code':process.returncode,'safe':False,'oom':oom,'log_tail':tail[-2000:]}
        if not oom:
            results.append(row);save();raise RuntimeError('实测出现非显存错误，停止搜索：'+str(logfile))
    results.append(row);save()
    print(json.dumps({'event':'benchmark_result',**{k:v for k,v in row.items() if k!='log_tail'}},ensure_ascii=False),flush=True)
    return row

lower=0;upper=None
batch=384
while upper is None:
    r=bench(batch)
    if r['safe']:
        lower=batch;batch+=128
    else:upper=batch
    if batch>1024:raise RuntimeError('搜索超过1024，请先复查GPU/数据管线')
while upper-lower>4:
    batch=((lower+upper)//8)*4
    if batch<=lower:batch=lower+4
    r=bench(batch)
    if r['safe']:lower=batch
    else:upper=batch
if lower==0:raise RuntimeError('首个批次均未通过，需要先测试较小batch')
# 同机比较较小批次吞吐，防止最大batch反而减速；用户要求尽量大，吞吐接近时优先最大。
small=max(4,(lower//8)*4)
if not any(x['batch']==small for x in results):bench(small)
safe=[x for x in results if x.get('safe')]
maxrow=next(x for x in safe if x['batch']==lower)
fastest=max(safe,key=lambda x:x['samples_per_second'])
chosen=fastest if fastest['samples_per_second']>maxrow['samples_per_second']*1.10 else maxrow
receipt={'state':'completed','updated_at':datetime.now().isoformat(),'reserve_mib':2048,
         'maximum_safe_batch':lower,'first_unsafe_batch':upper,
         'selected_batch':chosen['batch'],'selected_samples_per_second':chosen['samples_per_second'],
         'selection_reason':'最大可用batch；只有较小批次实测吞吐高出10%以上才选择更快批次',
         'results':results,'formal_training_started':False}
temp=REPORT.with_suffix('.tmp');temp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2));temp.replace(REPORT)
print('BATCH_SEARCH_COMPLETE '+json.dumps({k:v for k,v in receipt.items() if k!='results'},ensure_ascii=False),flush=True)
