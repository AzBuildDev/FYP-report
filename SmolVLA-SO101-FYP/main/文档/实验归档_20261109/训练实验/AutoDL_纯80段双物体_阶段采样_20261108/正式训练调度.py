"""独立正式训练，先实际保存16步检查点再严格恢复；不覆盖其他实验。"""
from pathlib import Path
from datetime import datetime
import json,os,subprocess
ROOT=Path(__file__).resolve().parent
LOG=ROOT/'日志';LOG.mkdir(exist_ok=True)
config=json.loads((ROOT/'训练参数_B.json').read_text())
output=ROOT/config['output_relative']
env=os.environ.copy()
env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
           PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072')
def state(value):
    value['time']=datetime.now().isoformat()
    p=LOG/'正式训练调度状态.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2));t.replace(p)
def run(resume=False,stop=None):
    args=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'启动训练.py'),'--arm','B','--run']
    if resume:args+=['--resume']
    if stop is not None:args+=['--stop-after',str(stop)]
    with (LOG/'正式训练.log').open('a') as log:
        p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state({'phase':'save_acceptance' if stop else 'training','pid':p.pid,'resume':resume,'stop_after':stop})
        return p.wait()
if output.exists():raise RuntimeError('正式输出已存在，不重复启动')
code=run(stop=16)
if code:
    state({'phase':'failed','during':'save_acceptance','exit_code':code});raise SystemExit(code)
summary=json.loads((output/'last/checkpoint_summary.json').read_text())
assert summary['step']==16 and summary['run_config']['batch_size']==config['batch_size']
for name in ('model.safetensors','training_state.pt','dataset_stats.json','training_provenance.json'):
    assert (output/'last'/name).stat().st_size>0
(LOG/'GPU保存验收.json').write_text(json.dumps({'status':'passed','step':16,'batch_size':config['batch_size'],
    'predictions':summary['validation']['predictions'],'checkpoint':str(output/'last')},ensure_ascii=False,indent=2))
code=run(resume=True)
state({'phase':'completed' if code==0 else 'failed','exit_code':code})
raise SystemExit(code)
