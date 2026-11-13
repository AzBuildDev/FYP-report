"""先验证实际检查点保存/恢复，再在screen中继续同一条训练。"""
from pathlib import Path
from datetime import datetime
import json,os,subprocess,time

ROOT=Path('/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107')
PYTHON=str(ROOT/'.venv/bin/python')
LOG=ROOT/'日志'
config=json.loads((ROOT/'训练参数_B.json').read_text(encoding='utf-8'))
output=ROOT/config['output_relative']
env=os.environ.copy()
env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
    PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072')
state_path=LOG/'正式训练调度状态.json'
def state(value):
    value['time']=datetime.now().isoformat()
    temporary=state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(state_path)
def run(resume=False,stop=None,logname='正式训练.log'):
    command=[PYTHON,'-u',str(ROOT/'启动训练.py'),'--arm','B','--run']
    if resume:command+=['--resume']
    if stop is not None:command+=['--stop-after',str(stop)]
    with (LOG/logname).open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state({'phase':'save_acceptance' if stop else 'training','pid':process.pid,'resume':resume,'stop_after':stop})
        return process.wait()

if output.exists():raise RuntimeError('正式输出已存在，不重复启动')
code=run(stop=16,logname='GPU保存验收.log')
if code:
    state({'phase':'failed','during':'gpu_save_acceptance','exit_code':code})
    raise SystemExit(code)
summary=json.loads((output/'last/checkpoint_summary.json').read_text(encoding='utf-8'))
assert summary['step']==16
for name in ('model.safetensors','training_state.pt','dataset_stats.json','training_provenance.json'):
    assert (output/'last'/name).stat().st_size>0
(LOG/'GPU保存验收.json').write_text(json.dumps({'status':'passed','step':16,
    'batch_size':config['batch_size'],'validation_predictions':summary['validation']['predictions'],
    'checkpoint':str(output/'last'),'resume_observation_pending':True},ensure_ascii=False,indent=2),encoding='utf-8')
code=run(resume=True)
state({'phase':'completed' if code==0 else 'failed','exit_code':code})
raise SystemExit(code)
