"""按实测批次保留96万窗口预算，独立screen启动并验证保存/恢复。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,math,os,subprocess,textwrap
ROOT=Path('/root/autodl-tmp/AutoDL_纯80段双物体_阶段采样_20261108')
benchmark=json.loads((ROOT/'显存实测/批次搜索结果.json').read_text())
if benchmark['state']!='completed':raise ValueError('批次搜索尚未完成')
batch=benchmark['selected_batch']
config_path=ROOT/'训练参数_B.json'
config=json.loads(config_path.read_text())
output=ROOT/config['output_relative']
if output.exists():raise RuntimeError('正式输出已存在，拒绝启动重复训练')
screen=subprocess.run(['screen','-ls'],capture_output=True,text=True)
if 'smolvla80train' in screen.stdout:raise RuntimeError('已有80段训练screen，拒绝重复启动')
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if gpu:raise RuntimeError('GPU仍有计算进程，需要先核对占用：'+gpu)
def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
write(ROOT/'配置调整前_batch140.json',config)
interval=max(1,round(32000/batch))
config.update(batch_size=batch,steps=math.ceil(960000/batch),warmup=max(1,math.ceil(16000/batch)),
              validate_every=interval,save_every=interval*5)
write(config_path,config)
experiment_path=ROOT/'实验说明.json'
experiment=json.loads(experiment_path.read_text())
experiment.update(control_experiment='用户要求在新GPU实测最大batch；保留96万样本窗口预算。数据子集、batch和更新步数均与100段实验不同，不能作为仅改变单双物体比例的严格对照。',
    actual_configuration=config,sampled_windows=batch*config['steps'],gpu='RTX PRO 6000 Blackwell Server Edition',
    batch_search=benchmark,training_started=False)
write(experiment_path,experiment)
wrapper=ROOT/'正式训练调度.py'
source='''"""独立正式训练，先实际保存16步检查点再严格恢复；不覆盖其他实验。"""
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
'''
wrapper.write_text(source,encoding='utf-8')
# 参数、实验说明及调度脚本都进入本实验清单，启动前会完整核验。
manifest=json.loads((ROOT/'文件清单.json').read_text())
indexed={x['path']:x for x in manifest}
for p in (config_path,experiment_path,wrapper,ROOT/'配置调整前_batch140.json'):
    rel=p.relative_to(ROOT).as_posix();indexed[rel]={'path':rel,'size':p.stat().st_size,'sha256':sha(p)}
write(ROOT/'文件清单.json',[indexed[k] for k in sorted(indexed)])
subprocess.run(['screen','-dmS','smolvla80train',str(ROOT/'.venv/bin/python'),'-u',str(wrapper)],cwd=ROOT,check=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
receipt={'state':'dispatched','started_at':datetime.now().isoformat(),'screen':'smolvla80train','config':config,
         'sampled_windows':batch*config['steps'],'maximum_safe_batch':benchmark['maximum_safe_batch'],
         'benchmark_samples_per_second':benchmark['selected_samples_per_second'],'formal_progress_pending':True}
write(ROOT/'新集群启动记录.json',receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
