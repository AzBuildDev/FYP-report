from pathlib import Path
import json,subprocess,shutil,datetime
R=Path('/root/autodl-tmp/AutoDL_纯80段双物体_阶段采样_20261108')
O=R/'输出/B_纯80段双物体_stage'
print('TIME',datetime.datetime.now().isoformat(),flush=True)
for args in (['screen','-ls'],['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],['pgrep','-af','百段阶段训练.py|正式训练调度.py|启动训练.py']):
    p=subprocess.run(args,capture_output=True,text=True,timeout=20)
    print(json.dumps({'command':args,'stdout':p.stdout,'stderr':p.stderr},ensure_ascii=False),flush=True)
print('DISK_GIB',shutil.disk_usage(R).free/1024**3,flush=True)
for rel in ('日志/正式训练调度状态.json','日志/GPU保存验收.json','新集群启动记录.json'):
    p=R/rel
    if p.exists():print(rel,p.read_text(),flush=True)
for p in (O/'metrics.jsonl',R/'日志/正式训练.log'):
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-14000));b=f.read()
        lines=b.decode('utf-8',errors='replace').splitlines()
        if p.name=='metrics.jsonl':
            for line in lines[-8:]:
                try:
                    row=json.loads(line)
                    print('METRIC',json.dumps({k:v for k,v in row.items() if k not in ('episode_metrics','train_episodes','validation_episodes')},ensure_ascii=False),flush=True)
                except Exception:pass
        else:
            print('LOG_TAIL','\n'.join(line[:1000] for line in lines[-6:]),flush=True)
            print('RECENT_ERRORS',[line[:1200] for line in lines if any(w in line for w in ('Traceback','OutOfMemoryError','RuntimeError','NaN','nan'))],flush=True)
for p in (O/'last/checkpoint_summary.json',):
    if p.exists():
        d=json.loads(p.read_text());c=d['run_config'];print('CHECKPOINT',json.dumps({'step':d['step'],'config':{k:c.get(k) for k in ('batch_size','steps','warmup','save_every','validate_every','annotation_sha256','initialization')},'metric':d.get('metric')},ensure_ascii=False),flush=True)
