from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import json,math,os,subprocess

root=Path('/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107')
output=root/'输出/B_纯100段_stage'
now=datetime.now(ZoneInfo('Asia/Shanghai'))
events=[]
if (output/'metrics.jsonl').exists():
    for line in (output/'metrics.jsonl').read_text(encoding='utf-8').splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
start_index=max([i for i,e in enumerate(events) if e.get('event')=='start'],default=0)
recent=events[start_index:]
trains=[e for e in recent if e.get('event')=='train']
latest=trains[-1] if trains else None
state_file=root/'日志/正式训练调度状态.json'
state=json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else None
config=json.loads((root/'训练参数_B.json').read_text(encoding='utf-8'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip()
result={'checked_at':now.isoformat(),'scheduler':state,'gpu':gpu,'config':config,
    'last_start':{k:v for k,v in recent[0].items() if k not in ('train_episodes','validation_episodes')} if recent else None,
    'latest_train':latest,'last_events':[{'event':e['event'],'step':e.get('step'),'time':e.get('time')} for e in events[-6:]],
    'disk_free_gib':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize/1024**3}
summary=output/'last/checkpoint_summary.json'
if summary.exists():
    value=json.loads(summary.read_text(encoding='utf-8'))
    result['checkpoint']={'step':value['step'],'best_score':value['best_score'],'predictions':value['validation']['predictions']}
if len(trains)>=2:
    first=trains[-3] if len(trains)>=3 else trains[0]
    last=trains[-1]
    rate=(last['elapsed_s']-first['elapsed_s'])/(last['step']-first['step'])
    validation_records=[e for e in events if e.get('event')=='validation']
    overhead=120.0
    if validation_records:
        validation=validation_records[0]
        initial=next(e for e in events if e.get('event')=='start')
        duration=(datetime.fromisoformat(validation['time'])-datetime.fromisoformat(initial['time'])).total_seconds()
        checkpoint=next((e for e in events if e.get('event')=='checkpoint_saved' and e['step']==validation['step']),None)
        save_s=(datetime.fromisoformat(checkpoint['time'])-datetime.fromisoformat(validation['time'])).total_seconds() if checkpoint else 15.0
        overhead=max(0,duration-validation['step']*rate)+save_s
    validations=math.ceil((config['steps']-last['step'])/config['validate_every'])
    remaining=(config['steps']-last['step'])*rate+validations*overhead
    result['estimate']={'step_s':rate,'validation_and_save_s':overhead,'remaining_validation_runs_approx':validations,
        'remaining_s':remaining,'finish_beijing':(now+timedelta(seconds=remaining)).isoformat(),
        'includes_validation_and_checkpoint_estimate':True,'derived_from_recent_logged_steps':last['step']-first['step']}
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
