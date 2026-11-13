from pathlib import Path
import json, subprocess, collections, os
root=Path('/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107')
out=root/'输出/B_纯100段_stage'
def read(p):
    try:return json.loads(p.read_text())
    except Exception as e:return str(e)
events=[]
for line in (out/'metrics.jsonl').read_text().splitlines():
    try:events.append(json.loads(line))
    except Exception:pass
train=[e for e in events if e.get('event')=='train']
val=[e for e in events if e.get('event')=='validation']
result={'event_counts':dict(collections.Counter(e.get('event') for e in events)),
 'starts':[e for e in events if e.get('event')=='start'],
 'train_first':train[:3], 'train_last':train[-4:],
 'validation':[{k:e.get(k) for k in ('step','time','task_balanced_mae','task_metrics','pre_marker_action_mae','first20_normalized_mae','best_score','improved')} for e in val],
 'tail_events':[{k:v for k,v in e.items() if k not in ('episode_metrics',)} for e in events[-2:]],
 'status':read(out/'status.json'), 'scheduler':read(root/'日志/正式训练调度状态.json'),
 'checkpoint_summaries':{}, 'log_tail':(root/'日志/正式训练.log').read_text(errors='replace').splitlines()[-12:]}
for p in sorted(out.glob('*/checkpoint_summary.json')):
    d=read(p)
    if isinstance(d,dict):result['checkpoint_summaries'][p.parent.name]={'step':d.get('step'),'validation':{k:v for k,v in d.get('validation',{}).items() if k!='episode_metrics'}}
for key,cmd in [('processes',['ps','-eo','pid,etimes,pcpu,pmem,args']),('screen',['screen','-ls']),('gpu',['nvidia-smi']),('disk',['df','-h',str(root)])]:
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
        lines=r.stdout.splitlines()
        if key=='processes':lines=[x for x in lines if any(t in x for t in ('百段','python','screen'))]
        result[key]={'code':r.returncode,'stdout':'\n'.join(lines),'stderr':r.stderr}
    except Exception as e:result[key]={'error':repr(e)}
print(json.dumps(result,ensure_ascii=False,indent=2))
