"""只读核对训练权重、任务和已保存动作；不加载GPU、不连接真实硬件。"""
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT.parent/'AutoDL_纯100段_阶段采样_20261107'
MODEL=ROOT/'模型/纯100段_阶段采样_20261107/step_003435'
OUT=ROOT/'运行记录/AutoDL纯100段部署_20261107'

def main():
    torch.set_num_threads(1)
    snapshot=json.loads((OUT/'训练诊断服务器快照.json').read_text(encoding='utf-8-sig'))
    summary=json.loads((MODEL/'checkpoint_summary.json').read_text(encoding='utf-8'))
    tasks={d['task_index']:d['task'] for d in map(json.loads,(PACKAGE/'训练数据/meta/tasks.jsonl').read_text().splitlines())}
    from 红蓝抓取任务 import TASKS
    task_errors=[]
    counts=Counter()
    speeds=[]
    tables={}
    for p in sorted((PACKAGE/'训练数据/data').glob('chunk-*/*.parquet')):
        t=pq.read_table(p).to_pydict()
        ep=int(t['episode_index'][0]);tables[ep]=t
        idx=set(t['task_index'])
        if len(idx)!=1 or next(iter(idx)) not in tasks:task_errors.append(ep)
        else:counts[tasks[next(iter(idx))]]+=1
        speeds.append(np.abs(np.diff(np.array(t['action']),axis=0)))
    speeds=np.concatenate(speeds)
    weight_groups={}
    with safe_open(str(PACKAGE/'模型/基础预训练模型/model.safetensors'),framework='pt',device='cpu') as b, safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as m:
        mapping={k.replace('model._orig_mod.','model.'):k for k in b.keys()}
        for key in m.keys():
            if key.startswith(('normalize_','unnormalize_')):continue
            group='vlm_frozen' if '.vlm.' in key else ('expert' if '.lm_expert.' in key else 'projections')
            row=weight_groups.setdefault(group,{'tensors':0,'changed_tensors':0,'nonfinite_tensors':0,'shape_mismatch':[],'missing_in_base':[],'max_abs_difference':0.0})
            row['tensors']+=1
            v=m.get_tensor(key)
            if not torch.isfinite(v).all():row['nonfinite_tensors']+=1
            if key not in mapping:row['missing_in_base'].append(key);continue
            ref=b.get_tensor(mapping[key])
            if ref.shape!=v.shape:row['shape_mismatch'].append(key);continue
            ref=ref.to(v.dtype)
            if not torch.equal(ref,v):
                row['changed_tensors']+=1
                row['max_abs_difference']=max(row['max_abs_difference'],float((ref.float()-v.float()).abs().max()))
    runs=[]
    candidates=sorted((ROOT/'运行记录').glob('实机抓取测试_20261107_22*.jsonl'))+sorted((ROOT/'运行记录').glob('实机抓取测试_20261108_01*.jsonl'))
    for p in candidates:
        events=[]
        for line in p.read_text().splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
        actions=[e for e in events if e.get('type')=='action']
        chunks=[e for e in events if e.get('type')=='async_chunk']
        if not actions:continue
        model=np.array([e['model_target'] for e in actions]);sent=np.array([e['sent_target'] for e in actions])
        diff=np.abs(model-sent)
        r={'log':p.name,'task':events[0]['task'],'model_identity':next((e for e in events if e.get('type')=='model_identity'),None),
           'actions':len(actions),'end':next((e for e in reversed(events) if e.get('type')=='end'),None),
           'errors':[e['error'] for e in events if e.get('type')=='error'],
           'inference_seconds_median':float(np.median([e['inference_s'] for e in chunks])) if chunks else None,
           'skipped_prefix_mean':float(np.mean([e['skip_prefix'] for e in chunks])) if chunks else None,
           'model_minus_sent_abs_mean_per_joint':diff.mean(0).tolist(),
           'model_minus_sent_abs_max_per_joint':diff.max(0).tolist(),
           'fraction_model_minus_sent_gt1_per_joint':(diff>1).mean(0).tolist()}
        runs.append(r)
    report={'checked_at':datetime.now().astimezone().isoformat(),'hardware_connected':False,'gpu_used':False,
        'checkpoint':str(MODEL),'checkpoint_step':summary['step'],'server_status':snapshot['status'],
        'server_scheduler':snapshot['scheduler'],
        'weights':weight_groups,'task_texts_match_inference':tasks=={0:TASKS['RC'],1:TASKS['BC']},
        'task_episode_counts':dict(counts),'episode_task_errors':task_errors,
        'action_step_abs_p95_per_joint':np.percentile(speeds,95,axis=0).tolist(),
        'action_step_fraction_exceeding_current_limit_0_5':(speeds>0.5).mean(0).tolist(),
        'local_summary_validation':{k:v for k,v in summary['validation'].items() if k!='episode_metrics'},
        'server_validation_curve':snapshot['validation'],'runs':runs,
        'limitations':['相同窗口总量不代表相同优化步数或效果。','动作误差指标不是选色成功率。','实际下发与模型输出差异包括限速、活动范围、跟随保护和夹爪保护，不能全部归因于单一保护。','未用此检查证明数据视觉标签全部正确，也未测量闭环选色成功率。']}
    output=OUT/'训练与实机偏差诊断_20261108.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','server_validation_curve','local_summary_validation')},ensure_ascii=False,indent=2))
    print(output)

if __name__=='__main__':main()
