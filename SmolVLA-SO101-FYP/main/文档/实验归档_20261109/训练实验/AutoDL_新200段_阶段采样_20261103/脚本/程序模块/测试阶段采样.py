"""离线核对真实200段的采样范围、颜色、划分与逐步恢复一致性；不发出硬件指令。"""
from pathlib import Path
from collections import Counter
import json,copy
import pyarrow.parquet as pq
from 阶段采样 import load_annotations,StageSampler
ROOT=Path(__file__).resolve().parents[2]

def main():
    split=json.loads((ROOT/'划分.json').read_text(encoding='utf-8'))
    annotations,report=load_annotations(ROOT/'训练数据',ROOT/'标注/阶段标注_200段.json',split)
    tables=[pq.read_table(p).to_pydict() for p in sorted((ROOT/'训练数据').glob('data/chunk-*/*.parquet'))]
    train=[tables[i] for i in split['train']]
    counts=Counter();crossed=0;selected=set()
    for batch in (16,32,64,128,256,384):
        sampler=StageSampler(train,annotations,batch)
        fresh=StageSampler(train,annotations,batch)
        for step in range(1,101):
            plan=sampler.plan(step)
            assert plan==fresh.plan(step)
            groups=Counter((r['stage'],r['task_index']) for r in plan)
            assert len(groups)==4 and set(groups.values())=={batch//4}
            assert plan!=sampler.plan(step+1)
            for r in plan:
                ep=r['episode'];row=annotations[ep];start=r['frame']
                assert ep in split['train'] and ep not in split['validation']
                assert 0<=start<=row['length']-50
                assert row['target']==('RC' if r['task_index']==0 else 'BC')
                if r['stage']=='pre_marker':
                    assert start<row['markers']['first_contact']
                    crossed+=int(start+49>=row['markers']['first_contact'])
                assert len(tables[ep]['action'][start:start+50])==50
                selected.add(ep);counts[f"batch{batch}_{r['stage']}_{r['task_index']}"]+=1
        # 预读相邻步不改变断点恢复所用样本；调用顺序不消耗全局采样状态。
        assert sampler.plan(88)==fresh.plan(88)
    assert len(selected)==160 and crossed>0
    for bad in (0,3,15):
        try:StageSampler(train,annotations,bad)
        except ValueError:pass
        else:raise AssertionError('非法batch未拒绝')
    wrong=copy.deepcopy(annotations);wrong[split['train'][0]]['target']='WRONG'
    try:StageSampler(train,wrong,32)
    except ValueError:pass
    else:raise AssertionError('颜色错配未拒绝')
    report.update(status='passed',tested_batches=[16,32,64,128,256,384],tested_steps_per_batch=100,
                  train_episodes_observed=len(selected),pre_marker_windows_crossing_marker=crossed,
                  exact_resume_sampling=True,counts=dict(counts),robot_commands_sent=0)
    (ROOT/'阶段采样核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
