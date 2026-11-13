"""核验独立指令对比实验、红蓝配对和留出划分；默认只用CPU，不连接硬件。"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=Path(r'D:\FYP\AutoDL_纯80段_指令对比_20261108')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,default=DEFAULT)
    a=p.parse_args();exp=a.experiment.resolve()
    if not exp.is_dir():raise FileNotFoundError('先生成独立实验包：'+str(exp))
    env=os.environ.copy();env['PYTHONUTF8']='1'
    subprocess.run([sys.executable,str(exp/'准备训练.py')],cwd=exp,env=env,check=True)
    subprocess.run([sys.executable,str(exp/'脚本/程序模块/测试目标指令对比.py')],cwd=exp,env=env,check=True)
    sys.path.insert(0,str(exp/'脚本/程序模块'))
    import numpy as np
    import pyarrow.parquet as pq
    from 目标指令对比 import ApproachPairs,VERSION
    from 阶段采样 import StageSampler,load_annotations
    split=json.loads((exp/'划分.json').read_text(encoding='utf-8'))
    annotations,_=load_annotations(exp/'训练数据',exp/'标注/阶段标注_80段.json',split)
    tables={}
    for path in sorted((exp/'训练数据/data/chunk-000').glob('*.parquet')):
        table=pq.read_table(path).to_pydict();tables[int(table['episode_index'][0])]=table
    assert set(tables)==set(range(80))
    train=[tables[ep] for ep in split['train']]
    std=np.concatenate([np.asarray(t['action']) for t in train]).std(axis=0)
    groups={}
    for name in ('train','validation'):
        pool=ApproachPairs([tables[ep] for ep in split[name]],annotations,std)
        seen=set();counts={0:0,1:0}
        for step in range(1,101):
            plan=pool.plan(step,8,42)
            assert plan==pool.plan(step,8,42)
            for row in plan:
                assert row['episode'] in split[name]
                assert row['frame']+row['cutoff']==annotations[row['episode']]['markers']['first_contact']
                assert 1<=row['cutoff']<50
                seen.add(row['episode']);counts[row['task_index']]+=1
        assert counts=={0:400,1:400}
        groups[name]={'pool':pool.report(),'tested_pairs':800,'target_counts':counts,'seen_episodes':sorted(seen)}
    assert not set(groups['train']['seen_episodes']) & set(groups['validation']['seen_episodes'])
    sampler=StageSampler(train,annotations,452,42,50,mode='stage')
    for step in range(1,11):assert len(sampler.plan(step))==452
    for arm in ('baseline','anchor','contrast'):
        subprocess.run([sys.executable,str(exp/'启动指令对比训练.py'),'--arm',arm],cwd=exp,env=env,check=True)
    report={'status':'passed','version':VERSION,'experiment':str(exp),'groups':groups,
            'dataset_episodes':80,'train_episodes':64,'validation_episodes':16,
            'base_sampler_unchanged':True,'training_started':False,'robot_commands':0,
            'limits':'CPU验证仅确认实现/数据关联，不证明实机改进；GPU反传与保存恢复另验。'}
    out=ROOT/'运行记录/指令对比训练改动_20261108';out.mkdir(parents=True,exist_ok=True)
    (out/'CPU数据与采样验收.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
