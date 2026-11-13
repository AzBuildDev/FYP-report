"""只读核对新200段录制、训练导出及推理任务标签，不连接机械臂。"""
from pathlib import Path
import sys,json,collections,hashlib
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'脚本/程序模块'))
from 红蓝抓取任务 import TASKS
source=Path(r'D:\FYP\训练数据采集\机械臂控制项目\任务演示\双相机演示_20261101_144759_857724')
export=Path(r'D:\FYP\AutoDL_新200段_4090D_20261102\训练数据')
def jl(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
tasks={p: {r['task_index']:r['task'] for r in jl(p/'meta/tasks.jsonl')} for p in (source,export)}
episodes={p:{r['episode_index']:r for r in jl(p/'meta/episodes.jsonl')} for p in (source,export)}
errors=[];counts=collections.Counter();rows=0;journals=0
for f in sorted((export/'data').rglob('*.parquet')):
 t=pq.read_table(f).to_pydict();ep=t['episode_index'][0];ids=set(t['task_index'])
 if len(ids)!=1:errors.append([ep,'mixed tasks'])
 text=tasks[export][t['task_index'][0]]
 obj=next((k for k,v in TASKS.items() if v==text),None)
 counts[obj]+=1;rows+=len(t['task_index'])
 orig=source/f.relative_to(export)
 st=pq.read_table(orig).to_pydict()
 if st!=t:errors.append([ep,'source/export parquet differ'])
 if episodes[export][ep]['tasks']!=[text]:errors.append([ep,'episode metadata mismatch'])
 if episodes[source][ep]['tasks']!=[text]:errors.append([ep,'source episode metadata mismatch'])
 journal=source/'逐帧动作备份'/f'第{ep+1:03d}段动作备份.jsonl'
 if journal.exists():
  with journal.open(encoding='utf-8') as stream:h=json.loads(next(stream))
  journals+=1
  if h.get('episode')!=ep or h.get('task')!=text or h.get('target_object_id')!=obj:
   errors.append([ep,'journal mismatch',h])
 else:errors.append([ep,'journal missing'])
video_count=0
for f in (export/'videos').rglob('*.mp4'):
 original=source/f.relative_to(export);video_count+=1
 if not original.exists() or original.stat().st_size!=f.stat().st_size:errors.append([str(f),'video missing or size mismatch'])
tokenizer=Tokenizer.from_file(str(ROOT/'模型/视觉语言处理器/tokenizer.json'))
tok={k:tokenizer.encode(v+'\n').ids for k,v in TASKS.items()}
cfg=json.loads((ROOT/'模型/新200段_batch32_20261102/step_020000/config.json').read_text(encoding='utf-8'))
report={'tasks':tasks[export],'mapping_matches':tasks[source]==tasks[export]=={0:TASKS['RC'],1:TASKS['BC']},'episodes':sum(counts.values()),'episode_counts':dict(counts),'frames':rows,'journals_checked':journals,'videos_size_checked':video_count,'errors':errors,'token_ids':tok,'tokens_differ':tok['RC']!=tok['BC'],'tokenizer_max_length':cfg.get('tokenizer_max_length'),'limitation':'核对标签、动作表和视频路径及大小；未逐段人工判定视频中实际抓取物体颜色。'}
out=ROOT/'运行记录/新200段标签链路核查_20261102.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True));print(str(out))
