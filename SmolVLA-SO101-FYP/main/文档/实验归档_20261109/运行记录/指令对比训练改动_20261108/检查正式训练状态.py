from pathlib import Path
import json,subprocess,time
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108')
p=r/'输出/contrast_纯80段/metrics.jsonl'
events=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
for kind in ['start','train','validation','checkpoint_saved','stopped_for_resume_test','completed']:
    found=[e for e in events if e.get('event')==kind]
    if found:
        e=found[-1]
        print(kind,json.dumps({k:v for k,v in e.items() if k in ['step','time','loss_avg50','elapsed_s','eta_s','lr','grad_norm','instruction','mae','improved']},ensure_ascii=False))
subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv'])
print('SCREEN');subprocess.run(['screen','-ls'])
log=r/'日志/正式指令对比训练.log'
tail=log.read_text(errors='replace')[-3000:]
if 'Traceback' in tail:print(tail)
summary=r/'输出/contrast_纯80段/last/checkpoint_summary.json'
if summary.exists():
    data=json.loads(summary.read_text());print('CHECKPOINT',json.dumps(data,ensure_ascii=False)[:1500])
