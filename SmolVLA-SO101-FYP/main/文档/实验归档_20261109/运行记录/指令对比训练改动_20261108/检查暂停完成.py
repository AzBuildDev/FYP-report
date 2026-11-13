from pathlib import Path
import subprocess,json
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108/输出/contrast_纯80段')
events=[json.loads(l) for l in (r/'metrics.jsonl').read_text().splitlines() if l.strip()]
print(json.dumps(events[-1],ensure_ascii=False))
s=json.loads((r/'last/checkpoint_summary.json').read_text())
print('LAST_STEP',s['step'],'PAUSE', (r/'PAUSE').exists(),'WRITING',(r/'last.writing').exists())
subprocess.run(['pgrep','-af','百段阶段训练.py|后台指令对比训练.py|启动指令对比训练.py'])
subprocess.run(['screen','-ls'])
subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv'])
