from pathlib import Path
import subprocess,json
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108')
subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw','--format=csv'])
subprocess.run(['screen','-ls'])
for f in sorted((r/'日志').glob('*.log'),key=lambda p:p.stat().st_mtime)[-2:]:
    print('\nLOG',f.name)
    print(f.read_text(errors='replace')[-3000:])
for f in (r/'输出').glob('*/metrics.jsonl'):
    print('METRICS',str(f));print('\n'.join(f.read_text().splitlines()[-4:]))
for f in (r/'输出').glob('*/last/checkpoint_summary.json'):
    print('LAST',f.read_text())
print('DISK');subprocess.run(['df','-h',str(r)])
