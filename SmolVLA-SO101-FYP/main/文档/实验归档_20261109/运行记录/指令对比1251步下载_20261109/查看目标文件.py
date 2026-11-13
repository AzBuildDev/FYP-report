from pathlib import Path
import json
p=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108/输出/contrast_纯80段/step_001251')
print('exists',p.exists())
for f in p.rglob('*'):
 if f.is_file():
  print(f.name,f.stat().st_size)
  if f.suffix=='.json':
   d=json.loads(f.read_text());print('keys',list(d))
   if f.name=='checkpoint_summary.json':print('step',d.get('step'))
