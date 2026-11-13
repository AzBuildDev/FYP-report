import json,subprocess,shutil
from pathlib import Path
root=Path('/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107')
result={'root_exists':root.exists(),'python_exists':(root/'.venv/bin/python').exists(),'free_gib':shutil.disk_usage(root).free/1024**3}
for key,cmd in [('gpu',['nvidia-smi']),('processes',['ps','-eo','pid,etimes,args'])]:
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
        result[key]={'returncode':p.returncode,'stdout':p.stdout if key=='gpu' else '\n'.join(x for x in p.stdout.splitlines() if any(k in x for k in ('训练','screen','python'))),'stderr':p.stderr}
    except Exception as e:result[key]={'error':repr(e)}
result['dev_gpu']=[str(p) for p in Path('/dev').glob('nvidia*')]
print(json.dumps(result,ensure_ascii=False,indent=2))
