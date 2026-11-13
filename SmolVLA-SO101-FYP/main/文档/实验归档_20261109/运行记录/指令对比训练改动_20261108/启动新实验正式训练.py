from pathlib import Path
import subprocess,os,json,shlex
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108')
selection=json.loads((r/'吞吐选择.json').read_text());best=selection['selected']
config=json.loads((r/'训练参数_contrast.json').read_text())
assert config['batch_size']==best['batch_size']
assert best['status']=='passed' and best['gpu_total_mib']-best['peak_gpu_reserved_mib']>=2048
if (r/config['output_relative']).exists():raise RuntimeError('正式输出已存在，拒绝重复启动')
runner=r/'后台指令对比训练.py'
runner.write_text('''from pathlib import Path
import subprocess,os,json,datetime
r=Path(__file__).resolve().parent
env=os.environ.copy();env.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(r/'上游框架/src'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
cmd=[str(r/'.venv/bin/python'),'-u',str(r/'启动指令对比训练.py'),'--arm','contrast','--run']
with (r/'日志/正式指令对比训练.log').open('a',buffering=1) as log:
    print('开始16步保存恢复验收',file=log,flush=True)
    subprocess.run(cmd+['--stop-after','16'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('16步保存完成，严格恢复继续正式训练',file=log,flush=True)
    subprocess.run(cmd+['--resume'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('正式训练正常完成',file=log,flush=True)
''')
subprocess.run(['screen','-dmS','smolvla80contrast',str(r/'.venv/bin/python'),'-u',str(runner)],cwd=r,check=True)
print(json.dumps({'screen':'smolvla80contrast','config':config,'benchmark':best,'started':True},ensure_ascii=False))
