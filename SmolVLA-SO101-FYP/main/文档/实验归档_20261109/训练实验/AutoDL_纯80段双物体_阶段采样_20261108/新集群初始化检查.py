"""只读校验迁移文件并检查新GPU环境；不运行优化器。"""
from pathlib import Path
import hashlib,json,os,subprocess
root=Path('/root/autodl-tmp/AutoDL_纯80段双物体_阶段采样_20261108')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
entries=json.loads((root/'文件清单.json').read_text())
for e in entries:
    p=root/e['path']
    if not p.is_file() or p.stat().st_size!=e['size'] or sha(p)!=e['sha256']:raise ValueError('文件仍在拷贝或校验失败：'+e['path'])
env=os.environ.copy();env.update(PYTHONPATH=str(root/'上游框架/src'),PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
python=str(root/'.venv/bin/python')
code='''import torch,json,psutil,os
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available(),
 'bf16':torch.cuda.is_bf16_supported(),'name':torch.cuda.get_device_name(0),'total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2,
 'compute':torch.cuda.get_device_capability(0),'ram_total_gib':psutil.virtual_memory().total/1024**3,'ram_available_gib':psutil.virtual_memory().available/1024**3},ensure_ascii=False))
'''
p=subprocess.run([python,'-c',code],cwd=root,env=env,capture_output=True,text=True,timeout=90)
if p.returncode:raise RuntimeError(p.stderr)
print('ENVIRONMENT '+p.stdout,flush=True)
subprocess.run([python,str(root/'准备训练.py')],cwd=root,env=env,check=True)
print(json.dumps({'file_verification':'passed','files':len(entries),'dataset_episodes':80,'gpu_training_started':False},ensure_ascii=False),flush=True)
