import subprocess
from pathlib import Path
for cmd in (['nvidia-smi'],['df','-h','/root/autodl-tmp'],['screen','-ls'],['pgrep','-af','python']):
    subprocess.run(cmd)
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108')
print('EXPERIMENT_EXISTS',r.exists())
if r.exists():
    subprocess.run([str(r/'.venv/bin/python'),'-c','import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'])
