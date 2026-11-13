from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile

base=Path('/root/autodl-tmp')
root=base/'AutoDL_纯100段_阶段采样_20261107'
archive=root.with_suffix('.zip')
expected='2e800a70cdced0b8c8caa0b66b37b2d99499cc3363226b0ab70df443a3fd5bb0'
digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
assert digest==expected,('archive checksum',digest)
print(json.dumps({'archive_sha256':digest,'status':'passed'},ensure_ascii=False),flush=True)
if root.exists():
    raise RuntimeError('保留已有目录，不覆盖')
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        target=(base/info.filename).resolve()
        assert target.is_relative_to(root.resolve()) and not info.is_dir(),info.filename
    z.extractall(base)
(root/'日志').mkdir(exist_ok=True)
print('完整训练包解压完成',flush=True)
subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(root/'.venv')],check=True)
python=str(root/'.venv/bin/python')
constraints=root/'部署依赖约束.txt'
constraints.write_text('torch==2.7.0\ntorchvision==0.22.0\n',encoding='utf-8')
requirements=['transformers==4.51.3','huggingface-hub>=0.34.2,<1.0','tokenizers>=0.21,<0.22',
    'datasets==3.6.0','draccus==0.10.0','diffusers>=0.27.2,<0.36','accelerate>=1.7,<2',
    'safetensors>=0.4.3','einops>=0.8','num2words>=0.5.14','av','pyarrow','pillow',
    'gymnasium>=0.29.1,<1','deepdiff>=7,<9','termcolor','opencv-python-headless','packaging',
    'jsonlines>=4,<5','wandb>=0.20,<0.22','imageio[ffmpeg]>=2.34,<3','pyserial>=3.5']
subprocess.run([python,'-m','pip','install','-c',str(constraints),*requirements],check=True)
env=os.environ.copy()
env.update(PYTHONPATH=str(root/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
probe="""import json,torch,transformers,av,pyarrow; from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'bf16':torch.cuda.is_bf16_supported(),'transformers':transformers.__version__})); assert torch.cuda.is_bf16_supported()"""
subprocess.run([python,'-c',probe],env=env,check=True)
subprocess.run([python,str(root/'准备训练.py')],env=env,cwd=root,check=True)
subprocess.run([python,str(root/'脚本/程序模块/测试阶段采样.py'),'--decode-batch'],env=env,cwd=root,check=True)
(root/'日志/服务器预检通过.json').write_text(json.dumps({'status':'passed','archive_sha256':digest,'training_started':False},ensure_ascii=False,indent=2),encoding='utf-8')
print('服务器环境、完整文件清单、采样及CPU批次预检通过；尚未开始GPU训练',flush=True)
