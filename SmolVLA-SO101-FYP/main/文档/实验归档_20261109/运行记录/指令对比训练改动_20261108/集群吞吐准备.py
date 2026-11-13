from pathlib import Path
import subprocess,os,json,hashlib,math,time
r=Path('/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108')
v=r/'.venv'
if not v.exists():v.symlink_to('/root/autodl-tmp/指令对比环境',target_is_directory=True)
py=str(v/'bin/python')
env=os.environ.copy();env.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(r/'上游框架/src'),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
logs=r/'日志';logs.mkdir(exist_ok=True)
for args in ([py,'准备训练.py'],[py,'脚本/程序模块/测试目标指令对比.py'],[py,'-c','from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; import torch; print(torch.__version__,torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported()']):
    subprocess.run(args,cwd=r,env=env,check=True)

config_path=r/'训练参数_contrast.json'
config=json.loads(config_path.read_text())
manifest_path=r/'文件清单.json'
def configure(batch):
    config.update(batch_size=batch,steps=math.ceil(960000/batch),warmup=math.ceil(16000/batch),save_every=math.ceil(160000/batch),validate_every=math.ceil(32000/batch))
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    m=json.loads(manifest_path.read_text())
    for e in m:
        if e['path']=='训练参数_contrast.json':e.update(size=config_path.stat().st_size,sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
results=[]
for batch in (256,384,448):
    configure(batch)
    path=logs/f'显存实测_batch{batch}.log'
    print('BENCHMARK_START',batch,flush=True)
    with path.open('w') as f:
        p=subprocess.run([py,'启动指令对比训练.py','--arm','contrast','--benchmark'],cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT)
    result_path=r/'显存实测'/f'contrast_batch{batch}'/'显存吞吐实测.json'
    if p.returncode:
        tail=path.read_text()[-6000:];print('BENCHMARK_FAILED',batch,tail,flush=True)
        if 'out of memory' in tail.lower():break
        raise RuntimeError('非显存错误，停止并排查')
    result=json.loads(result_path.read_text());results.append(result)
    print('BENCHMARK_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
    if result['gpu_total_mib']-result['peak_gpu_reserved_mib']<2048:break
valid=[b for b in results if b['gpu_total_mib']-b['peak_gpu_reserved_mib']>=2048]
if not valid:raise RuntimeError('没有通过的batch')
best=max(valid,key=lambda b:b['samples_per_second']);configure(best['batch_size'])
(r/'吞吐选择.json').write_text(json.dumps({'results':results,'selected':best,'formal_training_started':False},ensure_ascii=False,indent=2))
print('SELECTED',json.dumps(best,ensure_ascii=False),flush=True)
