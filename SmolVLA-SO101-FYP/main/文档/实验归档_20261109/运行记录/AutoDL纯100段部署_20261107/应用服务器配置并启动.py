from pathlib import Path
from datetime import datetime
import hashlib,json,math,os,subprocess

ROOT=Path('/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107')
mode='equal_sample_budget'
batch_report=json.loads((ROOT/'日志/最大稳定batch.json').read_text(encoding='utf-8'))
assert batch_report['status']=='passed'
batch=batch_report['selected_batch_size']
assert batch_report['benchmark']['safe'] and batch_report['benchmark']['headroom_mib']>=1536
assert not (ROOT/'输出').exists(),'已有训练输出，拒绝重复启动'
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
assert not any('百段阶段训练.py' in line or '正式训练调度.py' in line for line in processes.splitlines()),'训练进程仍在运行'
records=ROOT/'部署记录'
records.mkdir(exist_ok=True)
original=ROOT/'文件清单.json'
(records/'上传时文件清单.json').write_bytes(original.read_bytes())
config_path=ROOT/'训练参数_B.json'
(records/'上传时训练参数_B.json').write_bytes(config_path.read_bytes())
config=json.loads(config_path.read_text(encoding='utf-8'))
config['batch_size']=batch
if mode=='equal_sample_budget':
    for key,value in [('steps',30000),('warmup',500),('save_every',5000),('validate_every',1000)]:
        config[key]=math.ceil(value*32/batch)
    # 保存点与每五次验证对齐，避免分别取整后在相邻两步重复验证。
    config['save_every']=5*config['validate_every']
config.update(reference_batch_size=32,reference_steps=30000,reference_sample_budget=960000,
    budget_mode=mode,actual_sample_windows=config['batch_size']*config['steps'],
    note='按样本窗口数等比例换算训练/预热/验证/保存间隔；不同batch优化过程不完全等价；不自动提高学习率')
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
experiment_path=ROOT/'实验说明.json'
experiment=json.loads(experiment_path.read_text(encoding='utf-8'))
experiment['server_training']={'gpu':'RTX 5090 32GB','arm':'B','config':config,'batch_selection':batch_report}
experiment_path.write_text(json.dumps(experiment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
freeze=subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True)
(records/'服务器依赖锁定.txt').write_text(freeze,encoding='utf-8')
receipt={'time':datetime.now().isoformat(),'gpu':'RTX 5090 32GB','config':config,
    'batch_selection':batch_report,'user_authorized_start':True,
    'source_archive_sha256':'2e800a70cdced0b8c8caa0b66b37b2d99499cc3363226b0ab70df443a3fd5bb0',
    'local_original_archive_unchanged':True,'initialization':'official_base_fresh_optimizer'}
(records/'本次部署.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entries=json.loads(original.read_text(encoding='utf-8'))
paths={e['path'] for e in entries}
paths.update(p.relative_to(ROOT).as_posix() for folder in ['部署记录','运行工具'] for p in (ROOT/folder).rglob('*') if p.is_file())
paths.add('部署依赖约束.txt')
new=[]
for relative in sorted(paths):
    path=ROOT/relative
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    new.append({'path':relative,'size':path.stat().st_size,'sha256':digest})
original.write_text(json.dumps(new,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'准备训练.py')],cwd=ROOT,env=env,check=True)
subprocess.run(['screen','-dmS','smolvla100train','bash','-c',
    'cd /root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107 && exec .venv/bin/python -u 运行工具/正式训练调度.py > 日志/正式训练调度.log 2>&1'],check=True)
print(json.dumps({'started':True,'screen':'smolvla100train','config':config,'root':str(ROOT)},ensure_ascii=False),flush=True)
