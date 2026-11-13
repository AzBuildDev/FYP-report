"""仅同步新80段实验的配置和检查记录，不修改服务器训练。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,paramiko
L=Path(__file__).resolve().parent
R='/root/autodl-tmp/AutoDL_纯80段双物体_阶段采样_20261108'
B=L/'启动前配置备份';B.mkdir(exist_ok=True)
c=paramiko.SSHClient();c.load_host_keys(r'D:\FYP\ssh\known_hosts');c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.connect('connect.westd.seetacloud.com',port=50948,username='root',key_filename=r'D:\FYP\ssh\autodl_training_monitor',look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
files=['训练参数_B.json','实验说明.json','文件清单.json','配置调整前_batch140.json','正式训练调度.py','显存实测/批次搜索结果.json','日志/GPU保存验收.json','日志/正式训练调度状态.json']
try:
    c.get_transport().set_keepalive(20)
    s=c.open_sftp();results=[]
    for rel in files:
        data=s.open(R+'/'+rel,'rb').read();p=L/rel
        if p.exists() and p.read_bytes()!=data:
            old=B/rel;old.parent.mkdir(parents=True,exist_ok=True)
            if not old.exists():old.write_bytes(p.read_bytes())
        p.parent.mkdir(parents=True,exist_ok=True)
        t=p.with_suffix(p.suffix+'.tmp');t.write_bytes(data);t.replace(p)
        results.append({'path':rel,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    m=json.loads(s.open(R+'/输出/B_纯80段双物体_stage/last/checkpoint_summary.json','r').read())
    records=s.open(R+'/输出/B_纯80段双物体_stage/metrics.jsonl','r').read().decode('utf-8')
    events=[json.loads(x) for x in records.splitlines()]
    state={'checked_at':datetime.now().isoformat(),'host':'connect.westd.seetacloud.com','port':50948,'root':R,'files':results,'saved_step':m['step'],'last_start':next(x for x in reversed(events) if x['event']=='start'),'latest_train':next(x for x in reversed(events) if x['event']=='train'),'config':json.loads((L/'训练参数_B.json').read_text(encoding='utf-8'))}
    (L/'正式启动核对记录.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in state.items() if k not in ('files','last_start')},ensure_ascii=False,indent=2))
finally:c.close()
