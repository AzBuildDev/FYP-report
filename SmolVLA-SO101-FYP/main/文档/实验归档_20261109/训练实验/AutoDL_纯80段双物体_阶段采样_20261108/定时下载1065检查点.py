"""只读检查并完整下载独立1065步检查点；逐文件SHA256后原子发布。"""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import hashlib,inspect,json,os,shutil,time,paramiko

ROOT='/root/autodl-tmp/AutoDL_纯80段双物体_阶段采样_20261108'
REMOTE=ROOT+'/输出/B_纯80段双物体_stage/step_001065'
LOCAL=Path(r'D:\FYP\SmolVLA-SO101-FYP\模型\纯80段双物体_阶段采样_20261108\step_001065')
STAGING=Path(r'D:\FYP\下载缓存备份\纯80段_20261108_step_001065.partial')
REPORT=Path(r'D:\FYP\SmolVLA-SO101-FYP\运行记录\AutoDL纯80段下载_20261108')
EXPECTED_ANN='87b03225b182350eb009a177ac8fdb246327d76a8c7f58ce5fe2d44927b49662'
EXPECTED_DATA='e89ea5343c226211d8748a39fca2383d34347cddf2c304a0850347a3c15e10c6'
EXPECTED_BASE='8f8dc071d5b933e79edd2b73b8d6b5cca482ef0437c099ea3ec13ab978a38fc8'
TZ=timezone(timedelta(hours=8))
DEADLINE=datetime(2026,11,8,17,0,tzinfo=TZ)
REPORT.mkdir(parents=True,exist_ok=True)

def now():return datetime.now(TZ).isoformat()
def save(name,obj):
    p=REPORT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def connect():
    c=paramiko.SSHClient();c.load_host_keys(r'D:\FYP\ssh\known_hosts');c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('connect.westd.seetacloud.com',port=50948,username='root',key_filename=r'D:\FYP\ssh\autodl_training_monitor',look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
    c.get_transport().set_keepalive(20);return c

PROBE=r'''
from pathlib import Path
import hashlib,json
R=Path(ROOT);P=Path(REMOTE);O=P.parent
def read(p):return json.loads(p.read_text())
events=[json.loads(x) for x in (O/'metrics.jsonl').read_text().splitlines() if x.strip()]
result={'time':__import__('datetime').datetime.now().isoformat(),'ready':P.is_dir(),
        'latest_train':next((e for e in reversed(events) if e['event']=='train'),None),
        'latest_saved':next((e for e in reversed(events) if e['event']=='checkpoint_saved'),None)}
if result['ready']:
    result['summary']=read(P/'checkpoint_summary.json')
    result['provenance']=read(P/'training_provenance.json')
    result['normalization']=read(P/'dataset_stats.json')
    files=[]
    for p in sorted(P.rglob('*')):
        if p.is_symlink():raise RuntimeError('Checkpoint contains unexpected symlink')
        if not p.is_file():continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
        files.append({'path':p.relative_to(P).as_posix(),'size':p.stat().st_size,'sha256':h.hexdigest()})
    result['files']=files
print(json.dumps(result,ensure_ascii=False))
'''
def probe(c):
    ch=c.get_transport().open_session(timeout=20);ch.settimeout(240)
    ch.exec_command('/root/miniconda3/bin/python -u -')
    ch.sendall(('ROOT='+repr(ROOT)+'\nREMOTE='+repr(REMOTE)+'\n'+PROBE).encode('utf-8'));ch.shutdown_write()
    out=ch.makefile('rb').read();err=ch.makefile_stderr('rb').read();code=ch.recv_exit_status();ch.close()
    if code:raise RuntimeError('Remote probe failed: '+err.decode('utf-8',errors='replace')[-2000:])
    return json.loads(out)
def validate(d):
    s=d['summary'];c=s['run_config'];p=d['provenance']
    assert s['step']==1065 and c['batch_size']==452
    assert c['annotation_sha256']==p['annotation_sha256']==EXPECTED_ANN
    assert c['stage_sampling']['dataset_fingerprint']==EXPECTED_DATA
    assert c['base_weights_sha256']==EXPECTED_BASE
    assert c['initialization']=='official_base_fresh_optimizer'
    assert p['steps']==1065 and p['batch_size']==452 and p['status']=='completed'
    tr=c['train_episodes'];va=c['validation_episodes']
    assert len(tr)==len(set(tr))==64 and len(va)==len(set(va))==16
    assert not set(tr)&set(va) and set(tr)|set(va)==set(range(80))
    split=json.loads(Path(r'D:\FYP\AutoDL_纯80段双物体_阶段采样_20261108\划分.json').read_text(encoding='utf-8-sig'))
    assert tr==split['train'] and va==split['validation']
    assert p['training_data']==c['training_data']
    assert sorted(e['episode'] for e in p['training_data']['episodes'])==sorted(tr)
    assert sorted(e['episode'] for e in c['validation_data']['episodes'])==sorted(va)
    assert c['stage_sampling']['train_target_counts']=={'RC':32,'BC':32}
    assert c['stage_sampling']['validation_target_counts']=={'RC':8,'BC':8}
    files={f['path']:f for f in d['files']}
    required={'config.json','model.safetensors','training_state.pt','dataset_stats.json','training_provenance.json','checkpoint_summary.json'}
    assert required<=files.keys() and all(f['size']>0 for f in files.values())
    assert files['model.safetensors']['sha256']==p['checkpoint_sha256']
    for key in ('action','observation.state'):
        assert key in d['normalization']
    for rel in files:
        q=PurePosixPath(rel)
        if q.is_absolute() or '..' in q.parts or ':' in rel:raise ValueError('Invalid relative path')
    return files

def main():
    d=None
    while True:
        for attempt in range(3):
            try:
                c=connect()
                try:d=probe(c)
                finally:c.close()
                break
            except (OSError,paramiko.SSHException) as exc:
                if attempt==2:raise
                print('CONNECTION_RETRY',attempt+1,type(exc).__name__,flush=True);time.sleep(5)
        save('检查状态.json',{'checked_at':now(),'ready':d['ready'],'latest_train':d['latest_train'],'latest_saved':d['latest_saved']})
        print('PROBE',json.dumps({k:d[k] for k in ('ready','latest_train','latest_saved')},ensure_ascii=False),flush=True)
        if d['ready']:break
        if datetime.now(TZ)>=DEADLINE:raise TimeoutError('17:00 deadline reached; target unavailable')
        time.sleep(45)
    files=validate(d)
    save('服务器完整文件清单.json',d['files']);save('checkpoint_summary.json',d['summary']);save('training_provenance.json',d['provenance'])
    total=sum(f['size'] for f in files.values())
    if LOCAL.exists():target=LOCAL
    else:
        STAGING.mkdir(parents=True,exist_ok=True);target=STAGING
        if shutil.disk_usage(target).free<total+1024**3:raise OSError('Insufficient local disk space')
    verified=[]
    for rel,row in files.items():
        p=target/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not(p.exists() and p.stat().st_size==row['size'] and sha(p)==row['sha256']):
            if target==LOCAL:raise RuntimeError('Existing destination differs; refusing overwrite: '+rel)
            for attempt in range(3):
                try:
                    c=connect()
                    try:
                        s=c.open_sftp();s.get_channel().settimeout(120)
                        kwargs={'max_concurrent_prefetch_requests':32} if 'max_concurrent_prefetch_requests' in inspect.signature(s.get).parameters else {}
                        t=p.with_suffix(p.suffix+'.downloading')
                        print('DOWNLOAD_START',rel,row['size'],flush=True)
                        s.get(REMOTE+'/'+rel,str(t),**kwargs);s.close()
                    finally:c.close()
                    if t.stat().st_size!=row['size'] or sha(t)!=row['sha256']:raise ValueError('Download hash mismatch: '+rel)
                    t.replace(p);break
                except (OSError,paramiko.SSHException) as exc:
                    if attempt==2:raise
                    print('DOWNLOAD_RETRY',rel,attempt+1,type(exc).__name__,flush=True);time.sleep(5)
        verified.append(row);print('VERIFIED',rel,row['size'],flush=True)
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}==set(files)
    # 再次读取远程清单，确保整个传输期间独立检查点内容未变化。
    c=connect()
    try:after=probe(c)
    finally:c.close()
    assert after['ready'] and after['files']==d['files']
    if target!=LOCAL:
        LOCAL.parent.mkdir(parents=True,exist_ok=True);STAGING.rename(LOCAL)
    result={'status':'completed','finished_at':now(),'host':'connect.westd.seetacloud.com','port':50948,
            'remote':REMOTE,'local':str(LOCAL),'step':1065,'batch_size':452,'sampled_windows':481380,
            'batch16_equivalent_steps':30086.25,'files':len(verified),'bytes':total,
            'all_sha256_verified':True,'remote_unchanged_during_transfer':True,'robot_started':False,
            'server_training_modified':False,'model_selection_modified':False}
    save('本地逐文件校验.json',verified);save('下载完成.json',result)
    print('COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('本次执行错误.json',{'time':now(),'type':type(exc).__name__,'error':str(exc)})
        raise
