"""通过专用密钥部署已验收80段独立实验；不申请GPU或改动旧训练。"""
from pathlib import Path
import json
import sys
import paramiko

ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT.parent/'AutoDL_纯80段双物体_阶段采样_20261108'
REMOTE_SCRIPT=r'''
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile,os
source=Path('/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107')
root=source.with_name('AutoDL_纯80段双物体_阶段采样_20261108')
archive=source.with_name('纯80段部署小文件_20261108.tar.gz')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
if root.exists():raise ValueError('独立实验目录已存在，拒绝覆盖')
if shutil.disk_usage(source).free<10*1024**3:raise ValueError('服务器可用空间不足10GiB')
root.mkdir()
with tarfile.open(archive,'r:gz') as t:
    for member in t.getmembers():
        if not (root/member.name).resolve().is_relative_to(root) or member.issym() or member.islnk():
            raise ValueError('部署包路径异常')
    t.extractall(root,filter='data')
entries=json.loads((root/'文件清单.json').read_text())
copied=0
for entry in entries:
    dest=root/entry['path']
    if not dest.exists():
        src=source/entry['path']
        if not src.is_file() or src.stat().st_size!=entry['size'] or sha(src)!=entry['sha256']:
            raise ValueError('共享源文件与独立实验预期不符：'+entry['path'])
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest);copied+=1
    if dest.stat().st_size!=entry['size'] or sha(dest)!=entry['sha256']:
        raise ValueError('部署文件校验失败：'+entry['path'])
print(json.dumps({'files_verified':len(entries),'files_copied':copied},ensure_ascii=False),flush=True)
(root/'.venv').symlink_to(source/'.venv',target_is_directory=True)
environment=os.environ.copy()
environment.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(root/'上游框架/src'),
                   HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
python=str(root/'.venv/bin/python')
for args in ([python,str(root/'准备训练.py')],
             [python,str(root/'脚本/程序模块/测试阶段采样.py'),'--decode-batch'],
             [python,str(root/'启动训练.py'),'--arm','B']):
    subprocess.run(args,cwd=root,env=environment,check=True)
probe=subprocess.run([python,'-c','import torch,json;print(json.dumps({"cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count()}))'],
                     cwd=root,env=environment,capture_output=True,text=True,timeout=60)
receipt={'status':'prepared_on_cluster','root':str(root),'files_verified':len(entries),
         'gpu_probe':{'code':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr},
         'training_started':False,'paid_instance_modified':False,
         'train_episodes':64,'validation_episodes':16,'scene':'only dual-object episodes 0..79',
         'next_command':'cd '+str(root)+' && .venv/bin/python 启动训练.py --arm B --run'}
(root/'部署验收.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print('DEPLOYMENT_RESULT '+json.dumps(receipt,ensure_ascii=False),flush=True)
'''

def main():
    client=paramiko.SSHClient()
    client.load_host_keys(r'D:\FYP\ssh\known_hosts')
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect('connect.westb.seetacloud.com',port=31533,username='root',
        key_filename=r'D:\FYP\ssh\autodl_training_monitor',look_for_keys=False,allow_agent=False,
        timeout=20,auth_timeout=20,banner_timeout=20)
    output=LOCAL/'集群部署日志.txt'
    try:
        client.get_transport().set_keepalive(20)
        with client.open_sftp() as sftp:
            sftp.put(str(LOCAL.with_name(LOCAL.name+'_部署小文件.tar.gz')),
                     '/root/autodl-tmp/纯80段部署小文件_20261108.tar.gz')
        channel=client.get_transport().open_session(timeout=20)
        channel.set_combine_stderr(True);channel.settimeout(600)
        channel.exec_command('/root/miniconda3/bin/python -u -')
        channel.sendall(REMOTE_SCRIPT.encode('utf-8'));channel.shutdown_write()
        with channel.makefile('rb') as stream,output.open('wb') as saved:
            for line in stream:
                saved.write(line);saved.flush();sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
                text=line.decode('utf-8',errors='replace')
                if text.startswith('DEPLOYMENT_RESULT '):
                    (LOCAL/'集群部署验收.json').write_text(json.dumps(json.loads(text.removeprefix('DEPLOYMENT_RESULT ')),ensure_ascii=False,indent=2),encoding='utf-8')
        raise SystemExit(channel.recv_exit_status())
    finally:client.close()

if __name__=='__main__':main()
