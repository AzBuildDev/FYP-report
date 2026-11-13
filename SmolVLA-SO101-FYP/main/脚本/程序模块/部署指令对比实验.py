"""部署独立实验并进行CPU准备；不启动正式训练或硬件。"""
from pathlib import Path
import hashlib,json,shlex,sys
import paramiko

LOCAL=Path(r'D:\FYP\AutoDL_纯80段_指令对比_20261108')
REMOTE='/root/autodl-tmp/AutoDL_纯80段_指令对比_20261108'
OLD='/root/autodl-tmp/AutoDL_纯80段双物体_阶段采样_20261108'
REPORT=Path(r'D:\FYP\SmolVLA-SO101-FYP\运行记录\指令对比训练改动_20261108')

def main():
    c=paramiko.SSHClient();c.load_host_keys(r'D:\FYP\ssh\known_hosts')
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('connect.westd.seetacloud.com',port=50948,username='root',key_filename=r'D:\FYP\ssh\autodl_training_monitor',look_for_keys=False,allow_agent=False,timeout=20)
    c.get_transport().set_keepalive(20)
    def run(code):
        ch=c.get_transport().open_session();ch.set_combine_stderr(True)
        ch.exec_command('/root/miniconda3/bin/python -u -');ch.sendall(code.encode());ch.shutdown_write()
        data=ch.makefile('rb').read().decode('utf-8');rc=ch.recv_exit_status()
        if rc:raise RuntimeError(data)
        return data
    try:
        manifest=json.loads((LOCAL/'文件清单.json').read_text(encoding='utf-8'))
        plan=run('from pathlib import Path\nimport hashlib,json,os\nr=Path('+repr(REMOTE)+');o=Path('+repr(OLD)+')\nr.mkdir(exist_ok=True)\nm='+repr(manifest)+'\nmissing=[];reused=0\nfor e in m:\n p=r/e["path"];s=o/e["path"]\n def good(x):return x.is_file() and x.stat().st_size==e["size"] and hashlib.sha256(x.read_bytes()).hexdigest()==e["sha256"]\n if p.exists():\n  if not good(p):raise RuntimeError("Existing different file: "+str(p))\n  continue\n p.parent.mkdir(parents=True,exist_ok=True)\n if e["path"].split("/")[0] in ("训练数据","模型","上游框架") and good(s):\n  os.link(s,p);reused+=1\n else:missing.append(e["path"])\nprint(json.dumps({"missing":missing,"reused":reused},ensure_ascii=False))')
        plan=json.loads(plan);print('复用已校验静态文件:',plan['reused'],'待上传:',len(plan['missing']),flush=True)
        sftp=c.open_sftp()
        for rel in plan['missing']+['文件清单.json','数据检查.json']:
            sftp.put(str(LOCAL/rel),REMOTE+'/'+rel)
        sftp.close()
        code='''from pathlib import Path
import subprocess,os,json
r=Path(REMOTE_PATH);o=Path(OLD_PATH)
envdir=r/'.venv'
if not envdir.exists():envdir.symlink_to((o/'.venv').resolve(),target_is_directory=True)
env=os.environ.copy();env.update(PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(r/'上游框架/src'))
for args in ([str(envdir/'bin/python'),'准备训练.py'],[str(envdir/'bin/python'),'脚本/程序模块/测试目标指令对比.py'],[str(envdir/'bin/python'),'-c','import torch; print({"torch":torch.__version__,"cuda":torch.cuda.is_available(),"gpu_count":torch.cuda.device_count()})'],[str(envdir/'bin/python'),'启动指令对比训练.py','--arm','contrast']):
 subprocess.run(args,cwd=r,env=env,check=True)
'''.replace('REMOTE_PATH',repr(REMOTE)).replace('OLD_PATH',repr(OLD))
        result=run(code)
        print(result,flush=True)
        (REPORT/'集群部署准备.log').write_text(result,encoding='utf-8')
        (REPORT/'集群部署准备.json').write_text(json.dumps({'remote':REMOTE,'checks_passed':True,'training_started':False,'benchmark_completed':False,'reuse':plan['reused']},ensure_ascii=False,indent=2),encoding='utf-8')
    finally:c.close()

if __name__=='__main__':main()
