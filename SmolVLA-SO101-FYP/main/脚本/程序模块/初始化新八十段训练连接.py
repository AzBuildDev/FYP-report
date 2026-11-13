"""从标准输入读取一次性SSH口令，安装既有专用公钥；不保存口令。"""
import base64
import hashlib
import json
from pathlib import Path
import sys
import paramiko
import getpass

HOST='connect.westd.seetacloud.com'
PORT=50948
SSH=Path(r'D:\FYP\ssh')

class FirstUse(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self,client,hostname,key):
        digest=base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        print(json.dumps({'host':hostname,'host_key_type':key.get_name(),'host_key_sha256':digest}),flush=True)
        client.get_host_keys().add(hostname,key.get_name(),key)

def main():
    client=paramiko.SSHClient()
    known=SSH/'known_hosts'
    if known.exists():client.load_host_keys(str(known))
    client.set_missing_host_key_policy(FirstUse())
    print('WAITING_FOR_PASSWORD_ON_STDIN',flush=True)
    password=getpass.getpass('SSH password: ')
    try:
        client.connect(HOST,port=PORT,username='root',password=password,look_for_keys=False,allow_agent=False,
                       timeout=20,auth_timeout=20,banner_timeout=20)
        password=None
        client.save_host_keys(str(known))
        public=(SSH/'autodl_training_monitor.pub').read_text().strip()
        if not public.startswith(('ssh-ed25519 ','ssh-rsa ')):raise ValueError('专用公钥格式异常')
        script="""from pathlib import Path
import os,sys,json,subprocess
key=sys.stdin.readline().strip()
folder=Path('/root/.ssh');folder.mkdir(mode=0o700,exist_ok=True)
file=folder/'authorized_keys'
old=file.read_text() if file.exists() else ''
if key not in old.splitlines():
 with file.open('a') as f:f.write(('\\n' if old and not old.endswith('\\n') else '')+key+'\\n')
folder.chmod(0o700);file.chmod(0o600)
print('PUBLIC_KEY_INSTALLED')
"""
        # 脚本和公钥通过远程Python的参数/标准输入传输，不传私钥。
        import shlex
        _,stdout,stderr=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(script),timeout=30)
        stdout.channel.sendall((public+'\n').encode());stdout.channel.shutdown_write()
        print(stdout.read().decode());error=stderr.read().decode()
        if stdout.channel.recv_exit_status():raise RuntimeError(error)
    finally:
        password=None;client.close()
    check=paramiko.SSHClient();check.load_host_keys(str(known));check.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        check.connect(HOST,port=PORT,username='root',key_filename=str(SSH/'autodl_training_monitor'),
                      look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
        print('KEY_AUTH_VERIFIED',flush=True)
    finally:check.close()

if __name__=='__main__':main()
