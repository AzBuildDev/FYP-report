"""严格匹配已验证主机指纹，在当前GPU端口执行远程脚本。"""
import argparse,base64,hashlib
from pathlib import Path
import paramiko

def connect():
    known=Path(r'D:\FYP\ssh\known_hosts')
    class Verified(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self,client,hostname,key):
            fingerprint=base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
            if fingerprint!='liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc':
                raise RuntimeError('New host fingerprint: '+fingerprint)
            client.get_host_keys().add(hostname,key.get_name(),key)
            client.save_host_keys(str(known))
    c=paramiko.SSHClient();c.load_host_keys(str(known));c.set_missing_host_key_policy(Verified())
    c.connect('connect.westd.seetacloud.com',port=27362,username='root',key_filename=r'D:\FYP\ssh\autodl_training_monitor',look_for_keys=False,allow_agent=False,timeout=20)
    c.get_transport().set_keepalive(20)
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--script',type=Path,required=True);a=p.parse_args()
    c=connect()
    try:
        ch=c.get_transport().open_session();ch.set_combine_stderr(True)
        ch.exec_command('/root/miniconda3/bin/python -u -');ch.sendall(a.script.read_bytes());ch.shutdown_write()
        for line in ch.makefile('rb'):
            print(line.decode('utf-8',errors='replace'),end='',flush=True)
        raise SystemExit(ch.recv_exit_status())
    finally:c.close()
if __name__=='__main__':main()
