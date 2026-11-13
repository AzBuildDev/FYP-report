"""用专用密钥在新GPU集群执行指定Python脚本；严格核对已保存主机密钥。"""
import argparse
from pathlib import Path
import sys
import paramiko

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args()
    c=paramiko.SSHClient();c.load_host_keys(r'D:\FYP\ssh\known_hosts');c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('connect.westd.seetacloud.com',port=50948,username='root',
        key_filename=r'D:\FYP\ssh\autodl_training_monitor',look_for_keys=False,allow_agent=False,
        timeout=20,auth_timeout=20,banner_timeout=20)
    try:
        c.get_transport().set_keepalive(20)
        ch=c.get_transport().open_session(timeout=20);ch.set_combine_stderr(True);ch.settimeout(a.timeout)
        ch.exec_command('/root/miniconda3/bin/python -u -');ch.sendall(a.script.read_bytes());ch.shutdown_write()
        with ch.makefile('rb') as stream:
            for line in stream:sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
        raise SystemExit(ch.recv_exit_status())
    finally:c.close()

if __name__=='__main__':main()
