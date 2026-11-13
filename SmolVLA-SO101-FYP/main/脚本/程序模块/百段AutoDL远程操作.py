"""使用本机专用密钥执行指定远程脚本；凭据不写入项目。"""
import argparse
from pathlib import Path
import sys

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.load_host_keys(r"D:\FYP\ssh\known_hosts")
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect("connect.westb.seetacloud.com", port=31533, username="root",
                   key_filename=r"D:\FYP\ssh\autodl_training_monitor",
                   timeout=20, auth_timeout=20, banner_timeout=20,
                   look_for_keys=False, allow_agent=False)
    try:
        transport = client.get_transport()
        transport.set_keepalive(20)
        channel = transport.open_session(timeout=20)
        channel.set_combine_stderr(True)
        channel.settimeout(args.timeout)
        channel.exec_command("/root/miniconda3/bin/python -u -")
        channel.sendall(args.script.read_bytes())
        channel.shutdown_write()
        with channel.makefile("rb") as stream:
            for line in stream:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        raise SystemExit(channel.recv_exit_status())
    finally:
        client.close()


if __name__ == "__main__":
    main()
