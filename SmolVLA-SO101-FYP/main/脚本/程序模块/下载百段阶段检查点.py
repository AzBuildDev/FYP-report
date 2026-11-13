"""下载指定的独立阶段训练检查点并逐文件验签；不修改训练或连接机器人。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import stat
import subprocess
import time

import paramiko

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/AutoDL_纯100段_阶段采样_20261107/输出/B_纯100段_stage'
ANNOTATION = 'fbfa4f589b6800df86fcc9b39b930c3a34946661b3935c5d486cc23c5c67ccf1'
REPORTS = ROOT / '运行记录/AutoDL纯100段部署_20261107'


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=3435)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.step <= 0 or args.step % 1145:
        parser.error('只接受本轮固定独立保存步数，不下载可变last')
    name = f'step_{args.step:06d}'
    remote = REMOTE + '/' + name
    final = ROOT / '模型/纯100段_阶段采样_20261107' / name
    staging = ROOT / '运行环境/临时文件/百段阶段检查点下载' / name
    REPORTS.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.load_host_keys(r'D:\FYP\ssh\known_hosts')
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    for attempt in range(3):
        try:
            client.connect('connect.westb.seetacloud.com', port=31533, username='root',
                key_filename=r'D:\FYP\ssh\autodl_training_monitor', timeout=20,
                auth_timeout=20, banner_timeout=20, look_for_keys=False, allow_agent=False)
            break
        except (OSError, paramiko.SSHException):
            if attempt == 2:
                raise
            time.sleep(3)
    try:
        client.get_transport().set_keepalive(20)
        code = '''from pathlib import Path
import hashlib,json,stat
p=Path(REMOTE_PATH)
if not p.is_dir():
    print(json.dumps({'status':'pending','path':str(p)}));raise SystemExit(0)
summary=json.loads((p/'checkpoint_summary.json').read_text(encoding='utf-8'))
provenance=json.loads((p/'training_provenance.json').read_text(encoding='utf-8'))
assert summary['step']==STEP and summary['run_config']['batch_size']==140
assert summary['run_config']['annotation_sha256']==ANNOTATION_HASH
assert provenance['steps']==STEP and provenance['batch_size']==140 and provenance['status']=='completed'
assert provenance['annotation_sha256']==ANNOTATION_HASH
files=[]
for f in sorted(p.rglob('*')):
    assert not f.is_symlink()
    if not f.is_file():continue
    with f.open('rb') as s:sha=hashlib.file_digest(s,'sha256').hexdigest()
    files.append({'path':f.relative_to(p).as_posix(),'size':f.stat().st_size,'sha256':sha})
by_name={r['path']:r for r in files}
for n in ['config.json','model.safetensors','dataset_stats.json','checkpoint_summary.json','training_state.pt','training_provenance.json']:
    assert by_name[n]['size']>0
assert by_name['model.safetensors']['sha256']==provenance['checkpoint_sha256']
print(json.dumps({'status':'ready','path':str(p),'files':files,'step':STEP,'batch_size':140,'annotation_sha256':ANNOTATION_HASH},ensure_ascii=False))
'''.replace('REMOTE_PATH', repr(remote)).replace('STEP', str(args.step)).replace('ANNOTATION_HASH', repr(ANNOTATION))
        _, stdout, stderr = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(code), timeout=120)
        output, errors = stdout.read().decode('utf-8'), stderr.read().decode('utf-8')
        if stdout.channel.recv_exit_status():
            raise RuntimeError(errors or output)
        manifest = json.loads(output)
        if manifest['status'] != 'ready' or not args.download:
            print(json.dumps(manifest, ensure_ascii=False), flush=True)
            return
        manifest_path = REPORTS / f'{name}_服务器文件清单.json'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        target = final if final.exists() else staging
        target.mkdir(parents=True, exist_ok=True)
        for entry in manifest['files']:
                path = (target / entry['path']).resolve()
                if not path.is_relative_to(target.resolve()):
                    raise ValueError('文件路径越界')
                if path.is_file() and path.stat().st_size == entry['size'] and digest(path) == entry['sha256']:
                    print('复用已校验文件：' + entry['path'], flush=True)
                    continue
                if target == final:
                    raise ValueError('已有目标副本内容不同，保留现场：' + str(path))
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + '.partial')
                print(f"下载 {entry['path']} ({entry['size']} bytes)", flush=True)
                subprocess.run(['scp', '-P', '31533', '-i', r'D:\FYP\ssh\autodl_training_monitor',
                    '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=20',
                    '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=D:/FYP/ssh/known_hosts',
                    '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                    'root@connect.westb.seetacloud.com:' + remote + '/' + entry['path'], str(partial)],
                    check=True, timeout=600)
                if partial.stat().st_size != entry['size'] or digest(partial) != entry['sha256']:
                    raise ValueError('下载SHA256不一致：' + entry['path'])
                partial.replace(path)
        for entry in manifest['files']:
            path = target / entry['path']
            if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
                raise ValueError('最终文件校验失败：' + entry['path'])
        from 管理训练数据来源 import verify_checkpoint_stats
        verify_checkpoint_stats(target, json.loads((target / 'dataset_stats.json').read_text(encoding='utf-8')))
        if target != final:
            final.parent.mkdir(parents=True, exist_ok=True)
            target.rename(final)
        receipt = dict(status='passed', checked_at=datetime.now().isoformat(), model=str(final),
            step=args.step, batch_size=140, sample_windows=args.step * 140,
            equivalent_batch16_steps=args.step * 140 / 16, file_count=len(manifest['files']),
            bytes=sum(r['size'] for r in manifest['files']), sha256_verified=True,
            normalization_verified=True, source_manifest=str(manifest_path),
            training_modified=False, robot_commands_sent=0, offline_startup_pending=True)
        (REPORTS / f'{name}_下载验收.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
