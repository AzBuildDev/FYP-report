"""在AutoDL解压后验证清单、完整解码视频并生成本机检查报告；不训练。"""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'脚本/程序模块'), str(ROOT/'上游框架/src')]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

def verify_manifest():
    entries = json.loads((ROOT/'文件清单.json').read_text(encoding='utf-8'))
    for entry in entries:
        p = ROOT/entry['path']
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024**2), b''):
                h.update(chunk)
        if h.hexdigest() != entry['sha256']:
            raise RuntimeError(f'文件校验失败：{p}')
    print(f'清单校验通过：{len(entries)}个文件', flush=True)
    return entries

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--已核验报告', type=Path, help='导入与数据SHA256绑定的本地完整解码报告，避免重复解码')
    args = p.parse_args()
    entries = verify_manifest()
    if args.已核验报告:
        receipt = json.loads(args.已核验报告.read_text(encoding='utf-8'))
        data_entries = [e for e in entries if e['path'].startswith('训练数据/')]
        fingerprint = hashlib.sha256(json.dumps(data_entries, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        if receipt['verified_data_manifest_sha256'] != fingerprint or receipt['report']['status'] != 'passed':
            raise RuntimeError('已核验报告未绑定本次数据，不能导入')
        report = receipt['report']
        report['verification_origin'] = {'method':'本地逐帧完整解码，服务器核对全部文件SHA256后导入', 'original_dataset_root':report['dataset_root']}
        report['dataset_root'] = str((ROOT/'训练数据').resolve())
        (ROOT/'数据检查.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('全部数据SHA256与本地完整解码凭据一致；已生成服务器路径的数据报告。', flush=True)
        return
    from 检查演示数据 import inspect_dataset
    report, _, _ = inspect_dataset(ROOT/'训练数据', ROOT/'数据检查.json')
    if report['status'] != 'passed':
        raise RuntimeError('数据检查未通过，请查看数据检查.json')
    print('准备完成；尚未开始训练。', flush=True)

if __name__ == '__main__':
    main()
