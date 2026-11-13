"""核对全包SHA256、完整解码凭据、200段标注和采样边界，生成当前集群的检查报告。"""
from pathlib import Path
import json,hashlib,sys,subprocess
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'脚本/程序模块'),str(ROOT/'上游框架/src')]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    entries=json.loads((ROOT/'文件清单.json').read_text(encoding='utf-8'))
    for entry in entries:
        p=(ROOT/entry['path']).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file() or p.stat().st_size!=entry['size'] or digest(p)!=entry['sha256']:
            raise ValueError('文件清单不匹配：'+entry['path'])
    print(f'全部文件SHA256通过：{len(entries)}项',flush=True)
    receipt=json.loads((ROOT/'本地完整解码凭据.json').read_text(encoding='utf-8'))
    data_entries=[{k:e[k] for k in ('path','size','sha256')} for e in entries if e['path'].startswith('训练数据/')]
    # 旧凭据的字段顺序不影响sort_keys，文件顺序按旧清单沿用。
    fingerprint=hashlib.sha256(json.dumps(data_entries,ensure_ascii=False,sort_keys=True).encode('utf-8')).hexdigest()
    if fingerprint!=receipt['verified_data_manifest_sha256'] or receipt['report']['status']!='passed':
        raise ValueError('本批数据与完整逐帧解码凭据不一致')
    report=receipt['report']
    report.update(dataset_root=str((ROOT/'训练数据').resolve()),
                  verification_origin={'method':'全文件SHA256匹配已完成逐帧解码的数据；另验标注视频SHA256'})
    (ROOT/'数据检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    from 测试阶段采样 import main as test_sampling
    test_sampling()
    # 不自动训练；GPU实测及保存恢复验收由后续入口独立执行。
    print('数据/标注/采样准备完毕，尚未启动正式训练。',flush=True)
if __name__=='__main__':main()
