"""离线检查数据与抽样；不导入训练模型，不启动GPU更新。"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'脚本/程序模块'))
from 目标选择标注数据 import sha256_file,atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-decode',action='store_true',help='完整逐帧解码；首次准备必须执行')
    a=p.parse_args()
    if (ROOT/'文件清单.json').exists():
        manifest=json.loads((ROOT/'文件清单.json').read_text(encoding='utf-8'))
        for e in manifest:
            path=(ROOT/e['path']).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size!=e['size'] or sha256_file(path)!=e['sha256']:
                raise ValueError('文件SHA256不一致：'+e['path'])
        print('文件清单全部通过',flush=True)
    if a.full_decode:
        from 检查演示数据 import inspect_dataset
        report,_,_=inspect_dataset(ROOT/'训练数据',ROOT/'数据检查.json')
        if report['status']!='passed':raise ValueError(report['errors'])
    else:
        if not (ROOT/'数据检查.json').exists():
            raise ValueError('首次必须执行 python 准备训练.py --full-decode')
        report=json.loads((ROOT/'数据检查.json').read_text(encoding='utf-8'))
        if report.get('status')!='passed':raise ValueError('数据完整解码尚未通过')
        # 同内容训练包跨机器搬运后重写本机路径，不更改数据或统计。
        if not (ROOT/'文件清单.json').exists():raise ValueError('快速核验要求文件清单，或改用--full-decode')
        report['dataset_root']=str((ROOT/'训练数据').resolve())
        atomic_json(ROOT/'数据检查.json',report)
    from 测试阶段采样 import main as test_sampling
    test_sampling([])
    print('准备完成；未启动训练。先在AutoDL查看启动指令对比训练.py打印的计划。',flush=True)
if __name__=='__main__':main()
