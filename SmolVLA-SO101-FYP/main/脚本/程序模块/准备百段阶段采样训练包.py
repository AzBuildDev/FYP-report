"""准备独立AutoDL百段训练包；只复制/核验文件，不启动训练。"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from 目标选择标注数据 import DatasetCatalog, atomic_json, sha256_file

PROJECT = Path(__file__).resolve().parents[2]
TEMPLATE = PROJECT.parent / "AutoDL_新200段_阶段采样_20261103"
DATASET = PROJECT.parent / "训练数据采集/机械臂控制项目/任务演示/双相机演示_20261030_200119_173830"
ANNOTATION = PROJECT / "运行记录/目标选择时间标注/首次接触_50515e56d61e_训练导出.json"
DESTINATION = PROJECT.parent / "AutoDL_纯100段_阶段采样_20261107"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("训练模板已变化，拒绝猜测修改：" + old[:100])
    return source.replace(old, new, 1)


def trainer_source(template):
    source = template.read_text(encoding="utf-8")
    source = replace_once(source, "parser.add_argument('--seed', type=int, default=42)",
                          "parser.add_argument('--seed', type=int, default=42)\n    parser.add_argument('--sampling-mode', choices=('stage','full'), default='stage')")
    begin = source.index("    if len(tables) != 200")
    end = source.index("    train_tables, val_tables =", begin)
    source = source[:begin] + "    if len(tables) != annotation_report['episodes']:\n        raise ValueError('演示段数与标注不一致')\n" + source[end:]
    source = replace_once(source, "'sampling_mode': 'half_pre_marker_half_full_each_task_balanced',",
                          "'sampling_mode': args.sampling_mode,")
    source = replace_once(source, "annotations, args.batch_size, args.seed, cfg.chunk_size)",
                          "annotations, args.batch_size, args.seed, cfg.chunk_size, mode=args.sampling_mode)")
    source = replace_once(source, "'batch_size': args.batch_size, 'data_wait_s_avg50'",
                          "'sampling': sampler.statistics(step), 'batch_size': args.batch_size, 'data_wait_s_avg50'")
    source = replace_once(source, "if rss > 65536", "if rss > 32768")
    source = source.replace("'per_step_seed42'", "'per_step_seed_and_update'")
    source = replace_once(source, "'base_weights_sha256': file_sha256(ROOT / '模型/基础预训练模型/model.safetensors'),",
                          "'base_weights_sha256': file_sha256(ROOT / '模型/基础预训练模型/model.safetensors'),\n              'upstream_source_sha256': {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted((ROOT/'上游框架/src').rglob('*.py'))},\n              'runtime_versions': {'torch':torch.__version__, 'numpy':np.__version__},")
    source = source.replace("if step % args.save_every == 0 or step == args.steps:",
                            "if step % args.save_every == 0 or step == end:")
    # 不声明动作MAE可以识别选色；补充标记前输入的独立动作误差。
    source = replace_once(source, "def validate(policy, tables, videos, keys, tasks, stats, samples):",
                          "def validate(policy, tables, videos, keys, tasks, stats, samples, annotations):")
    source = replace_once(source, "records, combined, first, windows = [], [], [], []",
                          "records, combined, first, windows = [], [], [], []\n    pre_marker_errors = {0: [], 1: []}")
    source = replace_once(source, "combined.append(error)",
                          "combined.append(error)\n                        if int(i) < annotations[ep]['markers']['first_contact']:\n                            pre_marker_errors[int(table['task_index'][0])].append(float((error / std).mean()))")
    source = replace_once(source, "'first_action_normalized_mae':",
                          "'pre_marker_action_mae': {str(k):float(np.mean(v)) if v else None for k,v in pre_marker_errors.items()},\n            'color_selection_success_evaluated': False,\n            'first_action_normalized_mae':")
    source = replace_once(source, "validate(policy, val_tables, videos, keys, tasks, stats, args.validation_samples)",
                          "validate(policy, val_tables, videos, keys, tasks, stats, args.validation_samples, annotations)")
    return source


LAUNCHER = '''"""显式--run才训练；默认仅显示命令。A/B互用同一数据/预算但输出隔离。"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def command(config,arm,resume=False,stop_after=None):
    output=ROOT/config['output_relative']
    values=[sys.executable,'-u',str(ROOT/'脚本/程序模块/百段阶段训练.py'),
            '--dataset-root',str(ROOT/'训练数据'),'--data-check',str(ROOT/'数据检查.json'),
            '--annotations',str(ROOT/'标注/阶段标注_100段.json'),'--output-dir',str(output),
            '--sampling-mode',config['sampling_mode']]
    for key in ('steps','batch_size','warmup','save_every','validate_every','validation_samples','seed'):
        values+=['--'+key.replace('_','-'),str(config[key])]
    if resume:values+=['--resume',str(output/'last')]
    if stop_after is not None:values+=['--stop-after',str(stop_after)]
    return values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('A','B'),default='B')
    p.add_argument('--run',action='store_true',help='实际启动训练；未指定则只打印计划')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int,help='保存后停止，供另行授权的GPU保存恢复验证')
    a=p.parse_args()
    config=json.loads((ROOT/f'训练参数_{a.arm}.json').read_text(encoding='utf-8'))
    values=command(config,a.arm,a.resume,a.stop_after)
    print(json.dumps({'arm':a.arm,'config':config,'command':values,'training_started':bool(a.run)},ensure_ascii=False),flush=True)
    if not a.run:return
    if not (ROOT/'数据检查.json').is_file():p.error('先执行准备训练.py完成数据与采样核验')
    subprocess.run([sys.executable,str(ROOT/'准备训练.py')],cwd=ROOT,check=True)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
        PYTHONPATH=str(ROOT/'上游框架/src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='131072')
    subprocess.run(values,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
'''

PREPARE = '''"""离线检查数据与抽样；不导入训练模型，不启动GPU更新。"""
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
    print('准备完成；未启动训练。先在AutoDL查看启动训练.py打印的计划。',flush=True)
if __name__=='__main__':main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("输出已存在，保留现场；不要覆盖旧训练包")
    if not TEMPLATE.is_dir():
        raise ValueError("未找到已验收的阶段采样训练模板")
    catalog = DatasetCatalog(DATASET)
    if len(catalog.ids) != 100:
        raise ValueError("本入口仅准备这次100段")
    destination.mkdir(parents=True)
    for folder in ("脚本/程序模块", "标注", "日志"):
        (destination / folder).mkdir(parents=True)
    modules = destination / "脚本/程序模块"
    for name in ("微调预训练模型.py", "管理训练数据来源.py", "检查演示数据.py"):
        shutil.copy2(TEMPLATE / "脚本/程序模块" / name, modules / name)
    for name in ("阶段采样.py", "目标选择标注数据.py", "测试阶段采样.py"):
        shutil.copy2(Path(__file__).parent / name, modules / name)
    (modules / "百段阶段训练.py").write_text(trainer_source(TEMPLATE / "脚本/程序模块/新200段训练.py"), encoding="utf-8")
    shutil.copy2(ANNOTATION, destination / "标注/阶段标注_100段.json")
    import numpy as np
    rng = np.random.default_rng(42)
    val = []
    for target in ("RC", "BC"):
        ids = [ep for ep in catalog.ids if catalog.episodes[ep]["target"] == target]
        if len(ids) != 50:
            raise ValueError("本批应为红蓝各50段")
        for offset in range(0, len(ids), 10):
            val.extend(int(ep) for ep in rng.choice(ids[offset:offset + 10], 2, replace=False))
    split = {"seed": 42, "train": [ep for ep in catalog.ids if ep not in val], "validation": sorted(val),
             "method": "按颜色及每色录制顺序十分段分层，每组10段留2段；整段80/20划分；布局配对未核对，不声称布局独立验证"}
    atomic_json(destination / "划分.json", split)
    for arm, mode in (("A", "full"), ("B", "stage")):
        atomic_json(destination / f"训练参数_{arm}.json", {"batch_size": 32, "steps": 30000,
                    "warmup": 500, "save_every": 5000, "validate_every": 1000, "validation_samples": 10,
                    "seed": 42, "sampling_mode": mode, "output_relative": f"输出/{arm}_纯100段_{mode}",
                    "initialization": "official_base_fresh_optimizer", "continuous": False})
    atomic_json(destination / "实验说明.json", {"experiment": "纯100段阶段采样_20261107",
                "dataset_root_at_preparation": str(DATASET), "dataset_fingerprint": catalog.fingerprint,
                "annotation_source_sha256": sha256_file(ANNOTATION),
                "marker_storage": "原导出的first_contact字段及原帧号，不重新解释或改写标记",
                "sampling": "B每批32：标记前RC8/BC8、全段RC8/BC8；A全段RC16/BC16",
                "actual_pre_marker_fraction": "全段池也可抽到标记前，实际标记前输入比例可高于50%",
                "action_supervision": "同帧双摄/关节/指令及未来完整50步真实动作，可跨标记；不增加错误指令损失",
                "initialization": "官方基础模型、新优化器，从0计数；A/B各自独立；未继承旧检查点",
                "validation": "40红40蓝训练，10红10蓝验证；动作MAE不是选色成功率，后续实机单独评价",
                "budget": "A/B均候选batch32×30K；复用前次参考，不声称最优或保证改善",
                "gpu_training_acceptance": "pending; no optimizer updates authorized during preparation",
                "training_started": False})
    (destination / "启动训练.py").write_text(LAUNCHER, encoding="utf-8")
    (destination / "准备训练.py").write_text(PREPARE, encoding="utf-8")
    shutil.copy2(TEMPLATE / "安装依赖.sh", destination / "安装依赖.sh")
    for name in ("meta", "data", "videos"):
        shutil.copytree(DATASET / name, destination / "训练数据" / name)
        print("已复制数据：" + name, flush=True)
    for name in ("模型", "上游框架"):
        shutil.copytree(TEMPLATE / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        print("已复制：" + name, flush=True)
    print("训练包已生成：" + str(destination) + "；尚需准备训练.py --full-decode验收；未启动训练。", flush=True)


if __name__ == "__main__":
    main()
