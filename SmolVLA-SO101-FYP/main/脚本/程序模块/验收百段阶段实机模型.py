"""校验训练来源并做录制画面离线启动测试，通过后预选模型；不连接实机。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from 管理训练数据来源 import (dataset_source, load_training_source, select_tables,
    training_stats, verify_checkpoint_stats, file_sha256)
from 目标选择标注数据 import DatasetCatalog
from 选择推理模型 import validate_selection, remember_selection, discover_models
from 红蓝抓取任务 import TASKS

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT.parent / 'AutoDL_纯100段_阶段采样_20261107/训练数据'
PROCESSOR = ROOT / '模型/视觉语言处理器'
REPORTS = ROOT / '运行记录/AutoDL纯100段部署_20261107'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, default=3435)
    parser.add_argument('--prepare-selection', action='store_true')
    args = parser.parse_args()
    name = f'step_{args.step:06d}'
    model = ROOT / '模型/纯100段_阶段采样_20261107' / name
    selection = validate_selection(model, DATASET, PROCESSOR, ROOT)
    assert model.resolve() in discover_models(ROOT)
    summary = json.loads((model / 'checkpoint_summary.json').read_text(encoding='utf-8'))
    config = summary['run_config']
    assert summary['step'] == args.step and config['batch_size'] == 140
    catalog = DatasetCatalog(DATASET)
    assert catalog.fingerprint == config['stage_sampling']['dataset_fingerprint']
    info = json.loads((DATASET / 'meta/info.json').read_text(encoding='utf-8'))
    tasks = [json.loads(line) for line in (DATASET / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    tables = [pq.read_table(path).to_pydict() for path in sorted(DATASET.glob('data/chunk-*/*.parquet'))]
    train = select_tables(tables, config['train_episodes'])
    saved_source = load_training_source(model)
    local_source = dataset_source(DATASET, info, train, tasks)
    assert len(saved_source['episodes']) == len(train) == 80
    for key in ['fps', 'joint_names', 'state_names', 'total_frames']:
        assert local_source[key] == saved_source[key], key
    for actual, expected in zip(local_source['episodes'], saved_source['episodes']):
        for key in ['episode', 'frames', 'action_state_sha256', 'video_sha256']:
            assert actual[key] == expected[key], (actual['episode'], key)
    expected_stats = training_stats(train)
    saved_stats = json.loads((model / 'dataset_stats.json').read_text(encoding='utf-8'))
    for key, values in expected_stats.items():
        for statistic, expected in values.items():
            assert np.allclose(expected, saved_stats[key][statistic], rtol=1e-5, atol=1e-6)
    verify_checkpoint_stats(model, expected_stats)
    assert set(TASKS) == {'RC', 'BC'}
    print('80段训练来源、动作/视频指纹、归一化及模型选择检查通过', flush=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / '上游框架/src'), PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    subprocess.run([sys.executable, str(ROOT / '脚本/程序模块/模型启动离线验证.py'),
        '--task-inference', '--model-dir', str(model), '--dataset-root', str(DATASET),
        '--processor-dir', str(PROCESSOR)], cwd=ROOT, env=env, check=True, timeout=240)
    pointer = json.loads((ROOT / '运行记录/模型启动离线验证最新结果.json').read_text(encoding='utf-8'))
    report_path = Path(pointer['report'])
    offline = json.loads(report_path.read_text(encoding='utf-8'))
    assert pointer['status'] == offline['status'] == 'passed'
    assert file_sha256(report_path) == pointer['report_sha256']
    assert Path(offline['model']).resolve() == model.resolve()
    assert offline['robot_commands_sent'] == 0 and offline['hardware_tested'] is False
    previous = ROOT / '运行环境/私密配置/推理选择.json'
    if args.prepare_selection:
        if previous.exists():
            backup = REPORTS / f'{name}_预选前推理选择.json'
            if not backup.exists():
                backup.write_bytes(previous.read_bytes())
        remember_selection(selection, ROOT)
    receipt = {'status':'passed', 'checked_at':datetime.now().isoformat(), 'model':str(model),
        'step':args.step, 'batch_size':140, 'sample_windows':args.step * 140,
        'equivalent_batch16_steps':args.step * 140 / 16, 'source_episodes_verified':80,
        'dataset':str(DATASET), 'processor':str(PROCESSOR), 'dataset_and_video_fingerprints_verified':True,
        'normalization_verified':True, 'offline_report':str(report_path),
        'preselected':args.prepare_selection, 'available_targets':TASKS,
        'real_cameras_opened':False, 'robot_commands_sent':0, 'task_success_evaluated':False,
        'entry':str(ROOT / '脚本/日常操作/启动模型推理.bat')}
    (REPORTS / f'{name}_实机准备验收.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    download_receipt = REPORTS / f'{name}_下载验收.json'
    if download_receipt.exists():
        record = json.loads(download_receipt.read_text(encoding='utf-8'))
        record.update(offline_startup_pending=False, readiness_report=str(REPORTS / f'{name}_实机准备验收.json'))
        download_receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
