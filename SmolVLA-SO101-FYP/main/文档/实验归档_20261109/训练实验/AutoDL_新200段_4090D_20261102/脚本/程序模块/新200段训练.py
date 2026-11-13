"""双摄 SmolVLA 分阶段训练、固定留出验证及完整检查点恢复；不连接机械臂。"""
import argparse
import json
import math
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

from 微调预训练模型 import (
    ROOT, np, torch, PreTrainedConfig, FeatureType, PolicyFeature, SmolVLAPolicy,
    load_smolvla, VideoFrameCache, preload_frames,
)
import pyarrow.parquet as pq
from 管理训练数据来源 import (
    dataset_source, file_sha256, select_tables, training_stats, verify_checkpoint_stats, assess_sources,
)


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def fixed_split(total, seed):
    if total != 50:
        raise ValueError('本实验要求已按位置排序的 50 段演示，每组 10 段')
    rng = np.random.default_rng(seed)
    validation = sorted(int(i) for start in range(0, 50, 10)
                        for i in rng.choice(np.arange(start, start + 10), 2, replace=False))
    return [i for i in range(total) if i not in validation], validation


def read_data(root, check):
    report = json.loads(check.read_text(encoding='utf-8'))
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    if report['status'] != 'passed' or Path(report['dataset_root']).resolve() != root.resolve():
        raise ValueError('数据检查报告无效或对应其他数据目录')
    # 已完成逐帧视频检查；文件有变化则要求先重新检查。
    if any(p.stat().st_mtime_ns > check.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()):
        raise ValueError('数据文件在完整检查后发生变化，请重新检查')
    tables = [pq.read_table(p).to_pydict() for p in sorted(root.glob('data/chunk-*/*.parquet'))]
    if (len(tables) != info['total_episodes'] or len(tables) != report['total_episodes']
            or sum(len(t['action']) for t in tables) != report['total_frames']):
        raise ValueError('数据总数与完整检查报告不符')
    tasks = [json.loads(line) for line in (root / 'meta/tasks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return info, tables, tasks


def make_batch(records, videos, keys, tasks, chunk):
    states, actions, texts = [], [], []
    images = {k: [] for k in keys}
    for table, i in records:
        ep = int(table['episode_index'][0])
        states.append(table['observation.state'][i])
        actions.append(table['action'][i:i + chunk])
        texts.append(tasks[table['task_index'][i]])
        for key in keys:
            images[key].append(videos[key][ep][i])
    batch = {'observation.state': torch.tensor(np.asarray(states), dtype=torch.float32, device='cuda'),
             'action': torch.tensor(np.asarray(actions), dtype=torch.float32, device='cuda'), 'task': texts}
    batch.update({k: torch.from_numpy(np.stack(v)).permute(0, 3, 1, 2).to('cuda', torch.float32) / 255
                  for k, v in images.items()})
    return batch


def learning_rate_factor(index, warmup, total, minimum=0.025):
    if index < warmup:
        return (index + 1) / warmup
    progress = min(1.0, (index - warmup) / max(1, total - warmup - 1))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def validate(policy, tables, videos, keys, tasks, stats, samples):
    records, combined, first, windows = [], [], [], []
    std = np.asarray(stats['action']['std'], dtype=np.float32)
    policy.eval()
    try:
        # 验证不能改变随后训练所用的随机数状态。
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
            for table in tables:
                ep = int(table['episode_index'][0])
                indices = np.unique(np.linspace(0, len(table['action']) - policy.config.chunk_size, samples, dtype=int))
                errors = []
                for i in indices:
                    batch = make_batch([(table, int(i))], videos, keys, tasks, policy.config.chunk_size)
                    expected = np.asarray(table['action'][i:i + policy.config.chunk_size], dtype=np.float32)
                    for seed in (42, 43):
                        torch.manual_seed(seed + ep * 100000 + int(i))
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            predicted = policy.predict_action_chunk(batch)[0].float().cpu().numpy()
                        if predicted.shape != expected.shape or not np.isfinite(predicted).all():
                            raise ValueError('验证预测动作无效')
                        error = np.abs(predicted - expected)
                        combined.append(error)
                        first.append(error[:1])
                        windows.append(error[:20])
                        errors.append(float((error / std).mean()))
                records.append({'episode': ep, 'task_index': int(table['task_index'][0]),
                                'normalized_mae': float(np.mean(errors)), 'predictions': len(errors)})
    finally:
        policy.train()
    all_errors = np.concatenate(combined)
    per_task = {str(t):float(np.mean([r['normalized_mae'] for r in records if r['task_index']==t])) for t in (0,1)}
    return {'task_balanced_mae':float(np.mean(list(per_task.values()))), 'task_metrics':per_task, 'normalized_mae': float((all_errors / std).mean()),
            'first_action_normalized_mae': float((np.concatenate(first) / std).mean()),
            'first20_normalized_mae': float((np.concatenate(windows) / std).mean()),
            'mae_per_joint': all_errors.mean(axis=0).tolist(), 'episode_metrics': records,
            'predictions': len(combined), 'seeds': [42, 43], 'task_success_evaluated': False}


def save_checkpoint(target, policy, optimizer, scheduler, rng, step, best, config, source, stats, validation):
    target = Path(target)
    staging, previous = target.with_name(target.name + '.writing'), target.with_name(target.name + '.previous')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    policy.save_pretrained(staging)
    serial_stats = {k: {n: v.tolist() for n, v in data.items()} for k, data in stats.items()}
    write_json(staging / 'dataset_stats.json', serial_stats)
    verify_checkpoint_stats(staging, serial_stats)
    provenance = {'schema_version': 1, 'status': 'completed', 'training_data': source,
                  'checkpoint_sha256': file_sha256(staging / 'model.safetensors'),
                  'steps': step, 'batch_size': config['batch_size'], 'seed': config['seed'],
                  'created_at': datetime.now().isoformat(timespec='seconds')}
    write_json(staging / 'training_provenance.json', provenance)
    write_json(staging / 'checkpoint_summary.json', {'step': step, 'best_score': best,
                'validation': validation, 'run_config': config, 'metric': 'task_balanced_mae'})
    torch.save({'step': step, 'best_score': best, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'numpy_rng': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'python_rng': random.getstate(), 'run_config': config}, staging / 'training_state.pt')
    if previous.exists():
        shutil.rmtree(previous)
    if target.exists():
        target.rename(previous)
    staging.rename(target)
    if previous.exists():
        shutil.rmtree(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--data-check', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--validate-every', type=int, default=1000)
    parser.add_argument('--validation-samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, help='仅用于验证保存/恢复，达到指定更新步后正常退出')
    args = parser.parse_args()
    if args.batch_size != 16 or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('本入口要求单卡支持BF16，batch16')
    if min(args.steps, args.batch_size, args.save_every, args.validate_every, args.validation_samples) < 1 or not 0 < args.warmup < args.steps:
        parser.error('训练参数必须为正，预热小于总步数')
    if not args.resume and args.output_dir.exists():
        parser.error('输出目录已存在，请使用新目录或明确 --resume')
    info, tables, tasks_list = read_data(args.dataset_root, args.data_check)
    split = json.loads((ROOT/'划分.json').read_text(encoding='utf-8'))
    train_ids, val_ids = split['train'], split['validation']
    if len(tables) != 200 or len(train_ids) != 160 or len(val_ids) != 40 or set(train_ids) & set(val_ids):
        raise ValueError('新200段划分无效')
    for ids, expected in [(train_ids,80),(val_ids,20)]:
        for task_id in (0,1):
            if sum(int(tables[i]['task_index'][0])==task_id for i in ids)!=expected:
                raise ValueError('红蓝任务数不均衡')
    train_tables, val_tables = select_tables(tables, train_ids), select_tables(tables, val_ids)
    train_source = dataset_source(args.dataset_root, info, train_tables, tasks_list)
    val_source = dataset_source(args.dataset_root, info, val_tables, tasks_list)
    if not assess_sources(train_source, val_source)['held_out_validation']:
        raise ValueError('训练和验证存在内容重复')
    stats = training_stats(train_tables)
    config = {'dataset_root': str(args.dataset_root.resolve()), 'steps': args.steps,
              'batch_size': args.batch_size, 'gradient_accumulation': 1, 'warmup': args.warmup,
              'save_every': args.save_every, 'validate_every': args.validate_every,
              'validation_samples_per_episode': args.validation_samples, 'seed': args.seed,
              'train_episodes': train_ids, 'validation_episodes': val_ids,
              'early_stopping': False, 'lr_peak': 1e-4, 'lr_min': 2.5e-6,
              'augmentation': 'none', 'sampling_mode': 'red8_blue8', 'trainer_sha256': file_sha256(Path(__file__)),
              'split_sha256': file_sha256(ROOT/'划分.json'),
              'source_dependencies': {p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
              'base_weights_sha256': file_sha256(ROOT / '模型/基础预训练模型/model.safetensors'),
              'training_data': train_source, 'validation_data': val_source}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        write_json(args.output_dir / 'run_config.json', config)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.resume:
        summary = json.loads((args.resume / 'checkpoint_summary.json').read_text(encoding='utf-8'))
        if summary['run_config'] != config:
            raise ValueError('恢复时参数、源码或数据指纹变化，拒绝混用')
        verify_checkpoint_stats(args.resume, json.loads((args.resume / 'dataset_stats.json').read_text(encoding='utf-8')))
        provenance = json.loads((args.resume / 'training_provenance.json').read_text(encoding='utf-8'))
        if file_sha256(args.resume / 'model.safetensors') != provenance['checkpoint_sha256']:
            raise ValueError('恢复权重指纹不一致')
        resume_cfg = PreTrainedConfig.from_pretrained(args.resume, local_files_only=True)
        resume_cfg.vlm_model_name = str(ROOT/'模型/视觉语言处理器')
        resume_cfg.load_vlm_weights = False
        policy = SmolVLAPolicy.from_pretrained(args.resume, config=resume_cfg, local_files_only=True)
    else:
        cfg = PreTrainedConfig.from_pretrained(ROOT / '模型/基础预训练模型', local_files_only=True)
        keys = [k for k, v in info['features'].items() if v['dtype'] == 'video']
        if keys != ['observation.images.front', 'observation.images.wrist']:
            raise ValueError('需要前置、夹爪双摄顺序')
        cfg.input_features = {**{k: v for k, v in cfg.input_features.items() if v.type is not FeatureType.VISUAL},
                              **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *info['features'][k]['shape'][:2])) for k in keys}}
        cfg.device, cfg.push_to_hub = 'cuda', False
        cfg.vlm_model_name, cfg.load_vlm_weights = str(ROOT / '模型/视觉语言处理器'), False
        policy = SmolVLAPolicy(cfg, dataset_stats={k:{n:torch.from_numpy(v.copy()) for n,v in values.items()} for k,values in stats.items()})
        load_smolvla(policy, ROOT / '模型/基础预训练模型/model.safetensors',
                     device='cpu', checkpoint_keys_mapping='model._orig_mod.//model.')
    policy.to('cuda').train()
    params = [p for p in policy.parameters() if p.requires_grad]
    cfg = policy.config
    for parameter in params:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=cfg.optimizer_betas,
                                 eps=cfg.optimizer_eps, weight_decay=cfg.optimizer_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: learning_rate_factor(i, args.warmup, args.steps))
    start_step, best = 0, float('inf')
    if args.resume:
        # 文件由本程序在本机生成；包含优化器及 Python/NumPy 随机状态。
        saved = torch.load(args.resume / 'training_state.pt', map_location='cpu', weights_only=False)
        if saved['run_config'] != config:
            raise ValueError('优化器检查点配置不一致')
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        start_step, best = saved['step'], saved['best_score']
    keys, tasks = list(policy.config.image_features), {t['task_index']: t['task'] for t in tasks_list}
    cache = VideoFrameCache(max_bytes=2*1024**3)
    videos = {k: preload_frames(args.dataset_root, info, tables, k, cache) for k in keys}
    log = (args.output_dir / 'metrics.jsonl').open('a', encoding='utf-8', buffering=1)
    def emit(record):
        record['time'] = datetime.now().isoformat(timespec='seconds')
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        log.write(line + '\n')
        print(line, flush=True)
    emit({'event': 'start', 'step': start_step, 'train_episodes': train_ids, 'validation_episodes': val_ids,
          'gpu': torch.cuda.get_device_name(0), 'trainable_parameters': sum(p.numel() for p in params)})
    recent, start_time = [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    end = min(args.steps, args.stop_after or args.steps)
    if end <= start_step:
        raise ValueError('停止步数必须大于已恢复步数')
    for step in range(start_step + 1, end + 1):
        samples = []
        for task_id in (0,1):
            group = [t for t in train_tables if int(t['task_index'][0])==task_id]
            for _ in range(8):
                table = group[int(rng.integers(len(group)))]
                i = int(rng.integers(len(table['action']) - cfg.chunk_size + 1))
                samples.append((table, i))
        samples = [samples[int(i)] for i in rng.permutation(16)]
        batch = make_batch(samples, videos, keys, tasks, cfg.chunk_size)
        optimizer.zero_grad(set_to_none=True)
        lr_used = optimizer.param_groups[0]['lr']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError('训练损失不是有限数值')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        recent.append(float(loss.detach()))
        del batch, loss
        if step % 50 == 0 or step == start_step + 1:
            elapsed = time.perf_counter() - start_time
            emit({'event': 'train', 'step': step, 'loss_avg50': float(np.mean(recent[-50:])),
                  'lr': lr_used, 'grad_norm': float(grad_norm), 'elapsed_s': elapsed,
                  'eta_s': elapsed / (step - start_step) * (args.steps - step),
                  'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2})
            recent = recent[-50:]
        pause = (args.output_dir/'PAUSE').exists()
        if step % args.validate_every == 0 or step % args.save_every == 0 or step == end or pause:
            validation = validate(policy, val_tables, videos, keys, tasks, stats, args.validation_samples)
            improved = validation['task_balanced_mae'] < best
            best = min(best, validation['task_balanced_mae'])
            emit({'event': 'validation', 'step': step, 'best_score': best, 'improved': improved, **validation})
            save_checkpoint(args.output_dir / 'last', policy, optimizer, scheduler, rng, step, best, config, train_source, stats, validation)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(args.output_dir / f'step_{step:06d}', policy, optimizer, scheduler, rng, step, best, config, train_source, stats, validation)
            if improved:
                save_checkpoint(args.output_dir / 'best_val', policy, optimizer, scheduler, rng, step, best, config, train_source, stats, validation)
            write_json(args.output_dir / 'status.json', {'state': 'completed' if step == args.steps else ('paused' if pause else 'running'),
                       'step': step, 'total_steps': args.steps, 'best_score': best,
                       'last_checkpoint': str(args.output_dir / 'last'), 'updated_at': datetime.now().isoformat(timespec='seconds')})
            emit({'event': 'checkpoint_saved', 'step': step})
            if pause:
                log.close()
                return
    emit({'event': 'completed' if end == args.steps else 'stopped_for_resume_test', 'step': end})
    log.close()


if __name__ == '__main__':
    main()
