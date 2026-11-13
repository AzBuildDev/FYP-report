"""实验性指令对比：同一观察/动作/噪声，只交换RC/BC；不是抓取奖励。"""
from contextlib import contextmanager
import numpy as np
import torch

VERSION = 'instruction_margin_v1'
TASKS = {
    0: 'Pick up the red cube and place it in the target area.',
    1: 'Pick up the blue cube and place it in the target area.',
}


def load_training_checkpoint(policy_class, config, checkpoint):
    """先准备FP32训练参数再加载，防止默认BF16专家层把保存的FP32权重先舍入。"""
    from pathlib import Path
    policy = policy_class(config)
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    policy = policy_class._load_as_safetensor(policy,str(Path(checkpoint)/'model.safetensors'),'cpu',True)
    policy.to(config.device)
    policy.eval()
    return policy


class ApproachPairs:
    """只从调用者传入的划分抽样；窗口包含合爪前标记，不使用标记之后的对比误差。

    标记代表正确示范的到达阶段，不等同于检测到物理接触。
    动作幅度筛选只剔除近乎静止窗口，不能证明两个指令在该窗口必然互斥。
    """
    def __init__(self, tables, annotations, action_std, chunk=50, arm_dims=5, min_motion=0.1):
        self.chunk, self.arm_dims = chunk, arm_dims
        self.tables = {int(t['episode_index'][0]): t for t in tables}
        if len(self.tables) != len(tables) or not tables:
            raise ValueError('对比池的段号为空或重复')
        std = np.asarray(action_std, dtype=np.float64)[:arm_dims]
        if len(std) != arm_dims or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError('机械臂动作归一化统计无效')
        self.groups, self.windows, self.excluded = {0: [], 1: []}, {}, []
        for ep, table in self.tables.items():
            row = annotations[ep]
            task = int(table['task_index'][0])
            if task not in TASKS or set(table['task_index']) != {task} or row['task'] != TASKS[task]:
                raise ValueError('对比池任务与示范不一致')
            action = np.asarray(table['action'], dtype=np.float64)
            marker = row['markers']['first_contact']
            if type(marker) is not int or not 0 < marker < len(action) or not np.isfinite(action).all():
                raise ValueError('标记或动作无效')
            candidates = []
            for start in range(max(0, marker-chunk+1), min(marker-1, len(action)-chunk)+1):
                motion = np.ptp(action[start:marker+1, :arm_dims]/std, axis=0).max()
                if motion >= min_motion:
                    candidates.append({'episode': ep, 'frame': start, 'task_index': task,
                                       'cutoff': marker-start, 'marker': marker})
            if candidates:
                self.groups[task].append(ep)
                self.windows[ep] = candidates
            else:
                self.excluded.append(ep)
        if not all(self.groups.values()):
            raise ValueError('对比窗口必须同时覆盖红蓝；禁止静默退化')

    def plan(self, step, count=8, seed=42):
        if type(count) is not int or count < 2 or count % 2 or step < 1:
            raise ValueError('对比数必须是正偶数，step从1开始')
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 1065]))
        rows = []
        for task in (0, 1):
            for _ in range(count//2):
                ep = self.groups[task][int(rng.integers(len(self.groups[task])))]
                choices = self.windows[ep]
                rows.append(dict(choices[int(rng.integers(len(choices)))]))
        return [rows[int(i)] for i in rng.permutation(len(rows))]

    def records(self, plan):
        return [(self.tables[r['episode']], r['frame']) for r in plan]

    def report(self):
        return {'eligible_episodes': {str(k): sorted(v) for k, v in self.groups.items()},
                'excluded_static_or_invalid_window_episodes': sorted(self.excluded),
                'candidate_windows': sum(len(v) for v in self.windows.values()),
                'scope': 'marker_reaching_approach_only', 'selection_success_measured': False}


def swap_tasks(tasks):
    mapping = {TASKS[0]: TASKS[1], TASKS[1]: TASKS[0]}
    if any(t not in mapping for t in tasks):
        raise ValueError('只能交换已核对的红蓝完整指令')
    return [mapping[t] for t in tasks]


def masked_errors(losses, cutoffs, arm_dims):
    if losses.ndim != 3 or not 0 < arm_dims <= losses.shape[2]:
        raise ValueError('模型逐元素流匹配误差形状无效')
    cuts = torch.as_tensor(cutoffs, device=losses.device, dtype=torch.long)
    if cuts.shape != (losses.shape[0],) or bool(((cuts < 1) | (cuts >= losses.shape[1])).any()):
        raise ValueError('标记必须位于动作窗口内，且至少有两帧监督')
    mask = torch.arange(losses.shape[1], device=losses.device)[None, :] <= cuts[:, None]
    # 仅真实机械臂关节：不使用padding维度，也不让合爪/标记后搬运支配对比项。
    return (losses[:, :, :arm_dims].float()*mask[:, :, None]).sum((1, 2))/(mask.sum(1)*arm_dims)


def margin_objective(correct, wrong, margin=0.02, ranking_weight=1.0):
    if not 0 < margin < 1 or correct.shape != wrong.shape or not 0 <= ranking_weight <= 1:
        raise ValueError('对比间隔或逐样本误差无效')
    # 有界误差避免通过无限推高错误指令误差降低损失；正指令仍有直接模仿锚点。
    positive_score = correct/(1+correct)
    negative_score = wrong/(1+wrong)
    gap = negative_score-positive_score
    ranking = torch.relu(margin-gap).mean()
    objective = correct.mean()+ranking_weight*ranking
    metrics = {'correct_flow_error': float(correct.detach().mean()),
               'wrong_flow_error': float(wrong.detach().mean()),
               'bounded_gap': float(gap.detach().mean()),
               'correct_instruction_lower_error_fraction': float((wrong.detach()>correct.detach()).float().mean()),
               'margin_satisfied_fraction': float((gap.detach()>=margin).float().mean()),
               'ranking_loss': float(ranking.detach()),
               'positive_anchor_loss': float(correct.detach().mean()),
               'selection_success_measured': False}
    return objective, metrics


@contextmanager
def deterministic_policy_mode(policy):
    was_training = policy.training
    policy.eval()  # 禁用dropout；eval不会禁用梯度。
    try:
        yield
    finally:
        policy.train(was_training)


def paired_instruction_loss(policy, batch, cutoffs, arm_dims=5, margin=0.02, noise=None, time=None, ranking_weight=1.0):
    if policy.config.adapt_to_pi_aloha:
        raise ValueError('该实验未验证ALOHA动作转换，不允许隐式修改输入')
    n = len(batch['task'])
    device = batch['action'].device
    shape = (n, policy.config.chunk_size, policy.config.max_action_dim)
    if noise is None:
        noise = policy.model.sample_noise(shape, device)
    if time is None:
        time = policy.model.sample_time(n, device)
    if noise.shape != shape or time.shape != (n,):
        raise ValueError('配对噪声或时间维度错误')
    wrong_batch = {**batch, 'task': swap_tasks(batch['task'])}
    with deterministic_policy_mode(policy):
        _, positive_info = policy(dict(batch), noise=noise, time=time)
        positive = masked_errors(positive_info['losses_after_forward'], cutoffs, arm_dims)
        del positive_info
        _, negative_info = policy(wrong_batch, noise=noise, time=time)
        negative = masked_errors(negative_info['losses_after_forward'], cutoffs, arm_dims)
        del negative_info
    return margin_objective(positive, negative, margin, ranking_weight)


def backward_instruction_term(policy, pool, step, count, seed, batch_factory, weight, margin, warmup, microbatch=2, ranking_weight=1.0):
    """在原模仿损失backward之后调用；小批次逐次累积，不保留整个基础batch的计算图。"""
    if weight <= 0:
        return {'effective_weight': 0.0, 'pair_count': 0}
    if microbatch < 1:
        raise ValueError('对比microbatch必须为正')
    plan = pool.plan(step, count, seed)
    effective = weight*min(1.0, step/max(1, warmup))
    aggregate = {}
    for offset in range(0, len(plan), microbatch):
        rows = plan[offset:offset+microbatch]
        batch = batch_factory(pool.records(rows))
        loss, metrics = paired_instruction_loss(policy, batch, [r['cutoff'] for r in rows], pool.arm_dims, margin, ranking_weight=ranking_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('指令对比损失不是有限数')
        scale = len(rows)/len(plan)
        (effective*scale*loss).backward()
        for k, v in metrics.items():
            if not isinstance(v, bool):aggregate[k] = aggregate.get(k, 0.0)+scale*v
        del batch, loss
    return {**aggregate, 'effective_weight': effective, 'pair_count': len(plan),
            'selection_success_measured': False}


def evaluate_instruction_pairs(policy, pool, batch_factory, margin=0.02, samples_per_episode=2, seeds=(42,43)):
    """固定留出段、固定窗口与噪声；报告指令相容性，不冒充选色/抓取成功率。"""
    values = {0: [], 1: []}
    device = next(policy.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        for task in (0, 1):
            for ep in sorted(pool.groups[task]):
                choices = pool.windows[ep]
                indexes = np.unique(np.linspace(0,len(choices)-1,samples_per_episode,dtype=int))
                episode_metrics = []
                for index in indexes:
                    row = choices[int(index)]
                    batch = batch_factory(pool.records([row]))
                    for seed in seeds:
                        torch.manual_seed(seed+ep*100000+row['frame'])
                        _, metrics = paired_instruction_loss(policy,batch,[row['cutoff']],pool.arm_dims,margin)
                        episode_metrics.append(metrics)
                    del batch
                values[task].append({k:float(np.mean([m[k] for m in episode_metrics]))
                                     for k in episode_metrics[0] if not isinstance(episode_metrics[0][k],bool)})
    by_task = {str(t):{k:float(np.mean([m[k] for m in ms])) for k in ms[0]} for t,ms in values.items()}
    return {'version':VERSION,'by_task':by_task,
            'task_balanced_gap':float(np.mean([v['bounded_gap'] for v in by_task.values()])),
            'task_balanced_preference':float(np.mean([v['correct_instruction_lower_error_fraction'] for v in by_task.values()])),
            'evaluated_episodes':sum(map(len,values.values())), 'seeds':list(seeds),
            'selection_success_measured':False}
