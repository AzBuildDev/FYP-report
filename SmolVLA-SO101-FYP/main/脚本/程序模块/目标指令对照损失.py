"""实验用条件流匹配对照损失；未接入现有训练，不连接硬件。

只在人工确认不同目标应产生不同动作的帧上使用。归位、共同准备动作和
已放好后的帧不能自动当作负例。此方法是本项目的实验假设，非官方配方。
"""
import math

import torch


TASKS = {
    "RC": "Pick up the red cube and place it in the target area.",
    "BC": "Pick up the blue cube and place it in the target area.",
}


def masked_ranking_loss(correct_errors, wrong_errors, decision_mask, *, action_dim, margin):
    """对已审核帧计算 max(0, margin + 正确指令误差 - 错误指令误差)。

输入是条件流匹配的逐维平方误差，非预测抓取成功率。只比较真实电机维，
不让补齐到32维的通道满足目标约束。两个分支均保留梯度。
"""
    if correct_errors.shape != wrong_errors.shape or correct_errors.ndim != 3:
        raise ValueError("两个分支的损失必须都是相同的B×T×D张量")
    if decision_mask.shape != correct_errors.shape[:2] or decision_mask.dtype != torch.bool:
        raise ValueError("决策掩码必须是与动作对应的B×T布尔张量")
    if not 0 < action_dim <= correct_errors.shape[-1]:
        raise ValueError("真实动作维数无效")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin必须为有限正数")
    if not torch.isfinite(correct_errors).all() or not torch.isfinite(wrong_errors).all():
        raise ValueError("对照损失含非有限数值")
    correct = correct_errors[..., :action_dim].float().mean(-1)
    wrong = wrong_errors[..., :action_dim].float().mean(-1)
    if not decision_mask.any():
        return (correct.sum() + wrong.sum()) * 0
    return torch.relu(margin + correct - wrong)[decision_mask].mean()


def instruction_contrastive_objective(policy, batch, decision_mask, *, weight=0.1, margin=0.05):
    """可替换训练循环中的 loss, metrics = policy(batch)。

policy须为现有SmolVLAPolicy，batch保留真实动作与规范英文task。
decision_mask由人工核对的动作阶段生成；该函数不推断阶段或配对关系。
weight和margin是待消融的实验起点，不是已有论文验证的最佳值。
"""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight必须是有限非负数")
    actions = batch["action"]
    if actions.ndim != 3 or actions.shape[1] != policy.config.chunk_size:
        raise ValueError("实验入口要求完整动作窗口")
    if decision_mask.shape != actions.shape[:2] or decision_mask.dtype != torch.bool:
        raise ValueError("必须提供与本批动作帧对应的审核掩码")
    decision_mask = decision_mask.to(actions.device).clone()
    if "actions_id_pad" in batch:
        decision_mask &= ~batch["actions_id_pad"].to(actions.device)
    if weight == 0 or not decision_mask.any():
        loss, metrics = policy(batch)
        return loss, {**metrics, "contrastive_loss": 0.0, "decision_frames": int(decision_mask.sum())}
    mapping = {TASKS["RC"]: TASKS["BC"], TASKS["BC"]: TASKS["RC"]}
    if len(batch["task"]) != actions.shape[0] or any(t not in mapping for t in batch["task"]):
        raise ValueError("RC/BC任务文字不规范，拒绝生成错配条件")
    if policy.config.adapt_to_pi_aloha:
        raise ValueError("该实验未验证Aloha原地动作转换")
    noise = policy.model.sample_noise(
        (actions.shape[0], actions.shape[1], policy.config.max_action_dim), actions.device)
    flow_time = policy.model.sample_time(actions.shape[0], actions.device)
    # 两次前向保持相同的噪声、流匹配时间和随机层状态，只有任务文字不同。
    # 保存/恢复随机状态不会将负例分支的随机消耗带入下一次训练采样。
    cuda_devices = [actions.device.index if actions.device.index is not None else torch.cuda.current_device()] if actions.is_cuda else []
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(actions.device) if actions.is_cuda else None
    correct_loss, correct_metrics = policy(dict(batch), noise=noise, time=flow_time)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.set_rng_state(cpu_rng)
        if actions.is_cuda:
            torch.cuda.set_rng_state(cuda_rng, actions.device)
        wrong_batch = {**batch, "task": [mapping[t] for t in batch["task"]]}
        _, wrong_metrics = policy(wrong_batch, noise=noise, time=flow_time)
    ranking = masked_ranking_loss(
        correct_metrics["losses_after_rm_padding"], wrong_metrics["losses_after_rm_padding"],
        decision_mask, action_dim=actions.shape[-1], margin=margin)
    total = correct_loss + weight * ranking
    return total, {
        "loss": float(total.detach()), "flow_loss": float(correct_loss.detach()),
        "contrastive_loss": float(ranking.detach()), "contrastive_weight": weight,
        "contrastive_margin": margin, "decision_frames": int(decision_mask.sum()),
        "physical_success_evaluated": False,
    }
