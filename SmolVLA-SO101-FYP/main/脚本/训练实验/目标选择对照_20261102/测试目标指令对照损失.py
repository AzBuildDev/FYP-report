"""CPU验证实验损失的梯度、掩码和共享随机条件；不加载模型或连接硬件。"""
from types import SimpleNamespace

import torch

from 目标指令对照损失 import TASKS, instruction_contrastive_objective, masked_ranking_loss


def main():
    correct = torch.tensor([[[2., 9.], [200., 9.]]], requires_grad=True)
    wrong = torch.tensor([[[1., 999.], [0., 999.]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    loss = masked_ranking_loss(correct, wrong, mask, action_dim=1, margin=0.05)
    assert torch.isclose(loss, torch.tensor(1.05))
    loss.backward()
    assert torch.equal(correct.grad, torch.tensor([[[1., 0.], [0., 0.]]]))
    assert torch.equal(wrong.grad, torch.tensor([[[-1., 0.], [0., 0.]]]))
    satisfied = masked_ranking_loss(torch.zeros(1, 2, 1), torch.ones(1, 2, 1), mask, action_dim=1, margin=0.05)
    assert satisfied.item() == 0
    empty = masked_ranking_loss(correct, wrong, ~torch.ones_like(mask), action_dim=1, margin=0.05)
    assert empty.item() == 0

    class FakeFlow:
        def sample_noise(self, shape, device):
            return torch.randn(shape, device=device)
        def sample_time(self, count, device):
            return torch.rand(count, device=device)

    class FakePolicy:
        config = SimpleNamespace(chunk_size=2, max_action_dim=2, adapt_to_pi_aloha=False)
        model = FakeFlow()
        def __init__(self):
            self.calls = []
            self.parameter = torch.tensor(0.5, requires_grad=True)
        def __call__(self, batch, noise=None, time=None):
            self.calls.append((batch, noise, time, torch.rand(())))
            factor = 1 if batch['task'][0] == TASKS['RC'] else 2
            errors = (self.parameter * factor).square().expand(1, 2, 2)
            return errors.mean(), {'losses_after_rm_padding': errors}

    policy = FakePolicy()
    batch = {'action': torch.zeros(1, 2, 1), 'task': [TASKS['RC']]}
    combined, metrics = instruction_contrastive_objective(policy, batch, mask)
    assert len(policy.calls) == 2
    assert policy.calls[0][1] is policy.calls[1][1]
    assert policy.calls[0][2] is policy.calls[1][2]
    assert torch.equal(policy.calls[0][3], policy.calls[1][3])
    assert policy.calls[1][0]['task'] == [TASKS['BC']]
    assert batch['task'] == [TASKS['RC']]
    assert metrics['decision_frames'] == 1
    combined.backward()
    assert torch.isfinite(policy.parameter.grad)
    baseline = FakePolicy()
    instruction_contrastive_objective(baseline, batch, mask, weight=0)
    assert len(baseline.calls) == 1
    print('通过：梯度方向、忽略补齐维、忽略未审核帧、间隔满足后停止惩罚、空掩码、共享噪声/时间/随机层、不修改标签、零权重退回基线。')


if __name__ == '__main__':
    main()
