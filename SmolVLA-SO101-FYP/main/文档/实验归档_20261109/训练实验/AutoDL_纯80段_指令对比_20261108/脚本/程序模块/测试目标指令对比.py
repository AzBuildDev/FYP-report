"""指令对比的梯度、噪声配对、标记掩码和划分隔离检查；默认仅CPU。"""
import unittest
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from 目标指令对比 import ApproachPairs,TASKS,masked_errors,margin_objective,paired_instruction_loss,load_training_checkpoint


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(0.2))
        self.config = SimpleNamespace(adapt_to_pi_aloha=False,chunk_size=4,max_action_dim=8)
        self.calls=[]
    def forward(self,batch,noise=None,time=None):
        self.calls.append((dict(batch),noise.clone(),time.clone(),self.training))
        sign=torch.tensor([1 if t==TASKS[0] else -1 for t in batch['task']])[:,None,None]
        losses=(self.parameter*sign-torch.ones((len(sign),4,8)))**2
        return losses.mean(),{'losses_after_forward':losses}


class Tests(unittest.TestCase):
    def test_fp32_resume_does_not_round_through_bfloat16(self):
        from pathlib import Path
        from safetensors.torch import save_file,load_model
        class TinyPolicy(torch.nn.Module):
            def __init__(self,config):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(2,dtype=torch.bfloat16))
            @classmethod
            def _load_as_safetensor(cls,model,path,device,strict):
                load_model(model,path,device=device,strict=strict);return model
        weights=torch.tensor([0.1234567,0.9876543],dtype=torch.float32)
        with tempfile.TemporaryDirectory() as d:
            save_file({'weight':weights},str(Path(d)/'model.safetensors'))
            model=load_training_checkpoint(TinyPolicy,SimpleNamespace(device='cpu'),d)
            self.assertTrue(torch.equal(model.weight,weights))
            self.assertFalse(torch.equal(weights.to(torch.bfloat16).float(),weights))
    def test_mask_excludes_postmarker_padding_and_gripper(self):
        x=torch.ones(1,5,32,requires_grad=True)
        result=masked_errors(x,[2],5)
        self.assertEqual(result.item(),1)
        result.sum().backward()
        self.assertEqual(x.grad[:,3:,:].count_nonzero(),0)
        self.assertEqual(x.grad[:,:,5:].count_nonzero(),0)
        self.assertEqual(x.grad.count_nonzero(),15)

    def test_margin_gradient_and_saturation(self):
        good=torch.tensor([.3],requires_grad=True);bad=torch.tensor([.1],requires_grad=True)
        loss,_=margin_objective(good,bad);loss.backward()
        self.assertGreater(good.grad.item(),0)
        self.assertLess(bad.grad.item(),0)
        good=torch.tensor([.01],requires_grad=True);bad=torch.tensor([10.],requires_grad=True)
        loss,m=margin_objective(good,bad);loss.backward()
        self.assertEqual(m['ranking_loss'],0)
        self.assertEqual(bad.grad.item(),0)

    def test_exact_counterfactual_inputs_and_mode_restore(self):
        p=FakePolicy().train()
        batch={'task':[TASKS[0]],'action':torch.ones(1,4,6),
               'observation.state':torch.zeros(1,6),'observation.images.front':torch.zeros(1,3,8,8)}
        original_task=list(batch['task'])
        noise=torch.randn(1,4,8);time=torch.tensor([.6])
        loss,_=paired_instruction_loss(p,batch,[2],noise=noise,time=time)
        loss.backward()
        a,b=p.calls
        self.assertTrue(p.training)
        self.assertFalse(a[3]);self.assertFalse(b[3])
        self.assertTrue(torch.equal(a[1],b[1]));self.assertTrue(torch.equal(a[2],b[2]))
        self.assertEqual(a[0]['task'],[TASKS[0]]);self.assertEqual(b[0]['task'],[TASKS[1]])
        self.assertIs(a[0]['action'],b[0]['action'])
        self.assertIs(a[0]['observation.state'],b[0]['observation.state'])
        self.assertIs(a[0]['observation.images.front'],b[0]['observation.images.front'])
        self.assertEqual(batch['task'],original_task)
        self.assertTrue(torch.isfinite(p.parameter.grad))

    def test_pool_balance_determinism_and_no_split_leak(self):
        tables=[];annotations={}
        for ep in range(6):
            t=ep%2
            action=np.repeat(np.arange(80)[:,None],6,axis=1).tolist()
            tables.append({'episode_index':[ep]*80,'task_index':[t]*80,'action':action})
            annotations[ep]={'task':TASKS[t],'markers':{'first_contact':45}}
        pool=ApproachPairs(tables[:4],annotations,np.ones(6))
        plan=pool.plan(27,8,42)
        self.assertEqual(plan,pool.plan(27,8,42))
        self.assertEqual(sum(r['task_index']==0 for r in plan),4)
        self.assertTrue(all(r['episode']<4 for r in plan))
        self.assertTrue(all(r['frame']+r['cutoff']==45 for r in plan))
        self.assertTrue(all(1<=r['cutoff']<50 for r in plan))

    def test_unknown_instruction_rejected(self):
        p=FakePolicy()
        with self.assertRaises(ValueError):
            paired_instruction_loss(p,{'task':['RC'],'action':torch.zeros(1,4,6)},[2],noise=torch.zeros(1,4,8),time=torch.zeros(1))

    def test_anchor_control_does_not_push_wrong_branch(self):
        good=torch.tensor([.3],requires_grad=True);bad=torch.tensor([.1],requires_grad=True)
        loss,_=margin_objective(good,bad,ranking_weight=0)
        loss.backward()
        self.assertEqual(good.grad.item(),1.)
        self.assertEqual(bad.grad.item(),0.)

    def test_same_randomness_can_be_restored_between_auxiliary_updates(self):
        torch.manual_seed(42)
        saved=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(100003)
            torch.randn(100)
        self.assertTrue(torch.equal(saved,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
