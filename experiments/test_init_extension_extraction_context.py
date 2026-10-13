"""Check Qwen-style block hooks, attention masking, and error-state isolation."""
from types import SimpleNamespace
import random
import unittest

import numpy as np
import torch
from torch import nn

from experiments import init_extension_extraction_context as context
from experiments import init_extension as formulas
from experiments import init_extension_runtime as runtime
from experiments import next_ablations as accepted


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.proj=accepted.AblationLinear(nn.Linear(4,4),'next_base',2)
    def forward(self,x):return (torch.tanh(self.proj(x)),)


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(9,4);self.blocks=nn.ModuleList([Block(),Block()]);self.fail=False
    def forward(self,input_ids,attention_mask,use_cache=False):
        random.random();np.random.rand();torch.rand(1)
        x=self.embedding(input_ids)
        for block in self.blocks:
            x=block(x)[0]
            if self.fail:raise RuntimeError('intentional model error')
        return SimpleNamespace(logits=x*10000.)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123);self.model=Model();self.model.blocks[1].eval()
        self.modules={f'blocks.{i}.proj':block.proj for i,block in enumerate(self.model.blocks)}
        self.batches=[{'input_ids':torch.tensor([[1,2,8],[3,8,8]]),
                       'attention_mask':torch.tensor([[1,1,0],[1,0,0]])}]
    def state(self):
        return {'modes':[m.training for m in self.model.modules()],
                'cpu':torch.get_rng_state().clone(),'python':random.getstate(),'numpy':np.random.get_state(),
                'parameters':{n:p.detach().clone() for n,p in self.model.named_parameters()},
                'diagnostics':{n:(p._minimum_raw_norm.clone(),p._clamp_activations,p._norm_evaluations) for n,p in self.modules.items()}}
    def assert_state(self,before):
        self.assertEqual(before['modes'],[m.training for m in self.model.modules()])
        self.assertTrue(torch.equal(before['cpu'],torch.get_rng_state()));self.assertEqual(before['python'],random.getstate())
        after=np.random.get_state();self.assertEqual(before['numpy'][0],after[0]);np.testing.assert_array_equal(before['numpy'][1],after[1])
        self.assertEqual(before['numpy'][2:],after[2:])
        self.assertTrue(all(torch.equal(before['parameters'][n],p) for n,p in self.model.named_parameters()))
        for n,p in self.modules.items():
            v=before['diagnostics'][n];self.assertTrue(torch.equal(v[0],p._minimum_raw_norm));self.assertEqual(v[1:],(p._clamp_activations,p._norm_evaluations))
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))
    def test_masks_actual_block_outputs_and_restores_state(self):
        before=self.state();expected={}
        with runtime.temporary_factors(self.modules,None),torch.no_grad():
            hidden=self.model.embedding(self.batches[0]['input_ids'])
            for i,block in enumerate(self.model.blocks):
                hidden=block(hidden)[0];moment=formulas.ScalarMoments();moment.add(hidden,self.batches[0]['attention_mask'])
                expected[f'blocks.{i}']=moment.result()
        actual=context.observe_blocks(self.model,self.modules,list(expected),self.batches,None,'baseline',recipe='paper')
        self.assertEqual(actual,expected);self.assertEqual([r['count'] for r in actual.values()],[12,12])
        self.assertLess(max(row['var'] for row in actual.values()),1.)
        self.assert_state(before)
    def test_failure_restores_hooks_modes_factors_rng(self):
        self.model.fail=True;before=self.state()
        factors={n:(torch.full_like(p.lora_B,3.),torch.full_like(p.lora_A,-2.)) for n,p in self.modules.items()}
        with self.assertRaisesRegex(RuntimeError,'intentional model error'):
            context.observe_blocks(self.model,self.modules,['blocks.0','blocks.1'],self.batches,factors,'probe',recipe='source_v2')
        self.assert_state(before)


if __name__=='__main__':unittest.main()
