"""Storage conversion/RNG/order and finite optimizer checks for the supplement."""

import copy
import unittest
import torch
from torch import nn

from experiments import change_ablations as core
from experiments import change_factor_storage as supplement


class FactorStorageTests(unittest.TestCase):
    def test_same_fp32_draws_rounded_and_only_factor_storage_changes(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            for method in supplement.SUPPLEMENT_METHODS:
                base = nn.Linear(11, 7, device=device, dtype=torch.bfloat16)
                torch.manual_seed(17)
                reference = core.inject_adapters(copy.deepcopy(base), supplement.BASE_RECIPES[method], rank=3)
                reference_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state() if device == 'cuda' else None)
                torch.manual_seed(17)
                actual = supplement.inject_adapters(copy.deepcopy(base), method, rank=3)
                actual_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state() if device == 'cuda' else None)
                for value, target in zip(actual_rng, reference_rng):
                    if value is not None:
                        torch.testing.assert_close(value, target, rtol=0, atol=0)
                self.assertEqual(list(actual.state_dict()), list(reference.state_dict()))
                self.assertEqual([n for n,p in actual.named_parameters() if p.requires_grad], [n for n,p in reference.named_parameters() if p.requires_grad])
                for name in ('weight','bias','m'):
                    torch.testing.assert_close(getattr(actual,name),getattr(reference,name),rtol=0,atol=0)
                for name in ('lora_A','lora_B'):
                    self.assertEqual(getattr(actual,name).dtype,torch.bfloat16)
                    torch.testing.assert_close(getattr(actual,name),getattr(reference,name).bfloat16(),rtol=0,atol=0)

    def test_bf16_factor_gradients_and_adamw_state_with_and_without_autocast(self):
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            for method in supplement.SUPPLEMENT_METHODS:
                for autocast in (False,True):
                    layer=supplement.inject_adapters(nn.Linear(11,7,device=device,dtype=torch.bfloat16),method,rank=3)
                    with torch.no_grad():
                        layer.lora_A.normal_(std=.2)
                        layer.lora_B.normal_(std=.2)
                        layer.m.mul_(1.1)
                    optimizer=torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=.003)
                    x=torch.randn(5,11,device=device,dtype=torch.bfloat16,requires_grad=True)
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):
                        output=layer(x)
                        loss=output.float().square().mean()
                    self.assertEqual(output.dtype,torch.bfloat16)
                    loss.backward()
                    for p in (x,layer.lora_A,layer.lora_B,layer.m):
                        self.assertTrue(torch.isfinite(p.grad).all())
                        self.assertGreater(float(p.grad.norm()),0)
                    torch.nn.utils.clip_grad_norm_([p for p in layer.parameters() if p.requires_grad],.2)
                    optimizer.step()
                    for p in (layer.lora_A,layer.lora_B):
                        self.assertEqual(p.grad.dtype,torch.bfloat16)
                        self.assertEqual(optimizer.state[p]['exp_avg'].dtype,torch.bfloat16)
                        self.assertEqual(optimizer.state[p]['exp_avg_sq'].dtype,torch.bfloat16)
                    supplement.assert_no_clamp(layer)


if __name__=='__main__':
    unittest.main()
