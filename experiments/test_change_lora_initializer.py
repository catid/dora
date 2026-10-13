"""Validate initializer isolation and the unchanged factorized LoRA kernel."""

import copy
import unittest

import torch
from torch import nn

from experiments import adapters
from experiments import change_ablations as core
from experiments import change_lora_initializer as supplement


class LoRAInitializerTests(unittest.TestCase):
    def test_conventional_constructor_is_exact_passthrough(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            base = nn.Linear(11, 7, device=device, dtype=torch.bfloat16)
            torch.manual_seed(17)
            expected = adapters.inject_adapters(copy.deepcopy(base), 'lora', rank=3)
            expected_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state() if device == 'cuda' else None)
            torch.manual_seed(17)
            actual = supplement.inject_adapters(copy.deepcopy(base), 'lora_conventional', rank=3)
            actual_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state() if device == 'cuda' else None)
            self.assertEqual(list(actual.state_dict()), list(expected.state_dict()))
            for name, value in actual.state_dict().items():
                torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
            for value, target in zip(actual_rng, expected_rng):
                if value is not None:
                    torch.testing.assert_close(value, target, rtol=0, atol=0)

    def test_initializer_matches_historical_dora_factors_rng_and_order(self):
        for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
            for dtype in (torch.float32, torch.bfloat16):
                base = nn.Linear(11, 7, device=device, dtype=dtype)
                torch.manual_seed(17)
                reference = core.inject_adapters(copy.deepcopy(base), 'ab_original', rank=3)
                reference_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state() if device == 'cuda' else None)
                torch.manual_seed(17)
                actual = supplement.inject_adapters(copy.deepcopy(base), 'lora_historical_init', rank=3)
                actual_rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state() if device == 'cuda' else None)
                for value, target in zip(actual_rng, reference_rng):
                    if value is not None:
                        torch.testing.assert_close(value, target, rtol=0, atol=0)
                self.assertEqual([n for n,p in actual.named_parameters() if p.requires_grad], ['lora_B','lora_A'])
                self.assertFalse(actual.use_dora or actual.use_nora)
                self.assertIsNone(actual.m)
                for name in ('weight','bias','lora_A','lora_B'):
                    torch.testing.assert_close(getattr(actual,name),getattr(reference,name),rtol=0,atol=0)

    def test_forward_gradients_and_clipped_adamw_match_shared_lora_kernel(self):
        self.assertIs(supplement.HistoricalLoRALinear.forward, adapters.AdapterLinear.forward)
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            for dtype in (torch.float32,torch.bfloat16):
                for autocast in (False,True):
                    for method in supplement.SUPPLEMENT_METHODS:
                        base=nn.Linear(11,7,device=device,dtype=dtype)
                        torch.manual_seed(17)
                        actual=supplement.inject_adapters(copy.deepcopy(base),method,rank=3)
                        reference=adapters.AdapterLinear(copy.deepcopy(base),'lora',rank=3)
                        with torch.no_grad():
                            actual.lora_A.normal_(std=.2)
                            actual.lora_B.normal_(std=.2)
                        reference.load_state_dict(actual.state_dict())
                        # Match treatment registration order in clip reduction.
                        names=[n for n,p in actual.named_parameters() if p.requires_grad]
                        optimizers=[torch.optim.AdamW([dict(layer.named_parameters())[n] for n in names],lr=.003) for layer in (actual,reference)]
                        for _ in range(3):
                            x=torch.randn(5,11,device=device,dtype=dtype)
                            xs=[x.clone().requires_grad_(),x.clone().requires_grad_()]
                            outputs=[]
                            for layer,opt,xx in zip((actual,reference),optimizers,xs):
                                opt.zero_grad(set_to_none=True)
                                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):
                                    y=layer(xx)
                                    loss=y.float().square().mean()
                                loss.backward()
                                outputs.append(y)
                            torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
                            torch.testing.assert_close(xs[0].grad,xs[1].grad,rtol=0,atol=0)
                            for name in names:
                                p,q=getattr(actual,name),getattr(reference,name)
                                self.assertTrue(torch.isfinite(p.grad).all())
                                self.assertGreater(float(p.grad.norm()),0)
                                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                            for layer,opt in zip((actual,reference),optimizers):
                                torch.nn.utils.clip_grad_norm_([dict(layer.named_parameters())[n] for n in names],.2)
                                opt.step()
                            for name in names:
                                torch.testing.assert_close(getattr(actual,name),getattr(reference,name),rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
