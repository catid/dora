"""Actual standalone constructor/forward parity, including monitoring hooks."""

import copy
import unittest
import torch
from torch import nn
from dora_row import DoRALayer
from experiments import change_standalone_endpoint as supplement


class StandaloneEndpointTests(unittest.TestCase):
    def test_literal_constructor_rng_outputs_gradients_and_adamw(self):
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            for dtype in (torch.float32,torch.bfloat16):
                for autocast in (False,True):
                    base=nn.Linear(11,7,device=device,dtype=dtype)
                    torch.manual_seed(17)
                    reference=DoRALayer.from_linear(copy.deepcopy(base),rank=3)
                    reference_rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state() if device=='cuda' else None)
                    torch.manual_seed(17)
                    actual=supplement.inject_adapters(copy.deepcopy(base),'standalone_current',rank=3)
                    actual_rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state() if device=='cuda' else None)
                    self.assertIs(type(actual),DoRALayer)
                    self.assertEqual(list(actual.state_dict()),list(reference.state_dict()))
                    self.assertEqual([n for n,p in actual.named_parameters() if p.requires_grad],['m','lora_A','lora_B'])
                    for a,b in zip(actual_rng,reference_rng):
                        if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
                    for name,value in actual.state_dict().items():torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0)
                    self.assertEqual(actual.lora_A.dtype,dtype)
                    self.assertEqual(actual.m.dtype,torch.float32)
                    self.assertNotEqual(actual.weight.data_ptr(),base.weight.data_ptr())
                    self.assertFalse(actual.weight.requires_grad or actual.bias.requires_grad)
                    with torch.no_grad():
                        actual.lora_B.normal_(std=.2)
                        actual.m.mul_(1.1)
                    reference.load_state_dict(actual.state_dict())
                    opts=[torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=.003) for layer in (actual,reference)]
                    for step in range(3):
                        x=torch.randn(5,11,device=device,dtype=dtype)
                        outputs=[];inputs=[]
                        for layer,opt in zip((actual,reference),opts):
                            xx=x.clone().requires_grad_();inputs.append(xx)
                            opt.zero_grad(set_to_none=True)
                            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):
                                y=layer(xx);loss=y.float().square().mean()
                            loss.backward();outputs.append(y)
                        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
                        torch.testing.assert_close(inputs[0].grad,inputs[1].grad,rtol=0,atol=0)
                        for name in ('m','lora_A','lora_B'):
                            a,b=getattr(actual,name),getattr(reference,name)
                            self.assertTrue(torch.isfinite(a.grad).all())
                            self.assertGreater(float(a.grad.norm()),0)
                            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
                        for layer,opt in zip((actual,reference),opts):
                            torch.nn.utils.clip_grad_norm_([p for p in layer.parameters() if p.requires_grad],.2);opt.step()
                        for name,value in actual.state_dict().items():torch.testing.assert_close(value,reference.state_dict()[name],rtol=0,atol=0)
                    supplement.assert_no_clamp(actual)

    def test_monitor_rejects_real_floor_activation(self):
        base=nn.Linear(11,7,dtype=torch.bfloat16)
        with torch.no_grad():base.weight.zero_()
        layer=supplement.inject_adapters(base,'standalone_current',rank=3)
        with self.assertRaises(RuntimeError):supplement.assert_no_clamp(layer)


if __name__=='__main__':unittest.main()
