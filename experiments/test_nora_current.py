"""Independent NoRA formula, initializer pairing and zero-down controls."""
import copy
from dataclasses import asdict
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from dora import DoRALayer
from experiments import next_ablations as previous
from experiments import nora_current as ab
from experiments.test_next_ablations import NextAblationTests, DEVICES, rng


class CurrentNoRATests(unittest.TestCase):
    exact = NextAblationTests.exact
    trajectory = NextAblationTests.trajectory

    def test_two_by_two_recipes_change_only_nora_or_initializer(self):
        self.assertEqual(len(ab.REGISTRY), 4)
        for left, right, changed in (("current_dora", "current_dora_nora", ["nora"]),
                                     ("init_dora", "init_dora_nora", ["nora"]),
                                     ("current_dora", "init_dora", ["published_init"]),
                                     ("current_dora_nora", "init_dora_nora", ["published_init"])):
            a, b = asdict(ab.REGISTRY[left]), asdict(ab.REGISTRY[right])
            self.assertEqual([key for key in a if a[key] != b[key]], changed)
        for method, recipe in ab.REGISTRY.items():
            self.assertEqual(recipe.axis, 0)
            self.assertFalse(recipe.detach_norm or recipe.norm_fp32 or recipe.factorized or recipe.clamp_norm)
            self.assertTrue(recipe.magnitude_fp32)
            self.assertEqual(recipe.nora_epsilon, 1e-12)
            self.assertEqual(ab.recipe_manifest(method)["nora"], recipe.nora)

    def test_initialization_rng_order_storage_equal_within_each_pair(self):
        for device in DEVICES:
            for dtype in (torch.float32, torch.bfloat16, torch.float64):
                for off, on in (("current_dora", "current_dora_nora"), ("init_dora", "init_dora_nora")):
                    with self.subTest(device=device, dtype=dtype, pair=off):
                        base = nn.Linear(11, 7, device=device, dtype=dtype)
                        torch.manual_seed(42)
                        reference = previous.AblationLinear(copy.deepcopy(base), ab.SOURCE_METHODS[off], 3)
                        expected_rng = rng(device)
                        for method in (off, on):
                            torch.manual_seed(42)
                            actual = ab.AblationLinear(copy.deepcopy(base), method, 3)
                            for a,b in zip(rng(device), expected_rng):
                                if a is not None:self.exact(a,b)
                            self.assertEqual(list(actual.state_dict()), list(reference.state_dict()))
                            for name,value in reference.state_dict().items():
                                self.assertEqual(value.dtype,actual.state_dict()[name].dtype)
                                self.exact(value,actual.state_dict()[name])
                            self.assertEqual(actual.m.dtype, torch.float64 if dtype == torch.float64 else torch.float32)
                            diagnostic = actual.norm_diagnostics()
                            self.assertEqual(diagnostic['clamp_activations'],0)
                            self.assertEqual(diagnostic['nora_initial_zero_columns'],11 if method=='current_dora_nora' else 0)
                            self.assertEqual(diagnostic['nora_initial_epsilon_columns'],11 if method=='current_dora_nora' else 0)
                            self.assertEqual(diagnostic['nora_norm_evaluations'],1 if method==on else 0)

    def test_no_nora_variants_exact_previous_clipped_adamw(self):
        for device in DEVICES:
            for method in ('current_dora','init_dora'):
                for dtype in (torch.float32,torch.bfloat16):
                    for autocast in (False,True):
                        for grouped in (False,True):
                            with self.subTest(device=device,method=method,dtype=dtype,autocast=autocast,grouped=grouped):
                                base=nn.Linear(11,7,device=device,dtype=dtype,bias=False)
                                torch.manual_seed(23)
                                expected=previous.AblationLinear(copy.deepcopy(base),ab.SOURCE_METHODS[method],3)
                                torch.manual_seed(23)
                                actual=ab.AblationLinear(copy.deepcopy(base),method,3)
                                self.trajectory(actual,expected,device,dtype,autocast,grouped)

    def test_current_default_exact_public_class(self):
        for device in DEVICES:
            for dtype,autocast in ((torch.float32,False),(torch.bfloat16,True)):
                with self.subTest(device=device,dtype=dtype):
                    base=nn.Linear(11,7,device=device,dtype=dtype)
                    torch.manual_seed(52)
                    expected=DoRALayer(11,7,3,base.weight.detach().clone(),base.bias.detach().clone()).to(device=device)
                    torch.manual_seed(52)
                    actual=ab.AblationLinear(copy.deepcopy(base),'current_dora',3)
                    self.trajectory(actual,expected,device,dtype,autocast)

    def test_independent_fp64_dense_outputs_and_all_derivatives(self):
        for device in DEVICES:
            for method,recipe in ab.REGISTRY.items():
                with self.subTest(device=device,method=method):
                    layer=ab.AblationLinear(nn.Linear(11,7,device=device,dtype=torch.float64),method,3).double()
                    with torch.no_grad():
                        layer.lora_A.normal_(std=.2);layer.lora_B.normal_(std=.2);layer.m.mul_(1.2)
                    x=torch.randn(4,11,device=device,dtype=torch.float64,requires_grad=True)
                    xr=x.detach().clone().requires_grad_()
                    down,up,m=[t.detach().clone().requires_grad_() for t in (layer.lora_A,layer.lora_B,layer.m)]
                    effective=down/down.square().sum(dim=0,keepdim=True).sqrt().clamp_min(1e-12) if recipe.nora else down
                    adapted=layer.weight+up@effective
                    weight=m*(adapted/adapted.square().sum(dim=0,keepdim=True).sqrt())
                    expected=F.linear(xr,weight,layer.bias);actual=layer(x)
                    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                    probe=torch.randn_like(actual)
                    gradients=torch.autograd.grad((actual*probe).sum(),(x,layer.lora_A,layer.lora_B,layer.m))
                    references=torch.autograd.grad((expected*probe).sum(),(xr,down,up,m))
                    for value,reference in zip(gradients,references):
                        self.assertTrue(torch.isfinite(value).all())
                        self.assertGreater(float(value.norm()),0)
                        torch.testing.assert_close(value,reference,rtol=1e-11,atol=1e-11)

    def test_nora_exact_down_column_normalization_and_derivative(self):
        layer=ab.AblationLinear(nn.Linear(11,7,dtype=torch.float64),'init_dora_nora',3).double()
        with torch.no_grad():layer.lora_A.normal_(std=.2)
        expected=layer.lora_A/layer.lora_A.norm(dim=0,keepdim=True)
        self.exact(layer.effective_down(),expected)
        torch.testing.assert_close(layer.effective_down().norm(dim=0),torch.ones(11,dtype=torch.float64),atol=1e-15,rtol=1e-15)
        probe=torch.randn_like(layer.lora_A)
        gradient=torch.autograd.grad((layer.effective_down()*probe).sum(),layer.lora_A)[0]
        # The genuine normalization derivative is tangent to each input column.
        torch.testing.assert_close((gradient*layer.lora_A).sum(dim=0),torch.zeros(11,dtype=torch.float64),atol=1e-14,rtol=0)
        self.assertEqual(layer.norm_diagnostics()['nora_epsilon_column_events'],0)

    def test_literal_zero_down_epsilon_gradient_and_finite_clipped_updates(self):
        for device in DEVICES:
            for dtype,autocast in ((torch.float32,False),(torch.bfloat16,True)):
                with self.subTest(device=device,dtype=dtype):
                    base=nn.Linear(11,7,device=device,dtype=dtype)
                    torch.manual_seed(23);off=ab.AblationLinear(copy.deepcopy(base),'current_dora',3)
                    torch.manual_seed(23);on=ab.AblationLinear(copy.deepcopy(base),'current_dora_nora',3)
                    x=torch.randn(4,11,device=device,dtype=dtype);target=torch.randn(4,7,device=device)
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):
                        outputs=[layer(x) for layer in (off,on)]
                        self.exact(*outputs)
                        losses=[(y.float()-target).square().mean() for y in outputs]
                    for loss in losses:loss.backward()
                    torch.testing.assert_close(on.lora_A.grad,off.lora_A.grad/1e-12,atol=0,rtol=2e-6)
                    self.exact(on.m.grad,off.m.grad)
                    self.assertEqual(int(torch.count_nonzero(on.lora_B.grad)),0)
                    parameters=[p for p in on.parameters() if p.requires_grad]
                    opt=torch.optim.AdamW(parameters,lr=.001)
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                    self.assertTrue(torch.isfinite(norm));self.assertGreater(float(norm),1e8)
                    opt.step()
                    self.assertTrue(all(torch.isfinite(p).all() for p in parameters))
                    self.assertTrue(torch.all(on.lora_A.detach().norm(dim=0)>1e-12))
                    for _ in range(4):
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):
                            loss=(on(x).float()-target).square().mean()
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                        self.assertTrue(torch.isfinite(loss) and torch.isfinite(norm));opt.step()
                    diagnostic=on.assert_no_clamp()
                    self.assertEqual(diagnostic['nora_initial_zero_columns'],11)
                    self.assertEqual(diagnostic['nora_minimum_raw_column_norm'],0.)
                    self.assertEqual(diagnostic['nora_epsilon_column_events'],22)
                    self.assertEqual(diagnostic['nora_norm_evaluations'],6)
                    self.assertEqual(diagnostic['clamp_activations'],0)

    def test_injection_grouping_aliases_and_zero_down_diagnostics(self):
        model=nn.Module();model.first=nn.Linear(11,7,bias=False);model.alias=model.first;model.head=nn.Linear(7,2)
        ab.inject_adapters(model,'current_dora_nora',3,targets=('alias',))
        self.assertIs(model.first,model.alias);self.assertFalse(model.head.weight.requires_grad)
        self.assertEqual(ab.parameter_groups(model,.001)[0]['param_names'],['first.lora_B','first.lora_A'])
        self.assertEqual(ab.assert_no_clamp(model)['first']['nora_initial_zero_columns'],11)
        self.assertIn(previous.core.AblationLinear,ab.backend().adapter_types)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
