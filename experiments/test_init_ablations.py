"""Accepted-stream, distribution, geometry and training checks for initializers."""
import copy
from dataclasses import replace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments import init_ablations as ab
from experiments import nora_current as current
from experiments import test_next_ablations as helpers


FAMILIES = ('normal','uniform','rademacher','laplace','student_t','truncnormal',
            'orthogonal','sparse','dct','hadamard','cropped_hadamard')


class InitializerTests(unittest.TestCase):
    exact = helpers.NextAblationTests.exact
    trajectory = helpers.NextAblationTests.trajectory

    def test_spec_validation_and_explicit_rms_conventions(self):
        for kwargs in ({'mode':'bad'}, {'orientation':'bad'}, {'scale':0}, {'df':2},
                       {'sparsity':2}, {'kernel_length':float('nan')}, {'family':'bad'},
                       {'scale':2}, {'normalization':'sample_rms'}, {'kernel':'rbf'},
                       {'mode':'controlled','permute':True}):
            with self.assertRaises(ValueError): ab.InitSpec(**kwargs)
        analytic=ab.InitSpec(mode='controlled')
        sampled=replace(analytic,normalization='sample_rms')
        self.assertNotEqual(ab.recipe_manifest(analytic)['recipe_sha256'],ab.recipe_manifest(sampled)['recipe_sha256'])
        self.assertEqual(ab.local_seed(42,'a','up'),ab.local_seed(42,'a','up'))
        self.assertNotEqual(ab.local_seed(42,'a','up'),ab.local_seed(42,'b','up'))

    def test_exact_baseline_state_rng_and_five_clipped_adamw_steps(self):
        for device in helpers.DEVICES:
            for dtype,autocast in ((torch.float32,False),(torch.float32,True),(torch.bfloat16,True)):
                for grouped in (False,True):
                    with self.subTest(device=device,dtype=dtype,autocast=autocast,grouped=grouped):
                        base=nn.Linear(11,7,device=device,dtype=dtype)
                        torch.manual_seed(71); reference=current.AblationLinear(copy.deepcopy(base),'current_dora',3)
                        expected=helpers.rng(device)
                        torch.manual_seed(71); actual=ab.AblationLinear(copy.deepcopy(base),ab.InitSpec(),3)
                        for a,b in zip(helpers.rng(device),expected):
                            if a is not None:self.exact(a,b)
                        self.assertEqual(list(actual.state_dict()),list(reference.state_dict()))
                        self.trajectory(actual,reference,device,dtype,autocast,grouped)

    def test_all_families_orientations_preserve_rng_registration_and_zero_update(self):
        for device in helpers.DEVICES:
            base=nn.Sequential(nn.Linear(16,16,bias=False),nn.Linear(16,16)).to(device)
            for orientation in ('up','down'):
                for family in FAMILIES:
                    with self.subTest(device=device,orientation=orientation,family=family):
                        torch.manual_seed(51)
                        reference=copy.deepcopy(base);current.inject_adapters(reference,'current_dora',4)
                        expected=helpers.rng(device)
                        torch.manual_seed(51)
                        model=copy.deepcopy(base);spec=ab.InitSpec(mode='controlled',orientation=orientation,family=family)
                        ab.inject_adapters(model,spec,4,factor_seed=71,method=family)
                        for a,b in zip(helpers.rng(device),expected):
                            if a is not None:self.exact(a,b)
                        self.assertEqual(list(model.state_dict()),list(reference.state_dict()))
                        x=torch.randn(3,16,device=device)
                        self.exact(model(x),reference(x))
                        for name,module in model.named_modules():
                            if not isinstance(module,ab.AblationLinear):continue
                            self.assertEqual([n for n,p in module.named_parameters() if p.requires_grad],['m','lora_B','lora_A'])
                            self.assertEqual(module.norm_diagnostics()['initializer']['module_path'],name)
                            self.assertEqual(int(torch.count_nonzero(module.lora_B@module.lora_A)),0)
                            self.assertTrue(module.norm_diagnostics()['initializer']['initial_adapter_delta_is_zero'])

    def test_local_seed_reproducibility_independence_and_accepted_scaling(self):
        base=nn.Linear(16,16)
        spec=ab.InitSpec(mode='controlled',orientation='down')
        torch.manual_seed(10);first=ab.AblationLinear(copy.deepcopy(base),spec,4,module_path='a',factor_seed=42)
        torch.manual_seed(99);second=ab.AblationLinear(copy.deepcopy(base),spec,4,module_path='a',factor_seed=42)
        self.exact(first.lora_A,second.lora_A)
        third=ab.AblationLinear(copy.deepcopy(base),spec,4,module_path='b',factor_seed=42)
        self.assertFalse(torch.equal(first.lora_A,third.lora_A))
        torch.manual_seed(12);reference=ab.AblationLinear(copy.deepcopy(base),ab.InitSpec(),4)
        expected=helpers.rng('cpu')
        torch.manual_seed(12);scaled=ab.AblationLinear(copy.deepcopy(base),ab.InitSpec(mode='accepted_scaled',scale=.3),4)
        self.exact(scaled.lora_B,reference.lora_B*.3);self.exact(scaled.m,reference.m)
        self.exact(helpers.rng('cpu')[0],expected[0])

    def test_analytic_noise_variance_and_exact_sample_rms_are_distinct(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                spec=ab.InitSpec(mode='controlled',family=family,scale=.7)
                noise=ab.draw_factor(spec,(4096,8),torch.Generator().manual_seed(234))
                self.assertTrue(torch.isfinite(noise).all())
                self.assertLess(abs(float(noise.mean())),.015)
                self.assertLess(abs(float(noise.square().mean())/(.7**2/8)-1),.12)
                sampled=ab.draw_factor(replace(spec,normalization='sample_rms'),(128,8),torch.Generator().manual_seed(234))
                self.assertAlmostEqual(float(sampled.double().square().mean().sqrt()),.7/(8**.5),places=7)
        with self.assertRaises(ValueError):
            ab.draw_factor(ab.InitSpec(mode='controlled',family='hadamard'),(15,4),torch.Generator().manual_seed(1))
        with self.assertRaises(ValueError):
            ab.draw_factor(ab.InitSpec(mode='controlled',family='orthogonal'),(3,4),torch.Generator().manual_seed(1))

    def test_rank_kernels_covariance_and_structured_spectrum(self):
        for kernel in ('rbf','exponential','matern32'):
            spec=ab.InitSpec(mode='controlled',kernel=kernel,kernel_length=1.7)
            expected=ab.rank_covariance(4,spec)
            self.exact(expected.diag(),torch.ones(4,dtype=torch.float64))
            self.assertGreater(float(torch.linalg.eigvalsh(expected).min()),0)
            noise=ab.draw_factor(spec,(32768,4),torch.Generator().manual_seed(31)).double()
            torch.testing.assert_close(noise.T@noise/noise.shape[0]*4,expected,atol=.025,rtol=.025)
        for family in ('orthogonal','dct','hadamard'):
            spec=ab.InitSpec(mode='controlled',family=family)
            noise=ab.draw_factor(spec,(128,4),torch.Generator().manual_seed(31)).double()
            torch.testing.assert_close(noise.T@noise,torch.eye(4,dtype=torch.float64)*32,atol=1e-6,rtol=1e-6)

    def test_cropped_hadamard_independent_sylvester_reference(self):
        for width, rank in ((6, 4), (384, 8), (768, 2)):
            order = 1 << (width - 1).bit_length()
            # Construct the full Sylvester matrix by Kronecker products, without
            # using the production parity formula.
            reference = torch.ones(1, 1)
            while reference.shape[0] < order:
                reference = torch.kron(torch.tensor([[1., 1.], [1., -1.]]), reference)
            generator = torch.Generator().manual_seed(27)
            columns = torch.randperm(order, generator=generator)[:rank]
            rowsigns = torch.randint(2, (width, 1), generator=generator)*2-1
            colsigns = torch.randint(2, (1, rank), generator=generator)*2-1
            expected = reference[:width, columns]*rowsigns*colsigns*(.7/rank**.5)
            spec = ab.InitSpec(mode='controlled', family='cropped_hadamard', scale=.7)
            actual = ab.draw_factor(spec, (width, rank), torch.Generator().manual_seed(27))
            self.exact(actual, expected)
            for orientation in ('up', 'down'):
                base = nn.Linear(width, width, bias=False)
                layer = ab.AblationLinear(base, replace(spec, orientation=orientation), rank,
                                         module_path='projection', factor_seed=27)
                factor = layer.lora_B if orientation == 'up' else layer.lora_A.T
                self.assertEqual(tuple(factor.shape), (width, rank))
                self.assertAlmostEqual(float(factor.detach().double().square().mean()), .7**2/rank, places=7)
                self.assertEqual(int(torch.count_nonzero(layer.lora_B@layer.lora_A)), 0)

    def test_nonzero_uniform_uses_original_input_width_for_both_factors(self):
        for orientation in ('up','down'):
            spec=ab.InitSpec(mode='controlled',family='uniform',both_nonzero=True,
                             orientation=orientation,scaling='original_input',
                             scale=3**-.5,companion_scale=3**-.5)
            base=nn.Linear(11,7,bias=False)
            layer=ab.AblationLinear(base,spec,3,module_path='p',factor_seed=52)
            generator=torch.Generator().manual_seed(ab.local_seed(52,'p',orientation))
            first=(torch.rand((7 if orientation=='up' else 11,3),generator=generator)*2-1)/11**.5
            second=(torch.rand((11 if orientation=='up' else 7,3),generator=generator)*2-1)/11**.5
            up,down=(first,second.T) if orientation=='up' else (second,first.T)
            torch.testing.assert_close(layer.lora_B,up,atol=3e-8,rtol=1e-7)
            torch.testing.assert_close(layer.lora_A,down,atol=3e-8,rtol=1e-7)
            self.assertGreater(int(torch.count_nonzero(layer.lora_B@layer.lora_A)),0)
            x=torch.randn(4,11)
            reference=ab.AblationLinear(base,ab.InitSpec(),3)
            self.exact(layer(x),reference(x))

    def test_independent_fp64_forward_and_all_gradients_unchanged_by_initializer(self):
        for device in helpers.DEVICES:
            for orientation in ('up','down'):
                spec=ab.InitSpec(mode='controlled',orientation=orientation,family='uniform',normalization='sample_rms')
                layer=ab.AblationLinear(nn.Linear(11,7,dtype=torch.float64,device=device),spec,3).double()
                with torch.no_grad():
                    layer.lora_A.normal_(std=.2);layer.lora_B.normal_(std=.2);layer.m.mul_(1.2)
                x=torch.randn(4,11,dtype=torch.float64,device=device,requires_grad=True);xr=x.detach().clone().requires_grad_()
                down,up,m=[p.detach().clone().requires_grad_() for p in (layer.lora_A,layer.lora_B,layer.m)]
                v=layer.weight+up@down
                expected=F.linear(xr,m*(v/v.square().sum(dim=0,keepdim=True).sqrt()),layer.bias)
                actual=layer(x);torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                probe=torch.randn_like(actual)
                grads=torch.autograd.grad((actual*probe).sum(),(x,layer.lora_A,layer.lora_B,layer.m))
                refs=torch.autograd.grad((expected*probe).sum(),(xr,down,up,m))
                for a,b in zip(grads,refs):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)

    def test_five_finite_clipped_updates_all_families_fp32_bf16(self):
        for device in helpers.DEVICES:
            for dtype,autocast in ((torch.float32,False),(torch.bfloat16,True)):
                for orientation in ('up','down'):
                    for family in FAMILIES:
                        spec=ab.InitSpec(mode='controlled',family=family,orientation=orientation)
                        layer=ab.AblationLinear(nn.Linear(16,16,dtype=dtype,device=device),spec,4,factor_seed=76)
                        params=[p for p in layer.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.001)
                        x=torch.randn(4,16,dtype=dtype,device=device);target=torch.randn(4,16,device=device)
                        for _ in range(5):
                            opt.zero_grad(set_to_none=True)
                            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):loss=(layer(x).float()-target).square().mean()
                            self.assertTrue(torch.isfinite(loss));loss.backward()
                            self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(params,1.)))
                            opt.step();self.assertTrue(all(torch.isfinite(p).all() for p in params))
                        self.assertEqual(layer.assert_no_clamp()['clamp_activations'],0)

    def test_alias_injection_bound_backend_and_sparse_zero_vectors_reported(self):
        model=nn.Module();model.first=nn.Linear(16,16);model.alias=model.first;model.head=nn.Linear(16,2)
        spec=ab.InitSpec(mode='controlled',family='sparse',sparsity=.01)
        backend=ab.backend(spec,factor_seed=52)
        backend.inject_adapters(model,'sparse_test',4,targets=('alias',))
        self.assertIs(model.first,model.alias);self.assertFalse(model.head.weight.requires_grad)
        self.assertEqual(backend.parameter_groups(model,.001)[0]['param_names'],['first.lora_B','first.lora_A'])
        d=backend.assert_no_clamp(model)['first']
        self.assertGreater(d['initializer']['random_zero_vectors'],0)
        self.assertEqual(d['recipe_sha256'],backend.recipe_manifest('sparse_test')['recipe_sha256'])

    def test_compensated_callback_initial_identity_rng_buffers_and_reload(self):
        def callback(weight,rank,module_path):
            generator=torch.Generator().manual_seed(ab.local_seed(19,module_path,'up'))
            return torch.randn(weight.shape[0],rank,generator=generator)*.1,torch.randn(rank,weight.shape[1],generator=generator)*.1,{'source':'synthetic','module_path':module_path}
        for device in helpers.DEVICES:
            for dtype,autocast in ((torch.float32,False),(torch.bfloat16,True)):
                base=nn.Linear(11,7,dtype=dtype,device=device)
                torch.manual_seed(19);reference=ab.AblationLinear(copy.deepcopy(base),ab.InitSpec(),3)
                expected=helpers.rng(device)
                torch.manual_seed(19);layer=ab.AblationLinear(copy.deepcopy(base),ab.InitSpec(),3,module_path='projection',factor_initializer=callback,data_spec={'family':'synthetic'})
                for a,b in zip(helpers.rng(device),expected):
                    if a is not None:self.exact(a,b)
                self.exact(layer.m,reference.m);self.exact(layer.weight,base.weight)
                x=torch.randn(4,11,device=device,dtype=dtype)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):self.exact(layer(x),reference(x))
                self.assertEqual([n for n,p in layer.named_parameters() if p.requires_grad],['m','lora_B','lora_A'])
                self.assertEqual(set(dict(layer.named_buffers())),{'initial_up','initial_down'})
                opt=torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=.001)
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):loss=layer(x).float().square().mean()
                    loss.backward();self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(layer.parameters(),1.)));opt.step()
                restored=ab.AblationLinear(copy.deepcopy(base),ab.InitSpec(),3,module_path='projection',factor_initializer=callback,data_spec={'family':'synthetic'})
                restored.load_state_dict({n:p.detach() for n,p in layer.named_parameters() if p.requires_grad},strict=False)
                self.assertEqual(restored.norm_diagnostics()['initializer'],layer.norm_diagnostics()['initializer'])
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):self.exact(restored(x),layer(x))

    def test_compensated_fp64_output_all_derivatives_independent_and_rng_callback_rejected(self):
        def callback(weight,rank,path):
            generator=torch.Generator().manual_seed(34)
            return torch.randn(weight.shape[0],rank,generator=generator)*.1,torch.randn(rank,weight.shape[1],generator=generator)*.1,{'source':'synthetic'}
        for device in helpers.DEVICES:
            layer=ab.AblationLinear(nn.Linear(11,7,dtype=torch.float64,device=device),ab.InitSpec(),3,factor_initializer=callback,data_spec={'family':'synthetic'}).double()
            with torch.no_grad():layer.lora_A.add_(.03);layer.lora_B.mul_(1.3);layer.m.mul_(1.1)
            x=torch.randn(4,11,dtype=torch.float64,device=device,requires_grad=True);xr=x.detach().clone().requires_grad_()
            down,up,m=[p.detach().clone().requires_grad_() for p in (layer.lora_A,layer.lora_B,layer.m)]
            v=layer.weight+up@down-layer.initial_up@layer.initial_down
            expected=F.linear(xr,m*(v/v.square().sum(dim=0,keepdim=True).sqrt()),layer.bias)
            actual=layer(x);torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
            probe=torch.randn_like(actual)
            grads=torch.autograd.grad((actual*probe).sum(),(x,layer.lora_A,layer.lora_B,layer.m))
            refs=torch.autograd.grad((expected*probe).sum(),(xr,down,up,m))
            for a,b in zip(grads,refs):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
        def invalid(weight,rank,path):
            torch.rand(1)
            return callback(weight,rank,path)
        with self.assertRaisesRegex(ValueError,'global CPU RNG'):
            ab.AblationLinear(nn.Linear(11,7),ab.InitSpec(),3,factor_initializer=invalid,data_spec={'family':'bad'})

    def test_one_time_unit_vectors_bimi_and_dimension_scaling(self):
        for orientation in ('up','down'):
            spec=ab.InitSpec(mode='controlled',orientation=orientation,radial='unit_vectors')
            layer=ab.AblationLinear(nn.Linear(16,12),spec,4)
            factor=layer.lora_B if orientation=='up' else layer.lora_A.T
            torch.testing.assert_close(factor.norm(dim=1),torch.ones(factor.shape[0]),atol=1e-7,rtol=1e-7)
            self.assertEqual(layer.norm_diagnostics()['initializer']['target_entry_rms'],.5)
            bimi=ab.AblationLinear(nn.Linear(16,12),replace(spec,family='bimi',radial='none'),4)
            factor=bimi.lora_B if orientation=='up' else bimi.lora_A.T
            self.exact(factor,torch.eye(4)[torch.arange(factor.shape[0])%4])
        rank=ab.draw_factor(ab.InitSpec(mode='controlled'),(16,4),torch.Generator().manual_seed(17))
        fan=ab.draw_factor(ab.InitSpec(mode='controlled',scaling='fan_in'),(16,4),torch.Generator().manual_seed(17),16)
        absolute=ab.draw_factor(ab.InitSpec(mode='controlled',scaling='absolute'),(16,4),torch.Generator().manual_seed(17))
        self.exact(fan,rank*.5);self.exact(absolute,rank*2)

    def test_both_nonzero_families_peft_complementary_and_compensated_updates(self):
        specs=[ab.InitSpec(mode='controlled',family=family,both_nonzero=True,scaling='absolute',scale=.5,companion_scale=.3)
               for family in ('normal','uniform')]
        specs += [ab.InitSpec(mode='controlled',family='normal',both_nonzero=True,scaling='fan_in',scale=2**.5,companion_scale=2**.5),
                  ab.InitSpec(mode='controlled',family='peft_orthogonal',both_nonzero=True,scaling='absolute')]
        for device in helpers.DEVICES:
            for dtype,autocast in ((torch.float32,False),(torch.bfloat16,True)):
                base=nn.Linear(16,12,dtype=dtype,device=device)
                for spec in specs:
                    torch.manual_seed(14);reference=ab.AblationLinear(copy.deepcopy(base),ab.InitSpec(),4)
                    expected=helpers.rng(device)
                    torch.manual_seed(14);layer=ab.AblationLinear(copy.deepcopy(base),spec,4,factor_seed=37)
                    for a,b in zip(helpers.rng(device),expected):
                        if a is not None:self.exact(a,b)
                    self.assertTrue(layer.compensated)
                    self.assertIsNone(layer.zero_initialized_factor)
                    self.assertTrue(torch.count_nonzero(layer.lora_A)>0 and torch.count_nonzero(layer.lora_B)>0)
                    if spec.family=='peft_orthogonal':
                        self.assertLess(float((layer.lora_B.detach().double()@layer.lora_A.detach().double()).norm()),1e-7)
                    x=torch.randn(4,16,dtype=dtype,device=device)
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):self.exact(layer(x),reference(x))
                    opt=torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad],lr=.001)
                    for _ in range(5):
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=autocast):loss=layer(x).float().square().mean()
                        self.assertTrue(torch.isfinite(loss));loss.backward()
                        self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(layer.parameters(),1.)));opt.step()
                    self.assertTrue(all(torch.isfinite(p).all() for p in layer.parameters()))
        with self.assertRaisesRegex(ValueError,'even rank'):
            ab.AblationLinear(nn.Linear(16,12),specs[-1],3)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
