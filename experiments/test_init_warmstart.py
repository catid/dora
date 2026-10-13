import copy
import unittest
import torch
from experiments.init_warmstart import backend
from experiments import init_ablations as ab


class WarmstartTests(unittest.TestCase):
    def setUp(self):
        g=torch.Generator().manual_seed(500)
        self.w=torch.randn(8,6,generator=g)*.1
        x=torch.randn(24,6,generator=g);dy=torch.randn(24,8,generator=g)
        self.stats={'weight_grad':dy.T@x/24,'input_cov':x.T@x/24,
                    'output_fisher_diag':dy.square().mean(0),'training_population_size':100,
                    'input_gram_batches':[x.T@x]}
        self.calib={'modules':{'0':self.stats},'metadata':{'split':'train'}}

    def make(self,spec):
        model=torch.nn.Sequential(torch.nn.Linear(6,8,bias=False))
        with torch.no_grad():model[0].weight.copy_(self.w)
        backend(spec,calibration=self.calib).inject_adapters(model,'case',rank=2)
        return model

    def test_initial_nonzero_perturbation_manual_output_and_gradients(self):
        for spec in [{'method':'loraone'},{'method':'lorada'},
                     *[{'method':'cntlora','mode':m} for m in ['cross','self','shift']]]:
            with self.subTest(spec=spec):
                model=self.make(spec);layer=model[0]
                self.assertTrue(torch.equal(layer.weight,self.w))
                self.assertTrue(torch.equal(layer.m,self.w.norm(dim=0,keepdim=True)))
                self.assertFalse(layer.initialization['initial_adapter_delta_is_zero'])
                x=torch.arange(18,dtype=torch.float32).reshape(3,6)/20
                output=model(x);output.square().sum().backward()
                m=layer.m.detach().clone().requires_grad_();b=layer.lora_B.detach().clone().requires_grad_();a=layer.lora_A.detach().clone().requires_grad_()
                v=self.w+b@a;reference=x@(m*v/v.norm(dim=0,keepdim=True)).T
                reference.square().sum().backward()
                torch.testing.assert_close(output,reference)
                for actual,expected in [(layer.m,m),(layer.lora_B,b),(layer.lora_A,a)]:
                    torch.testing.assert_close(actual.grad,expected.grad,atol=2e-5,rtol=2e-5)
                    self.assertTrue(torch.isfinite(actual.grad).all())
                self.assertGreater(float(layer.lora_A.grad.norm()),0)
                self.assertGreater(float(layer.lora_B.grad.norm()),0)
                self.assertFalse(torch.equal(output,x@self.w.T))

    def test_adapted_column_policy_reconstructs_initial_perturbed_weight(self):
        model=self.make({'method':'loraone','magnitude_policy':'adapted_column'});layer=model[0]
        x=torch.randn(3,6)
        torch.testing.assert_close(model(x),x@(self.w+layer.lora_B@layer.lora_A).T)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_bfloat16_nonzero_initialization_and_gradient_flow(self):
        for method in ['loraone','lorada','cntlora']:
            model=torch.nn.Sequential(torch.nn.Linear(6,8,bias=False,device='cuda',dtype=torch.bfloat16))
            with torch.no_grad():model[0].weight.copy_(self.w.cuda())
            bound=backend({'method':method},calibration=self.calib)
            bound.inject_adapters(model,method,rank=2)
            layer=model[0]
            x=torch.randn(3,6,device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x)
            out.float().square().sum().backward()
            self.assertEqual(layer.weight.dtype,torch.bfloat16)
            for value in [layer.m,layer.lora_A,layer.lora_B]:
                self.assertEqual(value.dtype,torch.float32)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(float(value.grad.norm()),0)
            bound.assert_no_clamp(model)

    def test_rng_registration_and_trainable_checkpoint_reload(self):
        base=torch.nn.Sequential(torch.nn.Linear(6,8,bias=False))
        with torch.no_grad():base[0].weight.copy_(self.w)
        torch.manual_seed(22);control=copy.deepcopy(base);ab.inject_adapters(control,ab.InitSpec(),rank=2);rng=torch.get_rng_state()
        torch.manual_seed(22);bound=backend({'method':'loraone'},calibration=self.calib);model=copy.deepcopy(base);bound.inject_adapters(model,'one',2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual([n for n,p in model[0].named_parameters() if p.requires_grad],['m','lora_B','lora_A'])
        saved={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        restored=copy.deepcopy(base);bound.inject_adapters(restored,'one',2)
        restored.load_state_dict(saved,strict=False)
        x=torch.randn(3,6);self.assertTrue(torch.equal(model(x),restored(x)))
        self.assertEqual(bound.recipe_manifest('one'),backend({'method':'loraone'},factor_seed=43,calibration=self.calib).recipe_manifest('one'))
        fresh=backend({'method':'loraone'},calibration=self.calib)
        self.assertEqual(set(fresh.assert_no_clamp(model)),{'0'})


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
