import unittest
import copy
import torch
from experiments.init_loftq import NF4, NF4_BOUNDARIES, quantize_nf4, initialize_loftq, backend


class LoftQTests(unittest.TestCase):
    def test_nf4_levels_zero_groups_and_tail(self):
        w = torch.zeros(3, 11)
        q, meta, state = quantize_nf4(w, 32, return_state=True)
        self.assertTrue(torch.equal(q,w))
        self.assertEqual(meta['zero_groups'], 2)
        self.assertEqual(meta['padding_elements'], 31)
        self.assertTrue((state['indices']==7).all())
        levels = torch.tensor(NF4).repeat(2).view(2,16)
        q, _ = quantize_nf4(levels, 32)
        torch.testing.assert_close(q,levels,atol=0,rtol=0)

    def test_nf4_matches_explicit_nearest_levels(self):
        w = torch.randn(13,17,generator=torch.Generator().manual_seed(42))
        q, _, state = quantize_nf4(w,64,return_state=True)
        flat = torch.nn.functional.pad(w.flatten(),(0,(-w.numel())%64)).view(-1,64)
        norm = flat/flat.abs().amax(1)[:,None]
        codes = (norm[:,:,None]-torch.tensor(NF4)).abs().argmin(2)
        self.assertTrue(torch.equal(codes,state['indices'].long()))

    def test_exact_kernel_boundary_tie_breaking(self):
        boundaries=torch.tensor(NF4_BOUNDARIES)
        below=torch.nextafter(boundaries,torch.full_like(boundaries,-torch.inf))
        above=torch.nextafter(boundaries,torch.full_like(boundaries,torch.inf))
        values=torch.cat([torch.tensor([-1.,1.]),below,boundaries,above])
        values=torch.nn.functional.pad(values,(0,64-len(values))).view(1,64)
        _,_,state=quantize_nf4(values,64,return_state=True)
        codes=state['indices'].flatten()
        torch.testing.assert_close(codes[2:17].long(),torch.arange(15))
        torch.testing.assert_close(codes[17:32].long(),torch.arange(15))
        torch.testing.assert_close(codes[32:47].long(),torch.arange(15)+1)

    def test_loftq_one_step_optimal_residual_and_rng(self):
        w = torch.randn(16,12,generator=torch.Generator().manual_seed(10)).bfloat16()
        rng = torch.random.get_rng_state().clone()
        original = w.clone()
        q,b,a,m = initialize_loftq(w,4)
        direct,_ = quantize_nf4(w)
        sv = torch.linalg.svdvals(w.float()-direct)
        self.assertAlmostEqual(m['iterations'][0]['corrected_mse_to_original'],float(sv[4:].square().sum()/w.numel()),places=7)
        self.assertTrue(torch.equal(q,direct.bfloat16()))
        self.assertLess(m['iterations'][0]['corrected_mse_to_original'],m['iterations'][0]['quantization_mse_to_original'])
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(w,original))
        q2,b2,a2,_ = initialize_loftq(w,4,{'balance_gain':2,'adapter_scale':4})
        torch.testing.assert_close(4*b2@a2,b@a)
        self.assertTrue(torch.equal(q2,q))

    def test_quantized_backend_preserves_control_rng_and_paired_base(self):
        from experiments import init_ablations as ab
        base=torch.nn.Sequential(torch.nn.Linear(12,16,dtype=torch.bfloat16))
        original=base[0].weight.clone()
        torch.manual_seed(31)
        control=copy.deepcopy(base)
        ab.inject_adapters(control,ab.InitSpec(),rank=4)
        rng=torch.random.get_rng_state().clone()
        torch.manual_seed(31)
        treatment=copy.deepcopy(base)
        backend({'method':'loftq','iterations':2}).inject_adapters(treatment,'lq',4)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(control[0].m,treatment[0].m))
        self.assertTrue(torch.equal(original,base[0].weight))
        self.assertEqual([n for n,p in treatment[0].named_parameters() if p.requires_grad],['m','lora_B','lora_A'])
        torch.manual_seed(31)
        paired=copy.deepcopy(base)
        backend({'method':'quantizer_only','iterations':2,'base_source':'paired_loftq'}).inject_adapters(paired,'q',4)
        self.assertTrue(torch.equal(treatment[0].weight,paired[0].weight))
        self.assertTrue(torch.equal(control[0].lora_B,paired[0].lora_B))
        self.assertEqual(int(paired[0].lora_A.count_nonzero()),0)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=treatment(torch.randn(3,12))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(treatment[0].lora_A.grad).all())
        self.assertTrue(torch.isfinite(treatment[0].lora_B.grad).all())
        self.assertTrue(torch.isfinite(treatment[0].m.grad).all())
        fresh=backend({'method':'loftq','iterations':2})
        self.assertEqual(set(fresh.assert_no_clamp(treatment)),{'0'})

    def test_iteration_is_deterministic_and_invalid_settings_rejected(self):
        w = torch.randn(10,8,generator=torch.Generator().manual_seed(17))
        q,b,a,m = initialize_loftq(w,2,{'iterations':3})
        q2,b2,a2,m2 = initialize_loftq(w,2,{'iterations':3})
        self.assertTrue(torch.equal(q,q2) and torch.equal(b,b2) and torch.equal(a,a2))
        self.assertEqual(len(m['iterations']),3)
        for spec in [{'iterations':0},{'bits':8},{'blocksize':12}]:
            with self.assertRaises(ValueError):initialize_loftq(w,2,spec)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
