"""Independent equations and callback boundaries for the separate extension."""
import copy
import math
import unittest

import numpy as np
import torch

from experiments import init_extension as extension


class ExtensionTests(unittest.TestCase):
    def test_null_matches_dense_projector_and_singular_split(self):
        rng=np.random.default_rng(17)
        orthogonal=np.linalg.qr(rng.normal(size=(7,7)))[0]
        covariance=orthogonal@np.diag([.03,.2,.5,1.,2.,3.,4.])@orthogonal.T
        weight=rng.normal(size=(5,7))
        w=torch.tensor(weight,dtype=torch.float32);c=torch.tensor(covariance,dtype=torch.float32)
        old_w=w.clone();old_c=c.clone();state=torch.random.get_rng_state().clone()
        up,down,metadata=extension.initialize_lora_null(w,2,c)
        target=weight@orthogonal[:,:2]@orthogonal[:,:2].T
        np.testing.assert_allclose((up@down).numpy(),target,atol=3e-6,rtol=3e-5)
        values=np.linalg.svd(target,compute_uv=False)[:2]
        np.testing.assert_allclose((up.T@up).numpy(),np.diag(values),atol=4e-6,rtol=3e-5)
        np.testing.assert_allclose((down@down.T).numpy(),np.diag(values),atol=4e-6,rtol=3e-5)
        self.assertTrue(torch.equal(w,old_w));self.assertTrue(torch.equal(c,old_c))
        self.assertTrue(torch.equal(torch.random.get_rng_state(),state))
        self.assertEqual(metadata['calibration_role'],'task_only')
        self.assertTrue(metadata['compensation_required'])
        bu,bd,_=extension.initialize_lora_null(w,2,c,balance_gain=2.)
        self.assertTrue(torch.equal(bu,up*2));self.assertTrue(torch.equal(bd,down/2))

    def test_null_rejects_invalid_statistics_and_records_degeneracy(self):
        w=torch.eye(4)
        with self.assertRaises(ValueError):extension.initialize_lora_null(w,2,torch.diag(torch.tensor([-1.,1.,2.,3.])))
        with self.assertRaises(ValueError):extension.initialize_lora_null(w,2,torch.ones(4,3))
        with self.assertRaises(ValueError):extension.initialize_lora_null(w,2,w,calibration_role='unknown')
        up,down,metadata=extension.initialize_lora_null(w,2,w)
        self.assertEqual(metadata['boundary_eigenvalue_gap'],0.)
        self.assertEqual(torch.linalg.matrix_rank(up@down),2)

    def probes(self,eps=1e-8):
        base={'var':2.,'skew':-.2};sigma=.02;gamma=.2
        matrix=np.array([[2.,.4],[-.3,3.]])
        scale={'var':base['var']+matrix[0,0]*(sigma*sigma+eps),
               'skew':base['skew']+matrix[1,0]*(sigma*sigma+eps)}
        skew={'var':scale['var']+matrix[0,1]*(gamma+eps),
              'skew':scale['skew']+matrix[1,1]*(gamma+eps)}
        expected=np.array([.0009,.1]);target_delta=matrix@expected
        target={'var':base['var']+target_delta[0],'skew':base['skew']+target_delta[1]}
        return base,scale,skew,target,matrix,expected

    def test_paper_subtracts_scale_probe_and_solves_full_response(self):
        base,scale,skew,target,matrix,expected=self.probes()
        result=extension.solve_gme_response(base,scale,skew,target)
        np.testing.assert_allclose([[result['a'],result['b']],[result['c'],result['d']]],matrix,atol=1e-10)
        np.testing.assert_allclose([result['sigma2_raw'],result['gamma_raw']],expected,atol=1e-12)
        self.assertAlmostEqual(result['sigma'],.03);self.assertAlmostEqual(result['rho'],1.5)

    def test_v2_keeps_baseline_subtraction_divergence(self):
        base,scale,skew,target,_,_=self.probes()
        matrix=np.array([[(scale['var']-base['var'])/.02**2,(skew['var']-base['var'])/.2],
                         [(scale['skew']-base['skew'])/.02**2,(skew['skew']-base['skew'])/.2]])
        wanted=np.linalg.solve(matrix,[target['var']-base['var'],target['skew']-base['skew']])
        result=extension.solve_gme_response(base,scale,skew,target,recipe='source_v2')
        np.testing.assert_allclose([result['sigma2_raw'],result['gamma_raw']],wanted,atol=1e-12)
        self.assertNotEqual(result['b'],extension.solve_gme_response(base,scale,skew,target)['b'])

    def test_v1_uses_single_diagonal_probe(self):
        base,_,skew,target,_,_=self.probes()
        result=extension.solve_gme_response(base,None,skew,target,recipe='source_v1')
        expected_var=(target['var']-base['var'])/((skew['var']-base['var'])/.02**2)
        expected_skew=(target['skew']-base['skew'])/((skew['skew']-base['skew'])/.2)
        self.assertAlmostEqual(result['sigma2_raw'],expected_var)
        self.assertAlmostEqual(result['gamma_raw'],expected_skew)
        self.assertEqual(result['b'],0.);self.assertEqual(result['c'],0.)

    def test_singular_response_has_explicit_finite_fallback(self):
        base={'var':1.,'skew':0.};target={'var':2.,'skew':.1}
        for recipe in extension.GME_RECIPES:
            result=extension.solve_gme_response(base,base,base,target,recipe=recipe)
            self.assertEqual(result['sigma'],.05);self.assertEqual(result['gamma'],.8)
            self.assertTrue(all(math.isfinite(v) for v in result.values() if isinstance(v,float)))

    def test_paper_ridge_damping_matches_independent_regularized_least_squares(self):
        matrix=np.array([[.01,0.],[0.,.00001]])
        base={'var':1.,'skew':0.}
        scale={'var':1.+matrix[0,0]*(.02**2+1e-8),'skew':0.}
        skew={'var':scale['var'],'skew':matrix[1,1]*(.2+1e-8)}
        target={'var':1.+.00001,'skew':.01}
        wanted=np.linalg.solve(matrix.T@matrix+1e-6*np.eye(2),matrix.T@np.array([.00001,.01]))
        result=extension.solve_gme_response(base,scale,skew,target)
        self.assertEqual(result['solve_mode'],'paper_ridge_damped')
        np.testing.assert_allclose([result['sigma2_raw'],result['gamma_raw']],wanted,atol=1e-12)

    def test_paper_clips_response_coefficients_at_published_bounds(self):
        base={'var':1.,'skew':0.};scale={'var':2.,'skew':-1.};skew={'var':12.,'skew':9.}
        result=extension.solve_gme_response(base,scale,skew,{'var':1.001,'skew':.002})
        self.assertEqual([result[k] for k in ('a','b','c','d')],[10.,10.,-10.,10.])

    def test_masked_pooled_scalar_moments_match_direct_central_moments(self):
        values=torch.tensor([[[1.,2.],[10000.,10000.]],[[3.,6.],[7.,8.]]])
        mask=torch.tensor([[1,0],[1,1]])
        accumulator=extension.ScalarMoments();accumulator.add(values[:1],mask[:1]);accumulator.add(values[1:],mask[1:])
        result=accumulator.result();x=np.array([1.,2.,3.,6.,7.,8.])
        self.assertEqual(result['count'],6)
        self.assertAlmostEqual(result['var'],float(x.var()))
        self.assertAlmostEqual(result['skew'],float(((x-x.mean())**3).mean()/(x.var()+1e-8)**1.5))

    def test_population_gamma_statistics_and_source_empirical_matching(self):
        generator=torch.Generator().manual_seed(67)
        for skew in (.6,-.6):
            x=extension.sample_gamma((200000,),.2,skew,generator)
            self.assertLess(abs(float(x.mean())),.002)
            self.assertLess(abs(float(x.std(unbiased=False))-.2),.003)
            observed=float(((x-x.mean())/x.std(unbiased=False)).pow(3).mean())
            self.assertLess(abs(observed-skew),.04)
        x=extension.sample_gamma((50,60),.2,.6,generator,source_sample_matching=True)
        self.assertLess(abs(float(x.mean())),1e-7)
        self.assertAlmostEqual(float(x.std(unbiased=False)),.2,places=6)
        first=torch.Generator().manual_seed(99);second=torch.Generator().manual_seed(99)
        zero=extension.sample_gamma((20,30),.2,0.,first)
        floor=extension.sample_gamma((20,30),.2,.02,second)
        self.assertTrue(torch.equal(zero,floor))

    def test_callback_uses_coordinated_probes_and_preserves_rng(self):
        weights={'first':torch.eye(3),'second':torch.diag(torch.tensor([1.,2.,3.]))}
        observed=[]
        def callback(factors,stage):
            observed.append(stage)
            if factors is not None:
                self.assertEqual(set(factors),set(weights))
                for name,(up,down) in factors.items():
                    self.assertEqual(tuple(up.shape),(3,2));self.assertEqual(tuple(down.shape),(2,3))
            deltas={'baseline':0.,'probe_scale':.001,'probe_skew':.003}
            delta=deltas.get(stage,.002)
            return {0:{'var':1.+delta,'skew':.1+delta,'count':30},
                    1:{'var':2.+delta,'skew':.2+2*delta,'count':30}}
        state=torch.random.get_rng_state().clone()
        factors,metadata=extension.initialize_gme(weights,2,{'first':0,'second':1},callback,seed=7)
        self.assertEqual(observed,['baseline','probe_scale','probe_skew','refine_0','refine_1','final'])
        self.assertTrue(torch.equal(torch.random.get_rng_state(),state))
        again,_=extension.initialize_gme(weights,2,{'first':0,'second':1},callback,seed=7)
        for name in factors:
            for actual,repeat in zip(factors[name],again[name]):self.assertTrue(torch.equal(actual,repeat))
        self.assertEqual(metadata['target'],{'var':1.,'skew':float(torch.tensor(.1))})
        self.assertFalse(metadata['compensation_required'])
        self.assertEqual(metadata['observations']['final'][0]['count'],30.)
        observed.clear()
        extension.initialize_gme(weights,2,{'first':0,'second':1},callback,recipe='source_v1',refine_steps=0)
        self.assertEqual(observed,['baseline','probe_skew','final'])

    def test_callback_count_change_is_rejected(self):
        def callback(factors,stage):
            return {0:{'var':1.,'skew':0.,'count':30 if stage=='baseline' else 29}}
        with self.assertRaisesRegex(ValueError,'counts changed'):
            extension.initialize_gme({'w':torch.eye(3)},2,{'w':0},callback)

    def test_sampling_is_cpu_fp32_despite_default_dtype(self):
        previous=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            generator=torch.Generator(device='cpu').manual_seed(5)
            for skew in (0.,.2):
                sampled=extension.sample_gamma((3,4),.02,skew,generator)
                self.assertEqual(sampled.dtype,torch.float32);self.assertEqual(sampled.device.type,'cpu')
            def callback(factors,stage):return {0:{'var':1.,'skew':0.,'count':10}}
            factors,_=extension.initialize_gme({'w':torch.eye(3)},2,{'w':0},callback,refine_steps=0)
            for factor in factors['w']:self.assertEqual(factor.dtype,torch.float32)
        finally:torch.set_default_dtype(previous)

    def test_source_moment_floor_is_explicit(self):
        moments=extension.ScalarMoments();moments.add(torch.tensor([0.,0.,.0001]))
        paper=moments.result();source=moments.result(source_stability_floor=True)
        self.assertAlmostEqual(source['skew']/paper['skew'],(paper['var']+1e-8)**1.5/1e-8)

    def test_finite_family_settings_are_four_each_and_distinguish_source_recipes(self):
        families=extension.extension_families()
        self.assertEqual(len(families),4)
        self.assertTrue(all(len(f['configs'])==4 for f in families))
        self.assertEqual([f['configs'][1]['tau_b'] for f in families[1:]],[1e-4]*3)
        self.assertEqual(families[0]['calibration_requirements'],['corda_input_cov'])
        self.assertEqual(families[0]['configs'][2]['balance_gain'],1.)


if __name__=='__main__':unittest.main()
