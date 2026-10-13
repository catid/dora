"""Independent small-matrix identities for research initializer recipes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.init_data import METHODS, initialize_factors, backend


class FactorRecipeTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(7)
        self.w = torch.randn(8, 6, generator=gen)
        x = torch.randn(40, 6, generator=gen)
        dy = torch.randn(40, 8, generator=gen)
        self.c = {'input_cov': x.T @ x / 40,
                  'input_centered_cov': (x-x.mean(0)).T @ (x-x.mean(0)) / 40,
                  'input_mean': x.mean(0), 'input_count': 40,
                  'output_cov': (x @ self.w.T).T @ (x @ self.w.T) / 40,
                  'astra_output_cov': dy.T @ dy / 40,
                  'output_grad_cov': dy.T @ dy / 40,
                  'output_fisher_diag': dy.square().mean(0),
                  'weight_grad': dy.T @ x / 40, 'training_population_size': 10,
                  'input_gram_batches': [x[:20].T@x[:20], x[20:].T@x[20:]]}

    def run_recipe(self, method, **kwargs):
        return initialize_factors(self.w, 2, dict(method=method, **kwargs), self.c)

    def test_all_shapes_finite_rng_inputs_and_repeatability(self):
        before_w = self.w.clone()
        before_c = copy.deepcopy(self.c)
        rng = torch.random.get_rng_state().clone()
        for method in METHODS:
            kwargs = {'prefit_steps': 3} if method == 'inilora' else {}
            b, a, meta = self.run_recipe(method, **kwargs)
            b2, a2, _ = self.run_recipe(method, **kwargs)
            self.assertEqual(tuple(b.shape), (8, 2))
            self.assertEqual(tuple(a.shape), (2, 6))
            self.assertTrue(torch.equal(b, b2) and torch.equal(a, a2), method)
            self.assertTrue(torch.isfinite(b).all() and torch.isfinite(a).all())
            json.dumps(meta, allow_nan=False)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(self.w, before_w))
        for key in self.c:
            if isinstance(self.c[key], torch.Tensor):
                self.assertTrue(torch.equal(self.c[key], before_c[key]))

    def test_spectral_projection_and_factor_balance(self):
        u, s, vh = torch.linalg.svd(self.w, full_matrices=False)
        for method, sl in [('pissa', slice(0, 2)), ('milora', slice(-2, None))]:
            b, a, _ = self.run_recipe(method)
            ref = (u[:, sl] * s[sl]) @ vh[sl]
            torch.testing.assert_close(b @ a, ref)
            bg, ag, _ = self.run_recipe(method, balance_gain=3)
            torch.testing.assert_close(bg @ ag, ref)
        b, a, _ = self.run_recipe('olora')
        q, r = torch.linalg.qr(self.w, mode='reduced')
        torch.testing.assert_close(b @ a, q[:, :2] @ r[:2])

    def test_eva_and_tlora_zero_up_and_subspace(self):
        for method in ['eva', 'tlora']:
            b, a, _ = self.run_recipe(method)
            self.assertEqual(b.count_nonzero(), 0)
            torch.testing.assert_close(a @ a.T, torch.eye(2), atol=1e-5, rtol=1e-5)
        b, a, _ = self.run_recipe('eva', whiten=True)
        values = torch.linalg.eigvalsh(self.c['input_centered_cov'])[-2:].flip(0)
        torch.testing.assert_close((a*a).sum(1), (values * 40).rsqrt(), atol=1e-5, rtol=1e-5)

    def test_corda_spectral_reconstruction(self):
        b, a, meta = self.run_recipe('corda', damping=.02)
        c = self.c['input_cov']
        damp = c + meta['damping_used'] * c.diag().mean() * torch.eye(6)
        u, s, vh = torch.linalg.svd(self.w @ damp, full_matrices=False)
        torch.testing.assert_close((b@a)@damp, (u[:, :2]*s[:2])@vh[:2], atol=2e-5, rtol=2e-5)

    def test_loraone_negative_gradient_and_ga_unweighted_disjoint_basis(self):
        b, a, _ = self.run_recipe('loraone', gamma=4)
        u, s, vh = torch.linalg.svd(-self.c['weight_grad'], full_matrices=False)
        torch.testing.assert_close(b@a, (u[:, :2]*s[:2])@vh[:2]/s[0]/4)
        b, a, _ = self.run_recipe('loraga', gamma=16)
        c2 = 8**.5 / 16
        torch.testing.assert_close(b.T@b, torch.eye(2)*c2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a@a.T, torch.eye(2)*c2, atol=1e-6, rtol=1e-5)
        u, _, vh = torch.linalg.svd(self.c['weight_grad'], full_matrices=False)
        torch.testing.assert_close(b@b.T/c2, u[:, 2:4]@u[:, 2:4].T)
        torch.testing.assert_close(a.T@a/c2, vh[:2].T@vh[:2])

    def test_filet_matches_author_projected_diagonal_formula(self):
        b, a, _ = self.run_recipe('filet')
        v = torch.nn.functional.normalize(self.w.T@self.w, dim=0)
        u = torch.nn.functional.normalize(self.w@v, dim=0)
        score = (v.T@self.c['input_cov']@v).diag() * (u.T@self.c['output_grad_cov']@u).diag()
        ids = torch.topk(-score, 2).indices
        vals = score[ids].clamp_min(1e-8).sqrt()
        torch.testing.assert_close(b@a, (u[:,ids]*vals)@(vals[:,None]*v[:,ids].T))

    def test_canonical_signs_preserve_lorada_natural_gradient_projection(self):
        b, a, _ = self.run_recipe('lorada', gamma=1, variance_beta=0)
        sx = self.c['input_cov'] + 1e-6 * torch.eye(6)
        d = -torch.linalg.solve(sx, self.c['weight_grad'].T) / (self.c['output_fisher_diag']+1e-6)[None,:]
        d /= torch.linalg.svdvals(d)[0]
        u, _, _ = torch.linalg.svd(d, full_matrices=False)
        torch.testing.assert_close(b@a, d.T@u[:,:2]@u[:,:2].T, atol=2e-5, rtol=2e-5)

    def test_initialization_prefit_reduces_weight_error(self):
        _, _, meta = self.run_recipe('inilora', prefit_steps=30, initial_std=.05, prefit_lr=.02)
        self.assertLess(meta['final_mse'], meta['initial_mse'])

    def test_cache_reuse_and_integrity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            b, a, first = self.run_recipe('pissa', decomposition_cache_dir=directory)
            _, _, second = self.run_recipe('milora', decomposition_cache_dir=directory)
            self.assertFalse(first['decomposition_events'][0]['hit'])
            self.assertTrue(second['decomposition_events'][0]['hit'])
            self.assertEqual(len(list(Path(directory).glob('*.pt'))), 1)
            cache = next(Path(directory).glob('*.pt'))
            values = torch.load(cache, weights_only=True)
            values['values'][0][0, 0] += 1
            torch.save(values, cache)
            with self.assertRaises(ValueError):
                self.run_recipe('pissa', decomposition_cache_dir=directory)

    def test_backend_seed_independent_recipe_lazy_calibration_and_cpu_bf16_parity(self):
        from experiments import init_ablations as ab
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stats.pt'
            torch.save(self.c, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            calibration = {'module_files': {'0': str(path)}, 'module_sha256': {'0': digest},
                           'metadata': {'split': 'train', 'factor_cache_dir': directory+'/factors',
                                        'decomposition_cache_dir': directory+'/decompositions'}}
            for method in ['pissa', 'eva', 'loraga', 'lorada', 'filet']:
                first = backend({'method': method}, 42, calibration)
                second = backend({'method': method}, 43, calibration)
                self.assertEqual(first.recipe_manifest(method), second.recipe_manifest(method))
                base = torch.nn.Sequential(torch.nn.Linear(6, 8, dtype=torch.bfloat16))
                torch.manual_seed(55)
                reference = copy.deepcopy(base)
                ab.inject_adapters(reference, ab.InitSpec(), rank=2)
                torch.manual_seed(55)
                actual = copy.deepcopy(base)
                first.inject_adapters(actual, method, rank=2)
                self.assertTrue(torch.equal(actual[0].m, reference[0].m))
                x = torch.randn(4, 6)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    result, expected = actual(x), reference(x)
                self.assertTrue(torch.equal(result, expected), method)
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(actual[0].lora_B.grad).all())
                self.assertTrue(torch.isfinite(actual[0].lora_A.grad).all())
            calibration['module_sha256']['0'] = 'incorrect'
            broken = backend({'method': 'eva'}, 42, calibration)
            with self.assertRaises(ValueError):
                broken.inject_adapters(torch.nn.Sequential(torch.nn.Linear(6,8)), 'bad', 2)

    def test_in_memory_nested_statistics_bind_and_initialize(self):
        calibration = {'modules': {'0': self.c}, 'metadata': {'split': 'train'}}
        bound = backend({'method': 'cntlora'}, calibration=calibration)
        model = torch.nn.Sequential(torch.nn.Linear(6,8))
        bound.inject_adapters(model, 'cntlora', rank=2)
        self.assertTrue(model[0].compensated)

    def test_cnt_target_cache_preserves_formula_and_reuses_balances_and_seeds(self):
        for mode in ['cross', 'self', 'shift']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                b, a, _ = self.run_recipe('cntlora', mode=mode, seed=42)
                kwargs = {'mode': mode, 'decomposition_cache_dir': directory}
                bc, ac, first = self.run_recipe('cntlora', seed=42, **kwargs)
                torch.testing.assert_close(bc, b, rtol=0, atol=0)
                torch.testing.assert_close(ac, a, rtol=0, atol=0)
                self.assertFalse(first['cnt_target_cache_hit'])
                self.assertGreaterEqual(first['cnt_target_source_seconds'], 0)
                with patch('torch.linalg.pinv', side_effect=AssertionError('cache must skip pinv')):
                    bs, ass, seeded = self.run_recipe('cntlora', seed=999, **kwargs)
                    bb, ab, balanced = self.run_recipe('cntlora', seed=101, balance_gain=2, **kwargs)
                    bg, ag, gained = self.run_recipe('cntlora', seed=102, gain=2, **kwargs)
                for meta in [seeded, balanced, gained]:
                    self.assertTrue(meta['cnt_target_cache_hit'])
                    self.assertEqual(meta['cnt_target_source_seconds'], first['cnt_target_source_seconds'])
                # CNT uses deterministic solvers, so only the seed is invariant;
                # balance changes factors while gain also changes their product.
                torch.testing.assert_close(bs, b, rtol=0, atol=0)
                torch.testing.assert_close(ass, a, rtol=0, atol=0)
                torch.testing.assert_close(bb, b*2, rtol=0, atol=0)
                torch.testing.assert_close(ab, a/2, rtol=0, atol=0)
                torch.testing.assert_close(bg, b*2, rtol=0, atol=0)
                torch.testing.assert_close(ag, a*2, rtol=0, atol=0)
                self.assertFalse(torch.equal(bb, b))
                caches = list((Path(directory)/'cntlora_target').glob('*.pt'))
                self.assertEqual(len(caches), 1)
                saved = torch.load(caches[0], weights_only=True)
                saved['estimated'][0, 0] += 1
                torch.save(saved, caches[0])
                with self.assertRaisesRegex(ValueError, 'target cache integrity'):
                    self.run_recipe('cntlora', **kwargs)

    def test_cnt_target_cache_binds_grams_weight_solver_and_source(self):
        import experiments.init_data as data
        with tempfile.TemporaryDirectory() as directory:
            kwargs = {'method': 'cntlora', 'cnt_target_cache_dir': directory}
            initialize_factors(self.w, 2, kwargs, self.c)
            variants = [{'mode': 'self'}, {'mode': 'shift'}, {'damping': .01}, {'pinv_rtol': 1e-4}]
            for change in variants:
                _, _, meta = initialize_factors(self.w, 2, dict(kwargs, **change), self.c)
                self.assertFalse(meta['cnt_target_cache_hit'])
            _, _, meta = initialize_factors(self.w*2, 2, kwargs, self.c)
            self.assertFalse(meta['cnt_target_cache_hit'])
            changed = copy.deepcopy(self.c)
            changed['input_gram_batches'][0] *= 2
            _, _, meta = initialize_factors(self.w, 2, kwargs, changed)
            self.assertFalse(meta['cnt_target_cache_hit'])
            for target, value in [('torch.get_num_threads', torch.get_num_threads()+1),
                                  ('experiments.init_data._sha_file', 'changed-source')]:
                with patch(target, return_value=value):
                    _, _, meta = initialize_factors(self.w, 2, kwargs, self.c)
                    self.assertFalse(meta['cnt_target_cache_hit'])
            with patch.object(data.torch, '__version__', 'changed-torch'):
                _, _, meta = initialize_factors(self.w, 2, kwargs, self.c)
                self.assertFalse(meta['cnt_target_cache_hit'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_inilora_cuda_prefit_preserves_rng_and_returns_cpu(self):
        cpu_rng=torch.random.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state().clone()
        b,a,meta=self.run_recipe('inilora',prefit_steps=30,prefit_device='cuda',initial_std=.05,prefit_lr=.02)
        self.assertEqual(b.device.type,'cpu')
        self.assertEqual(a.dtype,torch.float32)
        self.assertTrue(torch.equal(cpu_rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng,torch.cuda.get_rng_state()))
        self.assertLess(meta['final_mse'],meta['initial_mse'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_inilora_prefit_disables_tf32_and_restores_task_policy(self):
        original=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            _,_,meta=self.run_recipe('inilora',prefit_steps=3,prefit_device='cuda',initial_std=.05)
            self.assertFalse(meta['prefit_allow_tf32'])
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32=original

    def test_prefit_cache_reuses_balance_variants_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs={'prefit_steps':5,'prefit_cache_dir':directory,'initial_std':.05}
            b,a,m=self.run_recipe('inilora',**kwargs)
            b2,a2,m2=self.run_recipe('inilora',balance_gain=2,**kwargs)
            self.assertFalse(m['prefit_cache_hit'])
            self.assertTrue(m2['prefit_cache_hit'])
            torch.testing.assert_close(b2,b*2,rtol=0,atol=0)
            torch.testing.assert_close(a2,a/2,rtol=0,atol=0)
            cache=next(Path(directory).glob('*.pt'))
            data=torch.load(cache,weights_only=True);data['up'][0,0]+=1;torch.save(data,cache)
            with self.assertRaises(ValueError):self.run_recipe('inilora',**kwargs)

    def test_weight_only_backend_does_not_read_irrelevant_calibration_files(self):
        calibration={'module_files':{'0':'/deliberately/nonexistent.pt'},
                     'module_sha256':{'0':'missing'},'metadata':{'split':'train'}}
        bound=backend({'method':'pissa'},calibration=calibration)
        model=torch.nn.Sequential(torch.nn.Linear(6,8))
        bound.inject_adapters(model,'pissa',rank=2)
        self.assertTrue(model[0].compensated)

    def test_missing_and_invalid_statistics_fail_loudly(self):
        with self.assertRaises(ValueError):
            initialize_factors(self.w, 2, {'method':'eva'})
        with self.assertRaises(ValueError):
            initialize_factors(self.w, 4, {'method':'loraga'}, self.c)
        with self.assertRaises(ValueError):
            self.run_recipe('lorada', population_size=0)
        with self.assertRaises(ValueError):
            self.run_recipe('pissa', factor_power=2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
