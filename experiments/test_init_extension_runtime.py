"""Independent state, cache and accepted-forward checks for extension wrappers."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from experiments import init_extension as formulas
from experiments import init_extension_runtime as runtime
from experiments import next_ablations as accepted


class ExtensionRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model=torch.nn.Sequential(torch.nn.Linear(7,5),torch.nn.Tanh(),torch.nn.Linear(5,4))

    def candidate(self,kind):
        family=formulas.extension_families()[0 if kind=='null' else 1]
        return {'candidate_id':kind,'spec':{'kind':family['kind'],'config':family['configs'][family['source_default_index']]}}

    def calibration(self,root):
        return {'modules':{'0':{'corda_input_cov':torch.diag(torch.arange(1,8,dtype=torch.float32))},
                           '2':{'corda_input_cov':torch.diag(torch.arange(1,6,dtype=torch.float32))}},
                'metadata':{'extension_factor_cache_dir':str(root),
                            'extension_gme_binding':{'task':'tiny_cpu_engineering','collector_sha':'fixture','train_ids':['a']}}}

    def test_null_centered_accepted_rng_forward_gradient_and_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate=self.candidate('null');cal=self.calibration(Path(temporary));base=copy.deepcopy(self.model)
            torch.manual_seed(41);reference=accepted.inject_adapters(copy.deepcopy(base),'next_base',rank=2)
            wanted_rng=torch.get_rng_state();torch.manual_seed(41)
            backend=runtime.backend_for(candidate,42,cal);model=backend.inject_adapters(copy.deepcopy(base),'null',rank=2)
            self.assertTrue(torch.equal(torch.get_rng_state(),wanted_rng))
            x=torch.randn(3,7);torch.testing.assert_close(model(x),reference(x),rtol=0,atol=0)
            model(x).square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:p.add_(p.grad,alpha=-.01)
            checkpoint={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
            reload=runtime.backend_for(candidate,42,cal).inject_adapters(copy.deepcopy(base),'null',rank=2)
            reload.load_state_dict(checkpoint,strict=False);torch.testing.assert_close(model(x),reload(x),rtol=0,atol=0)
            rows=runtime.backend_for(candidate,42,cal).assert_no_clamp(reload)
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['initializer']['callback_metadata']['factor_cache_hit'] for r in rows.values()))

    def callback(self,calibration,seen):
        def prepare(model,modules,config,seed):
            original_modes={m:m.training for m in model.modules()};model.eval()
            def observe(factors,stage):
                seen.append(stage)
                with runtime.temporary_factors(modules,factors):
                    value=model(torch.ones(4,7))
                    self.assertTrue(torch.isfinite(value).all())
                delta={'baseline':0.,'probe_scale':.001,'probe_skew':.003}.get(stage,.002)
                return {0:{'var':1.+delta,'skew':.1+delta,'count':20},
                        1:{'var':2.+delta,'skew':.2+2*delta,'count':16}}
            options={k:v for k,v in config.items() if k not in ('method','adapter_scale','magnitude_policy','moment_policy')}
            try:return formulas.initialize_gme({n:m.weight for n,m in modules.items()},2,{'0':0,'2':1},observe,seed=seed,**options)
            finally:
                for m,mode in original_modes.items():m.training=mode
        prepare.binding=calibration['metadata']['extension_gme_binding']
        return prepare

    def test_gme_full_callback_cache_preserves_weights_magnitude_rng_and_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate=self.candidate('gme');cal=self.calibration(Path(temporary));seen=[]
            torch.manual_seed(41);reference=accepted.inject_adapters(copy.deepcopy(self.model),'next_base',rank=2)
            wanted_rng=torch.get_rng_state();torch.manual_seed(41)
            backend=runtime.backend_for(candidate,42,cal,gme_prepare=self.callback(cal,seen))
            model=backend.inject_adapters(copy.deepcopy(self.model),'gme',rank=2)
            self.assertTrue(torch.equal(torch.get_rng_state(),wanted_rng))
            self.assertEqual(seen,['baseline','probe_scale','probe_skew','refine_0','refine_1','final'])
            for n,layer in model.named_modules():
                if isinstance(layer,accepted.AblationLinear):
                    expected=reference.get_submodule(n)
                    self.assertTrue(torch.equal(layer.weight,expected.weight));self.assertTrue(torch.equal(layer.m,expected.m))
                    self.assertEqual(layer._norm_evaluations,expected._norm_evaluations)
                    self.assertNotIn('forward',layer.__dict__)
            # Cache replay requires no callback and recreates identical factors.
            replay=runtime.backend_for(candidate,42,cal).inject_adapters(copy.deepcopy(self.model),'gme',rank=2)
            for (n,p),(rn,rp) in zip(model.named_parameters(),replay.named_parameters()):
                self.assertEqual(n,rn);torch.testing.assert_close(p,rp,rtol=0,atol=0)
            x=torch.randn(3,7);model(x).square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
            self.assertEqual(len(runtime.backend_for(candidate,42,cal).assert_no_clamp(replay)),2)

    def test_temporary_probe_restores_all_state_after_failure(self):
        model=accepted.inject_adapters(copy.deepcopy(self.model),'next_base',rank=2)
        modules={n:m for n,m in model.named_modules() if isinstance(m,accepted.AblationLinear)}
        before={n:p.clone() for n,p in model.named_parameters()};counts={n:m._norm_evaluations for n,m in modules.items()}
        expected=model(torch.ones(2,7)).detach()
        counts={n:m._norm_evaluations for n,m in modules.items()}
        with self.assertRaisesRegex(RuntimeError,'test callback'):
            with runtime.temporary_factors(modules,None):
                torch.testing.assert_close(model(torch.ones(2,7)),expected,rtol=0,atol=0)
                with torch.no_grad():
                    for layer in modules.values():layer.m.zero_();layer.lora_A.fill_(9)
                raise RuntimeError('test callback failure')
        for n,p in model.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        self.assertEqual({n:m._norm_evaluations for n,m in modules.items()},counts)
        self.assertTrue(all('forward' not in m.__dict__ for m in modules.values()))

    def test_bfloat16_baseline_keeps_accepted_normalization_rounding(self):
        base=torch.nn.Sequential(torch.nn.Linear(7,5,bias=False,dtype=torch.bfloat16))
        model=accepted.inject_adapters(copy.deepcopy(base),'next_base',rank=2)
        modules={'0':model[0]};x=torch.randn(3,7,dtype=torch.bfloat16)
        expected=model(x).detach()
        with torch.no_grad():model[0].lora_A.fill_(.1)
        changed=model[0].lora_A.clone()
        with runtime.temporary_factors(modules,None):
            torch.testing.assert_close(model(x),expected,rtol=0,atol=0)
        torch.testing.assert_close(model[0].lora_A,changed,rtol=0,atol=0)

    def test_gme_missing_observer_or_wrong_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate=self.candidate('gme');cal=self.calibration(Path(temporary))
            with self.assertRaisesRegex(ValueError,'requires task gme_prepare'):
                runtime.backend_for(candidate,42,cal).inject_adapters(copy.deepcopy(self.model),'gme',2)
            callback=self.callback(cal,[]);callback.binding={'wrong':True}
            with self.assertRaisesRegex(ValueError,'provenance differs'):
                runtime.backend_for(candidate,42,cal,gme_prepare=callback).inject_adapters(copy.deepcopy(self.model),'gme',2)

    def test_cache_tensor_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate=self.candidate('gme');cal=self.calibration(Path(temporary))
            runtime.backend_for(candidate,42,cal,gme_prepare=self.callback(cal,[])).inject_adapters(copy.deepcopy(self.model),'gme',2)
            path=next(Path(temporary).glob('*.pt'));payload=torch.load(path,weights_only=True)
            payload['factors']['0'][0].add_(1);torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                runtime.backend_for(candidate,42,cal).inject_adapters(copy.deepcopy(self.model),'gme',2)


if __name__=='__main__':unittest.main()
