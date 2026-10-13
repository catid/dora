import unittest

import torch

from experiments import init_ablations as ab
from experiments import inilora_nora_lora as pure


class PureLoRAIniLoRANoRATest(unittest.TestCase):
    def spec(self, scale=1.0, both=True):
        return ab.InitSpec(mode='controlled', family='normal',
                           both_nonzero=both, scaling='absolute',
                           scale=scale, companion_scale=scale)

    def make(self, *, nora, scale=1.0, both=True):
        torch.manual_seed(123)
        base = torch.nn.Linear(7, 5, bias=True)
        model = pure.inject_adapters(base, self.spec(scale, both), rank=3,
                                     factor_seed=42, nora=nora)
        layer = next(m for m in model.modules()
                     if isinstance(m, pure.AblationLinear))
        return model, layer, base

    def test_centered_start_matches_base_and_has_no_m(self):
        x = torch.randn(4, 7)
        for nora in (False, True):
            model, layer, base = self.make(nora=nora)
            self.assertNotIn('m', dict(model.named_parameters()))
            torch.testing.assert_close(model(x),
                                       torch.nn.functional.linear(x, base.weight,
                                                                   base.bias),
                                       rtol=0, atol=0)
            self.assertTrue(layer.norm_diagnostics()['dora_enabled'] is False)
        recipe = pure.recipe_manifest(self.spec(1.0, True), 'test', nora=True)
        self.assertEqual(recipe['trainable_registration_order'],
                         ['lora_B', 'lora_A'])

    def test_nora_normalizes_initial_and_live_products(self):
        model, layer, _ = self.make(nora=True)
        with torch.no_grad():
            live = torch.nn.functional.normalize(layer.lora_A, dim=0, eps=1e-12)
            initial = torch.nn.functional.normalize(layer.initial_down, dim=0,
                                                     eps=1e-12)
            self.assertTrue(torch.equal(live, initial))
            self.assertEqual(layer.norm_diagnostics()['nora_initial_zero_columns'],
                             0)

    def test_scale_is_linear_after_down_normalization(self):
        # With both factors multiplied by g, plain centered LoRA changes as
        # g^2, while NoRA removes the down-factor radial scale and changes as g.
        torch.manual_seed(1)
        base = torch.nn.Linear(7, 5, bias=False)
        one = pure.inject_adapters(base, self.spec(1.0), rank=3,
                                   factor_seed=42, nora=True)
        two = pure.inject_adapters(base, self.spec(2.0), rank=3,
                                   factor_seed=42, nora=True)
        a = next(m for m in one.modules() if isinstance(m, pure.AblationLinear))
        b = next(m for m in two.modules() if isinstance(m, pure.AblationLinear))
        with torch.no_grad():
            one_update = a.lora_B @ torch.nn.functional.normalize(a.lora_A, dim=0)
            one_update -= a.initial_up @ torch.nn.functional.normalize(a.initial_down, dim=0)
            two_update = b.lora_B @ torch.nn.functional.normalize(b.lora_A, dim=0)
            two_update -= b.initial_up @ torch.nn.functional.normalize(b.initial_down, dim=0)
        torch.testing.assert_close(two_update, 2 * one_update, rtol=1e-6, atol=1e-6)

    def test_zero_down_nora_baseline_is_finite(self):
        model, layer, base = self.make(nora=True, both=False)
        x = torch.randn(2, 7)
        y = model(x)
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(layer.norm_diagnostics()['nora_initial_zero_columns'], 7)
        self.assertEqual(layer.norm_diagnostics()['nora_initial_epsilon_columns'], 7)


if __name__ == '__main__':
    unittest.main()
