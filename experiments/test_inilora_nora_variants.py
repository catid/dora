import unittest

import torch

from experiments import init_ablations as ab
from experiments import inilora_nora_variants as variants


class PureLoRAVariantTests(unittest.TestCase):
    def spec(self, scale=1.0):
        return ab.InitSpec(mode='controlled', family='normal',
                           both_nonzero=True, scaling='absolute',
                           scale=scale, companion_scale=scale)

    def make(self, mode):
        torch.manual_seed(123)
        base = torch.nn.Linear(7, 5, bias=True)
        model = variants.inject_adapters(base, self.spec(), rank=3,
                                         factor_seed=42, mode=mode)
        return model, next(m for m in model.modules()
                           if isinstance(m, variants.AblationLinear)), base

    def test_all_variants_preserve_start_and_have_no_dora_magnitude(self):
        x = torch.randn(4, 7)
        for mode in variants.MODES:
            model, layer, base = self.make(mode)
            self.assertNotIn('m', dict(model.named_parameters()))
            torch.testing.assert_close(
                model(x), torch.nn.functional.linear(x, base.weight, base.bias),
                rtol=0, atol=0)

    def test_anchored_products_match_initial_factors(self):
        for mode in ('down_anchor', 'down_learned', 'up_anchor', 'both_anchor'):
            _, layer, _ = self.make(mode)
            with torch.no_grad():
                up = layer._up(layer.initial_up)
                down = layer._down(layer.initial_down)
                torch.testing.assert_close(up, layer.initial_up, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(down, layer.initial_down, rtol=1e-6, atol=1e-6)

    def test_learned_gain_is_optimizer_visible(self):
        model, _, _ = self.make('down_learned')
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any(name.endswith('log_gain') for name in names))
        groups = variants.parameter_groups(model, 1e-3)
        self.assertEqual(sum(g['parameter_count'] for g in groups),
                         sum(p.numel() for p in model.parameters() if p.requires_grad))

    def test_learned_gain_keeps_frozen_compensation_centered(self):
        _, layer, _ = self.make('down_learned')
        with torch.no_grad():
            layer.log_gain.add_(0.7)
        # Changing the live gain must not move the initial compensation.
        torch.testing.assert_close(
            layer._up(layer.initial_up) @ layer._down(layer.initial_down, initial=True),
            layer.initial_up @ (layer.initial_down *
                               (layer.initial_live_down_gain /
                                layer.initial_down_gain)),
            rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
