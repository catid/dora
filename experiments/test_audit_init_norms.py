from types import SimpleNamespace
import unittest

import torch

from experiments.audit_init_norms import audit_initial_norm, forward_denominator_minimum


class NormAuditTests(unittest.TestCase):
    def module(self):
        weight = torch.tensor([[0., 1.], [0., 2.]])
        return SimpleNamespace(weight=weight, m=weight.norm(dim=0, keepdim=True),
                               lora_B=torch.tensor([[1.], [2.]]),
                               lora_A=torch.tensor([[.1, 0.]]))

    def spec(self, **config):
        return {'kind': 'quantized', 'config': {
            'method': 'loftq', 'magnitude_policy': 'quantized', **config}}

    def test_zero_magnitude_with_positive_denominator_is_valid(self):
        module = self.module()
        before = {k: v.clone() for k, v in vars(module).items()}
        evidence = audit_initial_norm({'minimum_raw_norm': 0.}, module, self.spec())
        self.assertTrue(evidence['zero_explained_by_quantized_magnitude_initialization'])
        self.assertEqual(evidence['zero_initial_magnitude_columns'], 1)
        self.assertGreater(evidence['minimum_initial_forward_denominator'], 0)
        for k, v in before.items():
            self.assertTrue(torch.equal(v, getattr(module, k)))

    def test_zero_quantizer_only_denominator_is_rejected(self):
        module = self.module(); module.lora_A.zero_()
        with self.assertRaisesRegex(AssertionError, 'forward denominator'):
            audit_initial_norm({'minimum_raw_norm': 0.}, module,
                               self.spec(method='quantizer_only'))

    def test_adapter_cancellation_is_rejected(self):
        module = self.module(); module.lora_A[0, 1] = -1
        with self.assertRaisesRegex(AssertionError, 'forward denominator'):
            forward_denominator_minimum(module)

    def test_unexplained_zero_is_rejected(self):
        module = self.module(); module.weight[:, 0] = 1
        module.m = module.weight.norm(dim=0, keepdim=True)
        with self.assertRaises(AssertionError):
            audit_initial_norm({'minimum_raw_norm': 0.}, module, self.spec())

    def test_original_magnitude_policy_cannot_explain_zero(self):
        with self.assertRaises(AssertionError):
            audit_initial_norm({'minimum_raw_norm': 0.}, self.module(),
                               self.spec(magnitude_policy='original'))

    def test_magnitude_mismatch_is_rejected(self):
        module = self.module(); module.m.fill_(1)
        with self.assertRaisesRegex(AssertionError, 'Magnitude initialization'):
            audit_initial_norm({'minimum_raw_norm': 0.}, module, self.spec())

    def test_nonfinite_and_negative_diagnostics_are_rejected(self):
        for raw in [-1., float('nan'), float('inf')]:
            with self.subTest(raw=raw), self.assertRaises(AssertionError):
                audit_initial_norm({'minimum_raw_norm': raw}, self.module(), self.spec())

    def test_nonfinite_denominator_is_rejected(self):
        module = self.module(); module.lora_A.fill_(float('nan'))
        with self.assertRaisesRegex(AssertionError, 'forward denominator'):
            forward_denominator_minimum(module)

    def test_centered_update_uses_compensation(self):
        module = self.module()
        module.weight[:, 0] = .5
        module.compensated = True
        module.initial_up = module.lora_B.clone()
        module.initial_down = module.lora_A.clone()
        self.assertEqual(forward_denominator_minimum(module), float(module.weight.norm(dim=0).min()))
        module.weight[:, 0] = 0
        with self.assertRaisesRegex(AssertionError, 'forward denominator'):
            forward_denominator_minimum(module)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_matches_actual_bf16_runtime_denominator(self):
        from experiments.next_ablations import AblationLinear
        base = torch.nn.Linear(2, 2, bias=False, device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            base.weight.copy_(self.module().weight)
        layer = AblationLinear(base, 'next_base', rank=1)
        with torch.no_grad():
            layer.lora_B.copy_(self.module().lora_B)
            layer.lora_A.copy_(self.module().lora_A)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            expected = float(layer._weight_norm().detach().min())
            self.assertTrue(torch.isfinite(layer(torch.ones(1, 2, device='cuda'))).all())
        self.assertEqual(float(layer._minimum_raw_norm), 0.)
        self.assertEqual(forward_denominator_minimum(layer), expected)
        audit_initial_norm({'minimum_raw_norm': 0.}, layer, self.spec())


if __name__ == '__main__':
    unittest.main()
