"""Independent mathematical checks for the four task-comparison adapters."""

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.adapters import AdapterLinear, METHODS, inject_adapters, merge_adapters


class AdapterTests(unittest.TestCase):
    def test_shared_initialization_identity_and_parameter_counts(self):
        base = nn.Linear(7, 5, dtype=torch.float64)
        x = torch.randn(2, 3, 7, dtype=torch.float64)
        first_a = None
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(91)
                layer = AdapterLinear(copy.deepcopy(base), method, rank=3)
                torch.testing.assert_close(layer(x), base(x), rtol=1e-13, atol=1e-13)
                self.assertEqual(tuple(layer.lora_A.shape), (3, 7))
                self.assertEqual(tuple(layer.lora_B.shape), (5, 3))
                self.assertEqual(layer.lora_B.count_nonzero().item(), 0)
                count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                self.assertEqual(count, 3 * (7 + 5) + (5 if layer.use_dora else 0))
                if first_a is None:
                    first_a = layer.lora_A.detach().clone()
                else:
                    torch.testing.assert_close(layer.lora_A, first_a, rtol=0, atol=0)

    def test_dense_forward_and_backward_references(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(17)
                layer = AdapterLinear(nn.Linear(7, 5, dtype=torch.float64), method, rank=3)
                with torch.no_grad():
                    layer.lora_B.normal_(std=0.13)
                    if layer.m is not None:
                        layer.m.mul_(1.3)
                x = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
                a = layer.lora_A.detach().clone().requires_grad_()
                b = layer.lora_B.detach().clone().requires_grad_()
                ref_x = x.detach().clone().requires_grad_()
                down = a
                if method in ("nora", "dora_nora"):
                    # Independent equation 7: sum along rank, retaining gradient.
                    down = a / a.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
                adapted = layer.weight.detach() + b @ down
                refs = [ref_x, a, b]
                params = [x, layer.lora_A, layer.lora_B]
                if method in ("dora", "dora_nora"):
                    m = layer.m.detach().clone().requires_grad_()
                    denom = adapted.square().sum(dim=1, keepdim=True).sqrt().detach()
                    adapted = adapted * (m / denom)
                    refs.append(m)
                    params.append(layer.m)
                expected = ref_x @ adapted.T + layer.bias
                actual = layer(x)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                upstream = torch.randn_like(actual)
                actual_grads = torch.autograd.grad((actual * upstream).sum(), params)
                expected_grads = torch.autograd.grad((expected * upstream).sum(), refs)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)
                self.assertIsNone(layer.weight.grad)
                self.assertIsNone(layer.bias.grad)

    def test_nora_orientation_and_normalization_derivative(self):
        layer = AdapterLinear(nn.Linear(7, 5, dtype=torch.float64), "nora", rank=3)
        down = layer.effective_down()
        torch.testing.assert_close(down.square().sum(dim=0), torch.ones(7, dtype=torch.float64))
        grad = torch.autograd.grad((down * torch.randn_like(down)).sum(), layer.lora_A)[0]
        # The derivative of normalization projects onto each column's tangent
        # space. A detached denominator would fail this orthogonality check.
        torch.testing.assert_close((grad * layer.lora_A).sum(dim=0), torch.zeros(7, dtype=torch.float64),
                                   atol=1e-12, rtol=0)
        with torch.no_grad():
            layer.lora_A[:, 0].zero_()
        loss = layer.effective_down().sum()
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, layer.lora_A)[0]).all())

    def test_merge_includes_nonzero_update_and_nora_normalization(self):
        for method in METHODS:
            with self.subTest(method=method):
                layer = AdapterLinear(nn.Linear(7, 5, bias=False, dtype=torch.float64), method, rank=3).eval()
                with torch.no_grad():
                    layer.lora_B.normal_()
                x = torch.randn(2, 7, dtype=torch.float64)
                rng = torch.random.get_rng_state()
                merged = merge_adapters(layer)
                self.assertFalse(merged.training)
                self.assertIsNone(merged.bias)
                self.assertFalse(merged.weight.requires_grad)
                self.assertNotEqual(merged.weight.data_ptr(), layer.weight.data_ptr())
                torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
                torch.testing.assert_close(merged(x), layer(x), rtol=1e-12, atol=1e-12)

    def test_injection_targets_aliases_head_and_root(self):
        model = nn.Module()
        model.first_alias = nn.Linear(7, 5)
        model.selected_alias = model.first_alias
        model.head = nn.Linear(5, 2)
        result = inject_adapters(model, "dora+nora", targets=("selected_alias",))
        self.assertIs(result, model)
        self.assertIs(model.first_alias, model.selected_alias)
        self.assertIsInstance(model.first_alias, AdapterLinear)
        self.assertIsInstance(model.head, nn.Linear)
        self.assertFalse(model.head.weight.requires_grad)
        model.head.requires_grad_(True)
        self.assertTrue(model.head.weight.requires_grad)
        merge_adapters(model)
        self.assertIs(model.first_alias, model.selected_alias)
        root = inject_adapters(nn.Linear(3, 2), "lora", targets=lambda name, module: name == "")
        self.assertIsInstance(root, AdapterLinear)
        with self.assertRaises(ValueError):
            inject_adapters(nn.Linear(3, 2), "lora", targets=("missing",))

    def test_reduced_precision_base_keeps_fp32_adapters(self):
        for dtype in (torch.float16, torch.bfloat16):
            for method in METHODS:
                with self.subTest(dtype=dtype, method=method):
                    layer = AdapterLinear(nn.Linear(7, 5, dtype=dtype), method, rank=3)
                    self.assertEqual(layer.lora_A.dtype, torch.float32)
                    self.assertEqual(layer.lora_B.dtype, torch.float32)
                    if layer.m is not None:
                        self.assertEqual(layer.m.dtype, torch.float32)
                    x = torch.randn(3, 7, dtype=dtype)
                    out = layer(x)
                    self.assertEqual(out.dtype, dtype)
                    self.assertTrue(torch.isfinite(out).all())
                    out.float().square().mean().backward()
                    self.assertTrue(torch.isfinite(layer.lora_B.grad).all())
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        autocast_out = layer(x)
                    self.assertEqual(autocast_out.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(autocast_out).all())


if __name__ == "__main__":
    unittest.main()
