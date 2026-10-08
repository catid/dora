"""Independent numerical and conversion regressions for the DoRA layer.

Run with ``python -m unittest -v test_dora``.  CUDA checks skip automatically
when a CUDA device is unavailable; all mathematical reference tests use CPU.
"""

import importlib.util
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

import dora
from dora import DoRALayer, replace_linear_with_dora


class DoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def assert_close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, **kwargs)

    def assert_parameter_dtypes(self, layer, dtype):
        for parameter in (layer.weight, layer.lora_A, layer.lora_B, layer.bias):
            if parameter is not None:
                self.assertEqual(parameter.dtype, dtype)
        magnitude_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        self.assertEqual(layer.m.dtype, magnitude_dtype)

    def test_import_does_not_reset_random_generator(self):
        before = torch.random.get_rng_state().clone()
        spec = importlib.util.spec_from_file_location("_dora_import_test", dora.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_initial_conversion_preserves_linear_for_all_input_shapes(self):
        for bias in (False, True):
            for out_features in (1, 3):
                with self.subTest(bias=bias, out_features=out_features):
                    base = nn.Linear(5, out_features, bias=bias, dtype=torch.float64)
                    layer = DoRALayer.from_linear(base, rank=2)
                    self.assertEqual(layer.m.shape, (out_features, 1))
                    self.assert_close(layer.m, base.weight.norm(dim=1, keepdim=True))
                    for shape in ((5,), (7, 5), (2, 3, 5), (0, 5)):
                        x = torch.randn(shape, dtype=torch.float64)
                        self.assert_close(layer(x), base(x), rtol=1e-12, atol=1e-12)
                    self.assertEqual(layer.bias is None, not bias)

    def test_forward_and_gradients_match_detached_dense_reference(self):
        # A non-square matrix and nonzero updates distinguish the output-row
        # normalization from the original implementation's input-column norm.
        for bias in (None, torch.tensor([0.3, -1.1, 0.7], dtype=torch.float64)):
            with self.subTest(bias=bias is not None):
                weight = torch.randn(3, 5, dtype=torch.float64)
                layer = DoRALayer(5, 3, rank=2, weight=weight, bias=bias)
                with torch.no_grad():
                    layer.lora_A.normal_(0.0, 0.4)
                    layer.lora_B.normal_(0.0, 0.3)
                    layer.m.mul_(torch.tensor([[0.7], [1.2], [1.8]]))
                x = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                ar = layer.lora_A.detach().clone().requires_grad_()
                br = layer.lora_B.detach().clone().requires_grad_()
                mr = layer.m.detach().clone().requires_grad_()
                adapted = weight + ar @ br
                denominator = adapted.norm(dim=1, keepdim=True).detach()
                expected = F.linear(xr, mr * adapted / denominator, bias)
                actual = layer(x)
                self.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
                probe = torch.randn_like(expected)
                actual_loss = (actual * probe).sum() + actual.square().mean()
                expected_loss = (expected * probe).sum() + expected.square().mean()
                actual_loss.backward()
                expected_loss.backward()
                for name, actual_grad, reference_grad in (
                    ("input", x.grad, xr.grad),
                    ("A", layer.lora_A.grad, ar.grad),
                    ("B", layer.lora_B.grad, br.grad),
                    ("magnitude", layer.m.grad, mr.grad),
                ):
                    with self.subTest(gradient=name):
                        self.assert_close(actual_grad, reference_grad, rtol=1e-10, atol=1e-10)
                self.assertIsNone(layer.weight.grad)
                if layer.bias is not None:
                    self.assertIsNone(layer.bias.grad)

    def test_bias_is_not_scaled_by_magnitude(self):
        bias = torch.tensor([3.0, -2.0])
        layer = DoRALayer(4, 2, rank=1, weight=torch.ones(2, 4), bias=bias)
        with torch.no_grad():
            layer.m.mul_(7)
        self.assert_close(layer(torch.zeros(5, 4)), bias.expand(5, 2))

    def test_zero_weight_rows_have_finite_outputs_and_gradients(self):
        weight = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
        layer = DoRALayer(3, 2, rank=2, weight=weight)
        x = torch.randn(4, 3, requires_grad=True)
        self.assert_close(layer(x), F.linear(x, weight))
        layer(x).square().sum().backward()
        for value in (x.grad, layer.m.grad, layer.lora_A.grad, layer.lora_B.grad):
            self.assertTrue(torch.isfinite(value).all().item())

    def test_zero_weight_layer_can_learn_nonzero_outputs(self):
        layer = DoRALayer(3, 1, rank=1, weight=torch.zeros(1, 3))
        x, target = torch.ones(4, 3), torch.ones(4, 1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
        initial_loss = F.mse_loss(layer(x), target).item()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(layer(x), target)
            loss.backward()
            self.assertTrue(torch.isfinite(layer.lora_B.grad).all().item())
            optimizer.step()
        self.assertLess(F.mse_loss(layer(x), target).item(), initial_loss * 0.9)

    def test_small_nonzero_weights_preserve_initial_linear_output(self):
        # A fixed epsilon such as 1e-12 must not rescale untouched small rows.
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                weight = torch.tensor([[1.0, -2.0, 3.0]], dtype=dtype) * 1e-20
                layer = DoRALayer(3, 1, rank=1, weight=weight)
                x = torch.ones(2, 3, dtype=dtype)
                self.assert_close(layer(x), F.linear(x, weight), atol=0, rtol=1e-6)

    def test_frozen_base_is_independent_of_source_storage(self):
        original = nn.Linear(5, 3, dtype=torch.float64)
        layer = DoRALayer.from_linear(original, rank=2)
        saved_weight, saved_bias = original.weight.detach().clone(), original.bias.detach().clone()
        self.assertNotEqual(layer.weight.data_ptr(), original.weight.data_ptr())
        self.assertNotEqual(layer.bias.data_ptr(), original.bias.data_ptr())
        self.assertFalse(layer.weight.requires_grad)
        self.assertFalse(layer.bias.requires_grad)
        self.assertTrue(all(p.requires_grad for p in (layer.m, layer.lora_A, layer.lora_B)))
        with torch.no_grad():
            original.weight.add_(10)
            original.bias.add_(20)
        self.assert_close(layer.weight, saved_weight)
        self.assert_close(layer.bias, saved_bias)

    def test_fresh_layer_has_initialized_weight_and_no_bias(self):
        layer = DoRALayer(7, 3, rank=2)
        self.assertIsNone(layer.bias)
        self.assertTrue(torch.isfinite(layer.weight).all().item())
        self.assertGreater(layer.weight.abs().sum().item(), 0)
        self.assertLessEqual(layer.weight.abs().max().item(), 1 / math.sqrt(7))
        x = torch.randn(4, 7)
        self.assert_close(layer(x), F.linear(x, layer.weight))

    def test_invalid_dimensions_rank_and_weight_are_rejected(self):
        cases = (
            {"d_in": 0}, {"d_out": 0}, {"d_in": -1}, {"d_out": -1},
            {"rank": 0}, {"rank": -2}, {"rank": 1.5}, {"rank": True},
            {"weight": torch.randn(2, 3)},
            {"weight": torch.ones(3, 5, dtype=torch.int64)},
            {"bias": torch.randn(4)},
        )
        for override in cases:
            with self.subTest(override=override):
                kwargs = {"d_in": 5, "d_out": 3, "rank": 2}
                kwargs.update(override)
                with self.assertRaises((ValueError, TypeError)):
                    DoRALayer(**kwargs)

    def test_rank_larger_than_dimensions_is_valid(self):
        layer = DoRALayer(2, 1, rank=4)
        self.assertEqual(layer(torch.randn(3, 2)).shape, (3, 1))

    def test_optimizer_updates_adapters_and_keeps_base_frozen(self):
        layer = DoRALayer.from_linear(nn.Linear(5, 3), rank=2)
        x, target = torch.randn(32, 5), torch.randn(32, 3)
        base_weight, base_bias = layer.weight.clone(), layer.bias.clone()
        initial_loss = F.mse_loss(layer(x), target).item()
        initial_magnitude = layer.m.detach().clone()
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(layer(x), target).backward()
            optimizer.step()
        self.assertLess(F.mse_loss(layer(x), target).item(), initial_loss * 0.9)
        self.assertFalse(torch.equal(initial_magnitude, layer.m))
        self.assertTrue(torch.equal(base_weight, layer.weight))
        self.assertTrue(torch.equal(base_bias, layer.bias))

    def test_merge_matches_adapted_forward_and_is_independent(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                layer = DoRALayer.from_linear(nn.Linear(5, 3, bias=bias, dtype=torch.float64), rank=2)
                layer.eval()
                with torch.no_grad():
                    layer.lora_B.normal_()
                    layer.m.mul_(1.3)
                x = torch.randn(2, 4, 5, dtype=torch.float64)
                expected = layer(x).detach()
                merged = layer.to_linear()
                self.assertIsInstance(merged, nn.Linear)
                self.assertFalse(merged.training)
                self.assertEqual(merged.weight.dtype, layer.weight.dtype)
                self.assertEqual(merged.weight.device, layer.weight.device)
                self.assertEqual(merged.bias is None, not bias)
                self.assertTrue(all(not p.requires_grad for p in merged.parameters()))
                self.assert_close(merged(x), expected, rtol=1e-11, atol=1e-11)
                with torch.no_grad():
                    layer.m.add_(1)
                    if layer.bias is not None:
                        layer.bias.add_(1)
                self.assert_close(merged(x), expected, rtol=1e-11, atol=1e-11)

    def test_replacement_handles_root_nested_and_shared_layers(self):
        linear = nn.Linear(5, 3, bias=False, dtype=torch.float64).eval()
        x = torch.randn(2, 5, dtype=torch.float64)
        expected = linear(x)
        root = replace_linear_with_dora(linear, rank=2)
        self.assertIsInstance(root, DoRALayer)
        self.assertFalse(root.training)
        self.assert_close(root(x), expected)
        model = nn.Module()
        model.first = linear
        model.alias = linear
        model.nested = nn.Sequential(nn.ReLU(), linear)
        model.nested_alias = model.nested
        returned = replace_linear_with_dora(model, rank=2)
        self.assertIs(returned, model)
        self.assertIs(model.first, model.alias)
        self.assertIs(model.first, model.nested[1])
        self.assertIs(model.nested, model.nested_alias)
        self.assertIsInstance(model.nested[0], nn.ReLU)
        self.assertIsInstance(model.first, DoRALayer)
        self.assertFalse(model.first.training)
        self.assertEqual(model.first.lora_A.shape[1], 2)
        self.assert_close(model.first(x), expected)
        self.assertIs(replace_linear_with_dora(model, rank=2).first, model.first)

    def test_state_dict_round_trip(self):
        original = DoRALayer(5, 3, rank=2)
        with torch.no_grad():
            original.lora_B.normal_()
            original.m.mul_(1.5)
        restored = DoRALayer(5, 3, rank=2)
        restored.load_state_dict(original.state_dict())
        x = torch.randn(2, 5)
        self.assert_close(restored(x), original(x))

    def test_cpu_dtype_is_preserved(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                original = nn.Linear(5, 3, dtype=dtype)
                layer = DoRALayer.from_linear(original, rank=2)
                self.assert_parameter_dtypes(layer, dtype)
                x = torch.randn(2, 5, dtype=dtype, requires_grad=True)
                output = layer(x)
                self.assertEqual(output.dtype, dtype)
                self.assert_close(output, original(x), rtol=0.02, atol=0.02)
                output.float().square().sum().backward()
                self.assertTrue(torch.isfinite(layer.lora_B.grad).all().item())

    def test_cpu_autocast_preserves_activation_dtype_and_finite_gradients(self):
        original = nn.Linear(5, 3)
        layer = DoRALayer.from_linear(original, rank=2)
        x = torch.randn(2, 5, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(x)
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assert_close(output, original(x), rtol=0.02, atol=0.02)
            loss = output.float().square().sum()
        loss.backward()
        for value in (x.grad, layer.m.grad, layer.lora_A.grad, layer.lora_B.grad):
            self.assertTrue(torch.isfinite(value).all().item())

    def test_half_large_row_magnitude_stays_finite(self):
        weight = torch.full((1, 4), 40000.0, dtype=torch.float16)
        layer = DoRALayer(4, 1, rank=1, weight=weight)
        x = torch.full((1, 4), 1e-5, dtype=torch.float16)
        self.assertEqual(layer.m.dtype, torch.float32)
        self.assert_close(layer.m, torch.tensor([[80000.0]]))
        self.assert_close(layer(x), F.linear(x, weight))
        self.assert_close(layer.to_linear()(x), layer(x))

    def test_half_large_scale_keeps_representable_output_finite(self):
        layer = DoRALayer(1, 1, rank=1, weight=torch.tensor([[1e-4]], dtype=torch.float16))
        with torch.no_grad():
            layer.m.fill_(10.0)
        x = torch.ones(1, 1, dtype=torch.float16)
        self.assert_close(layer(x), torch.tensor([[10.0]], dtype=torch.float16))
        self.assert_close(layer.to_linear()(x), layer(x))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_device_dtype_forward_and_backward(self):
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                original = nn.Linear(16, 7, device="cuda", dtype=dtype)
                layer = DoRALayer.from_linear(original, rank=3)
                self.assertTrue(all(p.device == original.weight.device for p in layer.parameters()))
                self.assert_parameter_dtypes(layer, dtype)
                x = torch.randn(2, 3, 16, device="cuda", dtype=dtype, requires_grad=True)
                output = layer(x)
                self.assertEqual(output.dtype, dtype)
                self.assert_close(output, original(x), rtol=0.02, atol=0.02)
                output.float().square().sum().backward()
                for value in (x.grad, layer.m.grad, layer.lora_A.grad, layer.lora_B.grad):
                    self.assertTrue(torch.isfinite(value).all().item())
                self.assert_close(layer.to_linear()(x), output, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
