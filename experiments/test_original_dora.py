"""Parity with the archived historical source, not a rewritten reference."""

import ast
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import textwrap
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments import original_dora
from experiments.original_dora import OriginalDoRALinear


def historical_class():
    path = Path(__file__).resolve().parents[1] / original_dora.ORIGINAL_SOURCE_PATH
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != original_dora.ORIGINAL_SOURCE_SHA256:
        raise AssertionError("historical source hash mismatch")
    # Compile only the archived class: avoid its top-level manual_seed(0), demo,
    # and imports. Its actual constructor and forward remain untouched.
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "DoRALayer")
    namespace = {"torch": torch, "nn": nn, "F": F}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["DoRALayer"], node


class OriginalDoRATests(unittest.TestCase):
    def test_forward_syntax_matches_historical_source_and_import_preserves_rng(self):
        _, node = historical_class()
        expected = next(item for item in node.body if isinstance(item, ast.FunctionDef)
                        and item.name == "forward")
        actual = ast.parse(textwrap.dedent(inspect.getsource(OriginalDoRALinear.forward))).body[0]
        # The only changed forward operation is the final projection dispatch,
        # whose documented mixed-dtype boundary has a separate numerical test.
        self.assertEqual([ast.dump(item, include_attributes=False) for item in actual.body[:-1]],
                         [ast.dump(item, include_attributes=False) for item in expected.body[:-1]])
        torch.manual_seed(193)
        before = torch.random.get_rng_state()
        importlib.reload(original_dora)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)

    def test_original_initialization_rng_magnitude_dtype_and_freezing(self):
        historical, _ = historical_class()
        for dtype in (torch.float32, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                base = nn.Linear(7, 5, dtype=dtype).eval()
                torch.manual_seed(71)
                expected = historical(7, 5, rank=3, weight=base.weight.detach().clone(),
                                      bias=base.bias.detach().clone())
                expected_rng = torch.random.get_rng_state()
                torch.manual_seed(71)
                actual = OriginalDoRALinear(base, rank=3)
                torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
                for name in ("weight", "bias", "m", "lora_A", "lora_B"):
                    torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
                self.assertEqual(actual.m.dtype, dtype)
                self.assertEqual(actual.lora_A.dtype, torch.get_default_dtype())
                self.assertEqual(actual.lora_B.dtype, torch.get_default_dtype())
                self.assertEqual(tuple(actual.m.shape), (1, 7))
                self.assertEqual(tuple(actual.lora_A.shape), (5, 3))
                self.assertEqual(tuple(actual.lora_B.shape), (3, 7))
                self.assertFalse(actual.weight.requires_grad)
                self.assertFalse(actual.bias.requires_grad)
                self.assertFalse(actual.training)
                self.assertIsNone(actual.log_gain)
                self.assertEqual(sum(p.numel() for p in actual.parameters() if p.requires_grad), 43)

    def _parity(self, dtype, bias, autocast=False, device="cpu"):
        historical, _ = historical_class()
        torch.manual_seed(128)
        base = nn.Linear(7, 5, bias=bias, dtype=dtype)
        torch.manual_seed(71)
        # A zero historical bias supplies the independently equivalent biasless
        # reference without invoking the original uninitialized-bias fallback.
        expected = historical(7, 5, rank=3, weight=base.weight.detach().clone(),
                              bias=base.bias.detach().clone() if bias else torch.zeros(5, dtype=dtype))
        torch.manual_seed(71)
        actual = OriginalDoRALinear(base, rank=3)
        if dtype == torch.float64:
            # The original defaults to FP32 factors. Explicitly casting both
            # modules to FP64 permits a high-precision derivative comparison.
            actual.double()
            expected.double()
        actual.to(device)
        expected.to(device)
        with torch.no_grad():
            actual.lora_A.normal_(std=0.3)
            actual.lora_B.normal_(std=0.2)
            actual.m.mul_(torch.linspace(0.7, 1.3, 7, device=device).unsqueeze(0))
            expected.lora_A.copy_(actual.lora_A)
            expected.lora_B.copy_(actual.lora_B)
            expected.m.copy_(actual.m)
        x = torch.randn(2, 4, 7, dtype=dtype, device=device, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
            output, reference = actual(x), expected(ref_x)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        probe = torch.randn_like(output)
        actual_params = [x, actual.lora_A, actual.lora_B, actual.m]
        expected_params = [ref_x, expected.lora_A, expected.lora_B, expected.m]
        gradients = torch.autograd.grad((output * probe).sum(), actual_params)
        ref_gradients = torch.autograd.grad((reference * probe).sum(), expected_params)
        for grad, ref_grad in zip(gradients, ref_gradients):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 0)
            torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
        self.assertIsNone(actual.weight.grad)
        if bias:
            self.assertIsNone(actual.bias.grad)
        else:
            self.assertIsNone(actual.bias)

    def test_nonsquare_nonzero_forward_and_all_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for bias in (True, False):
                with self.subTest(dtype=dtype, bias=bias):
                    self._parity(dtype, bias)

    def test_bfloat16_autocast_matches_historical_source(self):
        for dtype in (torch.float32, torch.bfloat16):
            for bias in (True, False):
                with self.subTest(dtype=dtype, bias=bias):
                    self._parity(dtype, bias, autocast=True)

    def test_bfloat16_projection_boundary_without_autocast(self):
        historical, _ = historical_class()
        for bias in (True, False):
            with self.subTest(bias=bias):
                torch.manual_seed(71)
                base = nn.Linear(7, 5, bias=bias, dtype=torch.bfloat16)
                actual = OriginalDoRALinear(base, rank=3)
                with torch.no_grad():
                    actual.lora_B.normal_(std=0.2)
                    actual.m.mul_(1.25)
                expected = historical(7, 5, rank=3, weight=actual.weight.detach().clone(),
                                      bias=actual.bias.detach().clone() if bias else torch.zeros(5, dtype=torch.bfloat16))
                with torch.no_grad():
                    for name in ("lora_A", "lora_B", "m"):
                        getattr(expected, name).copy_(getattr(actual, name))
                x = torch.randn(2, 4, 7, dtype=torch.bfloat16, requires_grad=True)
                with self.assertRaises(RuntimeError):
                    expected(x)
                ref_x = x.detach().clone().requires_grad_()
                lora = torch.matmul(expected.lora_A, expected.lora_B)
                adapted = expected.weight + lora
                column_norm = adapted.norm(p=2, dim=0, keepdim=True)
                norm_adapted = adapted / column_norm
                calc_weights = expected.m * norm_adapted
                projected_bias = expected.bias.to(calc_weights.dtype) if bias else None
                reference = F.linear(ref_x.to(calc_weights.dtype), calc_weights, projected_bias).to(ref_x.dtype)
                output = actual(x)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(actual.m.dtype, torch.bfloat16)
                self.assertEqual(calc_weights.dtype, torch.float32)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                probe = torch.randn_like(output)
                params = [x, actual.lora_A, actual.lora_B, actual.m]
                refs = [ref_x, expected.lora_A, expected.lora_B, expected.m]
                grads = torch.autograd.grad((output * probe).sum(), params)
                ref_grads = torch.autograd.grad((reference * probe).sum(), refs)
                for grad, ref in zip(grads, ref_grads):
                    self.assertTrue(torch.isfinite(grad).all())
                    torch.testing.assert_close(grad, ref, rtol=0, atol=0)
                torch.testing.assert_close(actual.to_linear()(x), output, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_device_transfer_and_bfloat16_autocast(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._parity(dtype, True, autocast=True, device="cuda")

    def test_merge_and_zero_column_preserve_historical_semantics(self):
        base = nn.Linear(7, 5, bias=False)
        actual = OriginalDoRALinear(base, rank=3).eval()
        with torch.no_grad():
            actual.lora_B.normal_(std=0.2)
        x = torch.randn(4, 7)
        before = torch.random.get_rng_state()
        merged = actual.to_linear()
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        torch.testing.assert_close(merged(x), actual(x), rtol=0, atol=0)
        self.assertFalse(merged.training)
        self.assertTrue(all(not p.requires_grad for p in merged.parameters()))
        self.assertIsNone(merged.bias)
        with torch.no_grad():
            actual.weight[:, 0].zero_()
            actual.lora_B[:, 0].zero_()
            actual.m[:, 0].zero_()
        # Preserve the original zero-column singularity; do not silently clamp.
        self.assertFalse(torch.isfinite(actual(x)).all())


if __name__ == "__main__":
    unittest.main()
