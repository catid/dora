"""Original DoRA with only the measured FP32 magnitude-storage change."""
import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from dora import DoRALayer
from experiments.change_ablations import AblationLinear
from experiments.test_original_dora import historical_class


class OriginalMagnitudeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_native_norm_then_storage_cast(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                weight = torch.randn(7, 11, dtype=dtype)
                layer = DoRALayer(11, 7, rank=3, weight=weight, bias=torch.zeros(7, dtype=dtype))
                expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                self.assertEqual(layer.m.dtype, expected_dtype)
                self.assertEqual(layer.m.shape, (1, 11))
                torch.testing.assert_close(layer.m, weight.norm(dim=0, keepdim=True).to(expected_dtype), rtol=0, atol=0)
                self.assertEqual(layer.weight.dtype, dtype)
                self.assertEqual(layer.lora_A.dtype, torch.get_default_dtype())
                self.assertEqual(layer.lora_B.dtype, torch.get_default_dtype())
                if dtype in (torch.float16, torch.bfloat16):
                    self.assertFalse(torch.equal(layer.m, weight.float().norm(dim=0, keepdim=True)))

    def test_fp32_and_fp64_match_original_outputs_gradients_and_rng(self):
        original, _ = historical_class()
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                weight, bias = torch.randn(7, 11, dtype=dtype), torch.randn(7, dtype=dtype)
                torch.manual_seed(71)
                reference = original(11, 7, rank=3, weight=weight.clone(), bias=bias.clone())
                rng = torch.get_rng_state().clone()
                torch.manual_seed(71)
                layer = DoRALayer(11, 7, rank=3, weight=weight.clone(), bias=bias.clone())
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, value in reference.state_dict().items():
                    torch.testing.assert_close(layer.state_dict()[name], value, rtol=0, atol=0)
                with torch.no_grad():
                    layer.lora_B.normal_(std=.2)
                    layer.m.mul_(1.1)
                    reference.load_state_dict(layer.state_dict())
                x = torch.randn(2, 4, 11, dtype=dtype, requires_grad=True)
                y = x.detach().clone().requires_grad_()
                output, expected = layer(x), reference(y)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                probe = torch.randn_like(output)
                output.backward(probe); expected.backward(probe)
                for actual, target in [(x, y), *zip(layer.parameters(), reference.parameters())]:
                    if actual.grad is None:
                        self.assertIsNone(target.grad)
                    else:
                        torch.testing.assert_close(actual.grad, target.grad, rtol=0, atol=0)

    def test_full_column_norm_derivative(self):
        weight, bias = torch.randn(7, 11, dtype=torch.float64), torch.randn(7, dtype=torch.float64)
        layer = DoRALayer(11, 7, rank=3, weight=weight, bias=bias).double()
        with torch.no_grad():
            layer.lora_B.normal_(std=.3)
            layer.m.mul_(torch.linspace(.7, 1.3, 11))
        x = torch.randn(4, 11, dtype=torch.float64, requires_grad=True)
        xr, ar, br, mr = [t.detach().clone().requires_grad_() for t in (x, layer.lora_A, layer.lora_B, layer.m)]
        adapted = weight + ar @ br
        norm = adapted.square().sum(dim=0, keepdim=True).sqrt()
        expected = F.linear(xr, mr * (adapted / norm), bias)
        actual = layer(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        probe = torch.randn_like(actual)
        gradients = torch.autograd.grad((actual * probe).sum(), (x, layer.lora_A, layer.lora_B, layer.m))
        references = torch.autograd.grad((expected * probe).sum(), (xr, ar, br, mr), retain_graph=True)
        for value, reference in zip(gradients, references):
            torch.testing.assert_close(value, reference, atol=1e-11, rtol=1e-11)
        detached = F.linear(xr, mr * (adapted / norm.detach()), bias)
        detached_grad = torch.autograd.grad((detached * probe).sum(), br)[0]
        self.assertGreater(float((gradients[2] - detached_grad).abs().max()), 1e-3)

    def check_ablation_trajectory(self, device, dtype, autocast):
        base = nn.Linear(11, 7, dtype=dtype, device=device)
        torch.manual_seed(42)
        measured = AblationLinear(copy.deepcopy(base), 'ab_m_fp32', rank=3)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        layer = DoRALayer(11, 7, rank=3, weight=base.weight.detach().clone(), bias=base.bias.detach().clone()).to(device=device)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        params = [[p for p in module.parameters() if p.requires_grad] for module in (layer, measured)]
        optimizers = [torch.optim.AdamW(parameters, lr=.003, weight_decay=.01) for parameters in params]
        for step in range(5):
            x = torch.randn(5, 11, dtype=dtype, device=device)
            target = torch.randn(5, 7, device=device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                a, b = layer(x), measured(x)
                losses = [F.mse_loss(output.float(), target) for output in (a, b)]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for loss in losses:
                loss.backward()
            for value, expected in zip(*params):
                torch.testing.assert_close(value.grad, expected.grad, rtol=0, atol=0)
            for parameters, optimizer in zip(params, optimizers):
                torch.nn.utils.clip_grad_norm_(parameters, .2)
                optimizer.step()
            for value, expected in zip(*params):
                self.assertEqual(value.dtype, expected.dtype)
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
                for name, state in optimizers[0].state[value].items():
                    torch.testing.assert_close(state, optimizers[1].state[expected][name], rtol=0, atol=0)
            self.assertIsNone(layer.weight.grad)
            self.assertIsNone(layer.bias.grad)

    def test_cpu_matches_measured_single_change_through_adamw(self):
        for dtype, autocast in ((torch.float32, False), (torch.float32, True), (torch.bfloat16, True)):
            with self.subTest(dtype=dtype, autocast=autocast):
                self.check_ablation_trajectory('cpu', dtype, autocast)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_measured_single_change_through_adamw(self):
        for dtype, autocast in ((torch.float32, False), (torch.float32, True), (torch.bfloat16, True)):
            with self.subTest(dtype=dtype, autocast=autocast):
                self.check_ablation_trajectory('cuda', dtype, autocast)

    def test_small_magnitude_updates_survive_bf16_rounding(self):
        layer = DoRALayer(3, 2, rank=2, weight=torch.ones(2, 3, dtype=torch.bfloat16), bias=torch.zeros(2, dtype=torch.bfloat16))
        before = layer.m.detach().clone()
        optimizer = torch.optim.SGD([layer.m], lr=1e-5)
        layer.m.grad = torch.ones_like(layer.m)
        optimizer.step()
        self.assertTrue(torch.all(layer.m < before))
        self.assertTrue(torch.equal(layer.m.bfloat16(), before.bfloat16()))


if __name__ == '__main__':
    unittest.main()
