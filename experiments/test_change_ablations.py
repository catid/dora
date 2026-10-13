"""Endpoint, derivative, optimizer-trajectory and intervention checks.

Run CPU: CUDA_VISIBLE_DEVICES='' python -m unittest -v experiments.test_change_ablations
Run CUDA: CUDA_VISIBLE_DEVICES=1 python -m unittest -v experiments.test_change_ablations
"""

import copy
from dataclasses import asdict
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.adapters import AdapterLinear
from experiments.original_dora import OriginalDoRALinear
from experiments.test_original_dora import historical_class
from experiments.change_ablations import (
    AblationLinear, REGISTRY, assert_no_clamp, inject_adapters, parameter_groups,
    recipe_manifest,
)


DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


def rng_state(device):
    return (torch.random.get_rng_state(), torch.cuda.get_rng_state() if device == "cuda" else None)


def trainables(layer):
    return [(name, parameter) for name, parameter in layer.named_parameters() if parameter.requires_grad]


class ChangeAblationTests(unittest.TestCase):
    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def endpoint(self, method, dtype, device, bias=True):
        torch.manual_seed(987)
        base = nn.Linear(11, 7, bias=bias, dtype=dtype, device=device).eval()
        torch.manual_seed(71)
        expected = OriginalDoRALinear(copy.deepcopy(base), 3) if method == "ab_original" else AdapterLinear(copy.deepcopy(base), "dora", 3)
        expected_rng = rng_state(device)
        torch.manual_seed(71)
        actual = AblationLinear(copy.deepcopy(base), method, 3)
        actual_rng = rng_state(device)
        for current, reference in zip(actual_rng, expected_rng):
            if current is not None:
                self.assert_exact(current, reference)
        rename = {"lora_A": "lora_B", "lora_B": "lora_A"} if method == "ab_original" else {}
        expected_order = [rename.get(name, name) for name, _ in trainables(expected)]
        self.assertEqual([name for name, _ in trainables(actual)], expected_order)
        for name, parameter in expected.named_parameters():
            value = getattr(actual, rename.get(name, name))
            self.assertEqual(value.dtype, parameter.dtype)
            self.assert_exact(value, parameter)
        self.assertFalse(actual.training)
        return actual, expected, rename

    def test_recipes_are_unique_and_single_changes_are_single(self):
        self.assertEqual(len(REGISTRY), 12)
        self.assertEqual(len(set(REGISTRY.values())), 12)
        original = asdict(REGISTRY["ab_original"])
        for method, flag in (("ab_axis", "axis"), ("ab_detach", "detach_norm"),
                             ("ab_m_fp32", "magnitude_fp32"), ("ab_norm_fp32", "norm_fp32"),
                             ("ab_init", "published_init"), ("ab_factorized", "factorized")):
            changed = [key for key, value in asdict(REGISTRY[method]).items() if value != original[key]]
            self.assertEqual(changed, [flag])
        self.assertEqual(len({recipe_manifest(name)["recipe_sha256"] for name in REGISTRY}), 12)
        self.assertTrue(REGISTRY["ab_revised"].clamp_norm)
        self.assertFalse(any(recipe.clamp_norm for name, recipe in REGISTRY.items() if name != "ab_revised"))

    def test_original_constructor_matches_archived_class_and_rng(self):
        historical, _ = historical_class()
        for device in DEVICES:
            for dtype in (torch.float32, torch.bfloat16, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    base = nn.Linear(11, 7, dtype=dtype, device=device)
                    torch.manual_seed(17)
                    expected = historical(11, 7, rank=3, weight=base.weight.detach().clone(), bias=base.bias.detach().clone()).to(device)
                    expected_rng = rng_state(device)
                    torch.manual_seed(17)
                    actual = AblationLinear(copy.deepcopy(base), "ab_original", 3)
                    for value, reference in zip(rng_state(device), expected_rng):
                        if value is not None:
                            self.assert_exact(value, reference)
                    for name, other in (("weight", "weight"), ("bias", "bias"), ("m", "m"), ("lora_A", "lora_B"), ("lora_B", "lora_A")):
                        self.assert_exact(getattr(actual, name), getattr(expected, other))
                    self.assertEqual(actual.m.dtype, dtype)
                    self.assertEqual(actual.lora_A.dtype, torch.get_default_dtype())

    def test_endpoints_nonzero_outputs_all_gradients_and_dtypes(self):
        for device in DEVICES:
            for method in ("ab_original", "ab_revised"):
                for dtype in (torch.float32, torch.bfloat16, torch.float64):
                    for autocast in (False, True) if dtype != torch.float64 else (False,):
                        for bias in (True, False):
                            with self.subTest(device=device, method=method, dtype=dtype, autocast=autocast, bias=bias):
                                actual, expected, rename = self.endpoint(method, dtype, device, bias)
                                # Nonzero factors and changed magnitudes exercise
                                # denominator gradients and both factor gradients.
                                with torch.no_grad():
                                    for name, parameter in trainables(expected):
                                        if name == "m":
                                            parameter.mul_(torch.linspace(.7, 1.3, parameter.numel(), device=device).reshape(parameter.shape))
                                        else:
                                            parameter.normal_(std=.17)
                                        getattr(actual, rename.get(name, name)).copy_(parameter)
                                x = torch.randn(2, 4, 11, dtype=dtype, device=device, requires_grad=True)
                                reference_x = x.detach().clone().requires_grad_()
                                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                                    output, reference = actual(x), expected(reference_x)
                                self.assert_exact(output, reference)
                                self.assertEqual(output.dtype, reference.dtype)
                                probe = torch.randn_like(output)
                                gradients = torch.autograd.grad((output * probe).sum(), [x, *[p for _, p in trainables(actual)]])
                                ref_gradients = torch.autograd.grad((reference * probe).sum(), [reference_x, *[p for _, p in trainables(expected)]])
                                for value, target in zip(gradients, ref_gradients):
                                    self.assertTrue(torch.isfinite(value).all())
                                    self.assertGreater(float(value.norm()), 0)
                                    self.assert_exact(value, target)
                                actual.assert_no_clamp()

    def trajectory(self, actual, expected, dtype, device, autocast):
        actual_parameters = [parameter for _, parameter in trainables(actual)]
        expected_parameters = [parameter for _, parameter in trainables(expected)]
        actual_optimizer = torch.optim.AdamW(actual_parameters, lr=.003, weight_decay=.01)
        expected_optimizer = torch.optim.AdamW(expected_parameters, lr=.003, weight_decay=.01)
        for step in range(6):
            inputs = torch.randn(9, 11, dtype=dtype, device=device)
            targets = torch.randn(9, 7, device=device)
            actual_optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                output, reference = actual(inputs), expected(inputs)
                loss = F.mse_loss(output.float(), targets)
                reference_loss = F.mse_loss(reference.float(), targets)
            self.assert_exact(output, reference)
            loss.backward()
            reference_loss.backward()
            for value, target in zip(actual_parameters, expected_parameters):
                self.assert_exact(value.grad, target.grad)
            norm = torch.nn.utils.clip_grad_norm_(actual_parameters, .2)
            reference_norm = torch.nn.utils.clip_grad_norm_(expected_parameters, .2)
            self.assert_exact(norm, reference_norm)
            actual_optimizer.step()
            expected_optimizer.step()
            for value, target in zip(actual_parameters, expected_parameters):
                self.assert_exact(value, target)
                for key in actual_optimizer.state[value]:
                    self.assert_exact(actual_optimizer.state[value][key], expected_optimizer.state[target][key])
        actual.assert_no_clamp()

    def test_endpoint_clipped_adamw_trajectories_are_exact(self):
        for device in DEVICES:
            for method in ("ab_original", "ab_revised"):
                for dtype in (torch.float32, torch.bfloat16):
                    for autocast in (False, True):
                        with self.subTest(device=device, method=method, dtype=dtype, autocast=autocast):
                            actual, expected, _ = self.endpoint(method, dtype, device)
                            self.trajectory(actual, expected, dtype, device, autocast)

    def test_fp32_magnitude_storage_is_exact_noop_on_fp32_base(self):
        for device in DEVICES:
            for before, after in (("ab_original", "ab_m_fp32"), ("cum_axis_detach", "cum_axis_detach_m_fp32")):
                with self.subTest(device=device, before=before):
                    base = nn.Linear(11, 7, device=device)
                    torch.manual_seed(82)
                    expected = AblationLinear(copy.deepcopy(base), before, 3)
                    torch.manual_seed(82)
                    actual = AblationLinear(copy.deepcopy(base), after, 3)
                    self.trajectory(actual, expected, torch.float32, device, True)

    def test_magnitude_storage_and_norm_precision_are_separate(self):
        base = nn.Linear(11, 7, dtype=torch.bfloat16)
        original = AblationLinear(copy.deepcopy(base), "ab_original", 3)
        storage = AblationLinear(copy.deepcopy(base), "ab_m_fp32", 3)
        norm = AblationLinear(copy.deepcopy(base), "ab_norm_fp32", 3)
        self.assertEqual(original.m.dtype, torch.bfloat16)
        self.assertEqual(storage.m.dtype, torch.float32)
        self.assertEqual(norm.m.dtype, torch.bfloat16)
        self.assert_exact(storage.m, base.weight.norm(dim=0, keepdim=True).float())
        self.assert_exact(norm.m, torch.linalg.vector_norm(base.weight.float(), dim=0, keepdim=True).bfloat16())
        self.assertGreater(float((storage.m - base.weight.float().norm(dim=0, keepdim=True)).abs().max()), 0)
        # Native denominator uses ambient BF16 matmul; independent FP32 norm
        # must leave the numerator exactly in that original ambient arithmetic.
        for device in DEVICES:
            layer = AblationLinear(copy.deepcopy(base).to(device), "ab_norm_fp32", 3)
            with torch.no_grad():
                layer.lora_A.normal_(std=.2)
            x = torch.randn(4, 11, device=device, dtype=torch.bfloat16)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                numerator = layer.weight + layer.lora_B @ layer.lora_A
                with torch.autocast(device_type=device, enabled=False):
                    denominator = (layer.lora_B @ layer.lora_A + layer.weight).norm(dim=0, keepdim=True)
                reference = F.linear(x, layer.m * (numerator / denominator), layer.bias)
                self.assert_exact(layer(x), reference)

    def test_factorized_full_gradient_axis_zero_matches_dense_in_fp64(self):
        for device in DEVICES:
            base = nn.Linear(11, 7, dtype=torch.float64, device=device)
            torch.manual_seed(66)
            dense = AblationLinear(copy.deepcopy(base), "ab_original", 3).double()
            torch.manual_seed(66)
            factorized = AblationLinear(copy.deepcopy(base), "ab_factorized", 3).double()
            with torch.no_grad():
                dense.lora_A.normal_(std=.2)
                dense.m.mul_(1.2)
                factorized.load_state_dict(dense.state_dict())
            x = torch.randn(2, 4, 11, dtype=torch.float64, device=device, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            output, reference = factorized(x), dense(y)
            torch.testing.assert_close(output, reference, rtol=1e-12, atol=1e-12)
            probe = torch.randn_like(output)
            actual = torch.autograd.grad((output * probe).sum(), [x, *[p for _, p in trainables(factorized)]])
            expected = torch.autograd.grad((reference * probe).sum(), [y, *[p for _, p in trainables(dense)]])
            for value, target in zip(actual, expected):
                torch.testing.assert_close(value, target, rtol=1e-11, atol=1e-12)

    def test_full_vs_detached_gradient_and_zero_norm_engineering_policy(self):
        base = nn.Linear(11, 7)
        torch.manual_seed(19)
        full = AblationLinear(copy.deepcopy(base), "ab_original", 3)
        torch.manual_seed(19)
        detached = AblationLinear(copy.deepcopy(base), "ab_detach", 3)
        with torch.no_grad():
            full.lora_A.normal_(std=.3)
            detached.load_state_dict(full.state_dict())
        x = torch.randn(5, 11)
        self.assert_exact(full(x), detached(x))
        full(x).square().sum().backward()
        detached(x).square().sum().backward()
        self.assertGreater(float((full.lora_A.grad - detached.lora_A.grad).norm()), .01)
        zero = nn.Linear(11, 7, bias=False)
        with torch.no_grad():
            zero.weight.zero_()
        literal = AblationLinear(copy.deepcopy(zero), "ab_original", 3)
        revised = AblationLinear(copy.deepcopy(zero), "ab_revised", 3)
        self.assertFalse(torch.isfinite(literal(x)).all())
        self.assertTrue(torch.isfinite(revised(x)).all())
        with self.assertRaisesRegex(RuntimeError, "Norm floor activated"):
            revised.assert_no_clamp()

    def test_injection_aliases_freezing_groups_and_fp64_norm(self):
        model = nn.Module()
        model.first = nn.Linear(11, 7, dtype=torch.float64)
        model.alias = model.first
        model.head = nn.Linear(7, 2, dtype=torch.float64)
        result = inject_adapters(model, "ab_original", rank=3, targets=("alias",))
        self.assertIs(result.first, result.alias)
        self.assertFalse(result.head.weight.requires_grad)
        groups = parameter_groups(result, lr=.001, magnitude_lr_multiplier=.5)
        self.assertEqual(groups[0]["param_names"], ["first.lora_B", "first.lora_A"])
        self.assertEqual(groups[1]["param_names"], ["first.m"])
        self.assertEqual(groups[1]["weight_decay"], 0)
        self.assertEqual(groups[1]["lr"], .0005)
        self.assertEqual(sum(g["parameter_count"] for g in groups), sum(p.numel() for p in result.parameters() if p.requires_grad))
        assert_no_clamp(result)
        norm = AblationLinear(nn.Linear(11, 7, dtype=torch.float64), "ab_norm_fp32", 3)
        self.assertEqual(norm.m.dtype, torch.float64)
        self.assertEqual(norm._weight_norm().dtype, torch.float64)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
