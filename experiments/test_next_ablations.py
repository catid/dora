"""Independent next-recipe, public-default and FP32 reuse controls."""

import copy
from dataclasses import asdict
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from dora import DoRALayer
from experiments import change_ablations as old
from experiments import next_ablations as ab


DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


def params(layer):
    return [value for value in layer.parameters() if value.requires_grad]


def rng(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state() if device == "cuda" else None)


class NextAblationTests(unittest.TestCase):
    def exact(self, a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def trajectory(self, actual, reference, device, dtype, autocast, grouped=False):
        pa, pb = params(actual), params(reference)
        self.assertEqual(len(pa), len(pb))
        for a, b in zip(pa, pb):
            self.assertEqual(a.dtype, b.dtype)
            self.exact(a, b)
        groups = lambda model: ab.parameter_groups(model, lr=.003, magnitude_lr_multiplier=.3) if grouped else params(model)
        oa, ob = [torch.optim.AdamW(groups(layer), lr=.003, weight_decay=.01) for layer in (actual, reference)]
        for _ in range(5):
            x = torch.randn(4, 11, device=device, dtype=dtype)
            target = torch.randn(4, 7, device=device)
            oa.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                a, b = actual(x), reference(x)
                losses = [(out.float() - target).square().mean() for out in (a, b)]
            self.exact(a, b)
            for loss in losses:
                loss.backward()
            for a, b in zip(pa, pb):
                self.exact(a.grad, b.grad)
            self.exact(torch.nn.utils.clip_grad_norm_(pa, .2), torch.nn.utils.clip_grad_norm_(pb, .2))
            oa.step(); ob.step()
            for a, b in zip(pa, pb):
                self.exact(a, b)
                for key in oa.state[a]:
                    self.exact(oa.state[a][key], ob.state[b][key])

    def test_exact_single_additions_and_source_bound_manifests(self):
        self.assertEqual(len(ab.REGISTRY), 6)
        baseline = asdict(ab.REGISTRY["next_base"])
        expected = {"next_base": [], "next_axis": ["axis"], "next_detach": ["detach_norm"],
                    "next_norm_fp32": ["norm_fp32"], "next_init": ["published_init"],
                    "next_factorized": ["factorized"]}
        self.assertEqual(len(set(ab.REGISTRY.values())), 6)
        with self.assertRaises(TypeError):
            ab.REGISTRY["invalid"] = ab.REGISTRY["next_base"]
        hashes = set()
        for method, recipe in ab.REGISTRY.items():
            changes = [key for key, value in asdict(recipe).items() if value != baseline[key]]
            self.assertEqual(changes, expected[method])
            self.assertTrue(recipe.magnitude_fp32)
            self.assertFalse(recipe.clamp_norm)
            manifest = ab.recipe_manifest(method)
            self.assertEqual(manifest["changed_fields_from_baseline"], changes)
            self.assertEqual(manifest["source_sha256"], ab.SOURCE_SHA256)
            self.assertEqual(manifest["core_source_sha256"], old.SOURCE_SHA256)
            hashes.add(manifest["recipe_sha256"])
        self.assertEqual(len(hashes), 6)

    def test_initializers_rng_parameter_order_and_magnitude_precision(self):
        for device in DEVICES:
            for dtype in (torch.float32, torch.bfloat16, torch.float64):
                base = nn.Linear(11, 7, dtype=dtype, device=device)
                for method, recipe in ab.REGISTRY.items():
                    with self.subTest(device=device, dtype=dtype, method=method):
                        torch.manual_seed(19)
                        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                        if recipe.published_init:
                            down = torch.empty(3, 11, dtype=work_dtype, device=device)
                            up = torch.zeros(7, 3, dtype=work_dtype, device=device)
                            nn.init.kaiming_uniform_(down, a=math.sqrt(5))
                            order = ["lora_A", "lora_B", "m"]
                        else:
                            std = 1 / torch.sqrt(torch.tensor(3).float())
                            up = (torch.randn(7, 3) * std).to(device)
                            down = torch.zeros(3, 11).to(device)
                            order = ["m", "lora_B", "lora_A"]
                        expected_rng = rng(device)
                        weight = base.weight.to(work_dtype) if recipe.norm_fp32 else base.weight
                        expected_m = weight.norm(dim=recipe.axis, keepdim=True).to(work_dtype)
                        torch.manual_seed(19)
                        layer = ab.AblationLinear(copy.deepcopy(base), method, 3)
                        for value, expected in zip(rng(device), expected_rng):
                            if value is not None:
                                self.exact(value, expected)
                        self.exact(layer.lora_A, down); self.exact(layer.lora_B, up)
                        self.exact(layer.m, expected_m)
                        self.assertEqual(layer.m.dtype, work_dtype)
                        self.assertEqual([n for n, p in layer.named_parameters() if p.requires_grad], order)
                        if dtype == torch.bfloat16 and recipe.norm_fp32:
                            self.assertFalse(torch.equal(layer.m, expected_m.bfloat16().float()))

    def test_public_default_baseline_exact_clipped_adamw(self):
        for device in DEVICES:
            for dtype, autocast in ((torch.float32, False), (torch.float32, True), (torch.bfloat16, True)):
                for grouped in (False, True):
                    with self.subTest(device=device, dtype=dtype, autocast=autocast, grouped=grouped):
                        base = nn.Linear(11, 7, device=device, dtype=dtype)
                        torch.manual_seed(71)
                        reference = DoRALayer(11, 7, 3, base.weight.detach().clone(), base.bias.detach().clone()).to(device=device)
                        expected_rng = rng(device)
                        torch.manual_seed(71)
                        actual = ab.AblationLinear(copy.deepcopy(base), "next_base", 3)
                        for value, expected in zip(rng(device), expected_rng):
                            if value is not None:
                                self.exact(value, expected)
                        self.trajectory(actual, reference, device, dtype, autocast, grouped)

    def test_baseline_exact_previous_measured_fp32_magnitude_recipe(self):
        for device in DEVICES:
            for dtype in (torch.float32, torch.bfloat16):
                for autocast in (False, True):
                    for grouped in (False, True):
                        with self.subTest(device=device, dtype=dtype, autocast=autocast, grouped=grouped):
                            base = nn.Linear(11, 7, device=device, dtype=dtype, bias=False)
                            torch.manual_seed(77)
                            reference = old.AblationLinear(copy.deepcopy(base), "ab_m_fp32", 3)
                            torch.manual_seed(77)
                            actual = ab.AblationLinear(copy.deepcopy(base), "next_base", 3)
                            self.trajectory(actual, reference, device, dtype, autocast, grouped)

    def test_each_fp32_recipe_exact_old_single_change_for_reuse(self):
        for device in DEVICES:
            for method, source_method in ab.SOURCE_METHODS.items():
                for autocast in (False, True):
                    for grouped in (False, True):
                        with self.subTest(device=device, method=method, autocast=autocast, grouped=grouped):
                            base = nn.Linear(11, 7, device=device)
                            torch.manual_seed(77)
                            reference = old.AblationLinear(copy.deepcopy(base), source_method, 3)
                            torch.manual_seed(77)
                            actual = ab.AblationLinear(copy.deepcopy(base), method, 3)
                            self.trajectory(actual, reference, device, torch.float32, autocast, grouped)

    def test_all_recipes_match_independent_dense_fp64_outputs_and_derivatives(self):
        for device in DEVICES:
            for method, recipe in ab.REGISTRY.items():
                with self.subTest(device=device, method=method):
                    layer = ab.AblationLinear(nn.Linear(11, 7, device=device, dtype=torch.float64), method, 3).double()
                    with torch.no_grad():
                        layer.lora_A.normal_(std=.2); layer.lora_B.normal_(std=.2)
                        layer.m.mul_(torch.linspace(.7, 1.3, layer.m.numel(), device=device).reshape(layer.m.shape))
                    x = torch.randn(2, 4, 11, dtype=torch.float64, device=device, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    down, up, magnitude = [t.detach().clone().requires_grad_() for t in (layer.lora_A, layer.lora_B, layer.m)]
                    adapted = layer.weight + up @ down
                    denominator = adapted.square().sum(dim=recipe.axis, keepdim=True).sqrt()
                    if recipe.detach_norm:
                        denominator = denominator.detach()
                    expected = F.linear(xr, magnitude * (adapted / denominator), layer.bias)
                    actual = layer(x)
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    probe = torch.randn_like(actual)
                    actual_grad = torch.autograd.grad((actual * probe).sum(), (x, layer.lora_A, layer.lora_B, layer.m))
                    ref_grad = torch.autograd.grad((expected * probe).sum(), (xr, down, up, magnitude))
                    for value, reference in zip(actual_grad, ref_grad):
                        self.assertTrue(torch.isfinite(value).all())
                        self.assertGreater(float(value.norm()), 0)
                        torch.testing.assert_close(value, reference, atol=1e-11, rtol=1e-11)

    def test_norm_precision_uses_fp32_denominator_but_native_numerator(self):
        for device in DEVICES:
            base = nn.Linear(11, 7, dtype=torch.bfloat16, device=device)
            layer = ab.AblationLinear(base, "next_norm_fp32", 3)
            with torch.no_grad():
                layer.lora_A.normal_(std=.2); layer.m.mul_(1.2)
            x = torch.randn(4, 11, dtype=torch.bfloat16, device=device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                numerator = layer.weight + layer.lora_B @ layer.lora_A
                with torch.autocast(device_type=device, enabled=False):
                    denominator = (layer.lora_B @ layer.lora_A + layer.weight).norm(dim=0, keepdim=True)
                expected = F.linear(x, layer.m * (numerator / denominator), layer.bias)
                self.exact(layer(x), expected)

    def test_injection_aliases_biasless_groups_diagnostics_and_no_hidden_floor(self):
        model = nn.Module()
        model.first = nn.Linear(11, 7, bias=False)
        model.alias = model.first
        model.head = nn.Linear(7, 2)
        adapted = ab.inject_adapters(model, "next_base", rank=3, targets=("alias",))
        self.assertIs(adapted.first, adapted.alias)
        self.assertIsNone(adapted.first.bias)
        self.assertFalse(adapted.head.weight.requires_grad)
        self.assertEqual(ab.parameter_groups(adapted, .001)[0]["param_names"], ["first.lora_B", "first.lora_A"])
        self.assertEqual(ab.assert_no_clamp(adapted)["first"]["recipe_sha256"], ab.recipe_manifest("next_base")["recipe_sha256"])
        self.assertIn(old.AblationLinear, ab.backend().adapter_types)
        with torch.no_grad():
            adapted.first.weight.zero_()
        self.assertFalse(torch.isfinite(adapted.first(torch.randn(3, 11))).all())
        self.assertEqual(ab.assert_no_clamp(adapted)["first"]["norm_floor"], None)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
