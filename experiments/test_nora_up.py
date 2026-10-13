"""Independent up-factor normalization, unchanged-baseline and finite-start gates."""
import copy
from dataclasses import asdict
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from dora import DoRALayer
from experiments import nora_current as current
from experiments import nora_up as ab
from experiments import test_next_ablations as helpers


DEVICES = helpers.DEVICES


class UpNoRATests(unittest.TestCase):
    exact = helpers.NextAblationTests.exact
    trajectory = helpers.NextAblationTests.trajectory

    def test_only_up_normalization_changes_between_two_recipes(self):
        self.assertEqual(ab.METHODS, ('up_base', 'up_nora'))
        baseline = asdict(ab.REGISTRY['up_base'])
        treatment = asdict(ab.REGISTRY['up_nora'])
        self.assertEqual([key for key in baseline if baseline[key] != treatment[key]], ['nora'])
        for method, recipe in ab.REGISTRY.items():
            self.assertEqual(recipe.axis, 0)
            self.assertTrue(recipe.magnitude_fp32)
            self.assertFalse(recipe.detach_norm or recipe.norm_fp32 or recipe.factorized or recipe.clamp_norm or recipe.published_init)
            self.assertEqual((recipe.nora_factor, recipe.nora_axis, recipe.nora_epsilon), ('up', 1, 1e-12))
            manifest = ab.recipe_manifest(method)
            self.assertEqual(manifest['source_sha256'], ab.SOURCE_SHA256)
            self.assertEqual(manifest['inherited_current_source_sha256'], current.SOURCE_SHA256)
            self.assertIn('not paper NoRA', manifest['paper_relationship'])

    def test_initial_state_rng_order_storage_unchanged(self):
        for device in DEVICES:
            for dtype in (torch.float32, torch.bfloat16, torch.float64):
                base = nn.Linear(11, 7, dtype=dtype, device=device)
                torch.manual_seed(42)
                reference = current.AblationLinear(copy.deepcopy(base), 'current_dora', 3)
                expected_rng = helpers.rng(device)
                for method in ab.METHODS:
                    with self.subTest(device=device, dtype=dtype, method=method):
                        torch.manual_seed(42)
                        layer = ab.AblationLinear(copy.deepcopy(base), method, 3)
                        for actual, expected in zip(helpers.rng(device), expected_rng):
                            if actual is not None: self.exact(actual, expected)
                        self.assertEqual(list(layer.state_dict()), list(reference.state_dict()))
                        for name, value in reference.state_dict().items():
                            self.assertEqual(layer.state_dict()[name].dtype, value.dtype)
                            self.exact(layer.state_dict()[name], value)
                        self.assertEqual(layer.m.dtype, torch.float64 if dtype == torch.float64 else torch.float32)
                        d = layer.norm_diagnostics()
                        self.assertEqual(d['nora_factor'], 'up'); self.assertEqual(d['nora_axis'], 1)
                        self.assertEqual(d['nora_vector_count'], 7)
                        self.assertEqual(d['nora_initial_zero_vectors'], 0)
                        self.assertEqual(d['nora_initial_epsilon_vectors'], 0)
                        self.assertEqual(d['nora_norm_evaluations'], int(method == 'up_nora'))

    def test_up_base_exact_current_clipped_adamw_five_steps(self):
        for device in DEVICES:
            for dtype in (torch.float32, torch.bfloat16):
                for autocast in (False, True):
                    for grouped in (False, True):
                        with self.subTest(device=device, dtype=dtype, autocast=autocast, grouped=grouped):
                            base = nn.Linear(11, 7, dtype=dtype, device=device, bias=False)
                            torch.manual_seed(23); reference = current.AblationLinear(copy.deepcopy(base), 'current_dora', 3)
                            torch.manual_seed(23); actual = ab.AblationLinear(copy.deepcopy(base), 'up_base', 3)
                            self.trajectory(actual, reference, device, dtype, autocast, grouped)

    def test_up_base_exact_public_clipped_adamw_five_steps(self):
        for device in DEVICES:
            for dtype, autocast in ((torch.float32, False), (torch.float32, True), (torch.bfloat16, True)):
                with self.subTest(device=device, dtype=dtype, autocast=autocast):
                    base = nn.Linear(11, 7, dtype=dtype, device=device)
                    torch.manual_seed(52)
                    reference = DoRALayer(11, 7, 3, base.weight.detach().clone(), base.bias.detach().clone()).to(device=device)
                    torch.manual_seed(52); actual = ab.AblationLinear(copy.deepcopy(base), 'up_base', 3)
                    self.trajectory(actual, reference, device, dtype, autocast)

    def test_independent_fp64_dense_output_and_all_gradients(self):
        for device in DEVICES:
            for method, recipe in ab.REGISTRY.items():
                with self.subTest(device=device, method=method):
                    layer = ab.AblationLinear(nn.Linear(11, 7, dtype=torch.float64, device=device), method, 3).double()
                    with torch.no_grad():
                        layer.lora_A.normal_(std=.2); layer.lora_B.normal_(std=.2); layer.m.mul_(1.2)
                    x = torch.randn(2, 4, 11, dtype=torch.float64, device=device, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    down, up, m = [t.detach().clone().requires_grad_() for t in (layer.lora_A, layer.lora_B, layer.m)]
                    effective_up = up / up.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12) if recipe.nora else up
                    adapted = layer.weight + effective_up @ down
                    expected = F.linear(xr, m * (adapted / adapted.square().sum(dim=0, keepdim=True).sqrt()), layer.bias)
                    actual = layer(x)
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    probe = torch.randn_like(actual)
                    grads = torch.autograd.grad((actual * probe).sum(), (x, layer.lora_A, layer.lora_B, layer.m))
                    refs = torch.autograd.grad((expected * probe).sum(), (xr, down, up, m))
                    for value, reference in zip(grads, refs):
                        self.assertTrue(torch.isfinite(value).all()); self.assertGreater(float(value.norm()), 0)
                        torch.testing.assert_close(value, reference, atol=1e-11, rtol=1e-11)

    def test_up_row_normalization_derivative_and_epsilon_branch(self):
        for device in DEVICES:
            layer = ab.AblationLinear(nn.Linear(11, 7, dtype=torch.float64, device=device), 'up_nora', 3).double()
            with torch.no_grad(): layer.lora_B.normal_(std=.2)
            effective = layer.effective_up()
            self.exact(effective, layer.lora_B / layer.lora_B.norm(dim=1, keepdim=True))
            torch.testing.assert_close(effective.norm(dim=1), torch.ones(7, dtype=torch.float64, device=device), atol=1e-15, rtol=1e-15)
            probe = torch.randn_like(effective)
            grad = torch.autograd.grad((effective * probe).sum(), layer.lora_B)[0]
            torch.testing.assert_close((grad * layer.lora_B).sum(dim=1), torch.zeros(7, dtype=torch.float64, device=device), atol=1e-14, rtol=0)
            with torch.no_grad(): layer.lora_B[0].zero_()
            effective = layer.effective_up()
            grad = torch.autograd.grad((effective * probe).sum(), layer.lora_B)[0]
            torch.testing.assert_close(grad[0], probe[0] / 1e-12, atol=0, rtol=1e-15)
            self.assertTrue(torch.isfinite(effective).all() and torch.isfinite(grad).all())
            self.assertEqual(layer.norm_diagnostics()['nora_epsilon_vector_events'], 1)

    def test_initial_identity_bounded_first_grad_and_five_finite_updates(self):
        for device in DEVICES:
            for dtype, autocast in ((torch.float32, False), (torch.float32, True), (torch.bfloat16, True)):
                with self.subTest(device=device, dtype=dtype, autocast=autocast):
                    torch.manual_seed(33)
                    base = nn.Linear(11, 7, dtype=dtype, device=device)
                    torch.manual_seed(23); off = ab.AblationLinear(copy.deepcopy(base), 'up_base', 3)
                    torch.manual_seed(23); on = ab.AblationLinear(copy.deepcopy(base), 'up_nora', 3)
                    x = torch.randn(4, 11, device=device, dtype=dtype); target = torch.randn(4, 7, device=device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                        outputs = [layer(x) for layer in (off, on)]
                        self.exact(*outputs)
                        losses = [(value.float() - target).square().mean() for value in outputs]
                    for loss in losses: loss.backward()
                    self.exact(on.m.grad, off.m.grad)
                    self.assertEqual(int(torch.count_nonzero(on.lora_B.grad)), 0)
                    self.assertEqual(on.norm_diagnostics()['nora_epsilon_vector_events'], 0)
                    # This deterministic healthy initializer gives comparable finite gradients;
                    # unlike zero-factor normalization there is no 1/epsilon startup amplification.
                    ratio = float(on.lora_A.grad.norm() / off.lora_A.grad.norm())
                    self.assertGreater(ratio, .1); self.assertLess(ratio, 10.)
                    parameters = [p for p in on.parameters() if p.requires_grad]
                    opt = torch.optim.AdamW(parameters, lr=.001)
                    for index in range(5):
                        if index:
                            opt.zero_grad(set_to_none=True)
                            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
                                loss = (on(x).float() - target).square().mean()
                            self.assertTrue(torch.isfinite(loss)); loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                        self.assertTrue(torch.isfinite(norm)); opt.step()
                        self.assertTrue(all(torch.isfinite(p).all() for p in parameters))
                    d = on.assert_no_clamp()
                    self.assertEqual(d['nora_initial_zero_vectors'], 0)
                    self.assertEqual(d['nora_initial_epsilon_vectors'], 0)
                    self.assertEqual(d['nora_epsilon_vector_events'], 0)
                    self.assertGreater(d['nora_minimum_raw_vector_norm'], 1e-12)
                    self.assertEqual(d['nora_norm_evaluations'], 6)
                    self.assertEqual(d['norm_evaluations'], 6)

    def test_transposed_formula_identity(self):
        layer = ab.AblationLinear(nn.Linear(11, 7, dtype=torch.float64), 'up_nora', 3).double()
        with torch.no_grad(): layer.lora_A.normal_(std=.2)
        v = layer.weight + F.normalize(layer.lora_B, dim=1, eps=1e-12) @ layer.lora_A
        original = layer.m * (v / v.norm(dim=0, keepdim=True))
        transposed = layer.weight.T + layer.lora_A.T @ F.normalize(layer.lora_B.T, dim=0, eps=1e-12)
        transposed = layer.m.T * (transposed / transposed.norm(dim=1, keepdim=True))
        torch.testing.assert_close(original.T, transposed, atol=1e-15, rtol=1e-15)

    def test_alias_injection_groups_and_no_hidden_norm_floor(self):
        model = nn.Module(); model.first = nn.Linear(11, 7, bias=False); model.alias = model.first; model.head = nn.Linear(7, 2)
        ab.inject_adapters(model, 'up_nora', 3, targets=('alias',))
        self.assertIs(model.first, model.alias); self.assertFalse(model.head.weight.requires_grad)
        self.assertEqual(ab.parameter_groups(model, .001)[0]['param_names'], ['first.lora_B', 'first.lora_A'])
        d = ab.assert_no_clamp(model)['first']
        self.assertEqual(d['nora_initial_epsilon_vectors'], 0)
        self.assertEqual(d['nora_vector_count'], 7)
        self.assertIn(ab.core.AblationLinear, ab.backend().adapter_types)
        with torch.no_grad(): model.first.weight.zero_()
        self.assertFalse(torch.isfinite(model.first(torch.randn(3, 11))).all())
        self.assertIsNone(model.first.assert_no_clamp()['norm_floor'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
