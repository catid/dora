"""Independent math and optimizer-isolation tests for the two proposed mixtures."""

import copy
import unittest

import torch
from torch import nn

from experiments.second_round_adapters import AdapterLinear, METHODS, inject_adapters, merge_adapters, parameter_groups


class SecondRoundAdapterTests(unittest.TestCase):
    def test_dense_reference_gradients_all_methods(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(12)
                layer = AdapterLinear(nn.Linear(7, 5, dtype=torch.float64), method, rank=3)
                with torch.no_grad():
                    layer.lora_B.normal_(std=0.15)
                    if layer.m is not None:
                        layer.m.mul_(1.1)
                    if layer.log_gain is not None:
                        layer.log_gain.copy_(torch.linspace(-0.4, 0.4, 7))
                x = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
                params = [x, layer.lora_A, layer.lora_B]
                ref_x = x.detach().clone().requires_grad_()
                a = layer.lora_A.detach().clone().requires_grad_()
                b = layer.lora_B.detach().clone().requires_grad_()
                refs = [ref_x, a, b]
                down = a
                if method not in ("lora", "dora"):
                    down = a / a.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
                if layer.log_gain is not None:
                    gain = layer.log_gain.detach().clone().requires_grad_()
                    down = down * gain.exp()[None, :]
                    refs.append(gain)
                    params.append(layer.log_gain)
                weight = layer.weight.detach() + b @ down
                if layer.m is not None:
                    m = layer.m.detach().clone().requires_grad_()
                    weight = weight * (m / weight.square().sum(dim=1, keepdim=True).sqrt().detach())
                    refs.append(m)
                    params.append(layer.m)
                expected = ref_x @ weight.T + layer.bias
                actual = layer(x)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                upstream = torch.randn_like(actual)
                grads = torch.autograd.grad((actual * upstream).sum(), params)
                ref_grads = torch.autograd.grad((expected * upstream).sum(), refs)
                for grad, ref in zip(grads, ref_grads):
                    torch.testing.assert_close(grad, ref, atol=1e-11, rtol=1e-11)
                torch.testing.assert_close(merge_adapters(layer)(x), expected, atol=1e-12, rtol=1e-12)

    def test_matching_initialization_and_identity(self):
        base = nn.Linear(7, 5, dtype=torch.float64)
        x = torch.randn(2, 7, dtype=torch.float64)
        previous = None
        for method in METHODS:
            torch.manual_seed(33)
            layer = AdapterLinear(copy.deepcopy(base), method, rank=3)
            torch.testing.assert_close(layer(x), base(x), rtol=1e-13, atol=1e-13)
            if previous is not None:
                torch.testing.assert_close(layer.lora_A, previous, rtol=0, atol=0)
            previous = layer.lora_A.detach().clone()
            if layer.log_gain is not None:
                torch.testing.assert_close(layer.log_gain.exp(), torch.ones(7, dtype=torch.float64))

    def test_gain_restores_arbitrary_column_amplitudes(self):
        layer = AdapterLinear(nn.Linear(7, 5, dtype=torch.float64), "dora_nora_gain", rank=3)
        arbitrary = torch.randn_like(layer.lora_A) * torch.linspace(0.1, 3, 7)
        with torch.no_grad():
            layer.lora_A.copy_(arbitrary)
            layer.log_gain.copy_(arbitrary.norm(dim=0).log())
        torch.testing.assert_close(layer.effective_down(), arbitrary, rtol=1e-13, atol=1e-13)
        nora = AdapterLinear(nn.Linear(7, 5, dtype=torch.float64), "nora", rank=1)
        with torch.no_grad():
            nora.lora_B.fill_(2)
        update_norms = (nora.lora_B @ nora.effective_down()).norm(dim=0)
        torch.testing.assert_close(update_norms, update_norms[0].expand_as(update_norms))

    def test_optimizer_groups_no_duplicates_no_head_and_no_magnitude_decay(self):
        for method in METHODS:
            model = nn.Sequential(nn.Linear(7, 5), nn.Linear(5, 2))
            model = inject_adapters(model, method, rank=3, targets=("0",))
            model[1].requires_grad_(True)
            groups = parameter_groups(model, lr=0.01, magnitude_lr_multiplier=0.1, weight_decay=0.3)
            included = [p for group in groups for p in group["params"]]
            self.assertEqual(len(included), len({id(p) for p in included}))
            self.assertTrue({id(p) for p in included}.isdisjoint({id(p) for p in model[1].parameters()}))
            for group in groups:
                self.assertEqual(group["parameter_count"], sum(p.numel() for p in group["params"]))
                self.assertEqual(group["weight_decay"], 0.3 if group["group_name"] == "adapter_factors" else 0.0)
                self.assertEqual(group["lr"], 0.001 if group["group_name"] == "decoupled_magnitudes" else 0.01)
            # Real optimizer construction validates the metadata-bearing groups.
            optimizer = torch.optim.AdamW(groups)
            optimizer.zero_grad()
            model(torch.randn(4, 7)).square().mean().backward()
            optimizer.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in included))

    def test_mlr_forward_identical_and_magnitude_update_decoupled(self):
        base = nn.Linear(7, 5, dtype=torch.float64)
        torch.manual_seed(9)
        normal = AdapterLinear(copy.deepcopy(base), "dora_nora", rank=3)
        torch.manual_seed(9)
        slow = AdapterLinear(copy.deepcopy(base), "dora_nora_mlr", rank=3)
        x = torch.randn(4, 7, dtype=torch.float64)
        torch.testing.assert_close(normal(x), slow(x), atol=0, rtol=0)
        before = normal.m.detach().clone()
        optimizers = [torch.optim.SGD(parameter_groups(normal, 0.01, weight_decay=0)),
                      torch.optim.SGD(parameter_groups(slow, 0.01, magnitude_lr_multiplier=0.01, weight_decay=0))]
        for layer, optimizer in zip((normal, slow), optimizers):
            layer(x).sum().backward()
            optimizer.step()
        torch.testing.assert_close(slow.m - before, (normal.m - before) * 0.01, atol=1e-16, rtol=1e-10)

    def test_named_task_head_group_updates_with_adapters_and_preserves_base(self):
        for method in METHODS:
            with self.subTest(method=method):
                torch.manual_seed(17)
                model = nn.Sequential(nn.Linear(7, 5), nn.Linear(5, 2))
                model = inject_adapters(model, method, rank=3, targets=("0",))
                model[1].requires_grad_(True)
                groups = parameter_groups(model, 0.01, magnitude_lr_multiplier=0.1)
                head = list(model[1].named_parameters(prefix="1"))
                groups.append({"params": [parameter for _, parameter in head],
                               "param_names": [name for name, _ in head],
                               "group_name": "task_head", "lr": 0.01, "weight_decay": 0.01})
                optimizer = torch.optim.AdamW(groups)
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                inputs, targets = torch.randn(8, 7), torch.randn(8, 2)
                # A and gain initially receive no gradient because B starts at
                # zero. Two real steps cover their subsequent joint updates.
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    (model(inputs) - targets).square().mean().backward()
                    optimizer.step()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertTrue(torch.isfinite(parameter).all(), name)
                        self.assertFalse(torch.equal(parameter, before[name]), name)
                    else:
                        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                        self.assertIsNone(parameter.grad, name)


if __name__ == "__main__":
    unittest.main()
