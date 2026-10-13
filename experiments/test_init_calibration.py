"""Independent small-matrix references for initializer calibration statistics."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from experiments.init_calibration import CalibrationCollector


class CalibrationTests(unittest.TestCase):
    def check_case(self, device, autocast):
        torch.manual_seed(123)
        model = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Tanh(), nn.Linear(3, 2, bias=False)).to(device)
        model.train(); model[1].eval()
        reference = copy.deepcopy(model).eval()
        batches = [(torch.randn(2, 3, 4, device=device), torch.randn(2, 3, 2, device=device),
                    torch.tensor([[1, 1, 0], [1, 1, 1]], device=device).bool()),
                   (torch.randn(1, 3, 4, device=device), torch.randn(1, 3, 2, device=device),
                    torch.tensor([[1, 1, 0]], device=device).bool())]
        expected = {name: [] for name in ('0', '2')}
        for x, target, mask in batches:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                h = reference[0](x)
                z = reference[1](h)
                y = reference[2](z)
                loss = (y.float() - target).square().mean()
            dh, dy, gw0, gw2 = torch.autograd.grad(loss, (h, y, reference[0].weight, reference[2].weight))
            for name, inp, out, grad, wg in (('0', x, h, dh, gw0), ('2', z, y, dy, gw2)):
                expected[name].append((inp.detach().float().reshape(-1, inp.shape[-1]),
                                       out.detach().float().reshape(-1, out.shape[-1]),
                                       grad.detach().float().reshape(-1, grad.shape[-1]),
                                       wg.detach().float(), mask.reshape(-1), len(x)))
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        py_rng = random.getstate(); np_rng = np.random.get_state()
        flags = [m.training for m in model.modules()]
        required = [p.requires_grad for p in model.parameters()]
        initial = {n: p.detach().clone() for n,p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as directory:
            with CalibrationCollector(model, ('0', '2')) as collect:
                for i, (x, target, mask) in enumerate(batches):
                    collect.begin_batch(example_ids=[f'{i}:{j}' for j in range(len(x))], attention_mask=mask)
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                        y = model(x)
                        loss = (y.float() - target).square().mean()
                    collect.backward(loss, loss_weight=len(x))
                receipt = collect.save(Path(directory), {'task': 'independent tiny reference'})
            for name, records in expected.items():
                actual = torch.load(receipt['module_files'][name], weights_only=True)
                masked_x = torch.cat([x[mask] for x,y,g,wg,mask,n in records])
                masked_y = torch.cat([y[mask] for x,y,g,wg,mask,n in records])
                masked_g = torch.cat([g[mask] for x,y,g,wg,mask,n in records])
                mx, my = masked_x.mean(0), masked_y.mean(0)
                references = {
                    'input_cov': masked_x.T @ masked_x / len(masked_x),
                    'output_cov': masked_y.T @ masked_y / len(masked_y),
                    'input_mean': mx, 'output_mean': my,
                    'input_centered_cov': (masked_x-mx).T @ (masked_x-mx) / len(masked_x),
                    'output_centered_cov': (masked_y-my).T @ (masked_y-my) / len(masked_y),
                    'output_grad_cov': masked_g.T @ masked_g / len(masked_g),
                    'output_fisher_diag': masked_g.square().mean(0),
                    'weight_grad': sum(wg*n for x,y,g,wg,mask,n in records) / 3,
                    'filet_input_cov': sum(x.T@x/len(x) for x,y,g,wg,mask,n in records)/2,
                    'filet_output_grad_cov': sum(g.T@g/len(g) for x,y,g,wg,mask,n in records)/2,
                    'astra_output_cov': sum((y/y.max().abs()).T@(y/y.max().abs()) for x,y,g,wg,mask,n in records)/2,
                    'corda_input_cov': sum((x/x.max().abs()).T@(x/x.max().abs()) for x,y,g,wg,mask,n in records)/2,
                    'tlora_input_cov': sum((x-x.mean(0)).T@(x-x.mean(0)) for x,y,g,wg,mask,n in records)/2,
                }
                for key,value in references.items():
                    torch.testing.assert_close(actual[key], value.cpu(), rtol=2e-5, atol=2e-7, msg=key)
                self.assertEqual(actual['input_count'], 7)
                self.assertEqual(actual['population_size'], 3)
                for gram, (x,*_) in zip(actual['input_gram_batches'], records):
                    torch.testing.assert_close(gram, (x.T@x).cpu())
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else []):
            self.assertTrue(torch.equal(before, after))
        self.assertEqual(py_rng, random.getstate())
        self.assertEqual(np_rng[0], np.random.get_state()[0])
        np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
        self.assertEqual(flags, [m.training for m in model.modules()])
        self.assertEqual(required, [p.requires_grad for p in model.parameters()])
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)

    def test_cpu(self):
        self.check_case('cpu', False)

    def test_cpu_bf16(self):
        self.check_case('cpu', True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_bf16(self):
        self.check_case('cuda', True)


if __name__ == '__main__':
    unittest.main()
