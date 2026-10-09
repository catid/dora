"""Compare the live standalone layer with installed Hugging Face PEFT DoRA."""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.lora.layer import Linear


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/home/catid/dora'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    module_path = args.repo / 'dora.py'
    spec = importlib.util.spec_from_file_location('audited_dora', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(20261009)
    torch.set_num_threads(1)
    base = nn.Linear(7, 5, bias=True, dtype=torch.float64)
    actual = module.DoRALayer.from_linear(base, rank=3)
    official = get_peft_model(nn.Sequential(copy.deepcopy(base)), LoraConfig(
        r=3, lora_alpha=3, target_modules=['0'], use_dora=True,
        lora_dropout=0.0, bias='none'))
    layer = official.base_model.model[0]
    actual.train()
    official.train()
    with torch.no_grad():
        actual.lora_A.normal_(std=0.15)
        actual.lora_B.normal_(std=0.12)
        actual.m.mul_(torch.linspace(0.7, 1.3, 5, dtype=torch.float64)[:, None])
        layer.lora_B['default'].weight.copy_(actual.lora_A)
        layer.lora_A['default'].weight.copy_(actual.lora_B)
        layer.lora_magnitude_vector['default'].weight.copy_(actual.m.flatten())
    assert layer.scaling['default'] == 1
    assert not layer.fan_in_fan_out
    assert all(p.dtype == torch.float64 for p in actual.parameters())
    assert all(p.dtype == torch.float64 for p in official.parameters())
    torch.testing.assert_close(layer.base_layer.weight, actual.weight, rtol=0, atol=0)
    torch.testing.assert_close(layer.base_layer.bias, actual.bias, rtol=0, atol=0)

    x = torch.randn(4, 2, 7, dtype=torch.float64, requires_grad=True)
    xp = x.detach().clone().requires_grad_()
    target = torch.randn(4, 2, 5, dtype=torch.float64)
    weights = torch.rand(4, 2, 5, dtype=torch.float64) + 0.2
    y, yp = actual(x), official(xp)
    loss = ((y - target).square() * weights).sum()
    loss_peft = ((yp - target).square() * weights).sum()
    loss.backward()
    loss_peft.backward()
    pairs = {
        'output': (y, yp), 'loss': (loss, loss_peft),
        'input_gradient': (x.grad, xp.grad),
        'standalone_A_vs_peft_B_gradient': (actual.lora_A.grad, layer.lora_B['default'].weight.grad),
        'standalone_B_vs_peft_A_gradient': (actual.lora_B.grad, layer.lora_A['default'].weight.grad),
        'magnitude_gradient': (actual.m.grad.flatten(), layer.lora_magnitude_vector['default'].weight.grad),
    }
    errors = {}
    for name, (left, right) in pairs.items():
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        assert torch.linalg.vector_norm(right) > 0
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
        delta = left - right
        errors[name] = {
            'maximum_absolute_error': delta.abs().max().item(),
            'relative_l2_error': (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(right)).item(),
            'peft_l2_norm': torch.linalg.vector_norm(right).item(),
        }
    assert all(p.grad is None for p in (actual.weight, actual.bias,
                                       layer.base_layer.weight, layer.base_layer.bias))
    files = {'standalone': str(module_path), 'peft_dora': inspect.getfile(DoraLinearLayer),
             'peft_linear': inspect.getfile(Linear), 'audit_script': __file__}
    result = {
        'passed': True, 'command': [sys.executable, *sys.argv],
        'torch': torch.__version__, 'peft': importlib.metadata.version('peft'),
        'device': 'cpu', 'dtype': 'float64', 'seed': 20261009,
        'weight_shape': [5, 7], 'input_shape': [4, 2, 7], 'rank': 3,
        'lora_alpha': 3, 'effective_scaling': 1, 'adapter_dropout': 0,
        'both_low_rank_factors_nonzero': True, 'magnitudes_nontrivially_rescaled': True,
        'base_weight_and_bias_identical_and_gradient_free': True,
        'errors': errors, 'source_paths': files,
        'source_sha256': {name: sha(path) for name, path in files.items()},
        'scope': 'Actual installed PEFT forward/backward parity for one deterministic nonsquare nonzero-adapter CPU FP64 case. Confirms the detached-denominator variant; does not compare the full-gradient variant, train models, guarantee lower-precision identity, or establish task-quality superiority.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
