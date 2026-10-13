"""Reproduce API/numerical edge cases independently of task-quality rankings."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import runpy

import torch
from torch import nn

import dora_row as dora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    original = Path('results/2026-10-09-baseline-audit/original_dora.py')
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    assert original_sha == 'e1d25979459dce1ff5f458469bece1e928c8bae02844acc6259b606ac228174d'
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    legacy = runpy.run_path(str(original))
    assert not torch.equal(before, torch.get_rng_state())
    assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(0).get_state())
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    importlib.reload(dora)
    assert torch.equal(before, torch.get_rng_state())
    checks = [{'change': 'Import RNG side effect', 'original': 'Resets caller RNG to seed0',
               'revised': 'Preserves caller RNG state exactly', 'kind': 'engineering'}]

    base = nn.Sequential(nn.Linear(7, 5, bias=False))
    try:
        legacy['replace_linear_with_dora'](base)
    except AttributeError as error:
        failure = str(error)
    else:
        raise AssertionError('Historical biasless conversion unexpectedly succeeded')
    inputs = torch.randn(8, 7)
    expected = base(inputs).detach()
    converted = dora.replace_linear_with_dora(base)
    assert converted[0].bias is None
    torch.testing.assert_close(converted(inputs), expected, rtol=1e-5, atol=1e-6)
    checks.append({'change': 'Biasless conversion', 'original': failure,
                   'revised': 'Succeeds, absent bias retained, initial outputs preserved within FP32 tolerance',
                   'kind': 'engineering'})

    weight, bias = torch.zeros(5, 7), torch.zeros(5)
    old = legacy['DoRALayer'](7, 5, rank=3, weight=weight.clone(), bias=bias.clone())
    new = dora.DoRALayer(7, 5, rank=3, weight=weight.clone(), bias=bias.clone())
    assert torch.isnan(old(inputs)).all()
    output = new(inputs)
    assert torch.equal(output, torch.zeros_like(output))
    (output - 1).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
    assert float(new.lora_B.grad.norm()) > 0
    checks.append({'change': 'Zero-weight normalization guard', 'original': 'All output entries NaN for an all-zero weight',
                   'revised': 'Finite zero outputs and nonzero direction-update gradient',
                   'kind': 'numerical edge case',
                   'scope': 'Does not establish an effect on nonzero pretrained weights; task runs separately count floor activations.'})

    bf16 = nn.Linear(7, 5, dtype=torch.bfloat16)
    old = legacy['DoRALayer'](7, 5, rank=3, weight=bf16.weight.detach().clone(), bias=bf16.bias.detach().clone())
    try:
        old(inputs.bfloat16())
    except RuntimeError as error:
        failure = str(error)
    else:
        raise AssertionError('Historical mixed-dtype projection unexpectedly succeeded')
    new = dora.DoRALayer.from_linear(bf16, rank=3)
    output = new(inputs.bfloat16())
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    checks.append({'change': 'Standalone BF16 projection support outside autocast', 'original': failure,
                   'revised': 'Finite BF16 projection with base-dtype factors and FP32 magnitudes',
                   'kind': 'dtype support',
                   'scope': 'Not mathematical parity: the benchmark original adapter instead uses its explicitly documented projection cast.'})

    new = dora.DoRALayer(7, 5, rank=3)
    assert new.bias is None and torch.isfinite(new.weight).all()
    assert 0 < float(new.weight.abs().max()) <= 1 / math.sqrt(7)
    assert 'torch.Tensor(d_out, d_in)' in original.read_text()
    checks.append({'change': 'Standalone base initialization',
                   'original': 'Source allocates torch.Tensor(d_out, d_in) without initialization when weight is absent',
                   'revised': 'Finite nonzero Kaiming-uniform base within the expected bounds; absent bias stays absent',
                   'kind': 'engineering',
                   'scope': 'The historical uninitialized values are not scored; task runs supply pretrained weights.'})

    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(7, 5)
            self.right = self.left

    old, new = Shared(), Shared()
    legacy['replace_linear_with_dora'](old)
    new = dora.replace_linear_with_dora(new)
    assert old.left is not old.right
    assert new.left is new.right and isinstance(new.left, dora.DoRALayer)
    root = nn.Linear(7, 5)
    assert legacy['replace_linear_with_dora'](root) is None and isinstance(root, nn.Linear)
    assert isinstance(dora.replace_linear_with_dora(root), dora.DoRALayer)
    checks.append({'change': 'Root conversion and shared aliases',
                   'original': 'A root Linear is not replaced; shared aliases can split into adapted and unadapted branches',
                   'revised': 'Returns an adapted root and retains shared aliases', 'kind': 'API expansion/edge case',
                   'scope': 'Not a claimed quality gain on the measured task module graphs.'})

    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path('dora_row.py'), original]
    hashes = {}
    for source in sources:
        name = 'source/' + ('audit_engineering_changes.py' if source == Path(__file__) else str(source))
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    result = {'status': 'pass', 'torch': torch.__version__, 'device': 'cpu', 'checks': checks,
              'source_sha256': hashes,
              'scope': 'Concrete failures or API boundaries in the archived source, compared with the preserved revised implementation in dora_row.py. These checks do not establish downstream superiority, and no core code is changed.'}
    (args.output / 'engineering_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
