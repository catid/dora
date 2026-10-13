"""Require full-budget up-normalization baseline ViGGO reproduction against measured next-base runs.

Actual candidate and archived checkpoint payloads are mandatory for this parity
test. Portable reports retain its hash-bound evidence; they do not rerun or
pretend to revalidate omitted tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pair(reference, candidate, original):
    a, b = [json.loads((path / 'result.json').read_text()) for path in (reference, candidate)]
    fields = ('seed', 'lr', 'rank', 'steps', 'selected_step', 'validation_nll',
              'trainable_parameters', 'frozen_weights_unchanged', 'frozen_sha256')
    for field in fields:
        assert a[field] == b[field], (candidate, field)
    files = ['training.jsonl']
    if 'test' in a:
        for field, value in a['test'].items():
            if field != 'generation_seconds':
                assert value == b['test'][field]
        assert a['test_nll'] == b['test_nll']
        files.append('predictions.jsonl')
    for name in files:
        assert (reference / name).read_bytes() == (candidate / name).read_bytes(), (candidate, name)
    old, new = [torch.load(path / 'adapter.pt', map_location='cpu', weights_only=True)
                for path in (reference, candidate)]
    def renamed(name):
        prefix, suffix = name.rsplit('.', 1)
        return prefix + '.' + ({'lora_A': 'lora_B', 'lora_B': 'lora_A'}.get(suffix, suffix) if original else suffix)
    assert len(old) == len(new) == 216
    assert [renamed(name) for name in old] == list(new)
    assert sum(value.numel() for value in old.values()) == a['trainable_parameters']
    for name, tensor in old.items():
        new_name = renamed(name)
        assert tensor.dtype == new[new_name].dtype and tensor.shape == new[new_name].shape
        assert torch.isfinite(tensor).all() and torch.isfinite(new[new_name]).all()
        assert torch.equal(tensor, new[new_name]), (candidate, name)
    return {'reference': str(reference), 'candidate': str(candidate), 'seed': b['seed'],
            'lr': b['lr'], 'steps': b['steps'], 'tensors_bitwise_identical': len(old),
            'parameter_order_shapes_dtypes_identical': True,
            'files_bitwise_identical': files, 'scalar_metrics_identical': True,
            'reference_sha256': {name: sha(reference / name) for name in ['result.json', 'adapter.pt', *files]},
            'candidate_sha256': {name: sha(candidate / name) for name in ['result.json', 'adapter.pt', *files]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('/var/tmp/dora-bench/nora_current_bench/extraction'))
    p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    checks, missing = [], []
    for recipe, method, reference, original in [('up_base','current_dora',args.reference,False)]:
        tuning = json.loads((reference / 'tuning.json').read_text())
        lr = tuning['selected_learning_rates'][method]
        for seed in (42, 43, 44):
            candidate = args.root / 'finals' / f'{recipe}_lr{lr:g}_seed{seed}'
            if not (candidate / 'result.json').exists():
                missing.append(str(candidate)); continue
            checks.append(check_pair(reference / 'finals' / f'{method}_lr{lr:g}_seed{seed}', candidate, original))
        for lr in (0.0001, 0.0003):
            candidate = args.root / 'pilots' / f'{recipe}_lr{lr:g}_seed42'
            if not (candidate / 'result.json').exists():
                missing.append(str(candidate)); continue
            checks.append(check_pair(reference / 'pilots' / f'{method}_lr{lr:g}_seed42', candidate, original))
    assert args.allow_incomplete or not missing, missing
    result = {'status': 'partial' if missing else 'pass', 'checks': checks, 'missing': missing,
              'source_sha256': sha(Path(__file__))}
    target = args.root / ('baseline_history_partial.json' if missing else 'baseline_history_parity.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    source = args.root / 'audit_source' / Path(__file__).name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'fits_checked': len(checks),
                      'tensors_checked': sum(row['tensors_bitwise_identical'] for row in checks),
                      'missing_fits': len(missing)}))


if __name__ == '__main__':
    main()
