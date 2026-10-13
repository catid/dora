"""Controlled per-change ViGGO runs with fixed-rate and equally tuned lanes.

Uses the existing extraction training/evaluation loop via an explicit backend.
All pilots finish before held-out evaluation. A fit shared by both lanes is
stored once and referenced twice. Historical results are controls, never tuning
inputs except for the predeclared original-selected fixed learning rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import change_ablations as ab
from experiments import extraction as task


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend():
    return SimpleNamespace(adapter_types=(ab.AblationLinear, *task.ADAPTER_TYPES),
                           inject_adapters=ab.inject_adapters,
                           recipe_manifest=ab.recipe_manifest,
                           assert_no_clamp=ab.assert_no_clamp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('/var/tmp/dora-bench/original_bench/extraction'))
    p.add_argument('--methods', nargs='+', choices=ab.METHODS, default=list(ab.METHODS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--fixed-lr', type=float, default=0.0003)
    p.add_argument('--gate-evidence', type=Path, default=Path('/var/tmp/dora-bench/change_ablation_core_validation.json'))
    args = p.parse_args()
    gates = {
        'core': args.gate_evidence,
        'independent_review': Path('/var/tmp/dora-bench/change_ablation_independent_review.json'),
        'full_model_endpoints': Path('/var/tmp/dora-bench/change_bench/extraction_endpoint_control/control.json'),
    }
    gate_values = {name: json.loads(path.read_text()) for name, path in gates.items()}
    assert all(value['status'] == 'pass' for value in gate_values.values())
    core_sha = sha('experiments/change_ablations.py')
    assert gate_values['core']['core_source_sha256'] == core_sha
    assert gate_values['independent_review']['source_sha256'] == core_sha
    assert gate_values['full_model_endpoints']['source_sha256']['experiments/change_ablations.py'] == core_sha
    assert gate_values['full_model_endpoints']['source_sha256']['experiments/extraction.py'] == sha('experiments/extraction.py')
    reference = json.loads((args.reference / 'manifest.json').read_text())
    configuration = reference['configuration']
    for key in ('train_size', 'val_size', 'test_size', 'max_length', 'rank', 'batch_size',
                'generation_batch_size', 'pilot_steps', 'steps', 'eval_every', 'learning_rates'):
        setattr(args, key, configuration[key])
    original_tuning = json.loads((args.reference / 'tuning.json').read_text())
    assert args.fixed_lr == original_tuning['selected_learning_rates']['dora_original']
    assert args.rank == 8 and args.seeds == [42, 43, 44]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(task.MODEL, revision=task.MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    records, data_provenance = task.data(args, tokenizer)
    assert data_provenance == reference['dataset'], 'Original task data changed'
    assert task.MODEL_REVISION == reference['model_revision']
    assert hashlib.sha256(task.SYSTEM.encode()).hexdigest() == reference['system_prompt_sha256']
    sources = ['experiments/change_extraction.py', 'experiments/change_ablations.py',
               'experiments/extraction.py', 'experiments/adapters.py',
               'experiments/original_dora.py', 'dora_row.py']
    manifest = {
        'schema': 'dora-change-extraction-v1', 'task': 'ViGGO',
        'model': task.MODEL, 'model_revision': task.MODEL_REVISION,
        'dataset': data_provenance,
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'recipes': {method: ab.recipe_manifest(method) for method in args.methods},
        'source_sha256': {name: sha(name) for name in sources},
        'validation_gate_sha256': {name: sha(path) for name, path in gates.items()},
        'reference_manifest_sha256': sha(args.reference / 'manifest.json'),
        'reference_tuning_sha256': sha(args.reference / 'tuning.json'),
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(),
                        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')},
        'protocol': 'Same official ViGGO splits and deterministic subsets as original task. '
                    'Each recipe receives two 48-step seed42 LR pilots; minimum validation NLL '
                    'selects LR. Finals use 256 updates, checkpoint by validation NLL, seeds42-44. '
                    'Fixed lane uses original-selected LR0.0003 for every recipe; tuned lane uses '
                    'independent equally budgeted selection. Identical final fits are shared. '
                    'No held-out metric selects an LR or checkpoint. All norm-floor activations '
                    'must be zero. AdamW weight decay0.01 applies to every trainable parameter, '
                    'retaining historical first-round protocol.'}
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Configuration/source/data/environment changed; use a new output directory')
    task.write_json(path, manifest)
    for name, source in gates.items():
        target = args.output / 'gates' / (name + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in sources:
        target = args.output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    for name in ('manifest.json', 'tuning.json', 'task_audit.json'):
        target = args.output / 'reference' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.reference / name, target)
    model = AutoModelForCausalLM.from_pretrained(
        task.MODEL, revision=task.MODEL_REVISION, dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda()
    model.requires_grad_(False)
    adapter_backend = backend()

    def run_fit(method, seed, lr, stage):
        directory = args.output / stage / f'{method}_lr{lr:g}_seed{seed}'
        result_path = directory / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result['recipe'] == ab.recipe_manifest(method)
        else:
            result = task.fit(model, method, seed, lr,
                              args.pilot_steps if stage == 'pilots' else args.steps,
                              records, tokenizer, args, directory,
                              evaluate_test=stage == 'finals', backend=adapter_backend)
        assert result['frozen_weights_unchanged']
        control = json.loads((args.reference / 'dora_original_seed42' / 'result.json').read_text())
        assert result['frozen_sha256'] == control['frozen_sha256'], 'Frozen pretrained model differs'
        assert all(x['clamp_activations'] == 0 for x in result['norm_diagnostics'].values())
        return {**result, 'artifact_directory': str(directory.relative_to(args.output)),
                'result_sha256': sha(result_path)}

    pilots, selected = [], {}
    for method in args.methods:
        group = [run_fit(method, 42, lr, 'pilots') for lr in args.learning_rates]
        pilots.extend(group)
        selected[method] = min(group, key=lambda result: result['validation_nll'])['lr']
        task.write_json(args.output / 'tuning.json', {'pilots': pilots, 'selected_learning_rates': selected})
    final_fits, lanes = {}, []
    for seed in args.seeds:
        for method in args.methods:
            for lane, lr in (('fixed_original_lr', args.fixed_lr), ('validation_tuned', selected[method])):
                key = f'{method}_lr{lr:g}_seed{seed}'
                if key not in final_fits:
                    final_fits[key] = run_fit(method, seed, lr, 'finals')
                    print(json.dumps({'event': 'completed_fit', 'key': key,
                                      'exact_match': final_fits[key]['test']['exact_match']}), flush=True)
                lanes.append({'lane': lane, 'method': method, 'seed': seed, 'lr': lr, 'fit_key': key})
                task.write_json(args.output / 'results.json', {
                    'manifest': manifest, 'tuning': pilots, 'selected_learning_rates': selected,
                    'fits': final_fits, 'lanes': lanes,
                    'complete': len(lanes) == len(args.methods) * len(args.seeds) * 2})
    print(json.dumps({'event': 'study_completed', 'fits': len(final_fits), 'lanes': len(lanes)}), flush=True)


if __name__ == '__main__':
    main()
