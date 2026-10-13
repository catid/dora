"""ViGGO single-change runs from original DoRA plus FP32 magnitude baseline.

Uses the existing extraction training/evaluation loop via an explicit backend.
All pilots finish before held-out evaluation. A fit shared by both lanes is
stored once and referenced twice. Historical results are controls, never tuning
inputs except for the predeclared original-selected fixed learning rate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import next_ablations as ab
from experiments import change_ablations as previous
from experiments import extraction as task


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def backend():
    return SimpleNamespace(adapter_types=(ab.AblationLinear, previous.AblationLinear, *task.ADAPTER_TYPES),
                           inject_adapters=ab.inject_adapters,
                           recipe_manifest=ab.recipe_manifest,
                           assert_no_clamp=ab.assert_no_clamp)


def control(model, records, tokenizer, args, sources):
    destination = args.output / 'baseline_control'
    path = destination / 'control.json'
    if path.exists():
        result = json.loads(path.read_text())
        assert result['status'] == 'pass' and result['manifest_sha256'] == sha(args.output / 'manifest.json')
        for relative, digest in result['evidence_sha256'].items():
            assert sha(destination / relative) == digest
        return result
    from copy import copy
    config = copy(args)
    config.eval_every = 5
    old_backend = SimpleNamespace(adapter_types=(previous.AblationLinear, ab.AblationLinear, *task.ADAPTER_TYPES),
                                  inject_adapters=previous.inject_adapters,
                                  recipe_manifest=previous.recipe_manifest, assert_no_clamp=previous.assert_no_clamp)
    old_dir, new_dir = destination / 'ab_m_fp32', destination / 'next_base'
    old = task.fit(model, 'ab_m_fp32', 42, args.fixed_lr, 5, records, tokenizer, config, old_dir, backend=old_backend)
    new = task.fit(model, 'next_base', 42, args.fixed_lr, 5, records, tokenizer, config, new_dir, backend=backend())
    for key in ('selected_step','validation_nll','trainable_parameters','frozen_sha256'):
        assert old[key] == new[key], key
    assert (old_dir / 'training.jsonl').read_bytes() == (new_dir / 'training.jsonl').read_bytes()
    a,b = [torch.load(directory / 'adapter.pt', map_location='cpu', weights_only=True) for directory in (old_dir,new_dir)]
    assert list(a) == list(b) and len(a) == 216
    for name,value in a.items():
        assert value.dtype == b[name].dtype and value.shape == b[name].shape and torch.equal(value,b[name]), name
    result = {'status':'pass','manifest_sha256':sha(args.output / 'manifest.json'),
              'method':'next_base','reference_method':'ab_m_fp32','steps':5,
              'checkpoint_tensors_bitwise_identical':216,'training_log_bitwise_identical':True,
              'validation_nll_bitwise_identical':True,'frozen_sha256':old['frozen_sha256'],
              'source_sha256':{name:sha(name) for name in sources},
              'evidence_sha256':{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()}}
    task.write_json(path,result)
    print(json.dumps({'event':'baseline_control_pass','tensors':216,'steps':5}),flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('/var/tmp/dora-bench/change_ablation_runs/extraction'))
    p.add_argument('--methods', nargs='+', choices=ab.METHODS, default=list(ab.METHODS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--fixed-lr', type=float, default=0.0003)
    p.add_argument('--gate-evidence', type=Path, default=Path('/var/tmp/dora-bench/next_ablation_validation.json'))
    p.add_argument('--control-only', action='store_true')
    args = p.parse_args()
    gate = json.loads(args.gate_evidence.read_text())
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    assert gate['backend_source_sha256'] == sha('experiments/next_ablations.py') == ab.SOURCE_SHA256
    assert gate['recipes'] == {method: ab.recipe_manifest(method) for method in ab.METHODS}
    gates = {'backend': args.gate_evidence}
    reference = json.loads((args.reference / 'manifest.json').read_text())
    configuration = reference['configuration']
    for key in ('train_size', 'val_size', 'test_size', 'max_length', 'rank', 'batch_size',
                'generation_batch_size', 'pilot_steps', 'steps', 'eval_every', 'learning_rates'):
        setattr(args, key, configuration[key])
    original_tuning = json.loads((args.reference / 'tuning.json').read_text())
    assert args.fixed_lr == original_tuning['selected_learning_rates']['ab_m_fp32']
    assert args.rank == 8 and args.seeds == [42, 43, 44]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(task.MODEL, revision=task.MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    records, data_provenance = task.data(args, tokenizer)
    assert data_provenance == reference['dataset'], 'Original task data changed'
    assert task.MODEL_REVISION == reference['model_revision']
    archived_task = args.reference / 'source/experiments/extraction.py'
    assert sha(archived_task) == reference['source_sha256']['experiments/extraction.py']
    def task_ast(path):
        module = ast.parse(Path(path).read_text())
        # The old standalone CLI's provenance filename changed dora -> dora_row;
        # every task function, constant and import used here must be unchanged.
        module.body = [node for node in module.body if not isinstance(node, ast.FunctionDef) or node.name != 'main']
        return ast.dump(module, include_attributes=False)
    assert task_ast('experiments/extraction.py') == task_ast(archived_task)
    sources = ['experiments/next_extraction.py', 'experiments/next_ablations.py', 'experiments/change_ablations.py',
               'experiments/extraction.py', 'experiments/adapters.py',
               'experiments/original_dora.py', 'dora.py', 'dora_row.py']
    manifest = {
        'schema': 'dora-next-extraction-v1', 'task': 'ViGGO',
        'model': task.MODEL, 'model_revision': task.MODEL_REVISION,
        'dataset': data_provenance,
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'control_only'},
        'recipes': {method: ab.recipe_manifest(method) for method in args.methods},
        'source_sha256': {name: sha(name) for name in sources},
        'validation_gate_sha256': {name: sha(path) for name, path in gates.items()},
        'reference_manifest_sha256': sha(args.reference / 'manifest.json'),
        'reference_tuning_sha256': sha(args.reference / 'tuning.json'),
        'unchanged_task_loop': {'ast_identical_except_unused_cli_main': True,
                               'archived_source_sha256': sha(archived_task),
                               'current_source_sha256': sha('experiments/extraction.py')},
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(),
                        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')},
        'protocol': 'Six recipes from original+FP32 magnitude baseline; unchanged official ViGGO splits and deterministic subsets. '
                    'Each recipe receives two 48-step seed42 LR pilots; minimum validation NLL '
                    'selects LR. Finals use 256 updates, checkpoint by validation NLL, seeds42-44. '
                    'Fixed lane uses baseline-selected LR0.0003 for every recipe; tuned lane uses '
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
    control(model, records, tokenizer, args, sources)
    if args.control_only:
        return

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
        control = json.loads((args.reference / 'finals/ab_m_fp32_lr0.0003_seed42/result.json').read_text())
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
