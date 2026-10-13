"""One DoRA change recipe on the unchanged official Aircraft rank-2 protocol.

Each recipe receives four validation-only ten-epoch pilots, three twenty-epoch
fits at its selected rate, and three fits at original DoRA's selected LR .001.
When those rates coincide, the fixed-rate artifacts are byte-identical copies
with an explicit reuse manifest. Test scores never select hyperparameters.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace

import torch
import torchvision
import timm
from huggingface_hub import hf_hub_download

from experiments import change_ablations as changes
from experiments.second_round import vision
from experiments.vision import file_sha256, write_json, reuse_baseline_artifacts


def diagnostics(model):
    records = []
    for name, module in model.named_modules():
        if not isinstance(module, changes.AblationLinear):
            continue
        row = {"name": name, **module.assert_no_clamp()}
        factor = getattr(module, module.zero_initialized_factor)
        row['zero_initialized_factor_l2'] = factor.detach().norm().item()
        assert row['zero_initialized_factor_l2'] > 0, name
        assert row['clamp_activations'] == 0 and row['minimum_raw_norm'] > 0, name
        assert all(torch.isfinite(parameter).all() for parameter in module.parameters()), name
        records.append(row)
    assert len(records) == 48
    return records


BACKEND = SimpleNamespace(adapter_types=(changes.AblationLinear,),
                          inject_adapters=changes.inject_adapters,
                          parameter_groups=changes.parameter_groups,
                          adapter_diagnostics=diagnostics)


def copy_fit(source, destination):
    """Reuse an already measured, identical seed/rate/budget fit without relabeling."""
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ('config.json', 'result.json', 'epochs.jsonl', 'test_predictions.json', 'best_adapter_and_head.pt'):
        source_file, target = source / name, destination / name
        if target.exists():
            assert target.read_bytes() == source_file.read_bytes()
        else:
            shutil.copyfile(source_file, target)
        files[name] = file_sha256(target)
    result = json.loads((destination / 'result.json').read_text())
    assert result['adapter_learning_rate'] == .001 and result['epochs'] == 20
    return result, {'source_directory': str(source), 'destination_directory': str(destination),
                    'files_sha256': files, 'byte_identical': True,
                    'reason': 'Validation-selected LR equals prespecified original LR .001; same seed and full training protocol.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=changes.METHODS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/var/tmp/dora-bench/cache/aircraft'))
    parser.add_argument('--reference-root', type=Path, default=Path('/var/tmp/dora-bench/round2/vision/rank2_run'))
    parser.add_argument('--original-root', type=Path, default=Path('/var/tmp/dora-bench/original_bench/aircraft_rank2'))
    parser.add_argument('--gate-evidence', type=Path, required=True)
    parser.add_argument('--task-gate-evidence', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.epochs, args.pilot_epochs, args.batch_size, args.head_lr = 20, 10, 128, .001
    seeds, rank = (42, 43, 44), 2
    gate = json.loads(args.gate_evidence.read_text())
    current_core_sha = file_sha256('experiments/change_ablations.py')
    assert gate['status'] == 'pass' and gate['core_source_sha256'] == current_core_sha
    assert gate['cuda_tests']['status'] == 'pass'
    task_gate = json.loads(args.task_gate_evidence.read_text())
    assert task_gate['passed'] and task_gate['source_sha256']['experiments/change_ablations.py'] == current_core_sha
    assert {row['name'] for row in task_gate['cases']} == {'original_endpoint', 'revised_endpoint', 'original_magnitude_fp32_noop', 'cumulative_magnitude_fp32_noop'}
    assert all(row['passed'] for row in task_gate['cases'])
    for name, digest in task_gate['source_sha256'].items():
        assert file_sha256(name) == digest, name
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    old_manifest = args.reference_root / 'split_manifest.json'
    destination = args.output / 'split_manifest.json'
    if destination.exists():
        assert destination.read_bytes() == old_manifest.read_bytes()
    else:
        shutil.copyfile(old_manifest, destination)
    datasets, manifest = vision.datasets_and_manifest(args)
    checkpoint = hf_hub_download(vision.REPO, 'model.safetensors', revision=vision.REVISION,
                                cache_dir='/var/tmp/dora-bench/cache/huggingface')
    sources = ['experiments/change_aircraft.py', 'experiments/change_ablations.py',
               'experiments/change_aircraft_validate.py',
               'experiments/second_round/vision.py', 'experiments/second_round_adapters.py',
               'experiments/adapters.py', 'experiments/original_dora.py', 'experiments/vision.py', 'dora_row.py']
    recipe = changes.recipe_manifest(args.method)
    assert recipe['source_sha256'] == current_core_sha
    provenance = {'model': vision.MODEL, 'model_revision': vision.REVISION,
                  'checkpoint_sha256': file_sha256(checkpoint), 'dataset': 'FGVC-Aircraft variants',
                  'official_split_sizes': manifest['official_sizes'], 'split_sizes': manifest['effective_sizes'],
                  'split_manifest_sha256': file_sha256(args.output / 'split_manifest.json'),
                  'dataset_archive': json.loads((args.cache / 'download_provenance.json').read_text()),
                  'source_sha256': {name: file_sha256(name) for name in sources},
                  'transform': 'Remove bottom20 copyright pixels; train RRC224 scale[0.7,1]+horizontalflip; eval resize256+center224; bicubic; mean/std0.5',
                  'torch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': timm.__version__,
                  'python': platform.python_version(), 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'precision': 'FP32 base/adapters, BF16 autocast; TF32 permitted',
                  'command': [sys.executable, *sys.argv], 'recipe': recipe,
                  'core_parity_evidence_sha256': file_sha256(args.gate_evidence),
                  'task_parity_evidence_sha256': file_sha256(args.task_gate_evidence),
                  'reference_run_root': str(args.reference_root), 'original_run_root': str(args.original_root)}
    reference = json.loads((args.reference_root / 'protocol.json').read_text())['provenance']
    for name in ('model', 'model_revision', 'checkpoint_sha256', 'split_manifest_sha256', 'transform', 'torch', 'torchvision', 'timm', 'precision'):
        assert provenance[name] == reference[name], name
    original_selection = json.loads((args.original_root / 'rank_2/selected.json').read_text())
    assert original_selection['dora_original'] == {'lr': .001, 'magnitude_multiplier': 1.0}
    provenance['original_selection_sha256'] = file_sha256(args.original_root / 'rank_2/selected.json')
    for source, digest in provenance['source_sha256'].items():
        target = args.output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert file_sha256(target) == digest
        else:
            target.write_bytes(Path(source).read_bytes())
    for name, source in (('core_parity_evidence.json', args.gate_evidence),
                         ('task_parity_evidence.json', args.task_gate_evidence),
                         ('original_selected.json', args.original_root / 'rank_2/selected.json')):
        target = args.output / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copyfile(source, target)
    protocol = {'method': args.method, 'recipe': recipe, 'rank': rank, 'seeds': list(seeds),
                'pilot_seed': 42, 'pilot_epochs': 10, 'epochs': 20, 'batch_size': 128, 'head_lr': .001,
                'lr_grid': list(vision.BASE_LRS), 'fixed_original_lr': .001,
                'lanes': ['validation_tuned', 'fixed_original_lr'],
                'selection': 'Validation macro accuracy then cross-entropy; same four trials per recipe; no test selection.',
                'baseline_reuse': 'Historical trained-head controls copied byte-for-byte; every recipe starts from the same per-seed random classifier.',
                'timing_scope': 'Instrumented quality runs: every recipe tracks device norm minima and clamp counts; timings include that common diagnostic overhead and are not uninstrumented speed benchmarks.',
                'provenance': provenance}
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        write_json(protocol_path, protocol)
    baselines, reused = [], []
    for seed in seeds:
        row, attestation = reuse_baseline_artifacts(
            args.reference_root / 'baseline' / f'seed_{seed}', args.output / 'baseline' / f'seed_{seed}',
            {'seed': seed, 'epochs': 20, 'rank': 0, 'batch_size': 128, 'head_learning_rate': .001})
        baselines.append(row); reused.append(attestation)
    write_json(args.output / 'baseline_reuse.json', {'reused': reused})
    pilots = []
    for lr in vision.BASE_LRS:
        pilots.append(vision.run_one(args, datasets, checkpoint, provenance, args.method, rank, 42, lr, 1.0, 'pilot', backend=BACKEND))
        write_json(args.output / 'rank_2/pilot_results.json', pilots)
    winner = max(pilots, key=lambda row: (row['best_validation']['macro_class_accuracy'], -row['best_validation']['cross_entropy']))
    selected = winner['adapter_learning_rate']
    write_json(args.output / 'rank_2/selected.json', {args.method: {'lr': selected, 'magnitude_multiplier': 1.0}})
    finals = list(baselines)
    for seed in seeds:
        finals.append(vision.run_one(args, datasets, checkpoint, provenance, args.method, rank, seed, selected, 1.0, 'final', backend=BACKEND))
        write_json(args.output / 'results.json', finals)
    fixed_args = copy.copy(args)
    fixed_args.output = args.output / 'fixed_lr_0.001'
    fixed_args.output.mkdir(exist_ok=True)
    fixed, reuse = [], []
    for seed in seeds:
        if selected == .001:
            suffix = Path('rank_2/final') / args.method / f'seed_{seed}'
            row, attestation = copy_fit(args.output / suffix, fixed_args.output / suffix)
            reuse.append(attestation)
        else:
            row = vision.run_one(fixed_args, datasets, checkpoint, provenance, args.method, rank, seed, .001, 1.0, 'final', backend=BACKEND)
        fixed.append(row)
        write_json(fixed_args.output / 'results.json', fixed)
    write_json(fixed_args.output / 'fit_reuse.json', {'reused': reuse, 'selected_lr': selected, 'fixed_lr': .001})
    write_json(args.output / 'completion.json', {'status': 'complete', 'method': args.method,
               'recipe_sha256': recipe['recipe_sha256'], 'selected_lr': selected, 'pilots': 4,
               'tuned_finals': 3, 'fixed_lr_finals': 3, 'fixed_fits_reused': len(reuse),
               'new_epochs': 100 + (0 if selected == .001 else 60),
               'provenance_sha256': file_sha256(protocol_path)})


if __name__ == '__main__':
    main()
