"""Pure-LoRA IniLoRA/NoRA scale sweep on FGVC-Aircraft."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from huggingface_hub import hf_hub_download

from experiments import init_ablations as init
from experiments import inilora_nora_lora as pure
from experiments.second_round import vision


SCALES = (0.25, 0.5, 1.0, 2.0)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def candidates():
    standard_down = init.InitSpec(mode='controlled', orientation='down',
                                  family='uniform', scaling='fan_in',
                                  scale=1 / (3 ** .5))
    result = [
        {'id': 'lora', 'label': 'LoRA', 'nora': False, 'spec': standard_down},
        {'id': 'nora', 'label': 'NoRA', 'nora': True, 'spec': standard_down},
    ]
    for scale in SCALES:
        spec = init.InitSpec(mode='controlled', family='normal',
                             both_nonzero=True, scaling='absolute',
                             scale=scale, companion_scale=scale)
        result.extend([
            {'id': f'inilora_alpha_s{scale:g}', 'label': f'IniLoRA-alpha s={scale:g}',
             'nora': False, 'spec': spec},
            {'id': f'inilora_alpha_nora_s{scale:g}',
             'label': f'IniLoRA-alpha + NoRA s={scale:g}', 'nora': True, 'spec': spec},
        ])
    return result


def backend(candidate, seed):
    return pure.backend(candidate['spec'], factor_seed=seed, nora=candidate['nora'])


def recipe(candidate, seed=42):
    return backend(candidate, seed).recipe_manifest(candidate['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/var/tmp/dora-bench/cache/aircraft'))
    parser.add_argument('--reference-root', type=Path,
                        default=Path('/var/tmp/dora-bench/round2/vision/rank2_run'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.epochs, args.pilot_epochs, args.batch_size, args.head_lr, args.workers = 20, 10, 128, .001, 8
    args.reference_root = Path(args.reference_root)
    torch.set_num_threads(8); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    datasets, split_manifest = vision.datasets_and_manifest(args)
    checkpoint = hf_hub_download(vision.REPO, 'model.safetensors', revision=vision.REVISION,
                                 cache_dir='/var/tmp/dora-bench/cache/huggingface')
    reference = read(args.reference_root / 'protocol.json')['provenance']
    provenance = {'model': vision.MODEL, 'model_repository': vision.REPO,
                  'model_revision': vision.REVISION, 'checkpoint_sha256': sha(checkpoint),
                  'dataset': 'FGVC-Aircraft variants',
                  'official_split_sizes': split_manifest['official_sizes'],
                  'split_sizes': split_manifest['effective_sizes'],
                  'split_manifest_sha256': sha(args.output / 'split_manifest.json'),
                  'transform': 'Remove bottom20 copyright pixels; train RRC224 scale[0.7,1]+horizontalflip; eval resize256+center224; bicubic; mean/std0.5',
                  'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
                  'timm': __import__('timm').__version__,
                  'gpu': torch.cuda.get_device_name(0),
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'precision': 'FP32 base/adapters, BF16 autocast; TF32 permitted'}
    # Host package versions are recorded but are allowed to differ from the
    # prior baseline shard; tensor/checkpoint/split identity is the binding
    # parity requirement for this comparison.
    for key in ('model', 'model_revision', 'checkpoint_sha256',
                'split_manifest_sha256', 'transform'):
        assert provenance[key] == reference[key], key
    candidates_by_id = {row['id']: row for row in candidates()}
    manifest = {'schema': 'pure-lora-inilora-nora-aircraft-v1', 'task': 'aircraft',
                'rank': 2, 'seeds': args.seeds, 'scales': list(SCALES),
                'candidates': [{**{k: v for k, v in row.items() if k != 'spec'},
                                'spec': {'mode': row['spec'].mode,
                                         'orientation': row['spec'].orientation,
                                         'family': row['spec'].family,
                                         'both_nonzero': row['spec'].both_nonzero,
                                         'scaling': row['spec'].scaling,
                                         'scale': row['spec'].scale,
                                         'companion_scale': row['spec'].companion_scale}}
                               for row in candidates_by_id.values()],
                'configuration': {'epochs': args.epochs, 'pilot_epochs': args.pilot_epochs,
                                  'batch_size': args.batch_size, 'head_lr': args.head_lr,
                                  'learning_rates': list(vision.BASE_LRS)},
                'provenance': provenance,
                'source_sha256': {name: sha(name) for name in (
                    'experiments/pure_lora_aircraft.py', 'experiments/inilora_nora_lora.py',
                    'experiments/init_ablations.py', 'experiments/second_round/vision.py')},
                'protocol': 'Pure LoRA only; standard random-down/zero-up LoRA and NoRA '
                            'controls, centered IniLoRA-alpha and paired scale grid. '
                            'Validation macro accuracy then cross-entropy selects scale/LR; '
                            'only four selected candidates receive three-seed test finals. '
                            'No DoRA magnitude or adapted-weight normalization.'}
    path = args.output / 'manifest.json'
    if path.exists(): assert read(path) == manifest
    else: write(path, manifest)
    for name in manifest['source_sha256']:
        dest = args.output / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): assert sha(dest) == sha(name)
        else: shutil.copy2(name, dest)

    # The prior Aircraft shard was produced under a different torch/timm
    # build, so its randomly initialized classifier-head digest is not a
    # portable binding.  All four methods below share this run's seeded head;
    # checkpoint, split and transform identity were checked above.
    args.reference_root = None

    pilots = {}
    for candidate in candidates_by_id.values():
        trials = []
        for lr in vision.BASE_LRS:
            result = vision.run_one(args, datasets, checkpoint, provenance, candidate['id'],
                                    2, 42, lr, 1.0, 'pilot', backend=backend(candidate, 42))
            trials.append({'lr': lr, 'validation': result['best_validation'],
                           'result': result})
            write(args.output / 'pilot_results.json',
                  {**pilots, candidate['id']: {'trials': trials}})
        winner = max(trials, key=lambda x: (x['validation']['macro_class_accuracy'],
                                             -x['validation']['cross_entropy'], -x['lr']))
        pilots[candidate['id']] = {'candidate_id': candidate['id'], 'label': candidate['label'],
                                   'selected_lr': winner['lr'],
                                   'selected_validation': winner['validation'],
                                   'trials': [{k: v for k, v in row.items() if k != 'result'}
                                              for row in trials]}
        write(args.output / 'pilot_selection.json', pilots)

    selected_ids = {'lora', 'nora'}
    for prefix in ('inilora_alpha_s', 'inilora_alpha_nora_s'):
        family = [row for row in candidates_by_id.values() if row['id'].startswith(prefix)]
        selected_ids.add(max(family, key=lambda row: (
            pilots[row['id']]['selected_validation']['macro_class_accuracy'],
            -pilots[row['id']]['selected_validation']['cross_entropy']))['id'])
    finals = []
    for candidate in candidates_by_id.values():
        if candidate['id'] not in selected_ids: continue
        lr = pilots[candidate['id']]['selected_lr']
        for seed in args.seeds:
            result = vision.run_one(args, datasets, checkpoint, provenance, candidate['id'],
                                    2, seed, lr, 1.0, 'final', backend=backend(candidate, seed))
            finals.append({'candidate_id': candidate['id'], 'label': candidate['label'],
                           'seed': seed, 'lr': lr, 'result': result,
                           'recipe': recipe(candidate, seed)})
            write(args.output / 'final_results.json', finals)
    write(args.output / 'completion.json', {'status': 'complete',
        'selected_ids': sorted(selected_ids), 'pilot_count': len(pilots),
        'final_count': len(finals), 'test_used_for_selection': False,
        'manifest_sha256': sha(path)})


if __name__ == '__main__': main()
