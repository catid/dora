"""Search pure-LoRA IniLoRA/NoRA compatibility geometries on ViGGO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import extraction as task
from experiments import init_ablations as init
from experiments import inilora_nora_variants as pure


SCALES = (0.25, 0.5, 1.0, 2.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def candidates():
    result = []
    for scale in SCALES:
        spec = init.InitSpec(mode='controlled', family='normal',
                             both_nonzero=True, scaling='absolute',
                             scale=scale, companion_scale=scale)
        for mode in pure.MODES:
            result.append({'id': f'inilora_nora_{mode}_s{scale:g}',
                           'label': f'IniLoRA-alpha + NoRA[{mode}] s={scale:g}',
                           'mode': mode, 'spec': spec})
    return result


def backend(candidate, seed):
    return pure.backend(candidate['spec'], factor_seed=seed, mode=candidate['mode'])


def recipe(candidate, seed=42):
    return backend(candidate, seed).recipe_manifest(candidate['id'])


def model_and_fit(model, candidate, seed, lr, steps, records, tokenizer, args,
                  directory, final=False):
    return task.fit(model, candidate['id'], seed, lr, steps, records, tokenizer,
                    args, directory, evaluate_test=final,
                    backend=backend(candidate, seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-size', type=int, default=1024)
    parser.add_argument('--val-size', type=int, default=128)
    parser.add_argument('--test-size', type=int, default=256)
    parser.add_argument('--max-length', type=int, default=768)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--generation-batch-size', type=int, default=16)
    parser.add_argument('--pilot-steps', type=int, default=48)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=64)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[1e-4, 3e-4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(task.MODEL, revision=task.MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    records, provenance = task.data(args, tokenizer)
    candidates_by_id = {row['id']: row for row in candidates()}
    checkpoint = None
    manifest = {
        'schema': 'pure-lora-inilora-nora-variants-extraction-v1', 'task': 'ViGGO',
        'configuration': {key: getattr(args, key) for key in
            ('train_size', 'val_size', 'test_size', 'max_length', 'rank',
             'batch_size', 'generation_batch_size', 'pilot_steps', 'steps',
             'eval_every', 'learning_rates', 'seeds')},
        'dataset': provenance, 'scales': list(SCALES),
        'candidates': [{**{k: v for k, v in row.items() if k != 'spec'},
                        'spec': {'mode': row['spec'].mode,
                                 'orientation': row['spec'].orientation,
                                 'family': row['spec'].family,
                                 'both_nonzero': row['spec'].both_nonzero,
                                 'scaling': row['spec'].scaling,
                                 'scale': row['spec'].scale,
                                 'companion_scale': row['spec'].companion_scale}}
                       for row in candidates_by_id.values()],
        'source_sha256': {name: sha(name) for name in (
            'experiments/pure_lora_variant_extraction.py',
            'experiments/inilora_nora_variants.py', 'experiments/init_ablations.py',
            'experiments/extraction.py')},
        'protocol': 'Pure LoRA only; centered IniLoRA-alpha with five explicit '
                    'NoRA-compatible factor geometries and paired scale grid. '
                    'Validation target-token NLL selects geometry, scale and LR; '
                    'test generation occurs only for the selected final. '
                    'No DoRA magnitude or adapted-weight normalization.',
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(),
                        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')},
    }
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists(): assert read(manifest_path) == manifest
    else: write(manifest_path, manifest)
    for name in manifest['source_sha256']:
        destination = args.output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert sha(destination) == sha(name)
        else: shutil.copy2(name, destination)

    model = AutoModelForCausalLM.from_pretrained(
        task.MODEL, revision=task.MODEL_REVISION, dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().requires_grad_(False)
    pilots = {}
    for candidate in candidates_by_id.values():
        trials = []
        for lr in sorted(args.learning_rates):
            directory = args.output / 'pilots' / candidate['id'] / f'lr_{lr:g}'
            path = directory / 'result.json'
            if path.exists():
                row = read(path)
            else:
                result = model_and_fit(model, candidate, 42, lr, args.pilot_steps,
                                       records, tokenizer, args, directory)
                row = {'candidate_id': candidate['id'], 'label': candidate['label'],
                       'seed': 42, 'lr': lr, 'fit_id': str(directory.relative_to(args.output)),
                       'result': result, 'recipe': recipe(candidate),
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
                write(path, row)
            trials.append({'lr': lr, 'validation_nll': row['result']['validation_nll'],
                           'fit_id': row['fit_id']})
        winner = min(trials, key=lambda x: (x['validation_nll'], x['lr']))
        pilots[candidate['id']] = {'candidate_id': candidate['id'],
                                   'label': candidate['label'],
                                   'selected_lr': winner['lr'],
                                   'selected_validation_nll': winner['validation_nll'],
                                   'trials': trials}
        write(args.output / 'pilot_selection.json', pilots)

    selected_ids = {min(candidates_by_id.values(),
                        key=lambda row: pilots[row['id']]['selected_validation_nll'])['id']}

    finals = []
    for candidate in candidates_by_id.values():
        if candidate['id'] not in selected_ids: continue
        lr = pilots[candidate['id']]['selected_lr']
        for seed in args.seeds:
            directory = args.output / 'finals' / candidate['id'] / f'lr_{lr:g}' / f'seed_{seed}'
            path = directory / 'result.json'
            if path.exists():
                row = read(path)
            else:
                result = model_and_fit(model, candidate, seed, lr, args.steps,
                                       records, tokenizer, args, directory, final=True)
                row = {'candidate_id': candidate['id'], 'label': candidate['label'],
                       'seed': seed, 'lr': lr, 'fit_id': str(directory.relative_to(args.output)),
                       'result': result, 'recipe': recipe(candidate, seed),
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
                write(path, row)
            finals.append(row)
            write(args.output / 'final_results.json', finals)

    write(args.output / 'completion.json', {'status': 'complete',
        'selected_ids': sorted(selected_ids), 'pilot_count': len(pilots),
        'final_count': len(finals), 'test_used_for_selection': False,
        'manifest_sha256': sha(manifest_path)})


if __name__ == '__main__': main()
