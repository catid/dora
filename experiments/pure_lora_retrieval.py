"""Validation-selected pure-LoRA comparison for IniLoRA-alpha and NoRA.

This runner intentionally excludes DoRA from the method table.  It reuses the
already measured retrieval data/training functions, but injects the pure-LoRA
backend from ``inilora_nora_lora.py``.  Scale and learning-rate selection use
validation only; held-out test rankings are written only after every pilot is
complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from experiments import init_ablations as init
from experiments import inilora_nora_lora as pure
from experiments import nora_retrieval as old


SCALES = (0.25, 0.5, 1.0, 2.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def tensor_digest(named):
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def candidates():
    # Match the standard LoRA convention used by the task baselines: random
    # Kaiming down/input factor and zero up/output factor.  Using the
    # historical accepted initializer (random up, zero down) for NoRA would
    # deliberately start every normalized column in the epsilon branch.
    standard_down = init.InitSpec(mode='controlled', orientation='down',
                                  family='uniform', scaling='fan_in',
                                  scale=1 / (3 ** .5))
    result = [
        {'id': 'lora', 'label': 'LoRA', 'nora': False,
         'spec': standard_down},
        {'id': 'nora', 'label': 'NoRA', 'nora': True,
         'spec': standard_down},
    ]
    for scale in SCALES:
        token = f'{scale:g}'
        spec = init.InitSpec(mode='controlled', family='normal',
                             both_nonzero=True, scaling='absolute',
                             scale=scale, companion_scale=scale)
        result.extend([
            {'id': f'inilora_alpha_s{token}',
             'label': f'IniLoRA-alpha s={token}', 'nora': False, 'spec': spec},
            {'id': f'inilora_alpha_nora_s{token}',
             'label': f'IniLoRA-alpha + NoRA s={token}', 'nora': True, 'spec': spec},
        ])
    return result


def backend(candidate, seed):
    return pure.backend(candidate['spec'], factor_seed=seed,
                        nora=candidate['nora'])


def recipe(candidate, seed=42):
    return backend(candidate, seed).recipe_manifest(candidate['id'])


def new_model(candidate, seed):
    old.task.seed_everything(seed)
    model = AutoModel.from_pretrained(old.task.MODEL,
        revision=old.task.MODEL_REVISION, dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda()
    old.task.seed_everything(seed)
    return backend(candidate, seed).inject_adapters(
        model, candidate['id'], rank=8, targets=('query', 'value'))


def train_fit(model, tokenizer, data, negatives, args, candidate, seed, lr,
              directory):
    args.current_method, args.current_rank, args.learning_rate = candidate['id'], 8, lr
    if args.task == 'scifact':
        ds = data['scifact']
        return old.task.train(model, tokenizer, ds['corpus_tokens'],
            ds['query_tokens'], ds['relevance'], ds['splits'], args, seed,
            directory)
    return old.harder.train(model, tokenizer, data['nfcorpus'], negatives,
        args, candidate['id'], seed, lr, 1.0, directory,
        adapter_backend=backend(candidate, seed))


def evaluate(model, tokenizer, data, args, directory, *, final=False):
    result = {}
    for name, ds in data.items():
        splits = ('validation', 'test') if final and 'validation' in ds['splits'] else ('test',) if final else ('validation',)
        if not final and 'validation' not in ds['splits']:
            continue
        destination = directory / name
        destination.mkdir(parents=True, exist_ok=True)
        result[name] = old.task.evaluate(model, tokenizer, ds['corpus_tokens'],
            ds['query_tokens'], ds['relevance'], ds['splits'], args,
            destination, splits)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=('scifact', 'nfcorpus'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    if args.reference is None:
        args.reference = Path('/var/tmp/dora-bench/original_bench') / args.task
    reference = read(args.reference / 'provenance.json')
    cfg = reference['configuration']
    for name in ('epochs', 'batch_size', 'eval_batch_size', 'max_length',
                 'weight_decay', 'temperature', 'warmup_fraction',
                 'max_grad_norm', 'threads'):
        setattr(args, name, cfg[name])
    args.rank, args.device = 8, 'cuda:0'
    args.learning_rates = cfg['learning_rates']
    args.fixed_lr = .001 if args.task == 'scifact' else .0001
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')

    if args.task == 'scifact':
        args.validation_fraction, args.split_seed = cfg['validation_fraction'], cfg['split_seed']
        corpus, queries, relevance, splits, metadata = old.task.load_data(args)
        assert metadata == reference['dataset']
        assert splits == read(args.reference / 'splits.json')
        data = {'scifact': {'corpus': corpus, 'queries': queries,
            'relevance': relevance, 'splits': splits, 'metadata': metadata}}
        negatives = None
    else:
        args.hard_negatives = cfg['hard_negatives']
        data = old.harder.load_data()
        for name, ds in data.items():
            assert ds['metadata'] == reference['datasets'][name]
            assert ds['splits'] == read(args.reference / f'{name}_splits.json')
        negatives = read(args.reference / 'hard_negatives.json')
        shutil.copy2(args.reference / 'hard_negatives.json', args.output / 'hard_negatives.json')

    tokenizer = AutoTokenizer.from_pretrained(old.task.MODEL,
                                              revision=old.task.MODEL_REVISION)
    for name, ds in data.items():
        ds['corpus_tokens'] = old.task.tokenize_all(tokenizer, ds['corpus'], args.max_length)
        ds['query_tokens'] = old.task.tokenize_all(tokenizer, ds['queries'], args.max_length)
        write(args.output / f'{name}_splits.json', ds['splits'])
        write(args.output / f'{name}_qrels.json', {qid: ds['relevance'][qid]
            for ids in ds['splits'].values() for qid in ids})

    candidates_by_id = {row['id']: row for row in candidates()}
    checkpoint = hf_hub_download(old.task.MODEL, 'model.safetensors',
                                 revision=old.task.MODEL_REVISION)
    source_names = ['experiments/pure_lora_retrieval.py',
                    'experiments/inilora_nora_lora.py',
                    'experiments/init_ablations.py',
                    'experiments/change_ablations.py',
                    'experiments/nora_retrieval.py',
                    'experiments/retrieval.py',
                    'experiments/second_round/retrieval.py',
                    'experiments/adapters.py', 'dora.py', 'dora_row.py']
    manifest = {
        'schema': 'pure-lora-inilora-nora-retrieval-v1',
        'task': args.task, 'rank': 8, 'seeds': args.seeds,
        'model': {'repository': old.task.MODEL,
                  'revision': old.task.MODEL_REVISION,
                  'checkpoint_sha256': sha(checkpoint),
                  'frozen_dtype': 'bfloat16'},
        'configuration': {name: getattr(args, name) for name in
            ('epochs', 'batch_size', 'eval_batch_size', 'max_length',
             'weight_decay', 'temperature', 'warmup_fraction',
             'max_grad_norm', 'learning_rates', 'fixed_lr')},
        'datasets': {name: ds['metadata'] for name, ds in data.items()},
        'scales': list(SCALES),
        'candidates': [{**{k: v for k, v in row.items() if k != 'spec'},
                        'spec': {'mode': row['spec'].mode,
                                 'family': row['spec'].family,
                                 'both_nonzero': row['spec'].both_nonzero,
                                 'scaling': row['spec'].scaling,
                                 'scale': row['spec'].scale,
                                 'companion_scale': row['spec'].companion_scale}}
                       for row in candidates_by_id.values()],
        'source_sha256': {name: sha(name) for name in source_names},
        'reference_provenance_sha256': sha(args.reference / 'provenance.json'),
        'protocol': 'Pure LoRA only. LoRA and NoRA one-zero controls plus centered '
                    'IniLoRA-alpha and IniLoRA-alpha+NoRA scale grid. Every scale '
                    'and learning-rate pilot completes before validation selection; '
                    'test evaluation is restricted to selected final fits. No DoRA '
                    'magnitude or adapted-weight normalization is used.',
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(),
                        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
    }
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest
    else:
        write(manifest_path, manifest)
    for name in source_names:
        destination = args.output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha(destination) == sha(name)
        else:
            shutil.copy2(name, destination)

    # All validation pilots are completed before any held-out evaluation.
    pilots = {}
    for candidate in candidates_by_id.values():
        trials = []
        for lr in sorted(args.learning_rates):
            fit_id = f'pilots/{candidate["id"]}/lr_{lr:g}'
            directory = args.output / fit_id
            result_path = directory / 'result.json'
            if result_path.exists():
                row = read(result_path)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                model = new_model(candidate, 42)
                frozen = tensor_digest((n, p) for n, p in model.named_parameters()
                                       if not p.requires_grad)
                training = train_fit(model, tokenizer, data, negatives, args,
                                     candidate, 42, lr, directory)
                assert frozen == tensor_digest((n, p) for n, p in model.named_parameters()
                                               if not p.requires_grad)
                evaluation = evaluate(model, tokenizer, data, args, directory,
                                     final=False)
                row = {'status': 'pass', 'candidate_id': candidate['id'],
                       'label': candidate['label'], 'seed': 42, 'lr': lr,
                       'fit_id': fit_id, 'training': training,
                       'evaluation': evaluation,
                       'frozen_sha256': frozen,
                       'trainable_parameters': sum(p.numel() for p in model.parameters()
                                                   if p.requires_grad),
                       'recipe': recipe(candidate),
                       'diagnostics': backend(candidate, 42).assert_no_clamp(model),
                       'checkpoint_sha256': sha(directory / 'adapter.pt'),
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
                write(result_path, row)
                del model
                torch.cuda.empty_cache()
            value_key = args.task + '.validation.ndcg_at_10'
            metric = row['evaluation'][args.task]['validation']['ndcg_at_10']
            trials.append({'lr': lr, 'metric': metric,
                           'fit_id': row['fit_id'],
                           'checkpoint_sha256': row['checkpoint_sha256']})
        winner = max(trials, key=lambda x: (x['metric'], -x['lr']))
        pilots[candidate['id']] = {'candidate_id': candidate['id'],
                                   'label': candidate['label'],
                                   'selected_lr': winner['lr'],
                                   'selected_validation': winner['metric'],
                                   'trials': trials}
        write(args.output / 'pilot_selection.json', pilots)

    # Keep all scales in the pilot grid, but only carry the selected member of
    # each initializer family into the three-seed final comparison.
    selected_ids = {'lora', 'nora'}
    for prefix in ('inilora_alpha_s', 'inilora_alpha_nora_s'):
        family = [row for row in candidates_by_id.values()
                  if row['id'].startswith(prefix)]
        selected_ids.add(max(family,
                             key=lambda row: pilots[row['id']]['selected_validation'])['id'])
    finals = []
    for candidate in candidates_by_id.values():
        if candidate['id'] not in selected_ids:
            continue
        selected_lr = pilots[candidate['id']]['selected_lr']
        for seed in args.seeds:
            fit_id = f'finals/{candidate["id"]}/lr_{selected_lr:g}/seed_{seed}'
            directory = args.output / fit_id
            path = directory / 'result.json'
            if path.exists():
                row = read(path)
            else:
                directory.mkdir(parents=True, exist_ok=True)
                model = new_model(candidate, seed)
                frozen = tensor_digest((n, p) for n, p in model.named_parameters()
                                       if not p.requires_grad)
                training = train_fit(model, tokenizer, data, negatives, args,
                                     candidate, seed, selected_lr, directory)
                assert frozen == tensor_digest((n, p) for n, p in model.named_parameters()
                                               if not p.requires_grad)
                evaluation = evaluate(model, tokenizer, data, args, directory,
                                     final=True)
                row = {'status': 'pass', 'candidate_id': candidate['id'],
                       'label': candidate['label'], 'seed': seed,
                       'lr': selected_lr, 'fit_id': fit_id, 'training': training,
                       'evaluation': evaluation, 'frozen_sha256': frozen,
                       'recipe': recipe(candidate, seed),
                       'diagnostics': backend(candidate, seed).assert_no_clamp(model),
                       'checkpoint_sha256': sha(directory / 'adapter.pt'),
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
                write(path, row)
                del model
                torch.cuda.empty_cache()
            finals.append(row)
            write(args.output / 'final_results.json', finals)

    write(args.output / 'completion.json', {
        'status': 'complete', 'task': args.task,
        'candidate_count': len(candidates_by_id), 'pilot_count': len(pilots),
        'final_count': len(finals), 'test_used_for_selection': False,
        'manifest_sha256': sha(manifest_path),
    })


if __name__ == '__main__':
    main()
