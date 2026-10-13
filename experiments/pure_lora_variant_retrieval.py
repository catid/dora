"""Validation study for alternative pure-LoRA IniLoRA/NoRA mix geometries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from experiments import init_ablations as init
from experiments import inilora_nora_variants as variants
from experiments import pure_lora_retrieval as old
from experiments import retrieval as task
from experiments.second_round import retrieval as harder


SCALES = (0.25, 0.5, 1.0, 2.0)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def tensor_digest(named):
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def candidates():
    result = []
    for mode in variants.MODES:
        for scale in SCALES:
            spec = init.InitSpec(mode='controlled', family='normal',
                                 both_nonzero=True, scaling='absolute',
                                 scale=scale, companion_scale=scale)
            token = f'{scale:g}'
            result.append({
                'id': f'inilora_nora_{mode}_s{token}',
                'label': f'IniLoRA-alpha + NoRA[{mode}] s={token}',
                'mode': mode, 'spec': spec,
            })
    return result


def backend(candidate, seed):
    return variants.backend(candidate['spec'], factor_seed=seed,
                            mode=candidate['mode'])


def recipe(candidate, seed=42):
    return backend(candidate, seed).recipe_manifest(candidate['id'])


def new_model(candidate, seed):
    task.seed_everything(seed)
    model = AutoModel.from_pretrained(task.MODEL,
        revision=task.MODEL_REVISION, dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda()
    task.seed_everything(seed)
    return backend(candidate, seed).inject_adapters(
        model, candidate['id'], rank=8, targets=('query', 'value'))


def train_fit(model, tokenizer, data, negatives, args, candidate, seed, lr,
              directory):
    args.current_method, args.current_rank, args.learning_rate = candidate['id'], 8, lr
    if args.task == 'scifact':
        ds = data['scifact']
        return task.train(model, tokenizer, ds['corpus_tokens'],
            ds['query_tokens'], ds['relevance'], ds['splits'], args, seed,
            directory)
    return harder.train(model, tokenizer, data['nfcorpus'], negatives,
        args, candidate['id'], seed, lr, 1.0, directory,
        adapter_backend=backend(candidate, seed))


def evaluate(model, tokenizer, data, args, directory, *, final=False):
    result = {}
    for name, ds in data.items():
        splits = ('validation', 'test') if final and 'validation' in ds['splits'] else ('test',) if final else ('validation',)
        if not final and 'validation' not in ds['splits']:
            continue
        destination = directory / name; destination.mkdir(parents=True, exist_ok=True)
        result[name] = task.evaluate(model, tokenizer, ds['corpus_tokens'],
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
    reference = read(args.reference / 'provenance.json'); cfg = reference['configuration']
    for name in ('epochs', 'batch_size', 'eval_batch_size', 'max_length',
                 'weight_decay', 'temperature', 'warmup_fraction',
                 'max_grad_norm', 'threads'):
        setattr(args, name, cfg[name])
    args.rank, args.device = 8, 'cuda:0'; args.learning_rates = cfg['learning_rates']
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads); torch.set_float32_matmul_precision('highest')

    if args.task == 'scifact':
        args.validation_fraction, args.split_seed = cfg['validation_fraction'], cfg['split_seed']
        corpus, queries, relevance, splits, metadata = task.load_data(args)
        assert metadata == reference['dataset']; assert splits == read(args.reference / 'splits.json')
        data = {'scifact': {'corpus': corpus, 'queries': queries, 'relevance': relevance,
                            'splits': splits, 'metadata': metadata}}
        negatives = None
    else:
        args.hard_negatives = cfg['hard_negatives']; data = harder.load_data()
        for name, ds in data.items():
            assert ds['metadata'] == reference['datasets'][name]
            assert ds['splits'] == read(args.reference / f'{name}_splits.json')
        negatives = read(args.reference / 'hard_negatives.json')
        shutil.copy2(args.reference / 'hard_negatives.json', args.output / 'hard_negatives.json')

    tokenizer = AutoTokenizer.from_pretrained(task.MODEL, revision=task.MODEL_REVISION)
    for name, ds in data.items():
        ds['corpus_tokens'] = task.tokenize_all(tokenizer, ds['corpus'], args.max_length)
        ds['query_tokens'] = task.tokenize_all(tokenizer, ds['queries'], args.max_length)
        write(args.output / f'{name}_splits.json', ds['splits'])
        write(args.output / f'{name}_qrels.json', {qid: ds['relevance'][qid]
              for ids in ds['splits'].values() for qid in ids})

    rows = candidates(); by_id = {row['id']: row for row in rows}
    checkpoint = hf_hub_download(task.MODEL, 'model.safetensors', revision=task.MODEL_REVISION)
    source_names = ['experiments/pure_lora_variant_retrieval.py',
                    'experiments/inilora_nora_variants.py',
                    'experiments/pure_lora_retrieval.py',
                    'experiments/init_ablations.py', 'experiments/change_ablations.py',
                    'experiments/nora_retrieval.py', 'experiments/retrieval.py',
                    'experiments/second_round/retrieval.py', 'experiments/adapters.py']
    manifest = {
        'schema': 'pure-lora-inilora-nora-variant-retrieval-v1', 'task': args.task,
        'rank': 8, 'seeds': args.seeds, 'modes': list(variants.MODES),
        'scales': list(SCALES), 'model': {'repository': task.MODEL,
        'revision': task.MODEL_REVISION, 'checkpoint_sha256': sha(checkpoint),
        'frozen_dtype': 'bfloat16'},
        'configuration': {name: getattr(args, name) for name in
            ('epochs', 'batch_size', 'eval_batch_size', 'max_length', 'weight_decay',
             'temperature', 'warmup_fraction', 'max_grad_norm', 'learning_rates')},
        'datasets': {name: ds['metadata'] for name, ds in data.items()},
        'source_sha256': {name: sha(name) for name in source_names},
        'protocol': 'Validation-only geometry search over five pure-LoRA NoRA/IniLoRA '
                    'mixes and four scales. Test evaluation is restricted to the '
                    'best mode/scale per task, with a separately selected learning rate. '
                    'No DoRA magnitude or adapted-weight normalization.',
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(),
                        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
    }
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists(): assert read(manifest_path) == manifest
    else: write(manifest_path, manifest)
    for name in source_names:
        destination = args.output / 'source' / name; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): assert sha(destination) == sha(name)
        else: shutil.copy2(name, destination)

    pilots = {}
    for candidate in rows:
        trials = []
        for lr in sorted(args.learning_rates):
            fit_id = f'pilots/{candidate["id"]}/lr_{lr:g}'; directory = args.output / fit_id
            result_path = directory / 'result.json'
            if result_path.exists(): row = read(result_path)
            else:
                directory.mkdir(parents=True, exist_ok=True); model = new_model(candidate, 42)
                frozen = tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
                training = train_fit(model, tokenizer, data, negatives, args, candidate, 42, lr, directory)
                assert frozen == tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
                evaluation = evaluate(model, tokenizer, data, args, directory, final=False)
                row = {'status': 'pass', 'candidate_id': candidate['id'], 'label': candidate['label'],
                       'seed': 42, 'lr': lr, 'fit_id': fit_id, 'training': training,
                       'evaluation': evaluation, 'frozen_sha256': frozen,
                       'recipe': recipe(candidate), 'diagnostics': backend(candidate, 42).assert_no_clamp(model),
                       'checkpoint_sha256': sha(directory / 'adapter.pt'),
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
                write(result_path, row); del model; torch.cuda.empty_cache()
            metric = row['evaluation'][args.task]['validation']['ndcg_at_10']
            trials.append({'lr': lr, 'metric': metric, 'fit_id': row['fit_id'],
                           'checkpoint_sha256': row['checkpoint_sha256']})
        winner = max(trials, key=lambda x: (x['metric'], -x['lr']))
        pilots[candidate['id']] = {'candidate_id': candidate['id'], 'label': candidate['label'],
                                   'mode': candidate['mode'], 'scale': candidate['spec'].scale,
                                   'selected_lr': winner['lr'], 'selected_validation': winner['metric'],
                                   'trials': trials}
        write(args.output / 'pilot_selection.json', pilots)

    selected = max(pilots.values(), key=lambda x: (x['selected_validation'], -x['scale']))
    candidate = by_id[selected['candidate_id']]; finals = []
    for seed in args.seeds:
        lr = selected['selected_lr']; fit_id = f'finals/{candidate["id"]}/lr_{lr:g}/seed_{seed}'
        directory = args.output / fit_id; path = directory / 'result.json'
        if path.exists(): row = read(path)
        else:
            directory.mkdir(parents=True, exist_ok=True); model = new_model(candidate, seed)
            frozen = tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
            training = train_fit(model, tokenizer, data, negatives, args, candidate, seed, lr, directory)
            assert frozen == tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
            evaluation = evaluate(model, tokenizer, data, args, directory, final=True)
            row = {'status': 'pass', 'candidate_id': candidate['id'], 'label': candidate['label'],
                   'seed': seed, 'lr': lr, 'fit_id': fit_id, 'training': training,
                   'evaluation': evaluation, 'frozen_sha256': frozen,
                   'recipe': recipe(candidate, seed), 'diagnostics': backend(candidate, seed).assert_no_clamp(model),
                   'checkpoint_sha256': sha(directory / 'adapter.pt'),
                   'finished_utc': datetime.now(timezone.utc).isoformat()}
            write(path, row); del model; torch.cuda.empty_cache()
        finals.append(row); write(args.output / 'final_results.json', finals)
    write(args.output / 'completion.json', {'status': 'complete', 'task': args.task,
        'candidate_count': len(rows), 'pilot_count': len(pilots), 'final_count': len(finals),
        'selected_candidate': selected, 'test_used_for_selection': False,
        'manifest_sha256': sha(manifest_path)})


if __name__ == '__main__': main()
