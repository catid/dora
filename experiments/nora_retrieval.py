"""Four-cell initializer/NoRA study on accepted original plus FP32 magnitudes.

Delegates optimization, batches, encoding and scoring to the measured task
functions. A fit is uniquely identified by recipe, rate and seed; full-budget
seed42 validation fits and identical fixed/tuned fits are explicitly shared.
"""

import argparse
import copy
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

from experiments import nora_current as ab
from experiments import next_ablations as previous
from experiments import change_ablations as historical
from experiments import retrieval as task
from experiments.second_round import retrieval as harder


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


def new_model(method, seed, *, reference=None):
    task.seed_everything(seed)
    model = AutoModel.from_pretrained(task.MODEL, revision=task.MODEL_REVISION,
                                      dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    task.seed_everything(seed)
    if reference is not None:
        assert reference in ('next_base', 'next_init')
        return previous.inject_adapters(model, reference, rank=8, targets=('query', 'value'))
    return ab.inject_adapters(model, method, rank=8, targets=('query', 'value'))


def train_fit(model, tokenizer, data, negatives, args, method, seed, lr, directory, reference=None):
    args.current_method, args.current_rank, args.learning_rate = method, 8, lr
    if args.task == 'scifact':
        ds = data['scifact']
        return task.train(model, tokenizer, ds['corpus_tokens'], ds['query_tokens'], ds['relevance'], ds['splits'], args, seed, directory)
    return harder.train(model, tokenizer, data['nfcorpus'], negatives, args, method, seed, lr, 1.0,
                        directory, adapter_backend=ab)


def evaluate(model, tokenizer, data, args, directory, *, final=False):
    result = {}
    for name, ds in data.items():
        splits = ('validation', 'test') if final and 'validation' in ds['splits'] else ('test',) if final else ('validation',)
        if not final and 'validation' not in ds['splits']:
            continue
        destination = directory / name
        destination.mkdir(parents=True, exist_ok=True)
        result[name] = task.evaluate(model, tokenizer, ds['corpus_tokens'], ds['query_tokens'], ds['relevance'], ds['splits'], args, destination, splits)
    return result


def endpoint_control(tokenizer, data, negatives, args, manifest):
    """Five real MiniLM steps: exact baseline/init controls and finite NoRA trajectories."""
    destination = args.output / 'endpoint_control'
    gate_path = destination / 'control.json'
    if gate_path.exists():
        result = read(gate_path)
        assert result['status'] == 'pass' and result['manifest_sha256'] == sha(args.output / 'manifest.json')
        for name, expected in result['evidence_sha256'].items():
            assert sha(destination / name) == expected, name
        return result
    configuration = copy.copy(args)
    configuration.epochs = 1
    subset = copy.deepcopy(data)
    train_dataset = 'scifact' if args.task == 'scifact' else 'nfcorpus'
    subset[train_dataset]['splits']['train'] = subset[train_dataset]['splits']['train'][:160]
    controls = []
    for method, reference in [('current_dora','next_base'), ('init_dora','next_init'),
                              ('current_dora_nora',None), ('init_dora_nora',None)]:
        records = []
        for source in ((reference,None) if reference else (None,)):
            label = source or method
            directory = destination / label
            directory.mkdir(parents=True, exist_ok=True)
            model = new_model(method, 42, reference=source)
            frozen = tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
            initialized = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
            training = train_fit(model, tokenizer, subset, negatives, configuration, method, 42,
                                 args.fixed_lr, directory, reference=source)
            assert training['steps'] == 5
            assert frozen == tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
            diagnostics = (previous if source else ab).assert_no_clamp(model)
            rows = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
            for row in rows:
                row.pop('elapsed_seconds')
                assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
            state = torch.load(directory / 'adapter.pt', map_location='cpu', weights_only=True)
            assert all(torch.isfinite(tensor).all() for tensor in state.values())
            records.append({'rows': rows, 'initial': initialized, 'state': state,
                            'frozen': frozen, 'diagnostics': diagnostics, 'rng': rng})
            del model
            torch.cuda.empty_cache()
        current = records[-1]
        if reference:
            old = records[0]
            assert old['rows'] == current['rows'], f'{method}: loss/rate/clip trajectory changed'
            assert old['frozen'] == current['frozen']
            for key in old['rng']: assert torch.equal(old['rng'][key], current['rng'][key])
            for field in ('initial','state'):
                assert list(old[field]) == list(current[field])
                for name,tensor in old[field].items():
                    actual = current[field][name]
                    assert actual.dtype == tensor.dtype and actual.shape == tensor.shape
                    torch.testing.assert_close(actual,tensor,rtol=0,atol=0)
        controls.append({'method': method, 'reference': reference, 'steps': 5,
                         'logged_loss_learning_rate_clip_norm_bitwise': bool(reference),
                         'initialized_and_final_adapter_tensors_bitwise': bool(reference),
                         'initial_rng_bitwise': bool(reference),
                         'finite_trajectory_and_parameters': True, 'tensor_count': len(current['state']),
                         'first_step_gradient_norm': current['rows'][0]['gradient_norm'],
                         'frozen_sha256': current['frozen'], 'diagnostics': current['diagnostics']})
    result = {'status':'pass','task':args.task,'manifest_sha256':sha(args.output/'manifest.json'),
              'core_source_sha256':ab.SOURCE_SHA256,'controls':controls,
              'scope':'MiniLM BF16 base, actual task loop and fixed160-query subset. No-NoRA baseline and initializer control exactly reproduce next_base/next_init. NoRA treatments require finite five-step clipped AdamW trajectories; literal zero down-factor normalization epsilon usage is recorded and intentionally allowed. No downstream score inference.'}
    result['evidence_sha256'] = {str(path.relative_to(destination)):sha(path) for path in destination.rglob('*') if path.is_file() and path != gate_path}
    write(gate_path,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=('scifact', 'nfcorpus'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--methods', nargs='+', choices=ab.METHODS, default=list(ab.METHODS))
    parser.add_argument('--gate-evidence', type=Path, default=Path('/var/tmp/dora-bench/nora_current_validation.json'))
    parser.add_argument('--control-only', action='store_true')
    args = parser.parse_args()
    if args.reference is None:
        args.reference = Path('/var/tmp/dora-bench/original_bench') / args.task
    gate = read(args.gate_evidence)
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    assert gate['core_source_sha256'] == sha(Path(ab.__file__)) == ab.SOURCE_SHA256
    assert gate['inherited_next_source_sha256'] == sha(Path(previous.__file__)) == previous.SOURCE_SHA256
    assert gate['inherited_core_source_sha256'] == sha(Path(historical.__file__)) == historical.SOURCE_SHA256
    assert gate['standalone_source_sha256'] == sha(Path('dora.py'))
    assert gate['recipes'] == {method: ab.recipe_manifest(method) for method in ab.METHODS}
    reference = read(args.reference / 'provenance.json')
    cfg = reference['configuration']
    for name in ('epochs', 'batch_size', 'eval_batch_size', 'max_length', 'weight_decay', 'temperature',
                 'warmup_fraction', 'max_grad_norm', 'threads'):
        setattr(args, name, cfg[name])
    args.rank, args.device, args.seeds = 8, 'cuda:0', [42, 43, 44]
    args.learning_rates = cfg['learning_rates']
    args.fixed_lr = .001 if args.task == 'scifact' else .0001
    old_results = read(args.reference / 'results.json')
    selection_key = 'dora_original' if args.task == 'scifact' else 'dora_original_r8'
    assert old_results['tuning'][selection_key]['selected_learning_rate'] == args.fixed_lr
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    if args.task == 'scifact':
        args.validation_fraction, args.split_seed = cfg['validation_fraction'], cfg['split_seed']
        corpus, queries, relevance, splits, metadata = task.load_data(args)
        assert metadata == reference['dataset'] and splits == read(args.reference / 'splits.json')
        data = {'scifact': {'corpus': corpus, 'queries': queries, 'relevance': relevance,
                            'splits': splits, 'metadata': metadata}}
        negatives = None
    else:
        args.hard_negatives = cfg['hard_negatives']
        data = harder.load_data()
        for name, ds in data.items():
            assert ds['metadata'] == reference['datasets'][name]
            assert ds['splits'] == read(args.reference / f'{name}_splits.json')
        negatives = read(args.reference / 'hard_negatives.json')
        assert set(negatives) == set(data['nfcorpus']['splits']['train'])
        for qid, docs in negatives.items():
            assert len(docs) == args.hard_negatives and not set(docs) & set(data['nfcorpus']['relevance'][qid])
        shutil.copy2(args.reference / 'hard_negatives.json', args.output / 'hard_negatives.json')
    tokenizer = AutoTokenizer.from_pretrained(task.MODEL, revision=task.MODEL_REVISION)
    for name, ds in data.items():
        ds['corpus_tokens'] = task.tokenize_all(tokenizer, ds['corpus'], args.max_length)
        ds['query_tokens'] = task.tokenize_all(tokenizer, ds['queries'], args.max_length)
        write(args.output / f'{name}_splits.json', ds['splits'])
        write(args.output / f'{name}_qrels.json', {qid: ds['relevance'][qid] for ids in ds['splits'].values() for qid in ids})
    sources = ['experiments/nora_retrieval.py', 'experiments/nora_current.py', 'experiments/next_ablations.py', 'experiments/change_ablations.py', 'experiments/retrieval.py',
               'experiments/second_round/retrieval.py', 'experiments/adapters.py', 'experiments/second_round_adapters.py',
               'experiments/original_dora.py', 'dora_row.py', 'dora.py']
    checkpoint = hf_hub_download(task.MODEL, 'model.safetensors', revision=task.MODEL_REVISION)
    manifest = {'schema': 'dora-current-nora-retrieval-v1', 'task': args.task, 'rank': 8,
                'model': {'repository': task.MODEL, 'revision': task.MODEL_REVISION, 'checkpoint_sha256': sha(checkpoint), 'frozen_dtype': 'bfloat16'},
                'configuration': {name: getattr(args, name) for name in ('epochs','batch_size','eval_batch_size','max_length','weight_decay','temperature','warmup_fraction','max_grad_norm','seeds','learning_rates','fixed_lr','methods')},
                'datasets': {name: ds['metadata'] for name, ds in data.items()},
                'input_sha256': {path.name: sha(path) for path in args.output.glob('*_*.json') if path.name.endswith(('_splits.json','_qrels.json'))},
                'hard_negatives_sha256': sha(args.output / 'hard_negatives.json') if negatives is not None else None,
                'reference_provenance_sha256': sha(args.reference / 'provenance.json'),
                'reference_results_sha256': sha(args.reference / 'results.json'),
                'gate_evidence_sha256': sha(args.gate_evidence),
                'recipes': {method: ab.recipe_manifest(method) for method in args.methods},
                'source_sha256': {name: sha(name) for name in sources},
                'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'matmul_precision': torch.get_float32_matmul_precision()},
                'protocol': 'Same pinned checkpoint/splits/batches/five epochs/precision/task loop as original run. Every recipe gets the full original LR grid at seed42, validation nDCG selects LR with smaller-LR tie break. Fixed lane uses the prespecified original-selected LR; fresh FP32-magnitude baseline is independently validation-tuned as well. Final seeds42-44; identical recipe/seed/LR/budget fits explicitly shared. All candidate fits precede test evaluation. Seed42 full-budget validation fits are reused. No norm-floor activations accepted. Runtime includes on-device weight/down-factor diagnostics. NoRA uses differentiable rank-axis down-column normalization eps1e-12; zero-initialized down-factor epsilon use is explicit and accepted, never silently replaced by another initializer.'}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        assert read(manifest_path) == manifest, 'Source/configuration/input/gate changed: use a fresh output'
    else:
        write(manifest_path, manifest)
    for name in sources:
        destination = args.output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    for path in (args.gate_evidence, args.reference / 'provenance.json', args.reference / 'results.json', args.reference / 'task_audit.json'):
        if path.exists():
            destination = args.output / 'reference' / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    endpoint_control(tokenizer, data, negatives, args, manifest)
    if args.control_only:
        return
    manifest_sha = sha(manifest_path)
    result_path = args.output / 'results.json'
    results = read(result_path) if result_path.exists() else {'schema': 'dora-current-nora-retrieval-v1', 'task': args.task, 'rank': 8, 'manifest_sha256': manifest_sha, 'tuning': {}, 'runs': [], 'complete': False}
    assert results['manifest_sha256'] == manifest_sha
    validation_dataset = 'scifact' if args.task == 'scifact' else 'nfcorpus'

    def fit(method, seed, rate, final=False):
        fit_id = f'fits/{method}/lr_{rate:g}/seed_{seed}'
        directory = args.output / fit_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'result.json'
        existing = read(path) if path.exists() else None
        if existing is not None:
            assert existing['manifest_sha256'] == manifest_sha and existing['recipe_sha256'] == ab.recipe_manifest(method)['recipe_sha256']
            assert sha(directory / 'adapter.pt') == existing['checkpoint_sha256']
            if not final or 'test' in existing['evaluation'][validation_dataset]:
                return existing
        model = new_model(method, seed)
        if existing is None:
            frozen = tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
            record = {'method': method, 'seed': seed, 'rank': 8, 'learning_rate': rate, 'fit_id': fit_id,
                      'manifest_sha256': manifest_sha, 'recipe_sha256': ab.recipe_manifest(method)['recipe_sha256'],
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'trainable_parameter_order': [n for n, p in model.named_parameters() if p.requires_grad],
                      'initial_trainable_sha256': tensor_digest((n, p) for n, p in model.named_parameters() if p.requires_grad),
                      'frozen_sha256': frozen, 'started_utc': datetime.now(timezone.utc).isoformat()}
            record['training'] = train_fit(model, tokenizer, data, negatives, args, method, seed, rate, directory)
            assert frozen == tensor_digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
            record['frozen_weights_unchanged'] = True
            record['checkpoint_sha256'] = sha(directory / 'adapter.pt')
            record['evaluation'] = evaluate(model, tokenizer, data, args, directory, final=final)
            record['diagnostics'] = ab.assert_no_clamp(model)
        else:
            record = existing
            state = torch.load(directory / 'adapter.pt', map_location='cuda:0', weights_only=True)
            assert set(state) == {n for n, p in model.named_parameters() if p.requires_grad}
            model.load_state_dict(state, strict=False)
            record['evaluation'] = evaluate(model, tokenizer, data, args, directory, final=True)
            record['post_reload_diagnostics'] = ab.assert_no_clamp(model)
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(path, record)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({'event': 'fit', 'method': method, 'seed': seed, 'lr': rate, 'final': final, 'evaluation': record['evaluation']}), flush=True)
        return record

    for method in args.methods:
        trials = []
        for rate in sorted(args.learning_rates):
            row = fit(method, 42, rate)
            trials.append({'fit_id': row['fit_id'], 'learning_rate': rate, 'validation_ndcg_at_10': row['evaluation'][validation_dataset]['validation']['ndcg_at_10'],
                           'training_steps': row['training']['steps'], 'checkpoint_sha256': row['checkpoint_sha256']})
        winner = max(trials, key=lambda row: (row['validation_ndcg_at_10'], -row['learning_rate']))
        tuning = {'selected_learning_rate': winner['learning_rate'], 'fixed_learning_rate': args.fixed_lr, 'trials': trials,
                  'selection_metric': f'{validation_dataset}.validation.ndcg_at_10', 'test_used_for_selection': False}
        if method in results['tuning']:
            assert results['tuning'][method] == tuning
        results['tuning'][method] = tuning
        write(args.output / 'selections' / f'{method}.json', tuning)
        write(result_path, results)
        for seed in args.seeds:
            for rate in sorted({args.fixed_lr, winner['learning_rate']}):
                row = fit(method, seed, rate, final=True)
                lanes = [lane for lane, value in (('fixed',args.fixed_lr),('tuned',winner['learning_rate'])) if rate == value]
                for lane in lanes:
                    record = {**row, 'lane': lane, 'shared_lanes': lanes, 'reused_full_budget_validation_fit': seed == 42,
                              'reuse_note': 'Same fit_id means one measured recipe/seed/rate/budget fit, never an independent replicate.'}
                    results['runs'] = [r for r in results['runs'] if (r['method'],r['seed'],r['lane']) != (method,seed,lane)]
                    results['runs'].append(record)
                write(result_path, results)
    assert len(results['runs']) == len(args.methods) * 3 * 2
    results['complete'] = True
    results['unique_final_fits'] = len({row['fit_id'] for row in results['runs']})
    results['unique_trained_fits'] = len(list((args.output / 'fits').glob('*/*/*/adapter.pt')))
    write(result_path, results)
    print(json.dumps({'event':'complete','task':args.task,'runs':len(results['runs']),'unique_final_fits':results['unique_final_fits']}),flush=True)


if __name__ == '__main__':
    main()
