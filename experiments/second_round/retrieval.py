"""Hard-negative NFCorpus adaptation and held-out SciFact transfer.

CUDA_VISIBLE_DEVICES=1 HF_HOME=/var/tmp/dora-bench/cache \
  /var/tmp/dora-bench/venv/bin/python -m experiments.second_round.retrieval

Six methods, ranks 2/8, three seeds; four equally long validation-only trials.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from experiments.retrieval import (
    MODEL, MODEL_REVISION, embeddings, encode, evaluate, make_batch,
    seed_everything, tokenize_all, write_json,
)

DATA = {
    'nfcorpus': {'repository': 'BeIR/nfcorpus', 'revision': 'b5026a0e96e8a7ac4f95f482a596389289d46269',
                 'qrels': 'BeIR/nfcorpus-qrels', 'qrels_revision': 'a451b3b26d3ae1358f259c1a3a4dd61fcea35a65'},
    'scifact': {'repository': 'BeIR/scifact', 'revision': 'b3b5335604bf5ee3c4447671af975ea25143d4f5',
                'qrels': 'BeIR/scifact-qrels', 'qrels_revision': '2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029'},
}
METHODS = ['lora', 'dora', 'nora', 'dora_nora', 'dora_nora_mlr', 'dora_nora_gain']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(name):
    config = DATA[name]
    paths = {kind: hf_hub_download(config['repository'], f'{kind}/{kind}-00000-of-00001.parquet',
                                 repo_type='dataset', revision=config['revision']) for kind in ('corpus', 'queries')}
    corpus = {str(row['_id']): (row['title'] + '\n' + row['text']).strip()
              for row in pq.read_table(paths['corpus']).to_pylist()}
    queries = {str(row['_id']): row['text'] for row in pq.read_table(paths['queries']).to_pylist()}
    relevance, splits = {}, {}
    for split in ('train', 'dev', 'test') if name == 'nfcorpus' else ('test',):
        path = hf_hub_download(config['qrels'], f'{split}.tsv', repo_type='dataset', revision=config['qrels_revision'])
        paths[f'{split}_qrels'] = path
        rows = defaultdict(dict)
        with open(path) as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if int(row['score']) > 0:
                    rows[row['query-id']][row['corpus-id']] = int(row['score'])
        assert not (set(rows) & set(relevance)), 'Official query IDs must be disjoint'
        relevance.update(rows)
        splits['validation' if split == 'dev' else split] = sorted(rows)
    for qid, docs in relevance.items():
        assert qid in queries and docs and all(doc in corpus for doc in docs)
    metadata = {**config, 'source_sha256': {key: sha256(path) for key, path in paths.items()},
                'corpus_count': len(corpus), 'official_query_counts': {key: len(value) for key, value in splits.items()},
                'official_qrel_counts': {key: sum(len(relevance[q]) for q in value) for key, value in splits.items()}}
    return {'corpus': corpus, 'queries': queries, 'relevance': relevance, 'splits': splits, 'metadata': metadata}


def load_data():
    data = {name: load_dataset(name) for name in DATA}
    nf, sf = data['nfcorpus'], data['scifact']
    normalize = lambda text: ' '.join(text.lower().split())
    held_out = {normalize(nf['queries'][qid]) for split in ('validation', 'test') for qid in nf['splits'][split]}
    held_out.update(normalize(sf['queries'][qid]) for qid in sf['splits']['test'])
    excluded = [qid for qid in nf['splits']['train'] if normalize(nf['queries'][qid]) in held_out]
    nf['splits']['train'] = sorted(set(nf['splits']['train']) - set(excluded))
    nf['metadata']['excluded_train_ids_exact_heldout_text'] = excluded
    # Official development/test duplicate texts would invalidate model selection;
    # exclude those development rows while preserving the official test set.
    test_texts = {normalize(nf['queries'][qid]) for qid in nf['splits']['test']}
    test_texts.update(normalize(sf['queries'][qid]) for qid in sf['splits']['test'])
    excluded_dev = [qid for qid in nf['splits']['validation'] if normalize(nf['queries'][qid]) in test_texts]
    nf['splits']['validation'] = sorted(set(nf['splits']['validation']) - set(excluded_dev))
    nf['metadata']['excluded_validation_ids_exact_test_text'] = excluded_dev
    nf['metadata']['effective_query_counts'] = {key: len(value) for key, value in nf['splits'].items()}
    return data


@torch.inference_mode()
def mine_negatives(model, tokenizer, data, args):
    model.eval()
    docs, queries = sorted(data['corpus']), data['splits']['train']
    started = time.perf_counter()
    doc_vectors = encode(model, tokenizer, data['corpus_tokens'], docs, args.eval_batch_size, args.device)
    query_vectors = encode(model, tokenizer, data['query_tokens'], queries, args.eval_batch_size, args.device)
    rankings = (query_vectors @ doc_vectors.T).argsort(dim=1, descending=True).cpu().tolist()
    result = {qid: [docs[i] for i in ranks if docs[i] not in data['relevance'][qid]][:args.hard_negatives]
              for qid, ranks in zip(queries, rankings)}
    assert all(len(negatives) == args.hard_negatives and not (set(negatives) & set(data['relevance'][qid]))
               for qid, negatives in result.items())
    write_json(args.output_dir / 'hard_negatives.json', result)
    write_json(args.output_dir / 'mining.json', {
        'seconds': time.perf_counter() - started, 'query_count': len(result), 'negatives_per_query': args.hard_negatives,
        'selection': 'Highest frozen-MiniLM cosine scores excluding all relevant train qrels for the query. Fixed across every rank, method, seed, and epoch.',
        'sha256': sha256(args.output_dir / 'hard_negatives.json')})
    return result


def train(model, tokenizer, data, negatives, args, method, seed, learning_rate, multiplier, run_dir):
    from experiments.second_round_adapters import parameter_groups
    model.train()
    groups = parameter_groups(model, learning_rate, magnitude_lr_multiplier=multiplier, weight_decay=args.weight_decay)
    group_metadata = [{key: value for key, value in group.items() if key != 'params'} for group in groups]
    optimizer = torch.optim.AdamW(groups)
    initial_lrs = [group['lr'] for group in optimizer.param_groups]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    steps_per_epoch = math.ceil(len(data['splits']['train']) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, round(total_steps * args.warmup_fraction))
    generator = random.Random(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    started = time.perf_counter()
    step, epochs = 0, []
    with (run_dir / 'training.jsonl').open('w') as log:
        for epoch in range(args.epochs):
            ids = data['splits']['train'].copy()
            generator.shuffle(ids)
            losses = []
            for start in range(0, len(ids), args.batch_size):
                query_ids = ids[start:start + args.batch_size]
                # Stable de-duplication avoids giving popular hard negatives extra
                # weight merely because multiple queries mined the same document.
                positives = [generator.choice(sorted(data['relevance'][qid])) for qid in query_ids]
                doc_ids = list(dict.fromkeys(positives + [doc for qid in query_ids for doc in negatives[qid]]))
                positive_mask = torch.tensor([[doc in data['relevance'][qid] for doc in doc_ids] for qid in query_ids], device=args.device)
                assert positive_mask.any(dim=1).all()
                step += 1
                factor = step / warmup_steps if step <= warmup_steps else (total_steps - step + 1) / max(1, total_steps - warmup_steps)
                for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
                    group['lr'] = initial_lr * factor
                optimizer.zero_grad(set_to_none=True)
                qemb = embeddings(model, make_batch(tokenizer, data['query_tokens'], query_ids, args.device))
                demb = embeddings(model, make_batch(tokenizer, data['corpus_tokens'], doc_ids, args.device))
                scores = qemb @ demb.T / args.temperature
                loss = (scores.logsumexp(dim=1) - scores.masked_fill(~positive_mask, -torch.inf).logsumexp(dim=1)).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at step {step}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f'Nonfinite gradient at step {step}')
                optimizer.step()
                value = loss.item()
                losses.append(value)
                log.write(json.dumps({'epoch': epoch + 1, 'step': step, 'loss': value, 'learning_rates': [group['lr'] for group in optimizer.param_groups],
                                      'gradient_norm': grad_norm.item(), 'unique_documents': len(doc_ids), 'elapsed_seconds': time.perf_counter() - started}) + '\n')
                log.flush()
            summary = {'epoch': epoch + 1, 'mean_loss': statistics.mean(losses), 'elapsed_seconds': time.perf_counter() - started}
            epochs.append(summary)
            print(json.dumps({'event': 'epoch', 'method': method, 'rank': args.current_rank, 'seed': seed, **summary}), flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, run_dir / 'adapter.pt')
    return {'steps': step, 'epochs': epochs, 'seconds': seconds, 'queries_seen': len(data['splits']['train']) * args.epochs,
            'queries_per_second': len(data['splits']['train']) * args.epochs / seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'incremental_peak_allocated_bytes': torch.cuda.max_memory_allocated() - baseline_memory,
            'optimizer_groups': group_metadata}


def summarize(runs):
    groups = defaultdict(list)
    for record in runs:
        groups[f"{record['method']}_r{record['rank']}"].append(record)
    result = {}
    for key, records in groups.items():
        result[key] = {'seeds': [row['seed'] for row in records]}
        for dataset in ('nfcorpus', 'scifact'):
            result[key][dataset] = {}
            for metric in ('recall_at_10', 'mrr_at_10', 'ndcg_at_10'):
                values = [row['evaluation'][dataset]['test'][metric] for row in records]
                result[key][dataset][metric] = {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values) if len(values) > 1 else None, 'per_seed': values}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/var/tmp/dora-bench/round2/retrieval'))
    parser.add_argument('--methods', nargs='+', choices=['frozen', *METHODS], default=['frozen', *METHODS])
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 8])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--hard-negatives', type=int, default=4)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[3e-5, 1e-4, 3e-4, 1e-3])
    parser.add_argument('--mlr-learning-rates', type=float, nargs='+', default=[1e-4, 3e-4])
    parser.add_argument('--magnitude-multipliers', type=float, nargs='+', default=[0.1, 0.01])
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--warmup-fraction', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        parser.error('CUDA is required for this measured experiment')
    assert len(args.learning_rates) == len(args.mlr_learning_rates) * len(args.magnitude_multipliers)
    assert all(value > 0 for value in [args.epochs, args.batch_size, args.eval_batch_size, args.max_length, args.hard_negatives, *args.ranks])
    torch.cuda.set_device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    for name, dataset in data.items():
        dataset['corpus_tokens'] = tokenize_all(tokenizer, dataset['corpus'], args.max_length)
        dataset['query_tokens'] = tokenize_all(tokenizer, dataset['queries'], args.max_length)
        write_json(args.output_dir / f'{name}_splits.json', dataset['splits'])
        write_json(args.output_dir / f'{name}_qrels.json', {qid: dataset['relevance'][qid] for split in dataset['splits'].values() for qid in split})
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).resolve(), root / 'experiments/retrieval.py', root / 'experiments/adapters.py', root / 'experiments/second_round_adapters.py', root / 'dora.py']
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key not in ('resume', 'baseline_only')}
    metadata = {'utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv], 'configuration': configuration,
                'python': sys.version, 'platform': platform.platform(), 'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'huggingface_hub', 'numpy', 'pyarrow')},
                'cuda_runtime': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True).strip(),
                'model': {'repository': MODEL, 'revision': MODEL_REVISION, 'frozen_dtype': 'bfloat16', 'adapter_dtype': 'float32', 'pooling': 'FP32 mean pooling and L2 normalization'},
                'datasets': {name: ds['metadata'] for name, ds in data.items()},
                'source_sha256': {str(path.relative_to(root)): sha256(path) for path in source_paths if path.exists()},
                'protocol': {
                    'target_modules': 'Query/value in all six BERT self-attention layers; alpha/rank=1, adapter dropout=0.',
                    'objective': 'Multi-positive contrastive loss at temperature 0.05 over one seeded sampled relevant document plus four fixed mined negatives per query and in-batch negatives. Batch docs de-duplicated; all known query-relevant docs count in numerator.',
                    'selection': 'Fixed final epoch; four full-horizon candidates per method/rank selected using seed42 NFCorpus validation nDCG@10 only, tie prefers smaller base LR then smaller magnitude multiplier. Selected seed42 checkpoint reused; seeds43/44 use selected settings. Both test sets excluded from selection.',
                    'weight_decay': 'AdamW 0.01 for A/B only; magnitude m and log_gain have zero decay.',
                    'transfer': 'SciFact test-only, no SciFact training/development queries or qrels loaded.',
                    'ranking': 'Exhaustive cosine similarity, top10; nDCG gain 2^grade-1; macro query metrics.',
                    'inference': 'Unmerged adapters; three longest-batch warmups, synchronized wall times include CPU padding/transfers.',
                    'leakage_scope': 'Official split and exact lowercased whitespace-normalized query text audit within NFCorpus and against SciFact test. Shared corpus documents and paraphrases permitted by benchmark; pretrained-data contamination not audited.',
                }}
    if args.resume and (args.output_dir / 'provenance.json').exists():
        previous = json.loads((args.output_dir / 'provenance.json').read_text())
        assert previous['configuration'] == configuration, 'Resume configuration changed'
        assert previous['source_sha256'] == metadata['source_sha256'], 'Resume sources changed'
        metadata = previous
    else:
        write_json(args.output_dir / 'provenance.json', metadata)
        snapshot = args.output_dir / 'source'
        for source in source_paths:
            if source.exists():
                destination = snapshot / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    results, tuning = [], {}

    def new_model(method, rank, seed):
        seed_everything(seed)
        model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device)
        seed_everything(seed)
        if method == 'frozen':
            model.requires_grad_(False)
        else:
            from experiments.second_round_adapters import inject_adapters
            inject_adapters(model, method, rank=rank, targets=('query', 'value'))
        return model

    def evaluate_dataset(model, name, run_dir, splits):
        directory = run_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        ds = data[name]
        return evaluate(model, tokenizer, ds['corpus_tokens'], ds['query_tokens'], ds['relevance'], ds['splits'], args, directory, splits)

    def new_record(model, method, rank, seed, run_dir):
        return {'method': method, 'rank': rank, 'seed': seed, 'artifact_directory': str(run_dir),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'trainable_names': [name for name, p in model.named_parameters() if p.requires_grad]}

    def save_results():
        write_json(args.output_dir / 'results.json', {'provenance': metadata, 'tuning': tuning, 'runs': results, 'summary': summarize(results)})

    frozen_dir = args.output_dir / 'frozen_seed42'
    frozen_dir.mkdir(parents=True, exist_ok=True)
    needs_frozen = not (args.resume and (frozen_dir / 'result.json').exists())
    needs_negatives = not (args.resume and (args.output_dir / 'hard_negatives.json').exists())
    if needs_frozen or needs_negatives:
        model = new_model('frozen', 0, args.seeds[0])
        if needs_frozen:
            record = new_record(model, 'frozen', 0, args.seeds[0], frozen_dir)
            record['evaluation'] = {name: evaluate_dataset(model, name, frozen_dir, ('validation', 'test') if name == 'nfcorpus' else ('test',)) for name in data}
            write_json(frozen_dir / 'result.json', record)
            print(json.dumps({'event': 'baseline', 'evaluation': record['evaluation']}), flush=True)
        if needs_negatives:
            mine_negatives(model, tokenizer, data['nfcorpus'], args)
        del model
        torch.cuda.empty_cache()
    results.append(json.loads((frozen_dir / 'result.json').read_text()))
    negatives = json.loads((args.output_dir / 'hard_negatives.json').read_text())
    save_results()
    if args.baseline_only:
        return
    for rank in args.ranks:
        args.current_rank = rank
        for method in [method for method in args.methods if method != 'frozen']:
            candidates = []
            settings = [(lr, multiplier) for lr in args.mlr_learning_rates for multiplier in args.magnitude_multipliers] if method == 'dora_nora_mlr' else [(lr, 1.0) for lr in args.learning_rates]
            for lr, multiplier in sorted(settings):
                seed = args.seeds[0]
                directory = args.output_dir / 'tuning' / f'{method}_r{rank}_lr{lr:g}_m{multiplier:g}_seed{seed}'
                directory.mkdir(parents=True, exist_ok=True)
                if args.resume and (directory / 'result.json').exists():
                    record = json.loads((directory / 'result.json').read_text())
                else:
                    model = new_model(method, rank, seed)
                    record = new_record(model, method, rank, seed, directory)
                    record.update(learning_rate=lr, magnitude_lr_multiplier=multiplier)
                    record['training'] = train(model, tokenizer, data['nfcorpus'], negatives, args, method, seed, lr, multiplier, directory)
                    record['evaluation'] = {'nfcorpus': evaluate_dataset(model, 'nfcorpus', directory, ('validation',))}
                    write_json(directory / 'result.json', record)
                    print(json.dumps({'event': 'tuning', 'method': method, 'rank': rank, 'learning_rate': lr, 'magnitude_lr_multiplier': multiplier,
                                      'validation': record['evaluation']['nfcorpus']['validation'], 'training_seconds': record['training']['seconds']}), flush=True)
                    del model
                    torch.cuda.empty_cache()
                candidates.append(record)
            winner = max(candidates, key=lambda row: (row['evaluation']['nfcorpus']['validation']['ndcg_at_10'], -row['learning_rate'], -row['magnitude_lr_multiplier']))
            lr, multiplier = winner['learning_rate'], winner['magnitude_lr_multiplier']
            tuning[f'{method}_r{rank}'] = {'selected_learning_rate': lr, 'selected_magnitude_lr_multiplier': multiplier, 'candidates': candidates}
            save_results()
            for seed in args.seeds:
                directory = args.output_dir / f'{method}_r{rank}_seed{seed}'
                directory.mkdir(parents=True, exist_ok=True)
                if args.resume and (directory / 'result.json').exists():
                    record = json.loads((directory / 'result.json').read_text())
                else:
                    model = new_model(method, rank, seed)
                    record = new_record(model, method, rank, seed, directory)
                    record.update(learning_rate=lr, magnitude_lr_multiplier=multiplier)
                    if seed == args.seeds[0]:
                        checkpoint = Path(winner['artifact_directory']) / 'adapter.pt'
                        state = torch.load(checkpoint, map_location=args.device, weights_only=True)
                        expected = {name for name, p in model.named_parameters() if p.requires_grad}
                        assert set(state) == expected
                        model.load_state_dict(state, strict=False)
                        record['training'] = winner['training']
                        record['reused_tuning_checkpoint'] = str(checkpoint)
                        shutil.copy2(checkpoint, directory / 'adapter.pt')
                        shutil.copy2(checkpoint.parent / 'training.jsonl', directory / 'training.jsonl')
                    else:
                        record['training'] = train(model, tokenizer, data['nfcorpus'], negatives, args, method, seed, lr, multiplier, directory)
                    record['evaluation'] = {name: evaluate_dataset(model, name, directory, ('validation', 'test') if name == 'nfcorpus' else ('test',)) for name in data}
                    write_json(directory / 'result.json', record)
                    print(json.dumps({'event': 'result', 'method': method, 'rank': rank, 'seed': seed, 'evaluation': record['evaluation']}), flush=True)
                    del model
                    torch.cuda.empty_cache()
                results.append(record)
                save_results()
    print(json.dumps({'event': 'complete', 'runs': len(results), 'summary': summarize(results)}), flush=True)


if __name__ == '__main__':
    main()
