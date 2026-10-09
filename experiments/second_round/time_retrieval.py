"""Warm and repeat final retrieval inference measurements on fixed seed42 models."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

from experiments.retrieval import MODEL, MODEL_REVISION, encode, seed_everything, tokenize_all
from experiments.second_round.retrieval import load_data
from experiments.second_round_adapters import inject_adapters


@torch.inference_mode()
def measure(model, tokenizer, dataset, batch_size, repeats, device):
    corpus_ids = sorted(dataset['corpus'])
    query_ids = dataset['splits']['test']
    model.eval()
    # Warm every actual length bucket on both corpus and queries. This includes
    # the shape-dependent setup missed by a single longest-batch warmup.
    for _ in range(2):
        documents = encode(model, tokenizer, dataset['corpus_tokens'], corpus_ids, batch_size, device)
        queries = encode(model, tokenizer, dataset['query_tokens'], query_ids, batch_size, device)
        (queries @ documents.T).topk(10, dim=1)
    torch.cuda.synchronize()
    rows = []
    for _ in range(repeats):
        start = time.perf_counter()
        documents = encode(model, tokenizer, dataset['corpus_tokens'], corpus_ids, batch_size, device)
        torch.cuda.synchronize()
        corpus_seconds = time.perf_counter() - start
        start = time.perf_counter()
        queries = encode(model, tokenizer, dataset['query_tokens'], query_ids, batch_size, device)
        (queries @ documents.T).topk(10, dim=1)
        torch.cuda.synchronize()
        query_seconds = time.perf_counter() - start
        rows.append({'corpus_seconds': corpus_seconds, 'query_seconds': query_seconds,
                     'total_seconds': corpus_seconds + query_seconds})
    return {'corpus_documents': len(corpus_ids), 'queries': len(query_ids), 'repeats': rows,
            'median_corpus_seconds': statistics.median(row['corpus_seconds'] for row in rows),
            'median_query_seconds': statistics.median(row['query_seconds'] for row in rows),
            'median_total_seconds': statistics.median(row['total_seconds'] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    results = json.loads((args.root / 'results.json').read_text())
    config = SimpleNamespace(**results['provenance']['configuration'])
    torch.cuda.set_device(config.device)
    torch.set_num_threads(config.threads)
    torch.set_float32_matmul_precision('highest')
    data = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    for dataset in data.values():
        dataset['corpus_tokens'] = tokenize_all(tokenizer, dataset['corpus'], config.max_length)
        dataset['query_tokens'] = tokenize_all(tokenizer, dataset['queries'], config.max_length)
    rows = []
    for record in results['runs']:
        if record['seed'] != config.seeds[0]:
            continue
        seed_everything(record['seed'])
        model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16, attn_implementation='sdpa').to(config.device)
        seed_everything(record['seed'])
        if record['method'] != 'frozen':
            inject_adapters(model, record['method'], rank=record['rank'], targets=('query', 'value'))
            checkpoint = args.root / Path(record['artifact_directory']).name / 'adapter.pt'
            state = torch.load(checkpoint, map_location=config.device, weights_only=True)
            assert set(state) == {name for name, p in model.named_parameters() if p.requires_grad}
            model.load_state_dict(state, strict=False)
        model.requires_grad_(False)
        row = {'method': record['method'], 'rank': record['rank'], 'seed': record['seed'],
               'datasets': {name: measure(model, tokenizer, dataset, config.eval_batch_size, args.repeats, config.device) for name, dataset in data.items()}}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    result = {'protocol': 'Unmerged adapters; selected first-seed checkpoints; two complete corpus/query warmup passes per model and dataset, then synchronized repeated encode/exhaustive-top10 passes; repeat count is retained in each run. CPU token padding and transfers included.',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu': torch.cuda.get_device_name(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'runs': rows}
    (args.root / 'steady_state_timing.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
