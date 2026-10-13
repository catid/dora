"""Re-score original-DoRA retrieval and compare with the preserved task runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch
from huggingface_hub import hf_hub_download

METRICS = ('ndcg_at_10', 'mrr_at_10', 'recall_at_10')
METHOD = 'dora_original'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(ranking, relevant):
    grades = [relevant.get(document, 0) for document in ranking]
    ideal = sorted(relevant.values(), reverse=True)[:10]
    dcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
    idcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    return {'ndcg_at_10': dcg / idcg,
            'mrr_at_10': next((1 / (i + 1) for i, grade in enumerate(grades) if grade), 0),
            'recall_at_10': sum(grade > 0 for grade in grades) / len(relevant)}


def analyze(roots, output, allow_missing_checkpoints=False):
    hashes = {name: {} for name in roots}
    checkpoint_hashes = {name: {} for name in roots if name.startswith('original')}
    checkpoint_counts = {'verified': 0, 'omitted': 0}
    scored = 0

    def bind(root, relative):
        path = roots[root] / relative
        hashes[root][str(relative)] = digest(path)
        return path

    def read(root, relative):
        return json.loads(bind(root, relative).read_text())

    def checkpoint(root, relative, rank):
        path = roots[root] / relative
        if not path.exists():
            assert allow_missing_checkpoints, path
            checkpoint_counts['omitted'] += 1
            return
        checkpoint_hashes[root][str(relative)] = digest(path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert len(state) == 36
        for name, tensor in state.items():
            assert torch.isfinite(tensor).all()
            suffix = name.rsplit('.', 1)[-1]
            assert suffix in ('m', 'lora_A', 'lora_B')
            expected = {'m': ((1, 384), torch.bfloat16),
                        'lora_A': ((384, rank), torch.float32),
                        'lora_B': ((rank, 384), torch.float32)}[suffix]
            assert (tuple(tensor.shape), tensor.dtype) == expected
        checkpoint_counts['verified'] += 1

    def score(root, relative, ids, relevance, recorded):
        nonlocal scored
        rows = read(root, relative)
        assert [row['query_id'] for row in rows] == ids
        assert len(rows) == recorded['query_count']
        for row in rows:
            assert len(row['top_10']) == len(set(row['top_10'])) == 10
            expected = metrics(row['top_10'], relevance[row['query_id']])
            for key in METRICS:
                assert math.isclose(row[key], expected[key], abs_tol=1e-12)
        for key in METRICS:
            assert math.isclose(statistics.mean(row[key] for row in rows), recorded[key], abs_tol=1e-12)
        scored += len(rows)
        return np.array([[row[key] for row in rows] for key in METRICS], dtype=np.float64)

    def directory(row, second):
        return Path(row['artifact_directory']).name if second else f"{row['method']}_seed{row['seed']}"

    def training(root, relative, row, config, rank, train_count, second):
        lines = [json.loads(line) for line in bind(root, relative / 'training.jsonl').read_text().splitlines()]
        steps = math.ceil(train_count / config['batch_size']) * config['epochs']
        assert [line['step'] for line in lines] == list(range(1, steps + 1))
        assert all(math.isfinite(line[key]) for line in lines for key in ('loss', 'gradient_norm'))
        assert row['training']['steps'] == steps
        assert row['training']['queries_seen'] == train_count * config['epochs']
        assert len(row['training']['epochs']) == config['epochs']
        assert row['trainable_parameters'] == 12 * (2 * 384 * rank + 384)
        assert len(row['trainable_names']) == len(set(row['trainable_names'])) == 36
        if second:
            groups = row['training']['optimizer_groups']
            assert sum(group['parameter_count'] for group in groups) == row['trainable_parameters']
            assert all(group['lr'] == row['learning_rate'] for group in groups)
            assert all(group['weight_decay'] == (config['weight_decay'] if group['group_name'] == 'adapter_factors' else 0) for group in groups)
        checkpoint(root, relative / 'adapter.pt', rank)

    summary, comparisons, protocol = [], [], {}
    rng = np.random.default_rng(20261009)
    model_provenance = None
    for benchmark, second in (('scifact', False), ('nfcorpus', True)):
        original, reference = f'original_{benchmark}', f'reference_{benchmark}'
        new, old = read(original, 'results.json'), read(reference, 'results.json')
        config, prior = new['provenance']['configuration'], old['provenance']['configuration']
        assert {k: v for k, v in config.items() if k not in ('output_dir', 'methods')} == {
            k: v for k, v in prior.items() if k not in ('output_dir', 'methods')}
        assert config['seeds'] == [42, 43, 44]
        for key in ('repository', 'revision', 'frozen_dtype', 'pooling'):
            assert new['provenance']['model'][key] == old['provenance']['model'][key]
        model_provenance = new['provenance']['model']
        data_key = 'datasets' if second else 'dataset'
        assert new['provenance'][data_key] == old['provenance'][data_key]
        for name, expected in new['provenance']['source_sha256'].items():
            assert digest(bind(original, Path('source') / name)) == expected

        if second:
            splits = {name: read(original, f'{name}_splits.json') for name in ('nfcorpus', 'scifact')}
            gold = {name: read(original, f'{name}_qrels.json') for name in splits}
            matching = ['nfcorpus_splits.json', 'scifact_splits.json', 'nfcorpus_qrels.json',
                        'scifact_qrels.json', 'hard_negatives.json',
                        'frozen_seed42/nfcorpus/validation_per_query.json',
                        'frozen_seed42/nfcorpus/test_per_query.json', 'frozen_seed42/scifact/test_per_query.json']
            train_count = len(splits['nfcorpus']['train'])
        else:
            splits = {'scifact': read(original, 'splits.json')}
            gold_path = roots[original] / 'audit_qrels.json'
            if not gold_path.exists():
                provenance = new['provenance']['dataset']
                relevance = {}
                for split in ('train', 'test'):
                    path = hf_hub_download(provenance['qrels'], f'{split}.tsv', repo_type='dataset',
                                           revision=provenance['qrels_revision'], local_files_only=True)
                    assert digest(path) == provenance['source_sha256'][f'{split}_qrels']
                    with open(path) as stream:
                        for row in csv.DictReader(stream, delimiter='\t'):
                            if int(row['score']) > 0:
                                relevance.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
                gold_path.write_text(json.dumps(relevance, indent=2) + '\n')
            gold = {'scifact': read(original, 'audit_qrels.json')}
            matching = ['splits.json', 'frozen_seed42/validation_per_query.json', 'frozen_seed42/test_per_query.json']
            train_count = len(splits['scifact']['train'])
        for relative in matching:
            assert digest(bind(original, relative)) == digest(bind(reference, relative)), relative
        protocol[benchmark] = {'configuration_matches_except_methods_output': True,
                               'pinned_model_and_dataset_provenance_matches': True,
                               'byte_identical_inputs_and_frozen_predictions': matching}
        ranks = config['ranks'] if second else [config['rank']]
        assert len(new['runs']) == 1 + 3 * len(ranks)
        assert len(new['tuning']) == len(ranks)
        for rank in ranks:
            selection = new['tuning'][f'{METHOD}_r{rank}' if second else METHOD]
            candidates = selection['candidates']
            assert len(candidates) == len(config['learning_rates'])
            assert {row['learning_rate'] for row in candidates} == set(config['learning_rates'])
            dataset = 'nfcorpus' if second else 'scifact'
            val = lambda row: row['evaluation'][dataset]['validation'] if second else row['evaluation']['validation']
            winner = max(candidates, key=lambda row: (val(row)['ndcg_at_10'], -row['learning_rate']))
            assert selection['selected_learning_rate'] == winner['learning_rate']
            for row in candidates:
                assert row['method'] == METHOD and row['seed'] == 42
                relative = Path('tuning') / Path(row['artifact_directory']).name
                assert read(original, relative / 'result.json') == row
                training(original, relative, row, config, rank, train_count, second)
                prediction = relative / dataset / 'validation_per_query.json' if second else relative / 'validation_per_query.json'
                score(original, prediction, splits[dataset]['validation'], gold[dataset], val(row))

            for dataset in splits:
                arrays = {}
                for source, records in ((reference, old['runs']), (original, new['runs'])):
                    selected = [row for row in records if row['method'] == 'frozen' or not second or row['rank'] == rank]
                    methods = sorted({row['method'] for row in selected})
                    for method in methods:
                        if source == original and method == 'frozen':
                            row = next(row for row in selected if row['method'] == 'frozen')
                            prefix = Path(directory(row, second))
                            evaluation = row['evaluation'][dataset] if second else row['evaluation']
                            if second:
                                prefix = prefix / dataset
                            for split in ('validation', 'test') if 'validation' in splits[dataset] else ('test',):
                                score(source, prefix / f'{split}_per_query.json', splits[dataset][split], gold[dataset], evaluation[split])
                            continue
                        runs = sorted([row for row in selected if row['method'] == method], key=lambda row: row['seed'])
                        assert [row['seed'] for row in runs] == ([42] if method == 'frozen' else config['seeds'])
                        values = []
                        for row in runs:
                            relative = Path(directory(row, second))
                            evaluation = row['evaluation'][dataset] if second else row['evaluation']
                            prefix = relative / dataset if second else relative
                            values.append(score(source, prefix / 'test_per_query.json', splits[dataset]['test'], gold[dataset], evaluation['test']))
                            if source == original:
                                assert row['learning_rate'] == selection['selected_learning_rate']
                                assert read(source, relative / 'result.json') == row
                                # Once per checkpoint, not twice for the two evaluation datasets.
                                if dataset == ('nfcorpus' if second else 'scifact'):
                                    training(source, relative, row, config, rank, train_count, second)
                                    if row['seed'] == 42:
                                        a = roots[source] / relative / 'adapter.pt'
                                        b = roots[source] / 'tuning' / Path(winner['artifact_directory']).name / 'adapter.pt'
                                        if a.exists() and b.exists():
                                            assert digest(a) == digest(b)
                                    score(source, prefix / 'validation_per_query.json', splits[dataset]['validation'], gold[dataset], evaluation['validation'])
                        arrays[method] = np.stack(values)
                        summary.append({'benchmark': benchmark, 'dataset': dataset, 'rank': rank, 'method': method,
                                        'seeds': [row['seed'] for row in runs],
                                        'metrics': {key: {'mean': float(arrays[method][:, i].mean()),
                                                         'sample_sd': statistics.stdev(arrays[method][:, i].mean(1)) if len(runs) > 1 else None,
                                                         'per_seed': arrays[method][:, i].mean(1).tolist()} for i, key in enumerate(METRICS)}})
                seed_draws = rng.integers(3, size=(10000, 3))
                query_draws = rng.integers(len(splits[dataset]['test']), size=(10000, len(splits[dataset]['test'])))
                for method, values in arrays.items():
                    if method == METHOD:
                        continue
                    delta = arrays[METHOD] - values
                    row = {'benchmark': benchmark, 'dataset': dataset, 'rank': rank,
                           'candidate': METHOD, 'reference': method, 'metrics': {}}
                    for i, key in enumerate(METRICS):
                        sample = delta[:, i][seed_draws[:, :, None], query_draws[:, None, :]].mean((1, 2))
                        row['metrics'][key] = {'difference': float(delta[:, i].mean()),
                                               'ci95': np.quantile(sample, [0.025, 0.975]).tolist()}
                    comparisons.append(row)
    identity_path = roots['original_scifact'] / 'model_checkpoint_identity.json'
    if not identity_path.exists():
        cached_model = hf_hub_download(model_provenance['repository'], 'model.safetensors',
                                       revision=model_provenance['revision'], local_files_only=True)
        identity_path.write_text(json.dumps({
            'repository': model_provenance['repository'], 'revision': model_provenance['revision'],
            'filename': 'model.safetensors', 'sha256': digest(cached_model),
            'scope': 'File hashed from the shared immutable local model cache during the original-retrieval audit; retained identity is an attestation when the pretrained payload is unavailable.'}, indent=2) + '\n')
    identity = read('original_scifact', 'model_checkpoint_identity.json')
    assert identity['repository'] == model_provenance['repository'] and identity['revision'] == model_provenance['revision']
    result = {'status': 'pass', 'source_sha256': digest(__file__),
              'artifact_roots': {name: str(path) for name, path in roots.items()},
              'input_sha256': hashes, 'checkpoint_sha256': checkpoint_hashes,
              'audit': {'saved_query_records_rescored': scored, 'checkpoints': checkpoint_counts,
                        'protocol': protocol, 'shared_pinned_model_file_sha256': identity['sha256'],
                        'model_hash_scope': 'Current cached file for the same immutable model revision, plus exact frozen-ranking reproduction. Earlier retrieval manifests did not retain a separate checkpoint-file digest; historical frozen tensors are not independently rehashed.'},
              'summary': summary, 'comparisons': comparisons,
              'uncertainty': {'iterations': 10000, 'seed': 20261009,
                              'protocol': 'Crossed paired bootstrap of matched training seeds and shared test queries; common draws across references/metrics within each task/rank. Deterministic frozen baseline is broadcast over the candidate seed axis.',
                              'limitations': 'Exploratory percentile intervals, no multiple-comparison correction; three seeds, one training horizon and limited validation grids. No inference of broad task superiority or equal-quality runtime.'},
              'precision_scope': 'Original input-column full-gradient math with historical factor initialization. For BF16-base retrieval without autocast, original m remains BF16, factors/merged weights remain FP32, and only the F.linear boundary casts activation/bias to FP32 and output back to BF16. This is an explicit compatibility adaptation, not a literal mixed-dtype historical F.linear call.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, default=Path('/var/tmp/dora-bench/original_bench'))
    parser.add_argument('--reference-scifact', type=Path, default=Path('/var/tmp/dora-bench/runs/retrieval'))
    parser.add_argument('--reference-nfcorpus', type=Path, default=Path('/var/tmp/dora-bench/round2/retrieval'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-missing-checkpoints', action='store_true')
    args = parser.parse_args()
    roots = {'original_scifact': args.original_root / 'scifact', 'original_nfcorpus': args.original_root / 'nfcorpus',
             'reference_scifact': args.reference_scifact, 'reference_nfcorpus': args.reference_nfcorpus}
    result = analyze(roots, args.output or args.original_root / 'retrieval_analysis.json', args.allow_missing_checkpoints)
    print(json.dumps({'status': result['status'], 'audit': result['audit'], 'comparisons': len(result['comparisons'])}, indent=2))
