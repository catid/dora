"""Exploratory matched-seed, paired-query bootstrap for retrieval differences."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CONTRASTS = [('nora', 'lora'), ('dora', 'lora'), ('dora_nora', 'nora'), ('dora_nora_mlr', 'dora_nora'),
             ('dora_nora_gain', 'dora_nora'), ('dora_nora_mlr', 'nora'), ('dora_nora_gain', 'nora')]
METRICS = ('ndcg_at_10', 'mrr_at_10', 'recall_at_10')


def analyze(root, iterations=10000, seed=20261008):
    results = json.loads((root / 'results.json').read_text())
    config = results['provenance']['configuration']
    by_key = {(row['method'], row['rank'], row['seed']): row for row in results['runs']}
    generator = np.random.default_rng(seed)
    comparisons = []
    input_hashes = {}
    for dataset in ('nfcorpus', 'scifact'):
        for rank in config['ranks']:
            arrays = {}
            query_ids = None
            for method in config['methods']:
                if method == 'frozen':
                    continue
                values = []
                for run_seed in config['seeds']:
                    row = by_key[(method, rank, run_seed)]
                    directory = root / Path(row['artifact_directory']).name
                    raw_path = directory / dataset / 'test_per_query.json'
                    input_hashes[str(raw_path.relative_to(root))] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    raw = json.loads(raw_path.read_text())
                    ids = [q['query_id'] for q in raw]
                    if query_ids is None:
                        query_ids = ids
                    assert ids == query_ids
                    values.append([[q[metric] for q in raw] for metric in METRICS])
                arrays[method] = np.asarray(values, dtype=np.float64)  # seed, metric, query
            seeds, queries = len(config['seeds']), len(query_ids)
            seed_draws = generator.integers(seeds, size=(iterations, seeds))
            query_draws = generator.integers(queries, size=(iterations, queries))
            for candidate, reference in CONTRASTS:
                difference = arrays[candidate] - arrays[reference]
                row = {'dataset': dataset, 'rank': rank, 'candidate': candidate, 'reference': reference,
                       'seeds': config['seeds'], 'query_count': queries, 'metrics': {}}
                for metric_index, metric in enumerate(METRICS):
                    delta = difference[:, metric_index, :]
                    sampled = delta[seed_draws[:, :, None], query_draws[:, None, :]].mean(axis=(1, 2))
                    low, high = np.quantile(sampled, [0.025, 0.975])
                    row['metrics'][metric] = {
                        'mean_difference': float(delta.mean()), 'mean_difference_percentage_points': float(delta.mean() * 100),
                        'paired_seed_differences': delta.mean(axis=1).tolist(),
                        'percentile_95_interval': [float(low), float(high)],
                        'percentile_95_interval_percentage_points': [float(low * 100), float(high * 100)],
                        'bootstrap_fraction_positive': float((sampled > 0).mean()),
                    }
                comparisons.append(row)
    output = {'input_results_sha256': hashlib.sha256((root / 'results.json').read_bytes()).hexdigest(),
              'input_per_query_sha256': input_hashes, 'iterations': iterations, 'rng_seed': seed, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol': 'Crossed paired bootstrap: sample the matched training-seed axis and test-query axis independently with replacement, using the same sampled seeds/queries for both methods. Compute the mean paired metric difference, then percentile 95% intervals. Fixed common draws across all contrasts/metrics in a dataset/rank.',
              'limitations': 'Exploratory, no multiple-comparison correction; three training seeds provide limited seed-variance evidence. Test queries and training seeds are assumed exchangeable within this task; intervals do not establish cross-task generalization. Bootstrap fraction positive is descriptive, not a calibrated p-value.',
              'comparisons': comparisons}
    (root / 'comparisons.json').write_text(json.dumps(output, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    result = analyze(args.root, args.iterations, args.seed)
    for row in result['comparisons']:
        metric = row['metrics']['ndcg_at_10']
        print(row['dataset'], row['rank'], row['candidate'], '-', row['reference'],
              metric['mean_difference_percentage_points'], metric['percentile_95_interval_percentage_points'])
