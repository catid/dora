"""Recompute retrieval results from portable raw artifacts without model access."""
import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ('recall_at_10', 'mrr_at_10', 'ndcg_at_10')


def metrics(ranked, relevant):
    hits = [relevant.get(docid, 0) for docid in ranked]
    ideal = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(relevant.values(), reverse=True)[:10]))
    return {'recall_at_10': sum(grade > 0 for grade in hits) / len(relevant),
            'mrr_at_10': next((1 / (i + 1) for i, grade in enumerate(hits) if grade > 0), 0.0),
            'ndcg_at_10': sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(hits)) / ideal}


def validate(root, write_audit=True):
    result = json.loads((root / 'results.json').read_text())
    config = result['provenance']['configuration']
    splits = {name: json.loads((root / f'{name}_splits.json').read_text()) for name in ('nfcorpus', 'scifact')}
    qrels = {name: json.loads((root / f'{name}_qrels.json').read_text()) for name in splits}
    sources = result['provenance']['source_sha256']
    for name, expected in sources.items():
        assert hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() == expected, name
    negatives = json.loads((root / 'hard_negatives.json').read_text())
    mining = json.loads((root / 'mining.json').read_text())
    assert hashlib.sha256((root / 'hard_negatives.json').read_bytes()).hexdigest() == mining['sha256']
    assert set(negatives) == set(splits['nfcorpus']['train'])
    for qid, docs in negatives.items():
        assert len(docs) == len(set(docs)) == config['hard_negatives']
        assert not set(docs) & set(qrels['nfcorpus'][qid])
    expected_cells = {(method, rank) for method in config['methods'] if method != 'frozen' for rank in config['ranks']}
    assert len(result['runs']) == 1 + len(expected_cells) * len(config['seeds'])
    assert {(row['method'], row['rank']) for row in result['runs'] if row['method'] != 'frozen'} == expected_cells
    assert len(result['tuning']) == len(expected_cells)
    all_records = list(result['runs'])
    for key, selection in result['tuning'].items():
        candidates = selection['candidates']
        assert len(candidates) == len(config['learning_rates'])
        winner = max(candidates, key=lambda row: (row['evaluation']['nfcorpus']['validation']['ndcg_at_10'], -row['learning_rate'], -row['magnitude_lr_multiplier']))
        assert winner['learning_rate'] == selection['selected_learning_rate']
        assert winner['magnitude_lr_multiplier'] == selection['selected_magnitude_lr_multiplier']
        for row in candidates:
            assert set(row['evaluation']) == {'nfcorpus'}
            assert set(row['evaluation']['nfcorpus']).issuperset({'validation'})
            assert 'test' not in row['evaluation']['nfcorpus']
        for final in result['runs']:
            if final['method'] != winner['method'] or final['rank'] != winner['rank']:
                continue
            assert final['learning_rate'] == selection['selected_learning_rate']
            assert final['magnitude_lr_multiplier'] == selection['selected_magnitude_lr_multiplier']
            if final['seed'] == config['seeds'][0]:
                assert final['reused_tuning_checkpoint'] == str(Path(winner['artifact_directory']) / 'adapter.pt')
                for metric in METRICS:
                    assert abs(final['evaluation']['nfcorpus']['validation'][metric] - winner['evaluation']['nfcorpus']['validation'][metric]) < 1e-12
        all_records.extend(candidates)
    query_records, training_steps = 0, 0
    grouped = defaultdict(list)
    for row in all_records:
        original = Path(row['artifact_directory'])
        directory = root / ('tuning' if original.parent.name == 'tuning' else '') / original.name
        for dataset, evaluation in row['evaluation'].items():
            for split in ('validation', 'test'):
                if split not in evaluation:
                    continue
                detail = json.loads((directory / dataset / f'{split}_per_query.json').read_text())
                assert [d['query_id'] for d in detail] == splits[dataset][split]
                for d in detail:
                    assert len(d['top_10']) == len(set(d['top_10'])) == 10
                    assert len(d['scores']) == 10 and all(math.isfinite(v) for v in d['scores'])
                    assert all(a >= b for a, b in zip(d['scores'], d['scores'][1:]))
                    expected = metrics(d['top_10'], qrels[dataset][d['query_id']])
                    for metric in METRICS:
                        assert abs(expected[metric] - d[metric]) < 1e-12
                    query_records += 1
                for metric in METRICS:
                    assert abs(statistics.mean(d[metric] for d in detail) - evaluation[split][metric]) < 1e-12
        if row['method'] != 'frozen':
            train = row['training']
            steps = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
            expected_steps = math.ceil(len(splits['nfcorpus']['train']) / config['batch_size']) * config['epochs']
            assert len(steps) == train['steps'] == expected_steps
            assert [s['step'] for s in steps] == list(range(1, expected_steps + 1))
            assert all(math.isfinite(s['loss']) and math.isfinite(s['gradient_norm']) for s in steps)
            assert train['seconds'] > 0 and train['peak_allocated_bytes'] > 0
            groups = train['optimizer_groups']
            assert sum(group['parameter_count'] for group in groups) == row['trainable_parameters']
            assert set(name for group in groups for name in group['param_names']) == set(row['trainable_names'])
            for group in groups:
                if group['group_name'] != 'adapter_factors':
                    assert group['weight_decay'] == 0
                expected_lr = row['learning_rate'] * (row['magnitude_lr_multiplier'] if group['group_name'] == 'decoupled_magnitudes' else 1)
                assert abs(group['lr'] - expected_lr) < 1e-14
            training_steps += len(steps)
        if original.parent.name != 'tuning':
            grouped[f"{row['method']}_r{row['rank']}"].append(row)
    for key, rows in grouped.items():
        if rows[0]['method'] != 'frozen':
            assert sorted(row['seed'] for row in rows) == sorted(config['seeds'])
        for dataset in ('nfcorpus', 'scifact'):
            for metric in METRICS:
                assert abs(result['summary'][key][dataset][metric]['mean'] - statistics.mean(row['evaluation'][dataset]['test'][metric] for row in rows)) < 1e-12
    control_queries = 0
    if (root / 'initial_adapter_control.json').exists():
        control = json.loads((root / 'initial_adapter_control.json').read_text())
        for row in control['runs']:
            for dataset, evaluation in row['evaluation'].items():
                detail = json.loads((root / 'initial_control' / row['method'] / dataset / 'test_per_query.json').read_text())
                assert [d['query_id'] for d in detail] == splits[dataset]['test']
                for d in detail:
                    expected = metrics(d['top_10'], qrels[dataset][d['query_id']])
                    for metric in METRICS:
                        assert abs(expected[metric] - d[metric]) < 1e-12
                    control_queries += 1
                for metric in METRICS:
                    assert abs(statistics.mean(d[metric] for d in detail) - evaluation['test'][metric]) < 1e-12
    audit = {'status': 'pass', 'input_results_sha256': hashlib.sha256((root / 'results.json').read_bytes()).hexdigest(), 'final_records': len(result['runs']), 'tuning_records': len(all_records) - len(result['runs']),
             'raw_query_records_recomputed': query_records, 'initial_control_query_records_recomputed': control_queries,
             'logged_steps_checked_including_reused_checkpoints': training_steps,
             'negative_queries_checked': len(negatives), 'source_snapshots_verified': len(sources),
             'scope': 'Independent metric recomputation, selection, full trial budget, seed matrix, source hashes, negative relevance exclusions, optimizer groups, loss/gradient finiteness, and training log counts. Does not assert semantic correctness of corpus labels.'}
    if write_audit:
        (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(validate(parser.parse_args().root), indent=2))
