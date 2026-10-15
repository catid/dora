#!/usr/bin/env python3
"""Audit the immutable short-HPO launcher records without selecting trials."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/var/tmp/dora-adaptive-rank/runs/v3_hpo_20261015_retry'))
    parser.add_argument('--methods', type=Path, default=Path('configs/methods.yaml'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/hpo_integrity_v3.json'))
    args = parser.parse_args()
    method_ids = [row['id'] for row in yaml.safe_load(args.methods.read_text())['methods']]
    records, issues = [], []
    for directory in sorted(args.root.glob('*/')):
        status_path = directory / 'launcher_status.json'
        if not status_path.exists():
            continue
        launcher = json.loads(status_path.read_text())
        config_path, metadata_path = directory / 'hpo_config.json', directory / 'metadata.json'
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        state = launcher.get('status', 'unknown')
        record = {
            'run_id': directory.name,
            'method': config.get('method'),
            'trial_index': config.get('trial_index'),
            'seed': config.get('seed'),
            'launcher_status': state,
            'metadata_status': metadata.get('status'),
            'input_tokens': metadata.get('input_tokens'),
            'charged_gpu_seconds': metadata.get('charged_gpu_seconds'),
        }
        records.append(record)
        if config.get('method') not in method_ids:
            issues.append({'run_id': directory.name, 'kind': 'unknown_method', 'value': config.get('method')})
        if config.get('seed') != 1001:
            issues.append({'run_id': directory.name, 'kind': 'wrong_hpo_seed', 'value': config.get('seed')})
        if config.get('trial_index') not in range(12):
            issues.append({'run_id': directory.name, 'kind': 'trial_index_out_of_range', 'value': config.get('trial_index')})
        if state == 'complete' and metadata.get('status') not in {'complete', 'passed'}:
            issues.append({'run_id': directory.name, 'kind': 'terminal_without_complete_metadata', 'value': metadata.get('status')})
        if state == 'failed':
            issues.append({'run_id': directory.name, 'kind': 'hpo_failure', 'value': launcher.get('returncode')})

    trials_by_method = collections.defaultdict(list)
    complete_tokens = collections.Counter()
    for record in records:
        trials_by_method[record['method']].append(record['trial_index'])
        if record['launcher_status'] == 'complete':
            complete_tokens[record['input_tokens']] += 1
    duplicate_trials = {method: trials for method, trials in trials_by_method.items()
                        if len(trials) != len(set(trials))}
    missing_trials = {method: sorted(set(range(12)) - set(trials_by_method.get(method, [])))
                      for method in method_ids if sorted(set(range(12)) - set(trials_by_method.get(method, [])))}
    payload = {
        'schema': 'rankstudy-v3-hpo-integrity',
        'root': str(args.root),
        'selection_eligible': False,
        'expected_methods': len(method_ids),
        'expected_trials': len(method_ids) * 12,
        'launcher_counts': dict(collections.Counter(row['launcher_status'] for row in records)),
        'complete_records_with_metadata': sum(row['launcher_status'] == 'complete' and row['metadata_status'] in {'complete', 'passed'} for row in records),
        'complete_input_token_counts': {str(key): value for key, value in complete_tokens.items()},
        'duplicate_trials': duplicate_trials,
        'missing_trials_by_method': missing_trials,
        'issues': issues,
        'records': records,
        'note': 'Integrity audit only; no HPO result or configuration is selected here.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: payload[key] for key in ('launcher_counts', 'complete_records_with_metadata', 'duplicate_trials', 'issues')}))


if __name__ == '__main__':
    main()
