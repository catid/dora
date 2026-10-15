#!/usr/bin/env python3
"""Report completed v3 HPO measurements without selecting a method.

This report is intentionally descriptive: it includes every terminal trial,
keeps incomplete methods visible, and never writes frozen configurations or
opens the sealed audit/test data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/var/tmp/dora-adaptive-rank/runs/v3_hpo_20261015_retry'))
    parser.add_argument('--out-dir', type=Path, default=Path('reports'))
    parser.add_argument('--artifact', type=Path, default=Path('artifacts/hpo_live_v3.json'))
    args = parser.parse_args()

    records = []
    for directory in sorted(args.root.glob('*/')):
        metadata_path = directory / 'metadata.json'
        status_path = directory / 'launcher_status.json'
        config_path = directory / 'hpo_config.json'
        if not status_path.exists():
            continue
        launcher = json.loads(status_path.read_text())
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        records.append({
            'run_id': directory.name,
            'method': metadata.get('method', config.get('method', directory.name.split('_c')[0])),
            'trial_index': config.get('trial_index'),
            'lr': config.get('lr'),
            'scale': config.get('scale'),
            'penalty': config.get('penalty', 0.0),
            'status': launcher.get('status', metadata.get('status', 'unknown')),
            'returncode': launcher.get('returncode'),
            'dev_tune_endpoint_group_balanced_nll': metadata.get('dev_tune_endpoint_group_balanced_nll'),
            'dev_tune_endpoint_token_weighted_nll': metadata.get('dev_tune_endpoint_token_weighted_nll'),
            'charged_gpu_seconds': metadata.get('charged_gpu_seconds'),
            'metadata': str(metadata_path) if metadata_path.exists() else None,
            'selection_eligible': False,
        })

    by_method = defaultdict(list)
    for record in records:
        if record['status'] == 'complete' and record['dev_tune_endpoint_group_balanced_nll'] is not None:
            by_method[record['method']].append(float(record['dev_tune_endpoint_group_balanced_nll']))
    summary = []
    for method in sorted({record['method'] for record in records}):
        values = by_method.get(method, [])
        mean, sd = mean_sd(values)
        summary.append({
            'method': method,
            'complete': len(values),
            'expected': 12,
            'pending_or_nonterminal': 12 - len(values),
            'mean_endpoint_dev_nll': mean if values else None,
            'sd_endpoint_dev_nll': sd if values else None,
            'min_endpoint_dev_nll': min(values) if values else None,
            'max_endpoint_dev_nll': max(values) if values else None,
            'selection_eligible': False,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / 'hpo_live_v3.csv').open('w', newline='') as stream:
        fields = list(summary[0]) if summary else ['method']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    counts = defaultdict(int)
    for record in records:
        counts[record['status']] += 1
    payload = {
        'schema': 'rankstudy-v3-live-hpo-report',
        'root': str(args.root),
        'selection_eligible': False,
        'counts': dict(counts),
        'expected_trials': 492,
        'records': records,
        'method_summary': summary,
        'note': 'Descriptive HPO progress only. No method or configuration is selected from this report.',
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = [
        '# Live v3 HPO measurements', '',
        'This is a progress report over the corrected 492-trial short search. It is descriptive only: '
        'all rows remain selection-ineligible until the declared HPO completes and the registered tuning '
        'and screening stages run.', '',
        '| Method | Complete / 12 | Endpoint dev NLL mean ± SD | Range |',
        '|:--|--:|--:|--:|',
    ]
    for row in summary:
        if row['complete']:
            value = f"{row['mean_endpoint_dev_nll']:.6f} ± {row['sd_endpoint_dev_nll']:.6f}"
            interval = f"{row['min_endpoint_dev_nll']:.6f}–{row['max_endpoint_dev_nll']:.6f}"
        else:
            value, interval = 'pending', 'pending'
        lines.append(f"| {row['method']} | {row['complete']} / {row['expected']} | {value} | {interval} |")
    lines += ['', f"Terminal launcher records: **{sum(counts.values())}**; counts: `{dict(counts)}`.",
              '', 'The endpoint is development loss from the short horizon, not a final quality result.', '']
    (args.out_dir / 'hpo_live_v3.md').write_text('\n'.join(lines))
    print(json.dumps({'counts': dict(counts), 'methods': len(summary), 'selection_eligible': False}))


if __name__ == '__main__':
    main()
