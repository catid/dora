#!/usr/bin/env python3
"""Report full-horizon tuning progress without selecting configurations.

The tuning queue is deliberately descriptive until all 164 runs, the declared
diagnostics, and the independent screening stage are complete.  This report
keeps every launcher receipt and every endpoint measurement visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/var/tmp/dora-adaptive-rank/runs/v3_tuning_20261015'))
    parser.add_argument('--out-dir', type=Path, default=Path('reports'))
    parser.add_argument('--artifact', type=Path, default=Path('artifacts/tuning_live_v3.json'))
    args = parser.parse_args()

    records = []
    for directory in sorted(args.root.glob('*/')):
        status_path = directory / 'launcher_status.json'
        if not status_path.exists():
            continue
        launcher = json.loads(status_path.read_text())
        tuning = json.loads((directory / 'tuning_config.json').read_text()) if (directory / 'tuning_config.json').exists() else {}
        metadata = json.loads((directory / 'metadata.json').read_text()) if (directory / 'metadata.json').exists() else {}
        config = metadata.get('config', {})
        records.append({
            'run_id': directory.name,
            'method': tuning.get('method', metadata.get('method', config.get('method'))),
            'trial_index': tuning.get('trial_index'),
            'seed': tuning.get('seed', config.get('seed')),
            'lr': tuning.get('lr', config.get('lr')),
            'scale': tuning.get('scale', config.get('scale')),
            'penalty': tuning.get('penalty', config.get('penalty', 0.0)),
            'status': launcher.get('status', metadata.get('status', 'unknown')),
            'returncode': launcher.get('returncode'),
            'dev_tune_endpoint_group_balanced_nll': metadata.get('dev_tune_endpoint_group_balanced_nll'),
            'dev_tune_endpoint_token_weighted_nll': metadata.get('dev_tune_endpoint_token_weighted_nll'),
            'updates_finished': metadata.get('updates_finished'),
            'input_tokens': metadata.get('input_tokens'),
            'charged_gpu_seconds': metadata.get('charged_gpu_seconds'),
            'metadata_status': metadata.get('status'),
            'metadata': str(directory / 'metadata.json') if (directory / 'metadata.json').exists() else None,
            'selection_eligible': False,
        })

    by_method: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record['dev_tune_endpoint_group_balanced_nll']
        if record['status'] == 'complete' and value is not None:
            by_method[record['method']].append(float(value))
    methods = sorted({record['method'] for record in records if record['method']})
    summary = []
    for method in methods:
        values = by_method.get(method, [])
        mean, sd = mean_sd(values)
        summary.append({
            'method': method,
            'complete': len(values),
            'expected': 4,
            'pending_or_nonterminal': 4 - len(values),
            'mean_endpoint_dev_nll': mean if values else None,
            'sd_endpoint_dev_nll': sd if values else None,
            'min_endpoint_dev_nll': min(values) if values else None,
            'max_endpoint_dev_nll': max(values) if values else None,
            'selection_eligible': False,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0]) if summary else ['method']
    with (args.out_dir / 'tuning_live_v3.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    counts = Counter(record['status'] for record in records)
    payload = {
        'schema': 'rankstudy-v3-live-tuning-report',
        'root': str(args.root),
        'selection_eligible': False,
        'counts': dict(counts),
        'expected_runs': 164,
        'records': records,
        'method_summary': summary,
        'note': 'Descriptive full-horizon tuning progress only; no method or configuration is selected here.',
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = [
        '# Live v3 full-horizon tuning measurements', '',
        'This report is descriptive only. All rows remain selection-ineligible until the complete tuning '
        'queue, declared deletion diagnostics, and independent screening stage are finished.', '',
        '| Method | Complete / 4 | Endpoint dev NLL mean ± SD | Range |',
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
              '', 'Endpoint development NLL is not a final method-quality result.', '']
    (args.out_dir / 'tuning_live_v3.md').write_text('\n'.join(lines))
    print(json.dumps({'counts': dict(counts), 'methods': len(summary), 'selection_eligible': False}))


if __name__ == '__main__':
    main()
