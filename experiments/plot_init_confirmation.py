"""Render a compact bar chart from a complete, hash-bound confirmation report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SEEDS = [45, 46, 47, 48, 49]
ROLES = ['accepted_anchor', 'accepted_gaussian_scale', 'validation_winner']
LABELS = ['Accepted DoRA', 'Tuned Gaussian', 'Validation selected']
COLORS = ['#526779', '#41898b', '#bf773e']
DATASETS = [
    ('scifact', 'SciFact', 'Test nDCG@10 × 100'),
    ('nfcorpus', 'NFCorpus', 'Test nDCG@10 × 100'),
    ('scifact_transfer', 'NFCorpus → SciFact', 'Test nDCG@10 × 100'),
    ('aircraft', 'Aircraft', 'Test macro accuracy (%)'),
    ('extraction', 'ViGGO', 'Test exact JSON match (%)'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(summary_path, output):
    summary_path = Path(summary_path).resolve()
    manifest_path = summary_path.parent / 'report_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['schema'] == 'dora-confirm-report-artifacts-v1'
    assert sha(summary_path) == manifest['outputs'][summary_path.name]
    inputs_path = summary_path.parent / 'report_inputs.json'
    assert sha(inputs_path) == manifest['outputs']['report_inputs.json']
    payload = json.loads(inputs_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert payload['schema'] == 'dora-init-confirm-report-v1'
    assert summary['seeds'] == payload['seeds'] == SEEDS
    assert set(payload['tasks']) == {name for name, _, _ in DATASETS}
    rows = {(r['dataset'], r['role']): r for r in summary['results']}
    assert len(rows) == len(summary['results']) == 15
    assert set(rows) == {(name, role) for name, _, _ in DATASETS for role in ROLES}
    export = []
    for dataset, title, _ in DATASETS:
        data = payload['tasks'][dataset]
        for role, label in zip(ROLES, LABELS):
            row = rows[dataset, role]
            assert row['candidate_id'] == data['roles'][role]
            candidate = data['candidates'][row['candidate_id']]
            assert row['learning_rate'] == candidate['learning_rate']
            assert row['spec'] == candidate['spec']
            assert row['seeds'] == SEEDS
            if row['status'] == 'pass':
                assert row['n'] == len(row['values']) == 5 and not row['failed_seeds']
                assert all(math.isfinite(v) and 0 <= v <= 100 for v in row['values'])
                assert math.isclose(row['mean'], statistics.mean(row['values']), rel_tol=0, abs_tol=1e-10)
                assert math.isclose(row['sample_sd'], statistics.stdev(row['values']), rel_tol=0, abs_tol=1e-10)
            else:
                assert row['status'] == 'failed_confirmation_seed' and row['failed_seeds']
                assert 'mean' not in row and 'sample_sd' not in row
            export.append({'dataset': dataset, 'title': title, 'role': role, 'label': label,
                           'family_id': candidate['family_id'], 'spec': candidate['spec'],
                           **{k: row[k] for k in ('candidate_id', 'learning_rate', 'status', 'seeds',
                                                  'failed_seeds', 'fit_ids')},
                           'mean': row.get('mean'), 'sample_sd': row.get('sample_sd')})

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not list(output.iterdir()), 'Use a fresh presentation directory'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    limit = max([100] + [10 * math.ceil((r['mean'] + r['sample_sd'] + 9) / 10)
                        for r in export if r['status'] == 'pass'])
    for ax, (dataset, title, unit) in zip(axes.flat, DATASETS):
        for x, (role, color) in enumerate(zip(ROLES, COLORS)):
            row = rows[dataset, role]
            if row['status'] == 'pass':
                ax.bar(x, row['mean'], yerr=row['sample_sd'], color=color, width=.62,
                       capsize=4, edgecolor='white', linewidth=.8)
                ax.text(x, row['mean'] + row['sample_sd'] + 2,
                        f"{row['mean']:.2f}\n± {row['sample_sd']:.2f}",
                        ha='center', va='bottom', fontsize=9)
            else:
                ax.text(x, 5, 'Unavailable\nfailed seed', ha='center', fontsize=9, color='#555555')
        selected_id = payload['tasks'][dataset]['roles']['validation_winner']
        selected_family = payload['tasks'][dataset]['candidates'][selected_id]['family_id']
        selected_label = textwrap.fill('Selected: ' + selected_family.replace('_', ' '), width=34)
        ax.set_title(title + '\n' + selected_label, loc='left', fontweight='bold', fontsize=10)
        ax.set_xticks(range(3), ['Accepted', 'Tuned\nGaussian', 'Validation\nselected'])
        ax.set_ylabel(unit)
        ax.set_ylim(0, limit)
        ax.set_xlim(-.6, 2.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    legend = axes.flat[-1]
    legend.axis('off')
    legend.legend([Patch(facecolor=c) for c in COLORS], LABELS, loc='upper left', frameon=False)
    legend.text(.03, .61,
                'Mean ± sample SD across five new\ntraining seeds (45–49).\n\n'
                'Configurations selected using validation.\nNo selection by confirmation test scores.\n\n'
                'Held-out examples are reused.\nShared roles reuse the same fits.\n'
                'Paired intervals are in the full report.', va='top', linespacing=1.5, fontsize=10)
    fig.suptitle('DoRA initialization: fresh-seed confirmation', x=.02, ha='left', fontsize=17)
    fig.savefig(output / 'confirmation_bars.png', dpi=180)
    fig.savefig(output / 'confirmation_bars.svg')
    plt.close(fig)
    lines = ['| Dataset | Role | Selected candidate | Learning rate | Mean ± sample SD |',
             '|---|---|---|---:|---:|']
    for row in export:
        score = (f"{row['mean']:.2f} ± {row['sample_sd']:.2f}" if row['status'] == 'pass'
                 else 'Unavailable: failed seed')
        lines.append(f"| {row['title']} | {row['label']} | `{row['candidate_id']}` | "
                     f"{row['learning_rate']:.6g} | {score} |")
    (output / 'confirmation_table.md').write_text('\n'.join(lines) + '\n')
    (output / 'confirmation_rows.json').write_text(json.dumps(export, indent=2) + '\n')
    receipt = {'status': 'pass', 'scope': 'Presentation of completed confirmation results only; no new fitting or selection.',
               'seeds': SEEDS, 'rows': 15, 'source_sha256': sha(__file__),
               'inputs': {str(p): sha(p) for p in (summary_path, inputs_path, manifest_path)},
               'outputs': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    (output / 'presentation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = render(args.summary, args.output)
    print(json.dumps({'status': receipt['status'], 'rows': receipt['rows'], 'output': str(args.output)}))
