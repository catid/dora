"""Plot every predeclared synthetic teacher cell, retaining family boundaries."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('lora', 'dora', 'nora', 'dora_nora', 'dora_nora_mlr', 'dora_nora_gain')
LABELS = ('LoRA', 'DoRA', 'NoRA', 'DoRA+NoRA', '+ slow magnitudes', '+ input gains')
FAMILIES = {'row_scale': 'Row scaling', 'unit_column_low_rank': 'Unit-column update',
            'heterogeneous_column_low_rank': 'Varying column amplitudes', 'mixed_row_and_direction': 'Row scaling + direction'}


def make(root, output):
    source = json.loads((root / 'stratified_summary.json').read_text())
    assert source['audit']['passed'] and source['audit']['records_checked'] == 504
    cells = list(source['cells'].values())
    assert len(cells) == 28
    values = np.array([[cell['methods'][method]['test_relative_mse_mean'] for method in METHODS] for cell in cells])
    labels = []
    for cell in cells:
        t = cell['teacher']
        q = '' if t['family'] == 'row_scale' else f", q={t['teacher_direction_rank']}"
        labels.append(f"r={t['adapter_rank']} · {FAMILIES[t['family']]}{q} · {t['coordinates']}")
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'dora-round2-teacher-20261008'})
    fig, ax = plt.subplots(figsize=(15, 13))
    fig.subplots_adjust(left=.38, right=.91, top=.86, bottom=.115)
    image = ax.imshow(np.log10(np.maximum(values, 1e-10)), cmap='viridis', vmin=-10, vmax=0, aspect='auto')
    ax.set_xticks(range(6), LABELS, rotation=24, ha='left')
    ax.xaxis.tick_top()
    ax.set_yticks(range(28), labels)
    ax.tick_params(length=0)
    for i in range(28):
        for j in range(6):
            value = values[i, j]
            label = '<1e-10' if value < 1e-10 else (f'{value:.3f}' if value >= .01 else f'{value:.1e}')
            color = '#172554' if value > .003 else 'white'
            ax.text(j, i, label, ha='center', va='center', fontsize=8.5, color=color)
    for i in range(1, 28):
        if cells[i]['teacher']['coordinates'] != cells[i-1]['teacher']['coordinates'] or cells[i]['teacher']['adapter_rank'] != cells[i-1]['teacher']['adapter_rank']:
            ax.axhline(i - .5, color='white', linewidth=1.5)
    bar = fig.colorbar(image, ax=ax, fraction=.045, pad=.02)
    bar.set_label('log10(test MSE / frozen-model MSE); lower is better')
    fig.suptitle('Controlled weight adaptation: all 28 predeclared settings', fontsize=19, fontweight='bold', y=.982)
    fig.text(.5, .950, 'Three seeds · rank r=2 or 8 · true directional rank q=r−1 or r · four full-budget validation trials per method', ha='center', fontsize=10.5)
    fig.text(.03, .061, 'Each value is mean test MSE divided by the frozen model’s MSE; 1 means no improvement. No overall winner is computed.', fontsize=10)
    fig.text(.03, .039, 'Rescaled coordinates preserve the target function and labels. Gains add parameters. These synthetic results do not establish downstream superiority.', fontsize=9.5)
    fig.text(.03, .017, 'NoRA’s column-amplitude constraint concerns the normalized low-rank branch outside its epsilon clamp; DoRA also adapts output magnitudes.', fontsize=9.5)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        path = output / f'teacher_heatmap.{ext}'
        fig.savefig(path, dpi=160, facecolor='white', metadata={'Date': None} if ext == 'svg' else None)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)
    (output / 'teacher_summary.json').write_text(json.dumps(source, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/var/tmp/dora-bench/second_round/teacher'))
    parser.add_argument('--output', type=Path, default=Path('results/2026-10-08-round2'))
    args = parser.parse_args()
    make(args.root, args.output)
