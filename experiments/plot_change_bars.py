"""Render the requested bar chart directly from the audited paired contrasts."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('results/2026-10-09-change-ablations/comparisons.json'))
    p.add_argument('--output', type=Path, default=Path('results/2026-10-09-change-bars'))
    args = p.parse_args()
    rows = json.loads(args.input.read_text())
    methods = ['ab_axis', 'ab_detach', 'ab_m_fp32', 'ab_norm_fp32', 'ab_init', 'ab_factorized']
    labels = ['Output-row axis', 'Detach denominator', 'FP32 magnitudes', 'FP32 norm work',
              'Benchmark initializer', 'Factorized projection']
    tasks = [('scifact', 'SciFact direct', 'nDCG points'), ('nfcorpus', 'NFCorpus', 'nDCG points'),
             ('scifact_transfer', 'SciFact transfer', 'nDCG points'),
             ('aircraft', 'Aircraft (rank 2)', 'accuracy points'), ('viggo', 'ViGGO extraction', 'exact-match points')]
    available = {row['task'] for row in rows}
    # Task IDs are retained exactly from the audited reporter.
    tasks = [(('extraction' if key == 'viggo' and 'extraction' in available else key), title, unit)
             for key, title, unit in tasks]
    plt.rcParams.update({'font.size': 11, 'svg.hashsalt': 'dora-change-bars-20261009'})
    figure, axes = plt.subplots(1, 5, figsize=(19, 5.5), sharey=True)
    selected = []
    for ax, (task, title, unit) in zip(axes, tasks):
        for lane, offset, color, caption in [('fixed_original_lr', -.16, '#3978b5', 'Original learning rate'),
                                             ('validation_tuned', .16, '#d97a23', 'Equal validation tuning')]:
            cells = [next(row for row in rows if row['task'] == task and row['lane'] == lane
                          and row['kind'] == 'single_change' and row['candidate'] == method) for method in methods]
            selected.extend(cells)
            mean = np.array([row['difference'] for row in cells])
            low = np.array([row['ci95'][0] for row in cells])
            high = np.array([row['ci95'][1] for row in cells])
            assert np.all(low <= mean) and np.all(high >= mean)
            ax.barh(np.arange(6) + offset, mean, height=.28, color=color, alpha=.85, label=caption,
                    xerr=np.stack([mean - low, high - mean]),
                    error_kw={'ecolor': color, 'elinewidth': 1.2, 'capsize': 2})
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xlabel('Change from original\n(' + unit + ')', fontsize=10)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(np.arange(6), labels)
    axes[0].invert_yaxis()
    handles, captions = axes[0].get_legend_handles_labels()
    figure.legend(handles, captions, loc='lower center', ncol=2, bbox_to_anchor=(.55, .05), frameon=False)
    figure.suptitle('Add one change to the original DoRA recipe', fontsize=17, y=.98)
    figure.text(.55, .01, 'Bars: paired mean differences. Whiskers: exploratory 95% intervals; 3 seeds, no multiple-comparison correction.\n'
                'Panels use different scales. Benchmark initializer is a task-adapter change; the standalone class retains historical factor orientation.',
                ha='center', fontsize=9)
    figure.subplots_adjust(left=.14, right=.99, bottom=.24, top=.87, wspace=.25)
    args.output.mkdir(parents=True, exist_ok=True)
    png, svg = [args.output / ('single_change_bars.' + suffix) for suffix in ('png', 'svg')]
    figure.savefig(png, dpi=160, facecolor='white', metadata={'Software': 'Matplotlib'})
    figure.savefig(svg, metadata={'Date': None})
    plt.close(figure)
    (args.output / 'comparisons.json').write_bytes(args.input.read_bytes())
    (args.output / 'plot_change_bars.py').write_bytes(Path(__file__).read_bytes())
    summary = {'source_comparisons_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'plotted_contrasts': len(selected), 'source': str(args.input),
               'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
               'svg_sha256': hashlib.sha256(svg.read_bytes()).hexdigest(),
               'command': 'python plot_change_bars.py --input comparisons.json --output rebuilt'}
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
