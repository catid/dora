"""Aggregate the corrected pure-LoRA IniLoRA/NoRA compatibility sweep."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
ROOT = Path('/var/tmp/dora-bench/pure_lora_mix_v3')
BASELINE = BASE / 'summary.json'
TASKS = {
    'scifact': ('SciFact', 'nDCG@10 ×100'),
    'nfcorpus': ('NFCorpus', 'nDCG@10 ×100'),
    'extraction': ('ViGGO', 'exact match %'),
    'aircraft': ('Aircraft', 'macro accuracy %'),
}
METHODS = ['lora', 'nora', 'inilora', 'inilora_nora', 'compatibility']
LABELS = {
    'lora': 'LoRA',
    'nora': 'NoRA',
    'inilora': 'IniLoRA-alpha',
    'inilora_nora': 'Original IniLoRA + NoRA',
    'compatibility': 'Best compatible mix',
}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variant_metric(task, row):
    if task in ('scifact', 'nfcorpus'):
        return 100 * row['evaluation'][task]['test']['ndcg_at_10']
    if task == 'extraction':
        return 100 * row['result']['test']['exact_match']
    return 100 * row['result']['test']['macro_class_accuracy']


def selected(root, task):
    completion = load(root / 'completion.json')
    final_rows = load(root / 'final_results.json')
    if 'selected_candidate' in completion:
        candidate = completion['selected_candidate']
        candidate_id = candidate['candidate_id']
        label = candidate['label']
        scale = candidate['scale']
        mode = candidate['mode']
        lr = candidate['selected_lr']
    else:
        ids = completion['selected_ids']
        assert len(ids) == 1, (task, ids)
        candidate_id = ids[0]
        row = next(row for row in final_rows if row['candidate_id'] == candidate_id)
        label = row['label']
        mode = candidate_id.split('_nora_', 1)[1].rsplit('_s', 1)[0]
        scale = float(candidate_id.rsplit('s', 1)[1])
        lr = row['lr']
    rows = [row for row in final_rows if row['candidate_id'] == candidate_id]
    assert len(rows) == 3, (task, candidate_id, len(rows))
    scores = [variant_metric(task, row) for row in rows]
    return {
        'candidate_id': candidate_id,
        'label': label,
        'mode': mode,
        'scale': scale,
        'selected_lr': lr,
        'scores': scores,
        'mean': statistics.mean(scores),
        'sample_sd': statistics.stdev(scores),
        'completion': completion,
        'manifest_sha256': digest(root / 'manifest.json'),
        'pilot_selection_sha256': digest(root / 'pilot_selection.json'),
        'final_results_sha256': digest(root / 'final_results.json'),
        'artifact_root': str(root),
    }


def main():
    baseline = load(BASELINE)
    summary = {
        'schema': 'pure-lora-inilora-nora-compatibility-summary-v1',
        'protocol': 'The four previously reported controls are copied from the '
                    'locked baseline summary. The compatibility row is selected '
                    'from five NoRA factor geometries × four scales × the task LR '
                    'grid using validation only; only that row receives three '
                    'fresh held-out seeds.',
        'tasks': {},
    }
    for task, (label, metric_label) in TASKS.items():
        old = baseline['tasks'][task]
        methods = {key: old['methods'][key] for key in
                   ('lora', 'nora', 'inilora', 'inilora_nora')}
        methods['compatibility'] = selected(ROOT / task, task)
        summary['tasks'][task] = {
            'label': label,
            'metric': metric_label,
            'methods': methods,
        }

    (BASE / 'compatibility_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    lines = [
        '# Pure-LoRA IniLoRA-alpha + NoRA compatibility sweep', '',
        'The first four rows are the locked baseline controls. The final row is '
        'selected on validation from five factor geometries and four initializer '
        'scales; values are mean ± sample SD over fresh seeds 42–44.', '',
        '| Method | SciFact nDCG@10 ×100 | NFCorpus nDCG@10 ×100 | ViGGO exact match % | Aircraft macro accuracy % |',
        '|:--|--:|--:|--:|--:|',
    ]
    for key in METHODS:
        cells = [f"{summary['tasks'][task]['methods'][key]['mean']:.2f} ± "
                 f"{summary['tasks'][task]['methods'][key]['sample_sd']:.2f}"
                 for task in TASKS]
        lines.append(f"| {LABELS[key]} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Selected compatibility settings:', '',
              '| Task | Geometry | Scale | LR |', '|:--|:--|--:|--:|']
    for task, (label, _) in TASKS.items():
        item = summary['tasks'][task]['methods']['compatibility']
        lines.append(f"| {label} | {item['mode']} | {item['scale']:g} | "
                     f"{item['selected_lr']:g} |")
    (BASE / 'compatibility_table.md').write_text('\n'.join(lines) + '\n')

    colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
    x = np.arange(len(METHODS))
    for ax, task in zip(axes, TASKS):
        methods = summary['tasks'][task]['methods']
        means = [methods[key]['mean'] for key in METHODS]
        sds = [methods[key]['sample_sd'] for key in METHODS]
        ax.bar(x, means, yerr=sds, capsize=4, color=colors,
               error_kw={'elinewidth': 1, 'capthick': 1})
        ax.set_title(summary['tasks'][task]['label'])
        ax.set_ylabel(summary['tasks'][task]['metric'])
        ax.set_xticks(x, ['LoRA', 'NoRA', 'IniLoRA', 'Original\ncombo', 'Best\nmix'],
                      rotation=25, ha='right')
        ax.grid(axis='y', alpha=.25)
        item = methods['compatibility']
        ax.text(4, means[4] + sds[4] + .02, f"{item['mode']}\ns={item['scale']:g}",
                ha='center', va='bottom', fontsize=7)
    fig.suptitle('Pure-LoRA IniLoRA/NoRA compatibility sweep', fontsize=13)
    fig.savefig(BASE / 'compatibility_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
