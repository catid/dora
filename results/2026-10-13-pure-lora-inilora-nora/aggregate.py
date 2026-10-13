"""Aggregate the validation-selected pure-LoRA IniLoRA/NoRA runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/var/tmp/dora-bench/pure_lora_init_v1')
OUT = Path(__file__).resolve().parent
TASKS = {
    'scifact': ('SciFact', 'nDCG@10 ×100', 'retrieval'),
    'nfcorpus': ('NFCorpus', 'nDCG@10 ×100', 'retrieval'),
    'extraction': ('ViGGO', 'exact match %', 'extraction'),
    'aircraft': ('Aircraft', 'macro accuracy %', 'aircraft'),
}
ORDER = ['lora', 'nora', 'inilora', 'inilora_nora']
LABELS = {
    'lora': 'LoRA', 'nora': 'NoRA', 'inilora': 'IniLoRA-alpha',
    'inilora_nora': 'IniLoRA-alpha + NoRA',
}


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_ids(pilot):
    standalone = [k for k in pilot if k.startswith('inilora_alpha_s')]
    combined = [k for k in pilot if k.startswith('inilora_alpha_nora_s')]
    def rank(key):
        row = pilot[key]
        if 'selected_validation_nll' in row:
            return (row['selected_validation_nll'],)
        value = row['selected_validation']
        if isinstance(value, dict):
            return (-value['macro_class_accuracy'], value['cross_entropy'])
        return (-value,)
    return {
        'lora': 'lora', 'nora': 'nora',
        'inilora': min(standalone, key=rank),
        'inilora_nora': min(combined, key=rank),
    }


def metric(task, row):
    if task in ('scifact', 'nfcorpus'):
        return 100 * row['evaluation'][task]['test']['ndcg_at_10']
    if task == 'extraction':
        return 100 * row['result']['test']['exact_match']
    return 100 * row['result']['test']['macro_class_accuracy']


def main():
    summary = {'schema': 'pure-lora-inilora-nora-summary-v1', 'tasks': {}}
    for task, (label, metric_label, _) in TASKS.items():
        root = ROOT / task
        completion = load(root / 'completion.json')
        assert completion['status'] == 'complete', (task, completion)
        pilot = load(root / 'pilot_selection.json')
        selected = pick_ids(pilot)
        finals = load(root / 'final_results.json')
        assert len(finals) == 12, (task, len(finals))
        scores = {key: [] for key in ORDER}
        for row in finals:
            key = next(key for key, candidate in selected.items()
                       if row['candidate_id'] == candidate)
            scores[key].append(metric(task, row))
        assert all(len(values) == 3 for values in scores.values()), scores
        methods = {}
        for key in ORDER:
            values = scores[key]
            methods[key] = {
                'candidate_id': selected[key],
                'label': LABELS[key],
                'scores': values,
                'mean': statistics.mean(values),
                'sample_sd': statistics.stdev(values),
                'selected_lr': pilot[selected[key]]['selected_lr'],
            }
            if key in ('inilora', 'inilora_nora'):
                methods[key]['selected_scale'] = float(
                    selected[key].rsplit('s', 1)[1])
        summary['tasks'][task] = {
            'label': label, 'metric': metric_label, 'selected': selected,
            'methods': methods,
            'completion': completion,
            'manifest_sha256': digest(root / 'manifest.json'),
            'pilot_selection_sha256': digest(root / 'pilot_selection.json'),
            'final_results_sha256': digest(root / 'final_results.json'),
            'artifact_root': str(root),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [
        '# Pure-LoRA IniLoRA-alpha + NoRA comparison', '',
        'Validation-selected scale and learning rate; mean ± sample SD over seeds 42–44.', '',
        '| Method | SciFact nDCG@10 ×100 | NFCorpus nDCG@10 ×100 | ViGGO exact match % | Aircraft macro accuracy % |',
        '|:--|--:|--:|--:|--:|',
    ]
    for key in ORDER:
        cells = []
        for task in TASKS:
            item = summary['tasks'][task]['methods'][key]
            cells.append(f"{item['mean']:.2f} ± {item['sample_sd']:.2f}")
        lines.append(f"| {LABELS[key]} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Selected settings:', '',
              '| Task | LoRA LR | NoRA LR | IniLoRA scale / LR | IniLoRA + NoRA scale / LR |',
              '|:--|--:|--:|--:|--:|']
    for task, (label, _, _) in TASKS.items():
        methods = summary['tasks'][task]['methods']
        lines.append('| ' + ' | '.join([
            label, f"{methods['lora']['selected_lr']:g}",
            f"{methods['nora']['selected_lr']:g}",
            f"{methods['inilora']['selected_scale']:g} / {methods['inilora']['selected_lr']:g}",
            f"{methods['inilora_nora']['selected_scale']:g} / {methods['inilora_nora']['selected_lr']:g}",
        ]) + ' |')
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')

    colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756']
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), constrained_layout=True)
    x = np.arange(4)
    for ax, (task, (label, metric_label, _)) in zip(axes, TASKS.items()):
        methods = summary['tasks'][task]['methods']
        means = [methods[key]['mean'] for key in ORDER]
        sds = [methods[key]['sample_sd'] for key in ORDER]
        ax.bar(x, means, yerr=sds, capsize=4, color=colors,
               error_kw={'elinewidth': 1, 'capthick': 1})
        ax.set_title(label)
        ax.set_ylabel(metric_label)
        ax.set_xticks(x, ['LoRA', 'NoRA', 'IniLoRA', 'Ini+NoRA'], rotation=25,
                      ha='right')
        ax.grid(axis='y', alpha=.25)
        for index, key in enumerate(('inilora', 'inilora_nora')):
            ax.text(index + 2, means[index + 2] + sds[index + 2] + .02,
                    f"s={methods[key]['selected_scale']:g}",
                    ha='center', va='bottom', fontsize=8)
    fig.suptitle('Pure-LoRA initialization comparison (validation-retuned scale)',
                 fontsize=13)
    fig.savefig(OUT / 'pure_lora_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
