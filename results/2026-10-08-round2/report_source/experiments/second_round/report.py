"""Rebuild round2 tables/figures from portable predictions without model inference.

python -m experiments.second_round.report --raw-root /var/tmp/dora-bench/round2
Partial previews require --allow-partial and an explicit separate output directory.
"""
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from experiments.second_round.archive import bundle, sha256, verify_extracted
from experiments.second_round.analyze_cogs import check_predictions
from experiments.second_round.analyze_vision import record_directory, validate as validate_vision
from experiments.second_round.validate_retrieval import metrics as retrieval_metrics, validate as validate_retrieval
from experiments.second_round.validate_control_vision import validate as validate_vision_control

METHODS = ('lora', 'dora', 'nora', 'dora_nora', 'dora_nora_mlr', 'dora_nora_gain')
LABELS = {'baseline': 'Task baseline', 'lora': 'LoRA', 'dora': 'DoRA', 'nora': 'NoRA',
          'dora_nora': 'DoRA+NoRA', 'dora_nora_mlr': 'DoRA+NoRA (slow magnitudes)', 'dora_nora_gain': 'DoRA+NoRA (input gains)'}
SHORT_LABELS = ('LoRA', 'DoRA', 'NoRA', 'DoRA+\nNoRA', '+ slow\nmagnitudes', '+ input\ngains')
COLORS = dict(zip(METHODS, ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00')))
TASKS = {
    'aircraft': {'title': 'FGVC-Aircraft', 'metric': 'Macro accuracy (%)', 'baseline': 'Frozen ViT-B/16 backbone + trained classifier', 'ranks': (2, 8), 'baseline_seeds': 3},
    'nfcorpus': {'title': 'NFCorpus retrieval', 'metric': 'nDCG@10 ×100', 'baseline': 'Frozen MiniLM', 'ranks': (2, 8), 'baseline_seeds': 1},
    'scifact': {'title': 'SciFact transfer', 'metric': 'nDCG@10 ×100', 'baseline': 'Frozen MiniLM; no SciFact training', 'ranks': (2, 8), 'baseline_seeds': 1},
    'cogs': {'title': 'COGS generalization', 'metric': 'OOD atom exact match (%)', 'baseline': 'Frozen Qwen2.5-3B-Instruct', 'ranks': (8,), 'baseline_seeds': 1},
}


COGS_GENERATION_CAP_NOTE = (
    "The executed COGS runner recorded `hit_generation_cap`/`generation_cap_hits` by checking "
    "whether tokenizer EOS 151645 was absent. The pinned generation configuration also stops on "
    "EOS 151643, so these flags can falsely indicate token-limit exhaustion. Raw generated token "
    "IDs were not saved, preventing an exact retrospective cap audit. Saved text and its atom-set "
    "exact match, strict exact match, validity and F1 scores are unaffected. The archive retains "
    "the executed source; later EOS-detection or raw-token-logging fixes apply only to future runs "
    "and do not change these records."
)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=2e-7), (a, b)


def distribution(values):
    return {'mean': statistics.mean(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None, 'n': len(values), 'values': values}


def format_cell(value):
    if value['mean'] is None:
        return 'Pending'
    text = f"{value['mean']:.2f}"
    if value['sample_sd'] is not None:
        text += f" ± {value['sample_sd']:.2f}"
    return text


def sources(root, mapping):
    for name, expected in mapping.items():
        assert sha256(root / 'source' / name) == expected, root / name
    return len(mapping)


def finalize(name, records, provenance, uncertainty, audit):
    definition = TASKS[name]
    ranks, missing = {}, []
    for rank in definition['ranks']:
        selected = [row for row in records if row['rank'] == rank]
        summary = {}
        for method in ('baseline', *METHODS):
            rows = sorted((row for row in selected if row['method'] == method), key=lambda row: row['seed'] or 0)
            seeds = [row['seed'] for row in rows]
            assert len(seeds) == len(set(seeds)), (name, rank, method)
            expected = definition['baseline_seeds'] if method == 'baseline' else 3
            if len(rows) != expected:
                missing.append(f'{name} rank{rank} {method}: {len(rows)}/{expected}')
            item = distribution([row['value'] for row in rows])
            item['seeds'] = seeds
            item['trainable_parameters'] = sorted(set(row['trainable_parameters'] for row in rows))
            for field in ('training_seconds', 'peak_memory_bytes'):
                item[field] = distribution([row[field] for row in rows if row.get(field) is not None])
            summary[method] = item
        ranks[str(rank)] = {'runs': selected, 'summary': summary}
    return {**definition, 'task': name, 'ranks': ranks, 'complete': not missing, 'missing': missing,
            'provenance': provenance, 'paired_uncertainty': uncertainty, 'audit': audit}


def aircraft(raw):
    records, provenance, uncertainty, audit = [], {}, {}, {}
    for rank in (2, 8):
        root = raw / 'vision' / f'rank{rank}_run'
        rows = read(root / 'results.json', [])
        protocol = read(root / 'protocol.json', {})
        manifest = read(root / 'split_manifest.json', {})
        provenance[str(rank)] = protocol
        uncertainty[str(rank)] = read(root / 'analysis.json', {})
        if protocol:
            sources(root, protocol['provenance']['source_sha256'])
        labels = np.array([row['label'] for row in manifest.get('splits', {}).get('test', [])])
        count = np.bincount(labels, minlength=100) if labels.size else np.zeros(100)
        predictions_scored = 0
        for row in rows:
            assert row['test'] and row['rank'] in (0, rank)
            prediction = read(record_directory(root, row) / 'test_predictions.json')
            assert prediction is not None
            actual_labels, guesses = np.array(prediction['labels']), np.array(prediction['predictions'])
            assert np.array_equal(labels, actual_labels) and len(labels) == row['test']['count']
            correct = labels == guesses
            assert int(correct.sum()) == row['test']['correct']
            close(float(correct.mean()), row['test']['accuracy'])
            class_correct = np.bincount(labels[correct], minlength=100)
            close(float((class_correct / count).mean()), row['test']['macro_class_accuracy'])
            close(float(np.mean(prediction['true_class_nll'])), row['test']['cross_entropy'])
            close(float(np.mean(prediction['top5_correct'])), row['test']['top5_accuracy'])
            assert row['frozen_base_unchanged']
            predictions_scored += len(labels)
            records.append({'method': row['method'], 'rank': rank, 'adapter_rank': row['rank'], 'seed': row['seed'],
                            'value': 100 * row['test']['macro_class_accuracy'], 'test': row['test'],
                            'trainable_parameters': row['trainable_parameters'], 'learning_rate': row['adapter_learning_rate'],
                            'magnitude_lr_multiplier': row['magnitude_lr_multiplier'], 'selected_epoch': row['best_epoch'],
                            'training_seconds': row['training_wall_seconds'], 'peak_memory_bytes': row['peak_cuda_allocated_bytes']})
        if len(rows) == 21:
            _, _, _, _, full_audit = validate_vision(root, require_checkpoints=False)
            audit[str(rank)] = full_audit
        else:
            audit[str(rank)] = {'complete': False, 'available_final_records_scored': len(rows), 'predictions_scored': predictions_scored}
    task = finalize('aircraft', records, provenance, uncertainty, audit)
    task['initial_numerical_controls'] = {}
    for name in ('numerical_control', 'numerical_control_full'):
        root = raw / 'vision' / name
        if (root / 'result.json').exists():
            task['initial_numerical_controls'][name] = {
                'result': read(root / 'result.json'),
                'audit': validate_vision_control(root, write_audit=False),
                'mechanism': read(root / 'mechanism.json')}
    return task


def retrieval(raw):
    root = raw / 'retrieval'
    data = read(root / 'results.json', {})
    provenance = data.get('provenance', {})
    if provenance:
        sources(root, provenance['source_sha256'])
    controls = read(root / 'initial_adapter_control.json', {})
    uncertainty = read(root / 'comparisons.json', {})
    audit = validate_retrieval(root, write_audit=False) if len(data.get('runs', [])) == 37 else {'complete': False}
    tasks = {}
    for name in ('nfcorpus', 'scifact'):
        qrels = read(root / f'{name}_qrels.json', {})
        splits = read(root / f'{name}_splits.json', {})
        records = []
        for row in data.get('runs', []):
            directory = root / Path(row['artifact_directory']).name
            predictions = read(directory / name / 'test_per_query.json')
            assert [item['query_id'] for item in predictions] == splits['test']
            for item in predictions:
                expected = retrieval_metrics(item['top_10'], qrels[item['query_id']])
                for metric, value in expected.items():
                    close(value, item[metric])
            evaluation = row['evaluation'][name]['test']
            for metric in ('recall_at_10', 'mrr_at_10', 'ndcg_at_10'):
                close(statistics.mean(item[metric] for item in predictions), evaluation[metric])
            for rank in (2, 8) if row['method'] == 'frozen' else (row['rank'],):
                training = row.get('training', {})
                records.append({'method': 'baseline' if row['method'] == 'frozen' else row['method'], 'rank': rank,
                                'adapter_rank': row['rank'], 'seed': row['seed'], 'value': 100 * evaluation['ndcg_at_10'],
                                'test': evaluation, 'learning_rate': row.get('learning_rate'),
                                'magnitude_lr_multiplier': row.get('magnitude_lr_multiplier'),
                                'trainable_parameters': row['trainable_parameters'], 'training_seconds': training.get('seconds'),
                                'peak_memory_bytes': training.get('peak_allocated_bytes')})
        task = finalize(name, records, provenance, uncertainty, audit)
        task['initial_numerical_control'] = controls
        task['validation_tuning'] = data.get('tuning', {})
        tasks[name] = task
    return tasks


def cogs(raw):
    records, provenance = [], {}
    analysis = read(raw / 'cogs' / 'analysis.json', {})
    baseline = None
    predictions_scored, source_count, checkpoints_present, checkpoints_omitted = 0, 0, 0, 0
    for method in METHODS:
        root = raw / 'cogs' / method
        data = read(root / 'results.json', {})
        manifest = read(root / 'manifest.json', {})
        splits = read(root / 'split_manifest.json', {})
        if not manifest:
            continue
        provenance[method] = manifest
        source_count += sources(root, manifest['source_sha256'])
        current = read(root / 'baseline.json')
        if current:
            if baseline is None:
                baseline = current
                baseline_root, baseline_splits = root, splits
            else:
                assert current == baseline
        trials = data.get('tuning', [])
        if trials:
            assert len(trials) == 4
            winner = min(trials, key=lambda row: row['validation_nll'])
            for trial in trials:
                directory = root / Path(trial['directory']).name
                assert read(directory / 'result.json')['validation_nll'] == trial['validation_nll']
        for row in data.get('results', []):
            assert row['method'] == method and row['rank'] == 8 and row['frozen_weights_unchanged']
            assert row['lr'] == winner['lr'] and row['magnitude_lr_multiplier'] == winner['magnitude_lr_multiplier']
            directory = root / f'{method}_seed{row["seed"]}'
            checkpoint = directory / 'adapter.pt'
            if checkpoint.exists():
                assert sha256(checkpoint) == row['checkpoint_sha256']
                checkpoints_present += 1
            else:
                checkpoints_omitted += 1
            for split, key in (('iid', 'test'), ('ood', 'gen')):
                gold = {item['id']: item for item in splits[key]}
                check_predictions(directory / f'{split}_predictions.jsonl', gold, row[split])
                predictions_scored += len(gold)
            records.append({'method': method, 'rank': 8, 'adapter_rank': 8, 'seed': row['seed'],
                            'value': row['ood']['category_macro_exact'] * 100, 'test': row['ood'], 'iid': row['iid'],
                            'trainable_parameters': row['trainable_parameters'], 'learning_rate': row['lr'],
                            'magnitude_lr_multiplier': row['magnitude_lr_multiplier'], 'selected_step': row['selected_step'],
                            'training_seconds': row['train_and_validation_seconds'], 'peak_memory_bytes': row['peak_allocated_bytes']})
    if baseline is not None:
        # Baseline raw files live only in the original LoRA worker; other workers
        # reuse its exact manifest-checked baseline JSON.
        baseline_root = raw / 'cogs' / 'lora'
        for split, key in (('iid', 'test'), ('ood', 'gen')):
            gold = {item['id']: item for item in baseline_splits[key]}
            check_predictions(baseline_root / f'baseline_{split}.jsonl', gold, baseline[split])
            predictions_scored += len(gold)
        records.append({'method': 'baseline', 'rank': 8, 'adapter_rank': 0, 'seed': None, 'value': baseline['ood']['category_macro_exact'] * 100,
                        'test': baseline['ood'], 'iid': baseline['iid'], 'trainable_parameters': 0})
    audit = {'raw_predictions_rescored': predictions_scored, 'source_snapshots_verified': source_count,
             'final_checkpoint_hashes_verified': checkpoints_present, 'final_checkpoint_payloads_omitted': checkpoints_omitted,
             'full_local_audit_attestation': analysis,
             'generation_cap_audit_scope': COGS_GENERATION_CAP_NOTE,
             'scope': 'Report independently re-scores saved text and verifies present hashes and validation-only selection. Full trial/optimizer/checkpoint audit is retained from analyze_cogs; omitted checkpoint payloads cannot be rechecked from this archive.'}
    return finalize('cogs', records, provenance, analysis.get('paired_intervals', {}), audit)


def table(tasks, rank, output):
    names = [name for name, task in tasks.items() if str(rank) in task['ranks']]
    lines = ['| Method | ' + ' | '.join(tasks[name]['title'] for name in names) + ' |',
             '|:--|' + ':--|' * len(names)]
    csv_rows = []
    for method in ('baseline', *METHODS):
        values = [tasks[name]['ranks'][str(rank)]['summary'][method] for name in names]
        lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(format_cell(value) for value in values) + ' |')
        for name, value in zip(names, values):
            csv_rows.append({'rank': rank, 'task': name, 'method': method, 'mean': value['mean'], 'sample_sd': value['sample_sd'], 'seeds': value['n']})
    lines += ['', 'Values are mean ± sample standard deviation across three training seeds; frozen baselines have one deterministic evaluation. Aircraft uses three trained-head baseline seeds. All metrics are on a 0–100 scale; higher is better. These are separate tasks, so no cross-task average or winner is computed.', '',
              'Aircraft: macro accuracy. NFCorpus/SciFact: nDCG@10 ×100. COGS: atom-set exact match on the balanced 672-example OOD sample (32 per category).']
    (output / f'rank{rank}_table.md').write_text('\n'.join(lines) + '\n')
    with (output / f'rank{rank}_table.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=('rank', 'task', 'method', 'mean', 'sample_sd', 'seeds'), lineterminator='\n')
        writer.writeheader(); writer.writerows(csv_rows)


def plot(tasks, rank, output, complete):
    names = [name for name in tasks if str(rank) in tasks[name]['ranks']]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'dora-round2', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.77, wspace=.23, hspace=.67)
    fig.suptitle(f'Harder adaptation experiments · rank {rank}' + ('' if complete else ' · PARTIAL'), fontsize=22, weight='bold', y=.976)
    fig.text(.5, .934, 'Six methods · matched seeds · four validation-only tuning trials per method · mean ± seed SD', ha='center', color='#475569', fontsize=12)
    handles = [Patch(facecolor=COLORS[method], label=LABELS[method]) for method in METHODS]
    handles.append(Line2D([0], [0], color='#475569', linestyle='--', label='Task baseline'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .908), ncol=4, frameon=False, fontsize=11)
    for index, ax in enumerate(axes.flat):
        if index >= len(names):
            ax.set_visible(False)
            continue
        name, task = names[index], tasks[names[index]]
        summary = task['ranks'][str(rank)]['summary']
        ax.set_axisbelow(True); ax.grid(axis='y', color='#e2e8f0', linewidth=.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_ylim(0, 105); ax.set_yticks(range(0, 101, 20)); ax.set_xlim(-.6, 5.6)
        ax.set_xticks(range(6), SHORT_LABELS, fontsize=10)
        ax.set_title(task['title'], loc='left', weight='bold', fontsize=16, pad=28)
        counts = sorted({summary[method]['n'] for method in METHODS})
        ax.text(0, 1.045, f'n={"/".join(map(str, counts))} completed seeds per method', transform=ax.transAxes, fontsize=10, color='#475569')
        ax.set_ylabel(task['metric'])
        baseline = summary['baseline']['mean']
        if baseline is not None:
            ax.axhline(baseline, color='#475569', linestyle='--', linewidth=1.4)
        for position, method in enumerate(METHODS):
            value, error = summary[method]['mean'], summary[method]['sample_sd'] or 0
            if value is None:
                ax.text(position, 3, 'Pending', rotation=90, ha='center', va='bottom', color='#64748b')
                continue
            ax.bar(position, value, color=COLORS[method], width=.68, zorder=2)
            if error:
                ax.errorbar(position, value, yerr=error, fmt='none', color='#172554', capsize=3, lw=1.2, zorder=3)
            ax.text(position, value + error + 2, f'{value:.2f}', ha='center', fontsize=10)
        label = 'pending' if baseline is None else f'{baseline:.2f}'
        ax.text(0, -.29, f'Baseline: {task["baseline"]} ({label})', transform=ax.transAxes, fontsize=9.3, color='#475569')
    fig.text(.075, .069, 'All axes start at zero. Task-specific results and exploratory paired intervals do not establish a universal ranking.', fontsize=11, color='#334155')
    fig.text(.075, .043, 'SciFact is transfer-only after NFCorpus training. The input-gain variant adds trainable parameters; the slow-magnitude variant changes optimizer rates.', fontsize=10.5, color='#475569')
    for extension in ('png', 'svg'):
        path = output / f'rank{rank}_comparison.{extension}'
        fig.savefig(path, dpi=170, metadata={'Date': None} if extension == 'svg' else {})
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def secondary_tables(tasks, output):
    lines = ['| Method | IID atom exact | OOD atom exact | Structural macro | Lexical macro | OOD strict exact | OOD valid syntax | OOD atom micro F1 |',
             '|:--|--:|--:|--:|--:|--:|--:|--:|']
    selected = tasks['cogs']['ranks']['8']['runs']
    for method in ('baseline', *METHODS):
        rows = [row for row in selected if row['method'] == method]
        pairs = [('iid', 'atom_exact'), ('test', 'category_macro_exact'), ('test', 'structural_macro_exact'), ('test', 'lexical_macro_exact'),
                 ('test', 'strict_exact'), ('test', 'valid'), ('test', 'atom_micro_f1')]
        values = [distribution([row[group][metric] * 100 for row in rows]) for group, metric in pairs]
        lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(format_cell(value) for value in values) + ' |')
    lines += ['', 'COGS structural and lexical results remain separate: 18 lexical categories and 3 structural categories. Category sample sizes are equal. IID is a separate 128-example sample. Values are percentages; variability is across training seeds.']
    lines += ['', COGS_GENERATION_CAP_NOTE]
    (output / 'cogs_metrics.md').write_text('\n'.join(lines) + '\n')
    categories = sorted({category for row in selected for category in row['test']['categories']})
    lines = ['| COGS category | ' + ' | '.join(LABELS[method] for method in ('baseline', *METHODS)) + ' |', '|:--|' + '--:|' * 7]
    for category in categories:
        values = [distribution([row['test']['categories'][category]['atom_exact'] * 100 for row in selected if row['method'] == method]) for method in ('baseline', *METHODS)]
        lines.append('| ' + category + ' | ' + ' | '.join(format_cell(value) for value in values) + ' |')
    (output / 'cogs_categories.md').write_text('\n'.join(lines) + '\n')
    lines = ['| Task | Rank | Method | Recall@10 ×100 | MRR@10 ×100 | nDCG@10 ×100 |', '|:--|--:|:--|--:|--:|--:|']
    for name in ('nfcorpus', 'scifact'):
        for rank, rank_data in tasks[name]['ranks'].items():
            for method in ('baseline', *METHODS):
                rows = [row for row in rank_data['runs'] if row['method'] == method]
                values = [distribution([row['test'][metric] * 100 for row in rows]) for metric in ('recall_at_10', 'mrr_at_10', 'ndcg_at_10')]
                lines.append(f'| {name} | {rank} | {LABELS[method]} | ' + ' | '.join(format_cell(value) for value in values) + ' |')
    (output / 'retrieval_metrics.md').write_text('\n'.join(lines) + '\n')



def cogs_category_plot(task, output, complete):
    structural = {'cp_recursion', 'pp_recursion', 'obj_pp_to_subj_pp'}
    records = task['ranks']['8']['runs']
    categories = {category for row in records for category in row['test']['categories']}
    if not categories:
        return
    assert len(categories) == 21 and structural <= categories
    lexical = sorted(categories - structural)
    ordered = lexical + sorted(structural)
    methods = ('baseline', *METHODS)
    values = np.full((21, 7), np.nan)
    counts = []
    for column, method in enumerate(methods):
        rows = [row for row in records if row['method'] == method]
        counts.append(len(rows))
        for index, category in enumerate(ordered):
            assert all(row['test']['categories'][category]['count'] == 32 for row in rows)
            if rows:
                values[index, column] = 100 * statistics.mean(row['test']['categories'][category]['atom_exact'] for row in rows)
    fig, ax = plt.subplots(figsize=(16, 13))
    fig.subplots_adjust(left=.43, right=.91, top=.83, bottom=.12)
    color_map = plt.get_cmap('viridis').copy()
    color_map.set_bad('#e2e8f0')
    image = ax.imshow(np.ma.masked_invalid(values), cmap=color_map, vmin=0, vmax=100, aspect='auto')
    labels = ['Frozen\nbaseline', 'LoRA', 'DoRA', 'NoRA', 'DoRA+\nNoRA', '+ slow\nmagnitudes', '+ input\ngains']
    ax.set_xticks(range(7), [f'{label}\n(n={count})' for label, count in zip(labels, counts)], fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', pad=10)
    ax.set_yticks(range(21), ordered, fontsize=9.8)
    ax.tick_params(length=0)
    for row in range(21):
        for column in range(7):
            value = values[row, column]
            text = '—' if np.isnan(value) else f'{value:.1f}'
            color = '#475569' if np.isnan(value) else ('#0f172a' if value >= 60 else 'white')
            ax.text(column, row, text, ha='center', va='center', fontsize=10, color=color)
    ax.axhline(17.5, color='white', linewidth=4)
    ax.axhline(17.5, color='#334155', linewidth=.8)
    position = ax.get_position()
    fig.text(.035, position.y1 - position.height * 9/21, 'LEXICAL · 18 fixed categories', rotation=90,
             ha='center', va='center', fontsize=12, weight='bold', color='#334155')
    fig.text(.035, position.y0 + position.height * 1.5/21, 'STRUCTURAL · 3', rotation=90,
             ha='center', va='center', fontsize=11, weight='bold', color='#334155')
    bar = fig.colorbar(image, ax=ax, fraction=.035, pad=.025, ticks=range(0, 101, 20))
    bar.set_label('Atom-set exact match (%)', fontsize=11)
    fig.suptitle('COGS generalization · all 21 fixed categories' + ('' if complete else ' · PARTIAL'),
                 fontsize=21, weight='bold', y=.974)
    fig.text(.5, .937, 'Rank 8 · 32 fixed held-out examples per category · shared 0–100 scale', ha='center', fontsize=12, color='#475569')
    seed_caption = ('Cells average three matched training seeds; the frozen baseline has one deterministic evaluation.' if complete else
                    'Cells average the completed training seeds shown above (target: three per method); the frozen baseline has one evaluation.')
    fig.text(.07, .065, seed_caption, fontsize=10.5, color='#334155')
    fig.text(.07, .038, 'Lexical and structural categories are grouped separately. Categories are fixed by COGS, not selected by observed method performance.', fontsize=10.5, color='#475569')
    for extension in ('png', 'svg'):
        path = output / f'cogs_category_heatmap.{extension}'
        fig.savefig(path, dpi=170, metadata={'Date': None} if extension == 'svg' else {})
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def matched_magnitude_report(raw, output, require_complete):
    from experiments.second_round.matched_magnitude import analyze
    diagnostic = analyze(raw)
    recorded = read(raw / 'matched_magnitude_validation.json')
    if require_complete:
        assert diagnostic['complete'] and len(diagnostic['comparisons']) == 132
        assert diagnostic == recorded, 'Validation-only diagnostic differs from its captured evidence'
    write(output / 'matched_magnitude_validation.json', diagnostic)
    lines = ['This is a validation-only optimizer diagnostic at matched factor learning rate, rank, seed and trial budget. It does not measure test superiority or change any selection.', '',
             '| Task | Validation metric | Magnitude LR multiplier | Matched validation pairs | Strictly better / worse / tied | Median benefit in validation units |',
             '|:--|:--|--:|--:|:--|--:|']
    for row in diagnostic['descriptive_summary']:
        lines.append(f'| {row["task"]} | {row["metric"]} | {row["magnitude_lr_multiplier"]:g} | {row["pairs"]} | {row["strictly_better"]} / {row["strictly_worse"]} / {row["exact_ties"]} | {row["median_benefit"]:+.6g} |')
    lines += ['', 'Positive benefit means lower validation MSE/NLL or higher validation accuracy/nDCG, as appropriate. Metrics have different units and are not combined across tasks.', '', diagnostic['limitations']]
    if diagnostic['pending']:
        lines += ['', 'Incomplete preview: ' + json.dumps(diagnostic['pending'])]
    (output / 'matched_magnitude_validation.md').write_text('\n'.join(lines) + '\n')
    return {'complete': diagnostic['complete'], 'pairs': len(diagnostic['comparisons']),
            'input_files': len(diagnostic['input_sha256']), 'validation_only': True}


def resource_tables(tasks, output):
    lines = ['| Task | Rank | Method | Trainable parameters | Training seconds, mean ± SD | Peak allocated GiB, max |', '|:--|--:|:--|--:|--:|--:|']
    selection = ['| Task | Rank | Method | Base LR | Magnitude LR multiplier | Selected epoch/step |', '|:--|--:|:--|--:|--:|:--|']
    for name in ('aircraft', 'nfcorpus', 'cogs'):
        for rank, rank_data in tasks[name]['ranks'].items():
            for method in ('baseline', *METHODS):
                rows = [row for row in rank_data['runs'] if row['method'] == method]
                summary = rank_data['summary'][method]
                params = '/'.join(f'{count:,}' for count in summary['trainable_parameters']) or 'Pending'
                memory = summary['peak_memory_bytes']['values']
                peak = f'{max(memory) / 2**30:.3f}' if memory else '—'
                training_text = format_cell(summary['training_seconds']) if summary['training_seconds']['n'] else ('—' if rows else 'Pending')
                lines.append(f'| {name} | {rank} | {LABELS[method]} | {params} | {training_text} | {peak} |')
                rates = sorted({row['learning_rate'] for row in rows if row.get('learning_rate') is not None})
                multipliers = sorted({row['magnitude_lr_multiplier'] for row in rows if row.get('magnitude_lr_multiplier') is not None})
                checkpoints = [row.get('selected_epoch', row.get('selected_step', 'final epoch')) for row in rows if row.get('learning_rate') is not None]
                selection.append(f'| {name} | {rank} | {LABELS[method]} | {", ".join(f"{rate:g}" for rate in rates) or "—"} | {", ".join(f"{m:g}" for m in multipliers) or "—"} | {checkpoints or ("—" if rows else "Pending")} |')
    lines += ['', 'Timing scopes differ by task. Aircraft includes per-epoch validation and disk saves of improved checkpoints. COGS includes periodic validation and copies of the best state; its timer excludes final checkpoint save/restore and generation. Retrieval training time ends before the adapter save and excludes validation/test retrieval. Model loading and test inference are excluded throughout; compare recorded training times within a task. Retrieval has one training job for both retrieval datasets. Peak allocated memory differs from device-reserved memory. Parameter counts include the Aircraft classifier and extra input gains where applicable.', '',
              'Retrieval inference was measured but the first native frozen baseline incurred shape-specific startup cost; those cold/warm timings are retained in raw records and are not used for an inference-efficiency claim.']
    (output / 'runtime_memory.md').write_text('\n'.join(lines) + '\n')
    (output / 'selected_settings.md').write_text('\n'.join(selection) + '\n')


def comparison_rows(tasks):
    rows = []
    for rank, analysis in tasks['aircraft']['paired_uncertainty'].items():
        for row in analysis.get('comparisons', []):
            metric = row['metrics']['macro_class_accuracy']
            rows.append(('aircraft', row['rank'], row['candidate'], row['reference'], metric['mean_difference'], metric['percentile_95_interval']))
    for row in tasks['nfcorpus']['paired_uncertainty'].get('comparisons', []):
        metric = row['metrics']['ndcg_at_10']
        rows.append((row['dataset'], row['rank'], row['candidate'], row['reference'], metric['mean_difference'], metric['percentile_95_interval']))
    for scope in ('ood', 'lexical', 'structural'):
        for key, metric in tasks['cogs']['paired_uncertainty'].get(scope, {}).items():
            candidate, reference = key.split('_minus_')
            rows.append((f'cogs {scope}', 8, candidate, reference, metric['difference'], metric['ci95']))
    return rows


def uncertainty_table(tasks, output):
    lines = ['| Task | Rank | Contrast | Difference, percentage points | Exploratory 95% interval |', '|:--|--:|:--|--:|:--|']
    for name, rank, candidate, reference, difference, interval in comparison_rows(tasks):
        lines.append(f'| {name} | {rank} | {LABELS[candidate]} − {LABELS[reference]} | {100*difference:+.3f} | [{100*interval[0]:+.3f}, {100*interval[1]:+.3f}] |')
    lines += ['', '10,000 paired bootstrap replicates resample matched training seeds and held-out examples; Aircraft preserves class strata and COGS preserves category strata. Retrieval resamples shared queries. Intervals are exploratory, without multiple-comparison correction. Three seeds provide limited evidence about seed variability; intervals do not establish generalization to other tasks.']
    (output / 'paired_differences.md').write_text('\n'.join(lines) + '\n')



def difference_plot(tasks, output, complete):
    contrasts = [('nora', 'lora'), ('dora', 'lora'), ('dora_nora', 'nora'),
                 ('dora_nora_mlr', 'dora_nora'), ('dora_nora_gain', 'dora_nora')]
    labels = ['NoRA − LoRA', 'DoRA − LoRA', 'DoRA+NoRA − NoRA',
              'Slow magnitudes − DoRA+NoRA', 'Input gains − DoRA+NoRA']
    mapping = {(name, rank, first, second): (difference, interval)
               for name, rank, first, second, difference, interval in comparison_rows(tasks)}
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    fig.subplots_adjust(left=.19, right=.98, top=.84, bottom=.15, wspace=.75, hspace=.55)
    fig.suptitle('Paired method differences · rank 8' + ('' if complete else ' · PARTIAL'), fontsize=21, weight='bold', y=.965)
    fig.text(.5, .916, 'Matched training seeds and held-out examples · exploratory 95% bootstrap intervals', ha='center', fontsize=12, color='#475569')
    for ax, (name, title) in zip(axes.flat, [('aircraft', 'FGVC-Aircraft'), ('nfcorpus', 'NFCorpus retrieval'),
                                           ('scifact', 'SciFact transfer'), ('cogs ood', 'COGS generalization')]):
        ax.axvline(0, color='#475569', linestyle='--', linewidth=1.2)
        bound = .1
        for index, (first, second) in enumerate(contrasts):
            item = mapping.get((name, 8, first, second))
            if item is None:
                continue
            difference, interval = item
            mean, low, high = 100*difference, 100*interval[0], 100*interval[1]
            bound = max(bound, abs(mean), abs(low), abs(high))
            ax.hlines(index, low, high, color=COLORS[first], linewidth=2)
            ax.plot(mean, index, 'o', color=COLORS[first], markersize=6)
        ax.set_xlim(-bound*1.15, bound*1.15)
        ax.set_ylim(4.6, -.6)
        ax.set_yticks(range(5), labels, fontsize=10)
        ax.set_xlabel('Difference, percentage points')
        ax.set_title(title, loc='left', fontsize=15, weight='bold', pad=14)
        ax.grid(axis='x', color='#e2e8f0', linewidth=.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='y', length=0)
        if not any((name, 8, first, second) in mapping for first, second in contrasts):
            ax.text(.5, .5, 'Paired analysis pending', ha='center', va='center', transform=ax.transAxes, color='#64748b')
    fig.text(.19, .072, 'Positive values favor the first method. Each panel uses its own difference scale; zero marks no measured difference.', fontsize=10.5, color='#334155')
    fig.text(.19, .043, '10,000 replicates; no multiple-comparison correction. Three seeds provide limited evidence about seed variability.', fontsize=10.5, color='#475569')
    for extension in ('png', 'svg'):
        path = output / f'rank8_paired_differences.{extension}'
        fig.savefig(path, dpi=170, metadata={'Date': None} if extension == 'svg' else {})
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def numerical_controls(tasks, output):
    vision = tasks['aircraft']['initial_numerical_controls']
    lines = []
    full = vision.get('numerical_control_full')
    if full:
        result, audit = full['result'], full['audit']
        count = result['count']
        rows = {(row['method'], row['rank'], row['precision']): row for row in result['rows']}
        lines += [f'Aircraft: all {count:,} validation images, trained seed-42 baseline classifier, zero-update adapters. The FP32 control disables TF32; BF16 uses the training precision protocol. The shared trained classifier probes backbone sensitivity; it does not replay each adapter training run\'s initially random classifier. These are initialization diagnostics, not trained-adapter test results.', '',
                  '| Untrained backbone adapter | FP32 max logit drift from bare | BF16 max logit drift from bare | BF16 changed predictions | BF16 correct |',
                  '|:--|--:|--:|--:|--:|']
        for method in ('bare', *METHODS):
            rank = 0 if method == 'bare' else 2
            fp32, bf16 = rows[method, rank, 'fp32'], rows[method, rank, 'bf16']
            if method != 'bare':
                for precision in ('fp32', 'bf16'):
                    names = {f'{method}_r2_{precision}', f'{method}_r8_{precision}'}
                    pair = next(item for item in result['pairwise_adapter_comparisons'] if {item['first'], item['second']} == names)
                    assert pair['bitwise_equal'], 'Rank equality no longer holds; expand the numerical control table'
            lines.append(f'| {"Bare frozen backbone" if method == "bare" else LABELS[method]} | {fp32["max_absolute_logit_difference_from_bare"]:.6g} | {bf16["max_absolute_logit_difference_from_bare"]:.6g} | {bf16["argmax_disagreement_count_from_bare"]}/{count} | {bf16["probe_correct"]}/{count} |')
        fp32 = [row for row in result['rows'] if row['precision'] == 'fp32']
        pair = next(item for item in result['pairwise_adapter_comparisons'] if {item['first'], item['second']} == {'lora_r8_bf16', 'dora_r8_bf16'})
        lines += ['', f'Ranks 2 and 8 are bitwise equal within each method in this zero-update control. All FP32 adapter cases have {max(row["argmax_disagreement_count_from_bare"] for row in fp32)} changed predictions relative to bare (maximum logit drift {max(row["max_absolute_logit_difference_from_bare"] for row in fp32):.6g}). The bare BF16 result exactly matches the saved baseline validation record ({audit["full_validation_bare_bf16_correct"]}/{count} correct).', '',
                  f'LoRA and NoRA form one bitwise-equal BF16 family; the four DoRA-based methods form another. Between these families, {pair["argmax_disagreement_count"]}/{count} predictions change, with mean absolute logit drift {pair["mean_absolute_logit_difference"]:.6g} and maximum {pair["max_absolute_logit_difference"]:.6g}. The families differ by {abs(rows["lora", 2, "bf16"]["probe_correct"] - rows["dora", 2, "bf16"]["probe_correct"])} correct predictions overall. Disagreement counts measure prediction changes; they do not establish an accuracy loss of equal size or predict effects after training. Aircraft therefore does not have numerically identical BF16 initialization across all six methods. Tiny trained-method differences should be interpreted with this qualification.']
        mechanism = full.get('mechanism')
        if mechanism:
            compared = [row for row in mechanism['comparisons'] if row['reference'].startswith('lora_r8')]
            assert len(compared) == 2 and all(row['bitwise_equal'] for row in compared)
            bare = next(row for row in mechanism['comparisons'] if row['reference'] == 'bare_r0_bf16')
            lines += ['', f'A separate mechanism probe on the first {mechanism["count"]} validation images recalibrates a newly constructed rank-8 DoRA instance using GPU weight norms. Its maximum magnitude/norm ratio error drops from {mechanism["initial_ratio_error_max"]:.6g} to {mechanism["recalibrated_ratio_error_max"]:g}; its logits then match LoRA bitwise in both FP32 and BF16. This supports CPU/GPU norm reduction rounding as the source of the family difference in this probe. The recalibrated adapter retains the separate projection/bias numerical path: relative to bare BF16 it still changes {bare["argmax_disagreement_count"]}/{mechanism["count"]} predictions, with maximum logit drift {bare["max_absolute_logit_difference"]:.6g}. The probe does not modify any trained adapter, original control or reported task result.']
    original = vision.get('numerical_control')
    if original:
        result = original['result']
        lines += ['', f'The original {result["count"]}-image Aircraft control is also retained and independently re-scored. It covers the first four classes in manifest order, so its accuracy is not representative of the full validation split. Both controls retain all 26 logit arrays and 132 paired comparisons; the full control is the primary numerical diagnostic.', '']
    control = tasks['nfcorpus']['initial_numerical_control']
    lines += ['Retrieval initialization control:', '', '| Untrained model | NFCorpus nDCG@10 ×100 | SciFact nDCG@10 ×100 | Max probe embedding difference |', '|:--|--:|--:|--:|']
    for row in control.get('runs', []):
        lines.append(f'| {"Native frozen" if row["method"] == "frozen" else LABELS[row["method"]]} | {100*row["evaluation"]["nfcorpus"]["test"]["ndcg_at_10"]:.4f} | {100*row["evaluation"]["scifact"]["test"]["ndcg_at_10"]:.4f} | {row["max_absolute_probe_embedding_difference"]:.7f} |')
    lines += ['', 'Zero-update adapters split BF16 projection and bias addition while native Linear may fuse them. This changes rounding slightly. All six untrained adapter methods produced identical measured retrieval metrics; the control uses rank 2/seed 42 and 128 fixed document embeddings. Method-to-method comparisons share this numerical path. Native frozen baselines are retained visibly in the main tables.']
    (output / 'numerical_controls.md').write_text('\n'.join(lines) + '\n')


def capture_report_sources(output):
    root = Path(__file__).resolve().parents[2]
    target = output / 'report_source'
    names = ['experiments/__init__.py', 'experiments/second_round/__init__.py',
             'experiments/second_round/report.py', 'experiments/second_round/archive.py',
             'experiments/second_round/analyze_cogs.py', 'experiments/second_round/analyze_vision.py',
             'experiments/second_round/analyze_retrieval.py', 'experiments/second_round/validate_retrieval.py',
             'experiments/second_round/plot_teacher.py']
    names = sorted(set(names) | {str(path.relative_to(root)) for path in (root / 'experiments' / 'second_round').glob('*.py')})
    for name in names:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.chmod(0o644)
        destination.write_bytes((root / name).read_bytes())
    return target


def reproduce(output):
    text = '''The archive contains scored predictions, task/source manifests, training logs, local audit attestations, compact teacher evidence when available, and this report’s source. It excludes downstream model and adapter checkpoints. All included files have SHA-256 hashes. Publication uses deterministic 4 MiB parts named raw_artifacts.tar.gz.part-000 onward; raw_manifest.json and summary.json list their ordered names, byte counts and SHA-256 hashes, plus the complete archive hash. The final part can be smaller. Download every listed part, raw_manifest.json and report_source before rebuilding.

From this report directory, reassemble and check every part and the complete archive using only the Python standard library. The assembly command refuses missing, reordered, truncated or hash-mismatched parts and writes the archive atomically after successful verification. Then rebuild the downstream tables and PNG/SVG figures without a GPU, model downloads, or original absolute paths:

```bash
python report_source/experiments/second_round/archive.py --assemble-parts --report-dir .
mkdir raw
tar -xzf raw_artifacts.tar.gz -C raw
PYTHONPATH="$PWD/raw/report_source" CUDA_VISIBLE_DEVICES='' python -m experiments.second_round.report \\
  --raw-root "$PWD/raw" --output-dir "$PWD/rebuilt" --skip-bundle
```

The report verifies every archive-manifest hash before scoring and re-scores saved classification predictions, retrieval rankings and COGS text. It independently recomputes both Aircraft initialization controls and the separate GPU-norm recalibration probe from retained logit arrays. It also regenerates the matched-factor-LR magnitude diagnostic from validation-only trial files; that diagnostic does not use test scores or change the primary results. It validates available checkpoints, and explicitly counts omitted payloads. Full original checkpoint audits remain local audit attestations; omitted downstream weights cannot be revalidated from this compact archive. Rebuilding figures is not rerunning model inference or training.

To repeat the packaged offline reconstruction audit from this report directory:

```bash
PYTHONPATH="$PWD/raw/report_source" CUDA_VISIBLE_DEVICES='' python -m experiments.second_round.verify_report_rebuild \\
  --report-dir "$PWD" --work-parent /tmp --force-parts --audit-output "$PWD/offline_rebuild_audit_reproduced.json"
```

This reconstructs the archive exclusively from its published parts, ignoring any locally retained complete archive, then extracts a fresh read-only evidence tree, runs outside the repository with offline model-library settings, and compares generated figures, tables, diagnostics, sources and structured task scores. It verifies every input and part hash again afterward. The separate audit records part/complete-archive hashes, output/source hashes and environment; it stays outside that archive to avoid circular hashing. The verifier also uses parts automatically when the complete archive is absent.

Optional full task-level portable checks:

```bash
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.analyze_vision raw/vision/rank2_run --skip-checkpoints
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.analyze_vision raw/vision/rank8_run --skip-checkpoints
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.analyze_cogs --root raw/cogs --allow-missing-checkpoints
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.validate_retrieval raw/retrieval
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.validate_control_vision raw/vision/numerical_control --no-write
PYTHONPATH="$PWD/raw/report_source" python -m experiments.second_round.validate_control_vision raw/vision/numerical_control_full --no-write
```

Run the report before optional analysis scripts: those scripts can rewrite analysis files, changing archive-manifest hashes. The report also runs teacher_compact/verify.py: it regenerates all synthetic problem tensors exactly, verifies retained small adapter checkpoints, and independently reconstructs dense test predictions on CPU. Teacher scores and plots remain separate from downstream quality scores. The full report needs Python, NumPy, Matplotlib and CPU PyTorch because it runs the compact teacher verifier automatically. No GPU or model download is required; use the recorded package versions for exact reproduction. Original absolute paths in provenance describe original runs and do not locate files during this rebuild.
'''
    (output / 'reproduce.md').write_text(text + '\nCOGS diagnostic audit scope: ' + COGS_GENERATION_CAP_NOTE + '\n')



def require_final_audits(raw):
    verified = []
    for rank in (2, 8):
        root = raw / 'vision' / f'rank{rank}_run'
        analysis = read(root / 'analysis.json', {})
        assert analysis.get('audit', {}).get('status') == 'pass', f'Aircraft rank{rank} final audit missing'
        assert analysis['input_results_sha256'] == sha256(root / 'results.json'), 'Stale Aircraft audit'
        expected_paths = {str((record_directory(root, row) / 'test_predictions.json').relative_to(root))
                          for row in read(root / 'results.json')}
        prediction_hashes = analysis.get('input_predictions_sha256', {})
        assert len(expected_paths) == 21 and set(prediction_hashes) == expected_paths
        for name, digest in prediction_hashes.items():
            assert sha256(root / name) == digest, f'Stale Aircraft paired interval input: {name}'
        verified.append(f'aircraft rank{rank}')
    baseline_audit = read(raw / 'vision' / 'cross_shard_baseline_audit.json', {})
    assert baseline_audit.get('status') == 'pass', 'Cross-shard Aircraft baseline audit missing'
    pairs = baseline_audit.get('baseline_seed_pairs', [])
    assert len(pairs) == 3 and {row['seed'] for row in pairs} == {42, 43, 44}
    assert all(row['test_predictions_byte_identical'] and row['checkpoint_payloads_byte_identical'] for row in pairs)
    expected_paths = {f'rank{rank}_run/baseline/seed_{seed}/{name}' for rank in (2, 8)
                      for seed in (42, 43, 44) for name in ('result.json', 'test_predictions.json')}
    assert set(baseline_audit.get('input_sha256', {})) == expected_paths
    for name, digest in baseline_audit['input_sha256'].items():
        assert sha256(raw / 'vision' / name) == digest, f'Stale cross-shard Aircraft baseline audit: {name}'
    verified.append('aircraft cross-shard baselines')
    for name, count in (('numerical_control', 128), ('numerical_control_full', 3333)):
        root = raw / 'vision' / name
        audit = read(root / 'audit.json', {})
        assert audit.get('status') == 'pass' and audit.get('validation_images_checked') == count, f'Aircraft {name} audit missing'
        assert audit.get('input_result_sha256') == sha256(root / 'result.json'), f'Stale Aircraft {name} result audit'
        assert audit.get('input_logits_sha256') == sha256(root / 'logits.npz'), f'Stale Aircraft {name} logits audit'
        assert audit.get('cases_checked') == 26 and audit.get('pairwise_comparisons_checked') == 132
        if name == 'numerical_control_full':
            assert audit.get('full_validation_bare_bf16_matches_training_record'), 'Aircraft full-control baseline does not match training record'
            mechanism = audit.get('mechanism', {})
            assert mechanism.get('status') == 'pass' and mechanism.get('comparisons_checked') == 6
            assert mechanism.get('input_result_sha256') == sha256(root / 'mechanism.json'), 'Stale Aircraft mechanism result audit'
            assert mechanism.get('input_logits_sha256') == sha256(root / 'mechanism_logits.npz'), 'Stale Aircraft mechanism logits audit'
        verified.append(f'aircraft {name}')
    root = raw / 'retrieval'
    audit, comparison = read(root / 'audit.json', {}), read(root / 'comparisons.json', {})
    expected = sha256(root / 'results.json')
    assert audit.get('status') == 'pass' and audit.get('input_results_sha256') == expected, 'Missing or stale retrieval audit'
    assert comparison.get('input_results_sha256') == expected, 'Stale retrieval uncertainty'
    assert len(comparison.get('input_per_query_sha256', {})) == 72
    for name, digest in comparison['input_per_query_sha256'].items():
        assert sha256(root / name) == digest, f'Stale retrieval uncertainty input: {name}'
    verified.append('retrieval')
    root = raw / 'cogs'
    analysis = read(root / 'analysis.json', {})
    assert analysis.get('passed') and analysis.get('runs') == 18, 'COGS final audit missing'
    assert set(analysis.get('input_sha256', {})) == set(METHODS)
    for method, files in analysis['input_sha256'].items():
        assert set(files) == {'results.json', 'manifest.json', 'split_manifest.json', 'baseline.json'}
        for name, digest in files.items():
            assert sha256(root / method / name) == digest, f'Stale COGS audit: {method}/{name}'
    verified.append('cogs')
    teacher = raw / 'teacher_compact'
    assert read(teacher / 'portable_verification.json', {}).get('passed'), 'Compact teacher audit missing'
    verified.append('teacher compact')
    diagnostic = read(raw / 'matched_magnitude_validation.json', {})
    assert diagnostic.get('complete') and not diagnostic['pending'], 'Matched-magnitude validation diagnostic incomplete'
    assert len(diagnostic['comparisons']) == 132 and len(diagnostic['input_sha256']) == 77
    for name, digest in diagnostic['input_sha256'].items():
        assert sha256(raw / name) == digest, f'Stale matched-magnitude validation input: {name}'
    verified.append('matched magnitude validation only')
    return verified


def teacher_report(raw, output):
    root = raw / 'teacher_compact'
    if not (root / 'verify.py').exists():
        return {'available': False}
    target = output / 'teacher_recomputed.json'
    completed = subprocess.run([sys.executable, str(root / 'verify.py'), '--output', str(target)],
                               text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f'Teacher reconstruction failed: {completed.stderr}')
    verified = read(target)
    assert verified['passed'] and verified['final_run_count'] == 504
    from experiments.second_round.plot_teacher import make
    make(root, output)
    return {'available': True, 'passed': True, 'final_runs': verified['final_run_count'],
            'regenerated_tensor_count': verified['tensor_count'], 'independent_dense_weight_math': verified['independent_dense_weight_math'],
            'evidence_manifest_sha256': sha256(root / 'manifest.json'), 'verification_artifact': target.name,
            'scope': verified['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('/var/tmp/dora-bench/round2'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/2026-10-08-round2'))
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--skip-bundle', action='store_true')
    args = parser.parse_args()
    evidence = verify_extracted(args.raw_root)
    fresh_audits = require_final_audits(args.raw_root) if not args.allow_partial else []
    tasks = {'aircraft': aircraft(args.raw_root), **retrieval(args.raw_root), 'cogs': cogs(args.raw_root)}
    data_complete = all(task['complete'] for task in tasks.values())
    complete = data_complete and not args.allow_partial
    missing = [item for task in tasks.values() for item in task['missing']]
    if not data_complete and not args.allow_partial:
        parser.error('Incomplete task matrix: ' + '; '.join(missing))
    if not complete and args.output_dir == Path('results/2026-10-08-round2'):
        parser.error('Partial previews require a separate --output-dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    teacher = teacher_report(args.raw_root, args.output_dir)
    report_sources = capture_report_sources(args.output_dir)
    for name, task in tasks.items():
        write(args.output_dir / 'tasks' / f'{name}.json', task)
    for rank in (2, 8):
        table(tasks, rank, args.output_dir)
        plot(tasks, rank, args.output_dir, complete)
    secondary_tables(tasks, args.output_dir)
    cogs_category_plot(tasks["cogs"], args.output_dir, complete)
    matched_diagnostic = matched_magnitude_report(args.raw_root, args.output_dir, complete)
    resource_tables(tasks, args.output_dir)
    uncertainty_table(tasks, args.output_dir)
    difference_plot(tasks, args.output_dir, complete)
    numerical_controls(tasks, args.output_dir)
    reproduce(args.output_dir)
    archive = None if args.skip_bundle else bundle(args.raw_root, args.output_dir, report_sources)
    summary = {'complete': complete, 'data_complete': data_complete, 'preview': args.allow_partial, 'missing': missing, 'method_order': METHODS, 'method_labels': LABELS,
               'task_order': list(tasks), 'tasks': {name: {key: value for key, value in task.items() if key not in ('provenance', 'validation_tuning', 'initial_numerical_control')} for name, task in tasks.items()},
               'report_source_sha256': sha256(Path(__file__)), 'versions': {'python': sys.version, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
               'evidence_validation': evidence, 'fresh_passed_final_audits': fresh_audits, 'teacher_validation': teacher, 'matched_magnitude_diagnostic': matched_diagnostic, 'raw_archive': archive,
               'uncertainty': 'Sample SD across three training seeds; separate exploratory 10k paired bootstrap intervals. No aggregate task ranking.'}
    write(args.output_dir / 'summary.json', summary)
    print(json.dumps({'complete': complete, 'missing': missing, 'raw_archive': archive, 'output': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
