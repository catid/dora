"""Verify saved audit evidence and plot held-out toy MSE without rerunning training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = (
    ('original_dim0_fullgrad', 'Original: dim 0, full gradient', '#0072B2'),
    ('row_dim1_fullgrad', 'Row norm: full gradient', '#E69F00'),
    ('row_dim1_detached_dense', 'Row norm: detached, dense', '#009E73'),
    ('current_factorized', 'Current: detached, factorized', '#CC79A7'),
    ('lora', 'LoRA', '#D55E00'),
    ('original_magnitude_only', 'Original: magnitudes only', '#64748B'),
    ('continue_full_training', 'Continue full Linear training', '#94A3B8'),
    ('pretrained_reference', 'Pretrained Linear, no extra updates', '#CBD5E1'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate(root):
    manifest = root / 'manifest.json'
    checked = 0
    if manifest.exists():
        for row in read(manifest)['files']:
            path = root / row['path']
            assert path.is_file() and path.stat().st_size == row['bytes'], path
            assert sha(path) == row['sha256'], path
            checked += 1
    toy = read(root / 'toy_comparison.json')
    expected_sources = {'toy_compare.py': root / 'toy_compare.py',
                        'original_dora.py': root / 'original_dora.py',
                        'dora.py': root / 'gpu_source/dora.py'}
    for name, digest in toy['source_sha256'].items():
        assert sha(expected_sources[Path(name).name]) == digest
    seeds = toy['seeds']
    assert seeds == list(range(5)) and len(toy['runs']) == 35
    assert {(row['method'], row['seed']) for row in toy['runs']} == {
        (method, seed) for method, _, _ in METHODS[:-1] for seed in seeds}
    summary = {}
    for method, _, _ in METHODS[:-1]:
        rows = sorted((row for row in toy['runs'] if row['method'] == method), key=lambda row: row['seed'])
        values = [row['heldout_mse_after'] for row in rows]
        assert all(math.isfinite(value) and value >= 0 for value in values)
        mean, sd = statistics.mean(values), statistics.stdev(values)
        recorded = toy['summary'][method]
        assert values == recorded['values'] and mean == recorded['mean'] and sd == recorded['sample_sd']
        assert all(row['trainable_parameters'] == recorded['trainable_parameters'] for row in rows)
        summary[method] = {'mean': mean, 'sample_sd': sd, 'values': values,
                           'trainable_parameters': recorded['trainable_parameters']}
    baseline = []
    for seed in seeds:
        values = {row['heldout_mse_before'] for row in toy['runs'] if row['seed'] == seed}
        assert len(values) == 1
        baseline.append(values.pop())
    assert baseline == toy['baseline_heldout_mse']['values']
    assert statistics.mean(baseline) == toy['baseline_heldout_mse']['mean']
    summary['pretrained_reference'] = {'mean': statistics.mean(baseline), 'sample_sd': statistics.stdev(baseline),
                                       'values': baseline, 'trainable_parameters': 0}
    gpu = read(root / 'gpu_latency.json')
    gpu_audit = read(root / 'gpu_latency_validation.json')
    assert gpu['complete'] and gpu_audit['status'] == 'pass'
    assert sha(root / 'gpu_latency.json') == gpu_audit['input_results_sha256']
    assert sha(root / 'gpu_latency.py') == gpu['benchmark_script_sha256'] == gpu_audit['script_sha256']
    for name, digest in gpu['source_sha256'].items():
        assert sha(root / 'gpu_source' / name) == digest
    assert (root / 'original_dora.py').read_bytes() == (root / 'gpu_source/dora_original_bb97617.py').read_bytes()
    assert (root / 'original_dora.py').read_bytes().replace(b'dim=0', b'dim=1') == (root / 'gpu_source/dora_axis_only_dim1.py').read_bytes()
    count = 0
    for case in gpu['cases']:
        assert case['factorized_detached_dense_parity_passed']
        for row in case['measurements'].values():
            assert row['finite_after_timing']
            for mode in ('forward', 'forward_backward'):
                measured = row[mode]
                assert len(measured['sample_ms_per_call']) == 7
                assert all(math.isfinite(value) and value > 0 for value in measured['sample_ms_per_call'])
                assert statistics.median(measured['sample_ms_per_call']) == measured['median_ms_per_call']
                count += 7
    assert count == gpu_audit['raw_samples_recomputed'] == 168
    peft = read(root / 'peft_parity.json')
    assert peft['passed'] and peft['device'] == 'cpu' and peft['dtype'] == 'float64'
    assert sha(root / 'peft_parity.py') == peft['source_sha256']['audit_script']
    assert sha(root / 'gpu_source/dora.py') == peft['source_sha256']['standalone']
    assert all(math.isfinite(row['maximum_absolute_error']) and row['maximum_absolute_error'] <= 2e-15
               for row in peft['errors'].values())
    return {'toy': toy, 'summary': summary, 'manifest_files_verified': checked,
            'gpu_samples_recomputed': count, 'toy_runs_recomputed': 35,
            'external_source_scope': 'Third-party source hashes are retained as provenance; omitted source files and model training are not rechecked.'}


def plot(root, output, verified):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'savefig.facecolor': 'white'})
    summary = verified['summary']
    fig, ax = plt.subplots(figsize=(14, 8.6))
    fig.subplots_adjust(left=.285, right=.745, top=.80, bottom=.24)
    positions = [0, 1, 2, 3, 4, 5.6, 6.6, 7.6]
    labels = []
    offsets = [-.14, -.07, 0, .07, .14]
    for y, (method, label, color) in zip(positions, METHODS):
        row = summary[method]
        labels.append(label)
        ax.barh(y, row['mean'], height=.64, color=color, alpha=.90, zorder=2)
        ax.errorbar(row['mean'], y, xerr=row['sample_sd'], fmt='none', ecolor='#0F172A',
                    elinewidth=1.5, capsize=4, zorder=3)
        for value, offset in zip(row['values'], offsets):
            ax.scatter(value, y + offset, s=31, facecolor='white', edgecolor='#0F172A', linewidth=.9, zorder=4)
        ax.text(1.035, y, f"{row['mean']:.4f} ± {row['sample_sd']:.4f}   ({row['trainable_parameters']})",
                transform=ax.get_yaxis_transform(), va='center', fontsize=10.5, color='#0F172A')
    ax.set_yticks(positions, labels)
    ax.set_ylim(8.2, -.75)
    maximum = max(max(max(row['values']), row['mean'] + row['sample_sd']) for row in summary.values())
    ax.set_xlim(0, maximum * 1.07)
    ax.set_xlabel('Held-out mean squared error · lower is better', labelpad=12)
    ax.grid(axis='x', color='#E2E8F0')
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0, pad=10)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.axhline(4.8, color='#CBD5E1', linewidth=1)
    ax.text(1.035, 1.025, 'Mean ± seed SD   (trainable parameters)', transform=ax.transAxes,
            fontsize=10.5, color='#475569')
    fig.suptitle('Original implementation and ablations on the repository toy task',
                 fontsize=20, weight='bold', y=.963)
    fig.text(.5, .906, 'Five matched seeds · 10 → 1 sum regression · independent 10,000-example evaluation',
             ha='center', color='#475569', fontsize=12)
    legend = [Line2D([0], [0], color='#0F172A', linewidth=1.5, marker='|', markersize=10, label='Mean ± sample SD'),
              Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor='#0F172A', label='Individual seed')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .884), frameon=False, ncol=2)
    fig.text(.07, .145, 'Shared 100-epoch pretraining, then 5 epochs / 80 AdamW updates. Fixed LR 0.001; weight decay 0.01, including magnitudes.', fontsize=10.5, color='#334155')
    fig.text(.07, .11, 'Original uses input-column magnitudes; row methods use output-row magnitudes. Detaching the norm changes gradients.', fontsize=10.5, color='#334155')
    fig.text(.07, .075, 'Same-task continuation with one output: rank 4 imposes no meaningful low-rank capacity constraint. Optimizers are reset.', fontsize=10.5, color='#475569')
    fig.text(.07, .04, 'This synthetic toy does not establish downstream task superiority. The frozen-factor control matches all five recorded original MSEs.', fontsize=10.5, color='#475569')
    fig.savefig(output / 'toy_comparison.png', dpi=170, metadata={'Software': 'Matplotlib'})
    plt.close(fig)
    result = {'input_toy_sha256': sha(root / 'toy_comparison.json'), 'seeds': verified['toy']['seeds'],
              'methods': {method: {'label': label, **summary[method]} for method, label, _ in METHODS},
              'toy_runs_recomputed': verified['toy_runs_recomputed'],
              'scope': 'Recomputed means and sample SD from the retained per-seed runs; no new fitting or inference.'}
    (output / 'toy_chart_summary.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verified = validate(args.input_dir)
    output = args.output_dir or args.input_dir
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=True)
        plot(args.input_dir, output, verified)
    print(json.dumps({'passed': True, 'manifest_files_verified': verified['manifest_files_verified'],
                      'toy_runs_recomputed': verified['toy_runs_recomputed'],
                      'gpu_samples_recomputed': verified['gpu_samples_recomputed'],
                      'output': str(output), 'scope': verified['external_source_scope']}, indent=2))


if __name__ == '__main__':
    main()
