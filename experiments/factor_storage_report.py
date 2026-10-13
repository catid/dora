"""Separate report for BF16 factor storage on the two task-recipe endpoints."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

# Works both as a module in the checkout and as an archived standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import change_report as common

import matplotlib.pyplot as plt

METHODS = ['ab_original', 'storage_original_bf16', 'ab_revised', 'storage_revised_bf16']
LABELS = {'ab_original': 'Original task recipe / FP32 factors',
          'storage_original_bf16': 'Original task recipe / BF16 factors',
          'ab_revised': 'Revised task recipe / FP32 factors',
          'storage_revised_bf16': 'Revised task recipe / BF16 factors'}
PAIRS = [('storage_original_bf16', 'ab_original'), ('storage_revised_bf16', 'ab_revised')]
SCOPE = ('Separate exploratory SciFact rank8 supplement. It changes factor storage to BF16 after '
         'the exact same FP32 initialization draws. Magnitude storage and explicit norm-precision '
         'policies remain those of each task recipe. Factor gradients and AdamW moments inherit '
         'the factor dtype, so this treatment includes their reduced precision and the arithmetic '
         'implied by BF16 factors. It does not test native-BF16 random draws or reconstruct the '
         'full revised standalone initializer. FP32 controls are reused from the main change '
         'study; BF16 treatments use the same pinned model, splits, full validation LR grid, '
         'training budget and seeds 42–44. The fixed lane uses original-selected LR0.001; the '
         'tuned lane independently selects each treatment on validation nDCG. Repeated fit IDs '
         'identify one measurement shared between lanes. This is an additional exploratory '
         'analysis on an existing test set, with only three seeds and no multiple-comparison '
         'correction. Timing includes the common norm diagnostics.')

SCOPE += (' Official SciFact train/test provenance is retained: 726 training queries and an 81-query validation split after exact-text deduplication, 300 official test queries, 5,183 corpus documents, rank 8 and 115 updates per fit. Validation split seed 20261008; training seeds 42–44. This matches the prior local protocol, not the full DoRA paper benchmark suite.')


def compute(payload):
    data = payload['data']
    assert set(data['lanes']) == set(common.LANES)
    comparisons = []
    for lane, cells in data['lanes'].items():
        assert set(cells) == set(METHODS)
        for cell in cells.values():
            assert all(len(row) == len(data['example_ids']) for row in cell['scores'])
            cell['summary'] = common.summarize_cell(cell)
        for candidate, reference in PAIRS:
            result = common.contrast(cells[candidate]['scores'], cells[reference]['scores'])
            common.close(result['difference'], cells[candidate]['summary']['mean'] - cells[reference]['summary']['mean'])
            comparisons.append({'lane': lane, 'candidate': candidate, 'reference': reference, **result})
    payload['comparisons'] = comparisons


def render(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    common.matplotlib.rcParams['svg.hashsalt'] = 'dora-factor-storage-20261009'
    cells = payload['data']['lanes']
    rows = [[LABELS[method], *[common.fmt(cells[lane][method]['summary']) for lane in common.LANES]] for method in METHODS]
    (output / 'quality_table.md').write_text(common.markdown_table(['Recipe', 'Fixed LR / nDCG@10 ×100', 'Validation tuned / nDCG@10 ×100'], rows) + '\nMean ± sample SD over seeds 42–44. FP32 rows reuse the main study; BF16 rows are separate treatments.\n')
    effects = [[row['lane'], LABELS[row['candidate']], f"{row['difference']:+.3f}",
                f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]"] for row in payload['comparisons']]
    (output / 'paired_effects.md').write_text(common.markdown_table(['Lane', 'BF16 treatment minus matching FP32 control', 'nDCG×100 difference', 'Exploratory 95% interval'], effects))
    resources = []
    for lane in common.LANES:
        for method in METHODS:
            item = cells[lane][method]['summary']
            resources.append([lane, LABELS[method], ', '.join(f'{v:g}' for v in item['learning_rates']),
                              common.fmt(item['training_seconds']), f"{max(item['peak_memory_bytes']['values'])/2**30:.3f}"])
    (output / 'runtime_memory.md').write_text(common.markdown_table(['Lane', 'Recipe', 'LR by seed', 'Instrumented seconds, mean ± SD', 'Peak allocated GiB'], resources))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, lane in zip(axes, common.LANES):
        for index, method in enumerate(METHODS):
            item = cells[lane][method]['summary']
            axis.errorbar(index, item['mean'], yerr=item['sample_sd'], fmt='o', capsize=4,
                          color='#0072B2' if method in ('ab_original', 'ab_revised') else '#D55E00')
        axis.set_xticks(range(4), ['Original\nFP32', 'Original\nBF16', 'Revised task\nFP32', 'Revised task\nBF16'])
        axis.set_title('Fixed original LR' if lane == common.LANES[0] else 'Equal validation tuning')
        axis.set_ylabel('SciFact nDCG@10 ×100')
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Separate factor-storage supplement: same FP32 draws, then BF16 conversion')
    fig.text(.5, .015, 'Mean ± SD, 3 seeds. FP32 controls reused from main study. Not a complete standalone implementation comparison.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .92])
    fig.savefig(output / 'factor_storage_comparison.png', dpi=170)
    fig.savefig(output / 'factor_storage_comparison.svg', metadata={'Date': None})
    plt.close(fig)
    common.write(output / 'comparisons.json', payload['comparisons'])
    common.write(output / 'summary.json', payload)
    (output / 'scope.md').write_text(SCOPE + '\n')
    (output / 'reproduce.md').write_text('Verify raw_manifest.json and its archive/member hashes, then extract into an empty directory with Python tarfile filter="data". Run:\n\n```bash\npython source/experiments/factor_storage_report.py --rebuild report_inputs.json --output rebuilt\n```\n\nThe archived report verifies all retained input hashes, recomputes per-seed/query statistics and paired intervals, and regenerates outputs. NumPy and Matplotlib are required; no inference or omitted checkpoint replay is performed.\n')


def collect(raw, repo):
    evidence = common.Evidence(raw, repo)
    core = evidence.json(Path('/var/tmp/dora-bench/change_ablation_core_validation.json'))
    gate = evidence.json(Path('/var/tmp/dora-bench/change_factor_storage_validation.json'))
    assert gate['status'] == gate['cuda_tests']['status'] == 'pass'
    assert gate['supplement_source_sha256'] == common.sha(repo / 'experiments/change_factor_storage.py')
    for treatment_name, core_name in PAIRS:
        assert gate['recipes'][treatment_name]['base_recipe'] == core['recipes'][core_name]
    common.bound_references(evidence, core)
    common.bound_references(evidence, gate)
    review = Path('/var/tmp/dora-bench/change_factor_storage_independent_review.json')
    if review.exists():
        assessment = evidence.json(review)
        assert assessment['status'] == 'pass'
        for name, expected in assessment['source_sha256'].items():
            assert common.sha(repo / name) == expected, name
            evidence.add(repo / name)
    control = common.retrieval(evidence, core['recipes'], 'scifact')['scifact']
    recipes = {name: gate['recipes'][name] for name in METHODS[1::2]}
    treatment = common.retrieval(evidence, recipes, 'scifact', root_name='scifact_factor_storage')['scifact']
    assert control['example_ids'] == treatment['example_ids']
    for name in ('model', 'datasets', 'input_sha256'):
        assert control['manifest'][name] == treatment['manifest'][name], name
    config = control['manifest']['configuration']
    for name, value in config.items():
        if name != 'methods': assert treatment['manifest']['configuration'][name] == value, name
    data = copy.deepcopy(treatment)
    for lane in common.LANES:
        for method in ('ab_original', 'ab_revised'):
            data['lanes'][lane][method] = control['lanes'][lane][method]
    data['control_manifest'] = control['manifest']
    data['control_audit'] = control['audit']
    for path in (raw / 'scifact_factor_storage').rglob('*'):
        if path.is_file() and path.suffix in common.SUFFIXES and '__pycache__' not in path.parts: evidence.add(path)
    common.command_evidence(evidence, 'supplement_commands.json', '/var/tmp/dora-bench/run_retrieval_supplements.py')
    for name in ('experiments/change_report.py', 'experiments/factor_storage_report.py',
                 'experiments/change_factor_storage.py', 'experiments/test_change_factor_storage.py'):
        evidence.add(repo / name)
    return {'complete': True, 'kind': 'separate_factor_storage_supplement', 'data': data, 'scope': SCOPE,
            'core_gate': core, 'supplement_gate': gate,
            'source_sha256': common.sha(Path(__file__)), 'dependency_sha256': common.sha(Path(common.__file__)),
            'versions': {'numpy': common.np.__version__, 'matplotlib': common.matplotlib.__version__}}, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('/var/tmp/dora-bench/change_ablation_runs'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('results/2026-10-09-factor-storage'))
    parser.add_argument('--rebuild', type=Path)
    args = parser.parse_args()
    if args.rebuild:
        payload = common.read(args.rebuild)
        assert common.sha(Path(__file__)) == payload['source_sha256']
        assert common.sha(Path(common.__file__)) == payload['dependency_sha256']
        for row in payload['evidence']:
            path = args.rebuild.parent / row['path']
            assert path.stat().st_size == row['bytes'] and common.sha(path) == row['sha256'], path
        before = payload['comparisons']
        compute(payload)
        assert before == payload['comparisons']
        render(payload, args.output)
        common.write(args.output / 'rebuild_audit.json', {'status': 'pass', 'evidence_files_verified': len(payload['evidence']),
                     'scope': 'Retained evidence hashes verified; per-seed/query summaries and paired intervals recomputed; report regenerated without inference.'})
    else:
        try: payload, evidence = collect(args.raw_root.resolve(), args.repo.resolve())
        except common.NotReady as error: parser.exit(2, str(error) + '\n')
        compute(payload)
        args.output.mkdir(parents=True, exist_ok=True)
        common.bundle(payload, evidence, args.output)
        render(payload, args.output)
    print(common.json.dumps({'status': 'complete', 'supplement': 'factor_storage', 'output': str(args.output)}))


if __name__ == '__main__': main()
