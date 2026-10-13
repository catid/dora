"""Strict, portable reporting for the exploratory six-change DoRA study.

The five evaluation columns cover four training tasks. Collection requires both
learning-rate lanes, all twelve recipes, all three seeds, and task audits.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [42, 43, 44]
LANES = ['fixed_original_lr', 'validation_tuned']
METHODS = ['ab_original', 'ab_axis', 'ab_detach', 'ab_m_fp32', 'ab_norm_fp32',
           'ab_init', 'ab_factorized', 'cum_axis_detach', 'cum_axis_detach_m_fp32',
           'cum_axis_detach_m_fp32_norm_fp32', 'cum_axis_detach_m_fp32_norm_fp32_init', 'ab_revised']
LABELS = {'ab_original': 'Original', 'ab_axis': 'Axis only', 'ab_detach': 'Detach only',
          'ab_m_fp32': 'Magnitude FP32 only', 'ab_norm_fp32': 'Norm FP32 only',
          'ab_init': 'Initializer only', 'ab_factorized': 'Factorization only',
          'cum_axis_detach': 'Axis + detach', 'cum_axis_detach_m_fp32': '+ magnitude FP32',
          'cum_axis_detach_m_fp32_norm_fp32': '+ norm FP32',
          'cum_axis_detach_m_fp32_norm_fp32_init': '+ initializer', 'ab_revised': '+ factorization (revised task recipe)'}
OAT = METHODS[1:7]
CHAIN = ['ab_original', 'ab_axis', *METHODS[7:]]
CHANGES = ['Axis', 'Detach norm', 'Magnitude FP32', 'Norm FP32', 'Harness initializer', 'Factorization']
TASKS = {'scifact': ('SciFact direct r8', 'nDCG@10 ×100'),
         'nfcorpus': ('NFCorpus r8', 'nDCG@10 ×100'),
         'scifact_transfer': ('SciFact transfer r8', 'nDCG@10 ×100'),
         'aircraft': ('Aircraft r2', 'Macro accuracy %'),
         'extraction': ('ViGGO r8', 'JSON exact match %')}
SUFFIXES = {'.json', '.jsonl', '.py', '.txt', '.md', '.csv', '.log', '.png', '.jpg', '.npz'}
SCOPE = (
    'Exploratory attribution on tasks selected after inspecting original-versus-revised outcomes, '
    'not a fresh confirmatory benchmark. Existing held-out sets are reused; every learning-rate '
    'and checkpoint choice remains validation-only. Each lane includes seeds 42, 43 and 44. '
    'The fixed lane applies the original-selected rate to every recipe; the tuned lane gives '
    'every recipe the same validation-search budget. Fits shared between lanes or proven identity '
    'controls are identified explicitly.\n\nSingle-change effects compare each intervention to the '
    'original. Cumulative increments compare adjacent recipes along the declared path. These '
    'contrasts are distinct estimands and must not be added across paths. Axis changes alter '
    'magnitude layout and parameter count. The initializer intervention changes orientation, '
    'distribution and CPU-versus-device RNG policy as one task-harness recipe. '
    'Precision changes can alter the zero-update forward function; shared pretrained weights '
    'do not imply bitwise identical initial predictions across all recipes. First-batch losses '
    'are reported separately as numerical initialization diagnostics.\n\nThe revised '
    'task-recipe endpoint has a separate tiny-norm engineering floor; accepted quality runs must report zero '
    'floor activations. Timings include norm-diagnostic reductions and describe instrumented '
    'recipe runs, not optimized production performance. Exploratory 95% intervals use paired '
    'seed/example resampling with only three training seeds and no multiplicity correction. '
    'SciFact transfer evaluates the NFCorpus fit and is not a fifth training task.\n\n'
    'The six interventions connect the measured experimental task adapters; they do not exhaust '
    'all numeric changes to standalone dora.py. In particular, these task recipes retain FP32 '
    'factors at both endpoints, while the revised standalone class draws/stores factors in the '
    'base dtype on its device. The Kaiming initializer is specific to the task harness.'
)


class NotReady(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20): digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if not path.is_file(): raise NotReady(f'Missing {path}')
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def close(a, b, atol=2e-6):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=0), (a, b)


def stats(values):
    assert values and all(math.isfinite(x) for x in values)
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
            'values': values, 'n': len(values)}


class Evidence:
    def __init__(self, raw, repo):
        self.raw, self.repo, self.files = raw.resolve(), repo.resolve(), {}

    def add(self, path):
        path = Path(path).resolve()
        if path.suffix not in SUFFIXES: return
        assert path.is_file(), path
        if path.is_relative_to(self.raw): name = 'runs/' + str(path.relative_to(self.raw))
        elif path.is_relative_to(self.repo): name = 'source/' + str(path.relative_to(self.repo))
        elif path.is_relative_to(Path('/var/tmp/dora-bench')): name = 'controls/' + str(path.relative_to('/var/tmp/dora-bench'))
        elif path.is_relative_to(Path('/home/catid/dora')): name = 'prior/' + str(path.relative_to('/home/catid/dora'))
        else: raise ValueError(f'Unmapped evidence {path}')
        assert name not in self.files or self.files[name] == path
        self.files[name] = path
        return name

    def json(self, path):
        result = read(path)
        self.add(path)
        return result

    def jsonl(self, path):
        self.add(path)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit_inputs(evidence, audit, root):
    assert audit.get('status') == 'pass' or audit.get('passed') is True, root
    for name, value in audit.get('input_sha256', {}).items():
        mapping = value if isinstance(value, dict) else {name: value}
        base = Path(audit.get('artifact_roots', {}).get(name, root)) if isinstance(value, dict) else root
        for relative, expected in mapping.items():
            path = Path(relative) if Path(relative).is_absolute() else base / relative
            if path.suffix in {'.pt', '.bin', '.safetensors'}: continue
            assert sha(path) == expected, path
            evidence.add(path)


def require_audit(evidence, root):
    path = next((root / name for name in ('analysis.json', 'task_audit.json', 'audit.json') if (root / name).exists()), None)
    if path is None: raise NotReady(f'Missing independent audit under {root}')
    audit = evidence.json(path)
    audit_inputs(evidence, audit, root)
    return {'path': evidence.add(path), 'sha256': sha(path), 'details': audit}


def metric(scores, strata=None):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 2 and scores.shape[0] == 3 and np.isfinite(scores).all()
    if strata is None: return scores.mean(axis=1)
    labels = np.asarray(strata)
    return np.stack([scores[:, labels == label].mean(axis=1) for label in np.unique(labels)]).mean(axis=0)


def summarize_cell(cell, strata=None):
    records = sorted(cell['records'], key=lambda row: row['seed'])
    assert [row['seed'] for row in records] == SEEDS
    values = (metric(cell['scores'], strata) * 100).tolist()
    for row, value in zip(records, values): close(row['value'], value, 2e-5)
    result = stats(values)
    for name in ('training_seconds', 'peak_memory_bytes'):
        observed = [row[name] for row in records]
        result[name] = stats(observed)
    result['parameters'] = sorted({row['trainable_parameters'] for row in records})
    result['learning_rates'] = [row['learning_rate'] for row in records]
    result['fit_ids'] = [row['fit_id'] for row in records]
    return result


def contrast(candidate, reference, strata=None, replicates=10000):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    labels = np.zeros(delta.shape[1], dtype=int) if strata is None else np.asarray(strata)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    generator = np.random.default_rng(20261009)
    boot = np.zeros(replicates)
    for start in range(0, replicates, 100):
        n = min(100, replicates - start)
        seeds = generator.integers(3, size=(n, 3))
        samples = np.zeros(n)
        for indices in groups:
            examples = generator.choice(indices, size=(n, len(indices)))
            samples += delta[seeds[:, :, None], examples[:, None, :]].mean(axis=(1, 2)) / len(groups)
        boot[start:start + n] = samples
    values = metric(delta, strata) * 100
    return {'difference': float(values.mean()), 'paired_seed_differences': values.tolist(),
            'ci95': (np.quantile(boot, [.025, .975]) * 100).tolist(),
            'replicates': replicates, 'random_seed': 20261009,
            'resampling': 'Crossed paired seeds and examples; preserve class counts and equal class weights for Aircraft.'}


def compute(payload):
    assert set(payload['tasks']) == set(TASKS)
    comparisons, tuning_effects = [], []
    for task, data in payload['tasks'].items():
        assert set(data['lanes']) == set(LANES)
        strata = data.get('strata')
        for lane in LANES:
            cells = data['lanes'][lane]
            assert set(cells) == set(METHODS), (task, lane, set(cells))
            for method, cell in cells.items():
                assert len(cell['scores']) == 3 and all(len(row) == len(data['example_ids']) for row in cell['scores'])
                cell['summary'] = summarize_cell(cell, strata)
            cache = {}
            for kind, pairs in [('single_change', [(method, 'ab_original') for method in OAT]),
                                ('cumulative_increment', list(zip(CHAIN[1:], CHAIN[:-1])))]:
                for index, (candidate, reference) in enumerate(pairs):
                    pair = candidate, reference
                    if pair not in cache: cache[pair] = contrast(cells[candidate]['scores'], cells[reference]['scores'], strata)
                    result = cache[pair]
                    close(result['difference'], cells[candidate]['summary']['mean'] - cells[reference]['summary']['mean'], 2e-5)
                    comparisons.append({'task': task, 'lane': lane, 'kind': kind, 'change': CHANGES[index],
                                        'candidate': candidate, 'reference': reference, **result})
        for method in METHODS:
            fixed = data['lanes']['fixed_original_lr'][method]
            tuned = data['lanes']['validation_tuned'][method]
            result = contrast(tuned['scores'], fixed['scores'], strata)
            close(result['difference'], tuned['summary']['mean'] - fixed['summary']['mean'], 2e-5)
            tuning_effects.append({'task': task, 'method': method, 'candidate_lane': 'validation_tuned',
                                   'reference_lane': 'fixed_original_lr',
                                   'same_fits': tuned['summary']['fit_ids'] == fixed['summary']['fit_ids'], **result})
    payload['comparisons'] = comparisons
    payload['tuning_effects'] = tuning_effects


def fmt(item, places=2):
    return f"{item['mean']:.{places}f} ± {item['sample_sd']:.{places}f}"


def markdown_table(header, rows):
    return '| ' + ' | '.join(header) + ' |\n|:--|' + '--:|' * (len(header) - 1) + '\n' + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)


def protocol_table(payload, output):
    tasks = payload['tasks']
    scifact = tasks['scifact']['manifest']
    nfcorpus = tasks['nfcorpus']['manifest']
    vision = tasks['aircraft']['manifest']
    language = tasks['extraction']['manifest']
    sizes = [scifact['datasets']['scifact']['query_counts'],
             nfcorpus['datasets']['nfcorpus']['effective_query_counts'],
             {'train': 'NFCorpus fit', 'validation': 'NFCorpus only', 'test': len(tasks['scifact_transfer']['example_ids'])},
             tasks['aircraft']['split_manifest']['effective_sizes'], language['dataset']['sizes']]
    rules = ['Official SciFact train/test; 81-query validation split from deduplicated train (split seed 20261008).',
             'Official NFCorpus train/dev/test with exact-text train/validation deduplication.',
             'Official SciFact test evaluated after NFCorpus training; no SciFact tuning.',
             'Official Aircraft variants train/val/test with recorded exact-image deduplication.',
             'Deterministic subsets of official ViGGO train/validation/test after utterance deduplication (subset seed 2026).']
    budgets = ['5 epochs / 115 updates; batch32', '5 epochs / 405 updates; batch32', 'No extra fit',
               '20 epochs; batch128', '256 updates; batch8']
    searches = ['2 full-budget seed42 rates', '4 full-budget seed42 rates', 'Reuse NFCorpus selection',
                '4 seed42 pilots ×10 epochs', '2 seed42 pilots ×48 updates']
    rows = []
    for task, counts, rule, budget, search in zip(TASKS, sizes, rules, budgets, searches):
        rows.append([TASKS[task][0], counts['train'], counts.get('validation', counts.get('val')), counts['test'], budget, search, rule])
    (output / 'protocols.md').write_text(markdown_table(
        ['Evaluation', 'Train examples/queries', 'Validation', 'Test', 'Final budget per fit', 'Search per recipe', 'Split protocol'], rows)
        + '\nEvery adapted recipe uses seeds 42–44. Model/dataset revisions, hashes, LR grids, exact split IDs and scoring definitions are retained in summary.json and the raw archive. This reproduces the earlier local task protocols, not a claim to reproduce every original DoRA paper benchmark.\n')


def render(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['svg.hashsalt'] = 'dora-change-attribution-20261009'
    protocol_table(payload, output)
    for lane in LANES:
        header = ['Recipe', *[f'{title} / {unit}' for title, unit in TASKS.values()]]
        rows = [[LABELS[method], *[fmt(payload['tasks'][task]['lanes'][lane][method]['summary']) for task in TASKS]] for method in METHODS]
        (output / f'{lane}_table.md').write_text(markdown_table(header, rows) + '\nMean ± sample SD over seeds 42–44. Each column uses its own metric; no aggregate score.\n')
        with (output / f'{lane}_table.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
    for kind in ('single_change', 'cumulative_increment'):
        rows = [[TASKS[r['task']][0], r['lane'], r['change'], LABELS[r['candidate']], LABELS[r['reference']],
                 f"{r['difference']:+.3f}", f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"]
                for r in payload['comparisons'] if r['kind'] == kind]
        (output / f'{kind}_effects.md').write_text(markdown_table(['Task', 'Lane', 'Change', 'Candidate', 'Reference', 'Difference', 'Exploratory 95% interval'], rows) + '\nAccuracy differences are percentage points; retrieval differences are nDCG×100 points.\n')
        fig, axes = plt.subplots(1, 5, figsize=(19, 5.5), sharey=True)
        for ax, task in zip(axes, TASKS):
            for lane_index, lane in enumerate(LANES):
                records = [r for r in payload['comparisons'] if r['kind'] == kind and r['task'] == task and r['lane'] == lane]
                for index, row in enumerate(records):
                    position = index + (lane_index - .5) * .22
                    ax.plot(row['ci95'], [position] * 2, color=('#0072B2', '#D55E00')[lane_index], linewidth=1.3)
                    ax.plot(row['difference'], position, 'o', color=('#0072B2', '#D55E00')[lane_index], markersize=5)
            ax.axvline(0, color='grey', linewidth=1, linestyle=':')
            ax.set_title(TASKS[task][0], fontsize=11)
            ax.grid(axis='x', alpha=.2)
            ax.set_xlabel('Candidate − reference (points)')
            ax.set_yticks(range(6), CHANGES)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].invert_yaxis()
        fig.suptitle('Single-change effects versus original' if kind == 'single_change' else 'Successive increments along the cumulative path')
        fig.text(.5, .02, 'Blue: fixed original LR. Orange: equally tuned LR. Paired exploratory 95% intervals, 3 seeds; no multiple-comparison correction.', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .07, 1, .92])
        fig.savefig(output / f'{kind}_effects.png', dpi=160)
        fig.savefig(output / f'{kind}_effects.svg', metadata={'Date': None})
        plt.close(fig)
    tuning_rows = [[TASKS[row['task']][0], LABELS[row['method']], f"{row['difference']:+.3f}",
                    f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]", 'yes' if row['same_fits'] else 'no']
                   for row in payload['tuning_effects']]
    (output / 'learning_rate_effects.md').write_text(markdown_table(
        ['Task', 'Recipe', 'Tuned − fixed original LR', 'Exploratory 95% interval', 'Same fits reused'], tuning_rows)
        + '\nThis contrasts equally budgeted validation tuning with the prespecified original-selected rate. A zero from the same fit is a shared measurement, not an independently trained replicate.\n')
    resources = []
    for task, data in payload['tasks'].items():
        if task == 'scifact_transfer': continue
        for lane, cells in data['lanes'].items():
            for method in METHODS:
                item = cells[method]['summary']
                resources.append([TASKS[task][0], lane, LABELS[method], ', '.join(map(str, item['parameters'])),
                                  ', '.join(f'{x:g}' for x in item['learning_rates']), fmt(item['training_seconds']),
                                  f"{max(item['peak_memory_bytes']['values']) / 2**30:.3f}"])
    (output / 'runtime_memory.md').write_text(markdown_table(['Task', 'Lane', 'Recipe', 'Parameters', 'LR by seed', 'Instrumented seconds, mean ± SD', 'Peak allocated GiB'], resources) + '\nThe same fit can serve both lanes or an explicitly proven identity control; these rows are not summed compute costs. SciFact transfer adds no training fit. Timers retain task-specific scope.\n')
    initialization = [[LABELS[method], *[f"{payload['tasks'][task]['initialization'][method]['first_batch_loss']:.8f}"
                                        for task in ('scifact', 'nfcorpus', 'extraction')]] for method in METHODS]
    (output / 'initialization_diagnostics.md').write_text(markdown_table(
        ['Recipe', 'SciFact seed42 initial loss', 'NFCorpus seed42 initial loss', 'ViGGO seed42 initial loss'], initialization)
        + '\nLoss from the first real training batch before the first optimizer update. All LR candidates within each recipe are checked for identical initial loss. These task-specific losses are numerical initialization diagnostics, not held-out quality or directly comparable across tasks. Aircraft separately records the full-model startup identity error.\n')
    aliases = payload['tasks']['aircraft']['manifest']['aliases']
    identity_rows = [[LABELS[name], LABELS[item['source_method']], item['task_control_case'], item['scope']]
                     for name, item in aliases.items()]
    (output / 'identity_reuse.md').write_text(markdown_table(
        ['Aircraft recipe', 'Measured source recipe', 'Full-model control', 'Scope'], identity_rows)
        + '\nThese FP32 magnitude-storage interventions are exact no-ops in this workload. Their table entries and zero differences reuse proven identical source fits; they are not separately measured training replicates. Other identical-LR fits are also shared between fixed and tuned lanes. Each record retains its canonical fit_id in summary.json.\n')
    endpoint_rows = []
    for task, references in payload['endpoint_reference'].items():
        for method, old in references.items():
            cell = payload['tasks'][task]['lanes']['validation_tuned'][method]
            current = cell['summary']
            endpoint_rows.append([TASKS[task][0], LABELS[method], fmt(old['summary']), fmt(current),
                                  f"{current['mean'] - old['summary']['mean']:+.5f}",
                                  ', '.join(f"{row['learning_rate']:g}" for row in old['records']),
                                  ', '.join(f'{rate:g}' for rate in current['learning_rates'])])
    (output / 'endpoint_reproduction.md').write_text(markdown_table(
        ['Task', 'Endpoint', 'Prior mean ± SD', 'New tuned mean ± SD', 'New − prior', 'Prior LRs', 'New LRs'], endpoint_rows)
        + '\nPrior runs are reproducibility controls, not extra independent seeds. The new tuned endpoints use the same validation-search budget. Full per-seed scores, selected rates and fit identifiers are retained in summary.json. Numeric initialization and timing scopes are documented separately.\n')
    engineering = [[row['change'], row['kind'], row['original'], row['revised'], row.get('scope', 'CPU behavior check; no task-quality claim.')]
                   for row in payload['engineering']['checks']]
    (output / 'engineering_checks.md').write_text(markdown_table(
        ['Change', 'Kind', 'Archived original behavior', 'Revised standalone behavior', 'Scope'], engineering)
        + '\nThese are separate CPU correctness, dtype and API checks against source snapshots. They do not supply downstream-quality rankings or imply that an inactive edge case explains a task result.\n')
    write(output / 'comparisons.json', payload['comparisons'])
    write(output / 'summary.json', payload)
    (output / 'scope.md').write_text(SCOPE + '\n')
    (output / 'reproduce.md').write_text('Verify raw_manifest.json, concatenate only its archive_parts in listed order when split, and extract with Python tarfile filter="data" into an empty directory. Then run:\n\n```bash\npython source/experiments/change_report.py --rebuild report_inputs.json --output rebuilt\n```\n\nRebuild verifies all retained evidence hashes, derives seed statistics and crossed paired intervals from individual example scores, and recreates tables/figures without model inference. Tensor checkpoints and external model/dataset caches are omitted; task audits retain their local verification. NumPy and Matplotlib are required.\n')


def bundle(payload, evidence, output, limit=40_000_000):
    payload['evidence'] = [{'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size} for name, path in sorted(evidence.files.items())]
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    archive = output / 'raw_artifacts.tar.gz'
    with archive.open('wb') as stream, gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for item in payload['evidence']:
            path = evidence.files[item['path']]
            assert sha(path) == item['sha256']
            info = tarfile.TarInfo(item['path']); info.size = item['bytes']; info.mode = 0o644
            with path.open('rb') as src: tar.addfile(info, src)
        info = tarfile.TarInfo('report_inputs.json'); info.size = len(encoded); info.mode = 0o644
        tar.addfile(info, io.BytesIO(encoded))
    manifest = {'files': [*payload['evidence'], {'path': 'report_inputs.json', 'sha256': hashlib.sha256(encoded).hexdigest(), 'bytes': len(encoded)}],
                'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
                'exclusions': 'Serialized tensor checkpoints and external model/dataset caches; no inference replay.'}
    prior = {p for p in output.glob('raw_artifacts.tar.gz.part-*') if p.name.rsplit('.part-', 1)[-1].isdigit()}
    new = set()
    if archive.stat().st_size > limit:
        parts = []
        with archive.open('rb') as stream:
            while chunk := stream.read(limit):
                path = output / f'raw_artifacts.tar.gz.part-{len(parts):03d}'
                path.write_bytes(chunk); new.add(path)
                parts.append({'path': path.name, 'sha256': sha(path), 'bytes': len(chunk)})
        manifest['archive_parts'] = parts
        archive.unlink()
    else: manifest['archive_path'] = archive.name
    for path in prior - new: path.unlink()
    write(output / 'raw_manifest.json', manifest)


# Live task readers are kept below the report mathematics so portable replay
# requires neither torch nor the training modules.


def source_checks(evidence, root, manifest):
    mapping = manifest['source_sha256']
    assert mapping
    for name, expected in mapping.items():
        path = root / 'source' / name
        assert sha(path) == expected, path
        evidence.add(path)


def empty_lanes(methods=METHODS):
    return {lane: {method: {'records': [], 'scores': []} for method in methods} for lane in LANES}


def add_record(cell, row, scores):
    assert row['seed'] not in [existing['seed'] for existing in cell['records']]
    cell['records'].append(row)
    cell['scores'].append(scores)


def order_cells(data):
    for cells in data['lanes'].values():
        for cell in cells.values():
            ordered = sorted(zip(cell['records'], cell['scores']), key=lambda item: item[0]['seed'])
            cell['records'] = [item[0] for item in ordered]
            cell['scores'] = [item[1] for item in ordered]
    return data


def extraction(evidence, recipes):
    root = evidence.raw / 'extraction'
    raw = evidence.json(root / 'results.json')
    if not raw.get('complete'): raise NotReady('ViGGO matrix is incomplete')
    manifest = evidence.json(root / 'manifest.json')
    assert raw['manifest'] == manifest and manifest['recipes'] == recipes
    source_checks(evidence, root, manifest)
    splits = evidence.json(root / 'split_manifest.json')
    assert sha(root / 'split_manifest.json') == manifest['dataset']['split_manifest_sha256']
    gold = splits['test']
    assert len(gold) == 256 and len({row['id'] for row in gold}) == 256
    expected = {(lane, method, seed) for lane in LANES for method in METHODS for seed in SEEDS}
    assert len(raw['lanes']) == len(expected) and {(r['lane'], r['method'], r['seed']) for r in raw['lanes']} == expected
    data = {'example_ids': [row['id'] for row in gold], 'strata': None, 'lanes': empty_lanes(),
            'manifest': manifest, 'audit': require_audit(evidence, root), 'initialization': {}}
    history = evidence.json(root / 'endpoint_history_parity.json')
    assert history['status'] == 'pass' and not history['missing'] and len(history['checks']) == 10
    for check in history['checks']:
        for kind in ('reference', 'candidate'):
            for name, expected_hash in check[kind + '_sha256'].items():
                if Path(name).suffix == '.pt': continue
                path = Path(check[kind]) / name
                assert sha(path) == expected_hash, path
                evidence.add(path)
    for method in METHODS:
        pilots = [row for row in raw['tuning'] if row['method'] == method]
        assert len(pilots) == 2 and all(row['seed'] == 42 for row in pilots)
        first = [evidence.jsonl(root / row['artifact_directory'] / 'training.jsonl')[0]['loss'] for row in pilots]
        assert len(set(first)) == 1
        data['initialization'][method] = {'first_batch_loss': first[0], 'seed': 42,
                                          'rates': [row['lr'] for row in pilots], 'first_losses_by_rate': first}
    scored = {}
    for link in raw['lanes']:
        fit = raw['fits'][link['fit_key']]
        method, seed, lane = link['method'], link['seed'], link['lane']
        lr = manifest['configuration']['fixed_lr'] if lane == LANES[0] else raw['selected_learning_rates'][method]
        assert (fit['method'], fit['seed'], fit['lr']) == (method, seed, lr)
        assert fit['recipe'] == recipes[method] and fit['steps'] == 256
        assert fit['frozen_weights_unchanged']
        assert len(fit['norm_diagnostics']) == 72
        assert all(row['clamp_activations'] == 0 and row['minimum_raw_norm'] > 0 for row in fit['norm_diagnostics'].values())
        directory = root / fit['artifact_directory']
        assert sha(directory / 'result.json') == fit['result_sha256']
        evidence.add(directory / 'result.json')
        if link['fit_key'] not in scored:
            predictions = evidence.jsonl(directory / 'predictions.jsonl')
            mapping = {row['id']: row for row in predictions}
            assert len(mapping) == len(predictions) == len(gold) and set(mapping) == set(data['example_ids'])
            scores = []
            for target in gold:
                row = mapping[target['id']]
                assert row['gold'] == target['target'] and row['utterance'] == target['utterance']
                try: parsed = json.loads(row['text'].strip())
                except (ValueError, TypeError): parsed = None
                assert parsed == row['parsed']
                correct = parsed == target['target']
                assert correct == row['exact_match']
                scores.append(float(correct))
            close(statistics.mean(scores), fit['test']['exact_match'], 1e-12)
            scored[link['fit_key']] = scores
        add_record(data['lanes'][lane][method], {
            'seed': seed, 'value': fit['test']['exact_match'] * 100,
            'learning_rate': lr, 'training_seconds': fit['train_and_validation_seconds'],
            'peak_memory_bytes': fit['peak_allocated_bytes'], 'trainable_parameters': fit['trainable_parameters'],
            'fit_id': 'extraction/' + fit['artifact_directory'], 'selected_step': fit['selected_step'],
            'artifact_result_sha256': fit['result_sha256'], 'recipe_sha256': recipes[method]['recipe_sha256']},
            scored[link['fit_key']])
    return order_cells(data)


def aircraft(evidence, recipes):
    root = evidence.raw / 'aircraft_rank2'
    matrix = evidence.json(root / 'matrix.json')
    if matrix['status'] != 'complete': raise NotReady('Aircraft matrix is incomplete')
    assert matrix['recipes'] == recipes and matrix['methods'] == METHODS and matrix['seeds'] == SEEDS
    assert matrix['fixed_original_lr'] == .001 and matrix['rank'] == 2
    core_path, task_path = root / 'core_parity_evidence.json', root / 'task_parity_evidence.json'
    assert sha(core_path) == matrix['core_gate_sha256'] and sha(task_path) == matrix['task_gate_sha256']
    assert evidence.json(core_path)['status'] == 'pass'
    assert evidence.json(task_path)['passed'] is True
    data = {'lanes': empty_lanes(), 'manifest': matrix, 'audit': require_audit(evidence, root)}
    for path in root.glob('*historical*control.json'):
        control = evidence.json(path)
        if 'input_sha256' in control: audit_inputs(evidence, control, root)
    for method in METHODS:
        alias = matrix['aliases'].get(method)
        source_method = alias['source_method'] if alias else method
        if alias:
            assert alias['recipe'] == recipes[method] and alias['source_recipe'] == recipes[source_method]
            assert alias['task_gate_sha256'] == matrix['task_gate_sha256']
            assert alias['core_gate_sha256'] == matrix['core_gate_sha256']
            differing = {key for key in recipes[method] if recipes[method][key] != recipes[source_method][key]}
            assert differing == {'method', 'magnitude_fp32', 'recipe_sha256'}
        directory = root / source_method
        done = evidence.json(directory / 'completion.json')
        assert done['status'] == 'complete' and done['method'] == source_method
        protocol = evidence.json(directory / 'protocol.json')
        assert protocol['recipe'] == recipes[source_method]
        source_checks(evidence, directory, protocol['provenance'])
        split = evidence.json(directory / 'split_manifest.json')
        assert sha(directory / 'split_manifest.json') == protocol['provenance']['split_manifest_sha256']
        if 'split_manifest' in data: assert data['split_manifest'] == split
        else: data['split_manifest'] = split
        selection = evidence.json(directory / 'rank_2/selected.json')[source_method]
        assert selection['magnitude_multiplier'] == 1
        for lane in LANES:
            fit_root = directory if lane == 'validation_tuned' else directory / 'fixed_lr_0.001'
            rows = evidence.json(fit_root / 'results.json')
            rows = sorted((row for row in rows if row['method'] == source_method), key=lambda row: row['seed'])
            assert [row['seed'] for row in rows] == SEEDS
            for row in rows:
                seed = row['seed']
                lr = selection['lr'] if lane == 'validation_tuned' else .001
                assert row['adapter_learning_rate'] == lr and row['epochs'] == 20 and row['rank'] == 2
                assert row['frozen_base_unchanged']
                fit = fit_root / f'rank_2/final/{source_method}/seed_{seed}'
                assert evidence.json(fit / 'result.json') == row
                pred = evidence.json(fit / 'test_predictions.json')
                labels = np.asarray(pred['labels'])
                correct = labels == pred['predictions']
                assert len(labels) == 3333 and len(np.unique(labels)) == 100
                assert labels.tolist() == [item['label'] for item in split['splits']['test']]
                if 'strata' in data: assert data['strata'] == labels.tolist()
                else:
                    data['strata'] = labels.tolist()
                    data['example_ids'] = [item['image'] for item in split['splits']['test']]
                    data['example_identity_scope'] = 'Common ordered official test split; image paths, hashes and per-row class labels checked.'
                close(float(correct.mean()), row['test']['accuracy'])
                value = float(metric([correct, correct, correct], labels)[0] * 100)
                close(value, row['test']['macro_class_accuracy'] * 100, 2e-5)
                diagnostics = row['adapter_diagnostics']
                assert len(diagnostics) == 48
                assert all(item['clamp_activations'] == 0 and item['minimum_raw_norm'] > 0 for item in diagnostics)
                # Byte-copied lane fits retain their measured source identity.
                shared_lane = selection['lr'] == .001
                canonical_fit = directory / f'rank_2/final/{source_method}/seed_{seed}' if shared_lane else fit
                add_record(data['lanes'][lane][method], {
                    'seed': seed, 'value': value, 'learning_rate': lr,
                    'training_seconds': row['training_wall_seconds'], 'peak_memory_bytes': row['peak_cuda_allocated_bytes'],
                    'trainable_parameters': row['trainable_parameters'],
                    'fit_id': str(canonical_fit.relative_to(evidence.raw)), 'selected_epoch': row['best_epoch'],
                    'evaluated_method': source_method, 'identity_alias': alias,
                    'artifact_result_sha256': sha(fit / 'result.json'), 'recipe_sha256': recipes[method]['recipe_sha256']},
                    correct.astype(float).tolist())
    return order_cells(data)


def ranked_metrics(ranking, relevant):
    assert len(ranking) == len(set(ranking)) == 10
    grades = [relevant.get(document, 0) for document in ranking]
    ideal = sorted(relevant.values(), reverse=True)[:10]
    dcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
    idcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    assert idcg > 0 and all(grade > 0 for grade in relevant.values())
    return {'ndcg_at_10': dcg / idcg,
            'mrr_at_10': next((1 / (index + 1) for index, grade in enumerate(grades) if grade), 0),
            'recall_at_10': sum(grade > 0 for grade in grades) / len(relevant)}


def retrieval(evidence, recipes, task, root_name=None, diagnostic_count=12):
    methods = list(recipes)
    root = evidence.raw / (root_name or ('scifact' if task == 'scifact' else 'nfcorpus_rank8'))
    raw = evidence.json(root / 'results.json')
    if not raw.get('complete'): raise NotReady(f'{task} retrieval matrix is incomplete')
    manifest = evidence.json(root / 'manifest.json')
    assert raw['manifest_sha256'] == sha(root / 'manifest.json')
    assert manifest['recipes'] == recipes and manifest['rank'] == 8
    source_checks(evidence, root, manifest)
    for name, expected in manifest['input_sha256'].items():
        assert sha(root / name) == expected
        evidence.add(root / name)
    assert manifest['configuration']['seeds'] == SEEDS
    expected = {(lane, method, seed) for lane in ('fixed', 'tuned') for method in methods for seed in SEEDS}
    assert len(raw['runs']) == len(expected) and {(r['lane'], r['method'], r['seed']) for r in raw['runs']} == expected
    audit = require_audit(evidence, root)
    parity_path = root / 'endpoint_quality_parity.json'
    if parity_path.exists():
        parity = evidence.json(parity_path)
        assert parity['status'] == 'pass'
        for item in parity['rows']:
            reference = Path(item['reference'])
            if 'reference_predictions_sha256' in item:
                path = reference / 'test_per_query.json'
                assert sha(path) == item['reference_predictions_sha256']
                evidence.add(path)
            for dataset, bindings in item.get('prediction_sha256', {}).items():
                path = reference / dataset / 'test_per_query.json'
                if not path.exists(): path = reference / 'test_per_query.json'
                assert sha(path) == bindings['reference_sha256']
                evidence.add(path)
    datasets = ['scifact'] if task == 'scifact' else ['nfcorpus', 'scifact']
    data = {}
    for dataset in datasets:
        splits = evidence.json(root / f'{dataset}_splits.json')
        gold = evidence.json(root / f'{dataset}_qrels.json')
        ids = splits['test']
        assert len(ids) == len(set(ids))
        output_task = 'scifact_transfer' if task == 'nfcorpus' and dataset == 'scifact' else dataset
        data[output_task] = {'example_ids': ids, 'strata': None, 'lanes': empty_lanes(methods),
                             'manifest': manifest, 'audit': audit, 'initialization': {}}
        for method in methods:
            pilots = raw['tuning'][method]['trials']
            first = [evidence.jsonl(root / row['fit_id'] / 'training.jsonl')[0]['loss'] for row in pilots]
            assert len(set(first)) == 1
            data[output_task]['initialization'][method] = {'first_batch_loss': first[0], 'seed': 42,
                'rates': [row['learning_rate'] for row in pilots], 'first_losses_by_rate': first}
        cached = {}
        for row in raw['runs']:
            method, seed = row['method'], row['seed']
            lane = 'fixed_original_lr' if row['lane'] == 'fixed' else 'validation_tuned'
            selected = raw['tuning'][method]
            assert not selected['test_used_for_selection']
            assert selected['fixed_learning_rate'] == manifest['configuration']['fixed_lr']
            lr = selected['fixed_learning_rate' if lane == LANES[0] else 'selected_learning_rate']
            assert row['learning_rate'] == lr and row['rank'] == 8
            assert row['recipe_sha256'] == recipes[method]['recipe_sha256'] and row['frozen_weights_unchanged']
            for diagnostics in [row['diagnostics'], *([row['post_reload_diagnostics']] if 'post_reload_diagnostics' in row else [])]:
                assert len(diagnostics) == diagnostic_count
                assert all(item['clamp_activations'] == 0 and item['minimum_raw_norm'] > 0 for item in diagnostics.values())
            directory = root / row['fit_id']
            saved = evidence.json(directory / 'result.json')
            assert all(row[key] == value for key, value in saved.items())
            if row['fit_id'] not in cached:
                predictions = evidence.json(directory / dataset / 'test_per_query.json')
                mapping = {item['query_id']: item for item in predictions}
                assert len(mapping) == len(predictions) == len(ids) and set(mapping) == set(ids)
                values = []
                for qid in ids:
                    item = mapping[qid]
                    computed = ranked_metrics(item['top_10'], gold[qid])
                    for name, score in computed.items(): close(score, item[name], 1e-12)
                    assert len(item['scores']) == 10 and all(math.isfinite(score) for score in item['scores'])
                    assert item['scores'] == sorted(item['scores'], reverse=True)
                    values.append(computed['ndcg_at_10'])
                cached[row['fit_id']] = values
            values = cached[row['fit_id']]
            close(statistics.mean(values), row['evaluation'][dataset]['test']['ndcg_at_10'], 1e-12)
            add_record(data[output_task]['lanes'][lane][method], {
                'seed': seed, 'value': statistics.mean(values) * 100, 'learning_rate': lr,
                'training_seconds': row['training']['seconds'], 'peak_memory_bytes': row['training']['peak_allocated_bytes'],
                'trainable_parameters': row['trainable_parameters'], 'fit_id': root.name + '/' + row['fit_id'],
                'artifact_result_sha256': sha(directory / 'result.json'), 'recipe_sha256': recipes[method]['recipe_sha256'],
                'shared_lanes': row['shared_lanes'], 'reused_full_budget_validation_fit': row['reused_full_budget_validation_fit']}, values)
    return {key: order_cells(value) for key, value in data.items()}


def bound_references(evidence, value):
    if isinstance(value, list):
        for item in value: bound_references(evidence, item)
    elif isinstance(value, dict):
        for field in ('path', 'log'):
            if field in value and isinstance(value[field], str) and 'sha256' in value:
                path = Path(value[field])
                if path.is_absolute() and path.is_file():
                    assert sha(path) == value['sha256'], path
                    evidence.add(path)
        for item in value.values(): bound_references(evidence, item)


def command_evidence(evidence, filename, dispatcher):
    commands = evidence.json(evidence.raw / filename)
    assert commands and all(row.get('exit_code') == 0 for row in commands), filename
    dispatcher_hash = sha(Path(dispatcher))
    for row in commands:
        assert row['queue_source_sha256'] == dispatcher_hash
        if row.get('log'): evidence.add(Path(row['log']))
    evidence.add(Path(dispatcher))


def collect(raw, repo):
    evidence = Evidence(raw, repo)
    gate = evidence.json(Path('/var/tmp/dora-bench/change_ablation_core_validation.json'))
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    assert sha(repo / 'experiments/change_ablations.py') == gate['core_source_sha256']
    for name, expected in gate['source_sha256'].items():
        assert sha(repo / name) == expected, name
        evidence.add(repo / name)
    recipes = gate['recipes']
    assert list(recipes) == METHODS
    bound_references(evidence, gate)
    review = evidence.json(Path('/var/tmp/dora-bench/change_ablation_independent_review.json'))
    assert review['status'] == 'pass' and review['source_sha256'] == gate['core_source_sha256']
    assert len(review['checks']) == 48 and review['inactive_clamp_output_gradient_exact']
    assert all(row['finite_nonzero_all_gradients'] and row['clamp_activations'] == 0
               for row in review['checks'])
    report_review_path = Path('/var/tmp/dora-bench/change_report_independent_review.json')
    if report_review_path.exists():
        report_review = evidence.json(report_review_path)
        assert report_review['status'] == 'pass'
        assert report_review['source_sha256'] == sha(repo / 'experiments/change_report.py')
    tasks = retrieval(evidence, recipes, 'scifact')
    tasks.update(retrieval(evidence, recipes, 'nfcorpus'))
    tasks['aircraft'] = aircraft(evidence, recipes)
    tasks['extraction'] = extraction(evidence, recipes)
    engineering_root = raw / 'engineering_controls'
    engineering = evidence.json(engineering_root / 'engineering_audit.json')
    assert engineering['status'] == 'pass' and len(engineering['checks']) == 6
    for name, expected in engineering['source_sha256'].items():
        assert sha(engineering_root / name) == expected
        evidence.add(engineering_root / name)
    for folder in ('scifact', 'nfcorpus_rank8', 'aircraft_rank2', 'extraction', 'aircraft_task_validation', 'engineering_controls'):
        for path in (raw / folder).rglob('*'):
            if path.is_file() and path.suffix in SUFFIXES and '__pycache__' not in path.parts:
                evidence.add(path)
    for name in ('root_commands.json', 'extraction_commands.json', 'retrieval_commands.json',
                 'nfcorpus_postprocess_commands.json',
                 'extraction.log', 'scifact.log', 'nfcorpus_rank8.log',
                 'nfcorpus_control.log', 'nfcorpus_sweep.log'):
        if (raw / name).exists(): evidence.add(raw / name)
    evidence.add(Path('/var/tmp/dora-bench/audit_nfcorpus_when_complete.py'))
    for name in ('experiments/change_report.py', 'experiments/change_ablations.py', 'experiments/test_change_ablations.py'):
        evidence.add(repo / name)
    inventory = Path('/home/catid/dora/experiments/change_inventory.md')
    if inventory.exists(): evidence.add(inventory)
    previous = evidence.json(Path('/home/catid/dora/results/2026-10-09-original/summary.json'))
    prior_audit = evidence.json(Path('/home/catid/dora/results/2026-10-09-original/offline_rebuild_audit.json'))
    assert prior_audit['status'] == 'pass'
    endpoint_reference = {}
    for task, old_key in [('scifact', 'scifact'), ('nfcorpus', 'nfcorpus_r8'),
                          ('scifact_transfer', 'scifact_transfer_r8'), ('aircraft', 'aircraft_r2'),
                          ('extraction', 'extraction')]:
        endpoint_reference[task] = {}
        for method, old_method in [('ab_original', 'dora_original'), ('ab_revised', 'dora')]:
            rows = sorted((row for row in previous['tasks'][old_key]['records'] if row['method'] == old_method), key=lambda row: row['seed'])
            assert [row['seed'] for row in rows] == SEEDS
            summary = stats([row['value'] for row in rows])
            close(summary['mean'], previous['tasks'][old_key]['summary'][old_method]['mean'], 1e-10)
            endpoint_reference[task][method] = {'records': rows, 'summary': summary}
    payload = {'complete': True, 'tasks': tasks, 'recipes': recipes, 'scope': SCOPE, 'core_gate': gate,
               'independent_review': review,
               'endpoint_reference': endpoint_reference, 'engineering': engineering,
               'source_sha256': sha(repo / 'experiments/change_report.py'),
               'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}
    return payload, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('/var/tmp/dora-bench/change_ablation_runs'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('results/2026-10-09-change-ablations'))
    parser.add_argument('--rebuild', type=Path)
    args = parser.parse_args()
    if args.rebuild:
        payload = read(args.rebuild)
        assert sha(Path(__file__)) == payload['source_sha256']
        for item in payload['evidence']:
            path = args.rebuild.parent / item['path']
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
        expected = payload['comparisons'], payload['tuning_effects']
        compute(payload)
        assert (payload['comparisons'], payload['tuning_effects']) == expected
        render(payload, args.output)
        write(args.output / 'rebuild_audit.json', {'status': 'pass', 'evidence_files_verified': len(payload['evidence']),
              'source_sha256': sha(Path(__file__)), 'input_sha256': sha(args.rebuild),
              'scope': 'All retained inputs hash-verified, per-seed/example summaries and all paired intervals recomputed, outputs regenerated without inference.'})
    else:
        try: payload, evidence = collect(args.raw_root.resolve(), args.repo.resolve())
        except NotReady as error: parser.exit(2, str(error) + '\n')
        compute(payload)
        args.output.mkdir(parents=True, exist_ok=True)
        bundle(payload, evidence, args.output)
        render(payload, args.output)
    print(json.dumps({'status': 'complete', 'tasks': len(payload['tasks']), 'recipes': len(METHODS), 'lanes': LANES, 'output': str(args.output)}))


if __name__ == '__main__': main()
