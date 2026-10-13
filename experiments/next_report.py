"""Audited reporting and portable replay for changes added to FP32 magnitudes."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments import change_report as prior

METHODS = ['next_base', 'next_axis', 'next_detach', 'next_norm_fp32', 'next_init', 'next_factorized']
LABELS = dict(zip(METHODS, ['Current baseline', '+ row axis', '+ detach norm', '+ FP32 norm', '+ alternate initializer', '+ factorized projection']))
LANES = prior.LANES
TASKS = prior.TASKS
OLD = dict(zip(METHODS, ['ab_m_fp32', 'ab_axis', 'ab_detach', 'ab_norm_fp32', 'ab_init', 'ab_factorized']))
sha, read, write = prior.sha, prior.read, prior.write


def extraction(evidence, recipes):
    root = evidence.raw / 'extraction'
    raw = evidence.json(root / 'results.json')
    manifest = evidence.json(root / 'manifest.json')
    assert raw['complete'] and raw['manifest'] == manifest and manifest['recipes'] == recipes
    prior.source_checks(evidence, root, manifest)
    splits = evidence.json(root / 'split_manifest.json')
    assert sha(root / 'split_manifest.json') == manifest['dataset']['split_manifest_sha256']
    gold = splits['test']
    assert len(gold) == 256 and len({row['id'] for row in gold}) == 256
    expected = {(lane, method, seed) for lane in LANES for method in METHODS for seed in prior.SEEDS}
    assert len(raw['lanes']) == len(expected) and {(r['lane'], r['method'], r['seed']) for r in raw['lanes']} == expected
    data = {'example_ids': [row['id'] for row in gold], 'strata': None, 'lanes': prior.empty_lanes(METHODS),
            'manifest': manifest, 'audit': prior.require_audit(evidence, root)}
    for method in METHODS:
        pilots = [row for row in raw['tuning'] if row['method'] == method]
        assert len(pilots) == 2 and {p['lr'] for p in pilots} == {.0001, .0003}
        winner = min(pilots, key=lambda r: r['validation_nll'])
        assert raw['selected_learning_rates'][method] == winner['lr']
        for fit in pilots:
            assert fit['steps'] == 48 and fit['seed'] == 42 and 'test' not in fit
            evidence.add(root / fit['artifact_directory'] / 'result.json')
            evidence.add(root / fit['artifact_directory'] / 'training.jsonl')
    cache = {}
    for link in raw['lanes']:
        fit = raw['fits'][link['fit_key']]
        method, seed, lane = link['method'], link['seed'], link['lane']
        lr = manifest['configuration']['fixed_lr'] if lane == LANES[0] else raw['selected_learning_rates'][method]
        assert (fit['method'], fit['seed'], fit['lr']) == (method, seed, lr)
        assert fit['recipe'] == recipes[method] and fit['steps'] == 256 and fit['frozen_weights_unchanged']
        assert len(fit['norm_diagnostics']) == 72
        assert all(x['clamp_activations'] == 0 and x['minimum_raw_norm'] > 0 for x in fit['norm_diagnostics'].values())
        directory = root / fit['artifact_directory']
        assert sha(directory / 'result.json') == fit['result_sha256']
        evidence.add(directory / 'result.json'); evidence.add(directory / 'training.jsonl')
        if link['fit_key'] not in cache:
            predictions = evidence.jsonl(directory / 'predictions.jsonl')
            mapping = {row['id']: row for row in predictions}
            assert len(mapping) == len(predictions) == len(gold) and set(mapping) == set(data['example_ids'])
            values = []
            for target in gold:
                row = mapping[target['id']]
                assert row['gold'] == target['target'] and row['utterance'] == target['utterance']
                try: parsed = json.loads(row['text'].strip())
                except (ValueError, TypeError): parsed = None
                assert parsed == row['parsed']
                correct = parsed == target['target']
                assert correct == row['exact_match']
                values.append(float(correct))
            prior.close(statistics.mean(values), fit['test']['exact_match'], 1e-12)
            cache[link['fit_key']] = values
        prior.add_record(data['lanes'][lane][method], {
            'seed': seed, 'value': fit['test']['exact_match'] * 100, 'learning_rate': lr,
            'training_seconds': fit['train_and_validation_seconds'], 'peak_memory_bytes': fit['peak_allocated_bytes'],
            'trainable_parameters': fit['trainable_parameters'], 'fit_id': 'extraction/' + fit['artifact_directory'],
            'selected_step': fit['selected_step'], 'artifact_result_sha256': fit['result_sha256'],
            'recipe_sha256': recipes[method]['recipe_sha256']}, cache[link['fit_key']])
    return prior.order_cells(data)


def aircraft(evidence, recipes):
    from experiments.change_aircraft_analysis import Audit
    control_path = evidence.raw / 'aircraft_control/task_parity.json'
    control = evidence.json(control_path)
    assert control['passed'] and len(control['cases']) == 6 and all(r['passed'] for r in control['cases'])
    assert {r['candidate'] for r in control['cases']} == set(METHODS)
    assert control['base_and_adapter_dtype'] == 'float32' and control['autocast'] == 'bfloat16'
    assert control['rank'] == 2 and control['seed'] == 42
    for name, digest in control['source_sha256'].items():
        assert sha(evidence.repo / name) == digest
        evidence.add(evidence.repo / name)
    root = Path('/var/tmp/dora-bench/change_ablation_runs/aircraft_rank2')
    reference = Path('/var/tmp/dora-bench/round2/vision/rank2_run')
    audit = Audit(root, reference)
    old_summary_path = evidence.repo / 'results/2026-10-09-change-ablations/summary.json'
    historical = evidence.json(old_summary_path)['tasks']['aircraft']
    assert historical['manifest'] == audit.matrix
    assert historical['split_manifest'] == audit.manifest
    assert historical['strata'] == audit.labels.tolist()
    assert historical['example_ids'] == [item['image'] for item in audit.manifest['splits']['test']]
    assert evidence.json(evidence.raw / 'aircraft_control/split_manifest.json') == audit.manifest
    assert control['initial_head_sha256'] == audit.baselines[42]['initial_head_sha256']
    assert control['initial_backbone_sha256'] == audit.baselines[42]['initial_backbone_sha256']
    data = {key: copy.deepcopy(historical[key]) for key in ('example_ids', 'strata', 'split_manifest')}
    data['lanes'] = prior.empty_lanes(METHODS)
    data['reuse'] = {}
    cases = {r['candidate']: r for r in control['cases']}
    checks = []
    for method, old in OLD.items():
        # The previous ab_m_fp32 cell already explicitly aliases ab_original.
        source = 'ab_original' if old == 'ab_m_fp32' else old
        arrays, check = audit.method(source)
        checks.append(check)
        assert cases[method]['reference'] == old
        case = cases[method]
        assert case['initial_parameters']['bitwise_equal'] and case['initial_parameters']['max_absolute_error'] == 0
        assert [step['step'] for step in case['steps']] == [1, 2, 3]
        for step in case['steps']:
            assert step['logits_loss_clip_bitwise_equal']
            for field in ('gradients', 'parameters_after_adamw'):
                assert step[field]['bitwise_equal'] and step[field]['max_absolute_error'] == 0
        assert len(case['candidate_norm_diagnostics']) == 48
        for diagnostic in case['candidate_norm_diagnostics'].values():
            assert diagnostic['recipe_sha256'] == recipes[method]['recipe_sha256']
            assert diagnostic['magnitude_dtype'] == diagnostic['factor_dtype'] == 'torch.float32'
            assert diagnostic['norm_floor'] is None and diagnostic['clamp_activations'] == 0
            assert math.isfinite(diagnostic['minimum_raw_norm']) and diagnostic['minimum_raw_norm'] > 0
        flags = ('axis', 'detach_norm', 'magnitude_fp32', 'norm_fp32', 'published_init', 'factorized', 'clamp_norm')
        source_recipe = audit.matrix['recipes'][source]
        differing = [flag for flag in flags if recipes[method][flag] != source_recipe[flag]]
        assert differing == ['magnitude_fp32']
        assert recipes[method]['magnitude_fp32'] and not source_recipe['magnitude_fp32']
        if old != source:
            alias = audit.matrix['aliases'][old]
            assert alias['source_method'] == source and alias['recipe'] == audit.matrix['recipes'][old]
            assert alias['source_recipe'] == source_recipe
            assert [flag for flag in flags if alias['recipe'][flag] != alias['source_recipe'][flag]] == ['magnitude_fp32']
            assert alias['task_gate_sha256'] == audit.matrix['task_gate_sha256']
            assert alias['core_gate_sha256'] == audit.matrix['core_gate_sha256']
            assert next(r for r in audit.task['cases'] if r['name'] == alias['task_control_case'])['passed']
        protocol = audit.read('experiment', Path(source) / 'protocol.json')
        assert protocol['provenance']['checkpoint_sha256'] == control['checkpoint_sha256']
        for lane in LANES:
            cell = copy.deepcopy(historical['lanes'][lane][old])
            for i, row in enumerate(cell['records']):
                assert np.array_equal(cell['scores'][i], arrays[lane, row['seed']])
                assert row['recipe_sha256'] == audit.matrix['recipes'][old]['recipe_sha256']
                fit_root = Path(source) if lane == 'validation_tuned' else Path(source) / 'fixed_lr_0.001'
                fit = fit_root / f"rank_2/final/{source}/seed_{row['seed']}"
                measured = audit.read('experiment', fit / 'result.json')
                assert row['artifact_result_sha256'] == audit.bindings['experiment'][str(fit / 'result.json')]
                assert row['learning_rate'] == measured['adapter_learning_rate']
                assert row['training_seconds'] == measured['training_wall_seconds']
                assert row['peak_memory_bytes'] == measured['peak_cuda_allocated_bytes']
                assert row['trainable_parameters'] == measured['trainable_parameters']
                assert row['selected_epoch'] == measured['best_epoch']
                prior.close(row['value'], measured['test']['macro_class_accuracy'] * 100, 2e-5)
                row['reused_historical_fit'] = True
                row['historical_recipe_sha256'] = row['recipe_sha256']
                row['recipe_sha256'] = recipes[method]['recipe_sha256']
                row['measured_method'] = source
                row['requested_method'] = method
                row['fit_id'] = 'historical/' + row['fit_id']
            data['lanes'][lane][method] = cell
        data['reuse'][method] = {'historical_method': source, 'historical_cell': old,
                                'control_case': cases[method]['name'], 'reason': 'FP32 magnitude cast is exact identity for this FP32 workload'}
    for which, bindings in audit.bindings.items():
        for name, digest in bindings.items():
            path = audit.roots[which] / name
            assert sha(path) == digest
            evidence.add(path)
    receipt = {'status': 'pass', 'task': 'Aircraft exact FP32 reuse', 'control_sha256': sha(control_path),
               'historical_summary_sha256': sha(old_summary_path), 'checks': checks, 'counts': audit.counts,
               'input_sha256': audit.bindings, 'artifact_roots': {k: str(v) for k,v in audit.roots.items()},
               'reuse': data['reuse']}
    receipt_path = evidence.raw / 'aircraft_reuse_audit.json'
    write(receipt_path, receipt); evidence.add(receipt_path)
    data['audit'] = receipt
    data['manifest'] = historical['manifest']
    return prior.order_cells(data)


def compute(payload):
    comparisons = []
    assert set(payload['tasks']) == set(TASKS)
    for task, data in payload['tasks'].items():
        assert set(data['lanes']) == set(LANES)
        for lane, cells in data['lanes'].items():
            assert set(cells) == set(METHODS)
            for method, cell in cells.items():
                assert len(cell['scores']) == 3 and all(len(row) == len(data['example_ids']) for row in cell['scores'])
                cell['summary'] = prior.summarize_cell(cell, data['strata'])
            for method in METHODS[1:]:
                result = prior.contrast(cells[method]['scores'], cells['next_base']['scores'], data['strata'])
                prior.close(result['difference'], cells[method]['summary']['mean'] - cells['next_base']['summary']['mean'], 2e-5)
                comparisons.append({'task': task, 'lane': lane, 'candidate': method, 'reference': 'next_base', **result})
    payload['comparisons'] = comparisons


def microbenchmark(evidence, recipes):
    root = evidence.raw / 'microbenchmark'
    measured = evidence.json(root / 'results.json')
    audit = evidence.json(root / 'audit.json')
    assert measured['complete'] and audit['status'] == 'pass'
    assert audit['input_results_sha256'] == sha(root / 'results.json')
    assert measured['recipes'] == recipes and len(measured['cases']) == 36
    for name, digest in measured['source_sha256'].items():
        assert sha(root / 'sources' / name) == digest
        evidence.add(root / 'sources' / name)
    expected = {(method, precision, *shape) for method in METHODS
                for precision in measured['configuration']['precisions']
                for shape in measured['configuration']['shapes']}
    rows = {(r['method'], r['precision'], r['in_features'], r['out_features'], r['tokens']): r for r in measured['cases']}
    assert set(rows) == expected
    for key, row in rows.items():
        assert row['validation']['status'] == 'pass' and len(row['validation']['checks']) == 5
        assert all(r['bitwise_equal'] and r['max_absolute_error'] == 0 for r in row['validation']['checks'].values())
        baseline = rows[('next_base', *key[1:])]
        for mode in ('forward', 'forward_backward'):
            value = row[mode]
            samples = value['sample_ms_per_call']
            assert len(samples) == 7 and all(math.isfinite(x) and x > 0 for x in samples)
            assert statistics.median(samples) == value['median_ms_per_call']
            assert value['peak_incremental_allocated_bytes'] > 0
            assert row['relative_to_next_base'][mode]['speedup'] == baseline[mode]['median_ms_per_call'] / value['median_ms_per_call']
            assert row['relative_to_next_base'][mode]['peak_incremental_memory_ratio'] == value['peak_incremental_allocated_bytes'] / baseline[mode]['peak_incremental_allocated_bytes']
    return measured


def render(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['svg.hashsalt'] = 'dora-next-ablation-20261009'
    table = prior.markdown_table
    for lane in LANES:
        rows = [[LABELS[method], *[prior.fmt(payload['tasks'][task]['lanes'][lane][method]['summary']) for task in TASKS]] for method in METHODS]
        (output / (lane + '_table.md')).write_text(table(['Addition to FP32-magnitude baseline', *[v[0] + ' / ' + v[1] for v in TASKS.values()]], rows) + '\nMean ± sample SD over seeds 42–44. Aircraft results are explicitly reused after exact FP32 parity verification.\n')
    rows = []
    for row in payload['comparisons']:
        low, high = row['ci95']
        rows.append([TASKS[row['task']][0], row['lane'], LABELS[row['candidate']], f"{row['difference']:+.3f}", f"[{low:+.3f}, {high:+.3f}]"])
    (output / 'paired_effects.md').write_text(table(['Task', 'LR policy', 'Addition', 'Difference in points', 'Exploratory paired 95% interval'], rows) + '\nPositive favors the addition. Three training seeds; 10,000 crossed paired seed/example draws, stratified by class for Aircraft. No multiple-comparison correction.\n')
    rows = []
    for task, data in payload['tasks'].items():
        if task == 'scifact_transfer': continue
        for lane, cells in data['lanes'].items():
            for method in METHODS:
                s = cells[method]['summary']
                rows.append([TASKS[task][0], lane, LABELS[method], ', '.join(f'{v:g}' for v in s['learning_rates']), prior.fmt(s['training_seconds']), f"{max(s['peak_memory_bytes']['values'])/2**30:.3f}", ', '.join(map(str,s['parameters'])), 'Historical reuse' if task == 'aircraft' else 'Fresh'])
    (output / 'runtime_memory.md').write_text(table(['Task','LR policy','Addition','LRs','Instrumented seconds ± SD','Peak GiB','Parameters','Measurement'], rows) + '\nThese timings include task-loop validation and shared norm diagnostics as applicable. Fits shared between LR policies are counted once. They do not establish production throughput.\n')
    if 'microbenchmark' in payload:
        measured = payload['microbenchmark']
        rows = []
        for r in sorted(measured['cases'], key=lambda r: (r['precision'], r['in_features'], r['tokens'], METHODS.index(r['method']))):
            value = r['forward_backward']
            relative = r['relative_to_next_base']['forward_backward']
            rows.append([r['precision'], f"{r['in_features']} × {r['out_features']} / {r['tokens']}",
                         LABELS[r['method']], f"{value['median_ms_per_call']:.4f}",
                         f"{value['peak_incremental_allocated_bytes']/2**20:.2f}", f"{relative['speedup']:.3f}×"])
        (output / 'microbenchmark.md').write_text(table(['Precision', 'Weight / tokens', 'Addition', 'Forward + backward ms', 'Incremental peak MiB', 'Speedup vs baseline'], rows)
            + '\nRTX PRO 6000 Blackwell Max-Q, rank8, seven samples ×20 calls after ten warmups; synchronized forward/backward, input gradients, optimizer excluded. Existing tensors and allocator reservation excluded from incremental allocation. The shared read-only norm diagnostics are disabled only for timing, with bitwise output/all-gradient parity checked for every case. These single-layer results vary with shape and do not establish full-model training speed.\n')
        write(output / 'microbenchmark.json', measured)
    fig, axes = plt.subplots(2, 5, figsize=(17, 8.2), layout='constrained')
    for i, lane in enumerate(LANES):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            selected = [next(r for r in payload['comparisons'] if r['task']==task and r['lane']==lane and r['candidate']==m) for m in METHODS[1:]]
            values = np.array([r['difference'] for r in selected])
            lower = np.array([r['ci95'][0] for r in selected]); upper = np.array([r['ci95'][1] for r in selected])
            y = np.arange(5)
            ax.barh(y, values, color=['#26836b' if v>=0 else '#c56d54' for v in values], alpha=.85)
            ax.errorbar(values, y, xerr=np.array([np.maximum(values-lower, 0),np.maximum(upper-values, 0)]), fmt='none', ecolor='#283442', capsize=3)
            ax.axvline(0, color='#283442', lw=.8)
            ax.set_yticks(y, ['Row axis','Detach norm','FP32 norm','Initializer','Factorization'] if j==0 else ['']*5)
            ax.invert_yaxis(); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
            ax.set_title(TASKS[task][0] + ('\nHistorical FP32 reuse' if task=='aircraft' else ''), fontsize=10)
            ax.set_xlabel('Change in score (points)', fontsize=9)
            if j==0: ax.set_ylabel('Fixed baseline LR' if i==0 else 'Validation-tuned LR', fontsize=11)
    fig.suptitle('One addition to original DoRA + FP32 magnitudes\nPaired differences and exploratory 95% intervals; three seeds', fontsize=14)
    fig.savefig(output / 'next_change_bars.png', dpi=160)
    fig.savefig(output / 'next_change_bars.svg', metadata={'Date': None})
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout='constrained')
    for ax, task in zip(axes.flat, TASKS):
        selected = [next(r for r in payload['comparisons'] if r['task']==task and r['lane']=='validation_tuned' and r['candidate']==m) for m in METHODS[1:]]
        values = np.array([r['difference'] for r in selected])
        lower = np.array([r['ci95'][0] for r in selected]); upper = np.array([r['ci95'][1] for r in selected])
        y = np.arange(5)
        ax.barh(y, values, color=['#26836b' if v>=0 else '#c56d54' for v in values], alpha=.85)
        ax.errorbar(values, y, xerr=np.array([np.maximum(values-lower,0),np.maximum(upper-values,0)]), fmt='none', ecolor='#283442', capsize=3)
        ax.axvline(0, color='#283442', lw=.8)
        ax.set_yticks(y, ['Row axis','Detach norm','FP32 norm','Initializer','Factorization'])
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
        ax.set_title(TASKS[task][0] + (' (reused)' if task=='aircraft' else ''), fontsize=11)
        ax.set_xlabel('Change in score (points)', fontsize=10)
    axes[1,2].axis('off')
    axes[1,2].text(.05,.9,'Baseline: original + FP32 magnitudes\n\nEach bar adds one change.\nPositive values favor the addition.\n\nEqual validation tuning; three seeds.\nBars: mean paired difference.\nWhiskers: exploratory 95% interval.\nNo multiplicity correction.\n\nAircraft: proven FP32-equivalent fits.\nAxes use task-specific scales.',va='top',fontsize=11,linespacing=1.5)
    fig.suptitle('What helps after FP32 magnitudes?',fontsize=16)
    fig.savefig(output/'next_change_tuned.png',dpi=160)
    fig.savefig(output/'next_change_tuned.svg',metadata={'Date':None})
    plt.close(fig)
    write(output / 'summary.json', payload)
    write(output / 'comparisons.json', payload['comparisons'])
    (output / 'scope.md').write_text(payload['scope'] + '\n')
    (output / 'protocol.md').write_text(payload['protocol'])
    (output / 'reproduce.md').write_text('Verify raw_manifest.json, concatenate only its archive_parts in listed order when split, and extract with Python tarfile filter="data" into an empty directory. From that directory run:\n\n```bash\nPYTHONPATH=source python -m experiments.next_report --rebuild report_inputs.json --output rebuilt\n```\n\nThis verifies retained evidence hashes, recomputes all per-example summaries and paired intervals, and regenerates tables/figures. Checkpoint tensor payloads and pretrained caches are omitted; local task audits verify checkpoints. Source, configurations, training logs, predictions and audit receipts are retained. NumPy and Matplotlib are required.\n')


def collect(raw, repo):
    from experiments import next_ablations as ab
    evidence = prior.Evidence(raw, repo)
    gate = evidence.json(Path('/var/tmp/dora-bench/next_ablation_validation.json'))
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    assert gate['backend_source_sha256'] == gate['core_source_sha256'] == ab.SOURCE_SHA256
    assert gate['inherited_core_source_sha256'] == ab.core.SOURCE_SHA256
    assert gate['standalone_source_sha256'] == ab.PUBLIC_SOURCE_SHA256 == sha(repo / 'dora.py')
    for name, digest in gate['source_sha256'].items():
        assert sha(repo / name) == digest
        evidence.add(repo / name)
    prior.bound_references(evidence, gate)
    recipes = {method: ab.recipe_manifest(method) for method in METHODS}
    assert gate['recipes'] == recipes
    tasks = {}
    tasks.update(prior.retrieval(evidence, recipes, 'scifact', 'scifact'))
    tasks.update(prior.retrieval(evidence, recipes, 'nfcorpus', 'nfcorpus'))
    tasks['aircraft'] = aircraft(evidence, recipes)
    tasks['extraction'] = extraction(evidence, recipes)
    tasks = {key: tasks[key] for key in TASKS}
    speed = microbenchmark(evidence, recipes)
    for path in raw.rglob('*'):
        if path.is_file() and path.suffix in prior.SUFFIXES and path.suffix not in {'.png','.jpg'}:
            evidence.add(path)
    for name in ('experiments/next_report.py','experiments/change_report.py','experiments/check_next_report.py','experiments/next_protocol.md','experiments/next_ablations.py','experiments/next_aircraft_validate.py','experiments/change_aircraft_analysis.py','dora.py'):
        evidence.add(repo / name)
    payload = {'complete': True, 'baseline': 'Original implementation plus FP32 magnitude storage, FP64 retained',
               'recipes': recipes, 'tasks': tasks, 'gate': gate, 'microbenchmark': speed,
               'source_sha256': {name: sha(repo/name) for name in ('experiments/next_report.py','experiments/change_report.py','dora.py')},
               'protocol': (repo / 'experiments/next_protocol.md').read_text(),
               'scope': 'Exploratory next-step ablations using previously inspected task test sets. Each treatment adds exactly one declared recipe flag to original DoRA plus FP32 magnitude storage. Models/data/splits/rank/seeds/budgets remain matched. Test metrics never select learning rates or checkpoints. Fresh retrieval and ViGGO baseline/treatments; Aircraft reuses corresponding existing FP32 fits only after full-model bitwise parity and fresh artifact audits. Means and sample SD use seeds42-44; paired intervals use seed/example resampling, only three training seeds, 10,000 draws and no multiplicity correction. Initializer includes orientation/distribution/RNG policy; factorization includes floating-point association/projection precision. Task adapters retain shared pretrained-model compatibility handling. Training timings include shared norm diagnostics and task-specific validation scope. No universal ranking or independent confirmation is implied.'}
    return payload, evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=Path('/var/tmp/dora-bench/next_bench'))
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, default=Path('results/2026-10-09-next-change'))
    p.add_argument('--rebuild', type=Path)
    args=p.parse_args()
    if args.rebuild:
        payload=read(args.rebuild)
        assert sha(Path(__file__)) == payload['source_sha256']['experiments/next_report.py']
        assert sha(Path(prior.__file__)) == payload['source_sha256']['experiments/change_report.py']
        for name,digest in payload['source_sha256'].items():
            assert sha(args.rebuild.parent / 'source' / name)==digest
        for item in payload['evidence']:
            path=args.rebuild.parent / item['path']
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],path
        expected=copy.deepcopy(payload['comparisons'])
        compute(payload); assert payload['comparisons']==expected
        render(payload,args.output)
        write(args.output/'rebuild_audit.json',{'status':'pass','evidence_files_verified':len(payload['evidence']),'input_sha256':sha(args.rebuild)})
    else:
        payload,evidence=collect(args.raw_root.resolve(),args.repo.resolve())
        compute(payload)
        args.output.mkdir(parents=True,exist_ok=True)
        prior.bundle(payload,evidence,args.output)
        render(payload,args.output)
    print(json.dumps({'status':'complete','tasks':len(payload['tasks']),'recipes':len(METHODS),'output':str(args.output)}))


if __name__=='__main__': main()
