"""Audited four-cell DoRA/NoRA report with portable prediction and interval replay."""
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

METHODS = ['current_dora', 'current_dora_nora', 'init_dora', 'init_dora_nora']
LABELS = dict(zip(METHODS, ['Accepted DoRA', 'Accepted DoRA + NoRA', 'Alternate initializer', 'Alternate initializer + NoRA']))
LANES, TASKS = prior.LANES, prior.TASKS
CONTRASTS = [
    ('literal_nora', 'current_dora_nora', 'current_dora', 'NoRA: original initializer'),
    ('initialized_nora', 'init_dora_nora', 'init_dora', 'NoRA: alternate initializer'),
    ('initializer', 'init_dora', 'current_dora', 'Initializer alone'),
    ('combined', 'init_dora_nora', 'current_dora', 'Initializer + NoRA vs accepted'),
]
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
    from experiments.audit_nora_aircraft import Audit
    control = evidence.json(evidence.raw / 'aircraft_control/task_parity.json')
    assert control['passed'] and len(control['cases']) == 2
    assert {r['name'] for r in control['cases']} == {'accepted_default', 'initializer_control'}
    assert all(r['passed'] for r in control['cases'])
    reference = Path('/var/tmp/dora-bench/round2/vision/rank2_run')
    data = {'lanes': prior.empty_lanes(METHODS), 'audits': {}, 'manifests': {}}
    for method in METHODS:
        root = evidence.raw / 'aircraft' / method
        audit = prior.require_audit(evidence, root)
        checker = Audit(root, reference)
        arrays, checked = checker.method(method)
        assert checked['method'] == method
        protocol = evidence.json(root / 'protocol.json')
        assert protocol['recipe'] == recipes[method]
        assert checker.task == control
        data['audits'][method] = audit
        data['manifests'][method] = protocol
        ids = [row['image'] for row in checker.manifest['splits']['test']]
        if 'example_ids' not in data:
            data.update(example_ids=ids, strata=checker.labels.tolist(), split_manifest=checker.manifest,
                        manifest=protocol)
        assert data['example_ids'] == ids and data['split_manifest'] == checker.manifest
        for lane in LANES:
            folder = root if lane == 'validation_tuned' else root / 'fixed_lr_0.001'
            rows = evidence.json(folder / 'results.json')
            for row in rows:
                if row['method'] != method:
                    continue
                directory = folder / 'rank_2/final' / method / f"seed_{row['seed']}"
                fit_root = root if row['adapter_learning_rate'] == checked['selected_lr'] else folder
                # A selected .001 fit is copied to the fixed lane byte-for-byte.
                # Both records refer to its single actual training fit.
                fit = fit_root / 'rank_2/final' / method / f"seed_{row['seed']}"
                evidence.add(directory / 'result.json')
                evidence.add(directory / 'test_predictions.json')
                prior.add_record(data['lanes'][lane][method], {
                    'seed': row['seed'], 'value': row['test']['macro_class_accuracy'] * 100,
                    'learning_rate': row['adapter_learning_rate'], 'training_seconds': row['training_wall_seconds'],
                    'peak_memory_bytes': row['peak_cuda_allocated_bytes'], 'trainable_parameters': row['trainable_parameters'],
                    'fit_id': str(fit.relative_to(evidence.raw)), 'selected_epoch': row['best_epoch'],
                    'artifact_result_sha256': sha(directory / 'result.json'),
                    'recipe_sha256': recipes[method]['recipe_sha256'], 'fresh_fit': True,
                }, arrays[lane, row['seed']].tolist())
        for which, bindings in checker.bindings.items():
            for name, digest in bindings.items():
                path = checker.roots[which] / name
                assert sha(path) == digest
                evidence.add(path)
        for item in audit['details']['historical_control_parity']:
            for name, digest in item['reference_files_sha256'].items():
                path = Path(item['reference']) / name
                assert sha(path) == digest
                evidence.add(path)
    return prior.order_cells(data)


def compute(payload):
    comparisons = []
    assert set(payload['tasks']) == set(TASKS)
    for task, data in payload['tasks'].items():
        assert set(data['lanes']) == set(LANES)
        for lane, cells in data['lanes'].items():
            assert set(cells) == set(METHODS)
            for method, cell in cells.items():
                assert len(cell['scores']) == 3
                assert all(len(row) == len(data['example_ids']) for row in cell['scores'])
                cell['summary'] = prior.summarize_cell(cell, data['strata'])
            for kind, candidate, reference, label in CONTRASTS:
                result = prior.contrast(cells[candidate]['scores'], cells[reference]['scores'], data['strata'])
                prior.close(result['difference'], cells[candidate]['summary']['mean'] - cells[reference]['summary']['mean'], 2e-5)
                comparisons.append({'task': task, 'lane': lane, 'kind': kind, 'candidate': candidate,
                                    'reference': reference, 'label': label, **result})
    payload['comparisons'] = comparisons


def render(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['svg.hashsalt'] = 'accepted-dora-nora-20261009'
    table = prior.markdown_table
    prior.protocol_table(payload, output)
    for lane in LANES:
        rows = [[LABELS[m], *[prior.fmt(payload['tasks'][t]['lanes'][lane][m]['summary']) for t in TASKS]] for m in METHODS]
        (output / (lane + '_table.md')).write_text(table(['Recipe', *[v[0] + ' / ' + v[1] for v in TASKS.values()]], rows)
            + '\nMean ± sample SD over seeds 42–44. All four recipes freshly trained. Transfer evaluates NFCorpus fits.\n')
    rows = [[TASKS[r['task']][0], r['lane'], r['label'], f"{r['difference']:+.3f}",
             f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"] for r in payload['comparisons']]
    (output / 'paired_effects.md').write_text(table(['Task','LR policy','Contrast','Difference in points','Exploratory paired 95% interval'], rows)
        + '\nThree training seeds; 10,000 crossed paired seed/example draws, stratified by class for Aircraft. No multiple-comparison correction.\n')
    rows = []
    for task, data in payload['tasks'].items():
        if task == 'scifact_transfer':
            continue
        for lane, cells in data['lanes'].items():
            for method in METHODS:
                s = cells[method]['summary']
                rows.append([TASKS[task][0], lane, LABELS[method], ', '.join(f'{v:g}' for v in s['learning_rates']),
                             prior.fmt(s['training_seconds']), f"{max(s['peak_memory_bytes']['values'])/2**30:.3f}",
                             ', '.join(map(str,s['parameters']))])
    (output / 'runtime_memory.md').write_text(table(['Task','LR policy','Recipe','LRs','Instrumented seconds ± SD','Peak GiB','Parameters'], rows)
        + '\nFresh measurements include norm diagnostics and task-specific validation work. NoRA additionally tracks factor epsilon activity. Fits shared between LR policies count once. These measurements do not establish production throughput.\n')
    short = ['NoRA / original init', 'NoRA / alternate init', 'Initializer alone', 'Initializer + NoRA']
    coordinates = []
    for lane in LANES:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), layout='constrained')
        for ax, task in zip(axes.flat, TASKS):
            selected = [next(r for r in payload['comparisons'] if r['task']==task and r['lane']==lane and r['kind']==kind)
                        for kind, *_ in CONTRASTS]
            values = np.array([r['difference'] for r in selected])
            lower = np.array([r['ci95'][0] for r in selected]); upper = np.array([r['ci95'][1] for r in selected])
            y = np.arange(len(CONTRASTS))
            ax.barh(y, values, color=['#26836b' if v>=0 else '#c56d54' for v in values], alpha=.85)
            ax.errorbar(values,y,xerr=np.array([np.maximum(values-lower,0),np.maximum(upper-values,0)]),
                        fmt='none',ecolor='#283442',capsize=3)
            ax.axvline(0,color='#283442',lw=.8)
            ax.set_yticks(y,short); ax.invert_yaxis(); ax.grid(axis='x',alpha=.18); ax.set_axisbelow(True)
            ax.set_title(TASKS[task][0],fontsize=11); ax.set_xlabel('Change in score (points)',fontsize=10)
            coordinates.extend({'task': task, 'lane': lane, 'kind': row['kind'], 'y': int(i),
                                'value': row['difference'], 'ci95': row['ci95']} for i,row in enumerate(selected))
        axes[1,2].axis('off')
        axes[1,2].text(.04,.92,'Accepted DoRA:\noriginal + FP32 magnitudes\n\nFirst two bars isolate NoRA\nwithin each initializer. Last two\ncompare to accepted DoRA.\n\nThree seeds; fresh training.\nBars: mean paired difference.\nWhiskers: exploratory\n95% paired interval.\nNo multiplicity correction.\nTask-specific scales.',va='top',fontsize=10.5,linespacing=1.35)
        fig.suptitle('Adding NoRA to accepted DoRA — ' + ('equal validation tuning' if lane=='validation_tuned' else 'fixed baseline learning rates'),fontsize=15)
        stem = 'nora_tuned' if lane=='validation_tuned' else 'nora_fixed'
        fig.savefig(output / (stem+'.png'),dpi=160)
        fig.savefig(output / (stem+'.svg'),metadata={'Date':None})
        plt.close(fig)
    write(output/'chart_coordinates.json',coordinates)
    fig, axes = plt.subplots(2,3,figsize=(12.8,7.6),layout='constrained')
    score_coordinates=[]
    score_labels=['Accepted DoRA','+ NoRA / original init','Alternate init','Alternate init + NoRA']
    colors=['#336ba3','#c56d54','#9b8b67','#26836b']
    for ax,task in zip(axes.flat,TASKS):
        cells=payload['tasks'][task]['lanes']['validation_tuned']
        means=[cells[m]['summary']['mean'] for m in METHODS]
        deviations=[cells[m]['summary']['sample_sd'] for m in METHODS]
        y=np.arange(len(METHODS))
        ax.barh(y,means,color=colors,alpha=.9)
        ax.errorbar(means,y,xerr=deviations,fmt='none',ecolor='#283442',capsize=3)
        ax.set_yticks(y,score_labels);ax.invert_yaxis()
        ax.set_xlim(0,max(means[i]+deviations[i] for i in range(4))*1.22)
        for i,value in enumerate(means):
            ax.text(value+deviations[i]+.01*ax.get_xlim()[1],i,f'{value:.2f}',va='center',fontsize=9)
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.set_title(TASKS[task][0],fontsize=11);ax.set_xlabel(TASKS[task][1],fontsize=10)
        score_coordinates.extend({'task':task,'method':m,'y':i,'mean':means[i],'sample_sd':deviations[i]}
                                 for i,m in enumerate(METHODS))
    axes[1,2].axis('off')
    axes[1,2].text(.04,.92,'Accepted DoRA:\noriginal + FP32 magnitudes\n\nSame models, data and budgets.\nEqual validation-only LR search.\nSeeds 42, 43 and 44; fresh fits.\n\nBars: mean score; higher is better.\nWhiskers: sample SD.\nPreviously inspected test sets:\nexploratory comparison.\nPaired intervals in the report.',
                   va='top',fontsize=10.5,linespacing=1.35)
    fig.suptitle('Accepted DoRA with NoRA — initialization matters',fontsize=16)
    fig.savefig(output/'nora_scores.png',dpi=160)
    fig.savefig(output/'nora_scores.svg',metadata={'Date':None})
    plt.close(fig)
    write(output/'score_chart_coordinates.json',score_coordinates)
    write(output/'summary.json',payload); write(output/'comparisons.json',payload['comparisons'])
    (output/'scope.md').write_text(payload['scope']+'\n')
    (output/'protocol.md').write_text(payload['protocol'])
    (output/'reproduce.md').write_text('Verify raw_manifest.json, concatenate only its archive_parts in listed order when split, and extract with Python tarfile filter="data" into an empty directory. From that directory run:\n\n```bash\nPYTHONPATH=source python -m experiments.nora_current_report --rebuild report_inputs.json --output rebuilt\n```\n\nThis verifies retained evidence hashes, recomputes all per-example summaries and paired intervals, and regenerates tables/figures. Checkpoint tensor payloads and pretrained caches are omitted; local task audits verify checkpoints. Source, configurations, logs, predictions and audit receipts are retained. NumPy and Matplotlib are required.\n')


def collect(raw, repo):
    from experiments import nora_current as ab
    evidence = prior.Evidence(raw,repo)
    gate = evidence.json(raw/'backend_validation/nora_current_validation.json')
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    assert gate['backend_source_sha256'] == ab.SOURCE_SHA256
    assert gate['standalone_source_sha256'] == sha(repo/'dora.py')
    for name,digest in gate['source_sha256'].items():
        assert sha(repo/name)==digest
        evidence.add(repo/name)
    prior.bound_references(evidence,gate)
    recipes = {m:ab.recipe_manifest(m) for m in METHODS}
    assert gate['recipes']==recipes
    diagnostic=evidence.json(raw/'nora_first_step_diagnostic.json')
    diagnostic_review=evidence.json(raw/'orchestration/nora_first_step_independent_review.json')
    assert diagnostic_review['status']=='pass'
    assert sha(raw/'nora_first_step_diagnostic.json')==diagnostic_review['diagnostic_result_sha256']
    assert sha(raw/'orchestration/diagnose_nora_first_step.py')==diagnostic_review['diagnostic_source_sha256']==diagnostic['diagnostic_source_sha256']
    assert sha(raw/'orchestration/review_nora_first_step.py')==diagnostic_review['review_source_sha256']
    assert diagnostic['source_backend_sha256']==diagnostic_review['backend_source_sha256']==ab.SOURCE_SHA256
    assert {m:row['source_recipe'] for m,row in diagnostic['rows'].items()}==recipes
    tasks = {}
    tasks.update(prior.retrieval(evidence,recipes,'scifact','scifact'))
    tasks.update(prior.retrieval(evidence,recipes,'nfcorpus','nfcorpus'))
    tasks['aircraft'] = aircraft(evidence,recipes)
    tasks['extraction'] = extraction(evidence,recipes)
    tasks = {key:tasks[key] for key in TASKS}
    integrated = evidence.json(raw/'integrated_tests.json')
    assert integrated['status']=='pass' and integrated['test_methods']==77
    assert sha(raw/'integrated_tests.log')==integrated['log_sha256']
    for name,digest in integrated['source_sha256'].items():
        assert sha(repo/name)==digest
        evidence.add(repo/name)
    commands = {}
    for name in ('retrieval_commands.json','extraction_commands.json'):
        value = evidence.json(raw/name)
        rows = list(value.values()) if isinstance(value,dict) else value
        assert rows and all(row.get('exit_code')==0 for row in rows)
        for row in rows:
            source = row.get('dispatcher_source',row.get('dispatcher_source_path'))
            assert sha(raw/source)==row['dispatcher_sha256']
            if 'log_sha256' in row:
                assert sha(row['log'])==row['log_sha256']
            evidence.add(Path(row['log']))
        commands[name] = value
    aircraft_commands = sorted(raw.glob('aircraft_commands_gpu*.json'))
    assert aircraft_commands
    observed = []
    for path in aircraft_commands:
        rows = evidence.json(path)
        for row in rows:
            assert row['exit_code']==0 and sha(row['log'])==row['sha256']
            assert sha(raw/'run_aircraft.py')==row['queue_source_sha256']
            assert sha(repo/'experiments/nora_aircraft.py')==row['runner_source_sha256']
            observed.append(row['command'][row['command'].index('--method')+1])
        commands[path.name] = rows
    assert sorted(observed)==sorted(METHODS)
    fit_counts = {task:tasks[task]['audit']['details']['audit']['unique_trained_fits'] for task in ('scifact','nfcorpus')}
    extraction_audit = tasks['extraction']['audit']['details']
    fit_counts['extraction'] = extraction_audit['pilots']+extraction_audit['unique_final_fits']
    fit_counts['aircraft'] = sum(10-tasks['aircraft']['audits'][m]['details']['audit']['fixed_fits_reused'] for m in METHODS)
    for path in raw.rglob('*'):
        if path.is_file() and path.suffix in prior.SUFFIXES and path.suffix not in {'.png','.jpg'}:
            evidence.add(path)
    names = ('experiments/nora_current_report.py','experiments/change_report.py','experiments/check_nora_current_report.py',
             'experiments/nora_current_protocol.md','experiments/nora_current.py','experiments/nora_aircraft_validate.py',
             'experiments/audit_nora_aircraft.py','dora.py')
    for name in names:
        evidence.add(repo/name)
    sources = ('experiments/nora_current_report.py','experiments/change_report.py','dora.py')
    payload = {'complete':True,'baseline':'Accepted original DoRA plus FP32 magnitude storage, FP64 retained',
               'recipes':recipes,'tasks':tasks,'gate':gate,'integrated_tests':integrated,
               'commands':commands,'fresh_training_fits':fit_counts,'fresh_training_fits_total':sum(fit_counts.values()),
               'initialization_diagnostic':diagnostic,'initialization_diagnostic_review':diagnostic_review,
               'source_sha256':{name:sha(repo/name) for name in sources},
               'protocol':(repo/'experiments/nora_current_protocol.md').read_text(),
               'scope':'Exploratory two-by-two NoRA and initializer study on previously inspected test sets. All four recipes freshly trained with matched pinned models/data/splits/rank/seeds/budgets. All LR and checkpoint choices use validation only. Both fixed-baseline-LR and equally tuned lanes are reported. NoRA uses differentiable down-factor column normalization across rank, epsilon 1e-12; the literal original zero-down initializer enters epsilon and is explicitly distinct from the alternate Kaiming-down/zero-up treatment. Full DoRA column-norm gradients and FP32 magnitude storage are retained. Alternate initialization includes orientation/distribution/RNG and registration policy. Shared task adapters preserve pretrained storage/device, absent-bias and projection-dtype compatibility. Means and sample SD use seeds42-44; intervals use 10,000 crossed paired seed/example draws, stratified by Aircraft class, with no multiplicity correction. No universal ranking or independent confirmation is implied. Training timings include norm diagnostics and task-specific validation work. The public dora.py remains unchanged.'}
    return payload,evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=Path('/var/tmp/dora-bench/nora_current_bench'))
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,default=Path('results/2026-10-09-current-dora-nora'))
    p.add_argument('--rebuild',type=Path)
    args=p.parse_args()
    if args.rebuild:
        payload=read(args.rebuild)
        assert sha(Path(__file__))==payload['source_sha256']['experiments/nora_current_report.py']
        assert sha(Path(prior.__file__))==payload['source_sha256']['experiments/change_report.py']
        for name,digest in payload['source_sha256'].items():
            assert sha(args.rebuild.parent/'source'/name)==digest
        for item in payload['evidence']:
            path=args.rebuild.parent/item['path']
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


if __name__=='__main__':main()

