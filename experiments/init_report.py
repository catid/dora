"""Auditable reporting of the frozen DoRA initialization registries.

Collection requires successful final task audits. No table or plot chooses
families by test performance. A rebuild consumes retained normalized per-example
scores and verifies the evidence manifest without running model inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.change_report import contrast, metric, ranked_metrics

SEEDS = [42, 43, 44]
LANES = ['fixed_original_lr', 'validation_tuned']
TASKS = ['scifact', 'nfcorpus', 'aircraft', 'extraction']
# These are chosen from source definitions, before inspecting main test scores.
REPRESENTATIVES = ['accepted_anchor', 'accepted_gaussian_scale', 'normal_up',
                   'lora_kaiming_uniform_down', 'pissa', 'milora', 'olora', 'eva',
                   'corda_ipm', 'astra_tail', 'loraga_ArB2r', 'loraone_centered',
                   'filet', 'lorada_centered', 'cntlora_cross_centered',
                   'inilora_prefit', 'tlora_init_only', 'mica_init_only']
EXPECTED_FAMILIES = {'centered': 76, 'warmstart': 6, 'quantized': 9}
ALLOWED_KEYS = {'dialogue_act', 'name', 'release_year', 'developer', 'esrb', 'rating',
                'genres', 'player_perspective', 'has_multiplayer', 'platforms',
                'available_on_steam', 'has_linux_release', 'has_mac_release', 'exp_release_date', 'specifier'}
ALLOWED_ACTS = {'inform', 'confirm', 'recommend', 'request', 'give_opinion',
                'verify_attribute', 'suggest', 'request_attribute', 'request_explanation'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def close(actual, expected, tolerance=2e-6):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), (actual, expected)


def summary(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values) if len(values)>1 else None,
            'values': values, 'n': len(values)} if values else None


class Evidence:
    def __init__(self): self.files = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        name = str(path)
        if name not in self.files:
            assert path.is_file(), path
            self.files[name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        actual = self.files[name]['sha256']
        assert expected is None or expected == actual, (path, expected, actual)
        return actual

    def read(self, path, expected=None):
        self.add(path, expected)
        return json.loads(Path(path).read_text())

    def lines(self, path, expected=None):
        self.add(path, expected)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def audit_root(evidence, root, manifest):
    audit = evidence.read(root/'task_audit.json')
    assert audit['status'] == 'pass' and audit['phase'] == 'finals', root
    assert audit.get('manifest_sha256', sha(root/'manifest.json')) == sha(root/'manifest.json')
    for mapping_name in ['input_sha256', 'checkpoint_sha256']:
        for name, expected in audit.get(mapping_name, {}).items():
            mapping = expected if isinstance(expected, dict) else {name: expected}
            base = Path(audit.get('artifact_roots', {}).get(name, root)) if isinstance(expected, dict) else root
            for relative, value in mapping.items():
                path = Path(relative) if Path(relative).is_absolute() else base/relative
                evidence.add(path, value)
    for relative, value in manifest['source_sha256'].items(): evidence.add(root/'source'/relative, value)
    for relative, value in manifest.get('input_sha256', {}).items(): evidence.add(root/relative, value)
    return {'path': str(root/'task_audit.json'), 'sha256': evidence.add(root/'task_audit.json'),
            'scope': audit.get('scope'), 'checkpoint_hashes_verified': len(audit.get('checkpoint_sha256', {}))}


def selections(registry, selected, task):
    assert selected['test_used_for_selection'] is False
    assert selected['registry_sha256'] == digest(registry)
    pilots = {p['candidate_id']: p for p in selected['pilots']}
    candidates = {c['candidate_id']: c for c in registry['candidates']}
    assert len(pilots) == len(selected['pilots']) == len(candidates)
    assert set(pilots) == set(candidates)
    assert set(selected['selections']) == {c['family_id'] for c in candidates.values()}
    for family, choices in selected['selections'].items():
        assert set(choices) == set(LANES)
        for lane in LANES:
            eligible = [c for c in candidates.values() if c['family_id'] == family
                        and pilots[c['candidate_id']]['status'] == 'pass'
                        and (lane != LANES[0] or c['learning_rate'] == registry['fixed_lr'])]
            def key(c):
                pilot = pilots[c['candidate_id']]
                value = pilot['selection_value']; assert math.isfinite(value)
                secondary = pilot.get('selection_cross_entropy', pilot.get('selection_secondary', 0.))
                return (value if task == 'extraction' else -value, secondary, c['learning_rate'], c['candidate_id'])
            expected = min(eligible, key=key)['candidate_id'] if eligible else None
            assert choices[lane] == expected, (family, lane, expected, choices[lane])
    return candidates, pilots


def elapsed(row):
    from datetime import datetime
    if row.get('started_utc') and row.get('finished_utc'):
        return (datetime.fromisoformat(row['finished_utc'])-datetime.fromisoformat(row['started_utc'])).total_seconds()
    return None


def timing(row, task):
    result = row.get('result', row)
    if task in ('scifact', 'nfcorpus'):
        train = row.get('training', {})
        return {'training_seconds': train.get('seconds'), 'peak_memory_bytes': train.get('peak_allocated_bytes'),
                'initializer_seconds': None, 'total_fit_wall_seconds': elapsed(row),
                'trainable_parameters': row.get('trainable_parameters')}
    return {'training_seconds': result.get('training_wall_seconds' if task == 'aircraft' else 'train_and_validation_seconds'),
            'peak_memory_bytes': result.get('peak_cuda_allocated_bytes' if task == 'aircraft' else 'peak_allocated_bytes'),
            'initializer_seconds': row.get('initialization',{}).get('initializer_seconds'),
            'total_fit_wall_seconds': elapsed(row), 'trainable_parameters': result.get('trainable_parameters')}


def initialization_observation(row, task):
    result=row.get('result',row)
    diagnostics=(row.get('initial_diagnostics',{}) if task in ('scifact','nfcorpus') else
                 row.get('initialization',{}).get('diagnostics',result.get('norm_diagnostics',{})))
    values=diagnostics.values() if isinstance(diagnostics,dict) else diagnostics
    costs=[]
    for value in values:
        initial=value.get('initializer',{})
        meta=initial.get('callback_metadata',initial)
        fields={key:meta[key] for key in ('elapsed_seconds','factor_cache_hit','prefit_cache_hit',
                'prefit_source_seconds','cnt_target_cache_hit','cnt_target_source_seconds','decomposition_events') if key in meta}
        if fields:costs.append({'module_path':initial.get('module_path',value.get('name')),**fields})
    return {'initial_quality':row.get('initial_evaluation',result.get('initial_validation')),
            'module_preprocessing':costs,
            'timing_scope':'Factor-cache hits may retain historical source computation seconds; these are not new wall time. Full injection timer is separate.'}


def extraction_scores(rows, gold):
    assert [r['id'] for r in rows] == [g['id'] for g in gold]
    scores=[];valid=schema=tp=predicted=truth=0
    for row, target in zip(rows, gold):
        assert row['gold'] == target['target'] and row['utterance'] == target['utterance']
        try: parsed=json.loads(row['text'].strip()); valid+=1
        except (ValueError, TypeError): parsed=None
        assert parsed == row['parsed']
        good=(isinstance(parsed, dict) and set(parsed).issubset(ALLOWED_KEYS)
              and all(isinstance(v,str) for v in parsed.values()) and parsed.get('dialogue_act') in ALLOWED_ACTS)
        schema+=int(good); prediction=parsed if good else {}
        correct=prediction==target['target']; assert correct == row['exact_match'];scores.append(float(correct))
        tp+=len(set(prediction.items())&set(target['target'].items()));predicted+=len(prediction);truth+=len(target['target'])
    return scores, {'count':len(rows),'valid_json':valid/len(rows),'schema_valid':schema/len(rows),
                    'exact_match':statistics.mean(scores),'slot_micro_f1':2*tp/max(1,predicted+truth),
                    'correct_slots':tp,'predicted_slots':predicted,'gold_slots':truth}


def rescore(evidence, root, task, dataset, row, split, gold):
    if task in ('scifact','nfcorpus'):
        records=evidence.read(root/row['fit_id']/dataset/'test_per_query.json')
        ids=split['test']; mapping={r['query_id']:r for r in records}
        assert len(mapping)==len(records)==len(ids) and set(mapping)==set(ids)
        scores=[];per_metric={k:[] for k in ('ndcg_at_10','mrr_at_10','recall_at_10')}
        for qid in ids:
            item=mapping[qid];computed=ranked_metrics(item['top_10'],gold[qid])
            assert len(item['scores'])==10 and all(math.isfinite(s) for s in item['scores'])
            assert item['scores']==sorted(item['scores'],reverse=True)
            for key,value in computed.items():close(value,item[key],1e-12);per_metric[key].append(value)
            scores.append(computed['ndcg_at_10'])
        metrics={key:statistics.mean(values) for key,values in per_metric.items()}
        for key,value in metrics.items():close(value,row['evaluation'][dataset]['test'][key],1e-12)
        return ids,None,scores,metrics
    if task=='aircraft':
        pred=evidence.read(root/row['artifact_directory']/'test_predictions.json')
        labels=np.array([r['label'] for r in split['splits']['test']]);guesses=np.array(pred['predictions'])
        assert np.array_equal(labels,pred['labels']) and guesses.shape==labels.shape
        assert ((0<=guesses)&(guesses<len(split['classes']))).all()
        scores=(labels==guesses).astype(float)
        nll=np.array(pred['true_class_nll']);top5=np.array(pred['top5_correct'],dtype=float)
        assert nll.shape==top5.shape==labels.shape and np.isfinite(nll).all() and (nll>=0).all()
        assert np.isin(top5,[0,1]).all() and (top5>=scores).all()
        metrics={'count':len(labels),'correct':int(scores.sum()),'accuracy':float(scores.mean()),
                 'macro_class_accuracy':float(metric([scores]*3,labels)[0]),
                 'cross_entropy':float(nll.mean()),'top5_accuracy':float(top5.mean())}
        for key,value in metrics.items():close(value,row['result']['test'][key],3e-7)
        return [r['image'] for r in split['splits']['test']],labels.tolist(),scores.tolist(),metrics
    directory=root/row['artifact_directory']
    rows=evidence.lines(directory/'predictions.jsonl',row['predictions_sha256'])
    scores,metrics=extraction_scores(rows,split['test'])
    for key,value in metrics.items():close(value,row['result']['test'][key],1e-12)
    return [r['id'] for r in split['test']],None,scores,metrics


def collect_root(evidence, root, lane, task, fixture=False):
    root=Path(root).resolve();manifest=evidence.read(root/'manifest.json');registry=manifest['registry']
    assert registry['task']==task and registry.get('study_lane','centered')==lane
    assert registry['seeds']==SEEDS
    audit=audit_root(evidence,root,manifest)
    calibration=[]
    for path in sorted((root/'calibration').glob('**/manifest.json')):
        saved=evidence.read(path)
        calibration.append({'path':str(path),'sha256':evidence.add(path),'metadata':saved.get('metadata',{}),
                            'module_sha256':saved.get('module_sha256',{})})
    selected=evidence.read(root/'selections.json')
    assert selected['manifest_sha256']==sha(root/'manifest.json')
    candidates,pilots=selections(registry,selected,task)
    definitions={f['family_id']:f for f in registry.get('family_definitions',[])}
    if not fixture:
        assert len(definitions)==EXPECTED_FAMILIES[lane]
        assert set(definitions)==set(selected['selections'])
    inventory=[]
    for cid,candidate in candidates.items():
        if task=='extraction':path=root/'pilots'/cid/'receipt.json'
        elif task=='aircraft':path=root/'fits'/cid/'pilot/seed_42/fit.json'
        else:path=root/'fits'/cid/'seed_42/pilot.json'
        row=evidence.read(path);assert row['candidate_id']==cid and row['seed']==42
        result=row
        if task=='extraction' and row['status']=='pass':
            result={**row,'result':evidence.read(root/row['artifact_directory']/'result.json',row['result_sha256']),
                    'initialization':evidence.read(root/row['artifact_directory']/'initialization.json',row['initialization_sha256'])}
        inventory.append({'study_lane':lane,'task':task,**candidate,'pilot':pilots[cid],
                          'pilot_result_path':str(path),'pilot_timing':timing(result,task),
                          'initialization':initialization_observation(result,task),
                          'selected_lanes':[name for name,choice in selected['selections'][candidate['family_id']].items() if choice==cid],
                          'failure':row.get('failure',row.get('error'))})
    if task=='extraction':
        runs=[];paired=[]
        for family,choices in selected['selections'].items():
            record=evidence.read(root/'families'/f'{family}.json')
            assert record['manifest_sha256']==sha(root/'manifest.json')
            assert record['selections_sha256']==sha(root/'selections.json') and record['selections']==choices
            for name,cid in choices.items():
                if cid is None:continue
                for seed in SEEDS:
                    key=f'{cid}/seed_{seed}';row=record['fits'][key]
                    saved=evidence.read(root/'finals'/key/'receipt.json');assert row==saved
                    runs.append({**row,'candidate_id':cid,'family_id':family,'lane':name,'fit_id':f'finals/{key}'})
            for key,row in record.get('paired_control_fits',{}).items():
                assert row==evidence.read(root/'finals'/key/'receipt.json')
                paired.append({**row,'fit_id':f'finals/{key}'})
    else:
        raw=evidence.read(root/'results.json');assert raw['complete']
        assert raw['manifest_sha256']==sha(root/'manifest.json') and raw['selections_sha256']==sha(root/'selections.json')
        runs=raw['runs'];paired=raw.get('paired_control_fits',[])
    expected={(family,name,seed) for family,choices in selected['selections'].items()
              for name,cid in choices.items() if cid is not None for seed in SEEDS}
    assert len(runs)==len(expected) and {(r['family_id'],r['lane'],r['seed']) for r in runs}==expected
    datasets=['nfcorpus','scifact'] if task=='nfcorpus' else [task]
    output={}
    for dataset in datasets:
        split=evidence.read(root/(dataset+'_splits.json' if task in ('scifact','nfcorpus') else 'split_manifest.json'))
        gold=evidence.read(root/(dataset+'_qrels.json')) if task in ('scifact','nfcorpus') else None
        name='scifact_transfer' if task=='nfcorpus' and dataset=='scifact' else dataset
        data={'study_lane':lane,'training_task':task,'dataset':name,'root':str(root),'audit':audit,
              'registry_sha256':digest(registry),'manifest_sha256':sha(root/'manifest.json'),
              'protocol_sha256':manifest['protocol_sha256'],'configuration':manifest['configuration'],
              'model':manifest.get('model',manifest.get('provenance',{}).get('model')),
              'dataset_provenance':manifest.get('dataset',manifest.get('datasets',manifest.get('provenance',{}))),
              'source_sha256':manifest['source_sha256'],'calibration':calibration,
              'candidate_counts':registry.get('candidate_counts',{}),
              'parent_candidate_counts':registry.get('parent_candidate_counts',{}),
              'fixed_lr':registry.get('fixed_lr'),
              'family_definitions':definitions,'selections':selected['selections'],'cells':{},'fits':{}}
        for family,choices in selected['selections'].items():
            for lr_lane,cid in choices.items():
                data['cells'][family+'/'+lr_lane]={'family_id':family,'lane':lr_lane,'candidate_id':cid,
                    'status':'all_pilots_failed' if cid is None else 'pending','fit_keys':[]}
        for row in runs+paired:
            cid,seed=row['candidate_id'],row['seed'];candidate=candidates[cid]
            fit_key=f'{cid}/seed_{seed}'
            if fit_key not in data['fits']:
                if task=='extraction':
                    if row['status']=='pass':
                        row={**row,'result':evidence.read(root/row['artifact_directory']/'result.json',row['result_sha256']),
                             'initialization':evidence.read(root/row['artifact_directory']/'initialization.json',row['initialization_sha256'])}
                else:
                    path=root/row['fit_id']/('fit.json' if task=='aircraft' else 'result.json')
                    saved=evidence.read(path);assert all(row[k]==v for k,v in saved.items())
                record={'candidate_id':cid,'seed':seed,'spec':candidate['spec'],'learning_rate':candidate['learning_rate'],
                        'fit_id':str(root/row['fit_id']),'status':row['status'],**timing(row,task),
                        'failure':row.get('failure',row.get('error')),
                        'recipe_sha256':row.get('recipe_sha256',row.get('recipe',{}).get('recipe_sha256'))}
                record['initialization']=initialization_observation(row,task)
                if row['status']=='pass':
                    ids,strata,scores,metrics=rescore(evidence,root,task,dataset,row,split,gold)
                    if 'example_ids' in data:assert data['example_ids']==ids and data['strata']==strata
                    else:data.update(example_ids=ids,strata=strata)
                    record.update(scores=scores,metrics=metrics)
                    if task=='extraction':frozen=row['initialization'].get('frozen_sha256',row['result']['frozen_sha256'])
                    elif task=='aircraft':frozen=row['result']['frozen_state_before_sha256']
                    else:frozen=row['frozen_sha256']
                    record['frozen_sha256']=frozen
                data['fits'][fit_key]=record
            if 'lane' in row:
                cell=data['cells'][row['family_id']+'/'+row['lane']]
                assert cell['candidate_id']==cid
                assert fit_key not in cell['fit_keys'];cell['fit_keys'].append(fit_key)
        for cell in data['cells'].values():
            if cell['candidate_id'] is None:continue
            cell['fit_keys'].sort(key=lambda key:data['fits'][key]['seed'])
            assert [data['fits'][key]['seed'] for key in cell['fit_keys']]==SEEDS
            cell['status']='pass' if all(data['fits'][key]['status']=='pass' for key in cell['fit_keys']) else 'final_seed_failure'
            candidate=candidates[cell['candidate_id']]
            if candidate.get('spec',{}).get('config',{}).get('method')=='loftq':
                control=candidate['comparison_control_id'];assert candidates[control]['learning_rate']==candidate['learning_rate']
                cell['paired_control_keys']=[f'{control}/seed_{seed}' for seed in SEEDS]
                assert all(key in data['fits'] for key in cell['paired_control_keys'])
                for left,right in zip(cell['fit_keys'],cell['paired_control_keys']):
                    a,b=data['fits'][left],data['fits'][right]
                    if a['status']==b['status']=='pass':
                        assert a['frozen_sha256'] and a['frozen_sha256']==b['frozen_sha256']
        output[lane+'/'+name]=data
    return output,inventory


def scores_for(data,cell):
    return [data['fits'][key]['scores'] for key in cell['fit_keys']]


def factor_config(spec):
    config=dict(spec.get('config',spec));config.pop('magnitude_policy',None)
    return config


def calibration_signature(data):
    return sorted(digest({'seed':c['metadata'].get('classifier_seed',c['metadata'].get('calibration_seed')),
                          'module_sha256':c['module_sha256']}) for c in data['calibration'])


def compute(payload,replicates=10000):
    tables=[];comparisons=[]
    cache={}
    def compare(data,cell,reference_data,reference_cell,kind):
        identity={'study_lane':data['study_lane'],'dataset':data['dataset'],'family_id':cell['family_id'],
                  'lane':cell['lane'],'comparison':kind,'reference_study_lane':reference_data['study_lane'],
                  'reference_family_id':reference_cell['family_id'],'reference_lane':reference_cell['lane'],
                  'candidate_id':cell['candidate_id'],'reference_candidate_id':reference_cell['candidate_id']}
        if cell['status']!='pass' or reference_cell['status']!='pass':
            comparisons.append({**identity,'status':'unavailable_failed_or_missing_fit',
                                'candidate_status':cell['status'],'reference_status':reference_cell['status']})
            return
        assert data['example_ids']==reference_data['example_ids'] and data['strata']==reference_data['strata']
        key=(tuple(data['fits'][k]['fit_id'] for k in cell['fit_keys']),tuple(reference_data['fits'][k]['fit_id'] for k in reference_cell['fit_keys']),data['dataset'])
        if key not in cache:cache[key]=contrast(scores_for(data,cell),scores_for(reference_data,reference_cell),data['strata'],replicates)
        comparisons.append({**identity,'status':'pass',**cache[key]})
    for data in payload['tasks'].values():
        center=payload['tasks'].get('centered/'+data['dataset'])
        for cell in data['cells'].values():
            row={'study_lane':data['study_lane'],'dataset':data['dataset'],'family_id':cell['family_id'],
                 'lane':cell['lane'],'status':cell['status'],'candidate_id':cell['candidate_id']}
            if cell['status']=='pass':
                fits=[data['fits'][key] for key in cell['fit_keys']]
                row.update(summary((metric(scores_for(data,cell),data['strata'])*100).tolist()))
                row.update(n_examples=len(data['example_ids']),learning_rate=fits[0]['learning_rate'],spec=fits[0]['spec'],
                           fit_ids=[fit['fit_id'] for fit in fits])
                for field in ['training_seconds','initializer_seconds','peak_memory_bytes','total_fit_wall_seconds','trainable_parameters']:
                    values=[fit[field] for fit in fits]
                    row[field]=summary(values) if all(v is not None for v in values) else None
            else:row['failed_seeds']=[data['fits'][key]['seed'] for key in cell['fit_keys'] if data['fits'][key]['status']!='pass']
            tables.append(row)
            anchor_family=next((f for f in ('accepted_anchor','accepted') if f+'/'+cell['lane'] in data['cells']),None)
            if anchor_family:compare(data,cell,data,data['cells'][anchor_family+'/'+cell['lane']],'accepted_anchor_same_lane')
            if center and 'accepted_gaussian_scale/'+cell['lane'] in center['cells']:
                compare(data,cell,center,center['cells']['accepted_gaussian_scale/'+cell['lane']],'equally_tuned_gaussian_scale')
            if cell.get('paired_control_keys'):
                keys=cell['paired_control_keys'];fits=[data['fits'][key] for key in keys]
                reference={'family_id':'paired_nf4_control','lane':cell['lane'],'candidate_id':fits[0]['candidate_id'],
                           'fit_keys':keys,'status':'pass' if all(f['status']=='pass' for f in fits) else 'failed'}
                compare(data,cell,data,reference,'paired_same_base_same_lr_quantizer')
            if cell['lane']==LANES[1]:compare(data,cell,data,data['cells'][cell['family_id']+'/'+LANES[0]],'tuned_minus_fixed')
            if data['study_lane']=='warmstart' and cell['status']=='pass' and center:
                representative=data['fits'][cell['fit_keys'][0]]
                calibration_matches=calibration_signature(data)==calibration_signature(center)
                matches=[c for c in center['cells'].values() if calibration_matches and c['status']=='pass' and c['lane']==cell['lane']
                         and factor_config(center['fits'][c['fit_keys'][0]]['spec'])==factor_config(representative['spec'])
                         and center['fits'][c['fit_keys'][0]]['learning_rate']==representative['learning_rate']]
                for other in matches:compare(data,cell,center,other,'matched_config_lr_warm_minus_centered')
                if not matches:
                    comparisons.append({'study_lane':data['study_lane'],'dataset':data['dataset'],'family_id':cell['family_id'],
                                        'lane':cell['lane'],'comparison':'matched_config_lr_warm_minus_centered',
                                        'status':'unavailable_no_selected_config_lr_calibration_match'})
    return {'all_family_results':tables,'comparisons':comparisons,'replicates':replicates}


def csv_write(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def search_budgets(payload):
    """Count actual pilot opportunities once per training task, including failures."""
    tasks={(d['study_lane'],d['training_task']):d for d in payload['tasks'].values()}
    output=[]
    for (lane,task),data in sorted(tasks.items()):
        candidates=[c for c in payload['all_candidates'] if c['study_lane']==lane and c['task']==task]
        for scope,field,stored in [('family','family_id','candidate_counts'),
                                   ('parent','parent_family','parent_candidate_counts')]:
            groups={}
            for candidate in candidates:
                key=candidate.get(field,candidate['family_id'])
                groups.setdefault(key,[]).append(candidate)
            counts={name:len(rows) for name,rows in groups.items()}
            if data.get(stored):assert counts==data[stored],(lane,task,stored)
            for name,rows in sorted(groups.items()):
                output.append({'study_lane':lane,'training_task':task,'scope':scope,'name':name,
                    'family_count':len({r['family_id'] for r in rows}),
                    'initializer_count':len({(r['family_id'],digest(r['spec'])) for r in rows}),
                    'learning_rate_count':len({r['learning_rate'] for r in rows}),
                    'learning_rates':sorted({r['learning_rate'] for r in rows}),
                    'fixed_lr_candidate_count':sum(r['learning_rate']==data.get('fixed_lr') for r in rows),
                    'joint_candidate_count':len(rows),
                    'passed_pilots':sum(r['pilot']['status']=='pass' for r in rows),
                    'failed_pilots':sum(r['pilot']['status']!='pass' for r in rows)})
    return output


def render(payload,result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write(output/'summary.json',result)
    write(output/'all_candidates.json',payload['all_candidates'])
    csv_write(output/'all_families.csv',result['all_family_results'])
    csv_write(output/'all_comparisons.csv',result['comparisons'])
    csv_write(output/'all_candidates.csv',payload['all_candidates'])
    budgets=search_budgets(payload)
    write(output/'search_budgets.json',budgets)
    csv_write(output/'search_budgets.csv',budgets)
    fit_rows=[{'task':key,'fit_key':fit_key,**{k:v for k,v in fit.items() if k!='scores'}}
              for key,data in payload['tasks'].items() for fit_key,fit in data['fits'].items()]
    csv_write(output/'all_fits.csv',fit_rows)
    write(output/'all_fits.json',fit_rows)
    failures=[{'task':key,'fit_key':fit_key,**fit} for key,data in payload['tasks'].items()
              for fit_key,fit in data['fits'].items() if fit['status']!='pass']
    write(output/'failures.json',{'failed_final_fits':failures,
          'failed_pilots':[row for row in payload['all_candidates'] if row['pilot']['status']!='pass']})
    write(output/'report_inputs.json',payload)
    write(output/'evidence_manifest.json',payload['evidence'])
    title='Initialization report fixture — implementation validation only' if payload['fixture'] else 'DoRA initialization study'
    lines=[f'# {title}','',
           'All registered families are retained in the CSV and JSON outputs, including failed or unavailable cells. '
           'Means and sample standard deviations use seeds 42, 43, and 44. The fixed-rate lane still tunes initializer settings; '
           'the joint lane tunes initializer and learning rate using validation only. Three-seed results require every seed; '
           'failed seeds are never silently removed. Test examples are rescored from retained predictions.', '',
           'Differences are percentage points (nDCG ×100, Aircraft macro accuracy, or ViGGO JSON exact match). '
           'Exploratory 95% intervals use 10,000 crossed paired seed/example bootstrap draws, with class stratification and '
           'equal class weighting for Aircraft. They have no multiplicity correction. SciFact transfer uses NFCorpus-trained '
           'fits. Dense dequantized LoftQ timings do not measure compressed serving.', '',
           'The displayed families were declared in source before main-result inspection. They cover 14 literature recipe '
           'parents plus four random anchors; no test-score ranking selects the display. All warm-start and quantized '
           'families are displayed separately. Literature rows retain the frozen registry interpretation and adaptation boundaries.', '']
    index={(r['study_lane'],r['dataset'],r['family_id'],r['lane']):r for r in result['all_family_results']}
    comparisons={(r['study_lane'],r['dataset'],r['family_id'],r['lane'],r['comparison']):r for r in result['comparisons']}
    for task_key,data in payload['tasks'].items():
        lane,dataset=data['study_lane'],data['dataset']
        families=REPRESENTATIVES if lane=='centered' and not payload['fixture'] else list(data['selections'])
        lines += [f'## {task_key}','',f"Audit: `{data['audit']['sha256']}`. Test examples: {len(data.get('example_ids',[]))}.",'',
                  '| Family | Fixed rate mean ± SD | Joint validation tuning mean ± SD |','|---|---:|---:|']
        plotted=[]
        for family in families:
            cells=[index.get((lane,dataset,family,name)) for name in LANES]
            def fmt(r):return f"{r['mean']:.2f} ± {r['sample_sd']:.2f}" if r and r['status']=='pass' else (r['status'] if r else 'absent')
            lines.append(f'| {family} | {fmt(cells[0])} | {fmt(cells[1])} |')
            plotted.append((family,cells))
        lines += ['']
        lines += ['| Joint-tuned family | Δ vs accepted [95% CI] | Δ vs scale-tuned Gaussian [95% CI] |',
                  '|---|---:|---:|']
        for family in families:
            def effect(kind):
                item=comparisons.get((lane,dataset,family,LANES[1],kind))
                if not item:return 'unavailable'
                if item.get('status')!='pass':return item['status']
                return f"{item['difference']:+.2f} [{item['ci95'][0]:+.2f}, {item['ci95'][1]:+.2f}]"
            lines.append(f'| {family} | {effect("accepted_anchor_same_lane")} | {effect("equally_tuned_gaussian_scale")} |')
        lines += ['']
        if lane=='quantized':
            lines += ['| Joint-tuned LoftQ | Δ vs identical-base, same-LR NF4 control [95% CI] |','|---|---:|']
            for family in families:
                item=comparisons.get((lane,dataset,family,LANES[1],'paired_same_base_same_lr_quantizer'))
                if item:lines.append(f'| {family} | {effect("paired_same_base_same_lr_quantizer")} |')
            lines += ['']
        valid=[(family,cells) for family,cells in plotted if any(r and r['status']=='pass' for r in cells)]
        if valid:
            fig,ax=plt.subplots(figsize=(10,max(3,len(valid)*.34)))
            for offset,name,color in [(-.15,LANES[0],'#4477aa'),(.15,LANES[1],'#cc6677')]:
                index_lane=LANES.index(name)
                for i,(_,cells) in enumerate(valid):
                    r=cells[index_lane]
                    if r and r['status']=='pass':ax.errorbar(r['mean'],i+offset,xerr=r['sample_sd'],fmt='o',color=color,label=name if i==0 else None,capsize=2)
            ax.set_yticks(range(len(valid)),[v[0] for v in valid]);ax.invert_yaxis();ax.set_xlabel('Mean test score ×100; bars: sample SD across three seeds')
            ax.set_title(task_key);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
            name=task_key.replace('/','_')+'.png';fig.savefig(output/name,dpi=160);plt.close(fig)
            lines += [f'![{task_key}]({name})','']
    lines += ['## Interpretation boundaries','',
              'Centered methods use W + (BA − B₀A₀). Warm starts retain the nonzero initial update. '
              'Full EVA rank redistribution, TLoRA rank/scale redistribution and frozen down factors, MiCA frozen up factors, '
              'and LoRA-SB/LoRA-Pro structural or optimizer changes are outside this initializer-only comparison. '
              'The frozen registry carries the exact solver, covariance, sign, precision, and source-default adaptations.', '',
              'Training timers cover instrumented task training and its validation. `total_fit_wall_seconds`, when present, '
              'is the saved receipt interval: retrieval starts after injection, while Aircraft starts before injection. '
              '`initializer_seconds` separately records complete adapter injection for Aircraft/ViGGO when available. '
              'Missing timings remain null. Cached preprocessing and calibration '
              'costs are not inferred from training time. Identical fixed/joint fits share fit IDs and are not extra replicates.', '',
              'Rebuild with `python -m experiments.init_report --rebuild report_inputs.json --output rebuilt`. '
              'All referenced source, prediction, audit, and checkpoint hashes must still match; model inference is unnecessary.', '']
    (output/'report.md').write_text('\n'.join(lines))
    write(output/'report_manifest.json',{'schema':'dora-init-report-artifacts-v1','fixture':payload['fixture'],
          'source_sha256':sha(__file__),'outputs':{path.name:sha(path) for path in output.iterdir()
          if path.is_file() and path.name!='report_manifest.json'}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',default=[],help='STUDY_LANE:TASK=/absolute/task/root')
    parser.add_argument('--study-root',action='append',type=Path,default=[],help='Discover runs/{centered,warmstart,quantized}/{task}')
    parser.add_argument('--fixture',action='store_true',help='Allow smaller engineering registries and incomplete task coverage; label every artifact fixture')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rebuild',type=Path)
    args=parser.parse_args()
    if args.rebuild:
        receipt=json.loads((args.rebuild.parent/'report_manifest.json').read_text())
        assert sha(args.rebuild)==receipt['outputs'][args.rebuild.name],args.rebuild
        payload=json.loads(args.rebuild.read_text())
        for name,expected in payload['evidence'].items():
            assert sha(name)==expected['sha256'],name
    else:
        roots={}
        for study in args.study_root:
            for lane in EXPECTED_FAMILIES:
                for task in TASKS:
                    root=study/'runs'/lane/task
                    if root.exists():roots[(lane,task)]=root
        for entry in args.run:
            key,value=entry.split('=',1);lane,task=key.split(':');assert lane in EXPECTED_FAMILIES and task in TASKS
            assert (lane,task) not in roots;roots[(lane,task)]=Path(value)
        assert roots,'No task roots supplied'
        if not args.fixture:
            for lane in {key[0] for key in roots}:assert {task for l,task in roots if l==lane}==set(TASKS),lane
        evidence=Evidence();evidence.add(__file__)
        import experiments.change_report as bootstrap_source
        evidence.add(bootstrap_source.__file__)
        payload={'schema':'dora-init-report-v1','fixture':args.fixture,'representatives':REPRESENTATIVES,
                 'bootstrap_replicates':10000,'tasks':{},'all_candidates':[]}
        for (lane,task),root in sorted(roots.items()):
            data,candidates=collect_root(evidence,root,lane,task,args.fixture)
            payload['tasks'].update(data);payload['all_candidates'].extend(candidates)
            print(json.dumps({'collected':lane+'/'+task,'families':len(next(iter(data.values()))['selections'])}),flush=True)
        payload['evidence']=evidence.files
    result=compute(payload,payload['bootstrap_replicates']);render(payload,result,args.output)
    print(json.dumps({'status':'pass','fixture':payload['fixture'],'family_cells':len(result['all_family_results']),
                      'comparisons':len(result['comparisons']),'output':str(args.output)}),flush=True)


if __name__=='__main__': main()
