"""Separate reporting for the source-reviewed LoRA-Null/GME extension."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments import init_report as primary
from experiments import init_extension_registry as registry

FAMILIES=('accepted_anchor','accepted_gaussian_scale','lora_null_task_tail',
          'gme_paper','gme_source_v1','gme_source_v2')


def initialization_evidence(row,task):
    """Retain complete source/probe/fallback metadata, deduplicating model-wide GME."""
    result=row.get('result',row)
    diagnostics=(row.get('initial_diagnostics',{}) if task in ('scifact','nfcorpus') else
                 row.get('initialization',{}).get('diagnostics',result.get('norm_diagnostics',{})))
    if isinstance(diagnostics,list):
        diagnostics={value['name']:value for value in diagnostics}
    global_gme={};modules={}
    for name,value in (diagnostics or {}).items():
        initial=value.get('initializer',{});metadata=initial.get('callback_metadata')
        if not metadata:continue
        record={'initializer_kind':initial.get('initializer_kind'),'factor_seed':initial.get('factor_seed'),
                'minimum_raw_norm':value.get('minimum_raw_norm'),'clamp_activations':value.get('clamp_activations')}
        if 'observations' in metadata or 'gme' in metadata:
            key=primary.digest(metadata);global_gme[key]=metadata;record['gme_metadata_sha256']=key
        else:record['callback_metadata']=metadata
        modules[name]=record
    return {'modules':modules,'model_gme_metadata':global_gme,
            'initializer_seconds':row.get('initialization',{}).get('initializer_seconds'),
            'timing_scope':'Source computation seconds are cold costs; cache-hit wall time is separate. Retrieval fit timestamps begin after injection.'}


def read_fit_record(evidence,root,task,path):
    row=evidence.read(path)
    if task=='extraction' and row['status']=='pass':
        directory=Path(root)/row['artifact_directory']
        row={**row,'result':evidence.read(directory/'result.json',row['result_sha256']),
             'initialization':evidence.read(directory/'initialization.json',row['initialization_sha256'])}
    return row


def validate_registry(value,fixture=False):
    assert value['schema']=='dora-init-extension-v1' and value['study_lane']=='extension'
    assert value['seeds']==[42,43,44] and value['seed']==42
    assert value['test_used_for_candidate_definition'] is False
    families={f['family_id']:f for f in value['family_definitions']}
    if not fixture:
        assert set(families)==set(FAMILIES)
        settings=registry.TASKS[value['task']]
        for key,expected in settings.items():assert value[key]==expected
        assert len(value['candidates'])==21*len(value['learning_rates'])
    for family in families.values():
        expected='warmstart' if family['kind']=='extension_warmstart' else 'centered'
        assert family['initialization_lane']==expected
        assert family['kind'] in ('random','extension_centered','extension_warmstart')
    for candidate in value['candidates']:
        family=families[candidate['family_id']]
        assert candidate['initialization_lane']==family['initialization_lane']
        assert candidate['spec']['kind']==family['kind']
        assert candidate['spec']['config']==family['configs'][candidate['initializer_index']]
    actual={(c['family_id'],c['initializer_index'],c['learning_rate']) for c in value['candidates']}
    expected={(family['family_id'],index,lr) for family in families.values()
              for index in range(len(family['configs'])) for lr in value['learning_rates']}
    assert actual==expected and len(actual)==len(value['candidates'])
    assert value['candidate_counts']==dict(Counter(c['family_id'] for c in value['candidates']))
    assert value['parent_candidate_counts']==dict(Counter(c['parent_family'] for c in value['candidates']))
    return families


def collect_root(evidence,root,task,fixture=False):
    manifest=evidence.read(Path(root)/'manifest.json')
    definitions=validate_registry(manifest['registry'],fixture)
    # Primary collection verifies final audits, all source/input/checkpoint hashes,
    # validation-only selection and retained predictions. Its fixture switch only
    # skips main-grid family cardinality; extension cardinality is checked above.
    data,candidates=primary.collect_root(evidence,root,'extension',task,fixture=True)
    for candidate in candidates:
        raw=read_fit_record(evidence,root,task,candidate['pilot_result_path'])
        candidate['extension_initialization']=initialization_evidence(raw,task)
        if task in ('scifact','nfcorpus'):
            candidate['pilot_timing']['initializer_seconds']=raw.get('initialization',{}).get('initializer_seconds')
    for value in data.values():
        assert value['family_definitions']==definitions
        value['initial_control_observations']={}
        for name in ('control.json','initial_probes.json'):
            probes=Path(root)/'controls'/name
            if probes.exists():value['initial_control_observations'][name]=evidence.read(probes)
        for fit in value['fits'].values():
            file='receipt.json' if task=='extraction' else 'fit.json' if task=='aircraft' else 'result.json'
            raw=read_fit_record(evidence,root,task,Path(fit['fit_id'])/file)
            fit['extension_initialization']=initialization_evidence(raw,task)
            if task in ('scifact','nfcorpus'):
                fit['initializer_seconds']=raw.get('initialization',{}).get('initializer_seconds')
        for cell in value['cells'].values():
            cell['initialization_lane']=definitions[cell['family_id']]['initialization_lane']
            if cell['candidate_id'] is not None:
                candidate=next(c for c in manifest['registry']['candidates'] if c['candidate_id']==cell['candidate_id'])
                cell['same_lr_anchor_keys']=[f"{candidate['comparison_control_id']}/seed_{seed}" for seed in primary.SEEDS]
    return data,candidates


def compute(payload,replicates=10000):
    assert all(value['study_lane']=='extension' for value in payload['tasks'].values())
    result=primary.compute(payload,replicates)
    for data in payload['tasks'].values():
        for cell in data['cells'].values():
            if 'same_lr_anchor_keys' in cell:
                keys=cell['same_lr_anchor_keys']
                paired={'study_lane':'extension','dataset':data['dataset'],'family_id':cell['family_id'],
                    'lane':cell['lane'],'comparison':'same_lr_accepted_anchor','reference_study_lane':'extension',
                    'reference_family_id':'accepted_anchor','reference_lane':'paired_same_learning_rate',
                    'candidate_id':cell['candidate_id'],'reference_candidate_id':keys[0].split('/seed_')[0]}
                if cell['status']!='pass' or any(k not in data['fits'] or data['fits'][k]['status']!='pass' for k in keys):
                    paired['status']='unavailable_failed_or_missing_fit'
                else:
                    assert all(data['fits'][a]['learning_rate']==data['fits'][b]['learning_rate']
                               for a,b in zip(cell['fit_keys'],keys))
                    paired.update(status='pass',**primary.contrast(primary.scores_for(data,cell),
                        [data['fits'][k]['scores'] for k in keys],data['strata'],replicates))
                result['comparisons'].append(paired)
            reference=data['cells'].get('accepted_gaussian_scale/'+cell['lane'])
            if reference is None:continue
            row={'study_lane':'extension','dataset':data['dataset'],'family_id':cell['family_id'],
                 'lane':cell['lane'],'initialization_lane':cell['initialization_lane'],
                 'comparison':'equally_tuned_gaussian_scale','reference_study_lane':'extension',
                 'reference_family_id':'accepted_gaussian_scale','reference_lane':reference['lane'],
                 'candidate_id':cell['candidate_id'],'reference_candidate_id':reference['candidate_id']}
            if cell['status']!='pass' or reference['status']!='pass':
                row.update(status='unavailable_failed_or_missing_fit',candidate_status=cell['status'],
                           reference_status=reference['status'])
            else:
                row.update(status='pass',**primary.contrast(primary.scores_for(data,cell),
                    primary.scores_for(data,reference),data['strata'],replicates))
            result['comparisons'].append(row)
    for collection in ('all_family_results','comparisons'):
        for row in result[collection]:
            data=payload['tasks']['extension/'+row['dataset']]
            row['initialization_lane']=data['cells'][row['family_id']+'/'+row['lane']]['initialization_lane']
    return result


def render(payload,result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    primary.write(output/'summary.json',result)
    primary.csv_write(output/'all_families.csv',result['all_family_results'])
    primary.csv_write(output/'all_comparisons.csv',result['comparisons'])
    primary.write(output/'all_candidates.json',payload['all_candidates'])
    primary.csv_write(output/'all_candidates.csv',payload['all_candidates'])
    budgets=primary.search_budgets(payload)
    primary.write(output/'search_budgets.json',budgets);primary.csv_write(output/'search_budgets.csv',budgets)
    fits=[{'task':key,'fit_key':fit_key,**{k:v for k,v in fit.items() if k!='scores'}}
          for key,data in payload['tasks'].items() for fit_key,fit in data['fits'].items()]
    primary.write(output/'all_fits.json',fits);primary.csv_write(output/'all_fits.csv',fits)
    primary.write(output/'failures.json',{'failed_final_fits':[fit for fit in fits if fit['status']!='pass'],
        'failed_pilots':[row for row in payload['all_candidates'] if row['pilot']['status']!='pass']})
    primary.write(output/'initial_controls.json',{name:data.get('initial_control_observations')
                  for name,data in payload['tasks'].items()})
    primary.write(output/'report_inputs.json',payload);primary.write(output/'evidence_manifest.json',payload['evidence'])
    title='Extension report fixture — engineering evidence only' if payload['fixture'] else 'Separate DoRA initialization extension'
    lines=[f'# {title}','',
        'This extension is separate from the main initialization grid and fresh-seed confirmation. '
        'All registered families and failures are retained. Selection uses validation only, with fixed-rate and joint '
        'initializer/LR selection shown separately. Means and sample SD require all three training seeds 42–44.', '',
        'LoRA-Null task-tail is centered with accepted column DoRA and task-only source-normalized input Grams. '
        'It does not establish knowledge preservation. GME paper, source V1 and source V2 are distinct uncompensated '
        'warm starts, with original magnitude and different source response/sampling rules. Finite factors alone '
        'do not demonstrate successful moment equalization. Gap-Init is not tested on these unimodal checkpoints.', '',
        'There are four settings per extension child family, twelve across the GME parent, plus one accepted anchor '
        'and four equally tuned Gaussian controls. Complete parent/child search budgets are in search_budgets.csv. '
        'Initial quality, probe/fallback observations, cold/cache costs and failed cells are retained in the candidate/fit JSONs.', '']
    cells={(r['dataset'],r['family_id'],r['lane']):r for r in result['all_family_results']}
    comparisons={(r['dataset'],r['family_id'],r['comparison']):r for r in result['comparisons'] if r['lane']==primary.LANES[1]}
    for data in payload['tasks'].values():
        lines += [f"## {data['dataset']}",'',f"Final audit: `{data['audit']['sha256']}`.",'']
        for initialization_lane in ('centered','warmstart'):
            lines += [f'### {initialization_lane}','',
                '| Family | Fixed rate mean ± SD | Joint tuning mean ± SD | Joint Δ vs anchor [95% CI] | Joint Δ vs tuned Gaussian [95% CI] |',
                '|---|---:|---:|---:|---:|']
            for family in FAMILIES:
                definition=data['family_definitions'].get(family)
                if definition is None or definition['initialization_lane']!=initialization_lane:continue
                def score(lane):
                    row=cells[data['dataset'],family,lane]
                    return f"{row['mean']:.2f} ± {row['sample_sd']:.2f}" if row['status']=='pass' else row['status']
                def effect(kind):
                    row=comparisons.get((data['dataset'],family,kind))
                    if row is None:return 'unavailable'
                    if row['status']!='pass':return row['status']
                    return f"{row['difference']:+.2f} [{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]"
                lines.append(f'| {family} | {score(primary.LANES[0])} | {score(primary.LANES[1])} | '
                             f'{effect("accepted_anchor_same_lane")} | {effect("equally_tuned_gaussian_scale")} |')
            lines+=['']
    lines += ['Scores/differences are percentage points; retrieval nDCG is multiplied by100. Exploratory intervals '
        'use10000 crossed paired seed/example resamples with Aircraft class stratification and no multiplicity correction. '
        'NFCorpus transfer has no independent fit. Failed seeds are never removed to form a mean. '
        'Retrieval receipts that start after injection are not full end-to-end timings. '
        'No extension result changes the public DoRA implementation or ranks methods using a test-selected display list.','']
    (output/'report.md').write_text('\n'.join(lines))
    outputs={p.name:primary.sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='report_manifest.json'}
    primary.write(output/'report_manifest.json',{'schema':'dora-init-extension-report-manifest-v1',
        'fixture':payload['fixture'],'outputs':outputs})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path)
    parser.add_argument('--run',action='append',default=[],help='task=/absolute/task/root')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--rebuild',type=Path)
    args=parser.parse_args()
    if args.rebuild:
        manifest=primary.json.loads((args.rebuild.parent/'report_manifest.json').read_text())
        assert primary.sha(args.rebuild)==manifest['outputs'][args.rebuild.name]
        payload=primary.json.loads(args.rebuild.read_text())
        assert payload['schema']=='dora-init-extension-report-v1'
        for path,binding in payload['evidence'].items():assert primary.sha(path)==binding['sha256'],path
    else:
        roots={}
        if args.study_root:
            roots={task:args.study_root/'runs'/task for task in primary.TASKS
                   if (args.study_root/'runs'/task).exists()}
        for entry in args.run:
            task,path=entry.split('=',1);assert task in primary.TASKS and task not in roots
            roots[task]=Path(path)
        assert roots
        if not args.fixture:assert set(roots)==set(primary.TASKS)
        evidence=primary.Evidence();evidence.add(__file__);evidence.add(primary.__file__);evidence.add(registry.__file__)
        import experiments.change_report as bootstrap
        evidence.add(bootstrap.__file__)
        payload={'schema':'dora-init-extension-report-v1','fixture':args.fixture,
                 'bootstrap_replicates':10000,'tasks':{},'all_candidates':[]}
        for task,root in sorted(roots.items()):
            data,candidates=collect_root(evidence,root,task,args.fixture)
            payload['tasks'].update(data);payload['all_candidates'].extend(candidates)
        payload['evidence']=evidence.files
    result=compute(payload,payload['bootstrap_replicates']);render(payload,result,args.output)
    print({'status':'pass','fixture':payload['fixture'],'cells':len(result['all_family_results']),'output':str(args.output)},flush=True)


if __name__=='__main__':main()
