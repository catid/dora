"""Separate five-fresh-seed confirmation report; never pools main-grid seeds."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments import init_report as report

SEEDS=[45,46,47,48,49]
ROLES=['validation_winner','accepted_anchor','accepted_gaussian_scale']


def metric(scores,strata=None):
    values=np.asarray(scores,dtype=np.float64)
    assert values.ndim==2 and values.shape[0]==5 and values.shape[1]>0 and np.isfinite(values).all()
    if strata is None:return values.mean(axis=1)
    labels=np.asarray(strata);assert labels.shape==(values.shape[1],)
    return np.stack([values[:,labels==label].mean(axis=1) for label in np.unique(labels)]).mean(axis=0)


def contrast(candidate,reference,strata=None,replicates=10000):
    candidate=np.asarray(candidate,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
    assert candidate.shape==reference.shape
    delta=candidate-reference;values=metric(delta,strata)*100
    labels=np.zeros(delta.shape[1],dtype=int) if strata is None else np.asarray(strata)
    groups=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    generator=np.random.default_rng(20261009);boot=np.zeros(replicates)
    for start in range(0,replicates,100):
        n=min(100,replicates-start);seeds=generator.integers(5,size=(n,5));samples=np.zeros(n)
        for indices in groups:
            examples=generator.choice(indices,size=(n,len(indices)))
            samples+=delta[seeds[:,:,None],examples[:,None,:]].mean(axis=(1,2))/len(groups)
        boot[start:start+n]=samples
    return {'difference':float(values.mean()),'paired_seed_differences':values.tolist(),
            'ci95':(np.quantile(boot,[.025,.975])*100).tolist(),'replicates':replicates,'random_seed':20261009,
            'resampling':'Crossed paired five fresh training seeds and examples; Aircraft preserves class counts and equal class weights.'}


def verify_registry(evidence,registry):
    assert registry['schema']=='dora-init-confirmation-v1' and registry['seeds']==SEEDS
    assert registry['study_lane']=='centered' and registry['test_used_for_selection'] is False
    assert set(registry['roles'])==set(ROLES)
    parent=Path(registry['source_study_root'])
    manifest=evidence.read(parent/'manifest.json',registry['parent_manifest_sha256'])
    assert report.digest(manifest['registry'])==registry['parent_registry_sha256']
    selected=evidence.read(parent/'selections.json',registry['parent_selections_sha256'])
    audit=evidence.read(parent/'pilots_audit.json',registry['parent_pilots_audit_sha256'])
    assert audit['status']=='pass' and audit['phase']=='pilots'
    audit_inputs=audit.get('input_sha256',{})
    for name,value in registry['pilot_evidence_sha256'].items():
        assert audit_inputs.get(name)==value,(name,'pilot audit binding absent or changed')
        evidence.add(parent/name,value)
    # Stage summary files may grow when final audits run. Only immutable parent
    # inputs used for selection are required to match the pilot-stage audit.
    for name,value in [('manifest.json',registry['parent_manifest_sha256']),
                       ('selections.json',registry['parent_selections_sha256'])]:
        assert audit_inputs[name]==value
    task=registry['task'];candidates,pilots=report.selections(manifest['registry'],selected,task)
    successful=[cid for cid,p in pilots.items() if p['status']=='pass']
    def key(cid):
        row=pilots[cid];primary=row['selection_value'];secondary=row.get('selection_cross_entropy',row.get('selection_secondary',0.))
        assert math.isfinite(primary) and math.isfinite(secondary)
        return (primary if task=='extraction' else -primary,secondary,candidates[cid]['learning_rate'],cid)
    assert registry['roles']['validation_winner']==min(successful,key=key)
    for role in ROLES[1:]:
        assert registry['roles'][role]==min((cid for cid in successful if candidates[cid]['family_id']==role),key=key)
    actual={c['candidate_id']:c for c in registry['candidates']}
    assert len(actual)==len(registry['candidates']) and set(actual)==set(registry['roles'].values())
    for cid,candidate in actual.items():assert candidate==candidates[cid]
    return parent,manifest,actual


def collect(evidence,root,task):
    root=Path(root).resolve();manifest=evidence.read(root/'manifest.json');registry=manifest['registry']
    assert registry['task']==task
    parent,parent_manifest,candidates=verify_registry(evidence,registry)
    audit=report.audit_root(evidence,root,manifest)
    if task=='extraction':
        assert evidence.read(root/'task_audit.json')['complete_final_coverage'] is True
        assert manifest['source_study_manifest_sha256']==registry['parent_manifest_sha256']
    else:assert manifest['parent_manifest_sha256']==registry['parent_manifest_sha256']
    for name,value in parent_manifest['source_sha256'].items():assert manifest['source_sha256'][name]==value
    rows={}
    for cid in candidates:
        for seed in SEEDS:
            if task=='extraction':
                fit_id=f'finals/{cid}/seed_{seed}';row=evidence.read(root/fit_id/'receipt.json')
                if row['status']=='pass':
                    row={**row,'result':evidence.read(root/row['artifact_directory']/'result.json',row['result_sha256']),
                         'initialization':evidence.read(root/row['artifact_directory']/'initialization.json',row['initialization_sha256'])}
                row={**row,'fit_id':fit_id}
            else:
                fit_id=f'fits/{cid}/seed_{seed}';row=evidence.read(root/fit_id/('fit.json' if task=='aircraft' else 'result.json'))
            assert row['candidate_id']==cid and row['seed']==seed and row['manifest_sha256']==report.sha(root/'manifest.json')
            rows[cid,seed]=row
    if task!='extraction':
        result=evidence.read(root/'results.json');assert result['complete']
        assert result['manifest_sha256']==report.sha(root/'manifest.json')
        assert len(result['runs'])==15 and {(r['role'],r['seed']) for r in result['runs']}=={(role,seed) for role in ROLES for seed in SEEDS}
        for row in result['runs']:
            assert row['candidate_id']==registry['roles'][row['role']]
            assert all(row[key]==value for key,value in rows[row['candidate_id'],row['seed']].items())
    datasets=['nfcorpus','scifact'] if task=='nfcorpus' else [task];output={}
    for dataset in datasets:
        split=evidence.read(root/(dataset+'_splits.json' if task in ('scifact','nfcorpus') else 'split_manifest.json'))
        gold=evidence.read(root/(dataset+'_qrels.json')) if task in ('scifact','nfcorpus') else None
        name='scifact_transfer' if task=='nfcorpus' and dataset=='scifact' else dataset
        data={'task':task,'dataset':name,'root':str(root),'audit':audit,'roles':registry['roles'],
              'candidates':candidates,'fits':{},'registry':registry,'configuration':manifest['configuration']}
        for (cid,seed),row in rows.items():
            key=f'{cid}/seed_{seed}';record={'candidate_id':cid,'seed':seed,'fit_id':str(root/row['fit_id']),
                'status':row['status'],'learning_rate':candidates[cid]['learning_rate'],'spec':candidates[cid]['spec'],
                'failure':row.get('failure',row.get('error')),**report.timing(row,task),
                'initialization':report.initialization_observation(row,task)}
            if row['status']=='pass':
                ids,strata,scores,metrics=report.rescore(evidence,root,task,dataset,row,split,gold)
                if 'example_ids' in data:assert data['example_ids']==ids and data['strata']==strata
                else:data.update(example_ids=ids,strata=strata)
                record.update(scores=scores,metrics=metrics)
            data['fits'][key]=record
        output[name]=data
    return output


def compute(payload):
    rows=[];comparisons=[]
    for dataset,data in payload['tasks'].items():
        values={}
        for role,cid in data['roles'].items():
            fits=[data['fits'][f'{cid}/seed_{seed}'] for seed in SEEDS]
            complete=all(f['status']=='pass' for f in fits)
            row={'dataset':dataset,'role':role,'candidate_id':cid,'spec':data['candidates'][cid]['spec'],
                 'learning_rate':data['candidates'][cid]['learning_rate'],'seeds':SEEDS,
                 'status':'pass' if complete else 'failed_confirmation_seed','fit_ids':[f['fit_id'] for f in fits],
                 'failed_seeds':[f['seed'] for f in fits if f['status']!='pass']}
            if complete:
                values[role]=[f['scores'] for f in fits]
                row.update(report.summary((metric(values[role],data['strata'])*100).tolist()))
                for field in ('training_seconds','initializer_seconds','total_fit_wall_seconds','peak_memory_bytes'):
                    observed=[f[field] for f in fits];row[field]=report.summary(observed) if all(v is not None for v in observed) else None
            rows.append(row)
        for reference in ROLES[1:]:
            item={'dataset':dataset,'candidate_role':ROLES[0],'reference_role':reference,
                  'candidate_id':data['roles'][ROLES[0]],'reference_id':data['roles'][reference]}
            if ROLES[0] in values and reference in values:
                item.update(status='pass',**contrast(values[ROLES[0]],values[reference],data['strata']))
            else:item['status']='unavailable_failed_confirmation_seed'
            comparisons.append(item)
    return {'seeds':SEEDS,'results':rows,'comparisons':comparisons,'bootstrap_replicates':10000}


def render(payload,result,output):
    output.mkdir(parents=True,exist_ok=True)
    report.write(output/'report_inputs.json',payload);report.write(output/'summary.json',result)
    report.write(output/'evidence_manifest.json',payload['evidence'])
    report.csv_write(output/'all_roles.csv',result['results']);report.csv_write(output/'all_comparisons.csv',result['comparisons'])
    fits=[{'dataset':name,'fit_key':key,**{k:v for k,v in fit.items() if k!='scores'}}
          for name,data in payload['tasks'].items() for key,fit in data['fits'].items()]
    report.write(output/'all_fits.json',fits);report.csv_write(output/'all_fits.csv',fits)
    lines=['# Fresh-seed initialization confirmation','',
           'Configurations were selected using the complete seed-42 validation grid. These results use only fresh training '
           'seeds 45–49 and are separate from the primary seeds 42–44. Roles sharing a candidate reuse its fit; they do not '
           'add independent replicates. No configuration is selected by confirmation test scores. A failed seed makes the '
           'five-seed mean and its paired contrast unavailable.', '',
           'Metrics are nDCG ×100, Aircraft macro accuracy percent, and ViGGO JSON exact match percent. '
           'Means use five seeds; SD is the sample SD. Exploratory 95% intervals use 10,000 crossed paired seed/example '
           'bootstrap draws, with class stratification/equal class weights for Aircraft and no multiplicity correction. '
           'These are fresh training seeds on previously used held-out examples, not a fresh test dataset.', '',
           '| Dataset | Role | Candidate | Mean ± SD |','|---|---|---|---:|']
    for row in result['results']:
        value=f"{row['mean']:.2f} ± {row['sample_sd']:.2f}" if row['status']=='pass' else row['status']
        lines.append(f"| {row['dataset']} | {row['role']} | {row['candidate_id']} | {value} |")
    lines+=['','| Dataset | Winner comparison | Difference [95% CI] |','|---|---|---:|']
    for row in result['comparisons']:
        value=f"{row['difference']:+.2f} [{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]" if row['status']=='pass' else row['status']
        lines.append(f"| {row['dataset']} | {row['reference_role']} | {value} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    report.write(output/'report_manifest.json',{'schema':'dora-confirm-report-artifacts-v1','source_sha256':report.sha(__file__),
          'outputs':{path.name:report.sha(path) for path in output.iterdir() if path.is_file() and path.name!='report_manifest.json'}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',default=[],help='TASK=/absolute/confirmation/task/root')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--rebuild',type=Path)
    args=parser.parse_args()
    if args.rebuild:
        manifest=json.loads((args.rebuild.parent/'report_manifest.json').read_text())
        assert report.sha(args.rebuild)==manifest['outputs'][args.rebuild.name]
        payload=json.loads(args.rebuild.read_text())
        for name,value in payload['evidence'].items():assert report.sha(name)==value['sha256'],name
    else:
        roots={}
        for entry in args.run:
            task,path=entry.split('=',1);assert task in report.TASKS and task not in roots;roots[task]=Path(path)
        assert set(roots)==set(report.TASKS),'All four training tasks required for confirmation report'
        evidence=report.Evidence();evidence.add(__file__);evidence.add(report.__file__)
        import experiments.change_report as metric_source
        evidence.add(metric_source.__file__)
        payload={'schema':'dora-init-confirm-report-v1','seeds':SEEDS,'tasks':{}}
        for task,root in roots.items():payload['tasks'].update(collect(evidence,root,task))
        payload['evidence']=evidence.files
    result=compute(payload);render(payload,result,args.output)
    print(json.dumps({'status':'pass','datasets':len(payload['tasks']),'role_cells':len(result['results']),'output':str(args.output)}))


if __name__=='__main__':main()
