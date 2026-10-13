"""Independent CPU audit and paired comparisons for Aircraft per-change runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SEEDS = (42,43,44)
OAT = ('ab_axis','ab_detach','ab_m_fp32','ab_norm_fp32','ab_init','ab_factorized')
CHAIN = ('ab_original','ab_axis','cum_axis_detach','cum_axis_detach_m_fp32',
         'cum_axis_detach_m_fp32_norm_fp32','cum_axis_detach_m_fp32_norm_fp32_init','ab_revised')
FLAGS = ('axis','detach_norm','magnitude_fp32','norm_fp32','published_init','factorized','clamp_norm')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    def __init__(self, root, reference, allow_missing=False):
        self.root, self.reference = root, reference
        self.roots = {'experiment':root,'reference':reference}
        self.bindings = {name:{} for name in self.roots}
        self.allow_missing = allow_missing
        self.counts = {'epochs_checked':0,'prediction_rows_scored':0,'checkpoint_payloads_verified':0,'checkpoint_payloads_omitted':0}
        self.matrix = self.read('experiment','matrix.json')
        self.manifest = self.read('reference','split_manifest.json')
        self.labels = np.array([row['label'] for row in self.manifest['splits']['test']])
        self.counts_by_class = np.bincount(self.labels,minlength=100)
        assert len(self.labels)==3333 and (self.counts_by_class>0).all()
        self.baselines = {row['seed']:row for row in self.read('reference','results.json') if row['method']=='baseline'}
        assert set(self.baselines)==set(SEEDS)
        self.core = self.read('experiment','core_parity_evidence.json')
        self.task = self.read('experiment','task_parity_evidence.json')
        assert self.core['status']=='pass' and self.core['cuda_tests']['status']=='pass'
        assert self.task['passed'] and all(row['passed'] for row in self.task['cases'])
        assert sha(root/'core_parity_evidence.json')==self.matrix['core_gate_sha256']
        assert sha(root/'task_parity_evidence.json')==self.matrix['task_gate_sha256']
        assert self.core['core_source_sha256']==self.task['source_sha256']['experiments/change_ablations.py']
        assert self.matrix['recipes']==self.core['recipes']
        assert self.matrix['seeds']==list(SEEDS) and self.matrix['rank']==2
        assert self.matrix['lr_grid']==[1e-5,1e-4,3e-4,1e-3] and self.matrix['fixed_original_lr']==.001
        self.bind_file('experiment','scheduler_source.py',self.matrix['scheduler_sha256'])
        if 'dispatcher_handoff' in self.matrix:
            handoff=self.matrix['dispatcher_handoff']
            self.bind_file('experiment',handoff['source_path'],handoff['source_sha256'])
            self.bind_file('experiment','dispatcher_history/v1_matrix.json',handoff['previous_matrix_sha256'])
            self.bind_file('experiment','dispatcher_history/v1_queue_status.json',handoff['previous_status_sha256'])

    def bind_file(self,which,relative,expected=None):
        relative=str(relative); path=self.roots[which]/relative
        digest=sha(path)
        if expected is not None:assert digest==expected,(relative,digest,expected)
        self.bindings[which][relative]=digest
        return path

    def read(self,which,relative,lines=False):
        path=self.bind_file(which,relative)
        return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())

    def checkpoint(self,relative,expected):
        path=self.root/relative
        if path.exists():
            assert sha(path)==expected
            self.counts['checkpoint_payloads_verified']+=1
        else:
            assert self.allow_missing,relative
            self.counts['checkpoint_payloads_omitted']+=1

    def score(self,row,relative):
        prediction=self.read('experiment',Path(relative)/'test_predictions.json')
        assert np.array_equal(prediction['labels'],self.labels)
        guesses=np.array(prediction['predictions'])
        assert guesses.shape==self.labels.shape and ((0<=guesses)&(guesses<100)).all()
        correct=(guesses==self.labels).astype(float)
        nll,top5=np.array(prediction['true_class_nll']),np.array(prediction['top5_correct'],dtype=float)
        assert np.isfinite(nll).all() and (nll>=0).all() and np.isin(top5,[0,1]).all() and (top5>=correct).all()
        metrics={'count':3333,'correct':int(correct.sum()),'accuracy':float(correct.mean()),
                 'macro_class_accuracy':float((np.bincount(self.labels,weights=correct,minlength=100)/self.counts_by_class).mean()),
                 'cross_entropy':float(nll.mean()),'top5_accuracy':float(top5.mean())}
        for key,value in metrics.items():assert abs(value-row['test'][key])<3e-7,(relative,key)
        self.counts['prediction_rows_scored']+=3333
        return correct

    def run(self,row,relative,provenance,recipe,epochs):
        relative=Path(relative)
        assert self.read('experiment',relative/'result.json')==row
        config=self.read('experiment',relative/'config.json')
        assert all(row[key]==value for key,value in config.items())
        assert row['provenance']==provenance and row['method']==recipe['method']
        assert row['seed'] in SEEDS and row['rank']==2 and row['epochs']==epochs and row['batch_size']==128
        assert row['optimizer']=='AdamW' and row['gradient_clip_norm']==1 and row['label_smoothing']==.1
        assert row['head_learning_rate']==.001
        assert row['frozen_base_unchanged'] and row['frozen_state_before_sha256']==row['frozen_state_after_sha256']==row['initial_backbone_sha256']
        baseline=self.baselines[row['seed']]
        assert row['initial_head_sha256']==baseline['initial_head_sha256'] and row['initial_backbone_sha256']==baseline['initial_backbone_sha256']
        paired=row['reference_initialization']
        assert paired['initial_head_exact'] and paired['initial_backbone_exact']
        ref_relative=Path('baseline')/f"seed_{row['seed']}"/'result.json'
        assert self.read('reference',ref_relative)==baseline
        assert self.bindings['reference'][str(ref_relative)]==paired['result_sha256']
        assert row['initial_trainable_sha256']!=row['final_trainable_sha256']
        assert math.isfinite(row['initial_output_max_absolute_error'])
        expected_count=76900+12*(12288*2+(5376 if recipe['axis']==0 else 6912))
        assert row['trainable_parameters']==expected_count
        groups=row['optimizer_groups']
        assert {group['group_name'] for group in groups}=={'adapter_factors','magnitudes','classifier'}
        assert sum(group['parameter_count'] for group in groups)==expected_count
        names=[]
        for group in groups:
            names.extend(group['param_names'])
            assert group['weight_decay']==(0 if group['group_name']=='magnitudes' else .01)
            assert group['lr']==(.001 if group['group_name']=='classifier' else row['adapter_learning_rate'])
        assert len(names)==len(set(names)) and set(names)==set(row['trainable_names'])
        history=self.read('experiment',relative/'epochs.jsonl',lines=True)
        assert [entry['epoch'] for entry in history]==list(range(1,epochs+1))
        winner=max(history,key=lambda entry:(entry['validation']['macro_class_accuracy'],-entry['validation']['cross_entropy']))
        assert winner['epoch']==row['best_epoch'] and winner['validation']==row['best_validation']
        for epoch,entry in enumerate(history):
            scale=(epoch+1)/2 if epoch<2 else .5*(1+math.cos(math.pi*(epoch-2)/max(1,epochs-2)))
            assert np.allclose(entry['learning_rates'],[group['lr']*scale for group in groups],atol=0,rtol=1e-14)
            assert entry['train']['count']==3334 and entry['validation']['count']==3333
            assert math.isfinite(entry['train']['label_smoothed_cross_entropy']) and math.isfinite(entry['validation']['cross_entropy'])
        self.counts['epochs_checked']+=epochs
        diagnostics=row['adapter_diagnostics']
        assert len(diagnostics)==48
        for item in diagnostics:
            assert item['recipe_sha256']==recipe['recipe_sha256'] and item['normalization_axis']==recipe['axis']
            assert item['zero_initialized_factor']==recipe['zero_initialized_factor'] and item['zero_initialized_factor_l2']>0
            assert item['clamp_activations']==0 and item['minimum_raw_norm']>0 and math.isfinite(item['minimum_raw_norm'])
            assert item['magnitude_dtype']==item['factor_dtype']=='torch.float32'
            assert item['norm_evaluations']==2+54*epochs+(27 if row['evaluate_test'] else 0)
        self.checkpoint(relative/'best_adapter_and_head.pt',row['best_adapter_checkpoint_sha256'])
        if row['evaluate_test']:
            return self.score(row,relative)
        assert row['test'] is None and not (self.root/relative/'test_predictions.json').exists()

    def method(self,method):
        prefix=Path(method)
        protocol=self.read('experiment',prefix/'protocol.json')
        recipe=protocol['recipe'];provenance=protocol['provenance']
        assert recipe==self.matrix['recipes'][method] and recipe==provenance['recipe']
        payload={k:v for k,v in recipe.items() if k not in ('recipe_sha256','source_sha256')}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==recipe['recipe_sha256']
        assert recipe['source_sha256']==self.core['core_source_sha256']
        assert protocol['seeds']==list(SEEDS) and protocol['rank']==2 and protocol['epochs']==20 and protocol['pilot_epochs']==10
        assert protocol['fixed_original_lr']==.001 and protocol['lr_grid']==self.matrix['lr_grid']
        assert provenance['precision']=='FP32 base/adapters, BF16 autocast; TF32 permitted'
        for source,digest in provenance['source_sha256'].items():self.bind_file('experiment',prefix/'source'/source,digest)
        assert provenance['source_sha256']['experiments/change_ablations.py']==self.core['core_source_sha256']
        for source,digest in self.task['source_sha256'].items():
            assert provenance['source_sha256'][source]==digest,source
        assert self.read('experiment',prefix/'split_manifest.json')==self.manifest
        assert self.bindings['experiment'][str(prefix/'split_manifest.json')]==provenance['split_manifest_sha256']
        assert self.read('experiment',prefix/'core_parity_evidence.json')==self.core
        assert self.read('experiment',prefix/'task_parity_evidence.json')==self.task
        assert provenance['core_parity_evidence_sha256']==self.matrix['core_gate_sha256']
        assert provenance['task_parity_evidence_sha256']==self.matrix['task_gate_sha256']
        assert self.read('experiment',prefix/'original_selected.json')=={'dora_original':{'lr':.001,'magnitude_multiplier':1.}}
        assert self.bindings['experiment'][str(prefix/'original_selected.json')]==provenance['original_selection_sha256']
        reused=self.read('experiment',prefix/'baseline_reuse.json')['reused']
        assert len(reused)==3
        finals=self.read('experiment',prefix/'results.json')
        assert len(finals)==6
        for row in finals:
            if row['method']=='baseline':
                assert row==self.baselines[row['seed']]
                path=prefix/'baseline'/f"seed_{row['seed']}"
                assert self.read('experiment',path/'result.json')==row
        for item in reused:
            suffix=Path(*Path(item['destination_directory']).parts[-2:])
            assert suffix==Path(*Path(item['source_directory']).parts[-2:])
            for name,digest in item['files_sha256'].items():
                if name.endswith('.pt'):
                    self.checkpoint(prefix/suffix/name,digest)
                else:
                    self.bind_file('experiment',prefix/suffix/name,digest)
                    self.bind_file('reference',suffix/name,digest)
        pilots=self.read('experiment',prefix/'rank_2/pilot_results.json')
        assert len(pilots)==4 and {row['adapter_learning_rate'] for row in pilots}==set(self.matrix['lr_grid'])
        assert all(row['seed']==42 and row['stage']=='pilot' and not row['evaluate_test'] for row in pilots)
        winner=max(pilots,key=lambda row:(row['best_validation']['macro_class_accuracy'],-row['best_validation']['cross_entropy']))
        selected=self.read('experiment',prefix/'rank_2/selected.json')
        assert selected=={method:{'lr':winner['adapter_learning_rate'],'magnitude_multiplier':1.}}
        chosen=winner['adapter_learning_rate']
        initials={};arrays={}
        for row in pilots:
            relative=prefix/'rank_2/pilots'/method/f"lr_{row['adapter_learning_rate']:g}_m_1"
            self.run(row,relative,provenance,recipe,10)
            initials.setdefault(42,row['initial_trainable_sha256'])
            assert row['initial_trainable_sha256']==initials[42]
        tuned={row['seed']:row for row in finals if row['method']==method}
        assert set(tuned)==set(SEEDS)
        fixed=self.read('experiment',prefix/'fixed_lr_0.001/results.json')
        assert len(fixed)==3 and {row['seed'] for row in fixed}==set(SEEDS)
        for lane,rows,folder in (('validation_tuned',list(tuned.values()),prefix),('fixed_original_lr',fixed,prefix/'fixed_lr_0.001')):
            for row in rows:
                assert row['evaluate_test'] and row['stage']=='final'
                assert row['adapter_learning_rate']==(chosen if lane=='validation_tuned' else .001)
                relative=folder/'rank_2/final'/method/f"seed_{row['seed']}"
                arrays[lane,row['seed']]=self.run(row,relative,provenance,recipe,20)
                initials.setdefault(row['seed'],row['initial_trainable_sha256'])
                assert row['initial_trainable_sha256']==initials[row['seed']]
        fixed_reuse=self.read('experiment',prefix/'fixed_lr_0.001/fit_reuse.json')
        assert fixed_reuse['selected_lr']==chosen and fixed_reuse['fixed_lr']==.001
        assert len(fixed_reuse['reused'])==(3 if chosen==.001 else 0)
        for item in fixed_reuse['reused']:
            suffix=Path(*Path(item['destination_directory']).parts[-4:])
            assert suffix==Path(*Path(item['source_directory']).parts[-4:])
            for name,digest in item['files_sha256'].items():
                if name.endswith('.pt'):
                    if (self.root/prefix/suffix/name).exists():assert sha(self.root/prefix/suffix/name)==digest
                    elif not self.allow_missing:raise AssertionError(name)
                else:
                    assert self.bindings['experiment'][str(prefix/suffix/name)]==digest
                    assert self.bindings['experiment'][str(prefix/'fixed_lr_0.001'/suffix/name)]==digest
        completion=self.read('experiment',prefix/'completion.json')
        assert completion['status']=='complete' and completion['selected_lr']==chosen
        assert completion['new_epochs']==100+(0 if chosen==.001 else 60)
        assert completion['fixed_fits_reused']==len(fixed_reuse['reused'])
        return arrays,{'method':method,'selected_lr':chosen,'new_epochs':completion['new_epochs'],
                       'fixed_fits_reused':len(fixed_reuse['reused']),'trainable_parameters':tuned[42]['trainable_parameters'],
                       'all_norm_floor_activations_zero':True,'initial_heads_backbone_and_within_recipe_factors_paired':True}


def analyze(root,reference,allow_missing=False,only_method=None):
    audit=Audit(root,reference,allow_missing)
    if only_method:
        arrays,summary=audit.method(only_method)
        return {'status':'pass','method':only_method,'audit':{**audit.counts,**summary},'input_sha256':audit.bindings}
    matrix=audit.matrix
    assert matrix['status']=='complete'
    endpoint_controls={}
    for method in ('ab_original','ab_revised'):
        control=audit.read('experiment',f'{method}_historical_endpoint_control.json')
        assert control['passed'] and control['candidate']==method and len(control['fits'])==7
        audit.bind_file('experiment','historical_endpoint_control_source.py',control['source_sha256'])
        endpoint_controls[method]=control
    methods=matrix['methods'];measured=matrix['measured_methods'];aliases=matrix['aliases']
    assert set(methods)==set(measured)|set(aliases) and len(methods)==12 and len(measured)==10
    all_arrays={};checks=[]
    for method in measured:
        arrays,check=audit.method(method);checks.append(check)
        for (lane,seed),values in arrays.items():all_arrays[method,lane,seed]=values
    task_cases={row['name']:row for row in audit.task['cases']}
    for method,item in aliases.items():
        source=item['source_method']
        assert source in measured and item['recipe']==matrix['recipes'][method] and item['source_recipe']==matrix['recipes'][source]
        assert [key for key in FLAGS if item['recipe'][key]!=item['source_recipe'][key]]==['magnitude_fp32']
        assert item['recipe']['magnitude_fp32'] and not item['source_recipe']['magnitude_fp32']
        assert task_cases[item['task_control_case']]['passed']
        assert item['task_gate_sha256']==matrix['task_gate_sha256'] and item['core_gate_sha256']==matrix['core_gate_sha256']
        for lane in matrix['lanes']:
            for seed in SEEDS:all_arrays[method,lane,seed]=all_arrays[source,lane,seed]
        checks.append({'method':method,'exact_noop_alias_of':source,'scope':item['scope']})
    keys=[(method,lane) for method in methods for lane in matrix['lanes']]
    data=np.array([[all_arrays[method,lane,seed] for seed in SEEDS] for method,lane in keys])
    macro_weights=1/(100*audit.counts_by_class[audit.labels])
    seed_metrics=np.stack((data.mean(axis=2),data@macro_weights),axis=1)
    rng=np.random.default_rng(20261010)
    classes=[np.flatnonzero(audit.labels==value) for value in range(100)]
    samples=np.empty((len(keys),2,10000))
    for start in range(0,10000,64):
        size=min(64,10000-start)
        seeds=rng.integers(3,size=(size,3));images=np.empty((size,3333),dtype=np.int64)
        for values in classes:images[:,values]=rng.choice(values,size=(size,len(values)))
        for index in range(len(keys)):
            selected=data[index][seeds[:,:,None],images[:,None,:]].mean(axis=1)
            samples[index,0,start:start+size]=selected.mean(axis=1)
            samples[index,1,start:start+size]=selected@macro_weights
    summaries=[]
    for index,(method,lane) in enumerate(keys):
        summaries.append({'method':method,'lane':lane,'measurement_method':aliases.get(method,{}).get('source_method',method),
                          'metrics':{metric:{'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'values':values.tolist()}
                                     for metric,values in zip(('accuracy','macro_class_accuracy'),seed_metrics[index])}})
    comparisons=[]
    def comparison(kind,candidate,reference):
        i,j=keys.index(candidate),keys.index(reference);metrics={}
        for index,name in enumerate(('accuracy','macro_class_accuracy')):
            values=seed_metrics[i,index]-seed_metrics[j,index]
            draws=samples[i,index]-samples[j,index];draws[np.abs(draws)<1e-12]=0
            metrics[name]={'mean_difference':float(values.mean()),'paired_seed_differences':values.tolist(),
                           'percentile_95_interval':np.quantile(draws,[.025,.975]).tolist()}
        comparisons.append({'kind':kind,'candidate':{'method':candidate[0],'lane':candidate[1]},
                            'reference':{'method':reference[0],'lane':reference[1]},'metrics':metrics})
    for lane in matrix['lanes']:
        for method in OAT:comparison('one_change_from_original',(method,lane),('ab_original',lane))
        for reference,candidate in zip(CHAIN,CHAIN[1:]):comparison('successive_cumulative_change',(candidate,lane),(reference,lane))
        comparison('all_changes_endpoint',('ab_revised',lane),('ab_original',lane))
    for method in methods:comparison('validation_tuning_effect',(method,'validation_tuned'),(method,'fixed_original_lr'))
    return {'status':'pass','task':'official_aircraft_variants','rank':2,'seeds':list(SEEDS),
            'audit':{**audit.counts,'measured_recipes':10,'declared_recipes':12,'exact_noop_aliases':2,'new_training_epochs':sum(item.get('new_epochs',0) for item in checks),
                     'audited_epoch_rows_include_explicit_fixed_lane_copies':True,'methods':checks},
            'historical_endpoint_controls':endpoint_controls,
            'summary':summaries,'comparisons':comparisons,'input_sha256':audit.bindings,
            'artifact_roots':{key:str(path) for key,path in audit.roots.items()},
            'analysis_source_sha256':sha(Path(__file__)),
            'bootstrap':{'replicates':10000,'seed':20261010,'method':'Shared crossed paired draws of training seeds and class-stratified test images across every recipe/lane.',
                         'limitations':'Exploratory percentile intervals; no multiplicity correction; only three training seeds and one official task split; intervals conditional on validation-selected hyperparameters.'},
            'scope':'Each individual change is compared with original; cumulative increments use the immediately preceding recipe. Equal validation-only tuning and paired fixed LR .001 lanes are separate. Two FP32 magnitude-storage no-op recipes explicitly reuse identical source fits; no test-driven choices.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path,default=Path('/var/tmp/dora-bench/round2/vision/rank2_run'))
    parser.add_argument('--allow-missing-checkpoints',action='store_true')
    parser.add_argument('--method')
    args=parser.parse_args()
    result=analyze(args.root,args.reference_root,args.allow_missing_checkpoints,args.method)
    target=args.root/args.method/'audit.json' if args.method else args.root/'analysis.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    (target.parent/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':result['status'],'audit':result['audit']},indent=2))
