"""Strict independent budget, checkpoint and prediction audit for current DoRA plus NoRA on Aircraft."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

SEEDS = (42,43,44)
METHODS = ('current_dora','current_dora_nora','init_dora','init_dora_nora')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        cpu=tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(cpu.shape),cpu.dtype)).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_shapes():
    return {f'blocks.{block}.{suffix}':(inputs,outputs)
            for block in range(12)
            for suffix,inputs,outputs in (('attn.qkv',768,2304),('attn.proj',768,768),
                                         ('mlp.fc1',768,3072),('mlp.fc2',3072,768))}


def trainable_shapes():
    result={}
    for name,(inputs,outputs) in adapter_shapes().items():
        result.update({name+'.m':(1,inputs),name+'.lora_A':(2,inputs),name+'.lora_B':(outputs,2)})
    return {**result,'head.weight':(100,768),'head.bias':(100,)}


def check_recipe(recipe,method,core):
    payload={key:value for key,value in recipe.items()
             if key not in ('recipe_sha256','source_sha256','inherited_next_source_sha256',
                            'inherited_core_source_sha256','public_source_sha256')}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==recipe['recipe_sha256']
    assert recipe['schema']=='dora-current-nora-v1' and recipe['method']==method
    assert recipe['axis']==0 and recipe['magnitude_fp32']
    assert not any(recipe[key] for key in ('detach_norm','norm_fp32','factorized','clamp_norm'))
    alternate=method.startswith('init_')
    assert recipe['published_init']==alternate and recipe['nora']==method.endswith('_nora')
    assert recipe['nora_epsilon']==1e-12
    assert recipe['base_method']==('next_init' if alternate else 'next_base')
    assert recipe['zero_initialized_factor']==('lora_B' if alternate else 'lora_A')
    assert recipe['trainable_registration_order']==(['lora_A','lora_B','m'] if alternate else ['m','lora_B','lora_A'])
    for recipe_key,gate_key in (('source_sha256','backend_source_sha256'),
                               ('inherited_next_source_sha256','inherited_next_source_sha256'),
                               ('inherited_core_source_sha256','inherited_core_source_sha256'),
                               ('public_source_sha256','standalone_source_sha256')):
        assert recipe[recipe_key]==core[gate_key]


def check_diagnostics(item,recipe,inputs,evaluations):
    assert item['recipe_sha256']==recipe['recipe_sha256'] and item['normalization_axis']==0
    assert item['zero_initialized_factor']==recipe['zero_initialized_factor']
    assert math.isfinite(item['zero_initialized_factor_l2']) and item['zero_initialized_factor_l2']>0
    assert item['clamp_activations']==0 and item['norm_floor'] is None
    assert math.isfinite(item['minimum_raw_norm']) and item['minimum_raw_norm']>0
    assert item['magnitude_dtype']==item['factor_dtype']=='torch.float32'
    assert item['norm_evaluations']==evaluations
    assert item['nora_enabled']==recipe['nora'] and item['nora_epsilon']==recipe['nora_epsilon']
    initial_zero=inputs if recipe['nora'] and not recipe['published_init'] else 0
    assert item['nora_initial_zero_columns']==item['nora_initial_epsilon_columns']==initial_zero
    events=item['nora_epsilon_column_events']
    assert isinstance(events,int) and not isinstance(events,bool)
    if recipe['nora']:
        assert item['nora_norm_evaluations']==evaluations
        minimum=item['nora_minimum_raw_column_norm']
        assert math.isfinite(minimum) and minimum>=0
        assert initial_zero<=events<=inputs*evaluations
        if initial_zero:assert minimum==0 and events>=2*inputs
    else:
        assert item['nora_norm_evaluations']==events==0
        assert item['nora_minimum_raw_column_norm'] is None


class Audit:
    def __init__(self, root, reference, allow_missing=False):
        self.root, self.reference = root, reference
        self.roots = {'experiment':root,'reference':reference}
        self.bindings = {name:{} for name in self.roots}
        self.allow_missing = allow_missing
        self.counts = {'epochs_checked':0,'prediction_rows_scored':0,'checkpoint_payloads_verified':0,'checkpoint_payloads_omitted':0,'checkpoint_tensors_checked':0}
        self.manifest = self.read('reference','split_manifest.json')
        self.labels = np.array([row['label'] for row in self.manifest['splits']['test']])
        self.counts_by_class = np.bincount(self.labels,minlength=100)
        assert len(self.labels)==3333 and (self.counts_by_class>0).all()
        self.baselines = {row['seed']:row for row in self.read('reference','results.json') if row['method']=='baseline'}
        assert set(self.baselines)==set(SEEDS)
        self.core = self.read('experiment','core_parity_evidence.json')
        self.task = self.read('experiment','task_parity_evidence.json')
        assert self.core['status']==self.core['cpu_tests']['status']==self.core['cuda_tests']['status']=='pass'
        assert self.core['backend_source_sha256']==self.core['core_source_sha256']
        for field,name in (('backend_source_sha256','experiments/nora_current.py'),
                           ('inherited_next_source_sha256','experiments/next_ablations.py'),
                           ('inherited_core_source_sha256','experiments/change_ablations.py'),
                           ('standalone_source_sha256','dora.py')):
            assert self.core[field]==self.core['source_sha256'][name]
        assert set(self.core['recipes'])==set(METHODS)
        for method,recipe in self.core['recipes'].items():check_recipe(recipe,method,self.core)
        assert self.task['passed'] and all(row['passed'] for row in self.task['cases'])
        assert len(self.task['cases'])==2 and {row['name'] for row in self.task['cases']}=={'accepted_default','initializer_control'}
        assert self.task['rank']==2 and self.task['batch_size']==4
        assert self.task['autocast']=='bfloat16' and self.task['base_and_adapter_dtype']=='float32'
        for row in self.task['cases']:
            assert len(row['steps'])==3 and row['initial_parameters']['bitwise_equal']
            assert row['reference']==('next_base' if row['name']=='accepted_default' else 'next_init')
            assert row['candidate']==('current_dora' if row['name']=='accepted_default' else 'init_dora')
            for step,record in enumerate(row['steps'],1):
                assert record['step']==step and record['logits_loss_clip_bitwise_equal']
                assert record['gradients']['bitwise_equal'] and record['parameters_after_adamw']['bitwise_equal']
                assert math.isfinite(record['loss']) and math.isfinite(record['preclip_gradient_norm'])
        self.matrix = {'recipes':self.core['recipes'], 'core_gate_sha256':sha(root/'core_parity_evidence.json'),
                       'task_gate_sha256':sha(root/'task_parity_evidence.json'), 'lr_grid':[1e-5,1e-4,3e-4,1e-3]}

    def bind_file(self,which,relative,expected=None):
        relative=str(relative); path=self.roots[which]/relative
        digest=sha(path)
        if expected is not None:assert digest==expected,(relative,digest,expected)
        self.bindings[which][relative]=digest
        return path

    def read(self,which,relative,lines=False):
        path=self.bind_file(which,relative)
        return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())

    def checkpoint(self,relative,expected,shapes=None,expected_tensor_digest=None):
        path=self.root/relative
        if path.exists():
            assert sha(path)==expected
            if shapes is not None:
                state=torch.load(path,map_location='cpu',weights_only=True)
                assert set(state)==set(shapes),(relative,'checkpoint tensor names')
                for name,value in state.items():
                    assert isinstance(value,torch.Tensor) and tuple(value.shape)==shapes[name],(relative,name)
                    assert value.dtype==torch.float32 and torch.isfinite(value).all(),(relative,name)
                assert tensor_digest(state)==expected_tensor_digest,(relative,'trained tensor digest')
                self.counts['checkpoint_tensors_checked']+=len(state)
            self.counts['checkpoint_payloads_verified']+=1
        else:
            assert self.allow_missing,relative
            self.counts['checkpoint_payloads_omitted']+=1

    def score(self,row,relative):
        prediction=self.read('experiment',Path(relative)/'test_predictions.json')
        assert np.array_equal(prediction['labels'],self.labels)
        guesses=np.array(prediction['predictions'])
        assert guesses.shape==self.labels.shape and ((0<=guesses)&(guesses<100)).all()
        assert np.issubdtype(guesses.dtype,np.integer)
        correct=(guesses==self.labels).astype(float)
        nll,top5=np.array(prediction['true_class_nll']),np.array(prediction['top5_correct'],dtype=float)
        assert nll.shape==top5.shape==self.labels.shape
        confidence=np.array(prediction['confidence'])
        assert confidence.shape==self.labels.shape and np.isfinite(confidence).all()
        assert ((0<=confidence)&(confidence<=1)).all()
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
        shapes=trainable_shapes()
        assert row['trainable_names']==sorted(shapes)
        assert row['trainable_parameters']==expected_count
        groups=row['optimizer_groups']
        assert {group['group_name'] for group in groups}=={'adapter_factors','magnitudes','classifier'}
        assert sum(group['parameter_count'] for group in groups)==expected_count
        names=[]
        for group in groups:
            names.extend(group['param_names'])
            assert group['weight_decay']==(0 if group['group_name']=='magnitudes' else .01)
            assert group['lr']==(.001 if group['group_name']=='classifier' else row['adapter_learning_rate'])
            assert group['parameter_count']==sum(math.prod(shapes[name]) for name in group['param_names'])
            allowed={'head.weight','head.bias'} if group['group_name']=='classifier' else {
                name for name in shapes if name.endswith('.m')== (group['group_name']=='magnitudes') and not name.startswith('head.')}
            assert set(group['param_names'])==allowed
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
        expected_adapters=adapter_shapes()
        assert len(diagnostics)==48 and {item['name'] for item in diagnostics}==set(expected_adapters)
        for item in diagnostics:
            check_diagnostics(item,recipe,expected_adapters[item['name']][0],2+54*epochs+(27 if row['evaluate_test'] else 0))
        self.checkpoint(relative/'best_adapter_and_head.pt',row['best_adapter_checkpoint_sha256'],shapes,row['final_trainable_sha256'])
        if row['evaluate_test']:
            return self.score(row,relative)
        assert row['test'] is None and not (self.root/relative/'test_predictions.json').exists()

    def method(self,method):
        prefix=Path('.')
        protocol=self.read('experiment',prefix/'protocol.json')
        recipe=protocol['recipe'];provenance=protocol['provenance']
        assert protocol['schema']=='current-dora-nora-aircraft-v1' and protocol['method']==method
        assert recipe==self.matrix['recipes'][method] and recipe==provenance['recipe']
        assert recipe['source_sha256']==self.core['backend_source_sha256']
        assert protocol['seeds']==list(SEEDS) and protocol['rank']==2 and protocol['epochs']==20 and protocol['pilot_epochs']==10
        assert protocol['fixed_original_lr']==.001 and protocol['lr_grid']==self.matrix['lr_grid']
        assert provenance['precision']=='FP32 base/adapters, BF16 autocast; TF32 permitted'
        for source,digest in provenance['source_sha256'].items():self.bind_file('experiment',prefix/'source'/source,digest)
        assert provenance['source_sha256']['experiments/nora_current.py']==self.core['backend_source_sha256']
        for source,digest in self.core['source_sha256'].items():
            assert provenance['source_sha256'][source]==digest,source
        for source,digest in self.task['source_sha256'].items():
            assert provenance['source_sha256'][source]==digest,source
        assert self.task['model']==provenance['model'] and self.task['checkpoint_sha256']==provenance['checkpoint_sha256']
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
                    baseline=self.baselines[int(suffix.name.removeprefix('seed_'))]
                    self.checkpoint(prefix/suffix/name,digest,{'head.weight':(100,768),'head.bias':(100,)},baseline['final_trainable_sha256'])
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
        assert completion['method']==method and completion['recipe_sha256']==recipe['recipe_sha256']
        assert completion['provenance_sha256']==self.bindings['experiment'][str(prefix/'protocol.json')]
        assert completion['pilots']==4 and completion['tuned_finals']==completion['fixed_lr_finals']==3
        assert completion['new_epochs']==100+(0 if chosen==.001 else 60)
        assert completion['fixed_fits_reused']==len(fixed_reuse['reused'])
        return arrays,{'method':method,'selected_lr':chosen,'new_epochs':completion['new_epochs'],
                       'fixed_fits_reused':len(fixed_reuse['reused']),'trainable_parameters':tuned[42]['trainable_parameters'],
                       'all_norm_floor_activations_zero':True,'initial_heads_backbone_and_within_recipe_factors_paired':True,
                       'nora_enabled':recipe['nora'],
                       'nora_diagnostics_tuned_seed42':[{key:value for key,value in item.items() if key=='name' or key.startswith('nora_')}
                                                       for item in tuned[42]['adapter_diagnostics']]}



def audit(root, reference):
    checker=Audit(root,reference)
    protocol=checker.read('experiment','protocol.json')
    method=protocol['method']
    assert method in METHODS
    arrays, summary=checker.method(method)
    historical=[]
    if method in ('current_dora','init_dora'):
        source_method='ab_original' if method=='current_dora' else 'ab_init'
        oldroot=Path('/var/tmp/dora-bench/change_ablation_runs/aircraft_rank2')/source_method
        old_selection=json.loads((oldroot/'rank_2/selected.json').read_text())[source_method]['lr']
        assert old_selection==summary['selected_lr']
        paths=[]
        for rate in protocol['lr_grid']:
            paths.append((Path('rank_2/pilots')/method/f'lr_{rate:g}_m_1',Path('rank_2/pilots')/source_method/f'lr_{rate:g}_m_1'))
        for seed in SEEDS:
            paths.append((Path('rank_2/final')/method/f'seed_{seed}',Path('rank_2/final')/source_method/f'seed_{seed}'))
        for newrel,oldrel in paths:
            newdir,olddir=root/newrel,oldroot/oldrel
            new=json.loads((newdir/'result.json').read_text());old=json.loads((olddir/'result.json').read_text())
            for key in ('best_epoch','best_validation','test','initial_head_sha256','initial_backbone_sha256','initial_trainable_sha256','final_trainable_sha256'):
                assert new[key]==old[key],(method,newrel,key)
            a,b=[torch.load(d/'best_adapter_and_head.pt',map_location='cpu',weights_only=True) for d in (newdir,olddir)]
            assert list(a)==list(b)
            for name,value in a.items():
                assert value.dtype==b[name].dtype and torch.equal(value,b[name]),name
            logs=[]
            for directory in (newdir,olddir):
                rows=[json.loads(x) for x in (directory/'epochs.jsonl').read_text().splitlines()]
                logs.append([{k:v for k,v in row.items() if k not in ('training_seconds','epoch_seconds')} for row in rows])
            assert logs[0]==logs[1]
            prediction=newdir/'test_predictions.json'
            if prediction.exists():assert prediction.read_bytes()==(olddir/'test_predictions.json').read_bytes()
            historical.append({'candidate':str(newdir),'reference':str(olddir),'tensors_bitwise':len(a),
                               'history_bitwise':True,'predictions_byte_identical':prediction.exists(),
                               'reference_files_sha256':{p.name:sha(p) for p in olddir.iterdir() if p.is_file()},
                               'candidate_files_sha256':{p.name:sha(p) for p in newdir.iterdir() if p.is_file()}})
    source=root/'audit_source'/Path(__file__).name
    source.parent.mkdir(exist_ok=True);source.write_bytes(Path(__file__).read_bytes())
    checker.bind_file('experiment',str(source.relative_to(root)))
    out={'status':'pass','task':'current-dora-nora-aircraft','method':method,'audit':{**checker.counts,**summary},
         'historical_control_parity':historical,'input_sha256':checker.bindings,
         'artifact_roots':{k:str(v) for k,v in checker.roots.items()},'checker_source_sha256':sha(Path(__file__))}
    (root/'task_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('/var/tmp/dora-bench/round2/vision/rank2_run'))
    a=p.parse_args()
    out=audit(a.root,a.reference)
    print(json.dumps({'status':out['status'],'method':out['method'],'audit':out['audit']}))

if __name__=='__main__':main()
