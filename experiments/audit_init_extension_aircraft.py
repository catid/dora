"""Independent Aircraft initializer provenance, selection, tensor and metric audit."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from experiments import audit_init_extension_metadata as extension_checks


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def tensor_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        cpu=tensor.detach().cpu().contiguous();h.update(name.encode());h.update(str((tuple(cpu.shape),cpu.dtype)).encode())
        h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_shapes():
    return {f'blocks.{block}.{suffix}':(inputs,outputs) for block in range(12)
        for suffix,inputs,outputs in (('attn.qkv',768,2304),('attn.proj',768,768),('mlp.fc1',768,3072),('mlp.fc2',3072,768))}


def trainable_shapes():
    shapes={}
    for name,(inputs,outputs) in module_shapes().items():
        shapes.update({name+'.m':(1,inputs),name+'.lora_A':(2,inputs),name+'.lora_B':(outputs,2)})
    return {**shapes,'head.weight':(100,768),'head.bias':(100,)}


def diagnostics(rows,candidate,recipe,seed):
    shapes=module_shapes();assert len(rows)==48 and {r['name'] for r in rows}==set(shapes)
    kind=candidate['spec'].get('kind','random');spec=candidate['spec'].get('config',candidate['spec'])
    for row in rows:
        inputs,outputs=shapes[row['name']]
        assert row['recipe_sha256']==recipe['recipe_sha256'] and row['normalization_axis']==0
        assert row['norm_floor'] is None and row['clamp_activations']==0
        assert math.isfinite(row['minimum_raw_norm']) and row['minimum_raw_norm']>0
        assert row['factor_dtype']==row['magnitude_dtype']=='torch.float32'
        init=row['initializer'];assert init['factor_seed']==seed and init['module_path']==row['name']
        if init['initializer_kind'] in ('warmstart','quantized','extension_warmstart'):
            assert kind==init['initializer_kind']
            for key in ('up_sha256','down_sha256','pretrained_weight_sha256'):assert len(init[key])==64
            assert init['magnitude_policy']==spec.get('magnitude_policy','original')
            if kind in ('warmstart','extension_warmstart'):
                assert not init['compensated']
                if kind=='warmstart':assert math.isfinite(init['initial_delta_relative_norm']) and init['initial_delta_relative_norm']>=0
            else:assert len(init['quantized_weight_sha256'])==64 and init['quantized_weight_sha256']!=init['pretrained_weight_sha256']
            continue
        assert init['initial_adapter_delta_is_zero']
        if init['initializer_kind'] in ('data','random_both_nonzero','extension_centered'):
            assert init['compensated']
            for factor in ('up','down'):
                assert len(init['initial_'+factor+'_sha256'])==64
                stats=init[factor+'_statistics'];gram=torch.tensor(stats['rank_gram'],dtype=torch.float64)
                assert gram.shape==(2,2) and torch.isfinite(gram).all()
                torch.testing.assert_close(gram,gram.T,rtol=0,atol=1e-10)
                eigen=torch.linalg.eigvalsh(gram).clamp_min(0)
                torch.testing.assert_close(eigen,torch.tensor(stats['rank_gram_eigenvalues'],dtype=torch.float64),rtol=1e-10,atol=1e-10)
                assert math.isfinite(stats['entry_rms']) and stats['entry_rms']>=0
        else:
            assert kind=='random' and init['initializer_kind']=='random'
            orientation=spec.get('orientation','up');width=outputs if orientation=='up' else inputs
            assert row['zero_initialized_factor']==('lora_A' if orientation=='up' else 'lora_B')
            assert init['random_vector_count']==width and init['random_factor_rms']>0 and math.isfinite(init['random_factor_rms'])
            assert init['zero_factor_nonzero_entries']==0
            denominator=(2 if spec.get('radial','none')=='unit_vectors' else 1 if spec.get('scaling','rank')=='absolute'
                else inputs if spec.get('scaling','rank')=='original_input' or spec.get('scaling','rank')=='fan_in' and orientation=='down' else 2)
            assert math.isclose(init['target_entry_rms'],spec.get('scale',1)/math.sqrt(denominator),rel_tol=0,abs_tol=1e-15)
            assert math.isclose(init['random_factor_frobenius_norm'],init['random_factor_rms']*math.sqrt(width*2),rel_tol=1e-12)
            gram=torch.tensor(init['rank_gram'],dtype=torch.float64);assert gram.shape==(2,2) and torch.isfinite(gram).all()
            torch.testing.assert_close(gram,gram.T,rtol=0,atol=1e-10)
            eigen=torch.linalg.eigvalsh(gram).clamp_min(0)
            torch.testing.assert_close(eigen,torch.tensor(init['rank_gram_eigenvalues'],dtype=torch.float64),rtol=1e-10,atol=1e-10)


def audit(root,phase):
    bindings={};checkpoints={}
    own=root/'audit_source'/Path(__file__).name;own.parent.mkdir(exist_ok=True);own.write_bytes(Path(__file__).read_bytes());bindings[str(own.relative_to(root))]=sha(own)
    helper=root/'audit_source'/Path(extension_checks.__file__).name;helper.parent.mkdir(exist_ok=True);helper.write_bytes(Path(extension_checks.__file__).read_bytes());bindings[str(helper.relative_to(root))]=sha(helper)
    counts={'fits':0,'failed_fits':0,'epochs_checked':0,'prediction_rows_scored':0,'historical_accepted_fits_verified':0};unavailable_history=[]
    def read(relative,lines=False):
        path=root/relative;bindings[str(relative)]=sha(path)
        return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    manifest=read(Path('manifest.json'));registry=read(Path('reference/registry.json'));splits=read(Path('split_manifest.json'))
    assert manifest['schema']=='dora-init-extension-aircraft-v1' and manifest['registry']==registry
    assert registry['task']=='aircraft' and registry['rank']==2 and registry['seed']==42 and registry['seeds']==[42,43,44]
    assert registry['fixed_lr']==.001 and registry['learning_rates']==[.00001,.0001,.0003,.001]
    assert manifest['registry_sha256']==sha(root/'reference/registry.json')
    assert manifest['protocol_sha256']==registry['protocol_sha256']==sha(root/'reference/protocol.md')
    assert manifest['split_manifest_sha256']==sha(root/'split_manifest.json')
    assert splits['effective_sizes']=={'train':3334,'val':3333,'test':3333}
    config=manifest['configuration'];assert config['epochs']==20 and config['pilot_epochs']==10 and config['batch_size']==128
    assert config['head_lr']==config['fixed_lr']==.001 and config['seeds']==[42,43,44]
    for name,value in manifest['source_sha256'].items():assert sha(root/'source'/name)==value;bindings['source/'+name]=value
    gate=read(Path('reference/gate.json'));taskgate=read(Path('reference/task_gate.json'))
    assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    assert manifest['gate_sha256']==sha(root/'reference/gate.json') and manifest['task_gate_sha256']==sha(root/'reference/task_gate.json')
    assert taskgate['passed'] and taskgate['rank']==2 and taskgate['batch_size']==4
    assert len(taskgate['cases'])==1
    for row in taskgate['cases']:
        assert row['passed'] and row['initial_parameters']['bitwise_equal'] and len(row['steps'])==3
        for step in row['steps']:assert step['logits_loss_clip_bitwise_equal'] and step['gradients']['bitwise_equal'] and step['parameters_after_adamw']['bitwise_equal']
    for evidence in (gate,taskgate):
        for name,value in evidence['source_sha256'].items():assert manifest['source_sha256'][name]==value
    candidates={c['candidate_id']:c for c in registry['candidates']};assert len(candidates)==len(registry['candidates'])
    calibrations={}
    for seed,value in manifest['calibration_manifest_sha256'].items():
        path=Path('calibration')/f'seed_{seed}'/'manifest.json';calibration=read(path);assert sha(root/path)==value
        calibrations[int(seed)]=calibration
        extension_checks.calibration(calibration,'aircraft',manifest['source_sha256']['experiments/init_extension_observers.py'])
        meta=calibration['metadata'];assert meta['classifier_seed']==int(seed) and meta['train_only']
        assert meta['population_size']==128 and meta['training_population_size']==3334 and meta['batch_count']==16
        assert meta['batch_size']==8 and meta['total_loss_weight']==128 and meta['model_mode']=='eval'
        assert meta['calibration_seed']==20261009 and len(meta['example_ids'])==len(set(meta['example_ids']))==128
        train={row['image']:row for row in splits['splits']['train']}
        assert set(meta['example_ids'])<=set(train)
        assert not set(meta['example_ids'])&{r['image'] for split in ('val','test') for r in splits['splits'][split]}
        assert [image for batch in meta['batches'] for image in batch['image_ids']]==meta['example_ids']
        for batch in meta['batches']:
            assert batch['labels']==[train[image]['label'] for image in batch['image_ids']]
            assert math.isfinite(batch['loss'])
        assert set(calibration['module_files'])==set(module_shapes())
        for name,file in calibration['module_files'].items():
            assert sha(file)==calibration['module_sha256'][name]
            stats=torch.load(file,map_location='cpu',weights_only=True)
            assert stats['population_size']==128 and stats['batch_count']==16 and stats['forward_count']==stats['backward_count']==16
            for key,tensor in stats.items():
                if torch.is_tensor(tensor):assert torch.isfinite(tensor).all(),(name,key)
            checkpoints[str(Path(file))]=sha(file)
    assert registry['study_lane']==manifest['study_lane']=='extension'
    controls=read(Path('controls/control.json'));assert controls['status']=='pass' and controls['manifest_sha256']==sha(root/'manifest.json')
    assert {digest(row['spec']) for row in controls['controls']}=={digest(c['spec']) for c in candidates.values()}
    assert controls['unique_specs']==len(controls['controls'])
    for row in controls['controls']:
        lane=candidates[row['candidate_id']]['initialization_lane']
        if row.get('failure_stage')=='initialization':
            assert row['status']=='numerical_failure';extension_checks.numerical_failure(row['failure']);continue
        assert row['training_rng_matches_accepted'] and row['frozen_weights_unchanged']
        if lane=='centered':assert row['initial_probe_bitwise_accepted']
        else:
            assert row['study_lane']==lane and row['initial_validation']['count']==3333
            assert all(math.isfinite(v) and v>=0 for v in row['initial_probe_drift'].values())
        candidate=candidates[row['candidate_id']];recipe=manifest['recipes']['42'][row['candidate_id']]
        diagnostics(row['initial_diagnostics'],candidate,recipe,42)
        extension_checks.diagnostics(row['initial_diagnostics'],candidate,calibrations[42],'aircraft')
        if row['status']=='pass':
            assert row['optimizer_updates_completed']==1 and math.isfinite(row['initial_first_loss']) and math.isfinite(row['first_step_gradient_norm'])
            diagnostics(row['diagnostics'],candidate,recipe,42)
        else:
            assert row['status']=='numerical_failure';extension_checks.numerical_failure(row['failure'])
    controls_by_spec={digest(row['spec']):row for row in controls['controls']}
    for candidate in candidates.values():
        if candidate['spec'].get('kind')!='quantized' or candidate['spec'].get('config',{}).get('method')!='loftq':continue
        assert candidate.get('comparison_control_id') in candidates
        comparator=candidates[candidate['comparison_control_id']]
        assert comparator['spec']['kind']=='quantized' and comparator['spec']['config']['method']=='quantizer_only'
        for key,default in (('bits',4),('blocksize',64),('iterations',1),('magnitude_policy','original')):
            assert candidate['spec']['config'].get(key,default)==comparator['spec']['config'].get(key,default)
        if candidate['spec']['config'].get('iterations',1)>1:assert comparator['spec']['config'].get('base_source')=='paired_loftq'
        treatment=controls_by_spec[digest(candidate['spec'])];control=controls_by_spec[digest(comparator['spec'])]
        if treatment.get('failure_stage')=='initialization' or control.get('failure_stage')=='initialization':continue
        contrast=treatment['comparison_controls'][candidate['comparison_control_id']]
        assert contrast['same_frozen_base'] and all(math.isfinite(contrast[key]) and contrast[key]>=0 for key in ('mse','maximum_absolute_error','relative_l2'))
        assert treatment['initial_frozen_sha256']==control['initial_frozen_sha256']
        left=torch.load(root/'controls'/(treatment['candidate_id']+'.pt'),map_location='cpu',weights_only=True)['probe']
        right=torch.load(root/'controls'/(control['candidate_id']+'.pt'),map_location='cpu',weights_only=True)['probe']
        expected={'maximum_absolute_error':float((left-right).abs().max()),'mse':float((left-right).square().mean()),'relative_l2':float((left-right).norm()/right.norm())}
        for key,value in expected.items():assert math.isclose(contrast[key],value,rel_tol=0,abs_tol=1e-10)
    for name,value in controls['evidence_sha256'].items():assert sha(root/'controls'/name)==value;checkpoints['controls/'+name]=value
    if phase=='controls':return {'status':'pass','phase':phase,'input_sha256':bindings,'checkpoint_sha256':checkpoints}
    expected_shapes=trainable_shapes();fits={};historical=[]
    for path in sorted((root/'fits').glob('*/*/*/fit.json')):
        fit=read(path.relative_to(root));fits[fit['fit_id']]=fit;candidate=candidates[fit['candidate_id']];counts['fits']+=1;lane=candidate['initialization_lane']
        assert fit['candidate_sha256']==digest(candidate) and fit['spec']==candidate['spec'] and fit['family_id']==candidate['family_id']
        assert fit['manifest_sha256']==sha(root/'manifest.json') and fit['seed'] in (42,43,44) and fit['rank']==2
        recipe=manifest['recipes'][str(fit['seed'])][fit['candidate_id']];assert fit['recipe_sha256']==recipe['recipe_sha256']
        directory=Path(fit['artifact_directory']);history=read(directory/'epochs.jsonl',lines=True) if (root/directory/'epochs.jsonl').exists() else []
        if fit['status']=='numerical_failure':
            extension_checks.numerical_failure(fit['failure'])
            assert fit['failure_stage'] in ('initialization','training')
            assert fit['completed_epochs']==len(history)
            assert fit['completed_optimizer_updates_lower_bound']==len(history)*math.ceil(3334/128)
            assert fit['completed_optimizer_updates_exact']==(0 if fit['failure_stage']=='initialization' else None)
            assert [r['epoch'] for r in history]==list(range(1,len(history)+1))
            assert all(math.isfinite(r['train']['label_smoothed_cross_entropy']) and math.isfinite(r['validation']['cross_entropy']) for r in history)
            counts['failed_fits']+=1;continue
        assert fit['status']=='pass';row=read(directory/'result.json');assert row==fit['result']
        config=read(directory/'config.json');assert all(row[key]==value for key,value in config.items())
        assert row['method']==fit['candidate_id'] and row['seed']==fit['seed'] and row['adapter_learning_rate']==candidate['learning_rate']
        assert row['provenance']['recipe']==recipe and row['provenance']['source_sha256']==manifest['source_sha256']
        epochs=10 if fit['stage']=='pilot' else 20;assert row['epochs']==epochs and row['batch_size']==128
        assert row['stage']==fit['stage'] and row['evaluate_test']==(fit['stage']=='final')
        assert row['label_smoothing']==.1 and row['gradient_clip_norm']==1 and row['head_learning_rate']==.001
        assert row['frozen_base_unchanged'] and row['frozen_state_before_sha256']==row['frozen_state_after_sha256']
        if candidate['spec'].get('kind')=='quantized':
            assert fit['initialization']['deterministic_recreation_verified']
            assert row['frozen_state_before_sha256']==fit['initialization']['expected_frozen_sha256']!=row['initial_backbone_sha256']
        else:assert row['frozen_state_before_sha256']==row['initial_backbone_sha256']
        if lane!='centered':
            assert row['study_lane']==lane and row['initial_validation']['count']==3333
            assert all(math.isfinite(row[key]) and row[key]>=0 for key in ('initial_probe_mse','initial_probe_relative_l2','initial_output_max_absolute_error'))
            initial_prediction=read(directory/'initial_validation_predictions.json')
            assert sha(root/directory/'initial_validation_predictions.json')==row['initial_validation_predictions_sha256']
            initial_labels=np.array([r['label'] for r in splits['splits']['val']])
            assert np.array_equal(initial_prediction['labels'],initial_labels)
            initial_guesses=np.array(initial_prediction['predictions']);assert initial_guesses.shape==initial_labels.shape
            assert ((0<=initial_guesses)&(initial_guesses<100)).all()
            initial_correct=(initial_guesses==initial_labels).astype(float)
            initial_nll=np.array(initial_prediction['true_class_nll']);initial_top5=np.array(initial_prediction['top5_correct'],dtype=float)
            assert initial_nll.shape==initial_top5.shape==initial_labels.shape and np.isfinite(initial_nll).all() and (initial_nll>=0).all()
            assert np.isin(initial_top5,[0,1]).all() and (initial_top5>=initial_correct).all()
            initial_metrics={'count':3333,'correct':int(initial_correct.sum()),'accuracy':float(initial_correct.mean()),
                'cross_entropy':float(initial_nll.mean()),'top5_accuracy':float(initial_top5.mean()),
                'macro_class_accuracy':float((np.bincount(initial_labels,weights=initial_correct,minlength=100)/np.bincount(initial_labels,minlength=100)).mean())}
            for key,value in initial_metrics.items():assert abs(row['initial_validation'][key]-value)<3e-7
            counts['prediction_rows_scored']+=3333

        assert row['reference_initialization']['initial_head_exact'] and row['reference_initialization']['initial_backbone_exact']
        reference_path=Path(row['reference_initialization']['result_path']);prior=json.loads(reference_path.read_text())
        assert sha(reference_path)==row['reference_initialization']['result_sha256']
        for key in ('initial_head_sha256','initial_backbone_sha256'):assert row[key]==prior[key]
        assert row['initial_trainable_sha256']!=row['final_trainable_sha256']
        assert row['trainable_names']==sorted(expected_shapes) and row['trainable_parameters']==sum(math.prod(shape) for shape in expected_shapes.values())
        groups=row['optimizer_groups'];assert {g['group_name'] for g in groups}=={'adapter_factors','magnitudes','classifier'}
        names=[]
        for group in groups:
            names.extend(group['param_names']);assert group['weight_decay']==(0 if group['group_name']=='magnitudes' else .01)
            assert group['lr']==(.001 if group['group_name']=='classifier' else candidate['learning_rate'])
            assert group['parameter_count']==sum(math.prod(expected_shapes[n]) for n in group['param_names'])
        assert len(names)==len(set(names))==len(expected_shapes) and set(names)==set(expected_shapes)
        assert [entry['epoch'] for entry in history]==list(range(1,epochs+1))
        best=max(history,key=lambda e:(e['validation']['macro_class_accuracy'],-e['validation']['cross_entropy']))
        assert row['best_epoch']==best['epoch'] and row['best_validation']==best['validation']
        for epoch,entry in enumerate(history):
            scale=(epoch+1)/2 if epoch<2 else .5*(1+math.cos(math.pi*(epoch-2)/max(1,epochs-2)))
            assert np.allclose(entry['learning_rates'],[g['lr']*scale for g in groups],rtol=1e-14,atol=0)
            assert entry['train']['count']==3334 and entry['validation']['count']==3333
            assert math.isfinite(entry['train']['label_smoothed_cross_entropy']) and math.isfinite(entry['validation']['cross_entropy'])
        counts['epochs_checked']+=epochs
        diagnostics(fit['initialization']['diagnostics'],candidate,recipe,fit['seed'])
        extension_checks.diagnostics(fit['initialization']['diagnostics'],candidate,calibrations[fit['seed']],'aircraft')
        diagnostics(row['adapter_diagnostics'],candidate,recipe,fit['seed'])
        reload=fit['reload_validation'];assert reload['bitwise_checkpoint_probe'] and reload['initial_factors_recreated'] and reload['frozen_base_verified']
        probe_path=Path(fit['fit_id'])/'probe_replay.pt';assert sha(root/probe_path)==reload['probe_sha256']
        probe=torch.load(root/probe_path,map_location='cpu',weights_only=True)
        torch.testing.assert_close(probe['trained_probe'],probe['replayed_probe'],rtol=0,atol=0)
        checkpoints[str(probe_path)]=reload['probe_sha256']
        checkpoint=directory/'best_adapter_and_head.pt';assert sha(root/checkpoint)==fit['checkpoint_sha256']==row['best_adapter_checkpoint_sha256']
        state=torch.load(root/checkpoint,map_location='cpu',weights_only=True);assert set(state)==set(expected_shapes)
        assert tensor_digest(state)==row['final_trainable_sha256']
        for name,value in state.items():assert tuple(value.shape)==expected_shapes[name] and value.dtype==torch.float32 and torch.isfinite(value).all()
        checkpoints[str(checkpoint)]=sha(root/checkpoint)
        if fit['stage']=='pilot':assert row['test'] is None and not (root/directory/'test_predictions.json').exists()
        else:
            prediction=read(directory/'test_predictions.json');labels=np.array([r['label'] for r in splits['splits']['test']])
            assert np.array_equal(prediction['labels'],labels)
            guesses=np.array(prediction['predictions']);assert guesses.shape==labels.shape and ((0<=guesses)&(guesses<100)).all()
            correct=(guesses==labels).astype(float);nll=np.array(prediction['true_class_nll']);top5=np.array(prediction['top5_correct'],dtype=float)
            assert nll.shape==top5.shape==labels.shape and np.isfinite(nll).all() and (nll>=0).all()
            assert np.isin(top5,[0,1]).all() and (top5>=correct).all()
            metrics={'count':3333,'correct':int(correct.sum()),'accuracy':float(correct.mean()),'cross_entropy':float(nll.mean()),
                'top5_accuracy':float(top5.mean()),'macro_class_accuracy':float((np.bincount(labels,weights=correct,minlength=100)/np.bincount(labels,minlength=100)).mean())}
            for key,value in metrics.items():assert abs(row['test'][key]-value)<3e-7,(path,key)
            counts['prediction_rows_scored']+=3333
        kind=candidate['spec'].get('kind','random');spec=candidate['spec'].get('config',candidate['spec'])
        if kind=='random' and spec.get('mode','accepted')=='accepted':
            prior_root=Path('/var/tmp/dora-bench/nora_current_bench/aircraft/current_dora')
            paths=([prior_root/'rank_2/pilots/current_dora'/f"lr_{candidate['learning_rate']:g}_m_1"] if fit['stage']=='pilot' else
                [prior_root/'rank_2/final/current_dora'/f"seed_{fit['seed']}",prior_root/'fixed_lr_0.001/rank_2/final/current_dora'/f"seed_{fit['seed']}"])
            old=extension_checks.exact_history(paths,{'seed':fit['seed'],'adapter_learning_rate':candidate['learning_rate'],'epochs':epochs,'rank':2})
            if old is None:
                unavailable_history.append({'fit_id':fit['fit_id'],'reason':'No historical accepted fit with exact seed, learning rate and epoch budget'})
                continue
            prior=json.loads((old/'result.json').read_text())
            for key in ('best_epoch','best_validation','test','initial_head_sha256','initial_backbone_sha256','initial_trainable_sha256','final_trainable_sha256'):assert row[key]==prior[key],(path,key)
            oldstate=torch.load(old/'best_adapter_and_head.pt',map_location='cpu',weights_only=True)
            assert list(state)==list(oldstate)
            for name,value in state.items():torch.testing.assert_close(value,oldstate[name],rtol=0,atol=0)
            oldhistory=[json.loads(x) for x in (old/'epochs.jsonl').read_text().splitlines()]
            clean=lambda values:[{k:v for k,v in r.items() if k not in ('training_seconds','epoch_seconds')} for r in values]
            assert clean(history)==clean(oldhistory)
            if fit['stage']=='final':assert (old/'test_predictions.json').read_bytes()==(root/directory/'test_predictions.json').read_bytes()
            historical.append({'fit_id':fit['fit_id'],'reference':str(old),'bitwise_checkpoint':True,'exact_history':True,'exact_predictions':fit['stage']=='final'})
            counts['historical_accepted_fits_verified']+=1
    if phase=='fits':return {'status':'pass','phase':phase,'task':'aircraft','source_sha256':sha(__file__),
        'input_sha256':bindings,'checkpoint_sha256':checkpoints,'audit':counts,
        'scope':'Engineering partial-fit audit; no registry selection or completion claim.'}
    selection=read(Path('selections.json'));pilot_gate=read(Path('pilots_complete.json'))
    assert selection['registry_sha256']==digest(registry) and selection['all_pilots_complete_before_test'] and not selection['test_used_for_selection']
    assert pilot_gate['status']=='pass' and pilot_gate['selections_sha256']==sha(root/'selections.json')
    assert {r['candidate_id'] for r in selection['pilots']}==set(candidates) and len(selection['pilots'])==len(candidates)
    pilot_byid={r['candidate_id']:r for r in selection['pilots']}
    for pilot in selection['pilots']:
        fit=fits[pilot['fit_id']];assert sha(root/pilot['fit_id']/'fit.json')==pilot['pilot_sha256'] and fit['stage']=='pilot' and fit['seed']==42
        assert pilot['status']==fit['status']
        if fit['status']=='pass':
            assert pilot['selection_value']==fit['result']['best_validation']['macro_class_accuracy']
            assert pilot['selection_cross_entropy']==fit['result']['best_validation']['cross_entropy']
    reconstructed={}
    for family in sorted({c['family_id'] for c in candidates.values()}):
        reconstructed[family]={}
        for lane in ('fixed_original_lr','validation_tuned'):
            eligible=[c for c in candidates.values() if c['family_id']==family and pilot_byid[c['candidate_id']]['status']=='pass'
                and (lane!='fixed_original_lr' or c['learning_rate']==.001)]
            reconstructed[family][lane]=min(eligible,key=lambda c:(-pilot_byid[c['candidate_id']]['selection_value'],
                pilot_byid[c['candidate_id']]['selection_cross_entropy'],c['learning_rate'],c['candidate_id']))['candidate_id'] if eligible else None
    assert selection['selections']==reconstructed
    if phase=='finals':
        results=read(Path('results.json'));assert results['complete'] and results['selections_sha256']==sha(root/'selections.json')
        expected={(family,lane,seed) for family,choices in reconstructed.items() for lane,cid in choices.items() if cid is not None for seed in (42,43,44)}
        assert len(results['runs'])==len(expected) and {(r['family_id'],r['lane'],r['seed']) for r in results['runs']}==expected
        for row in results['runs']:
            assert row['candidate_id']==reconstructed[row['family_id']][row['lane']]
            assert all(row[key]==value for key,value in fits[row['fit_id']].items())
        pairs=results.get('paired_control_fits',[])
        expected_pairs={(family,cid,seed) for family,choices in reconstructed.items() for cid in set(choices.values())
            if cid is not None and candidates[cid]['comparison_control_id']!=cid for seed in (42,43,44)}
        assert len(pairs)==len(expected_pairs) and {(r['for_family_id'],r['for_candidate_id'],r['seed']) for r in pairs}==expected_pairs
        for pair in pairs:
            treatment=candidates[pair['for_candidate_id']];control=candidates[treatment['comparison_control_id']]
            assert pair['candidate_id']==control['candidate_id'] and pair['learning_rate']==treatment['learning_rate']
            assert all(pair[key]==value for key,value in fits[pair['fit_id']].items())
            treatment_fit=next(value for value in fits.values() if value['candidate_id']==pair['for_candidate_id'] and value['seed']==pair['seed'] and value['stage']=='final')
            if pair['status']==treatment_fit['status']=='pass':
                assert pair['result']['frozen_state_before_sha256']==treatment_fit['result']['frozen_state_before_sha256']
        assert results['unique_trained_fits']==len(fits) and results['unique_final_fits']==len({r['fit_id'] for r in results['runs']+pairs})
    else:assert all(fit['stage']=='pilot' for fit in fits.values())
    return {'status':'pass','phase':phase,'task':'aircraft','source_sha256':sha(__file__),'input_sha256':bindings,
        'checkpoint_sha256':checkpoints,'audit':counts,'historical_accepted_parity':historical,'unavailable_historical_parity':unavailable_history,
        'scope':'Independent immutable registry/selection/calibration binding, historical accepted tensor/history identity, completeepoch/rate/classifier checks, checkpointtensors and rawprediction rescore. No inference rerun.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('controls','fits','pilots','finals'),required=True);args=parser.parse_args()
    torch.set_num_threads(2);result=audit(args.root,args.phase)
    path=args.root/('task_audit.json' if args.phase=='finals' else args.phase+'_audit.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result.get('audit',{'status':'pass','phase':args.phase}),indent=2))
