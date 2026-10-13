"""Independent fresh-seed confirmation coverage, budgets, tensors and metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiments import audit_init_retrieval as retrieval
from experiments import audit_init_aircraft as aircraft


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def audit(root,phase):
    bindings={};checkpoints={};counts={'fits':0,'failed_fits':0,'optimizer_updates_checked':0,'epochs_checked':0,'predictions_rescored':0}
    def read(relative,lines=False):
        path=root/relative;bindings[str(relative)]=sha(path)
        return [json.loads(x) for x in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    manifest=read(Path('manifest.json'));registry=read(Path('reference/registry.json'));task=registry['task'];is_aircraft=task=='aircraft'
    assert manifest['registry']==registry and registry['schema']=='dora-init-confirmation-v1'
    assert registry['seeds']==[45,46,47,48,49] and not registry['test_used_for_selection'] and registry['study_lane']=='centered'
    assert set(registry['roles'])=={'validation_winner','accepted_anchor','accepted_gaussian_scale'}
    assert registry['control_id']==registry['roles']['accepted_anchor']
    assert manifest['registry_sha256']==sha(root/'reference/registry.json')
    assert registry['protocol_sha256']==manifest['protocol_sha256']==sha(root/'reference/protocol.md')
    parent=Path(registry['source_study_root']);parent_manifest=json.loads((parent/'manifest.json').read_text())
    parent_registry=parent_manifest['registry'];selection=json.loads((parent/'selections.json').read_text())
    assert sha(parent/'manifest.json')==registry['parent_manifest_sha256']==manifest['parent_manifest_sha256']
    assert digest(parent_registry)==registry['parent_registry_sha256'] and sha(parent/'selections.json')==registry['parent_selections_sha256']
    assert not selection['test_used_for_selection']
    parent_audit=json.loads((parent/'pilots_audit.json').read_text())
    assert parent_audit['status']=='pass' and sha(parent/'pilots_audit.json')==registry['parent_pilots_audit_sha256']
    for name,value in parent_manifest['source_sha256'].items():assert manifest['source_sha256'][name]==value
    for name,value in registry['pilot_evidence_sha256'].items():assert sha(parent/name)==value
    byid={c['candidate_id']:c for c in registry['candidates']};all_candidates={c['candidate_id']:c for c in parent_registry['candidates']}
    successful={p['candidate_id']:p for p in selection['pilots'] if p['status']=='pass'}
    def key(cid):
        row=successful[cid];return (-row['selection_value'],row.get('selection_cross_entropy',0),all_candidates[cid]['learning_rate'],cid)
    assert registry['roles']['validation_winner']==min(successful,key=key)
    for role in ('accepted_anchor','accepted_gaussian_scale'):
        assert registry['roles'][role]==min((cid for cid in successful if all_candidates[cid]['family_id']==role),key=key)
    assert set(byid)==set(registry['roles'].values())
    for cid,candidate in byid.items():assert candidate==all_candidates[cid]
    for name,value in manifest['source_sha256'].items():assert sha(root/'source'/name)==value;bindings['source/'+name]=value
    gate=read(Path('reference/gate.json'));assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    assert manifest['gate_sha256']==sha(root/'reference/gate.json')
    for name,value in gate['source_sha256'].items():assert manifest['source_sha256'][name]==value
    controls=read(Path('controls/control.json'));assert controls['status']=='pass' and controls['seed']==45 and controls['manifest_sha256']==sha(root/'manifest.json')
    if is_aircraft:
        assert controls['initial_parameters']['bitwise_equal'] and len(controls['steps'])==3 and controls['frozen_weights_unchanged']
        for step in controls['steps']:assert step['logits_loss_clip_bitwise_equal'] and step['gradients']['bitwise_equal'] and step['parameters_after_adamw']['bitwise_equal']
        splits=read(Path('split_manifest.json'));assert splits['effective_sizes']=={'train':3334,'val':3333,'test':3333}
        assert manifest['split_manifest_sha256']==sha(root/'split_manifest.json')
        references={}
        for seed,value in manifest['initialization_reference_sha256'].items():
            path=Path('initialization_reference/baseline')/f'seed_{seed}'/'result.json';reference=read(path);assert sha(root/path)==value
            assert reference['seed']==int(seed) and not reference['training_performed'] and not reference['historical_trained_baseline']
            assert reference['provenance']['checkpoint_sha256']==manifest['provenance']['checkpoint_sha256'];references[int(seed)]=reference
        for seed,value in manifest['calibration_manifest_sha256'].items():
            path=Path('calibration')/f'seed_{seed}'/'manifest.json';calibration=read(path);assert sha(root/path)==value
            meta=calibration['metadata'];assert meta['classifier_seed']==int(seed) and meta['initial_head_sha256']==references[int(seed)]['initial_head_sha256']
            assert meta['train_only'] and meta['population_size']==128 and meta['training_population_size']==3334
            assert meta['batch_count']==16 and meta['total_loss_weight']==128 and len(set(meta['example_ids']))==128
            assert set(meta['example_ids'])<={r['image'] for r in splits['splits']['train']}
            for name,file in calibration['module_files'].items():assert sha(file)==calibration['module_sha256'][name];checkpoints[str(Path(file).relative_to(root))]=sha(file)
    else:
        assert controls['steps']==5 and controls['initial_final_rng_trajectory_bitwise']
        for name,value in controls['evidence_sha256'].items():assert sha(root/'controls'/name)==value;bindings['controls/'+name]=value
        splits=read(Path(task+'_splits.json'));expected_queries=726 if task=='scifact' else 2582
        assert len(splits['train'])==expected_queries
        if manifest['calibration_reference']:
            path=Path(manifest['calibration_reference']);assert sha(path)==manifest['calibration_manifest_sha256']==parent_manifest['calibration_manifest_sha256']
            calibration=json.loads(path.read_text());assert calibration['metadata']['train_only']
            for name,file in calibration['module_files'].items():assert sha(file)==calibration['module_sha256'][name]
    if phase=='controls':return {'status':'pass','phase':phase,'input_sha256':bindings,'checkpoint_sha256':checkpoints}
    fits={};pattern='*/*/fit.json' if is_aircraft else '*/*/result.json'
    for path in sorted((root/'fits').glob(pattern)):
        fit=read(path.relative_to(root));candidate=byid[fit['candidate_id']];fits[fit['fit_id']]=fit;counts['fits']+=1
        assert fit['seed'] in registry['seeds'] and fit['learning_rate']==candidate['learning_rate'] and fit['spec']==candidate['spec']
        assert fit['manifest_sha256']==sha(root/'manifest.json')
        recipe=manifest['recipes'][str(fit['seed'])][fit['candidate_id']];assert fit['recipe_sha256']==recipe['recipe_sha256']
        if fit['status']=='numerical_failure':
            failure=fit['failure'];assert failure['type'] in ('RuntimeError','ValueError','_LinAlgError')
            assert any(token in failure['message'].lower() for token in ('nonfinite','failed to converge','did not converge','convergence failed','corda inverse','singular matrix','ill-conditioned'))
            assert fit['failure_stage'] in ('initialization','training')
            count_key='completed_epochs' if is_aircraft else 'completed_steps'
            log=Path(fit['artifact_directory'])/'epochs.jsonl' if is_aircraft else Path(fit['fit_id'])/'training.jsonl'
            rows=read(log,lines=True) if (root/log).exists() else []
            assert fit[count_key]==len(rows)
            counts['failed_fits']+=1;continue
        assert fit['status']=='pass'
        initial=fit['initialization']['initial_parity'] if is_aircraft else fit['initial_parity']
        assert initial['seed']==fit['seed'] and initial['bitwise_accepted_probe'] and initial['cpu_cuda_rng_equal']
        replay=fit['reload_validation'];assert replay['bitwise_checkpoint_probe'] and replay['initial_compensation_buffers_verified'] and replay['frozen_base_verified']
        replay_path=Path(fit['fit_id'])/'probe_replay.pt';assert sha(root/replay_path)==replay['probe_sha256']
        payload=torch.load(root/replay_path,map_location='cpu',weights_only=True)
        torch.testing.assert_close(payload['trained_probe'],payload['replayed_probe'],rtol=0,atol=0)
        checkpoints[str(replay_path)]=replay['probe_sha256']
        if is_aircraft:
            directory=Path(fit['artifact_directory']);row=read(directory/'result.json');assert row==fit['result']
            config=read(directory/'config.json');assert all(row[k]==v for k,v in config.items())
            assert row['epochs']==20 and row['stage']=='final' and row['evaluate_test'] and row['batch_size']==128
            assert row['label_smoothing']==.1 and row['gradient_clip_norm']==1 and row['head_learning_rate']==.001
            assert row['frozen_base_unchanged'] and row['frozen_state_before_sha256']==row['frozen_state_after_sha256']==row['initial_backbone_sha256']
            for name in ('initial_head_sha256','initial_backbone_sha256'):assert row[name]==references[fit['seed']][name]
            shapes=aircraft.trainable_shapes();assert row['trainable_names']==sorted(shapes)
            assert row['trainable_parameters']==sum(math.prod(shape) for shape in shapes.values())
            history=read(directory/'epochs.jsonl',lines=True);assert [r['epoch'] for r in history]==list(range(1,21))
            best=max(history,key=lambda r:(r['validation']['macro_class_accuracy'],-r['validation']['cross_entropy']))
            assert row['best_epoch']==best['epoch'] and row['best_validation']==best['validation']
            groups=row['optimizer_groups'];assert {g['group_name'] for g in groups}=={'adapter_factors','magnitudes','classifier'}
            names=[]
            for group in groups:
                names.extend(group['param_names']);assert group['weight_decay']==(0 if group['group_name']=='magnitudes' else .01)
                assert group['lr']==(.001 if group['group_name']=='classifier' else candidate['learning_rate'])
            assert len(names)==len(set(names))==len(shapes) and set(names)==set(shapes)
            for epoch,entry in enumerate(history):
                scale=(epoch+1)/2 if epoch<2 else .5*(1+math.cos(math.pi*(epoch-2)/18))
                assert np.allclose(entry['learning_rates'],[g['lr']*scale for g in groups],rtol=1e-14,atol=0)
                assert entry['train']['count']==3334 and entry['validation']['count']==3333
                assert math.isfinite(entry['train']['label_smoothed_cross_entropy']) and math.isfinite(entry['validation']['cross_entropy'])
            counts['epochs_checked']+=20;counts['optimizer_updates_checked']+=540
            aircraft.diagnostics(fit['initialization']['diagnostics'],candidate,recipe,fit['seed']);aircraft.diagnostics(row['adapter_diagnostics'],candidate,recipe,fit['seed'])
            checkpoint=directory/'best_adapter_and_head.pt';assert sha(root/checkpoint)==fit['checkpoint_sha256']==row['best_adapter_checkpoint_sha256']
            state=torch.load(root/checkpoint,map_location='cpu',weights_only=True);assert set(state)==set(shapes)
            assert aircraft.tensor_digest(state)==row['final_trainable_sha256']
            for name,value in state.items():assert tuple(value.shape)==shapes[name] and value.dtype==torch.float32 and torch.isfinite(value).all()
            prediction=read(directory/'test_predictions.json');labels=np.array([r['label'] for r in splits['splits']['test']]);assert np.array_equal(prediction['labels'],labels)
            guesses=np.array(prediction['predictions']);assert guesses.shape==labels.shape and ((0<=guesses)&(guesses<100)).all()
            correct=(guesses==labels).astype(float);nll=np.array(prediction['true_class_nll']);top5=np.array(prediction['top5_correct'],dtype=float)
            assert np.isfinite(nll).all() and (nll>=0).all() and np.isin(top5,[0,1]).all() and (top5>=correct).all()
            values={'count':3333,'correct':int(correct.sum()),'accuracy':float(correct.mean()),'cross_entropy':float(nll.mean()),'top5_accuracy':float(top5.mean()),
                'macro_class_accuracy':float((np.bincount(labels,weights=correct,minlength=100)/np.bincount(labels,minlength=100)).mean())}
            for key,value in values.items():assert abs(row['test'][key]-value)<3e-7
            counts['predictions_rescored']+=3333
        else:
            directory=Path(fit['fit_id']);steps_per_epoch=math.ceil(expected_queries/32);steps=5*steps_per_epoch
            history=read(directory/'training.jsonl',lines=True);assert len(history)==fit['training']['steps']==steps
            assert fit['training']['queries_seen']==expected_queries*5 and fit['frozen_weights_unchanged']
            warmup=max(1,round(steps*manifest['configuration']['warmup_fraction']))
            for index,row in enumerate(history,1):
                assert row['step']==index and row['epoch']==(index-1)//steps_per_epoch+1
                assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
                scale=index/warmup if index<=warmup else (steps-index+1)/max(1,steps-warmup)
                rates=row.get('learning_rates',[row.get('learning_rate')]);assert len(rates)==(2 if task=='nfcorpus' else 1)
                assert all(math.isclose(v,candidate['learning_rate']*scale,rel_tol=0,abs_tol=1e-15) for v in rates)
            counts['optimizer_updates_checked']+=steps
            retrieval.check_diagnostics(fit['initial_diagnostics'],candidate,recipe,fit['seed']);retrieval.check_diagnostics(fit['diagnostics'],candidate,recipe,fit['seed'])
            checkpoint=directory/'adapter.pt';assert sha(root/checkpoint)==fit['checkpoint_sha256']
            state=torch.load(root/checkpoint,map_location='cpu',weights_only=True)
            assert list(state)==fit['trainable_parameter_order'] and len(state)==36 and sum(t.numel() for t in state.values())==fit['trainable_parameters']==78336
            for name,value in state.items():
                suffix=name.rsplit('.',1)[-1];shape=(8,384) if suffix=='lora_A' else (384,8) if suffix=='lora_B' else (1,384)
                assert tuple(value.shape)==shape and value.dtype==torch.float32 and torch.isfinite(value).all()
            for dataset,results in fit['evaluation'].items():
                ds_splits=read(Path(dataset+'_splits.json'));qrels=read(Path(dataset+'_qrels.json'))
                for split,metrics in results.items():
                    if split not in ('validation','test'):continue
                    predictions=read(directory/dataset/(split+'_per_query.json'));byqid={r['query_id']:r for r in predictions}
                    assert len(byqid)==len(predictions)==len(ds_splits[split]) and set(byqid)==set(ds_splits[split])
                    values=[]
                    for qid in ds_splits[split]:
                        item=byqid[qid];score=retrieval.metrics(item['top_10'],qrels[qid]);values.append(score)
                        for key,value in score.items():assert math.isclose(value,item[key],rel_tol=0,abs_tol=1e-12)
                        assert len(item['scores'])==10 and all(math.isfinite(v) for v in item['scores']) and item['scores']==sorted(item['scores'],reverse=True)
                    for key in values[0]:assert math.isclose(sum(v[key] for v in values)/len(values),metrics[key],rel_tol=0,abs_tol=1e-12)
                    counts['predictions_rescored']+=len(values)
        checkpoints[str(checkpoint)]=sha(root/checkpoint)
    results=read(Path('results.json'));assert results['complete'] and results['manifest_sha256']==sha(root/'manifest.json')
    expected={(role,seed) for role in registry['roles'] for seed in registry['seeds']}
    assert len(results['runs'])==len(expected) and {(r['role'],r['seed']) for r in results['runs']}==expected
    assert len(fits)==results['unique_trained_fits']==len(byid)*5
    for row in results['runs']:
        assert row['candidate_id']==registry['roles'][row['role']]
        assert all(row[key]==value for key,value in fits[row['fit_id']].items())
        assert row['shared_roles']==[key for key,value in registry['roles'].items() if value==row['candidate_id']]
    return {'status':'pass','task':task,'phase':phase,'source_sha256':sha(__file__),'input_sha256':bindings,'checkpoint_sha256':checkpoints,'audit':counts,
        'scope':'Five untouchedconfirmationseeds after completepilotvalidation-onlyselection. Exactfullbudgets, trainabletensorpayloads, rawpredictionmetrics and deduplicatedthree-rolecoverage.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('controls','finals'),required=True);args=p.parse_args()
    torch.set_num_threads(2);result=audit(args.root,args.phase)
    path=args.root/('task_audit.json' if args.phase=='finals' else 'controls_audit.json');path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result.get('audit',{'status':'pass','phase':args.phase}),indent=2))
