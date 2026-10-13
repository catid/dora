"""Independent staged-registry, budget, checkpoint and ranking audit for init retrieval."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments import audit_init_extension_metadata as extension_checks

LANES=('fixed_original_lr','validation_tuned')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def metrics(ranking,relevance):
    assert len(ranking)==len(set(ranking))==10
    grades=[relevance.get(doc,0) for doc in ranking]
    ideal=sorted(relevance.values(),reverse=True)[:10]
    denominator=sum((2**grade-1)/math.log2(i+2) for i,grade in enumerate(ideal))
    assert denominator>0
    return {'ndcg_at_10':sum((2**grade-1)/math.log2(i+2) for i,grade in enumerate(grades))/denominator,
            'mrr_at_10':next((1/(i+1) for i,grade in enumerate(grades) if grade),0),
            'recall_at_10':sum(grade>0 for grade in grades)/len(relevance)}


def spec_parts(spec):return spec.get('kind','random'),spec.get('config',spec)


def check_diagnostics(values,candidate,recipe,seed):
    assert len(values)==12
    kind,spec=spec_parts(candidate['spec'])
    orientation=spec.get('orientation','up')
    for name,row in values.items():
        assert row['recipe_sha256']==recipe['recipe_sha256']
        assert row['normalization_axis']==0 and row['norm_floor'] is None and row['clamp_activations']==0
        assert row['minimum_raw_norm']>0 and math.isfinite(row['minimum_raw_norm'])
        assert row['magnitude_dtype']==row['factor_dtype']=='torch.float32'
        init=row['initializer']
        assert init['factor_seed']==seed and init['module_path']==name
        if init['initializer_kind'] in ('warmstart','quantized','extension_warmstart'):
            assert kind==init['initializer_kind']
            for key in ('up_sha256','down_sha256','pretrained_weight_sha256'):assert len(init[key])==64
            assert init['magnitude_policy']==spec.get('magnitude_policy','original')
            if kind in ('warmstart','extension_warmstart'):
                assert not init['compensated']
                if kind=='warmstart':assert math.isfinite(init['initial_delta_relative_norm']) and init['initial_delta_relative_norm']>=0
            else:
                assert len(init['quantized_weight_sha256'])==64 and init['quantized_weight_sha256']!=init['pretrained_weight_sha256']
            continue
        if init['initializer_kind'] in ('data','random_both_nonzero','extension_centered'):
            assert init['compensated'] and init['initial_adapter_delta_is_zero']
            for factor in ('up','down'):
                assert len(init['initial_'+factor+'_sha256'])==64
                stats=init[factor+'_statistics'];gram=torch.tensor(stats['rank_gram'],dtype=torch.float64)
                assert gram.shape==(8,8) and torch.isfinite(gram).all()
                torch.testing.assert_close(gram,gram.T,rtol=0,atol=1e-10)
                eigen=torch.linalg.eigvalsh(gram).clamp_min(0)
                torch.testing.assert_close(eigen,torch.tensor(stats['rank_gram_eigenvalues'],dtype=torch.float64),rtol=1e-10,atol=1e-10)
                assert math.isfinite(stats['entry_rms']) and stats['entry_rms']>=0
            assert row['zero_initialized_factor'] in (None,'lora_A','lora_B')
            continue
        assert kind=='random' and init['initializer_kind']=='random'
        assert row['zero_initialized_factor']==('lora_A' if orientation=='up' else 'lora_B')
        assert init['random_factor_orientation']==orientation
        assert init['zero_factor_nonzero_entries']==0 and init['initial_adapter_delta_is_zero']
        assert init['random_vector_count']==384
        assert 0<=init['random_zero_vectors']<=384
        assert math.isfinite(init['random_factor_rms']) and init['random_factor_rms']>0
        denominator=(8 if spec.get('radial','none')=='unit_vectors' else 1 if spec.get('scaling','rank')=='absolute'
                     else 384 if spec.get('scaling','rank')=='original_input' or spec.get('scaling','rank')=='fan_in' and orientation=='down' else 8)
        assert math.isclose(init['target_entry_rms'],spec.get('scale',1)/math.sqrt(denominator),rel_tol=0,abs_tol=1e-15)
        assert math.isclose(init['random_factor_frobenius_norm'],init['random_factor_rms']*math.sqrt(384*8),rel_tol=1e-12)
        gram=torch.tensor(init['rank_gram'],dtype=torch.float64)
        assert gram.shape==(8,8) and torch.isfinite(gram).all()
        torch.testing.assert_close(gram,gram.T,rtol=0,atol=1e-10)
        eigenvalues=torch.linalg.eigvalsh(gram).clamp_min(0)
        torch.testing.assert_close(eigenvalues,torch.tensor(init['rank_gram_eigenvalues'],dtype=torch.float64),rtol=1e-10,atol=1e-10)
        weights=eigenvalues/eigenvalues.sum();positive=weights[weights>0]
        rank=float(torch.exp(-(positive*positive.log()).sum()))
        assert math.isclose(rank,init['rank_effective_rank'],rel_tol=1e-10,abs_tol=1e-10)
        if spec.get('normalization','analytic_rms')=='sample_rms' or spec.get('radial','none')=='unit_vectors':
            assert math.isclose(init['random_factor_rms'],init['target_entry_rms'],rel_tol=1e-6)


def audit(root,phase):
    bindings={};checkpoint_bindings={}
    own=root/'audit_source'/Path(__file__).name;own.parent.mkdir(exist_ok=True);own.write_bytes(Path(__file__).read_bytes());bindings[str(own.relative_to(root))]=sha(own)
    helper=root/'audit_source'/Path(extension_checks.__file__).name;helper.parent.mkdir(exist_ok=True);helper.write_bytes(Path(extension_checks.__file__).read_bytes());bindings[str(helper.relative_to(root))]=sha(helper)
    def read(relative,lines=False):
        path=root/relative;bindings[str(relative)]=sha(path)
        return [json.loads(line) for line in path.read_text().splitlines()] if lines else json.loads(path.read_text())
    manifest=read(Path('manifest.json'));registry=read(Path('reference/registry.json'))
    assert manifest['schema']=='dora-init-extension-retrieval-v1' and manifest['registry']==registry
    assert manifest['registry_sha256']==sha(root/'reference/registry.json')
    assert manifest['protocol_sha256']==registry['protocol_sha256']==sha(root/'reference/protocol.md')
    assert registry['schema']=='dora-init-extension-v1' and registry['task']==manifest['task']
    assert registry['rank']==8 and registry['seed']==42 and registry['seeds']==[42,43,44]
    candidates={row['candidate_id']:row for row in registry['candidates']}
    assert len(candidates)==len(registry['candidates']) and registry['control_id'] in candidates
    for name,value in manifest['source_sha256'].items():
        assert sha(root/'source'/name)==value;bindings['source/'+name]=value
    gate=read(Path('reference/gate.json'))
    assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    assert sha(root/'reference/gate.json')==manifest['gate_evidence_sha256']
    for name,value in gate['source_sha256'].items():assert manifest['source_sha256'][name]==value
    for name,value in manifest['input_sha256'].items():assert sha(root/name)==value;bindings[name]=value
    config=manifest['configuration'];task=manifest['task']
    expected_queries=726 if task=='scifact' else 2582
    expected_rates=[.0001,.001] if task=='scifact' else [.00003,.0001,.0003,.001]
    assert registry['learning_rates']==expected_rates
    fixed=.001 if task=='scifact' else .0001
    assert registry['fixed_lr']==config['fixed_lr']==fixed
    assert config['epochs']==5 and config['batch_size']==32 and config['seeds']==[42,43,44]
    splits=read(Path(task+'_splits.json'));assert len(splits['train'])==expected_queries
    steps_per_epoch=math.ceil(expected_queries/32);steps=5*steps_per_epoch
    warmup=max(1,round(steps*config['warmup_fraction']))
    if task=='nfcorpus':assert sha(root/'hard_negatives.json')==manifest['hard_negatives_sha256']
    if manifest['calibration_manifest_sha256'] is not None:
        calibration=read(Path('calibration/manifest.json'))
        assert sha(root/'calibration/manifest.json')==manifest['calibration_manifest_sha256']
        extension_checks.calibration(calibration,task,manifest['source_sha256']['experiments/init_extension_observers.py'])
        meta=calibration['metadata'];assert meta['train_only'] and meta['population_size']==128 and meta['batch_count']==16
        assert meta['batch_size']==8 and meta['calibration_seed']==20261009 and meta['total_loss_weight']==128
        assert len(meta['example_ids'])==len(set(meta['example_ids']))==128
        assert set(meta['example_ids'])<=set(splits['train'])
        assert not set(meta['example_ids'])&(set(splits['validation'])|set(splits['test']))
        assert len(calibration['module_files'])==12
        for name,path in calibration['module_files'].items():
            assert sha(path)==calibration['module_sha256'][name]
            statistics=torch.load(path,map_location='cpu',weights_only=True)
            assert statistics['population_size']==128 and statistics['batch_count']==16
            assert statistics['forward_count']==statistics['backward_count']==32
            for key,tensor in statistics.items():
                if torch.is_tensor(tensor):assert torch.isfinite(tensor).all()
            checkpoint_bindings[str(Path(path))]=sha(path)
    assert registry['study_lane']==manifest['study_lane']=='extension'
    controls=read(Path('controls/control.json'))
    assert controls['status']=='pass' and controls['manifest_sha256']==sha(root/'manifest.json')
    accepted=[row for row in controls['controls'] if row['accepted_historical_bitwise_trajectory']]
    assert len(accepted)==1 and accepted[0]['candidate_id']==registry['control_id']
    assert accepted[0]['steps_completed']==accepted[0]['steps_requested']==5
    for row in controls['controls']:
        lane=candidates[row['candidate_id']]['initialization_lane']
        if row.get('failure_stage')=='initialization':
            assert row['status']=='numerical_failure';extension_checks.numerical_failure(row['failure']);continue
        assert row['training_rng_matches_accepted']
        if lane=='centered':assert row['initial_probe_bitwise_accepted']
        else:
            assert row['study_lane']==lane
            assert all(math.isfinite(v) and v>=0 for v in row['initial_probe_drift'].values())
        assert row['tested_learning_rate']==fixed
        check_diagnostics(row['initial_diagnostics'],candidates[row['candidate_id']],manifest['recipes']['42'][row['candidate_id']],42)
        extension_checks.diagnostics(row['initial_diagnostics'],candidates[row['candidate_id']],calibration,task)
        if row['status']=='pass':
            assert row['finite_trajectory_and_parameters'] and row['steps_completed']==row['steps_requested']
            check_diagnostics(row['diagnostics'],candidates[row['candidate_id']],manifest['recipes']['42'][row['candidate_id']],42)
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
        assert treatment['frozen_sha256']==control['frozen_sha256']
        left=torch.load(root/'controls'/(treatment['candidate_id']+'_initial.pt'),map_location='cpu',weights_only=True)['probe']
        right=torch.load(root/'controls'/(control['candidate_id']+'_initial.pt'),map_location='cpu',weights_only=True)['probe']
        expected={'maximum_absolute_error':float((left-right).abs().max()),'mse':float((left-right).square().mean()),'relative_l2':float((left-right).norm()/right.norm())}
        for key,value in expected.items():assert math.isclose(contrast[key],value,rel_tol=0,abs_tol=1e-10)
    for name,value in controls['evidence_sha256'].items():
        path=Path('controls')/name;assert sha(root/path)==value
        (checkpoint_bindings if path.suffix=='.pt' else bindings)[str(path)]=value
    if phase=='controls':return {'status':'pass','phase':phase,'input_sha256':bindings,'checkpoint_sha256':checkpoint_bindings}
    fits={};logged=predictions_scored=failed=0
    for path in sorted((root/'fits').glob('*/*/result.json')):
        relative=path.relative_to(root);row=read(relative);candidate=candidates[row['candidate_id']];lane=candidate['initialization_lane']
        assert row['fit_id']==str(relative.parent) and row['method']==candidate['candidate_id']
        assert row['family_id']==candidate['family_id'] and row['spec']==candidate['spec']
        assert row['learning_rate']==candidate['learning_rate'] and row['rank']==8 and row['seed'] in (42,43,44)
        assert row['candidate_sha256']==digest({'spec':candidate['spec'],'learning_rate':candidate['learning_rate']})
        assert row['manifest_sha256']==sha(root/'manifest.json')
        recipe=manifest['recipes'][str(row['seed'])][row['method']]
        assert row['recipe_sha256']==recipe['recipe_sha256']
        log=read(relative.parent/'training.jsonl',lines=True) if (root/relative.parent/'training.jsonl').exists() else []
        for i,item in enumerate(log,1):
            assert item['step']==i and item['epoch']==(i-1)//steps_per_epoch+1
            assert math.isfinite(item['loss']) and math.isfinite(item['gradient_norm'])
            scale=i/warmup if i<=warmup else (steps-i+1)/max(1,steps-warmup)
            rates=item.get('learning_rates',[item.get('learning_rate')])
            assert len(rates)==(2 if task=='nfcorpus' else 1)
            assert all(math.isclose(v,row['learning_rate']*scale,rel_tol=0,abs_tol=1e-15) for v in rates)
        logged+=len(log)
        if row['status']=='numerical_failure':
            extension_checks.numerical_failure(row['failure'])
            assert row['completed_steps']==len(log) and row['evaluation']=={}
            failed+=1;fits[row['fit_id']]=row;continue
        assert row['status']=='pass' and row['frozen_weights_unchanged']
        if candidate['spec'].get('kind')=='quantized':assert row['deterministic_recreation_verified']
        if lane!='centered':assert set(row['initial_evaluation'][task])>={'validation'}
        assert len(log)==row['training']['steps']==steps
        assert row['training']['queries_seen']==expected_queries*5
        check_diagnostics(row['initial_diagnostics'],candidate,recipe,row['seed'])
        extension_checks.diagnostics(row['initial_diagnostics'],candidate,calibration,task)
        check_diagnostics(row['diagnostics'],candidate,recipe,row['seed'])
        if 'post_reload_diagnostics' in row:check_diagnostics(row['post_reload_diagnostics'],candidate,recipe,row['seed'])
        checkpoint=relative.parent/'adapter.pt';assert sha(root/checkpoint)==row['checkpoint_sha256']
        state=torch.load(root/checkpoint,map_location='cpu',weights_only=True)
        assert list(state)==row['trainable_parameter_order']
        assert len(state)==36 and sum(t.numel() for t in state.values())==row['trainable_parameters']==78336
        for name,tensor in state.items():
            suffix=name.rsplit('.',1)[-1]
            assert tuple(tensor.shape)==((8,384) if suffix=='lora_A' else (384,8) if suffix=='lora_B' else (1,384))
            assert tensor.dtype==torch.float32 and torch.isfinite(tensor).all()
        checkpoint_bindings[str(checkpoint)]=row['checkpoint_sha256']
        reload=row['reload_validation'];assert reload['bitwise_checkpoint_probe'] and reload['initial_factors_recreated'] and reload['frozen_base_verified']
        probe_path=relative.parent/'probe_replay.pt';assert sha(root/probe_path)==reload['probe_sha256']
        probe=torch.load(root/probe_path,map_location='cpu',weights_only=True)
        torch.testing.assert_close(probe['trained_probe'],probe['replayed_probe'],rtol=0,atol=0)
        checkpoint_bindings[str(probe_path)]=reload['probe_sha256']
        for evaluation_prefix,evaluation_values in [('',row['evaluation']),('initial',row.get('initial_evaluation',{}))]:
            for dataset,values in evaluation_values.items():
                ids=read(Path(dataset+'_splits.json'));qrels=read(Path(dataset+'_qrels.json'))
                for split,scores in values.items():
                    if split not in ('validation','test'):continue
                    if phase=='pilots':assert split=='validation'
                    predictions=read(relative.parent/evaluation_prefix/dataset/(split+'_per_query.json'))
                    byid={r['query_id']:r for r in predictions}
                    assert len(byid)==len(predictions)==len(ids[split]) and set(byid)==set(ids[split])
                    computed=[]
                    for qid in ids[split]:
                        item=byid[qid];computed.append(metrics(item['top_10'],qrels[qid]))
                        for key,value in computed[-1].items():assert math.isclose(value,item[key],rel_tol=0,abs_tol=1e-12)
                        assert len(item['scores'])==10 and all(math.isfinite(v) for v in item['scores'])
                        assert item['scores']==sorted(item['scores'],reverse=True)
                    for key in computed[0]:assert math.isclose(sum(r[key] for r in computed)/len(computed),scores[key],rel_tol=0,abs_tol=1e-12)
                    predictions_scored+=len(computed)
        fits[row['fit_id']]=row
    if phase=='fits':return {'status':'pass','phase':phase,'task':task,'source_sha256':sha(__file__),
        'input_sha256':bindings,'checkpoint_sha256':checkpoint_bindings,
        'audit':{'fits':len(fits),'failed_fits':failed,'logged_optimizer_updates':logged,'predictions_rescored':predictions_scored},
        'scope':'Engineering partial-fit audit; no registry selection or completion claim.'}
    selection=read(Path('selections.json'));pilot_gate=read(Path('pilots_complete.json'))
    assert pilot_gate['status']=='pass' and pilot_gate['selections_sha256']==sha(root/'selections.json')
    assert selection['all_pilots_complete_before_test'] and not selection['test_used_for_selection']
    assert selection['registry_sha256']==digest(registry)
    assert len(selection['pilots'])==len(candidates)
    for row in selection['pilots']:
        candidate=candidates[row['candidate_id']];path=Path(row['fit_id'])/'pilot.json';pilot=read(path)
        assert row['pilot_sha256']==sha(root/path)
        assert pilot['seed']==42 and pilot['status']==row['status']
        assert pilot['manifest_sha256']==sha(root/'manifest.json')
        if row['status']=='pass':
            assert row['selection_value']==pilot['evaluation'][task]['validation']['ndcg_at_10']
            assert 'test' not in pilot['evaluation'][task]
            assert row['checkpoint_sha256']==pilot['checkpoint_sha256']
        else:assert row['status']=='numerical_failure' and row['selection_value'] is None
    pilot_by_id={r['candidate_id']:r for r in selection['pilots']}
    reconstructed={}
    for family in sorted({c['family_id'] for c in candidates.values()}):
        reconstructed[family]={}
        for lane in LANES:
            eligible=[c for c in candidates.values() if c['family_id']==family and pilot_by_id[c['candidate_id']]['status']=='pass'
                and (lane!='fixed_original_lr' or c['learning_rate']==fixed)]
            reconstructed[family][lane]=min(eligible,key=lambda c:(-pilot_by_id[c['candidate_id']]['selection_value'],c['learning_rate'],c['candidate_id']))['candidate_id'] if eligible else None
    assert selection['selections']==reconstructed
    parity=[];unavailable_history=[]
    historical=Path('/var/tmp/dora-bench/nora_current_bench')/task
    for fit_id,row in fits.items():
        kind,spec=spec_parts(row['spec'])
        if kind!='random' or spec.get('mode','accepted')!='accepted':continue
        prior=historical/'fits/current_dora'/f"lr_{row['learning_rate']:g}"/f"seed_{row['seed']}"
        assert row['status']=='pass'
        prior=extension_checks.exact_history([prior],{'seed':row['seed'],'learning_rate':row['learning_rate'],'rank':8})
        if prior is None:
            unavailable_history.append({'fit_id':fit_id,'reason':'No historical accepted fit with exact seed and learning rate'})
            continue
        previous=json.loads((prior/'result.json').read_text());assert previous['training']['steps']==steps
        expected=torch.load(prior/'adapter.pt',map_location='cpu',weights_only=True)
        actual=torch.load(root/fit_id/'adapter.pt',map_location='cpu',weights_only=True)
        assert list(expected)==list(actual)
        for name,value in expected.items():torch.testing.assert_close(value,actual[name],rtol=0,atol=0)
        prediction_hashes={}
        for dataset,ev in row['evaluation'].items():
            for split in ('validation','test'):
                if split not in ev:continue
                old=prior/dataset/(split+'_per_query.json');new=root/fit_id/dataset/(split+'_per_query.json')
                if old.exists():
                    assert old.read_bytes()==new.read_bytes();prediction_hashes[dataset+'/'+split]=sha(new)
                else:unavailable_history.append({'fit_id':fit_id,'dataset':dataset,'split':split,'reason':'Historical fullbudget fit has no saved predictions for this split'})
        parity.append({'fit_id':fit_id,'reference':str(prior),'tensor_count':36,'bitwise_tensors':True,
                       'byte_identical_predictions':True,'prediction_sha256':prediction_hashes})
    if phase=='finals':
        results=read(Path('results.json'))
        assert results['complete'] and results['selections_sha256']==sha(root/'selections.json')
        expected={(family,lane,seed) for family,choices in reconstructed.items() for lane,candidate in choices.items() if candidate is not None for seed in (42,43,44)}
        assert len(results['runs'])==len(expected) and {(r['family_id'],r['lane'],r['seed']) for r in results['runs']}==expected
        for row in results['runs']:
            assert row['candidate_id']==reconstructed[row['family_id']][row['lane']]
            assert all(row[k]==v for k,v in fits[row['fit_id']].items())
        pairs=results.get('paired_control_fits',[])
        expected_pairs={(family,cid,seed) for family,choices in reconstructed.items() for cid in set(choices.values())
            if cid is not None and candidates[cid]['comparison_control_id']!=cid for seed in (42,43,44)}
        assert len(pairs)==len(expected_pairs) and {(r['for_family_id'],r['for_candidate_id'],r['seed']) for r in pairs}==expected_pairs
        for pair in pairs:
            treatment=candidates[pair['for_candidate_id']];control=candidates[treatment['comparison_control_id']]
            assert pair['candidate_id']==control['candidate_id'] and pair['learning_rate']==treatment['learning_rate']
            assert all(pair[key]==value for key,value in fits[pair['fit_id']].items())
            treatment_fit=next(value for value in fits.values() if value['candidate_id']==pair['for_candidate_id'] and value['seed']==pair['seed'])
            if pair['status']==treatment_fit['status']=='pass':
                assert pair['frozen_sha256']==treatment_fit['frozen_sha256']
        assert results['unique_trained_fits']==len(fits)
        assert results['unique_final_fits']==len({r['fit_id'] for r in results['runs']+pairs})
    return {'status':'pass','phase':phase,'task':task,'source_sha256':sha(__file__),
        'input_sha256':bindings,'checkpoint_sha256':checkpoint_bindings,'artifact_roots':{'task':str(root)},
        'audit':{'candidates':len(candidates),'families':len(reconstructed),'fits':len(fits),'failed_fits':failed,
                 'logged_optimizer_updates':logged,'predictions_rescored':predictions_scored,
                 'checkpoint_payloads_verified':len(checkpoint_bindings),'historical_accepted_fits_verified':len(parity)},
        'historical_control_parity':parity,'unavailable_historical_parity':unavailable_history,
        'scope':'Independent validation selection, immutablepilot and registry binding, fullupdate/rate budgets, real checkpoint tensors, rawranking rescore and acceptedbaseline historicalidentity. No inference rerun; numericalfailures remain failures.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',required=True,choices=('controls','fits','pilots','finals'));args=p.parse_args()
    torch.set_num_threads(2)
    source=args.root/'audit_source'/Path(__file__).name;source.parent.mkdir(exist_ok=True);source.write_bytes(Path(__file__).read_bytes())
    result=audit(args.root,args.phase)
    destination=args.root/('task_audit.json' if args.phase=='finals' else args.phase+'_audit.json')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result.get('audit',{'status':result['status'],'phase':args.phase}),indent=2))
