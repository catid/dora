"""Independent audit of rank-axis up-factor normalization on accepted DoRA."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranked_metrics(ranking, relevant):
    assert len(ranking) == len(set(ranking)) == 10
    grades = [relevant.get(document, 0) for document in ranking]
    ideal = sorted(relevant.values(), reverse=True)[:10]
    dcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
    idcg = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
    assert idcg > 0 and all(grade > 0 for grade in relevant.values())
    return {'ndcg_at_10': dcg / idcg,
            'mrr_at_10': next((1 / (index + 1) for index, grade in enumerate(grades) if grade), 0),
            'recall_at_10': sum(grade > 0 for grade in grades) / len(relevant)}


def check_diagnostics(diagnostics, method):
    assert len(diagnostics)==12
    nora=method.endswith('_nora')
    for value in diagnostics.values():
        assert value['clamp_activations']==0 and math.isfinite(value['minimum_raw_norm']) and value['minimum_raw_norm']>0
        assert value['normalization_axis']==0 and value['norm_floor'] is None
        assert value['zero_initialized_factor']=='lora_A'
        assert value['magnitude_dtype']==value['factor_dtype']=='torch.float32'
        assert value['nora_enabled']==nora
        assert value['nora_factor']=='up' and value['nora_axis']==1 and value['nora_vector_count']==384
        assert value['nora_epsilon']==1e-12
        assert value['nora_initial_zero_vectors']==value['nora_initial_epsilon_vectors']==0
        if nora:
            assert math.isfinite(value['nora_minimum_raw_vector_norm']) and value['nora_minimum_raw_vector_norm']>=0
            assert 0<=value['nora_epsilon_vector_events']<=384*value['nora_norm_evaluations']
            assert value['nora_norm_evaluations']==value['norm_evaluations']>0
        else:
            assert value['nora_epsilon_vector_events']==value['nora_norm_evaluations']==0
            assert value['nora_minimum_raw_vector_norm'] is None


def audit(root):
    bindings = {}
    def read(relative, lines=False):
        path = root / relative
        bindings[str(relative)] = sha(path)
        text = path.read_text()
        return [json.loads(row) for row in text.splitlines()] if lines else json.loads(text)
    manifest = read(Path('manifest.json'))
    results = read(Path('results.json'))
    assert results['complete'] and results['manifest_sha256'] == sha(root / 'manifest.json')
    assert manifest['schema'] == results['schema'] == 'dora-up-nora-retrieval-v1'
    assert results['task'] == manifest['task'] and results['rank'] == manifest['rank'] == 8
    config = manifest['configuration']
    assert config['methods'] == ['up_base','up_nora']
    assert all(recipe['magnitude_fp32'] and not recipe['clamp_norm'] for recipe in manifest['recipes'].values())
    for method,recipe in manifest['recipes'].items():
        assert recipe['axis']==0 and not recipe['detach_norm'] and not recipe['norm_fp32'] and not recipe['factorized']
        assert not recipe['published_init']
        assert recipe['nora']==method.endswith('_nora') and recipe['nora_epsilon']==1e-12
        assert recipe['nora_factor']=='up' and recipe['nora_axis']==1
    gate = read(Path('reference/nora_up_validation.json'))
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    assert sha(root/'reference/nora_up_validation.json') == manifest['gate_evidence_sha256']
    assert gate['recipes'] == manifest['recipes']
    assert gate['core_source_sha256'] == manifest['source_sha256']['experiments/nora_up.py']
    assert gate['inherited_current_source_sha256'] == manifest['source_sha256']['experiments/nora_current.py']
    assert gate['inherited_next_source_sha256'] == manifest['source_sha256']['experiments/next_ablations.py']
    assert gate['inherited_core_source_sha256'] == manifest['source_sha256']['experiments/change_ablations.py']
    assert gate['standalone_source_sha256'] == manifest['source_sha256']['dora.py']
    assert config['seeds'] == [42,43,44] and config['epochs'] == 5 and config['batch_size'] == 32
    task = manifest['task']
    dataset = 'scifact' if task == 'scifact' else 'nfcorpus'
    fixed = .001 if task == 'scifact' else .0001
    rates = [.0001,.001] if task == 'scifact' else [.00003,.0001,.0003,.001]
    assert config['learning_rates'] == rates and config['fixed_lr'] == fixed
    expected_train_queries = 726 if task == 'scifact' else 2582
    splits = read(Path(f'{dataset}_splits.json'))
    assert len(splits['train']) == expected_train_queries
    epoch_steps = math.ceil(expected_train_queries / config['batch_size'])
    expected_steps = epoch_steps * config['epochs']
    warmup = max(1, round(expected_steps * config['warmup_fraction']))
    for name, expected in manifest['source_sha256'].items():
        path = Path('source') / name
        assert sha(root / path) == expected
        bindings[str(path)] = expected
    for name, expected in manifest['input_sha256'].items():
        assert sha(root / name) == expected
        bindings[name] = expected
    if task == 'nfcorpus':
        assert sha(root / 'hard_negatives.json') == manifest['hard_negatives_sha256']
        bindings['hard_negatives.json'] = sha(root / 'hard_negatives.json')
    endpoint = read(Path('endpoint_control/control.json'))
    assert endpoint['status'] == 'pass' and endpoint['manifest_sha256'] == sha(root / 'manifest.json')
    assert {row['method'] for row in endpoint['controls']} == set(config['methods'])
    for row in endpoint['controls']:
        assert row['steps'] == 5 and row['finite_trajectory_and_parameters']
        assert math.isfinite(row['first_step_gradient_norm'])
        check_diagnostics(row['diagnostics'],row['method'])
        if row['method'] == 'up_base':
            assert row['reference'] == 'current_dora'
            assert row['logged_loss_learning_rate_clip_norm_bitwise'] and row['initialized_and_final_adapter_tensors_bitwise']
            assert row['initial_rng_bitwise']
        else:
            assert row['reference'] is None
            assert row['raw_initial_tensors_rng_and_first_loss_match_baseline']
    endpoint_checkpoints = {}
    verified_endpoint_checkpoints = omitted_endpoint_checkpoints = 0
    for name,digest in endpoint['evidence_sha256'].items():
        path=Path('endpoint_control')/name
        if path.suffix == '.pt':
            endpoint_checkpoints[str(path)] = digest
            verified_endpoint_checkpoints += 1
        assert sha(root/path)==digest
        if path.suffix != '.pt':bindings[str(path)]=digest
    fits = {}
    checkpoints = {}
    verified_checkpoints = omitted_checkpoints = logged_steps = 0
    for path in sorted((root/'fits').glob('*/*/*/result.json')):
        relative=path.relative_to(root)
        row=read(relative)
        fit_id=str(relative.parent)
        assert row['fit_id']==fit_id and row['manifest_sha256']==sha(root/'manifest.json')
        method=row['method'];recipe=manifest['recipes'][method]
        core_recipe=recipe.get('base_recipe',recipe)
        assert row['recipe_sha256']==recipe['recipe_sha256']
        assert row['rank']==8 and row['seed'] in config['seeds'] and row['learning_rate'] in rates
        assert row['training']['steps']==expected_steps and row['training']['queries_seen']==expected_train_queries*5
        assert row['frozen_weights_unchanged']
        logs=read(relative.parent/'training.jsonl',lines=True)
        assert len(logs)==expected_steps and [v['step'] for v in logs]==list(range(1,expected_steps+1))
        for step,item in enumerate(logs,1):
            assert item['epoch']==(step-1)//epoch_steps+1
            assert math.isfinite(item['loss']) and math.isfinite(item['gradient_norm'])
            factor=step/warmup if step<=warmup else (expected_steps-step+1)/max(1,expected_steps-warmup)
            learning_rates=item.get('learning_rates',[item.get('learning_rate')])
            assert len(learning_rates)==(2 if task=='nfcorpus' else 1)
            assert all(math.isclose(value,row['learning_rate']*factor,abs_tol=1e-15,rel_tol=0) for value in learning_rates)
        logged_steps+=len(logs)
        if task=='nfcorpus':
            groups=row['training']['optimizer_groups']
            assert [g['group_name'] for g in groups]==['adapter_factors','magnitudes']
            assert [g['weight_decay'] for g in groups]==[config['weight_decay'],0]
            assert sum(g['parameter_count'] for g in groups)==row['trainable_parameters']
            assert all(g['lr']==row['learning_rate'] for g in groups)
        check_diagnostics(row['diagnostics'],method)
        if 'post_reload_diagnostics' in row:
            check_diagnostics(row['post_reload_diagnostics'],method)
        checkpoint=relative.parent/'adapter.pt'
        checkpoints[str(checkpoint)]=row['checkpoint_sha256']
        if (root/checkpoint).exists():
            assert sha(root/checkpoint)==row['checkpoint_sha256']
            state=torch.load(root/checkpoint,map_location='cpu',weights_only=True)
            assert list(state)==row['trainable_parameter_order']
            assert len(state)==36 and sum(t.numel() for t in state.values())==row['trainable_parameters']==78336
            factor_dtype=torch.bfloat16 if recipe.get('factor_storage')=='bfloat16' else torch.float32
            magnitude_dtype=torch.float32 if core_recipe['magnitude_fp32'] else torch.bfloat16
            for name,tensor in state.items():
                assert torch.isfinite(tensor).all()
                suffix=name.rsplit('.',1)[-1]
                shape=(8,384) if suffix=='lora_A' else (384,8) if suffix=='lora_B' else (384,1) if core_recipe['axis']==1 else (1,384)
                assert tuple(tensor.shape)==shape
                assert tensor.dtype==(magnitude_dtype if suffix=='m' else factor_dtype)
                if suffix==core_recipe['zero_initialized_factor']:
                    assert float(tensor.float().norm())>0
            verified_checkpoints+=1
        else:
            raise AssertionError(f'Missing required local checkpoint: {root/checkpoint}')
        fits[fit_id]=row
    assert len({row['frozen_sha256'] for row in fits.values()})==1
    expected_runs={(method,seed,lane) for method in config['methods'] for seed in config['seeds'] for lane in ('fixed','tuned')}
    assert {(r['method'],r['seed'],r['lane']) for r in results['runs']}==expected_runs and len(results['runs'])==len(expected_runs)
    for method in config['methods']:
        selection=read(Path('selections')/f'{method}.json')
        assert selection==results['tuning'][method]
        trials=selection['trials']
        assert [r['learning_rate'] for r in trials]==rates
        assert selection['test_used_for_selection'] is False and selection['fixed_learning_rate']==fixed
        assert selection['selection_metric']==f'{dataset}.validation.ndcg_at_10'
        for trial in trials:
            fit=fits[trial['fit_id']]
            assert fit['method']==method and fit['seed']==42 and fit['learning_rate']==trial['learning_rate']
            assert trial['checkpoint_sha256']==fit['checkpoint_sha256'] and trial['training_steps']==expected_steps
            assert trial['validation_ndcg_at_10']==fit['evaluation'][dataset]['validation']['ndcg_at_10']
        winner=max(trials,key=lambda r:(r['validation_ndcg_at_10'],-r['learning_rate']))
        assert selection['selected_learning_rate']==winner['learning_rate']
    for row in results['runs']:
        fit=fits[row['fit_id']]
        for key,value in fit.items():assert row[key]==value,(row['fit_id'],key)
        expected_rate=fixed if row['lane']=='fixed' else results['tuning'][row['method']]['selected_learning_rate']
        assert row['learning_rate']==expected_rate
        expected_lanes=[lane for lane,rate in (('fixed',fixed),('tuned',results['tuning'][row['method']]['selected_learning_rate'])) if rate==expected_rate]
        assert row['shared_lanes']==expected_lanes and row['reused_full_budget_validation_fit']==(row['seed']==42)
        assert 'test' in row['evaluation'][dataset]
    assert results['unique_final_fits']==len({r['fit_id'] for r in results['runs']})
    assert results['unique_trained_fits']==len(fits)
    rescored = 0
    for fit_id, row in fits.items():
        for name, evaluations in row['evaluation'].items():
            dataset_splits = read(Path(name + '_splits.json'))
            qrels = read(Path(name + '_qrels.json'))
            for split, metrics in evaluations.items():
                if split not in ('validation', 'test'): continue
                predictions = read(Path(fit_id) / name / (split + '_per_query.json'))
                byid = {item['query_id']: item for item in predictions}
                assert len(byid) == len(predictions) == len(dataset_splits[split])
                assert set(byid) == set(dataset_splits[split])
                computed = []
                for qid in dataset_splits[split]:
                    item = byid[qid]
                    scores = ranked_metrics(item['top_10'], qrels[qid])
                    for key,value in scores.items():
                        assert math.isclose(value,item[key],abs_tol=1e-12,rel_tol=0)
                    assert len(item['scores']) == 10 and all(math.isfinite(value) for value in item['scores'])
                    assert item['scores'] == sorted(item['scores'],reverse=True)
                    computed.append(scores)
                for key in ('ndcg_at_10','mrr_at_10','recall_at_10'):
                    assert math.isclose(sum(item[key] for item in computed)/len(computed),metrics[key],abs_tol=1e-12,rel_tol=0)
                rescored += len(computed)
    historical_root = Path('/var/tmp/dora-bench/nora_current_bench') / task
    historical = json.loads((historical_root/'results.json').read_text())
    parity = []
    unique = set()
    for method,reference_method in [('up_base','current_dora')]:
        for seed in config['seeds']:
            for lane in ('fixed','tuned'):
                current = next(row for row in results['runs'] if row['method']==method and row['seed']==seed and row['lane']==lane)
                prior = next(row for row in historical['runs'] if row['method']==reference_method and row['seed']==seed and row['lane']==lane)
                assert current['learning_rate'] == prior['learning_rate']
                if current['fit_id'] in unique: continue
                unique.add(current['fit_id'])
                olddir = historical_root/prior['fit_id'];newdir = root/current['fit_id']
                expected = torch.load(olddir/'adapter.pt',map_location='cpu',weights_only=True)
                actual = torch.load(newdir/'adapter.pt',map_location='cpu',weights_only=True)
                assert list(expected) == list(actual)
                for name,tensor in expected.items():
                    assert tensor.dtype == actual[name].dtype and tensor.shape == actual[name].shape
                    torch.testing.assert_close(actual[name],tensor,rtol=0,atol=0)
                pred_hashes = {}
                for dataset in ('nfcorpus','scifact') if task=='nfcorpus' else ('scifact',):
                    oldpred = olddir/dataset/'test_per_query.json';newpred = newdir/dataset/'test_per_query.json'
                    assert oldpred.read_bytes() == newpred.read_bytes()
                    pred_hashes[dataset] = {'reference_sha256':sha(oldpred),'current_sha256':sha(newpred)}
                parity.append({'method':method,'reference_method':reference_method,'seed':seed,
                               'learning_rate':current['learning_rate'],'shared_lanes':current['shared_lanes'],
                               'reference':str(olddir),'current':current['fit_id'],
                               'tensor_count':len(actual),'bitwise_tensors':True,'byte_identical_test_predictions':True,
                               'checkpoint_sha256':{'reference':sha(olddir/'adapter.pt'),'current':sha(newdir/'adapter.pt')},
                               'prediction_sha256':pred_hashes})
    parity_receipt = {'status':'pass','source_sha256':sha(__file__),'rows':parity,
                      'scope':'Fresh full-budget up_base exactly reproduces current_dora tensors bitwise and test predictions byte-identically; fixed and tuned lanes retained without duplicate fit counting.'}
    (root/'endpoint_quality_parity.json').write_text(json.dumps(parity_receipt,indent=2)+'\n')
    # Bind raw ranking predictions and all stored JSON independently of whether
    # metric replay consumes them; checkpoint payload hashes remain separate.
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.py','.txt') and path.name not in ('task_audit.json',):
            bindings[str(path.relative_to(root))]=sha(path)
    return {'status':'pass','task':task,'rank':8,'source_sha256':sha(__file__),
            'input_results_sha256':sha(root/'results.json'),'input_sha256':bindings,
            'checkpoint_sha256':checkpoints,'endpoint_checkpoint_sha256':endpoint_checkpoints,'artifact_roots':{'task':str(root)},
            'audit':{'recipes':len(config['methods']),'lane_records':len(results['runs']),'unique_trained_fits':len(fits),
                     'unique_final_fits':results['unique_final_fits'],'logged_optimizer_updates':logged_steps,
                     'checkpoint_payloads_verified':verified_checkpoints,'checkpoint_payloads_omitted':omitted_checkpoints,
                     'endpoint_checkpoint_payloads_verified':verified_endpoint_checkpoints,'endpoint_checkpoint_payloads_omitted':omitted_endpoint_checkpoints,
                     'full_budget_validation_only_selection':True,'fit_sharing_explicit':True,'norm_clamp_activations':0},
            'baseline_historical_parity':parity,
            'predictions_rescored':rescored,
            'scope':'Recomputed all saved query-ranking metrics and exact baseline parity against current_dora. Recomputed training-update/LR budgets, checked actual adapter tensors/shapes/dtypes and frozen attestations, reconstructed validation selection and shared fixed/tuned fits. Optimizer state payloads are not saved. Ranking metrics are independently rescored here from qrels/predictions. No training or model inference rerun.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    source=args.root/'audit_source'/Path(__file__).name
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    result=audit(args.root)
    (args.root/'task_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['audit'],indent=2))
