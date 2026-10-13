"""Strict independent initializer ViGGO artifact, history and reload audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from experiments import extraction as task
from experiments import init_common as common
from experiments import init_extraction as runner
from experiments import nora_current as old
from experiments import audit_init_norms as norm_audit_helpers
from experiments.check_nora_up_extraction_history import check_pair


def predictions(directory, expected):
    rows = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == len(expected) and [row['id'] for row in rows] == list(expected)
    valid = schema = exact = tp = predicted = truth = 0
    for row in rows:
        gold = expected[row['id']]
        assert row['gold'] == gold['target'] and row['utterance'] == gold['utterance']
        parsed = None
        try:
            parsed = json.loads(row['text'].strip())
            valid += 1
        except (TypeError, ValueError):
            pass
        assert parsed == row['parsed']
        good = (isinstance(parsed,dict) and set(parsed).issubset(task.ALLOWED_KEYS)
                and all(isinstance(value,str) for value in parsed.values())
                and parsed.get('dialogue_act') in task.ALLOWED_ACTS)
        schema += int(good)
        pred = parsed if good else {}
        match = pred == gold['target']
        exact += int(match)
        assert row['exact_match'] == match
        tp += len(set(pred.items()) & set(gold['target'].items()))
        predicted += len(pred); truth += len(gold['target'])
    return {'count':len(rows),'valid_json':valid/len(rows),'schema_valid':schema/len(rows),
            'exact_match':exact/len(rows),'slot_micro_f1':2*tp/max(1,predicted+truth),
            'correct_slots':tp,'predicted_slots':predicted,'gold_slots':truth}


def snapshot(model, adapter_types):
    """Independent serialization of the initializer and compensation buffers."""
    modules = {}
    for name,module in model.named_modules():
        if isinstance(module,adapter_types):
            names = [key for key,p in module.named_parameters(recurse=False) if p.requires_grad]
            buffers = list(module.named_buffers(recurse=False))
            modules[name] = {'trainable_names':names,
                'trainable_sha256':common.tensor_digest((key,p) for key,p in module.named_parameters(recurse=False) if p.requires_grad),
                'buffer_names':[key for key,_ in buffers], 'buffer_sha256':common.tensor_digest(buffers),
                'frozen_weight_sha256':common.tensor_digest([('weight',module.weight)])}
    return modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('/var/tmp/dora-bench/nora_current_bench/extraction'))
    parser.add_argument('--phase',choices=('controls','pilots','finals'),default='finals')
    args = parser.parse_args();root = args.root
    manifest = common.read(root/'manifest.json');manifest_sha = common.sha(root/'manifest.json')
    assert manifest['schema'] == 'dora-init-extraction-v1'
    registry = common.load_registry(root/'registry.json',root/'protocol.md','extraction')
    study_lane = runner.validate_study_lane(registry)
    assert manifest.get('study_lane','centered') == study_lane
    assert registry == manifest['registry']
    assert common.sha(root/'registry.json') == manifest['registry_file_sha256']
    assert common.sha(root/'protocol.md') == manifest['protocol_sha256']
    assert common.sha(args.reference/'manifest.json') == manifest['reference_manifest_sha256']
    historical = common.read(args.reference/'manifest.json')
    for name,digest in manifest['source_sha256'].items():
        assert common.sha(root/'source'/name) == common.sha(name) == digest, name
    gate = common.read(root/'backend_gate.json')
    assert common.sha(root/'backend_gate.json') == manifest['backend_gate_sha256']
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    for name,digest in gate['source_sha256'].items():
        assert manifest['source_sha256'][name] == digest
    config = manifest['configuration']
    assert config['rank'] == 8 and config['pilot_steps'] == 48 and config['steps'] == 256
    assert config['batch_size'] == 8 and config['eval_every'] == 64
    assert config['learning_rates'] == [.0001,.0003]
    assert manifest['dataset'] == historical['dataset']
    assert common.sha(root/'split_manifest.json') == manifest['dataset']['split_manifest_sha256']
    split = common.read(root/'split_manifest.json')
    assert {key:len(rows) for key,rows in split.items()} == {'train':1024,'validation':128,'test':256}
    ids,texts = set(),set()
    for rows in split.values():
        these_ids = {row['id'] for row in rows}
        these_texts = {task.normalized_text(row['utterance']) for row in rows}
        assert len(these_ids) == len(these_texts) == len(rows)
        assert not ids & these_ids and not texts & these_texts
        ids.update(these_ids);texts.update(these_texts)
    candidates = {row['candidate_id']:row for row in registry['candidates']}
    calibration = common.read(root/'calibration/manifest.json') if (root/'calibration/manifest.json').exists() else None
    modules = [f'model.layers.{layer}.self_attn.{projection}' for layer in range(36) for projection in ('q_proj','v_proj')]
    if calibration:
        import random
        chosen = list(range(len(split['train'])));random.Random(20261009).shuffle(chosen)
        chosen_ids = [split['train'][index]['id'] for index in chosen[:128]]
        assert calibration['metadata']['example_ids'] == chosen_ids
        assert calibration['metadata']['calibration_split'] == 'train only'
        assert calibration['metadata']['task_source_manifest'] == manifest['source_sha256']
        assert calibration['metadata']['source_sha256'] == manifest['source_sha256']['experiments/init_calibration.py']
        assert calibration['metadata']['training_population_size'] == len(split['train'])
        assert list(calibration['module_files']) == modules
        assert sum(batch['target_tokens'] for batch in calibration['metadata']['batches']) == calibration['metadata']['total_loss_weight']
        for name,path in calibration['module_files'].items():
            assert common.sha(path) == calibration['module_sha256'][name]
            stats = torch.load(path,map_location='cpu',weights_only=True)
            assert stats['population_size'] == 128 and stats['batch_count'] == 16
            assert stats['forward_count'] == stats['backward_count'] == 16
            assert stats['input_count'] == stats['output_count'] == stats['output_grad_count']
            assert len(stats['input_gram_batches']) == 16
            for key,value in stats.items():
                if torch.is_tensor(value):assert torch.isfinite(value).all(), (name,key)
                elif isinstance(value,list):assert all(torch.isfinite(part).all() for part in value)
    for name,digest in manifest['checkpoint_sha256'].items():
        assert common.sha(hf_hub_download(task.MODEL,name,revision=task.MODEL_REVISION)) == digest
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(task.MODEL,revision=task.MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    records,provenance = task.data(SimpleNamespace(**config,output=root),tokenizer)
    assert provenance == manifest['dataset']
    model = AutoModelForCausalLM.from_pretrained(task.MODEL,revision=task.MODEL_REVISION,
                dtype=torch.bfloat16,attn_implementation='sdpa').cuda().requires_grad_(False)
    frozen_sha = task.frozen_digest(model)
    original_projections = runner.original_projection_parameters(model)
    if calibration:
        group_std={suffix:round(sum(float(weight.detach().float().std()) for name,(weight,bias) in original_projections.items()
            if name.endswith('.'+suffix))/sum(name.endswith('.'+suffix) for name in original_projections),3)
            for suffix in ('q_proj','v_proj')}
        assert calibration['metadata']['inilora_group_std']==group_std
    shapes = {}
    for name in modules:
        output = 2048 if name.endswith('q_proj') else 256
        for suffix,shape in [('m',(1,2048)),('lora_B',(output,8)),('lora_A',(8,2048))]:
            shapes[name+'.'+suffix] = shape
    expected_gold = {row['id']:row for row in split['test']}
    checks=[];history=[];checkpoint_hashes={};row_count=0;prediction_count=0;failed=[];audited_fits=set()

    def check(directory,candidate,seed,steps,test=False,reference_backend=False):
        nonlocal row_count,prediction_count
        if str(directory) in audited_fits:return
        audited_fits.add(str(directory))
        receipt = common.read(directory/'receipt.json')
        assert receipt['candidate_id'] == candidate['candidate_id'] and receipt['seed'] == seed
        runner.verify_receipt(root,receipt,manifest_sha)
        if receipt['status'] != 'pass':
            assert steps != 5 and receipt['status']=='failed_numerically'
            error_class={'RuntimeError':RuntimeError,'ValueError':ValueError}[receipt['exception_type']]
            error=error_class(receipt['error'])
            assert runner.initialization_failure(error) or runner.numerical_failure(error)
            assert receipt['failure_stage'] in ('initialization','training')
            if 'initialization_failure_from_control_sha256' in receipt:
                assert receipt['initialization_failure_from_control_sha256']==common.sha(root/'controls/initial_probes.json')
                assert receipt['failure_stage']=='initialization' and receipt['steps_completed']==0
            else:
                log=directory/'training.jsonl'
                assert receipt['steps_completed']==(len(log.read_text().splitlines()) if log.exists() else 0)
            assert 0<=receipt['steps_completed']<=steps
            failed.append({'artifact_directory':str(directory.relative_to(root)),'receipt':receipt})
            return
        assert receipt['steps'] == steps and receipt['evaluate_test'] == test
        result = common.read(directory/'result.json')
        assert result['seed'] == seed and result['steps'] == steps and result['lr'] == candidate['learning_rate']
        assert result['rank'] == 8 and result['frozen_weights_unchanged']
        kind,_ = common.spec_parts(candidate['spec'])
        initial = common.read(directory/'initialization.json')
        fit_frozen = initial.get('frozen_sha256',frozen_sha)
        assert result['frozen_sha256'] == fit_frozen
        if kind!='quantized':assert fit_frozen == frozen_sha
        assert result['trainable_parameters'] == sum(math.prod(shape) for shape in shapes.values())
        assert math.isfinite(result['train_and_validation_seconds']) and result['train_and_validation_seconds'] > 0
        logs = [json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
        assert [row['step'] for row in logs] == list(range(1,steps+1))
        for row in logs:
            assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) and row['gradient_norm'] >= 0
            assert row['lr'] == result['lr']*min(1.,row['step']/max(1,steps*.1))
        eval_every = 5 if steps == 5 else config['eval_every']
        validation = [row for row in logs if 'validation_nll' in row]
        assert [row['step'] for row in validation] == [step for step in range(1,steps+1) if step%eval_every==0 or step==steps]
        assert all(math.isfinite(row['validation_nll']) for row in validation)
        selected = min(validation,key=lambda row:row['validation_nll'])
        assert result['selected_step'] == selected['step'] and result['validation_nll'] == selected['validation_nll']
        assert receipt['selection_value'] == result['validation_nll']
        row_count += len(logs)
        backend = common.backend_for(candidate,seed,calibration)
        if reference_backend:
            backend = SimpleNamespace(adapter_types=backend.adapter_types,
                inject_adapters=lambda model,method,rank,targets:old.inject_adapters(model,'current_dora',rank,targets),
                recipe_manifest=lambda method:old.recipe_manifest('current_dora'))
        assert result['recipe'] == receipt['recipe'] == backend.recipe_manifest(candidate['candidate_id'])
        state = torch.load(directory/'adapter.pt',map_location='cpu',weights_only=True)
        assert list(state) == list(shapes) and len(state) == 216
        for name,value in state.items():
            assert tuple(value.shape) == shapes[name] and value.dtype == torch.float32 and torch.isfinite(value).all()
        checkpoint_hashes[str((directory/'adapter.pt').relative_to(root))] = common.sha(directory/'adapter.pt')
        diagnostic = result['norm_diagnostics'];assert list(diagnostic) == modules
        for name,d in diagnostic.items():
            assert d['normalization_axis'] == 0 and d['norm_floor'] is None and d['clamp_activations'] == 0
            # This legacy counter includes magnitude initialization. Recreate
            # the initializer below to explain any zero and check denominators.
            assert d['minimum_raw_norm'] >= 0 and math.isfinite(d['minimum_raw_norm'])
            assert d['factor_dtype'] == d['magnitude_dtype'] == 'torch.float32'
            assert d['norm_evaluations'] > steps
            assert d['recipe_sha256'] == result['recipe']['recipe_sha256']
            if not reference_backend:
                assert d['initializer']['module_path'] == name and d['initializer']['factor_seed'] == seed
                if study_lane=='centered':assert d['initializer']['initial_adapter_delta_is_zero']
        runner.restore_original_projections(model,backend,original_projections,frozen_sha)
        task.seed_all(seed)
        backend.inject_adapters(model,candidate['candidate_id'],rank=8,targets=('q_proj','v_proj'))
        recreated = snapshot(model,backend.adapter_types)
        norm_evidence = {name:norm_audit_helpers.audit_initial_norm(diagnostic[name],module,candidate['spec'])
                         for name,module in model.named_modules() if isinstance(module,backend.adapter_types)}
        assert set(norm_evidence) == set(diagnostic)
        reload = common.read(directory/'reload_validation.json')
        assert initial['status'] == reload['status'] == 'pass'
        assert initial['modules'] == reload['initial_modules_recreated'] == recreated
        assert reload['checkpoint_tensors'] == 216 and reload['logits_bitwise_equal']
        assert reload['probe_split'] == 'train' and reload['probe_ids'] == [row['id'] for row in records['train'][:2]]
        assert reload['frozen_sha256'] == fit_frozen == task.frozen_digest(model)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad:p.copy_(state[name])
        for name,module in model.named_modules():
            if isinstance(module,backend.adapter_types):
                norm_evidence[name]['minimum_checkpoint_forward_denominator'] = norm_audit_helpers.forward_denominator_minimum(module)
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits = model(**task.batch(records['train'][:2],tokenizer),use_cache=False).logits.detach().cpu()
        assert list(logits.shape) == reload['probe_logits_shape']
        assert common.tensor_digest([('logits',logits)]) == reload['probe_logits_sha256']
        if test:
            assert math.isfinite(result['test_nll'])
            scored = predictions(directory,expected_gold)
            for key,value in scored.items():assert result['test'][key] == value,(directory,key)
            prediction_count += scored['count']
        else:
            assert 'test' not in result and 'test_nll' not in result and not (directory/'predictions.jsonl').exists()
        if candidate['candidate_id'] == registry['control_id'] and steps != 5:
            previous = args.reference/('pilots' if not test else 'finals')/f'current_dora_lr{candidate["learning_rate"]:g}_seed{seed}'
            history.append(check_pair(previous,directory,False))
        checks.append({'artifact_directory':str(directory.relative_to(root)),'steps':steps,
                       'norm_scope_evidence':norm_evidence,
                       'checkpoint_tensors':len(state),'initial_modules_recreated':len(recreated),
                       'probe_logits_bitwise_equal':True,'test_predictions_rescored':len(expected_gold) if test else 0})
        print(json.dumps({'audited':str(directory.relative_to(root)),'steps':steps,'reload':'bitwise'}),flush=True)

    control = common.read(root/'controls/control.json')
    assert control['status'] == 'pass' and control['manifest_sha256'] == manifest_sha
    anchor = candidates[registry['control_id']]
    for name,reference_backend in [('reference',True),('anchor',False)]:
        check(root/'controls'/name,anchor,42,5,reference_backend=reference_backend)
    assert (root/'controls/reference/training.jsonl').read_bytes() == (root/'controls/anchor/training.jsonl').read_bytes()
    ref,actual = [torch.load(root/'controls'/name/'adapter.pt',map_location='cpu',weights_only=True) for name in ('reference','anchor')]
    assert list(ref) == list(actual) and all(torch.equal(ref[name],actual[name]) for name in ref)
    initial_checks = []
    initial_probes = common.read(root/'controls/initial_probes.json')
    assert initial_probes['status']=='pass' and initial_probes['manifest_sha256']==manifest_sha
    assert initial_probes['study_lane']==study_lane and not initial_probes['test_used']
    assert initial_probes['probe_split']=='train' and initial_probes['quality_split']=='validation'
    assert initial_probes['probe_ids']==[row['id'] for row in records['train'][:2]]
    assert set(initial_probes['candidates'])==set(candidates)
    initial_logits={};accepted_logits=None;accepted_rng=None
    initial_order=[registry['control_id'],*[cid for cid in candidates if cid!=registry['control_id']]]
    for cid in initial_order:
        candidate=candidates[cid]
        row=initial_probes['candidates'][cid]
        if 'reused_spec_probe' in row:
            assert candidates[row['reused_spec_probe']]['spec']==candidate['spec']
            continue
        if row.get('status')=='failed_numerically':
            assert row['failure_stage']=='initialization' and row['steps_completed']==0
            error_class={'RuntimeError':RuntimeError,'ValueError':ValueError}[row['exception_type']]
            assert runner.initialization_failure(error_class(row['error']))
            initial_checks.append({'candidate_id':cid,'failed_numerically':True,'reason':row['error']})
            continue
        backend=common.backend_for(candidate,42,calibration)
        runner.restore_original_projections(model,backend,original_projections,frozen_sha)
        task.seed_all(42)
        backend.inject_adapters(model,cid,rank=8,targets=('q_proj','v_proj'))
        assert snapshot(model,backend.adapter_types)==row['initialization']
        assert task.frozen_digest(model)==row['initial_frozen_sha256']
        assert row['original_frozen_sha256']==frozen_sha
        rng=common.tensor_digest([('cpu',torch.get_rng_state()),('cuda',torch.cuda.get_rng_state())])
        assert rng==row['training_rng_sha256']
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(**task.batch(records['train'][:2],tokenizer),use_cache=False)
            logits=output.logits.detach().cpu();loss=float(output.loss)
        if accepted_logits is None:
            assert cid==registry['control_id']
            accepted_logits=logits;accepted_rng=rng
        assert rng==accepted_rng and row['training_rng_matches_accepted']
        assert common.tensor_digest([('logits',logits)])==row['probe_logits_sha256']
        assert loss==row['probe_loss'] and list(logits.shape)==row['probe_logits_shape']
        difference=logits.float()-accepted_logits.float()
        assert row['logit_error_vs_accepted']=={'rmse':float(difference.square().mean().sqrt()),'maximum_absolute':float(difference.abs().max())}
        assert row['accepted_logits_bitwise_equal']==torch.equal(logits,accepted_logits)
        kind,_=common.spec_parts(candidate['spec'])
        if study_lane=='centered' or kind=='random':assert torch.equal(logits,accepted_logits)
        assert task.nll(model,records['validation'],tokenizer,config['batch_size'])==row['initial_validation_nll']
        if 'probe_payload' in row:
            assert common.sha(root/row['probe_payload'])==row['probe_payload_sha256']
            assert torch.equal(torch.load(root/row['probe_payload'],map_location='cpu',weights_only=True),logits)
        if study_lane=='quantized':initial_logits[cid]=logits
        initial_checks.append({'candidate_id':cid,'initial_full_logits_recreated':True,'initial_validation_nll_reinferred':True})
    for cid,candidate in candidates.items():
        kind,config_=common.spec_parts(candidate['spec'])
        if kind!='quantized' or config_.get('method','loftq')!='loftq':continue
        def resolve(key):
            row=initial_probes['candidates'][key]
            return row.get('reused_spec_probe',key)
        actual_cid,control_cid=resolve(cid),resolve(candidate['comparison_control_id'])
        row,control_row=[initial_probes['candidates'][key] for key in (actual_cid,control_cid)]
        if row.get('status')=='failed_numerically' or control_row.get('status')=='failed_numerically':continue
        assert row['initial_frozen_sha256']==control_row['initial_frozen_sha256']
        delta=initial_logits[actual_cid].float()-initial_logits[control_cid].float()
        expected_pair={'candidate_id':candidate['comparison_control_id'],'frozen_base_identical':True,
            'logit_rmse':float(delta.square().mean().sqrt()),'logit_maximum_absolute':float(delta.abs().max()),
            'initial_validation_nll_difference':row['initial_validation_nll']-control_row['initial_validation_nll']}
        assert row['quantizer_controls'][candidate['comparison_control_id']]==expected_pair
    selected = None
    if args.phase in ('pilots','finals'):
        receipts=[]
        for cid,candidate in candidates.items():
            check(root/'pilots'/cid,candidate,42,48)
            receipts.append(common.read(root/'pilots'/cid/'receipt.json'))
        expected_selection = common.select_candidates(registry,receipts,minimize=True)
        expected_selection['manifest_sha256'] = manifest_sha
        selected = common.read(root/'selections.json')
        assert selected == expected_selection
    if args.phase == 'finals':
        for family,lanes in selected['selections'].items():
            family_record = common.read(root/'families'/f'{family}.json')
            assert family_record['selections'] == lanes and family_record['manifest_sha256'] == manifest_sha
            assert family_record['selections_sha256'] == common.sha(root/'selections.json')
            expected_keys = {f'{cid}/seed_{seed}' for cid in lanes.values() if cid is not None for seed in registry['seeds']}
            assert set(family_record['fits']) == expected_keys
            for key in sorted(expected_keys):
                cid,seed_text = key.split('/');seed=int(seed_text.removeprefix('seed_'))
                check(root/'finals'/cid/seed_text,candidates[cid],seed,256,True)
                assert family_record['fits'][key] == common.read(root/'finals'/cid/seed_text/'receipt.json')
            expected_paired={f'{candidates[cid]["comparison_control_id"]}/seed_{seed}'
                for cid in lanes.values() if cid is not None
                if common.spec_parts(candidates[cid]['spec'])[0]=='quantized'
                and common.spec_parts(candidates[cid]['spec'])[1].get('method','loftq')=='loftq'
                for seed in registry['seeds']}
            assert set(family_record.get('paired_control_fits',{}))==expected_paired
            for key in sorted(expected_paired):
                cid,seed_text=key.split('/');seed=int(seed_text.removeprefix('seed_'))
                check(root/'finals'/cid/seed_text,candidates[cid],seed,256,True)
                assert family_record['paired_control_fits'][key]==common.read(root/'finals'/cid/seed_text/'receipt.json')
            for treatment_id in {cid for cid in lanes.values() if cid is not None}:
                kind,cfg=common.spec_parts(candidates[treatment_id]['spec'])
                if kind!='quantized' or cfg.get('method','loftq')!='loftq':continue
                paired_id=candidates[treatment_id]['comparison_control_id']
                assert candidates[paired_id]['learning_rate']==candidates[treatment_id]['learning_rate']
                for seed in registry['seeds']:
                    treatment=common.read(root/'finals'/treatment_id/f'seed_{seed}'/'receipt.json')
                    paired=common.read(root/'finals'/paired_id/f'seed_{seed}'/'receipt.json')
                    if treatment['status']=='pass' and paired['status']=='pass':
                        a,b=[common.read(root/'finals'/cid/f'seed_{seed}'/'initialization.json') for cid in (treatment_id,paired_id)]
                        assert a['frozen_sha256']==b['frozen_sha256']
    path = root/'audit_source'/Path(__file__).name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(Path(__file__).read_bytes())
    helper_source = Path(norm_audit_helpers.__file__)
    helper_copy = root/'audit_source'/helper_source.name
    if helper_copy.exists():assert helper_copy.read_bytes() == helper_source.read_bytes()
    else:helper_copy.write_bytes(helper_source.read_bytes())
    common.write(root/'baseline_history_parity.json',{'status':'pass','checks':history,'source_sha256':common.sha(__file__)})
    inputs = {str(path.relative_to(root)):common.sha(path) for path in root.rglob('*') if path.is_file()
              and path.suffix != '.pt' and path.name not in ('task_audit.json','controls_audit.json','pilots_audit.json')}
    audit = {'status':'pass','phase':args.phase,'source_sha256':common.sha(__file__),'manifest_sha256':manifest_sha,
             'norm_audit_helper_source_sha256':common.sha(helper_source),
             'fits_checked':len(checks),'logged_steps_checked':row_count,'predictions_rescored':prediction_count,
             'checkpoint_tensors_checked':216*len(checks),'checkpoint_sha256':checkpoint_hashes,
             'checks':checks,'historical_baseline_fits_bitwise_equal':len(history),
             'initial_probe_checks':initial_checks,'study_lane':study_lane,'failed_numerically':failed,
             'calibration_train_examples':128 if calibration else 0,'frozen_sha256':frozen_sha,'input_sha256':inputs,
             'scope':'Strict checkpoint payload audit; all initial factors and compensation buffers recreated; saved train-only full-logit probe replayed bitwise on CUDA. All prediction JSON metrics rescored. Test NLL retained from original inference, not independently re-inferred. Fixture is implementation validation, not the main family comparison.'}
    common.write(root/('task_audit.json' if args.phase=='finals' else args.phase+'_audit.json'),audit)
    print(json.dumps({key:value for key,value in audit.items() if key not in ('input_sha256','checkpoint_sha256','checks')}),flush=True)


if __name__ == '__main__':
    main()
