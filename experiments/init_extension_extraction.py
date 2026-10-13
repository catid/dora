"""Separately frozen LoRA-Null/GME extension with the unchanged ViGGO task loop."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from experiments import extraction as task
from experiments import init_extension_common as common
from experiments import init_extension_extraction_context as context
from experiments import nora_current as old


def prepare_calibration(records, tokenizer, output, source_manifest, reference):
    source=reference/'calibration/manifest.json'
    original=common.read(source);parent=common.read(reference/'manifest.json')
    assert original['metadata']['task_source_manifest']==parent['source_sha256']
    assert original['metadata']['source_sha256']==common.sha('experiments/init_calibration.py')
    assert parent['source_sha256']['experiments/extraction.py']==common.sha('experiments/extraction.py')
    indices=list(range(len(records['train'])));random.Random(20261009).shuffle(indices)
    ids=[records['train'][index]['id'] for index in indices[:128]]
    assert original['metadata']['example_ids']==ids
    assert original['metadata']['calibration_split']=='train only'
    assert not set(ids)&{row['id'] for split in ('validation','test') for row in records[split]}
    for name,file in original['module_files'].items():assert common.sha(file)==original['module_sha256'][name]
    result=copy.deepcopy(original)
    result['metadata']['extension_parent_calibration_sha256']=common.sha(source)
    result['metadata']['extension_parent_manifest_sha256']=common.sha(reference/'manifest.json')
    result['metadata']['extension_parent_root']=str(reference.resolve())
    result['metadata']['extension_source_sha256']=source_manifest
    result['metadata']['extension_factor_cache_dir']=str((output/'extension_factor_cache').resolve())
    prepare=context.make_gme_prepare(records,tokenizer,result,output,source_manifest)
    result['metadata']['extension_gme_binding']=prepare.binding
    common.write_once(output/'calibration/manifest.json',result)
    return result,prepare


def load_records(args,tokenizer):
    with (args.output/'.split_manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:return task.data(args,tokenizer)
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def numerical_failure(error):
    return isinstance(error, RuntimeError) and any(text in str(error).lower() for text in
                ('nonfinite training loss', 'nonfinite adapter gradient', 'nonfinite parameter'))


def initialization_failure(error):
    return isinstance(error,(RuntimeError,ValueError)) and any(text in str(error).lower() for text in
        ('nonfinite','failed to converge','did not converge','convergence failed','corda inverse',
         'algorithm failed to converge','singular matrix','ill-conditioned'))


validate_study_lane = common.validate_study_lane


def original_projection_parameters(model):
    return {name:(module.weight,module.bias) for name,module in model.named_modules()
            if isinstance(module,torch.nn.Linear) and name.endswith(('.q_proj','.v_proj'))}


def restore_original_projections(model, backend, original, expected_digest):
    """Quantized adapters replace W; retained original Parameters remain intact."""
    task.strip_adapters(model,backend.adapter_types)
    modules = dict(model.named_modules())
    for name,(weight,bias) in original.items():
        assert isinstance(modules[name],torch.nn.Linear)
        modules[name].weight,modules[name].bias = weight,bias
    model.requires_grad_(False)
    assert task.frozen_digest(model) == expected_digest, 'Original pretrained parameters changed'


def record_quantizer_comparison(row, control_id, comparison):
    """A deduplicated initializer may have distinct same-rate control IDs."""
    values=row.setdefault('quantizer_controls',{})
    assert control_id not in values or values[control_id]==comparison
    values[control_id]=comparison


def adapter_snapshot(model, adapter_types):
    """Bind the exact initial factors and compensation buffers without storing W."""
    result = {}
    for name, module in model.named_modules():
        if not isinstance(module, adapter_types):
            continue
        trainable = [(key, value) for key, value in module.named_parameters(recurse=False) if value.requires_grad]
        buffers = list(module.named_buffers(recurse=False))
        result[name] = {'trainable_sha256': common.tensor_digest(trainable),
                        'trainable_names': [key for key, _ in trainable],
                        'buffer_sha256': common.tensor_digest(buffers),
                        'buffer_names': [key for key, _ in buffers],
                        'frozen_weight_sha256': common.tensor_digest([('weight', module.weight)])}
    assert len(result) == 72
    return result


def verify_receipt(output, receipt, manifest_sha):
    assert receipt['manifest_sha256'] == manifest_sha
    if receipt['status'] != 'pass':
        assert receipt['status'] == 'failed_numerically'
        return
    directory = output / receipt['artifact_directory']
    for name, key in (('result.json','result_sha256'), ('adapter.pt','checkpoint_sha256'),
                      ('training.jsonl','training_log_sha256'), ('initialization.json','initialization_sha256'),
                      ('reload_validation.json','reload_validation_sha256')):
        assert common.sha(directory/name) == receipt[key], (directory,name)
    if receipt['evaluate_test']:
        assert common.sha(directory/'predictions.jsonl') == receipt['predictions_sha256']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate-evidence', type=Path, required=True)
    p.add_argument('--calibration-reference',type=Path,required=True)
    p.add_argument('--phase', choices=('controls', 'pilots', 'select', 'finals'), required=True)
    p.add_argument('--candidate-ids', nargs='+')
    p.add_argument('--family-ids', nargs='+')
    p.add_argument('--reference', type=Path, default=Path('/var/tmp/dora-bench/nora_current_bench/extraction'))
    args = p.parse_args()
    registry = common.load_registry(args.registry, args.protocol, 'extraction')
    study_lane = validate_study_lane(registry)
    gate = common.read(args.gate_evidence)
    assert gate['status'] == gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    for name, digest in gate['source_sha256'].items():
        assert common.sha(name) == digest, name
    candidates = {row['candidate_id']: row for row in registry['candidates']}
    requested = args.candidate_ids or list(candidates)
    assert set(requested) <= set(candidates)
    reference = common.read(args.reference / 'manifest.json')
    for key in ('train_size', 'val_size', 'test_size', 'max_length', 'rank', 'batch_size',
                'generation_batch_size', 'pilot_steps', 'steps', 'eval_every', 'learning_rates'):
        setattr(args, key, reference['configuration'][key])
    assert args.rank == registry['rank']
    assert args.learning_rates == registry['learning_rates']
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(task.MODEL, revision=task.MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    records, provenance = load_records(args, tokenizer)
    assert provenance == reference['dataset']
    assert common.sha(args.output/'split_manifest.json') == provenance['split_manifest_sha256']
    sources = ['experiments/init_extension_extraction.py','experiments/audit_init_extension_extraction.py',
               'experiments/init_extension_extraction_context.py','experiments/init_extension_common.py',
               'experiments/init_extension_registry.py','experiments/init_extension_runtime.py','experiments/init_extension.py',
               'experiments/init_extraction.py', 'experiments/init_common.py', 'experiments/init_ablations.py',
               'experiments/init_data.py', 'experiments/init_calibration.py', 'experiments/extraction.py',
               'experiments/nora_current.py', 'experiments/next_ablations.py', 'experiments/change_ablations.py',
               'experiments/adapters.py', 'experiments/original_dora.py', 'dora.py', 'dora_row.py']
    sources = list(dict.fromkeys([*sources, *gate['source_sha256']]))
    for name in ('experiments/init_warmstart.py','experiments/init_loftq.py'):
        if Path(name).exists() and name not in sources:sources.append(name)
    source_hashes = {name: common.sha(name) for name in sources}
    index_path = hf_hub_download(task.MODEL, 'model.safetensors.index.json', revision=task.MODEL_REVISION)
    weight_index = common.read(index_path)
    checkpoint_hashes = {'model.safetensors.index.json': common.sha(index_path)}
    for name in sorted(set(weight_index['weight_map'].values())):
        checkpoint_hashes[name] = common.sha(hf_hub_download(task.MODEL, name, revision=task.MODEL_REVISION))
    manifest = {'schema': 'dora-init-extension-extraction-v1', 'registry': registry,
                'study_lane':study_lane,'calibration_reference':str(args.calibration_reference.resolve()),
                'calibration_reference_manifest_sha256':common.sha(args.calibration_reference/'manifest.json'),
                'registry_file_sha256': common.sha(args.registry), 'source_sha256': source_hashes,
                'reference_manifest_sha256': common.sha(args.reference/'manifest.json'),
                'dataset': provenance, 'model': task.MODEL, 'model_revision': task.MODEL_REVISION,
                'checkpoint_sha256': checkpoint_hashes,
                'configuration': {key: getattr(args,key) for key in ('train_size','val_size','test_size','max_length',
                                  'rank','batch_size','generation_batch_size','pilot_steps','steps','eval_every','learning_rates')},
                'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()},
                'protocol_sha256': common.sha(args.protocol),
                'backend_gate_sha256': common.sha(args.gate_evidence),
                'test_selection_policy': 'No test generation before every registered pilot has completed and selections are frozen'}
    common.write_once(args.output/'manifest.json', manifest)
    for source in sources:
        destination = args.output/'source'/source
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert common.sha(destination) == source_hashes[source]
        else:
            shutil.copyfile(source, destination)
    for name, source in (('registry.json',args.registry), ('protocol.md',args.protocol),
                         ('backend_gate.json',args.gate_evidence)):
        destination = args.output/name
        if destination.exists():
            assert common.sha(destination) == common.sha(source)
        else:
            shutil.copyfile(source,destination)
    manifest_sha = common.sha(args.output/'manifest.json')

    def select():
        rows = [common.read(args.output/'pilots'/cid/'receipt.json') for cid in candidates]
        for row in rows:
            verify_receipt(args.output, row, manifest_sha)
        result = common.select_candidates(registry, rows, minimize=True)
        result['manifest_sha256'] = manifest_sha
        common.write_once(args.output/'selections.json', result)
        return result
    if args.phase == 'select':
        select(); return

    model = AutoModelForCausalLM.from_pretrained(task.MODEL, revision=task.MODEL_REVISION,
                   dtype=torch.bfloat16, attn_implementation='sdpa').cuda().requires_grad_(False)
    original_frozen = task.frozen_digest(model)
    original_projections = original_projection_parameters(model)
    assert len(original_projections) == 72
    calibration,gme_prepare = prepare_calibration(records,tokenizer,args.output,source_hashes,args.calibration_reference)
    def make_backend(candidate,seed):
        return common.backend_for(candidate,seed,calibration,gme_prepare=gme_prepare)

    def run_unlocked(candidate, seed, steps, directory, final=False, *, reference_backend=None):
        directory.mkdir(parents=True,exist_ok=True)
        receipt_path = directory/'receipt.json'
        backend = reference_backend or make_backend(candidate,seed)
        kind,_ = common.spec_parts(candidate['spec'])
        recipe = backend.recipe_manifest(candidate['candidate_id'])
        if receipt_path.exists():
            receipt = common.read(receipt_path)
            assert receipt['manifest_sha256'] == manifest_sha and receipt['recipe'] == recipe
            assert receipt['candidate_id']==candidate['candidate_id'] and receipt['seed']==seed
            if receipt['status']=='pass':assert receipt['steps']==steps and receipt['evaluate_test']==final
            verify_receipt(args.output, receipt, manifest_sha)
            return receipt
        probes_path=args.output/'controls/initial_probes.json'
        if probes_path.exists() and reference_backend is None:
            probe_rows=common.read(probes_path)['candidates']
            control_probe=probe_rows[candidate['candidate_id']]
            if 'reused_spec_probe' in control_probe:control_probe=probe_rows[control_probe['reused_spec_probe']]
            if control_probe.get('status')=='failed_numerically' and control_probe['failure_stage']=='initialization':
                receipt={'status':'failed_numerically','failure_stage':'initialization','seed':seed,
                    'candidate_id':candidate['candidate_id'],'recipe':recipe,'manifest_sha256':manifest_sha,
                    'error':control_probe['error'],'exception_type':control_probe['exception_type'],
                    'steps_completed':0,'initialization_failure_from_control_sha256':common.sha(probes_path)}
                common.write(receipt_path,receipt);return receipt
        initial = {}
        restore_original_projections(model,backend,original_projections,original_frozen)
        def capture_injection(model, method, rank, targets):
            started = time.perf_counter()
            result = backend.inject_adapters(model, method, rank=rank, targets=targets)
            torch.cuda.synchronize()
            initial.update(status='pass', modules=adapter_snapshot(model,backend.adapter_types),
                           initializer_seconds=time.perf_counter()-started,
                           frozen_sha256=task.frozen_digest(model))
            assert initial['frozen_sha256'] == original_frozen
            common.write(directory/'initialization.json', initial)
            return result
        tracked_backend = SimpleNamespace(adapter_types=backend.adapter_types, inject_adapters=capture_injection,
                              recipe_manifest=backend.recipe_manifest, assert_no_clamp=backend.assert_no_clamp)
        try:
            result = task.fit(model, candidate['candidate_id'], seed, candidate['learning_rate'], steps,
                              records, tokenizer, args, directory, evaluate_test=final, backend=tracked_backend)
        except (RuntimeError,ValueError) as error:
            if not (numerical_failure(error) or initialization_failure(error)):
                raise
            log_path=directory/'training.jsonl'
            steps_completed=len(log_path.read_text().splitlines()) if log_path.exists() else 0
            receipt = {'status': 'failed_numerically', 'candidate_id': candidate['candidate_id'],
                       'recipe': recipe, 'error': str(error), 'manifest_sha256': manifest_sha, 'seed':seed,
                       'exception_type':type(error).__name__,'steps_completed':steps_completed,
                       'failure_stage':'training' if initial else 'initialization'}
            common.write(receipt_path,receipt)
            return receipt
        expected_frozen = initial['frozen_sha256']
        assert result['frozen_weights_unchanged'] and result['frozen_sha256'] == expected_frozen
        assert result['recipe'] == recipe
        # Trainable-only checkpoints omit compensation buffers. Recreate them
        # deterministically, then require exact logits on a train-only probe.
        probe = task.batch(records['train'][:2],tokenizer)
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            expected = model(**probe,use_cache=False).logits.detach().cpu()
        task.strip_adapters(model,backend.adapter_types)
        task.seed_all(seed)
        backend.inject_adapters(model,candidate['candidate_id'],rank=args.rank,targets=('q_proj','v_proj'))
        regenerated = adapter_snapshot(model,backend.adapter_types)
        assert initial['modules'] == regenerated, 'Recreated initial factors/buffers differ'
        state = torch.load(directory/'adapter.pt',map_location='cpu',weights_only=True)
        names = [name for name,p in model.named_parameters() if p.requires_grad]
        assert names == list(state)
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad:
                    assert p.shape == state[name].shape and p.dtype == state[name].dtype
                    p.copy_(state[name])
        assert task.frozen_digest(model) == expected_frozen
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            actual = model(**probe,use_cache=False).logits.detach().cpu()
        assert torch.equal(expected,actual), 'Checkpoint reload logits differ'
        common.write(directory/'reload_validation.json',
                     {'status':'pass','initial_modules_recreated':regenerated,
                      'probe_split':'train','probe_ids':[row['id'] for row in records['train'][:2]],
                      'probe_logits_shape':list(expected.shape),
                      'probe_logits_sha256':common.tensor_digest([('logits',expected)]),
                      'checkpoint_tensors':len(state),'logits_bitwise_equal':True,
                      'frozen_sha256':expected_frozen,'original_frozen_sha256':original_frozen})
        del expected, actual, state
        receipt = {'status':'pass','candidate_id':candidate['candidate_id'],'seed':seed,
                   'manifest_sha256':manifest_sha,'recipe':recipe,'steps':steps,
                   'selection_value':result['validation_nll'],
                   'result_sha256':common.sha(directory/'result.json'),
                   'checkpoint_sha256':common.sha(directory/'adapter.pt'),
                   'training_log_sha256':common.sha(directory/'training.jsonl'),
                   'initialization_sha256':common.sha(directory/'initialization.json'),
                   'reload_validation_sha256':common.sha(directory/'reload_validation.json'),
                   'evaluate_test':final,
                   'predictions_sha256':common.sha(directory/'predictions.jsonl') if final else None,
                   'artifact_directory':str(directory.relative_to(args.output)),
                   'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
        common.write(receipt_path,receipt)
        return receipt

    def run(candidate, seed, steps, directory, final=False, *, reference_backend=None):
        # Different selected families may share the same-rate accepted control.
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'.fit.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                return run_unlocked(candidate,seed,steps,directory,final,reference_backend=reference_backend)
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)

    control_path = args.output/'controls/control.json'
    if args.phase == 'controls':
        anchor = candidates[registry['control_id']]
        old_backend = SimpleNamespace(adapter_types=make_backend(anchor,42).adapter_types,
                       inject_adapters=lambda model, method, rank, targets: old.inject_adapters(model,'current_dora',rank,targets),
                       recipe_manifest=lambda method: old.recipe_manifest('current_dora'), assert_no_clamp=old.assert_no_clamp)
        old_dir, new_dir = args.output/'controls/reference', args.output/'controls/anchor'
        control_args = copy.copy(args)
        args.eval_every = 5
        before = run(anchor,42,5,old_dir,reference_backend=old_backend)
        after = run(anchor,42,5,new_dir)
        args.eval_every = control_args.eval_every
        assert before['status'] == after['status'] == 'pass'
        assert common.read(old_dir/'initialization.json')['modules'] == common.read(new_dir/'initialization.json')['modules']
        assert (old_dir/'training.jsonl').read_bytes() == (new_dir/'training.jsonl').read_bytes()
        state_a, state_b = [torch.load(path/'adapter.pt', map_location='cpu', weights_only=True) for path in (old_dir,new_dir)]
        assert list(state_a) == list(state_b)
        for name,value in state_a.items():
            torch.testing.assert_close(state_b[name],value,rtol=0,atol=0)
        common.write_once(control_path, {'status':'pass','manifest_sha256':manifest_sha,
                          'reference':before,'anchor':after,'tensors_bitwise_equal':len(state_a),
                          'training_logs_bitwise_equal':True})
        # Every distinct specification receives an actual train-only initial
        # forward before pilot selection. Intentional starts retain measured
        # drift; centered recipes must match accepted logits bitwise.
        probe_path = args.output/'controls/initial_probes.json'
        if not probe_path.exists():
            probe = task.batch(records['train'][:2],tokenizer)
            probe_rows={};seen={};accepted_logits=None;accepted_rng=None
            ordered=[anchor,*[candidate for candidate in candidates.values() if candidate is not anchor]]
            for candidate in ordered:
                spec_key=common.digest_json(candidate['spec'])
                if spec_key in seen:
                    probe_rows[candidate['candidate_id']]={'reused_spec_probe':seen[spec_key]}
                    continue
                backend=make_backend(candidate,42)
                restore_original_projections(model,backend,original_projections,original_frozen)
                task.seed_all(42)
                try:
                    backend.inject_adapters(model,candidate['candidate_id'],rank=args.rank,targets=('q_proj','v_proj'))
                except (RuntimeError,ValueError) as error:
                    if candidate is anchor or not initialization_failure(error):raise
                    probe_rows[candidate['candidate_id']]={'status':'failed_numerically','failure_stage':'initialization',
                        'error':str(error),'exception_type':type(error).__name__,'steps_completed':0,
                        'recipe':backend.recipe_manifest(candidate['candidate_id'])}
                    seen[spec_key]=candidate['candidate_id'];continue
                rng=common.tensor_digest([('cpu',torch.get_rng_state()),('cuda',torch.cuda.get_rng_state())])
                model.eval()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(**probe,use_cache=False)
                    logits=output.logits.detach().cpu();loss=float(output.loss)
                if not torch.isfinite(logits).all() or not math.isfinite(loss):
                    if candidate is anchor:raise RuntimeError('Nonfinite accepted initial forward')
                    probe_rows[candidate['candidate_id']]={'status':'failed_numerically','failure_stage':'initialization',
                        'error':'Nonfinite initial forward logits or loss','exception_type':'RuntimeError','steps_completed':0,
                        'recipe':backend.recipe_manifest(candidate['candidate_id'])}
                    seen[spec_key]=candidate['candidate_id'];continue
                if accepted_logits is None:accepted_logits=logits;accepted_rng=rng
                assert rng==accepted_rng, 'Initializer changed downstream training RNG'
                difference=logits.float()-accepted_logits.float()
                kind,_=common.spec_parts(candidate['spec'])
                if candidate['initialization_lane']=='centered':assert torch.equal(logits,accepted_logits)
                validation_nll=task.nll(model,records['validation'],tokenizer,args.batch_size)
                if not math.isfinite(validation_nll):
                    if candidate is anchor:raise RuntimeError('Nonfinite accepted initial validation NLL')
                    probe_rows[candidate['candidate_id']]={'status':'failed_numerically','failure_stage':'initialization',
                        'error':'Nonfinite initial validation NLL','exception_type':'RuntimeError','steps_completed':0,
                        'recipe':backend.recipe_manifest(candidate['candidate_id'])}
                    seen[spec_key]=candidate['candidate_id'];continue
                probe_rows[candidate['candidate_id']]={'status':'pass','recipe':backend.recipe_manifest(candidate['candidate_id']),
                    'initialization':adapter_snapshot(model,backend.adapter_types),
                    'initial_norm_diagnostics':backend.assert_no_clamp(model),
                    'initial_frozen_sha256':task.frozen_digest(model),'original_frozen_sha256':original_frozen,
                    'probe_loss':loss,'initial_validation_nll':validation_nll,
                    'probe_logits_shape':list(logits.shape),'probe_logits_sha256':common.tensor_digest([('logits',logits)]),
                    'accepted_logits_bitwise_equal':torch.equal(logits,accepted_logits),
                    'logit_error_vs_accepted':{'rmse':float(difference.square().mean().sqrt()),'maximum_absolute':float(difference.abs().max())},
                    'training_rng_sha256':rng,'training_rng_matches_accepted':True}
                if candidate['initialization_lane']=='warmstart':
                    payload_path=args.output/'controls'/(candidate['candidate_id']+'_initial_logits.pt')
                    torch.save(logits,payload_path)
                    probe_rows[candidate['candidate_id']]['probe_payload']=str(payload_path.relative_to(args.output))
                    probe_rows[candidate['candidate_id']]['probe_payload_sha256']=common.sha(payload_path)
                seen[spec_key]=candidate['candidate_id']
            common.write_once(probe_path,{'status':'pass','study_lane':study_lane,'manifest_sha256':manifest_sha,
                'probe_split':'train','probe_ids':[row['id'] for row in records['train'][:2]],
                'quality_split':'validation','test_used':False,'candidates':probe_rows})
        return
    control = common.read(control_path)
    assert control['status'] == 'pass' and control['manifest_sha256'] == manifest_sha
    initial_probes=common.read(args.output/'controls/initial_probes.json')
    assert initial_probes['status']=='pass' and initial_probes['manifest_sha256']==manifest_sha
    if args.phase == 'pilots':
        for cid in requested:
            row = run(candidates[cid],42,args.pilot_steps,args.output/'pilots'/cid)
            print(json.dumps({'event':'pilot','candidate_id':cid,'status':row['status'],
                              'validation_nll':row.get('selection_value')}),flush=True)
        return
    selection = select()
    existing = common.read(args.output/'selections.json')
    assert existing == selection
    families = args.family_ids or list(selection['selections'])
    assert set(families) <= set(selection['selections'])
    for family in families:
        lanes = selection['selections'][family]
        fits = {}
        paired_control_fits = {}
        for seed in registry['seeds']:
            for cid in sorted({cid for cid in lanes.values() if cid is not None}):
                receipt = run(candidates[cid],seed,args.steps,args.output/'finals'/cid/f'seed_{seed}',True)
                fits[f'{cid}/seed_{seed}'] = receipt
                control_id=candidates[cid]['comparison_control_id']
                if control_id!=cid:
                    paired_control_fits[f'{control_id}/seed_{seed}']=run(candidates[control_id],seed,args.steps,
                        args.output/'finals'/control_id/f'seed_{seed}',True)
        common.write_once(args.output/'families'/f'{family}.json',
                          {'family_id':family,'selections':lanes,'fits':fits,
                           'paired_control_fits':paired_control_fits,
                           'manifest_sha256':manifest_sha,'selections_sha256':common.sha(args.output/'selections.json')})


if __name__ == '__main__':
    main()
