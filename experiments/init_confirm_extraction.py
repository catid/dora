"""Fresh-seed confirmation of frozen validation-selected ViGGO configurations.

The historical task.fit training, checkpoint selection and generation loop is
unchanged. This module adds provenance, accepted parity and initialization gates.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from experiments import extraction as task
from experiments import init_common as common
from experiments import init_confirm_common as confirmation
from experiments import init_extraction as baseline
from experiments import nora_current as old

adapter_snapshot = baseline.adapter_snapshot
verify_receipt = baseline.verify_receipt
numerical_failure = baseline.numerical_failure
initialization_failure = baseline.initialization_failure


def load_records(args, tokenizer):
    """Serialize the historical loader's shared split-manifest temporary file."""
    with (args.output/'.split_manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            return task.data(args,tokenizer)
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def load_calibration(study_root, reference):
    path = study_root/'calibration/manifest.json'
    assert path.exists(), 'Selected data initializer requires its original train-only calibration manifest'
    calibration = common.read(path)
    assert calibration['metadata']['task_source_manifest'] == reference['source_sha256']
    assert calibration['metadata']['calibration_split'] == 'train only'
    for name, file in calibration['module_files'].items():
        assert common.sha(file) == calibration['module_sha256'][name]
    return calibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate-evidence',type=Path,required=True)
    p.add_argument('--phase',choices=('controls','finals'),required=True)
    p.add_argument('--candidate-ids',nargs='+')
    p.add_argument('--seeds',type=int,nargs='+')
    args=p.parse_args()
    registry=confirmation.load_registry(args.registry,args.protocol,args.study_root,'extraction')
    reference=common.read(args.study_root/'manifest.json')
    assert reference['study_lane']=='centered'
    candidates={row['candidate_id']:row for row in registry['candidates']}
    requested=args.candidate_ids or list(candidates)
    seeds=args.seeds or registry['seeds']
    assert set(requested)<=set(candidates) and set(seeds)<=set(registry['seeds'])
    assert len(seeds)==len(set(seeds)) and len(requested)==len(set(requested))
    gate=common.read(args.gate_evidence)
    assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    for name,digest in gate['source_sha256'].items():assert common.sha(name)==digest,name
    for name,digest in reference['source_sha256'].items():assert common.sha(name)==digest,name
    for key,value in reference['configuration'].items():setattr(args,key,value)
    assert args.rank==registry['rank']==8 and args.steps==256 and args.pilot_steps==48
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(task.MODEL,revision=task.MODEL_REVISION)
    tokenizer.pad_token=tokenizer.eos_token
    records,provenance=load_records(args,tokenizer)
    assert provenance==reference['dataset']
    assert common.sha(args.output/'split_manifest.json')==provenance['split_manifest_sha256']
    for name,digest in reference['checkpoint_sha256'].items():
        assert common.sha(hf_hub_download(task.MODEL,name,revision=task.MODEL_REVISION))==digest
    sources=list(dict.fromkeys([*reference['source_sha256'],*gate['source_sha256'],
        'experiments/init_confirm_common.py','experiments/init_confirm_extraction.py',
        'experiments/audit_init_confirm_extraction.py']))
    source_hashes={name:common.sha(name) for name in sources}
    calibration=load_calibration(args.study_root,reference) if any(
        common.spec_parts(row['spec'])[0]=='data' for row in candidates.values()) else None
    manifest={'schema':'dora-init-confirm-extraction-v1','registry':registry,
        'study_lane':'centered','registry_file_sha256':common.sha(args.registry),
        'source_sha256':source_hashes,'source_study_root':str(args.study_root.resolve()),
        'source_study_manifest_sha256':common.sha(args.study_root/'manifest.json'),
        'source_selections_sha256':common.sha(args.study_root/'selections.json'),
        'dataset':provenance,'model':task.MODEL,'model_revision':task.MODEL_REVISION,
        'checkpoint_sha256':reference['checkpoint_sha256'],'configuration':reference['configuration'],
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
        'protocol_sha256':common.sha(args.protocol),'backend_gate_sha256':common.sha(args.gate_evidence),
        'calibration_manifest_sha256':common.sha(args.study_root/'calibration/manifest.json') if calibration else None,
        'test_selection_policy':'Registry provenance recorded verbatim; fresh seeds45-49; checkpoint selection uses validation NLL only'}
    common.write_once(args.output/'manifest.json',manifest)
    for source in sources:
        destination=args.output/'source'/source;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert common.sha(destination)==source_hashes[source]
        else:shutil.copyfile(source,destination)
    for name,source in (('registry.json',args.registry),('protocol.md',args.protocol),
                        ('backend_gate.json',args.gate_evidence)):
        destination=args.output/name
        if destination.exists():assert common.sha(destination)==common.sha(source)
        else:shutil.copyfile(source,destination)
    manifest_sha=common.sha(args.output/'manifest.json')
    model=AutoModelForCausalLM.from_pretrained(task.MODEL,revision=task.MODEL_REVISION,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().requires_grad_(False)
    original_frozen=task.frozen_digest(model)
    original_projections=baseline.original_projection_parameters(model)
    assert len(original_projections)==72

    def run(candidate, seed, steps, directory, final=False, *, reference_backend=None):
        directory.mkdir(parents=True,exist_ok=True)
        receipt_path = directory/'receipt.json'
        backend = reference_backend or common.backend_for(candidate, seed, calibration)
        kind,_ = common.spec_parts(candidate['spec'])
        recipe = backend.recipe_manifest(candidate['candidate_id'])
        if receipt_path.exists():
            receipt = common.read(receipt_path)
            assert receipt['manifest_sha256'] == manifest_sha and receipt['recipe'] == recipe
            assert receipt['candidate_id'] == candidate['candidate_id'] and receipt['seed'] == seed
            if receipt['status'] == 'pass':
                assert receipt['steps'] == steps and receipt['evaluate_test'] == final
            verify_receipt(args.output, receipt, manifest_sha)
            return receipt
        initial = {}
        baseline.restore_original_projections(model,backend,original_projections,original_frozen)
        def capture_injection(model, method, rank, targets):
            started = time.perf_counter()
            result = backend.inject_adapters(model, method, rank=rank, targets=targets)
            torch.cuda.synchronize()
            initial.update(status='pass', modules=adapter_snapshot(model,backend.adapter_types),
                           initializer_seconds=time.perf_counter()-started,
                           frozen_sha256=task.frozen_digest(model))
            assert initial['frozen_sha256'] == original_frozen
            if final:
                probes = common.read(args.output/'controls/initial_probes.json')['candidates']
                probe = probes[f'{candidate["candidate_id"]}/seed_{seed}']
                if 'reused_spec_probe' in probe:probe = probes[probe['reused_spec_probe']]
                assert probe['status'] == 'pass' and probe['initialization'] == initial['modules']
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

    control_path=args.output/'controls/control.json'
    anchor=candidates[registry['control_id']]
    assert common.spec_parts(anchor['spec'])==('random',{'mode':'accepted'})
    if args.phase=='controls':
        old_backend=SimpleNamespace(adapter_types=common.backend_for(anchor,45).adapter_types,
            inject_adapters=lambda model,method,rank,targets:old.inject_adapters(model,'current_dora',rank,targets),
            recipe_manifest=lambda method:old.recipe_manifest('current_dora'),assert_no_clamp=old.assert_no_clamp)
        before_dir,after_dir=args.output/'controls/reference',args.output/'controls/anchor'
        original_eval_every=args.eval_every;args.eval_every=5
        before=run(anchor,45,5,before_dir,reference_backend=old_backend)
        after=run(anchor,45,5,after_dir)
        args.eval_every=original_eval_every
        assert before['status']==after['status']=='pass'
        assert common.read(before_dir/'initialization.json')['modules']==common.read(after_dir/'initialization.json')['modules']
        assert (before_dir/'training.jsonl').read_bytes()==(after_dir/'training.jsonl').read_bytes()
        states=[torch.load(path/'adapter.pt',map_location='cpu',weights_only=True) for path in (before_dir,after_dir)]
        assert list(states[0])==list(states[1])
        assert all(torch.equal(states[0][name],states[1][name]) for name in states[0])
        common.write_once(control_path,{'status':'pass','manifest_sha256':manifest_sha,'seed':45,
            'reference':before,'anchor':after,'tensors_bitwise_equal':len(states[0]),'training_logs_bitwise_equal':True})
        probe_path=args.output/'controls/initial_probes.json'
        if not probe_path.exists():
            probe=task.batch(records['train'][:2],tokenizer);probe_rows={}
            ordered=[anchor,*[row for row in candidates.values() if row is not anchor]]
            for seed in registry['seeds']:
                accepted_logits=None;accepted_rng=None;seen={}
                for candidate in ordered:
                    cid=candidate['candidate_id'];key=f'{cid}/seed_{seed}'
                    spec_key=common.digest_json(candidate['spec'])
                    if spec_key in seen:
                        probe_rows[key]={'reused_spec_probe':seen[spec_key]};continue
                    backend=common.backend_for(candidate,seed,calibration)
                    baseline.restore_original_projections(model,backend,original_projections,original_frozen)
                    task.seed_all(seed)
                    try:
                        backend.inject_adapters(model,cid,rank=args.rank,targets=('q_proj','v_proj'))
                    except (RuntimeError,ValueError) as error:
                        if candidate is anchor or not initialization_failure(error):raise
                        probe_rows[key]={'status':'failed_numerically','failure_stage':'initialization',
                            'error':str(error),'exception_type':type(error).__name__,'steps_completed':0,
                            'recipe':backend.recipe_manifest(cid)}
                        seen[spec_key]=key;continue
                    rng=common.tensor_digest([('cpu',torch.get_rng_state()),('cuda',torch.cuda.get_rng_state())])
                    model.eval()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(**probe,use_cache=False);logits=output.logits.detach().cpu();loss=float(output.loss)
                    assert torch.isfinite(logits).all() and math.isfinite(loss)
                    if accepted_logits is None:accepted_logits=logits;accepted_rng=rng
                    assert torch.equal(logits,accepted_logits),'Centered initial full logits changed'
                    assert rng==accepted_rng,'Initializer changed downstream training RNG'
                    assert task.frozen_digest(model)==original_frozen
                    validation_nll=task.nll(model,records['validation'],tokenizer,args.batch_size)
                    assert math.isfinite(validation_nll)
                    probe_rows[key]={'status':'pass','candidate_id':cid,'seed':seed,
                        'recipe':backend.recipe_manifest(cid),'initialization':adapter_snapshot(model,backend.adapter_types),
                        'initial_frozen_sha256':original_frozen,'probe_loss':loss,'initial_validation_nll':validation_nll,
                        'probe_logits_shape':list(logits.shape),'probe_logits_sha256':common.tensor_digest([('logits',logits)]),
                        'training_rng_sha256':rng,'accepted_logits_bitwise_equal':True,'training_rng_matches_accepted':True}
                    seen[spec_key]=key
                    print(json.dumps({'event':'confirmation_initial_probe','candidate_id':cid,'seed':seed,'status':'pass'}),flush=True)
            common.write_once(probe_path,{'status':'pass','manifest_sha256':manifest_sha,'study_lane':'centered',
                'probe_split':'train','probe_ids':[row['id'] for row in records['train'][:2]],
                'quality_split':'validation','test_used':False,'candidates':probe_rows})
        return
    control=common.read(control_path);assert control['status']=='pass' and control['manifest_sha256']==manifest_sha
    probes=common.read(args.output/'controls/initial_probes.json')
    assert probes['status']=='pass' and probes['manifest_sha256']==manifest_sha
    for cid in requested:
        for seed in seeds:
            key=f'{cid}/seed_{seed}';probe=probes['candidates'][key]
            if 'reused_spec_probe' in probe:probe=probes['candidates'][probe['reused_spec_probe']]
            if probe['status']=='failed_numerically':
                directory=args.output/'finals'/cid/f'seed_{seed}'
                receipt={'status':'failed_numerically','failure_stage':'initialization','seed':seed,'candidate_id':cid,
                    'recipe':common.backend_for(candidates[cid],seed,calibration).recipe_manifest(cid),
                    'manifest_sha256':manifest_sha,'error':probe['error'],'exception_type':probe['exception_type'],
                    'steps_completed':0,'initialization_failure_from_control_sha256':common.sha(args.output/'controls/initial_probes.json')}
                common.write_once(directory/'receipt.json',receipt)
            else:
                receipt=run(candidates[cid],seed,args.steps,args.output/'finals'/cid/f'seed_{seed}',True)
            print(json.dumps({'event':'confirmation_fit','candidate_id':cid,'seed':seed,'status':receipt['status']}),flush=True)


if __name__=='__main__':main()
