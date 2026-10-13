"""Registry-bound initializer study on the frozen Aircraft rank-2 task.

Training, augmentation, checkpoint selection and scoring use vision.run_one
unchanged. Each training seed gets its own train-only calibration and exact
historical random classifier. Pilot and final budgets remain ten/twenty epochs.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
import timm
import torchvision
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from experiments import init_extension_common as common
from experiments import init_extension_observers as observers
GME_PREPARE = {}
from experiments import nora_current as accepted
from experiments.second_round import vision
from experiments.vision import seed_everything, tensor_digest

LANES=('fixed_original_lr','validation_tuned')


def copy_once(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert common.sha(destination)==common.sha(source),destination
    else:shutil.copyfile(source,destination)


def base_model(checkpoint,seed,args):
    seed_everything(seed)
    model=timm.create_model(vision.MODEL,pretrained=False)
    model.load_state_dict(load_file(checkpoint),strict=True)
    model.reset_classifier(100)
    prior=common.read(args.reference_root/'baseline'/f'seed_{seed}'/'result.json')
    assert tensor_digest(model.head.state_dict().items())==prior['initial_head_sha256']
    assert tensor_digest((n,p) for n,p in model.named_parameters() if not n.startswith('head.'))==prior['initial_backbone_sha256']
    return model


def calibration(args,datasets,checkpoint,seed):
    source=args.study_root/'calibration'/f'seed_{seed}'/'manifest.json'
    parent=common.read(args.study_root/'manifest.json')
    assert common.sha(source)==parent['calibration_manifest_sha256'][str(seed)]
    result=observers.bind_calibration(source,args.output/'calibration'/f'seed_{seed}'/'manifest.json',args.output,'aircraft',seed)
    GME_PREPARE[seed]=observers.aircraft_prepare(result,datasets,args)
    return result


def bound_backend(candidate,seed,calibrations,capture=None,verify_context=None):
    backend=common.backend_for(candidate,seed,calibrations.get(seed),gme_prepare=GME_PREPARE.get(seed))
    kind=common.spec_parts(candidate['spec'])[0]
    def diagnose(model):
        values=backend.assert_no_clamp(model)
        assert len(values)==48
        if capture is not None and next(model.parameters()).is_cuda:
            mode=model.training;model.eval()
            probe=torch.randn(2,3,224,224,generator=torch.Generator().manual_seed(999)).cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):capture['_trained_probe']=model(probe).cpu()
            model.train(mode)
        return [{'name':name,**row} for name,row in values.items()]
    def inject(model,method,rank,targets):
        started=time.perf_counter()
        result=backend.inject_adapters(model,method,rank=rank,targets=targets)
        if capture is not None:capture.update(initializer_seconds=time.perf_counter()-started,diagnostics=diagnose(result),initial_trainable_sha256=tensor_digest(
            (n,p) for n,p in result.named_parameters() if p.requires_grad))
        return result
    def verify_initial_frozen(model,frozen):
        if kind!='quantized':return
        assert verify_context is not None
        checkpoint,args=verify_context
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        rng_python=random.getstate();rng_numpy=np.random.get_state()
        recreated=base_model(checkpoint,seed,args)
        backend.inject_adapters(recreated,candidate['candidate_id'],rank=2,targets=lambda n,m:n.endswith(vision.TARGET_SUFFIXES))
        recreated.head.requires_grad_(True)
        expected=tensor_digest((n,p) for n,p in recreated.named_parameters() if not p.requires_grad)
        assert expected==frozen
        assert tensor_digest((n,p) for n,p in recreated.named_parameters() if p.requires_grad)==tensor_digest((n,p) for n,p in model.named_parameters() if p.requires_grad)
        if capture is not None:capture.update(deterministic_recreation_verified=True,expected_frozen_sha256=expected)
        del recreated
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);random.setstate(rng_python);np.random.set_state(rng_numpy)
    return SimpleNamespace(adapter_types=backend.adapter_types,parameter_groups=backend.parameter_groups,
        inject_adapters=inject,adapter_diagnostics=diagnose,recipe_manifest=backend.recipe_manifest,
        initializer_kind=kind,verify_initial_frozen=verify_initial_frozen)


def initialization_failure(error):
    message=str(error).lower()
    return isinstance(error,(RuntimeError,ValueError)) and any(token in message for token in (
        'nonfinite','failed to converge','did not converge','convergence failed','cor da inverse',
        'corda inverse','algorithm failed to converge','singular matrix','ill-conditioned'))


def controls(args,registry,datasets,checkpoint,calibrations,manifest):
    path=args.output/'controls/control.json'
    if path.exists():
        result=common.read(path);assert result['manifest_sha256']==common.sha(args.output/'manifest.json')
        for name,value in result['evidence_sha256'].items():assert common.sha(args.output/'controls'/name)==value
        return result
    dataset=copy.copy(datasets['train']);dataset.transform=datasets['val'].transform
    samples=[dataset[i] for i in range(4)]
    images=torch.stack([x for x,y in samples]).cuda();labels=torch.tensor([y for x,y in samples],device='cuda')
    byid={c['candidate_id']:c for c in registry['candidates']};anchor=byid[registry['control_id']]
    assert common.spec_parts(anchor['spec'])==('random',{'mode':'accepted'})
    ordered=[anchor];seen={common.digest_json(anchor['spec'])}
    for candidate in registry['candidates']:
        key=common.digest_json(candidate['spec'])
        if key not in seen:ordered.append(candidate);seen.add(key)
    rows=[];baseline=None;path.parent.mkdir(exist_ok=True)
    assert registry['study_lane']=='extension'
    for candidate in ordered:
        lane=candidate['initialization_lane']
        model=base_model(checkpoint,42,args);backend=bound_backend(candidate,42,calibrations,verify_context=(checkpoint,args))
        try:
            backend.inject_adapters(model,candidate['candidate_id'],2,lambda n,m:n.endswith(vision.TARGET_SUFFIXES))
        except (RuntimeError,ValueError) as error:
            if candidate['candidate_id']==registry['control_id'] or not initialization_failure(error):raise
            rows.append({'candidate_id':candidate['candidate_id'],'spec':candidate['spec'],'seed':42,
                'status':'numerical_failure','failure_stage':'initialization','failure':{'type':type(error).__name__,'message':str(error)},
                'optimizer_updates_completed':0,'tested_learning_rate':args.fixed_lr/2})
            del model;torch.cuda.empty_cache();continue
        model.head.requires_grad_(True);model.cuda().eval()
        rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):probe=model(images).cpu()
        if not torch.isfinite(probe).all():
            if candidate['candidate_id']==registry['control_id']:raise RuntimeError('Nonfinite accepted initial probe')
            zero_columns={name:(layer.weight.detach().float().norm(dim=0)==0).nonzero().flatten().tolist()
                for name,layer in model.named_modules() if hasattr(layer,'initialization')
                and (layer.weight.detach().float().norm(dim=0)==0).any()}
            rows.append({'candidate_id':candidate['candidate_id'],'spec':candidate['spec'],'seed':42,
                'status':'numerical_failure','failure_stage':'initialization','optimizer_updates_completed':0,
                'failure':{'type':'RuntimeError','message':'Nonfinite initializer output',
                    'nonfinite_probe_values':int((~torch.isfinite(probe)).sum()),'zero_norm_quantized_columns':zero_columns},
                'tested_learning_rate':args.fixed_lr/2})
            torch.save({'probe':probe,'rng':rng},path.parent/(candidate['candidate_id']+'.pt'))
            del model;torch.cuda.empty_cache();continue
        initial=backend.adapter_diagnostics(model)
        frozen=tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        if baseline is None:baseline={'probe':probe,'rng':rng,'frozen':frozen}
        for name,value in rng.items():assert torch.equal(value,baseline['rng'][name]),name
        identity=torch.equal(probe,baseline['probe'])
        if lane=='centered':assert identity
        if backend.initializer_kind=='quantized':
            assert lane=='quantized' and frozen!=baseline['frozen'];backend.verify_initial_frozen(model,frozen)
        else:assert frozen==baseline['frozen']
        drift={'maximum_absolute_error':float((probe-baseline['probe']).abs().max()),
               'mse':float((probe-baseline['probe']).square().mean()),
               'relative_l2':float((probe-baseline['probe']).norm()/baseline['probe'].norm())}
        initial_quality=None
        if lane!='centered':initial_quality=vision.evaluate(model,vision.loaders_for(datasets,args,42)['val'])
        groups=backend.parameter_groups(model,args.fixed_lr/2)
        groups.append({'params':list(model.head.parameters()),'param_names':['head.weight','head.bias'],'lr':args.head_lr/2,'weight_decay':.01})
        optimizer=torch.optim.AdamW(groups);model.train();optimizer.zero_grad(set_to_none=True)
        failure=None;loss_value=norm_value=None;updates_completed=0
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(images).float(),labels,label_smoothing=.1)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss_value=float(loss.detach());loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
            norm_value=float(norm);optimizer.step();updates_completed=1
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite parameter')
        except (RuntimeError,ValueError) as error:
            if not initialization_failure(error):raise
            failure={'type':type(error).__name__,'message':str(error)}
        assert frozen==tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        rows.append({'candidate_id':candidate['candidate_id'],'spec':candidate['spec'],'seed':42,
            'status':'numerical_failure' if failure else 'pass','failure':failure,
            'training_rng_matches_accepted':True,'initial_probe_bitwise_accepted':identity,'initial_probe_drift':drift,
            'initial_validation':initial_quality,'initial_frozen_sha256':frozen,'study_lane':lane,
            'initial_diagnostics':initial,'diagnostics':None if failure else backend.adapter_diagnostics(model),
            'initial_first_loss':loss_value,'first_step_gradient_norm':norm_value,'frozen_weights_unchanged':True,
            'tested_learning_rate':args.fixed_lr/2,'optimizer_updates_completed':updates_completed})
        torch.save({'probe':probe,'rng':rng},path.parent/(candidate['candidate_id']+'.pt'))
        del model,optimizer;torch.cuda.empty_cache()
    byspec={common.digest_json(row['spec']):row for row in rows}
    for candidate in registry['candidates']:
        if not candidate.get('comparison_control_id'):continue
        if candidate['spec'].get('kind')!='quantized' or candidate['spec'].get('config',{}).get('method')!='loftq':continue
        row=byspec[common.digest_json(candidate['spec'])]
        comparator=byid[candidate['comparison_control_id']]
        other=byspec[common.digest_json(comparator['spec'])]
        if row.get('failure_stage')=='initialization' or other.get('failure_stage')=='initialization':continue
        assert row['initial_frozen_sha256']==other['initial_frozen_sha256'],'Paired quantized treatment/control base mismatch'
        left=torch.load(path.parent/(row['candidate_id']+'.pt'),map_location='cpu',weights_only=True)['probe']
        right=torch.load(path.parent/(other['candidate_id']+'.pt'),map_location='cpu',weights_only=True)['probe']
        row.setdefault('comparison_controls',{})[candidate['comparison_control_id']]={
            'same_frozen_base':True,'maximum_absolute_error':float((left-right).abs().max()),
            'mse':float((left-right).square().mean()),'relative_l2':float((left-right).norm()/right.norm())}
    result={'status':'pass','manifest_sha256':common.sha(args.output/'manifest.json'),'controls':rows,
        'unique_specs':len(ordered),'task_gate_sha256':manifest['task_gate_sha256'],
        'scope':'Accepted taskgate has3realminibatch bitwise fulltrajectory parity; here every successfulinitializer gets1real4imageupdate and downstream RNG parity. Centered initialBF16logits matchaccepted; explicitwarmstart/quantized lanes recordinitialquality/drift andpairedquantizerbase checks. Numericalfailures retained.'}
    result['evidence_sha256']={p.name:common.sha(p) for p in path.parent.glob('*.pt')}
    common.write_once(path,result);return result


def prepare(args,registry):
    gate=common.read(args.gate_evidence);taskgate=common.read(args.task_gate_evidence)
    assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    assert taskgate['passed'] and all(row['passed'] for row in taskgate['cases'])
    for evidence in (gate,taskgate):
        for name,value in evidence['source_sha256'].items():assert common.sha(name)==value,name
    args.output.mkdir(parents=True,exist_ok=True)
    copy_once(args.reference_root/'split_manifest.json',args.output/'split_manifest.json')
    datasets,splits=vision.datasets_and_manifest(args)
    checkpoint=hf_hub_download(vision.REPO,'model.safetensors',revision=vision.REVISION,cache_dir='/var/tmp/dora-bench/cache/huggingface')
    has_data=any(common.spec_parts(c['spec'])[0] in ('extension_centered','extension_warmstart') for c in registry['candidates'])
    calibrations={seed:calibration(args,datasets,checkpoint,seed) for seed in args.seeds} if has_data else {}
    sources=sorted(set(gate['source_sha256'])|set(taskgate['source_sha256'])|{
        'experiments/init_extension_aircraft.py','experiments/init_extension.py','experiments/init_extension_runtime.py','experiments/init_extension_registry.py','experiments/init_extension_common.py','experiments/init_extension_observers.py','experiments/init_aircraft.py','experiments/init_aircraft_runtime.py','experiments/init_common.py','experiments/init_data.py','experiments/init_calibration.py',
        'experiments/second_round/vision.py','experiments/vision.py','experiments/init_ablations.py','experiments/nora_current.py',
        'experiments/next_ablations.py','experiments/change_ablations.py','experiments/second_round_adapters.py','dora.py'})
    reference=common.read(args.reference_root/'protocol.json')['provenance']
    provenance={key:reference[key] for key in ('model','model_revision','checkpoint_sha256','split_manifest_sha256','transform','precision')}
    assert provenance['checkpoint_sha256']==common.sha(checkpoint) and provenance['split_manifest_sha256']==common.sha(args.output/'split_manifest.json')
    provenance.update(source_sha256={name:common.sha(name) for name in sources},torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__)
    manifest={'schema':'dora-init-extension-aircraft-v1','task':'aircraft','study_lane':registry.get('study_lane','centered'),'registry':registry,'registry_sha256':common.sha(args.registry),
        'protocol_sha256':common.sha(args.protocol),'source_sha256':provenance['source_sha256'],'provenance':provenance,
        'gate_sha256':common.sha(args.gate_evidence),'task_gate_sha256':common.sha(args.task_gate_evidence),
        'split_manifest_sha256':common.sha(args.output/'split_manifest.json'),
        'calibration_manifest_sha256':{str(seed):common.sha(args.output/'calibration'/f'seed_{seed}'/'manifest.json') for seed in calibrations},
        'recipes':{str(seed):{c['candidate_id']:bound_backend(c,seed,calibrations).recipe_manifest(c['candidate_id'])
            for c in registry['candidates']} for seed in args.seeds},
        'configuration':{name:getattr(args,name) for name in ('epochs','pilot_epochs','batch_size','head_lr','fixed_lr','seeds','workers')},
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
        'scope':'Official deduplicated Aircraftsplit, unchanged10epochpilots/20epochfinals and validationbestcheckpoint. Perseed calibrationuses128trainimages withthatseeds unchangedrandomclassifier. Allpilotsbeforetest.'}
    common.write_once(args.output/'manifest.json',manifest)
    for name in sources:copy_once(name,args.output/'source'/name)
    for name,source in (('registry.json',args.registry),('protocol.md',args.protocol),('gate.json',args.gate_evidence),('task_gate.json',args.task_gate_evidence)):
        copy_once(source,args.output/'reference'/name)
    return datasets,checkpoint,calibrations,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('registry','protocol','output','gate-evidence','task-gate-evidence','study-root'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=('controls','pilots','select','finals','finalize'),required=True)
    parser.add_argument('--candidate-ids',nargs='+');parser.add_argument('--family-ids',nargs='+')
    parser.add_argument('--cache',type=Path,default=Path('/var/tmp/dora-bench/cache/aircraft'))
    parser.add_argument('--reference-root',type=Path,default=Path('/var/tmp/dora-bench/round2/vision/rank2_run'))
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    args.epochs,args.pilot_epochs,args.batch_size,args.head_lr,args.fixed_lr,args.seeds=20,10,128,.001,.001,[42,43,44]
    registry=common.load_registry(args.registry,args.protocol,'aircraft');assert registry['learning_rates']==list(vision.BASE_LRS)
    assert registry['fixed_lr']==args.fixed_lr
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if args.phase in ('select','finalize'):
        manifest=common.read(args.output/'manifest.json');assert manifest['registry']==registry
        for name,value in manifest['source_sha256'].items():assert common.sha(name)==value,name
        assert manifest['gate_sha256']==common.sha(args.gate_evidence) and manifest['task_gate_sha256']==common.sha(args.task_gate_evidence)
        datasets=checkpoint=calibrations=None
        control=common.read(args.output/'controls/control.json');assert control['status']=='pass'
    else:
        datasets,checkpoint,calibrations,manifest=prepare(args,registry)
        control=controls(args,registry,datasets,checkpoint,calibrations,manifest)
    if args.phase=='controls':return
    manifest_sha=common.sha(args.output/'manifest.json');byid={c['candidate_id']:c for c in registry['candidates']}
    def _fit(candidate,seed,stage):
        cid=candidate['candidate_id'];fit_id=f'fits/{cid}/{stage}/seed_{seed}'
        root=args.output/fit_id;path=root/'fit.json'
        recipe=manifest['recipes'][str(seed)][cid]
        if path.exists():
            row=common.read(path);assert row['manifest_sha256']==manifest_sha and row['recipe_sha256']==recipe['recipe_sha256']
            if row['status']=='pass':assert common.sha(args.output/row['artifact_directory']/'best_adapter_and_head.pt')==row['checkpoint_sha256']
            return row
        configuration=copy.copy(args);configuration.output=root
        capture={};backend=bound_backend(candidate,seed,calibrations,capture,verify_context=(checkpoint,args))
        relative=Path('rank_2')/('pilots' if stage=='pilot' else 'final')/cid/(f"lr_{candidate['learning_rate']:g}_m_1" if stage=='pilot' else f'seed_{seed}')
        artifact=root/relative
        row={'candidate_id':cid,'family_id':candidate['family_id'],'spec':candidate['spec'],'seed':seed,'stage':stage,
            'learning_rate':candidate['learning_rate'],'rank':2,'fit_id':fit_id,'artifact_directory':str(artifact.relative_to(args.output)),
            'manifest_sha256':manifest_sha,'recipe_sha256':recipe['recipe_sha256'],'candidate_sha256':common.digest_json(candidate),
            'started_utc':datetime.now(timezone.utc).isoformat(),'environment':{'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu':torch.cuda.get_device_name(),'command':[sys.executable,*sys.argv]}}
        provenance={**manifest['provenance'],'recipe':recipe,'manifest_sha256':manifest_sha}
        try:
            lane=candidate['initialization_lane']
            if lane=='centered':
                result=vision.run_one(configuration,datasets,checkpoint,provenance,cid,2,seed,candidate['learning_rate'],1.,stage,backend)
            else:
                from experiments.init_aircraft_runtime import run_one
                result=run_one(configuration,datasets,checkpoint,provenance,cid,2,seed,candidate['learning_rate'],1.,stage,backend,study_lane=lane)
            row.update(status='pass',result=result,checkpoint_sha256=result['best_adapter_checkpoint_sha256'])
            recreated=base_model(checkpoint,seed,args)
            replay_backend=bound_backend(candidate,seed,calibrations)
            replay_backend.inject_adapters(recreated,cid,rank=2,targets=lambda n,m:n.endswith(vision.TARGET_SUFFIXES))
            recreated.head.requires_grad_(True)
            assert tensor_digest((n,p) for n,p in recreated.named_parameters() if not p.requires_grad)==result['frozen_state_before_sha256']
            initial={r['name']:r['initializer'] for r in capture['diagnostics']}
            for record in replay_backend.adapter_diagnostics(recreated):
                for key in ('initial_up_sha256','initial_down_sha256','up_sha256','down_sha256','pretrained_weight_sha256'):
                    if key in initial[record['name']]:assert record['initializer'][key]==initial[record['name']][key]
            recreated.load_state_dict(torch.load(artifact/'best_adapter_and_head.pt',map_location='cpu',weights_only=True),strict=False)
            recreated.cuda().eval();probe=torch.randn(2,3,224,224,generator=torch.Generator().manual_seed(999)).cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):replay=recreated(probe).cpu()
            trained=capture.pop('_trained_probe');torch.testing.assert_close(trained,replay,rtol=0,atol=0)
            torch.save({'trained_probe':trained,'replayed_probe':replay},root/'probe_replay.pt')
            row['reload_validation']={'bitwise_checkpoint_probe':True,'initial_factors_recreated':True,'frozen_base_verified':True,
                                      'probe_sha256':common.sha(root/'probe_replay.pt')}
            del recreated;torch.cuda.empty_cache()
        except (RuntimeError,ValueError) as error:
            if not initialization_failure(error):raise
            history_path=artifact/'epochs.jsonl'
            completed_epochs=len(history_path.read_text().splitlines()) if history_path.exists() else 0
            row.update(status='numerical_failure',failure={'type':type(error).__name__,'message':str(error)},
                failure_stage='training' if history_path.exists() else 'initialization',completed_epochs=completed_epochs,
                completed_optimizer_updates_lower_bound=completed_epochs*math.ceil(3334/args.batch_size),
                completed_optimizer_updates_exact=None if history_path.exists() else 0,
                partial_epoch_update_count='Unavailable: unchanged inherited runner logs completed epochs only' if history_path.exists() else 'No optimizer update began')
        capture.pop('_trained_probe',None)
        row.update(initialization=capture,finished_utc=datetime.now(timezone.utc).isoformat())
        common.write_once(path,row);return row
    def fit(candidate,seed,stage):
        lock=args.output/'fits'/candidate['candidate_id']/stage/f'seed_{seed}'/'fit.lock'
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            return _fit(candidate,seed,stage)

    if args.phase=='pilots':
        assert not (args.output/'selections.json').exists() and not (args.output/'results.json').exists()
        selected=set(args.candidate_ids or byid);assert selected<=set(byid)
        for candidate in registry['candidates']:
            if candidate['candidate_id'] in selected:fit(candidate,42,'pilot')
        return
    pilots=[]
    for candidate in registry['candidates']:
        path=args.output/'fits'/candidate['candidate_id']/'pilot/seed_42/fit.json';row=common.read(path)
        assert row['manifest_sha256']==manifest_sha and row['candidate_sha256']==common.digest_json(candidate)
        record={key:row[key] for key in ('candidate_id','family_id','fit_id','status','learning_rate')}
        record['pilot_sha256']=common.sha(path)
        if row['status']=='pass':
            assert row['result']['test'] is None and not row['result']['evaluate_test']
            record.update(selection_value=row['result']['best_validation']['macro_class_accuracy'],
                selection_cross_entropy=row['result']['best_validation']['cross_entropy'],checkpoint_sha256=row['checkpoint_sha256'])
        else:record.update(selection_value=None,selection_cross_entropy=None,failure=row['failure'])
        pilots.append(record)
    pilot_byid={row['candidate_id']:row for row in pilots};families=sorted({c['family_id'] for c in registry['candidates']});selections={}
    for family in families:
        selections[family]={}
        for lane in LANES:
            eligible=[c for c in registry['candidates'] if c['family_id']==family and pilot_byid[c['candidate_id']]['status']=='pass'
                and (lane!='fixed_original_lr' or c['learning_rate']==args.fixed_lr)]
            selections[family][lane]=(min(eligible,key=lambda c:(-pilot_byid[c['candidate_id']]['selection_value'],
                pilot_byid[c['candidate_id']]['selection_cross_entropy'],c['learning_rate'],c['candidate_id']))['candidate_id'] if eligible else None)
    selected={'schema':'dora-init-selections-v1','manifest_sha256':manifest_sha,'registry_sha256':common.digest_json(registry),
        'test_used_for_selection':False,'all_pilots_complete_before_test':True,'pilots':pilots,'selections':selections,
        'selection':'Maximum validationmacro thenminimum validationCE, lowerLR, candidateID'}
    selection_path=args.output/'selections.json'
    if not selection_path.exists():assert args.phase=='select'
    common.write_once(selection_path,selected)
    if args.phase=='select':
        common.write_once(args.output/'pilots_complete.json',{'status':'pass','manifest_sha256':manifest_sha,
            'selections_sha256':common.sha(selection_path),'pilots':len(pilots),'families':len(families)});return
    pilot_gate=common.read(args.output/'pilots_complete.json');assert pilot_gate['status']=='pass' and pilot_gate['selections_sha256']==common.sha(selection_path)
    chosen=set(args.family_ids or families);assert chosen<=set(families)
    for family in families if args.phase=='finals' else []:
        if family not in chosen:continue
        records=[];paired_records=[]
        for seed in args.seeds:
            for cid in sorted({c for c in selections[family].values() if c is not None}):
                row=fit(byid[cid],seed,'final')
                candidate=byid[cid]
                if candidate['comparison_control_id']!=candidate['candidate_id']:
                    comparator=byid[candidate['comparison_control_id']];assert comparator['learning_rate']==candidate['learning_rate']
                    paired=fit(comparator,seed,'final');paired_records.append({**paired,'for_candidate_id':cid,'for_family_id':family})
                lanes=[lane for lane in LANES if selections[family][lane]==cid]
                records.extend({**row,'lane':lane,'shared_lanes':lanes} for lane in lanes)
        common.write_once(args.output/'family_results'/f'{family}.json',{'family_id':family,'complete':True,
            'manifest_sha256':manifest_sha,'selections_sha256':common.sha(selection_path),'runs':records,'paired_control_fits':paired_records})
    if args.phase=='finals' and args.family_ids:return
    records=[];paired_records=[]
    for family in families:
        row=common.read(args.output/'family_results'/f'{family}.json')
        assert row['complete'] and row['manifest_sha256']==manifest_sha and row['selections_sha256']==common.sha(selection_path)
        records.extend(row['runs']);paired_records.extend(row.get('paired_control_fits',[]))
    common.write_once(args.output/'results.json',{'schema':'dora-init-extension-aircraft-v1','task':'aircraft','complete':True,
        'manifest_sha256':manifest_sha,'selections_sha256':common.sha(selection_path),'runs':records,'paired_control_fits':paired_records,
        'unique_trained_fits':len(list((args.output/'fits').glob('*/*/*/fit.json'))),
        'unique_final_fits':len({r['fit_id'] for r in records+paired_records}),
        'unavailable_lanes':[{'family_id':family,'lane':lane} for family,choices in selections.items() for lane,cid in choices.items() if cid is None]})
    print(json.dumps({'event':'complete','task':'aircraft','families':len(families)}),flush=True)


if __name__=='__main__':main()
