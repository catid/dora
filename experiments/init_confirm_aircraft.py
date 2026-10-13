"""Five fresh-seed Aircraft confirmations with per-seed original-head calibration."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import json
import os
import random
import numpy as np
from pathlib import Path
import shutil
import sys
import torch
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from experiments import init_common as common
from experiments import init_confirm_common as confirmation
from experiments import init_aircraft as runner
from experiments import init_aircraft_validate as parity
from experiments import nora_current as accepted
from experiments.second_round import vision
from experiments.vision import seed_everything,tensor_digest


def copy_once(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert common.sha(source)==common.sha(destination)
    else:shutil.copyfile(source,destination)


def fresh_initialization_references(args,checkpoint):
    # These are independently regenerated untrained initialization receipts,
    # explicitly not reused historical trained head-only baseline results.
    for seed in args.seeds:
        seed_everything(seed);model=timm.create_model(vision.MODEL,pretrained=False)
        model.load_state_dict(load_file(checkpoint),strict=True);model.reset_classifier(100)
        receipt={'schema':'dora-fresh-untrained-initialization-v1','seed':seed,
            'initial_head_sha256':tensor_digest(model.head.state_dict().items()),
            'initial_backbone_sha256':tensor_digest((n,p) for n,p in model.named_parameters() if not n.startswith('head.')),
            'provenance':{'checkpoint_sha256':common.sha(checkpoint),'model':vision.MODEL,'model_revision':vision.REVISION},
            'training_performed':False,'historical_trained_baseline':False,
            'scope':'New seed-specific untrained classifier and pretrained backbone; used only to verify deterministic initialization.'}
        common.write_once(args.reference_root/'baseline'/f'seed_{seed}'/'result.json',receipt)
        del model


def prepare(args,registry):
    parent=common.read(args.study_root/'manifest.json');gate=common.read(args.gate_evidence)
    assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    for name,value in gate['source_sha256'].items():assert common.sha(name)==value,name
    for name,value in parent['source_sha256'].items():assert common.sha(name)==value,name
    args.epochs,args.pilot_epochs,args.batch_size,args.head_lr,args.fixed_lr,args.seeds=20,10,128,.001,.001,registry['seeds']
    args.output.mkdir(parents=True,exist_ok=True);args.reference_root=args.output/'initialization_reference'
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    copy_once(args.study_root/'split_manifest.json',args.output/'split_manifest.json')
    datasets,splits=vision.datasets_and_manifest(args)
    checkpoint=hf_hub_download(vision.REPO,'model.safetensors',revision=vision.REVISION,cache_dir='/var/tmp/dora-bench/cache/huggingface')
    assert common.sha(checkpoint)==parent['provenance']['checkpoint_sha256']
    fresh_initialization_references(args,checkpoint)
    has_data=any(common.spec_parts(c['spec'])[0]=='data' for c in registry['candidates'])
    calibrations={seed:runner.calibration(args,datasets,checkpoint,seed) for seed in args.seeds} if has_data else {}
    sources=sorted(set(parent['source_sha256'])|set(gate['source_sha256'])|{
        'experiments/init_confirm_aircraft.py','experiments/init_confirm_common.py','experiments/init_aircraft_validate.py'})
    provenance={**parent['provenance'],'source_sha256':{name:common.sha(name) for name in sources}}
    manifest={'schema':'dora-init-confirm-aircraft-v1','task':'aircraft','registry':registry,'registry_sha256':common.sha(args.registry),
        'protocol_sha256':common.sha(args.protocol),'parent_manifest_sha256':common.sha(args.study_root/'manifest.json'),
        'gate_sha256':common.sha(args.gate_evidence),'source_sha256':provenance['source_sha256'],'provenance':provenance,
        'split_manifest_sha256':common.sha(args.output/'split_manifest.json'),
        'configuration':{'epochs':20,'pilot_epochs':10,'batch_size':128,'head_lr':.001,'seeds':args.seeds,'workers':args.workers},
        'initialization_reference_kind':'Fresh independently generated untrained initialization; no historical trained fit exists for45..49',
        'initialization_reference_sha256':{str(seed):common.sha(args.reference_root/'baseline'/f'seed_{seed}'/'result.json') for seed in args.seeds},
        'calibration_manifest_sha256':{str(seed):common.sha(args.output/'calibration'/f'seed_{seed}'/'manifest.json') for seed in calibrations},
        'recipes':{str(seed):{c['candidate_id']:runner.bound_backend(c,seed,calibrations).recipe_manifest(c['candidate_id']) for c in registry['candidates']} for seed in args.seeds}}
    common.write_once(args.output/'manifest.json',manifest)
    for name in sources:copy_once(Path(name),args.output/'source'/name)
    for source,name in ((args.registry,'registry.json'),(args.protocol,'protocol.md'),(args.gate_evidence,'gate.json')):copy_once(source,args.output/'reference'/name)
    return datasets,checkpoint,calibrations,manifest


def controls(args,registry,datasets,checkpoint,calibrations):
    path=args.output/'controls/control.json'
    if path.exists():
        row=common.read(path);assert row['status']=='pass' and row['manifest_sha256']==common.sha(args.output/'manifest.json');return
    candidate=next(c for c in registry['candidates'] if c['candidate_id']==registry['control_id'])
    assert common.spec_parts(candidate['spec'])==('random',{'mode':'accepted'})
    base=runner.base_model(checkpoint,45,args);initial_rng=torch.get_rng_state()
    batches=[];inputs=[]
    for index in range(3):
        seed_everything(20261009+index);samples=[datasets['train'][i] for i in range(4)]
        images=torch.stack([x for x,y in samples]);labels=torch.tensor([y for x,y in samples])
        inputs.append({'augmentation_seed':20261009+index,'train_indices':list(range(4)),
            'labels':labels.tolist(),'tensor_sha256':tensor_digest([('images',images),('labels',labels)])})
        batches.append((images.cuda(),labels.cuda()))
    backend=runner.bound_backend(candidate,45,calibrations)
    left,right=copy.deepcopy(base),copy.deepcopy(base)
    torch.set_rng_state(initial_rng);accepted.inject_adapters(left,'current_dora',rank=2,targets=lambda n,m:n.endswith(vision.TARGET_SUFFIXES))
    expected_rng=torch.get_rng_state();torch.set_rng_state(initial_rng)
    backend.inject_adapters(right,candidate['candidate_id'],rank=2,targets=lambda n,m:n.endswith(vision.TARGET_SUFFIXES))
    assert torch.equal(expected_rng,torch.get_rng_state())
    left.head.requires_grad_(True);right.head.requires_grad_(True)
    initial=parity.compare(left,right);left.cuda();right.cuda()
    def optimizer(model,implementation):
        groups=implementation.parameter_groups(model,candidate['learning_rate']/2)
        groups.append({'params':list(model.head.parameters()),'param_names':['head.weight','head.bias'],'lr':.0005,'weight_decay':.01})
        return torch.optim.AdamW(groups)
    lopt,ropt=optimizer(left,accepted),optimizer(right,backend);steps=[]
    frozen=tensor_digest((n,p) for n,p in left.named_parameters() if not p.requires_grad)
    for index,(images,labels) in enumerate(batches):
        rng=torch.cuda.get_rng_state();lv=parity.backward(left,lopt,images,labels);after=torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng);rv=parity.backward(right,ropt,images,labels);assert torch.equal(after,torch.cuda.get_rng_state())
        for first,second in zip(lv,rv):assert torch.equal(first,second)
        grads=parity.compare(left,right,gradients=True);lopt.step();ropt.step();parameters=parity.compare(left,right)
        steps.append({'step':index+1,'loss':float(lv[1]),'preclip_gradient_norm':float(lv[2]),
            'logits_loss_clip_bitwise_equal':True,'gradients':grads,'parameters_after_adamw':parameters})
    assert frozen==tensor_digest((n,p) for n,p in right.named_parameters() if not p.requires_grad)
    common.write_once(path,{'status':'pass','seed':45,'manifest_sha256':common.sha(args.output/'manifest.json'),
        'initial_parameters':initial,'inputs':inputs,'steps':steps,'frozen_weights_unchanged':True,
        'scope':'Fresh45 fullViT acceptedoriginalvsnew backend exact3realtrainminibatch logits/loss/gradients/AdamW. No historical45trainedbaseline assumed.'})
    del left,right,base,lopt,ropt,batches;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registry','protocol','study-root','output','gate-evidence'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('controls','fits','finalize'),required=True);p.add_argument('--candidate-ids',nargs='+');p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--cache',type=Path,default=Path('/var/tmp/dora-bench/cache/aircraft'));p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    requested_seeds=args.seeds;registry=confirmation.load_registry(args.registry,args.protocol,args.study_root,'aircraft')
    if args.phase=='finalize':
        manifest=common.read(args.output/'manifest.json');assert manifest['registry']==registry
        for name,value in manifest['source_sha256'].items():assert common.sha(name)==value,name
    else:
        datasets,checkpoint,calibrations,manifest=prepare(args,registry)
        controls(args,registry,datasets,checkpoint,calibrations)
    if args.phase=='controls':return
    byid={c['candidate_id']:c for c in registry['candidates']};manifest_sha=common.sha(args.output/'manifest.json')
    if args.phase=='fits':
        requested=set(args.candidate_ids or byid);seeds=requested_seeds or registry['seeds']
        assert requested<=set(byid) and set(seeds)<=set(registry['seeds'])
        for cid in byid:
            if cid not in requested:continue
            candidate=byid[cid]
            for seed in seeds:
                root=args.output/'fits'/cid/f'seed_{seed}';path=root/'fit.json';root.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    row=common.read(path);assert row['manifest_sha256']==manifest_sha
                    if row['status']=='pass':assert common.sha(args.output/row['artifact_directory']/'best_adapter_and_head.pt')==row['checkpoint_sha256']
                    continue
                capture={};backend=runner.bound_backend(candidate,seed,calibrations,capture)
                original_inject=backend.inject_adapters;original_diagnose=backend.adapter_diagnostics
                def checked_inject(model,method,rank,targets):
                    model=original_inject(model,method,rank,targets)
                    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();python_rng=random.getstate();numpy_rng=np.random.get_state()
                    reference=runner.base_model(checkpoint,seed,args)
                    accepted.inject_adapters(reference,'current_dora',rank=2,targets=targets)
                    assert torch.equal(cpu_rng,torch.get_rng_state())
                    assert all(torch.equal(left,right) for left,right in zip(cuda_rng,torch.cuda.get_rng_state_all()))
                    probe=torch.randn(2,3,224,224,generator=torch.Generator().manual_seed(999))
                    reference.eval();model.eval()
                    with torch.no_grad():
                        expected=reference(probe);actual=model(probe)
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    capture['initial_parity']={'seed':seed,'bitwise_accepted_probe':True,'cpu_cuda_rng_equal':True}
                    torch.save({'initial_probe':actual},root/'initial_probe.pt')
                    capture['initial_probe_sha256']=common.sha(root/'initial_probe.pt')
                    del reference
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);random.setstate(python_rng);np.random.set_state(numpy_rng)
                    return model
                def checked_diagnose(model):
                    result=original_diagnose(model)
                    if next(model.parameters()).is_cuda:
                        mode=model.training;model.eval()
                        probe=torch.randn(2,3,224,224,generator=torch.Generator().manual_seed(999)).cuda()
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):value=model(probe).cpu()
                        torch.save(value,root/'trained_probe.pt');model.train(mode)
                    return result
                backend.inject_adapters=checked_inject;backend.adapter_diagnostics=checked_diagnose

                configuration=copy.copy(args);configuration.output=root
                artifact=root/'rank_2/final'/cid/f'seed_{seed}';recipe=manifest['recipes'][str(seed)][cid]
                row={'candidate_id':cid,'family_id':candidate['family_id'],'spec':candidate['spec'],'rank':2,'stage':'final',
                    'seed':seed,'learning_rate':candidate['learning_rate'],'fit_id':str(root.relative_to(args.output)),
                    'artifact_directory':str(artifact.relative_to(args.output)),'manifest_sha256':manifest_sha,'recipe_sha256':recipe['recipe_sha256'],
                    'started_utc':datetime.now(timezone.utc).isoformat(),'environment':{'gpu':torch.cuda.get_device_name(),
                        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'command':[sys.executable,*sys.argv]}}
                provenance={**manifest['provenance'],'recipe':recipe,'manifest_sha256':manifest_sha}
                try:
                    result=vision.run_one(configuration,datasets,checkpoint,provenance,cid,2,seed,candidate['learning_rate'],1.,'final',backend)
                    row.update(status='pass',result=result,checkpoint_sha256=result['best_adapter_checkpoint_sha256'])
                    recreated=runner.base_model(checkpoint,seed,args);reload_backend=runner.bound_backend(candidate,seed,calibrations)
                    reload_backend.inject_adapters(recreated,cid,rank=2,targets=lambda n,m:n.endswith(vision.TARGET_SUFFIXES));recreated.head.requires_grad_(True)
                    diagnostics=reload_backend.adapter_diagnostics(recreated)
                    old={r['name']:r for r in capture['diagnostics']}
                    for record in diagnostics:
                        for key in ('initial_up_sha256','initial_down_sha256'):
                            if key in record['initializer']:assert record['initializer'][key]==old[record['name']]['initializer'][key]
                    assert tensor_digest((n,p) for n,p in recreated.named_parameters() if not p.requires_grad)==result['frozen_state_before_sha256']
                    state=torch.load(artifact/'best_adapter_and_head.pt',map_location='cpu',weights_only=True)
                    recreated.load_state_dict(state,strict=False);recreated.cuda().eval()
                    probe=torch.randn(2,3,224,224,generator=torch.Generator().manual_seed(999)).cuda()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):replayed=recreated(probe).cpu()
                    trained=torch.load(root/'trained_probe.pt',map_location='cpu',weights_only=True)
                    torch.testing.assert_close(trained,replayed,rtol=0,atol=0)
                    torch.save({'trained_probe':trained,'replayed_probe':replayed},root/'probe_replay.pt')
                    row['reload_validation']={'bitwise_checkpoint_probe':True,'initial_compensation_buffers_verified':True,
                        'frozen_base_verified':True,'probe_sha256':common.sha(root/'probe_replay.pt')}
                    del recreated;torch.cuda.empty_cache()

                except (RuntimeError,ValueError) as error:
                    if not runner.initialization_failure(error):raise
                    row.update(status='numerical_failure',failure={'type':type(error).__name__,'message':str(error)},
                        failure_stage='training' if (artifact/'epochs.jsonl').exists() else 'initialization',
                        completed_epochs=len((artifact/'epochs.jsonl').read_text().splitlines()) if (artifact/'epochs.jsonl').exists() else 0)
                row.update(initialization=capture,finished_utc=datetime.now(timezone.utc).isoformat());common.write_once(path,row)
        return
    records=[]
    for role,cid in registry['roles'].items():
        for seed in registry['seeds']:
            row=common.read(args.output/'fits'/cid/f'seed_{seed}'/'fit.json');assert row['manifest_sha256']==manifest_sha
            records.append({**row,'role':role,'shared_roles':[key for key,value in registry['roles'].items() if value==cid]})
    common.write_once(args.output/'results.json',{'schema':'dora-init-confirm-aircraft-v1','task':'aircraft','complete':True,
        'manifest_sha256':manifest_sha,'runs':records,'unique_trained_fits':len(byid)*len(registry['seeds'])})


if __name__=='__main__':main()
