"""Fresh-seed confirmation of validation-selected retrieval initializers.

The registry is independently rederived from the complete centered pilot grid.
Seeds45..49 use unchanged task training/evaluation and train-only calibration.
"""
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
from transformers import AutoTokenizer

from experiments import init_common as common
from experiments import init_confirm_common as confirmation
from experiments import init_retrieval as runner


def copy_once(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert common.sha(source)==common.sha(destination)
    else:shutil.copyfile(source,destination)


def prepare(args,registry):
    parent=common.read(args.study_root/'manifest.json');gate=common.read(args.gate_evidence)
    assert parent['task']==args.task and gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    for name,value in gate['source_sha256'].items():assert common.sha(name)==value,name
    for name,value in parent['source_sha256'].items():assert common.sha(name)==value,name
    for key,value in parent['configuration'].items():setattr(args,key,value)
    args.seeds=registry['seeds'];args.rank=8;args.device='cuda:0';args.threads=4
    args.fixed_lr=registry['fixed_lr'];args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('highest')
    historical=common.read(args.study_root/'reference/reference_provenance.json');cfg=historical['configuration']
    if args.task=='scifact':
        args.validation_fraction=cfg['validation_fraction'];args.split_seed=cfg['split_seed']
        corpus,queries,relevance,splits,metadata=runner.task.load_data(args)
        data={'scifact':dict(corpus=corpus,queries=queries,relevance=relevance,splits=splits,metadata=metadata)}
        negatives=None
    else:
        args.hard_negatives=cfg['hard_negatives'];data=runner.harder.load_data()
        negatives=common.read(args.study_root/'hard_negatives.json')
        copy_once(args.study_root/'hard_negatives.json',args.output/'hard_negatives.json')
    tokenizer=AutoTokenizer.from_pretrained(runner.task.MODEL,revision=runner.task.MODEL_REVISION)
    for name,ds in data.items():
        assert ds['splits']==common.read(args.study_root/(name+'_splits.json'))
        assert ds['metadata']==parent['datasets'][name]
        ds['corpus_tokens']=runner.task.tokenize_all(tokenizer,ds['corpus'],args.max_length)
        ds['query_tokens']=runner.task.tokenize_all(tokenizer,ds['queries'],args.max_length)
        for suffix in ('_splits.json','_qrels.json'):copy_once(args.study_root/(name+suffix),args.output/(name+suffix))
    # MiniLM has no newly randomized task head: reuse the exact train-only
    # calibration evidence, not statistics from any trained candidate.
    calibration_path=args.study_root/'calibration/manifest.json'
    runner.CALIBRATION=common.read(calibration_path) if calibration_path.exists() else None
    if runner.CALIBRATION is not None:
        assert common.sha(calibration_path)==parent['calibration_manifest_sha256']
        for name,file in runner.CALIBRATION['module_files'].items():assert common.sha(file)==runner.CALIBRATION['module_sha256'][name]
    sources=sorted(set(parent['source_sha256'])|set(gate['source_sha256'])|{
        'experiments/init_confirm_retrieval.py','experiments/init_confirm_common.py'})
    manifest={'schema':'dora-init-confirm-retrieval-v1','task':args.task,'registry':registry,
        'registry_sha256':common.sha(args.registry),'protocol_sha256':common.sha(args.protocol),
        'parent_manifest_sha256':common.sha(args.study_root/'manifest.json'),'gate_sha256':common.sha(args.gate_evidence),
        'source_sha256':{name:common.sha(name) for name in sources},'model':parent['model'],
        'configuration':{**parent['configuration'],'seeds':registry['seeds']},
        'calibration_reference':str(calibration_path) if runner.CALIBRATION is not None else None,
        'calibration_manifest_sha256':parent['calibration_manifest_sha256'],
        'recipes':{str(seed):{c['candidate_id']:runner.backend_for(c,seed).recipe_manifest(c['candidate_id']) for c in registry['candidates']} for seed in args.seeds}}
    common.write_once(args.output/'manifest.json',manifest)
    for name in sources:copy_once(Path(name),args.output/'source'/name)
    for source,name in ((args.registry,'registry.json'),(args.protocol,'protocol.md'),(args.gate_evidence,'gate.json')):
        copy_once(source,args.output/'reference'/name)
    return tokenizer,data,negatives,manifest


def controls(args,registry,tokenizer,data,negatives):
    receipt=args.output/'controls/control.json'
    if receipt.exists():
        result=common.read(receipt);assert result['manifest_sha256']==common.sha(args.output/'manifest.json')
        for name,value in result['evidence_sha256'].items():assert common.sha(receipt.parent/name)==value
        return
    candidate=next(c for c in registry['candidates'] if c['candidate_id']==registry['control_id'])
    assert common.spec_parts(candidate['spec'])==('random',{'mode':'accepted'})
    subset=copy.deepcopy(data);subset[args.task]['splits']['train']=subset[args.task]['splits']['train'][:160]
    settings=copy.copy(args);settings.epochs=1;states=[]
    for reference in (True,False):
        path=receipt.parent/('historical_formula' if reference else 'confirmation_backend');path.mkdir(parents=True,exist_ok=True)
        model=runner.new_model(candidate,45,reference=reference)
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
        frozen=runner.tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        training=runner.train_fit(model,tokenizer,subset,negatives,settings,candidate,45,candidate['learning_rate'],path)
        assert training['steps']==5
        assert frozen==runner.tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        final=torch.load(path/'adapter.pt',map_location='cpu',weights_only=True)
        rows=[json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
        for row in rows:row.pop('elapsed_seconds')
        states.append(dict(initial=initial,final=final,rng=rng,rows=rows,frozen=frozen))
        del model;torch.cuda.empty_cache()
    left,right=states
    assert left['rows']==right['rows'] and left['frozen']==right['frozen']
    for group in ('initial','final','rng'):
        assert list(left[group])==list(right[group])
        for name,value in left[group].items():torch.testing.assert_close(value,right[group][name],rtol=0,atol=0)
    common.write_once(receipt,{'status':'pass','seed':45,'steps':5,'initial_final_rng_trajectory_bitwise':True,
        'manifest_sha256':common.sha(args.output/'manifest.json'),
        'evidence_sha256':{str(p.relative_to(receipt.parent)):common.sha(p) for p in receipt.parent.rglob('*') if p.is_file()},
        'scope':'Fresh45 accepted initializer vs originalaccepted DoRA fullreal5minibatch trajectory, exacttrainables/RNG/loss/rates; no historical45trainedfit assumed.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registry','protocol','study-root','output','gate-evidence'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--task',choices=('scifact','nfcorpus'),required=True);p.add_argument('--phase',choices=('controls','fits','finalize'),required=True)
    p.add_argument('--candidate-ids',nargs='+');p.add_argument('--seeds',nargs='+',type=int);args=p.parse_args()
    requested_seeds=args.seeds
    registry=confirmation.load_registry(args.registry,args.protocol,args.study_root,args.task)
    if args.phase=='finalize':
        manifest=common.read(args.output/'manifest.json');assert manifest['registry']==registry
        for name,value in manifest['source_sha256'].items():assert common.sha(name)==value,name
    else:
        tokenizer,data,negatives,manifest=prepare(args,registry)
        controls(args,registry,tokenizer,data,negatives)
    if args.phase=='controls':return
    byid={c['candidate_id']:c for c in registry['candidates']};manifest_sha=common.sha(args.output/'manifest.json')
    if args.phase=='fits':
        cids=set(args.candidate_ids or byid);seeds=requested_seeds or registry['seeds']
        assert cids<=set(byid) and set(seeds)<=set(registry['seeds'])
        for cid in byid:
            if cid not in cids:continue
            candidate=byid[cid]
            for seed in seeds:
                path=args.output/'fits'/cid/f'seed_{seed}';path.mkdir(parents=True,exist_ok=True)
                receipt=path/'result.json'
                if receipt.exists():
                    result=common.read(receipt);assert result['manifest_sha256']==manifest_sha
                    if result['status']=='pass':assert common.sha(path/'adapter.pt')==result['checkpoint_sha256']
                    continue
                result={'candidate_id':cid,'family_id':candidate['family_id'],'spec':candidate['spec'],'rank':8,
                    'seed':seed,'learning_rate':candidate['learning_rate'],'fit_id':str(path.relative_to(args.output)),
                    'manifest_sha256':manifest_sha,'recipe_sha256':manifest['recipes'][str(seed)][cid]['recipe_sha256'],
                    'started_utc':datetime.now(timezone.utc).isoformat(),'environment':{'gpu':torch.cuda.get_device_name(),
                    'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'command':[sys.executable,*sys.argv]}}
                model=None
                try:
                    model=runner.new_model(candidate,seed)
                    result['initial_diagnostics']=runner.diagnostics_of(model,candidate,seed)
                    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();python_rng=random.getstate();numpy_rng=np.random.get_state()
                    reference=runner.new_model(next(c for c in registry['candidates'] if c['candidate_id']==registry['control_id']),seed,reference=True)
                    assert torch.equal(cpu_rng,torch.get_rng_state())
                    assert all(torch.equal(left,right) for left,right in zip(cuda_rng,torch.cuda.get_rng_state_all()))
                    probe_ids=data[args.task]['splits']['train'][:8]
                    probe_batch=runner.task.make_batch(tokenizer,data[args.task]['query_tokens'],probe_ids,args.device)
                    model.eval();reference.eval()
                    with torch.no_grad():
                        expected_probe=runner.task.embeddings(reference,probe_batch).cpu()
                        initial_probe=runner.task.embeddings(model,probe_batch).cpu()
                    torch.testing.assert_close(expected_probe,initial_probe,rtol=0,atol=0)
                    result['initial_parity']={'seed':seed,'probe_train_query_ids':probe_ids,'bitwise_accepted_probe':True,'cpu_cuda_rng_equal':True}
                    del reference
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);random.setstate(python_rng);np.random.set_state(numpy_rng)

                    result['trainable_parameter_order']=[n for n,t in model.named_parameters() if t.requires_grad]
                    result['trainable_parameters']=sum(t.numel() for t in model.parameters() if t.requires_grad)
                    result['frozen_sha256']=runner.tensor_digest((n,t) for n,t in model.named_parameters() if not t.requires_grad)
                    result['training']=runner.train_fit(model,tokenizer,data,negatives,args,candidate,seed,candidate['learning_rate'],path)
                    assert result['frozen_sha256']==runner.tensor_digest((n,t) for n,t in model.named_parameters() if not t.requires_grad)
                    result.update(status='pass',frozen_weights_unchanged=True,checkpoint_sha256=common.sha(path/'adapter.pt'),
                        evaluation=runner.evaluate(model,tokenizer,data,args,path,final=True),diagnostics=runner.diagnostics_of(model,candidate,seed))
                    model.eval()
                    with torch.no_grad():trained_probe=runner.task.embeddings(model,probe_batch).cpu()
                    recreated=runner.new_model(candidate,seed)
                    diagnostics=runner.diagnostics_of(recreated,candidate,seed)
                    for name,initial in result['initial_diagnostics'].items():
                        for key in ('initial_up_sha256','initial_down_sha256'):
                            if key in initial['initializer']:assert diagnostics[name]['initializer'][key]==initial['initializer'][key]
                    assert result['frozen_sha256']==runner.tensor_digest((n,t) for n,t in recreated.named_parameters() if not t.requires_grad)
                    state=torch.load(path/'adapter.pt',map_location='cuda:0',weights_only=True);recreated.load_state_dict(state,strict=False);recreated.eval()
                    with torch.no_grad():replayed_probe=runner.task.embeddings(recreated,probe_batch).cpu()
                    torch.testing.assert_close(trained_probe,replayed_probe,rtol=0,atol=0)
                    torch.save({'initial_probe':initial_probe,'trained_probe':trained_probe,'replayed_probe':replayed_probe},path/'probe_replay.pt')
                    result['reload_validation']={'bitwise_checkpoint_probe':True,'initial_compensation_buffers_verified':True,
                        'frozen_base_verified':True,'probe_sha256':common.sha(path/'probe_replay.pt')}
                    del recreated
                except (RuntimeError,ValueError) as error:
                    if not runner.initialization_failure(error):raise
                    result.update(status='numerical_failure',failure={'type':type(error).__name__,'message':str(error)},evaluation={},
                        failure_stage='training' if (path/'training.jsonl').exists() else 'initialization',
                        completed_steps=len((path/'training.jsonl').read_text().splitlines()) if (path/'training.jsonl').exists() else 0)
                result['finished_utc']=datetime.now(timezone.utc).isoformat();common.write_once(receipt,result)
                del model;torch.cuda.empty_cache()
                print(json.dumps({'event':'confirmation_fit','candidate_id':cid,'seed':seed,'status':result['status']}),flush=True)
        return
    records=[]
    for role,cid in registry['roles'].items():
        for seed in registry['seeds']:
            result=common.read(args.output/'fits'/cid/f'seed_{seed}'/'result.json')
            assert result['manifest_sha256']==manifest_sha
            records.append({**result,'role':role,'shared_roles':[key for key,value in registry['roles'].items() if value==cid]})
    common.write_once(args.output/'results.json',{'schema':'dora-init-confirm-retrieval-v1','complete':True,
        'task':args.task,'manifest_sha256':manifest_sha,'runs':records,'unique_trained_fits':len(byid)*len(registry['seeds'])})


if __name__=='__main__':main()
