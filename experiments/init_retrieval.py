"""Staged, registry-bound initializer experiments on unchanged retrieval tasks.

Delegates optimization, batches, encoding and scoring to the measured task
functions. A fit is uniquely identified by recipe, rate and seed; full-budget
seed42 validation fits and identical fixed/tuned fits are explicitly shared.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from experiments import init_ablations as ab
from experiments import nora_current as previous
from experiments import next_ablations as inherited_next
from experiments import change_ablations as historical
from experiments import retrieval as task
from experiments.second_round import retrieval as harder

CALIBRATION = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_once(path,value):
    path=Path(path)
    if path.exists():assert read(path)==value,path
    else:write(path,value)


def read(path):
    return json.loads(Path(path).read_text())


def tensor_digest(named):
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def spec_of(candidate):
    from experiments import init_common
    kind,config=init_common.spec_parts(candidate['spec'])
    assert kind=='random'
    return ab.InitSpec(**config)


def candidate_digest(candidate):
    payload={'spec':candidate['spec'],'learning_rate':candidate['learning_rate']}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def new_model(candidate, seed, *, reference=False):
    task.seed_everything(seed)
    model = AutoModel.from_pretrained(task.MODEL, revision=task.MODEL_REVISION,
                                      dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    task.seed_everything(seed)
    if reference:
        return previous.inject_adapters(model,'current_dora',rank=8,targets=('query','value'))
    return backend_for(candidate,seed).inject_adapters(model,candidate['candidate_id'],
                                                      rank=8,targets=('query','value'))


def train_fit(model, tokenizer, data, negatives, args, candidate, seed, lr, directory):
    method=candidate['candidate_id']
    args.current_method, args.current_rank, args.learning_rate = method, 8, lr
    if args.task == 'scifact':
        ds = data['scifact']
        return task.train(model, tokenizer, ds['corpus_tokens'], ds['query_tokens'], ds['relevance'], ds['splits'], args, seed, directory)
    return harder.train(model, tokenizer, data['nfcorpus'], negatives, args, method, seed, lr, 1.0,
                        directory, adapter_backend=backend_for(candidate,seed))


def evaluate(model, tokenizer, data, args, directory, *, final=False):
    result = {}
    for name, ds in data.items():
        splits = ('validation', 'test') if final and 'validation' in ds['splits'] else ('test',) if final else ('validation',)
        if not final and 'validation' not in ds['splits']:
            continue
        destination = directory / name
        destination.mkdir(parents=True, exist_ok=True)
        result[name] = task.evaluate(model, tokenizer, ds['corpus_tokens'], ds['query_tokens'], ds['relevance'], ds['splits'], args, destination, splits)
    return result


def backend_for(candidate, seed):
    from experiments import init_common
    return init_common.backend_for(candidate, seed, calibration=CALIBRATION)


def recipe_of(candidate):
    return backend_for(candidate,42).recipe_manifest(candidate['candidate_id'])


def diagnostics_of(model, candidate, seed):
    result=backend_for(candidate,seed).assert_no_clamp(model)
    assert len(result)==12,'Expected every retrieval query/value adapter to produce diagnostics'
    return result


def numerical_failure(error):
    return isinstance(error,RuntimeError) and any(text in str(error).lower() for text in
        ('nonfinite loss','nonfinite gradient','nonfinite parameter','nonfinite evaluation'))


def prepare_calibration(tokenizer,data,negatives,args,registry):
    from experiments import init_common
    if not any(init_common.spec_parts(c['spec'])[0] in ('data','warmstart') for c in registry['candidates']):return None
    from experiments.init_calibration import CalibrationCollector
    directory=args.output/'calibration';path=directory/'manifest.json'
    ds=data[args.task];generator=random.Random(20261009)
    ids=ds['splits']['train'].copy();generator.shuffle(ids);ids=ids[:128]
    assert len(ids)==128 and not set(ids)&set(ds['splits']['validation']) and not set(ids)&set(ds['splits']['test'])
    if path.exists():
        result=read(path)
        assert result['metadata']['example_ids']==ids
        assert result['metadata']['calibration_seed']==20261009
        assert result['metadata']['source_sha256']==sha('experiments/init_calibration.py')
        for name,file in result['module_files'].items():assert sha(file)==result['module_sha256'][name]
        return result
    task.seed_everything(42)
    model=AutoModel.from_pretrained(task.MODEL,revision=task.MODEL_REVISION,dtype=torch.bfloat16,
                                    attn_implementation='sdpa').cuda()
    groups={}
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear) and name.endswith(('.query','.value')):
            groups.setdefault(name.rsplit('.',1)[-1],[]).append(float(module.weight.detach().float().std()))
    group_std={name:round(sum(values)/len(values),3) for name,values in groups.items()}
    batches=[]
    with CalibrationCollector(model,targets=('query','value')) as collector:
        for start in range(0,len(ids),8):
            qids=ids[start:start+8]
            positives=[generator.choice(sorted(ds['relevance'][qid])) for qid in qids]
            docs=(list(dict.fromkeys(positives+[doc for qid in qids for doc in negatives[qid]]))
                  if negatives is not None else positives)
            positive_mask=torch.tensor([[doc in ds['relevance'][qid] for doc in docs] for qid in qids],device=args.device)
            assert positive_mask.any(dim=1).all()
            query=task.make_batch(tokenizer,ds['query_tokens'],qids,args.device)
            document=task.make_batch(tokenizer,ds['corpus_tokens'],docs,args.device)
            collector.begin_batch(example_ids=qids,attention_mask=query['attention_mask'])
            qemb=task.embeddings(model,query)
            collector.set_mask(document['attention_mask'])
            demb=task.embeddings(model,document)
            scores=qemb@demb.T/args.temperature
            loss=(scores.logsumexp(dim=1)-scores.masked_fill(~positive_mask,-torch.inf).logsumexp(dim=1)).mean()
            collector.backward(loss,loss_weight=len(qids))
            batches.append({'query_ids':qids,'positive_document_ids':positives,'document_ids':docs,
                            'query_nonpadding_tokens':int(query['attention_mask'].sum()),
                            'document_nonpadding_tokens':int(document['attention_mask'].sum()),
                            'positive_mask_sha256':tensor_digest([('positive_mask',positive_mask)]),
                            'loss':float(loss.detach())})
        result=collector.save(directory,{'task':args.task,'calibration_seed':20261009,'batch_size':8,
            'train_only':True,'selection':'Fixed seed20261009 shuffle of exact task train query IDs, first128; no val/test query IDs',
            'loss':'Unchanged multi-positive contrastive logsumexp loss, same temperature and corpus/hard negatives; evalmode, batch8',
            'batches':batches,'training_population_size':len(ds['splits']['train']),'inilora_group_std':group_std,
            'factor_cache_dir':str((args.output/'factor_cache').resolve()),
            'decomposition_cache_dir':str((args.output/'decomposition_cache').resolve()),
            'model_repository':task.MODEL,'model_revision':task.MODEL_REVISION,
            'split_manifest_sha256':sha(args.output/f'{args.task}_splits.json')})
    del model;torch.cuda.empty_cache()
    return result


def initialization_failure(error):
    message=str(error).lower()
    return isinstance(error,(RuntimeError,ValueError)) and any(token in message for token in (
        'nonfinite','failed to converge','did not converge','convergence failed','cor da inverse',
        'corda inverse','algorithm failed to converge','singular matrix','ill-conditioned'))


def controls(tokenizer, data, negatives, args, registry):
    destination=args.output/'controls';destination.mkdir(exist_ok=True)
    receipt=destination/'control.json'
    if receipt.exists():
        value=read(receipt)
        assert value['status']=='pass' and value['manifest_sha256']==sha(args.output/'manifest.json')
        for name,digest in value['evidence_sha256'].items():assert sha(destination/name)==digest,name
        return value
    configuration=copy.copy(args);configuration.epochs=1
    subset=copy.deepcopy(data);dataset=args.task
    subset[dataset]['splits']['train']=subset[dataset]['splits']['train'][:160]
    byid={c['candidate_id']:c for c in registry['candidates']}
    control=byid[registry['control_id']]
    assert spec_of(control).mode=='accepted'
    ordered=[control];seen={json.dumps(control['spec'],sort_keys=True)}
    for candidate in registry['candidates']:
        key=json.dumps(candidate['spec'],sort_keys=True)
        if key not in seen:ordered.append(candidate);seen.add(key)
    records=[];baseline=None
    lane=registry.get('study_lane','centered');assert lane in ('centered','warmstart','quantized')
    for candidate in ordered:
        states=[]
        reference=candidate['candidate_id']==registry['control_id']
        control_data=copy.deepcopy(subset)
        if not reference:control_data[dataset]['splits']['train']=control_data[dataset]['splits']['train'][:32]
        for original in ([True,False] if reference else [False]):
            label='historical_current_dora' if original else candidate['candidate_id']
            directory=destination/label;directory.mkdir(exist_ok=True)
            try:
                model=new_model(candidate,42,reference=original)
            except (RuntimeError,ValueError) as error:
                if reference or not initialization_failure(error):raise
                records.append({'candidate_id':candidate['candidate_id'],'spec':candidate['spec'],
                    'recipe_sha256':recipe_of(candidate)['recipe_sha256'],'steps_requested':1,'steps_completed':0,
                    'status':'numerical_failure','failure_stage':'initialization','failure':{'type':type(error).__name__,'message':str(error)},
                    'tested_learning_rate':args.fixed_lr,'accepted_historical_bitwise_trajectory':False})
                states=[];break
            frozen=tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
            initialized={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            assert all(torch.isfinite(t).all() for t in initialized.values()),'Invalid nonfinite initializer'
            rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
            initial_diagnostics=None if original else diagnostics_of(model,candidate,42)
            model.eval()
            probe_ids=data[dataset]['splits']['train'][:8]
            with torch.no_grad():
                probe=task.embeddings(model,task.make_batch(tokenizer,data[dataset]['query_tokens'],probe_ids,args.device)).cpu()
            failure=None
            try:
                training=train_fit(model,tokenizer,control_data,negatives,configuration,candidate,42,args.fixed_lr,directory)
                assert training['steps']==(5 if reference else 1)
                if not all(torch.isfinite(t).all() for t in model.parameters()):raise RuntimeError('Nonfinite parameter after control')
            except RuntimeError as error:
                if not numerical_failure(error) or reference:raise
                failure={'type':type(error).__name__,'message':str(error)}
            assert frozen==tensor_digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
            diagnostics=(previous.assert_no_clamp(model) if original else diagnostics_of(model,candidate,42)) if failure is None else None
            rows=[json.loads(line) for line in (directory/'training.jsonl').read_text().splitlines()]
            for row in rows:
                row.pop('elapsed_seconds')
                assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
            state=torch.load(directory/'adapter.pt',map_location='cpu',weights_only=True) if failure is None else None
            if state is not None:assert all(torch.isfinite(t).all() for t in state.values())
            states.append({'initialized':initialized,'state':state,'rng':rng,'rows':rows,
                           'frozen':frozen,'diagnostics':diagnostics,'initial_diagnostics':initial_diagnostics,
                           'failure':failure,'probe':probe})
            del model;torch.cuda.empty_cache()
        if not states:continue
        actual=states[-1]
        if reference:
            old=states[0]
            assert old['rows']==actual['rows'] and old['frozen']==actual['frozen']
            for key in old['rng']:assert torch.equal(old['rng'][key],actual['rng'][key])
            for field in ('initialized','state'):
                assert list(old[field])==list(actual[field])
                for name,tensor in old[field].items():
                    assert tensor.dtype==actual[field][name].dtype and tensor.shape==actual[field][name].shape
                    torch.testing.assert_close(tensor,actual[field][name],rtol=0,atol=0)
            baseline=actual
        assert baseline is not None
        for key in baseline['rng']:assert torch.equal(baseline['rng'][key],actual['rng'][key])
        identity=torch.equal(baseline['probe'],actual['probe'])
        kind=candidate['spec'].get('kind','random')
        if lane=='centered':assert identity
        if kind=='quantized':assert lane=='quantized' and actual['frozen']!=baseline['frozen']
        else:assert actual['frozen']==baseline['frozen']
        drift={'maximum_absolute_error':float((actual['probe']-baseline['probe']).abs().max()),
               'mse':float((actual['probe']-baseline['probe']).square().mean()),
               'relative_l2':float((actual['probe']-baseline['probe']).norm()/baseline['probe'].norm())}
        torch.save({'probe':actual['probe'],'rng':actual['rng']},destination/(candidate['candidate_id']+'_initial.pt'))
        matching=[c['candidate_id'] for c in registry['candidates'] if c['spec']==candidate['spec'] and c['learning_rate']==args.fixed_lr]
        records.append({'candidate_id':candidate['candidate_id'],'spec':candidate['spec'],
                        'recipe_sha256':recipe_of(candidate)['recipe_sha256'],'steps_requested':5 if reference else 1,
                        'steps_completed':len(actual['rows']),'status':'numerical_failure' if actual['failure'] else 'pass',
                        'failure':actual['failure'],'tested_learning_rate':args.fixed_lr,
                        'matching_pilot_candidate_ids':matching,
                        'accepted_historical_bitwise_trajectory':reference,
                        'training_rng_matches_accepted':True,'initial_probe_bitwise_accepted':identity,'initial_probe_drift':drift,'study_lane':lane,
                        'finite_trajectory_and_parameters':actual['failure'] is None,
                        'initial_first_loss':actual['rows'][0]['loss'] if actual['rows'] else None,
                        'first_step_gradient_norm':actual['rows'][0]['gradient_norm'] if actual['rows'] else None,
                        'frozen_sha256':actual['frozen'],'diagnostics':actual['diagnostics'],
                        'initial_diagnostics':actual['initial_diagnostics']})
    byspec={json.dumps(row['spec'],sort_keys=True):row for row in records}
    for candidate in registry['candidates']:
        if not candidate.get('comparison_control_id'):continue
        if candidate['spec'].get('kind')!='quantized' or candidate['spec'].get('config',{}).get('method')!='loftq':continue
        row=byspec[json.dumps(candidate['spec'],sort_keys=True)]
        comparator=byid[candidate['comparison_control_id']]
        other=byspec[json.dumps(comparator['spec'],sort_keys=True)]
        if row.get('failure_stage')=='initialization' or other.get('failure_stage')=='initialization':continue
        assert row['frozen_sha256']==other['frozen_sha256'],'Paired quantized treatment/control base mismatch'
        left=torch.load(destination/(row['candidate_id']+'_initial.pt'),map_location='cpu',weights_only=True)['probe']
        right=torch.load(destination/(other['candidate_id']+'_initial.pt'),map_location='cpu',weights_only=True)['probe']
        row.setdefault('comparison_controls',{})[candidate['comparison_control_id']]={
            'same_frozen_base':True,'maximum_absolute_error':float((left-right).abs().max()),
            'mse':float((left-right).square().mean()),'relative_l2':float((left-right).norm()/right.norm())}
    value={'status':'pass','task':args.task,'manifest_sha256':sha(args.output/'manifest.json'),
           'controls':records,'unique_specs':len(ordered),
           'scope':'Accepted anchor: five real task updates on fixed160-query subset, exact current_dora initial/final tensors/RNG/loss/rate/clipping. Other specs: one update on32queries at fixedLR. Every successful initializer preserves downstream training RNG. Centered lane requires exact initial embedding parity; explicit warmstart/quantized lanes record initial perturbations and paired quantizer-base identity. Numerical control failures remain scoped to observed spec/LR.'}
    value['evidence_sha256']={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file() and p!=receipt}
    write(receipt,value)
    return value


def prepare(args, registry):
    global CALIBRATION
    gate=read(args.gate_evidence)
    assert gate['status']==gate['cpu_tests']['status']==gate['cuda_tests']['status']=='pass'
    for name,digest in gate['source_sha256'].items():assert sha(name)==digest,name
    assert gate.get('backend_source_sha256',gate.get('core_source_sha256'))==ab.SOURCE_SHA256
    reference=read(args.reference/'provenance.json');cfg=reference['configuration']
    for name in ('epochs','batch_size','eval_batch_size','max_length','weight_decay','temperature',
                 'warmup_fraction','max_grad_norm','threads'):setattr(args,name,cfg[name])
    args.rank,args.device,args.seeds=8,'cuda:0',[42,43,44]
    args.fixed_lr=.001 if args.task=='scifact' else .0001
    assert registry['rank']==8 and registry['seeds']==args.seeds and registry['seed']==42
    assert registry['fixed_lr']==args.fixed_lr and registry['learning_rates']==cfg['learning_rates']
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('highest')
    if args.task=='scifact':
        args.validation_fraction,args.split_seed=cfg['validation_fraction'],cfg['split_seed']
        corpus,queries,relevance,splits,metadata=task.load_data(args)
        assert metadata==reference['dataset'] and splits==read(args.reference/'splits.json')
        data={'scifact':{'corpus':corpus,'queries':queries,'relevance':relevance,'splits':splits,'metadata':metadata}}
        negatives=None
    else:
        args.hard_negatives=cfg['hard_negatives'];data=harder.load_data()
        for name,ds in data.items():
            assert ds['metadata']==reference['datasets'][name]
            assert ds['splits']==read(args.reference/f'{name}_splits.json')
        negatives=read(args.reference/'hard_negatives.json')
        assert set(negatives)==set(data['nfcorpus']['splits']['train'])
        for qid,docs in negatives.items():
            assert len(docs)==args.hard_negatives and not set(docs)&set(data['nfcorpus']['relevance'][qid])
        negative_path=args.output/'hard_negatives.json'
        if negative_path.exists():assert sha(negative_path)==sha(args.reference/'hard_negatives.json')
        else:shutil.copy2(args.reference/'hard_negatives.json',negative_path)
    tokenizer=AutoTokenizer.from_pretrained(task.MODEL,revision=task.MODEL_REVISION)
    for name,ds in data.items():
        ds['corpus_tokens']=task.tokenize_all(tokenizer,ds['corpus'],args.max_length)
        ds['query_tokens']=task.tokenize_all(tokenizer,ds['queries'],args.max_length)
        write_once(args.output/f'{name}_splits.json',ds['splits'])
        write_once(args.output/f'{name}_qrels.json',{qid:ds['relevance'][qid] for ids in ds['splits'].values() for qid in ids})
    CALIBRATION=prepare_calibration(tokenizer,data,negatives,args,registry)
    source_names=['experiments/init_retrieval.py','experiments/init_common.py','experiments/init_ablations.py',
        'experiments/nora_current.py','experiments/next_ablations.py','experiments/change_ablations.py',
        'experiments/retrieval.py','experiments/second_round/retrieval.py','experiments/adapters.py',
        'experiments/second_round_adapters.py','experiments/original_dora.py','dora.py','dora_row.py']
    # The common/data-aware modules are deliberately source-bound once available.
    for optional in ('experiments/init_data.py','experiments/init_calibration.py','experiments/init_data_adapter.py','experiments/init_loftq.py','experiments/init_warmstart.py'):
        if Path(optional).exists():source_names.append(optional)
    source_names=list(dict.fromkeys([*source_names,*gate['source_sha256']]))
    checkpoint=hf_hub_download(task.MODEL,'model.safetensors',revision=task.MODEL_REVISION)
    manifest={'schema':'dora-init-retrieval-v1','task':args.task,'rank':8,'study_lane':registry.get('study_lane','centered'),
        'registry':registry,'registry_sha256':sha(args.registry),'protocol_sha256':sha(args.protocol),
        'recipes':{c['candidate_id']:recipe_of(c) for c in registry['candidates']},
        'model':{'repository':task.MODEL,'revision':task.MODEL_REVISION,'checkpoint_sha256':sha(checkpoint),'frozen_dtype':'bfloat16'},
        'configuration':{name:getattr(args,name) for name in ('epochs','batch_size','eval_batch_size','max_length','weight_decay','temperature','warmup_fraction','max_grad_norm','seeds','fixed_lr')},
        'datasets':{name:ds['metadata'] for name,ds in data.items()},
        'input_sha256':{p.name:sha(p) for p in args.output.glob('*.json') if p.name.endswith(('_splits.json','_qrels.json'))},
        'hard_negatives_sha256':sha(args.output/'hard_negatives.json') if negatives else None,
        'calibration_manifest_sha256':sha(args.output/'calibration/manifest.json') if CALIBRATION is not None else None,
        'gate_evidence_sha256':sha(args.gate_evidence),'source_sha256':{name:sha(name) for name in source_names},
        'reference_provenance_sha256':sha(args.reference/'provenance.json'),
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                       'matmul_precision':torch.get_float32_matmul_precision()},
        'protocol':'All candidate seed42 full-budget validation pilots finish before immutable per-family selections. Fixed lane only selects candidates at fixedLR; tuned lane jointly selects initializer and LR. Test evaluation only in separately invoked finals phase after verifying every pilot/selection artifact. Actual identical candidate/seed/rate/budget fits are shared, never extra replicates. Initializer factor seed equals training seed, with isolated recipe-local RNG and original downstream training RNG.'}
    path=args.output/'manifest.json'
    if path.exists():assert read(path)==manifest,'Source/configuration/gate/registry changed; use a fresh output'
    else:write(path,manifest)
    for name in source_names:
        destination=args.output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert sha(destination)==sha(name)
        else:shutil.copy2(name,destination)
    for source,name in ((args.gate_evidence,'gate.json'),(args.registry,'registry.json'),(args.protocol,'protocol.md'),
                        (args.reference/'provenance.json','reference_provenance.json')):
        destination=args.output/'reference'/name;destination.parent.mkdir(exist_ok=True)
        if destination.exists():assert sha(destination)==sha(source)
        else:shutil.copy2(source,destination)
    return tokenizer,data,negatives,manifest


def main():
    from experiments import init_common
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=('scifact','nfcorpus'))
    p.add_argument('--registry',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--phase',required=True,choices=('controls','pilots','select','finals','finalize'))
    p.add_argument('--candidate-ids',nargs='+');p.add_argument('--family-ids',nargs='+')
    p.add_argument('--gate-evidence',type=Path,default=Path('/var/tmp/dora-bench/init_ablation_validation.json'))
    args=p.parse_args()
    if args.reference is None:args.reference=Path('/var/tmp/dora-bench/original_bench')/args.task
    registry=init_common.load_registry(args.registry,args.protocol,args.task)
    if args.phase in ('select','finalize'):
        manifest=read(args.output/'manifest.json')
        assert manifest['registry']==registry and manifest['registry_sha256']==sha(args.registry)
        for name,digest in manifest['source_sha256'].items():assert sha(name)==digest,name
        assert manifest['gate_evidence_sha256']==sha(args.gate_evidence)
        args.fixed_lr=manifest['configuration']['fixed_lr'];args.seeds=registry['seeds']
        tokenizer=data=negatives=None
        control=read(args.output/'controls/control.json');assert control['status']=='pass'
    else:
        tokenizer,data,negatives,manifest=prepare(args,registry)
        control=controls(tokenizer,data,negatives,args,registry)
    if args.phase=='controls':return
    manifest_sha=sha(args.output/'manifest.json')
    byid={c['candidate_id']:c for c in registry['candidates']}

    def fit(candidate,seed,final=False):
        method=candidate['candidate_id'];rate=candidate['learning_rate'];recipe=manifest['recipes'][method]
        fit_id=f'fits/{method}/seed_{seed}';directory=args.output/fit_id;directory.mkdir(parents=True,exist_ok=True)
        path=directory/'result.json';row=read(path) if path.exists() else None
        if row is not None:
            assert row['manifest_sha256']==manifest_sha and row['recipe_sha256']==recipe['recipe_sha256']
            if seed==42 and not final and not (directory/'pilot.json').exists():write(directory/'pilot.json',row)
            if row['status']=='numerical_failure':return row
            assert sha(directory/'adapter.pt')==row['checkpoint_sha256']
            if not final or 'test' in row['evaluation'][args.task]:
                return row
        try:
            model=new_model(candidate,seed)
        except (RuntimeError,ValueError) as error:
            if not initialization_failure(error):raise
            row={'status':'numerical_failure','failure_stage':'initialization','candidate_id':method,'method':method,
                'family_id':candidate['family_id'],'spec':candidate['spec'],'candidate_sha256':candidate_digest(candidate),
                'seed':seed,'rank':8,'learning_rate':rate,'fit_id':fit_id,'manifest_sha256':manifest_sha,
                'recipe_sha256':recipe['recipe_sha256'],'completed_steps':0,'evaluation':{},
                'failure':{'type':type(error).__name__,'message':str(error)},
                'finished_utc':datetime.now(timezone.utc).isoformat()}
            write(path,row)
            if seed==42 and not final:write_once(directory/'pilot.json',row)
            torch.cuda.empty_cache();return row
        if row is None:
            frozen=tensor_digest((n,t) for n,t in model.named_parameters() if not t.requires_grad)
            row={'status':'pass','candidate_id':method,'method':method,'family_id':candidate['family_id'],'spec':candidate['spec'],
                'candidate_sha256':candidate_digest(candidate),'seed':seed,'rank':8,'learning_rate':rate,'fit_id':fit_id,
                'manifest_sha256':manifest_sha,'recipe_sha256':recipe['recipe_sha256'],
                'trainable_parameters':sum(t.numel() for t in model.parameters() if t.requires_grad),
                'trainable_parameter_order':[n for n,t in model.named_parameters() if t.requires_grad],
                'initial_trainable_sha256':tensor_digest((n,t) for n,t in model.named_parameters() if t.requires_grad),
                'frozen_sha256':frozen,'started_utc':datetime.now(timezone.utc).isoformat()}
            row['initial_diagnostics']=diagnostics_of(model,candidate,seed)
            if registry.get('study_lane','centered')!='centered':
                row['initial_evaluation']=evaluate(model,tokenizer,data,args,directory/'initial',final=False)
            if candidate['spec'].get('kind')=='quantized':
                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                rng_python=random.getstate();import numpy as np
                rng_numpy=np.random.get_state()
                recreated=new_model(candidate,seed)
                assert frozen==tensor_digest((n,t) for n,t in recreated.named_parameters() if not t.requires_grad)
                assert row['initial_trainable_sha256']==tensor_digest((n,t) for n,t in recreated.named_parameters() if t.requires_grad)
                row['deterministic_recreation_verified']=True
                del recreated;torch.cuda.empty_cache()
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);random.setstate(rng_python);np.random.set_state(rng_numpy)
            row['environment']={'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                                'gpu':torch.cuda.get_device_name(),'command':[sys.executable,*sys.argv]}
            try:
                row['training']=train_fit(model,tokenizer,data,negatives,args,candidate,seed,rate,directory)
                if not all(torch.isfinite(t).all() for t in model.parameters()):raise RuntimeError('Nonfinite parameter after fit')
                assert frozen==tensor_digest((n,t) for n,t in model.named_parameters() if not t.requires_grad)
                row['frozen_weights_unchanged']=True;row['checkpoint_sha256']=sha(directory/'adapter.pt')
                row['evaluation']=evaluate(model,tokenizer,data,args,directory,final=final)
                row['diagnostics']=diagnostics_of(model,candidate,seed)
            except RuntimeError as error:
                if not numerical_failure(error):raise
                row.update(status='numerical_failure',failure={'type':type(error).__name__,'message':str(error)},
                           completed_steps=len((directory/'training.jsonl').read_text().splitlines()),evaluation={})
        else:
            assert row['frozen_sha256']==tensor_digest((n,t) for n,t in model.named_parameters() if not t.requires_grad)
            recreated=diagnostics_of(model,candidate,seed)
            for name,record in row['initial_diagnostics'].items():
                initial=record['initializer']
                for key in ('initial_up_sha256','initial_down_sha256','up_sha256','down_sha256','pretrained_weight_sha256','quantized_weight_sha256','magnitude_policy'):
                    if key in initial:assert recreated[name]['initializer'][key]==initial[key]
            state=torch.load(directory/'adapter.pt',map_location='cuda:0',weights_only=True)
            assert set(state)=={n for n,t in model.named_parameters() if t.requires_grad}
            model.load_state_dict(state,strict=False)
            row['evaluation']=evaluate(model,tokenizer,data,args,directory,final=True)
            row['post_reload_diagnostics']=diagnostics_of(model,candidate,seed)
        row['finished_utc']=datetime.now(timezone.utc).isoformat();write(path,row)
        if seed==42 and not final:
            pilot=directory/'pilot.json'
            if pilot.exists():assert read(pilot)==row
            else:write(pilot,row)
        del model;torch.cuda.empty_cache()
        print(json.dumps({'event':'fit','candidate_id':method,'seed':seed,'lr':rate,'final':final,'status':row['status'],'evaluation':row['evaluation']}),flush=True)
        return row

    if args.phase=='pilots':
        assert not (args.output/'results.json').exists(),'Pilot phase cannot run after finals started'
        assert not list((args.output/'family_results').glob('*.json')),'Pilot phase cannot run after finals started'
        assert not (args.output/'selections.json').exists(),'Pilot phase cannot run after selections frozen'
        selected_candidates=set(args.candidate_ids or byid)
        assert selected_candidates<=set(byid)
        for candidate in registry['candidates']:
            if candidate['candidate_id'] in selected_candidates:fit(candidate,42)
        return
    pilots=[]
    for candidate in registry['candidates']:
        directory=args.output/f"fits/{candidate['candidate_id']}/seed_42"
        path=directory/'pilot.json';row=read(path)
        assert row['candidate_sha256']==candidate_digest(candidate) and row['manifest_sha256']==manifest_sha
        assert row['seed']==42
        pilot={'candidate_id':candidate['candidate_id'],'family_id':candidate['family_id'],
               'status':row['status'],'learning_rate':candidate['learning_rate'],
               'fit_id':row['fit_id'],'pilot_sha256':sha(path)}
        if row['status']=='pass':
            assert 'test' not in row['evaluation'][args.task]
            assert row['checkpoint_sha256']==sha(directory/'adapter.pt')
            pilot.update(selection_value=row['evaluation'][args.task]['validation']['ndcg_at_10'],
                         checkpoint_sha256=row['checkpoint_sha256'])
        else:
            assert row['status']=='numerical_failure'
            pilot.update(selection_value=None,failure=row['failure'])
        pilots.append(pilot)
    selected=init_common.select_candidates(registry,pilots,minimize=False)
    selected.update(manifest_sha256=manifest_sha,registry_file_sha256=sha(args.registry),
                    selection_metric=f'{args.task}.validation.ndcg_at_10',all_pilots_complete_before_test=True)
    selections=selected['selections'];families=sorted(selections)
    selection_path=args.output/'selections.json'
    if selection_path.exists():assert read(selection_path)==selected,'Frozen selection changed'
    else:
        assert args.phase=='select','Finals requires prior select phase and frozen selections'
        write(selection_path,selected)
    if args.phase=='select':
        write(args.output/'pilots_complete.json',{'status':'pass','manifest_sha256':manifest_sha,
            'selections_sha256':sha(selection_path),'pilots':len(pilots),'families':len(families)})
        return
    pilot_gate=read(args.output/'pilots_complete.json')
    assert pilot_gate['status']=='pass' and pilot_gate['selections_sha256']==sha(selection_path)
    chosen_families=set(args.family_ids or families)
    assert chosen_families<=set(families)
    for family in families if args.phase=='finals' else []:
        if family not in chosen_families:continue
        result_path=args.output/'family_results'/f'{family}.json'
        results=read(result_path) if result_path.exists() else {'schema':'dora-init-retrieval-v1','task':args.task,'rank':8,
            'family_id':family,'manifest_sha256':manifest_sha,'selections_sha256':sha(selection_path),'runs':[],'paired_control_fits':[],'complete':False}
        assert results['manifest_sha256']==manifest_sha and results['selections_sha256']==sha(selection_path)
        for seed in args.seeds:
            for method in sorted({value for value in selections[family].values() if value is not None}):
                row=fit(byid[method],seed,final=True)
                candidate=byid[method]
                if candidate['spec'].get('kind')=='quantized' and candidate['spec'].get('config',{}).get('method')=='loftq':
                    comparator=byid[candidate['comparison_control_id']];assert comparator['learning_rate']==candidate['learning_rate']
                    paired=fit(comparator,seed,final=True)
                    pair={**paired,'for_candidate_id':method,'for_family_id':family}
                    results.setdefault('paired_control_fits',[])
                    results['paired_control_fits']=[p for p in results['paired_control_fits'] if (p['for_candidate_id'],p['seed'])!=(method,seed)]
                    results['paired_control_fits'].append(pair)
                lanes=[lane for lane in ('fixed_original_lr','validation_tuned') if selections[family][lane]==method]
                for lane in lanes:
                    results['runs']=[v for v in results['runs'] if (v['family_id'],v['seed'],v['lane'])!=(family,seed,lane)]
                    results['runs'].append({**row,'lane':lane,'shared_lanes':lanes,'reused_full_budget_validation_fit':seed==42})
                write(result_path,results)
        results['complete']=True;write(result_path,results)
    if args.phase=='finals' and args.family_ids:return
    results={'schema':'dora-init-retrieval-v1','task':args.task,'rank':8,'manifest_sha256':manifest_sha,
             'selections_sha256':sha(selection_path),'runs':[],'paired_control_fits':[],'complete':False}
    for family in families:
        value=read(args.output/'family_results'/f'{family}.json')
        assert value['complete'] and value['family_id']==family and value['selections_sha256']==sha(selection_path)
        results['runs'].extend(value['runs']);results['paired_control_fits'].extend(value.get('paired_control_fits',[]))
    result_path=args.output/'results.json'
    assert len(results['runs'])==sum(value is not None for choices in selections.values() for value in choices.values())*3
    results['unavailable_lanes']=[{'family_id':family,'lane':lane,'reason':'all eligible pilots failed numerically'} for family,choices in selections.items() for lane,value in choices.items() if value is None]
    results['complete']=True;results['unique_final_fits']=len({r['fit_id'] for r in results['runs']+results['paired_control_fits']})
    results['unique_trained_fits']=len(list((args.output/'fits').glob('*/*/result.json')))
    results['successful_checkpoint_fits']=len(list((args.output/'fits').glob('*/*/adapter.pt')))
    write(result_path,results)
    print(json.dumps({'event':'complete','task':args.task,'families':len(families),'unique_trained_fits':results['unique_trained_fits']}),flush=True)


if __name__=='__main__':main()
