#!/usr/bin/env python3
"""Audited native runner. Pipelines and DDP are separate explicit stages.

This executable refuses to silently approximate an unsupported registry cell.
Large sweeps remain gated until smoke, resume and reference checks pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import sdpa_kernel, SDPBackend

from rankstudy.v2.data import collate, microbatches, read_jsonl
from rankstudy.v3.adapters import adapted, inject, compact
from rankstudy.v3.allocation import Candidate, TokenSchedule, allocate, uniform_allocate
from rankstudy.v3.checkpoint import backbone_hash, json_write, save_checkpoint, load_checkpoint
from rankstudy.v3.reference import reference_allocator, reference_after_step, reference_penalty
from rankstudy.v3.scoring import SlotEMA, token_subsample


def numerator(logits, labels):
    pred, labels = logits[:, :-1].float(), labels[:, 1:]
    count = labels.ne(-100).sum()
    # Flatten token rows: nightly's NLLLoss2d CUDA reduction rejects strict
    # determinism; the ordinary 2D classifier reduction is deterministic.
    loss = torch.nn.functional.cross_entropy(pred.reshape(-1,pred.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')
    return loss, count


@torch.no_grad()
def evaluate(model, rows, pad, device):
    model.eval()
    for m in adapted(model).values():
        m.begin_update(tracking=False)
    records=[]
    for i in range(0,len(rows),2):
        batch=collate(rows[i:i+2],pad,device)
        logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
        for j,row in enumerate(rows[i:i+2]):
            loss, count=numerator(logits[j:j+1],batch['labels'][j:j+1])
            records.append({'group_id':row.get('group_id'), 'loss_sum':float(loss),'supervised_tokens':int(count),'nll':float(loss/count)})
    model.train()
    return {'group_balanced_nll':sum(r['nll'] for r in records)/len(records),
            'token_weighted_nll':sum(r['loss_sum'] for r in records)/sum(r['supervised_tokens'] for r in records),
            'groups':records}


def counts(model):
    mods=adapted(model)
    return {'active_slots':sum(int(m.mask.sum()) for m in mods.values()),
            'active_trainable_scalars':sum(int(m.mask.sum())*m.slot_cost for m in mods.values()),
            'allocated_trainable_scalars':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen_adapter_bytes':sum(b.numel()*b.element_size() for m in mods.values() for b in m.buffers(recurse=False)),
            'rank_map':{n:int(m.mask.sum()) for n,m in mods.items()}}


def lr_at(step,total,peak):
    warmup=max(1,round(.03*total))
    if step < warmup:
        return peak*(step+1)/warmup
    progress=(step-warmup)/max(1,total-warmup-1)
    return peak*(.1+.9*.5*(1+math.cos(math.pi*min(1,progress))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B')
    ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'))
    ap.add_argument('--manifest-root',type=Path,default=Path('artifacts/manifests_v2'))
    ap.add_argument('--seed',type=int,default=1001)
    ap.add_argument('--updates',type=int,default=128)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--scale',type=float,default=1.)
    ap.add_argument('--penalty',type=float,default=0.)
    ap.add_argument('--target-rank',type=int,default=8)
    ap.add_argument('--max-rank',type=int,default=16)
    ap.add_argument('--targets',nargs='+',default=['q_proj','o_proj'])
    ap.add_argument('--microbatch',type=int,default=2)
    ap.add_argument('--backend',choices=['CUDNN_ATTENTION','FLASH_ATTENTION','EFFICIENT_ATTENTION','MATH'],default='EFFICIENT_ATTENTION')
    ap.add_argument('--checkpoint-every',type=int,default=16)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--stop-after',type=int)
    ap.add_argument('--dev-limit',type=int,default=0,help='Smoke only; never select from partial dev')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--profile',action='store_true')
    args=ap.parse_args()
    if args.dev_limit and not args.smoke:
        raise ValueError('partial development evaluation allowed only in labeled smoke tests')
    if torch.__version__ != '2.16.0.dev20261013+cu130' or not torch.cuda.is_available():
        raise RuntimeError('pinned CUDA13 nightly and real CUDA execution required')
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
    specs={m['id']:m for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']}
    spec=specs[args.method]
    if spec['group'] in {'restart','restart_control','compress'}:
        raise NotImplementedError('use the audited multi-stage runner; never substitute native training')
    adalora=spec['adalora']; custom_online=spec['allocator']!='none' and not adalora
    triplet=spec['triplet']
    if (spec['group']=='triplet_ortho' or adalora) and args.penalty<=0:
        raise ValueError('positive declared penalty required')
    if args.out.exists() and (args.out/'metadata.json').exists() and args.resume is None:
        raise FileExistsError('immutable run already exists; use another run ID')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest_path=args.manifest_root/f'train_seed{args.seed}.json'
    manifest=json.loads(manifest_path.read_text())
    updates=manifest['updates'][:args.updates]
    if len(updates)!=args.updates:
        raise ValueError('insufficient sealed manifest updates')
    train=read_jsonl(args.data_root/'train.jsonl')
    dev=read_jsonl(args.data_root/'dev_tune.jsonl')
    if args.dev_limit:
        dev=dev[:args.dev_limit]
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    files=[Path(__file__),*Path('src/rankstudy/v3').rglob('*.py'),Path('src/rankstudy/v2/data.py'),Path('configs/methods.yaml')]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    invariant_keys=['method','model','data_root','seed','updates','lr','scale','penalty','target_rank','max_rank','targets','microbatch','backend']
    resume_invariants={k:config[k] for k in invariant_keys}
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    meta={'status':'running','protocol':'adaptive-rank-v3','method':args.method,'config':config,
          'config_sha256':digest,'source_revision_at_start':revision,'source_hashes':source_hashes,
          'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'torch':torch.__version__,'device':torch.cuda.get_device_name(),'backend_requested':args.backend,
          'topology':'single_gpu','smoke':args.smoke,'seed':args.seed,
          'deterministic_algorithms':True,'sdpa_exception_receipt':'artifacts/sdpa_v3_oracle.json',
          'progress_units':'optimizer_updates_upstream_reference' if adalora else 'input_tokens',
          'act_sampling':'up to 256 valid tokens per effective batch; shared deterministic seed/ordinal sample'}
    json_write(args.out/'config.json',config); json_write(args.out/'metadata.json',meta)
    setup_start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
    model.config.use_cache=False
    target_shapes={n:(m.out_features,m.in_features) for n,m in model.named_modules()
                   if isinstance(m,torch.nn.Linear) and n.split('.')[-1] in args.targets}
    target_cost=sum(args.target_rank*(d+n) for d,n in target_shapes.values())
    rank=args.target_rank if spec['allocator']=='none' and spec['group']!='wide' else args.max_rank
    pattern={name:rank for name in target_shapes}
    if triplet and not custom_online:
        cells=[Candidate(name,i,d+n+1,0.) for name,(d,n) in target_shapes.items() for i in range(rank)]
        keep,_=uniform_allocate(cells,target_cost)
        pattern={name:sum(n==name for n,_ in keep) for name in target_shapes}
    if args.resume:
        desc=json.loads((args.resume/'manifest.json').read_text())
        pattern={name:m['rank'] for name,m in desc['modules'].items()}
    # For reference s is the nominal multiplier; record alpha and offset exactly.
    alpha=args.scale*args.max_rank if adalora else args.scale
    inject(model,rank,spec['init'],alpha,args.seed,spec['score'],tuple(args.targets),pattern)
    mods=adapted(model)
    initial_cost=sum(args.max_rank*m.slot_cost for m in mods.values())
    schedule=TokenSchedule([u['input_tokens'] for u in updates],initial_cost,target_cost)
    original_hash=backbone_hash(model)
    meta.update({'target_shapes':target_shapes,'P_target':target_cost,'initial_counts':counts(model),
                 'original_backbone_sha256':original_hash,'actual_multiplier':float(next(iter(mods.values())).multiplier),
                 'reference_alpha':alpha if adalora else None})
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)
    ema=SlotEMA(uncertainty='UNCERTAINTY' in args.method)
    ref=reference_allocator(model,args.max_rank,args.target_rank,args.updates) if adalora else None
    start_step,tokens,supervised,charged,compacted=0,0,0,0.,False
    events=[]
    if args.resume:
        extra=load_checkpoint(args.resume,model,opt,'cuda')
        if extra['manifest_sha256']!=meta['manifest_sha256'] or extra['backbone_sha256']!=original_hash:
            raise ValueError('resume provenance mismatch')
        if extra['method']!=args.method or extra['updates']!=args.updates:
            raise ValueError('resume method/horizon mismatch')
        if extra.get('resume_invariants')!=resume_invariants:
            raise ValueError('resume hyperparameter/backend mismatch or unaudited older checkpoint')
        start_step,tokens,supervised,charged,compacted=(extra[k] for k in ['step','input_tokens','supervised_tokens','charged_s','compacted'])
        events=extra['pruning_events']
        accounting=args.resume/'accounting.json'
        if accounting.exists():
            charged+=json.loads(accounting.read_text())['checkpoint_write_s']
        ema.load_state_dict(extra['ema'])
        if ref is not None:
            for key,value in extra['reference_state'].items():
                setattr(ref,key,value)
    meta['setup_s']=time.perf_counter()-setup_start
    json_write(args.out/'metadata.json',meta)
    backend=getattr(SDPBackend,args.backend)
    completed=start_step
    with sdpa_kernel(backend):
        if args.smoke and not args.resume:
            before=evaluate(model,dev,tokenizer.pad_token_id,'cuda')
            json_write(args.out/'smoke_before.json',before)
        for step in range(start_step,args.updates):
            if args.stop_after is not None and step>=args.stop_after:
                break
            torch.cuda.synchronize(); begin=time.perf_counter()
            update=updates[step]
            mods=adapted(model)
            tracking=custom_online and not compacted
            for m in mods.values(): m.begin_update(tracking and spec['allocator']!='global_random')
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups: group['lr']=lr_at(step,args.updates,args.lr)
            batches=[collate([train[i] for i in ix],tokenizer.pad_token_id)
                     for ix in microbatches(update['row_indices'],train,args.microbatch)]
            samples=token_subsample([b['attention_mask'] for b in batches],args.seed,step+1) if tracking and spec['score']=='ACT' else [None]*len(batches)
            loss_sum=0.; nt=0; ns=0
            profiler=(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                      if args.profile and step==start_step else None)
            if profiler: profiler.__enter__()
            for batch,indices in zip(batches,samples):
                batch={k:v.to('cuda') for k,v in batch.items()}
                for m in mods.values(): m._sample_indices=indices
                logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False).logits
                part,count=numerator(logits,batch['labels'])
                part.backward()
                loss_sum+=float(part.detach()); ns+=int(count); nt+=int(batch['attention_mask'].sum())
            if profiler:
                profiler.__exit__(None,None,None)
                kernels=sorted({event.name for event in profiler.events() if 'attention' in event.name.lower() or 'cudnn' in event.name.lower()})
                json_write(args.out/'attention_profile.json',{'requested':args.backend,'events':kernels})
                if args.backend=='CUDNN_ATTENTION' and not any('cudnn_attention' in key.lower() for key in kernels):
                    raise RuntimeError('requested cuDNN attention was not observed')
                if args.backend=='EFFICIENT_ATTENTION' and not any('efficient_attention' in key.lower() for key in kernels):
                    raise RuntimeError('requested efficient attention was not observed')
            if nt!=update['input_tokens'] or ns!=update['supervised_tokens']:
                raise ValueError('executed exposure differs from sealed manifest')
            params=[p for p in model.parameters() if p.requires_grad]
            for p in params:
                if p.grad is not None: p.grad.div_(ns)
            task_grad_norm=float(torch.stack([p.grad.float().square().sum() for p in params if p.grad is not None]).sum().sqrt())
            budget,event,freeze=schedule.at(step+1,tokens+nt)
            if tracking:
                if spec['allocator']=='global_random':
                    scores={name:torch.tensor([random.Random(f'{args.seed}/{name}/{int(i)}').random() for i in m.slot_ids],device='cuda') for name,m in mods.items()}
                else:
                    scores={name:m.task_score(ns) for name,m in mods.items()}
                ema.update(scores)
                if event:
                    cells=[Candidate(name,int(sid),m.slot_cost,float(q))
                           for name,m in mods.items() for sid,q,active in zip(m.slot_ids,ema.value(name),m.mask) if active]
                    allocator=uniform_allocate if spec['allocator']=='uniform_gate' else allocate
                    keep,receipt=allocator(cells,budget,floor=1)
                    for name,m in mods.items():
                        m.set_mask(torch.tensor([(name,int(i)) in keep for i in m.slot_ids],device='cuda'),opt)
                    events.append({'step':step+1,'input_tokens':tokens+nt,**receipt,'rank_map':counts(model)['rank_map']})
            reg=torch.zeros((),device='cuda')
            if triplet and args.penalty:
                reg=args.penalty*torch.stack([m.penalty() for m in mods.values()]).mean()
            elif adalora:
                reg=reference_penalty(model,args.penalty)
            if reg.requires_grad: reg.backward()
            for m in mods.values(): m.freeze_pruned(opt)
            grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
            opt.step()
            if adalora: reference_after_step(model,ref,step+1)
            if custom_online and freeze:
                opt=compact(model,opt); compacted=True
                assert counts(model)['active_trainable_scalars']<=target_cost
            tokens+=nt; supervised+=ns
            completed=step+1
            torch.cuda.synchronize(); duration=time.perf_counter()-begin; charged+=duration
            row={'step':step+1,'input_tokens':nt,'supervised_tokens':ns,'input_tokens_total':tokens,
                 'task_loss':loss_sum/ns,'regularizer':float(reg.detach()),'task_gradient_norm':task_grad_norm,
                 'total_gradient_norm':grad_norm,'clipped':grad_norm>1.,'lr':lr_at(step,args.updates,args.lr),
                 'step_s':duration,'charged_s':charged,'score_updates':ema.updates,'compacted':compacted,**counts(model)}
            with (args.out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='rank_map'}),flush=True)
            if (step+1)%args.checkpoint_every==0 or freeze or step+1==args.updates or step+1==args.stop_after:
                ck_start=time.perf_counter()
                reference_state={key:getattr(ref,key) for key in ['ipt','exp_avg_ipt','exp_avg_unc','rank_pattern']} if ref else {}
                ck_path=args.out/f'checkpoint_{step+1:04d}'
                save_checkpoint(ck_path,model,opt,
                    {'step':step+1,'method':args.method,'updates':args.updates,'input_tokens':tokens,'supervised_tokens':supervised,
                     'charged_s':charged,'compacted':compacted,'ema':ema.state_dict(),'reference_state':reference_state,
                     'resume_invariants':resume_invariants,'pruning_events':events,
                     'manifest_sha256':meta['manifest_sha256'],'backbone_sha256':original_hash})
                ck_elapsed=time.perf_counter()-ck_start
                charged+=ck_elapsed
                json_write(ck_path/'accounting.json',{'checkpoint_write_s':ck_elapsed})
        eval_start=time.perf_counter()
        dev_metrics=evaluate(model,dev,tokenizer.pad_token_id,'cuda')
        eval_s=time.perf_counter()-eval_start
    final_hash=backbone_hash(model)
    if final_hash!=original_hash: raise RuntimeError('original frozen backbone changed')
    final=counts(model)
    finished=completed==args.updates
    if finished and spec['group']!='wide' and not adalora and final['active_trainable_scalars']>target_cost:
        raise RuntimeError('endpoint budget violated')
    meta.update({'status':'complete' if finished else 'interrupted_checkpoint','updates_finished':completed,
                 'input_tokens':tokens,'supervised_tokens':supervised,'charged_gpu_seconds':charged,
                 'evaluation_s':eval_s,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                 'final_counts':final,'dev_tune_endpoint_group_balanced_nll':dev_metrics['group_balanced_nll'],
                 'dev_tune_endpoint_token_weighted_nll':dev_metrics['token_weighted_nll'],
                 'backbone_unchanged':True,'budget_delta':final['active_trainable_scalars']-target_cost})
    json_write(args.out/'dev_tune.json',dev_metrics); json_write(args.out/'pruning.json',events)
    json_write(args.out/'metadata.json',meta)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if '--out' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--out')+1])/'metadata.json'
            if output.exists():
                failed=json.loads(output.read_text())
                if failed.get('status')=='running':
                    failed.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
                    json_write(output,failed)
        raise
