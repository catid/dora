#!/usr/bin/env python3
"""Run one actual study training job; all metrics are immutable JSONL."""
from __future__ import annotations
import argparse,json,time,random,hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rankstudy.adapters import inject,modules
from rankstudy.data import CompletionDataset,stream_batches
from rankstudy.allocator import Slot,allocate
from rankstudy.scores import raw_scores,displacement_scores,activation_scores,adalora_scores,ema_update,uncertainty_update
from rankstudy.registry import build_registry

def loss_sum_count(logits,labels):
    flat_logits=logits[...,:-1,:].float().reshape(-1,logits.shape[-1])
    flat_labels=labels[...,1:].reshape(-1)
    valid=flat_labels.ne(-100)
    return torch.nn.functional.cross_entropy(flat_logits,flat_labels,ignore_index=-100,reduction='sum'), valid.sum()

def loss_fn(logits,labels):
    """Global supervised-token mean for one effective batch."""
    total,n=loss_sum_count(logits,labels)
    return total / max(1, int(n))

def safe_stream_batches(ds, seed):
    """Length-aware microbatches with a fixed token-memory cap.

    The effective update still accumulates by token count. Splitting only long
    padded batches preserves example order/IDs while avoiding a rare pair of
    2048-token examples exceeding card memory.
    """
    for batch in stream_batches(ds,seed,batch_size=2):
        if batch['input_ids'].shape[0] == 2 and int(batch['input_ids'].shape[0]*batch['input_ids'].shape[1]) > 1800:
            for i in range(2):
                yield {k:v[i:i+1] for k,v in batch.items()}
        else:
            yield batch

@torch.no_grad()
def evaluate(model, ds, seed, device):
    model.eval(); total=0.; count=0
    for batch in stream_batches(ds,seed,batch_size=2):
        batch={k:v.to(device) for k,v in batch.items() if k!='group_id'}
        logits=model(**batch).logits; labels=batch['labels'][...,1:]; pred=logits[...,:-1,:]
        valid=labels.ne(-100); total += float(torch.nn.functional.cross_entropy(pred.float().reshape(-1,pred.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')); count += int(valid.sum())
    model.train(); return total/max(1,count)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',required=True); ap.add_argument('--init',choices=['L','I','T0','T1','ADALORA'],default=None); ap.add_argument('--scale',type=float,default=1.); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--seed',type=int,default=2001); ap.add_argument('--updates',type=int,default=8); ap.add_argument('--data-root',default='artifacts/data'); ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B'); ap.add_argument('--out',required=True); ap.add_argument('--rank',type=int,default=16); ap.add_argument('--target-rank',type=int,default=8); args=ap.parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.set_float32_matmul_precision('high'); device='cuda'
    reg={x.id:x for x in build_registry()}; spec=reg.get(args.method); init=args.init or (spec.init if spec else 'L'); adalora=(args.method=='ADALORA_REF' or init=='ADALORA'); triplet=init in ('T0','T1')
    tok=AutoTokenizer.from_pretrained(args.model); tok.padding_side='right'; tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    model=inject(model,args.rank,('q_proj','o_proj'),init,args.scale,args.method,args.seed,triplet,adalora=adalora)
    ds=CompletionDataset(Path(args.data_root)/'train.jsonl',tok,2048); dev=CompletionDataset(Path(args.data_root)/'dev_tune.jsonl',tok,2048); params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=0.)
    warmup=max(1,int(round(.03*args.updates)))
    def lr_factor(step):
        if step < warmup: return (step+1)/warmup
        q=(step-warmup)/max(1,args.updates-warmup-1)
        return .10+.90*.5*(1+torch.cos(torch.tensor(q*3.141592653589793))).item()
    for group in opt.param_groups: group['lr']=args.lr*lr_factor(0)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); config_bytes=(json.dumps(vars(args),sort_keys=True,indent=2)+'\n').encode(); (out/'config.json').write_bytes(config_bytes)
    # Single-example microbatches keep peak activation memory within the
    # measured 97.9-GiB cards even for 2048-token rows; effective updates still
    # accumulate to >=8192 tokens.
    metrics=[]; score_ema={}; unc={}; epoch=0; it=iter(safe_stream_batches(ds,args.seed+epoch)); model.train(); t0=time.time(); opt.zero_grad(set_to_none=True)
    for step in range(args.updates):
        # Reference AdaLoRA keeps dormant E coordinates zero during the
        # forward pass, but does not freeze their optimizer update before the
        # next allocation event.  This preserves its reactivation semantics.
        for m in modules(model):
            m.begin_score_accum()
            if m.adalora:
                with torch.no_grad(): m.E.data[~m.active_mask, 0] = 0
        input_tokens=0; supervised_tokens=0; task_loss_sum=0.0
        while input_tokens < 8192:
            try: raw=next(it)
            except StopIteration:
                epoch += 1; it=iter(safe_stream_batches(ds,args.seed+epoch)); raw=next(it)
            batch={k:v.to(device) for k,v in raw.items() if k!='group_id'}
            valid_x=batch.get('attention_mask').detach().bool() if 'attention_mask' in batch else None
            for m in modules(model): m._score_valid.append(valid_x.reshape(-1) if valid_x is not None else None)
            outp=model(**batch); part,n=loss_sum_count(outp.logits,batch['labels']); part.backward()
            input_tokens += int(batch['attention_mask'].sum()); supervised_tokens += int(n); task_loss_sum += float(part.detach())
        # Backward was on an unnormalized sum across microbatches. Normalize
        # the task gradient once by the complete effective update.
        denom=max(1,supervised_tokens)
        for p in params:
            if p.grad is not None: p.grad.div_(denom)
        for m in modules(model):
            if m._gate_grad_accum is not None: m._gate_grad_accum.div_(denom)
        pending_keep=None
        if spec and spec.allocator!='none' and step and (step%16==0 or step==args.updates-1):
            allslots=[]
            for mi,m in enumerate(modules(model)):
                if spec.allocator=='global_random':
                    q=torch.tensor([random.Random(args.seed*1000003+mi*1009+j).random() for j in range(m.rank)],device=device)
                elif spec.score=='ADALORA': q=adalora_scores(m.adapter)
                elif spec.score=='RAW': q=raw_scores(m.adapter)
                elif spec.score=='DISP': q=displacement_scores(m.adapter)
                elif spec.score=='ACT':
                    xs=torch.cat(m._score_x,dim=0)
                    vals=[v for v in m._score_valid if v is not None]
                    vv=torch.cat(vals,dim=0) if vals else None
                    if vv is not None:
                        valid_idx=torch.nonzero(vv,as_tuple=False).flatten()
                        if valid_idx.numel()>256:
                            g=torch.Generator(device='cpu').manual_seed(args.seed*1009+step*9176+mi)
                            valid_idx=valid_idx[torch.randperm(valid_idx.numel(),generator=g)[:256].to(valid_idx.device)]
                            xs=xs[valid_idx]; vv=None
                    q=activation_scores(m.adapter,xs,vv)
                else: q=m.score_gate()
                score_ema[mi]=ema_update(score_ema.get(mi),q,.85)
                if 'UNCERTAINTY' in args.method or m.adalora:
                    unc[mi]=uncertainty_update(unc.get(mi),q,score_ema[mi],.85); q=score_ema[mi]*unc[mi]
                slot_cost=m.A.shape[1]+m.B.shape[0]+(1 if (m.triplet or m.adalora) else 0)
                for j,v in enumerate(q.tolist()):
                    # Custom allocators are monotonic. Faithful AdaLoRA keeps
                    # its reference reactivation semantics and scores all E
                    # coordinates.
                    if (not m.adalora) and (not bool(m.active_mask[j])): continue
                    allslots.append(Slot(str(mi),j,slot_cost,float(v)))
            initial=sum(m.rank*(m.A.shape[1]+m.B.shape[0]+(1 if (m.triplet or m.adalora) else 0)) for m in modules(model))
            target=sum(args.target_rank*(m.A.shape[1]+m.B.shape[0]+(1 if (m.triplet or m.adalora) else 0)) for m in modules(model))
            frac=max(0,min(1,(step/args.updates-.1)/.7)); budget=round(target+(initial-target)*(1-frac)**3)
            floor={} if any(m.adalora for m in modules(model)) else {str(i):1 for i in range(len(modules(model)))}
            if spec.allocator=='uniform_gate':
                # Equal-cost q/o modules use a uniform rank pattern, while
                # retaining the highest gate scores within each module.
                costs=[m.A.shape[1]+m.B.shape[0]+(1 if (m.triplet or m.adalora) else 0) for m in modules(model)]
                if len(set(costs)) != 1:
                    keep=allocate(allslots,budget,floor)
                else:
                    total_slots=max(0,budget//costs[0]); base,extra=divmod(total_slots,len(modules(model))); keep=set()
                    for mi,m in enumerate(modules(model)):
                        want=min(m.rank,base+(mi<extra)); group=[s for s in allslots if s.module==str(mi)]
                        keep.update((s.module,s.index) for s in sorted(group,key=lambda s:(-s.score,s.index))[:want])
            else:
                keep=allocate(allslots,budget,floor)
            if any(m.adalora for m in modules(model)):
                pending_keep=keep
            else:
                for mi,m in enumerate(modules(model)): m.set_mask(torch.tensor([(str(mi),j) in keep for j in range(m.rank)],device=device))
        reg_loss=torch.zeros((),device=device)
        if spec and spec.triplet and spec.penalty:
            reg_loss=spec.penalty*sum(m.orth_penalty() for m in modules(model))/max(1,len(modules(model)))
        elif any(m.adalora for m in modules(model)):
            # Reference AdaLoRA's documented coefficient is 0.1 in the
            # original implementation; this is separate from normalized
            # custom triplet orthogonality ablations.
            reg_loss=0.1*sum(m.adalora_orth_penalty() for m in modules(model))/max(1,len(modules(model)))
        if reg_loss.requires_grad:
            reg_loss.backward()
        for m in modules(model):
            mask=m.active_mask.float()
            if not m.adalora:
                if m.A.grad is not None: m.A.grad.mul_(mask[:,None])
                if m.B.grad is not None: m.B.grad.mul_(mask[None,:])
            # Faithful AdaLoRA leaves A/B/E gradients live for dormant
            # coordinates so the reference allocator can reactivate them.
            if m.adalora:
                pass
            if not m.adalora and hasattr(m,'E') and m.E.grad is not None: m.E.grad.mul_(mask[:,None])
            if hasattr(m,'lam') and m.lam.grad is not None: m.lam.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if pending_keep is not None:
            for mi,m in enumerate(modules(model)): m.set_mask(torch.tensor([(str(mi),j) in pending_keep for j in range(m.rank)],device=device))
        for group in opt.param_groups: group['lr']=args.lr*lr_factor(step+1)
        # Adam moments for deleted coordinates must not reintroduce them.
        for m in modules(model):
            mask=m.active_mask.float()
            state_views = ([] if m.adalora else [(m.A,mask[:,None]),(m.B,mask[None,:])] + ([(m.lam,mask)] if hasattr(m,'lam') else []))
            for p,view in state_views:
                state=opt.state.get(p,{})
                for key in ('exp_avg','exp_avg_sq'):
                    if key in state: state[key].mul_(view)
        opt.zero_grad(set_to_none=True)
        metrics.append({'step':step+1,'loss':task_loss_sum/denom+float(reg_loss.detach()),'task_loss':task_loss_sum/denom,'lr':opt.param_groups[0]['lr'],'supervised_tokens':supervised_tokens,'input_tokens':input_tokens,'active_slots':sum(int(m.active_mask.sum()) for m in modules(model)),'elapsed_s':time.time()-t0})
        print(json.dumps(metrics[-1]),flush=True)
    dev_loss=evaluate(model,dev,args.seed+991,device)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    metadata={'method':args.method,'init':init,'lr':args.lr,'scale':args.scale,'rank':args.rank,'target_rank':args.target_rank,'seed':args.seed,'updates':args.updates,'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'status':'complete','allocator_update_order':'pre_optimizer_shared_runner; faithful AdaLoRA reference ordering differs and is recorded as a compatibility exception','final_active_slots':metrics[-1]['active_slots'],'input_tokens':sum(x['input_tokens'] for x in metrics),'supervised_tokens':sum(x['supervised_tokens'] for x in metrics),'elapsed_s':metrics[-1]['elapsed_s'],'peak_memory_bytes':torch.cuda.max_memory_allocated(),'dev_tune_endpoint_nll':dev_loss}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__': main()
