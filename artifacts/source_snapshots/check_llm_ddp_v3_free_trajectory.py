#!/usr/bin/env python3
"""Real Qwen BF16 single-process/DDP parity through physical compaction.

Invoke using torchrun on two idle GPUs. Unequal local microbatch/token counts,
no_sync accumulation, signed score reduction, rank-zero allocation and a fresh
DDP wrapper/optimizer mapping after compaction are all exercised.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoModelForCausalLM,AutoTokenizer
from rankstudy.v2.data import read_jsonl,collate
from rankstudy.v3.adapters import adapted,inject,compact
from rankstudy.v3.allocation import Candidate,allocate
from rankstudy.v3.checkpoint import json_write,backbone_hash
from rankstudy.v3.distributed import normalize_task_gradients,broadcast_masks
from scripts.run_one_v3 import numerator


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B')
    ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'))
    a=ap.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.set_num_threads(1)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    train=read_jsonl(a.data_root/'train.jsonl')
    probe=json.loads((a.data_root/'probe_source_indices.json').read_text())
    lookup={r['source_index']:r for r in train};rows=[lookup[i] for i in probe[:16]]
    records=[]
    for init,score in [('I','GATE'),('L','ACT'),('T0','GATE'),('T1','GATE')]:
        torch.manual_seed(882)
        model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
        model.config.use_cache=False;inject(model,4,init,1.,33,score)
        original=backbone_hash(model);oracle=copy.deepcopy(model)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.,fused=True)
        refopt=torch.optim.AdamW([p for p in oracle.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.,fused=True)
        ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],broadcast_buffers=False)
        for step in range(4):
            batch_rows=rows[4*step:4*step+4]
            opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
            for m in adapted(model).values():m.begin_update(True)
            for m in adapted(oracle).values():m.begin_update(True)
            local_count=0
            local=[0] if rank==0 else [1,2,3]
            # Same deterministic ACT sample: first <=64 tokens from each of
            # four examples, totaling <=256 across the effective global batch.
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                for j,index in enumerate(local):
                    b=collate([batch_rows[index]],tok.eos_token_id,'cuda')
                    for m in adapted(model).values():
                        m._sample_indices=torch.arange(min(64,int(b['attention_mask'].sum()))) if score=='ACT' else None
                    with ddp.no_sync() if j+1<len(local) else nullcontext():
                        loss,count=numerator(ddp(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits,b['labels'])
                        loss.backward();local_count+=int(count)
                total=normalize_task_gradients(ddp,local_count,ddp_averaged=True)
                oracle_count=0
                for row in batch_rows:
                    b=collate([row],tok.eos_token_id,'cuda')
                    for m in adapted(oracle).values():
                        m._sample_indices=torch.arange(min(64,int(b['attention_mask'].sum()))) if score=='ACT' else None
                    loss,count=numerator(oracle(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits,b['labels'])
                    loss.backward();oracle_count+=int(count)
                for p in oracle.parameters():
                    if p.grad is not None:p.grad.div_(oracle_count)
            assert total==oracle_count
            params=[p for p in model.parameters() if p.requires_grad]
            refparams=[p for p in oracle.parameters() if p.requires_grad]
            graderr=torch.stack([(p.grad-q.grad).square().sum() for p,q in zip(params,refparams)]).sum().sqrt()
            gradnorm=torch.stack([p.grad.square().sum() for p in refparams]).sum().sqrt()
            relative=float(graderr/gradnorm.clamp_min(1e-20))
            assert relative<5e-5,('gradient mismatch',init,score,step,relative)
            got={n:m.task_score(total) for n,m in adapted(model).items()}
            want={n:m.task_score(total) for n,m in adapted(oracle).items()}
            err=torch.cat([got[n]-want[n] for n in got]).norm()
            norm=torch.cat(list(want.values())).norm()
            score_error=float(err/norm.clamp_min(1e-20))
            assert score_error<5e-5,('score mismatch',init,score,step,score_error)
            mask_equal=True
            if step==1:
                def decision(mods,scores):
                    cells=[Candidate(n,int(i),m.slot_cost,float(v)) for n,m in mods.items() for i,v in zip(m.slot_ids,scores[n])]
                    budget=sum(2*(m.base.in_features+m.base.out_features) for m in mods.values())
                    return allocate(cells,budget,1)[0]
                keep=decision(adapted(model),got);expected=decision(adapted(oracle),want)
                mask_equal=keep==expected;assert mask_equal
                masks={n:torch.tensor([(n,int(i)) in keep for i in m.slot_ids]) for n,m in adapted(model).items()} if rank==0 else None
                broadcast_masks(ddp,opt,masks)
                for n,m in adapted(oracle).items():m.set_mask(torch.tensor([(n,int(i)) in expected for i in m.slot_ids],device='cuda'),refopt)
            torch.nn.utils.clip_grad_norm_(params,1.);torch.nn.utils.clip_grad_norm_(refparams,1.)
            opt.step();refopt.step()
            parameter_error=max(float((p-q).abs().max()) for p,q in zip(params,refparams))
            assert parameter_error<2e-6,('optimizer mismatch',init,step,parameter_error)
            if step==1:
                # No DDP parameter set is mutated: destroy wrapper, slice both
                # optimizer mappings, then construct a new reducer on survivors.
                del ddp,params,refparams,loss;gc.collect();dist.barrier()
                opt=compact(model,opt);refopt=compact(oracle,refopt)
                ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank],broadcast_buffers=False)
                for n,m in adapted(model).items():
                    assert torch.equal(m.slot_ids,adapted(oracle)[n].slot_ids)
                dist.barrier()
            records.append({'init':init,'score':score,'step':step+1,'global_supervised_tokens':total,
                            'gradient_relative_l2_error':relative,'score_relative_l2_error':score_error,
                            'maximum_parameter_absolute_error':parameter_error,'mask_equal':mask_equal,
                            'compacted':step>=1})
        assert backbone_hash(model)==original==backbone_hash(oracle)
        del ddp,opt,refopt,model,oracle;gc.collect();torch.cuda.empty_cache();dist.barrier()
    if rank==0:
        json_write(a.out,{'status':'passed','topology':'2-rank real Qwen1.5B DDP/NCCL versus single-process oracle',
                         'backend':'EFFICIENT_ATTENTION','records':records,
                         'scope':'I_GATE, L_ACT, T0_GATE, T1_GATE through optimizer and DDP rebuild; other score families have separate tiny-layer parity',
                         'no_sync':True,'local_microbatches':[1,3],'original_backbone_unchanged':True,
                         'tolerances':{'gradient_relative_l2':5e-5,'score_relative_l2':5e-5,'parameter_max_absolute':2e-6}})
        print(json.dumps({'status':'passed','cases':len(records)}),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
