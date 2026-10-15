#!/usr/bin/env python3
"""Run with torchrun --standalone --nproc-per-node=2 on two idle allocated GPUs.

Compare the exact same global examples to single-process task gradients, signed
gate reduction and resulting masks. Unequal local token counts are intentional.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import torch
import torch.distributed as dist
from torch import nn
from rankstudy.v3.adapters import AdapterLinear
from rankstudy.v3.allocation import Candidate,allocate
from rankstudy.v3.distributed import normalize_task_gradients,broadcast_masks


def main():
    rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    torch.use_deterministic_algorithms(True)
    records=[]
    for init in ['L','I','T0','T1']:
        for score in ['GATE']+(['RAW','DISP','ACT'] if init in ['L','I'] else []):
            torch.manual_seed(882)
            model=nn.Sequential(AdapterLinear(nn.Linear(7,6,bias=False),4,init,1.,33,score)).cuda().double()
            with torch.no_grad():
                for p in model[0].parameters(recurse=False): p.add_(torch.randn_like(p)*.1)
            oracle=copy.deepcopy(model)
            ddp=torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank])
            # Data constructed from CPU stream, identical on both processes.
            g=torch.Generator().manual_seed(991)
            x=torch.randn(7,7,generator=g,dtype=torch.float64).cuda()
            target=torch.randn(7,6,generator=g,dtype=torch.float64).cuda()
            ddp.module[0].begin_update(); oracle[0].begin_update()
            selected=slice(0,2) if rank==0 else slice(2,7)
            if score=='ACT':
                ddp.module[0]._sample_indices=torch.arange(x[selected].shape[0])
                oracle[0]._sample_indices=torch.arange(7)
            # Numerators, not separately averaged losses from unequal batches.
            (ddp(x[selected])-target[selected]).square().sum().backward()
            (oracle(x)-target).square().sum().backward()
            denominator=normalize_task_gradients(ddp,x[selected].shape[0]*6,ddp_averaged=True)
            for p in oracle.parameters():
                if p.grad is not None: p.grad.div_(42)
            got=ddp.module[0].task_score(denominator)
            want=oracle[0].task_score(42)
            max_grad=max(float((a.grad-b.grad).abs().max()) for a,b in zip(model.parameters(),oracle.parameters()) if a.requires_grad)
            max_score=float((got-want).abs().max())
            torch.testing.assert_close(got,want,atol=1e-10,rtol=1e-10)
            assert max_grad<1e-10
            cells=[Candidate('0',i,13,float(s)) for i,s in enumerate(got)]
            kept,_=allocate(cells,26,1)
            expected,_=allocate([Candidate('0',i,13,float(s)) for i,s in enumerate(want)],26,1)
            assert kept==expected
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0.)
            masks={'0':torch.tensor([('0',i) in kept for i in range(4)])} if rank==0 else None
            broadcast_masks(ddp,opt,masks)
            assert model[0].mask.tolist()==[('0',i) in expected for i in range(4)]
            records.append({'init':init,'score':score,'max_abs_gradient_error':max_grad,
                            'max_abs_score_error':max_score,'mask_equal':True,'global_supervised_tokens':denominator})
            del ddp
            dist.barrier()
    if rank==0:
        receipt={'status':'passed','topology':'torchrun 2-rank DDP/NCCL','torch':torch.__version__,
                 'records':records,'limitation':'Tiny-layer parity; full-LLM DDP and wrapper rebuild tests remain pending.'}
        Path('artifacts/ddp_v3_parity.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2))
    dist.destroy_process_group()


if __name__=='__main__': main()
