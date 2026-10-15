"""Compress the actual update on original W0 using only thin factors.

QR/SVD is performed in FP64 on CPU by default. Exports are ordinary FP32 LoRA
factors with an explicit fixed scale, never an initialization shift in W0.
"""
from __future__ import annotations
import math
import hashlib
import json
from pathlib import Path
from functools import reduce

import torch
from torch import nn
from safetensors.torch import save_file

from .adapters import AdapterLinear, adapted
from .checkpoint import load_adapter, json_write


@torch.no_grad()
def actual_factors(module, dtype=torch.float64, device='cpu'):
    def t(name):
        return getattr(module,name).detach().to(device=device,dtype=dtype)
    mask=module.mask.detach().to(device)
    if module.init=='L':
        c,d=t('B')[:,mask],t('A')[mask]
    elif module.init=='I':
        a0,b0,u,v=(t(n) for n in ['A0','B0','U','V'])
        c=torch.cat((b0[:,mask],v[:,mask]),1)
        d=torch.cat((u[mask],(a0+u)[mask]),0)
    elif module.init in {'T0','T1'}:
        p0,q0,l0,v,u,z=(t(n) for n in ['P0','Q0','lambda0','dP','dQ','dlambda'])
        c=torch.cat((p0[:,mask],v[:,mask]),1)
        d=torch.cat(((l0[:,None]*u+z[:,None]*(q0+u))[mask],((l0+z)[:,None]*(q0+u))[mask]),0)
    elif module.init=='ADALORA':
        # Actual E values, including any native between-event reactivation.
        c,d=t('lora_B'),t('lora_E')*t('lora_A')
    else:
        raise ValueError(module.init)
    return c*float(module.multiplier),d


def thin_svd(c,d):
    if c.ndim!=2 or d.ndim!=2 or c.shape[1]!=d.shape[0]:
        raise ValueError('incompatible thin factors')
    if not c.shape[1]:
        return c.clone(),torch.empty(0,dtype=c.dtype,device=c.device),d.T.clone()
    qc,rc=torch.linalg.qr(c,mode='reduced')
    qd,rd=torch.linalg.qr(d.T,mode='reduced')
    u,s,vh=torch.linalg.svd(rc@rd.T,full_matrices=False)
    return qc@u,s,qd@vh.T


def numerical_rank(s,d,n):
    if not s.numel(): return 0
    tol=max(d,n)*torch.finfo(s.dtype).eps*float(s.max())
    return int((s>tol).sum())


def prefix_ranks(spectra,costs,budget):
    """Exact equal-cost/feasible DP, otherwise prefix value-per-cost heuristic."""
    names=sorted(spectra)
    if set(names)!=set(costs) or any(costs[n]<=0 for n in names):
        raise ValueError('invalid costs')
    energy={n:[float(x)**2 for x in spectra[n]] for n in names}
    if any(any(a+1e-14<b for a,b in zip(v,v[1:])) for v in energy.values()):
        raise ValueError('spectra must be descending')
    ranks={n:0 for n in names}
    divisor=reduce(math.gcd,costs.values()) if names else 1
    if len(set(costs.values()))<=1:
        algorithm='equal_cost_exact_global_energy'
        items=sorted([(v,n,i) for n in names for i,v in enumerate(energy[n])],key=lambda x:(-x[0],x[1],x[2]))
        count=budget//next(iter(costs.values())) if names else 0
        for _,name,_ in items[:count]: ranks[name]+=1
    elif budget//divisor<=8192:
        algorithm='exact_prefix_integer_dp'
        dp={0:(0.,())}
        for name in names:
            pref=[0.]
            for value in energy[name]: pref.append(pref[-1]+value)
            nxt={}
            for used,(score,pattern) in dp.items():
                for rank,value in enumerate(pref):
                    total=used+rank*(costs[name]//divisor)
                    if total>budget//divisor: break
                    proposal=(score+value,pattern+(rank,))
                    old=nxt.get(total)
                    if old is None or proposal[0]>old[0] or (proposal[0]==old[0] and proposal[1]<old[1]):
                        nxt[total]=proposal
            dp=nxt
        _,(_,chosen)=min(dp.items(),key=lambda kv:(-kv[1][0],-kv[0],kv[1][1]))
        ranks=dict(zip(names,chosen))
    else:
        algorithm='prefix_value_per_cost_greedy'
        room=budget
        while True:
            feasible=[n for n in names if ranks[n]<len(energy[n]) and costs[n]<=room]
            if not feasible: break
            name=min(feasible,key=lambda n:(-energy[n][ranks[n]]/costs[n],n))
            ranks[name]+=1; room-=costs[name]
    used=sum(ranks[n]*costs[n] for n in names)
    kept=sum(sum(energy[n][:ranks[n]]) for n in names)
    return ranks,{'algorithm':algorithm,'budget':budget,'spent':used,'underspend':budget-used,'retained_energy':kept,'floor':0}


def balanced_factors(left,s,right,rank,scale=1.):
    if scale<=0 or rank<0 or rank>s.numel():
        raise ValueError('invalid rank/scale')
    root=(s[:rank].clamp_min(0)/scale).sqrt()
    b=left[:,:rank]*root
    a=root[:,None]*right[:,:rank].T
    return a,b


@torch.no_grad()
def compress_modules(modules,target_rank,scale=1.,policy='uniform'):
    spectra={}; decompositions={}; costs={}; report={}
    for name,module in sorted(modules.items()):
        c,d=actual_factors(module)
        left,s,right=thin_svd(c,d)
        decompositions[name]=(left,s,right)
        spectra[name]=s
        costs[name]=module.base.in_features+module.base.out_features
        report[name]={'singular_values':s.tolist(),'numerical_rank':numerical_rank(s,c.shape[0],d.shape[1]),
                      'source_slots':module.rank,'source_active_slots':int(module.mask.sum())}
    budget=target_rank*sum(costs.values())
    if policy=='global':
        ranks,allocation=prefix_ranks(spectra,costs,budget)
    elif policy in {'uniform','exact'}:
        ranks={n:min(target_rank,len(s)) if policy=='uniform' else report[n]['numerical_rank'] for n,s in spectra.items()}
        used=sum(ranks[n]*costs[n] for n in ranks)
        allocation={'algorithm':policy,'budget':budget,'spent':used,'underspend':budget-used,'diagnostic_above_budget':policy=='exact'}
    else:
        raise ValueError(policy)
    factors={}
    for name,(left,s,right) in decompositions.items():
        a,b=balanced_factors(left,s,right,ranks[name],scale)
        factors[name]={'A':a.float(),'B':b.float()}
        report[name].update({'export_rank':ranks[name],'discarded_frobenius_norm':float(s[ranks[name]:].square().sum().sqrt())})
    return factors,ranks,{'allocation':allocation,'modules':report,'export_scale':scale,'policy':policy,'qr_svd_dtype':'float64','qr_svd_device':'cpu'}


def checkpoint_modules(path,shapes):
    """Read only small adapter tensors; meta-device base placeholders hold shapes."""
    path=Path(path); desc=json.loads((path/'manifest.json').read_text())
    root=nn.Module()
    for name,spec in desc['modules'].items():
        if name not in shapes: raise ValueError('missing target shape')
        owner=root
        bits=name.split('.')
        for bit in bits[:-1]:
            if not hasattr(owner,bit): owner.add_module(bit,nn.Module())
            owner=getattr(owner,bit)
        d,n=shapes[name]
        base=nn.Linear(n,d,bias=False,device='meta')
        owner.add_module(bits[-1],AdapterLinear(base,spec['rank'],spec['init'],spec['scale'],score=spec['score']))
    load_adapter(path,root,'cpu')
    return adapted(root)


def write_export(path,factors,scale,report,source):
    """Standalone ordinary LoRA on W0; no optimizer or hidden backbone state."""
    path=Path(path); path.mkdir(parents=True,exist_ok=False)
    tensors={}; modules={}
    for name,factor in sorted(factors.items()):
        r=factor['A'].shape[0]
        tensors.update({f'{name}.{key}':value.cpu().contiguous() for key,value in factor.items()})
        tensors[f'{name}.mask']=torch.ones(r,dtype=torch.bool)
        tensors[f'{name}.slot_ids']=torch.arange(r)
        tensors[f'{name}.scale']=torch.tensor(float(scale),dtype=torch.float64)
        modules[name]={'init':'L','score':'none','rank':r,'scale':scale,'slot_ids':list(range(r))}
    save_file(tensors,str(path/'adapter.safetensors'))
    manifest={'schema':'rankstudy-v3-ordinary-export','complete':True,'modules':modules,
              'files':{'adapter.safetensors':hashlib.sha256((path/'adapter.safetensors').read_bytes()).hexdigest()},
              'source':source,'unchanged_original_backbone_required':True}
    json_write(path/'manifest.json',manifest); json_write(path/'compression.json',report)
