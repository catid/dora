#!/usr/bin/env python3
"""Small deterministic diagnostics for score calibration and gauge effects."""
from __future__ import annotations
import hashlib, json, random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch import nn
from rankstudy.v3.adapters import AdapterLinear


def digest(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(module,x,target,gate=None):
    module.begin_update(gate is not None)
    if gate is not None: module._gate=gate
    return ((module(x)-target)**2).mean()


def one(init,seed):
    torch.manual_seed(seed);random.seed(seed)
    base=nn.Linear(7,5,bias=False);base.weight.data.normal_(0,.2)
    module=AdapterLinear(base,4,init,1.,seed=seed,score='GATE')
    with torch.no_grad():
        if init=='L':module.B.normal_(0,.15);module.A.normal_(0,.15)
        else:module.U.normal_(0,.15);module.V.normal_(0,.15)
    x=torch.randn(32,7);target=torch.randn(32,5)
    before_digest=digest(module);state_before={k:v.detach().clone() for k,v in module.state_dict().items()};rng_before=torch.get_rng_state().clone()
    module.begin_update(True);loss=((module(x)-target)**2).mean();loss.backward()
    signed=(-module._gate.grad.detach()).tolist()
    base_loss=float(loss.detach())
    singles=[]
    for i in range(4):
        old=module.mask.clone();module.mask[i]=False
        singles.append(float((((module(x)-target)**2).mean()-base_loss).detach()));module.mask.copy_(old)
    pair_old=module.mask.clone();module.mask[0]=False;module.mask[1]=False
    pair=float((((module(x)-target)**2).mean()-base_loss).detach());module.mask.copy_(pair_old)
    eps=1e-2;curvature=[]
    for i in range(4):
        gate=torch.ones(4);gate[i]+=eps;plus=float(evaluate(module,x,target,gate))
        gate=torch.ones(4);gate[i]-=eps;minus=float(evaluate(module,x,target,gate))
        gate=torch.ones(4);center=float(evaluate(module,x,target,gate))
        curvature.append((plus+minus-2*center)/(eps*eps))
    # Gauge-rescale current and origin factors together. The centered update
    # is unchanged algebraically, while coordinate scores can be compared.
    pre_gauge=float(evaluate(module,x,target,torch.ones(4)))
    if init=='L':
        with torch.no_grad():module.A[0].mul_(2.5);module.B[:,0].div_(2.5)
    else:
        with torch.no_grad():
            module.A0[0].div_(2.5);module.U[0].div_(2.5)
            module.B0[:,0].mul_(2.5);module.V[:,0].mul_(2.5)
    post_gauge=float(evaluate(module,x,target,torch.ones(4)))
    # Restore every tensor and RNG state after the shadow probes.
    with torch.no_grad():
        for key,value in state_before.items():
            target_param=module
            parts=key.split('.')
            for part in parts[:-1]: target_param=getattr(target_param,part)
            getattr(target_param,parts[-1]).copy_(value)
    torch.set_rng_state(rng_before)
    restored_digest=digest(module)
    out={'init':init,'seed':seed,'base_loss':base_loss,'signed_first_order_deletion':signed,
         'actual_single_deletion':singles,'curvature_proxy':curvature,'pair_deletion_01':pair,
         'pair_interaction_01':pair-singles[0]-singles[1],
         'gauge_forward_abs_delta':abs(post_gauge-pre_gauge),
         'adapter_digest_before':before_digest,'adapter_digest_after_probe':restored_digest,
         'rng_digest_before':hashlib.sha256(rng_before.numpy().tobytes()).hexdigest(),
         'restoration_verified':restored_digest==before_digest,
         'restoration_expected':True}
    return out


def main():
    # Each result is independent; explicitly record all seeds and sources.
    results=[one(init,2903+i) for i,init in enumerate(('L','I'))]
    for r in results:
        r['status']='passed' if r['gauge_forward_abs_delta']<1e-5 else 'failed'
    out=Path('artifacts/mechanism_diagnostics_v3.json');out.write_text(json.dumps({'schema':'rankstudy-v3-mechanism-diagnostics','status':'complete','selection_eligible':False,'results':results},indent=2)+'\n')
    print(json.dumps({'status':'complete','results':len(results)}))


if __name__=='__main__':main()
