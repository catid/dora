"""BF16-backbone / FP32-adapter linear modules for the study."""
from __future__ import annotations
import math
import hashlib
import torch
from torch import nn
from .algebra import CenteredTwoFactor, CenteredTriplet, FaithfulAdaLoRA


class StudyLinear(nn.Module):
    def __init__(self, base, rank, init='L', scale=1.0, method='L_FIXED', seed=0,
                 triplet=False, penalty=0.0, adalora=False):
        super().__init__(); self.base=base; self.rank=int(rank); self.scale=float(scale)
        self.method=method; self.init=init; self.triplet=triplet; self.adalora=adalora; self.penalty=float(penalty)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None: self.base.bias.requires_grad_(False)
        g=torch.Generator(device='cpu').manual_seed(int(seed) & ((1<<63)-1))
        d,n=base.out_features,base.in_features
        if adalora:
            A0=torch.randn(rank,n,generator=g,dtype=torch.float32)*.02
            B0=torch.randn(d,rank,generator=g,dtype=torch.float32)*.02
            self.adapter=FaithfulAdaLoRA(A0,B0,scale=scale)
        elif triplet:
            P0=torch.randn(d,rank,generator=g,dtype=torch.float32)*.5
            Q0=torch.randn(rank,n,generator=g,dtype=torch.float32)*.5
            lam0=torch.ones(rank) if init=='T1' else torch.zeros(rank)
            self.adapter=CenteredTriplet(P0,Q0,lam0,scale=scale)
            self.P,self.Q,self.lam=self.adapter.P,self.adapter.Q,self.adapter.lam
        elif init=='I':
            A0=torch.randn(rank,n,generator=g,dtype=torch.float32)*.5
            B0=torch.randn(d,rank,generator=g,dtype=torch.float32)*.5
            self.adapter=CenteredTwoFactor(A0,B0,scale=scale)
        else:
            bound=1/math.sqrt(n)
            A0=torch.empty(rank,n,dtype=torch.float32).uniform_(-bound,bound,generator=g)
            B0=torch.zeros(d,rank,dtype=torch.float32)
            self.adapter=CenteredTwoFactor(A0,B0,scale=scale)
        self.register_buffer('active_mask',torch.ones(rank,dtype=torch.bool))
        self._last_x=None; self._last_grad_out=None; self._gate_grad_accum=None; self._score_x=[]; self._score_valid=[]

    def begin_score_accum(self):
        self._gate_grad_accum = None; self._score_x=[]; self._score_valid=[]

    @property
    def A(self): return self.adapter.Q if self.triplet else self.adapter.A
    @property
    def B(self): return self.adapter.P if self.triplet else self.adapter.B
    @property
    def A0(self): return self.adapter.Q0 if self.triplet else self.adapter.A0
    @property
    def B0(self): return self.adapter.P0 if self.triplet else self.adapter.B0

    @property
    def E(self):
        if not self.adalora:
            raise AttributeError('E is only defined for faithful AdaLoRA')
        return self.adapter.E

    def set_mask(self,mask):
        mask=mask.to(self.active_mask.device,dtype=torch.bool); self.active_mask.copy_(mask); self.adapter.set_mask(mask)

    def _adapter_output(self,x):
        xf=x.float().reshape(-1,x.shape[-1]); m=self.active_mask.float()
        if self.adalora:
            return (xf @ self.adapter.delta_expanded().T).reshape(*x.shape[:-1],self.base.out_features)
        elif self.triplet:
            return (xf @ self.adapter.delta_expanded().T).reshape(*x.shape[:-1],self.base.out_features)
        elif self.init=='I':
            U=(self.A-self.A0)*m[:,None]; V=(self.B-self.B0)*m[None,:]
            y=(xf @ U.T) @ self.B0.T + (xf @ (self.A0+U).T) @ V.T
        else:
            y=(xf @ (self.A*m[:,None]).T) @ (self.B*m[None,:]).T
        return y.reshape(*x.shape[:-1],self.base.out_features)*self.scale

    def forward(self,x):
        base=self.base(x.to(dtype=self.base.weight.dtype)); ad=self._adapter_output(x)
        x_flat=x.detach().float().reshape(-1,x.shape[-1]); self._last_x=x_flat; self._score_x.append(x_flat)
        def save_grad(g):
            gf=g.detach().float().reshape(-1,g.shape[-1]); self._last_grad_out=gf
            term=gf.T @ x_flat
            self._gate_grad_accum = term if self._gate_grad_accum is None else self._gate_grad_accum + term
            return g
        if ad.requires_grad: ad.register_hook(save_grad)
        return base + ad.to(dtype=base.dtype)

    def score_gate(self):
        if self._last_x is None or self._last_grad_out is None: return torch.zeros(self.rank,device=self.A.device)
        G=self._gate_grad_accum if self._gate_grad_accum is not None else self._last_grad_out.T @ self._last_x
        slots=self.adapter.slot_deltas()
        return (slots.to(G.dtype) * G[None]).flatten(1).sum(1).abs().detach()

    def orth_penalty(self):
        if not self.triplet or self.active_mask.sum()<2: return self.A.sum()*0
        P=self.P[:,self.active_mask]; Q=self.Q[self.active_mask,:]
        P=P/(P.norm(dim=0,keepdim=True).clamp_min(1e-8)); Q=Q/(Q.norm(dim=1,keepdim=True).clamp_min(1e-8))
        def off(x): return x-torch.diag(torch.diagonal(x))
        return (off(P.T@P).square().sum()+off(Q@Q.T).square().sum())/max(1,2*P.shape[1]*(P.shape[1]-1))

    def adalora_orth_penalty(self):
        """Upstream AdaLoRA unnormalized Frobenius factor penalty."""
        if not self.adalora:
            raise RuntimeError('only faithful AdaLoRA has this penalty')
        r=self.A.shape[0]; eye=torch.eye(r,device=self.A.device,dtype=self.A.dtype)
        return 0.5*((self.A@self.A.T-eye).norm(p='fro') + (self.B.T@self.B-eye).norm(p='fro'))


def inject(model,rank=16,target_modules=('q_proj','o_proj'),init='L',scale=1.,method='L_FIXED',seed=0,triplet=False,penalty=0.,adalora=False):
    model.requires_grad_(False)
    selected=[]
    for name,module in list(model.named_modules()):
        if isinstance(module,nn.Linear) and any(name==t or name.endswith('.'+t) for t in target_modules): selected.append((name,module))
    if not selected: raise ValueError('no target modules')
    for name,module in selected:
        parent=model; bits=name.split('.')
        for bit in bits[:-1]: parent=getattr(parent,bit)
        stable=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
        setattr(parent,bits[-1],StudyLinear(module,rank,init,scale,method,seed+stable%100000,triplet,penalty,adalora).to(module.weight.device))
    return model


def modules(model): return [m for m in model.modules() if isinstance(m,StudyLinear)]
