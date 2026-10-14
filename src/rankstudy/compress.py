"""Thin-QR/small-core SVD compression of centered updates."""
from __future__ import annotations
import torch


def factorized_update(A0,B0,A,B,scale=1.0):
    # Delta = scale [B0,V] [U; A0+U].
    U=A-A0; V=B-B0
    C=torch.cat([B0,V],dim=1)
    D=torch.cat([U,A0+U],dim=0)
    return C,D


def thin_core_svd(A0,B0,A,B,scale=1.0,energy=None,rank=None):
    C,D=factorized_update(A0,B0,A,B,scale)
    Qc,Rc=torch.linalg.qr(C,mode='reduced')
    Qd,Rd=torch.linalg.qr(D.T,mode='reduced')
    core=scale*(Rc @ Rd.T)
    uc,s,vt=torch.linalg.svd(core,full_matrices=False)
    if rank is None and energy is not None:
        cumulative=torch.cumsum(s.square(),0)/s.square().sum().clamp_min(torch.finfo(s.dtype).tiny)
        rank=int(torch.searchsorted(cumulative,torch.tensor(energy,device=s.device)).item()+1)
    if rank is None: rank=len(s)
    rank=max(0,min(int(rank),len(s)))
    left=Qc@uc[:,:rank]
    right=Qd@vt[:rank,:].T
    vals=s[:rank]
    return left,vals,right,{'full_rank':len(s),'retained_rank':rank,'singular_values':s.detach().cpu().tolist(),'reconstruction_error_fro':float(s[rank:].detach().square().sum().sqrt())}


def balanced_lora_factors(left, singular_values, right, scale=1.0):
    """Return ordinary LoRA B,A whose product is the retained update."""
    if left.shape[1] != len(singular_values) or right.shape[1] != len(singular_values):
        raise ValueError('factor/rank mismatch')
    # Put the square root of sigma in both factors and fold the requested
    # multiplier into B.  The exported product is exactly left diag(s) right^T.
    root=singular_values.clamp_min(0).sqrt()
    B=left*root[None,:]
    A=right.T*root[:,None]
    if scale <= 0: raise ValueError('scale must be positive')
    return B*scale,A


def compress_centered(A0,B0,A,B,scale=1.0,rank=None,energy=None):
    left,s,right,meta=thin_core_svd(A0,B0,A,B,scale=scale,rank=rank,energy=energy)
    Bx,Ax=balanced_lora_factors(left,s,right,scale=1.0)
    meta['export_rank']=int(s.numel()); meta['export_scale']=1.0
    return Bx,Ax,meta
