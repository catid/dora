#!/usr/bin/env python3
"""Record native SDPA support, numerical accuracy, causality and determinism."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import warnings
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


def main():
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(49)
    records=[]
    for padding in [False, True]:
        q=torch.randn(2,12,17,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn_like(q,requires_grad=True); v=torch.randn_like(q,requires_grad=True)
        weights=torch.randn_like(q).float()
        lengths=torch.tensor([17,9],device='cuda') if padding else torch.tensor([17,17],device='cuda')
        valid=torch.arange(17,device='cuda')[None,:]<lengths[:,None]
        mask=torch.ones(17,17,device='cuda',dtype=torch.bool).tril()[None,None]&valid[:,None,None,:]
        q64,k64,v64=[t.detach().double().requires_grad_() for t in [q,k,v]]
        with sdpa_kernel(SDPBackend.MATH):
            oracle=torch.nn.functional.scaled_dot_product_attention(q64,k64,v64,attn_mask=mask)
            og=torch.autograd.grad((oracle*weights).sum(),(q64,k64,v64))
        for backend in ['CUDNN_ATTENTION','FLASH_ATTENTION','EFFICIENT_ATTENTION','MATH']:
            row={'backend':backend,'padding':padding}
            with warnings.catch_warnings(record=True) as caught:
                try:
                    vals=[]
                    with sdpa_kernel(getattr(SDPBackend,backend)):
                        for _ in range(2):
                            # Causal flag without an explicit mask for unpadded shapes.
                            out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask if padding else None,is_causal=not padding)
                            grads=torch.autograd.grad((out.float()*weights).sum(),(q,k,v))
                            vals.append([out,*grads])
                    row['bitwise_repeat']=all(torch.equal(a,b) for a,b in zip(*vals))
                    row['max_abs_forward_error']=float((vals[0][0].double()-oracle).abs().max())
                    row['max_abs_gradient_error']=max(float((a.double()-b).abs().max()) for a,b in zip(vals[0][1:],og))
                    row['relative_l2_gradient_error']=max(float((a.double()-b).norm()/b.norm()) for a,b in zip(vals[0][1:],og))
                    row['passed']=(row['bitwise_repeat'] and row['max_abs_forward_error']<.02 and row['relative_l2_gradient_error']<.02)
                except RuntimeError as exc:
                    row['passed']=False; row['error']=str(exc)
                row['warnings']=[str(w.message) for w in caught]
            records.append(row)
    result={'torch':torch.__version__,'device':torch.cuda.get_device_name(),'deterministic_algorithms':True,
            'policy':'EFFICIENT_ATTENTION for all v3 primary candidates, pending actual-model profiling and exact resume',
            'exception':'cuDNN SDPA rejects deterministic execution; native flash rejects explicit padding mask. Keep original causal/padding pattern.',
            'records':records}
    Path('artifacts/sdpa_v3_oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(r['passed'] for r in records if r['backend']=='EFFICIENT_ATTENTION')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
