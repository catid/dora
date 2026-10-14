#!/usr/bin/env python3
"""Small frozen-matrix control used before interpreting LLM runs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from rankstudy.algebra import CenteredTwoFactor

def one(seed,init,anisotropic,noise,steps):
 g=torch.Generator().manual_seed(seed); modules=[]; teachers=[]; xs=[]; ys=[]
 ranks=[0,0,1,1,2,4,8,16]
 for k in range(8):
  d=n=16; W0=torch.randn(d,n,generator=g)*.1
  x=torch.randn(128,n,generator=g); x=x@torch.diag(torch.linspace(3.,.3,n)) if anisotropic else x
  r=ranks[k]
  if r:
   u=torch.randn(d,r,generator=g); v=torch.randn(r,n,generator=g); sv=torch.linspace(1.,.1,r); delta=(u*sv[None,:])@v
  else: delta=torch.zeros_like(W0)
  y=x@(W0+delta).T
  if noise: y += .03*torch.randn(y.shape,generator=g)
  A0=torch.randn(16,n,generator=g)*(.5 if init=='I' else 1/n**.5); B0=torch.randn(d,16,generator=g)*.5 if init=='I' else torch.zeros(d,16)
  a=CenteredTwoFactor(A0,B0); modules.append(a); teachers.append((W0,delta)); xs.append(x); ys.append(y)
 opt=torch.optim.Adam([p for a in modules for p in (a.A,a.B)],lr=.03)
 for _ in range(steps):
  opt.zero_grad(); loss=0
  for a,(W0,_),x,y in zip(modules,teachers,xs,ys): loss += ((x@W0.T+a(x)-y)**2).mean()
  loss.backward(); opt.step()
 slots=[]
 for mi,a in enumerate(modules):
  e=torch.stack([(xs[mi]@d.T).square().mean() for d in a.slot_deltas()]);
  for j,v in enumerate(e): slots.append((float(v.detach()),mi,j))
 slots.sort(reverse=True); out=[]
 for budget in (8,16,32):
  keep={(mi,j) for _,mi,j in slots[:budget]}; err=0; oracle=[]
  for mi,a in enumerate(modules):
   mask=torch.tensor([ (mi,j) in keep for j in range(16)]); a.set_mask(mask); err += float(((xs[mi]@teachers[mi][0].T+a(xs[mi])-ys[mi])**2).mean())
  out.append({'budget_slots':budget,'energy_selected_mse':err/8})
 return {'seed':seed,'init':init,'anisotropic':anisotropic,'noise':noise,'steps':steps,'results':out}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/synthetic_controls.json'); ap.add_argument('--steps',type=int,default=100); args=ap.parse_args()
 rows=[one(s,i,a,n,args.steps) for s in (11,12) for i in ('L','I') for a in (False,True) for n in (False,True)]
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(rows,indent=2)+'\n'); print(json.dumps({'rows':len(rows),'out':args.out}))
if __name__=='__main__': main()
