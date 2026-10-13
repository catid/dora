"""Independent formula audit against pinned primary recipes, not task quality."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from experiments.init_data import initialize_factors


def canonical_svd(w):
    u,s,vh=torch.linalg.svd(w,full_matrices=False)
    signs=u[u.abs().argmax(0),torch.arange(u.shape[1])].sign()
    signs[signs==0]=1
    return u*signs,s,vh*signs[:,None]


def canonical_eigh(c):
    e,v=torch.linalg.eigh(c)
    signs=v[v.abs().argmax(0),torch.arange(v.shape[1])].sign()
    signs[signs==0]=1
    return e,v*signs


def run():
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(228)
    w=torch.randn(9,7,generator=generator)
    x=torch.randn(35,7,generator=generator)
    dy=torch.randn(35,9,generator=generator)
    gradient=dy.T@x/len(x)
    cov=x.T@x/len(x)
    centered=(x-x.mean(0)).T@(x-x.mean(0))/len(x)
    input_grams=[x[:15].T@x[:15],x[15:].T@x[15:]]
    corda=sum(z.T@z/(z.max().abs()**2) for z in [x[:15],x[15:]])/2
    tlora=sum((z-z.mean(0)).T@(z-z.mean(0)) for z in [x[:15],x[15:]])/2
    y=torch.randn(35,9,generator=generator)
    astra=sum(z.T@z/(z.max().abs()**2) for z in [y[:15],y[15:]])/2
    sy=dy.T@dy/len(x)
    stats={'input_cov':cov,'input_centered_cov':centered,'input_count':len(x),
           'corda_input_cov':corda,'tlora_input_cov':tlora,'astra_output_cov':astra,
           'filet_input_cov':cov,'filet_output_grad_cov':sy,
           'weight_grad':gradient,'output_fisher_diag':dy.square().mean(0),
           'training_population_size':500,'input_gram_batches':input_grams}
    rows=[]
    def check(name,spec,ref_b,ref_a,*,product_only=False):
        b,a,meta=initialize_factors(w,2,{'method':name,**spec},stats)
        if product_only:
            tolerance=1e-4 if name=='cntlora' else 2e-5
            torch.testing.assert_close(b@a,ref_b@ref_a,atol=tolerance,rtol=tolerance)
        else:
            torch.testing.assert_close(b,ref_b,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(a,ref_a,atol=2e-5,rtol=2e-5)
        rows.append({'method':name,'spec':spec,'maximum_product_error':float((b@a-ref_b@ref_a).abs().max()),
                     'checked':'factor_product' if product_only else 'both_factors',
                     'source_initial_product':meta['source_initial_product'],'adaptations':meta['adaptations']})
        return b,a,meta
    u,s,vh=canonical_svd(w)
    check('pissa',{},u[:,:2]*s[:2].sqrt(),s[:2,None].sqrt()*vh[:2])
    check('milora',{},u[:,-2:]*s[-2:].sqrt(),s[-2:,None].sqrt()*vh[-2:])
    check('mica',{},u[:,-2:],torch.zeros(2,7))
    q,r=torch.linalg.qr(w,mode='reduced');check('olora',{},q[:,:2],r[:2])
    _,sx,vhx=torch.linalg.svd(x-x.mean(0),full_matrices=False)
    check('eva',{},torch.zeros(9,2),canonical_eigh(centered)[1][:,-2:].flip(1).T)
    # Whitening is independently derived from activation SVD rather than covariance powers.
    b,a,meta=initialize_factors(w,2,{'method':'eva','whiten':True},stats)
    torch.testing.assert_close(a.T@a,vhx[:2].T@torch.diag(1/sx[:2])@vhx[:2],atol=2e-6,rtol=2e-5)
    rows.append({'method':'eva_whiten','checked':'down_gram_equals_V_diag_inverse_activation_singular_value_Vt'})
    c=corda+.01*corda.diag().mean()*torch.eye(7)
    u,s,vh=canonical_svd(w@c);v=vh@torch.linalg.inv(c)
    check('corda',{},u[:,:2]*s[:2].sqrt(),s[:2,None].sqrt()*v[:2])
    check('corda',{'selection':'bottom'},u[:,-2:]*s[-2:].sqrt(),s[-2:,None].sqrt()*v[-2:])
    # Near-singular matrix triggers the declared damping retry, with no silent inverse fallback.
    basis=torch.randn(7,3,generator=generator)
    singular=dict(stats,input_cov=basis@basis.T)
    _,_,retry=initialize_factors(w,2,{'method':'corda','covariance_key':'input_cov','damping':1e-9},singular)
    assert retry['damping_used']>1e-9 and retry['inverse_residual']<.05
    rows.append({'method':'corda_retry','damping_used':retry['damping_used'],'inverse_residual':retry['inverse_residual']})
    _,v=canonical_eigh(astra)
    check('astra',{},v[:,:2],v[:,:2].T@w)
    ug,sg,vhg=canonical_svd(gradient);scale=9**.25/math.sqrt(16)
    for direction,ai,bi in [('ArB2r',[0,1],[2,3]),('A2rBr',[2,3],[0,1]),('ArBr',[1,3],[0,2])]:
        check('loraga',{'direction':direction},ug[:,bi]*scale,vhg[ai]*scale)
    roots=(sg[:2]/sg[0]).sqrt()/math.sqrt(128)
    check('loraone',{},-ug[:,:2]*roots,roots[:,None]*vhg[:2])
    v=F.normalize(w.T@w,dim=0);u=F.normalize(w@v,dim=0)
    energies=(v.T@cov@v).diag()*(u.T@sy@u).diag()
    ix=torch.topk(-energies,2).indices;roots=energies[ix].clamp_min(1e-8).sqrt()
    check('filet',{},u[:,ix]*roots,roots[:,None]*v[:,ix].T)
    # Independent Cholesky natural-gradient and full guidance reference.
    sx=cov+1e-6*torch.eye(7);chol=torch.linalg.cholesky(sx)
    iy=(stats['output_fisher_diag']+1e-6).reciprocal()
    d=-torch.cholesky_solve(gradient.T,chol)*iy[None,:]
    norm=torch.linalg.svdvals(d)[0];d/=norm
    coefficient=iy.sum()/500/norm**2
    h=-d@d.T+.03*coefficient*torch.cholesky_inverse(chol)
    _,v=canonical_eigh((h+h.T)/2);a=v[:,:2]
    check('lorada',{},d.T@a/32,a.T/32,product_only=True)
    for mode in ['cross','self','shift']:
        targets=[]
        for c in input_grams:
            if mode=='cross':t=w@torch.linalg.pinv(c,rtol=1e-6,hermitian=True)
            elif mode=='shift':t=torch.linalg.pinv(w.T-c@w.T,rtol=1e-6)
            else:
                e,v=canonical_eigh(c);half=v.flip(1)*e.flip(0).clamp_min(0).sqrt()
                t=w@torch.linalg.pinv(half,rtol=1e-6)
            targets.append(t)
        delta=(w+sum(targets))/3-w
        u,s,vh=canonical_svd(delta)
        check('cntlora',{'mode':mode},u[:,:2]*s[:2].sqrt(),s[:2,None].sqrt()*vh[:2],product_only=True)
    _,_,vh=canonical_svd(w@tlora);check('tlora',{},torch.zeros(9,2),vh[:2])
    # Execute source-shaped U/P Adam with StepLR before step and identical local draws.
    g=torch.Generator().manual_seed(123)
    b=(torch.randn(9,2,generator=g)*.04).requires_grad_();p=(torch.randn(7,2,generator=g)*.04).requires_grad_()
    opt=torch.optim.Adam([b,p],lr=.0005,foreach=False)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=5000,gamma=.9)
    for step in range(5001):
        opt.zero_grad();loss=((b@p.T-w)**2).mean();loss.backward();scheduler.step();opt.step()
    check('inilora',{'seed':123,'prefit_steps':5001,'initial_std':.04},b.detach(),p.detach().T)
    source=Path(__file__).with_name('init_data.py')
    return {'status':'pass','cases':len(rows),'rows':rows,
            'implementation_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Primary formula and stated solver/sign adaptations; no task performance claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':result['cases'],'output':str(args.output)}))
