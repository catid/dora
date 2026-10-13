"""Exercise every frozen-grid configuration on independent finite calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from experiments import init_registry as registry
from experiments import init_data, init_warmstart, init_loftq, init_ablations


def audit(output,device='cuda'):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    root=Path(output).parent/'registry_smoke_cache';root.mkdir(parents=True,exist_ok=True)
    results=[]
    for rank in [2,8]:
        g=torch.Generator().manual_seed(335+rank)
        # Non-power-of-two rectangular dimensions expose cropped basis assumptions.
        weight=torch.randn(21,19,generator=g)*.05
        x=torch.randn(70,19,generator=g)
        dy=torch.randn(70,21,generator=g)
        y=torch.randn(70,21,generator=g)
        xc=x-x.mean(0)
        stats={'input_cov':x.T@x/70,'input_centered_cov':xc.T@xc/70,'input_count':70,
               'corda_input_cov':x.T@x/x.max().abs().square(),
               'tlora_input_cov':xc.T@xc,
               'astra_output_cov':y.T@y/y.max().abs().square(),
               'filet_input_cov':x.T@x/70,'filet_output_grad_cov':dy.T@dy/70,
               'output_grad_cov':dy.T@dy/70,'output_fisher_diag':dy.square().mean(0),
               'weight_grad':dy.T@x/70,'training_population_size':1024,
               'input_gram_batches':[x[:35].T@x[:35],x[35:].T@x[35:]]}
        calibration={'modules':{'0':stats},'metadata':{'split':'train',
                     'decomposition_cache_dir':str(root), 'factor_cache_dir':str(root/'factors'),
                     'inilora_group_std':{'0':.05}}}
        for lane in ['centered','warmstart','quantized']:
            for family in registry.families(lane):
                for index,config in enumerate(family['configs']):
                    start=time.perf_counter();seed=17
                    base=torch.nn.Sequential(torch.nn.Linear(19,21,bias=False,dtype=torch.bfloat16,device=device))
                    with torch.no_grad():base[0].weight.copy_(weight.to(device))
                    kind=family['kind']
                    if kind=='random':bound=init_ablations.backend(init_ablations.InitSpec(**config),seed)
                    elif kind=='data':bound=init_data.backend(config,seed,calibration)
                    elif kind=='warmstart':bound=init_warmstart.backend(config,seed,calibration)
                    else:bound=init_loftq.backend(config,seed,calibration)
                    torch.manual_seed(85)
                    expected=torch.nn.Sequential(torch.nn.Linear(19,21,bias=False,dtype=torch.bfloat16,device=device))
                    with torch.no_grad():expected[0].weight.copy_(weight.to(device))
                    torch.manual_seed(91);init_ablations.backend(init_ablations.InitSpec(),seed).inject_adapters(expected,'ref',rank)
                    rng=torch.get_rng_state().clone()
                    torch.manual_seed(91);bound.inject_adapters(base,family['family_id'],rank)
                    assert torch.equal(rng,torch.get_rng_state())
                    assert [n for n,p in base[0].named_parameters() if p.requires_grad]==['m','lora_B','lora_A']
                    probe=torch.randn(4,19,generator=g).to(device)
                    with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16):
                        actual=base(probe);reference=expected(probe)
                    assert torch.isfinite(actual).all()
                    if lane=='centered':assert torch.equal(actual,reference),(family['family_id'],index,rank)
                    actual.float().square().mean().backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in base.parameters() if p.requires_grad)
                    init=base[0].norm_diagnostics()['initializer']
                    results.append({'lane':lane,'family_id':family['family_id'],'index':index,'rank':rank,
                                    'source_default':family['source_default_index']==index,
                                    'config':config,'status':'pass','seconds':time.perf_counter()-start,
                                    'initial_probe_max_error':float((actual-reference).detach().abs().max()),
                                    'initialization_kind':init['initializer_kind']})
        print(json.dumps({'rank':rank,'completed':len(results)}),flush=True)
    names=['init_registry.py','initialization_protocol.md','init_ablations.py','init_data.py','init_warmstart.py','init_loftq.py','audit_init_registry.py']
    evidence={'status':'pass','cases':len(results),'rows':results,'device':device,'torch':str(torch.__version__),
              'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
              'scope':'Every registered config at both ranks, rectangular BF16 engineering probe; no task quality claim'}
    Path(output).write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda');args=parser.parse_args();audit(args.output,args.device)
