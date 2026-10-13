"""Independent dense quantization-backend checks; no downstream accuracy claim."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from experiments import init_ablations as accepted
from experiments.init_loftq import backend, quantize_nf4


def run(device='cpu'):
    torch.set_num_threads(2)
    gen=torch.Generator().manual_seed(309)
    weight=(torch.randn(24,16,generator=gen)*.03).bfloat16()
    base=torch.nn.Sequential(torch.nn.Linear(16,24,dtype=torch.bfloat16,device=device))
    with torch.no_grad():base[0].weight.copy_(weight.to(device))
    original=copy.deepcopy(base.state_dict())
    fixed_x=torch.randn(5,16,generator=gen).to(device)
    q,quantization,state=quantize_nf4(weight,return_state=True)
    saturation={'lower_code_count':int((state['indices']==0).sum()),
                'upper_code_count':int((state['indices']==15).sum()),
                'code_count':int(state['indices'].numel()),
                'all_indices_valid':bool(((state['indices']>=0)&(state['indices']<=15)).all())}
    rows=[]
    for method,iterations,policy in [('quantizer_only',1,'original'),('loftq',1,'original'),
                                     ('loftq',3,'original'),('loftq',1,'quantized')]:
        config={'method':method,'iterations':iterations,'magnitude_policy':policy}
        torch.manual_seed(800)
        control=copy.deepcopy(base);accepted.inject_adapters(control,accepted.InitSpec(),rank=4)
        expected_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(800)
        bound=backend(config,factor_seed=800)
        model=copy.deepcopy(base);bound.inject_adapters(model,'audit',rank=4)
        assert torch.equal(expected_rng,torch.random.get_rng_state())
        layer=model[0]
        assert not layer.weight.requires_grad and not layer.bias.requires_grad
        assert layer.weight.dtype==torch.bfloat16
        assert all(p.dtype==torch.float32 for p in [layer.m,layer.lora_A,layer.lora_B])
        assert [name for name,p in layer.named_parameters() if p.requires_grad]==['m','lora_B','lora_A']
        if policy=='original':assert torch.equal(layer.m,control[0].m)
        else:torch.testing.assert_close(layer.m,layer.weight.norm(dim=0,keepdim=True).float(),rtol=0,atol=0)
        if method=='quantizer_only':
            assert torch.equal(layer.weight.cpu(),q.bfloat16())
            assert torch.equal(layer.lora_B,control[0].lora_B)
            assert layer.lora_A.count_nonzero()==0
        snapshot=copy.deepcopy(model.state_dict())
        recreated=copy.deepcopy(base);bound.inject_adapters(recreated,'audit',rank=4)
        recreated.load_state_dict(snapshot,strict=True)
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16):
            output=model(fixed_x);replayed=recreated(fixed_x)
        assert torch.equal(output,replayed)
        output.float().square().mean().backward()
        grads={name:float(parameter.grad.float().norm()) for name,parameter in layer.named_parameters() if parameter.requires_grad}
        assert all(torch.isfinite(parameter.grad).all() for parameter in [layer.m,layer.lora_A,layer.lora_B])
        if method=='loftq':assert grads['lora_A']>0 and grads['lora_B']>0
        else:assert grads['lora_A']>0 and grads['lora_B']==0
        bound.assert_no_clamp(model)
        delta=layer.lora_B.detach().float()@layer.lora_A.detach().float()
        error=float((layer.weight.float()+delta-weight.to(device).float()).square().mean())
        rows.append({'config':config,'recipe_sha256':bound.recipe_manifest('audit')['recipe_sha256'],
                     'full_weight_reconstruction_mse':error,'first_step_gradient_norms':grads,
                     'state_reload_bitwise':True,'m_dtype':str(layer.m.dtype),
                     'base_dtype':str(layer.weight.dtype),'trainable_count':sum(p.numel() for p in layer.parameters() if p.requires_grad)})
    for key,value in base.state_dict().items():assert torch.equal(value,original[key])
    files=['init_loftq.py','init_data.py','init_ablations.py','audit_init_loftq.py','test_init_loftq.py']
    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files}
    return {'status':'pass','device':str(device),'torch':str(torch.__version__),'source_sha256':sources,
            'quantizer':quantization,'saturation':saturation,'rows':rows,
            'scope':'Tiny independent engineering audit; no task performance result or compressed-runtime claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.device);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'rows':len(result['rows']),'output':str(args.output)}))
