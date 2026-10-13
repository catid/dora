"""Preserved revised standalone DoRALayer endpoint, with read-only norm checks."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from dora_row import DoRALayer
from experiments import adapters
from experiments import change_ablations as core

SUPPLEMENT_METHODS = ('standalone_current',)
METHODS = ('ab_original', 'ab_revised', *SUPPLEMENT_METHODS)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def recipe_manifest(method):
    if method in ('ab_original','ab_revised'):
        return core.recipe_manifest(method)
    if method != 'standalone_current':
        raise ValueError(method)
    payload={'schema':'dora-standalone-endpoint-v1','method':method,'kind':'standalone_endpoint',
             'constructor':'Preserved revised dora_row.DoRALayer.from_linear, no constructor or forward overrides',
             'implementation_sha256':hashlib.sha256(Path('dora_row.py').read_bytes()).hexdigest(),
             'factor_names':'lora_A=[output,rank] up; lora_B=[rank,input] down',
             'factor_storage':'bfloat16','magnitude_storage':'float32','axis':1,
             'zero_initialized_factor':'lora_B','trainable_registration_order':['m','lora_A','lora_B'],
             'initialization':'Native BF16 up normal with std 1/sqrt(rank) drawn on base device; down zero',
             'scope':'Complete standalone recipe. Includes native BF16 initialization/storage/arithmetic and row norm, detached FP32 denominator, factorized projection, FP32 magnitude and floor policy. Read-only norm monitoring rejects any floor activation.'}
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    return {**payload,'recipe_sha256':hashlib.sha256(encoded).hexdigest(),'source_sha256':SOURCE_SHA256}


@torch.no_grad()
def monitor_norm(module, inputs):
    with torch.autocast(device_type=module.weight.device.type,enabled=False):
        weight,a,b=module.weight,module.lora_A,module.lora_B
        if weight.dtype in (torch.float16,torch.bfloat16):
            weight,a,b=weight.float(),a.float(),b.float()
        adapted=a@b
        adapted.add_(weight)
        raw=torch.linalg.vector_norm(adapted,dim=1,keepdim=True)
        floor=torch.finfo(module.weight.dtype).tiny
        minimum=raw.min()
        count=(raw<floor).sum()
        module._standalone_min_norm=minimum if module._standalone_min_norm is None else torch.minimum(module._standalone_min_norm,minimum)
        module._standalone_floor_count=count if module._standalone_floor_count is None else module._standalone_floor_count+count
        module._standalone_norm_calls+=1


def inject_adapters(model,method,rank=8,targets=None):
    if method in ('ab_original','ab_revised'):
        return core.inject_adapters(model,method,rank,targets)
    if method != 'standalone_current':
        raise ValueError(method)
    if isinstance(targets,str):targets=(targets,)
    if targets is None:matches=lambda name,module:True
    elif callable(targets):matches=targets
    else:
        names=tuple(targets)
        matches=lambda name,module:any(name==item or name.endswith('.'+item) for item in names)
    selected={id(module) for name,module in model.named_modules(remove_duplicate=False)
              if isinstance(module,nn.Linear) and matches(name,module)}
    if not selected:raise ValueError('no nn.Linear modules matched adapter targets')
    model.requires_grad_(False)
    def convert(module):
        if id(module) not in selected:return module
        result=DoRALayer.from_linear(module,rank=rank)
        result._standalone_min_norm=None
        result._standalone_floor_count=None
        result._standalone_norm_calls=0
        # Initial zero-update norm is exactly the constructor's magnitude norm.
        monitor_norm(result,())
        result.register_forward_pre_hook(monitor_norm)
        return result
    return adapters._rewrite_modules(model,convert)


def assert_no_clamp(model):
    diagnostics=core.assert_no_clamp(model)
    for name,module in model.named_modules():
        if type(module) is not DoRALayer:continue
        assert hasattr(module,'_standalone_norm_calls')
        value={'clamp_activations':int(module._standalone_floor_count.item()),
               'minimum_raw_norm':float(module._standalone_min_norm.item()),
               'norm_evaluations':module._standalone_norm_calls,
               'factor_dtype':str(module.lora_A.dtype),'magnitude_dtype':str(module.m.dtype),
               'recipe_sha256':recipe_manifest('standalone_current')['recipe_sha256']}
        if value['clamp_activations'] or not torch.isfinite(module._standalone_min_norm):
            raise RuntimeError('Standalone norm floor used or nonfinite norm observed')
        diagnostics[name]=value
    return diagnostics


parameter_groups=core.parameter_groups
