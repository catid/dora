"""Supplemental BF16 factor-storage treatments; the six-change core is frozen.

Draw the same FP32 factors using each core endpoint's initialization policy,
then round/store only those factor parameters in BF16 before optimizer creation.
This changes factor/gradient/AdamW storage and arithmetic implied by those dtypes.
The recipe's explicit denominator-precision policy and magnitude dtype remain.
Native-BF16 random draws and the revised standalone initializer are not tested.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from experiments import change_ablations as core


SUPPLEMENT_METHODS = ('storage_original_bf16', 'storage_revised_bf16')
METHODS = ('ab_original', 'ab_revised', *SUPPLEMENT_METHODS)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BASE_RECIPES = {'ab_original': 'ab_original', 'ab_revised': 'ab_revised',
                'storage_original_bf16': 'ab_original', 'storage_revised_bf16': 'ab_revised'}


def recipe_manifest(method):
    base = BASE_RECIPES[method]
    payload = {'schema': 'dora-factor-storage-supplement-v1', 'method': method,
               'base_recipe': core.recipe_manifest(base),
               'factor_storage': 'bfloat16' if method in SUPPLEMENT_METHODS else 'float32',
               'initialization': 'Unchanged base-recipe FP32 draws followed by factor-only BF16 storage conversion' if method in SUPPLEMENT_METHODS else 'Unchanged core control',
               'scope': 'Factor, gradient and AdamW-state storage plus arithmetic implied by dtype; explicit norm/magnitude policies retained. Not native-BF16 RNG or a complete standalone endpoint.'}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return {**payload, 'recipe_sha256': hashlib.sha256(encoded).hexdigest(), 'source_sha256': SOURCE_SHA256}


def inject_adapters(model, method, rank=8, targets=None):
    result = core.inject_adapters(model, BASE_RECIPES[method], rank, targets)
    if method in SUPPLEMENT_METHODS:
        for module in result.modules():
            if isinstance(module, core.AblationLinear):
                if module.weight.dtype != torch.bfloat16:
                    raise ValueError('Supplemental storage experiment requires BF16 frozen weights')
                # Reassign existing parameter names so their registration order
                # remains the core endpoint's order; casts consume no RNG.
                module.lora_A = nn.Parameter(module.lora_A.detach().to(torch.bfloat16))
                module.lora_B = nn.Parameter(module.lora_B.detach().to(torch.bfloat16))
                module.factor_storage_supplement = method
    return result


def assert_no_clamp(model):
    result = core.assert_no_clamp(model)
    for name, module in model.named_modules():
        if isinstance(module, core.AblationLinear):
            if hasattr(module, 'factor_storage_supplement'):
                result[name]['factor_storage_supplement'] = module.factor_storage_supplement
                result[name]['supplement_recipe_sha256'] = recipe_manifest(module.factor_storage_supplement)['recipe_sha256']
                if module.lora_A.dtype != torch.bfloat16 or module.lora_B.dtype != torch.bfloat16:
                    raise RuntimeError('Factor storage treatment changed')
    return result


parameter_groups = core.parameter_groups
