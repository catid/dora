"""Bounded LoRA initializer supplement, retaining the measured LoRA forward.

The historical treatment changes the initialization bundle: random factor
orientation/distribution, CPU draw device, zero factor and registration order.
It adds neither DoRA normalization nor a dense adapter projection.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from experiments import adapters
from experiments import change_ablations as core


SUPPLEMENT_METHODS = ('lora_conventional', 'lora_historical_init')
METHODS = ('ab_original', 'ab_revised', *SUPPLEMENT_METHODS)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class HistoricalLoRALinear(adapters.AdapterLinear):
    """Same factorized LoRA forward, with the literal historical factor draws."""

    def __init__(self, base, rank=8):
        nn.Module.__init__(self)
        if not isinstance(base, nn.Linear):
            raise TypeError('base must be an nn.Linear')
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError('rank must be a positive integer')
        if not base.weight.is_floating_point():
            raise ValueError('base weight must have a floating-point dtype')
        self.method, self.use_dora, self.use_nora = 'lora', False, False
        self.in_features, self.out_features, self.rank = base.in_features, base.out_features, rank
        self.weight = base.weight
        self.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        std_dev = 1 / torch.sqrt(torch.tensor(rank, device='cpu').float())
        up = torch.randn(self.out_features, rank, device='cpu') * std_dev
        down = torch.zeros(rank, self.in_features, device='cpu')
        self.lora_B = nn.Parameter(up.to(device=self.weight.device))
        self.lora_A = nn.Parameter(down.to(device=self.weight.device))
        self.register_parameter('m', None)
        self.train(base.training)


def recipe_manifest(method):
    if method in ('ab_original', 'ab_revised'):
        return core.recipe_manifest(method)
    if method not in SUPPLEMENT_METHODS:
        raise ValueError(method)
    historical = method == 'lora_historical_init'
    payload = {'schema': 'lora-initializer-supplement-v1', 'method': method,
               'adapter_kind': 'lora', 'factor_storage': 'float32',
               'zero_initialized_factor': 'lora_A' if historical else 'lora_B',
               'trainable_registration_order': ['lora_B', 'lora_A'] if historical else ['lora_A', 'lora_B'],
               'forward': 'Unmodified experiments.adapters.AdapterLinear.forward, use_dora=False, use_nora=False',
               'initialization': 'CPU up normal scaled by 1/sqrt(rank), down zero; historical registration order' if historical else 'Device down Kaiming-uniform with a=sqrt(5), up zero; conventional registration order',
               'scope': 'Initializer bundle only: orientation, distribution, RNG device, zero factor and registration order. Factorized forward and effective scale 1 retained; no DoRA normalization.'}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return {**payload, 'recipe_sha256': hashlib.sha256(encoded).hexdigest(), 'source_sha256': SOURCE_SHA256}


def inject_adapters(model, method, rank=8, targets=None):
    if method in ('ab_original', 'ab_revised'):
        return core.inject_adapters(model, method, rank, targets)
    if method == 'lora_conventional':
        return adapters.inject_adapters(model, 'lora', rank, targets)
    if method != 'lora_historical_init':
        raise ValueError(method)
    if isinstance(targets, str):
        targets = (targets,)
    if targets is None:
        matches = lambda name, module: True
    elif callable(targets):
        matches = targets
    else:
        names = tuple(targets)
        matches = lambda name, module: any(name == item or name.endswith('.' + item) for item in names)
    selected = {id(module) for name, module in model.named_modules(remove_duplicate=False)
                if isinstance(module, nn.Linear) and matches(name, module)}
    if not selected:
        raise ValueError('no nn.Linear modules matched adapter targets')
    model.requires_grad_(False)
    return adapters._rewrite_modules(model, lambda module: HistoricalLoRALinear(module, rank)
                                     if id(module) in selected else module)


assert_no_clamp = core.assert_no_clamp
parameter_groups = core.parameter_groups
