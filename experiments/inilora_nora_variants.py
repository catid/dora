"""Pure-LoRA compatibility variants for the IniLoRA-alpha/NoRA study.

The original combination uses unit-norm NoRA down columns.  That discards the
per-column radial scale carried by a two-factor IniLoRA initialization.  This
module keeps the same centered pure-LoRA update while exposing a small set of
explicit alternatives:

``down_unit``
    The previously measured paper-style NoRA combination.
``down_anchor``
    Normalize down columns but multiply by their frozen initial norms.
``down_learned``
    The same anchored parameterization with trainable positive log gains.
``up_anchor``
    Normalize up rows and multiply by their frozen initial norms.
``both_anchor``
    Apply both anchored normalizations.

The anchored variants preserve the initial factor product exactly and do not
use a DoRA magnitude or adapted-weight normalization.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from experiments import init_ablations as base
from experiments import change_ablations as core


MODES = ('down_unit', 'down_anchor', 'down_learned', 'up_anchor', 'both_anchor')
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class AblationLinear(base.AblationLinear):
    """Centered pure-LoRA adapter with an explicit NoRA factor geometry."""

    def __init__(self, layer, spec, rank=8, module_path='', factor_seed=42,
                 method=None, factor_initializer=None, data_spec=None,
                 mode='down_unit'):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        # Use the shared initializer only; this class owns all normalization.
        super().__init__(layer, spec, rank, module_path, factor_seed, method,
                         factor_initializer, data_spec)
        self.m = None
        self.mode = mode
        self.nora_enabled = True
        self.nora_epsilon = 1e-12
        self._nora_norm_evaluations = 0

        use_down = mode.startswith('down') or mode == 'both_anchor'
        use_up = mode.startswith('up') or mode == 'both_anchor'
        if use_down:
            down_gain = self.initial_down.detach().norm(p=2, dim=0)
            if bool((down_gain <= self.nora_epsilon).any()):
                raise ValueError('Initial down factor contains an epsilon column')
            self.register_buffer('initial_down_gain', down_gain)
            if mode == 'down_learned':
                self.log_gain = nn.Parameter(down_gain.log())
                # Center against the exact initial value used by the live
                # exp(log_gain) path.  norm(log(norm)) round-trips can differ
                # by a few ulps; in BF16 dense projection that can trip the
                # zero-update identity check.  Keep this detached snapshot so
                # later gain updates cannot move the compensation term.
                self.register_buffer('initial_live_down_gain',
                                     self.log_gain.detach().exp())
        if use_up:
            up_gain = self.initial_up.detach().norm(p=2, dim=1)
            if bool((up_gain <= self.nora_epsilon).any()):
                raise ValueError('Initial up factor contains an epsilon row')
            self.register_buffer('initial_up_gain', up_gain)
        self._down_gain_initial = (self.initial_down.detach().norm(p=2, dim=0)
                                   if use_down else None)
        self._up_gain_initial = (self.initial_up.detach().norm(p=2, dim=1)
                                 if use_up else None)

    def _gain(self, which, value, *, initial=False):
        if which == 'down':
            if self.mode == 'down_learned' and not initial:
                return self.log_gain.exp().to(device=value.device, dtype=value.dtype)
            if self.mode == 'down_learned' and initial:
                return self.initial_live_down_gain.to(device=value.device,
                                                      dtype=value.dtype)
            return self.initial_down_gain.to(device=value.device, dtype=value.dtype)
        return self.initial_up_gain.to(device=value.device, dtype=value.dtype)

    def _down(self, value, *, initial=False):
        if self.mode in ('down_unit', 'down_anchor', 'down_learned', 'both_anchor'):
            self._nora_norm_evaluations += 1
            result = F.normalize(value, p=2, dim=0, eps=self.nora_epsilon)
            if self.mode in ('down_anchor', 'down_learned', 'both_anchor'):
                result = result * self._gain('down', value, initial=initial)
            return result
        return value

    def _up(self, value, *, initial=False):
        if self.mode in ('up_anchor', 'both_anchor'):
            self._nora_norm_evaluations += 1
            return F.normalize(value, p=2, dim=1, eps=self.nora_epsilon) * self._gain('up', value, initial=initial)[:, None]
        return value

    def forward(self, x):
        live_up = self._up(self.lora_B)
        live_down = self._down(self.lora_A)
        if self.compensated:
            initial_up = self._up(self.initial_up, initial=True)
            initial_down = self._down(self.initial_down, initial=True)
            update = live_up @ live_down - initial_up @ initial_down
        else:
            update = live_up @ live_down
        return core._project(x, self.weight + update, self.bias)

    def norm_diagnostics(self):
        if self.mode == 'down_learned':
            gain = self.log_gain.detach().exp()
            gain_range = [float(gain.min()), float(gain.max())]
            trainable_gain = True
        else:
            gain_range = None
            trainable_gain = False
        return {
            'normalization_axis': 0 if self.mode.startswith('down') or self.mode == 'both_anchor' else 1,
            'zero_initialized_factor': self.zero_initialized_factor,
            'clamp_activations': 0,
            'norm_evaluations': 0,
            'norm_floor': None,
            'magnitude_dtype': None,
            'factor_dtype': str(self.lora_A.dtype),
            'recipe_sha256': recipe_manifest(self.init_spec, self.method,
                                             self.data_spec, self.mode)['recipe_sha256'],
            'dora_enabled': False,
            'nora_enabled': True,
            'nora_mode': self.mode,
            'nora_epsilon': self.nora_epsilon,
            'nora_norm_evaluations': self._nora_norm_evaluations,
            'trainable_log_gain': trainable_gain,
            'gain_range': gain_range,
            'initial_update_is_zero': True,
        }


def recipe_manifest(spec, method=None, data_spec=None, mode='down_unit'):
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    payload = base.recipe_manifest(spec, method, data_spec)
    payload.update({
        'schema': 'pure-lora-inilora-nora-variants-v1',
        'method': method or 'inilora_alpha_nora_variant',
        'control_method': 'pure_lora',
        'base_method': 'pure_lora',
        'dora_enabled': False,
        'nora_enabled': True,
        'nora_mode': mode,
        'nora_epsilon': 1e-12,
        'nora_initial_product_centered': True,
        'nora_normalizes_initial_factor': True,
        'initial_gain_policy': ('trainable log gain initialized to initial norm'
                                if mode == 'down_learned' else
                                'frozen initial factor norm' if mode != 'down_unit'
                                else 'none'),
        'formula': {
            'down_unit': 'W + B@normalize(A) - B0@normalize(A0)',
            'down_anchor': 'W + B@(normalize(A)*||A0||) - B0@A0',
            'down_learned': 'W + B@(normalize(A)*exp(log_gain)) - B0@A0',
            'up_anchor': 'W + (normalize(B)*||B0||)@A - B0@A0',
            'both_anchor': 'W + (normalize(B)*||B0||)@(normalize(A)*||A0||) - B0@A0',
        }[mode],
        'trainable_parameter_policy': ('factors plus down log gains; no DoRA magnitude'
                                        if mode == 'down_learned' else
                                        'factors only; no DoRA magnitude'),
        'source_sha256': SOURCE_SHA256,
    })
    payload['trainable_registration_order'] = (
        ['lora_B', 'lora_A', 'log_gain'] if mode == 'down_learned'
        else ['lora_B', 'lora_A'])
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    payload['recipe_sha256'] = hashlib.sha256(encoded).hexdigest()
    return payload


def inject_adapters(model, spec, rank=8, targets=None, factor_seed=42,
                    method=None, factor_initializer=None, data_spec=None,
                    mode='down_unit'):
    if isinstance(targets, str):
        targets = (targets,)
    if targets is None:
        matches = lambda name, module: True
    elif callable(targets):
        matches = targets
    else:
        suffixes = tuple(targets)
        matches = lambda name, module: any(
            name == x or name.endswith('.' + x) for x in suffixes)
    selected = {id(module): name for name, module in
                reversed(list(model.named_modules(remove_duplicate=False)))
                if isinstance(module, nn.Linear) and matches(name, module)}
    if not selected:
        raise ValueError('no nn.Linear modules matched adapter targets')
    canonical = {id(module): name for name, module in
                 reversed(list(model.named_modules()))}
    model.requires_grad_(False)
    replacements = {}

    def rewrite(module):
        if id(module) in replacements:
            return replacements[id(module)]
        if id(module) in selected:
            result = AblationLinear(module, spec, rank, canonical[id(module)],
                                    factor_seed, method, factor_initializer,
                                    data_spec, mode=mode)
        else:
            result = module
        replacements[id(module)] = result
        if result is module:
            for name, child in list(module._modules.items()):
                if child is not None:
                    setattr(module, name, rewrite(child))
        return result

    return rewrite(model)


def parameter_groups(model, lr, magnitude_lr_multiplier=1.0, weight_decay=0.01):
    groups = core.parameter_groups(model, lr,
                                   magnitude_lr_multiplier=magnitude_lr_multiplier,
                                   weight_decay=weight_decay)
    gains = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad and name.rsplit('.', 1)[-1] == 'log_gain']
    if gains:
        groups.append({'group_name': 'nora_gains', 'params': [p for _, p in gains],
                       'param_names': [n for n, _ in gains], 'lr': lr,
                       'weight_decay': 0.0,
                       'parameter_count': sum(p.numel() for _, p in gains)})
    return groups


def backend(spec, factor_seed=42, *, factor_initializer=None, data_spec=None,
            mode='down_unit'):
    from experiments.adapters import ADAPTER_TYPES
    diagnostics = lambda model: {
        name: module.norm_diagnostics() for name, module in
        model.named_modules() if isinstance(module, AblationLinear)
    }
    return SimpleNamespace(
        adapter_types=(AblationLinear, base.AblationLinear, *ADAPTER_TYPES),
        inject_adapters=lambda model, method, rank=8, targets=None:
            inject_adapters(model, spec, rank, targets, factor_seed, method,
                            factor_initializer, data_spec, mode=mode),
        recipe_manifest=lambda method: recipe_manifest(spec, method, data_spec,
                                                       mode=mode),
        parameter_groups=parameter_groups,
        assert_no_clamp=diagnostics,
        adapter_diagnostics=diagnostics,
    )
