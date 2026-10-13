"""Pure-LoRA IniLoRA/NoRA combinations.

This module deliberately does not use DoRA's magnitude parameter or adapted
weight normalization.  It is the small, task-runner-compatible backend for
the initializer study requested in the IniLoRA-alpha + NoRA follow-up:

    y = x (W + B A_eff - B0 A0_eff)^T + b

where ``A_eff`` is either ``A`` (centered IniLoRA-alpha) or NoRA's
differentiably column-normalized down factor.  When both factors are nonzero,
the frozen initial product is transformed by the same normalization as the
live product; otherwise the accepted one-zero initializer is retained.
"""
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


SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class AblationLinear(base.AblationLinear):
    """A centered pure-LoRA adapter with optional NoRA down normalization."""

    def __init__(self, layer, spec, rank=8, module_path='', factor_seed=42,
                 method=None, factor_initializer=None, data_spec=None,
                 nora=True):
        super().__init__(layer, spec, rank, module_path, factor_seed, method,
                         factor_initializer, data_spec)
        self.nora_enabled = bool(nora)
        self.nora_epsilon = 1e-12
        # The shared initializer parent constructs DoRA's m for historical
        # compatibility.  Remove it before the optimizer is built: this
        # backend must expose only LoRA factors as trainable parameters.
        self.m = None
        self._nora_minimum_raw_column_norm = None
        self._nora_epsilon_column_events = None
        self._nora_norm_evaluations = 0
        self._nora_initial_zero_columns = None
        self._nora_initial_epsilon_columns = None
        if self.nora_enabled:
            self._track_nora(initial=True)

    def _track_nora(self, initial=False):
        raw = self.lora_A.detach().norm(p=2, dim=0)
        minimum = raw.amin()
        count = (raw < self.nora_epsilon).sum()
        if self._nora_minimum_raw_column_norm is None:
            self._nora_minimum_raw_column_norm = minimum
            self._nora_epsilon_column_events = count
        else:
            self._nora_minimum_raw_column_norm = torch.minimum(
                self._nora_minimum_raw_column_norm.to(minimum.device), minimum)
            self._nora_epsilon_column_events = (
                self._nora_epsilon_column_events.to(count.device) + count)
        self._nora_norm_evaluations += 1
        if initial:
            self._nora_initial_zero_columns = (raw == 0).sum()
            self._nora_initial_epsilon_columns = count.clone()

    def _effective_down(self, value):
        if not self.nora_enabled:
            return value
        self._track_nora()
        # This is NoRA's differentiable normalization, not a detached
        # diagnostic or a norm floor.
        return F.normalize(value, p=2, dim=0, eps=self.nora_epsilon)

    def forward(self, x):
        live = self._effective_down(self.lora_A)
        if self.compensated:
            initial = self._effective_down(self.initial_down)
            update = self.lora_B @ live - self.initial_up @ initial
        else:
            update = self.lora_B @ live
        adapted = self.weight + update
        return core._project(x, adapted, self.bias)

    def norm_diagnostics(self):
        return {
            'normalization_axis': 0,
            'zero_initialized_factor': self.zero_initialized_factor,
            'minimum_raw_norm': float(self._nora_minimum_raw_column_norm)
                if self.nora_enabled else None,
            'clamp_activations': 0,
            'norm_evaluations': 0,
            'norm_floor': None,
            'magnitude_dtype': None,
            'factor_dtype': str(self.lora_A.dtype),
            'recipe_sha256': recipe_manifest(self.init_spec, self.method,
                                             self.data_spec,
                                             self.nora_enabled)['recipe_sha256'],
            'nora_enabled': self.nora_enabled,
            'nora_factor': 'down' if self.nora_enabled else None,
            'nora_axis': 0 if self.nora_enabled else None,
            'nora_epsilon': self.nora_epsilon if self.nora_enabled else None,
            'nora_initial_zero_columns': int(self._nora_initial_zero_columns)
                if self.nora_enabled else 0,
            'nora_initial_epsilon_columns': int(self._nora_initial_epsilon_columns)
                if self.nora_enabled else 0,
            'nora_minimum_raw_column_norm': float(self._nora_minimum_raw_column_norm)
                if self.nora_enabled else None,
            'nora_epsilon_column_events': int(self._nora_epsilon_column_events)
                if self.nora_enabled else 0,
            'nora_norm_evaluations': self._nora_norm_evaluations,
            'dora_enabled': False,
        }


def recipe_manifest(spec, method=None, data_spec=None, nora=True):
    payload = base.recipe_manifest(spec, method, data_spec)
    payload.update({
        'schema': 'pure-lora-inilora-nora-v1',
        'method': method or ('inilora_alpha_nora' if nora else 'inilora_alpha'),
        'dora_enabled': False,
        'nora_enabled': bool(nora),
        'nora_factor': 'down' if nora else None,
        'nora_axis': 0 if nora else None,
        'nora_epsilon': 1e-12 if nora else None,
        'nora_initial_product_centered': True if nora else None,
        'nora_normalizes_initial_factor': True if nora else None,
        'control_method': 'pure_lora',
        'base_method': 'pure_lora',
        'scope': ('Pure LoRA centered initialization with differentiable '
                  'NoRA down normalization' if nora else
                  'Pure LoRA centered initialization without NoRA'),
        'source_sha256': SOURCE_SHA256,
        'base_initializer_source_sha256': base.SOURCE_SHA256,
        'public_source_sha256': base.PUBLIC_SOURCE_SHA256,
        'formula': ('W + B@normalize(A) - B0@normalize(A0)'
                    if nora else 'W + B@A - B0@A0'),
        'trainable_parameter_policy': 'factors only; no DoRA magnitude parameter',
    })
    # The shared initializer parent creates DoRA's ``m`` temporarily so that
    # it can reuse the established factor-draw path.  This backend removes it
    # before optimization; keep the manifest aligned with the actual
    # trainable registration rather than inheriting DoRA's order.
    payload['trainable_registration_order'] = ['lora_B', 'lora_A']
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    payload['recipe_sha256'] = hashlib.sha256(encoded).hexdigest()
    return payload


def inject_adapters(model, spec, rank=8, targets=None, factor_seed=42,
                    method=None, factor_initializer=None, data_spec=None,
                    nora=True):
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
                                    data_spec, nora=nora)
        else:
            result = module
        replacements[id(module)] = result
        if result is module:
            for name, child in list(module._modules.items()):
                if child is not None:
                    setattr(module, name, rewrite(child))
        return result

    return rewrite(model)


def backend(spec, factor_seed=42, *, factor_initializer=None, data_spec=None,
            nora=True):
    from experiments.adapters import ADAPTER_TYPES
    diagnostics = lambda model: {
        name: module.norm_diagnostics() for name, module in
        model.named_modules() if isinstance(module, AblationLinear)
    }
    return SimpleNamespace(
        adapter_types=(AblationLinear, base.AblationLinear, *ADAPTER_TYPES),
        inject_adapters=lambda model, method, rank=8, targets=None:
            inject_adapters(model, spec, rank, targets, factor_seed, method,
                            factor_initializer, data_spec, nora=nora),
        recipe_manifest=lambda method: recipe_manifest(spec, method, data_spec,
                                                       nora=nora),
        parameter_groups=core.parameter_groups,
        assert_no_clamp=diagnostics,
        adapter_diagnostics=diagnostics,
    )
