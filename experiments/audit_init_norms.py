"""Distinguish magnitude initialization norms from DoRA denominators in audits.

The historical minimum_raw_norm counter combines both. A zero quantized-base
column may initialize a zero magnitude while a LoftQ residual supplies a strictly
positive denominator. This helper observes tensors; it never changes a floor,
parameter, forward pass, or historical diagnostic.
"""
from contextlib import nullcontext
import math

import torch


def forward_denominator_minimum(module):
    """Recompute the accepted dense column denominator under task autocast."""
    scope = (torch.autocast('cuda', dtype=torch.bfloat16)
             if module.weight.is_cuda else nullcontext())
    with torch.no_grad(), scope:
        update = module.lora_B @ module.lora_A
        if getattr(module, 'compensated', False):
            update = update - module.initial_up @ module.initial_down
        adapted = module.weight + update
        norms = adapted.norm(p=2, dim=0, keepdim=True)
    assert torch.isfinite(norms).all() and (norms > 0).all(), 'Invalid forward denominator'
    return float(norms.min())


def audit_initial_norm(diagnostic, module, spec):
    """Validate an unmodified legacy diagnostic against a recreated initializer.

The caller must independently verify initializer/checkpoint hashes, finite
training losses and gradients, and checkpoint reloads. This records initial
denominator evidence, not a claim to have observed every past training step.
    """
    raw = diagnostic['minimum_raw_norm']
    assert math.isfinite(raw) and raw >= 0, 'Invalid combined norm diagnostic'
    evidence = {'minimum_initial_forward_denominator': forward_denominator_minimum(module),
                'legacy_minimum_raw_norm': raw,
                'legacy_scope': 'magnitude initialization and forward denominators'}
    if raw == 0:
        kind, config = spec.get('kind', 'random'), spec.get('config', spec)
        assert kind == 'quantized' and config.get('method', 'loftq') == 'loftq'
        assert config.get('magnitude_policy', 'original') == 'quantized'
        with torch.no_grad():
            base_norm = module.weight.norm(p=2, dim=0, keepdim=True)
        assert torch.isfinite(base_norm).all() and base_norm.min() == 0
        assert torch.equal(module.m, base_norm.to(module.m.dtype)), 'Magnitude initialization differs'
        evidence.update(zero_explained_by_quantized_magnitude_initialization=True,
                        zero_initial_magnitude_columns=int((base_norm == 0).sum()))
    return evidence
