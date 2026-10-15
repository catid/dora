"""Thin integration of unmodified original AdaLoRA (pinned, MIT licensed).

Keep original threshold ties, post-optimizer element EMAs, denominator offset,
and final-window masking on every step. This is not the custom allocator.
"""
from __future__ import annotations

from .adapters import adapted
from .vendor.adalora.adalora import RankAllocator, compute_orth_regu


def reference_allocator(model, initial_rank, target_rank, updates):
    return RankAllocator(model, initial_rank, target_rank,
                         init_warmup=int(.10*updates), final_warmup=int(.20*updates),
                         mask_interval=16, beta1=.85, beta2=.85, total_step=updates)


def reference_after_step(model, allocator, update):
    result = allocator.update_and_mask(model, update)
    # Mask bookkeeping reflects actual E, including reactivation between events.
    for m in adapted(model).values():
        m.mask.copy_(m.lora_E[:, 0].detach().ne(0))
    return result


def reference_penalty(model, gamma):
    return compute_orth_regu(model, regu_weight=gamma)
