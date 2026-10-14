"""Slot-importance estimators from the preregistered protocol."""
from __future__ import annotations
import torch


def raw_scores(adapter):
    """mean(|A grad_A|)+mean(|B grad_B|) per slot."""
    if adapter.A.grad is None or adapter.B.grad is None: raise RuntimeError("missing factor gradients")
    return (torch.mean((adapter.A.detach()*adapter.A.grad).abs(), dim=1) +
            torch.mean((adapter.B.detach()*adapter.B.grad).abs(), dim=0)).detach()


def displacement_scores(adapter):
    if adapter.A.grad is None or adapter.B.grad is None: raise RuntimeError("missing factor gradients")
    return (torch.mean((adapter.U.detach()*adapter.A.grad).abs(), dim=1) +
            torch.mean((adapter.V.detach()*adapter.B.grad).abs(), dim=0)).detach()


def adalora_scores(adapter):
    """Faithful AdaLoRA slot score before EMA/uncertainty aggregation."""
    if adapter.A.grad is None or adapter.B.grad is None or adapter.E.grad is None:
        raise RuntimeError("missing AdaLoRA factor gradients")
    return (torch.mean((adapter.A.detach()*adapter.A.grad).abs(), dim=1) +
            torch.mean((adapter.B.detach()*adapter.B.grad).abs(), dim=0) +
            (adapter.E.detach()[:, 0]*adapter.E.grad[:, 0]).abs()).detach()


def gate_scores(adapter, grad_delta):
    """Absolute Frobenius deletion sensitivity from dL/dDeltaW."""
    slots = adapter.slot_deltas().to(dtype=grad_delta.dtype)
    return (slots * grad_delta[None, :, :]).flatten(1).sum(1).abs().detach()


def activation_scores(adapter, x, valid=None):
    """Mean ||D_i x||² using low-rank identities, with FP32 accumulation."""
    x = x.reshape(-1, x.shape[-1]).to(dtype=adapter.A.dtype)
    if valid is not None: x = x[valid.reshape(-1)]
    if x.numel() == 0: return torch.zeros(adapter.rank, device=adapter.A.device)
    U, A0, V, B0 = adapter.U, adapter.A0, adapter.V, adapter.B0
    out = []
    for i in range(adapter.rank):
        u = x @ U[i]
        a = x @ A0[i]
        z = B0[:, i][None, :] * u[:, None] + V[:, i][None, :] * (a+u)[:, None]
        out.append((adapter.scale*z).float().square().sum(dim=1).mean())
    return torch.stack(out).detach()


def ema_update(state, q, beta=0.85):
    q = q.detach().float()
    if state is None: return q.clone()
    return beta*state + (1-beta)*q


def uncertainty_update(state, q, smoothed, beta=0.85):
    q, smoothed = q.detach().float(), smoothed.detach().float()
    delta = (q-smoothed).abs()
    if state is None: return (1-beta)*delta
    return beta*state + (1-beta)*delta
