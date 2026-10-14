"""Centered adapter algebra used by the adaptive-rank study."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


def _mask(mask, dtype, device):
    return mask.to(device=device, dtype=dtype).reshape(-1)


class CenteredTwoFactor(nn.Module):
    """Frozen-origin two-factor adapter with a persistent slot mask.

    A is (r,n), B is (d,r); `scale` is fixed and independent of active rank.
    The trainable representation is the displacement from the initial factors,
    while the direct factors remain available for raw scoring and export.
    """
    def __init__(self, A0: torch.Tensor, B0: torch.Tensor, scale: float = 1.0,
                 mask=None):
        super().__init__()
        if A0.ndim != 2 or B0.ndim != 2 or A0.shape[0] != B0.shape[1]:
            raise ValueError("factor shapes must be A=(r,n), B=(d,r)")
        self.register_buffer("A0", A0.detach().clone())
        self.register_buffer("B0", B0.detach().clone())
        self.A = nn.Parameter(A0.detach().clone())
        self.B = nn.Parameter(B0.detach().clone())
        self.scale = float(scale)
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if mask is None:
            mask = torch.ones(A0.shape[0], dtype=torch.bool, device=A0.device)
        if mask.shape != (A0.shape[0],):
            raise ValueError("mask shape mismatch")
        self.register_buffer("mask", mask.to(dtype=torch.bool).clone())

    @property
    def rank(self):
        return self.A.shape[0]

    @property
    def U(self):
        return self.A - self.A0

    @property
    def V(self):
        return self.B - self.B0

    def set_mask(self, mask):
        if tuple(mask.shape) != (self.rank,):
            raise ValueError("mask shape mismatch")
        self.mask.copy_(mask.to(device=self.mask.device, dtype=torch.bool))

    def delta_direct(self):
        m = _mask(self.mask, self.A.dtype, self.A.device)
        return self.scale * ((self.B * m) @ self.A -
                             (self.B0 * m) @ self.A0)

    def delta_expanded(self):
        """Evaluate BA-B0A0 without subtracting large products."""
        m = _mask(self.mask, self.A.dtype, self.A.device)
        A0 = self.A0 * m[:, None]
        U = self.U * m[:, None]
        B0 = self.B0 * m[None, :]
        V = self.V * m[None, :]
        return self.scale * (B0 @ U + V @ A0 + V @ U)

    def delta_matrix(self, expanded=True):
        return self.delta_expanded() if expanded else self.delta_direct()

    def forward(self, x, expanded=True):
        return x @ self.delta_matrix(expanded=expanded).transpose(-1, -2)

    def slot_deltas(self):
        """Return D_i matrices with shape (r,d,n), including scale and mask."""
        m = self.mask.to(dtype=self.A.dtype)
        return self.scale * m[:, None, None] * (
            self.B0.transpose(0, 1)[:, :, None] * self.U[:, None, :] +
            self.V.transpose(0, 1)[:, :, None] * self.A0[:, None, :] +
            self.V.transpose(0, 1)[:, :, None] * self.U[:, None, :])


class CenteredTriplet(nn.Module):
    """Centered P diag(lambda) Q adapter with explicit amplitudes."""
    def __init__(self, P0, Q0, lambda0=None, scale=1.0, mask=None):
        super().__init__()
        if P0.ndim != 2 or Q0.ndim != 2 or P0.shape[1] != Q0.shape[0]:
            raise ValueError("P=(d,r), Q=(r,n) shapes required")
        r = P0.shape[1]
        if lambda0 is None:
            lambda0 = torch.zeros(r, dtype=P0.dtype, device=P0.device)
        if lambda0.shape != (r,):
            raise ValueError("lambda shape mismatch")
        self.register_buffer("P0", P0.detach().clone())
        self.register_buffer("Q0", Q0.detach().clone())
        self.register_buffer("lambda0", lambda0.detach().clone())
        self.P = nn.Parameter(P0.detach().clone())
        self.Q = nn.Parameter(Q0.detach().clone())
        self.lam = nn.Parameter(lambda0.detach().clone())
        self.scale = float(scale)
        if mask is None:
            mask = torch.ones(r, dtype=torch.bool, device=P0.device)
        self.register_buffer("mask", mask.to(dtype=torch.bool).clone())

    @property
    def rank(self): return self.P.shape[1]
    @property
    def VP(self): return self.P - self.P0
    @property
    def UQ(self): return self.Q - self.Q0
    @property
    def Ul(self): return self.lam - self.lambda0

    def set_mask(self, mask): self.mask.copy_(mask.to(self.mask.device, torch.bool))

    def delta_direct(self):
        m = self.mask.to(self.P.dtype)
        return self.scale * ((self.P * (self.lam*m)[None, :]) @ self.Q -
                             (self.P0 * (self.lambda0*m)[None, :]) @ self.Q0)

    def delta_expanded(self):
        m = self.mask.to(self.P.dtype)
        p0 = self.P0 * m[None, :]
        vp = self.VP * m[None, :]
        q0 = self.Q0 * m[:, None]
        uq = self.UQ * m[:, None]
        l0 = self.lambda0 * m
        ul = self.Ul * m
        # Expansion of (P0+V)diag(l0+u)(Q0+U)-P0diag(l0)Q0.
        terms = [p0 @ (l0[:, None] * uq),
                 (vp * l0[None, :]) @ q0,
                 p0 @ (ul[:, None] * q0),
                 (vp * l0[None, :]) @ uq,
                 p0 @ (ul[:, None] * uq),
                 (vp * ul[None, :]) @ q0,
                 (vp * ul[None, :]) @ uq]
        return self.scale * sum(terms)

    def delta_matrix(self, expanded=True):
        return self.delta_expanded() if expanded else self.delta_direct()

    def slot_deltas(self):
        """Per-component centered triplet updates, shape (r,d,n)."""
        m = self.mask.to(self.P.dtype)
        return self.scale * m[:, None, None] * (
            self.P.transpose(0, 1)[:, :, None] * self.lam[:, None, None] * self.Q[:, None, :]
            - self.P0.transpose(0, 1)[:, :, None] * self.lambda0[:, None, None] * self.Q0[:, None, :]
        )

    def forward(self, x, expanded=True):
        return x @ self.delta_matrix(expanded).transpose(-1, -2)


class FaithfulAdaLoRA(nn.Module):
    """Minimal faithful AdaLoRA SVD adapter.

    This deliberately keeps the reference parameterization separate from the
    centered custom triplet: A and B are nonzero N(0,.02), E starts at zero,
    and the reference uses ``scaling / (ranknum + 1e-5)``.  ``ranknum`` is
    persistent metadata in the reference implementation; masking slots does
    not silently change that denominator.
    """
    def __init__(self, A0, B0, scale=1.0, ranknum=None, mask=None):
        super().__init__()
        if A0.ndim != 2 or B0.ndim != 2 or A0.shape[0] != B0.shape[1]:
            raise ValueError("factor shapes must be A=(r,n), B=(d,r)")
        self.register_buffer("A0", A0.detach().clone())
        self.register_buffer("B0", B0.detach().clone())
        self.A = nn.Parameter(A0.detach().clone())
        self.B = nn.Parameter(B0.detach().clone())
        self.E = nn.Parameter(torch.zeros(A0.shape[0], 1, dtype=A0.dtype, device=A0.device))
        self.scale = float(scale)
        self.register_buffer("ranknum", torch.tensor(float(A0.shape[0] if ranknum is None else ranknum), dtype=A0.dtype, device=A0.device))
        if mask is None:
            mask = torch.ones(A0.shape[0], dtype=torch.bool, device=A0.device)
        self.register_buffer("mask", mask.to(dtype=torch.bool).clone())

    @property
    def rank(self):
        return self.A.shape[0]

    @property
    def U(self):
        return self.A - self.A0

    @property
    def V(self):
        return self.B - self.B0

    def set_mask(self, mask):
        if tuple(mask.shape) != (self.rank,):
            raise ValueError("mask shape mismatch")
        mask = mask.to(device=self.mask.device, dtype=torch.bool)
        self.mask.copy_(mask)
        # Faithful AdaLoRA masks singular amplitudes in-place; the reference
        # keeps A/B but deleted E coordinates contribute no update.
        with torch.no_grad():
            self.E[~mask, 0] = 0

    @property
    def multiplier(self):
        return self.scale / (self.ranknum + 1e-5)

    def delta_direct(self):
        e = self.E[:, 0]
        return self.multiplier * ((self.B * e[None, :]) @ self.A)

    def delta_expanded(self):
        # E is the only centered/zero-start coordinate in faithful AdaLoRA;
        # this form avoids materializing a masked dense update and is exact.
        e = self.E[:, 0]
        return self.multiplier * ((self.B * e[None, :]) @ self.A)

    def delta_matrix(self, expanded=True):
        return self.delta_expanded()

    def slot_deltas(self):
        return self.multiplier * (
            self.B.transpose(0, 1)[:, :, None] * self.E[:, None, :] * self.A[:, None, :]
        )

    def forward(self, x, expanded=True):
        return x @ self.delta_matrix(expanded=expanded).transpose(-1, -2)


def rank_of_matrix(delta, tol=None):
    s = torch.linalg.svdvals(delta.double())
    if tol is None: tol = max(delta.shape) * torch.finfo(s.dtype).eps * (s.max() if s.numel() else 1)
    return int((s > tol).sum())
