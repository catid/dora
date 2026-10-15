"""Audited low-rank branches. v2 is retained unchanged as historical evidence.

All trainable coordinates are FP32 unless explicitly cast for numerical tests.
Only diagnostic methods materialize a dense update. Gates are ephemeral tensors,
not Parameters, and therefore cannot enter AdamW or gradient clipping.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext

import torch
from torch import nn


class AdapterLinear(nn.Module):
    def __init__(self, base, rank, init="L", scale=1., seed=0, score="none"):
        super().__init__()
        if rank < 0 or scale <= 0 or init not in {"L", "I", "T0", "T1", "ADALORA"}:
            raise ValueError("invalid adapter configuration")
        self.base = base
        self.base.requires_grad_(False)
        self.init, self.score = init, score
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))
        self.register_buffer("mask", torch.ones(rank, dtype=torch.bool))
        self.register_buffer("slot_ids", torch.arange(rank))
        g = torch.Generator().manual_seed(seed)
        d, n = base.out_features, base.in_features
        if init == "L":
            a = torch.empty(rank, n).uniform_(-1 / math.sqrt(n), 1 / math.sqrt(n), generator=g)
            self.A = nn.Parameter(a)
            self.B = nn.Parameter(torch.zeros(d, rank))
            # DISP needs the frozen origin; other L controls store none.
            if score == "DISP":
                self.register_buffer("A0", a.clone())
        elif init == "I":
            self.register_buffer("A0", torch.randn(rank, n, generator=g) * .5)
            self.register_buffer("B0", torch.randn(d, rank, generator=g) * .5)
            self.U = nn.Parameter(torch.zeros(rank, n))
            self.V = nn.Parameter(torch.zeros(d, rank))
        elif init in {"T0", "T1"}:
            self.register_buffer("P0", torch.randn(d, rank, generator=g) * .5)
            self.register_buffer("Q0", torch.randn(rank, n, generator=g) * .5)
            self.register_buffer("lambda0", torch.full((rank,), float(init == "T1")))
            self.dP = nn.Parameter(torch.zeros(d, rank))
            self.dQ = nn.Parameter(torch.zeros(rank, n))
            self.dlambda = nn.Parameter(torch.zeros(rank))
        else:
            # Names deliberately match the vendored original RankAllocator.
            self.lora_A = nn.Parameter(torch.randn(rank, n, generator=g) * .02)
            self.lora_B = nn.Parameter(torch.randn(d, rank, generator=g) * .02)
            self.lora_E = nn.Parameter(torch.zeros(rank, 1))
            self.register_buffer("ranknum", torch.tensor([float(rank)]))
        self.tracking = False
        self._gate = None
        self._act_sum = None
        self._act_count = 0
        self._sample_indices = None

    @property
    def rank(self):
        return self.mask.numel()

    @property
    def multiplier(self):
        if self.init == "ADALORA":
            return self.scale / (self.ranknum + 1e-5)
        return self.scale

    @property
    def slot_cost(self):
        return self.base.in_features + self.base.out_features + int(self.init in {"T0", "T1", "ADALORA"})

    def axes(self):
        return {"A": 0, "B": 1, "A0": 0, "B0": 1, "U": 0, "V": 1,
                "P0": 1, "Q0": 0, "lambda0": 0, "dP": 1, "dQ": 0,
                "dlambda": 0, "mask": 0, "slot_ids": 0,
                "lora_A": 0, "lora_B": 1, "lora_E": 0}

    def begin_update(self, tracking=True):
        self.tracking = tracking
        p = next(p for n, p in self.named_parameters() if not n.startswith("base."))
        self._gate = (torch.ones(self.rank, dtype=p.dtype, device=p.device, requires_grad=True)
                      if tracking and self.score == "GATE" else None)
        self._act_sum, self._act_count, self._sample_indices = None, 0, None

    def factors(self):
        if self.init == "L":
            return self.A, self.B
        if self.init == "I":
            return self.A0 + self.U, self.B0 + self.V
        if self.init in {"T0", "T1"}:
            return self.Q0 + self.dQ, self.P0 + self.dP
        return self.lora_A, self.lora_B

    def branch(self, x, gate=None):
        # Disable autocast for the entire small branch, preserving displacements.
        device_type = x.device.type
        with torch.autocast(device_type, enabled=False) if device_type in {"cpu", "cuda"} else nullcontext():
            dtype = next(p.dtype for n, p in self.named_parameters() if not n.startswith("base."))
            x = x.to(dtype)
            m = self.mask.to(dtype)
            if gate is not None:
                m = m * gate
            if self.init == "L":
                y = ((x @ self.A.T) * m) @ self.B.T
            elif self.init == "I":
                u, a = x @ self.U.T, x @ self.A0.T
                y = (u * m) @ self.B0.T + ((a + u) * m) @ self.V.T
            elif self.init in {"T0", "T1"}:
                q, u = x @ self.Q0.T, x @ self.dQ.T
                # Group the seven expansion terms into two low-rank products.
                y = ((self.lambda0 * u + self.dlambda * (q + u)) * m) @ self.P0.T
                y = y + (((self.lambda0 + self.dlambda) * (q + u)) * m) @ self.dP.T
            else:
                # Reference masks E in-place at events. No hidden persistent mask.
                y = ((x @ self.lora_A.T) * self.lora_E[:, 0]) @ self.lora_B.T
            return y * self.multiplier.to(dtype)

    def forward(self, x):
        base = self.base(x.to(self.base.weight.dtype))
        ad = self.branch(x, self._gate if self.tracking else None)
        if self.tracking and self.score == "ACT" and self._sample_indices is not None:
            with torch.no_grad():
                flat = x.detach().reshape(-1, x.shape[-1])[self._sample_indices]
                value = self.activation_energy(flat)
                self._act_sum = value if self._act_sum is None else self._act_sum + value
                self._act_count += flat.shape[0]
        return base + ad.to(base.dtype)

    def activation_energy(self, x):
        """Sum energy over sampled tokens using Gram scalars, never T×d×r."""
        dtype = self.factors()[0].dtype
        x = x.to(dtype)
        if self.init == "L":
            z = x @ self.A.T
            value = z.square().sum(0) * self.B.square().sum(0)
        elif self.init == "I":
            u, a = x @ self.U.T, x @ self.A0.T
            value = (u.square().sum(0) * self.B0.square().sum(0)
                     + (a + u).square().sum(0) * self.V.square().sum(0)
                     + 2 * (u * (a + u)).sum(0) * (self.B0 * self.V).sum(0))
        else:
            raise ValueError("ACT registered only for two-factor methods")
        return value.clamp_min(0) * self.scale.to(dtype).square() * self.mask

    def task_score(self, supervised_tokens=1):
        """Read normalized pre-clip factor gradients; gates normalized separately."""
        with torch.no_grad():
            if self.score == "GATE":
                if self._gate is None or self._gate.grad is None:
                    raise RuntimeError("missing gate gradient")
                return (self._gate.grad / supervised_tokens).abs()
            if self.score == "ACT":
                if not self._act_count:
                    raise RuntimeError("no valid ACT samples")
                return self._act_sum / self._act_count
            if self.score in {"RAW", "DISP"}:
                a, b = self.factors()
                ga, gb = (self.A.grad, self.B.grad) if self.init == "L" else (self.U.grad, self.V.grad)
                if self.score == "DISP":
                    a, b = (self.A - self.A0, self.B) if self.init == "L" else (self.U, self.V)
                return (a * ga).abs().mean(1) + (b * gb).abs().mean(0)
        raise ValueError("no task score configured")

    def penalty(self):
        if self.init not in {"T0", "T1"}:
            raise ValueError("custom penalty requires triplets")
        q, p = self.factors()
        p, q = p[:, self.mask], q[self.mask]
        r = q.shape[0]
        if r < 2:
            return (p.sum() + q.sum()) * 0
        p = p / p.norm(dim=0, keepdim=True).clamp_min(1e-8)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
        def off(x):
            return x - torch.diag_embed(torch.diagonal(x))
        return (off(p.T @ p).square().sum() + off(q @ q.T).square().sum()) / (2*r*(r-1))

    @torch.no_grad()
    def set_mask(self, mask, optimizer=None):
        mask = mask.to(device=self.mask.device, dtype=torch.bool)
        if mask.shape != self.mask.shape:
            raise ValueError("mask shape mismatch")
        if self.init != "ADALORA" and bool((mask & ~self.mask).any()):
            raise ValueError("custom pruning must be monotonic")
        self.mask.copy_(mask)
        if self.init == "ADALORA":
            self.lora_E[~mask] = 0
        elif optimizer is not None:
            # Clear moments BEFORE step at an allocation event.
            self.freeze_pruned(optimizer)

    @torch.no_grad()
    def freeze_pruned(self, optimizer):
        if self.init == "ADALORA":
            return
        for name, p in self.named_parameters(recurse=False):
            shape = [1] * p.ndim
            shape[self.axes()[name]] = self.rank
            m = self.mask.reshape(shape)
            if p.grad is not None:
                p.grad.mul_(m)
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = optimizer.state.get(p, {}).get(key)
                if value is not None:
                    value.mul_(m)

    def thin_factors(self):
        """C,D such that the actual update on unchanged W0 equals C@D."""
        m = self.mask
        if self.init == "L":
            c, d = self.B[:, m], self.A[m]
        elif self.init == "I":
            c = torch.cat((self.B0[:, m], self.V[:, m]), 1)
            d = torch.cat((self.U[m], (self.A0 + self.U)[m]), 0)
        elif self.init in {"T0", "T1"}:
            # Cancellation-free factors for the same grouped seven-term expansion.
            q = self.Q0 + self.dQ
            c = torch.cat((self.P0[:, m], self.dP[:, m]), 1)
            d = torch.cat(((self.lambda0[:, None]*self.dQ + self.dlambda[:, None]*q)[m],
                           ((self.lambda0 + self.dlambda)[:, None]*q)[m]), 0)
        else:
            c, d = self.lora_B, self.lora_E * self.lora_A
        return c * self.multiplier.to(c.dtype), d


def adapted(model):
    return {name: m for name, m in model.named_modules() if isinstance(m, AdapterLinear)}


def inject(model, rank, init, scale, seed, score="none", targets=("q_proj", "o_proj"), pattern=None):
    model.requires_grad_(False)
    targets_found = [(name, m) for name, m in model.named_modules()
                     if isinstance(m, nn.Linear) and name.split(".")[-1] in targets]
    if not targets_found:
        raise ValueError("no target modules")
    for name, base in targets_found:
        parent, _, attr = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        stable = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        r = rank if pattern is None else pattern[name]
        layer = AdapterLinear(base, r, init, scale, (seed + stable) % (2**63), score).to(base.weight.device)
        setattr(owner, attr, layer)
    return model


def compact(model, optimizer):
    """Single-process rebuild; callers must unwrap/rebuild DDP around this.

    Survivor parameters, frozen origins, stable IDs, scale and Adam states are
    sliced together. Reference AdaLoRA retains upstream dormant storage.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise ValueError("unwrap DDP and rebuild it after compaction")
    old_params = dict(model.named_parameters())
    selections = {}
    for name, m in adapted(model).items():
        if m.init == "ADALORA":
            continue
        keep = torch.nonzero(m.mask).flatten()
        new = AdapterLinear(m.base, len(keep), m.init, float(m.scale), score=m.score).to(m.mask.device)
        with torch.no_grad():
            for key, value in m.state_dict().items():
                if key.startswith("base."):
                    continue
                if key in m.axes():
                    value = value.index_select(m.axes()[key], keep)
                value = value.detach().clone()
                if key in new._parameters:
                    setattr(new, key, nn.Parameter(value))
                else:
                    setattr(new, key, value)
        for pn, _ in m.named_parameters(recurse=False):
            selections[f"{name}.{pn}" if name else pn] = (m.axes()[pn], keep)
        parent, _, attr = name.rpartition(".")
        if not name:
            raise ValueError("compact a containing model, not a root AdapterLinear")
        setattr(model.get_submodule(parent) if parent else model, attr, new)
    new_params = dict(model.named_parameters())
    reverse = {id(p): name for name, p in old_params.items()}
    groups = [{**{k: v for k, v in g.items() if k != "params"},
               "params": [new_params[reverse[id(p)]] for p in g["params"]]}
              for g in optimizer.param_groups]
    rebuilt = type(optimizer)(groups)
    for name, old in old_params.items():
        if old not in optimizer.state:
            continue
        state = {}
        for key, value in optimizer.state[old].items():
            if torch.is_tensor(value):
                if value.shape == old.shape and name in selections:
                    axis, keep = selections[name]
                    value = value.index_select(axis, keep)
                value = value.clone()
            state[key] = value
        rebuilt.state[new_params[name]] = state
    return rebuilt
