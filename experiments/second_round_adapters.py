"""Controlled second-round adapters and explicit optimizer parameter groups.

The two proposed mixtures are experimental variants, not claims about the
published NoRA method. ``dora_nora_mlr`` changes only magnitude learning rate;
``dora_nora_gain`` learns positive per-input gains after NoRA normalization.
Original experiment implementations and results remain unchanged.
"""

import math

import torch
from torch import nn

from experiments.adapters import AdapterLinear as FirstRoundAdapter, _rewrite_modules


METHODS = ("lora", "dora", "nora", "dora_nora", "dora_nora_mlr", "dora_nora_gain")


class AdapterLinear(FirstRoundAdapter):
    """Gain restores column amplitude freedom; MLR leaves the forward unchanged."""

    def __init__(self, base, method, rank=8):
        method = method.lower().replace("+", "_")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        inherited_method = "dora_nora" if method in ("dora_nora_mlr", "dora_nora_gain") else method
        super().__init__(base, inherited_method, rank=rank)
        self.method = method
        if method == "dora_nora_gain":
            self.log_gain = nn.Parameter(self.lora_A.new_zeros(self.in_features))
        else:
            self.register_parameter("log_gain", None)

    def effective_down(self):
        down = super().effective_down()
        if self.log_gain is not None:
            down = down * self.log_gain.exp().unsqueeze(0)
        return down


def inject_adapters(model, method, rank=8, targets=None):
    """Freeze the model, replace selected linears, and return its possibly new root.

    Targets follow the first-round API: None, exact/dotted-suffix names, or a
    predicate ``(name, module) -> bool``. Re-enable task heads separately.
    """
    method = method.lower().replace("+", "_")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if isinstance(targets, str):
        targets = (targets,)
    if targets is None:
        matches = lambda name, module: True
    elif callable(targets):
        matches = targets
    else:
        names = tuple(targets)
        matches = lambda name, module: any(name == item or name.endswith("." + item) for item in names)
    selected = {id(module) for name, module in model.named_modules(remove_duplicate=False)
                if isinstance(module, nn.Linear) and matches(name, module)}
    if not selected:
        raise ValueError("no nn.Linear modules matched adapter targets")
    model.requires_grad_(False)
    return _rewrite_modules(model, lambda module: AdapterLinear(module, method, rank=rank)
                            if id(module) in selected else module)


def merge_adapters(model):
    return _rewrite_modules(model, lambda module: module.to_linear()
                            if isinstance(module, AdapterLinear) else module)


def parameter_groups(model, lr, magnitude_lr_multiplier=1.0, weight_decay=0.01):
    """Return adapter-only optimizer groups; append task-head groups separately.

    Factors use the requested decay; magnitude and log-gain parameters never
    decay. The magnitude multiplier applies only to ``dora_nora_mlr``. Group
    metadata contains names/counts and is accepted by ordinary PyTorch optimizers.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(magnitude_lr_multiplier) or magnitude_lr_multiplier <= 0:
        raise ValueError("magnitude_lr_multiplier must be finite and positive")
    buckets = {"adapter_factors": [], "magnitudes": [], "decoupled_magnitudes": [], "input_gains": []}
    names = {name: [] for name in buckets}
    seen = set()
    for prefix, module in model.named_modules():
        if not isinstance(module, AdapterLinear):
            continue
        members = [("lora_A", "adapter_factors"), ("lora_B", "adapter_factors"),
                   ("m", "decoupled_magnitudes" if module.method == "dora_nora_mlr" else "magnitudes"),
                   ("log_gain", "input_gains")]
        for name, group in members:
            parameter = getattr(module, name)
            if parameter is not None and parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                buckets[group].append(parameter)
                names[group].append(f"{prefix}.{name}" if prefix else name)
    groups = []
    for name, parameters in buckets.items():
        if not parameters:
            continue
        groups.append({"params": parameters, "lr": lr * (magnitude_lr_multiplier if name == "decoupled_magnitudes" else 1),
                       "weight_decay": weight_decay if name == "adapter_factors" else 0.0,
                       "group_name": name, "parameter_count": sum(p.numel() for p in parameters),
                       "param_names": names[name]})
    if not groups:
        raise ValueError("model has no trainable second-round adapters")
    return groups
