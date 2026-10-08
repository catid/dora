"""Controlled LoRA/DoRA/NoRA adapters for the task experiments.

All methods use the same down-factor draw, zero up-factor, rank, and effective
scale 1 (conventional LoRA alpha=rank), with no adapter dropout. NoRA follows
equations 7--9 of https://arxiv.org/abs/2608.31036: normalize each down-factor
column across the rank dimension on every forward, retaining its derivative.
The combined method applies that update inside DoRA's weight decomposition,
matching the DoRA-style NoRA parameterization in the paper's Table 5.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from dora import DoRALayer


METHODS = ("lora", "dora", "nora", "dora_nora")


def _method_name(method):
    method = method.lower().replace("+", "_")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return method


class AdapterLinear(DoRALayer):
    """A shared implementation with conventional A=[rank,in], B=[out,rank].

    The frozen base parameters retain their original storage and dtype. Adapter
    parameters are FP32, except when a FP64 base is supplied for reference tests.
    Move/cast the base before injection; later ``model.half()`` also casts the
    adapters according to ordinary PyTorch semantics.
    """

    def __init__(self, base, method, rank=8):
        nn.Module.__init__(self)
        self.method = _method_name(method)
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not base.weight.is_floating_point():
            raise ValueError("base weight must have a floating-point dtype")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.use_dora = self.method in ("dora", "dora_nora")
        self.use_nora = self.method in ("nora", "dora_nora")
        self.weight = base.weight
        self.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        adapter_dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
        factory = {"device": self.weight.device, "dtype": adapter_dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.use_dora:
            with torch.no_grad():
                self.m = nn.Parameter(self._row_norm(self.weight).to(adapter_dtype))
        else:
            self.register_parameter("m", None)
        self.train(base.training)

    def effective_down(self):
        """Differentiable column normalization; this is not NoRA-init."""
        if self.use_nora:
            return F.normalize(self.lora_A, p=2, dim=0, eps=1e-12)
        return self.lora_A

    @torch.no_grad()
    def _weight_norm(self, down=None):
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            if down is None:
                down = self.effective_down()
            adapted = self.lora_B @ down
            adapted.add_(self.weight)
            return self._row_norm(adapted)

    def forward(self, x):
        down = self.effective_down()
        base_output = F.linear(x, self.weight)
        # Autocast handles mixed input/adapter types without allocating an
        # unnecessary FP32 copy of the activation. Explicit BF16 inputs also
        # work outside autocast by converting only the adapter branch.
        adapter_input = x if torch.is_autocast_enabled(x.device.type) else x.to(down.dtype)
        update = F.linear(F.linear(adapter_input, down), self.lora_B)
        output = base_output + update
        if self.use_dora:
            scale = (self.m / self._weight_norm(down)).flatten()
            output = output * scale
        if self.bias is not None:
            output = output + self.bias
        return output.to(base_output.dtype)

    @classmethod
    def from_linear(cls, layer, method="dora", rank=8):
        return cls(layer, method=method, rank=rank)

    @torch.no_grad()
    def to_linear(self):
        """Export an independent frozen linear layer, including NoRA's norm."""
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            adapted = self.lora_B @ self.effective_down()
            adapted.add_(self.weight)
            if self.use_dora:
                adapted = adapted * (self.m / self._row_norm(adapted))
            merged = adapted.to(self.weight.dtype)
        result = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                           device="meta", dtype=self.weight.dtype)
        result.weight = nn.Parameter(merged, requires_grad=False)
        if self.bias is not None:
            result.bias = nn.Parameter(self.bias.detach().clone(), requires_grad=False)
        return result.train(self.training)

    def extra_repr(self):
        return f"method={self.method}, " + super().extra_repr()


def _rewrite_modules(model, transform):
    replacements = {}

    def replace(module):
        if id(module) in replacements:
            return replacements[id(module)]
        result = transform(module)
        replacements[id(module)] = result
        if result is module:
            for name, child in list(module._modules.items()):
                if child is not None:
                    setattr(module, name, replace(child))
        return result

    return replace(model)


def inject_adapters(model, method, rank=8, targets=None):
    """Freeze a model, replace selected linears, and return its possibly new root.

    ``targets`` is None (all linears), a ``(name, module) -> bool`` predicate,
    or an iterable of exact/dotted-suffix module names such as ``('query',
    'value')``. Shared aliases remain shared, including when only one alias
    matches. Re-enable task heads explicitly after injection. Recreate the
    optimizer afterward. No selected modules is an error rather than a silent
    frozen-model experiment.
    """
    method = _method_name(method)
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
    selected = {
        id(module) for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, nn.Linear) and matches(name, module)
    }
    if not selected:
        raise ValueError("no nn.Linear modules matched adapter targets")
    model.requires_grad_(False)
    return _rewrite_modules(
        model, lambda module: AdapterLinear(module, method, rank) if id(module) in selected else module
    )


def merge_adapters(model):
    """Merge adapters in place and return the possibly replaced root."""
    return _rewrite_modules(model, lambda module: module.to_linear() if isinstance(module, AdapterLinear) else module)
