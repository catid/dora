"""Explicit single-change and cumulative recipes between measured DoRA endpoints.

Historical factors keep their actual CPU RNG draw and registration sequence,
but use conventional names: lora_A is down and lora_B is up. The precision
flags are independent: magnitude_fp32 casts an already computed initial norm;
norm_fp32 changes initial/denominator norm arithmetic, not the dense numerator.
Factorization also changes floating-point association and projection precision.

The revised endpoint's tiny norm floor is a separate engineering policy, not a
seventh swept algorithm change. Quality runners must call assert_no_clamp()
after evaluation/training: a run where this policy activates is not an accepted
six-change ablation. Diagnostics remain on-device until requested.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Recipe:
    axis: int = 0
    detach_norm: bool = False
    magnitude_fp32: bool = False
    norm_fp32: bool = False
    published_init: bool = False
    factorized: bool = False
    clamp_norm: bool = False


_original = Recipe()
_axis = replace(_original, axis=1)
_axis_detach = replace(_axis, detach_norm=True)
_magnitude = replace(_axis_detach, magnitude_fp32=True)
_norm = replace(_magnitude, norm_fp32=True)
_init = replace(_norm, published_init=True)
REGISTRY = MappingProxyType({
    "ab_original": _original,
    "ab_axis": _axis,
    "ab_detach": replace(_original, detach_norm=True),
    "ab_m_fp32": replace(_original, magnitude_fp32=True),
    "ab_norm_fp32": replace(_original, norm_fp32=True),
    "ab_init": replace(_original, published_init=True),
    "ab_factorized": replace(_original, factorized=True),
    "cum_axis_detach": _axis_detach,
    "cum_axis_detach_m_fp32": _magnitude,
    "cum_axis_detach_m_fp32_norm_fp32": _norm,
    "cum_axis_detach_m_fp32_norm_fp32_init": _init,
    "ab_revised": replace(_init, factorized=True, clamp_norm=True),
})
METHODS = tuple(REGISTRY)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def recipe_manifest(method):
    recipe = REGISTRY[method]
    payload = {"schema": "dora-change-ablation-v1", "method": method,
               **asdict(recipe),
               "factor_names": "lora_A=[rank,input] down; lora_B=[output,rank] up",
               "zero_initialized_factor": "lora_B" if recipe.published_init else "lora_A",
               "trainable_registration_order": ["lora_A", "lora_B", "m"] if recipe.published_init else ["m", "lora_B", "lora_A"],
               "norm_precision_scope": "FP32 (FP64 retained) initial norm and independently computed denominator; ambient dense numerator unchanged" if recipe.norm_fp32 else "Ambient native initial norm and denominator",
               "floor_scope": "Separate engineering guard; quality runs rejected if active" if recipe.clamp_norm else "No clamp"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "recipe_sha256": hashlib.sha256(encoded).hexdigest(), "source_sha256": SOURCE_SHA256}


def _project(x, weight, bias):
    mismatch = x.dtype != weight.dtype or (bias is not None and bias.dtype != weight.dtype)
    if mismatch and not torch.is_autocast_enabled(x.device.type):
        projected_bias = None if bias is None else bias.to(weight.dtype)
        return F.linear(x.to(weight.dtype), weight, projected_bias).to(x.dtype)
    return F.linear(x, weight, bias)


class AblationLinear(nn.Module):
    use_dora = True
    use_nora = False

    def __init__(self, base, method, rank=8):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be an nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not base.weight.is_floating_point():
            raise ValueError("base weight must be floating point")
        self.method = method
        self.recipe = REGISTRY[method]
        self.norm_axis = self.recipe.axis
        self.in_features, self.out_features, self.rank = base.in_features, base.out_features, rank
        self.weight = base.weight
        self.weight.requires_grad_(False)
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self._minimum_raw_norm = None
        self._clamp_activations = None
        self._norm_evaluations = 0
        self.zero_initialized_factor = "lora_B" if self.recipe.published_init else "lora_A"
        if self.recipe.published_init:
            # Match the published task class's dtype, RNG device and parameter
            # registration order (important for gradient clipping reductions).
            dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
            factory = {"device": self.weight.device, "dtype": dtype}
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, **factory))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.m = nn.Parameter(self._initial_magnitude())
        else:
            # Historical order is magnitude, up, down, despite renamed factors.
            self.m = nn.Parameter(self._initial_magnitude())
            std_dev = 1 / torch.sqrt(torch.tensor(rank, device="cpu").float())
            up = torch.randn(self.out_features, rank, device="cpu") * std_dev
            down = torch.zeros(rank, self.in_features, device="cpu")
            self.lora_B = nn.Parameter(up.to(device=self.weight.device))
            self.lora_A = nn.Parameter(down.to(device=self.weight.device))
        self.register_parameter("log_gain", None)
        self.train(base.training)

    def _track_norm(self, raw):
        # Plain tensor attributes deliberately stay outside state_dict and frozen
        # weight digests. Move once if a CPU-initialized adapter moves to CUDA.
        minimum = raw.detach().amin()
        count = (raw.detach() < torch.finfo(self.weight.dtype).tiny).sum() if self.recipe.clamp_norm else torch.zeros((), device=raw.device, dtype=torch.int64)
        if self._minimum_raw_norm is None:
            self._minimum_raw_norm = minimum
            self._clamp_activations = count
        else:
            self._minimum_raw_norm = torch.minimum(self._minimum_raw_norm.to(minimum.device), minimum)
            self._clamp_activations = self._clamp_activations.to(count.device) + count
        self._norm_evaluations += 1

    def _norm(self, weight):
        if self.recipe.norm_fp32:
            if weight.dtype in (torch.float16, torch.bfloat16):
                weight = weight.float()
            result = torch.linalg.vector_norm(weight, dim=self.norm_axis, keepdim=True)
        else:
            result = weight.norm(p=2, dim=self.norm_axis, keepdim=True)
        self._track_norm(result)
        if self.recipe.clamp_norm:
            result = result.clamp_min(torch.finfo(self.weight.dtype).tiny)
        return result

    def _initial_magnitude(self):
        with torch.no_grad():
            scope = torch.autocast(device_type=self.weight.device.type, enabled=False) if self.recipe.norm_fp32 else nullcontext()
            with scope:
                magnitude = self._norm(self.weight)
            # Compute first, cast second: storage promotion alone must preserve
            # the BF16-rounded native initial norm instead of recomputing it.
            dtype = (torch.float64 if self.weight.dtype == torch.float64 else torch.float32) if self.recipe.magnitude_fp32 else self.weight.dtype
            return magnitude.to(dtype)

    def _weight_norm(self, adapted=None):
        grad_scope = torch.no_grad() if self.recipe.detach_norm else nullcontext()
        with grad_scope:
            if self.recipe.norm_fp32:
                with torch.autocast(device_type=self.weight.device.type, enabled=False):
                    dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
                    denominator_weight = self.lora_B.to(dtype) @ self.lora_A.to(dtype)
                    denominator_weight.add_(self.weight)
                    return self._norm(denominator_weight)
            if adapted is None:
                adapted = self.weight + torch.matmul(self.lora_B, self.lora_A)
            return self._norm(adapted)

    def effective_down(self):
        return self.lora_A

    def forward(self, x):
        if not self.recipe.factorized:
            # Keep the original numerator evaluation, operation order, ambient
            # autocast and explicit mixed-dtype projection boundary unchanged.
            lora = torch.matmul(self.lora_B, self.lora_A)
            adapted = self.weight + lora
            norm = self._weight_norm(adapted)
            norm_adapted = adapted / norm
            calc_weights = self.m * norm_adapted
            return _project(x, calc_weights, self.bias)
        if self.norm_axis == 1:
            # Exact task AdapterLinear order, including BF16 branch arithmetic.
            base_output = F.linear(x, self.weight)
            adapter_input = x if torch.is_autocast_enabled(x.device.type) else x.to(self.lora_A.dtype)
            update = F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
            output = base_output + update
            scale = (self.m / self._weight_norm()).flatten()
            output = output * scale
            if self.bias is not None:
                output = output + self.bias
            return output.to(base_output.dtype)
        # Input-column normalization scales the input. This is algebraically
        # equivalent in real arithmetic, but deliberately exposes changed
        # association and reduced-precision evaluation as the factorization step.
        scaled_input = x * (self.m / self._weight_norm()).flatten()
        autocast = torch.is_autocast_enabled(x.device.type)
        base_weight = self.weight if autocast else self.weight.to(scaled_input.dtype)
        base_output = F.linear(scaled_input, base_weight)
        adapter_input = scaled_input if autocast else scaled_input.to(self.lora_A.dtype)
        output = base_output + F.linear(F.linear(adapter_input, self.lora_A), self.lora_B)
        if self.bias is not None:
            output = output + self.bias
        return output.to(base_output.dtype if autocast else x.dtype)

    def norm_diagnostics(self):
        return {"normalization_axis": self.norm_axis,
                "zero_initialized_factor": self.zero_initialized_factor,
                "minimum_raw_norm": float(self._minimum_raw_norm),
                "clamp_activations": int(self._clamp_activations),
                "norm_evaluations": self._norm_evaluations,
                "norm_floor": torch.finfo(self.weight.dtype).tiny if self.recipe.clamp_norm else None,
                "magnitude_dtype": str(self.m.dtype), "factor_dtype": str(self.lora_A.dtype),
                "recipe_sha256": recipe_manifest(self.method)["recipe_sha256"]}

    def assert_no_clamp(self):
        diagnostics = self.norm_diagnostics()
        if diagnostics["clamp_activations"]:
            raise RuntimeError("Norm floor activated: reject this run from the six-change quality ablation")
        return diagnostics


def inject_adapters(model, method, rank=8, targets=None):
    if method not in REGISTRY:
        raise ValueError(f"method must be one of {METHODS}")
    if isinstance(targets, str):
        targets = (targets,)
    if targets is None:
        matches = lambda name, module: True
    elif callable(targets):
        matches = targets
    else:
        suffixes = tuple(targets)
        matches = lambda name, module: any(name == item or name.endswith("." + item) for item in suffixes)
    selected = {id(module) for name, module in model.named_modules(remove_duplicate=False)
                if isinstance(module, nn.Linear) and matches(name, module)}
    if not selected:
        raise ValueError("no nn.Linear modules matched adapter targets")
    model.requires_grad_(False)
    replacements = {}

    def rewrite(module):
        if id(module) in replacements:
            return replacements[id(module)]
        result = AblationLinear(module, method, rank) if id(module) in selected else module
        replacements[id(module)] = result
        if result is module:
            for name, child in list(module._modules.items()):
                if child is not None:
                    setattr(module, name, rewrite(child))
        return result

    return rewrite(model)


def parameter_groups(model, lr, magnitude_lr_multiplier=1.0, weight_decay=0.01):
    """Second-round factor decay and zero magnitude decay, retaining factor order."""
    factors, magnitudes = [], []
    factor_names, magnitude_names = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.rsplit(".", 1)[-1] in ("lora_A", "lora_B"):
            factors.append(parameter)
            factor_names.append(name)
        elif name.rsplit(".", 1)[-1] == "m":
            magnitudes.append(parameter)
            magnitude_names.append(name)
    result = []
    for group_name, parameters, names, decay, rate in (
        ("adapter_factors", factors, factor_names, weight_decay, lr),
        ("magnitudes", magnitudes, magnitude_names, 0.0, lr * magnitude_lr_multiplier),
    ):
        if parameters:
            result.append({"group_name": group_name, "params": parameters, "param_names": names,
                           "lr": rate, "weight_decay": decay,
                           "parameter_count": sum(p.numel() for p in parameters)})
    return result


def assert_no_clamp(model):
    return {name: module.assert_no_clamp() for name, module in model.named_modules()
            if isinstance(module, AblationLinear)}
