"""Normalize the historically random UP factor: a transposed NoRA-inspired variant.

Paper NoRA normalizes the down factor's columns. This experiment instead
normalizes up-factor rows across rank and retains the original zero down factor.
NoRA-off is the accepted original DoRA plus FP32 magnitude storage exactly.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from experiments import change_ablations as core
from experiments import next_ablations as previous
from experiments import nora_current as current


@dataclass(frozen=True)
class Recipe(core.Recipe):
    nora: bool = False
    nora_epsilon: float = 1e-12
    nora_factor: str = "up"
    nora_axis: int = 1


SOURCE_METHODS = MappingProxyType({"up_base": "next_base", "up_nora": "next_base"})
REGISTRY = MappingProxyType({
    method: Recipe(**asdict(previous.REGISTRY[base]), nora=method == "up_nora")
    for method, base in SOURCE_METHODS.items()
})
METHODS = tuple(REGISTRY)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def recipe_manifest(method):
    recipe = REGISTRY[method]
    payload = {
        "schema": "dora-up-nora-v1", "method": method, **asdict(recipe),
        "factor_names": "lora_A=[rank,input] down; lora_B=[output,rank] up",
        "zero_initialized_factor": "lora_A",
        "trainable_registration_order": ["m", "lora_B", "lora_A"],
        "base_method": SOURCE_METHODS[method], "control_method": "current_dora",
        "initialization": "Historical CPU up normal std1/sqrt(rank), down zero",
        "normalization": "Differentiable up-factor row L2 normalization across rank, every forward" if recipe.nora else "No factor normalization",
        "dora_formula": "m * (W + effective_up @ down) / column_norm(W + effective_up @ down), full denominator derivative, dense projection, implicit scale1",
        "magnitude_storage": "FP32, retaining FP64 for FP64 base; native initial norm then cast",
        "normalized_factor_initialization": "Random up-factor rows; zero down-factor remains unnormalized",
        "floor_scope": "No DoRA weight-norm floor; up-factor epsilon branch separately reported",
        "compatibility": "Unchanged common pretrained storage/device, absent-bias and final mixed-dtype projection handling",
        "paper_relationship": "Transposed NoRA-inspired variant, not paper NoRA: paper normalizes down columns, this recipe normalizes up rows. Under transposition of W, up rows become down columns and column-wise DoRA becomes row-wise DoRA; task and gradient geometry are not thereby equivalent.",
        "reference_paper": current.PAPER,
        "epsilon_policy": "1e-12 retained from prior local factor-normalization experiments",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "recipe_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_sha256": SOURCE_SHA256,
            "inherited_current_source_sha256": current.SOURCE_SHA256,
            "inherited_next_source_sha256": previous.SOURCE_SHA256,
            "inherited_core_source_sha256": core.SOURCE_SHA256,
            "public_source_sha256": previous.PUBLIC_SOURCE_SHA256}


class AblationLinear(previous.AblationLinear):
    def __init__(self, base, method, rank=8):
        if method not in REGISTRY:
            raise ValueError(f"method must be one of {METHODS}")
        super().__init__(base, SOURCE_METHODS[method], rank)
        self.method = method
        self.recipe = REGISTRY[method]
        self.use_nora = self.recipe.nora
        self._nora_minimum_raw_vector_norm = None
        self._nora_epsilon_vector_events = None
        self._nora_norm_evaluations = 0
        self._nora_initial_zero_vectors = None
        self._nora_initial_epsilon_vectors = None
        if self.use_nora:
            self._track_nora(initial=True)

    def _track_nora(self, initial=False):
        # Detached device tensors avoid introducing a CPU sync on every forward.
        raw = self.lora_B.detach().norm(p=2, dim=1)
        minimum = raw.amin()
        count = (raw < self.recipe.nora_epsilon).sum()
        if self._nora_minimum_raw_vector_norm is None:
            self._nora_minimum_raw_vector_norm = minimum
            self._nora_epsilon_vector_events = count
        else:
            self._nora_minimum_raw_vector_norm = torch.minimum(self._nora_minimum_raw_vector_norm.to(minimum.device), minimum)
            self._nora_epsilon_vector_events = self._nora_epsilon_vector_events.to(count.device) + count
        self._nora_norm_evaluations += 1
        if initial:
            self._nora_initial_zero_vectors = (raw == 0).sum()
            self._nora_initial_epsilon_vectors = count.clone()

    def effective_up(self):
        if self.use_nora:
            self._track_nora()
            return F.normalize(self.lora_B, p=2, dim=1, eps=self.recipe.nora_epsilon)
        return self.lora_B

    def forward(self, x):
        lora = torch.matmul(self.effective_up(), self.lora_A)
        adapted = self.weight + lora
        norm = self._weight_norm(adapted)
        norm_adapted = adapted / norm
        calc_weights = self.m * norm_adapted
        return core._project(x, calc_weights, self.bias)

    def norm_diagnostics(self):
        return {"normalization_axis": self.norm_axis,
                "zero_initialized_factor": self.zero_initialized_factor,
                "minimum_raw_norm": float(self._minimum_raw_norm),
                "clamp_activations": int(self._clamp_activations),
                "norm_evaluations": self._norm_evaluations, "norm_floor": None,
                "magnitude_dtype": str(self.m.dtype), "factor_dtype": str(self.lora_A.dtype),
                "recipe_sha256": recipe_manifest(self.method)["recipe_sha256"],
                "nora_enabled": self.use_nora, "nora_epsilon": self.recipe.nora_epsilon,
                "nora_factor": "up", "nora_axis": 1, "nora_vector_count": self.out_features,
                "nora_initial_zero_vectors": int(self._nora_initial_zero_vectors) if self.use_nora else 0,
                "nora_initial_epsilon_vectors": int(self._nora_initial_epsilon_vectors) if self.use_nora else 0,
                "nora_minimum_raw_vector_norm": float(self._nora_minimum_raw_vector_norm) if self.use_nora else None,
                "nora_epsilon_vector_events": int(self._nora_epsilon_vector_events) if self.use_nora else 0,
                "nora_norm_evaluations": self._nora_norm_evaluations}


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


parameter_groups = core.parameter_groups


def assert_no_clamp(model):
    return {name: module.assert_no_clamp() for name, module in model.named_modules()
            if isinstance(module, AblationLinear)}


def backend():
    from experiments.adapters import ADAPTER_TYPES
    return SimpleNamespace(adapter_types=(core.AblationLinear, *ADAPTER_TYPES),
                           inject_adapters=inject_adapters, recipe_manifest=recipe_manifest,
                           assert_no_clamp=assert_no_clamp)
