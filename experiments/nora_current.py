"""NoRA additions to current DoRA, with an explicit initializer control.

Eq. 7--9 of arXiv:2608.31036 normalize the DOWN factor's input columns
across rank. The current default initializes that factor to zero, so its
literal NoRA treatment starts inside epsilon. This is recorded, not repaired.
The alternate-initializer pair uses the already measured next_init recipe.
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


@dataclass(frozen=True)
class Recipe(core.Recipe):
    nora: bool = False
    nora_epsilon: float = 1e-12


SOURCE_METHODS = MappingProxyType({
    "current_dora": "next_base", "current_dora_nora": "next_base",
    "init_dora": "next_init", "init_dora_nora": "next_init",
})
REGISTRY = MappingProxyType({
    name: Recipe(**asdict(previous.REGISTRY[base]), nora=name.endswith("_nora"))
    for name, base in SOURCE_METHODS.items()
})
METHODS = tuple(REGISTRY)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PAPER = {"url": "https://arxiv.org/abs/2608.31036", "version": "2608.31036v1",
         "equations": "7--9; random down / zero up initialization described in Eq. 2 and 11",
         "cached_text_sha256": "5a513f5aafc42159bf07840c19f673122e34440090aa3dca3c747315580a1a9a",
         "epsilon_policy": "1e-12 matches prior local NoRA adapters; paper Eq.7 specifies a small stability constant without this exact value"}


def recipe_manifest(method):
    recipe = REGISTRY[method]
    payload = {
        "schema": "dora-current-nora-v1", "method": method, **asdict(recipe),
        "factor_names": "lora_A=[rank,input] down; lora_B=[output,rank] up",
        "zero_initialized_factor": "lora_B" if recipe.published_init else "lora_A",
        "trainable_registration_order": ["lora_A", "lora_B", "m"] if recipe.published_init else ["m", "lora_B", "lora_A"],
        "base_method": SOURCE_METHODS[method],
        "initialization": "Device down Kaiming-uniform, up zero; existing alternate initializer" if recipe.published_init else "Historical CPU up normal std1/sqrt(rank), down zero",
        "normalization": "Differentiable down-factor column L2 normalization across rank, every forward" if recipe.nora else "No down-factor normalization",
        "dora_formula": "m * (W + up @ effective_down) / column_norm(W + up @ effective_down), full denominator derivative, dense projection, implicit scale1",
        "magnitude_storage": "FP32, retaining FP64 for FP64 base; native initial norm then cast",
        "zero_down_scope": "Literal current initializer begins inside epsilon; derivative at zero is I/epsilon; not the paper's random-down initializer" if recipe.nora and not recipe.published_init else "No normalized zero-down initializer",
        "floor_scope": "No DoRA weight-norm floor; NoRA epsilon branch is separate and reported",
        "compatibility": "Unchanged common pretrained storage/device, absent-bias and final mixed-dtype projection handling",
        "paper": PAPER,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "recipe_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_sha256": SOURCE_SHA256,
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
        self._nora_minimum_raw_column_norm = None
        self._nora_epsilon_column_events = None
        self._nora_norm_evaluations = 0
        self._nora_initial_zero_columns = None
        self._nora_initial_epsilon_columns = None
        if self.use_nora:
            self._track_nora(initial=True)

    def _track_nora(self, initial=False):
        # Detached diagnostics stay on the same device; no host synchronization
        # is added to training. Epsilon use here is not a DoRA norm-floor event.
        raw = self.lora_A.detach().norm(p=2, dim=0)
        minimum = raw.amin()
        count = (raw < self.recipe.nora_epsilon).sum()
        if self._nora_minimum_raw_column_norm is None:
            self._nora_minimum_raw_column_norm = minimum
            self._nora_epsilon_column_events = count
        else:
            self._nora_minimum_raw_column_norm = torch.minimum(self._nora_minimum_raw_column_norm.to(minimum.device), minimum)
            self._nora_epsilon_column_events = self._nora_epsilon_column_events.to(count.device) + count
        self._nora_norm_evaluations += 1
        if initial:
            self._nora_initial_zero_columns = (raw == 0).sum()
            self._nora_initial_epsilon_columns = count.clone()

    def effective_down(self):
        if self.use_nora:
            self._track_nora()
            return F.normalize(self.lora_A, p=2, dim=0, eps=self.recipe.nora_epsilon)
        return self.lora_A

    def forward(self, x):
        lora = torch.matmul(self.lora_B, self.effective_down())
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
                "nora_initial_zero_columns": int(self._nora_initial_zero_columns) if self.use_nora else 0,
                "nora_initial_epsilon_columns": int(self._nora_initial_epsilon_columns) if self.use_nora else 0,
                "nora_minimum_raw_column_norm": float(self._nora_minimum_raw_column_norm) if self.use_nora else None,
                "nora_epsilon_column_events": int(self._nora_epsilon_column_events) if self.use_nora else 0,
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
