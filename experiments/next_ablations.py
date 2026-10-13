"""Five single additions to the original-plus-FP32-magnitude default.

The previous core is frozen. Every recipe preserves its task compatibility
policy: pretrained storage/device, absent bias, and mixed-dtype projection.
Those policies are shared controls rather than new algorithm interventions.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import torch
from torch import nn

from experiments import change_ablations as core


SOURCE_METHODS = MappingProxyType({
    "next_base": "ab_original",
    "next_axis": "ab_axis",
    "next_detach": "ab_detach",
    "next_norm_fp32": "ab_norm_fp32",
    "next_init": "ab_init",
    "next_factorized": "ab_factorized",
})
REGISTRY = MappingProxyType({
    name: replace(core.REGISTRY[old], magnitude_fp32=True)
    for name, old in SOURCE_METHODS.items()
})
METHODS = tuple(REGISTRY)
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PUBLIC_SOURCE_SHA256 = hashlib.sha256((Path(__file__).resolve().parents[1] / "dora.py").read_bytes()).hexdigest()


def recipe_manifest(method):
    recipe = REGISTRY[method]
    baseline = asdict(REGISTRY["next_base"])
    payload = {
        "schema": "dora-next-ablation-v1", "method": method,
        **asdict(recipe),
        "baseline": "Original dora.py plus FP32 magnitude storage; FP64 retained",
        "changed_fields_from_baseline": [key for key, value in asdict(recipe).items()
                                         if value != baseline[key]],
        "factor_names": "lora_A=[rank,input] down; lora_B=[output,rank] up",
        "zero_initialized_factor": "lora_B" if recipe.published_init else "lora_A",
        "trainable_registration_order": ["lora_A", "lora_B", "m"] if recipe.published_init else ["m", "lora_B", "lora_A"],
        "norm_precision_scope": "FP32 initial norm and denominator outside autocast; FP64 retained; ambient dense numerator unchanged" if recipe.norm_fp32 else "Ambient native initial norm and denominator; initial norm cast to magnitude storage afterward",
        "magnitude_dtype": "FP32, retaining FP64 for FP64 base",
        "floor_scope": "No clamp",
        "compatibility": "All recipes retain pretrained storage/device, absent bias, and the prior common mixed-dtype projection cast",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "recipe_sha256": hashlib.sha256(encoded).hexdigest(),
            "source_sha256": SOURCE_SHA256,
            "core_source_sha256": core.SOURCE_SHA256,
            "public_source_sha256": PUBLIC_SOURCE_SHA256}


class AblationLinear(core.AblationLinear):
    def __init__(self, base, method, rank=8):
        if method not in REGISTRY:
            raise ValueError(f"method must be one of {METHODS}")
        super().__init__(base, SOURCE_METHODS[method], rank)
        self.method = method
        assert self.recipe == REGISTRY[method]

    def _initial_magnitude(self):
        # Set the storage policy BEFORE calculating the norm. Computing a
        # native-storage parameter and casting later would round the FP32-norm
        # treatment through BF16 and silently change the intervention.
        self.recipe = replace(self.recipe, magnitude_fp32=True)
        return super()._initial_magnitude()

    def norm_diagnostics(self):
        return {"normalization_axis": self.norm_axis,
                "zero_initialized_factor": self.zero_initialized_factor,
                "minimum_raw_norm": float(self._minimum_raw_norm),
                "clamp_activations": int(self._clamp_activations),
                "norm_evaluations": self._norm_evaluations,
                "norm_floor": None,
                "magnitude_dtype": str(self.m.dtype), "factor_dtype": str(self.lora_A.dtype),
                "recipe_sha256": recipe_manifest(self.method)["recipe_sha256"]}


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
                           inject_adapters=inject_adapters,
                           recipe_manifest=recipe_manifest,
                           assert_no_clamp=assert_no_clamp)
