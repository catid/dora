"""Dense NF4 reference quantizer and alternating LoftQ initialization.

This is a quantization substudy, not a start-preserving BF16 initializer. The
output is a dense dequantized base, so no compressed-memory/throughput claim is
possible. NF4 codebook and group64 defaults match bitsandbytes/PEFT sources
pinned in initialization_literature.md and the cached loftq/bnb manifests.
"""
from __future__ import annotations

import math
import time

import torch

from experiments.init_data import _matrix, _positive, _sha_file, _svd, _tensor_hash

NF4 = (-1., -.6961928009986877, -.5250730514526367, -.39491748809814453,
       -.28444138169288635, -.18477343022823334, -.09105003625154495, 0.,
       .07958029955625534, .16093020141124725, .24611230194568634,
       .33791524171829224, .44070982933044434, .5626170039176941,
       .7229568362236023, 1.)
# Exact CUDA dQuantizeNF4 decision thresholds, not float32-averaged levels.
NF4_BOUNDARIES = (-0.8480964303016663, -0.6106328964233398, -0.4599952697753906,
                  -0.33967941999435425, -0.23460739850997925, -0.13791173696517944,
                  -0.045525018125772476, 0.03979014977812767, 0.1202552542090416,
                  0.2035212516784668, 0.2920137643814087, 0.3893125355243683,
                  0.5016633868217468, 0.6427869200706482, 0.8614783883094788)
# Above are exact float32 C literal values. Parsing the source decimals first
# as Python float64 and then as float32 double-rounds four boundaries!


@torch.no_grad()
def quantize_nf4(weight, blocksize=64, *, return_state=False):
    """Quantize contiguous flat groups, returning FP32 dense dequantized values.

    No double quantization. Groups use their absolute maximum. Zero groups map
    to NF4's exact zero. Partial final groups are padded with zeros only for
    computing an unchanged absolute maximum. Indices are unpacked uint8 here.
    """
    if weight.ndim != 2:
        raise ValueError('NF4 reference expects a weight matrix')
    if blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError('Unsupported bitsandbytes NF4 blocksize')
    w = _matrix(weight, weight.shape, 'weight')
    length = w.numel()
    padding = (-length) % blocksize
    flat = torch.nn.functional.pad(w.flatten(), (0, padding)).view(-1, blocksize)
    maxima = flat.abs().amax(1)
    normalized = flat * maxima.clamp_min(torch.finfo(torch.float32).tiny).reciprocal()[:, None]
    levels = torch.tensor(NF4, dtype=torch.float32)
    boundaries = torch.tensor(NF4_BOUNDARIES, dtype=torch.float32)
    indices = torch.bucketize(normalized, boundaries, right=False).to(torch.uint8)
    result = (levels[indices.long()] * maxima[:, None]).flatten()[:length].reshape(w.shape)
    metadata = {'quantizer': 'dense_reference_nf4', 'bits': 4, 'blocksize': blocksize,
                'compress_statistics': False, 'codebook': list(NF4),
                'decision_boundaries': list(NF4_BOUNDARIES),
                'normalization_arithmetic': 'FP32 multiply by reciprocal absmax',
                'scale': 'flat contiguous group absolute maximum, float32',
                'original_shape': list(w.shape), 'padding_elements': padding,
                'zero_groups': int((maxima == 0).sum()),
                'quantization_mse': float((result-w).square().mean()),
                'dense_dequantized_sha256': _tensor_hash(result),
                'source_sha256': _sha_file(__file__)}
    if return_state:
        return result, metadata, {'indices': indices, 'absmax': maxima, 'length': length}
    return result, metadata


@torch.no_grad()
def initialize_loftq(weight, rank, spec=None):
    """Return (dense_base, up, down, metadata), with no compensation.

    Dense base uses the original weight dtype, as in PEFT loftq_init; factors
    return CPU FP32. The independent quantizer-only control is quantize_nf4(W)
    cast to the same base dtype, paired with the accepted one-zero initializer.
    DoRA magnitude must be chosen explicitly by the runner (original W or
    dense quantized W); both are integration variants of source LoRA.
    """
    started = time.perf_counter()
    spec = dict(spec or {})
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= min(weight.shape):
        raise ValueError('rank must be within both weight dimensions')
    iterations = int(spec.get('iterations', 1))
    if iterations <= 0:
        raise ValueError('LoftQ iterations must be positive')
    if int(spec.get('bits', 4)) != 4:
        raise ValueError('This source-matched reference implements NF4 only')
    blocksize = int(spec.get('blocksize', 64))
    scale = _positive(spec, 'adapter_scale', 1)
    balance = _positive(spec, 'balance_gain', 1)
    w = _matrix(weight, weight.shape, 'weight')
    residual = w.clone()
    rows = []
    for step in range(iterations):
        quantized, quantizer = quantize_nf4(residual, blocksize)
        error = w - quantized
        u, values, vh = _svd(error, spec.get('decomposition_cache_dir'))
        roots = values[:rank].sqrt()
        up, down = u[:, :rank] * roots, roots[:, None] * vh[:rank]
        reconstruction = quantized + up @ down
        rows.append({'iteration': step+1, 'quantization_mse_to_original': float((quantized-w).square().mean()),
                     'corrected_mse_to_original': float((reconstruction-w).square().mean()),
                     'selected_residual_singular_values': values[:rank].tolist()})
        residual = w - up @ down
    up = (up * balance / math.sqrt(scale)).contiguous()
    down = (down / balance / math.sqrt(scale)).contiguous()
    base = quantized.to(dtype=weight.dtype).contiguous()
    metadata = {'method': 'loftq_nf4_dense_simulation', 'spec': spec,
                'source_sha256': _sha_file(__file__), 'weight_sha256': _tensor_hash(weight),
                'rank': rank, 'adapter_scale': scale, 'iterations': rows,
                'quantizer': quantizer, 'base_storage_dtype': str(base.dtype),
                'factor_storage_dtype': 'torch.float32', 'compensation': 'none',
                'initial_identity': 'approximate quantization residual reconstruction',
                'magnitude_policy': 'runner must explicitly choose original or quantized base magnitude',
                'paper_boundary': 'LoftQ factor initialization; dense storage and normalized DoRA integration are adaptations',
                'base_sha256': _tensor_hash(base), 'up_sha256': _tensor_hash(up), 'down_sha256': _tensor_hash(down),
                'after_base_cast_corrected_mse': float((base.float()+scale*(up@down)-w).square().mean()),
                'elapsed_seconds': time.perf_counter()-started}
    return base, up, down, metadata


def backend(config, factor_seed=42, calibration=None):
    """Separate dense quantization-study backend using accepted DoRA arithmetic.

    method=loftq keeps residual factors; method=quantizer_only uses original
    one-zero factors. Set base_source=paired_loftq for the control to share the
    exact final optimized base of a multi-iteration LoftQ treatment. The default
    direct base is Q(W), exactly matching one-iteration LoftQ's base.
    """
    import json
    from types import SimpleNamespace
    from torch import nn
    from experiments import next_ablations as previous
    from experiments.init_data import _json_hash
    config = json.loads(json.dumps(dict(config), allow_nan=False))
    method = config.get('method', 'loftq')
    if method not in {'loftq', 'quantizer_only'}:
        raise ValueError('Quantized study method must be loftq or quantizer_only')
    policy = config.get('magnitude_policy', 'original')
    if policy not in {'original', 'quantized'}:
        raise ValueError('magnitude_policy must be original or quantized')
    base_source = config.get('base_source', 'direct')
    if base_source not in {'direct', 'paired_loftq'}:
        raise ValueError('base_source must be direct or paired_loftq')
    if float(config.get('adapter_scale', 1)) != 1:
        raise ValueError('Accepted normalized DoRA runtime has implicit adapter_scale=1')

    def recipe(name):
        payload = {'schema': 'dora-init-quantized-v1', 'method': name,
                   'config': config, 'initializer_kind': 'quantized',
                   'source_sha256': _sha_file(__file__),
                   'factor_generator_source_sha256': _sha_file(__import__('experiments.init_data', fromlist=['x']).__file__),
                   'runtime_source_sha256': previous.SOURCE_SHA256,
                   'runtime_core_source_sha256': previous.core.SOURCE_SHA256,
                   'quantizer': 'NF4 dense dequantized simulation; no compression claim',
                   'magnitude_policy': policy, 'compensation': 'none',
                   'initial_identity': 'not required; measure quantization/reconstruction error',
                   'trainable_registration_order': ['m', 'lora_B', 'lora_A'],
                   'factor_seed_policy': 'accepted global CPU draws; explicit fit seed controls RNG',
                   'formula': 'accepted column-norm DoRA on dense quantized base plus BA'}
        return dict(payload, recipe_sha256=_json_hash(payload))

    class QuantizedLinear(previous.AblationLinear):
        def __init__(self, base, name, rank, module_path):
            super().__init__(base, 'next_base', rank)
            original = self.weight.detach().cpu()
            if method == 'loftq' or base_source == 'paired_loftq':
                qbase, up, down, details = initialize_loftq(original, rank, config)
            else:
                qbase, details = quantize_nf4(original, int(config.get('blocksize', 64)))
                qbase = qbase.to(dtype=original.dtype)
            # Replace, never mutate the pretrained tensor supplied by the caller.
            self.weight = nn.Parameter(qbase.to(device=self.weight.device), requires_grad=False)
            if policy == 'quantized':
                with torch.no_grad():
                    self.m.copy_(self._initial_magnitude())
            if method == 'loftq':
                with torch.no_grad():
                    self.lora_B.copy_(up.to(device=self.lora_B.device))
                    self.lora_A.copy_(down.to(device=self.lora_A.device))
                self.zero_initialized_factor = 'neither'
            self.method = name
            self.initialization = {'initializer_kind': 'quantized', 'module_path': module_path,
                                   'factor_seed': factor_seed, 'quantization': details,
                                   'base_source': base_source, 'magnitude_policy': policy,
                                   'pretrained_weight_sha256': _tensor_hash(original),
                                   'quantized_weight_sha256': _tensor_hash(self.weight),
                                   'initial_adapter_delta_is_zero': method == 'quantizer_only',
                                   'up_sha256': _tensor_hash(self.lora_B),
                                   'down_sha256': _tensor_hash(self.lora_A)}

        def norm_diagnostics(self):
            return {'normalization_axis': self.norm_axis,
                    'zero_initialized_factor': self.zero_initialized_factor,
                    'minimum_raw_norm': float(self._minimum_raw_norm),
                    'clamp_activations': int(self._clamp_activations),
                    'norm_evaluations': self._norm_evaluations, 'norm_floor': None,
                    'magnitude_dtype': str(self.m.dtype), 'factor_dtype': str(self.lora_A.dtype),
                    'recipe_sha256': recipe(self.method)['recipe_sha256'],
                    'initializer': self.initialization}

    def inject(model, name, rank=8, targets=None):
        if isinstance(targets, str):
            targets = (targets,)
        matches = ((lambda path, module: True) if targets is None else targets if callable(targets)
                   else lambda path, module: any(path == suffix or path.endswith('.'+suffix) for suffix in targets))
        selected = {id(module) for path, module in model.named_modules(remove_duplicate=False)
                    if isinstance(module, nn.Linear) and matches(path, module)}
        if not selected:
            raise ValueError('No Linear modules matched quantized study targets')
        paths = {id(module): path for path, module in reversed(list(model.named_modules()))}
        model.requires_grad_(False)
        replacements = {}
        def replace(module):
            if id(module) in replacements:
                return replacements[id(module)]
            result = QuantizedLinear(module, name, rank, paths[id(module)]) if id(module) in selected else module
            replacements[id(module)] = result
            if result is module:
                for key, child in list(module._modules.items()):
                    if child is not None:
                        setattr(module, key, replace(child))
            return result
        return replace(model)

    return SimpleNamespace(adapter_types=previous.backend().adapter_types,
                           inject_adapters=inject, recipe_manifest=recipe,
                           parameter_groups=previous.parameter_groups,
                           assert_no_clamp=lambda model: {name: layer.assert_no_clamp()
                               for name, layer in model.named_modules() if isinstance(layer, previous.AblationLinear)
                               and getattr(layer, 'initialization', {}).get('initializer_kind') == 'quantized'})
