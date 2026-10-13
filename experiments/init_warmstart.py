"""Intentional nonzero initialization, isolated from centered DoRA experiments."""
from __future__ import annotations

import json
from types import SimpleNamespace
import torch
from torch import nn
from experiments import init_data
from experiments import next_ablations as previous


def backend(config, factor_seed=42, calibration=None):
    config = json.loads(json.dumps(dict(config), allow_nan=False))
    if config.get('method') not in {'loraone', 'lorada', 'cntlora'}:
        raise ValueError('Warm-start lane supports source LoRA-One, LoRA-DA and CNTLoRA')
    magnitude_policy = config.get('magnitude_policy', 'original')
    if magnitude_policy not in {'original', 'adapted_column'}:
        raise ValueError('Warm-start magnitude_policy must be original or adapted_column')
    factor_spec = {key: value for key,value in config.items() if key != 'magnitude_policy'}
    data = init_data.backend(factor_spec, factor_seed, calibration)

    def recipe(name):
        payload = {'schema': 'dora-init-warmstart-v1', 'method': name,
                   'config': config, 'initializer_kind': 'warmstart',
                   'source_sha256': init_data._sha_file(__file__),
                   'factor_generator_source_sha256': init_data._sha_file(init_data.__file__),
                   'runtime_source_sha256': previous.SOURCE_SHA256,
                   'runtime_core_source_sha256': previous.core.SOURCE_SHA256,
                   'calibration_sha256': data.data_spec['calibration_sha256'],
                   'compensation': 'none; source intended nonzero factor product retained',
                   'magnitude_policy': magnitude_policy,
                   'initial_identity': 'not required; report initial function perturbation',
                   'geometry': 'accepted column-normalized dense DoRA; source LoRA geometry differs',
                   'trainable_registration_order': ['m','lora_B','lora_A'],
                   'stream_policy': 'consume accepted global draws then local callback',
                   'formula': 'm*(W+BA)/column_norm(W+BA), full denominator gradients'}
        return dict(payload, recipe_sha256=init_data._json_hash(payload))

    class WarmstartLinear(previous.AblationLinear):
        def __init__(self, base, name, rank, module_path):
            super().__init__(base, 'next_base', rank)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(self.weight.device) if self.weight.is_cuda else None
            up, down, metadata = data.factor_initializer(self.weight, rank, module_path)
            if not torch.equal(cpu_rng,torch.get_rng_state()):
                raise ValueError('Warm-start initializer changed global CPU RNG')
            if cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state(self.weight.device)):
                raise ValueError('Warm-start initializer changed global CUDA RNG')
            with torch.no_grad():
                self.lora_B.copy_(up.to(device=self.lora_B.device))
                self.lora_A.copy_(down.to(device=self.lora_A.device))
                delta = self.lora_B @ self.lora_A
                adapted = self.weight + delta
                if magnitude_policy == 'adapted_column':
                    self.m.copy_(adapted.norm(p=2,dim=0,keepdim=True).to(self.m.dtype))
            self.method = name
            self.zero_initialized_factor = 'neither'
            self.initialization = {
                'initializer_kind':'warmstart', 'factor_seed':factor_seed,
                'module_path':module_path, 'compensated':False,
                'initial_adapter_delta_is_zero':bool(delta.count_nonzero()==0),
                'initial_delta_frobenius_norm':float(delta.double().norm()),
                'initial_delta_relative_norm':float(delta.double().norm()/self.weight.double().norm()),
                'magnitude_policy':magnitude_policy,
                'up_sha256':init_data._tensor_hash(self.lora_B),
                'down_sha256':init_data._tensor_hash(self.lora_A),
                'pretrained_weight_sha256':init_data._tensor_hash(self.weight),
                'callback_metadata':metadata,
                'runtime_cost':'one factor product per forward, same accepted DoRA arithmetic',
                'reload_policy':'bound factor recipe and unchanged pretrained W; trainable checkpoint overwrites m,B,A'}

        def norm_diagnostics(self):
            return {'normalization_axis':self.norm_axis, 'zero_initialized_factor':self.zero_initialized_factor,
                    'minimum_raw_norm':float(self._minimum_raw_norm),
                    'clamp_activations':int(self._clamp_activations),
                    'norm_evaluations':self._norm_evaluations, 'norm_floor':None,
                    'magnitude_dtype':str(self.m.dtype),'factor_dtype':str(self.lora_A.dtype),
                    'recipe_sha256':recipe(self.method)['recipe_sha256'],
                    'initializer':self.initialization}

    def inject(model,name,rank=8,targets=None):
        if isinstance(targets,str):targets=(targets,)
        matches=((lambda path,module:True) if targets is None else targets if callable(targets)
                 else lambda path,module:any(path==suffix or path.endswith('.'+suffix) for suffix in targets))
        selected={id(module) for path,module in model.named_modules(remove_duplicate=False)
                  if isinstance(module,nn.Linear) and matches(path,module)}
        if not selected:raise ValueError('No Linear modules matched warm-start targets')
        paths={id(module):path for path,module in reversed(list(model.named_modules()))}
        model.requires_grad_(False);replacements={}
        def replace(module):
            if id(module) in replacements:return replacements[id(module)]
            result=WarmstartLinear(module,name,rank,paths[id(module)]) if id(module) in selected else module
            replacements[id(module)]=result
            if result is module:
                for key,child in list(module._modules.items()):
                    if child is not None:setattr(module,key,replace(child))
            return result
        return replace(model)

    return SimpleNamespace(adapter_types=previous.backend().adapter_types,
                           inject_adapters=inject,recipe_manifest=recipe,
                           parameter_groups=previous.parameter_groups,
                           assert_no_clamp=lambda model:{name:layer.assert_no_clamp()
                               for name,layer in model.named_modules() if isinstance(layer,previous.AblationLinear)
                               and getattr(layer,'initialization',{}).get('initializer_kind')=='warmstart'})
