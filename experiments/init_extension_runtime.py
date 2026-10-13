"""Separate extension initialization around the unchanged accepted DoRA runtime.

GME callbacks run coordinated model probes, then return a source-bound factor
pack. No extension method changes the inherited DoRA training forward.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments import init_ablations as centered
from experiments import init_common as common
from experiments import init_data
from experiments import init_extension as formulas
from experiments import next_ablations as accepted


def tensor_sha(value):return init_data._tensor_hash(value)
def json_sha(value):return init_data._json_hash(value)
def source_sha(path):return init_data._sha_file(path)


def calibration_binding(calibration):
    if calibration is None:return {}
    binding={'metadata':calibration.get('metadata',{}),'module_sha256':calibration.get('module_sha256',{})}
    if 'modules' in calibration:
        binding['in_memory_statistics_sha256']={name:json_sha(init_data._statistic_binding(row))
                                               for name,row in calibration['modules'].items()}
    return binding


@contextmanager
def preserve_rng():
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python=random.getstate();numpy=np.random.get_state()
    try:yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        random.setstate(python);np.random.set_state(numpy)


@contextmanager
def temporary_factors(modules,factors):
    """Install all factors, or accepted zero-update DoRA, for one probe.

Restore factors, magnitudes, diagnostic counters and instance forwards on every
exit, including exceptions. Frozen weight/bias objects are never replaced.
"""
    if factors is not None and set(factors)!=set(modules):raise ValueError('Probe factor modules differ')
    saved={}
    for name,layer in modules.items():
        saved[name]={'up':layer.lora_B.detach().clone(),'down':layer.lora_A.detach().clone(),
            'm':layer.m.detach().clone(),'forward_present':'forward' in layer.__dict__,
            'forward':layer.__dict__.get('forward'),
            'diagnostics':{key:copy.deepcopy(getattr(layer,key)) for key in
                           ('_minimum_raw_norm','_clamp_activations','_norm_evaluations')}}
    try:
        with torch.no_grad():
            for name,layer in modules.items():
                if factors is None:
                    # GME source constructs DoRA before measuring its baseline.
                    # Keep native column-normalization rounding rather than
                    # substituting a raw pretrained Linear projection.
                    layer.lora_A.zero_()
                else:
                    up,down=factors[name]
                    if up.shape!=layer.lora_B.shape or down.shape!=layer.lora_A.shape:
                        raise ValueError('Probe factor shape mismatch')
                    if not torch.isfinite(up).all() or not torch.isfinite(down).all():
                        raise ValueError('Nonfinite extension probe factors')
                    layer.lora_B.copy_(up);layer.lora_A.copy_(down)
        yield
    finally:
        with torch.no_grad():
            for name,layer in modules.items():
                row=saved[name];layer.lora_B.copy_(row['up']);layer.lora_A.copy_(row['down']);layer.m.copy_(row['m'])
                if row['forward_present']:layer.forward=row['forward']
                else:layer.__dict__.pop('forward',None)
                for key,value in row['diagnostics'].items():setattr(layer,key,value)


def _cache(path,key,compute):
    started=time.perf_counter()
    if path is not None and path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['bindings']!=key:raise ValueError('Extension factor cache binding mismatch')
        hit=True
    else:
        factors,metadata=compute()
        payload={'bindings':key,'factors':factors,'metadata':metadata,
                 'factor_sha256':{name:[tensor_sha(up),tensor_sha(down)] for name,(up,down) in factors.items()},
                 'source_compute_seconds':time.perf_counter()-started}
        hit=False
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(f'.{time.time_ns()}.tmp');torch.save(payload,temporary);temporary.replace(path)
    for name,(up,down) in payload['factors'].items():
        if up.dtype!=torch.float32 or down.dtype!=torch.float32 or up.device.type!='cpu' or down.device.type!='cpu':
            raise ValueError('Extension factor pack must be CPU FP32')
        if not torch.isfinite(up).all() or not torch.isfinite(down).all():raise ValueError('Nonfinite extension factor pack')
        if [tensor_sha(up),tensor_sha(down)]!=payload['factor_sha256'][name]:raise ValueError('Extension factor cache checksum mismatch')
    return payload['factors'],{**payload['metadata'],'factor_cache_hit':hit,
        'source_compute_seconds':payload['source_compute_seconds'],'actual_cache_or_compute_seconds':time.perf_counter()-started}


def backend_for(candidate,factor_seed=42,calibration=None,*,gme_prepare=None):
    kind,config=common.spec_parts(candidate['spec'])
    if kind=='random':return common.backend_for(candidate,factor_seed,calibration)
    if kind not in ('extension_centered','extension_warmstart'):raise ValueError('Unknown extension kind')
    config=json.loads(json.dumps(config,allow_nan=False))
    expected='lora_null' if kind=='extension_centered' else 'gme'
    if config.get('method')!=expected or config.get('adapter_scale',1.)!=1.:
        raise ValueError('Extension method/kind or adapter scale mismatch')
    binding=calibration_binding(calibration);calibration_sha=json_sha(binding)
    metadata={} if calibration is None else calibration.get('metadata',{})
    cache_root=metadata.get('extension_factor_cache_dir')
    generator_sources={'math':source_sha(formulas.__file__),'runtime':source_sha(__file__),
                       'accepted':accepted.SOURCE_SHA256,'core':accepted.core.SOURCE_SHA256,
                       'centered':source_sha(centered.__file__)}

    def recipe(method):
        payload={'schema':'dora-init-extension-runtime-v1','method':method,'config':config,
            'initializer_kind':kind,'factor_seed':factor_seed,'source_sha256':generator_sources,
            'calibration_sha256':calibration_sha,'normalization_axis':0,'norm_floor':None,
            'compensation':'W+(BA-B0A0)' if kind=='extension_centered' else 'none; intentional GME warm start',
            'magnitude_policy':'original','trainable_registration_order':['m','lora_B','lora_A'],
            'stream_policy':'accepted global CPU draws; local factor generator; probe RNG restored'}
        return {**payload,'recipe_sha256':json_sha(payload)}

    def key_for(weights,rank):
        return {'config':config,'seed':factor_seed,'calibration_sha256':calibration_sha,
            'sources':generator_sources,'torch':str(torch.__version__),'cpu_threads':torch.get_num_threads(),
            'weights':{name:tensor_sha(value) for name,value in weights.items()},'rank':rank,
            'devices':{name:str(value.device) for name,value in weights.items()},
            'dtypes':{name:str(value.dtype) for name,value in weights.items()},
            'tf32':torch.backends.cuda.matmul.allow_tf32,'observer':metadata.get('extension_gme_binding')}

    def path_for(key):return None if not cache_root else Path(cache_root)/(json_sha(key)+'.pt')

    def null_factors(weight,rank,name):
        if calibration is None:raise ValueError('LoRA-Null requires bound task input covariance')
        key=key_for({name:weight},rank)
        def compute():
            if 'modules' in calibration:stats=calibration['modules'][name]
            else:
                path=calibration['module_files'][name]
                init_data._verify_calibration_file(path,calibration['module_sha256'][name])
                stats=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            covariance=config['covariance_key']
            if covariance not in stats:raise ValueError(f'Missing LoRA-Null covariance {covariance}')
            up,down,record=formulas.initialize_lora_null(weight,rank,stats[covariance],
                calibration_role=config['calibration_role'],balance_gain=config['balance_gain'])
            return {name:(up,down)},record
        pack,record=_cache(path_for(key),key,compute)
        return *pack[name],record

    def inject(model,method,rank=8,targets=None):
        started=time.perf_counter()
        if kind=='extension_centered':
            base=centered.backend(centered.InitSpec(),factor_seed,factor_initializer=null_factors,
                data_spec={'config':config,'calibration_sha256':calibration_sha,'source_sha256':generator_sources})
            model=base.inject_adapters(model,method,rank,targets)
            modules={name:layer for name,layer in model.named_modules() if isinstance(layer,centered.AblationLinear)}
            for layer in modules.values():
                layer._extension_recipe=recipe(method)
                layer._initialization_diagnostics['initializer_kind']=kind
        else:
            if config.get('magnitude_policy')!='original':raise ValueError('GME requires original magnitudes')
            model=accepted.inject_adapters(model,'next_base',rank,targets)
            modules={name:layer for name,layer in model.named_modules() if isinstance(layer,accepted.AblationLinear)}
            if not modules:raise ValueError('No extension modules injected')
            weights={name:layer.weight for name,layer in modules.items()};key=key_for(weights,rank)
            weight_hashes={name:tensor_sha(layer.weight) for name,layer in modules.items()}
            magnitude_hashes={name:tensor_sha(layer.m) for name,layer in modules.items()}
            def compute():
                if gme_prepare is None:raise ValueError('Uncached GME injection requires task gme_prepare')
                if not metadata.get('extension_gme_binding') or getattr(gme_prepare,'binding',None)!=metadata['extension_gme_binding']:
                    raise ValueError('GME task observer provenance differs from bound calibration metadata')
                with preserve_rng():return gme_prepare(model,modules,config,factor_seed)
            with preserve_rng():pack,record=_cache(path_for(key),key,compute)
            if set(pack)!=set(modules):raise ValueError('GME factor pack modules differ')
            with torch.no_grad():
                for name,layer in modules.items():
                    if tensor_sha(layer.weight)!=weight_hashes[name] or tensor_sha(layer.m)!=magnitude_hashes[name]:
                        raise ValueError('GME callback changed original weight or magnitude')
                    up,down=pack[name]
                    if up.shape!=layer.lora_B.shape or down.shape!=layer.lora_A.shape:raise ValueError('GME factor shape mismatch')
                    layer.lora_B.copy_(up);layer.lora_A.copy_(down)
                    layer._extension_recipe=recipe(method);layer.zero_initialized_factor='neither'
                    layer.initialization={'initializer_kind':kind,'factor_seed':factor_seed,'module_path':name,
                        'compensated':False,'magnitude_policy':'original','up_sha256':tensor_sha(layer.lora_B),
                        'down_sha256':tensor_sha(layer.lora_A),'pretrained_weight_sha256':weight_hashes[name],
                        'initial_magnitude_sha256':magnitude_hashes[name],'callback_metadata':record}
        if not modules:raise ValueError('No extension modules injected')
        for layer in modules.values():
            if list(name for name,p in layer.named_parameters() if p.requires_grad)!=['m','lora_B','lora_A']:
                raise ValueError('Extension trainable registration order changed')
        model._extension_initialization_receipt={'initializer_seconds':time.perf_counter()-started,
            'module_count':len(modules),'recipe_sha256':recipe(method)['recipe_sha256'],
            'kind':kind,'factor_seed':factor_seed}
        return model

    def diagnose(model):
        rows={}
        for name,layer in model.named_modules():
            if not isinstance(layer,accepted.AblationLinear) or not hasattr(layer,'_extension_recipe'):continue
            row=layer.assert_no_clamp();row['recipe_sha256']=layer._extension_recipe['recipe_sha256']
            row['initializer']=layer._initialization_diagnostics if isinstance(layer,centered.AblationLinear) else layer.initialization
            rows[name]=row
        return rows

    return SimpleNamespace(adapter_types=accepted.backend().adapter_types,inject_adapters=inject,
        parameter_groups=accepted.parameter_groups,recipe_manifest=recipe,assert_no_clamp=diagnose,
        initializer_kind=kind)
