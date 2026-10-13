"""Qwen block-moment calibration for the separately frozen initializer extension."""
from __future__ import annotations

from contextlib import contextmanager
import random
import time
import copy

import numpy as np
import torch

from experiments import extraction as task
from experiments import init_common as common
from experiments import init_extension as formulas


@contextmanager
def preserve_execution_state(model):
    """Restore module mode flags and every RNG stream this callback may touch."""
    modes=[(module,module.training) for module in model.modules()]
    cpu=torch.get_rng_state();python=random.getstate();numpy=np.random.get_state()
    devices={p.device for p in model.parameters() if p.is_cuda}
    cuda={device:torch.cuda.get_rng_state(device) for device in devices}
    try:
        yield
    finally:
        for module,mode in modes:module.training=mode
        torch.set_rng_state(cpu);random.setstate(python);np.random.set_state(numpy)
        for device,state in cuda.items():torch.cuda.set_rng_state(state,device)


def observe_blocks(model,selected_modules,block_paths,batches,factors,stage,*,recipe):
    """Measure actual block outputs, with attention masks and no final-norm mix."""
    from experiments.init_extension_runtime import temporary_factors
    blocks=dict(model.named_modules())
    assert all(path in blocks for path in block_paths)
    moments={name:formulas.ScalarMoments() for name in block_paths}
    handles=[];active_mask=None;calls={name:0 for name in block_paths}
    def hook(name):
        def collect(module,inputs,output):
            hidden=output[0] if isinstance(output,(tuple,list)) else output
            assert torch.is_tensor(hidden) and hidden.ndim==3
            assert active_mask is not None and tuple(active_mask.shape)==tuple(hidden.shape[:-1])
            moments[name].add(hidden,active_mask)
            calls[name]+=1
        return collect
    with preserve_execution_state(model):
        try:
            for name in block_paths:handles.append(blocks[name].register_forward_hook(hook(name)))
            with temporary_factors(selected_modules,factors),torch.inference_mode():
                model.eval()
                device=next(model.parameters()).device
                for batch in batches:
                    active_mask=batch['attention_mask']
                    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        model(input_ids=batch['input_ids'],attention_mask=active_mask,use_cache=False)
            assert set(calls.values())=={len(batches)},(stage,calls)
            return {name:moment.result(source_stability_floor=recipe!='paper') for name,moment in moments.items()}
        finally:
            for handle in handles:handle.remove()


def make_gme_prepare(records,tokenizer,calibration,output,source_manifest):
    """Bind the exact 128 training examples and provide the shared runtime callback."""
    ids=calibration['metadata']['example_ids'];assert len(ids)==len(set(ids))==128
    rows={row['id']:row for row in records['train']}
    assert set(ids)<=set(rows)
    assert not set(ids)&{row['id'] for split in ('validation','test') for row in records[split]}
    chosen=[rows[id_] for id_ in ids]
    batch_rows=[chosen[start:start+8] for start in range(0,len(chosen),8)]
    calibration_binding={'example_ids':ids,'input_ids':[row['input_ids'] for row in chosen],
        'batch_size':8,'padding':'right, excluded by attention_mask','token_scope':'all valid prompt and target tokens',
        'loss':'none; labels are not passed during GME observations',
        'block_scope':'actual decoder block outputs before final model norm; 36 blocks',
        'scalar_pooling':'valid token-by-hidden entries, FP64 raw sums',
        'baseline_policy':'accepted_zero_update_dora',
        'base_dtype':'bfloat16','factor_and_magnitude_dtype':'float32','autocast':'cuda bfloat16'}

    def prepare(model,selected_modules,config,seed):
        paths=list(selected_modules)
        expected=[f'model.layers.{layer}.self_attn.{projection}' for layer in range(36) for projection in ('q_proj','v_proj')]
        assert paths==expected
        module_layers={name:'.'.join(name.split('.')[:3]) for name in paths}
        block_paths=list(dict.fromkeys(module_layers.values()))
        started=time.perf_counter()
        weights={name:module.weight.detach().cpu().clone() for name,module in selected_modules.items()}
        options={name:value for name,value in config.items() if name not in ('method','adapter_scale','magnitude_policy','moment_policy')}
        assert options['recipe'] in formulas.GME_RECIPES
        assert config.get('magnitude_policy','original')=='original'
        assert config.get('moment_policy','paper' if options['recipe']=='paper' else 'source_stability_floor')==(
            'paper' if options['recipe']=='paper' else 'source_stability_floor')
        with preserve_execution_state(model):
            batches=[task.batch(part,tokenizer) for part in batch_rows]
            def observe(factors,stage):
                return observe_blocks(model,selected_modules,block_paths,batches,factors,stage,recipe=options['recipe'])
            factors,details=formulas.initialize_gme(weights,{name:module.lora_A.shape[0] for name,module in selected_modules.items()},
                module_layers,observe,seed=seed,**options)
        assert list(factors)==paths
        hashes={name:common.tensor_digest(zip(('up','down'),pair)) for name,pair in factors.items()}
        metadata={'task':'ViGGO','calibration_binding_sha256':common.digest_json(calibration_binding),
            'factor_sha256':hashes,'gme':details,'cold_initializer_seconds':time.perf_counter()-started}
        return factors,metadata

    parent_calibration=copy.deepcopy(calibration)
    parent_calibration['metadata'].pop('extension_gme_binding',None)
    prepare.binding={'schema':'dora-extension-qwen-block-observer-v1','task':'ViGGO',
        'calibration':calibration_binding,'calibration_manifest_sha256':common.digest_json(parent_calibration),
        'source_manifest':source_manifest,'taskcollector_source_sha256':common.sha(__file__),
        'checkpoint':{'model':task.MODEL,'revision':task.MODEL_REVISION},
        'module_layers':{f'model.layers.{layer}.self_attn.{projection}':f'model.layers.{layer}'
            for layer in range(36) for projection in ('q_proj','v_proj')},
        'moment_domain':'one Qwen decoder domain; actual pre-final-norm block outputs'}
    return prepare
