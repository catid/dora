"""Train-only whole-block moment observers for retrieval and Aircraft GME."""
from __future__ import annotations
import copy
from pathlib import Path
import re
import time

import torch

from experiments import init_extension as formulas
from experiments import init_extension_runtime as runtime
from experiments import init_common as common
from experiments import retrieval
from experiments.second_round import vision
from experiments.vision import tensor_digest


def bind_calibration(source,destination,output,task,seed=None):
    value=common.read(source)
    for name,path in value['module_files'].items():
        if common.sha(path)!=value['module_sha256'][name]:raise ValueError('Parent calibration changed')
    meta=value['metadata']
    assert meta['train_only'] and meta['population_size']==128
    assert meta['model_repository']==(vision.REPO if task=='aircraft' else retrieval.MODEL)
    assert meta['model_revision']==(vision.REVISION if task=='aircraft' else retrieval.MODEL_REVISION)
    binding={'task':task,'parent_calibration_manifest_sha256':common.sha(source),
        'observer_source_sha256':common.sha(__file__),'model_repository':meta['model_repository'],
        'model_revision':meta['model_revision'],'classifier_seed':seed,
        'baseline_policy':'accepted_zero_update_dora','moment_domain':'whole_transformer_blocks',
        'mask_policy':'all CLS and patch tokens' if task=='aircraft' else 'attention-mask valid query and document tokens',
        'order_policy':'unchanged parent train-only calibration batches',
        'precision':'FP32 base/factors; BF16 CUDA autocast' if task=='aircraft' else 'BF16 base; FP32 factors; unchanged retrieval forward',
        'scalar_accumulation':'FP64 raw sums; paper/source denominator specified per recipe',
        'batch_manifest_sha256':common.digest_json(meta['batches'])}
    value=copy.deepcopy(value)
    value['metadata'].update(extension_gme_binding=binding,
        extension_factor_cache_dir=str((Path(output)/'extension_factor_cache'/('shared' if seed is None else f'seed_{seed}')).resolve()),
        source_calibration_manifest=str(Path(source).resolve()),
        source_calibration_manifest_sha256=common.sha(source))
    common.write_once(destination,value)
    return value


def _prepare(model,modules,config,seed,calibration,forward_batches,task):
    initial_device=next(model.parameters()).device
    modes={module:module.training for module in model.modules()};hooks=[];mask={'value':None};moments={};probes={};calls={}
    pattern=r'^blocks\.(\d+)\.' if task=='aircraft' else r'^encoder\.layer\.(\d+)\.'
    module_layers={name:int(re.match(pattern,name).group(1)) for name in modules}
    block_names={index:('blocks.' if task=='aircraft' else 'encoder.layer.')+str(index) for index in set(module_layers.values())}
    baseline=None
    def hook(index):
        def observe(module,inputs,output):
            hidden=output[0] if isinstance(output,(tuple,list)) else output
            moments[index].add(hidden,None if task=='aircraft' else mask['value'])
            calls[index]+=1
        return observe
    def observe(factors,stage):
        nonlocal moments,baseline,calls
        started=time.perf_counter();moments={index:formulas.ScalarMoments() for index in block_names}
        samples=[];calls={index:0 for index in block_names};batch_count=0
        with runtime.temporary_factors(modules,factors),torch.no_grad():
            for index,batch in enumerate(forward_batches()):
                batch_count+=1
                if task=='aircraft':
                    with torch.autocast('cuda',dtype=torch.bfloat16):value=model(batch)
                else:
                    mask['value']=batch['attention_mask'];value=retrieval.embeddings(model,batch)
                if index==0:samples.append(value[:4].detach().float().cpu())
        assert batch_count==(16 if task=='aircraft' else 32)
        assert all(value==batch_count for value in calls.values()),'GME whole-block observation call count changed'
        torch.cuda.synchronize();probe=torch.cat(samples)
        if baseline is None:baseline=probe
        probes[stage]={'forward_batch_count':batch_count,'block_hook_calls':dict(calls),'seconds':time.perf_counter()-started,'probe':probe.tolist(),
            'probe_sha256':tensor_digest([('probe',probe)]),
            'max_absolute_drift':float((probe-baseline).abs().max()),
            'mean_squared_drift':float((probe-baseline).square().mean())}
        return {index:item.result(eps=config['eps'],source_stability_floor=config['moment_policy']=='source_stability_floor')
                for index,item in moments.items()}
    options={key:value for key,value in config.items() if key not in ('method','adapter_scale','magnitude_policy','moment_policy')}
    try:
        with runtime.preserve_rng():
            model.cuda().eval()
            for index,path in sorted(block_names.items()):hooks.append(model.get_submodule(path).register_forward_hook(hook(index)))
            factors,metadata=formulas.initialize_gme({name:module.weight for name,module in modules.items()},
                {name:module.rank for name,module in modules.items()},module_layers,observe,seed=seed,**options)
            metadata.update(task_observer_binding=calibration['metadata']['extension_gme_binding'],
                module_layers=module_layers,block_names=block_names,probes=probes,
                baseline_policy='accepted_zero_update_dora',
                calibration_uses_labels=False,actual_task_head_seed=calibration['metadata'].get('classifier_seed'))
            return factors,metadata
    finally:
        for handle in hooks:handle.remove()
        model.to(initial_device)
        for module,mode in modes.items():module.training=mode


def retrieval_prepare(calibration,tokenizer,data,args):
    dataset=data[args.task];records=calibration['metadata']['batches']
    assert len(records)==16 and sum(len(r['query_ids']) for r in records)==128
    assert all(set(r['query_ids'])<=set(dataset['splits']['train']) for r in records)
    def batches():
        for row in records:
            for key,tokens,count_key in [('query_ids','query_tokens','query_nonpadding_tokens'),
                                         ('document_ids','corpus_tokens','document_nonpadding_tokens')]:
                value=retrieval.make_batch(tokenizer,dataset[tokens],row[key],args.device)
                assert int(value['attention_mask'].sum())==row[count_key]
                yield value
    def prepare(model,modules,config,seed):return _prepare(model,modules,config,seed,calibration,batches,args.task)
    prepare.binding=calibration['metadata']['extension_gme_binding']
    return prepare


def aircraft_prepare(calibration,datasets,args):
    dataset=copy.copy(datasets['train']);dataset.transform=datasets['val'].transform
    records=calibration['metadata']['batches'];assert len(records)==16
    def batches():
        for row in records:
            samples=[dataset[i] for i in row['indices']]
            images=torch.stack([x for x,y in samples]).cuda();labels=torch.tensor([y for x,y in samples],device='cuda')
            assert labels.tolist()==row['labels']
            assert tensor_digest([('images',images),('labels',labels)])==row['tensor_sha256']
            yield images
    def prepare(model,modules,config,seed):return _prepare(model,modules,config,seed,calibration,batches,'aircraft')
    prepare.binding=calibration['metadata']['extension_gme_binding']
    return prepare
