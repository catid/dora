#!/usr/bin/env python3
"""Separate-process original-backbone reload and actual-update export audit.

Uses only the presealed 64-example training probe. Exact-refactor tolerances are
fixed below before execution. Truncated distortion is measured, never forced to
pass an exact-parity tolerance. The ordinary recovery process is separate again.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from safetensors.torch import save_file,load_file
from rankstudy.v3.checkpoint import json_write,backbone_hash,load_adapter
from rankstudy.v3.compression import checkpoint_modules,compress_modules,write_export,actual_factors
from rankstudy.v3.adapters import inject,adapted


def emit(a):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from rankstudy.v2.data import read_jsonl,collate
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1);torch.manual_seed(999)
    desc=json.loads((a.adapter/'manifest.json').read_text())
    spec=next(iter(desc['modules'].values()))
    pattern={n:m['rank'] for n,m in desc['modules'].items()}
    model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
    inject(model,1,spec['init'],spec['scale'],0,spec['score'],
           tuple(sorted({n.split('.')[-1] for n in pattern})),pattern)
    load_adapter(a.adapter,model,'cuda'); model.eval()
    for m in adapted(model).values():m.begin_update(False)
    original=backbone_hash(model)
    if a.backbone_arithmetic=='fp32_oracle':
        # Same stored BF16 weight values, promoted for an arithmetic diagnostic.
        # Primary training/evaluation remains BF16. Do not claim this is another
        # trained checkpoint or use these losses for method selection.
        for p in model.parameters():
            if not p.requires_grad:p.data=p.data.float()
    arithmetic_hash=backbone_hash(model)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    by_source={r['source_index']:r for r in read_jsonl(a.data_root/'train.jsonl')}
    indices=json.loads((a.data_root/'probe_source_indices.json').read_text())
    assert len(indices)==64 and len(set(indices))==64 and all(i in by_source for i in indices)
    rows=[by_source[i] for i in indices]
    losses=[];last=[];begin=time.perf_counter()
    with torch.no_grad(),sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        for row in rows:
            b=collate([row],tok.eos_token_id,'cuda')
            logits=model(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits.float()
            pred,labels=logits[:,:-1],b['labels'][:,1:]
            loss=torch.nn.functional.cross_entropy(pred.reshape(-1,pred.shape[-1]),labels.reshape(-1),ignore_index=-100,reduction='sum')
            count=int(labels.ne(-100).sum())
            losses.append({'source_index':row['source_index'],'group_id':row['group_id'],
                           'nll':float(loss/count),'loss_sum':float(loss),'supervised_tokens':count})
            last.append(logits[0,-2].cpu())
    torch.cuda.synchronize()
    a.out.mkdir(parents=True,exist_ok=False)
    save_file({'last_supervised_position_logits':torch.stack(last)},str(a.out/'logits.safetensors'))
    final_hash=backbone_hash(model)
    if final_hash!=arithmetic_hash:raise RuntimeError('evaluation changed frozen backbone values')
    json_write(a.out/'metrics.json',{'pid':os.getpid(),'adapter':str(a.adapter),'backbone_sha256':original,
        'probe_sha256':hashlib.sha256((a.data_root/'probe_source_indices.json').read_bytes()).hexdigest(),
        'groups':losses,'group_balanced_nll':sum(r['nll'] for r in losses)/len(losses),
        'token_weighted_nll':sum(r['loss_sum'] for r in losses)/sum(r['supervised_tokens'] for r in losses),
        'probe_gpu_seconds':time.perf_counter()-begin,'backend':'EFFICIENT_ATTENTION',
        'backbone_arithmetic':a.backbone_arithmetic,'arithmetic_backbone_sha256':arithmetic_hash})


def compare(a):
    torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=False)
    export_desc=json.loads((a.export/'manifest.json').read_text())
    source=Path(export_desc['source']['checkpoint']);shapes=export_desc['source']['target_shapes']
    modules=checkpoint_modules(source,shapes)
    factors,_,receipt=compress_modules(modules,8,1.,policy='exact')
    exact=a.out/'exact_export'
    write_export(exact,factors,1.,receipt,export_desc['source'])
    # Low-rank reconstruction check avoids materializing a dense matrix per layer.
    g=torch.Generator().manual_seed(2411);errors={}
    for name,m in modules.items():
        c,d=actual_factors(m);x=torch.randn(13,d.shape[1],dtype=torch.float64,generator=g)
        expected=(x@d.T)@c.T
        got=(x@factors[name]['A'].double().T)@factors[name]['B'].double().T
        errors[name]=float((got-expected).norm()/expected.norm().clamp_min(1e-30))
    policy={'exact_factor_relative_error_max':2e-6,'exact_logits_relative_rms_max':.002,
            'exact_logits_absolute_error_max':.25,'exact_group_nll_abs_difference_max':.005,
            'probe':'sealed 64 training examples, no dev/test','selection_eligible':False}
    if a.backbone_arithmetic=='fp32_oracle':
        policy.update(exact_logits_relative_rms_max=2e-5,exact_logits_absolute_error_max=.002,
                      exact_group_nll_abs_difference_max=5e-5)
    policy['backbone_arithmetic']=a.backbone_arithmetic
    json_write(a.out/'policy.json',policy)
    records={}
    for label,adapter in [('native',source),('native_repeat',source),('exact',exact),('truncated',a.export)]:
        cmd=[sys.executable,str(Path(__file__)), '--emit','--adapter',str(adapter),'--out',str(a.out/label),
             '--model',a.model,'--data-root',str(a.data_root),'--backbone-arithmetic',a.backbone_arithmetic]
        with (a.out/f'{label}.log').open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        records[label]=json.loads((a.out/label/'metrics.json').read_text())
    native=load_file(str(a.out/'native/logits.safetensors'))['last_supervised_position_logits'].double()
    comparisons={}
    for label in ['exact','truncated']:
        got=load_file(str(a.out/label/'logits.safetensors'))['last_supervised_position_logits'].double()
        comparisons[label]={'logits_relative_rms':float((got-native).norm()/native.norm()),
                            'logits_max_absolute_difference':float((got-native).abs().max()),
                            'group_nll_difference':records[label]['group_balanced_nll']-records['native']['group_balanced_nll'],
                            'token_nll_difference':records[label]['token_weighted_nll']-records['native']['token_weighted_nll']}
    hashes={r['backbone_sha256'] for r in records.values()}
    checks={'same_original_backbone':hashes=={export_desc['source']['original_backbone_sha256']},
            'four_fresh_processes':len({r['pid'] for r in records.values()})==4,
            'native_reload_bitwise_equal':torch.equal(native.float(),load_file(str(a.out/'native_repeat/logits.safetensors'))['last_supervised_position_logits']),
            'factor_reconstruction':max(errors.values())<=policy['exact_factor_relative_error_max'],
            'exact_relative_rms':comparisons['exact']['logits_relative_rms']<=policy['exact_logits_relative_rms_max'],
            'exact_absolute_logits':comparisons['exact']['logits_max_absolute_difference']<=policy['exact_logits_absolute_error_max'],
            'exact_nll':abs(comparisons['exact']['group_nll_difference'])<=policy['exact_group_nll_abs_difference_max']}
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'policy':policy,
            'comparisons':comparisons,'factor_relative_errors':errors,'probe_metrics':records,
            'source':str(source),'export':str(a.export),'exact_refactor_diagnostic_above_budget':True}
    json_write(a.out/'audit.json',result);print(json.dumps({k:result[k] for k in ['status','checks','comparisons']}),flush=True)
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',type=Path);p.add_argument('--adapter',type=Path)
    p.add_argument('--emit',action='store_true');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B')
    p.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'))
    p.add_argument('--backbone-arithmetic',choices=['bf16','fp32_oracle'],default='bf16')
    a=p.parse_args()
    emit(a) if a.emit else compare(a)
