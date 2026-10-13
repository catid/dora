"""Check full-model initial logits and RNG for the frozen initializer fixture."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments import extraction as task
from experiments import init_common as common


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    manifest=common.read(root/'manifest.json')
    for name,digest in manifest['source_sha256'].items():
        assert common.sha(name)==digest and common.sha(root/'source'/name)==digest
    assert Path(task.__file__).resolve().is_relative_to(root/'source')
    assert Path(common.__file__).resolve().is_relative_to(root/'source')
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(task.MODEL,revision=task.MODEL_REVISION)
    tokenizer.pad_token=tokenizer.eos_token
    records,provenance=task.data(SimpleNamespace(**manifest['configuration'],output=root),tokenizer)
    assert provenance==manifest['dataset']
    calibration=common.read(root/'calibration/manifest.json')
    model=AutoModelForCausalLM.from_pretrained(task.MODEL,revision=task.MODEL_REVISION,
             dtype=torch.bfloat16,attn_implementation='sdpa').cuda().requires_grad_(False)
    frozen=task.frozen_digest(model)
    candidates=manifest['registry']['candidates'];assert candidates[0]['candidate_id']==manifest['registry']['control_id']
    baseline=None;checks=[]
    batch=task.batch(records['train'][:2],tokenizer)
    for candidate in candidates:
        backend=common.backend_for(candidate,42,calibration)
        task.strip_adapters(model,backend.adapter_types);task.seed_all(42)
        backend.inject_adapters(model,candidate['candidate_id'],rank=8,targets=('q_proj','v_proj'))
        assert task.frozen_digest(model)==frozen
        rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(**batch,use_cache=False)
            logits=output.logits.detach().cpu();loss=float(output.loss)
        if baseline is None:baseline={'logits':logits,'loss':loss,'rng':rng}
        assert torch.equal(baseline['logits'],logits) and baseline['loss']==loss
        for name,state in rng.items():assert torch.equal(state,baseline['rng'][name])
        checks.append({'candidate_id':candidate['candidate_id'],'recipe':backend.recipe_manifest(candidate['candidate_id']),
            'loss':loss,'logits_shape':list(logits.shape),'logits_sha256':common.tensor_digest([('logits',logits)]),
            'rng_sha256':common.tensor_digest(rng.items()),'full_logits_bitwise_accepted':True,
            'training_rng_bitwise_accepted':True,'frozen_parameters_unchanged':True})
        print(json.dumps({'candidate_id':candidate['candidate_id'],'initial_logits':'bitwise','loss':loss}),flush=True)
    source=root/'audit_source'/Path(__file__).name;source.parent.mkdir(parents=True,exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    common.write(root/'initial_function_audit.json',{'status':'pass','seed':42,'probe_split':'train',
        'probe_ids':[row['id'] for row in records['train'][:2]],'manifest_sha256':common.sha(root/'manifest.json'),
        'source_sha256':common.sha(__file__),'checks':checks,
        'scope':'Independent full-model BF16 initial forward and original training RNG equality for four frozen fixture recipes. No optimizer step or test examples.'})


if __name__=='__main__':main()
