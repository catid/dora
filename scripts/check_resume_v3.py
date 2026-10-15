#!/usr/bin/env python3
"""Compare uninterrupted/resumed checkpoints, including scores and RNG state."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('uninterrupted',type=Path); ap.add_argument('resumed',type=Path)
    ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    a=load_file(str(args.uninterrupted/'adapter.safetensors')); b=load_file(str(args.resumed/'adapter.safetensors'))
    sa=torch.load(args.uninterrupted/'training_state.pt',map_location='cpu',weights_only=True)
    sb=torch.load(args.resumed/'training_state.pt',map_location='cpu',weights_only=True)
    checks={'adapter_tensors_bitwise_equal':equal(a,b),'optimizer_state_bitwise_equal':equal(sa['optimizer'],sb['optimizer']),
            'score_ema_bitwise_equal':equal(sa['extra']['ema'],sb['extra']['ema']),
            'rng_state_bitwise_equal':all(equal(sa[k],sb[k]) for k in ['torch_rng','cuda_rng','python_rng']),
            'pruning_events_equal':equal(sa['extra']['pruning_events'],sb['extra']['pruning_events']),
            'exact_exposure_equal':all(sa['extra'][k]==sb['extra'][k] for k in ['input_tokens','supervised_tokens'])}
    result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,
            'uninterrupted':str(args.uninterrupted),'resumed':str(args.resumed),
            'backend':sa['extra']['resume_invariants']['backend'],'endpoint_step':sa['extra']['step'],
            'compaction_exercised':sa['extra']['compacted']}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not all(checks.values()): raise SystemExit(1)


if __name__=='__main__': main()
