"""Compare the first full recipe pilot with the historical original DoRA run."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',type=Path,default=Path('/var/tmp/dora-bench/original_bench/aircraft_rank2/rank_2/pilots/dora_original/lr_1e-05_m_1'))
parser.add_argument('--candidate',type=Path,default=Path('/var/tmp/dora-bench/change_ablation_runs/aircraft_rank2/ab_original/rank_2/pilots/ab_original/lr_1e-05_m_1'))
parser.add_argument('--output',type=Path,default=Path('/var/tmp/dora-bench/change_ablation_runs/aircraft_rank2/historical_pilot_control.json'))
args=parser.parse_args()
read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
a,b=read(args.reference/'epochs.jsonl'),read(args.candidate/'epochs.jsonl')
assert len(a)==len(b)==10
for left,right in zip(a,b):
    for key in ('epoch','learning_rates','train','validation','best_epoch'):
        assert left[key]==right[key],key
left=torch.load(args.reference/'best_adapter_and_head.pt',weights_only=True,map_location='cpu')
right=torch.load(args.candidate/'best_adapter_and_head.pt',weights_only=True,map_location='cpu')
renamed={}
for name,value in left.items():
    if name.endswith('.lora_A'):name=name[:-6]+'lora_B'
    elif name.endswith('.lora_B'):name=name[:-6]+'lora_A'
    renamed[name]=value
assert set(renamed)==set(right)
assert all(torch.equal(renamed[k],right[k]) for k in renamed)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'passed':True,'pilot_lr':1e-5,'epochs':10,'seed':42,'training_batch_size':128,
        'all_epoch_metrics_and_learning_rates_exact':True,
        'all_checkpoint_tensors_bitwise_equal_after_factor_name_mapping':True,'checkpoint_tensors':len(right),
        'factor_name_mapping':'historical lora_A(up)->recipe lora_B; historical lora_B(down)->recipe lora_A',
        'inputs':{str(p):sha(p) for directory in (args.reference,args.candidate) for p in (directory/'epochs.jsonl',directory/'result.json',directory/'best_adapter_and_head.pt')},
        'source_sha256':sha(Path(__file__)),
        'scope':'Full official Aircraft training/validation pilot, zero test evaluation; excludes elapsed-time equality.'}
args.output.write_text(json.dumps(record,indent=2)+'\n')
args.output.with_name('historical_pilot_control_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'passed':True,'checkpoint_tensors':len(right)}))
