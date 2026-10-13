"""Audit completed recipe endpoints against the measured historical task runs."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('/var/tmp/dora-bench/change_ablation_runs/aircraft_rank2'))
parser.add_argument('--method',choices=('ab_original','ab_revised'),required=True)
args=parser.parse_args()
reference_root=Path('/var/tmp/dora-bench/original_bench/aircraft_rank2' if args.method=='ab_original' else '/var/tmp/dora-bench/round2/vision/rank2_run')
reference_method='dora_original' if args.method=='ab_original' else 'dora'
candidate_root=args.root/args.method
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selected=read(candidate_root/'rank_2/selected.json')[args.method]
assert selected==read(reference_root/'rank_2/selected.json')[reference_method]
fits=[];inputs={}
for stage in ('pilots','final'):
 candidate=read(candidate_root/('rank_2/pilot_results.json' if stage=='pilots' else 'results.json'))
 previous=read(reference_root/('rank_2/pilot_results.json' if stage=='pilots' else 'results.json'))
 for row in candidate:
  if row['method']!=args.method:continue
  match=[other for other in previous if other['method']==reference_method and other['seed']==row['seed'] and other['adapter_learning_rate']==row['adapter_learning_rate']]
  assert len(match)==1
  other=match[0]
  suffix=f"lr_{row['adapter_learning_rate']:g}_m_1" if stage=='pilots' else f"seed_{row['seed']}"
  old=reference_root/'rank_2'/stage/reference_method/suffix
  new=candidate_root/'rank_2'/stage/args.method/suffix
  for key in ('seed','epochs','batch_size','adapter_learning_rate','best_validation','test','best_epoch','initial_head_sha256','initial_backbone_sha256','trainable_parameters'):
   assert row[key]==other[key],(stage,suffix,key)
  a=[json.loads(line) for line in (old/'epochs.jsonl').read_text().splitlines()]
  b=[json.loads(line) for line in (new/'epochs.jsonl').read_text().splitlines()]
  assert len(a)==len(b)==row['epochs']
  for left,right in zip(a,b):
   for key in ('epoch','learning_rates','train','validation','best_epoch'):assert left[key]==right[key],(stage,suffix,key)
  left=torch.load(old/'best_adapter_and_head.pt',weights_only=True,map_location='cpu')
  right=torch.load(new/'best_adapter_and_head.pt',weights_only=True,map_location='cpu')
  renamed={}
  for name,value in left.items():
   if args.method=='ab_original':
    if name.endswith('.lora_A'):name=name[:-6]+'lora_B'
    elif name.endswith('.lora_B'):name=name[:-6]+'lora_A'
   renamed[name]=value
  assert set(renamed)==set(right) and all(torch.equal(renamed[name],right[name]) for name in renamed),(stage,suffix)
  names=['result.json','epochs.jsonl','best_adapter_and_head.pt']
  if stage=='final':
   assert (old/'test_predictions.json').read_bytes()==(new/'test_predictions.json').read_bytes()
   names.append('test_predictions.json')
  for directory in (old,new):
   for name in names:inputs[str(directory/name)]=sha(directory/name)
  fits.append({'stage':stage,'seed':row['seed'],'lr':row['adapter_learning_rate'],'epochs':row['epochs'],
               'all_epoch_metrics_and_learning_rates_exact':True,'selected_checkpoint_tensors_bitwise_equal':len(right),
               'test_predictions_byte_identical':stage=='final'})
assert len(fits)==7
result={'passed':True,'candidate':args.method,'reference':reference_method,'selected_lr':selected['lr'],
        'fits':fits,'input_sha256':inputs,'source_sha256':sha(Path(__file__)),
        'scope':'Full four-pilot and three-final task protocol; all epoch metrics, selected checkpoints, test predictions exactly reproduce the historical endpoint. Factor names are mapped for original; elapsed times are excluded.'}
output=args.root/f'{args.method}_historical_endpoint_control.json'
output.write_text(json.dumps(result,indent=2)+'\n')
args.root.joinpath('historical_endpoint_control_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'passed':True,'method':args.method,'fits':7,'epochs':sum(row['epochs'] for row in fits)}))
