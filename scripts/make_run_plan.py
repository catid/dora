#!/usr/bin/env python3
import json,hashlib,datetime
from pathlib import Path
import yaml
from rankstudy.registry import build_registry

def main():
 cfg=yaml.safe_load(Path('configs/study.yaml').read_text()); rows=[]
 for m in build_registry():
  for lr in cfg['peak_learning_rates']:
   for scale in cfg['forward_scales']:
    rows.append({'run_id':f'hpo_{m.id}_lr{lr:g}_s{scale:g}','stage':'B_hpo','method_id':m.id,'seed':1001,'updates':128,'lr':lr,'scale':scale,'status':'pending'})
  for seed in cfg['tuning_seeds']:
   rows.append({'run_id':f'tune_{m.id}_{seed}_a','stage':'B_tuning','method_id':m.id,'seed':seed,'updates':512,'status':'pending','depends_on':f'hpo_{m.id}'})
  for seed in cfg['screening_seeds']:
   rows.append({'run_id':f'screen_{m.id}_{seed}','stage':'B_screen','method_id':m.id,'seed':seed,'updates':512,'status':'pending'})
 # Required confirmation, rank4, export, compute, larger model, hetero targets are declared now.
 for m in build_registry():
  for seed in cfg['confirmation_seeds']:
   rows.append({'run_id':f'confirm_{m.id}_{seed}','stage':'C_confirmation','method_id':m.id,'seed':seed,'updates':512,'status':'pending'})
 for m in ['L_FIXED','I_FIXED','ADALORA_REF','L_RAW','L_GATE','I_RAW','I_GATE']:
  for seed in [4001,4002,4003]: rows.append({'run_id':f'rank4_{m}_{seed}','stage':'C_rank4','method_id':m,'seed':seed,'rank':4,'max_rank':8,'updates':512,'status':'pending'})
 for m in ['L_FIXED','I_FIXED','ADALORA_REF']:
  for seed in [5001,5002,5003]: rows.append({'run_id':f'export_{m}_{seed}','stage':'D_export','method_id':m,'seed':seed,'updates':512,'status':'pending'})
 for m in ['L_FIXED','I_FIXED','ADALORA_REF']:
  for seed in [6001,6002,6003]: rows.append({'run_id':f'compute_{m}_{seed}','stage':'D_compute','method_id':m,'seed':seed,'updates':512,'status':'pending'})
 for m in ['L_FIXED','I_FIXED','ADALORA_REF']:
  for seed in [7001,7002,7003]: rows.append({'run_id':f'7b_{m}_{seed}','stage':'E_7b','method_id':m,'seed':seed,'updates':1024,'model':'Qwen/Qwen2.5-Math-7B','status':'pending'})
 for m in ['L_FIXED','I_FIXED','ADALORA_REF']:
  for seed in [8001,8002,8003]: rows.append({'run_id':f'hetero_{m}_{seed}','stage':'E_hetero','method_id':m,'seed':seed,'updates':512,'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],'status':'pending'})
 plan={'schema':'adaptive-rank-run-plan-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_gpus':4,'study_config':cfg,'method_count':len(build_registry()),'runs':rows,'coverage':{'total':len(rows),'pending':len(rows),'complete':0,'failed':0,'blocked':0,'not_in_scope':0}}
 out=Path('artifacts/run_plan.json'); out.write_text(json.dumps(plan,indent=2)+'\n'); print(len(rows),hashlib.sha256(out.read_bytes()).hexdigest())
 Path('artifacts/coverage.json').write_text(json.dumps({'schema':'adaptive-rank-coverage-v1','plan_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'counts':plan['coverage'],'runs':[]},indent=2)+'\n')
if __name__=='__main__': main()
