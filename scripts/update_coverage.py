#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import sys
sys.path.insert(0,'src')
from rankstudy.registry import build_registry

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--plan',default='artifacts/run_plan.json'); ap.add_argument('--roots',nargs='+',required=True); ap.add_argument('--out',default='artifacts/coverage.json'); args=ap.parse_args()
 p=json.loads(Path(args.plan).read_text()); rows=[]
 blocked_methods={x.id for x in build_registry() if x.group in {'restart','restart_control','compress'}}
 for run in p['runs']:
  rid=run['run_id']; status='pending'; reason=''
  if run.get('method_id') in blocked_methods:
   status='blocked'; reason='pipeline_not_implemented_in_current_scope'
  for root in args.roots:
   root_path = Path(root)
   # Output directory names are intentionally more detailed than plan IDs:
   # selected tuning runs use METHOD_c{1,2}_seedNNNN, while the plan uses
   # tune_METHOD_NNNN_a. Match by stage/method/seed instead of brittle string
   # slicing (method IDs themselves contain underscores).
   candidates=[]
   if run.get('stage') == 'B_hpo':
    candidates = [root_path / rid, root_path / rid.removeprefix('hpo_')]
   elif run.get('stage') == 'B_tuning':
    candidates = list(root_path.glob(f"{run['method_id']}_c*_seed{run['seed']}"))
   elif run.get('stage') == 'B_screen':
    candidates = list(root_path.glob(f"{run['method_id']}_seed{run['seed']}"))
   else:
    candidates = [root_path / rid]
   for d in candidates:
    m=d/'metadata.json'; s=d/'launcher_status.json'
    if m.exists():
     try: md=json.loads(m.read_text())
     except json.JSONDecodeError: md={}
     if md.get('status') == 'complete':
      status='complete'; break
     if md.get('status') in {'failed','error'}:
      status='failed'; reason=md.get('failure_reason', md.get('returncode','metadata_failed'))
    if s.exists():
     try: sd=json.loads(s.read_text())
     except json.JSONDecodeError: sd={}
     if sd.get('status') == 'failed':
      status='failed'; reason=sd.get('returncode','launcher_failed')
   if status != 'pending': break
   if status not in {'pending'}: break
  rows.append({'run_id':rid,'stage':run.get('stage'),'method_id':run.get('method_id'),'status':status,'failure_reason':reason})
 counts={x:sum(r['status']==x for r in rows) for x in ('pending','complete','failed','blocked','not_in_scope')}
 out={'schema':'adaptive-rank-coverage-v1','plan_sha256':hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),'counts':{'total':len(rows),**counts},'runs':rows}
 Path(args.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['counts']))
if __name__=='__main__': main()
