#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--plan',default='artifacts/run_plan.json'); ap.add_argument('--roots',nargs='+',required=True); ap.add_argument('--out',default='artifacts/coverage.json'); args=ap.parse_args()
 p=json.loads(Path(args.plan).read_text()); rows=[]
 for run in p['runs']:
  rid=run['run_id']; status='pending'; reason=''
  for root in args.roots:
   prefixes=[rid]
   if rid.startswith('hpo_'): prefixes.append(rid[4:])
   if rid.startswith('tune_'): prefixes.append(rid[5:])
   for pref in prefixes:
    for d in Path(root).glob(f'{pref}*'):
     m=d/'metadata.json'; s=d/'launcher_status.json'
     if m.exists() and json.loads(m.read_text()).get('status')=='complete': status='complete'; break
     if s.exists() and json.loads(s.read_text()).get('status')=='failed': status='failed'; reason=json.loads(s.read_text()).get('returncode')
   if status!='pending': break
  rows.append({'run_id':rid,'stage':run.get('stage'),'method_id':run.get('method_id'),'status':status,'failure_reason':reason})
 counts={x:sum(r['status']==x for r in rows) for x in ('pending','complete','failed','blocked','not_in_scope')}
 out={'schema':'adaptive-rank-coverage-v1','plan_sha256':hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),'counts':{'total':len(rows),**counts},'runs':rows}
 Path(args.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['counts']))
if __name__=='__main__': main()
