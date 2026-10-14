#!/usr/bin/env python3
"""Build immutable CSV summaries from native run directories."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict

def read_rows(root):
 rows=[]
 for d in Path(root).iterdir() if Path(root).exists() else []:
  c=d/'config.json'; m=d/'metadata.json'; s=d/'launcher_status.json'
  if not c.exists(): continue
  cf=json.loads(c.read_text()); md=json.loads(m.read_text()) if m.exists() else {}
  st=json.loads(s.read_text()) if s.exists() else {}
  rows.append({**cf,**md,'run_dir':str(d),'launcher_status':st.get('status','pending')})
 return rows

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
 rows=read_rows(args.root); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
 fields=sorted({k for r in rows for k in r})
 with out.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 by=defaultdict(list)
 for r in rows:
  if r.get('status')=='complete' and r.get('dev_tune_endpoint_nll') is not None: by[r.get('method')].append(r)
 best=[]
 for method,rs in sorted(by.items()):
  x=min(rs,key=lambda r:float(r['dev_tune_endpoint_nll'])); best.append({'method':method,'n':len(rs),'best_dev_tune_nll':float(x['dev_tune_endpoint_nll']),'lr':x.get('lr'),'scale':x.get('scale'),'seed':x.get('seed')})
 (out.parent/(out.stem+'_best.json')).write_text(json.dumps(best,indent=2)+'\n')
 print(json.dumps({'rows':len(rows),'complete':sum(r.get('status')=='complete' for r in rows),'best_methods':len(best)}))
if __name__=='__main__': main()
