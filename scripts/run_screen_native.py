#!/usr/bin/env python3
"""Run three fresh native screening seeds from frozen tuning selections."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
from collections import defaultdict
from rankstudy.registry import build_registry

SUPPORTED={"fixed","wide","reference","online","allocation_control","triplet","triplet_ortho","online_uncertainty"}

def select(tune_root):
 by=defaultdict(list)
 for d in Path(tune_root).iterdir() if Path(tune_root).exists() else []:
  c=d/'config.json'; m=d/'metadata.json'
  if c.exists() and m.exists():
   cf=json.loads(c.read_text()); md=json.loads(m.read_text())
   if md.get('status')=='complete' and md.get('dev_tune_endpoint_nll') is not None: by[cf['method']].append((float(md['dev_tune_endpoint_nll']),cf))
 out={}
 for mid,rows in by.items():
  rows.sort(key=lambda x:x[0]); out[mid]=rows[0][1]
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tune-root',required=True); ap.add_argument('--out-root',required=True); ap.add_argument('--gpus',default='0,1,2,3'); ap.add_argument('--updates',type=int,default=512); ap.add_argument('--seeds',default='2001,2002,2003'); ap.add_argument('--python',default='.venv/bin/python'); args=ap.parse_args()
 specmap={x.id:x for x in build_registry()}; chosen=select(args.tune_root); jobs=[]
 for mid,cf in sorted(chosen.items()):
  spec=specmap[mid]
  if spec.group not in SUPPORTED: continue
  rank=8 if spec.group=='fixed' or (spec.group in {'triplet','triplet_ortho'} and spec.allocator=='none') else 16
  for seed in [int(x) for x in args.seeds.split(',')]: jobs.append({'method':mid,'init':spec.init,'lr':cf['lr'],'scale':cf['scale'],'seed':seed,'rank':rank,'target_rank':8,'updates':args.updates,'out':str(Path(args.out_root)/f'{mid}_seed{seed}')})
 gpus=[x.strip() for x in args.gpus.split(',') if x.strip()]; free=list(gpus); active={}; i=0; Path(args.out_root).mkdir(parents=True,exist_ok=True)
 while i<len(jobs) or active:
  while i<len(jobs) and free:
   j=jobs[i]; i+=1; out=Path(j['out']); out.mkdir(parents=True,exist_ok=True)
   cmd=[args.python,'scripts/run_one.py']
   for k in ('method','init','scale','lr','seed','updates','rank','target_rank','out'): cmd += [f'--{k.replace("_","-")}',str(j[k])]
   gpu=free.pop(0); env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':'src','HF_HOME':os.environ.get('HF_HOME','/var/tmp/dora-adaptive-rank/hf')})
   log=(out/'stdout.log').open('w'); p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT); active[p.pid]=(p,log,j,gpu)
   print(json.dumps({'event':'launch','pid':p.pid,'gpu':gpu,'method':j['method'],'seed':j['seed']}),flush=True)
  time.sleep(1)
  for pid,(p,log,j,gpu) in list(active.items()):
   rc=p.poll()
   if rc is None: continue
   log.close(); del active[pid]; free.append(gpu); free.sort(key=gpus.index)
   (Path(j['out'])/'launcher_status.json').write_text(json.dumps({'status':'complete' if rc==0 else 'failed','returncode':rc,'stage':'screening','gpu':gpu},indent=2)+'\n')
   print(json.dumps({'event':'finish','pid':pid,'status':'complete' if rc==0 else 'failed','gpu':gpu,'method':j['method'],'seed':j['seed']}),flush=True)
 print(json.dumps({'jobs':len(jobs),'methods':len(chosen),'out_root':args.out_root}))
if __name__=='__main__': main()
