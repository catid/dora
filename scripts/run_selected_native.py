#!/usr/bin/env python3
"""Full-horizon native tuning for every supported registered cell.

Selects the best two distinct HPO (endpoint dev_tune) configurations per cell,
then runs each from scratch on tuning seeds 1002 and 1003.
"""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
from collections import defaultdict
from rankstudy.registry import build_registry

SUPPORTED={"fixed","wide","reference","online","allocation_control","triplet","triplet_ortho","online_uncertainty"}

def load_jobs(hpo_root, seed_list, out_root, updates, resume=False):
  by=defaultdict(list)
  for d in Path(hpo_root).iterdir():
    c=d/'config.json'; m=d/'metadata.json'
    if not c.exists() or not m.exists(): continue
    cf=json.loads(c.read_text()); md=json.loads(m.read_text())
    if md.get('status')!='complete': continue
    by[cf['method']].append((float(md.get('dev_tune_endpoint_nll',1e9)),cf))
  specmap={x.id:x for x in build_registry()}
  jobs=[]
  for mid,vals in sorted(by.items()):
    if mid not in specmap or specmap[mid].group not in SUPPORTED: continue
    vals.sort(key=lambda z:(z[0],z[1]['lr'],z[1]['scale']))
    chosen=[]
    for score,cf in vals:
      key=(cf['lr'],cf['scale'])
      if key not in {(x['lr'],x['scale']) for x in chosen}:
        chosen.append(cf)
      if len(chosen)>=2: break
    spec=specmap[mid]
    rank=8 if spec.group in {'fixed'} or (spec.group in {'triplet','triplet_ortho'} and spec.allocator=='none') else 16
    for ci,cf in enumerate(chosen):
      for seed in seed_list:
        out=Path(out_root)/f'{mid}_c{ci+1}_seed{seed}'
        if resume and (out/'metadata.json').exists(): continue
        jobs.append({'method':mid,'init':spec.init,'lr':cf['lr'],'scale':cf['scale'],'seed':seed,'rank':rank,'target_rank':8,'updates':updates,'out':str(out)})
  return jobs

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--hpo-root',required=True); ap.add_argument('--out-root',required=True); ap.add_argument('--gpus',default='0,1,2,3'); ap.add_argument('--updates',type=int,default=512); ap.add_argument('--seeds',default='1002,1003'); ap.add_argument('--python',default='.venv/bin/python'); ap.add_argument('--microbatch',type=int,default=1); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
 jobs=load_jobs(args.hpo_root,[int(x) for x in args.seeds.split(',')],args.out_root,args.updates,args.resume); Path(args.out_root).mkdir(parents=True,exist_ok=True)
 gpus=[x.strip() for x in args.gpus.split(',') if x.strip()]; free=list(gpus); active={}; i=0
 while i<len(jobs) or active:
  while i<len(jobs) and free:
   j=jobs[i]; i+=1; out=Path(j['out']); out.mkdir(parents=True,exist_ok=True)
   cmd=[args.python,'scripts/run_one.py']
   for k in ('method','init','scale','lr','seed','updates','rank','target_rank','out'): cmd += [f'--{k.replace("_","-")}',str(j[k])]
   cmd += ['--microbatch',str(args.microbatch)]
   gpu=free.pop(0); env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':'src','HF_HOME':os.environ.get('HF_HOME','/var/tmp/dora-adaptive-rank/hf')})
   log=(out/'stdout.log').open('w'); p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
   active[p.pid]=(p,log,j,gpu); print(json.dumps({'event':'launch','pid':p.pid,'gpu':gpu,**{k:j[k] for k in ('method','lr','scale','seed')}}),flush=True)
  time.sleep(1)
  for pid,(p,log,j,gpu) in list(active.items()):
   rc=p.poll()
   if rc is None: continue
   log.close(); del active[pid]; free.append(gpu); free.sort(key=gpus.index)
   (Path(j['out'])/'launcher_status.json').write_text(json.dumps({'status':'complete' if rc==0 else 'failed','returncode':rc,'stage':'full_tuning','gpu':gpu},indent=2)+'\n')
   print(json.dumps({'event':'finish','pid':pid,'status':'complete' if rc==0 else 'failed','gpu':gpu,'method':j['method'],'seed':j['seed']}),flush=True)
 print(json.dumps({'jobs':len(jobs),'out_root':args.out_root}))
if __name__=='__main__': main()
