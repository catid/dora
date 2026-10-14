#!/usr/bin/env python3
"""Run the registered native HPO cells round-robin on available GPUs.

Restart/compression pipelines are deliberately excluded until their dedicated
pipeline implementations land; they remain blocked in coverage rather than
being mislabeled as native runs.
"""
from __future__ import annotations
import argparse, itertools, json, os, random, subprocess, time
from pathlib import Path
import yaml
from rankstudy.registry import build_registry

SUPPORTED_GROUPS={"fixed","wide","reference","online","allocation_control","triplet","triplet_ortho","online_uncertainty"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpus',default='0,1,2,3'); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--updates',type=int,default=128); ap.add_argument('--resume',action='store_true'); ap.add_argument('--root',default='/var/tmp/dora-adaptive-rank/runs/hpo'); ap.add_argument('--python',default='.venv/bin/python'); args=ap.parse_args()
    cfg=yaml.safe_load(Path('configs/study.yaml').read_text()); root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for spec in build_registry():
        if spec.group not in SUPPORTED_GROUPS: continue
        for lr,scale in itertools.product(cfg['peak_learning_rates'],cfg['forward_scales']):
            rows.append((spec,lr,scale))
    random.Random(1337).shuffle(rows)
    if args.limit: rows=rows[:args.limit]
    gpus=[x.strip() for x in args.gpus.split(',') if x.strip()]
    pending=[]; complete=[]
    for spec,lr,scale in rows:
        out=root/f'{spec.id}_lr{lr:g}_s{scale:g}'
        meta=out/'metadata.json'
        if args.resume and meta.exists():
            try:
                if json.loads(meta.read_text()).get('status')=='complete': complete.append(str(out)); continue
            except Exception: pass
        rank=8 if spec.group=='fixed' or (spec.group=='triplet' and spec.allocator=='none') or (spec.group=='triplet_ortho' and spec.allocator=='none') else 16
        pending.append((spec,lr,scale,out,rank))
    active={}; idx=0; free_gpus=list(gpus)
    while idx<len(pending) or active:
        while idx<len(pending) and free_gpus:
            spec,lr,scale,out,rank=pending[idx]; idx+=1
            out.mkdir(parents=True,exist_ok=True)
            cmd=[args.python,'scripts/run_one.py','--method',spec.id,'--init',spec.init,'--scale',str(scale),'--lr',str(lr),'--seed',str(cfg['seed_hpo']),'--updates',str(args.updates),'--rank',str(rank),'--target-rank',str(cfg['final_average_rank']),'--out',str(out)]
            gpu=free_gpus.pop(0)
            env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu; env['PYTHONPATH']='src'; env.setdefault('HF_HOME','/var/tmp/dora-adaptive-rank/hf')
            log=(out/'stdout.log').open('w')
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[p.pid]=(p,log,out,spec.id,lr,scale,gpu)
            print(json.dumps({'event':'launch','pid':p.pid,'method':spec.id,'lr':lr,'scale':scale,'gpu':gpu}),flush=True)
        time.sleep(1)
        for pid,(p,log,out,mid,lr,scale,gpu) in list(active.items()):
            rc=p.poll()
            if rc is None: continue
            log.close(); del active[pid]
            free_gpus.append(gpu); free_gpus.sort(key=gpus.index)
            status='complete' if rc==0 and (out/'metadata.json').exists() else 'failed'
            (out/'launcher_status.json').write_text(json.dumps({'status':status,'returncode':rc,'method':mid,'lr':lr,'scale':scale,'gpu':gpu},indent=2)+'\n')
            print(json.dumps({'event':'finish','pid':pid,'method':mid,'status':status,'gpu':gpu}),flush=True)
    print(json.dumps({'submitted':len(rows),'already_complete':len(complete),'root':str(root)}))
if __name__=='__main__': main()
