#!/usr/bin/env python3
"""Run the separate-process common-export audits for Stage-D roots."""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out-root',type=Path,required=True); ap.add_argument('--gpus',default='0,1,2,3')
    ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B')
    ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'))
    args=ap.parse_args(); args.out_root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for run in sorted(args.root.iterdir()):
        adapter=run/'ordinary_export'; md=run/'metadata.json'
        if not adapter.exists() or not md.exists(): continue
        jobs.append({'run':run.name,'adapter':str(adapter),'out':str(args.out_root/run.name)})
    (args.out_root/'plan.json').write_text(json.dumps({'root':str(args.root),'jobs':jobs},indent=2)+'\n')
    rng=random.Random(20261026); rng.shuffle(jobs); gpus=[x.strip() for x in args.gpus.split(',') if x.strip()]
    free=list(gpus); active={}; i=0
    while i<len(jobs) or active:
        while i<len(jobs) and free:
            job=jobs[i]; i+=1; gpu=free.pop(0); out=Path(job['out'])
            # check_export_v3 owns the audit directory and intentionally
            # creates it with exist_ok=False. Keep launcher receipts/logs next
            # to it rather than pre-creating the audit output.
            status=args.out_root/(job['run']+'.launcher_status.json')
            if status.exists() and json.loads(status.read_text()).get('status')=='complete':
                free.append(gpu); free.sort(key=gpus.index); continue
            cmd=[sys.executable,'scripts/check_export_v3.py','--export',job['adapter'],'--out',str(out),
                 '--model',args.model,'--data-root',str(args.data_root)]
            env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':str(Path.cwd()/'src'),
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HOME':env.get('HF_HOME','/var/tmp/dora-adaptive-rank/hf')})
            log=(args.out_root/(job['run']+'.stdout.log')).open('w'); p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            status.write_text(json.dumps({'status':'running','pid':p.pid,'gpu':gpu,'command':cmd},indent=2)+'\n')
            active[p.pid]=(p,log,job,gpu)
        time.sleep(1)
        for pid,(p,log,job,gpu) in list(active.items()):
            rc=p.poll()
            if rc is None: continue
            log.close(); del active[pid]; free.append(gpu); free.sort(key=gpus.index)
            status.write_text(json.dumps({'status':'complete' if rc==0 else 'failed','returncode':rc,'gpu':gpu},indent=2)+'\n')
    print(json.dumps({'status':'complete','jobs':len(jobs),'out_root':str(args.out_root)}))

if __name__=='__main__': main()
