#!/usr/bin/env python3
"""Balanced 12-trial v3 HPO queue for all 41 registered cells.

Every trial is an immutable subprocess record.  Native and multi-stage cells
use their audited v3 runners; no v2 implementation or quarantined result is
read.  The launcher does not select a winner; it only produces the declared
short-run records for the subsequent tuning stage.
"""
from __future__ import annotations
import argparse, concurrent.futures, fcntl, json, os, random, subprocess, sys, time
from pathlib import Path
import yaml


def trial_design(method):
    trials=[]
    for i,lr in enumerate((1e-6,1e-5,1e-4,1e-3)):
        for j,scale in enumerate((.25,1.,4.)):
            penalty=0.
            if method.endswith('_ORTHO'): penalty=(1e-4,1e-3,1e-2)[(i+j)%3]
            if method=='ADALORA_REF': penalty=(.01,.1,1.)[(i+j)%3]
            canonical=False
            if method=='ADALORA_REF' and i==3 and j==1:
                lr,penalty,canonical=8e-4,.1,True
            trials.append({'trial_index':len(trials),'lr':lr,'scale':scale,'penalty':penalty,
                           'canonical_reference_replacement':canonical})
    return trials


def make_jobs(root, methods, updates, seed, target_rank, max_rank):
    jobs=[]
    for spec in methods:
        method=spec['id']
        for trial in trial_design(method):
            out=root/f"{method}_c{trial['trial_index']:02d}"
            jobs.append({**trial,'method':method,'seed':seed,'updates':updates,
                         'target_rank':target_rank,'max_rank':max_rank,'out':str(out),
                         'pipeline':spec['group'] in {'restart','restart_control','compress'}})
    return jobs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--gpus',default='0,1,2,3'); ap.add_argument('--updates',type=int,default=128)
    ap.add_argument('--seed',type=int,default=1001); ap.add_argument('--target-rank',type=int,default=8)
    ap.add_argument('--max-rank',type=int,default=16); ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B')
    ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'))
    ap.add_argument('--manifest-root',type=Path,default=Path('artifacts/manifests_v2')); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--targets',nargs='+',default=['q_proj','o_proj'])
    ap.add_argument('--method-ids',default='',help='Optional comma-separated registered subset; default is all methods.')
    args=ap.parse_args(); args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'.queue.lock').open('a+'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=json.loads(Path('artifacts/protocol_execution_gate.json').read_text())
    if not gate.get('v3_large_sweep_allowed'): raise RuntimeError('v3 primary execution gate is closed')
    methods=yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']
    if args.method_ids:
        wanted=set(x.strip() for x in args.method_ids.split(',') if x.strip())
        methods=[m for m in methods if m['id'] in wanted]
        if len(methods)!=len(wanted): raise ValueError(f'unknown method id in --method-ids: {sorted(wanted-{m["id"] for m in methods})}')
    else:
        # Exploratory hybrids are opt-in and never silently enlarge the
        # preregistered 41-cell primary tournament.
        methods=[m for m in methods if m.get('group')!='hybrid']
    jobs=make_jobs(args.root,methods,args.updates,args.seed,args.target_rank,args.max_rank)
    rng=random.Random(20261015); rng.shuffle(jobs)
    plan_path=args.root/'plan.json'
    plan={'schema':'rankstudy-v3-hpo-queue','method_count':len(methods),'trial_count':len(jobs),
          'methods':[m['id'] for m in methods],'updates':args.updates,'search_seed':args.seed,
          'target_rank':args.target_rank,'max_rank':args.max_rank,'targets':args.targets,
          'trial_design':'lr={1e-6,1e-5,1e-4,1e-3} x scale={.25,1,4}; balanced declared penalty cycle',
          'selection_eligible':False,'jobs':jobs}
    if plan_path.exists():
        old=json.loads(plan_path.read_text())
        if old.get('trial_count')!=len(jobs) or old.get('methods')!=[m['id'] for m in methods]:
            raise ValueError('immutable HPO plan changed')
    else:
        plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    pending=[]
    for job in jobs:
        out=Path(job['out']); out.mkdir(parents=True,exist_ok=True)
        if args.resume and (out/'metadata.json').exists():
            md=json.loads((out/'metadata.json').read_text())
            if md.get('status') in {'complete','failed'}: continue
        config={k:v for k,v in job.items() if k!='out'}; (out/'hpo_config.json').write_text(json.dumps(config,indent=2)+'\n')
        pending.append(job)
    gpu_ids=[x.strip() for x in args.gpus.split(',') if x.strip()]; free=list(gpu_ids); active={}; index=0
    def launch(job,gpu):
        out=Path(job['out']); cmd=[sys.executable,'scripts/run_pipeline_v3.py' if job['pipeline'] else 'scripts/run_one_v3.py',
             '--method',job['method'],'--out',str(out),'--updates',str(job['updates']),'--seed',str(job['seed']),
             '--lr',str(job['lr']),'--scale',str(job['scale']),'--target-rank',str(job['target_rank']),
             '--max-rank',str(job['max_rank']),'--model',args.model,'--data-root',str(args.data_root),
             '--manifest-root',str(args.manifest_root),'--penalty',str(job['penalty']),
             '--targets',*args.targets]
        env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':str(Path.cwd()/'src'),
             'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HOME':os.environ.get('HF_HOME','/var/tmp/dora-adaptive-rank/hf')})
        log=(out/'stdout.log').open('w'); proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/'launcher_status.json').write_text(json.dumps({'status':'running','pid':proc.pid,'gpu':gpu,'command':cmd},indent=2)+'\n')
        return proc,log
    while index<len(pending) or active:
        while index<len(pending) and free:
            job=pending[index]; index+=1; gpu=free.pop(0); proc,log=launch(job,gpu)
            active[proc.pid]=(proc,log,job,gpu); print(json.dumps({'event':'launch','pid':proc.pid,'gpu':gpu,'method':job['method'],'trial':job['trial_index']}),flush=True)
        time.sleep(1)
        for pid,(proc,log,job,gpu) in list(active.items()):
            rc=proc.poll()
            if rc is None: continue
            log.close(); del active[pid]; free.append(gpu); free.sort(key=gpu_ids.index)
            (Path(job['out'])/'launcher_status.json').write_text(json.dumps({'status':'complete' if rc==0 else 'failed','returncode':rc,'gpu':gpu},indent=2)+'\n')
            print(json.dumps({'event':'finish','pid':pid,'gpu':gpu,'method':job['method'],'status':'complete' if rc==0 else 'failed'}),flush=True)
    print(json.dumps({'jobs_submitted':len(pending),'root':str(args.root)}))


if __name__=='__main__': main()
