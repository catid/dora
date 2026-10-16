#!/usr/bin/env python3
"""Run two full-horizon tuning seeds for every registered v3 cell."""
from __future__ import annotations
import argparse, concurrent.futures, fcntl, json, os, random, subprocess, sys, time
from pathlib import Path
import yaml


def select(hpo_root, methods):
    candidates={m['id']:[] for m in methods}
    for path in sorted(hpo_root.glob('*/metadata.json')):
        meta=json.loads(path.read_text()); method=meta.get('method')
        if method not in candidates or meta.get('status')!='complete': continue
        score=meta.get('dev_tune_endpoint_group_balanced_nll')
        if score is None: continue
        cfg=json.loads((path.parent/'hpo_config.json').read_text())
        candidates[method].append((float(score),float(cfg['lr']),float(cfg['scale']),float(cfg.get('penalty',0.)),cfg))
    selected={}
    for method,rows in candidates.items():
        rows.sort(key=lambda x:(x[0],x[1],x[2],x[3],x[4]['trial_index']))
        seen=set(); keep=[]
        for row in rows:
            key=(row[1],row[2],row[3])
            if key in seen: continue
            seen.add(key); keep.append({'hpo_score':row[0],'config':row[4]})
            if len(keep)==2: break
        selected[method]=keep
    return selected


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hpo-root',type=Path,required=True);ap.add_argument('--out-root',type=Path,required=True);ap.add_argument('--gpus',default='0,1,2,3');ap.add_argument('--updates',type=int,default=512);ap.add_argument('--seeds',default='1002,1003');ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B');ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'));ap.add_argument('--manifest-root',type=Path,default=Path('artifacts/manifests_v2'));ap.add_argument('--resume',action='store_true');ap.add_argument('--method-ids',default='',help='Optional comma-separated registered subset; default is all methods.');args=ap.parse_args();args.out_root.mkdir(parents=True,exist_ok=True)
    gate=json.loads(Path('artifacts/protocol_execution_gate.json').read_text())
    if not gate.get('v3_large_sweep_allowed'): raise RuntimeError('v3 primary execution gate is closed')
    methods=yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']
    if args.method_ids:
        wanted=set(x.strip() for x in args.method_ids.split(',') if x.strip()); methods=[m for m in methods if m['id'] in wanted]
        if len(methods)!=len(wanted): raise ValueError(f'unknown method id in --method-ids: {sorted(wanted-{m["id"] for m in methods})}')
    else:
        methods=[m for m in methods if m.get('group')!='hybrid']
    selected=select(args.hpo_root,methods)
    (args.out_root/'selected_configs.json').write_text(json.dumps(selected,indent=2)+'\n')
    jobs=[]
    for spec in methods:
        for ci,item in enumerate(selected[spec['id']],1):
            cfg=item['config']
            for seed in map(int,args.seeds.split(',')):
                out=args.out_root/f"{spec['id']}_c{cfg['trial_index']:02d}_s{seed}"
                if args.resume and (out/'metadata.json').exists(): continue
                out.mkdir(parents=True,exist_ok=True)
                tuning_cfg={'method':spec['id'],'lr':cfg['lr'],'scale':cfg['scale'],'penalty':cfg.get('penalty',0.),'trial_index':cfg['trial_index'],'seed':seed,'hpo_score':item['hpo_score']}
                (out/'tuning_config.json').write_text(json.dumps(tuning_cfg,indent=2)+'\n')
                jobs.append({**tuning_cfg,'out':str(out),'pipeline':spec['group'] in {'restart','restart_control','compress'}})
    rng=random.Random(20261016);rng.shuffle(jobs);gpus=[x.strip() for x in args.gpus.split(',') if x.strip()];free=list(gpus);active={};i=0
    def launch(job,gpu):
        out=Path(job['out']); cmd=[sys.executable,'scripts/run_pipeline_v3.py' if job['pipeline'] else 'scripts/run_one_v3.py','--method',job['method'],'--out',str(out),'--updates',str(args.updates),'--seed',str(job['seed']),'--lr',str(job['lr']),'--scale',str(job['scale']),'--penalty',str(job['penalty']),'--model',args.model,'--data-root',str(args.data_root),'--manifest-root',str(args.manifest_root)]
        env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':str(Path.cwd()/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HOME':os.environ.get('HF_HOME','/var/tmp/dora-adaptive-rank/hf')});log=(out/'stdout.log').open('w');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);(out/'launcher_status.json').write_text(json.dumps({'status':'running','pid':proc.pid,'gpu':gpu,'command':cmd},indent=2)+'\n');return proc,log
    while i<len(jobs) or active:
        while i<len(jobs) and free:
            job=jobs[i];i+=1;gpu=free.pop(0);proc,log=launch(job,gpu);active[proc.pid]=(proc,log,job,gpu);print(json.dumps({'event':'launch','pid':proc.pid,'gpu':gpu,'method':job['method'],'seed':job['seed']}),flush=True)
        time.sleep(1)
        for pid,(proc,log,job,gpu) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();del active[pid];free.append(gpu);free.sort(key=gpus.index);(Path(job['out'])/'launcher_status.json').write_text(json.dumps({'status':'complete' if rc==0 else 'failed','returncode':rc,'gpu':gpu},indent=2)+'\n');print(json.dumps({'event':'finish','pid':pid,'gpu':gpu,'method':job['method'],'status':'complete' if rc==0 else 'failed'}),flush=True)
    print(json.dumps({'jobs_submitted':len(jobs),'methods_with_hpo':sum(bool(x) for x in selected.values()),'out_root':str(args.out_root)}))


if __name__=='__main__':main()
