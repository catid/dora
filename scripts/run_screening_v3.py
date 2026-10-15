#!/usr/bin/env python3
"""Freeze one full-horizon configuration per cell and run three fresh screens."""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import yaml


def frozen_configs(tuning_root, methods):
    grouped=defaultdict(lambda: defaultdict(list))
    for path in sorted(tuning_root.glob('*/metadata.json')):
        md=json.loads(path.read_text());
        if md.get('status')!='complete': continue
        if md.get('dev_tune_endpoint_group_balanced_nll') is None: continue
        cfg=json.loads((path.parent/'tuning_config.json').read_text())
        key=(float(cfg['lr']),float(cfg['scale']),float(cfg.get('penalty',0.)))
        grouped[md.get('method')][key].append(float(md['dev_tune_endpoint_group_balanced_nll']))
    out={}
    for spec in methods:
        candidates=[]
        for key,values in grouped[spec['id']].items():
            if len(values)!=2: continue
            candidates.append({'lr':key[0],'scale':key[1],'penalty':key[2],
                               'tuning_mean_group_balanced_nll':sum(values)/len(values),
                               'tuning_values':values})
        if candidates: out[spec['id']]=sorted(candidates,key=lambda x:(x['tuning_mean_group_balanced_nll'],x['lr'],x['scale'],x['penalty']))[0]
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tuning-root',type=Path,required=True);ap.add_argument('--out-root',type=Path,required=True);ap.add_argument('--gpus',default='0,1,2,3');ap.add_argument('--seeds',default='2001,2002,2003');ap.add_argument('--updates',type=int,default=512);ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B');ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'));ap.add_argument('--manifest-root',type=Path,default=Path('artifacts/manifests_v2'));args=ap.parse_args();args.out_root.mkdir(parents=True,exist_ok=True)
    methods=yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']; frozen=frozen_configs(args.tuning_root,methods)
    if len(frozen)!=len(methods): raise RuntimeError(f'cannot freeze all 41 methods; got {len(frozen)}')
    (args.out_root/'frozen_configs.json').write_text(json.dumps(frozen,indent=2)+'\n')
    jobs=[]
    for spec in methods:
        cfg=frozen[spec['id']]
        for seed in map(int,args.seeds.split(',')):
            out=args.out_root/f"{spec['id']}_s{seed}";out.mkdir(parents=True,exist_ok=True)
            jobs.append({'method':spec['id'],'seed':seed,'lr':cfg['lr'],'scale':cfg['scale'],'penalty':cfg['penalty'],'out':str(out),'pipeline':spec['group'] in {'restart','restart_control','compress'}})
    rng=random.Random(20261017);rng.shuffle(jobs);gpus=[x.strip() for x in args.gpus.split(',') if x.strip()];free=list(gpus);active={};i=0
    def launch(job,gpu):
        out=Path(job['out']);cmd=[sys.executable,'scripts/run_pipeline_v3.py' if job['pipeline'] else 'scripts/run_one_v3.py','--method',job['method'],'--out',str(out),'--updates',str(args.updates),'--seed',str(job['seed']),'--lr',str(job['lr']),'--scale',str(job['scale']),'--penalty',str(job['penalty']),'--model',args.model,'--data-root',str(args.data_root),'--manifest-root',str(args.manifest_root)]
        env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':str(Path.cwd()/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HOME':os.environ.get('HF_HOME','/var/tmp/dora-adaptive-rank/hf')});log=(out/'stdout.log').open('w');p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);(out/'launcher_status.json').write_text(json.dumps({'status':'running','pid':p.pid,'gpu':gpu,'command':cmd},indent=2)+'\n');return p,log
    while i<len(jobs) or active:
        while i<len(jobs) and free:
            job=jobs[i];i+=1;gpu=free.pop(0);p,log=launch(job,gpu);active[p.pid]=(p,log,job,gpu);print(json.dumps({'event':'launch','pid':p.pid,'gpu':gpu,'method':job['method'],'seed':job['seed']}),flush=True)
        time.sleep(1)
        for pid,(p,log,job,gpu) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            log.close();del active[pid];free.append(gpu);free.sort(key=gpus.index);(Path(job['out'])/'launcher_status.json').write_text(json.dumps({'status':'complete' if rc==0 else 'failed','returncode':rc,'gpu':gpu},indent=2)+'\n');print(json.dumps({'event':'finish','pid':pid,'gpu':gpu,'method':job['method'],'status':'complete' if rc==0 else 'failed'}),flush=True)
    print(json.dumps({'jobs':len(jobs),'methods':len(frozen),'out_root':str(args.out_root)}))


if __name__=='__main__':main()
