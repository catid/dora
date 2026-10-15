#!/usr/bin/env python3
"""Balanced real-model correctness smoke for all 41 cells, never HPO.

Requires explicitly idle allocated GPUs. Each method receives the same 32-update
engineering configuration; no result selects hyperparameters or a method family.
Failed trials remain terminal and are not overwritten or silently retried.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.root/'.queue.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpus=a.gpus.split(',');visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible and not set(gpus)<=set(visible.split(',')):
        raise ValueError('requested GPUs exceed inherited CUDA_VISIBLE_DEVICES')
    specs=yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']
    groups={}
    for m in specs:groups.setdefault(m['group'],[]).append(m)
    rng=random.Random(20261015)
    for v in groups.values():rng.shuffle(v)
    keys=sorted(groups);rng.shuffle(keys);ordered=[]
    while any(groups.values()):
        for key in keys:
            if groups[key]:ordered.append(groups[key].pop())
    assignments=[[] for _ in gpus]
    for i,m in enumerate(ordered):assignments[i%len(gpus)].append(m)
    plan={'purpose':'engineering correctness, not selection','updates':32,'lr':1e-4,'scale':1.,'seed':1001,
          'dev_limit':16,'methods':[m['id'] for m in ordered],
          'assignments':{gpu:[m['id'] for m in jobs] for gpu,jobs in zip(gpus,assignments)}}
    path=a.root/'plan.json'
    if path.exists():
        if not a.resume or json.loads(path.read_text())!=plan:raise ValueError('queue exists or changed')
    else:path.write_text(json.dumps(plan,indent=2)+'\n')
    def lane(item):
        gpu,jobs=item
        for spec in jobs:
            method=spec['id'];out=a.root/method
            if (out/'metadata.json').exists():
                md=json.loads((out/'metadata.json').read_text())
                if md['status'] in {'complete','failed'}:
                    print(json.dumps({'method':method,'status':md['status'],'retained_terminal':True}),flush=True);continue
                raise RuntimeError(f'nonterminal prior process requires explicit inspection: {out}')
            pipeline=spec['group'] in {'restart','restart_control','compress'}
            script='scripts/run_pipeline_v3.py' if pipeline else 'scripts/run_one_v3.py'
            cmd=[sys.executable,script,'--method',method,'--out',str(out),'--updates','32',
                 '--lr','1e-4','--scale','1','--smoke','--dev-limit','16','--penalty',
                 str(.1 if spec['adalora'] else spec['penalty'])]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            with (a.root/f'{method}.log').open('x') as log:
                begin=time.perf_counter();proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                record={'pid':proc.pid,'command':cmd,'gpu':gpu,'start_unix_s':time.time(),'status':'running'}
                launch=a.root/f'{method}.launcher.json';launch.write_text(json.dumps(record,indent=2)+'\n')
                code=proc.wait();record.update(status='complete' if code==0 else 'failed',returncode=code,
                                               all_in_process_wall_s=time.perf_counter()-begin)
                launch.write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps({'method':method,'gpu':gpu,'returncode':code}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lane,zip(gpus,assignments)))


if __name__=='__main__':main()
