#!/usr/bin/env python3
"""Run the declared 12-trial HPO for every executable native registry cell."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path
import sys

# Make direct ``python scripts/run_hpo_v2.py`` invocation reproducible without
# requiring callers to remember a repository-specific PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rankstudy.registry import build_registry

SUPPORTED = {"fixed", "wide", "reference", "online", "allocation_control", "triplet", "triplet_ortho", "online_uncertainty"}

def jobs(root, data_root, manifest_root, updates, seed, rank, target_rank):
    lrs = (1e-6, 1e-5, 1e-4, 1e-3)
    scales = (0.25, 1.0, 4.0)
    out = []
    for spec in build_registry():
        if spec.group not in SUPPORTED:
            continue
        r = target_rank if spec.group in {"fixed"} or (spec.group in {"triplet", "triplet_ortho"} and spec.allocator == "none") else rank
        trial_index = 0
        for lr in lrs:
            for scale in scales:
                # The protocol gives orthogonality candidates the same twelve
                # LR/scale trials, with the three registered penalty weights
                # assigned in a deterministic balanced cycle.
                penalty = (1e-4, 1e-3, 1e-2)[trial_index % 3] if spec.group == "triplet_ortho" else 0.0
                name = f"hpo_{spec.id}_lr{lr:g}_s{scale:g}"
                out.append({"method": spec.id, "init": spec.init, "lr": lr, "scale": scale,
                            "penalty": penalty,
                            "seed": seed, "rank": r, "target_rank": target_rank,
                            "updates": updates, "data_root": str(data_root),
                            # run_one_v2 resolves the seed-specific manifest from
                            # this directory and records the exact file in metadata.
                            # Passing the file itself would append the seed a
                            # second time and fail before model loading.
                            "manifest": str(Path(manifest_root)),
                            "out": str(Path(root) / name)})
                trial_index += 1
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2')); p.add_argument('--manifest-root',type=Path,default=Path('artifacts/manifests_v2')); p.add_argument('--updates',type=int,default=128); p.add_argument('--seed',type=int,default=1001); p.add_argument('--gpus',default='0,1,2,3'); p.add_argument('--python',default='.venv-study/bin/python'); p.add_argument('--microbatch',type=int,default=2); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True); gpu_ids=[x.strip() for x in a.gpus.split(',') if x.strip()]; free=list(gpu_ids); active={}; queue=[]
    for j in jobs(a.root,a.data_root,a.manifest_root,a.updates,a.seed,16,8):
        out=Path(j['out']);
        if a.resume and (out/'metadata.json').exists(): continue
        queue.append(j)
    i=0
    while i<len(queue) or active:
        while i<len(queue) and free:
            j=queue[i]; i+=1; out=Path(j['out']); out.mkdir(parents=True,exist_ok=True); gpu=free.pop(0)
            cmd=[a.python,'scripts/run_one_v2.py']
            for k in ('method','init','scale','lr','penalty','seed','updates','data_root','manifest','out','rank','target_rank'): cmd += [f'--{k.replace("_","-")}',str(j[k])]
            cmd += ['--microbatch',str(a.microbatch)]
            env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':'src','HF_HOME':os.environ.get('HF_HOME','/var/tmp/dora-adaptive-rank/hf'),'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
            log=(out/'stdout.log').open('w'); proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[proc.pid]=(proc,log,j,gpu); print(json.dumps({'event':'launch','pid':proc.pid,'gpu':gpu,'method':j['method'],'lr':j['lr'],'scale':j['scale']}),flush=True)
            (out/'launcher_status.json').write_text(json.dumps({'status':'running','pid':proc.pid,'gpu':gpu,'stage':'v2_hpo'},indent=2)+'\n')
        time.sleep(1)
        for pid,(proc,log,j,gpu) in list(active.items()):
            rc=proc.poll()
            if rc is None: continue
            log.close(); del active[pid]; free.append(gpu); free.sort(key=gpu_ids.index)
            (Path(j['out'])/'launcher_status.json').write_text(json.dumps({'status':'complete' if rc==0 else 'failed','returncode':rc,'gpu':gpu,'stage':'v2_hpo'},indent=2)+'\n')
            print(json.dumps({'event':'finish','pid':pid,'gpu':gpu,'status':'complete' if rc==0 else 'failed','method':j['method']}),flush=True)
    print(json.dumps({'jobs':len(queue),'root':str(a.root)}))
if __name__=='__main__': main()
