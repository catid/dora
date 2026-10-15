#!/usr/bin/env python3
"""Audited restart, train/compress and common 75/25 deployment pipelines.

All stages invoke the same native runner on disjoint contiguous slices of one
sealed stream. No pilot weights enter a restart student. No backbone is merged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import yaml
from rankstudy.v3.checkpoint import json_write
from rankstudy.v3.compression import checkpoint_modules,compress_modules,write_export
from rankstudy.v3.pipeline import token_boundary,stage_seed,shuffle_compatible


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=1001); ap.add_argument('--updates',type=int,default=128)
    ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--scale',type=float,default=1.)
    ap.add_argument('--penalty',type=float,default=0.)
    ap.add_argument('--target-rank',type=int,default=8); ap.add_argument('--max-rank',type=int,default=16)
    ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'))
    ap.add_argument('--manifest-root',type=Path,default=Path('artifacts/manifests_v2'))
    ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B')
    ap.add_argument('--targets',nargs='+',default=['q_proj','o_proj'])
    ap.add_argument('--microbatch',type=int,default=2)
    ap.add_argument('--smoke',action='store_true'); ap.add_argument('--dev-limit',type=int,default=0)
    ap.add_argument('--deployment-export',action='store_true')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    specs={m['id']:m for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']}
    spec=specs[a.method]
    restart=spec['group'] in {'restart','restart_control'}
    compress=spec['group']=='compress'
    if not (restart or compress or a.deployment_export):
        raise ValueError('native non-pipelines must use run_one_v3.py')
    if a.dev_limit and not a.smoke: raise ValueError('partial dev is smoke-only')
    torch.set_num_threads(1)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (a.out/'config.json').exists():
        if not a.resume or json.loads((a.out/'config.json').read_text())!=config:
            raise ValueError('immutable pipeline exists or resume configuration changed')
        old=json.loads((a.out/'metadata.json').read_text())
        if old['status']=='complete':
            print(json.dumps({'already_complete':str(a.out)})); return
    a.out.mkdir(parents=True,exist_ok=True)
    json_write(a.out/'config.json',config)
    manifest_path=a.manifest_root/f'train_seed{a.seed}.json'
    manifest=json.loads(manifest_path.read_text()); updates=manifest['updates'][:a.updates]
    if len(updates)!=a.updates: raise ValueError('insufficient sealed stream')
    cut,cut_receipt=token_boundary(updates,.75)
    source_end=cut if compress or a.deployment_export else a.updates
    meta={'status':'running','protocol':'adaptive-rank-v3','method':a.method,'smoke':a.smoke,
          'config':config,'config_sha256':config_hash,'seed':a.seed,
          'source_revision_at_start':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
          'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__),Path('scripts/run_one_v3.py'),*Path('src/rankstudy/v3').rglob('*.py')]},
          'standalone_pilot_charge':True,'physical_pilot_cache_savings_s':0.,'deployment_export':a.deployment_export}
    json_write(a.out/'metadata.json',meta)
    start=time.perf_counter(); stages=[]; transition_s=0.

    def stage(name,method,length,offset=0,stop=None,adapter_seed=None,pattern=None,initial=None,penalty=0.,final=False):
        out=a.out/name
        cmd=[sys.executable,'scripts/run_one_v3.py','--method',method,'--out',str(out),
             '--updates',str(length),'--stream-offset',str(offset),'--seed',str(a.seed),
             '--lr',str(a.lr),'--scale',str(a.scale),'--target-rank',str(a.target_rank),'--max-rank',str(a.max_rank),
             '--data-root',str(a.data_root),'--manifest-root',str(a.manifest_root),'--model',a.model,
             '--microbatch',str(a.microbatch),'--parent-run',str(a.out),'--penalty',str(penalty),'--targets',*a.targets]
        if adapter_seed is not None: cmd+=['--adapter-seed',str(adapter_seed)]
        if pattern is not None: cmd+=['--pattern',str(pattern)]
        if initial is not None: cmd+=['--initial-adapter',str(initial)]
        if stop is not None: cmd+=['--stop-after',str(stop)]
        if a.smoke: cmd+=['--smoke','--dev-limit',str(a.dev_limit)]
        if not final: cmd+=['--skip-eval']
        executed=length if stop is None else stop
        end=out/f'checkpoint_{executed:04d}'
        valid=False
        if (out/'command.json').exists():
            if json.loads((out/'command.json').read_text())!=cmd: raise ValueError('stage command changed')
            md=json.loads((out/'metadata.json').read_text()) if (out/'metadata.json').exists() else {}
            valid=(md.get('status') in {'complete','interrupted_checkpoint'} and (end/'manifest.json').exists())
            # Never restart a still-live child after losing observation.
            launch=json.loads((out/'launcher.json').read_text()) if (out/'launcher.json').exists() else {}
            pid=launch.get('pid')
            if pid and Path(f'/proc/{pid}/cmdline').exists() and str(out).encode() in Path(f'/proc/{pid}/cmdline').read_bytes():
                raise RuntimeError('pipeline child is still live')
        if not valid:
            out.mkdir(exist_ok=True)
            json_write(out/'command.json',cmd)
            with (out/'stdout.log').open('a') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                json_write(out/'launcher.json',{'pid':proc.pid,'command':cmd,'status':'running'})
                code=proc.wait()
            json_write(out/'launcher.json',{'pid':proc.pid,'command':cmd,'status':'complete' if code==0 else 'failed','returncode':code})
            if code: raise RuntimeError(f'{name} failed; see {out}/stdout.log')
        md=json.loads((out/'metadata.json').read_text())
        if stages and md['original_backbone_sha256']!=stages[0]['original_backbone_sha256']:
            raise ValueError('pipeline stages used different original backbones')
        stages.append({'name':name,'metadata':str(out/'metadata.json'),'checkpoint':str(end),
                       'original_backbone_sha256':md['original_backbone_sha256'],
                       'input_tokens':md['input_tokens'],'supervised_tokens':md['supervised_tokens'],
                       'charged_gpu_seconds':md['charged_gpu_seconds'],'setup_s':md['setup_s'],'evaluation_s':md['evaluation_s'],
                       'cached_physical_execution':valid})
        return end,md

    if restart:
        pilot_end,pilot_receipt=token_boundary(updates,.20)
        pilot_init=a.method.split('_PILOT_')[0] if '_PILOT_' in a.method else spec['init']
        student_init=spec['init']
        pilot,pm=stage('pilot',pilot_init+'_GATE',pilot_end,adapter_seed=stage_seed(a.seed,'pilot'))
        pattern=pm['final_counts']['rank_map']
        if 'UNIFORM_RESTART' in a.method:
            pattern={name:a.target_rank for name in pattern}
        elif 'SHUFFLED_SELF_RESTART' in a.method:
            pattern=shuffle_compatible(pattern,pm['target_shapes'],stage_seed(a.seed,'pattern_shuffle'))
        # This JSON is the ONLY pilot-derived data supplied to the new student.
        pattern_path=a.out/'student_rank_pattern.json'; json_write(pattern_path,pattern)
        meta.update(pilot_boundary=pilot_receipt,pilot_initialization=pilot_init,student_initialization=student_init,
                    pilot_seed=stage_seed(a.seed,'pilot'),student_seed=stage_seed(a.seed,'student'),
                    pilot_weights_discarded=True)
        student_total=a.updates-pilot_end
        source,sm=stage('student',student_init+'_FIXED',student_total,offset=pilot_end,
                         stop=source_end-pilot_end if a.deployment_export else None,
                         adapter_seed=stage_seed(a.seed,'student'),pattern=pattern_path,final=not a.deployment_export)
    else:
        source_method=spec['init']+'_WIDE' if compress else a.method
        source,sm=stage('source',source_method,a.updates,stop=source_end,penalty=a.penalty)

    if compress or a.deployment_export:
        begin=time.perf_counter()
        source_modules=checkpoint_modules(source,sm['target_shapes'])
        policy='global' if compress and 'GLOBAL_SVD' in a.method else 'uniform'
        factors,pattern,receipt=compress_modules(source_modules,a.target_rank,a.scale,policy)
        export=a.out/'ordinary_export'
        if not export.exists():
            write_export(export,factors,a.scale,receipt,{'checkpoint':str(source),'original_backbone_sha256':sm['original_backbone_sha256'],
                         'model':a.model,'target_shapes':sm['target_shapes']})
        transition_s+=time.perf_counter()-begin
        final,fm=stage('recovery','L_FIXED',a.updates-source_end,offset=source_end,
                       adapter_seed=stage_seed(a.seed,'recovery'),initial=export,final=True)
        meta.update(export_boundary=cut_receipt,compression_policy=policy,export=str(export),
                    compression_receipt=str(export/'compression.json'),recovery_optimizer_reset=True)
    else:
        final,fm=source,sm
    inputs=sum(s['input_tokens'] for s in stages); supervised=sum(s['supervised_tokens'] for s in stages)
    if inputs!=sum(u['input_tokens'] for u in updates) or supervised!=sum(u['supervised_tokens'] for u in updates):
        raise RuntimeError('pipeline duplicated or dropped stream exposure')
    meta.update(status='complete',updates_finished=a.updates,stages=stages,
                input_tokens=inputs,supervised_tokens=supervised,final_checkpoint=str(final),
                P_target=fm['P_target'],final_counts=fm['final_counts'],budget_delta=fm['budget_delta'],
                charged_gpu_seconds=sum(s['charged_gpu_seconds'] for s in stages)+transition_s,
                transition_s=transition_s,setup_s=sum(s['setup_s'] for s in stages),
                evaluation_s=sum(s['evaluation_s'] for s in stages),wall_s=time.perf_counter()-start,
                original_backbone_sha256=fm['original_backbone_sha256'],backbone_unchanged=fm['backbone_unchanged'],
                dev_tune_endpoint_group_balanced_nll=fm['dev_tune_endpoint_group_balanced_nll'],
                dev_tune_endpoint_token_weighted_nll=fm['dev_tune_endpoint_token_weighted_nll'])
    json_write(a.out/'metadata.json',meta)
    print(json.dumps({'status':'complete','method':a.method,'input_tokens':inputs,'charged_gpu_seconds':meta['charged_gpu_seconds']}),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if '--out' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--out')+1])/'metadata.json'
            if path.exists():
                md=json.loads(path.read_text())
                if md.get('status')=='running':
                    md.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}');json_write(path,md)
        raise
