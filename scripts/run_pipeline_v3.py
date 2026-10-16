#!/usr/bin/env python3
"""Audited restart, train/compress and common 75/25 deployment pipelines.

All stages invoke the same native runner on disjoint contiguous slices of one
sealed stream. No pilot weights enter a restart student. No backbone is merged.
"""
from __future__ import annotations
import argparse
import fcntl
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
from rankstudy.v3.pipeline import token_boundary,stage_seed,shuffle_compatible,export_policy
from rankstudy.v3.stage_execution import run_stage,attempt_costs,PlannedInterruption,complete_checkpoint


def main():
    global _OWNED_OUTPUT
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
    ap.add_argument('--gpu-budget-s',type=float,
                    help='Optional hard end-to-end charged-GPU-second budget. '
                         'Source and recovery are capped with a reserved recovery '
                         'interval; incomplete transitions are classified infeasible.')
    ap.add_argument('--transition-reserve-s',type=float,default=1.0,
                    help='Quality-blind charged-second reserve for export/compression.')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--interrupt-stage',choices=['pilot','student','source','recovery'])
    ap.add_argument('--interrupt-after',type=int,default=4)
    a=ap.parse_args()
    if not a.smoke:
        gate=json.loads(Path('artifacts/protocol_execution_gate.json').read_text())
        if not gate.get('v3_large_sweep_allowed'):
            raise RuntimeError('primary v3 execution gate is closed; use --smoke only for declared engineering checks')
    specs={m['id']:m for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']}
    spec=specs[a.method]
    restart=spec['group'] in {'restart','restart_control'}
    compress=spec['group']=='compress'
    if not (restart or compress or a.deployment_export):
        raise ValueError('native non-pipelines must use run_one_v3.py')
    if a.dev_limit and not a.smoke: raise ValueError('partial dev is smoke-only')
    if a.interrupt_stage and not a.smoke: raise ValueError('planned interruption is smoke-only')
    torch.set_num_threads(1)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()
            if k not in {'resume','interrupt_stage','interrupt_after'}}
    config['common_export_policy']='global_singular_energy' if a.deployment_export else None
    if a.gpu_budget_s is not None:
        if a.gpu_budget_s <= 0 or a.transition_reserve_s < 0:
            raise ValueError('invalid GPU budget or transition reserve')
        if not a.deployment_export:
            raise ValueError('GPU budget branch requires --deployment-export')
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    a.out.mkdir(parents=True,exist_ok=True)
    parent_lock=(a.out/'.pipeline.lock').open('a+')
    fcntl.flock(parent_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=None
    if (a.out/'config.json').exists():
        if not a.resume or json.loads((a.out/'config.json').read_text())!=config:
            raise ValueError('immutable pipeline exists or resume configuration changed')
        old=json.loads((a.out/'metadata.json').read_text())
        if old['status']=='complete':
            print(json.dumps({'already_complete':str(a.out)})); return
        history=a.out/'parent_attempts';history.mkdir(exist_ok=True)
        json_write(history/f'{len(list(history.glob("*.json"))):04d}.json',old)
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
    if old and old['source_hashes']!=meta['source_hashes']:
        raise ValueError('pipeline source changed; use a new run ID or an explicitly audited migration')
    json_write(a.out/'metadata.json',meta)
    _OWNED_OUTPUT=a.out/'metadata.json'
    start=time.perf_counter(); stages=[]; transition_s=0.

    def stage(name,method,length,offset=0,stop=None,adapter_seed=None,pattern=None,initial=None,penalty=0.,final=False,budget_cap=None):
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
        if budget_cap is not None: cmd+=['--max-charged-s',str(max(1e-6,budget_cap))]
        if a.smoke: cmd+=['--smoke','--dev-limit',str(a.dev_limit)]
        if not final: cmd+=['--skip-eval']
        executed=length if stop is None else stop
        end,md,execution=run_stage(out,cmd,executed,resume=a.resume,
                                  interrupt_after=a.interrupt_after if a.interrupt_stage==name else None,
                                  allow_incomplete=budget_cap is not None)
        costs=attempt_costs(execution['attempts'])
        if stages and md['original_backbone_sha256']!=stages[0]['original_backbone_sha256']:
            raise ValueError('pipeline stages used different original backbones')
        stages.append({'name':name,'metadata':str(Path(execution['selected_attempt'])/'metadata.json'),'checkpoint':str(end),
                       'original_backbone_sha256':md['original_backbone_sha256'],
                       'input_tokens':md['input_tokens'],'supervised_tokens':md['supervised_tokens'],
                       'charged_gpu_seconds':sum(c['observed_training_gpu_seconds'] for c in costs),
                       'logical_trajectory_gpu_seconds':md['charged_gpu_seconds'],
                       'setup_s':sum(c['setup_s'] for c in costs),'evaluation_s':sum(c['evaluation_s'] for c in costs),
                       'attempt_costs':costs,'process_wall_s':sum(c['process_wall_s'] or 0. for c in costs),
                       'all_in_cost_complete':all(c['process_wall_s'] is not None for c in costs),
                       'cached_physical_execution':execution['cached_physical_execution']})
        meta['stages']=stages
        json_write(a.out/'metadata.json',meta)
        return end,md,execution

    source_cap=None
    recovery_cap=None
    if a.gpu_budget_s is not None:
        # Reserve the final 25% recovery interval and a measured transition
        # allowance before assigning the source (pilot plus student) budget.
        recovery_reserve=a.gpu_budget_s*(a.updates-source_end)/a.updates
        source_cap=a.gpu_budget_s-recovery_reserve-a.transition_reserve_s
        if source_cap <= 0:
            raise ValueError('GPU budget leaves no source interval')

    if restart:
        pilot_end,pilot_receipt=token_boundary(updates,.20)
        pilot_init=a.method.split('_PILOT_')[0] if '_PILOT_' in a.method else spec['init']
        student_init=spec['init']
        pilot_cap=(source_cap*pilot_end/source_end) if source_cap is not None else None
        pilot,pm,_=stage('pilot',pilot_init+'_GATE',pilot_end,
                          adapter_seed=stage_seed(a.seed,'pilot'),budget_cap=pilot_cap)
        if source_cap is not None and pm.get('updates_finished') != pilot_end:
            meta.update(status='infeasible_budget',failure_reason='pilot exhausted its reserved GPU budget',
                        budget_s=a.gpu_budget_s,budget_source_cap_s=source_cap,
                        budget_pilot_updates_finished=pm.get('updates_finished'),
                        budget_pilot_updates_required=pilot_end,final_checkpoint=str(pilot),
                        stages=stages,charged_gpu_seconds=sum(s['charged_gpu_seconds'] for s in stages),
                        transition_s=0.0,backbone_unchanged=pm.get('backbone_unchanged',True))
            json_write(a.out/'metadata.json',meta)
            print(json.dumps({'status':'infeasible_budget','method':a.method,
                              'updates_finished':pm.get('updates_finished'),
                              'updates_required':pilot_end,'charged_gpu_seconds':meta['charged_gpu_seconds']}),flush=True)
            return
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
        student_cap=(source_cap-stages[-1]['charged_gpu_seconds']) if source_cap is not None else None
        source,sm,_=stage('student',student_init+'_FIXED',student_total,offset=pilot_end,
                         stop=source_end-pilot_end if a.deployment_export else None,
                         adapter_seed=stage_seed(a.seed,'student'),pattern=pattern_path,
                         final=not a.deployment_export,budget_cap=student_cap)
        if source_cap is not None and sm.get('updates_finished') != source_end-pilot_end:
            meta.update(status='infeasible_budget',failure_reason='restart student exhausted its reserved GPU budget',
                        budget_s=a.gpu_budget_s,budget_source_cap_s=source_cap,
                        budget_student_updates_finished=sm.get('updates_finished'),
                        budget_student_updates_required=source_end-pilot_end,final_checkpoint=str(source),
                        stages=stages,charged_gpu_seconds=sum(s['charged_gpu_seconds'] for s in stages),
                        transition_s=0.0,backbone_unchanged=sm.get('backbone_unchanged',True))
            json_write(a.out/'metadata.json',meta)
            print(json.dumps({'status':'infeasible_budget','method':a.method,
                              'updates_finished':sm.get('updates_finished'),
                              'updates_required':source_end-pilot_end,
                              'charged_gpu_seconds':meta['charged_gpu_seconds']}),flush=True)
            return
    else:
        source_method=spec['init']+'_WIDE' if compress else a.method
        source,sm,source_execution=stage('source',source_method,a.updates,stop=source_end,
                                         penalty=a.penalty,budget_cap=source_cap)
        if a.gpu_budget_s is not None and sm.get('updates_finished') != source_end:
            meta.update(status='infeasible_budget',failure_reason='source stage exhausted its reserved GPU budget',
                        budget_s=a.gpu_budget_s,budget_source_cap_s=source_cap,
                        budget_source_updates_finished=sm.get('updates_finished'),
                        budget_source_updates_required=source_end,final_checkpoint=str(source),
                        stages=stages,charged_gpu_seconds=sum(s['charged_gpu_seconds'] for s in stages),
                        transition_s=0.0,backbone_unchanged=sm.get('backbone_unchanged',True))
            json_write(a.out/'metadata.json',meta)
            print(json.dumps({'status':'infeasible_budget','method':a.method,
                              'updates_finished':sm.get('updates_finished'),
                              'updates_required':source_end,'charged_gpu_seconds':meta['charged_gpu_seconds']}),flush=True)
            return

    if compress or a.deployment_export:
        begin=time.perf_counter()
        # One common deployment operator for every source family, including
        # train/compress methods at their natural 75% event (never a second SVD).
        # Native uniform/global compression cells remain distinct outside this
        # new common pipeline. Identical common branches can be physically reused.
        policy=export_policy(a.method,a.deployment_export)
        export=a.out/'ordinary_export'
        if export.exists():
            if not complete_checkpoint(export): raise ValueError('incomplete existing export; preserve and audit it')
            desc=json.loads((export/'manifest.json').read_text())
            if desc['source']['checkpoint']!=str(source): raise ValueError('cached export has another source')
            receipt=json.loads((export/'compression.json').read_text())
            if receipt['policy']!=policy or receipt['export_scale']!=a.scale: raise ValueError('cached export policy changed')
            transition_s=json.loads((export/'transition_accounting.json').read_text())['transition_s']
            meta['physical_transition_cache_savings_s']=transition_s
        else:
            source_modules=checkpoint_modules(source,sm['target_shapes'])
            factors,pattern,receipt=compress_modules(source_modules,a.target_rank,a.scale,policy)
            if a.deployment_export and receipt['allocation']['underspend']:
                # Never silently call a smaller equal-cost export size-matched.
                costs={m.base.in_features+m.base.out_features for m in source_modules.values()}
                if len(costs)==1:
                    raise ValueError('common equal-cost export cannot fill its declared budget')
            write_export(export,factors,a.scale,receipt,{'checkpoint':str(source),'original_backbone_sha256':sm['original_backbone_sha256'],
                         'model':a.model,'target_shapes':sm['target_shapes']})
            transition_s=time.perf_counter()-begin
            json_write(export/'transition_accounting.json',{'transition_s':transition_s})
        recovery_cap=None
        if a.gpu_budget_s is not None:
            used=sum(s['charged_gpu_seconds'] for s in stages)+transition_s
            recovery_cap=a.gpu_budget_s-used
            if recovery_cap <= 0:
                meta.update(status='infeasible_budget',failure_reason='compression consumed the remaining recovery budget',
                            budget_s=a.gpu_budget_s,budget_used_before_recovery_s=used,
                            final_checkpoint=str(source),stages=stages,charged_gpu_seconds=used,
                            transition_s=transition_s,backbone_unchanged=sm.get('backbone_unchanged',True))
                json_write(a.out/'metadata.json',meta)
                print(json.dumps({'status':'infeasible_budget','method':a.method,
                                  'charged_gpu_seconds':used}),flush=True)
                return
        final,fm,recovery_execution=stage('recovery','L_FIXED',a.updates-source_end,offset=source_end,
                       adapter_seed=stage_seed(a.seed,'recovery'),initial=export,final=True,
                       budget_cap=recovery_cap)
        if a.gpu_budget_s is not None and fm.get('updates_finished') != a.updates-source_end:
            # ``stages`` already contains the measured partial recovery
            # charge.  Do not add it a second time when writing the
            # infeasibility receipt.
            used=sum(s['charged_gpu_seconds'] for s in stages)+transition_s
            meta.update(status='infeasible_budget',failure_reason='recovery stage did not reach the required final interval',
                        budget_s=a.gpu_budget_s,budget_used_s=used,
                        budget_recovery_updates_finished=fm.get('updates_finished'),
                        budget_recovery_updates_required=a.updates-source_end,
                        final_checkpoint=str(final),stages=stages,charged_gpu_seconds=used,
                        transition_s=transition_s,backbone_unchanged=fm.get('backbone_unchanged',True))
            json_write(a.out/'metadata.json',meta)
            print(json.dumps({'status':'infeasible_budget','method':a.method,
                              'updates_finished':fm.get('updates_finished'),
                              'updates_required':a.updates-source_end,'charged_gpu_seconds':used}),flush=True)
            return
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
                standalone_stage_process_wall_s=sum(s['process_wall_s'] for s in stages),
                all_in_cost_complete=all(s['all_in_cost_complete'] for s in stages),
                original_backbone_sha256=fm['original_backbone_sha256'],backbone_unchanged=fm['backbone_unchanged'],
                dev_tune_endpoint_group_balanced_nll=fm['dev_tune_endpoint_group_balanced_nll'],
                dev_tune_endpoint_token_weighted_nll=fm['dev_tune_endpoint_token_weighted_nll'])
    if a.gpu_budget_s is not None:
        meta.update(budget_s=a.gpu_budget_s,budget_utilization=meta['charged_gpu_seconds']/a.gpu_budget_s,
                    budget_transition_reserve_s=a.transition_reserve_s,
                    budget_recovery_cap_s=recovery_cap)
    json_write(a.out/'metadata.json',meta)
    print(json.dumps({'status':'complete','method':a.method,'input_tokens':inputs,'charged_gpu_seconds':meta['charged_gpu_seconds']}),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if '--out' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--out')+1])/'metadata.json'
            if path.exists() and globals().get('_OWNED_OUTPUT')==path:
                md=json.loads(path.read_text())
                if md.get('status')=='running':
                    md.update(status='interrupted_checkpoint' if isinstance(exc,PlannedInterruption) else 'failed',
                              failure_reason=f'{type(exc).__name__}: {exc}');json_write(path,md)
        raise
