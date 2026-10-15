#!/usr/bin/env python3
"""Run v3 HPO -> tuning -> fresh screening in declared order.

The HPO launcher may already be running in another session. This supervisor
only observes its immutable launcher records and starts the next stage once all
492 trials are terminal and successful; it never converts a failure into a
selection or overwrites a run.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path


def statuses(root):
    rows=[]
    for path in sorted(root.glob('*/launcher_status.json')):
        rows.append(json.loads(path.read_text()))
    return rows


def wait_for(root, expected, interval):
    while True:
        rows=statuses(root); counts={}
        for row in rows: counts[row.get('status')]=counts.get(row.get('status'),0)+1
        print(json.dumps({'root':str(root),'counts':counts,'expected':expected}),flush=True)
        if len(rows)>=expected and sum(v for k,v in counts.items() if k not in {'complete','failed'})==0:
            return counts
        time.sleep(interval)


def run(cmd):
    print(json.dumps({'event':'start','command':cmd}),flush=True)
    result=subprocess.run(cmd)
    if result.returncode: raise SystemExit(result.returncode)


def verify_hpo_integrity(root):
    """Require the declared 41x12 search before any tuning selection."""
    # Keep this check independent of the report writer: it validates the
    # immutable launcher/config/metadata records that selection consumes.
    import yaml
    method_ids = [row['id'] for row in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']]
    seen = {method: [] for method in method_ids}
    issues = []
    completed_token_counts = []
    for path in sorted(root.glob('*/launcher_status.json')):
        launcher = json.loads(path.read_text())
        config_path = path.parent / 'hpo_config.json'
        metadata_path = path.parent / 'metadata.json'
        if not config_path.exists():
            issues.append(f'{path.parent.name}: missing hpo_config.json'); continue
        config = json.loads(config_path.read_text())
        method, trial, seed = config.get('method'), config.get('trial_index'), config.get('seed')
        if method not in seen: issues.append(f'{path.parent.name}: unknown method {method}'); continue
        if seed != 1001 or trial not in range(12):
            issues.append(f'{path.parent.name}: seed/trial mismatch ({seed}, {trial})')
        seen[method].append(trial)
        if launcher.get('status') == 'complete':
            if not metadata_path.exists():
                issues.append(f'{path.parent.name}: complete launcher without metadata')
            else:
                metadata = json.loads(metadata_path.read_text())
                if metadata.get('status') not in {'complete', 'passed'}:
                    issues.append(f'{path.parent.name}: launcher complete but metadata status is {metadata.get("status")}')
                if metadata.get('input_tokens') is not None:
                    completed_token_counts.append(metadata['input_tokens'])
        if launcher.get('status') == 'failed':
            issues.append(f'{path.parent.name}: failed launcher')
    missing = {method: sorted(set(range(12)) - set(trials)) for method, trials in seen.items()
               if sorted(set(range(12)) - set(trials))}
    duplicate = {method: trials for method, trials in seen.items() if len(trials) != len(set(trials))}
    if len(set(completed_token_counts)) > 1:
        issues.append(f'inconsistent completed input-token counts: {sorted(set(completed_token_counts))}')
    if missing or duplicate or issues or len(sum(seen.values(), [])) != 492:
        raise RuntimeError(json.dumps({'missing': missing, 'duplicate': duplicate, 'issues': issues,
                                       'records': len(sum(seen.values(), []))}, sort_keys=True))


def run_declared_diagnostics(out_root, checkpoint_root, data_root, model):
    """Run the two registered diagnostic seeds sequentially on one free GPU."""
    import os
    out_root.mkdir(parents=True, exist_ok=True)
    for method in ('L_GATE','I_GATE'):
        for seed in (2901,2902):
            summary=out_root/f'{method}_s{seed}_summary.json'
            if summary.exists():
                continue
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONPATH']=str(Path.cwd()/'src')
            cmd=[sys.executable,'scripts/run_deletion_diagnostics_v3.py','--method',method,'--seed',str(seed),
                 '--checkpoint-root',str(checkpoint_root),'--out',str(out_root),'--model',model,'--data-root',str(data_root)]
            print(json.dumps({'event':'diagnostic_start','method':method,'seed':seed}),flush=True)
            result=subprocess.run(cmd,env=env)
            if result.returncode: raise SystemExit(result.returncode)
    records=list(out_root.glob('*_s*_summary.json'))
    if len(records)!=4: raise RuntimeError(f'declared diagnostics incomplete: {len(records)}/4')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hpo-root',type=Path,required=True);ap.add_argument('--tuning-root',type=Path,required=True);ap.add_argument('--screening-root',type=Path,required=True);ap.add_argument('--gpus',default='0,1,2,3');ap.add_argument('--model',default='/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B');ap.add_argument('--data-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/data_v2'));ap.add_argument('--manifest-root',type=Path,default=Path('artifacts/manifests_v2'));ap.add_argument('--poll-seconds',type=int,default=30);args=ap.parse_args()
    counts=wait_for(args.hpo_root,492,args.poll_seconds)
    if counts.get('failed',0): raise RuntimeError(f'HPO contains {counts["failed"]} failed trials; rescue policy must run before tuning')
    verify_hpo_integrity(args.hpo_root)
    run_declared_diagnostics(Path('/var/tmp/dora-adaptive-rank/runs/v3_deletion_diagnostics_declared_20261015_final'), Path('/var/tmp/dora-adaptive-rank/runs/v3_deletion_replay_20261015'), args.data_root, args.model)
    run([sys.executable,'scripts/run_tuning_v3.py','--hpo-root',str(args.hpo_root),'--out-root',str(args.tuning_root),'--gpus',args.gpus,'--model',args.model,'--data-root',str(args.data_root),'--manifest-root',str(args.manifest_root)])
    tuning_rows=statuses(args.tuning_root)
    if len(tuning_rows)<164 or any(row.get('status')!='complete' for row in tuning_rows): raise RuntimeError('tuning did not complete all 164 jobs')
    run([sys.executable,'scripts/run_screening_v3.py','--tuning-root',str(args.tuning_root),'--out-root',str(args.screening_root),'--gpus',args.gpus,'--model',args.model,'--data-root',str(args.data_root),'--manifest-root',str(args.manifest_root)])
    screening_rows=statuses(args.screening_root)
    if len(screening_rows)<123 or any(row.get('status')!='complete' for row in screening_rows): raise RuntimeError('screening did not complete all 123 jobs')
    print(json.dumps({'status':'stage_B_complete','hpo':counts,'tuning_jobs':len(tuning_rows),'screening_jobs':len(screening_rows)}))


if __name__=='__main__':main()
