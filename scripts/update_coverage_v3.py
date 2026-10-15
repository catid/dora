#!/usr/bin/env python3
"""Coverage of declared jobs; raw diagnostic work is reported separately.

Passing a narrow test does not mark an entire broad correctness gate passed.
Gate receipts must explicitly attest that gate's scope before admission here.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',type=Path,default=Path('artifacts/run_plan_v3.json'))
    ap.add_argument('--root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/runs/v3'))
    ap.add_argument('--synthetic-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/runs/v3_synthetic_retained_20261015'))
    ap.add_argument('--out',type=Path,default=Path('artifacts/coverage_v3.json'))
    ap.add_argument('--hpo-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/runs/v3_hpo_20261015_retry'))
    ap.add_argument('--tuning-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/runs/v3_tuning_20261015'))
    ap.add_argument('--screening-root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/runs/v3_screening_20261015'))
    args=ap.parse_args()
    plan=json.loads(args.plan.read_text()); rows=[]
    for run in plan['runs']:
        path=args.root/run['run_id']/'metadata.json'
        # The Stage B launchers use isolated roots and intentionally retain
        # their own immutable names.  Map those records back to the declared
        # plan IDs so coverage is live and machine-readable while jobs run.
        if run['stage']=='B_hpo':
            path=args.hpo_root/f"{run['method_id']}_c{int(run['run_id'].rsplit('c', 1)[1]):02d}"/'metadata.json'
        elif run['stage']=='B_screen':
            path=args.screening_root/f"{run['method_id']}_s{run['seed']}"/'metadata.json'
        elif run['stage']=='B_tuning':
            method, seed = run['method_id'], int(run['seed'])
            candidates=[]
            for candidate in sorted(args.tuning_root.glob(f'{method}_c*_s{seed}')):
                config=candidate/'tuning_config.json'
                try:
                    trial=int(json.loads(config.read_text())['trial_index']) if config.exists() else 10**9
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    trial=10**9
                candidates.append((trial, candidate))
            candidates.sort(key=lambda item: (item[0], str(item[1])))
            index=int(run.get('shortlist_index', 0))
            if index < len(candidates): path=candidates[index][1]/'metadata.json'
        if run['stage']=='A_synthetic':
            path=args.synthetic_root/run['run_id']/'metadata.json'
        if run['stage']=='A_full_horizon_diagnostic':
            path=Path('/var/tmp/dora-adaptive-rank/runs/v3_full_horizon_checks_20261015')/run['method_id']/'metadata.json'
        status,reason='pending',''
        launcher = path.parent/'launcher_status.json'
        if path.exists() or launcher.exists():
            if not path.exists() and launcher.exists():
                live=json.loads(launcher.read_text())
                if live.get('status')=='running':
                    rows.append({'run_id':run['run_id'],'stage':run['stage'],'method_id':run['method_id'],
                                 'selection_role':run['selection_role'],'status':'pending',
                                 'reason':'launcher running; terminal receipt pending','conditional':run.get('conditional'),
                                 'evidence':str(launcher)})
                    continue
                if live.get('status') in {'complete', 'failed'}:
                    rows.append({'run_id':run['run_id'],'stage':run['stage'],'method_id':run['method_id'],
                                 'selection_role':run['selection_role'],
                                 'status':'failed' if live.get('status')=='failed' else 'pending',
                                 'reason':'launcher terminal but metadata receipt is missing',
                                 'conditional':run.get('conditional'),'evidence':str(launcher)})
                    continue
            record=json.loads(path.read_text())
            state=record['status']
            if state in {'complete','passed'}:
                status='passed'
            elif state=='failed':
                status='failed'; reason=record.get('failure_reason','')
            elif state=='running':
                reason='terminal receipt pending; inspect live PID separately'
        rows.append({'run_id':run['run_id'],'stage':run['stage'],'method_id':run['method_id'],
                     'selection_role':run['selection_role'],'status':status,'reason':reason,
                     'conditional':run.get('conditional'),'evidence':str(path) if path.exists() else None})
    counts={s:sum(r['status']==s for r in rows) for s in ['passed','failed','blocked','pending','not_in_scope']}
    quarantine=json.loads(Path('artifacts/protocol_v2_quarantine.json').read_text())
    smoke_path=Path('reports/smoke_v3_summary.json')
    smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    result={'schema':'rankstudy-v3-coverage','plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            'counts':{'total':len(rows),**counts},'runs':rows,
            'separate_historical_accounting':{'v2_quarantined_completed_trials':len(quarantine['runs']),
                'v3_completed_smoke_executions':smoke.get('completed',0),'v3_failed_smoke_attempts':smoke.get('failed',0)},
            'note':'No broad gate is inferred from a narrow test. Conditional jobs stay pending until resolved by frozen rules.',
            'live_roots': {'hpo':str(args.hpo_root),'tuning':str(args.tuning_root),'screening':str(args.screening_root)}}
    args.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['counts']))
    with Path('reports/coverage_v3.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


if __name__=='__main__': main()
