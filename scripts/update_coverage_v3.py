#!/usr/bin/env python3
"""Coverage of declared jobs; raw diagnostic work is reported separately.

Passing a narrow test does not mark an entire broad correctness gate passed.
Gate receipts must explicitly attest that gate's scope before admission here.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',type=Path,default=Path('artifacts/run_plan_v3.json'))
    ap.add_argument('--root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/runs/v3'))
    ap.add_argument('--out',type=Path,default=Path('artifacts/coverage_v3.json')); args=ap.parse_args()
    plan=json.loads(args.plan.read_text()); rows=[]
    for run in plan['runs']:
        path=args.root/run['run_id']/'metadata.json'
        if run['stage']=='A_full_horizon_diagnostic':
            path=Path('/var/tmp/dora-adaptive-rank/runs/v3_full_horizon_checks_20261015')/run['method_id']/'metadata.json'
        status,reason='pending',''
        if path.exists():
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
            'note':'No broad gate is inferred from a narrow test. Conditional jobs stay pending until resolved by frozen rules.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['counts']))


if __name__=='__main__': main()
