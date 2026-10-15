#!/usr/bin/env python3
"""Generate smoke evidence from raw records; never an HPO selection input."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--roots',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    rows=[]; failures=[]
    for root in args.roots:
        for p in sorted(root.glob('*/metadata.json')):
            m=json.loads(p.read_text())
            if m.get('status')=='failed':
                failures.append({'run_id':p.parent.name,'path':str(p.parent),'method':m['method'],
                    'reason':m.get('failure_reason'),'metadata_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            if not m.get('smoke') or m['status']!='complete':
                continue
            before=p.parent/'smoke_before.json'
            prior=json.loads(before.read_text())['group_balanced_nll'] if before.exists() else None
            children=[json.loads(Path(stage['metadata']).read_text()) for stage in m.get('stages',[])]
            backend=m.get('backend_requested') or ','.join(sorted({child['backend_requested'] for child in children}))
            deterministic=m.get('deterministic_algorithms',bool(children) and all(child.get('deterministic_algorithms',False) for child in children))
            rows.append({'run_id':p.parent.name,'method':m['method'],'status':m['status'],
                         'execution_kind':'pipeline' if children else 'native',
                         'updates':m['updates_finished'],'seed':m['seed'],
                         'backend':backend,'deterministic':deterministic,
                         'partial_dev_groups':m['config']['dev_limit'],
                         'before_nll':prior,'after_nll':m['dev_tune_endpoint_group_balanced_nll'],
                         'input_tokens':m['input_tokens'],'active_trainable_scalars':m['final_counts']['active_trainable_scalars'],
                         'allocated_trainable_scalars':m['final_counts']['allocated_trainable_scalars'],
                         'P_target':m['P_target'],'budget_delta':m['budget_delta'],
                         'charged_gpu_seconds':m['charged_gpu_seconds'],
                         'profiled':m['config'].get('profile',False) or any(c['config'].get('profile',False) for c in children),
                         'original_backbone_unchanged':m['backbone_unchanged'],
                         'path':str(p.parent),'metadata_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                         'selection_eligible':False})
    out=Path('reports/smoke_v3.csv')
    if rows:
        with out.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    Path('reports/smoke_v3_summary.json').write_text(json.dumps({'type':'engineering_smoke_not_tournament',
        'partial_dev_only':True,'completed':len(rows),'failed':len(failures),'rows':rows,'failures':failures},indent=2)+'\n')
    if failures:
        with Path('reports/failures_v3.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(failures[0])); writer.writeheader(); writer.writerows(failures)
    print(json.dumps({'completed_smoke_executions':len(rows),'failed_smoke_executions':len(failures),'primary_selection_eligible':0}))


if __name__=='__main__': main()
