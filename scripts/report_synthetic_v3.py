#!/usr/bin/env python3
"""Report retained synthetic runs without selecting an LLM configuration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/var/tmp/dora-adaptive-rank/runs/v3_synthetic_retained_20261015'))
    a=ap.parse_args();rows=[];failures=[];wall=0.;checkpoints=0
    for path in sorted(a.root.glob('*/metadata.json')):
        md=json.loads(path.read_text());wall+=md.get('cpu_wall_s',0.)
        if md['status']!='complete':failures.append({'path':str(path),'status':md['status'],'reason':md.get('failure_reason')});continue
        for r in json.loads((path.parent/'results.json').read_text()):
            for stage in r['stages']:
                ck=Path(stage['checkpoint']);desc=json.loads((ck/'manifest.json').read_text())
                assert desc['complete']
                assert all(hashlib.sha256((ck/n).read_bytes()).hexdigest()==h for n,h in desc['files'].items())
                checkpoints+=1
            rows.append({k:r[k] for k in ['method','seed','covariance','noise','budget_slots_equivalent',
                'P_target','active_scalars','allocated_scalars','frozen_adapter_bytes','dev_function_mse','dev_observed_mse','data_sha256']}
                |{'oracle_dev_function_mse':r['oracle']['dev_function_mse'],
                  'slots_on_zero_teacher_modules':r['rank_map']['layers.0']+r['rank_map']['layers.1'],
                  'rank_map_json':json.dumps(r['rank_map'],sort_keys=True),'source':str(path.parent/'results.json')})
    # Every paired candidate sees exactly the same problem and sample draws.
    by_problem={}
    for r in rows:by_problem.setdefault((r['seed'],r['covariance'],r['noise']),set()).add(r['data_sha256'])
    assert all(len(h)==1 for h in by_problem.values())
    with Path('reports/synthetic_v3.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    methods=[m['id'] for m in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']]
    aggregate=[]
    for method in methods:
        for budget in [16,32,64]:
            for covariance in ['isotropic','anisotropic']:
                for noise in [False,True]:
                    cells=[r for r in rows if (r['method'],r['budget_slots_equivalent'],r['covariance'],r['noise'])==(method,budget,covariance,noise)]
                    vals=[r['dev_function_mse'] for r in cells]
                    if not vals:continue
                    aggregate.append({'method':method,'budget_slots_equivalent':budget,'covariance':covariance,'noise':noise,
                        'n_data_seeds':len(vals),'mean_function_mse':statistics.mean(vals),
                        'sd_function_mse':statistics.stdev(vals) if len(vals)>1 else 0.,'median_function_mse':statistics.median(vals)})
    with Path('reports/synthetic_v3_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]),lineterminator='\n');writer.writeheader();writer.writerows(aggregate)
    fig,axes=plt.subplots(2,2,figsize=(24,13),sharex=True)
    for ax,(cov,noise) in zip(axes.flat,[(c,n) for c in ['isotropic','anisotropic'] for n in [False,True]]):
        subset={r['method']:r for r in aggregate if r['budget_slots_equivalent']==32 and r['covariance']==cov and r['noise']==noise}
        vals=[subset[m]['mean_function_mse'] for m in methods]
        sds=[subset[m]['sd_function_mse'] for m in methods]
        ax.bar(np.arange(len(methods)),vals,color=['#999999' if m.endswith('_WIDE') else '#3b6fa5' for m in methods])
        ax.errorbar(np.arange(len(methods)),vals,yerr=sds,fmt='none',ecolor='black',linewidth=.6,capsize=1.5)
        ax.set_yscale('log');ax.set_title(f'{cov}; target noise={0.03 if noise else 0}');ax.set_ylabel('Dev function MSE (log scale)')
        ax.grid(axis='y',alpha=.2);ax.set_xticks(np.arange(len(methods)),methods,rotation=90,fontsize=7)
    fig.suptitle('Synthetic controls: 32-slot-equivalent budget; mean ± SD over two data seeds\nCommon LR/scale; gray wide controls exceed target budget; not a tuned LLM comparison')
    fig.tight_layout();Path('reports/figures').mkdir(exist_ok=True);fig.savefig('reports/figures/synthetic_v3_rank4.png',dpi=150);plt.close(fig)
    extra=[]
    for root in [Path('/var/tmp/dora-adaptive-rank/runs/v3'),Path('/var/tmp/dora-adaptive-rank/runs/v3_synthetic_engineering_20261015')]:
        ms=[json.loads(p.read_text()) for p in root.glob('A_syn_*/metadata.json')]
        extra.append({'root':str(root),'completed_jobs':sum(m['status']=='complete' for m in ms),
                      'summed_job_wall_seconds':sum(m.get('cpu_wall_s',0.) for m in ms),
                      'role':'earlier engineering/retention-incomplete executions; charged separately, not extra independent seeds'})
    receipt={'status':'passed' if len(rows)==984 and not failures else 'incomplete','retained_jobs':len({r['source'] for r in rows}),
             'fits':len(rows),'verified_stage_checkpoints':checkpoints,'summed_job_wall_seconds':wall,'failures':failures,
             'paired_data_hashes_equal':True,'historical_engineering_cost':extra,
             'missing_stage_A_scope':['signed deletion/curvature/gauge diagnostics','joint deletion damage and recovery diagnostics'],
             'selection_eligible':False}
    Path('artifacts/synthetic_v3_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    lookup={(r['method'],r['budget_slots_equivalent'],r['covariance'],r['noise']):r for r in aggregate}
    fixed_l=lookup[('L_FIXED',32,'isotropic',False)]['mean_function_mse']
    fixed_i=lookup[('I_FIXED',32,'isotropic',False)]['mean_function_mse']
    compress_i=lookup[('I_GLOBAL_SVD',32,'isotropic',False)]['mean_function_mse']
    lines=['# Corrected synthetic controls','',
        '| Measured finding at average rank four | Evidence | Limit |',
        '|---|---|---|',
        f'| Fixed standard LoRA has lower clean isotropic function error than fixed IniLoRA | MSE {fixed_l:.6f} versus {fixed_i:.6f} | Common LR/scale, one teacher and two data seeds; not tuned superiority. |',
        f'| IniLoRA global compression/recovery substantially changes the outcome | MSE {compress_i:.6f}, versus fixed IniLoRA {fixed_i:.6f} | Source width, refactorization, optimizer reset and recovery are combined; their separate causal contributions remain unmeasured. |',
        '| No LLM method ranking follows from this toy task | All primary LLM tuning/independent screening remains pending | Synthetic allocation/compression behavior is a diagnostic. |', '',
        f"Completed {receipt['retained_jobs']} jobs / {len(rows)} fits, with {checkpoints} verified adapter/optimizer stage checkpoints.",
        '', 'All 41 cells receive the same 100-update LR=0.01, scale=1 configuration at three feasible scalar budgets. '
        'Two data seeds share one fixed known teacher. This is a common-configuration diagnostic, not equally tuned performance '
        'and not a basis for selecting LLM finalists. Wide controls exceed target budget; reference AdaLoRA retains its native extra coefficients.',
        '', 'The CSV files preserve every seed, noise/covariance case, parameter count and rank map. '
        'The known-teacher covariance-weighted oracle is explicitly separate from learned methods. '
        'Repeated engineering executions are counted as cost, not independent seed replications.',
        '', '![All-method synthetic comparison at average rank four](figures/synthetic_v3_rank4.png)', '',
        'Raw rows: `synthetic_v3.csv`; mean/SD/median: `synthetic_v3_summary.csv`; evidence/costs: `../artifacts/synthetic_v3_audit.json`.',
        '', 'At the 32-slot-equivalent budget, every registered cell is shown below. Values are mean ± sample SD '
        'over two data seeds; lower function error is better. Wide controls exceed the final budget, and faithful '
        'AdaLoRA retains its native E-scalar excess. Seed-level counts and all three budgets are in the CSV.', '',
        '| Method | Isotropic clean | Isotropic noisy | Anisotropic clean | Anisotropic noisy |',
        '|---|---:|---:|---:|---:|']
    for method in methods:
        vals=[lookup[(method,32,c,n)] for c,n in [('isotropic',False),('isotropic',True),('anisotropic',False),('anisotropic',True)]]
        lines.append('| '+method+' | '+' | '.join(f"{r['mean_function_mse']:.5f} ± {r['sd_function_mse']:.5f}" for r in vals)+' |')
    lines+=['', 'Signed deletion, finite-difference curvature, gauge and interacting-group diagnostics remain pending. '
        'Two data seeds have low power and do not establish broad method superiority.']
    Path('reports/synthetic_v3.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:receipt[k] for k in ['status','fits','verified_stage_checkpoints']}))


if __name__=='__main__':main()
