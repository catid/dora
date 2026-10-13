"""Separate SciFact evaluation of the preserved revised standalone DoRALayer."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import change_report as common
import matplotlib.pyplot as plt

METHODS = ['ab_original', 'ab_revised', 'standalone_current']
LABELS = {'ab_original': 'Original implementation', 'ab_revised': 'Revised task recipe',
          'standalone_current': 'Preserved revised standalone DoRALayer'}
PAIRS = [('standalone_current', 'ab_original'), ('standalone_current', 'ab_revised')]
SCOPE = ('Separate exploratory SciFact rank8 evaluation of the preserved revised dora_row.DoRALayer.from_linear '
         'constructor and forward, without overrides. This is a complete standalone recipe '
         'comparison, not an isolated attribution: native BF16 device draws/storage, historical '
         'normal-up/zero-down orientation, row axis, detached FP32 norm, FP32 magnitude and '
         'factorized projection all coexist. Both preserved task-recipe controls reuse main-study '
         'fits; they are not new seeds. The actual standalone receives the same two-rate '
         'validation grid, training budget and seeds 42–44. Fixed LR is 0.001; tuned selection '
         'uses validation nDCG only. Read-only forward hooks recompute the norm to verify zero '
         'floor activations, adding a dense update/norm calculation each forward. Its recorded '
         'timing therefore cannot establish production speed relative to the task adapters. '
         'This existing-test-set comparison is exploratory, with three seeds and no multiplicity '
         'correction. Results concern this one SciFact protocol and do not replace the broader '
         'task-recipe ablation matrix.')

SCOPE += (' Official SciFact train/test provenance is retained: 726 training queries and an 81-query validation split after exact-text deduplication, 300 official test queries, 5,183 corpus documents, rank 8 and 115 updates per fit. Validation split seed 20261008; training seeds 42–44. This matches the prior local protocol, not the full DoRA paper benchmark suite.')


def compute(payload):
    data = payload['data']; comparisons = []
    assert set(data['lanes']) == set(common.LANES)
    for lane, cells in data['lanes'].items():
        assert set(cells) == set(METHODS)
        for cell in cells.values(): cell['summary'] = common.summarize_cell(cell)
        for candidate, reference in PAIRS:
            value = common.contrast(cells[candidate]['scores'], cells[reference]['scores'])
            common.close(value['difference'], cells[candidate]['summary']['mean'] - cells[reference]['summary']['mean'])
            comparisons.append({'lane': lane, 'candidate': candidate, 'reference': reference, **value})
    payload['comparisons'] = comparisons


def render(payload, output):
    output.mkdir(parents=True, exist_ok=True)
    common.matplotlib.rcParams['svg.hashsalt'] = 'dora-standalone-endpoint-20261009'
    cells = payload['data']['lanes']
    rows = [[LABELS[m], *[common.fmt(cells[lane][m]['summary']) for lane in common.LANES]] for m in METHODS]
    (output / 'quality_table.md').write_text(common.markdown_table(['Recipe', 'Fixed LR / nDCG@10 ×100', 'Validation tuned / nDCG@10 ×100'], rows) + '\nMean ± sample SD, seeds 42–44. Actual standalone is newly measured; both task-recipe controls reuse main-study fits.\n')
    rows = [[r['lane'], LABELS[r['candidate']] + ' − ' + LABELS[r['reference']], f"{r['difference']:+.3f}",
             f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"] for r in payload['comparisons']]
    (output / 'paired_effects.md').write_text(common.markdown_table(['Lane', 'Contrast', 'nDCG×100 difference', 'Exploratory 95% interval'], rows))
    rows=[]
    for lane in common.LANES:
        for method in METHODS:
            s=cells[lane][method]['summary']
            rows.append([lane,LABELS[method],', '.join(map(str,s['parameters'])),', '.join(f'{v:g}' for v in s['learning_rates']),common.fmt(s['training_seconds']),f"{max(s['peak_memory_bytes']['values'])/2**30:.3f}"])
    (output/'runtime_memory.md').write_text(common.markdown_table(['Lane','Recipe','Parameters','LR by seed','Training seconds, mean ± SD','Peak allocated GiB'],rows)+'\nStandalone timing includes an additional dense update/norm check per forward; core task timings include lighter norm tracking. Identical fit IDs represent one fit shared between lanes.\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis,lane in zip(axes,common.LANES):
        for index,method in enumerate(METHODS):
            s=cells[lane][method]['summary']
            axis.errorbar(index,s['mean'],yerr=s['sample_sd'],fmt='o',capsize=4,color=['#D55E00','#0072B2','#009E73'][index])
        axis.set_xticks(range(3),['Original\nimplementation','Revised\ntask recipe','Preserved revised\nstandalone'])
        axis.set_title('Fixed original LR' if lane==common.LANES[0] else 'Equal validation tuning')
        axis.set_ylabel('SciFact nDCG@10 ×100');axis.grid(axis='y',alpha=.2);axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Preserved revised standalone DoRALayer on the same SciFact protocol')
    fig.text(.5,.015,'Mean ± SD, 3 seeds. Task-recipe controls reused. Complete standalone endpoint, not isolated attribution.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(output/'standalone_endpoint_comparison.png',dpi=170)
    fig.savefig(output/'standalone_endpoint_comparison.svg',metadata={'Date':None});plt.close(fig)
    common.write(output/'comparisons.json',payload['comparisons']);common.write(output/'summary.json',payload)
    (output/'scope.md').write_text(SCOPE+'\n')
    (output/'reproduce.md').write_text('Verify raw_manifest.json and its archive/member hashes, then extract with Python tarfile filter="data". From the extracted directory:\n\n```bash\npython source/experiments/standalone_endpoint_report.py --rebuild report_inputs.json --output rebuilt\n```\n\nRebuild verifies retained evidence hashes, recomputes per-seed/query statistics and paired intervals, and regenerates outputs. NumPy and Matplotlib are required. No inference or omitted checkpoint replay.\n')


def collect(raw,repo):
    evidence=common.Evidence(raw,repo)
    core=evidence.json(Path('/var/tmp/dora-bench/change_ablation_core_validation.json'))
    gate=evidence.json(Path('/var/tmp/dora-bench/change_standalone_endpoint_validation.json'))
    assert gate['status']==gate['cuda_tests']['status']=='pass'
    assert gate['supplement_source_sha256']==common.sha(repo/'experiments/change_standalone_endpoint.py')
    common.bound_references(evidence,core);common.bound_references(evidence,gate)
    review=Path('/var/tmp/dora-bench/change_standalone_endpoint_independent_review.json')
    if review.exists():
        assessment=evidence.json(review)
        assert assessment['status']=='pass'
        for name,expected in assessment['source_sha256'].items():
            assert common.sha(repo/name)==expected,name
            evidence.add(repo/name)
    control=common.retrieval(evidence,core['recipes'],'scifact')['scifact']
    recipes={name:gate['recipes'][name] for name in ['standalone_current']}
    assert recipes['standalone_current']['implementation_sha256']==common.sha(repo/'dora_row.py')
    data=common.retrieval(evidence,recipes,'scifact',root_name='scifact_standalone_endpoint')['scifact']
    assert data['example_ids']==control['example_ids']
    for key in ('model','datasets','input_sha256'):assert data['manifest'][key]==control['manifest'][key],key
    for key,value in control['manifest']['configuration'].items():
        if key!='methods':assert data['manifest']['configuration'][key]==value,key
    for lane in common.LANES:
        for method in ('ab_original','ab_revised'):data['lanes'][lane][method]=control['lanes'][lane][method]
    data['control_manifest']=control['manifest'];data['control_audit']=control['audit']
    for path in (raw/'scifact_standalone_endpoint').rglob('*'):
        if path.is_file() and path.suffix in common.SUFFIXES and '__pycache__' not in path.parts:evidence.add(path)
    common.command_evidence(evidence, 'standalone_commands.json', '/var/tmp/dora-bench/run_standalone_endpoint.py')
    for name in ('experiments/change_report.py','experiments/standalone_endpoint_report.py','experiments/change_standalone_endpoint.py','experiments/test_change_standalone_endpoint.py'):
        evidence.add(repo/name)
    return {'complete':True,'kind':'actual_standalone_endpoint','data':data,'scope':SCOPE,
            'core_gate':core,'supplement_gate':gate,'source_sha256':common.sha(Path(__file__)),
            'dependency_sha256':common.sha(Path(common.__file__)),
            'versions':{'numpy':common.np.__version__,'matplotlib':common.matplotlib.__version__}},evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=Path('/var/tmp/dora-bench/change_ablation_runs'))
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,default=Path('results/2026-10-09-standalone-endpoint'))
    p.add_argument('--rebuild',type=Path);args=p.parse_args()
    if args.rebuild:
        payload=common.read(args.rebuild)
        assert common.sha(Path(__file__))==payload['source_sha256'] and common.sha(Path(common.__file__))==payload['dependency_sha256']
        for row in payload['evidence']:
            path=args.rebuild.parent/row['path'];assert path.stat().st_size==row['bytes'] and common.sha(path)==row['sha256'],path
        before=payload['comparisons'];compute(payload);assert payload['comparisons']==before;render(payload,args.output)
        common.write(args.output/'rebuild_audit.json',{'status':'pass','evidence_files_verified':len(payload['evidence']),
                     'scope':'All retained inputs hash-verified; seed/query summaries and paired intervals recomputed; outputs regenerated without inference.'})
    else:
        try:payload,evidence=collect(args.raw_root.resolve(),args.repo.resolve())
        except common.NotReady as error:p.exit(2,str(error)+'\n')
        compute(payload);args.output.mkdir(parents=True,exist_ok=True);common.bundle(payload,evidence,args.output);render(payload,args.output)
    print(common.json.dumps({'status':'complete','supplement':'standalone_endpoint','output':str(args.output)}))


if __name__=='__main__':main()
