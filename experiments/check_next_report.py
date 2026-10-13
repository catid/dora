"""Independent raw-score, chart-coordinate and archive replay audit of next_report.

No model or tensor checkpoint is loaded. Strict task audits separately establish
local checkpoint integrity. This checker recomputes quality from archived raw
predictions, reconstructs paired intervals, then requires exact report outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile

import numpy as np

METHODS = ['next_base', 'next_axis', 'next_detach', 'next_norm_fp32', 'next_init', 'next_factorized']
LANES = ['fixed_original_lr', 'validation_tuned']
TASKS = ['scifact', 'nfcorpus', 'scifact_transfer', 'aircraft', 'extraction']
OUTPUTS = ['fixed_original_lr_table.md', 'validation_tuned_table.md', 'paired_effects.md',
           'runtime_memory.md', 'microbenchmark.md', 'microbenchmark.json',
           'next_change_bars.png', 'next_change_bars.svg', 'next_change_tuned.png',
           'next_change_tuned.svg', 'summary.json', 'comparisons.json', 'scope.md',
           'protocol.md', 'reproduce.md']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def close(a, b, tolerance=1e-9):
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, (a,b)


def ndcg(ranking, relevant):
    assert len(ranking) == len(set(ranking)) == 10
    gains = [2.0 ** relevant.get(doc, 0) - 1 for doc in ranking]
    ideal = sorted([2.0 ** grade - 1 for grade in relevant.values()], reverse=True)[:10]
    denominator = sum(value / math.log2(i + 2) for i,value in enumerate(ideal))
    assert denominator > 0
    return sum(value / math.log2(i + 2) for i,value in enumerate(gains)) / denominator


def per_seed(values, strata):
    matrix = np.asarray(values, dtype=np.float64)
    assert matrix.ndim == 2 and matrix.shape[0] == 3 and np.isfinite(matrix).all()
    if strata is None: return matrix.mean(axis=1) * 100
    labels = np.asarray(strata)
    assert matrix.shape[1] == len(labels)
    return np.array([matrix[:,labels == label].mean(axis=1) for label in np.unique(labels)]).mean(axis=0) * 100


def independent_interval(candidate, baseline, strata):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    labels = np.asarray(strata) if strata is not None else np.zeros(delta.shape[1], dtype=np.int64)
    categories = [np.where(labels == label)[0] for label in np.unique(labels)]
    rng = np.random.default_rng(20261009)
    draws = []
    for _ in range(100):
        seed_draws = rng.integers(3, size=(100,3))
        sampled = np.zeros(100)
        for category in categories:
            example_draws = rng.choice(category, size=(100,len(category)))
            selected = delta[seed_draws[:,:,None], example_draws[:,None,:]]
            sampled += selected.mean(axis=(1,2)) / len(categories)
        draws.extend(sampled.tolist())
    return np.quantile(draws,[.025,.975]) * 100


def raw_values(extracted, task, data, record, index):
    ids = data['example_ids']
    if task in ('scifact','nfcorpus','scifact_transfer'):
        dataset = 'scifact' if task == 'scifact_transfer' else task
        trained = 'nfcorpus' if task == 'scifact_transfer' else task
        directory = extracted / 'runs' / record['fit_id']
        assert sha(directory/'result.json') == record['artifact_result_sha256']
        qrels = read(extracted/'runs'/trained/(dataset+'_qrels.json'))
        splits = read(extracted/'runs'/trained/(dataset+'_splits.json'))
        assert splits['test'] == ids
        predictions = read(directory/dataset/'test_per_query.json')
        byid = {item['query_id']:item for item in predictions}
        assert len(byid) == len(predictions) == len(ids) and set(byid) == set(ids)
        values = [ndcg(byid[qid]['top_10'],qrels[qid]) for qid in ids]
        for qid,value in zip(ids,values): close(value,byid[qid]['ndcg_at_10'],1e-12)
        return values
    if task == 'extraction':
        directory = extracted/'runs'/record['fit_id']
        assert sha(directory/'result.json') == record['artifact_result_sha256']
        gold = read(extracted/'runs/extraction/split_manifest.json')['test']
        assert [row['id'] for row in gold] == ids
        predictions = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        byid = {row['id']:row for row in predictions}
        assert len(byid) == len(predictions) == len(gold) and set(byid) == set(ids)
        values=[]
        for target in gold:
            row=byid[target['id']]
            assert row['gold']==target['target'] and row['utterance']==target['utterance']
            try: decoded=json.loads(row['text'].strip())
            except (ValueError,TypeError): decoded=None
            assert decoded==row['parsed']
            value=float(decoded==target['target'])
            assert bool(value)==row['exact_match']
            values.append(value)
        return values
    assert task=='aircraft'
    result_paths = [extracted/name for name in index[record['artifact_result_sha256']] if name.endswith('/result.json')]
    assert result_paths
    arrays=[]
    for result_path in result_paths:
        raw=read(result_path)
        if 'test' not in raw or 'macro_class_accuracy' not in raw['test']: continue
        prediction=read(result_path.parent/'test_predictions.json')
        assert prediction['labels']==data['strata']
        assert len(prediction['labels'])==len(prediction['predictions'])==len(ids)==3333
        assert len(set(prediction['labels']))==100
        values=[float(a==b) for a,b in zip(prediction['labels'],prediction['predictions'])]
        arrays.append(values)
        close(float(per_seed([values]*3,data['strata'])[0]),raw['test']['macro_class_accuracy']*100,2e-5)
    assert arrays and all(row==arrays[0] for row in arrays)
    return arrays[0]


def verify_raw(payload, extracted, manifest):
    assert payload['complete'] and list(payload['tasks'])==TASKS
    assert set(payload['recipes'])==set(METHODS)
    flags=('axis','detach_norm','magnitude_fp32','norm_fp32','published_init','factorized','clamp_norm')
    expected_flags=dict(zip(METHODS,[[],['axis'],['detach_norm'],['norm_fp32'],['published_init'],['factorized']]))
    base=payload['recipes']['next_base']
    for method,recipe in payload['recipes'].items():
        assert recipe['magnitude_fp32'] and not recipe['clamp_norm']
        changed=[flag for flag in flags if recipe[flag]!=base[flag]]
        assert changed==recipe['changed_fields_from_baseline']==expected_flags[method]
    index={}
    for row in manifest['files']: index.setdefault(row['sha256'],[]).append(row['path'])
    cells=records=prediction_rows=0
    for task,data in payload['tasks'].items():
        assert set(data['lanes'])==set(LANES)
        if task=='aircraft':
            assert [row['image'] for row in data['split_manifest']['splits']['test']]==data['example_ids']
            assert [row['label'] for row in data['split_manifest']['splits']['test']]==data['strata']
        for lane,methods in data['lanes'].items():
            assert set(methods)==set(METHODS)
            for method,cell in methods.items():
                assert len(cell['records'])==len(cell['scores'])==3
                assert [r['seed'] for r in cell['records']]==[42,43,44]
                recomputed=[]
                for record,stored in zip(cell['records'],cell['scores']):
                    values=raw_values(extracted,task,data,record,index)
                    np.testing.assert_allclose(values,stored,rtol=0,atol=1e-12)
                    recomputed.append(values)
                    records+=1;prediction_rows+=len(values)
                means=per_seed(recomputed,data['strata']).tolist()
                for record,value in zip(cell['records'],means): close(record['value'],value,2e-5)
                summary=cell['summary']
                assert summary['n']==3
                np.testing.assert_allclose(summary['values'],means,rtol=0,atol=1e-9)
                close(summary['mean'],statistics.mean(means));close(summary['sample_sd'],statistics.stdev(means))
                assert summary['fit_ids']==[r['fit_id'] for r in cell['records']]
                assert summary['learning_rates']==[r['learning_rate'] for r in cell['records']]
                if task=='aircraft': assert all(r['reused_historical_fit'] for r in cell['records'])
                cells+=1
    expected={(task,lane,method) for task in TASKS for lane in LANES for method in METHODS[1:]}
    comparisons=payload['comparisons']
    assert len(comparisons)==50 and {(r['task'],r['lane'],r['candidate']) for r in comparisons}==expected
    for row in comparisons:
        assert row['reference']=='next_base' and row['replicates']==10000 and row['random_seed']==20261009
        data=payload['tasks'][row['task']];methods=data['lanes'][row['lane']]
        candidate=methods[row['candidate']]['scores'];baseline=methods['next_base']['scores']
        differences=per_seed(np.asarray(candidate)-np.asarray(baseline),data['strata'])
        np.testing.assert_allclose(differences,row['paired_seed_differences'],rtol=0,atol=1e-9)
        close(float(differences.mean()),row['difference'])
        np.testing.assert_allclose(independent_interval(candidate,baseline,data['strata']),row['ci95'],rtol=0,atol=1e-9)
    assert (cells,records)==(60,180)
    return {'cells':cells,'seed_records':records,'record_prediction_rows_rescored':prediction_rows,'paired_contrasts':len(comparisons)}


def verify_microbenchmark(payload, extracted):
    measured=read(extracted/'runs/microbenchmark/results.json')
    assert measured==payload['microbenchmark'] and measured['complete']
    assert len(measured['cases'])==36 and measured['recipes']==payload['recipes']
    rows={(r['method'],r['precision'],r['in_features'],r['out_features'],r['tokens']):r
          for r in measured['cases']}
    assert len(rows)==36
    samples=0
    for key,row in rows.items():
        base=rows[('next_base',*key[1:])]
        assert row['validation']['status']=='pass'
        assert len(row['validation']['checks'])==5
        assert all(v['bitwise_equal'] and v['max_absolute_error']==0 for v in row['validation']['checks'].values())
        for mode in ('forward','forward_backward'):
            timing=row[mode];values=timing['sample_ms_per_call']
            assert len(values)==7 and all(math.isfinite(v) and v>0 for v in values)
            assert statistics.median(values)==timing['median_ms_per_call']
            assert timing['peak_incremental_allocated_bytes']>0
            close(row['relative_to_next_base'][mode]['speedup'],base[mode]['median_ms_per_call']/timing['median_ms_per_call'],1e-12)
            close(row['relative_to_next_base'][mode]['peak_incremental_memory_ratio'],timing['peak_incremental_allocated_bytes']/base[mode]['peak_incremental_allocated_bytes'],1e-12)
            samples+=len(values)
    return {'cases':36,'timing_samples':samples,'scope':'Raw timing medians, relative speed/memory ratios and diagnostic-disabled parity receipts verified.'}


CAPTURE = '''import json,runpy,sys
from pathlib import Path
import matplotlib.axes
import numpy as np
captured={"bars":[],"errors":[]}
barh=matplotlib.axes.Axes.barh
errorbar=matplotlib.axes.Axes.errorbar
def bars(self,y,width,*args,**kwargs):
 captured["bars"].append({"y":np.asarray(y).tolist(),"values":np.asarray(width).tolist()})
 return barh(self,y,width,*args,**kwargs)
def errors(self,x,y,*args,**kwargs):
 captured["errors"].append({"y":np.asarray(y).tolist(),"values":np.asarray(x).tolist(),"xerr":np.asarray(kwargs["xerr"]).tolist()})
 return errorbar(self,x,y,*args,**kwargs)
matplotlib.axes.Axes.barh=bars
matplotlib.axes.Axes.errorbar=errors
sys.argv=["experiments.next_report","--rebuild","report_inputs.json","--output","rebuilt"]
runpy.run_module("experiments.next_report",run_name="__main__")
Path("plot_capture.json").write_text(json.dumps(captured,indent=2)+"\\n")
'''


def verify_chart(payload, captured):
    order=[(task,lane) for lane in LANES for task in TASKS]+[(task,'validation_tuned') for task in TASKS]
    assert len(captured['bars'])==len(captured['errors'])==len(order)==15
    for (task,lane),bar,error in zip(order,captured['bars'],captured['errors']):
        rows=[next(row for row in payload['comparisons'] if (row['task'],row['lane'],row['candidate'])==(task,lane,method)) for method in METHODS[1:]]
        means=[r['difference'] for r in rows]
        assert bar['y']==error['y']==list(range(5))
        assert bar['values']==error['values']==means
        expected=[[max(r['difference']-r['ci95'][0],0) for r in rows],
                  [max(r['ci95'][1]-r['difference'],0) for r in rows]]
        assert error['xerr']==expected
    return {'bar_series':15,'plotted_mean_differences':75,'paired_interval_whiskers':75,
            'scope':'Captured exact Matplotlib bar and errorbar arguments during archived-source rebuild; every value matches audited contrast.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/2026-10-09-next-change'))
    args=p.parse_args();output=args.output.resolve()
    manifest=read(output/'raw_manifest.json');payload=read(output/'summary.json')
    work=Path(tempfile.mkdtemp(prefix='next_report_offline_',dir='/var/tmp/dora-bench'))
    archive=work/'bundle.tar.gz'
    parts=manifest.get('archive_parts',[{'path':manifest.get('archive_path'),'bytes':manifest['archive_bytes'],'sha256':manifest['archive_sha256']}])
    with archive.open('wb') as stream:
        for row in parts:
            path=output/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            with path.open('rb') as source: shutil.copyfileobj(source,stream)
    assert archive.stat().st_size==manifest['archive_bytes'] and sha(archive)==manifest['archive_sha256']
    extracted=work/'extracted';extracted.mkdir()
    with tarfile.open(archive,'r:gz') as tar:
        assert sorted(tar.getnames())==sorted(row['path'] for row in manifest['files'])
        tar.extractall(extracted,filter='data')
    for row in manifest['files']:
        path=extracted/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    assert read(extracted/'report_inputs.json')==payload
    assert read(output/'comparisons.json')==payload['comparisons']
    raw=verify_raw(payload,extracted,manifest)
    micro=verify_microbenchmark(payload,extracted)
    wrapper=work/'capture_rebuild.py';wrapper.write_text(CAPTURE)
    env={**os.environ,'PYTHONPATH':str(extracted/'source'),'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1'}
    command=[sys.executable,str(wrapper)]
    result=subprocess.run(command,cwd=extracted,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (work/'rebuild.stdout.log').write_text(result.stdout);(work/'rebuild.stderr.log').write_text(result.stderr)
    assert result.returncode==0,(result.returncode,result.stdout,result.stderr)
    chart=verify_chart(payload,read(extracted/'plot_capture.json'))
    checked=[]
    for name in OUTPUTS:
        assert sha(output/name)==sha(extracted/'rebuilt'/name),name
        checked.append({'path':name,'sha256':sha(output/name)})
    shutil.copyfile(Path(__file__),output/'offline_rebuild_source.py')
    audit={'status':'pass','checker_source_sha256':sha(Path(__file__)),
           'raw_manifest_sha256':sha(output/'raw_manifest.json'),'archive_sha256':manifest['archive_sha256'],
           'parts_verified':len(parts),'members_verified':len(manifest['files']),'raw_quality_checks':raw,
           'microbenchmark_checks':micro,
           'chart_data_checks':chart,'exact_outputs':checked,'rebuild_audit':read(extracted/'rebuilt/rebuild_audit.json'),
           'command':command,'work_directory':str(work),
           'scope':'Fresh archived-source rebuild with every retained file hash verified. Independently rescored predictions for every cell, recomputed all paired intervals and captured chart arguments, then required all15 derived outputs byte-identical. Tensor checkpoint payloads are omitted; no inference replay.'}
    (output/'offline_rebuild_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':'pass','members':len(manifest['files']),'exact_outputs':len(checked),'raw':raw,'chart':chart,'work_directory':str(work)},indent=2))


if __name__=='__main__':main()
