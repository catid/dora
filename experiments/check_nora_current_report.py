"""Independent raw-score, chart-coordinate and archive replay audit of nora_current_report.

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

METHODS = ['current_dora', 'current_dora_nora', 'init_dora', 'init_dora_nora']
PAIRS = [('current_dora_nora','current_dora'), ('init_dora_nora','init_dora'),
         ('init_dora','current_dora'), ('init_dora_nora','current_dora')]
KINDS = ['literal_nora','initialized_nora','initializer','combined']
LABELS = ['Accepted DoRA','Accepted DoRA + NoRA','Alternate initializer','Alternate initializer + NoRA']
CONTRAST_LABELS = ['NoRA: original initializer','NoRA: alternate initializer',
                   'Initializer alone','Initializer + NoRA vs accepted']
TASK_LABELS = ['SciFact direct r8','NFCorpus r8','SciFact transfer r8','Aircraft r2','ViGGO r8']
LANES = ['fixed_original_lr', 'validation_tuned']
TASKS = ['scifact', 'nfcorpus', 'scifact_transfer', 'aircraft', 'extraction']
OUTPUTS = ['fixed_original_lr_table.md', 'validation_tuned_table.md', 'paired_effects.md',
           'runtime_memory.md', 'protocols.md', 'nora_fixed.png', 'nora_fixed.svg',
           'nora_tuned.png', 'nora_tuned.svg', 'summary.json', 'comparisons.json',
           'chart_coordinates.json', 'scope.md', 'protocol.md', 'reproduce.md',
           'nora_scores.png', 'nora_scores.svg', 'score_chart_coordinates.json']


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
    directory = extracted / 'runs' / record['fit_id']
    assert sha(directory/'result.json') == record['artifact_result_sha256']
    raw = read(directory/'result.json')
    assert raw['seed'] == record['seed']
    if task in ('scifact','nfcorpus','scifact_transfer'):
        close(raw['learning_rate'],record['learning_rate'])
        close(raw['training']['seconds'],record['training_seconds'])
        assert raw['training']['peak_allocated_bytes']==record['peak_memory_bytes']
    elif task=='extraction':
        close(raw['lr'],record['learning_rate'])
        close(raw['train_and_validation_seconds'],record['training_seconds'])
        assert raw['peak_allocated_bytes']==record['peak_memory_bytes']
        assert raw['selected_step']==record['selected_step']
    else:
        close(raw['adapter_learning_rate'],record['learning_rate'])
        close(raw['training_wall_seconds'],record['training_seconds'])
        assert raw['peak_cuda_allocated_bytes']==record['peak_memory_bytes']
        assert raw['best_epoch']==record['selected_epoch']
        assert record['fresh_fit'] and raw['method'] in METHODS
    assert raw['trainable_parameters']==record['trainable_parameters']
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
    assert payload['gate']['status']==payload['gate']['cpu_tests']['status']==payload['gate']['cuda_tests']['status']=='pass'
    assert payload['gate']['recipes']==payload['recipes']
    for method,recipe in payload['recipes'].items():
        hashed={key:value for key,value in recipe.items() if key not in
                ('recipe_sha256','source_sha256','inherited_next_source_sha256',
                 'inherited_core_source_sha256','public_source_sha256')}
        assert hashlib.sha256(json.dumps(hashed,sort_keys=True,separators=(',',':')).encode()).hexdigest()==recipe['recipe_sha256']
        assert recipe['axis']==0 and recipe['magnitude_fp32'] and not recipe['clamp_norm']
        assert not recipe['detach_norm'] and not recipe['norm_fp32'] and not recipe['factorized']
        assert recipe['published_init']==method.startswith('init_')
        assert recipe['nora']==method.endswith('_nora') and recipe['nora_epsilon']==1e-12
        assert recipe['zero_initialized_factor']==('lora_B' if method.startswith('init_') else 'lora_A')
        assert recipe['base_method']==('next_init' if method.startswith('init_') else 'next_base')
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
                    measured=read(extracted/'runs'/record['fit_id']/'result.json')
                    assert measured['method']==method
                    assert record['recipe_sha256']==payload['recipes'][method]['recipe_sha256']
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
                for field in ('training_seconds','peak_memory_bytes'):
                    observed=[r[field] for r in cell['records']]
                    assert summary[field]['n']==3 and summary[field]['values']==observed
                    close(summary[field]['mean'],statistics.mean(observed))
                    close(summary[field]['sample_sd'],statistics.stdev(observed))
                assert summary['parameters']==sorted({r['trainable_parameters'] for r in cell['records']})
                if task=='aircraft': assert all(not r.get('reused_historical_fit',False) and not r['fit_id'].startswith('historical/') for r in cell['records'])
                cells+=1
    expected={(task,lane,candidate,reference) for task in TASKS for lane in LANES for candidate,reference in PAIRS}
    comparisons=payload['comparisons']
    assert len(comparisons)==40 and {(r['task'],r['lane'],r['candidate'],r['reference']) for r in comparisons}==expected
    for row in comparisons:
        assert (row['candidate'],row['reference']) in PAIRS and row['replicates']==10000 and row['random_seed']==20261009
        index=PAIRS.index((row['candidate'],row['reference']))
        assert row['kind']==KINDS[index] and row['label']==CONTRAST_LABELS[index]
        data=payload['tasks'][row['task']];methods=data['lanes'][row['lane']]
        candidate=methods[row['candidate']]['scores'];baseline=methods[row['reference']]['scores']
        differences=per_seed(np.asarray(candidate)-np.asarray(baseline),data['strata'])
        np.testing.assert_allclose(differences,row['paired_seed_differences'],rtol=0,atol=1e-9)
        close(float(differences.mean()),row['difference'])
        np.testing.assert_allclose(independent_interval(candidate,baseline,data['strata']),row['ci95'],rtol=0,atol=1e-9)
    assert (cells,records)==(40,120)
    return {'cells':cells,'seed_records':records,'record_prediction_rows_rescored':prediction_rows,'paired_contrasts':len(comparisons)}


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
sys.argv=["experiments.nora_current_report","--rebuild","report_inputs.json","--output","rebuilt"]
runpy.run_module("experiments.nora_current_report",run_name="__main__")
Path("plot_capture.json").write_text(json.dumps(captured,indent=2)+"\\n")
'''


def verify_chart(payload, captured, coordinates, score_coordinates):
    order=[(task,lane) for lane in LANES for task in TASKS]
    assert len(order)==10 and len(captured['bars'])==len(captured['errors'])==15
    expected_coordinates=[]
    for (task,lane),bar,error in zip(order,captured['bars'],captured['errors']):
        rows=[next(row for row in payload['comparisons'] if (row['task'],row['lane'],row['candidate'],row['reference'])==(task,lane,candidate,reference)) for candidate,reference in PAIRS]
        means=[r['difference'] for r in rows]
        assert bar['y']==error['y']==list(range(4))
        assert bar['values']==error['values']==means
        expected=[[max(r['difference']-r['ci95'][0],0) for r in rows],
                  [max(r['ci95'][1]-r['difference'],0) for r in rows]]
        assert error['xerr']==expected
        expected_coordinates.extend({'task':task,'lane':lane,'kind':row['kind'],'y':i,
                                     'value':row['difference'],'ci95':row['ci95']} for i,row in enumerate(rows))
    assert coordinates==expected_coordinates
    expected_scores=[]
    for task,bar,error in zip(TASKS,captured['bars'][10:],captured['errors'][10:]):
        data=payload['tasks'][task]
        means=[];deviations=[]
        for i,method in enumerate(METHODS):
            values=per_seed(data['lanes']['validation_tuned'][method]['scores'],data['strata']).tolist()
            mean,sd=statistics.mean(values),statistics.stdev(values)
            means.append(mean);deviations.append(sd)
            expected_scores.append({'task':task,'method':method,'y':i,'mean':mean,'sample_sd':sd})
        assert bar['y']==error['y']==list(range(4))
        np.testing.assert_allclose(bar['values'],means,rtol=0,atol=1e-9)
        np.testing.assert_allclose(error['values'],means,rtol=0,atol=1e-9)
        np.testing.assert_allclose(error['xerr'],deviations,rtol=0,atol=1e-9)
    assert score_coordinates==expected_scores
    return {'bar_series':15,'plotted_mean_differences':40,'paired_interval_whiskers':40,
            'absolute_score_means':20,'sample_sd_whiskers':20,
            'scope':'Captured exact Matplotlib bar and errorbar arguments during archived-source rebuild; every value matches audited contrast.'}


def verify_tables(payload, output):
    def rows(name):
        lines=[line for line in (output/name).read_text().splitlines() if line.startswith('|')]
        return [[value.strip() for value in line.split('|')[1:-1]] for line in lines[2:]]
    def formatted(values):
        return f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}'
    for lane in LANES:
        expected=[]
        for method,label in zip(METHODS,LABELS):
            expected.append([label,*[formatted(per_seed(payload['tasks'][task]['lanes'][lane][method]['scores'],
                payload['tasks'][task]['strata']).tolist()) for task in TASKS]])
        assert rows(lane+'_table.md')==expected
    expected=[]
    for row in payload['comparisons']:
        expected.append([TASK_LABELS[TASKS.index(row['task'])],row['lane'],row['label'],
                         f"{row['difference']:+.3f}",f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}]"])
    assert rows('paired_effects.md')==expected
    runtime=[]
    for task in TASKS:
        if task=='scifact_transfer':continue
        for lane in LANES:
            for method,label in zip(METHODS,LABELS):
                records=payload['tasks'][task]['lanes'][lane][method]['records']
                runtime.append([TASK_LABELS[TASKS.index(task)],lane,label,
                    ', '.join(f"{r['learning_rate']:g}" for r in records),
                    formatted([r['training_seconds'] for r in records]),
                    f"{max(r['peak_memory_bytes'] for r in records)/2**30:.3f}",
                    ', '.join(map(str,sorted({r['trainable_parameters'] for r in records})))])
    assert rows('runtime_memory.md')==runtime
    return {'quality_cells':40,'paired_effect_rows':40,'runtime_rows':32,
            'scope':'Parsed displayed Markdown cells and independently checked labels, order, rounding, scores, intervals, rates, timings, memory and parameter counts.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/2026-10-09-current-dora-nora'))
    args=p.parse_args();output=args.output.resolve()
    manifest=read(output/'raw_manifest.json');payload=read(output/'summary.json')
    work=Path(tempfile.mkdtemp(prefix='nora_current_report_offline_',dir='/var/tmp/dora-bench'))
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
    assert sha(extracted/'source/experiments/check_nora_current_report.py')==sha(Path(__file__))
    assert read(extracted/'report_inputs.json')==payload
    assert read(output/'comparisons.json')==payload['comparisons']
    raw=verify_raw(payload,extracted,manifest)
    wrapper=work/'capture_rebuild.py';wrapper.write_text(CAPTURE)
    env={**os.environ,'PYTHONPATH':str(extracted/'source'),'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1'}
    command=[sys.executable,str(wrapper)]
    result=subprocess.run(command,cwd=extracted,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (work/'rebuild.stdout.log').write_text(result.stdout);(work/'rebuild.stderr.log').write_text(result.stderr)
    assert result.returncode==0,(result.returncode,result.stdout,result.stderr)
    chart=verify_chart(payload,read(extracted/'plot_capture.json'),read(output/'chart_coordinates.json'),
                       read(output/'score_chart_coordinates.json'))
    tables=verify_tables(payload,output)
    checked=[]
    for name in OUTPUTS:
        assert sha(output/name)==sha(extracted/'rebuilt'/name),name
        checked.append({'path':name,'sha256':sha(output/name)})
    shutil.copyfile(Path(__file__),output/'offline_rebuild_source.py')
    audit={'status':'pass','checker_source_sha256':sha(Path(__file__)),
           'raw_manifest_sha256':sha(output/'raw_manifest.json'),'archive_sha256':manifest['archive_sha256'],
           'parts_verified':len(parts),'members_verified':len(manifest['files']),'raw_quality_checks':raw,
           'chart_data_checks':chart,'table_data_checks':tables,'exact_outputs':checked,'rebuild_audit':read(extracted/'rebuilt/rebuild_audit.json'),
           'command':command,'work_directory':str(work),
           'scope':'Fresh archived-source rebuild with every retained file hash verified. Independently rescored predictions for every cell, recomputed all paired intervals and captured chart arguments, then required all18 derived outputs byte-identical. Tensor checkpoint payloads are omitted; no inference replay.'}
    (output/'offline_rebuild_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':'pass','members':len(manifest['files']),'exact_outputs':len(checked),'raw':raw,'chart':chart,'work_directory':str(work)},indent=2))


if __name__=='__main__':main()
