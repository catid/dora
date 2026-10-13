"""Collect recorded fit costs from completed, hash-bound initialization reports.

This is reporting only: it never loads a model, changes a fit, or estimates a
missing duration. Repeated report views share physical fits; retrieval pilot
and final snapshots share training but retain their separate evaluations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SCHEMAS = {
    'dora-init-report-v1', 'dora-init-extension-report-v1',
    'dora-init-confirm-report-v1', 'dora-init-cost-engineering-input-v1',
}
RETRIEVAL = {'scifact', 'nfcorpus'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Evidence:
    """Verify every report evidence file, including files not used for costs."""
    def __init__(self):
        self.verified = {}

    def verify(self, path, expected, size=None):
        path = Path(path).resolve()
        require(path.is_file(), f'Missing evidence: {path}')
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if str(path) not in self.verified or self.verified[str(path)]['stamp'] != stamp:
            actual = sha(path)
            after = path.stat()
            require(stamp == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    f'Evidence changed while hashing: {path}')
            self.verified[str(path)] = {'sha256': actual, 'bytes': stat.st_size, 'stamp': stamp}
        entry = self.verified[str(path)]
        require(entry['sha256'] == expected, f'Hash mismatch: {path}')
        require(size is None or size == stat.st_size, f'Size mismatch: {path}')
        return path

    def read(self, path, expected):
        path = self.verify(path, expected)
        # Hash the bytes actually parsed, not just an earlier stat/hash result.
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected, f'Hash mismatch: {path}')
        return json.loads(content)


def load_report(path, evidence):
    path = Path(path).resolve()
    manifest_path = path.with_name('report_manifest.json')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    outputs = manifest.get('outputs', {})
    require(path.name in outputs, f'Report manifest does not bind {path.name}')
    for name, expected in outputs.items():
        target = (path.parent / name).resolve()
        require(target.is_relative_to(path.parent), 'Report output escapes report directory')
        evidence.verify(target, expected)
    payload = evidence.read(path, outputs[path.name])
    require(payload.get('schema') in SCHEMAS, f'Unsupported report schema: {payload.get("schema")}')
    if payload['schema'] == 'dora-init-cost-engineering-input-v1':
        require(payload.get('fixture') is True, 'Engineering cost input must be marked fixture')
    bindings = {}
    for name, value in payload['evidence'].items():
        target = Path(name)
        require(target.is_absolute(), f'Report evidence path must be absolute: {name}')
        expected = value['sha256'] if isinstance(value, dict) else value
        target = evidence.verify(target, expected, value.get('bytes') if isinstance(value, dict) else None)
        if str(target) in bindings:
            require(bindings[str(target)] == expected, f'Conflicting evidence aliases: {target}')
        bindings[str(target)] = expected
    return payload, bindings, {
        'path': str(path), 'sha256': outputs[path.name],
        'report_manifest_path': str(manifest_path),
        'report_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'schema': payload['schema'], 'fixture': bool(payload.get('fixture') or payload.get('engineering_only')),
        'verified_evidence_files': len(bindings),
    }


def references(payload, report_path):
    """Use normalized fit paths and explicit pilot paths, never guess candidates."""
    roots = {}
    for view, data in payload['tasks'].items():
        task = data.get('training_task', data.get('task'))
        require(task in RETRIEVAL | {'aircraft', 'extraction'}, f'Unknown training task: {task}')
        root = Path(data['root']).resolve()
        lane = data.get('study_lane', payload.get('study_lane', 'confirmation'))
        roots.setdefault((lane, task), set()).add(root)
        for key, fit in data.get('fits', {}).items():
            base = Path(fit['fit_id'])
            base = (base if base.is_absolute() else root / base).resolve()
            filename = 'receipt.json' if task == 'extraction' else 'fit.json' if task == 'aircraft' else 'result.json'
            selected = [name for name, cell in data.get('cells', {}).items()
                        if key in cell.get('fit_keys', []) or key in cell.get('paired_control_keys', [])]
            yield {'root': root, 'task': task, 'source': base / filename, 'stage': 'final',
                   'normalized': fit, 'view': {'report': report_path, 'task_view': view,
                   'study_lane': lane, 'dataset': data.get('dataset'), 'fit_key': key,
                   'selected_cells': selected, 'stage': 'final'}}
    for candidate in payload.get('all_candidates', []):
        task, lane = candidate['task'], candidate['study_lane']
        choices = roots.get((lane, task), set())
        source = Path(candidate['pilot_result_path']).resolve()
        matches = [root for root in choices if source.is_relative_to(root)]
        require(len(matches) == 1, f'Ambiguous pilot root: {source}')
        yield {'root': matches[0], 'task': task, 'source': source, 'stage': 'pilot',
               'normalized': {'candidate_id': candidate['candidate_id'], 'seed': 42,
                              'status': candidate['pilot']['status']},
               'pilot_sha256': candidate['pilot'].get('pilot_sha256'),
               'view': {'report': report_path, 'task_view': f'{lane}/{task}',
               'study_lane': lane, 'dataset': None, 'family_id': candidate.get('family_id'),
               'selected_cells': candidate.get('selected_lanes', []), 'stage': 'pilot'}}


def number(value, field):
    if value is None:
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, f'Invalid duration at {field}: {value}')
    return value


def timing_fields(value, prefix=''):
    """Retain nested timing observations, including nonadditive cached metadata."""
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f'{prefix}.{key}' if prefix else key
            if (key == 'seconds' or key.endswith('_seconds')) and (child is None or isinstance(child, (int, float))):
                rows.append({'field': path, 'seconds': number(child, path)})
            elif isinstance(child, (dict, list)):
                rows.extend(timing_fields(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(timing_fields(child, f'{prefix}[{index}]'))
    return rows


def read_snapshot(ref, bindings, evidence):
    source = ref['source'].resolve()
    require(str(source) in bindings, f'Raw fit is not bound by report evidence: {source}')
    expected = bindings[str(source)]
    require(ref.get('pilot_sha256') in (None, expected), f'Pilot receipt binding differs: {source}')
    raw = evidence.read(source, expected)
    for key in ('candidate_id', 'seed', 'status'):
        require(raw.get(key) == ref['normalized'].get(key), f'Normalized/raw {key} mismatch: {source}')
    task, root = ref['task'], ref['root']
    result, initial = raw.get('result', raw), raw.get('initialization', {})
    linked = {}
    if task == 'extraction' and raw.get('status') == 'pass':
        directory = (root / raw['artifact_directory']).resolve()
        require(directory.is_relative_to(root), f'Fit directory escapes task root: {directory}')
        for filename, hash_key in [('result.json', 'result_sha256'), ('initialization.json', 'initialization_sha256')]:
            linked_path = directory / filename
            value = evidence.read(linked_path, raw[hash_key])
            if str(linked_path) in bindings:
                require(bindings[str(linked_path)] == raw[hash_key], f'Receipt/evidence binding differs: {linked_path}')
            linked[filename] = {'path': str(linked_path), 'sha256': raw[hash_key]}
            if filename == 'result.json': result = value
            else: initial = value
    else:
        directory = (root / raw.get('artifact_directory', raw.get('fit_id', str(source.parent)))).resolve()
    require(directory.is_relative_to(root), f'Fit directory escapes task root: {directory}')
    if task in RETRIEVAL:
        train = raw.get('training', {}).get('seconds')
        train_scope = 'Training loop only; evaluation is separate.'
    elif task == 'aircraft':
        train = result.get('training_wall_seconds')
        train_scope = 'Epoch loop including validation and checkpoint selection/writes; no separate validation timer.'
    else:
        train = result.get('train_and_validation_seconds')
        train_scope = 'Training and periodic validation/checkpoint selection; excludes initial warmup and test NLL.'
    measurements = [{**value, 'source': str(source), 'source_sha256': expected}
                    for value in timing_fields(raw)]
    if linked:
        for name, value in [('result.json', result), ('initialization.json', initial)]:
            measurements.extend({**row, 'source': linked[name]['path'],
                                 'source_sha256': linked[name]['sha256']}
                                for row in timing_fields(value))
    snapshot = {'path': str(source), 'sha256': expected, 'stage': ref['stage'], 'status': raw['status'],
                'linked_sources': linked, 'started_utc': raw.get('started_utc'),
                'finished_utc': raw.get('finished_utc'), 'raw_cost_fields': measurements}
    return str(directory), raw, result, initial, snapshot, number(train, 'training_seconds'), train_scope


def retrieval_evaluation(value):
    datasets = {}
    for dataset, row in value.items():
        if not isinstance(row, dict):
            continue
        datasets[dataset] = {
            'corpus_encoding_seconds': number(row.get('corpus_encoding_seconds'), f'{dataset}.corpus_encoding_seconds'),
            'reported_total_seconds': number(row.get('total_seconds'), f'{dataset}.total_seconds'),
            'query_encoding_and_ranking_seconds': {
                split: number(item.get('query_encoding_and_ranking_seconds'), f'{dataset}.{split}')
                for split, item in row.items() if isinstance(item, dict)
                and split in ('validation', 'test')},
        }
    return datasets


def collect(paths):
    evidence = Evidence()
    fits, inputs, seen_reports = {}, [], set()
    for path in paths:
        path = str(Path(path).resolve())
        if path in seen_reports:
            continue
        seen_reports.add(path)
        payload, bindings, provenance = load_report(path, evidence)
        inputs.append(provenance)
        for ref in references(payload, path):
            physical, raw, result, initial, snapshot, train, train_scope = read_snapshot(ref, bindings, evidence)
            identity = (ref['task'], physical)
            fit = fits.setdefault(identity, {
                'physical_fit_directory': physical, 'training_task': ref['task'],
                'candidate_id': raw['candidate_id'], 'seed': raw['seed'], 'views': [], 'snapshots': {},
                'training_seconds': None, 'initializer_seconds': None, 'training_scope': train_scope,
                'initializer_scope': 'Recorded injection timer only; cache history and repeated reload/audit initialization are not added.',
                'evaluation_invocations': {}, 'status': raw['status'],
            })
            require((fit['candidate_id'], fit['seed']) == (raw['candidate_id'], raw['seed']),
                    f'Conflicting physical fit identity: {physical}')
            if ref['view'] not in fit['views']:
                fit['views'].append(ref['view'])
            key = snapshot['path']
            if key in fit['snapshots']:
                require(fit['snapshots'][key]['sha256'] == snapshot['sha256'], f'Conflicting fit snapshot: {key}')
                continue
            fit['snapshots'][key] = snapshot
            if ref['stage'] == 'final':
                fit['status'] = raw['status']
            for field, value in [('training_seconds', train),
                                 ('initializer_seconds', number(initial.get('initializer_seconds'), 'initializer_seconds'))]:
                if value is not None:
                    require(fit[field] in (None, value), f'Conflicting repeated {field}: {physical}')
                    fit[field] = value
            if ref['task'] in RETRIEVAL:
                for field, stage in [('initial_evaluation', 'initial'), ('evaluation', ref['stage'])]:
                    if not raw.get(field):
                        continue
                    # Initial evaluations are copied from pilot into final. Other
                    # evaluations are distinct when the final snapshot is extended.
                    token = 'initial' if stage == 'initial' else digest({
                        'evaluation': raw[field], 'finished_utc': raw.get('finished_utc')})
                    evaluation = {'kind': 'retrieval', 'stage': stage,
                                  'datasets': retrieval_evaluation(raw[field]),
                                  'scope': 'One evaluation invocation per dataset. Parent total includes corpus and split children; do not add them together.',
                                  'sources': []}
                    saved = fit['evaluation_invocations'].setdefault(token, evaluation)
                    require(saved['datasets'] == evaluation['datasets'], f'Conflicting evaluation copy: {physical}')
                    saved['sources'].append({'path': key, 'sha256': snapshot['sha256'], 'field': field})
            else:
                field = 'test_seconds' if ref['task'] == 'aircraft' else 'generation_seconds'
                value = result.get(field) if field == 'test_seconds' else (result.get('test') or {}).get(field)
                token = field
                evaluation = {'kind': ref['task'], 'stage': ref['stage'], 'field': field,
                              'seconds': number(value, field), 'sources': [{'path': key, 'sha256': snapshot['sha256']}],
                              'scope': 'Held-out test evaluation timer.' if field == 'test_seconds' else
                              'Test generation, parsing and prediction writing; excludes separate test NLL inference.'}
                prior = fit['evaluation_invocations'].setdefault(token, evaluation)
                require(prior['seconds'] == evaluation['seconds'], f'Conflicting evaluation timing: {physical}')
    output = []
    for fit in fits.values():
        fit['snapshots'] = sorted(fit['snapshots'].values(), key=lambda row: (row['stage'], row['path']))
        fit['evaluation_invocations'] = list(fit['evaluation_invocations'].values())
        fit['missing_cost_fields'] = [key for key in ('training_seconds', 'initializer_seconds') if fit[key] is None]
        fit['stages'] = sorted({view['stage'] for view in fit['views']})
        fit['seed42_pilot_training_reused'] = (fit['training_task'] in RETRIEVAL and fit['seed'] == 42
                                              and fit['stages'] == ['final', 'pilot']
                                              and fit['training_seconds'] is not None)
        fit['physical_fit_id'] = digest([fit['training_task'], fit['physical_fit_directory']])
        output.append(fit)
    return {'schema': 'dora-init-cost-report-v1', 'inputs': inputs,
            'fixture': any(row['fixture'] for row in inputs), 'fits': sorted(output, key=lambda row: row['physical_fit_directory']),
            'verified_evidence_files': {name: {key: row[key] for key in ('sha256', 'bytes')}
                                        for name, row in evidence.verified.items()},
            'scope': 'Recorded physical fit costs only. No total wall time or inferred missing durations. Excludes controls, audits, queue gaps, and uncaptured reload/calibration work.'}


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def render(payload, output):
    output = Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), f'Use an empty cost output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'costs.json', payload)
    write_csv(output / 'fits.csv', [{key: value for key, value in row.items() if key != 'snapshots'} for row in payload['fits']])
    raw = []
    for fit in payload['fits']:
        for snapshot in fit['snapshots']:
            for value in snapshot['raw_cost_fields']:
                raw.append({'physical_fit_id': fit['physical_fit_id'], 'training_task': fit['training_task'],
                            'candidate_id': fit['candidate_id'], 'seed': fit['seed'], 'stage': snapshot['stage'],
                            'source': snapshot['path'], 'source_sha256': snapshot['sha256'], **value,
                            'aggregation': 'Observation only; nested/cumulative/cache fields may overlap. Do not sum raw rows.'})
    write_csv(output / 'raw_costs.csv', raw)
    (output / 'README.md').write_text(
        '# Recorded fit costs' + (' — engineering fixture' if payload['fixture'] else '') + '\n\n'
        f"Collected {len(payload['fits'])} physical fits from {len(payload['inputs'])} hash-verified reports. "
        'No model inference or metric changes were performed.\n\n'
        '`costs.json` retains every source snapshot, raw timing field and report view; `fits.csv` has one row per physical fit. '
        '`raw_costs.csv` contains the literal timed observations with field paths and hashes. Raw rows are not additive.\n\n'
        'Physical paths deduplicate fixed/tuned views, transfer datasets, repeated input reports, and retrieval seed-42 pilot/final training. '
        'Distinct pilot and final retrieval evaluations remain separate, including each corpus encoding. The final result overwrites '
        'the pilot evaluation, so both hashed snapshots are necessary. Initial evaluations copied between snapshots count once.\n\n'
        'Retrieval evaluation totals include corpus encoding plus query/ranking split times. Never add those totals to their children '
        'or count the NFCorpus-trained SciFact transfer view again. Aircraft training includes epoch validation; Qwen training includes '
        'periodic validation. Aircraft test timing and Qwen generation timing are separate. Qwen test-NLL time is unavailable.\n\n'
        'Initialization uses only the recorded injection timer. Nested preprocessing/decomposition/prefit timings can describe historical '
        'cache creation, appear under many modules, or overlap that timer. They are retained as observations, never added as new fit cost. '
        'Missing values stay null/blank; a recorded zero stays zero. No total is fabricated, and timestamp intervals are not used because '
        'they can span scheduling gaps. Controls, audits, shared calibration, reloads, failed partial work without timers and device warmup '
        'are outside this accounting. Failed fits are retained with their available measurements.\n\n'
        'Every input report output and evidence file was verified against its recorded SHA256. Receipt-linked result/initialization '
        'files are additionally checked against the receipt hashes. This verifies provenance, not the completeness of historical timers.\n')
    write(output / 'cost_manifest.json', {'schema': 'dora-init-cost-artifacts-v1', 'source_sha256': sha(__file__),
          'outputs': {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True,
                        help='Completed report_inputs.json files; sibling report_manifest.json files are required')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = collect(args.inputs)
    render(payload, args.output)
    print(json.dumps({'status': 'pass', 'physical_fits': len(payload['fits']),
                      'reports': len(payload['inputs']), 'fixture': payload['fixture'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
