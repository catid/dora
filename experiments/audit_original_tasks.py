"""Audit the original-adapter reruns against the recorded task protocols.

Checks saved evidence without rerunning downstream models. Teacher checkpoints
are small enough to reconstruct independently on CPU. Paths in the output are
historical provenance; the report bundles the compact evidence separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def same_configuration(new, old, ignored):
    for key in set(new) | set(old):
        if key not in ignored:
            assert new.get(key) == old.get(key), (key, new.get(key), old.get(key))


def sources(root, manifest, directory):
    checks = {}
    for name, expected in manifest['source_sha256'].items():
        path = root / directory / name
        actual = sha(path)
        assert actual == expected, path
        checks[name] = actual
    return checks


def training_log(path, steps, selection=None):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1, steps + 1)), path
    assert all(math.isfinite(r['loss']) for r in rows), path
    if selection is not None:
        candidates = [r for r in rows if 'validation_nll' in r]
        winner = min(candidates, key=lambda r: r['validation_nll'])
        assert winner['step'] == selection['selected_step'], path
        assert winner['validation_nll'] == selection['validation_nll'], path
    return sha(path)


def language(root, reference, task):
    current, old = read(root / 'results.json'), read(reference / 'results.json')
    new_manifest, old_manifest = current['manifest'], old['manifest']
    for key in ('model', 'model_revision', 'dataset'):
        assert new_manifest[key] == old_manifest[key], key
    if task == 'extraction':
        assert new_manifest['system_prompt_sha256'] == old_manifest['system_prompt_sha256']
    same_configuration(new_manifest['configuration'], old_manifest['configuration'],
                       {'output', 'methods', 'baseline_from'})
    assert sha(root / 'split_manifest.json') == sha(reference / 'split_manifest.json')
    source_checks = sources(root, new_manifest, 'source')
    rows = current['results']
    assert {(r['method'], r['seed']) for r in rows} == {('dora_original', s) for s in (42, 43, 44)}
    trials = current['tuning']
    expected_rates = new_manifest['configuration']['learning_rates']
    assert len(trials) == len(expected_rates)
    assert {r['lr'] for r in trials} == set(expected_rates)
    assert all(r['seed'] == 42 and r['method'] == 'dora_original' for r in trials)
    winner = min(trials, key=lambda r: r['validation_nll'])
    assert all(r['lr'] == winner['lr'] for r in rows)
    expected_frozen = {r['frozen_sha256'] for r in old['results']}
    checks = []
    for i, result in enumerate(trials + rows):
        is_trial = i < len(trials)
        if is_trial:
            directory = (Path(result['directory']) if task == 'cogs'
                         else root / f"pilot_dora_original_{result['lr']:g}")
        else:
            directory = root / f"dora_original_seed{result['seed']}"
        checkpoint = directory / 'adapter.pt'
        checkpoint_hash = sha(checkpoint)
        if 'checkpoint_sha256' in result:
            assert checkpoint_hash == result['checkpoint_sha256']
        assert result['frozen_weights_unchanged'] and result['frozen_sha256'] in expected_frozen
        steps = new_manifest['configuration']['pilot_steps'] if task == 'extraction' and is_trial else new_manifest['configuration']['steps']
        assert result['steps'] == steps
        log_dir = Path(result['reused_winning_full_budget_trial']) if 'reused_winning_full_budget_trial' in result else directory
        log_hash = training_log(log_dir / 'training.jsonl', steps, result)
        checks.append({'directory': str(directory), 'checkpoint_sha256': checkpoint_hash,
                       'training_log_sha256': log_hash, 'steps_verified': steps})
    if task == 'cogs':
        assert current['baseline'] == old['baseline']
    else:
        assert sha(root / 'baseline_predictions.jsonl') == sha(reference / 'baseline_predictions.jsonl')
        assert current['baseline']['test_nll'] == old['baseline']['test_nll']
    return {'task': task, 'matched_model_revision': new_manifest['model_revision'],
            'split_manifest_sha256': sha(root / 'split_manifest.json'),
            'dataset_and_training_configuration_match': True, 'frozen_base_hashes_match': True,
            'baseline_match': True, 'validation_only_selection_verified': True,
            'source_checks': source_checks, 'runs': checks,
            'scope': 'Verifies protocol, logged budgets/selection, checkpoint/source hashes and frozen-base attestations. Prediction metrics are independently recomputed by the report.'}


def teacher(root, reference):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4)
    data, old = read(root / 'results.json'), read(reference / 'results.json')
    same_configuration(data['provenance']['config'], old['provenance']['config'], {'output', 'methods'})
    source_checks = sources(root, data['provenance'], 'executed_sources')
    rows = data['runs']
    cell_ids = {r['cell_id'] for r in old['runs']}
    assert len(cell_ids) == 28 and len(rows) == 84
    assert {(r['cell_id'], r['seed'], r['method']) for r in rows} == {
        (cell, seed, 'dora_original') for cell in cell_ids for seed in (42, 43, 44)}
    checks = []
    for cell in sorted(cell_ids):
        problem = torch.load(root / cell / 'problem.pt', weights_only=True, map_location='cpu')
        prior = torch.load(reference / cell / 'problem.pt', weights_only=True, map_location='cpu')
        assert problem.keys() == prior.keys()
        assert all(torch.equal(problem[k], prior[k]) for k in problem), cell
        selection = read(root / cell / 'dora_original' / 'selection.json')
        assert len(selection['trials']) == 4
        assert {r['learning_rate'] for r in selection['trials']} == {.001, .003, .01, .03}
        winner = min(selection['trials'], key=lambda r: r['validation']['relative_to_frozen'])
        assert selection['selected_lr'] == winner['learning_rate']
        for result in (r for r in rows if r['cell_id'] == cell):
            checkpoint = root / cell / 'dora_original' / f"seed{result['seed']}.pt"
            assert sha(checkpoint) == result['checkpoint']['sha256']
            state = torch.load(checkpoint, weights_only=True, map_location='cpu')
            assert result['learning_rate'] == selection['selected_lr']
            assert result['frozen_weight_unchanged']
            adapted = problem['base_weight'] + state['lora_A'] @ state['lora_B']
            effective = state['m'] * (adapted / adapted.norm(dim=0, keepdim=True))
            loss = F.mse_loss(F.linear(problem['x_test'], effective), problem['y_test']).item()
            assert math.isclose(loss, result['test']['mse'], rel_tol=2e-4, abs_tol=1e-8), (cell, loss, result['test'])
            checks.append({'cell': cell, 'seed': result['seed'], 'cpu_mse': loss,
                           'recorded_gpu_mse': result['test']['mse'], 'checkpoint_sha256': sha(checkpoint)})
    return {'task': 'teacher', 'same_problem_tensors_all_cells': True, 'cell_count': 28,
            'source_checks': source_checks, 'cpu_checkpoint_checks': checks,
            'scope': 'Exact problem tensors matched to prior run; all 84 original checkpoints evaluated independently with the dense column-normalized formula on CPU.'}


def sdxl(root, reference):
    current, old = read(root / 'results.json'), read(reference / 'results.json')
    manifest, prior = current['provenance'], old['provenance']
    for key in ('model', 'model_revision', 'weight_variant', 'dataset', 'dataset_revision', 'clip', 'dino'):
        assert manifest[key] == prior[key], key
    for key in ('resolution', 'steps', 'trial_steps', 'inference_steps', 'rank', 'no_checkpointing'):
        assert manifest['config'][key] == prior['config'][key], key
    for key in ('seed', 'rank', 'targets', 'train_indices', 'validation_index', 'test_index',
                'lr_candidates', 'optimizer', 'weight_decay', 'grad_clip', 'lr_schedule'):
        assert manifest['controls'][key] == prior['controls'][key], key
    new_photos, old_photos = read(root / 'dataset_manifest.json'), read(reference / 'dataset_manifest.json')
    for a, b in zip(new_photos['records'], old_photos['records'], strict=True):
        for key in ('index', 'split', 'sha256', 'time_ids'):
            assert a[key] == b[key], key
    cache = Path(manifest['conditioning_cache'])
    cache_metadata = read(cache.with_suffix('.json'))
    assert sha(cache) == cache_metadata['sha256']
    source_checks = sources(root, manifest, 'executed_sources')
    assert {r['method'] for r in current['results']} == {'baseline', 'dora_original'}
    result = next(r for r in current['results'] if r['method'] == 'dora_original')
    baseline = next(r for r in current['results'] if r['method'] == 'baseline')
    old_baseline = next(r for r in old['results'] if r['method'] == 'baseline')
    baseline_errors = []
    for split in ('validation', 'test'):
        for a, b in zip(baseline[split]['rows'], old_baseline[split]['rows'], strict=True):
            assert (a['timestep'], a['noise_seed']) == (b['timestep'], b['noise_seed'])
            baseline_errors.append(abs(a['mse'] - b['mse']))
            assert math.isclose(a['mse'], b['mse'], rel_tol=1e-6, abs_tol=1e-8)
    assert result['frozen_sha256_before'] == result['frozen_sha256_after']
    assert result['frozen_sha256_before'] in {r.get('frozen_sha256_before') for r in old['results']}
    assert sha(root / 'dora_original' / 'adapter.pt') == result['checkpoint']['sha256']
    log_hashes = {'dora_original/training.jsonl': training_log(root / 'dora_original/training.jsonl', 500)}
    if manifest['controls']['seed'] == 42:
        assert len(result['trials']) == 2
        assert {r['lr'] for r in result['trials']} == {1e-5, 1e-4}
        winner = min(result['trials'], key=lambda r: r['validation']['mean_mse'])
        assert result['selected_lr'] == winner['lr']
        for row in result['trials']:
            relative = f"dora_original/trial_lr_{row['lr']:g}/training.jsonl"
            log_hashes[relative] = training_log(root / relative, 50)
    else:
        selection = read(Path(manifest['lr_selection']['source']))
        assert selection['selection_seed'] == 42
        assert selection['selected_lrs']['dora_original'] == result['selected_lr']
        assert sha(Path(manifest['lr_selection']['source'])) == manifest['lr_selection']['sha256']
    for row in current['results']:
        for sample in row['samples']:
            assert sha(Path(sample['path'])) == sample['sha256']
    return {'task': 'sdxl', 'seed': manifest['controls']['seed'], 'same_model_data_budget': True,
            'frozen_base_hashes_match': True, 'baseline_probe_max_abs_error': max(baseline_errors),
            'conditioning_cache_sha256': sha(cache), 'validation_only_selection_verified': True,
            'source_checks': source_checks, 'training_logs': log_hashes,
            'checkpoint_sha256': result['checkpoint']['sha256'],
            'scope': 'Checks original protocol, shared latent cache, baseline probes, budgets, validation-only selection, frozen base, source/checkpoint/image hashes. Image-quality superiority is not established by denoising MSE.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['extraction', 'cogs', 'teacher', 'sdxl'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    result = (sdxl(args.root, args.reference) if args.task == 'sdxl'
              else teacher(args.root, args.reference) if args.task == 'teacher'
              else language(args.root, args.reference, args.task))
    source = args.root / 'audit_source' / Path(__file__).name
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(Path(__file__).read_bytes())
    result.update(passed=True, status='pass', audit_script_sha256=sha(Path(__file__)),
                  input_results_sha256=sha(args.root / 'results.json'), reference=str(args.reference),
                  input_sha256={str(path.relative_to(args.root)): sha(path)
                                for path in sorted(args.root.rglob('*'))
                                if path.is_file() and path.suffix in ('.json', '.jsonl', '.py', '.txt')
                                and path.name != 'task_audit.json'})
    (args.root / 'task_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'task': args.task, 'output': str(args.root / 'task_audit.json')}))


if __name__ == '__main__':
    main()
