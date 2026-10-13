"""Independently audit the completed two-recipe up-normalization ViGGO matrix and saved predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.extraction import ALLOWED_ACTS, ALLOWED_KEYS


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--allow-missing-checkpoints', action='store_true',
                   help='Portable metadata/metric replay using an existing strict task_audit.json; writes portable_task_audit.json, never replaces the strict attestation')
    args = p.parse_args()
    root = args.root
    prior = read(root / 'task_audit.json') if args.allow_missing_checkpoints else None
    if prior is not None:
        assert prior['status'] == 'pass' and prior['checkpoint_payloads_omitted'] == 0
        for name, expected in prior['input_sha256'].items():
            assert sha(root / name) == expected, name
    data, manifest = read(root / 'results.json'), read(root / 'manifest.json')
    assert data['complete'] and data['manifest'] == manifest
    assert manifest['schema'] == 'dora-nora-up-extraction-v1' and manifest['task'] == 'ViGGO'
    config, recipes = manifest['configuration'], manifest['recipes']
    assert config['seeds'] == [42, 43, 44] and len(recipes) == 2
    assert list(recipes) == config['methods']
    assert config['learning_rates'] == [.0001, .0003] and config['fixed_lr'] == .0003
    assert config['rank'] == 8 and config['pilot_steps'] == 48 and config['steps'] == 256
    assert config['batch_size'] == 8 and config['eval_every'] == 64
    assert len(data['lanes']) == 12
    for name, expected in manifest['source_sha256'].items():
        assert sha(root / 'source' / name) == expected
    for name, expected in manifest['validation_gate_sha256'].items():
        gate = root / 'gates' / (name + '.json')
        assert sha(gate) == expected and read(gate)['status'] == 'pass'
    gate = read(root / 'gates/backend.json')
    assert gate['backend_source_sha256'] == manifest['source_sha256']['experiments/nora_up.py']
    assert gate['recipes'] == recipes
    assert gate['cpu_tests']['status'] == gate['cuda_tests']['status'] == 'pass'
    assert all(recipe['magnitude_fp32'] for recipe in recipes.values())
    assert set(recipes) == {'up_base','up_nora'}
    for method, recipe in recipes.items():
        assert recipe['axis'] == 0 and not recipe['detach_norm'] and not recipe['norm_fp32']
        assert not recipe['factorized'] and not recipe['clamp_norm']
        assert recipe['nora'] == method.endswith('_nora')
        assert not recipe['published_init']
        assert recipe['nora_factor'] == 'up' and recipe['nora_axis'] == 1
        assert recipe['nora_epsilon'] == 1e-12
    baseline_control = read(root / 'baseline_control/control.json')
    assert baseline_control['status'] == 'pass' and baseline_control['manifest_sha256'] == sha(root / 'manifest.json')
    assert {check['method'] for check in baseline_control['checks']} == {'up_base'}
    for check in baseline_control['checks']:
        assert check['checkpoint_tensors_bitwise_identical'] == 216 and check['steps'] == 5
        assert check['training_log_bitwise_identical'] and check['validation_nll_bitwise_identical']
    baseline_verified = baseline_omitted = 0
    for name, expected in baseline_control['evidence_sha256'].items():
        path = root / 'baseline_control' / name
        if path.suffix == '.pt' and not path.exists() and args.allow_missing_checkpoints:
            assert prior['baseline_checkpoint_sha256'][str(path.relative_to(root))] == expected
            baseline_omitted += 1
        else:
            assert sha(path) == expected
            if path.suffix == '.pt': baseline_verified += 1
    split = read(root / 'split_manifest.json')
    assert sha(root / 'split_manifest.json') == manifest['dataset']['split_manifest_sha256']
    assert {name: len(rows) for name, rows in split.items()} == {'train': 1024, 'validation': 128, 'test': 256}
    seen_text = set()
    for rows in split.values():
        assert len({row['id'] for row in rows}) == len(rows)
        texts = {' '.join(row['utterance'].lower().split()) for row in rows}
        assert len(texts) == len(rows)
        assert not texts & seen_text
        seen_text.update(texts)
    expected_gold = {row['id']: row for row in split['test']}
    assert len(expected_gold) == 256
    tuning = read(root / 'tuning.json')
    assert tuning['pilots'] == data['tuning'] and len(tuning['pilots']) == 4
    assert tuning['selected_learning_rates'] == data['selected_learning_rates']
    assert set(data['selected_learning_rates']) == set(recipes)
    history = read(root / 'baseline_history_parity.json')
    assert history['status'] == 'pass' and history['missing'] == [] and len(history['checks']) == 5
    assert history['source_sha256'] == sha(root / 'audit_source/check_nora_up_extraction_history.py')
    history_candidates = set()
    for check in history['checks']:
        assert check['tensors_bitwise_identical'] == 216 and check['scalar_metrics_identical']
        assert check['parameter_order_shapes_dtypes_identical']
        candidate = Path(check['candidate'])
        # Absolute historical paths remain provenance; current artifacts can be relocated.
        relative = Path(candidate.parent.name) / candidate.name
        history_candidates.add(str(relative))
        for name, expected in check['candidate_sha256'].items():
            path = root / relative / name
            if path.suffix == '.pt' and not path.exists() and args.allow_missing_checkpoints:
                assert prior['checkpoint_sha256'][str(path.relative_to(root))] == expected
            else:
                assert sha(path) == expected
    expected_history = {f'pilots/{method}_lr{lr:g}_seed42' for method in ('up_base',) for lr in config['learning_rates']}
    expected_history.update(f'finals/{method}_lr{data["selected_learning_rates"][method]:g}_seed{seed}'
                            for method in ('up_base',) for seed in config['seeds'])
    assert history_candidates == expected_history
    expected_frozen = baseline_control['frozen_sha256']
    observed_fits, checkpoint_tensors, scored_predictions = set(), 0, 0
    checkpoint_hashes, observed_directories = {}, set()
    checkpoint_payloads_verified = checkpoint_payloads_omitted = 0
    nora_diagnostics = []
    expected_modules = [f'model.layers.{index}.self_attn.{projection}'
                        for index in range(36) for projection in ('q_proj', 'v_proj')]

    def check_fit(row, steps, test):
        nonlocal expected_frozen, checkpoint_tensors, scored_predictions, checkpoint_payloads_verified, checkpoint_payloads_omitted
        directory = root / row['artifact_directory']
        assert row['artifact_directory'] == f'{"finals" if test else "pilots"}/{row["method"]}_lr{row["lr"]:g}_seed{row["seed"]}'
        assert row['artifact_directory'] not in observed_directories
        observed_directories.add(row['artifact_directory'])
        result = read(directory / 'result.json')
        assert sha(directory / 'result.json') == row['result_sha256']
        assert row == {**result, 'artifact_directory': row['artifact_directory'],
                       'result_sha256': row['result_sha256']}
        assert row['recipe'] == recipes[row['method']]
        recipe = recipes[row['method']]
        assert row['seed'] in config['seeds'] and row['lr'] in config['learning_rates']
        assert row['steps'] == steps and row['rank'] == 8 and row['frozen_weights_unchanged']
        expected_frozen = expected_frozen or row['frozen_sha256']
        assert row['frozen_sha256'] == expected_frozen
        logs = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
        assert [line['step'] for line in logs] == list(range(1, steps + 1))
        for entry in logs:
            assert math.isfinite(entry['loss']) and math.isfinite(entry['gradient_norm'])
            assert entry['gradient_norm'] >= 0
            assert math.isclose(entry['lr'], row['lr'] * min(1., entry['step'] / max(1, steps * .1)), rel_tol=1e-12)
        expected_validation_steps = [step for step in range(1, steps + 1) if step % config['eval_every'] == 0 or step == steps]
        assert [line['step'] for line in logs if 'validation_nll' in line] == expected_validation_steps
        assert all(math.isfinite(line['validation_nll']) for line in logs if 'validation_nll' in line)
        selected = min((line for line in logs if 'validation_nll' in line), key=lambda line: line['validation_nll'])
        assert row['selected_step'] == selected['step'] and row['validation_nll'] == selected['validation_nll']
        diagnostics = row['norm_diagnostics']
        assert list(diagnostics) == expected_modules
        for diagnostic_name, diagnostic in diagnostics.items():
            assert diagnostic['clamp_activations'] == 0 and math.isfinite(diagnostic['minimum_raw_norm']) and diagnostic['minimum_raw_norm'] > 0
            assert diagnostic['recipe_sha256'] == recipes[row['method']]['recipe_sha256']
            assert diagnostic['normalization_axis'] == recipe['axis']
            assert diagnostic['zero_initialized_factor'] == recipe['zero_initialized_factor']
            assert diagnostic['norm_evaluations'] > steps
            assert diagnostic['factor_dtype'] == 'torch.float32'
            assert diagnostic['magnitude_dtype'] == ('torch.float32' if recipe['magnitude_fp32'] else 'torch.bfloat16')
            assert diagnostic['norm_floor'] is None
            assert diagnostic['nora_enabled'] == recipe['nora']
            assert diagnostic['nora_epsilon'] == 1e-12
            assert diagnostic['nora_factor'] == 'up' and diagnostic['nora_axis'] == 1
            expected_vectors = 2048 if diagnostic_name.endswith('q_proj') else 256
            assert diagnostic['nora_vector_count'] == expected_vectors
            if recipe['nora']:
                assert diagnostic['nora_norm_evaluations'] == diagnostic['norm_evaluations']
                raw_min = diagnostic['nora_minimum_raw_vector_norm']
                assert math.isfinite(raw_min) and raw_min > 0
                expected_initial = 0
                assert diagnostic['nora_initial_zero_vectors'] == expected_initial
                assert diagnostic['nora_initial_epsilon_vectors'] == expected_initial
                assert diagnostic['nora_epsilon_vector_events'] == 0, 'Unexpected epsilon branch; reject up-normalization quality run'
                assert diagnostic['nora_epsilon_vector_events'] >= expected_initial
            else:
                assert diagnostic['nora_minimum_raw_vector_norm'] is None
                assert all(diagnostic[name] == 0 for name in ('nora_norm_evaluations','nora_initial_zero_vectors',
                           'nora_initial_epsilon_vectors','nora_epsilon_vector_events'))
        nora_diagnostics.append({'artifact_directory': row['artifact_directory'], 'method': row['method'],
                                 'first_gradient_norm': logs[0]['gradient_norm'],
                                 'maximum_gradient_norm': max(entry['gradient_norm'] for entry in logs),
                                 'nora_initial_epsilon_vectors_total': sum(d['nora_initial_epsilon_vectors'] for d in diagnostics.values()),
                                 'nora_epsilon_vector_events_total': sum(d['nora_epsilon_vector_events'] for d in diagnostics.values()),
                                 'nora_epsilon_scope': 'Up-row vector-events summed across all 72 adapters and all recorded normalization calls, including constructor and warmup; expected zero for random-up initialization, separate from the DoRA weight-norm floor.'})
        shapes = {}
        for module in expected_modules:
            output = 2048 if module.endswith('q_proj') else 256
            module_shapes = {'lora_A': (8, 2048), 'lora_B': (output, 8),
                             'm': (output, 1) if recipe['axis'] == 1 else (1, 2048)}
            for suffix in recipe['trainable_registration_order']:
                shapes[f'{module}.{suffix}'] = module_shapes[suffix]
        assert row['trainable_parameters'] == sum(math.prod(shape) for shape in shapes.values())
        checkpoint = directory / 'adapter.pt'
        checkpoint_name = str(checkpoint.relative_to(root))
        if checkpoint.exists():
            checkpoint_hashes[checkpoint_name] = sha(checkpoint)
            if prior is not None:
                assert checkpoint_hashes[checkpoint_name] == prior['checkpoint_sha256'][checkpoint_name]
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert list(state) == list(shapes) and len(state) == 216
            for name, value in state.items():
                assert tuple(value.shape) == shapes[name] and torch.isfinite(value).all()
                module, suffix = name.rsplit('.', 1)
                diagnostic = diagnostics[module]
                assert str(value.dtype) == diagnostic['magnitude_dtype' if suffix == 'm' else 'factor_dtype']
                if suffix == diagnostic['zero_initialized_factor']:
                    assert float(value.float().norm()) > 0
            checkpoint_tensors += len(state)
            checkpoint_payloads_verified += 1
        else:
            assert args.allow_missing_checkpoints, checkpoint
            checkpoint_hashes[checkpoint_name] = prior['checkpoint_sha256'][checkpoint_name]
            checkpoint_payloads_omitted += 1
        if not test:
            assert 'test' not in row and not (directory / 'predictions.jsonl').exists()
            return
        assert math.isfinite(row['test_nll'])
        predictions = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == len(expected_gold) and len({line['id'] for line in predictions}) == len(expected_gold)
        assert {line['id'] for line in predictions} == set(expected_gold)
        valid = schema = exact = tp = predicted = truth = 0
        for line in predictions:
            gold = expected_gold[line['id']]['target']
            assert line['gold'] == gold and line['utterance'] == expected_gold[line['id']]['utterance']
            parsed = None
            try:
                parsed = json.loads(line['text'].strip())
                valid += 1
            except (ValueError, TypeError):
                pass
            assert line['parsed'] == parsed
            is_schema = (isinstance(parsed, dict) and set(parsed).issubset(ALLOWED_KEYS)
                         and all(isinstance(v, str) for v in parsed.values())
                         and parsed.get('dialogue_act') in ALLOWED_ACTS)
            schema += int(is_schema)
            pred = parsed if is_schema else {}
            match = pred == gold
            exact += int(match)
            assert line['exact_match'] == match
            tp += len(set(gold.items()) & set(pred.items()))
            predicted += len(pred)
            truth += len(gold)
        metrics = {'count': len(predictions), 'valid_json': valid / len(predictions),
                   'schema_valid': schema / len(predictions), 'exact_match': exact / len(predictions),
                   'slot_micro_f1': 2 * tp / max(1, predicted + truth),
                   'correct_slots': tp, 'predicted_slots': predicted, 'gold_slots': truth}
        for name, value in metrics.items():
            assert row['test'][name] == value, (directory, name)
        scored_predictions += len(predictions)

    for method in recipes:
        pilots = [row for row in tuning['pilots'] if row['method'] == method]
        assert [row['lr'] for row in pilots] == config['learning_rates']
        assert all(row['seed'] == 42 for row in pilots)
        assert min(pilots, key=lambda row: row['validation_nll'])['lr'] == data['selected_learning_rates'][method]
        for row in pilots:
            check_fit(row, 48, False)
    expected_lanes = {(lane, method, seed) for lane in ('fixed_original_lr', 'validation_tuned')
                      for method in recipes for seed in config['seeds']}
    assert {(row['lane'], row['method'], row['seed']) for row in data['lanes']} == expected_lanes
    for row in data['lanes']:
        expected_lr = config['fixed_lr'] if row['lane'] == 'fixed_original_lr' else data['selected_learning_rates'][row['method']]
        assert row['lr'] == expected_lr
        fit = data['fits'][row['fit_key']]
        assert (fit['method'], fit['seed'], fit['lr']) == (row['method'], row['seed'], expected_lr)
        observed_fits.add(row['fit_key'])
    assert observed_fits == set(data['fits'])
    expected_fit_keys = {f'{method}_lr{lr:g}_seed{seed}' for method in recipes for seed in config['seeds']
                         for lr in {config['fixed_lr'], data['selected_learning_rates'][method]}}
    assert set(data['fits']) == expected_fit_keys
    for fit in data['fits'].values():
        check_fit(fit, 256, True)
    actual_directories = {str(path.parent.relative_to(root)) for stage in ('pilots', 'finals')
                          for path in (root / stage).glob('*/result.json')}
    assert actual_directories == observed_directories
    target = root / 'audit_source' / Path(__file__).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(__file__).read_bytes())
    artifact_hashes = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
                       if path.is_file() and path.name not in ('task_audit.json', 'portable_task_audit.json') and path.suffix != '.pt'}
    audit = {'status': 'pass', 'task': 'ViGGO original-initializer up-factor normalization', 'lanes': len(data['lanes']),
             'unique_final_fits': len(data['fits']), 'pilots': len(tuning['pilots']),
             'checkpoint_tensors_checked': checkpoint_tensors, 'predictions_rescored': scored_predictions,
             'checkpoint_payloads_verified': checkpoint_payloads_verified, 'checkpoint_payloads_omitted': checkpoint_payloads_omitted,
             'baseline_checkpoint_payloads_verified': baseline_verified, 'baseline_checkpoint_payloads_omitted': baseline_omitted,
             'checkpoint_sha256': checkpoint_hashes,
             'baseline_checkpoint_sha256': {str(Path('baseline_control')/name): digest for name,digest in baseline_control['evidence_sha256'].items() if name.endswith('.pt')},
             'nora_diagnostics': nora_diagnostics,
             'prior_strict_audit_sha256': sha(root / 'task_audit.json') if args.allow_missing_checkpoints else None,
             'frozen_sha256': expected_frozen, 'input_sha256': artifact_hashes,
             'scope': 'Sources, data IDs/gold, recipe/precision and clamp diagnostics, budgets, '
                      'validation selection, checkpoint finiteness and updates, complete fixed/tuned matrix; '
                      'all generation metrics recomputed from prediction text. Test NLL is not reinferred. '
                      'Optimizer state payloads were not saved. Portable omission mode retains checkpoint hashes '
                      'from the required prior strict audit; omitted payloads are not revalidated.'}
    (root / ('portable_task_audit.json' if args.allow_missing_checkpoints else 'task_audit.json')).write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('input_sha256', 'checkpoint_sha256','baseline_checkpoint_sha256','nora_diagnostics')}))


if __name__ == '__main__':
    main()
