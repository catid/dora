"""Recompute COGS scores from saved text and audit the complete six-method grid."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

METHODS = ('lora', 'dora', 'nora', 'dora_nora', 'dora_nora_mlr', 'dora_nora_gain')
STRUCTURAL = {'cp_recursion', 'pp_recursion', 'obj_pp_to_subj_pp'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def parse(text):
    """Independent small parser: words cannot silently absorb prose whitespace."""
    matches = list(re.finditer(r'\*|[A-Za-z][A-Za-z_]*|\d+|[().,;_]', text))
    end = 0
    tokens = []
    for match in matches:
        if text[end:match.start()].strip():
            return None
        tokens.append(match.group())
        end = match.end()
    if text[end:].strip() or not tokens:
        return None
    i, result = 0, []
    try:
        while i < len(tokens):
            atom = ''
            if tokens[i] == '*':
                atom, i = '*', i + 1
            if not re.fullmatch(r'[A-Za-z][A-Za-z_]*', tokens[i]) or tokens[i] == 'AND':
                return None
            atom += tokens[i]; i += 1
            while tokens[i] == '.':
                i += 1
                if not re.fullmatch(r'[A-Za-z][A-Za-z_]*', tokens[i]):
                    return None
                atom += '.' + tokens[i]; i += 1
            if tokens[i] != '(':
                return None
            atom += '('; i += 1
            while True:
                if tokens[i] in ('x', 'x_'):
                    # The tokenizer accepts both "x _ 1" and "x_1" forms.
                    if tokens[i] == 'x':
                        i += 1
                        if tokens[i] != '_':
                            return None
                    i += 1
                    if not tokens[i].isdigit():
                        return None
                    atom += 'x_' + tokens[i]; i += 1
                elif re.fullmatch(r'[A-Z][A-Za-z_]*', tokens[i]):
                    atom += tokens[i]; i += 1
                else:
                    return None
                if tokens[i] == ',':
                    atom += ','; i += 1
                    continue
                if tokens[i] != ')':
                    return None
                atom += ')'; i += 1
                break
            result.append(atom)
            if i < len(tokens):
                if tokens[i] not in ('AND', ';'):
                    return None
                i += 1
                if i == len(tokens):
                    return None
    except IndexError:
        return None
    return frozenset(result) if len(set(result)) == len(result) else None


def check_predictions(path, gold, recorded):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(gold) == recorded['count']
    assert {r['id'] for r in rows} == set(gold)
    assert digest(path) == recorded['predictions_sha256']
    for row in rows:
        assert row['gold'] == gold[row['id']]['target']
        assert row['utterance'] == gold[row['id']]['utterance']
        assert row['category'] == gold[row['id']]['category']
        actual, expected = parse(row['text']), parse(row['gold'])
        assert expected is not None
        assert row['valid'] == (actual is not None)
        assert row['atom_exact'] == (actual == expected)
        assert row['strict_exact'] == (re.sub(r'\s+', '', row['text']) == re.sub(r'\s+', '', row['gold']))
        guessed = actual or frozenset()
        assert row['correct_atoms'] == len(guessed & expected)
        assert row['predicted_atoms'] == len(guessed)
        assert row['gold_atoms'] == len(expected)
    categories = {name: statistics.mean(r['atom_exact'] for r in rows if r['category'] == name)
                  for name in sorted({r['category'] for r in rows})}
    assert set(recorded['categories']) == set(categories)
    for name, exact in categories.items():
        selected = [row for row in rows if row['category'] == name]
        assert recorded['categories'][name] == {
            'count': len(selected), 'atom_exact': exact,
            'strict_exact': statistics.mean(row['strict_exact'] for row in selected)}
    assert recorded['generation_cap_hits'] == sum(row['hit_generation_cap'] for row in rows)
    checks = {'atom_exact': statistics.mean(r['atom_exact'] for r in rows),
              'strict_exact': statistics.mean(r['strict_exact'] for r in rows),
              'valid': statistics.mean(r['valid'] for r in rows),
              'category_macro_exact': statistics.mean(categories.values()),
              'atom_micro_f1': 2 * sum(r['correct_atoms'] for r in rows) / max(1, sum(r['predicted_atoms'] + r['gold_atoms'] for r in rows))}
    for name, value in checks.items():
        assert math.isclose(value, recorded[name], abs_tol=1e-12), (path, name, value, recorded[name])
    for group, condition in [('structural_macro_exact', lambda x: x in STRUCTURAL),
                             ('lexical_macro_exact', lambda x: x not in STRUCTURAL)]:
        values = [value for name, value in categories.items() if condition(name)]
        expected = statistics.mean(values) if values else None
        assert expected == recorded[group]
    return {row['id']: float(row['atom_exact']) for row in rows}


def paired(delta, seed):
    # Equal category sample sizes make this a category-stratified macro estimate.
    rng = np.random.default_rng(seed)
    seeds, categories, examples = delta.shape
    scores = []
    for _ in range(10000):
        chosen_seeds = rng.integers(seeds, size=seeds)
        chosen_examples = rng.integers(examples, size=(categories, examples))
        scores.append(float(delta[chosen_seeds[:, None, None], np.arange(categories)[None, :, None], chosen_examples[None, :, :]].mean()))
    return {'difference': float(delta.mean()), 'ci95': np.quantile(scores, [.025, .975]).tolist(),
            'replicates': 10000, 'scope': 'Matched seeds and shared examples, stratified by category; exploratory percentile interval.'}


def check_checkpoint(directory, recorded, require_checkpoints):
    checkpoint = directory / 'adapter.pt'
    if checkpoint.exists():
        assert digest(checkpoint) == recorded['checkpoint_sha256']
        return 1
    assert not require_checkpoints, checkpoint
    return 0


def check_training(directory, recorded, configuration, require_checkpoints=True):
    """Verify full optimization budgets and validation-only checkpoint selection."""
    rows = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
    steps = configuration['steps']
    assert recorded['steps'] == steps
    assert [row['step'] for row in rows] == list(range(1, steps + 1))
    assert all(math.isfinite(row[key]) for row in rows for key in ('loss', 'gradient_norm', 'lr'))
    validation = [row for row in rows if 'validation_nll' in row]
    expected_steps = [step for step in range(1, steps + 1)
                      if step % configuration['eval_every'] == 0 or step == steps]
    assert [row['step'] for row in validation] == expected_steps
    assert all(math.isfinite(row['validation_nll']) for row in validation)
    best = min(validation, key=lambda row: row['validation_nll'])
    assert best['step'] == recorded['selected_step']
    assert best['validation_nll'] == recorded['validation_nll']
    saved = load(directory / 'result.json')
    for key, value in saved.items():
        assert recorded[key] == value, (directory, key)
    assert recorded['frozen_weights_unchanged']
    groups = recorded['optimizer_groups']
    names = [name for group in groups for name in group['param_names']]
    assert len(names) == len(set(names))
    assert sum(group['parameter_count'] for group in groups) == recorded['trainable_parameters']
    # Qwen2.5-3B: 36 layers, q=[2048,2048], v=[256,2048].
    expected_parameters = 36 * configuration['rank'] * (4096 + 2304)
    if recorded['method'].startswith('dora'):
        expected_parameters += 36 * 2304
    if recorded['method'] == 'dora_nora_gain':
        expected_parameters += 36 * 4096
    assert recorded['trainable_parameters'] == expected_parameters
    for group in groups:
        name = group['group_name']
        assert group['weight_decay'] == (0.01 if name == 'adapter_factors' else 0.0)
        expected_lr = recorded['lr'] * (recorded['magnitude_lr_multiplier']
                                       if name == 'decoupled_magnitudes' else 1.0)
        assert group['lr'] == expected_lr
    return check_checkpoint(directory, recorded, require_checkpoints)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/var/tmp/dora-bench/round2/cogs'))
    p.add_argument('--allow-missing-checkpoints', action='store_true',
                   help='Re-score portable evidence; report omitted checkpoint payloads explicitly.')
    args = p.parse_args()
    rows, predictions, baseline = [], {}, None
    split_reference, sources = None, []
    source_reference, configuration_reference, frozen_reference = None, None, None
    checkpoints_verified = 0
    input_hashes = {}
    for method in METHODS:
        root = args.root / method
        run, manifest, splits = load(root / 'results.json'), load(root / 'manifest.json'), load(root / 'split_manifest.json')
        input_hashes[method] = {name: digest(root / name) for name in
                                ('results.json', 'manifest.json', 'split_manifest.json', 'baseline.json')}
        assert len(run['results']) == 3 and {r['seed'] for r in run['results']} == {42, 43, 44}
        assert run['manifest'] == manifest
        configuration = manifest['configuration']
        comparable_configuration = {k: v for k, v in configuration.items()
                                    if k not in ('output', 'methods', 'baseline_from')}
        if source_reference is None:
            source_reference = manifest['source_sha256']
            configuration_reference = comparable_configuration
        assert source_reference == manifest['source_sha256']
        assert configuration_reference == comparable_configuration
        assert len(run['tuning']) == 4
        expected_candidates = ({(lr, multiplier) for lr in configuration['mlr_learning_rates']
                                for multiplier in configuration['magnitude_multipliers']}
                               if method == 'dora_nora_mlr' else
                               {(lr, 1.0) for lr in configuration['learning_rates']})
        assert {(row['lr'], row['magnitude_lr_multiplier']) for row in run['tuning']} == expected_candidates
        for trial in run['tuning']:
            assert trial['method'] == method and trial['seed'] == 42
            checkpoints_verified += check_training(root / Path(trial['directory']).name, trial, configuration,
                                                   not args.allow_missing_checkpoints)
            if frozen_reference is None:
                frozen_reference = trial['frozen_sha256']
            assert frozen_reference == trial['frozen_sha256']
        if split_reference is None:
            split_reference = splits
        assert splits == split_reference
        for path, expected in manifest['source_sha256'].items():
            assert digest(root / 'source' / path) == expected
            sources.append({'method': method, 'path': path, 'sha256': expected})
        current_baseline = load(root / 'baseline.json')
        if baseline is None:
            baseline = current_baseline
        assert current_baseline == baseline
        best = min(run['tuning'], key=lambda row: row['validation_nll'])
        assert run['selected'][method]['directory'] == best['directory']
        for row in run['results']:
            assert row['method'] == method and row['rank'] == 8
            assert row['lr'] == best['lr'] and row['magnitude_lr_multiplier'] == best['magnitude_lr_multiplier']
            assert row['frozen_weights_unchanged']
            assert row['steps'] == configuration['steps']
            assert row['frozen_sha256'] == frozen_reference
            directory = root / f'{method}_seed{row["seed"]}'
            if row['seed'] == 42:
                checkpoints_verified += check_checkpoint(directory, row, not args.allow_missing_checkpoints)
                assert row['reused_winning_full_budget_trial'] == best['directory']
                assert row['checkpoint_sha256'] == best['checkpoint_sha256']
                assert row['selected_step'] == best['selected_step']
                assert row['validation_nll'] == best['validation_nll']
            else:
                checkpoints_verified += check_training(directory, row, configuration,
                                                       not args.allow_missing_checkpoints)
            for split, data_key in [('iid', 'test'), ('ood', 'gen')]:
                gold = {r['id']: r for r in splits[data_key]}
                scores = check_predictions(directory / f'{split}_predictions.jsonl', gold, row[split])
                predictions[(method, row['seed'], split)] = scores
            rows.append({k: v for k, v in row.items() if k != 'optimizer_groups'})
    for split, data_key in [('iid', 'test'), ('ood', 'gen')]:
        check_predictions(args.root / 'lora' / f'baseline_{split}.jsonl', {r['id']: r for r in split_reference[data_key]}, baseline[split])
    arrays = {}
    for method in METHODS:
        for split, data_key in [('iid', 'test'), ('ood', 'gen')]:
            cats = sorted({r['category'] for r in split_reference[data_key]})
            ids = [[r['id'] for r in split_reference[data_key] if r['category'] == category] for category in cats]
            arrays[(method, split)] = np.array([[[predictions[(method, seed, split)][key] for key in keys] for keys in ids] for seed in (42, 43, 44)])
    ood_categories = sorted({r['category'] for r in split_reference['gen']})
    structural_mask = np.array([name in STRUCTURAL for name in ood_categories])
    assert structural_mask.sum() == 3 and len(ood_categories) == 21
    category_summary = {}
    for method in METHODS:
        arrays[(method, 'structural')] = arrays[(method, 'ood')][:, structural_mask, :]
        arrays[(method, 'lexical')] = arrays[(method, 'ood')][:, ~structural_mask, :]
        category_summary[method] = {name: {'mean': float(arrays[(method, 'ood')][:, i, :].mean()),
                                          'by_seed': arrays[(method, 'ood')][:, i, :].mean(axis=1).tolist()}
                                    for i, name in enumerate(ood_categories)}
    contrasts = [('nora', 'lora'), ('dora', 'lora'), ('dora_nora', 'nora'), ('dora_nora_mlr', 'dora_nora'),
                 ('dora_nora_gain', 'dora_nora'), ('dora_nora_mlr', 'nora'), ('dora_nora_gain', 'nora')]
    intervals = {split: {f'{left}_minus_{right}': paired(arrays[(left, split)] - arrays[(right, split)], 20261008 + i)
                        for i, (left, right) in enumerate(contrasts)} for split in ('iid', 'ood', 'lexical', 'structural')}
    result = {'passed': True, 'runs': len(rows), 'baseline': baseline, 'results': rows,
              'paired_intervals': intervals, 'verified_sources': sources,
              'category_summary': category_summary,
              'input_sha256': input_hashes,
              'verified_training_fits': 36,
              'adapter_checkpoint_hashes_verified': checkpoints_verified,
              'adapter_checkpoint_payloads_omitted': 42 - checkpoints_verified,
              'shared_frozen_sha256': frozen_reference,
              'metric_scope': 'Recomputed valid syntax, atom-set exact match, strict string match and atom F1 from saved text using an independent parser. OOD intervals preserve all 21 category strata. Verified every training step, validation checkpoint minimum, and matching source/configuration/frozen-weight hashes across workers.',
              'analysis_source_sha256': digest(Path(__file__))}
    (args.root / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    source = args.root / 'analysis_source'
    source.mkdir(exist_ok=True)
    (source / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'passed': True, 'runs': len(rows), 'intervals': intervals}, indent=2))


if __name__ == '__main__':
    main()
