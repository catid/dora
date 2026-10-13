"""Shared provenance and selection utilities for the initialization study.

Only new study modules depend on this file. Historical runners remain frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_once(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Frozen artifact changed: {path}'
    else:
        write(path, value)


def tensor_digest(named):
    import torch
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_registry(path, protocol, task=None):
    """Reject incomplete or silently enlarged grids before model work begins."""
    registry = read(path)
    assert registry['schema'] == 'dora-init-study-v1'
    assert registry['protocol_sha256'] == sha(protocol)
    if task is not None:
        assert registry['task'] == task
    assert registry['task'] in ('scifact', 'nfcorpus', 'aircraft', 'extraction')
    assert registry['rank'] == (2 if registry['task'] == 'aircraft' else 8)
    assert registry['seed'] == 42 and registry['seeds'] == [42, 43, 44]
    assert registry['fixed_lr'] in registry['learning_rates']
    candidates = registry['candidates']
    ids = [row['candidate_id'] for row in candidates]
    assert candidates and len(ids) == len(set(ids))
    assert registry['control_id'] in ids
    unique, families = set(), {}
    for row in candidates:
        assert row['family_id'] and row['spec'] and row['candidate_id']
        assert '/' not in row['family_id'] and row['family_id'] not in ('.', '..')
        assert '/' not in row['candidate_id'] and row['candidate_id'] not in ('.', '..')
        assert row['learning_rate'] in registry['learning_rates']
        key = (row['family_id'], digest_json(row['spec']), row['learning_rate'])
        assert key not in unique, f'Duplicate tuning candidate: {row["candidate_id"]}'
        unique.add(key)
        families.setdefault(row['family_id'], []).append(row)
    assert all(any(row['learning_rate'] == registry['fixed_lr'] for row in group)
               for group in families.values())
    if 'candidate_counts' in registry:
        assert registry['candidate_counts'] == {family: len(rows) for family, rows in families.items()}
    validate_study_lane(registry)
    return registry


def spec_parts(spec):
    """The wrapper separates random distributions from published factorizations."""
    return (spec.get('kind', 'random'), spec.get('config', spec))


def validate_study_lane(registry):
    """Keep nonidentity studies and their quantizer controls explicit."""
    lane = registry.get('study_lane', 'centered')
    assert lane in ('centered', 'warmstart', 'quantized')
    candidates = {row['candidate_id']: row for row in registry['candidates']}
    assert spec_parts(candidates[registry['control_id']]['spec']) == ('random', {'mode':'accepted'})
    allowed = {'centered': ('random','data'), 'warmstart': ('random','warmstart'),
               'quantized': ('random','quantized')}
    for candidate in candidates.values():
        kind, config = spec_parts(candidate['spec'])
        assert kind in allowed[lane], (lane, kind)
        if lane != 'centered' and kind == 'random':
            assert config == {'mode':'accepted'}
        if kind == 'warmstart':
            assert config.get('magnitude_policy','original') in ('original','adapted_column')
        if kind == 'quantized' and config.get('method','loftq') == 'loftq':
            paired = candidates[candidate['comparison_control_id']]
            paired_kind, control = spec_parts(paired['spec'])
            assert paired_kind == 'quantized' and control['method'] == 'quantizer_only'
            assert paired['learning_rate'] == candidate['learning_rate'], 'Quantizer control LR differs'
            for key, default in (('bits',4), ('blocksize',64), ('magnitude_policy','original')):
                assert config.get(key,default) == control.get(key,default), key
            iterations = config.get('iterations',1)
            assert control.get('iterations',1) == iterations
            if iterations > 1:
                assert control.get('base_source') == 'paired_loftq'
    return lane


def backend_for(candidate, seed, calibration=None):
    from experiments import init_ablations as ab
    kind, config = spec_parts(candidate['spec'])
    if kind == 'random':
        return ab.backend(ab.InitSpec(**config), factor_seed=seed)
    if kind == 'data':
        # The data backend owns compensation semantics and its calibration gate.
        from experiments import init_data
        return init_data.backend(config, factor_seed=seed, calibration=calibration)
    if kind == 'quantized':
        from experiments import init_loftq
        return init_loftq.backend(config, factor_seed=seed)
    if kind == 'warmstart':
        from experiments import init_warmstart
        return init_warmstart.backend(config, factor_seed=seed, calibration=calibration)
    raise ValueError(f'Unknown initializer kind: {kind}')


def candidate_manifest(candidate, seed=42, calibration=None):
    backend = backend_for(candidate, seed, calibration)
    return backend.recipe_manifest(candidate['candidate_id'])


def select_candidates(registry, pilot_rows, *, minimize=False):
    """Freeze fixed-rate and joint winners only after the complete grid finishes.

    Each pilot row supplies candidate_id, status, selection_value and bindings.
    Numerical failures are retained; a wholly failed family has no winner.
    """
    by_id = {row['candidate_id']: row for row in pilot_rows}
    assert len(by_id) == len(pilot_rows) == len(registry['candidates'])
    assert set(by_id) == {row['candidate_id'] for row in registry['candidates']}
    families = {}
    for candidate in registry['candidates']:
        families.setdefault(candidate['family_id'], []).append(candidate)
    selections = {}
    for family, rows in families.items():
        choices = {}
        for lane in ('fixed_original_lr', 'validation_tuned'):
            eligible = [row for row in rows if by_id[row['candidate_id']]['status'] == 'pass'
                        and (lane != 'fixed_original_lr' or row['learning_rate'] == registry['fixed_lr'])]
            def key(candidate):
                value = by_id[candidate['candidate_id']]['selection_value']
                secondary = by_id[candidate['candidate_id']].get('selection_secondary', 0.)
                return (value if minimize else -value, secondary,
                        candidate['learning_rate'], candidate['candidate_id'])
            choices[lane] = min(eligible, key=key)['candidate_id'] if eligible else None
        selections[family] = choices
    return {'schema': 'dora-init-selections-v1', 'registry_sha256': digest_json(registry),
            'test_used_for_selection': False, 'minimize_selection_value': minimize,
            'pilots': pilot_rows, 'selections': selections}
