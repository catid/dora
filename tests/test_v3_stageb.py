import json
from pathlib import Path

import pytest
import yaml

from scripts.run_stageB_v3 import verify_hpo_integrity


def _write_hpo_root(root: Path, complete: bool = True):
    methods = [row['id'] for row in yaml.safe_load(Path('configs/methods.yaml').read_text())['methods']]
    for method in methods:
        for trial in range(12):
            directory = root / f'{method}_c{trial:02d}'
            directory.mkdir(parents=True)
            (directory / 'hpo_config.json').write_text(json.dumps({
                'method': method, 'trial_index': trial, 'seed': 1001,
            }))
            (directory / 'launcher_status.json').write_text(json.dumps({
                'status': 'complete' if complete else 'running',
            }))
            (directory / 'metadata.json').write_text(json.dumps({'status': 'complete', 'input_tokens': 1234}))


def test_hpo_integrity_accepts_declared_grid(tmp_path):
    _write_hpo_root(tmp_path)
    verify_hpo_integrity(tmp_path)


def test_hpo_integrity_rejects_duplicate_trial(tmp_path):
    _write_hpo_root(tmp_path)
    duplicate = tmp_path / 'L_FIXED_c11'
    config = json.loads((duplicate / 'hpo_config.json').read_text())
    config['trial_index'] = 0
    (duplicate / 'hpo_config.json').write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match='duplicate'):
        verify_hpo_integrity(tmp_path)


def test_hpo_integrity_rejects_inconsistent_token_exposure(tmp_path):
    _write_hpo_root(tmp_path)
    metadata = tmp_path / 'L_FIXED_c00' / 'metadata.json'
    metadata.write_text(json.dumps({'status': 'complete', 'input_tokens': 1235}))
    with pytest.raises(RuntimeError, match='token'):
        verify_hpo_integrity(tmp_path)
