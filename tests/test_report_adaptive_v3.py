import json

from scripts.report_adaptive import endpoint_nll, rows


def test_rows_reads_stage_b_tuning_config_and_group_balanced_endpoint(tmp_path):
    run = tmp_path / 'L_FIXED_c00_s1002'
    run.mkdir()
    (run / 'tuning_config.json').write_text(json.dumps({
        'method': 'L_FIXED', 'lr': 1e-4, 'scale': 1.0, 'penalty': 0.0, 'seed': 1002,
    }))
    (run / 'metadata.json').write_text(json.dumps({
        'status': 'complete', 'dev_tune_endpoint_group_balanced_nll': 0.25,
        'dev_tune_endpoint_token_weighted_nll': 0.3,
    }))
    (run / 'launcher_status.json').write_text(json.dumps({'status': 'complete'}))
    record = rows(tmp_path)[0]
    assert record['method'] == 'L_FIXED'
    assert endpoint_nll(record) == 0.25


def test_endpoint_nll_legacy_fallback():
    assert endpoint_nll({'dev_tune_endpoint_nll': 0.4}) == 0.4
