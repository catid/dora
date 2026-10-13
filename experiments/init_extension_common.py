"""Extension-only adapter; the frozen main common module remains unchanged."""
from experiments.init_common import (sha, digest_json, read, write, write_once,
                                     tensor_digest, spec_parts, select_candidates)
from experiments.init_extension_registry import load_registry


def validate_study_lane(registry):
    assert registry['schema']=='dora-init-extension-v1' and registry['study_lane']=='extension'
    for candidate in registry['candidates']:
        kind,config=spec_parts(candidate['spec'])
        assert kind in ('random','extension_centered','extension_warmstart')
        expected='warmstart' if kind=='extension_warmstart' else 'centered'
        assert candidate['initialization_lane']==expected
        if kind=='extension_warmstart':assert config['magnitude_policy']=='original'
    return 'extension'


def backend_for(candidate, seed, calibration=None, *, gme_prepare=None):
    from experiments import init_extension_runtime
    return init_extension_runtime.backend_for(candidate,seed,calibration,gme_prepare=gme_prepare)


def candidate_manifest(candidate, seed=42, calibration=None):
    return backend_for(candidate,seed,calibration).recipe_manifest(candidate['candidate_id'])
