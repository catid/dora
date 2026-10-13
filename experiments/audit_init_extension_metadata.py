"""Independent checks of extension observer and factor-generation receipts."""
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def calibration(value,task,observer_sha):
    meta=value['metadata'];source=Path(meta['source_calibration_manifest'])
    assert sha(source)==meta['source_calibration_manifest_sha256']
    parent=json.loads(source.read_text())
    assert value['module_files']==parent['module_files'] and value['module_sha256']==parent['module_sha256']
    assert all(meta[key]==item for key,item in parent['metadata'].items())
    binding=meta['extension_gme_binding']
    assert binding['task']==task and binding['parent_calibration_manifest_sha256']==sha(source)
    assert binding['observer_source_sha256']==observer_sha
    assert binding['batch_manifest_sha256']==digest(meta['batches'])
    assert binding['model_repository']==meta['model_repository'] and binding['model_revision']==meta['model_revision']
    assert binding['baseline_policy']=='accepted_zero_update_dora' and binding['moment_domain']=='whole_transformer_blocks'
    assert binding['classifier_seed']==(meta['classifier_seed'] if task=='aircraft' else None)
    return binding


def diagnostics(rows,candidate,calibration_value,task):
    kind=candidate['spec'].get('kind','random')
    if kind=='random':return
    config=candidate['spec']['config'];rows=list(rows.values()) if isinstance(rows,dict) else rows
    blocks=12 if task=='aircraft' else 6
    width=768 if task=='aircraft' else 384
    batch_count=16 if task=='aircraft' else 32
    metadata=calibration_value['metadata']
    expected_count=(128*197 if task=='aircraft' else sum(r['query_nonpadding_tokens']+r['document_nonpadding_tokens'] for r in metadata['batches']))*width
    seen=set()
    for row in rows:
        init=row['initializer'];record=init['callback_metadata']
        assert type(record['factor_cache_hit']) is bool
        assert all(math.isfinite(record[key]) and record[key]>=0 for key in ('source_compute_seconds','actual_cache_or_compute_seconds'))
        if kind=='extension_centered':
            assert record['method']=='lora_null' and record['calibration_role']==config['calibration_role']=='task_only'
            assert record['compensation_required'] and record['balance_gain']==config['balance_gain']
            eigen=record['tail_eigenvalues'];assert len(eigen)==(2 if task=='aircraft' else 8)
            assert all(math.isfinite(v) for v in eigen) and eigen==sorted(eigen)
            assert 0<=record['rank_space_reconstruction_relative_error']<1e-3
            continue
        assert kind=='extension_warmstart' and record['method']=='gme'
        # Whole-model metadata is repeated per module; validate each distinct payload once.
        key=digest(record)
        if key in seen:continue
        seen.add(key)
        assert record['recipe']==config['recipe'] and record['tau_b']==config['tau_b']
        assert record['seed']==init['factor_seed'] and record['required_moment_policy']==config['moment_policy']
        assert record['task_observer_binding']==metadata['extension_gme_binding']
        assert record['baseline_policy']=='accepted_zero_update_dora' and not record['calibration_uses_labels']
        assert record['actual_task_head_seed']==metadata.get('classifier_seed')
        assert len(record['block_names'])==blocks and len(record['module_layers'])==len(rows)
        assert set(map(int,record['block_names']))==set(range(blocks))
        stages={'baseline','probe_skew','final'}|{f'refine_{i}' for i in range(config['refine_steps'])}
        if config['recipe']!='source_v1':stages.add('probe_scale')
        assert set(record['observations'])==set(record['probes'])==stages
        base=torch.tensor(record['probes']['baseline']['probe'],dtype=torch.float32)
        for stage,values in record['observations'].items():
            assert set(map(int,values))==set(range(blocks))
            for value in values.values():
                assert value['count']==expected_count and value['var']>=0
                assert all(math.isfinite(v) for v in value.values())
            probe=record['probes'][stage]
            assert probe['forward_batch_count']==batch_count
            assert set(map(int,probe['block_hook_calls']))==set(range(blocks))
            assert set(probe['block_hook_calls'].values())=={batch_count}
            tensor=torch.tensor(probe['probe'],dtype=torch.float32)
            assert tensor.shape==base.shape and torch.isfinite(tensor).all()
            value=hashlib.sha256();value.update(b'probe');value.update(str((tuple(tensor.shape),tensor.dtype)).encode());value.update(tensor.numpy().tobytes())
            assert value.hexdigest()==probe['probe_sha256']
            assert math.isclose(float((tensor-base).abs().max()),probe['max_absolute_drift'],rel_tol=0,abs_tol=1e-12)
            assert math.isclose(float((tensor-base).square().mean()),probe['mean_squared_drift'],rel_tol=0,abs_tol=1e-12)


def exact_history(paths,expected):
    """An unavailable seed/rate/budget has no historical parity claim."""
    for path in map(Path,paths):
        result=path/'result.json'
        if not result.exists():continue
        value=json.loads(result.read_text())
        if all(value.get(key)==item for key,item in expected.items()):return path
    return None


def numerical_failure(record):
    assert record['type'] in ('RuntimeError','ValueError')
    message=record['message'].lower()
    assert any(token in message for token in ('nonfinite','failed to converge','did not converge',
        'convergence failed','cor da inverse','corda inverse','algorithm failed to converge','singular matrix','ill-conditioned'))
