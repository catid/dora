"""Forward/backward timing for next recipes with diagnostic reductions disabled.

CUDA_VISIBLE_DEVICES=3 python -m experiments.next_microbenchmark --output ...
No optimizer steps, quality measurements, or production-throughput claims.
"""

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import torch
from torch import nn

from experiments import next_ablations as ab


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def disable_tracking(raw):
    """The only timed-instance override: omit read-only diagnostic reductions."""
    pass


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def validate(instrumented, timed, x, autocast):
    a, b = [x.detach().clone().requires_grad_() for _ in range(2)]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        actual, expected = timed(a), instrumented(b)
    torch.manual_seed(311)
    probe = torch.randn_like(expected) / math.sqrt(expected.numel())
    actual.backward(probe); expected.backward(probe)
    pairs = {'output': (actual, expected), 'input_gradient': (a.grad, b.grad)}
    for name in ('lora_A', 'lora_B', 'm'):
        pairs[name + '_gradient'] = (getattr(timed, name).grad, getattr(instrumented, name).grad)
    checked = {}
    for name, (value, reference) in pairs.items():
        assert torch.isfinite(value).all() and torch.isfinite(reference).all(), name
        assert torch.equal(value, reference), name
        assert torch.count_nonzero(value) > 0, name
        checked[name] = {'bitwise_equal': True, 'dtype': str(value.dtype),
                         'l2': float(value.float().norm()), 'max_absolute_error': 0.0}
    assert timed.weight.grad is None and timed.bias.grad is None
    diagnostics = instrumented.assert_no_clamp()
    assert math.isfinite(diagnostics['minimum_raw_norm']) and diagnostics['minimum_raw_norm'] > 0
    assert diagnostics['clamp_activations'] == 0
    assert timed._norm_evaluations + 1 == instrumented._norm_evaluations
    timed.zero_grad(set_to_none=True); instrumented.zero_grad(set_to_none=True)
    return {'status': 'pass', 'checks': checked, 'instrumented_diagnostics': diagnostics,
            'override': 'Instance _track_norm only; forward, norm, projection and parameter code unchanged'}


def measure(layer, x, backward, autocast, args):
    def run():
        if backward:
            layer.zero_grad(set_to_none=True)
            x.grad = None
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                output = layer(x)
                loss = output.float().square().mean()
            loss.backward()
        else:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                layer(x)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    layer.zero_grad(set_to_none=True); x.grad = None
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.samples):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        for _ in range(args.iterations):
            run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / args.iterations / 1e6)
    peak = torch.cuda.max_memory_allocated() - baseline
    assert peak > 0, 'Invalid incremental allocation: live baseline changed during measurement'
    layer.zero_grad(set_to_none=True); x.grad = None
    return {'sample_ms_per_call': samples, 'median_ms_per_call': statistics.median(samples),
            'peak_incremental_allocated_bytes': peak, 'preexisting_allocated_bytes': baseline}


def nvidia_smi():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu',
                                    '--format=csv,noheader'], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    assert args.warmup >= 0 and args.samples > 0 and args.iterations > 0
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.json').exists():
        raise RuntimeError('Use a fresh output directory to preserve prior timing evidence')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gate_path = Path('/var/tmp/dora-bench/next_ablation_validation.json')
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'pass' and gate['backend_source_sha256'] == ab.SOURCE_SHA256
    root = Path(__file__).resolve().parents[1]
    sources = ['experiments/next_microbenchmark.py', 'experiments/next_ablations.py',
               'experiments/change_ablations.py', 'dora.py']
    hashes = {name: sha(root / name) for name in sources}
    for name in sources:
        target = args.output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    (args.output / 'sources' / gate_path.name).write_bytes(gate_path.read_bytes())
    props = torch.cuda.get_device_properties(0)
    result = {
        'schema': 'dora-next-microbenchmark-v1', 'complete': False,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'device': props.name, 'compute_capability': [props.major, props.minor],
                        'gpu_total_memory_bytes': props.total_memory, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'cpu_threads': torch.get_num_threads(), 'matmul_precision': torch.get_float32_matmul_precision(),
                        'tf32': torch.backends.cuda.matmul.allow_tf32, 'nvidia_smi_before': nvidia_smi(),
                        'git_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()},
        'command': [sys.executable, *sys.argv], 'source_sha256': hashes, 'validation_gate_sha256': sha(gate_path),
        'recipes': {name: ab.recipe_manifest(name) for name in ab.METHODS},
        'configuration': {'shapes': [[1024,1024,32],[1024,1024,256],[4096,4096,32]], 'rank': 8,
                          'precisions': {'float32': 'FP32 frozen weight/bias/input/factors/magnitude; autocast disabled; TF32 disabled',
                                         'bfloat16_autocast': 'BF16 frozen weight/bias/input; FP32 factors/magnitude; BF16 CUDA autocast; per-recipe explicit norm scope unchanged'},
                          'warmup_calls': args.warmup, 'samples': args.samples, 'calls_per_sample': args.iterations,
                          'base_seed': 42, 'adapter_seed': 71, 'nonzero_seed': 991, 'case_order_seed': 20261009},
        'methodology': {'projection': 'One biased Linear, rank8; constructor policies preserved. Set only its initially zero factor to nonzero normal std0.01 and rescale m by a deterministic0.9..1.1 ramp before checks/timing.',
                        'validation': 'Nonzero output/input gradient/all trainable gradients bitwise equal to instrumented instance for every shape/precision/recipe.',
                        'timing': 'Synchronized sample wall time, includes per-call gradient clearing and autocast context; excludes optimizer and validation. Forward uses inference_mode. Forward+backward uses FP32 squared-mean loss with input gradients.',
                        'memory': 'Max memory_allocated minus live allocation after warmup and clearing gradients, includes input/adapter grads and autograd work; excludes preexisting weight/factors/magnitude/input and caching-allocator reservation.',
                        'diagnostics': 'Only the instance _track_norm callback disabled. Constructor and quality semantics unchanged; previous task-loop timings included these read-only reductions.',
                        'scope': 'Task-agnostic layer speed/memory measurements, no quality claim. Square layers keep parameter counts matched; full training models can behave differently.'},
        'cases': []}
    order_rng = random.Random(20261009)
    for precision in result['configuration']['precisions']:
        autocast = precision == 'bfloat16_autocast'
        dtype = torch.bfloat16 if autocast else torch.float32
        for d_in, d_out, tokens in result['configuration']['shapes']:
            torch.manual_seed(42)
            base = nn.Linear(d_in, d_out, device='cuda', dtype=dtype)
            x = torch.randn(tokens, d_in, device='cuda', dtype=dtype, requires_grad=True)
            base_hash = tensor_hash(base.weight)
            methods = list(ab.METHODS); order_rng.shuffle(methods)
            measured = []
            for method in methods:
                torch.manual_seed(71)
                instrumented = ab.AblationLinear(copy.deepcopy(base), method, 8)
                torch.manual_seed(991)
                with torch.no_grad():
                    getattr(instrumented, instrumented.zero_initialized_factor).normal_(std=.01)
                    instrumented.m.mul_(torch.linspace(.9,1.1,instrumented.m.numel(),device='cuda').reshape(instrumented.m.shape))
                timed = copy.deepcopy(instrumented)
                # A bound MethodType stored on its own instance creates a
                # self-reference cycle whose later collection can corrupt the
                # live-allocation baseline. This plain function has no cycle.
                timed._track_norm = disable_tracking
                validation = validate(instrumented, timed, x, autocast)
                del instrumented
                torch.cuda.synchronize()
                record = {'method': method, 'precision': precision, 'in_features': d_in, 'out_features': d_out,
                          'tokens': tokens, 'rank': 8, 'execution_order': methods.index(method),
                          'frozen_weight_sha256': base_hash, 'recipe_sha256': ab.recipe_manifest(method)['recipe_sha256'],
                          'parameter_dtypes': {name: str(p.dtype) for name,p in timed.named_parameters()},
                          'trainable_parameters': sum(p.numel() for p in timed.parameters() if p.requires_grad),
                          'validation': validation,
                          'forward': measure(timed, x, False, autocast, args),
                          'forward_backward': measure(timed, x, True, autocast, args)}
                assert torch.isfinite(timed.m).all()
                measured.append(record)
                print(json.dumps({'method':method,'precision':precision,'shape':[d_in,d_out,tokens],
                                  'forward_backward_ms':record['forward_backward']['median_ms_per_call'],
                                  'incremental_MiB':record['forward_backward']['peak_incremental_allocated_bytes']/2**20}),flush=True)
                del timed
                torch.cuda.empty_cache()
            baseline = next(row for row in measured if row['method'] == 'next_base')
            for record in measured:
                record['relative_to_next_base'] = {mode: {
                    'speedup': baseline[mode]['median_ms_per_call'] / record[mode]['median_ms_per_call'],
                    'peak_incremental_memory_ratio': record[mode]['peak_incremental_allocated_bytes'] / baseline[mode]['peak_incremental_allocated_bytes']}
                    for mode in ('forward', 'forward_backward')}
            result['cases'].extend(measured)
            (args.output / 'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            del base,x
    assert len(result['cases']) == 36 and all(row['validation']['status'] == 'pass' for row in result['cases'])
    result['complete'] = True
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    result['environment']['nvidia_smi_after'] = nvidia_smi()
    (args.output / 'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','cases':len(result['cases']),'output':str(args.output)}),flush=True)


if __name__ == '__main__':
    main()
