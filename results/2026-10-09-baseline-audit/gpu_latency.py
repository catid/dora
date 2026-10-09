"""Exact-original/axis-only/detached/factorized FP32 CUDA microbenchmark.

Run: CUDA_VISIBLE_DEVICES=1 /var/tmp/dora-bench/venv/bin/python gpu_latency.py
No optimizer updates, model-quality claims or original/current equivalence claims.
"""
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from types import SimpleNamespace
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'gpu_source'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def probe(model, x, upstream):
    model.zero_grad(set_to_none=True)
    current_x = x.detach().clone().requires_grad_()
    output = model(current_x)
    output.backward(upstream)
    values = {'output': output, 'input_gradient': current_x.grad}
    values.update({name + '_gradient': getattr(model, name).grad for name in ('m', 'lora_A', 'lora_B')})
    assert not model.weight.requires_grad and not model.bias.requires_grad
    assert model.weight.grad is None and model.bias.grad is None
    saved, diagnostics = {}, {}
    for name, value in values.items():
        assert value is not None and torch.isfinite(value).all().item(), name
        saved[name] = value.detach().cpu()
        diagnostics[name] = {'shape': list(value.shape), 'finite': True,
                             'l2_norm': value.detach().double().norm().item(),
                             'max_absolute': value.detach().abs().max().item()}
    model.zero_grad(set_to_none=True)
    return saved, diagnostics


def differences(first, second):
    delta = first.double() - second.double()
    return {'max_absolute': delta.abs().max().item(),
            'relative_l2': (delta.norm() / second.double().norm().clamp_min(1e-30)).item()}


def main():
    original = load('dora_original', SOURCE / 'dora_original_bb97617.py')
    axis = load('dora_axis_only', SOURCE / 'dora_axis_only_dim1.py')
    current = load('dora', SOURCE / 'dora.py')
    existing = load('existing_benchmark', SOURCE / 'benchmark_dora.py')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    args = SimpleNamespace(warmup=10, samples=7, iterations=20)
    methods = ('original_dim0_full_gradient', 'axis_only_dim1_full_gradient',
               'dense_dim1_detached_reference', 'current_factorized_dim1_detached')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
              'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(device),
                              'device': str(device), 'threads': torch.get_num_threads(),
                              'matmul_precision': torch.get_float32_matmul_precision(),
                              'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                              'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version', '--format=csv,noheader'], text=True)},
              'source_sha256': {p.name: sha(p) for p in sorted(SOURCE.glob('*.py'))},
              'benchmark_script_sha256': sha(Path(__file__)),
              'configuration': {'dtype': 'float32', 'rank': 8, 'seed': 1234,
                                'warmup': 10, 'samples': 7, 'calls_per_sample': 20,
                                'cases': [[1024, 1024, 32], [1024, 1024, 256], [4096, 4096, 32]],
                                'measurement_order': list(methods)},
              'semantics': {
                  methods[0]: 'Exact DoRALayer source at bb97617; dim0 column normalization; full norm gradient. CPU constructor then whole module moved to GPU; forward unmodified.',
                  methods[1]: 'Exact original source with only both dim=0 occurrences replaced by dim=1; row normalization; full norm gradient.',
                  methods[2]: 'DenseDoRA reference captured from existing benchmark_dora.py; row normalization with detached norm.',
                  methods[3]: 'Current dora.py DoRALayer; row normalization with detached norm and factorized activation path.'},
              'methodology': {'weights': 'Identical copied FP32 W, bias, A, B in all methods; B is nonzero N(0,.02). Magnitude uses CPU norm of W on the method axis times a shared U(.8,1.2) vector. All row-based methods use identical m. Original column magnitudes intentionally retain original semantics.',
                              'forward': 'Existing benchmark.measure with inference_mode.',
                              'forward_backward': 'Existing benchmark.measure: clear adapter/input gradients; forward; FP32 squared-mean loss; backward including input gradients. No optimizer step.',
                              'timing': 'Synchronized wall-clock batches; seven samples each averaging20 calls, after10 warmup calls; same protocol as original reported microbenchmark.',
                              'memory': 'max_memory_allocated minus resident allocation after clearing gradients; excludes already-resident model/input tensors, includes temporary tensors and gradients. All four models resident; no optimizer state.',
                              'probe': 'Shared Gaussian upstream/sqrt(output.numel()); finite output/input/all-adapter gradients; only current vs detached dense is required to match.',
                              'limits': 'Original, row full-gradient and row detached variants have different mathematical or gradient semantics. Latency and memory comparisons do not establish task quality or semantic equivalence. Fixed method order and one device/run; raw timings retained.'},
              'cases': []}
    for width, token_counts in ((1024, (32, 256)), (4096, (32,))):
        torch.manual_seed(1234)
        base = torch.nn.Linear(width, width, dtype=torch.float32)
        shared_a = torch.randn(width, 8) / math.sqrt(8)
        shared_b = torch.randn(8, width) * .02
        multiplier = torch.empty(width).uniform_(.8, 1.2)
        original_model = original.DoRALayer(width, width, 8, base.weight.detach().clone(), base.bias.detach().clone())
        axis_model = axis.DoRALayer(width, width, 8, base.weight.detach().clone(), base.bias.detach().clone())
        factorized = current.DoRALayer(width, width, 8, base.weight.detach().clone(), base.bias.detach().clone())
        with torch.no_grad():
            for model in (original_model, axis_model, factorized):
                model.lora_A.copy_(shared_a)
                model.lora_B.copy_(shared_b)
                model.m.mul_(multiplier.reshape(model.m.shape))
            # Make the row-based parameter point exactly identical, independently
            # of tiny implementation differences in CPU norm accumulation.
            axis_model.m.copy_(factorized.m)
        reference = existing.DenseDoRA(factorized)
        models = dict(zip(methods, (original_model, axis_model, reference, factorized)))
        shared_hashes = {name: tensor_sha(getattr(original_model, name)) for name in ('weight', 'bias', 'lora_A', 'lora_B')}
        for model in models.values():
            assert all(tensor_sha(getattr(model, name)) == digest for name, digest in shared_hashes.items())
            model.to(device)
        for tokens in token_counts:
            generator = torch.Generator().manual_seed(5678 + tokens)
            cpu_x = torch.randn(tokens, width, generator=generator)
            x = cpu_x.to(device).requires_grad_()
            upstream = (torch.randn(tokens, width, generator=generator) / math.sqrt(tokens * width)).to(device)
            case = {'width': width, 'tokens': tokens, 'input_sha256': tensor_sha(cpu_x),
                    'shared_parameter_sha256': shared_hashes,
                    'measurements': {}, 'probes': {}, 'comparisons': {}}
            probes = {}
            for name, model in models.items():
                probes[name], case['probes'][name] = probe(model, x, upstream)
                case['measurements'][name] = {
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'magnitude_shape': list(model.m.shape), 'magnitude_sha256': tensor_sha(model.m),
                    'forward': existing.measure(model, x, False, args, device),
                    'forward_backward': existing.measure(model, x, True, args, device)}
                _, post = probe(model, x, upstream)
                case['measurements'][name]['finite_after_timing'] = all(row['finite'] for row in post.values())
                print(width, tokens, name, json.dumps(case['measurements'][name]), flush=True)
            for first, second, label in ((methods[3], methods[2], 'factorized_vs_detached_dense'),
                                         (methods[1], methods[2], 'full_gradient_vs_detached_row'),
                                         (methods[0], methods[3], 'original_vs_current_different_semantics')):
                case['comparisons'][label] = {}
                for name in ('output', 'input_gradient', 'lora_A_gradient', 'lora_B_gradient', 'm_gradient'):
                    left, right = probes[first][name], probes[second][name]
                    # Original column magnitudes and row magnitudes index different
                    # directions, so their magnitude gradients are not comparable.
                    if left.shape != right.shape:
                        case['comparisons'][label][name] = {'comparable': False, 'reason': 'Different magnitude axes/shapes'}
                        continue
                    case['comparisons'][label][name] = differences(left, right)
                    if label == 'factorized_vs_detached_dense':
                        torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-5)
            case['factorized_detached_dense_parity_passed'] = True
            report['cases'].append(case)
            (ROOT / 'gpu_latency.json').write_text(json.dumps(report, indent=2) + '\n')
            del x, upstream, probes
        del models, original_model, axis_model, reference, factorized, model, base, shared_a, shared_b, multiplier
        gc.collect()
        torch.cuda.empty_cache()
    report['complete'] = True
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (ROOT / 'gpu_latency.json').write_text(json.dumps(report, indent=2) + '\n')
    print('COMPLETE', str(ROOT / 'gpu_latency.json'), flush=True)


if __name__ == '__main__':
    main()
