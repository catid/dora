"""Validate and benchmark dense, factorized, and merged DoRA on one device.

Example: python benchmark_dora.py --device cuda:0 --output benchmark_results.json
CUDA timings include synchronization; each raw sample averages --iterations calls.
"""

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from dora import DoRALayer


class DenseDoRA(nn.Module):
    """Independent dense reference: normalize output rows and detach the norm."""

    def __init__(self, layer):
        super().__init__()
        for name in ("weight", "bias", "lora_A", "lora_B", "m"):
            self.register_parameter(name, copy.deepcopy(getattr(layer, name)))

    def forward(self, x):
        work_dtype = (
            torch.float32
            if self.weight.dtype in (torch.float16, torch.bfloat16)
            else self.weight.dtype
        )
        adapted = self.weight.to(work_dtype) + (
            self.lora_A.to(work_dtype) @ self.lora_B.to(work_dtype)
        )
        with torch.no_grad():
            norm = adapted.norm(dim=1, keepdim=True).clamp_min(
                torch.finfo(self.weight.dtype).tiny
            )
        effective = (adapted * (self.m / norm)).to(self.weight.dtype)
        return F.linear(x, effective, self.bias)


def difference(actual, expected):
    delta = (actual.detach().double() - expected.detach().double())
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.detach().double().norm().clamp_min(1e-30)).item(),
    }


def validate(layer, reference, merged, x):
    """Exercise nonzero adapters and compare both activation and all gradients."""
    if x.dtype == torch.float64:
        rtol, atol = 1e-9, 1e-10
    elif x.dtype == torch.float32:
        rtol, atol = 2e-4, 2e-5
    elif x.dtype == torch.float16:
        rtol, atol = 1e-2, 5e-3
    else:
        rtol, atol = 8e-2, 4e-2
    actual_x = x.detach().clone().requires_grad_()
    expected_x = x.detach().clone().requires_grad_()
    actual = layer(actual_x)
    expected = reference(expected_x)
    # Normalize the probe so absolute gradient tolerances do not depend on
    # token count or output width (large reductions amplify roundoff).
    upstream = torch.randn_like(actual) / math.sqrt(actual.numel())
    actual.backward(upstream)
    expected.backward(upstream)
    pairs = {"output": (actual, expected), "input_grad": (actual_x.grad, expected_x.grad)}
    for name in ("lora_A", "lora_B", "m"):
        pairs[name + "_grad"] = (getattr(layer, name).grad, getattr(reference, name).grad)
    with torch.no_grad():
        pairs["merged_output"] = (merged(x), expected)
    metrics = {}
    for name, (value, target) in pairs.items():
        torch.testing.assert_close(
            value, target, rtol=rtol, atol=atol, msg=lambda detail: f"{name}: {detail}"
        )
        metrics[name] = difference(value, target)
    layer.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return {"passed": True, "rtol": rtol, "atol": atol, "comparisons": metrics}


def measure(model, x, backward, args, device):
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run():
        if backward:
            model.zero_grad(set_to_none=True)
            x.grad = None
            model(x).float().square().mean().backward()
        else:
            with torch.inference_mode():
                model(x)

    for _ in range(args.warmup):
        run()
    synchronize()
    model.zero_grad(set_to_none=True)
    x.grad = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    samples = []
    for _ in range(args.samples):
        synchronize()
        start = time.perf_counter_ns()
        for _ in range(args.iterations):
            run()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / args.iterations / 1e6)
    peak = (
        torch.cuda.max_memory_allocated(device) - baseline
        if device.type == "cuda"
        else None
    )
    model.zero_grad(set_to_none=True)
    x.grad = None
    return {
        "sample_ms_per_call": samples,
        "median_ms_per_call": statistics.median(samples),
        "peak_incremental_allocated_bytes": peak,
    }


def provenance(device):
    source = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "git_revision": revision,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("dora.py", "benchmark_dora.py")
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            gpu_name=properties.name,
            gpu_total_memory_bytes=properties.total_memory,
            gpu_compute_capability=[properties.major, properties.minor],
        )
        try:
            result["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            result["nvidia_smi"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float64", "float16", "bfloat16"), default="float32")
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("in_features", "out_features", "rank", "samples", "iterations", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or any(tokens < 1 for tokens in args.tokens):
        parser.error("warmup must be nonnegative and tokens must be positive")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    base = nn.Linear(args.in_features, args.out_features, device=device, dtype=dtype)
    layer = DoRALayer(
        args.in_features, args.out_features, args.rank, weight=base.weight, bias=base.bias
    )
    with torch.no_grad():
        layer.lora_B.normal_(std=0.02)
        layer.m.mul_(torch.empty_like(layer.m).uniform_(0.8, 1.2))
    reference = DenseDoRA(layer)
    merged = layer.to_linear()
    config = vars(args).copy()
    config["output"] = str(args.output) if args.output else None
    report = {
        "environment": provenance(device),
        "configuration": config,
        "methodology": {
            "reference": "Dense DoRA; output-row norm of W + A @ B detached from autograd.",
            "gradient_validation": "Gaussian vector-Jacobian probe divided by sqrt(output.numel()).",
            "forward": "torch.inference_mode; no autograd graph",
            "forward_backward": "Clear gradients; forward; float32 squared mean loss; backward including input gradients.",
            "timing": "Synchronized wall-clock batch; each sample is mean milliseconds per call.",
            "memory": "CUDA max_memory_allocated minus live baseline after clearing gradients; excludes existing model/input tensors.",
            "merged": "to_linear conversion excluded from inference timing; adapters fixed for inference.",
        },
        "cases": [],
    }
    for tokens in args.tokens:
        x = torch.randn(tokens, args.in_features, device=device, dtype=dtype, requires_grad=True)
        case = {"tokens": tokens, "validation": validate(layer, reference, merged, x), "measurements": {}}
        for name, model in (("dense", reference), ("factorized", layer), ("merged", merged)):
            measurements = {"forward": measure(model, x, False, args, device)}
            if name != "merged":
                measurements["forward_backward"] = measure(model, x, True, args, device)
            case["measurements"][name] = measurements
        case["speedup_dense_over_factorized"] = {
            mode: case["measurements"]["dense"][mode]["median_ms_per_call"]
            / case["measurements"]["factorized"][mode]["median_ms_per_call"]
            for mode in ("forward", "forward_backward")
        }
        report["cases"].append(case)
        print(f"tokens={tokens}: " + "; ".join(
            f"{name} forward={values['forward']['median_ms_per_call']:.3f} ms"
            + (f", forward+backward={values['forward_backward']['median_ms_per_call']:.3f} ms" if "forward_backward" in values else "")
            for name, values in case["measurements"].items()
        ), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
