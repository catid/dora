"""Balanced teacher diagnostics for magnitude, direction, and column amplitudes.

These are controlled mechanism tests, not downstream quality benchmarks. The
same functions are observed in white and rescaled input coordinates. Teacher
families deliberately include both LoRA-representable and DoRA-representable
updates; results are stratified instead of declaring a universal winner.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.second_round_adapters import AdapterLinear, METHODS, parameter_groups


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cells(ranks):
    for rank in ranks:
        for coordinates in ("white", "rescaled"):
            yield rank, "row_scale", 0, coordinates
            for family in ("unit_column_low_rank", "heterogeneous_column_low_rank", "mixed_row_and_direction"):
                for teacher_rank in (rank - 1, rank):
                    yield rank, family, teacher_rank, coordinates


def make_problem(rank, family, teacher_rank, coordinates, args):
    family_index = ("row_scale", "unit_column_low_rank", "heterogeneous_column_low_rank", "mixed_row_and_direction").index(family)
    seed = 20261008 + 1000 * rank + 100 * family_index + teacher_rank
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(args.d_out, args.d_in, generator=generator) / math.sqrt(args.d_in)
    drift = torch.sign(torch.randn(args.d_out, generator=generator)) * (0.5 + 0.5 * torch.rand(args.d_out, generator=generator))
    if family == "row_scale":
        delta = drift[:, None] * base
    else:
        up = torch.linalg.qr(torch.randn(args.d_out, teacher_rank, generator=generator), mode="reduced").Q
        down = F.normalize(torch.randn(teacher_rank, args.d_in, generator=generator), dim=0)
        if family == "heterogeneous_column_low_rank":
            gains = torch.exp(torch.linspace(-1.2, 1.2, args.d_in)[torch.randperm(args.d_in, generator=generator)])
            down = down * gains[None, :]
        directional = up @ down
        directional *= base.norm() / directional.norm()
        delta = directional if family != "mixed_row_and_direction" else directional + drift[:, None] * base
    delta *= args.strength * base.norm() / delta.norm()
    teacher = base + delta
    feature_scale = torch.ones(args.d_in)
    if coordinates == "rescaled":
        feature_scale = torch.logspace(-0.6, 0.6, args.d_in)[torch.randperm(args.d_in, generator=generator)]
    # Keep the latent datasets identical between white/rescaled coordinates.
    data_generator = torch.Generator().manual_seed(seed + 1_000_000)
    data = {"base_weight": base / feature_scale, "teacher_weight": teacher / feature_scale,
            "canonical_base": base, "canonical_teacher": teacher, "feature_scale": feature_scale}
    for name, count in (("train", args.train_size), ("validation", args.validation_size), ("test", args.test_size)):
        z = torch.randn(count, args.d_in, generator=data_generator)
        data[f"x_{name}"] = z * feature_scale
        data[f"y_{name}"] = F.linear(z, teacher)
    singular_values = torch.linalg.svdvals(delta.double())
    metadata = {"adapter_rank": rank, "family": family, "teacher_direction_rank": teacher_rank,
                "coordinates": coordinates, "seed": seed, "strength": args.strength,
                "teacher_delta_numerical_rank": int(torch.linalg.matrix_rank(delta.double(), atol=1e-6)),
                "lora_population_rank_bound": float(singular_values[rank:].square().sum() / singular_values.square().sum()),
                "teacher_mean_abs_relative_row_norm_change": float((teacher.norm(dim=1) / base.norm(dim=1) - 1).abs().mean()),
                "coordinate_scale_range": [float(feature_scale.min()), float(feature_scale.max())]}
    return data, metadata


@torch.no_grad()
def weight_metrics(layer, data):
    effective = layer.to_linear().weight
    target, base = data["teacher_weight"], data["base_weight"]
    delta = target - base
    scaling = data["feature_scale"]
    column_amplitudes = (effective - base).norm(dim=0)
    target_amplitudes = delta.norm(dim=0)
    cosine = F.cosine_similarity(effective, target, dim=1).clamp(-1, 1)
    metrics = {"weighted_weight_error_ratio": float(((effective - target) * scaling).square().sum() /
                                                       (delta * scaling).square().sum()),
               "row_norm_ratio_rmse": float(((effective.norm(dim=1) - target.norm(dim=1)) /
                                               base.norm(dim=1)).square().mean().sqrt()),
               "mean_target_direction_error_degrees": float(cosine.acos().mean() * 180 / math.pi),
               "column_amplitude_relative_rmse": float((column_amplitudes - target_amplitudes).square().sum().sqrt() /
                                                        target_amplitudes.square().sum().sqrt()),
               "learned_mean_abs_relative_row_norm_change": float((effective.norm(dim=1) / base.norm(dim=1) - 1).abs().mean())}
    if layer.use_nora:
        metrics["normalized_down_column_error"] = float((F.normalize(layer.lora_A, dim=0).norm(dim=0) - 1).abs().max())
    if layer.log_gain is not None:
        gains = layer.log_gain.exp()
        metrics["gain_min"] = float(gains.min())
        metrics["gain_max"] = float(gains.max())
        metrics["gain_std"] = float(gains.std())
    return metrics


@torch.no_grad()
def mse_metrics(layer, data, split):
    raw = F.mse_loss(layer(data[f"x_{split}"]), data[f"y_{split}"])
    baseline = F.mse_loss(F.linear(data[f"x_{split}"], data["base_weight"]), data[f"y_{split}"])
    return {"mse": float(raw), "frozen_mse": float(baseline), "relative_to_frozen": float(raw / baseline)}


def new_layer(data, method, rank, seed):
    torch.manual_seed(seed)
    base = nn.Linear(data["base_weight"].shape[1], data["base_weight"].shape[0], bias=False, device="meta")
    base.weight = nn.Parameter(data["base_weight"].clone(), requires_grad=False)
    return AdapterLinear(base, method, rank=rank)


def fit(data, method, rank, seed, lr, multiplier, args):
    layer = new_layer(data, method, rank, seed)
    groups = parameter_groups(layer, lr, magnitude_lr_multiplier=multiplier, weight_decay=0.0)
    optimizer = torch.optim.AdamW(groups, fused=True)
    parameters = [parameter for group in groups for parameter in group["params"]]
    baseline_loss = F.mse_loss(F.linear(data["x_train"], data["base_weight"]), data["y_train"])
    checkpoints = {0, 1, 10, 25, 50, 100, 200, 400, args.steps}
    curves = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(args.steps + 1):
        if step in checkpoints:
            validation = mse_metrics(layer, data, "validation")
            if not math.isfinite(validation["mse"]):
                raise RuntimeError(f"Nonfinite result: {method}, lr={lr}, step={step}")
            curves.append({"step": step, "validation": validation, **weight_metrics(layer, data)})
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(layer(data["x_train"]), data["y_train"]) / baseline_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not torch.equal(layer.weight, data["base_weight"]):
        raise RuntimeError("Frozen base weight changed")
    result = {"seed": seed, "method": method, "rank": rank, "learning_rate": lr,
              "magnitude_lr_multiplier": multiplier, "steps": args.steps, "seconds": elapsed,
              "trainable_parameters": sum(parameter.numel() for parameter in parameters),
              "optimizer_groups": [{key: value for key, value in group.items() if key != "params"} for group in groups],
              "curves": curves, "validation": curves[-1]["validation"],
              "weight_metrics": weight_metrics(layer, data), "frozen_weight_unchanged": True}
    return layer, result


def candidates(method):
    if method == "dora_nora_mlr":
        return [(lr, multiplier) for lr in (0.003, 0.03) for multiplier in (0.1, 0.01)]
    return [(lr, 1.0) for lr in (0.001, 0.003, 0.01, 0.03)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/var/tmp/dora-bench/second_round/teacher"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--d-in", type=int, default=128)
    parser.add_argument("--d-out", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    if args.steps < 1 or any(rank < 2 for rank in args.ranks):
        parser.error("steps must be positive; main grid ranks must be at least2")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    source_root = Path(__file__).resolve().parents[2]
    sources = {"teacher.py": Path(__file__), "second_round_adapters.py": source_root / "experiments/second_round_adapters.py",
               "adapters.py": source_root / "experiments/adapters.py", "dora.py": source_root / "dora.py"}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance = {"config": config, "source_sha256": {name: sha256(path) for name, path in sources.items()},
                  "torch": torch.__version__, "python": platform.python_version(), "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "protocol": {"precision": "FP32, highest matmul precision", "optimizer": "AdamW",
                               "weight_decay": 0, "grad_clip": 1, "training": "full batch; noiseless linear teacher",
                               "selection": "four full-budget candidates on first-seed validation; selected seed reused",
                               "objective": "train MSE divided by frozen-model train MSE",
                               "coordinates": "x=z*s and W=Wcanonical/s; underlying function/data unchanged",
                               "not_downstream_quality": True}}
    provenance_path = args.output / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["config"] != config or previous["source_sha256"] != provenance["source_sha256"]:
            raise RuntimeError("Existing experiment configuration/source differs; use a fresh output directory")
    write_json(provenance_path, provenance)
    archive = args.output / "executed_sources"
    archive.mkdir(exist_ok=True)
    for name, path in sources.items():
        (archive / name).write_bytes(path.read_bytes())

    # Exact rank1 capacity diagnostic for a positive, heterogeneous-amplitude
    # rank1 update: NoRA columns must equal ±b. The optimal common amplitude is
    # the mean. LoRA and normalized columns with learned gains represent it.
    amplitudes = torch.exp(torch.linspace(-1.2, 1.2, args.d_in)).double()
    write_json(args.output / "rank1_capacity_check.json", {
        "amplitudes": amplitudes.tolist(), "lora_exact_relative_error": 0,
        "gain_exact_relative_error": 0,
        "nora_best_possible_relative_frobenius_error": float((amplitudes - amplitudes.mean()).square().sum() /
                                                            amplitudes.square().sum()),
        "scope": "Unconstrained learned output vector; all target column amplitudes positive; exact NoRA unit columns."})

    final_records = []
    total_started = time.perf_counter()
    for index, (rank, family, teacher_rank, coordinates) in enumerate(cells(args.ranks)):
        if args.max_cells is not None and index >= args.max_cells:
            break
        cell_id = f"r{rank}_{family}_q{teacher_rank}_{coordinates}"
        directory = args.output / cell_id
        directory.mkdir(exist_ok=True)
        data_cpu, metadata = make_problem(rank, family, teacher_rank, coordinates, args)
        if not (directory / "problem.pt").exists():
            torch.save(data_cpu, directory / "problem.pt")
        metadata["problem_sha256"] = sha256(directory / "problem.pt")
        write_json(directory / "problem.json", metadata)
        data = {name: value.to(device) for name, value in data_cpu.items()}
        for method in args.methods:
            method_dir = directory / method
            method_dir.mkdir(exist_ok=True)
            if (method_dir / "results.json").exists():
                final_records.extend(json.loads((method_dir / "results.json").read_text())["runs"])
                continue
            trials = []
            winner_layer, winner = None, None
            for lr, multiplier in candidates(method):
                layer, record = fit(data, method, rank, args.seeds[0], lr, multiplier, args)
                trials.append(record)
                if winner is None or record["validation"]["relative_to_frozen"] < winner["validation"]["relative_to_frozen"]:
                    winner, winner_layer = record, layer
            write_json(method_dir / "selection.json", {"trials": trials, "selected_lr": winner["learning_rate"],
                                                       "selected_magnitude_lr_multiplier": winner["magnitude_lr_multiplier"]})
            records = []
            for seed in args.seeds:
                if seed == args.seeds[0]:
                    layer, record = winner_layer, copy.deepcopy(winner)
                    record["reused_full_budget_tuning_checkpoint"] = True
                else:
                    layer, record = fit(data, method, rank, seed, winner["learning_rate"],
                                        winner["magnitude_lr_multiplier"], args)
                record.update(cell_id=cell_id, teacher=metadata, test=mse_metrics(layer, data, "test"))
                state = {name: parameter.detach().cpu() for name, parameter in layer.named_parameters() if parameter.requires_grad}
                checkpoint = method_dir / f"seed{seed}.pt"
                torch.save(state, checkpoint)
                record["checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
                records.append(record)
            write_json(method_dir / "results.json", {"runs": records})
            final_records.extend(records)
            print(json.dumps({"cell": cell_id, "method": method,
                              "selected_lr": winner["learning_rate"],
                              "magnitude_lr_multiplier": winner["magnitude_lr_multiplier"],
                              "test_relative_mse": [r["test"]["relative_to_frozen"] for r in records],
                              "elapsed_seconds": time.perf_counter() - total_started}), flush=True)
        del data
        torch.cuda.empty_cache()
        write_json(args.output / "results.json", {"provenance": provenance, "runs": final_records,
                                                  "elapsed_seconds": time.perf_counter() - total_started})
    print(f"COMPLETE {len(final_records)} final runs in {time.perf_counter() - total_started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
