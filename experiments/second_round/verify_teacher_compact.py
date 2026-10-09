"""Standalone CPU verifier for a compact teacher evidence directory.

PyTorch and NumPy are required. No repository imports, original paths, CUDA device,
or network access are used. Exact tensor hashes detect generator/library drift.
"""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_fingerprint(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}


def captured_generator(path):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_problem"]
    assert len(functions) == 1
    # Execute the exact captured generator function in a minimal CPU namespace.
    # Excluding top-level imports avoids depending on live experiment modules.
    namespace = {"torch": torch, "F": F, "math": math}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["make_problem"]


def dense_weight(base, state, method):
    expected = {"lora_A", "lora_B"}
    if method not in ("lora", "nora"):
        expected.add("m")
    if method == "dora_nora_gain":
        expected.add("log_gain")
    assert set(state) == expected
    assert all(tensor.device.type == "cpu" and torch.isfinite(tensor).all() for tensor in state.values())
    down = state["lora_A"]
    if method not in ("lora", "dora"):
        down = down / torch.linalg.vector_norm(down, dim=0, keepdim=True).clamp_min(1e-12)
    if method == "dora_nora_gain":
        down = down * torch.exp(state["log_gain"])[None, :]
    weight = base + state["lora_B"] @ down
    if "m" in state:
        denominator = torch.linalg.vector_norm(weight, dim=1, keepdim=True).clamp_min(torch.finfo(base.dtype).tiny)
        weight = state["m"] * weight / denominator
    return weight


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-4, abs_tol=2e-8), (actual, expected)


@torch.inference_mode()
def verify(root, output=None):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["format"] == "dora-teacher-compact-v1"
    for relative, expected in manifest["files_sha256"].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root) and file_hash(path) == expected, relative
    recorded = json.loads((root / "recorded_results.json").read_text())
    summary = json.loads((root / "stratified_summary.json").read_text())
    config = argparse.Namespace(**recorded["provenance"]["config"])
    generator = captured_generator(root / "executed_sources" / "teacher.py")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    by_cell = defaultdict(list)
    for run in recorded["runs"]:
        by_cell[run["cell_id"]].append(run)
    total_tensors, total_runs, total_candidates = 0, 0, 0
    max_relative_difference = 0.0
    reconstructed = {}
    for cell, entry in manifest["cells"].items():
        specification = entry["generator_arguments"]
        data, _ = generator(*specification, config)
        actual_fingerprints = {name: tensor_fingerprint(tensor) for name, tensor in data.items()}
        assert actual_fingerprints == entry["problem_tensor_fingerprints"], f"Exact problem regeneration failed: {cell}"
        total_tensors += len(data)
        baseline = float(F.mse_loss(data["x_test"] @ data["base_weight"].T, data["y_test"]))
        reconstructed[cell] = {"teacher": summary["cells"][cell]["teacher"], "methods": {}}
        for method in config.methods:
            selection = json.loads((root / "selections" / cell / f"{method}.json").read_text())
            assert len(selection["trials"]) == 4
            winner = min(selection["trials"], key=lambda row: row["validation"]["relative_to_frozen"])
            assert winner["learning_rate"] == selection["selected_lr"]
            assert winner["magnitude_lr_multiplier"] == selection["selected_magnitude_lr_multiplier"]
            assert all(row["steps"] == config.steps for row in selection["trials"])
            total_candidates += len(selection["trials"])
            runs = [run for run in by_cell[cell] if run["method"] == method]
            assert sorted(run["seed"] for run in runs) == sorted(config.seeds)
            values, weight_errors = [], []
            for run in runs:
                assert run["steps"] == config.steps
                assert run["learning_rate"] == winner["learning_rate"]
                assert run["magnitude_lr_multiplier"] == winner["magnitude_lr_multiplier"]
                checkpoint = root / "checkpoints" / cell / method / f"seed{run['seed']}.pt"
                assert file_hash(checkpoint) == run["checkpoint"]["sha256"]
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                weight = dense_weight(data["base_weight"], state, method)
                mse = float(F.mse_loss(data["x_test"] @ weight.T, data["y_test"]))
                relative = mse / baseline
                delta = (data["teacher_weight"] - data["base_weight"]) * data["feature_scale"]
                error = (weight - data["teacher_weight"]) * data["feature_scale"]
                weight_error = float(error.square().sum() / delta.square().sum())
                close(baseline, run["test"]["frozen_mse"])
                close(mse, run["test"]["mse"])
                close(relative, run["test"]["relative_to_frozen"])
                close(weight_error, run["weight_metrics"]["weighted_weight_error_ratio"])
                max_relative_difference = max(max_relative_difference, abs(relative - run["test"]["relative_to_frozen"]))
                values.append(relative)
                weight_errors.append(weight_error)
                total_runs += 1
            mean = statistics.mean(values)
            close(mean, summary["cells"][cell]["methods"][method]["test_relative_mse_mean"])
            reconstructed[cell]["methods"][method] = {
                "test_relative_mse_mean": mean,
                "test_relative_mse_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "test_relative_mse_per_seed": values,
                "weighted_weight_error_ratio_mean": statistics.mean(weight_errors),
                "selected_learning_rate": winner["learning_rate"],
                "selected_magnitude_lr_multiplier": winner["magnitude_lr_multiplier"],
            }
    assert total_runs == len(recorded["runs"]) == manifest["final_run_count"]
    result = {"passed": True, "format": manifest["format"], "tensor_count": total_tensors,
              "final_run_count": total_runs, "validation_candidate_count": total_candidates,
              "runtime_torch": torch.__version__, "recorded_torch": recorded["provenance"]["torch"],
              "all_problem_tensors_regenerated_with_exact_hashes": True,
              "independent_dense_weight_math": True, "cuda_used": False,
              "max_absolute_difference_from_recorded_gpu_test_relative_mse": max_relative_difference,
              "scope": "Recomputes selected-checkpoint test MSE and weighted weight error. Verifies saved validation selection, without retraining or independently recomputing training curves. CPU/GPU floating-point differences use relative tolerance 1e-4 and absolute tolerance 2e-8.",
              "cells": reconstructed}
    if output is not None:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = verify(options.root, options.output)
    print(json.dumps({key: value for key, value in result.items() if key != "cells"}, indent=2))
