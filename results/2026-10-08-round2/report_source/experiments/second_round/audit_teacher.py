"""Independent artifact checks and stratified summaries for teacher experiments."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from experiments.second_round.teacher import candidates, cells, new_layer, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/var/tmp/dora-bench/second_round/teacher"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.output
    result = json.loads((root / "results.json").read_text())
    config = result["provenance"]["config"]
    source_root = Path(__file__).resolve().parents[2]
    live_sources = {"teacher.py": Path(__file__).with_name("teacher.py"),
                    "second_round_adapters.py": source_root / "experiments/second_round_adapters.py",
                    "adapters.py": source_root / "experiments/adapters.py", "dora.py": source_root / "dora.py"}
    for name, expected in result["provenance"]["source_sha256"].items():
        assert sha(root / "executed_sources" / name) == expected, name
        assert sha(live_sources[name]) == expected, f"Live reconstruction source changed: {name}"
    expected_cells = list(cells(config["ranks"]))
    if config["max_cells"] is not None:
        expected_cells = expected_cells[:config["max_cells"]]
    assert len(result["runs"]) == len(expected_cells) * len(config["methods"]) * len(config["seeds"])
    groups = defaultdict(list)
    for record in result["runs"]:
        groups[record["cell_id"]].append(record)
    summaries = {}
    records_checked = 0
    max_relative_metric_difference = 0.0
    max_coordinate_output_difference = 0.0
    selection_trials_checked = 0
    minimum_normalized_raw_column_norm = float("inf")
    white_data = {}
    for rank, family, teacher_rank, coordinates in expected_cells:
        cell_id = f"r{rank}_{family}_q{teacher_rank}_{coordinates}"
        directory = root / cell_id
        metadata = json.loads((directory / "problem.json").read_text())
        assert sha(directory / "problem.pt") == metadata["problem_sha256"]
        data = torch.load(directory / "problem.pt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(tensor).all() for tensor in data.values())
        for split in ("train", "validation", "test"):
            output = F.linear(data[f"x_{split}"], data["teacher_weight"])
            diff = float((output - data[f"y_{split}"]).abs().max())
            max_coordinate_output_difference = max(max_coordinate_output_difference, diff)
            torch.testing.assert_close(output, data[f"y_{split}"], rtol=1e-4, atol=4e-6)
        pairing = (rank, family, teacher_rank)
        if coordinates == "white":
            white_data[pairing] = data
        else:
            white = white_data.pop(pairing)
            for key in ("canonical_base", "canonical_teacher", "y_train", "y_validation", "y_test"):
                torch.testing.assert_close(data[key], white[key], rtol=0, atol=0)
            for split in ("train", "validation", "test"):
                torch.testing.assert_close(data[f"x_{split}"] / data["feature_scale"], white[f"x_{split}"], rtol=2e-7, atol=2e-7)
        summaries[cell_id] = {"teacher": metadata, "methods": {}}
        for method in config["methods"]:
            method_dir = directory / method
            selection = json.loads((method_dir / "selection.json").read_text())
            assert len(selection["trials"]) == len(candidates(method)) == 4
            assert {(r["learning_rate"], r["magnitude_lr_multiplier"]) for r in selection["trials"]} == set(candidates(method))
            winner = min(selection["trials"], key=lambda r: r["validation"]["relative_to_frozen"])
            assert selection["selected_lr"] == winner["learning_rate"]
            assert selection["selected_magnitude_lr_multiplier"] == winner["magnitude_lr_multiplier"]
            selection_trials_checked += len(selection["trials"])
            records = [r for r in groups[cell_id] if r["method"] == method]
            assert sorted(r["seed"] for r in records) == sorted(config["seeds"])
            for record in selection["trials"] + records:
                assert record["steps"] == config["steps"]
                assert record["curves"][-1]["step"] == config["steps"]
                assert record["frozen_weight_unchanged"]
                assert all(g["weight_decay"] == 0 for g in record["optimizer_groups"])
            for record in records:
                assert record["learning_rate"] == winner["learning_rate"]
                assert record["magnitude_lr_multiplier"] == winner["magnitude_lr_multiplier"]
                if record["seed"] == config["seeds"][0]:
                    assert record["validation"] == winner["validation"]
                    assert record["curves"] == winner["curves"]
                checkpoint = Path(record["checkpoint"]["path"])
                assert sha(checkpoint) == record["checkpoint"]["sha256"]
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                layer = new_layer(data, method, rank, record["seed"])
                expected = {name for name, parameter in layer.named_parameters() if parameter.requires_grad}
                assert set(state) == expected
                assert all(torch.isfinite(tensor).all() for tensor in state.values())
                layer.load_state_dict(state, strict=False)
                if layer.use_nora:
                    minimum_normalized_raw_column_norm = min(minimum_normalized_raw_column_norm,
                                                             float(layer.lora_A.detach().norm(dim=0).min()))
                with torch.no_grad():
                    # Dense merged reference independently checks low-rank forward metrics.
                    output = F.linear(data["x_test"], layer.to_linear().weight)
                    mse = float(F.mse_loss(output, data["y_test"]))
                    frozen = float(F.mse_loss(F.linear(data["x_test"], data["base_weight"]), data["y_test"]))
                    relative = mse / frozen
                    delta = data["teacher_weight"] - data["base_weight"]
                    error = (layer.to_linear().weight - data["teacher_weight"]) * data["feature_scale"]
                    weight_ratio = float(error.square().sum() / (delta * data["feature_scale"]).square().sum())
                difference = abs(relative - record["test"]["relative_to_frozen"])
                max_relative_metric_difference = max(max_relative_metric_difference, difference)
                torch.testing.assert_close(torch.tensor(relative), torch.tensor(record["test"]["relative_to_frozen"]), rtol=1e-4, atol=2e-8)
                torch.testing.assert_close(torch.tensor(weight_ratio), torch.tensor(record["weight_metrics"]["weighted_weight_error_ratio"]), rtol=1e-4, atol=2e-8)
                if method == "lora":
                    assert weight_ratio + 1e-6 >= metadata["lora_population_rank_bound"]
                records_checked += 1
            values = [r["test"]["relative_to_frozen"] for r in records]
            summaries[cell_id]["methods"][method] = {
                "test_relative_mse_mean": statistics.mean(values),
                "test_relative_mse_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "test_relative_mse_per_seed": values,
                "selected_learning_rate": selection["selected_lr"],
                "selected_magnitude_lr_multiplier": selection["selected_magnitude_lr_multiplier"],
                "trainable_parameters": records[0]["trainable_parameters"],
                "fit_seconds_mean": statistics.mean(r["seconds"] for r in records),
                "weighted_weight_error_ratio_mean": statistics.mean(r["weight_metrics"]["weighted_weight_error_ratio"] for r in records),
                "row_norm_ratio_rmse_mean": statistics.mean(r["weight_metrics"]["row_norm_ratio_rmse"] for r in records),
                "mean_target_direction_error_degrees": statistics.mean(r["weight_metrics"]["mean_target_direction_error_degrees"] for r in records),
                "column_amplitude_relative_rmse_mean": statistics.mean(r["weight_metrics"]["column_amplitude_relative_rmse"] for r in records),
            }
    audit = {"passed": True, "records_checked": records_checked,
             "selection_trials_checked": selection_trials_checked,
             "source_snapshots_and_all_checkpoint_hashes_verified": True,
             "live_reconstruction_source_matches_executed_snapshots": True,
             "lora_weight_errors_respect_analytic_population_rank_bound": True,
             "coordinate_pairs_share_exact_canonical_weights_and_labels": True,
             "max_coordinate_output_absolute_difference": max_coordinate_output_difference,
             "max_test_relative_mse_cpu_dense_vs_gpu_factorized_absolute_difference": max_relative_metric_difference,
             "minimum_raw_A_column_norm_over_normalized_final_checkpoints": (
                 minimum_normalized_raw_column_norm if minimum_normalized_raw_column_norm < float("inf") else None),
             "normalization_eps": 1e-12,
             "total_actual_fits": selection_trials_checked + len(expected_cells) * len(config["methods"]) * (len(config["seeds"]) - 1),
             "runner_elapsed_seconds": result["elapsed_seconds"],
             "audit_source_sha256": sha(__file__), "results_sha256": sha(root / "results.json")}
    write_json(root / "stratified_summary.json", {"audit": audit, "cells": summaries})
    write_json(root / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
