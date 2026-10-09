"""Audit Aircraft artifacts and quantify matched-seed, paired-image differences.

This CPU-only analysis is independent of the training and adapter modules.
The class-stratified bootstrap keeps all 100 classes and their sample counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("lora", "dora", "nora", "dora_nora", "dora_nora_mlr", "dora_nora_gain")
CONTRASTS = (("dora", "lora"), ("nora", "lora"), ("dora_nora", "nora"),
             ("dora_nora_mlr", "dora_nora"), ("dora_nora_gain", "dora_nora"),
             ("dora_nora_mlr", "nora"), ("dora_nora_gain", "nora"),
             ("dora_nora", "lora"), ("dora_nora_mlr", "lora"), ("dora_nora_gain", "lora"))


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def record_directory(root, row):
    if row["stage"] == "pilot":
        return root / f"rank_{row['rank']}" / "pilots" / row["method"] / (
            f"lr_{row['adapter_learning_rate']:g}_m_{row['magnitude_lr_multiplier']:g}")
    if row["method"] == "baseline":
        return root / "baseline" / f"seed_{row['seed']}"
    return root / f"rank_{row['rank']}" / "final" / row["method"] / f"seed_{row['seed']}"


def validate(root, require_checkpoints=True):
    protocol = read(root / "protocol.json")
    manifest = read(root / "split_manifest.json")
    results = read(root / "results.json")
    seeds, ranks = protocol["seeds"], protocol["ranks"]
    assert len(seeds) == len(set(seeds)) == 3
    assert protocol["epochs"] == 20 and protocol["pilot_epochs"] == 10
    assert protocol["trials_per_method_per_rank"] == 4
    assert len(results) == len(seeds) * (1 + len(ranks) * len(METHODS))
    keys = {(row["method"], row["rank"], row["seed"]) for row in results}
    assert keys == {(method, rank, seed) for rank in ranks for seed in seeds for method in METHODS} | {
        ("baseline", 0, seed) for seed in seeds}
    assert sha256(root / "split_manifest.json") == protocol["provenance"]["split_manifest_sha256"]
    assert manifest["official_sizes"] == {"train": 3334, "val": 3333, "test": 3333}
    all_images = [row for values in manifest["splits"].values() for row in values]
    assert len(all_images) == len({row["image"] for row in all_images}) == len({row["sha256"] for row in all_images})
    for name, expected in protocol["provenance"]["source_sha256"].items():
        assert sha256(root / "source" / name) == expected
    records = list(results)
    for rank in ranks:
        pilots = read(root / f"rank_{rank}" / "pilot_results.json")
        selected = read(root / f"rank_{rank}" / "selected.json")
        assert len(pilots) == len(METHODS) * 4 and set(selected) == set(METHODS)
        for method in METHODS:
            trials = [row for row in pilots if row["method"] == method]
            assert len(trials) == 4
            expected = {(lr, multiplier) for lr in protocol["mlr_base_lr_grid"]
                        for multiplier in protocol["magnitude_multipliers"]} if method == "dora_nora_mlr" else {
                            (lr, 1.0) for lr in protocol["standard_and_gain_lr_grid"]}
            assert {(row["adapter_learning_rate"], row["magnitude_lr_multiplier"]) for row in trials} == expected
            assert all(row["seed"] == protocol["pilot_seed"] and row["rank"] == rank and
                       row["stage"] == "pilot" and row["test"] is None and not row["evaluate_test"] for row in trials)
            winner = max(trials, key=lambda row: (row["best_validation"]["macro_class_accuracy"],
                                                  -row["best_validation"]["cross_entropy"]))
            assert selected[method] == {"lr": winner["adapter_learning_rate"],
                                        "magnitude_multiplier": winner["magnitude_lr_multiplier"]}
            finals = [row for row in results if row["method"] == method and row["rank"] == rank]
            assert all(row["adapter_learning_rate"] == selected[method]["lr"] and
                       row["magnitude_lr_multiplier"] == selected[method]["magnitude_multiplier"] for row in finals)
        records.extend(pilots)
    labels = np.array([row["label"] for row in manifest["splits"]["test"]])
    counts = np.bincount(labels, minlength=100)
    assert (counts > 0).all()
    head_hashes, source_hashes, frozen_hashes = {}, set(), set()
    epoch_count, prediction_count, checkpoint_count = 0, 0, 0
    arrays = {}
    for row in records:
        directory = record_directory(root, row)
        assert read(directory / "result.json") == row
        assert row["provenance"] == protocol["provenance"]
        assert row["frozen_base_unchanged"] and row["frozen_state_before_sha256"] == row["frozen_state_after_sha256"]
        assert row["initial_backbone_sha256"] == row["frozen_state_before_sha256"]
        assert row["initial_trainable_sha256"] != row["final_trainable_sha256"]
        assert row["initial_output_max_absolute_error"] < 2e-5
        checkpoint = directory / "best_adapter_and_head.pt"
        if checkpoint.exists():
            assert sha256(checkpoint) == row["best_adapter_checkpoint_sha256"]
            checkpoint_count += 1
        else:
            assert not require_checkpoints, checkpoint
        source_hashes.add(row["source_state_sha256"])
        frozen_hashes.add(row["initial_backbone_sha256"])
        head_hashes.setdefault(row["seed"], set()).add(row["initial_head_sha256"])
        expected_epochs = protocol["pilot_epochs"] if row["stage"] == "pilot" else protocol["epochs"]
        assert row["epochs"] == expected_epochs
        history = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
        assert [entry["epoch"] for entry in history] == list(range(1, expected_epochs + 1))
        best = max(history, key=lambda item: (item["validation"]["macro_class_accuracy"],
                                              -item["validation"]["cross_entropy"]))
        assert row["best_epoch"] == best["epoch"] and row["best_validation"] == best["validation"]
        for entry in history:
            assert entry["train"]["count"] == manifest["effective_sizes"]["train"]
            assert entry["validation"]["count"] == manifest["effective_sizes"]["val"]
            assert math.isfinite(entry["train"]["label_smoothed_cross_entropy"])
            assert math.isfinite(entry["validation"]["cross_entropy"])
            assert entry["epoch_seconds"] >= entry["training_seconds"] > 0
        assert row["training_wall_seconds"] >= sum(entry["epoch_seconds"] for entry in history) - 1e-5
        assert row["peak_cuda_allocated_bytes"] > 0
        epoch_count += len(history)
        groups = row["optimizer_groups"]
        assert sum(group["parameter_count"] for group in groups) == row["trainable_parameters"]
        # ViT-B/16: twelve blocks, four targeted projections per block.
        expected_parameters = 768 * 100 + 100
        if row["method"] != "baseline":
            expected_parameters += 12 * 12288 * row["rank"]
        if row["method"].startswith("dora"):
            expected_parameters += 12 * 6912
        if row["method"] == "dora_nora_gain":
            expected_parameters += 12 * 5376
        assert row["trainable_parameters"] == expected_parameters
        grouped_names = [name for group in groups for name in group.get("param_names", ["head.weight", "head.bias"])]
        assert len(grouped_names) == len(set(grouped_names)) and set(grouped_names) == set(row["trainable_names"])
        for group in groups:
            name = group["group_name"]
            assert group["weight_decay"] == (0.01 if name in ("classifier", "adapter_factors") else 0)
            lr = row["head_learning_rate"] if name == "classifier" else row["adapter_learning_rate"]
            if name == "decoupled_magnitudes":
                lr *= row["magnitude_lr_multiplier"]
            assert group["lr"] == lr
        diagnostics = row["adapter_diagnostics"]
        assert len(diagnostics) == (0 if row["method"] == "baseline" else 48)
        for module in diagnostics:
            assert module["up_factor_l2"] > 0
            assert module.get("normalized_column_relative_error", 0) < 1e-5
            assert module.get("merged_magnitude_relative_error", 0) < 1e-4
            if "positive_gain_range" in module:
                assert 0 < module["positive_gain_range"][0] <= module["positive_gain_range"][1] < math.inf
        if row["stage"] == "pilot":
            assert not (directory / "test_predictions.json").exists()
            continue
        assert row["stage"] == "final" and row["evaluate_test"]
        prediction = read(directory / "test_predictions.json")
        assert all(len(values) == len(labels) for values in prediction.values())
        assert np.array_equal(prediction["labels"], labels)
        guesses = np.array(prediction["predictions"])
        assert ((guesses >= 0) & (guesses < 100)).all()
        correct = (guesses == labels).astype(float)
        nll, confidence = np.array(prediction["true_class_nll"]), np.array(prediction["confidence"])
        top5 = np.array(prediction["top5_correct"], dtype=float)
        assert np.isfinite(nll).all() and (nll >= 0).all()
        assert np.isfinite(confidence).all() and ((confidence >= 0) & (confidence <= 1)).all()
        true_probability = np.exp(-nll)
        assert (true_probability <= confidence + 2e-6).all()
        assert np.allclose(true_probability[correct.astype(bool)], confidence[correct.astype(bool)], atol=2e-6, rtol=2e-6)
        assert np.isin(top5, [0, 1]).all() and (top5 >= correct).all()
        metrics = {"count": len(labels), "correct": int(correct.sum()), "accuracy": correct.mean(),
                   "macro_class_accuracy": (np.bincount(labels, weights=correct, minlength=100) / counts).mean(),
                   "top5_accuracy": top5.mean(), "cross_entropy": nll.mean()}
        for name, expected in metrics.items():
            assert abs(expected - row["test"][name]) < (3e-7 if name == "cross_entropy" else 1e-7), (directory, name)
        arrays[(row["method"], row["rank"], row["seed"])] = correct
        prediction_count += len(labels)
    assert len(source_hashes) == len(frozen_hashes) == 1
    assert all(len(values) == 1 for values in head_hashes.values())
    audit = {"status": "pass", "final_records": len(results), "pilot_records": len(records) - len(results),
             "epochs_checked": epoch_count, "raw_test_predictions_recomputed": prediction_count,
             "adapter_checkpoint_hashes_verified": checkpoint_count,
             "adapter_checkpoint_payloads_omitted": len(records) - checkpoint_count,
             "source_snapshots_verified": len(protocol["provenance"]["source_sha256"]),
             "seeded_heads_paired": True, "frozen_pretrained_backbone_paired": True,
             "scope": "Metrics recomputed from saved predictions; complete trial and refit budgets, validation selection, source and checkpoint hashes, frozen states, paired initializations, optimizer groups, adapter diagnostics, disjoint image hashes. Does not independently evaluate image labels or raw logits."}
    return protocol, results, labels, arrays, audit


def analyze(root, iterations=10000, seed=20261008, require_checkpoints=True):
    protocol, results, labels, arrays, audit = validate(root, require_checkpoints)
    generator = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == label) for label in range(100)]
    macro_weights = 1 / (100 * np.bincount(labels)[labels])
    methods = ("baseline", *METHODS)
    comparisons, summary = [], []
    for rank in protocol["ranks"]:
        data = np.array([[arrays[(method, 0 if method == "baseline" else rank, run_seed)]
                          for run_seed in protocol["seeds"]] for method in methods])
        sampled_accuracy = np.empty((len(methods), iterations))
        sampled_macro = np.empty_like(sampled_accuracy)
        for start in range(0, iterations, 64):
            size = min(64, iterations - start)
            seed_draws = generator.integers(len(protocol["seeds"]), size=(size, len(protocol["seeds"])))
            image_draws = np.empty((size, len(labels)), dtype=np.int64)
            for indices in classes:
                image_draws[:, indices] = generator.choice(indices, size=(size, len(indices)))
            for index in range(len(methods)):
                sampled = data[index][seed_draws[:, :, None], image_draws[:, None, :]].mean(axis=1)
                sampled_accuracy[index, start:start + size] = sampled.mean(axis=1)
                sampled_macro[index, start:start + size] = sampled @ macro_weights
        for index, method in enumerate(methods):
            selected = [row for row in results if row["method"] == method and row["rank"] == (0 if method == "baseline" else rank)]
            metrics = {metric: {"mean": float(np.mean([row["test"][metric] for row in selected])),
                                "sample_sd": float(np.std([row["test"][metric] for row in selected], ddof=1)),
                                "by_seed": {str(row["seed"]): row["test"][metric] for row in selected}}
                       for metric in ("accuracy", "macro_class_accuracy", "top5_accuracy", "cross_entropy")}
            metrics["accuracy"]["percentile_95_interval"] = np.quantile(sampled_accuracy[index], [.025, .975]).tolist()
            metrics["macro_class_accuracy"]["percentile_95_interval"] = np.quantile(sampled_macro[index], [.025, .975]).tolist()
            summary.append({"method": method, "rank": rank, "metrics": metrics,
                            "trainable_parameters": selected[0]["trainable_parameters"],
                            "adapter_learning_rate": selected[0]["adapter_learning_rate"],
                            "magnitude_lr_multiplier": selected[0]["magnitude_lr_multiplier"],
                            "training_wall_seconds_mean": float(np.mean([row["training_wall_seconds"] for row in selected])),
                            "peak_cuda_allocated_bytes_max": max(row["peak_cuda_allocated_bytes"] for row in selected),
                            "best_epochs": [row["best_epoch"] for row in selected]})
        for candidate, reference in CONTRASTS:
            first, second = methods.index(candidate), methods.index(reference)
            difference = data[first] - data[second]
            metrics = {}
            for metric, samples, seed_values in (
                ("accuracy", sampled_accuracy[first] - sampled_accuracy[second], difference.mean(axis=1)),
                ("macro_class_accuracy", sampled_macro[first] - sampled_macro[second], difference @ macro_weights),
            ):
                # Exact discrete ties must not become positive/negative from roundoff.
                samples[np.abs(samples) < 1e-12] = 0
                seed_values[np.abs(seed_values) < 1e-12] = 0
                metrics[metric] = {"mean_difference": float(seed_values.mean()),
                                   "paired_seed_differences": seed_values.tolist(),
                                   "percentile_95_interval": np.quantile(samples, [.025, .975]).tolist(),
                                   "bootstrap_fraction_positive": float((samples > 0).mean())}
            comparisons.append({"rank": rank, "candidate": candidate, "reference": reference, "metrics": metrics})
    prediction_hashes = {str((record_directory(root, row) / "test_predictions.json").relative_to(root)):
                         sha256(record_directory(root, row) / "test_predictions.json") for row in results}
    output = {"source_sha256": sha256(Path(__file__)), "input_results_sha256": sha256(root / "results.json"),
              "input_predictions_sha256": prediction_hashes,
              "bootstrap_iterations": iterations, "bootstrap_rng_seed": seed, "audit": audit,
              "bootstrap_protocol": "Crossed paired bootstrap: resample three matched training seeds and independently resample test images within each of the 100 classes, preserving each class count. Common draws across methods and metrics; percentile 95% intervals. Macro accuracy weights classes equally; accuracy weights images equally.",
              "limitations": "Exploratory intervals without multiple-comparison correction. Three seeds give limited seed-variance evidence. Image independence is assumed within class; aircraft registration and photographer grouping are unavailable. Fractions positive are descriptive, not calibrated p-values. Validation search uses 10 epochs and final refits 20 epochs; no test-driven choices. Timed training includes per-epoch validation and saving improved checkpoints, excluding model setup and final test evaluation.",
              "summary": summary, "comparisons": comparisons}
    (root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--skip-checkpoints", action="store_true", help="Allow a portable archive that intentionally omits checkpoint payloads")
    args = parser.parse_args()
    output = analyze(args.root, args.iterations, args.seed, require_checkpoints=not args.skip_checkpoints)
    print(json.dumps(output["audit"], indent=2))
    for row in output["summary"]:
        print(row["method"], row["rank"], row["metrics"]["accuracy"]["mean"], row["metrics"]["macro_class_accuracy"]["mean"])
