"""Audit and report original DoRA on the previously published task protocols.

Live collection: python -m experiments.original_report --raw-root PATH --output PATH
Portable rebuild: python original_report.py --rebuild EXTRACTED/report_inputs.json --output PATH
Collection requires every requested task and its independent audit. Checkpoints
are verified by task audits, then intentionally omitted from the portable archive.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import statistics
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = [42, 43, 44]
ORIGINAL = "dora_original"
METHODS = ["baseline", ORIGINAL, "lora", "dora", "nora", "dora_nora", "dora_nora_mlr", "dora_nora_gain"]
LABELS = {"baseline": "Task baseline", ORIGINAL: "Original DoRA", "lora": "LoRA", "dora": "Revised DoRA",
          "nora": "NoRA", "dora_nora": "DoRA+NoRA", "dora_nora_mlr": "+ slow magnitudes", "dora_nora_gain": "+ input gains"}
COLORS = {ORIGINAL: "#D55E00", "lora": "#0072B2", "dora": "#E69F00", "nora": "#009E73"}
ROUND1 = [("flowers", "Flowers102", "Top-1 accuracy (%)", "vision"),
          ("scifact", "SciFact trained directly", "nDCG@10 ×100", "retrieval"),
          ("extraction", "ViGGO", "JSON exact match (%)", "extraction"),
          ("sdxl", "SDXL", "Denoising MSE ↓", "sdxl")]
ROUND2 = [("aircraft", "FGVC-Aircraft", "Macro accuracy (%)", "aircraft"),
          ("nfcorpus", "NFCorpus", "nDCG@10 ×100", "nfcorpus"),
          ("scifact_transfer", "SciFact transfer", "nDCG@10 ×100", "scifact"),
          ("cogs", "COGS OOD", "Atom-set exact match (%)", "cogs")]
PORTABLE_SUFFIXES = {".json", ".jsonl", ".py", ".md", ".txt", ".csv", ".tsv", ".png", ".jpg", ".npz", ".log"}
LIMITATIONS = (
    "The original implementation keeps dim=0 column magnitudes, the full norm gradient, its original normal up-factor/zero down-factor initialization, and base-dtype magnitudes. "
    "Revised DoRA uses dim=1 row magnitudes, a detached norm, FP32 magnitudes, and the task adapters' shared Kaiming-uniform down-factor/zero up-factor initialization. It computes norms outside autocast in FP32 with a base-dtype tiny-value clamp, and applies separate base and low-rank projections before output scaling. This compares complete implementations, not an isolated axis, detachment, precision, initialization, or projection ablation. "
    "Original factors are drawn on CPU, then moved without a dtype cast. Frozen base weights/biases retain their dtype; missing bias stays absent. Outside autocast only, a mismatched dense projection casts input/bias to the effective weight dtype and the result back to input dtype. No explicit norm promotion, epsilon, detached gradient, or autocast override is added. "
    "Previously published comparator runs are reused, not retrained; task baselines denote pretrained/head-only models, not original DoRA. Equal rank does not imply equal parameter count, initialization, numerical paths, or optimal hyperparameters. "
    "Intervals are exploratory matched-seed/example bootstrap percentiles with only three training seeds and no multiple-comparison correction. Seeds pair data/checkpoint controls, not identical factor draws. "
    "Timing compares recorded fixed budgets in separate execution sessions, not time to equal quality; cold kernels and checkpoint/validation overhead can differ. SDXL uses one subject and one held-out image; denoising loss is not image quality. Teacher cells are a separate synthetic diagnostic, not independent downstream evidence."
)


class NotReady(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if not path.exists():
        raise NotReady(f"Missing required artifact: {path}")
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def close(actual, expected, tolerance=2e-7):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0), (actual, expected)


def distribution(values):
    assert values and all(math.isfinite(value) for value in values)
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "values": values}


def summarize(records):
    result = {}
    for method in METHODS:
        rows = sorted((row for row in records if row["method"] == method), key=lambda row: row["seed"] or 0)
        if not rows:
            continue
        seeds = [row["seed"] for row in rows]
        assert len(seeds) == len(set(seeds)), (method, seeds)
        if method != "baseline":
            assert seeds == SEEDS, (method, seeds)
        item = {**distribution([row["value"] for row in rows]), "seeds": seeds,
                "trainable_parameters": sorted({row["trainable_parameters"] for row in rows})}
        for name in ("training_seconds", "peak_memory_bytes"):
            values = [row[name] for row in rows if row.get(name) is not None]
            item[name] = distribution(values) if values else None
        result[method] = item
    assert ORIGINAL in result
    return result


class Evidence:
    def __init__(self, raw, repo):
        self.raw, self.repo = raw.resolve(), repo.resolve()
        self.files = {}

    def add(self, path):
        path = Path(path).resolve()
        if path.suffix not in PORTABLE_SUFFIXES:
            return
        if path.is_relative_to(self.raw):
            logical = "original_bench/" + str(path.relative_to(self.raw))
        elif path.is_relative_to(self.repo / "results/2026-10-08"):
            logical = "comparators/round1/" + str(path.relative_to(self.repo / "results/2026-10-08"))
        elif path.is_relative_to(self.repo / "results/2026-10-08-round2"):
            logical = "comparators/round2/" + str(path.relative_to(self.repo / "results/2026-10-08-round2"))
        elif path.is_relative_to(Path("/var/tmp/dora-bench")):
            logical = "reference_raw/" + str(path.relative_to("/var/tmp/dora-bench"))
        elif path.is_relative_to(self.repo):
            logical = "report_source/" + str(path.relative_to(self.repo))
        else:
            raise ValueError(f"Unmapped evidence path: {path}")
        assert path.is_file()
        previous = self.files.get(logical)
        assert previous is None or previous == path
        self.files[logical] = path

    def json(self, path):
        value = read(path)
        self.add(path)
        return value

    def lines(self, path):
        self.add(path)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def check_sources(evidence, root, provenance):
    mapping = provenance.get("source_sha256", {})
    assert mapping, root
    for name, expected in mapping.items():
        candidates = [root / "source" / name, root / "executed_sources" / name,
                      root / "executed_sources" / Path(name).name]
        path = next((p for p in candidates if p.exists() and sha(p) == expected), None)
        assert path is not None, (root, name, expected)
        evidence.add(path)


def require_audit(evidence, root):
    candidates = [root / "task_audit.json", root / "analysis.json", root / "audit.json"]
    path = next((path for path in candidates if path.exists()), None)
    if path is None:
        raise NotReady(f"Required task audit not yet available: {root}")
    value = evidence.json(path)
    passed = value.get("passed") is True or value.get("status") == "pass" or value.get("audit", {}).get("status") == "pass" or value.get("audit", {}).get("passed") is True
    assert passed, (path, value)
    for key in ("input_results_sha256", "results_sha256"):
        if key in value:
            assert sha(root / "results.json") == value[key], path
    return {"artifact": str(path.relative_to(evidence.raw)), "sha256": sha(path), "details": value}


def check_configuration(actual, reference, ignored=()):
    ignored = {"output", "methods", "baseline_from", "reference_root", "reuse_baselines", *ignored}
    checked = []
    for name, expected in reference.items():
        if name not in ignored:
            assert name in actual and actual[name] == expected, (name, actual.get(name), expected)
            checked.append(name)
    return checked


def comparator(evidence, round_name, task):
    folder = "2026-10-08" if round_name == "round1" else "2026-10-08-round2"
    return evidence.json(evidence.repo / "results" / folder / "tasks" / f"{task}.json")


def comparator_records(task, rank, round_name):
    source = task["runs"] if round_name == "round1" else task["ranks"][str(rank)]["runs"]
    rows = []
    for row in source:
        rows.append({**row, "rank": rank, "training_seconds": row.get("training_seconds", row.get("training_wall_seconds"))})
    for method, recorded in (task["summary"] if round_name == "round1" else task["ranks"][str(rank)]["summary"]).items():
        values = [row["value"] for row in rows if row["method"] == method]
        close(statistics.mean(values), recorded["mean"], 1e-10)
        if len(values) > 1:
            close(statistics.stdev(values), recorded.get("sample_sd", recorded.get("sample_std")), 1e-10)
    return rows


def vision_rows(evidence, root, rank, flowers=False):
    results = evidence.json(root / "results.json")
    original = [row for row in results if row["method"] == ORIGINAL]
    if len(original) != 3:
        raise NotReady(f"{root.name}: {len(original)}/3 final seeds")
    manifest = evidence.json(root / "split_manifest.json")
    classes = 102 if flowers else 100
    rows = []
    for row in original:
        directory = root / f"seed_{row['seed']}" / ORIGINAL if flowers else root / f"rank_{rank}" / "final" / ORIGINAL / f"seed_{row['seed']}"
        predictions = evidence.json(directory / "test_predictions.json")
        labels, guesses = np.array(predictions["labels"]), np.array(predictions["predictions"])
        correct = labels == guesses
        assert len(labels) == row["test"]["count"] == (6149 if flowers else 3333)
        assert int(correct.sum()) == row["test"]["correct"]
        close(float(correct.mean()), row["test"]["accuracy"])
        counts = np.bincount(labels, minlength=classes)
        macro = (np.bincount(labels[correct], minlength=classes) / counts).mean()
        close(float(macro), row["test"]["macro_class_accuracy"])
        assert row["frozen_base_unchanged"]
        rows.append({"method": ORIGINAL, "seed": row["seed"], "rank": rank,
                     "value": 100 * row["test"]["accuracy" if flowers else "macro_class_accuracy"],
                     "test": row["test"], "trainable_parameters": row["trainable_parameters"],
                     "training_seconds": row["training_wall_seconds"], "peak_memory_bytes": row["peak_cuda_allocated_bytes"],
                     "learning_rate": row["adapter_learning_rate"], "selected_epoch": row["best_epoch"]})
    return rows


def retrieval_rows(evidence, root, rank, dataset=None):
    data = evidence.json(root / "results.json")
    selected = [row for row in data["runs"] if row["method"] == ORIGINAL and row.get("rank", 8) == rank]
    if len(selected) != 3:
        raise NotReady(f"{root.name} rank {rank}: {len(selected)}/3 final seeds")
    rows = []
    for row in selected:
        test = row["evaluation"][dataset]["test"] if dataset else row["evaluation"]["test"]
        directory = root / (Path(row["artifact_directory"]).name if "artifact_directory" in row
                            else f"{ORIGINAL}_seed{row['seed']}")
        prediction = directory / dataset / "test_per_query.json" if dataset else directory / "test_per_query.json"
        per_query = evidence.json(prediction)
        assert len(per_query) == test["query_count"]
        for metric in ("ndcg_at_10", "recall_at_10", "mrr_at_10"):
            close(statistics.mean(p[metric] for p in per_query), test[metric], 1e-12)
        rows.append({"method": ORIGINAL, "seed": row["seed"], "rank": rank, "value": 100 * test["ndcg_at_10"],
                     "test": test, "trainable_parameters": row["trainable_parameters"],
                     "training_seconds": row["training"]["seconds"], "peak_memory_bytes": row["training"]["peak_allocated_bytes"],
                     "learning_rate": row["learning_rate"]})
    return rows


def extraction_predictions(evidence, path, gold, recorded):
    rows = evidence.lines(path)
    assert [row["id"] for row in rows] == [row["id"] for row in gold]
    correct = []
    for row, target in zip(rows, gold):
        assert row["gold"] == target["target"] and row["utterance"] == target["utterance"]
        try:
            parsed = json.loads(row["text"].strip())
        except (ValueError, TypeError):
            parsed = None
        assert parsed == row["parsed"]
        exact = parsed == target["target"]
        assert exact == row["exact_match"]
        correct.append(float(exact))
    assert len(correct) == recorded["count"] == 256
    close(statistics.mean(correct), recorded["exact_match"], 1e-12)
    return correct


def language_rows(evidence, root, previous, cogs=False):
    data = evidence.json(root / "results.json")
    selected = data["results"]
    if len(selected) != 3:
        raise NotReady(f"{root.name}: {len(selected)}/3 final seeds")
    manifest = evidence.json(root / "manifest.json")
    old_manifest = previous["provenance"]["lora"] if cogs else previous["provenance"]
    check_configuration(manifest["configuration"], old_manifest["configuration"])
    for name in ("model", "model_revision", "dataset"):
        assert manifest[name] == old_manifest[name], (root, name)
    if not cogs:
        assert manifest["system_prompt_sha256"] == old_manifest["system_prompt_sha256"]
    check_sources(evidence, root, manifest)
    splits = evidence.json(root / "split_manifest.json")
    assert sha(root / "split_manifest.json") == manifest["dataset"]["split_manifest_sha256"]
    rows, arrays = [], {}
    if cogs:
        from experiments.second_round.analyze_cogs import check_predictions
        evidence.add(evidence.repo / "experiments/second_round/analyze_cogs.py")
    for row in selected:
        assert row["method"] == ORIGINAL and row["rank"] == 8 and row["frozen_weights_unchanged"]
        directory = root / f"{ORIGINAL}_seed{row['seed']}"
        if cogs:
            scores = {}
            for split, metric in (("test", "iid"), ("gen", "ood")):
                path = directory / f"{metric}_predictions.jsonl"
                evidence.add(path)
                mapping = check_predictions(path, {item["id"]: item for item in splits[split]}, row[metric])
                scores[metric] = [mapping[item["id"]] for item in splits[split]]
            value = 100 * row["ood"]["atom_exact"]
            arrays[row["seed"]] = scores["ood"]
        else:
            arrays[row["seed"]] = extraction_predictions(evidence, directory / "predictions.jsonl", splits["test"], row["test"])
            value = 100 * row["test"]["exact_match"]
        rows.append({"method": ORIGINAL, "seed": row["seed"], "rank": 8, "value": value,
                     "test": row["ood"] if cogs else row["test"], "iid": row.get("iid"),
                     "trainable_parameters": row["trainable_parameters"], "learning_rate": row["lr"],
                     "selected_step": row["selected_step"], "training_seconds": row["train_and_validation_seconds"],
                     "peak_memory_bytes": row["peak_allocated_bytes"]})
    return rows, arrays, splits


def language_comparisons(evidence, root, previous, arrays, splits, cogs=False):
    methods = [method for method in METHODS if method not in ("baseline", ORIGINAL)
               and any(row["method"] == method for row in (previous["ranks"]["8"]["runs"] if cogs else previous["runs"]))]
    gold = splits["gen" if cogs else "test"]
    strata = np.array([row["category"] for row in gold]) if cogs else np.zeros(len(gold), dtype=int)
    original = np.array([arrays[seed] for seed in SEEDS])
    comparisons = []
    for method in methods:
        reference = []
        old_rows = previous["ranks"]["8"]["runs"] if cogs else previous["runs"]
        for seed in SEEDS:
            recorded = next(row for row in old_rows if row["method"] == method and row["seed"] == seed)
            if cogs:
                from experiments.second_round.analyze_cogs import check_predictions
                path = Path("/var/tmp/dora-bench/round2/cogs") / method / f"{method}_seed{seed}" / "ood_predictions.jsonl"
                evidence.add(path)
                scores = check_predictions(path, {row["id"]: row for row in gold}, recorded["test"])
                reference.append([scores[row["id"]] for row in gold])
            else:
                path = Path(previous["raw_directory"]) / f"{method}_seed{seed}" / "predictions.jsonl"
                reference.append(extraction_predictions(evidence, path, gold, recorded["test"]))
        delta = original - np.array(reference)
        generator = np.random.default_rng(20261009)
        sampled = np.empty(10000)
        groups = [np.flatnonzero(strata == category) for category in np.unique(strata)]
        for start in range(0, len(sampled), 100):
            size = min(100, len(sampled) - start)
            seed_draw = generator.integers(3, size=(size, 3))
            images = np.empty((size, len(gold)), dtype=int)
            for indices in groups:
                images[:, indices] = generator.choice(indices, size=(size, len(indices)))
            sampled[start:start + size] = delta[seed_draw[:, :, None], images[:, None, :]].mean(axis=(1, 2))
        comparisons.append({"candidate": ORIGINAL, "reference": method,
                            "mean_difference": float(delta.mean() * 100),
                            "paired_seed_differences": (delta.mean(axis=1) * 100).tolist(),
                            "percentile_95_interval": (np.quantile(sampled, [.025, .975]) * 100).tolist(),
                            "units": "percentage points", "bootstrap_replicates": 10000,
                            "protocol": "Crossed paired resampling of three seeds and examples; preserve COGS categories." if cogs else "Crossed paired resampling of three seeds and examples."})
    return comparisons


def sdxl_rows(evidence, previous):
    rows, audits = [], []
    old = previous["provenance"]
    selection_root = evidence.raw / "sdxl_seed42"
    selected_lrs = evidence.json(selection_root / "selected_lrs.json")
    assert selected_lrs["selection_seed"] == 42
    selection_path = selection_root / ORIGINAL / "selection.json"
    assert sha(selection_path) == selected_lrs["selection_file_sha256"][ORIGINAL]
    evidence.add(selection_path)
    for seed in SEEDS:
        root = evidence.raw / f"sdxl_seed{seed}"
        data = evidence.json(root / "results.json")
        provenance = evidence.json(root / "provenance.json")
        for name in ("model", "model_revision", "weight_variant", "dataset", "dataset_revision"):
            assert provenance[name] == old[name], name
        check_configuration(provenance["config"], old["config"], ignored=("device", "seed", "skip_baseline", "conditioning_cache"))
        check_configuration(provenance["controls"], old["controls"], ignored=("seed", "train_precision"))
        assert provenance["controls"]["seed"] == seed
        check_sources(evidence, root, provenance)
        selected = [row for row in data["results"] if row["method"] == ORIGINAL]
        assert len(selected) == 1
        row = selected[0]
        probes = row["test"]["rows"]
        assert {(p["timestep"], p["noise_seed"]) for p in probes} == {
            (t, n) for t in (100, 300, 500, 700, 900) for n in (1001, 1002, 1003, 1004)}
        assert len(probes) == 20
        close(statistics.mean(p["mse"] for p in probes), row["test"]["mean_mse"], 1e-12)
        close(statistics.pstdev(p["mse"] for p in probes), row["test"]["std_across_noise_timestep"], 1e-12)
        assert row["frozen_sha256_before"] == row["frozen_sha256_after"]
        trials = row["trials"]
        if seed == 42:
            assert sorted(trial["lr"] for trial in trials) == [1e-5, 1e-4]
            assert row["selected_lr"] == min(trials, key=lambda trial: trial["validation"]["mean_mse"])["lr"]
        else:
            assert not trials
            assert Path(provenance["config"]["selected_lrs_json"]) == selection_root / "selected_lrs.json"
        assert row["selected_lr"] == selected_lrs["selected_lrs"][ORIGINAL]
        for sample in row["samples"]:
            path = root / ORIGINAL / "samples" / Path(sample["path"] if "path" in sample else sample["image"]).name
            if not path.exists():
                path = Path(sample.get("path", sample.get("image", "")))
            assert sha(path) == sample["sha256"]
            evidence.add(path)
        image_metrics = dict(row["image_metrics"])
        diagnostic_path = root / "class_clip_metrics.json"
        if diagnostic_path.exists():
            diagnostic = evidence.json(diagnostic_path)
            assert diagnostic["post_hoc"] and not diagnostic["used_for_model_selection"]
            item = next(item for item in diagnostic["results"] if item["method"] == ORIGINAL and item["seed"] == seed)
            assert [item["image_sha256"] for item in item["per_prompt"]] == [sample["sha256"] for sample in row["samples"]]
            close(statistics.mean(item["cosine"] for item in item["per_prompt"]), item["class_only_clip_mean"], 1e-10)
            image_metrics["class_only_clip_mean_post_hoc"] = item["class_only_clip_mean"]
        rows.append({"method": ORIGINAL, "rank": 8, "seed": seed, "value": row["test"]["mean_mse"],
                     "test": row["test"], "image_metrics": image_metrics,
                     "trainable_parameters": row["checkpoint"]["trainable_parameters"],
                     "training_seconds": row["training"]["seconds"], "peak_memory_bytes": row["training"]["peak_allocated_bytes"],
                     "learning_rate": row["selected_lr"]})
        audits.append(require_audit(evidence, root))
    return rows, audits


def teacher(evidence):
    root = evidence.raw / "teacher"
    data = evidence.json(root / "results.json")
    provenance = evidence.json(root / "provenance.json")
    check_sources(evidence, root, provenance)
    records = data.get("results", data.get("runs", []))
    if len(records) != 84:
        raise NotReady(f"Teacher: {len(records)}/84 final records")
    previous = evidence.json(evidence.repo / "results/2026-10-08-round2/teacher_summary.json")
    cells = {}
    for key, old in previous["cells"].items():
        selected = sorted((row for row in records if row["cell_id"] == key), key=lambda row: row["seed"])
        assert [row["seed"] for row in selected] == SEEDS
        assert all(row["method"] == ORIGINAL for row in selected)
        for row in selected:
            assert row["teacher"] == old["teacher"], key
        values = [row["test"]["relative_to_frozen"] for row in selected]
        methods = {method: {"mean": row["test_relative_mse_mean"], "sample_sd": row["test_relative_mse_sample_std"],
                            "values": row["test_relative_mse_per_seed"], "trainable_parameters": row["trainable_parameters"]}
                   for method, row in old["methods"].items()}
        for item in methods.values():
            close(statistics.mean(item["values"]), item["mean"], 1e-12)
            close(statistics.stdev(item["values"]), item["sample_sd"], 1e-12)
        methods[ORIGINAL] = {**distribution(values), "trainable_parameters": selected[0]["trainable_parameters"]}
        cells[key] = {"teacher": old["teacher"], "methods": methods}
    assert len(cells) == 28
    return {"cells": cells, "audit": require_audit(evidence, root),
            "scope": "Separate 28-cell constructed-teacher diagnostic. Initialization seeds repeat one problem per cell; no cross-task aggregate or independent-cell significance claim."}


def harvest_audit_inputs(evidence, value, default_root):
    """Preserve input files explicitly hash-bound by independent task audits."""
    roots = value.get("artifact_roots", {})
    mapping = value.get("input_sha256", {})
    for name, expected in mapping.items():
        if isinstance(expected, dict):
            root = Path(roots.get(name, str(default_root)))
            for relative, digest in expected.items():
                path = Path(relative) if Path(relative).is_absolute() else root / relative
                if path.suffix in (".pt", ".safetensors", ".bin"):
                    continue
                assert sha(path) == digest, path
                evidence.add(path)
        elif isinstance(expected, str):
            path = Path(name) if Path(name).is_absolute() else default_root / name
            if path.suffix in (".pt", ".safetensors", ".bin"):
                continue
            assert sha(path) == expected, path
            evidence.add(path)


def collect(raw, repo):
    evidence = Evidence(raw, repo)
    tasks = {}
    for key, title, metric, previous_key in ROUND1:
        previous = comparator(evidence, "round1", previous_key)
        old_rows = comparator_records(previous, 8, "round1")
        comparisons = []
        if key == "flowers":
            root = raw / key
            new_rows = vision_rows(evidence, root, 8, flowers=True)
            audit = require_audit(evidence, root)
            comparisons = audit["details"].get("comparisons", [])
            harvest_audit_inputs(evidence, audit["details"], root)
        elif key == "scifact":
            root = raw / key
            new_rows = retrieval_rows(evidence, root, 8)
            audit = {"external_analysis": "retrieval_analysis.json"}
        elif key == "extraction":
            root = raw / key
            new_rows, arrays, splits = language_rows(evidence, root, previous)
            audit = require_audit(evidence, root)
            comparisons = language_comparisons(evidence, root, previous, arrays, splits)
        else:
            new_rows, audit = sdxl_rows(evidence, previous)
        records = [*old_rows, *new_rows]
        tasks[key] = {"title": title, "metric": metric, "round": 1, "rank": 8, "records": records,
                      "summary": summarize(records), "audit": audit, "comparisons": comparisons}
    for key, title, metric, previous_key in ROUND2:
        previous = comparator(evidence, "round2", previous_key)
        for rank in ((8,) if key == "cogs" else (2, 8)):
            old_rows = comparator_records(previous, rank, "round2")
            comparisons = []
            if key == "aircraft":
                root = raw / f"aircraft_rank{rank}"
                new_rows = vision_rows(evidence, root, rank)
                audit = require_audit(evidence, root)
                comparisons = audit["details"].get("comparisons", [])
                harvest_audit_inputs(evidence, audit["details"], root)
            elif key in ("nfcorpus", "scifact_transfer"):
                root = raw / "nfcorpus"
                new_rows = retrieval_rows(evidence, root, rank, "nfcorpus" if key == "nfcorpus" else "scifact")
                audit = {"external_analysis": "retrieval_analysis.json"}
            else:
                root = raw / key
                new_rows, arrays, splits = language_rows(evidence, root, previous, cogs=True)
                audit = require_audit(evidence, root)
                comparisons = language_comparisons(evidence, root, previous, arrays, splits, cogs=True)
            records = [*old_rows, *new_rows]
            tasks[f"{key}_r{rank}"] = {"title": title, "metric": metric, "round": 2, "rank": rank,
                                      "records": records, "summary": summarize(records), "audit": audit, "comparisons": comparisons}
    retrieval_analysis = evidence.json(raw / "retrieval_analysis.json")
    assert retrieval_analysis.get("passed") is True or retrieval_analysis.get("status") == "pass" or retrieval_analysis.get("audit", {}).get("status") == "pass"
    harvest_audit_inputs(evidence, retrieval_analysis, raw)
    diagnostic = teacher(evidence)
    harvest_audit_inputs(evidence, diagnostic["audit"]["details"], raw / "teacher")
    for task in tasks.values():
        audits = task["audit"] if isinstance(task["audit"], list) else [task["audit"]]
        for audit in audits:
            if "details" in audit:
                artifact = raw / audit["artifact"]
                harvest_audit_inputs(evidence, audit["details"], artifact.parent)
    # Save all portable original artifacts, including trial logs and executed
    # sources. Checkpoints/conditioning caches are intentionally excluded.
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix in PORTABLE_SUFFIXES and "__pycache__" not in path.parts:
            evidence.add(path)
    for folder, names in (("2026-10-08", ("summary.json", "validation.json")),
                          ("2026-10-08-round2", ("summary.json", "offline_rebuild_audit.json", "test_validation.json"))):
        for name in names:
            evidence.add(repo / "results" / folder / name)
    evidence.add(repo / "experiments/original_report.py")
    evidence.add(repo / "experiments/original_dora.py")
    evidence.add(repo / "experiments/test_original_dora.py")
    evidence.add(repo / "results/2026-10-09-baseline-audit/original_dora.py")
    for name in ("experiments/audit_original_tasks.py", "experiments/original_retrieval_analysis.py"):
        if (repo / name).exists():
            evidence.add(repo / name)
    payload = {"complete": True, "method_order": METHODS, "method_labels": LABELS, "tasks": tasks,
               "teacher": diagnostic, "retrieval_analysis": retrieval_analysis, "limitations": LIMITATIONS,
               "scope": "Seven downstream tasks: Flowers102, SciFact retrieval, ViGGO, SDXL, Aircraft, NFCorpus, COGS; SciFact transfer is a second evaluation of the NFCorpus fit. Separate 28-cell teacher diagnostic.",
               "source_sha256": sha(repo / "experiments/original_report.py"),
               "report_versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    return payload, evidence


def formatted(item, decimals=2):
    if item is None:
        return "—"
    result = f"{item['mean']:.{decimals}f}"
    if item["sample_sd"] is not None:
        result += f" ± {item['sample_sd']:.{decimals}f}"
    return result


def table(payload, keys, output, name):
    tasks = [payload["tasks"][key] for key in keys]
    header = ["Method", *[task["title"] + " / " + task["metric"] for task in tasks]]
    methods = [method for method in METHODS if any(method in task["summary"] for task in tasks)]
    rows = [[LABELS[method], *[formatted(task["summary"].get(method), 6 if key == "sdxl" else 2)
                             for key, task in zip(keys, tasks)]] for method in methods]
    text = "| " + " | ".join(header) + " |\n|:--|" + "--:|" * len(tasks) + "\n"
    text += "".join("| " + " | ".join(row) + " |\n" for row in rows)
    text += "\nMean ± sample SD over seeds 42–44. Comparator measurements are reused from the previous reports; original DoRA uses the same task budgets and validation search. Each column has its own metric and does not contribute to an aggregate ranking.\n"
    (output / f"{name}.md").write_text(text)
    with (output / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def chart(payload, keys, output, name):
    methods = list(COLORS)
    fig, axes = plt.subplots(1 if len(keys) <= 3 else 2, min(3, len(keys)) if len(keys) <= 3 else 2,
                             figsize=(12, 4.2) if len(keys) <= 3 else (12, 8.2), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        task = payload["tasks"][key]
        for index, method in enumerate(methods):
            value = task["summary"][method]
            ax.errorbar(index, value["mean"], yerr=value["sample_sd"], fmt="o", color=COLORS[method],
                        capsize=5, markersize=8, linewidth=1.5)
        baseline = task["summary"].get("baseline")
        if baseline:
            ax.axhline(baseline["mean"], color="#777777", linestyle=":", linewidth=1, label="Task baseline")
        ax.set_xticks(range(4), ["Original\nDoRA", "LoRA", "Revised\nDoRA", "NoRA"])
        ax.set_title(task["title"], loc="left", fontweight="bold")
        ax.set_ylabel(task["metric"])
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Original implementation on the same benchmark protocols", fontsize=15)
    fig.text(.5, .015, "Mean ± SD, 3 seeds. Dotted: task baseline. Prior comparator runs reused; separate axes and no aggregate score.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .95])
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def resource_table(payload, output):
    lines = ["| Task | Rank | Method | Parameters | Training seconds, mean ± SD | Peak allocated GiB, max |",
             "|:--|--:|:--|--:|--:|--:|"]
    for key, task in payload["tasks"].items():
        # SciFact transfer is evaluation of the NFCorpus fit, not extra training.
        if key.startswith("scifact_transfer"):
            continue
        for method in METHODS:
            item = task["summary"].get(method)
            if item is None:
                continue
            memory = item["peak_memory_bytes"]
            peak = f"{max(memory['values']) / 2**30:.3f}" if memory else "—"
            lines.append(f"| {task['title']} | {task['rank']} | {LABELS[method]} | {', '.join(map(str, item['trainable_parameters']))} | {formatted(item['training_seconds'])} | {peak} |")
    lines.append("\nTimes retain each task's recorded scope. Vision and language include scheduled validation/checkpoint work; retrieval and SDXL time training. Setup and final generation are excluded. Runs occurred in separate sessions, so cold kernels and system overhead may differ. These are fixed-budget measurements, not time to equal quality. Peak allocated memory includes resident task models and is not reserved memory.\n")
    (output / "runtime_memory.md").write_text("\n".join(lines))


def secondary_tables(payload, output):
    cogs = payload["tasks"]["cogs_r8"]
    lines = ["COGS metrics are means across three seeds. Structural and lexical categories remain separate; generation-cap flags are not compared across runner versions.\n",
             "| Method | IID exact % | OOD exact % | Lexical exact % | Structural exact % |",
             "|:--|--:|--:|--:|--:|"]
    methods = [method for method in METHODS if method != "baseline" and method in cogs["summary"]]
    for method in methods:
        rows = [row for row in cogs["records"] if row["method"] == method]
        fields = [statistics.mean(row["iid"]["atom_exact"] for row in rows),
                  statistics.mean(row["test"]["atom_exact"] for row in rows),
                  statistics.mean(row["test"]["lexical_macro_exact"] for row in rows),
                  statistics.mean(row["test"]["structural_macro_exact"] for row in rows)]
        lines.append("| " + LABELS[method] + " | " + " | ".join(f"{value * 100:.2f}" for value in fields) + " |")
    (output / "cogs_metrics.md").write_text("\n".join(lines) + "\n")
    categories = sorted(cogs["records"][0]["test"]["categories"])
    lines = ["| COGS OOD category | " + " | ".join(LABELS[method] for method in methods) + " |",
             "|:--|" + "--:|" * len(methods)]
    for category in categories:
        values = [100 * statistics.mean(row["test"]["categories"][category]["atom_exact"]
                                        for row in cogs["records"] if row["method"] == method) for method in methods]
        lines.append("| " + category + " | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    (output / "cogs_categories.md").write_text("\n".join(lines) + "\n\nEach category has 32 shared examples per seed. These are descriptive category means, not independent hypothesis tests.\n")
    sdxl = payload["tasks"]["sdxl"]
    lines = ["SDXL uses one subject, three training photos, and one validation/test photo each. Denoising loss, CLIP alignment and DINO similarity do not establish overall image quality. Class-only CLIP is a post-hoc diagnostic on unchanged generated images and was not used for selection.\n",
             "| Method | Denoising MSE | Identifier CLIP | Class-only CLIP | DINO subject similarity |", "|:--|--:|--:|--:|--:|"]
    for method in METHODS:
        rows = [row for row in sdxl["records"] if row["method"] == method]
        if not rows:
            continue
        values = []
        for key in ("clip_text_image_cosine_mean", "class_only_clip_mean_post_hoc", "dino_train_subject_cosine_mean"):
            scores = [row["image_metrics"][key] for row in rows if key in row["image_metrics"]]
            values.append(formatted(distribution(scores), 4) if len(scores) == len(rows) else "—")
        lines.append("| " + LABELS[method] + " | " + formatted(sdxl["summary"][method], 6) + " | " + " | ".join(values) + " |")
    (output / "sdxl_metrics.md").write_text("\n".join(lines) + "\n")


def flatten_comparisons(payload):
    rows = []
    for key, task in payload["tasks"].items():
        for row in task["comparisons"]:
            if "mean_difference" in row:
                rows.append({"task": key, "rank": task["rank"], **row})
            elif "metrics" in row:
                metric = "accuracy" if key == "flowers" else "macro_class_accuracy"
                value = row["metrics"][metric]
                rows.append({"task": key, "rank": task["rank"], "candidate": row["candidate"], "reference": row["reference"],
                             "mean_difference": value["mean_difference"] * 100,
                             "percentile_95_interval": [v * 100 for v in value["percentile_95_interval"]],
                             "units": "percentage points"})
    # Retrieval's independently rescored query comparisons retain their native
    # source schema in summary.json; normalize their table schema here.
    for row in payload["retrieval_analysis"].get("comparisons", []):
        value = row["metrics"]["ndcg_at_10"]
        task = "scifact" if row["benchmark"] == "scifact" else ("nfcorpus" if row["dataset"] == "nfcorpus" else "scifact_transfer")
        rows.append({"task": task, "rank": row["rank"], "candidate": row["candidate"], "reference": row["reference"],
                     "mean_difference": value["difference"] * 100,
                     "percentile_95_interval": [v * 100 for v in value["ci95"]], "units": "nDCG×100 points"})
    for row in rows:
        if row["reference"] == "frozen":
            row["reference"] = "baseline"
        key = row["task"] if row["task"] in payload["tasks"] else f"{row['task']}_r{row['rank']}"
        summary = payload["tasks"][key]["summary"]
        close(row["mean_difference"], summary[row["candidate"]]["mean"] - summary[row["reference"]]["mean"], 2e-5)
    return rows


def comparison_table(payload, output):
    rows = flatten_comparisons(payload)
    write(output / "comparisons.json", rows)
    lines = ["| Task | Rank | Contrast | Difference | Exploratory 95% interval |", "|:--|--:|:--|--:|:--|"]
    for row in rows:
        difference = row.get("mean_difference", row.get("mean_delta"))
        interval = row.get("percentile_95_interval", row.get("percentile_95_ci"))
        if difference is None or interval is None:
            continue
        candidate, reference = row.get("candidate", ORIGINAL), row.get("reference")
        if reference is None:
            continue
        lines.append(f"| {row.get('task', row.get('dataset', 'retrieval'))} | {row.get('rank', 8)} | {LABELS.get(candidate, candidate)} − {LABELS.get(reference, reference)} | {difference:+.3f} | [{interval[0]:+.3f}, {interval[1]:+.3f}] |")
    lines.append("\nOriginal-minus-comparator differences. Accuracy/exact-match values are percentage points; retrieval values are nDCG×100 points. Paired seed/example intervals use 10,000 draws, preserve relevant class/category strata, and are exploratory without multiple-comparison correction. Three seeds do not establish a universal ranking. SDXL has no claim of image-quality superiority.\n")
    (output / "paired_differences.md").write_text("\n".join(lines))


def teacher_outputs(payload, output):
    cells = payload["teacher"]["cells"]
    methods = [method for method in METHODS if method != "baseline"]
    lines = ["| Teacher cell | " + " | ".join(LABELS[method] for method in methods) + " |",
             "|:--|" + "--:|" * len(methods)]
    values = []
    for key, cell in cells.items():
        row = [cell["methods"][method]["mean"] for method in methods]
        values.append(row)
        lines.append("| " + key + " | " + " | ".join(f"{value:.3g}" for value in row) + " |")
    lines.append("\nMean held-out MSE divided by frozen-model MSE; lower is better. These are 28 related synthetic cells with three initialization seeds each, shown separately from downstream benchmarks. No universal winner or independent-cell significance claim.\n")
    (output / "teacher_table.md").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(12, 12))
    matrix = np.log10(np.maximum(np.array(values), 1e-12))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto", vmin=-12, vmax=0)
    ax.set_xticks(range(len(methods)), [LABELS[method] for method in methods], rotation=35, ha="right")
    ax.set_yticks(range(len(cells)), list(cells))
    ax.set_title("Separate teacher diagnostic: log10(relative test MSE)", loc="left")
    fig.colorbar(im, ax=ax, label="log10(MSE / frozen MSE); lower is better")
    fig.tight_layout()
    fig.savefig(output / "teacher_comparison.png", dpi=160)
    fig.savefig(output / "teacher_comparison.svg", metadata={"Date": None})
    plt.close(fig)


def render(payload, output):
    assert payload["complete"]
    assert set(payload["tasks"]) == {"flowers", "scifact", "extraction", "sdxl", "aircraft_r2", "aircraft_r8",
                                       "nfcorpus_r2", "nfcorpus_r8", "scifact_transfer_r2", "scifact_transfer_r8", "cogs_r8"}
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "original-dora-20261009"
    # Always derive summaries afresh from individual seeded records, including
    # during portable replay. Never promote a partial or edited aggregate.
    for task in payload["tasks"].values():
        recomputed = summarize(task["records"])
        assert recomputed == task["summary"]
    first = [row[0] for row in ROUND1]
    harder = [row[0] + "_r8" for row in ROUND2]
    low_rank = [row[0] + "_r2" for row in ROUND2[:-1]]
    for keys, name in ((first, "first_round"), (harder, "rank8"), (low_rank, "rank2")):
        table(payload, keys, output, name + "_table")
        chart(payload, keys, output, name + "_comparison")
    resource_table(payload, output)
    secondary_tables(payload, output)
    comparison_table(payload, output)
    teacher_outputs(payload, output)
    write(output / "summary.json", payload)
    notes = payload["scope"] + "\n\n" + LIMITATIONS + "\n"
    extension = payload["tasks"]["flowers"]["audit"].get("details", {}).get("diagnostic_extension")
    if extension:
        notes += ("\nFlowers seed44 required relaxing only the startup identity diagnostic from absolute tolerance 1e-5 to 2e-5. "
                  "The literal historical class and the compatibility adapter matched bitwise in the recorded full-model FP32 control; "
                  "the failed attempt performed no optimizer steps or test evaluation. A separately snapshotted continuation completed seed44. "
                  "Its source versions, failed log, control and continuation manifest remain in the archive. This did not change adapter math.\n")
    notes += ("\nCOGS uses the same text-scoring rules and fixed generation budget. The current runner corrects EOS-cap diagnostics and retains raw token IDs; "
              "historical generation-cap flags are not directly comparable, while text-based exact-match scores remain comparable.\n")
    (output / "scope_and_portability.md").write_text(notes)
    (output / "reproduce.md").write_text(
        "The raw archive contains original task results, validation trials, predictions, split manifests, executed sources, independent audits, and compact prior comparator evidence. Prepared SDXL photos and generated images are retained. Serialized tensor payloads (model/adapter checkpoints, conditioning caches, and teacher problem tensors) are omitted; their locally verified hashes and generation provenance remain in audits.\n\n"
        "Verify raw_manifest.json hashes before extraction. If split, concatenate exactly the archive_parts paths listed in that manifest, in their listed order; do not concatenate unlisted files. Extract into an empty directory using Python tarfile with filter='data'. Then, from the extracted directory:\n\n"
        "```bash\npython report_source/experiments/original_report.py --rebuild report_inputs.json --output rebuilt\n```\n\n"
        "The rebuild verifies every retained input hash, recomputes seed summaries from individual records, and regenerates tables and plots without model inference. The raw predictions and task audit scripts support independent metric re-scoring. NumPy and Matplotlib are required; matching versions reproduce plot rendering most closely.\n")


def bundle(payload, evidence, output, part_bytes=40_000_000):
    evidence_rows = [{"archive_path": name, "sha256": sha(path), "bytes": path.stat().st_size}
                     for name, path in sorted(evidence.files.items())]
    payload["evidence"] = evidence_rows
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    archive = output / "raw_artifacts.tar.gz"
    previous_parts = {path for path in output.glob("raw_artifacts.tar.gz.part-*")
                      if path.name.rsplit(".part-", 1)[-1].isdigit()}
    with archive.open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for item in evidence_rows:
                    path = evidence.files[item["archive_path"]]
                    assert sha(path) == item["sha256"], ("Evidence changed during collection", path)
                    info = tarfile.TarInfo(item["archive_path"])
                    info.size, info.mtime, info.mode = item["bytes"], 0, 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                info = tarfile.TarInfo("report_inputs.json")
                info.size, info.mtime, info.mode = len(encoded), 0, 0o644
                tar.addfile(info, io.BytesIO(encoded))
    manifest = {"files": [*evidence_rows, {"archive_path": "report_inputs.json", "bytes": len(encoded),
                                          "sha256": hashlib.sha256(encoded).hexdigest()}],
                "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
                "exclusions": "Serialized tensor payloads: pretrained/adapter checkpoints, conditioning caches and teacher problem tensors. Prepared SDXL photos and generated images are retained; bulk external benchmark image corpora are not bundled. Local audits retain checkpoint/tensor hashes and generation provenance.",
                "audit_scope": "Original results/trials/predictions/sources plus exact comparator files read by task analyses and this report."}
    if archive.stat().st_size > part_bytes:
        parts = []
        new_parts = set()
        with archive.open("rb") as stream:
            index = 0
            while chunk := stream.read(part_bytes):
                path = archive.with_name(archive.name + f".part-{index:03d}")
                path.write_bytes(chunk)
                new_parts.add(path)
                parts.append({"path": path.name, "bytes": len(chunk), "sha256": sha(path)})
                index += 1
        manifest["archive_parts"] = parts
        archive.unlink()
    else:
        manifest["archive_path"] = archive.name
        new_parts = set()
    for obsolete in previous_parts - new_parts:
        obsolete.unlink()
    write(output / "raw_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/var/tmp/dora-bench/original_bench"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/2026-10-09-original"))
    parser.add_argument("--rebuild", type=Path)
    args = parser.parse_args()
    if args.rebuild:
        payload = read(args.rebuild)
        assert sha(Path(__file__)) == payload["source_sha256"], "Use the archived report source for replay"
        for row in payload["evidence"]:
            path = args.rebuild.parent / row["archive_path"]
            assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], path
        render(payload, args.output)
        write(args.output / "rebuild_audit.json", {"status": "pass", "evidence_files_verified": len(payload["evidence"]),
                                                   "input_sha256": sha(args.rebuild), "source_sha256": sha(Path(__file__)),
                                                   "scope": "All retained inputs hash-verified; all method means/sample SD recomputed from individual seed records; tables/figures regenerated without model inference."})
    else:
        try:
            payload, evidence = collect(args.raw_root.resolve(), args.repo.resolve())
        except NotReady as error:
            parser.exit(2, str(error) + "\n")
        args.output.mkdir(parents=True, exist_ok=True)
        bundle(payload, evidence, args.output)
        render(payload, args.output)
    print(json.dumps({"complete": True, "tasks": len(payload["tasks"]), "teacher_cells": len(payload["teacher"]["cells"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
