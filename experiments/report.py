"""Build the four-task report from measured artifacts, without using a GPU.

python -m experiments.report
Use --allow-partial with a separate --output-dir for progress previews only.
Raw JSON/JSONL/log/NPZ files are archived; model weights and images are excluded.
"""

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


METHODS = ("lora", "dora", "nora", "dora_nora")
LABELS = {"baseline": "Task baseline", "lora": "LoRA", "dora": "DoRA", "nora": "NoRA", "dora_nora": "DoRA+NoRA"}
COLORS = {"lora": "#0072B2", "dora": "#E69F00", "nora": "#009E73", "dora_nora": "#CC79A7"}
TASKS = {
    "vision": {"title": "Flowers102 classification", "metric": "Test top-1 accuracy", "unit": "%", "direction": "higher", "baseline": "Frozen ViT backbone + trained head", "seeds": 3, "baseline_seeds": 3},
    "retrieval": {"title": "SciFact retrieval", "metric": "Test nDCG@10", "unit": "×100", "direction": "higher", "baseline": "Frozen MiniLM encoder", "seeds": 3, "baseline_seeds": 1},
    "extraction": {"title": "ViGGO → JSON extraction", "metric": "Test exact object match", "unit": "%", "direction": "higher", "baseline": "Frozen Qwen2.5-3B-Instruct", "seeds": 3, "baseline_seeds": 1},
    "sdxl": {"title": "SDXL subject adaptation", "metric": "Held-out denoising MSE", "unit": "MSE", "direction": "lower", "baseline": "Base SDXL", "seeds": 3, "baseline_seeds": 1},
}


def load_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def median_present(records, field):
    values = [record[field] for record in records if record.get(field) is not None]
    return statistics.median(values) if values else None


def distribution(values):
    return {"mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "values": values}


def read_task(name, root, extra_roots=()):
    aggregate = load_json(root / "results.json")
    sources = [root / "results.json"] if aggregate is not None else []
    provenance, tuning, records = {}, None, []
    if name == "vision":
        raw = aggregate or []
        provenance = load_json(root / "provenance.json", {})
        tuning = {"selected_learning_rates": load_json(root / "selected_learning_rates.json", {}),
                  "candidates": [{"method": row["method"], "learning_rate": row["adapter_learning_rate"], "validation": row["best_validation"]} for row in load_json(root / "pilot_results.json", [])]}
        for row in raw:
            if not row.get("test"):
                continue
            assert row["rank"] == 8, "This report's declared rank must match the experiment"
            records.append({"method": row["method"], "seed": row["seed"], "value": 100 * row["test"]["accuracy"],
                            "test": row["test"], "learning_rate": row["adapter_learning_rate"], "selected_epoch": row["best_epoch"],
                            "trainable_parameters": row["trainable_parameters"], "training_wall_seconds": row["training_wall_seconds"],
                            "peak_memory_bytes": row["peak_cuda_allocated_bytes"], "frozen_base_unchanged": row["frozen_base_unchanged"],
                            "checkpoint_sha256": row["checkpoint_sha256"], "adapter_checkpoint_sha256": row["best_adapter_checkpoint_sha256"]})
    elif name == "retrieval":
        data = aggregate or {}
        provenance = data.get("provenance", load_json(root / "provenance.json", {}))
        if provenance:
            assert provenance["configuration"]["rank"] == 8
        tuning = {method: {"selected_learning_rate": item["selected_learning_rate"],
                           "candidates": [{"learning_rate": row["learning_rate"], "validation": row["evaluation"]["validation"]} for row in item["candidates"]]}
                  for method, item in data.get("tuning", {}).items()}
        for row in data.get("runs", []):
            training = row.get("training", {})
            records.append({"method": "baseline" if row["method"] == "frozen" else row["method"], "seed": row["seed"],
                            "value": 100 * row["evaluation"]["test"]["ndcg_at_10"], "test": row["evaluation"]["test"],
                            "learning_rate": row.get("learning_rate"), "trainable_parameters": row["trainable_parameters"],
                            "training_wall_seconds": training.get("seconds"), "peak_memory_bytes": training.get("peak_allocated_bytes"),
                            "corpus_encoding_seconds": row["evaluation"]["corpus_encoding_seconds"]})
    elif name == "extraction":
        data = aggregate or {}
        provenance = data.get("manifest", load_json(root / "manifest.json", {}))
        if provenance:
            assert provenance["configuration"]["rank"] == 8
        tuning = {"selected_learning_rates": data.get("selected_learning_rates", {}),
                  "candidates": [{"method": row["method"], "learning_rate": row["lr"], "validation_nll": row["validation_nll"]} for row in data.get("tuning", [])]}
        baseline = data.get("baseline", load_json(root / "baseline.json"))
        if baseline:
            records.append({"method": "baseline", "seed": None, "value": 100 * baseline["test"]["exact_match"],
                            "test": baseline["test"], "test_nll": baseline["test_nll"], "trainable_parameters": 0})
        for row in data.get("results", []):
            records.append({"method": row["method"], "seed": row["seed"], "value": 100 * row["test"]["exact_match"],
                            "test": row["test"], "test_nll": row["test_nll"], "learning_rate": row["lr"],
                            "selected_step": row["selected_step"], "validation_nll": row["validation_nll"],
                            "trainable_parameters": row["trainable_parameters"], "training_wall_seconds": row["train_and_validation_seconds"],
                            "peak_memory_bytes": row["peak_allocated_bytes"], "frozen_base_unchanged": row["frozen_weights_unchanged"]})
    else:
        data = aggregate or {}
        provenance = data.get("provenance", load_json(root / "provenance.json", {}))
        if provenance:
            assert provenance["controls"]["rank"] == 8
        raw = data.get("results")
        if raw is None:
            raw = []
            for method in ("baseline", *METHODS):
                path = root / method / "result.json"
                row = load_json(path)
                if row:
                    sources.append(path)
                    raw.append(row)
        raw = [dict(row, seed=row.get("seed", provenance.get("controls", {}).get("seed", 42))) for row in raw]
        additional_provenance = {}
        for extra_root in extra_roots:
            extra = load_json(extra_root / "results.json", {})
            extra_provenance = extra.get("provenance", load_json(extra_root / "provenance.json", {}))
            if extra_provenance:
                additional_provenance[str(extra_root)] = extra_provenance
            if extra:
                sources.append(extra_root / "results.json")
                extra_rows = extra.get("results", [])
            else:
                extra_rows = []
                for method in METHODS:
                    path = extra_root / method / "result.json"
                    row = load_json(path)
                    if row:
                        sources.append(path)
                        extra_rows.append(row)
            for row in extra_rows:
                if row["method"] != "baseline":
                    raw.append(dict(row, seed=row.get("seed", extra_provenance.get("controls", {}).get("seed"))))
        provenance = {**provenance, "additional_seed_provenance": additional_provenance}
        class_diagnostics, diagnostic_provenance = {}, {}
        for diagnostic_root in (root, *extra_roots):
            diagnostic = load_json(diagnostic_root / "class_clip_metrics.json", {})
            if diagnostic:
                assert diagnostic["post_hoc"] and not diagnostic["used_for_model_selection"]
                assert not diagnostic["generated_images_changed"] and not diagnostic["original_metrics_changed"]
                diagnostic_provenance[str(diagnostic_root)] = {key: value for key, value in diagnostic.items() if key != "results"}
                for item in diagnostic["results"]:
                    class_diagnostics[(item["method"], item["seed"])] = item
        provenance["post_hoc_class_clip_diagnostic"] = diagnostic_provenance
        tuning = {}
        for row in raw:
            training = row.get("training", {})
            diagnostic = class_diagnostics.get((row["method"], row["seed"]))
            image_metrics = dict(row["image_metrics"])
            if diagnostic:
                image_metrics["class_only_clip_mean_post_hoc"] = diagnostic["class_only_clip_mean"]
            records.append({"method": row["method"], "seed": row["seed"], "value": row["test"]["mean_mse"],
                            "test": row["test"], "validation": row["validation"], "image_metrics": image_metrics, "learning_rate": row.get("selected_lr"),
                            "class_clip_diagnostic": diagnostic, "generated_image_sha256": [sample["sha256"] for sample in row["samples"]],
                            "trainable_parameters": row.get("checkpoint", {}).get("trainable_parameters", row.get("trainable_parameters")),
                            "training_wall_seconds": training.get("seconds"), "peak_memory_bytes": training.get("peak_allocated_bytes"),
                            "frozen_base_unchanged": row.get("frozen_sha256_before") == row.get("frozen_sha256_after") if row["method"] != "baseline" else None,
                            "adapter_checkpoint_sha256": row.get("checkpoint", {}).get("sha256")})
            if row["method"] != "baseline" and row.get("trials"):
                tuning[row["method"]] = {"selected_learning_rate": row["selected_lr"],
                                         "candidates": [{"learning_rate": trial["lr"], "validation_mse": trial["validation"]["mean_mse"]} for trial in row["trials"]]}
    summary, missing = {}, []
    for row in records:
        if row.get("frozen_base_unchanged") is False:
            raise ValueError(f"Frozen-base audit failed in {name}/{row['method']}/{row['seed']}")
    for method in ("baseline", *METHODS):
        selected = sorted((row for row in records if row["method"] == method), key=lambda row: row["seed"] or 0)
        seeds = [row["seed"] for row in selected]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate final seeds in {name}/{method}: {seeds}")
        values = [row["value"] for row in selected]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite metric in {name}/{method}")
        expected = TASKS[name]["baseline_seeds" if method == "baseline" else "seeds"]
        if len(values) != expected:
            missing.append(f"{method}: {len(values)}/{expected} completed")
        summary[method] = {
            "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "seeds": seeds, "values": values,
            "median_training_wall_seconds": median_present(selected, "training_wall_seconds"),
            "median_peak_memory_bytes": median_present(selected, "peak_memory_bytes"),
            "max_recorded_peak_memory_bytes": max((row["peak_memory_bytes"] for row in selected if row.get("peak_memory_bytes") is not None), default=None),
            "trainable_parameters": sorted(set(row["trainable_parameters"] for row in selected if row.get("trainable_parameters") is not None)),
        }
        if name == "vision":
            secondary = {"macro_class_accuracy_percent": [100 * row["test"]["macro_class_accuracy"] for row in selected]}
        elif name == "retrieval":
            secondary = {metric + "_times_100": [100 * row["test"][metric] for row in selected] for metric in ("recall_at_10", "mrr_at_10")}
        elif name == "extraction":
            secondary = {metric + "_percent": [100 * row["test"][metric] for row in selected] for metric in ("valid_json", "schema_valid", "slot_micro_f1")}
            secondary["target_token_nll"] = [row["test_nll"] for row in selected]
        else:
            secondary = {metric: [row["image_metrics"][metric] for row in selected] for metric in ("clip_text_image_cosine_mean", "dino_train_subject_cosine_mean")}
            secondary["class_only_clip_mean_post_hoc"] = [row["image_metrics"]["class_only_clip_mean_post_hoc"] for row in selected if "class_only_clip_mean_post_hoc" in row["image_metrics"]]
            if len(secondary["class_only_clip_mean_post_hoc"]) != len(selected):
                missing.append(f"{method}: class-only CLIP available for {len(secondary['class_only_clip_mean_post_hoc'])}/{len(selected)} finished seeds")
        summary[method]["secondary_metrics"] = {metric: distribution(values) for metric, values in secondary.items()}
    paired_uncertainty = load_json(root / "paired_bootstrap.json")
    if paired_uncertainty is None and name == "vision":
        paired_uncertainty = load_json(root / "summary_audited.json", {}).get("paired_bootstrap_vs_lora")
    return {"task": name, **TASKS[name], "complete": not missing, "missing": missing,
            "provenance": provenance, "validation_tuning": tuning, "summary": summary,
            "runs": records, "raw_directory": str(root),
            "source_files": {str(path): sha256(path) for path in sources},
            "paired_uncertainty": paired_uncertainty}


def format_value(value, decimals=2):
    if value["mean"] is None:
        return "Pending"
    text = f"{value['mean']:.{decimals}f}"
    if value["sample_std"] is not None:
        text += f" ± {value['sample_std']:.{decimals}f}"
    return text


def validate_artifacts(tasks, roots, extra_roots, output, cache):
    """Recompute scored metrics from raw predictions rather than trusting summaries."""
    audited = {"vision": [], "retrieval": [], "extraction": [], "sdxl": [], "sources": []}

    def close(actual, expected, context, tolerance=1e-10):
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
            raise ValueError(f"Metric audit failed for {context}: recomputed {actual}, recorded {expected}")

    for row in tasks["vision"]["runs"]:
        path = roots["vision"] / f"seed_{row['seed']}" / row["method"] / "test_predictions.json"
        predictions = load_json(path)
        labels, guesses = np.array(predictions["labels"]), np.array(predictions["predictions"])
        assert len(labels) == len(guesses) == row["test"]["count"]
        correct = int((labels == guesses).sum())
        assert correct == row["test"]["correct"]
        close(correct / len(labels), row["test"]["accuracy"], str(path))
        counts = np.bincount(labels, minlength=102)
        correct_by_class = np.bincount(labels[labels == guesses], minlength=102)
        close(float(np.mean(correct_by_class / np.maximum(counts, 1))), row["test"]["macro_class_accuracy"], str(path), 2e-7)
        audited["vision"].append({"method": row["method"], "seed": row["seed"], "examples": len(labels), "correct": correct})

    if tasks["retrieval"]["runs"]:
        dataset = tasks["retrieval"]["provenance"]["dataset"]
        hub_relative = Path("datasets--BeIR--scifact-qrels") / "snapshots" / dataset["qrels_revision"] / "test.tsv"
        candidates = [output / "audit_inputs" / "retrieval_test_qrels.tsv",
                      roots["retrieval"].parent / "audit_inputs" / "retrieval_test_qrels.tsv",
                      Path(os.environ.get("HF_HOME", str(cache))) / "hub" / hub_relative,
                      cache / "hub" / hub_relative,
                      cache / "huggingface" / "hub" / hub_relative]
        qrels_path = next((path for path in candidates if path.exists() and sha256(path) == dataset["source_sha256"]["test_qrels"]), None)
        if qrels_path is None:
            raise FileNotFoundError("Pinned SciFact test qrels needed for independent metric validation")
        qrels = {}
        with qrels_path.open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if int(row["score"]) > 0:
                    qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        target = output / "audit_inputs" / "retrieval_test_qrels.tsv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(qrels_path.read_bytes())
        for row in tasks["retrieval"]["runs"]:
            method = "frozen" if row["method"] == "baseline" else row["method"]
            path = roots["retrieval"] / f"{method}_seed{row['seed']}" / "test_per_query.json"
            predictions = load_json(path)
            assert len(predictions) == len(qrels) == row["test"]["query_count"]
            assert {item["query_id"] for item in predictions} == set(qrels)
            recomputed = {"recall_at_10": [], "mrr_at_10": [], "ndcg_at_10": []}
            for item in predictions:
                ranks = {document: index + 1 for index, document in enumerate(item["top_10"])}
                assert len(ranks) == 10
                assert all(first >= second for first, second in zip(item["scores"], item["scores"][1:]))
                relevant = qrels[item["query_id"]]
                matched = set(ranks) & set(relevant)
                dcg = sum((2 ** relevant[document] - 1) / math.log2(ranks[document] + 1) for document in matched)
                ideal = sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(sorted(relevant.values(), reverse=True)[:10]))
                metrics = {"recall_at_10": len(matched) / len(relevant),
                           "mrr_at_10": max((1 / ranks[document] for document in matched), default=0),
                           "ndcg_at_10": dcg / ideal}
                for metric, value in metrics.items():
                    close(value, item[metric], f"{path}:{item['query_id']}:{metric}")
                    recomputed[metric].append(value)
            for metric, values in recomputed.items():
                close(statistics.mean(values), row["test"][metric], f"{path}:{metric}")
            audited["retrieval"].append({"method": row["method"], "seed": row["seed"], "queries": len(predictions), "ranking_metrics_recomputed_from_pinned_qrels": True})

    if tasks["extraction"]["runs"]:
        source = roots["extraction"] / "source" / "experiments" / "extraction.py"
        tree = ast.parse(source.read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("ALLOWED_KEYS", "ALLOWED_ACTS", "SYSTEM"):
                        constants[target.id] = ast.literal_eval(node.value)
        allowed_keys, allowed_acts = constants["ALLOWED_KEYS"], constants["ALLOWED_ACTS"]
        assert "specifier" in allowed_keys and "specifier" in constants["SYSTEM"], "Exclude the invalid pre-correction extraction run"
        manifests = load_json(roots["extraction"] / "split_manifest.json")
        normalized_splits = []
        for split, rows in manifests.items():
            texts = {" ".join(row["utterance"].lower().split()) for row in rows}
            assert len(texts) == len(rows)
            normalized_splits.append(texts)
            for row in rows:
                assert set(row["target"]) <= allowed_keys and row["target"]["dialogue_act"] in allowed_acts
                assert all(isinstance(value, str) for value in row["target"].values())
        assert all(not first.intersection(second) for index, first in enumerate(normalized_splits) for second in normalized_splits[index + 1:])
        gold_by_id = {row["id"]: row for row in manifests["test"]}
        for row in tasks["extraction"]["runs"]:
            path = roots["extraction"] / ("baseline_predictions.jsonl" if row["method"] == "baseline" else f"{row['method']}_seed{row['seed']}/predictions.jsonl")
            predictions = [json.loads(line) for line in path.read_text().splitlines() if line]
            assert len(predictions) == len(gold_by_id) == row["test"]["count"]
            assert {item["id"] for item in predictions} == set(gold_by_id)
            valid = schema = exact = correct_slots = predicted_slots = gold_slots = 0
            for item in predictions:
                gold = gold_by_id[item["id"]]
                assert item["gold"] == gold["target"] and item["utterance"] == gold["utterance"]
                parsed = None
                try:
                    parsed = json.loads(item["text"].strip())
                    valid += 1
                except (json.JSONDecodeError, ValueError):
                    pass
                is_schema = (isinstance(parsed, dict) and set(parsed) <= allowed_keys
                             and all(isinstance(value, str) for value in parsed.values())
                             and parsed.get("dialogue_act") in allowed_acts)
                schema += int(is_schema)
                scored = parsed if is_schema else {}
                is_exact = scored == gold["target"]
                assert item["exact_match"] == is_exact
                exact += int(is_exact)
                predicted = set(scored.items())
                expected = set(gold["target"].items())
                correct_slots += len(predicted.intersection(expected))
                predicted_slots += len(predicted)
                gold_slots += len(expected)
            checks = {"valid_json": valid / len(predictions), "schema_valid": schema / len(predictions),
                      "exact_match": exact / len(predictions), "slot_micro_f1": 2 * correct_slots / max(1, predicted_slots + gold_slots),
                      "correct_slots": correct_slots, "predicted_slots": predicted_slots, "gold_slots": gold_slots}
            for metric, value in checks.items():
                close(value, row["test"][metric], f"{path}:{metric}")
            audited["extraction"].append({"method": row["method"], "seed": row["seed"], "examples": len(predictions), "exact_objects": exact,
                                          "valid_json": valid, "schema_valid": schema, "specifier_in_schema_and_prompt": True})

    for row in tasks["sdxl"]["runs"]:
        probes = row["test"]["rows"]
        assert len(probes) == 20
        assert {(item["timestep"], item["noise_seed"]) for item in probes} == {(timestep, seed) for timestep in (100, 300, 500, 700, 900) for seed in (1001, 1002, 1003, 1004)}
        values = [item["mse"] for item in probes]
        close(statistics.mean(values), row["test"]["mean_mse"], f"SDXL/{row['method']}/{row['seed']}/mean")
        close(statistics.pstdev(values), row["test"]["std_across_noise_timestep"], f"SDXL/{row['method']}/{row['seed']}/probe_std")
        diagnostic = row.get("class_clip_diagnostic")
        if diagnostic:
            assert len(diagnostic["per_prompt"]) == 4
            assert [item["image_sha256"] for item in diagnostic["per_prompt"]] == row["generated_image_sha256"]
            close(statistics.mean(item["cosine"] for item in diagnostic["per_prompt"]), diagnostic["class_only_clip_mean"], f"SDXL/{row['method']}/{row['seed']}/class_clip")
            close(diagnostic["original_identifier_clip_mean"], row["image_metrics"]["clip_text_image_cosine_mean"], f"SDXL/{row['method']}/{row['seed']}/original_clip", 1e-6)
        audited["sdxl"].append({"method": row["method"], "seed": row["seed"], "fixed_probes": 20, "probe_mean": statistics.mean(values)})

    source_roots = [(name, root, tasks[name]["provenance"]) for name, root in roots.items()]
    source_roots.extend((root.name, root, load_json(root / "provenance.json", {})) for root in extra_roots)
    for name, root, provenance in source_roots:
        hashes = provenance.get("source_sha256", {})
        if name == "vision" and provenance:
            hashes = {"experiments/vision.py": provenance["script_sha256"], "experiments/adapters.py": provenance["adapters_sha256"]}
        for filename, expected in hashes.items():
            paths = [root / "source" / filename, root / "executed_sources" / Path(filename).name]
            matching = next((path for path in paths if path.exists() and sha256(path) == expected), None)
            if matching is None:
                raise ValueError(f"Recorded source hash lacks matching snapshot: {name}/{filename}")
            audited["sources"].append({"task": name, "path": str(matching), "sha256": expected, "verified": True})
        if provenance:
            dependency = next((path for path in (root / "source" / "dora.py", root / "executed_sources" / "dora.py") if path.exists()), None)
            if dependency is None:
                raise ValueError(f"Inherited dora.py dependency snapshot missing: {name}")
            if not any(Path(entry["path"]) == dependency for entry in audited["sources"]):
                audited["sources"].append({"task": name, "path": str(dependency), "sha256": sha256(dependency), "verified": "Dependency snapshot; see task provenance for historical hash coverage"})
        diagnostic = load_json(root / "class_clip_metrics.json", {})
        if diagnostic:
            scorer = root / "executed_sources" / "sdxl_clip_diagnostic.py"
            assert scorer.exists() and sha256(scorer) == diagnostic["source_sha256"]
            audited["sources"].append({"task": name, "path": str(scorer), "sha256": sha256(scorer), "verified": True, "post_hoc": True})
        bootstrap = load_json(root / "paired_bootstrap.json", {})
        for hash_field, relative_path in (("source_audit_script_sha256", "audit.py"),
                                           ("analysis_script_sha256", "source/experiments/analyze_extraction.py"),
                                           ("reproduction_script_sha256", "source/experiments/analyze_retrieval.py")):
            if hash_field in bootstrap:
                analysis_source = root / relative_path
                assert analysis_source.exists() and sha256(analysis_source) == bootstrap[hash_field]
                audited["sources"].append({"task": name, "path": str(analysis_source), "sha256": bootstrap[hash_field], "verified": True, "analysis": True})
        if "sample_arrays_sha256" in bootstrap:
            assert sha256(root / "paired_bootstrap_samples.npz") == bootstrap["sample_arrays_sha256"]
    audited["passed"] = True
    audited["scope"] = "Recomputed vision accuracy from labels, retrieval ranking metrics from pinned qrels, JSON/schema/exact/F1 from generated text and gold, SDXL means from fixed probes, and matched recorded source hashes to archived executed .py files. No model weights were rerun."
    write_json(output / "validation.json", audited)
    return {"passed": True, "details": "validation.json", "run_counts": {name: len(audited[name]) for name in TASKS}, "verified_source_files": len(audited["sources"])}


def make_table(tasks, output):
    lines = ["| Method | Flowers102 top-1 % ↑ | SciFact nDCG@10 ×100 ↑ | ViGGO exact match % ↑ | SDXL denoising MSE ↓ |",
             "|:--|--:|--:|--:|--:|"]
    for method in ("baseline", *METHODS):
        values = [format_value(tasks[name]["summary"][method], 4 if name == "sdxl" else 2) for name in TASKS]
        lines.append("| " + " | ".join([LABELS[method], *values]) + " |")
    lines += ["", "Values are means ± sample standard deviations over three training seeds for each adapted task. "
              "The vision baseline also has three seeds; other baselines are fixed. Each SDXL seed's MSE is averaged "
              "over 20 fixed noise/timestep probes on one held-out photo. The error bar measures variation across training seeds, not across those probes.",
              "", "Baselines: frozen ViT backbone with a trained classifier; frozen MiniLM; frozen Qwen2.5-3B-Instruct; base SDXL.",
              "", "All adapter methods use rank 8 and validation-only learning-rate selection. "
              "SDXL denoising MSE measures a noise-prediction objective, **not image quality**. "
              "The SDXL study contains only three training photos and one validation/test photo each of one subject. "
              "Flowers102 is near the pretrained model's accuracy ceiling. These tasks do not establish a universal method ranking.",
              "", "Runtime fields in summary.json retain each task's recorded training wall-time scope; vision and JSON timing include validation. "
              "DoRA methods have extra magnitude parameters. Raw predictions, training logs, split manifests, and provenance are in raw_artifacts.tar.gz.", ""]
    (output / "table.md").write_text("\n".join(lines))
    sdxl_lines = ["| Method | Held-out denoising MSE ↓ | Original-prompt CLIP ↑ | Class-only CLIP, post hoc ↑ | DINO training-subject cosine ↑ |",
                  "|:--|--:|--:|--:|--:|"]
    for method in ("baseline", *METHODS):
        row = tasks["sdxl"]["summary"][method]
        sdxl_lines.append("| " + " | ".join([LABELS[method], format_value(row, 5),
                                             format_value(row["secondary_metrics"]["clip_text_image_cosine_mean"], 4),
                                             format_value(row["secondary_metrics"]["class_only_clip_mean_post_hoc"], 4),
                                             format_value(row["secondary_metrics"]["dino_train_subject_cosine_mean"], 4)]) + " |")
    sdxl_lines += ["", "Adapted results are mean ± sample SD over three training seeds. Each seed uses the same held-out noise probes "
                   "and four predeclared generation prompts/seeds. DINO compares generated images to the three training photos; "
                   "the shared orange backdrop can influence similarity. CLIP and DINO are separate proxies, not human judgments of quality or definitive identity. "
                   "The seed-42 contact sheet in sdxl_samples.jpg accompanies these numbers.",
                   "", "Learning rates were selected using short 50-step validation trials, followed by fixed 500-step refits. "
                   "This limited search does not establish each method's best achievable result; a rate selected at 50 steps may be suboptimal at 500. "
                   "Final validation losses are retained alongside trial losses in tasks/sdxl.json.", ""]
    sdxl_lines += ["The class-only CLIP diagnostic was added after training: scoring prompts replace the unfamiliar identifier phrase `sks dog` "
                   "with `a dog`. Generated images, original CLIP measurements, denoising metrics, selected rates, and checkpoints are unchanged. "
                   "This diagnostic was not used for model selection and remains an embedding proxy.", ""]
    (output / "sdxl_metrics.md").write_text("\n".join(sdxl_lines))


def make_runtime_tables(tasks, output):
    headers = {"vision": "Flowers102 train+val", "retrieval": "SciFact train", "extraction": "ViGGO train+val", "sdxl": "SDXL train"}
    lines = ["| Method | " + " | ".join(label + " (s)" for label in headers.values()) + " |", "|:--|--:|--:|--:|--:|"]
    memory = ["| Method | Flowers102 (GiB) | SciFact (GiB) | ViGGO (GiB) | SDXL (GiB) |", "|:--|--:|--:|--:|--:|"]
    for method in ("baseline", *METHODS):
        times, peaks = [], []
        for name in TASKS:
            row = tasks[name]["summary"][method]
            seconds, peak = row["median_training_wall_seconds"], row["max_recorded_peak_memory_bytes"]
            times.append("—" if seconds is None else f"{seconds:.1f}")
            peaks.append("—" if peak is None else f"{peak / 2**30:.2f}")
        lines.append("| " + " | ".join([LABELS[method], *times]) + " |")
        memory.append("| " + " | ".join([LABELS[method], *peaks]) + " |")
    lines += ["", "Median recorded final-training wall time across three seeds; excludes downloads, model/setup work, and generated-image/text evaluation. "
              "Vision and JSON extraction include their scheduled validation passes and checkpoint bookkeeping. Retrieval and SDXL record training loops only. "
              "Learning-rate search trials are additional work and are retained separately in the raw artifacts. "
              "All task budgets are fixed within a task; these seconds do not measure time to equal quality. Frozen baselines have no training; the vision baseline trains its classifier.", ""]
    memory += ["", "Maximum recorded peak allocated CUDA memory across each method's three final seeds, including resident model components. "
               "These are full-task allocations, not adapter-only memory or reserved memory. Vision's recorded peak can include selected-checkpoint evaluation; "
               "extraction includes validation; retrieval and SDXL record training peaks. GPU memory scopes differ across tasks, so compare methods within a column.", ""]
    (output / "runtime.md").write_text("\n".join(lines))
    (output / "peak_memory.md").write_text("\n".join(memory))


def reproduction_instructions(output):
    (output / "reproduce.md").write_text("""The raw archive contains metrics, predictions, split manifests, exact executed source snapshots, and verification logs. It excludes model weights and full-resolution images.

From the repository root, with the experiment Python environment installed:

```bash
restored_results=/tmp/dora-results-restored
mkdir -p "$restored_results"
tar -xzf results/2026-10-08/raw_artifacts.tar.gz -C "$restored_results"
python -m experiments.report \\
  --vision-root "$restored_results/vision" \\
  --retrieval-root "$restored_results/retrieval" \\
  --extraction-root "$restored_results/extraction" \\
  --sdxl-root "$restored_results/sdxl" \\
  --sdxl-extra-roots "$restored_results/sdxl_seed43" "$restored_results/sdxl_seed44" \\
  --verification-root "$restored_results/verification" \\
  --output-dir "$restored_results/rebuilt-report" --skip-images --skip-bundle
```

This recomputes the reported metrics from archived predictions and generates the tables and PNG/SVG chart without a GPU, model downloads, or original absolute paths. It does not rerun inference. The committed SDXL montage remains available separately; regenerating it requires the original full-resolution PNGs. Absolute paths inside provenance describe the original run and are not needed for this metric/chart reproduction command.

When the original full-resolution generated PNGs are available, rerun the separate CPU class-only CLIP diagnostic with:

```bash
python -m experiments.sdxl_clip_diagnostic --run-dirs \\
  /var/tmp/dora-bench/sdxl /var/tmp/dora-bench/sdxl_seed43 /var/tmp/dora-bench/sdxl_seed44
```

This writes all three class_clip_metrics.json files. It changes scoring text only, leaving generations, model selection, and original CLIP scores unchanged. It requires the generated PNGs and the pinned CLIP checkpoint; it cannot run from the compact metric archive alone.
""")


def make_plot(tasks, output, complete):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 15,
                         "axes.labelsize": 12, "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.18, top=0.78, hspace=0.60, wspace=0.24)
    fig.suptitle("Four adaptation experiments" + (" — partial results" if not complete else ""), fontsize=22, fontweight="bold", y=0.977)
    fig.text(0.5, 0.928, "Rank 8 · validation-only learning-rate selection · matched pretrained checkpoints within each task", ha="center", fontsize=12, color="#475569")
    legend = [Patch(facecolor=COLORS[method], label=LABELS[method]) for method in METHODS]
    legend.append(Line2D([0], [0], color="#475569", linewidth=1.7, linestyle="--", label="Task baseline"))
    fig.legend(handles=legend, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.905), frameon=False, fontsize=12)
    for ax, (name, task) in zip(axes.flat, tasks.items()):
        values = [task["summary"][method]["mean"] for method in METHODS]
        deviations = [task["summary"][method]["sample_std"] or 0 for method in METHODS]
        positions = np.arange(4)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#94a3b8")
        baseline = task["summary"]["baseline"]["mean"]
        maximum = max([value + error for value, error in zip(values, deviations) if value is not None] + ([baseline] if baseline is not None else []) + [0.01])
        upper = 112 if name != "sdxl" else maximum * 1.32
        ax.set_ylim(0, upper)
        ax.set_xlim(-0.6, 3.6)
        if name != "sdxl":
            ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(positions, [LABELS[method] for method in METHODS], fontsize=12)
        arrow = "↑" if task["direction"] == "higher" else "↓"
        ax.set_title(task["title"] + " " + arrow, loc="left", pad=34, fontweight="bold")
        counts = sorted(set(task["summary"][method]["n"] for method in METHODS))
        if name == "sdxl":
            detail = f"n={','.join(map(str, counts))} seeds · mean ± seed SD; MSE is not image quality"
        else:
            detail = f"n={','.join(map(str, counts))} seeds · bars show mean ± seed SD"
        ax.text(0, 1.065, detail, transform=ax.transAxes, fontsize=10, color="#475569")
        ax.set_ylabel(task["metric"] + (" (%)" if task["unit"] == "%" else (" ×100" if task["unit"] == "×100" else "")))
        if baseline is not None:
            ax.axhline(baseline, color="#475569", linestyle="--", linewidth=1.5, zorder=3)
        for index, method in enumerate(METHODS):
            value = values[index]
            if value is None:
                ax.text(index, upper * 0.08, "Pending", ha="center", rotation=90, color="#64748b", fontsize=11)
                continue
            ax.bar(index, value, color=COLORS[method], width=0.64, zorder=2)
            if deviations[index]:
                ax.errorbar(index, value, yerr=deviations[index], fmt="none", color="#172554", capsize=4, linewidth=1.3, zorder=4)
            ax.text(index, value + deviations[index] + upper * 0.025, f"{value:.4f}" if name == "sdxl" else f"{value:.2f}", ha="center", va="bottom", fontsize=11, fontweight="medium")
        baseline_text = f"{baseline:.4f}" if baseline is not None and name == "sdxl" else (f"{baseline:.2f}" if baseline is not None else "pending")
        ax.text(0, -0.24, f"Baseline: {task['baseline']} ({baseline_text})", transform=ax.transAxes, fontsize=9.5, color="#475569")
    fig.text(0.085, 0.063, "Higher is better for the first three panels; lower is better for SDXL. All axes start at zero.", fontsize=11, color="#334155")
    fig.text(0.085, 0.037, "SDXL: 3 train / 1 validation / 1 test photo; 20 fixed noise/timestep probes. Small-task results do not imply a universal ranking.", fontsize=10.5, color="#475569")
    for suffix in ("png", "svg"):
        destination = output / f"comparison.{suffix}"
        fig.savefig(destination, dpi=180)
        if suffix == "svg":
            destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n")
    plt.close(fig)


def make_sdxl_montage(root, output):
    """Use all predeclared seed-42 samples, with readable row and prompt labels."""
    aggregate = load_json(root / "results.json", {})
    records = {row["method"]: row for row in aggregate.get("results", [])}
    if not all(method in records for method in ("baseline", *METHODS)):
        return None
    from PIL import Image

    fig, axes = plt.subplots(5, 4, figsize=(13, 16))
    fig.subplots_adjust(left=0.14, right=0.995, top=0.905, bottom=0.035, hspace=0.035, wspace=0.015)
    fig.suptitle("SDXL subject adaptation · fixed prompts, training seed 42", fontsize=19, fontweight="bold", y=0.977)
    fig.text(0.56, 0.952, "Identical generation seeds across methods; all four predeclared prompts shown", ha="center", fontsize=11, color="#475569")
    column_titles = ("Original subject prompt", "Beach prompt", "Snowy forest prompt", "Watercolor prompt")
    sources = []
    for row_index, method in enumerate(("baseline", *METHODS)):
        samples = records[method]["samples"]
        assert len(samples) == 4
        for column, sample in enumerate(samples):
            source = Path(sample["path"])
            assert sha256(source) == sample["sha256"]
            ax = axes[row_index, column]
            with Image.open(source) as picture:
                ax.imshow(picture.convert("RGB"))
            ax.set_axis_off()
            if row_index == 0:
                ax.set_title(column_titles[column], fontsize=12, pad=10)
            sources.append({"method": method, "prompt": sample["prompt"], "generation_seed": sample["seed"], "source": str(source), "sha256": sample["sha256"]})
        position = axes[row_index, 0].get_position()
        label = "Frozen SDXL" if method == "baseline" else LABELS[method]
        fig.text(0.13, position.y0 + position.height / 2, label, ha="right", va="center", fontsize=13, fontweight="bold", color=COLORS.get(method, "#475569"))
    fig.text(0.14, 0.015, "Illustrative generations, not a human-rated quality comparison. The same subject has only three training photos.", fontsize=10, color="#475569")
    destination = output / "sdxl_samples.jpg"
    fig.savefig(destination, dpi=150, pil_kwargs={"quality": 90})
    plt.close(fig)
    return {"path": destination.name, "sha256": sha256(destination), "training_seed": 42, "samples": sources}


def bundle_artifacts(roots, output):
    allowed = {".json", ".jsonl", ".log", ".npz", ".py", ".tsv"}
    files, manifest = [], []
    for task, root in roots.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in allowed or path.stat().st_size > 32 * 1024 * 1024:
                continue
            archive_name = str(Path(task) / path.relative_to(root))
            files.append((path, archive_name))
            manifest.append({"archive_path": archive_name, "source": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    with (output / "raw_artifacts.tar.gz").open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path, name in files:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))
    write_json(output / "raw_manifest.json", {"files": manifest, "excluded": "Model checkpoints, images, caches, unlisted suffixes, and individual files larger than 32 MiB.",
                                               "archive_sha256": sha256(output / "raw_artifacts.tar.gz")})
    return {"archive": "raw_artifacts.tar.gz", "manifest": "raw_manifest.json", "file_count": len(files),
            "bytes": (output / "raw_artifacts.tar.gz").stat().st_size, "sha256": sha256(output / "raw_artifacts.tar.gz")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-root", type=Path, default=Path("/var/tmp/dora-bench/runs/vision"))
    parser.add_argument("--retrieval-root", type=Path, default=Path("/var/tmp/dora-bench/runs/retrieval"))
    parser.add_argument("--extraction-root", type=Path, default=Path("/var/tmp/dora-bench/runs/extraction"))
    parser.add_argument("--sdxl-root", type=Path, default=Path("/var/tmp/dora-bench/sdxl"))
    parser.add_argument("--sdxl-extra-roots", type=Path, nargs="*", default=[Path("/var/tmp/dora-bench/sdxl_seed43"), Path("/var/tmp/dora-bench/sdxl_seed44")])
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-10-08"))
    parser.add_argument("--cache-root", type=Path, default=Path("/var/tmp/dora-bench/cache"))
    parser.add_argument("--verification-root", type=Path, default=Path("/var/tmp/dora-bench/verification"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-bundle", action="store_true", help="Useful for progress previews")
    parser.add_argument("--skip-images", action="store_true", help="Rebuild tables/charts from archived metrics without original generation PNGs")
    args = parser.parse_args()
    roots = {name: getattr(args, name + "_root") for name in TASKS}
    tasks = {name: read_task(name, root, args.sdxl_extra_roots if name == "sdxl" else ()) for name, root in roots.items()}
    complete = all(task["complete"] for task in tasks.values())
    if not complete and not args.allow_partial:
        parser.error("Incomplete task results: " + "; ".join(f"{name}: {', '.join(task['missing'])}" for name, task in tasks.items() if not task["complete"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_copy = args.output_dir / "report_source" / "report.py"
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    source_copy.write_bytes(Path(__file__).read_bytes())
    validation = validate_artifacts(tasks, roots, args.sdxl_extra_roots, args.output_dir, args.cache_root)
    for name, task in tasks.items():
        write_json(args.output_dir / "tasks" / f"{name}.json", task)
    make_table(tasks, args.output_dir)
    make_runtime_tables(tasks, args.output_dir)
    reproduction_instructions(args.output_dir)
    make_plot(tasks, args.output_dir, complete)
    sample_images = {}
    montage = None if args.skip_images else make_sdxl_montage(roots["sdxl"], args.output_dir)
    if montage:
        sample_images["sdxl_samples.jpg"] = montage
    for source_name, destination_name in (("dataset_contact_sheet.jpg", "sdxl_dataset.jpg"),):
        source_image = roots["sdxl"] / source_name
        if not args.skip_images and source_image.exists():
            destination = args.output_dir / destination_name
            destination.write_bytes(source_image.read_bytes())
            sample_images[destination_name] = {"source": str(source_image), "sha256": sha256(source_image)}
    archive_roots = {**roots, **{root.name: root for root in args.sdxl_extra_roots},
                     "verification": args.verification_root, "audit_inputs": args.output_dir / "audit_inputs", "report_source": args.output_dir / "report_source"}
    archive = None if args.skip_bundle else bundle_artifacts(archive_roots, args.output_dir)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "complete": complete,
               "report_source_sha256": sha256(Path(__file__)), "report_command": [sys.executable, *sys.argv],
               "report_environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "task_order": list(TASKS), "method_order": list(METHODS),
               "method_colors": COLORS, "tasks": {name: {key: value for key, value in task.items() if key not in ("provenance", "runs", "validation_tuning")} for name, task in tasks.items()},
               "uncertainty": "Sample standard deviation across three training seeds for each adapter/task. SDXL first averages 20 fixed noise/timestep probes per training seed; variability among those probes is not used as seed uncertainty.",
               "validation": validation, "raw_archive": archive, "sdxl_images": sample_images,
               "sdxl_montage_protocol": "Seed-42 baseline/LoRA/DoRA/NoRA/DoRA+NoRA rows; four predeclared prompts with shared generation seeds. The montage is illustrative, not a human-rated image-quality score."}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"complete": complete, "output": str(args.output_dir), "counts": {name: {method: row["n"] for method, row in task["summary"].items()} for name, task in tasks.items()}, "raw_archive": archive}, indent=2))


if __name__ == "__main__":
    main()
