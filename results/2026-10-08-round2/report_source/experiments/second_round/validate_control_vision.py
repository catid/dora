"""Recompute the Aircraft numerical control from saved logits, without a GPU."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(first, second):
    delta = first.astype(np.float64) - second.astype(np.float64)
    flat = delta.reshape(-1)
    return float(np.abs(delta).max()), math.sqrt(float(flat @ flat) / len(flat)), int(np.sum(first.argmax(1) != second.argmax(1)))


def validate(root, write_audit=True):
    result_path, logits_path = root / "result.json", root / "logits.npz"
    result = json.loads(result_path.read_text())
    arrays = dict(np.load(logits_path, allow_pickle=False))
    assert digest(logits_path) == result["logits_sha256"]
    assert digest(root / "control_vision.py") == result["source_sha256"]
    source_root = root.parent / "rank2_run"
    for source, expected in result["training_source_sha256"].items():
        assert digest(source_root / "source" / source) == expected
    manifest_path = source_root / "split_manifest.json"
    assert digest(manifest_path) == result["split_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    count = result["count"]
    assert result["split"] == "validation" and 1 <= count <= len(manifest["splits"]["val"]) and result["ranks"] == [2, 8]
    assert result["image_ids"] == [row["image"] for row in manifest["splits"]["val"][:count]]
    labels = arrays.pop("labels")
    assert np.array_equal(labels, [row["label"] for row in manifest["splits"]["val"][:count]])
    baseline = json.loads((source_root / "baseline/seed_42/result.json").read_text())
    assert result["baseline_head_checkpoint_sha256"] == baseline["best_adapter_checkpoint_sha256"]
    assert result["baseline_head_tensor_sha256"] == baseline["final_trainable_sha256"]
    assert result["baseline_backbone_tensor_sha256"] == baseline["initial_backbone_sha256"]
    assert result["checkpoint_sha256"] == baseline["provenance"]["checkpoint_sha256"]
    methods = ("lora", "dora", "nora", "dora_nora", "dora_nora_mlr", "dora_nora_gain")
    cases = [("bare", 0), *((method, rank) for rank in (2, 8) for method in methods)]
    names = {f"{method}_r{rank}_{precision}" for method, rank in cases for precision in ("fp32", "bf16")}
    assert set(arrays) == names
    assert all(array.shape == (count, 100) and array.dtype == np.float32 and np.isfinite(array).all() for array in arrays.values())
    assert len(result["rows"]) == len(names)
    assert {f"{row['method']}_r{row['rank']}_{row['precision']}" for row in result["rows"]} == names
    for row in result["rows"]:
        name = f"{row['method']}_r{row['rank']}_{row['precision']}"
        maximum, rms, disagreements = differences(arrays[name], arrays[f"bare_r0_{row['precision']}"])
        assert abs(maximum - row["max_absolute_logit_difference_from_bare"]) < 1e-12
        assert abs(rms - row["rms_logit_difference_from_bare"]) < 1e-12
        if "mean_absolute_logit_difference_from_bare" in row:
            mean = np.abs(arrays[name].astype(np.float64) - arrays[f"bare_r0_{row['precision']}"]).mean()
            assert abs(mean - row["mean_absolute_logit_difference_from_bare"]) < 1e-12
        assert disagreements == row["argmax_disagreement_count_from_bare"]
        assert int(np.sum(arrays[name].argmax(1) == labels)) == row["probe_correct"]
    expected_pairs = set()
    for precision in ("fp32", "bf16"):
        method_names = sorted(name for name in names if name.endswith(precision) and not name.startswith("bare"))
        expected_pairs.update(itertools.combinations(method_names, 2))
        equal = all(np.array_equal(arrays[first].view(np.uint32), arrays[second].view(np.uint32))
                    for first, second in itertools.combinations(method_names, 2))
        assert equal == result["all_adapters_bitwise_equal_by_precision"][precision]
    pairs = result["pairwise_adapter_comparisons"]
    assert len(pairs) == len(expected_pairs)
    assert {tuple(sorted((row["first"], row["second"]))) for row in pairs} == expected_pairs
    for row in pairs:
        first, second = arrays[row["first"]], arrays[row["second"]]
        maximum, _, disagreements = differences(first, second)
        assert np.array_equal(first.view(np.uint32), second.view(np.uint32)) == row["bitwise_equal"]
        assert abs(maximum - row["max_absolute_logit_difference"]) < 1e-12
        if "mean_absolute_logit_difference" in row:
            assert abs(np.abs(first.astype(np.float64) - second).mean() - row["mean_absolute_logit_difference"]) < 1e-12
        assert disagreements == row["argmax_disagreement_count"]
    maximum, rms, disagreements = differences(arrays["bare_r0_bf16"], arrays["bare_r0_fp32"])
    audit = {"status": "pass", "source_sha256": digest(Path(__file__)),
             "input_result_sha256": digest(result_path), "input_logits_sha256": digest(logits_path),
             "cases_checked": len(names), "pairwise_comparisons_checked": len(pairs),
             "validation_images_checked": count,
             "baseline_bf16_vs_fp32": {"max_absolute_logit_difference": maximum,
                                        "rms_logit_difference": rms, "argmax_disagreement_count": disagreements},
             "scope": "Recomputed all per-case and pairwise logit/argmax/probe-accuracy statistics; verified validation image IDs/labels, source snapshots, checkpoint/head provenance, and saved-logit hash. Does not independently re-render images or rerun the model."}
    if count == len(manifest["splits"]["val"]):
        actual = int(np.sum(arrays["bare_r0_bf16"].argmax(1) == labels))
        expected = baseline["best_validation"]["correct"]
        audit["full_validation_bare_bf16_matches_training_record"] = actual == expected
        audit["full_validation_bare_bf16_correct"] = actual
        audit["training_record_best_validation_correct"] = expected
    magnitude_errors = result.get("initial_cpu_magnitude_vs_gpu_norm", [])
    for row in magnitude_errors:
        values = row["by_module"].values()
        assert all(math.isfinite(value) and value >= 0 for value in values)
        assert row["max_absolute_ratio_error"] == max(values, default=0.0)
        assert len(row["by_module"]) == (48 if row["method"].startswith("dora") else 0)
    if magnitude_errors:
        assert len(magnitude_errors) == len(cases)
        audit["max_initial_cpu_magnitude_vs_gpu_norm_ratio_error"] = max(row["max_absolute_ratio_error"] for row in magnitude_errors)
    mechanism_path = root / "mechanism.json"
    if mechanism_path.exists():
        mechanism = json.loads(mechanism_path.read_text())
        raw_path = root / "mechanism_logits.npz"
        raw = dict(np.load(raw_path, allow_pickle=False))
        probe_count = mechanism["count"]
        assert probe_count == min(128, count) and np.array_equal(raw.pop("labels"), labels[:probe_count])
        assert mechanism["source_control_result_sha256"] == digest(result_path)
        assert mechanism["logits_sha256"] == digest(raw_path)
        assert all(array.shape == (probe_count, 100) and array.dtype == np.float32 and np.isfinite(array).all() for array in raw.values())
        for precision in ("fp32", "bf16"):
            for reference in ("bare_r0", "lora_r8", "dora_r8"):
                name = f"{reference}_{precision}"
                assert np.array_equal(raw[name].view(np.uint32), arrays[name][:probe_count].view(np.uint32))
        assert len(mechanism["comparisons"]) == 6
        for row in mechanism["comparisons"]:
            candidate, reference = raw[f"dora_gpu_m_{row['precision']}"], raw[row["reference"]]
            maximum, _, disagreements = differences(candidate, reference)
            assert np.array_equal(candidate.view(np.uint32), reference.view(np.uint32)) == row["bitwise_equal"]
            assert abs(maximum - row["max_absolute_logit_difference"]) < 1e-12
            assert abs(np.abs(candidate.astype(np.float64) - reference).mean() - row["mean_absolute_logit_difference"]) < 1e-12
            assert disagreements == row["argmax_disagreement_count"]
        for label in ("initial", "recalibrated"):
            values = mechanism[f"{label}_ratio_error_by_module"]
            assert len(values) == 48 and all(math.isfinite(value) and value >= 0 for value in values.values())
            assert max(values.values()) == mechanism[f"{label}_ratio_error_max"]
        audit["mechanism"] = {"status": "pass", "input_result_sha256": digest(mechanism_path),
                              "input_logits_sha256": digest(raw_path), "comparisons_checked": 6,
                              "scope": "Recomputed six paired logit comparisons and verified original-probe equality to full-control logits, source linkage, and reported magnitude-ratio summary consistency; magnitude values themselves require a model rerun."}
    if write_audit:
        (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--no-write", action="store_true", help="Audit without modifying an extracted artifact tree")
    args = parser.parse_args()
    print(json.dumps(validate(args.root, write_audit=not args.no_write), indent=2))
