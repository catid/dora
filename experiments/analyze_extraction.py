"""CPU-only paired bootstrap for the frozen ViGGO extraction benchmark results.

Run after all four methods finish for seeds 42, 43, and 44::

    python -m experiments.analyze_extraction

The bootstrap resamples training seeds and shared test examples with replacement.
A draw uses the same seed and example indices for both methods, preserving the
pairing of repeated predictions on the same held-out examples. These exploratory
intervals describe one fixed benchmark with only three training seeds; they do
not establish broad statistical or task-independent superiority.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ("lora", "dora", "nora", "dora_nora")
SEEDS = (42, 43, 44)
COMPARISONS = (("nora", "lora"), ("dora", "lora"), ("dora_nora", "nora"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_correctness(root):
    """Recompute exact match from raw text and enforce pairing and provenance."""
    results_path = root / "results.json"
    payload = json.loads(results_path.read_text())
    results = payload["results"]
    expected_runs = {(method, seed) for method in METHODS for seed in SEEDS}
    found_runs = {(row["method"], row["seed"]) for row in results}
    if len(results) != len(expected_runs) or found_runs != expected_runs:
        raise ValueError(f"Need all 12 final runs; currently have {len(results)}")
    split_path = root / "split_manifest.json"
    manifest = json.loads(split_path.read_text())
    test = manifest["test"]
    if len(test) != 256 or len({item["id"] for item in test}) != 256:
        raise ValueError("Expected exactly 256 unique official test examples")
    provenance = payload["manifest"]["dataset"]
    if sha256(split_path) != provenance["split_manifest_sha256"]:
        raise ValueError("Split manifest SHA256 differs from training provenance")
    ids = [item["id"] for item in test]
    scores = np.zeros((len(METHODS), len(SEEDS), len(test)), dtype=np.bool_)
    hashes = {"results.json": sha256(results_path), "split_manifest.json": sha256(split_path)}
    for method_index, method in enumerate(METHODS):
        for seed_index, seed in enumerate(SEEDS):
            path = root / f"{method}_seed{seed}" / "predictions.jsonl"
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            if [row["id"] for row in predictions] != ids:
                raise ValueError(f"Test example order or IDs differ: {path}")
            for index, (prediction, expected) in enumerate(zip(predictions, test)):
                if prediction["gold"] != expected["target"] or prediction["utterance"] != expected["utterance"]:
                    raise ValueError(f"Test input/target differs: {path}, item {index}")
                try:
                    parsed = json.loads(prediction["text"].strip())
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if parsed != prediction["parsed"]:
                    raise ValueError(f"Archived parse differs from generated text: {path}, item {index}")
                # The manifest's nonempty gold JSON objects are valid targets.
                # Equality with gold therefore also implies schema validity.
                exact = parsed == expected["target"]
                if exact != prediction["exact_match"]:
                    raise ValueError(f"Archived exact-match flag differs: {path}, item {index}")
                scores[method_index, seed_index, index] = exact
            run = next(row for row in results if row["method"] == method and row["seed"] == seed)
            if run["test"]["count"] != len(test):
                raise ValueError(f"Wrong reported test count: {method}, seed {seed}")
            recomputed = scores[method_index, seed_index].mean()
            if abs(recomputed - run["test"]["exact_match"]) > 1e-14:
                raise ValueError(f"Reported exact match differs: {method}, seed {seed}")
            hashes[str(path.relative_to(root))] = sha256(path)
    return scores, ids, hashes


def paired_bootstrap(scores, repetitions, rng_seed):
    differences = np.stack([
        scores[METHODS.index(left)].astype(np.float64) - scores[METHODS.index(right)]
        for left, right in COMPARISONS
    ])
    rng = np.random.default_rng(rng_seed)
    draws = np.empty((repetitions, len(COMPARISONS)))
    for iteration in range(repetitions):
        seed_indices = rng.integers(0, scores.shape[1], size=scores.shape[1])
        example_indices = rng.integers(0, scores.shape[2], size=scores.shape[2])
        # Test examples are shared across seeds, so use one matched example
        # resample across all selected seeds rather than treating repeats as
        # independent new test examples.
        draws[iteration] = differences[:, seed_indices].mean(axis=1)[:, example_indices].mean(axis=1) * 100
    comparisons = {}
    for index, (left, right) in enumerate(COMPARISONS):
        comparisons[f"{left}_minus_{right}"] = {
            "left": left, "right": right,
            "mean_exact_match_difference_percentage_points": float(differences[index].mean() * 100),
            "per_seed_difference_percentage_points": {
                str(seed): float(differences[index, seed_index].mean() * 100)
                for seed_index, seed in enumerate(SEEDS)
            },
            "percentile_95_interval": np.quantile(draws[:, index], [0.025, 0.975]).tolist(),
        }
    return comparisons, draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/var/tmp/dora-bench/runs/extraction"))
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.replicates <= 0:
        raise ValueError("replicates must be positive")
    scores, ids, hashes = load_correctness(args.input)
    comparisons, draws = paired_bootstrap(scores, args.replicates, args.rng_seed)
    arrays_path = args.input / "paired_bootstrap_samples.npz"
    np.savez_compressed(arrays_path, correctness=scores, example_ids=np.asarray(ids),
                        methods=np.asarray(METHODS), seeds=np.asarray(SEEDS),
                        comparisons=np.asarray([f"{a}_minus_{b}" for a, b in COMPARISONS]),
                        difference_draws_percentage_points=draws)
    output = {
        "task": "ViGGO structured extraction", "metric": "JSON object exact match",
        "seeds": list(SEEDS), "test_examples": len(ids), "final_runs": 12,
        "replicates": args.replicates, "rng_seed": args.rng_seed,
        "method": "Paired crossed bootstrap: resample three matched training seeds and 256 shared test examples with replacement; percentile 95% intervals.",
        "caveat": "Exploratory intervals from only three training seeds and one fixed held-out test set. They do not establish broad statistical or task-independent superiority; no test-driven tuning was performed.",
        "raw_predictions_independently_verified": True,
        "method_exact_match_percent": {
            method: {"mean": float(scores[index].mean() * 100),
                     "per_seed": {str(seed): float(scores[index, seed_index].mean() * 100)
                                  for seed_index, seed in enumerate(SEEDS)}}
            for index, method in enumerate(METHODS)
        },
        "comparisons": comparisons,
        "input_sha256": hashes,
        "analysis_script_sha256": sha256(__file__),
        "numpy_version": np.__version__,
        "sample_arrays": arrays_path.name,
        "sample_arrays_sha256": sha256(arrays_path),
    }
    output_path = args.input / "paired_bootstrap.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
