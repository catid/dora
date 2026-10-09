"""Validation-only optimizer diagnostic at matched factor LR and training seed.

Uses existing full-budget trials only. No new fitting, test-data selection, or
comparison of independently selected base LRs. Counts are descriptive, not tests
of significance across independent tasks or training seeds.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


REFERENCE = "dora_nora"
CANDIDATE = "dora_nora_mlr"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    root = Path(root)
    inputs, rows, pending = {}, [], []

    def read(path, fallback=None):
        if not path.exists():
            return fallback
        inputs[str(path.relative_to(root))] = digest(path)
        return json.loads(path.read_text())

    def compare(task, condition, rank, references, candidates, rate_key, score,
                metric, higher, budget_key, selected_key=None, secondary=None):
        assert len(references) <= 4 and len(candidates) <= 4
        reference_rates = {row[rate_key]: row for row in references}
        assert len(reference_rates) == len(references)
        count = 0
        for candidate in candidates:
            rate = candidate[rate_key]
            if rate not in reference_rates:
                continue
            reference = reference_rates[rate]
            assert reference["method"] == REFERENCE and candidate["method"] == CANDIDATE
            assert reference["seed"] == candidate["seed"]
            assert reference["rank"] == candidate["rank"] == rank
            assert reference[budget_key] == candidate[budget_key]
            assert reference["magnitude_lr_multiplier"] == 1
            assert candidate["magnitude_lr_multiplier"] in (0.1, 0.01)
            if "head_learning_rate" in reference:
                assert reference["head_learning_rate"] == candidate["head_learning_rate"]
            if "frozen_sha256" in reference:
                assert reference["frozen_sha256"] == candidate["frozen_sha256"]
            reference_value, candidate_value = score(reference), score(candidate)
            delta = candidate_value - reference_value
            row = {"task": task, "condition": condition, "rank": rank,
                   "seed": reference["seed"], "factor_learning_rate": rate,
                   "magnitude_lr_multiplier": candidate["magnitude_lr_multiplier"],
                   "metric": metric, "higher_is_better": higher,
                   "reference_validation": reference_value, "candidate_validation": candidate_value,
                   "candidate_minus_reference": delta, "benefit": delta if higher else -delta,
                   "budget": {budget_key: reference[budget_key]},
                   "reference_method": REFERENCE, "candidate_method": CANDIDATE}
            if selected_key:
                row["reference_selected_checkpoint"] = reference[selected_key]
                row["candidate_selected_checkpoint"] = candidate[selected_key]
            if secondary:
                row["secondary_validation"] = {
                    "metric": "cross_entropy", "higher_is_better": False,
                    "reference": secondary(reference), "candidate": secondary(candidate)}
            rows.append(row)
            count += 1
        if count != 4:
            pending.append({"task": task, "condition": condition, "rank": rank,
                            "matched_pairs_available": count, "matched_pairs_expected": 4})

    teacher = root / "teacher_compact"
    manifest = read(teacher / "manifest.json", {})
    if not manifest:
        pending.append({"task": "teacher", "reason": "Compact teacher manifest unavailable"})
    for cell, specification in manifest.get("cells", {}).items():
        rank = specification["generator_arguments"][0]
        trials = {method: read(teacher / "selections" / cell / f"{method}.json", {}).get("trials", [])
                  for method in (REFERENCE, CANDIDATE)}
        compare("teacher", cell, rank, trials[REFERENCE], trials[CANDIDATE], "learning_rate",
                lambda row: row["validation"]["relative_to_frozen"], "relative_validation_mse", False, "steps")

    for rank in (2, 8):
        trials = {}
        for method in (REFERENCE, CANDIDATE):
            paths = sorted((root / "retrieval" / "tuning").glob(f"{method}_r{rank}_*/result.json"))
            candidates = [read(path) for path in paths]
            assert all(set(row["evaluation"]) == {"nfcorpus"}
                       and "test" not in row["evaluation"]["nfcorpus"] for row in candidates)
            trials[method] = [{**row, "epochs": len(row["training"]["epochs"])} for row in candidates]
        compare("retrieval", "nfcorpus", rank, trials[REFERENCE], trials[CANDIDATE], "learning_rate",
                lambda row: row["evaluation"]["nfcorpus"]["validation"]["ndcg_at_10"],
                "validation_ndcg_at_10", True, "epochs")

    trials = {method: read(root / "cogs" / method / "tuning.json", {}).get("trials", [])
              for method in (REFERENCE, CANDIDATE)}
    compare("cogs", "iid_validation", 8, trials[REFERENCE], trials[CANDIDATE], "lr",
            lambda row: row["validation_nll"], "validation_target_token_nll", False, "steps", "selected_step")

    for rank in (2, 8):
        pilots = read(root / "vision" / f"rank{rank}_run" / f"rank_{rank}" / "pilot_results.json", [])
        trials = {method: [row for row in pilots if row["method"] == method] for method in (REFERENCE, CANDIDATE)}
        for trial in trials[REFERENCE] + trials[CANDIDATE]:
            assert trial["test"] is None and not trial["evaluate_test"]
        compare("aircraft", "validation", rank, trials[REFERENCE], trials[CANDIDATE], "adapter_learning_rate",
                lambda row: row["best_validation"]["macro_class_accuracy"], "validation_macro_accuracy", True,
                "epochs", "best_epoch", lambda row: row["best_validation"]["cross_entropy"])

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["metric"], row["magnitude_lr_multiplier"])].append(row)
    summaries = []
    for (task, metric, multiplier), records in sorted(grouped.items()):
        benefits = [row["benefit"] for row in records]
        summaries.append({"task": task, "metric": metric, "magnitude_lr_multiplier": multiplier,
                          "pairs": len(records), "strictly_better": sum(value > 0 for value in benefits),
                          "strictly_worse": sum(value < 0 for value in benefits),
                          "exact_ties": sum(value == 0 for value in benefits),
                          "median_reference_validation": statistics.median(row["reference_validation"] for row in records),
                          "median_candidate_validation": statistics.median(row["candidate_validation"] for row in records),
                          "median_benefit": statistics.median(benefits)})
    return {"analysis": "validation-only matched-factor-LR magnitude optimizer diagnostic",
            "complete": not pending, "pending": pending, "comparisons": rows, "descriptive_summary": summaries,
            "input_sha256": inputs, "analysis_source_sha256": digest(__file__),
            "protocol": "Compare the original combination (magnitude multiplier 1) with multipliers 0.1/0.01 at the same factor learning rate, rank, training seed and full trial budget. Teacher/retrieval use final-horizon validation; COGS/Aircraft use the best validation checkpoint within their common budget. Only validation trial artifacts are loaded; no test metric is used and no searches are altered.",
            "limitations": "Exploratory optimizer diagnostic at one tuning seed. Shared-LR cases and synthetic cells are not independent statistical replicates; no confidence interval or test-superiority inference is made. Strict better/worse counts include arbitrarily small floating-point differences. Candidate methods may choose different validation-best checkpoints. Joint-search conclusions remain conditional on tested grids and horizons."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/var/tmp/dora-bench/round2"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    output = args.output or args.root / "matched_magnitude_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": result["complete"], "pending": result["pending"],
                      "descriptive_summary": result["descriptive_summary"]}, indent=2))
