#!/usr/bin/env python3
"""Aggregate separate-process common-export audit receipts.

The checker intentionally uses strict BF16 logit tolerances.  A receipt can
therefore fail those tolerances while still proving that the export reloads,
reconstructs its factors, and preserves probe NLL.  Keep those cases separate
from structural audit failures instead of reducing them to one boolean.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def classify(audit: dict) -> str:
    checks = audit.get("checks", {})
    structural = all(checks.get(k, False) for k in (
        "same_original_backbone", "four_fresh_processes",
        "native_reload_bitwise_equal", "factor_reconstruction", "exact_nll"))
    strict = all(checks.get(k, False) for k in (
        "exact_relative_rms", "exact_absolute_logits"))
    if structural and strict:
        return "passed"
    if structural and not strict:
        return "bf16_logit_rounding_failure"
    return "structural_failure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for audit_path in sorted(args.root.glob("*/audit.json")):
        run = audit_path.parent.name
        audit = json.loads(audit_path.read_text())
        checks = audit.get("checks", {})
        exact = audit.get("comparisons", {}).get("exact", {})
        truncated = audit.get("comparisons", {}).get("truncated", {})
        factor = audit.get("factor_relative_errors", {})
        launcher_path = args.root / f"{run}.launcher_status.json"
        launcher = json.loads(launcher_path.read_text()) if launcher_path.exists() else {}
        rows.append({
            "run": run,
            "audit_status": audit.get("status"),
            "classification": classify(audit),
            "launcher_status": launcher.get("status"),
            "launcher_returncode": launcher.get("returncode"),
            "same_original_backbone": checks.get("same_original_backbone"),
            "four_fresh_processes": checks.get("four_fresh_processes"),
            "native_reload_bitwise_equal": checks.get("native_reload_bitwise_equal"),
            "factor_reconstruction": checks.get("factor_reconstruction"),
            "exact_relative_rms_check": checks.get("exact_relative_rms"),
            "exact_absolute_logits_check": checks.get("exact_absolute_logits"),
            "exact_nll_check": checks.get("exact_nll"),
            "exact_logits_relative_rms": exact.get("logits_relative_rms"),
            "exact_logits_max_absolute_difference": exact.get("logits_max_absolute_difference"),
            "exact_group_nll_difference": exact.get("group_nll_difference"),
            "exact_token_nll_difference": exact.get("token_nll_difference"),
            "truncated_logits_relative_rms": truncated.get("logits_relative_rms"),
            "truncated_logits_max_absolute_difference": truncated.get("logits_max_absolute_difference"),
            "truncated_group_nll_difference": truncated.get("group_nll_difference"),
            "truncated_token_nll_difference": truncated.get("token_nll_difference"),
            "factor_relative_error_max": max(factor.values()) if factor else None,
            "exact_refactor_diagnostic_above_budget": audit.get("exact_refactor_diagnostic_above_budget"),
            "export": audit.get("export"),
        })

    if not rows:
        raise SystemExit(f"no audit receipts under {args.root}")
    fields = list(rows[0])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    counts = Counter(row["classification"] for row in rows)
    payload = {
        "root": str(args.root),
        "jobs": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "rows": rows,
        "interpretation": {
            "bf16_logit_rounding_failure": (
                "Structural export checks and exact probe NLL passed, but the strict "
                "full-model BF16 logit tolerances failed. Preserve as a failure; "
                "do not relabel it as exact numerical equality."
            )
        },
    }
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Stage-D separate-process export audits",
        "",
        f"Audited {len(rows)} exports from `{args.root}`.",
        "",
        "The checker uses strict BF16 full-model logit tolerances. Structural "
        "checks and probe NLL are reported separately from BF16 rounding failures.",
        "",
        "| Classification | Count |",
        "|:--|--:|",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "| Run | Classification | Exact RMS | Exact max abs | Exact token NLL delta | Truncated token NLL delta | Factor max rel. error |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return "" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append("| `{run}` | `{classification}` | {rms} | {mx} | {nll} | {tnll} | {factor} |".format(
            run=row["run"], classification=row["classification"],
            rms=fmt("exact_logits_relative_rms"), mx=fmt("exact_logits_max_absolute_difference"),
            nll=fmt("exact_token_nll_difference"), tnll=fmt("truncated_token_nll_difference"),
            factor=fmt("factor_relative_error_max")))
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"jobs": len(rows), "classification_counts": dict(sorted(counts.items())),
                      "csv": str(args.csv), "json": str(args.json), "markdown": str(args.markdown)}))


if __name__ == "__main__":
    main()
