#!/usr/bin/env python3
"""Aggregate equal charged-GPU-second Stage-D pipeline receipts."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def stage_detail(entry):
    path = Path(entry.get("metadata", ""))
    md = json.loads(path.read_text()) if path.exists() else {}
    required = md.get("config", {}).get("stop_after")
    if required is None:
        required = md.get("config", {}).get("updates")
    return {
        "name": entry.get("name"),
        "updates_finished": md.get("updates_finished"),
        "updates_required": required,
        "charged_gpu_seconds": float(entry.get("charged_gpu_seconds", 0.0)),
        "input_tokens": entry.get("input_tokens"),
        "supervised_tokens": entry.get("supervised_tokens"),
        "checkpoint": entry.get("checkpoint"),
    }


def classify(md, stages):
    status = md.get("status")
    if status == "complete":
        return "complete"
    reason = md.get("failure_reason", "")
    if "pilot" in reason:
        return "infeasible_pilot_budget"
    if "source" in reason:
        return "infeasible_source_budget"
    if "recovery" in reason:
        return "infeasible_recovery_budget"
    if stages and any(s["name"] == "recovery" and s["updates_finished"] != s["updates_required"] for s in stages):
        return "infeasible_recovery_budget"
    return "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/metadata.json")):
        md = json.loads(path.read_text())
        run = path.parent.name
        stages = [stage_detail(x) for x in md.get("stages", [])]
        source = next((s for s in stages if s["name"] in {"source", "student"}), None)
        recovery = next((s for s in stages if s["name"] == "recovery"), None)
        # Use stage accounting as authoritative.  Older receipts from the
        # first launcher version double-counted partial recovery in the
        # top-level failure field; summing stages avoids that error while
        # preserving the original receipt unchanged.
        charged = sum(s["charged_gpu_seconds"] for s in stages) + float(md.get("transition_s", 0.0) or 0.0)
        status_path = path.parent / "launcher_status.json"
        launcher = json.loads(status_path.read_text()) if status_path.exists() else {}
        rows.append({
            "run": run,
            "role": run.split("_")[0],
            "method": md.get("method"),
            "seed": md.get("seed"),
            "repeat": run.rsplit("_rep", 1)[-1] if "_rep" in run else None,
            "pipeline_status": md.get("status"),
            "classification": classify(md, stages),
            "launcher_status": launcher.get("status"),
            "budget_s": md.get("budget_s", md.get("config", {}).get("gpu_budget_s")),
            "charged_gpu_seconds_stage_sum": charged,
            "charged_gpu_seconds_receipt": md.get("charged_gpu_seconds"),
            "transition_s": md.get("transition_s", 0.0),
            "source_updates_finished": source["updates_finished"] if source else None,
            "source_updates_required": source["updates_required"] if source else None,
            "source_gpu_seconds": source["charged_gpu_seconds"] if source else None,
            "recovery_updates_finished": recovery["updates_finished"] if recovery else None,
            "recovery_updates_required": recovery["updates_required"] if recovery else None,
            "recovery_gpu_seconds": recovery["charged_gpu_seconds"] if recovery else None,
            "input_tokens": sum(s["input_tokens"] or 0 for s in stages),
            "supervised_tokens": sum(s["supervised_tokens"] or 0 for s in stages),
            "dev_nll": md.get("dev_tune_endpoint_group_balanced_nll"),
            "failure_reason": md.get("failure_reason"),
        })
    if not rows:
        raise SystemExit(f"no pipeline receipts under {args.root}")
    fields = list(rows[0])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=fields); out.writeheader(); out.writerows(rows)
    counts = Counter(r["classification"] for r in rows)
    by_method = defaultdict(list)
    for row in rows: by_method[row["method"]].append(row)
    summary = {}
    for method, values in sorted(by_method.items()):
        good = [r for r in values if r["classification"] == "complete"]
        summary[method] = {
            "runs": len(values), "complete": len(good),
            "infeasible": len(values) - len(good),
            "complete_nll_values": [r["dev_nll"] for r in good if r["dev_nll"] is not None],
            "charged_seconds": [r["charged_gpu_seconds_stage_sum"] for r in values],
        }
    payload = {"root": str(args.root), "jobs": len(rows),
               "budget_s": rows[0]["budget_s"], "classification_counts": dict(sorted(counts.items())),
               "summary_by_method": summary, "rows": rows,
               "preliminary_root": "/var/tmp/dora-adaptive-rank/runs/v3_stageD_compute_20261016",
               "preliminary_root_note": "The first one-second-reserve calibration attempt is retained separately and excluded from this retry summary."}
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Stage-D equal charged-GPU-second comparison", "",
             f"The retry queue contains {len(rows)} jobs with a common cap of {rows[0]['budget_s']} charged GPU-seconds.",
             "The first one-second-reserve attempt remains preserved separately and is excluded.", "",
             "| Classification | Count |", "|:--|--:|"]
    for key, val in sorted(counts.items()): lines.append(f"| `{key}` | {val} |")
    lines += ["", "| Method | Runs | Complete | Infeasible | Complete NLL mean |", "|:--|--:|--:|--:|--:|"]
    for method, value in sorted(summary.items()):
        nll = value["complete_nll_values"]
        mean = "" if not nll else f"{sum(nll)/len(nll):.6f}"
        lines.append(f"| `{method}` | {value['runs']} | {value['complete']} | {value['infeasible']} | {mean} |")
    lines += ["", "A complete row finished its source, transition, and required ordinary-LoRA recovery within the cap. "
              "Infeasible rows retain their partial checkpoints and measured charged work; they are not quality scores.", ""]
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text("\n".join(lines))
    print(json.dumps({"jobs": len(rows), "classification_counts": dict(sorted(counts.items())),
                      "csv": str(args.csv), "json": str(args.json), "markdown": str(args.markdown)}))


if __name__ == "__main__": main()
