#!/usr/bin/env python3
"""Summarize Stage-E HPO/tuning/screening receipts without selecting on tests."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def rows(root: Path, stage: str):
    output = []
    for path in sorted(root.glob("*/metadata.json")):
        md = json.loads(path.read_text())
        cfg = md.get("config", {})
        output.append({
            "run": path.parent.name,
            "method": md.get("method"),
            "status": md.get("status"),
            "stage": stage,
            "seed": md.get("seed", cfg.get("seed")),
            "lr": cfg.get("lr"), "scale": cfg.get("scale"), "penalty": cfg.get("penalty", 0.0),
            "targets": ",".join(cfg.get("targets", [])),
            "model": cfg.get("model"),
            "updates_finished": md.get("updates_finished"), "updates": cfg.get("updates"),
            "input_tokens": md.get("input_tokens"), "supervised_tokens": md.get("supervised_tokens"),
            "charged_gpu_seconds": md.get("charged_gpu_seconds"),
            "dev_nll": md.get("dev_tune_endpoint_group_balanced_nll"),
            "dev_token_nll": md.get("dev_tune_endpoint_token_weighted_nll"),
            "failure_reason": md.get("failure_reason"),
        })
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stage", required=True, choices=["hpo", "tuning", "screening"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args()
    data = rows(args.root, args.stage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(data[0]) if data else ["run", "method", "status", "stage"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    complete = [r for r in data if r["status"] == "complete"]
    by = defaultdict(list)
    for r in complete: by[r["method"]].append(r)
    summary = {}
    for method, vals in sorted(by.items()):
        scores = [r["dev_nll"] for r in vals if r["dev_nll"] is not None]
        summary[method] = {"complete": len(vals), "nll_values": scores,
                           "nll_mean": statistics.mean(scores) if scores else None,
                           "nll_sd": statistics.stdev(scores) if len(scores) > 1 else None}
    payload = {"root": str(args.root), "stage": args.stage, "jobs": len(data),
               "status_counts": dict(Counter(r["status"] for r in data)),
               "summary_by_method": summary, "rows": data}
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [f"# Stage-E {args.stage} report", "", f"Jobs: {len(data)}.", "",
             "| Method | Complete | Mean endpoint dev NLL | SD |", "|:--|--:|--:|--:|"]
    for method, item in sorted(summary.items()):
        mean = "" if item["nll_mean"] is None else f"{item['nll_mean']:.6f}"
        sd = "" if item["nll_sd"] is None else f"{item['nll_sd']:.6f}"
        lines.append(f"| `{method}` | {item['complete']} | {mean} | {sd} |")
    lines += ["", "Selection is based only on the declared development endpoint. "
              "Official GSM8K/MATH-500 results are intentionally absent until the selection freeze.", ""]
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text("\n".join(lines))
    print(json.dumps({"jobs": len(data), "status_counts": dict(Counter(r["status"] for r in data)),
                      "csv": str(args.out), "json": str(args.json), "markdown": str(args.markdown)}))


if __name__ == "__main__": main()
