#!/usr/bin/env python3
"""Summarize the opt-in IniLoRA-alpha x AdaLoRA-SVD experiment.

The script consumes immutable HPO/tuning/screening metadata and never selects
from test data.  It also accepts the existing Stage-C confirmation CSV so the
exploratory hybrid can be compared with already-frozen references.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def records(root: Path, stage: str):
    rows = []
    for path in sorted(root.glob("*/metadata.json")):
        try:
            md = json.loads(path.read_text())
        except Exception:
            continue
        if md.get("status") != "complete":
            continue
        score = md.get("dev_tune_endpoint_group_balanced_nll")
        if score is None:
            continue
        cfg_name = "hpo_config.json" if stage == "hpo" else "tuning_config.json"
        cfg_path = path.parent / cfg_name
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        rows.append({"stage": stage, "method": md.get("method"), "seed": md.get("seed"),
                     "nll": float(score), "lr": cfg.get("lr"), "scale": cfg.get("scale"),
                     "penalty": cfg.get("penalty", 0.0), "path": str(path.parent)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpo", type=Path, required=True)
    ap.add_argument("--tuning", type=Path, required=True)
    ap.add_argument("--screening", type=Path, required=True)
    ap.add_argument("--screening-extra", type=Path, action="append", default=[])
    ap.add_argument("--confirmation", type=Path, default=Path("reports/confirmation_summary.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    hpo, tuning = records(args.hpo, "hpo"), records(args.tuning, "tuning")
    screening = records(args.screening, "screening")
    for extra in args.screening_extra:
        screening.extend(records(extra, "screening"))
    summary = []
    for method in sorted({r["method"] for r in hpo + tuning + screening}):
        hs = [r for r in hpo if r["method"] == method]
        ts = [r for r in tuning if r["method"] == method]
        ss = [r for r in screening if r["method"] == method]
        best = min(hs, key=lambda r: (r["nll"], r["lr"], r["scale"], r["penalty"])) if hs else {}
        tmean = statistics.mean(r["nll"] for r in ts) if ts else None
        svals = [r["nll"] for r in ss]
        summary.append({"method": method, "hpo_trials": len(hs),
                        "hpo_best_nll": best.get("nll"), "hpo_best_lr": best.get("lr"),
                        "hpo_best_scale": best.get("scale"), "hpo_best_penalty": best.get("penalty"),
                        "tuning_runs": len(ts), "tuning_mean_nll": tmean,
                        "screening_runs": len(ss),
                        "screening_mean_nll": statistics.mean(svals) if svals else None,
                        "screening_sd_nll": statistics.stdev(svals) if len(svals) > 1 else None})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"hpo": hpo, "tuning": tuning, "screening": screening,
                                    "summary": summary}, indent=2) + "\n")
    with (args.out.with_suffix(".csv")).open("w", newline="") as f:
        fields = list(summary[0]) if summary else ["method"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    if args.confirmation.exists():
        with args.confirmation.open(newline="") as f:
            refs = list(csv.DictReader(f))
        by_method = {r.get("method"): r for r in refs}
        for row in summary:
            ref = by_method.get(row["method"])
            if ref:
                row["stageC_mean_nll"] = float(ref["mean_dev_group_balanced_nll"])
        args.out.write_text(json.dumps({"hpo": hpo, "tuning": tuning, "screening": screening,
                                        "summary": summary}, indent=2) + "\n")
    print(json.dumps({"hpo": len(hpo), "tuning": len(tuning), "screening": len(screening),
                      "methods": sorted({r["method"] for r in hpo + tuning + screening}),
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
