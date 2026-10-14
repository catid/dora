#!/usr/bin/env python3
"""Build machine-readable summaries from corrected v2 run directories."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rankstudy.registry import build_registry


FIELDS = ["run_id", "method", "stage", "seed", "lr", "scale", "penalty",
          "updates", "input_tokens", "supervised_tokens", "elapsed_s",
          "dev_group_balanced_nll", "dev_token_weighted_nll", "status",
          "source_revision", "path"]


def collect(roots):
    rows = []
    for root in roots:
        root = Path(root)
        for metadata_path in sorted(root.glob("**/metadata.json")):
            # A worker writes metadata immediately before its launcher records
            # terminal status.  Do not promote that narrow interval into a
            # completed result in live reports; coverage treats it as
            # pending/running until the launcher exits cleanly.
            launcher_path = metadata_path.parent / "launcher_status.json"
            if launcher_path.exists():
                try:
                    launcher = json.loads(launcher_path.read_text())
                except Exception:
                    launcher = {}
                if launcher.get("status") == "running":
                    continue
            try:
                md = json.loads(metadata_path.read_text())
            except Exception as exc:
                rows.append({"run_id": metadata_path.parent.name, "status": "invalid",
                             "path": str(metadata_path.parent), "error": repr(exc)})
                continue
            method = md.get("method", metadata_path.parent.name)
            stage = ("hpo" if "hpo" in metadata_path.parent.name else
                     "pipeline" if md.get("pilot_steps") or md.get("wide_steps") else "native")
            rows.append({"run_id": metadata_path.parent.name, "method": method,
                         "stage": stage, "seed": md.get("seed"), "lr": md.get("lr"),
                         "scale": md.get("scale"), "penalty": md.get("penalty", 0.0),
                         "updates": md.get("updates"), "input_tokens": md.get("input_tokens"),
                         "supervised_tokens": md.get("supervised_tokens"),
                         "elapsed_s": md.get("elapsed_s"),
                         "dev_group_balanced_nll": md.get("dev_tune_endpoint_group_balanced_nll",
                                                          md.get("group_balanced_nll")),
                         "dev_token_weighted_nll": md.get("dev_tune_endpoint_token_weighted_nll",
                                                           md.get("token_weighted_nll")),
                         "status": md.get("status", "unknown"),
                         "source_revision": md.get("source_revision"),
                         "path": str(metadata_path.parent)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/v2_hpo.csv"))
    ap.add_argument("--summary", type=Path, default=Path("reports/v2_summary.json"))
    args = ap.parse_args()
    rows = collect(args.roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    completed = [r for r in rows if r.get("status") == "complete" and r.get("dev_group_balanced_nll") is not None]
    best = {}
    for r in completed:
        method = r["method"]
        if method not in best or float(r["dev_group_balanced_nll"]) < float(best[method]["dev_group_balanced_nll"]):
            best[method] = r
    summary = {"schema": "adaptive-rank-v2-report-v1", "roots": [str(x) for x in args.roots],
               "registered_methods": [x.id for x in build_registry()],
               "rows": len(rows), "complete_rows": len(completed),
               "best_by_method": best}
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "complete": len(completed), "methods": len(best)}))


if __name__ == "__main__":
    main()
