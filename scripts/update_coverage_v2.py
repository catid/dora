#!/usr/bin/env python3
"""Update coverage for the corrected v2 run plan from immutable run artifacts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def candidates(run, roots):
    rid = run["run_id"]
    for root in roots:
        root = Path(root)
        if run.get("stage") == "B_hpo":
            yield root / rid
        elif run.get("stage") in {"B_tuning", "B_screen"}:
            # Later launchers may use a descriptive suffix; exact IDs are
            # preferred, then stage/method/seed wildcard matching.
            yield root / rid
            method, seed = run.get("method_id"), run.get("seed")
            if run.get("stage") == "B_tuning":
                yield from root.glob(f"{method}_c*_seed{seed}")
            else:
                yield from root.glob(f"{method}_seed{seed}")
        else:
            yield root / rid


def inspect(run, roots):
    status, reason = "pending", ""
    for d in candidates(run, roots):
        metadata = d / "metadata.json"
        launcher = d / "launcher_status.json"
        if metadata.exists():
            try:
                value = json.loads(metadata.read_text())
            except Exception as exc:
                return "failed", f"invalid_metadata:{exc!r}"
            if value.get("status") == "complete":
                return "complete", ""
            if value.get("status") in {"failed", "error"}:
                status, reason = "failed", str(value.get("failure_reason", "metadata_failed"))
        if launcher.exists():
            try:
                value = json.loads(launcher.read_text())
            except Exception as exc:
                return "failed", f"invalid_launcher_status:{exc!r}"
            if value.get("status") == "running":
                # An in-flight run is still pending from the study's terminal
                # status vocabulary; it must not be counted as a failure.
                status, reason = "pending", "running"
            elif value.get("status") == "failed":
                status, reason = "failed", str(value.get("returncode", "launcher_failed"))
    return status, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=Path("artifacts/run_plan.json"))
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/coverage_v2.json"))
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = []
    for run in plan["runs"]:
        status, reason = inspect(run, args.roots)
        rows.append({"run_id": run["run_id"], "stage": run.get("stage"),
                     "method_id": run.get("method_id"), "status": status,
                     "failure_reason": reason})
    counts = {key: sum(x["status"] == key for x in rows)
              for key in ("pending", "complete", "failed", "blocked", "not_in_scope")}
    output = {"schema": "adaptive-rank-coverage-v2", "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
              "roots": [str(x) for x in args.roots], "counts": {"total": len(rows), **counts}, "runs": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["counts"]))


if __name__ == "__main__":
    main()
