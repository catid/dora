#!/usr/bin/env python3
"""Select up to two HPO configurations per registered method by dev_tune NLL."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rankstudy.registry import build_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/hpo_selected_v2.json"))
    args = ap.parse_args()
    by_method = {m.id: [] for m in build_registry()}
    for p in sorted(args.root.glob("*/metadata.json")):
        try:
            md = json.loads(p.read_text())
        except Exception:
            continue
        if md.get("status") != "complete" or md.get("dev_tune_endpoint_group_balanced_nll") is None:
            continue
        if md.get("method") in by_method:
            by_method[md["method"]].append({**md, "path": str(p.parent)})
    selected = {}
    for method, rows in by_method.items():
        selected[method] = sorted(rows, key=lambda x: (
            float(x["dev_tune_endpoint_group_balanced_nll"]),
            float(x.get("lr", 0.0)), float(x.get("scale", 0.0)), str(x["path"])))[:2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "adaptive-rank-v2-hpo-selection-v1",
                                    "root": str(args.root), "selected": selected}, indent=2) + "\n")
    print(json.dumps({"methods_with_results": sum(bool(v) for v in selected.values()),
                      "registered_methods": len(selected),
                      "selected_rows": sum(len(v) for v in selected.values())}))


if __name__ == "__main__":
    main()
