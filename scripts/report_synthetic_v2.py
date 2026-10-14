#!/usr/bin/env python3
"""Render the frozen-matrix control artifact without hand-entered values."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("artifacts/synthetic_controls_v2.json"))
    ap.add_argument("--csv", type=Path, default=Path("reports/synthetic_controls_v2.csv"))
    ap.add_argument("--markdown", type=Path, default=Path("reports/synthetic_controls_v2.md"))
    args = ap.parse_args()
    rows = json.loads(args.input.read_text())
    flat = []
    for row in rows:
        for result in row["results"]:
            flat.append({
                "seed": row["seed"], "init": row["init"],
                "anisotropic": row["anisotropic"], "noise": row["noise"],
                "steps": row["steps"], **result,
            })
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)

    grouped = defaultdict(list)
    for row in flat:
        grouped[(row["init"], row["anisotropic"], row["noise"], row["budget_slots"])].append(row["energy_selected_mse"])
    lines = [
        "# Frozen-matrix Stage A control",
        "",
        "This report is generated from `artifacts/synthetic_controls_v2.json`. It uses two seeds, eight independent 16×16 modules, teacher ranks `[0,0,1,1,2,4,8,16]`, 100 Adam steps, slot budgets 8/16/32, isotropic and anisotropic Gaussian inputs, and clean/noisy targets.",
        "",
        "Values are mean MSE over the eight modules and the two seeds in each cell; this is a mechanistic control, not an LLM quality result.",
        "",
        "| Init | Anisotropic | Noise | Budget | Mean MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped, key=lambda x: (x[0], x[1], x[2], x[3])):
        vals = grouped[key]
        lines.append(f"| {key[0]} | {str(bool(key[1])).lower()} | {str(bool(key[2])).lower()} | {key[3]} | {sum(vals)/len(vals):.6f} |")
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(flat), "csv": str(args.csv), "markdown": str(args.markdown)}))


if __name__ == "__main__":
    main()
