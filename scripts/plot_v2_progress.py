#!/usr/bin/env python3
"""Plot completed corrected-v2 rows without selecting from incomplete methods."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    with args.csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "complete":
                continue
            value = row.get("dev_group_balanced_nll")
            if value not in (None, ""):
                grouped[row["method"]].append(float(value))
    if not grouped:
        raise SystemExit("no complete rows with dev_group_balanced_nll")

    best = sorted((min(values), method) for method, values in grouped.items())
    labels = [method for _, method in best]
    values = [value for value, _ in best]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 4.5))
    ax.bar(labels, values, color="#4472c4")
    ax.set_ylabel("best completed group-balanced dev NLL")
    ax.set_title("Corrected v2 HPO progress (incomplete methods omitted)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
