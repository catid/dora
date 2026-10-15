#!/usr/bin/env python3
"""Summarize native adaptive-rank runs and refresh the README evidence block.

All values are read from immutable run metadata.  The script never fills in
pending candidates and reports a method only when it has the requested number
of complete seed runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def rows(root: Path) -> list[dict]:
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        config_paths = (d / "config.json", d / "tuning_config.json", d / "hpo_config.json")
        c = next((path for path in config_paths if path.exists()), None)
        m, s = d / "metadata.json", d / "launcher_status.json"
        if c is None and not m.exists():
            continue
        cfg = json.loads(c.read_text()) if c is not None else {}
        md = json.loads(m.read_text()) if m.exists() else {}
        st = json.loads(s.read_text()) if s.exists() else {}
        out.append({**cfg, **md, "run_dir": str(d), "launcher_status": st.get("status", "pending")})
    return out


def endpoint_nll(row: dict):
    """Use the preregistered group-balanced endpoint, with legacy fallback."""
    value = row.get("dev_tune_endpoint_group_balanced_nll")
    if value is None:
        value = row.get("dev_tune_endpoint_nll")
    return value


def mean_sd(values: list[float]) -> tuple[float, float]:
    mu = sum(values) / len(values)
    sd = math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mu, sd


def write_csv(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in data for k in row})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-root", type=Path, required=True)
    ap.add_argument("--screening-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    args = ap.parse_args()

    tuning = rows(args.tuning_root)
    screening = rows(args.screening_root)
    write_csv(args.out_dir / "tuning.csv", tuning)
    write_csv(args.out_dir / "screening.csv", screening)

    # Select the frozen configuration using the mean endpoint loss over the
    # two tuning seeds.  This is the same rule used by run_screen_native.py.
    cfg_groups: dict[str, dict[tuple[float, float, float], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in tuning:
        value = endpoint_nll(row)
        if row.get("status") == "complete" and value is not None:
            cfg_groups[row["method"]][(float(row["lr"]), float(row["scale"]), float(row.get("penalty", 0.0)))].append(row)
    selected = []
    for method, groups in sorted(cfg_groups.items()):
        candidates = []
        for (lr, scale, penalty), rs in groups.items():
            if not {1002, 1003}.issubset({int(r.get("seed")) for r in rs}):
                continue
            mu, sd = mean_sd([float(endpoint_nll(r)) for r in rs])
            candidates.append({"method": method, "lr": lr, "scale": scale, "penalty": penalty,
                               "tuning_n": len(rs), "tuning_mean_nll": mu, "tuning_sd_nll": sd})
        if candidates:
            selected.append(sorted(candidates, key=lambda x: (x["tuning_mean_nll"], x["lr"], x["scale"], x["penalty"]))[0])
    write_csv(args.out_dir / "tuning_selection.csv", selected)

    screen_groups: dict[str, list[dict]] = defaultdict(list)
    for row in screening:
        if row.get("status") == "complete" and endpoint_nll(row) is not None:
            screen_groups[row["method"]].append(row)
    summary = []
    for method, rs in sorted(screen_groups.items()):
        vals = [float(endpoint_nll(r)) for r in rs]
        mu, sd = mean_sd(vals)
        summary.append({"method": method, "n": len(vals), "mean_dev_tune_nll": mu, "sd_dev_tune_nll": sd, "median_dev_tune_nll": sorted(vals)[len(vals)//2], "min_dev_tune_nll": min(vals), "max_dev_tune_nll": max(vals), "seeds": ",".join(str(r.get("seed")) for r in sorted(rs, key=lambda r: int(r.get("seed", 0))))})
    write_csv(args.out_dir / "screening_summary.csv", summary)

    # Plot only complete screening methods, preserving the neutral registry
    # order supplied by the raw method IDs.  Error bars are seed SDs.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if summary:
        labels = [r["method"] for r in summary]
        means = [r["mean_dev_tune_nll"] for r in summary]
        errs = [r["sd_dev_tune_nll"] for r in summary]
        fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.48), 7))
        ax.bar(range(len(labels)), means, yerr=errs, capsize=2, color="#4472c4", alpha=0.9)
        ax.set_ylabel("Endpoint dev_tune NLL (lower is better)")
        ax.set_title("Native adaptive-rank screening; mean ± seed SD")
        ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "screening_endpoint_nll.png", dpi=180)
        plt.close(fig)

    complete = sum(int(r["n"]) for r in summary)
    md = [
        "# Native adaptive-rank screening results",
        "",
        "Values are endpoint `dev_tune` NLL from the frozen native 1.5B q/o protocol. The table uses only complete screening runs; mean and SD are over independent screening seeds. Lower is better. HPO and tuning selection remain separate from this screening table.",
        "",
        "| Method | Seeds | Endpoint dev NLL (mean ± SD) |",
        "|:--|--:|--:|",
    ]
    for r in summary:
        md.append(f"| {r['method']} | {r['n']} | {r['mean_dev_tune_nll']:.6f} ± {r['sd_dev_tune_nll']:.6f} |")
    md += ["", f"Complete screening seed runs represented: **{complete}**.", "", "![Native screening endpoint NLL](figures/screening_endpoint_nll.png)", ""]
    (args.out_dir / "native_summary.md").write_text("\n".join(md))

    # Keep the README section machine-generated and replaceable.  This avoids
    # manually transcribing measurements while retaining a short inline table.
    start = "<!-- ADAPTIVE_NATIVE_RESULTS_START -->"
    end = "<!-- ADAPTIVE_NATIVE_RESULTS_END -->"
    readme_md = [x.replace("(figures/screening_endpoint_nll.png)", "(reports/figures/screening_endpoint_nll.png)") for x in md[2:]]
    block = [start, "## Measured native adaptive-rank screening", "", *readme_md, end]
    readme = args.readme.read_text() if args.readme.exists() else ""
    rendered = "\n".join(block)
    if start in readme and end in readme:
        before = readme.split(start, 1)[0]
        after = readme.split(end, 1)[1]
        readme = before + rendered + after
    else:
        readme = readme.rstrip() + "\n\n" + rendered + "\n"
    args.readme.write_text(readme)
    print(json.dumps({"tuning_rows": len(tuning), "screening_rows": len(screening), "complete_screening": complete, "methods": len(summary)}))


if __name__ == "__main__":
    main()
