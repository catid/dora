#!/usr/bin/env python3
"""Run the declared equal charged-GPU-second Stage-D pipelines.

The calibration median is read from immutable calibration receipts (or can be
passed explicitly).  Each pipeline receives the same end-to-end cap and a
fixed transition reserve.  A candidate that cannot finish its required source
and recovery intervals is retained as ``infeasible_budget`` rather than being
quietly allowed extra time.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


ROLES = {
    "FIXED_L": "L_FIXED", "FIXED_I": "I_FIXED", "REFERENCE": "ADALORA_REF",
    "ONLINE_L": "L_GATE", "ONLINE_I": "I_DISP", "TRIPLET_T0": "T0_GATE",
    "TRIPLET_T1": "T1_GATE", "RESTART_L": "I_PILOT_L_STUDENT",
    "RESTART_I": "I_PILOT_I_STUDENT", "COMPRESS_L": "L_UNIFORM_SVD",
    "COMPRESS_I": "I_UNIFORM_SVD",
}


def frozen_configs(path: Path):
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["method"]] = {"lr": float(row["lr"]), "scale": float(row["scale"]),
                                   "penalty": float(row.get("penalty", 0.0))}
    missing = sorted(set(ROLES.values()) - set(out))
    if missing:
        raise RuntimeError(f"frozen tuning config missing methods: {missing}")
    return out


def calibration_median(root: Path) -> float:
    values = []
    for path in sorted(root.glob("*/metadata.json")):
        data = json.loads(path.read_text())
        if data.get("method") == "L_FIXED" and data.get("status") == "complete":
            values.append(float(data["charged_gpu_seconds"]))
    if len(values) < 3:
        raise RuntimeError(f"expected at least three valid calibration receipts under {root}")
    values.sort()
    return values[len(values) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", type=Path, default=Path("reports/tuning_selection.csv"))
    ap.add_argument("--calibration-root", type=Path,
                    default=Path("/var/tmp/dora-adaptive-rank/runs/v3_stageD_calibration_20261016_retry"))
    ap.add_argument("--budget-s", type=float, default=None,
                    help="Frozen quality-blind cap; defaults to the calibration median.")
    ap.add_argument("--transition-reserve-s", type=float, default=0.2,
                    help="Measured transition reserve; 0.2 s covers the observed 0.043--0.157 s range.")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seeds", default="6001,6002,6003")
    ap.add_argument("--repeats", default="0,1")
    ap.add_argument("--updates", type=int, default=512)
    ap.add_argument("--target-rank", type=int, default=8)
    ap.add_argument("--max-rank", type=int, default=16)
    ap.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--data-root", type=Path, default=Path("/var/tmp/dora-adaptive-rank/data_v2"))
    ap.add_argument("--manifest-root", type=Path, default=Path("artifacts/manifests_v2"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    configs = frozen_configs(args.tuning)
    budget = calibration_median(args.calibration_root) if args.budget_s is None else args.budget_s
    if budget <= 0 or args.transition_reserve_s < 0:
        raise ValueError("invalid budget or transition reserve")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    repeats = [int(x) for x in args.repeats.split(",") if x.strip()]
    missing = [s for s in seeds if not (args.manifest_root / f"train_seed{s}.json").exists()]
    if missing:
        raise RuntimeError(f"sealed training manifests missing before launch: {missing}")
    jobs = []
    for role, method in ROLES.items():
        for seed in seeds:
            for repeat in repeats:
                jobs.append({"role": role, "method": method, "seed": seed, "repeat": repeat,
                             **configs[method],
                             "out": str(args.out_root / f"{role}_{method}_s{seed}_rep{repeat}")})
    plan = {"stage": "D_compute", "budget_s": budget,
            "transition_reserve_s": args.transition_reserve_s,
            "calibration_root": str(args.calibration_root),
            "frozen_tuning": str(args.tuning), "jobs": jobs}
    (args.out_root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rng = random.Random(20261027)
    rng.shuffle(jobs)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    free, active, index = list(gpus), {}, 0
    while index < len(jobs) or active:
        while index < len(jobs) and free:
            job = jobs[index]
            index += 1
            gpu = free.pop(0)
            out = Path(job["out"])
            out.mkdir(parents=True, exist_ok=True)
            status = out / "launcher_status.json"
            if status.exists():
                old = json.loads(status.read_text())
                if old.get("status") in {"complete", "failed"} and not args.resume:
                    raise RuntimeError(f"existing terminal job requires --resume: {out}")
                if old.get("status") == "complete" and args.resume:
                    free.append(gpu); free.sort(key=gpus.index)
                    continue
            cmd = [sys.executable, "scripts/run_pipeline_v3.py", "--method", job["method"],
                   "--out", str(out), "--seed", str(job["seed"]), "--updates", str(args.updates),
                   "--lr", str(job["lr"]), "--scale", str(job["scale"]),
                   "--penalty", str(job["penalty"]), "--target-rank", str(args.target_rank),
                   "--max-rank", str(args.max_rank), "--model", args.model,
                   "--data-root", str(args.data_root), "--manifest-root", str(args.manifest_root),
                   "--deployment-export", "--gpu-budget-s", str(budget),
                   "--transition-reserve-s", str(args.transition_reserve_s)]
            if args.resume:
                cmd.append("--resume")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(Path.cwd() / "src"),
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "HF_HOME": env.get("HF_HOME", "/var/tmp/dora-adaptive-rank/hf")})
            log = (out / "stdout.log").open("a")
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            status.write_text(json.dumps({"status": "running", "pid": p.pid, "gpu": gpu,
                                          "role": job["role"], "method": job["method"],
                                          "seed": job["seed"], "repeat": job["repeat"],
                                          "budget_s": budget, "command": cmd}, indent=2) + "\n")
            active[p.pid] = (p, log, job, gpu)
            print(json.dumps({"event": "launch", "pid": p.pid, "gpu": gpu,
                              "role": job["role"], "seed": job["seed"], "repeat": job["repeat"]}), flush=True)
        time.sleep(1)
        for pid, (p, log, job, gpu) in list(active.items()):
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            del active[pid]
            free.append(gpu); free.sort(key=gpus.index)
            out = Path(job["out"])
            (out / "launcher_status.json").write_text(json.dumps({
                "status": "complete" if rc == 0 else "failed", "returncode": rc,
                "gpu": gpu, "role": job["role"], "method": job["method"],
                "seed": job["seed"], "repeat": job["repeat"]}, indent=2) + "\n")
            print(json.dumps({"event": "finish", "pid": pid, "gpu": gpu,
                              "role": job["role"], "seed": job["seed"], "repeat": job["repeat"],
                              "status": "complete" if rc == 0 else "failed"}), flush=True)
    print(json.dumps({"status": "complete", "jobs": len(jobs), "budget_s": budget,
                      "out_root": str(args.out_root)}))


if __name__ == "__main__":
    main()
