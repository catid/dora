#!/usr/bin/env python3
"""Launch the declared Stage-D calibration and common export branches.

This launcher consumes only frozen Stage-C method/config records.  It keeps
each subprocess in its own immutable directory and records failed attempts;
the common export pipeline itself is implemented by run_pipeline_v3.py.
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


def jobs(configs, stage, root, seeds):
    if stage == "calibration":
        return [{"role": "CALIBRATION_L", "method": "L_FIXED", "seed": seed,
                 **configs["L_FIXED"], "out": str(root / f"L_FIXED_s{seed}"), "pipeline": False}
                for seed in seeds]
    return [{"role": role, "method": method, "seed": seed, **configs[method],
             "out": str(root / f"{role}_{method}_s{seed}"), "pipeline": True}
            for role, method in ROLES.items() for seed in seeds]


def launch(job, gpu, args):
    out = Path(job["out"]); out.mkdir(parents=True, exist_ok=True)
    status = out / "launcher_status.json"
    if status.exists():
        old = json.loads(status.read_text())
        if old.get("status") == "complete":
            return None, None
        if not args.resume:
            raise RuntimeError(f"incomplete existing job requires --resume: {out}")
    exe = "scripts/run_pipeline_v3.py" if job["pipeline"] else "scripts/run_one_v3.py"
    cmd = [sys.executable, exe, "--method", job["method"], "--out", str(out),
           "--seed", str(job["seed"]), "--updates", str(args.updates),
           "--lr", str(job["lr"]), "--scale", str(job["scale"]),
           "--penalty", str(job["penalty"]), "--target-rank", str(args.target_rank),
           "--max-rank", str(args.max_rank), "--model", args.model,
           "--data-root", str(args.data_root), "--manifest-root", str(args.manifest_root)]
    if job["pipeline"]:
        cmd.append("--deployment-export")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(Path.cwd() / "src"),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "HF_HOME": env.get("HF_HOME", "/var/tmp/dora-adaptive-rank/hf")})
    log = (out / "stdout.log").open("a")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    status.write_text(json.dumps({"status": "running", "pid": p.pid, "gpu": str(gpu),
                                  "command": cmd, "role": job["role"]}, indent=2) + "\n")
    return p, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calibration", "export"], required=True)
    ap.add_argument("--tuning", type=Path, default=Path("reports/tuning_selection.csv"))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seeds", default="6101,6102,6103,6104,6105")
    ap.add_argument("--updates", type=int, default=512)
    ap.add_argument("--target-rank", type=int, default=8)
    ap.add_argument("--max-rank", type=int, default=16)
    ap.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B")
    ap.add_argument("--data-root", type=Path, default=Path("/var/tmp/dora-adaptive-rank/data_v2"))
    ap.add_argument("--manifest-root", type=Path, default=Path("artifacts/manifests_v2"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); args.out_root.mkdir(parents=True, exist_ok=True)
    configs = frozen_configs(args.tuning)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    missing = [seed for seed in seeds if not (args.manifest_root / f"train_seed{seed}.json").exists()]
    if missing:
        raise RuntimeError(f"sealed training manifests missing before launch: {missing}")
    plan = jobs(configs, args.stage, args.out_root, seeds)
    (args.out_root / "plan.json").write_text(json.dumps({"stage": args.stage, "jobs": plan,
                                                            "frozen_tuning": str(args.tuning)}, indent=2) + "\n")
    rng = random.Random(20261024 if args.stage == "calibration" else 20261025); rng.shuffle(plan)
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    free, active, index = list(gpus), {}, 0
    while index < len(plan) or active:
        while index < len(plan) and free:
            job = plan[index]; index += 1; gpu = free.pop(0)
            launched = launch(job, gpu, args)
            if launched[0] is None:
                free.append(gpu); free.sort(key=gpus.index); continue
            p, log = launched; active[p.pid] = (p, log, job, gpu)
            print(json.dumps({"event": "launch", "pid": p.pid, "gpu": gpu,
                              "role": job["role"], "method": job["method"], "seed": job["seed"]}), flush=True)
        time.sleep(1)
        for pid, (p, log, job, gpu) in list(active.items()):
            rc = p.poll()
            if rc is None: continue
            log.close(); del active[pid]; free.append(gpu); free.sort(key=gpus.index)
            (Path(job["out"]) / "launcher_status.json").write_text(json.dumps({
                "status": "complete" if rc == 0 else "failed", "returncode": rc, "gpu": str(gpu),
                "role": job["role"], "method": job["method"], "seed": job["seed"]}, indent=2) + "\n")
            print(json.dumps({"event": "finish", "pid": pid, "gpu": gpu,
                              "role": job["role"], "method": job["method"],
                              "status": "complete" if rc == 0 else "failed"}), flush=True)
    print(json.dumps({"status": "complete", "stage": args.stage, "jobs": len(plan),
                      "out_root": str(args.out_root)}))


if __name__ == "__main__":
    main()
