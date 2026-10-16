#!/usr/bin/env python3
"""Run the bounded Stage-E representative HPO/tuning/screening queues.

Two independent plans are supported: q/o confirmation on the 7B model and
attention+MLP confirmation on the 1.5B model.  The same eleven representative
IDs and 12-trial HPO quota are used in both plans; only the declared model,
target set, horizon, and paired seed streams differ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHODS = [
    "L_FIXED", "I_FIXED", "ADALORA_REF", "L_GATE", "I_DISP", "T0_GATE",
    "T1_GATE", "L_PILOT_L_STUDENT", "I_PILOT_I_STUDENT", "L_UNIFORM_SVD",
    "I_UNIFORM_SVD",
]


def run(command, log: Path, resume: bool):
    if resume:
        command = list(command) + ["--resume"]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(map(str, command)) + "\n")
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["7b", "hetero"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-7B")
    ap.add_argument("--data-root", type=Path, default=Path("/var/tmp/dora-adaptive-rank/data_v2"))
    ap.add_argument("--manifest-root", type=Path, default=Path("artifacts/manifests_v2"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.kind == "7b":
        hpo_updates, full_updates = 256, 1024
        hpo_seed, tuning_seeds, screen_seeds = 7101, "7102,7103", "7001,7002,7003"
        targets = ["q_proj", "o_proj"]
        model_tag = "7b"
    else:
        hpo_updates, full_updates = 128, 512
        hpo_seed, tuning_seeds, screen_seeds = 8101, "8102,8103", "8001,8002,8003"
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model_tag = "hetero"

    args.root.mkdir(parents=True, exist_ok=True)
    method_ids = ",".join(METHODS)
    hpo = args.root / "hpo"
    tuning = args.root / "tuning"
    screening = args.root / "screening"
    plan = {
        "schema": "rankstudy-stageE-v3",
        "kind": args.kind,
        "model": args.model,
        "targets": targets,
        "methods": METHODS,
        "hpo_updates": hpo_updates,
        "full_updates": full_updates,
        "hpo_seed": hpo_seed,
        "tuning_seeds": tuning_seeds,
        "screening_seeds": screen_seeds,
        "hpo_quota": 12,
        "selection_metric": "dev_tune_endpoint_group_balanced_nll",
        "hpo": str(hpo), "tuning": str(tuning), "screening": str(screening),
    }
    (args.root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    commands = [
        [sys.executable, "scripts/run_hpo_v3.py", "--root", str(hpo), "--gpus", args.gpus,
         "--updates", str(hpo_updates), "--seed", str(hpo_seed), "--target-rank", "8",
         "--max-rank", "16", "--model", args.model, "--data-root", str(args.data_root),
         "--manifest-root", str(args.manifest_root), "--method-ids", method_ids,
         "--targets", *targets],
        [sys.executable, "scripts/run_tuning_v3.py", "--hpo-root", str(hpo), "--out-root", str(tuning),
         "--gpus", args.gpus, "--updates", str(full_updates), "--seeds", tuning_seeds,
         "--model", args.model, "--data-root", str(args.data_root), "--manifest-root", str(args.manifest_root),
         "--method-ids", method_ids, "--targets", *targets],
        [sys.executable, "scripts/run_screening_v3.py", "--tuning-root", str(tuning), "--out-root", str(screening),
         "--gpus", args.gpus, "--updates", str(full_updates), "--seeds", screen_seeds,
         "--model", args.model, "--data-root", str(args.data_root), "--manifest-root", str(args.manifest_root),
         "--method-ids", method_ids, "--targets", *targets],
    ]
    status_path = args.root / "status.json"
    for index, command in enumerate(commands, 1):
        status_path.write_text(json.dumps({"stage": index, "command": command,
                                           "plan": str(args.root / "plan.json")}, indent=2) + "\n")
        run(command, args.root / f"stage_{index}.log", args.resume)
    status_path.write_text(json.dumps({"stage": "complete", "status": "complete",
                                       "plan": str(args.root / "plan.json")}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "kind": model_tag, "root": str(args.root)}))


if __name__ == "__main__":
    main()
