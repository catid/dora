#!/usr/bin/env python3
"""Run two full-horizon tuning configurations per completed v2 HPO cell."""
from __future__ import annotations
import argparse, json, os, subprocess, time, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rankstudy.registry import build_registry

SUPPORTED = {"fixed", "wide", "reference", "online", "allocation_control",
             "triplet", "triplet_ortho", "online_uncertainty"}


def select(hpo_root):
    by = defaultdict(list)
    for d in sorted(Path(hpo_root).iterdir()):
        c, m = d / "config.json", d / "metadata.json"
        if not c.exists() or not m.exists():
            continue
        cf, md = json.loads(c.read_text()), json.loads(m.read_text())
        score = md.get("dev_tune_endpoint_group_balanced_nll")
        if md.get("status") == "complete" and score is not None:
            by[cf["method"]].append((float(score), cf))
    specmap = {x.id: x for x in build_registry()}
    selected = {}
    for mid, rows in by.items():
        if mid not in specmap or specmap[mid].group not in SUPPORTED:
            continue
        rows.sort(key=lambda x: (x[0], float(x[1]["lr"]), float(x[1]["scale"])))
        chosen = []
        seen = set()
        for _, cfg in rows:
            key = (float(cfg["lr"]), float(cfg["scale"]), float(cfg.get("penalty", 0.0)))
            if key in seen:
                continue
            seen.add(key); chosen.append(cfg)
            if len(chosen) == 2:
                break
        selected[mid] = chosen
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpo-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--updates", type=int, default=512)
    ap.add_argument("--seeds", default="1002,1003")
    ap.add_argument("--python", default=".venv-study/bin/python")
    ap.add_argument("--microbatch", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    specmap = {x.id: x for x in build_registry()}
    jobs = []
    for mid, configs in sorted(select(args.hpo_root).items()):
        spec = specmap[mid]
        rank = 8 if spec.group == "fixed" or (spec.group in {"triplet", "triplet_ortho"} and spec.allocator == "none") else 16
        for ci, cfg in enumerate(configs, 1):
            for seed in map(int, args.seeds.split(",")):
                out = args.out_root / f"{mid}_c{ci}_seed{seed}"
                if args.resume and (out / "metadata.json").exists():
                    continue
                out.mkdir(parents=True, exist_ok=True)
                (out / "launcher_status.json").write_text(json.dumps({"status": "pending", "stage": "v2_tuning"}, indent=2) + "\n")
                jobs.append({"method": mid, "init": spec.init, "lr": cfg["lr"], "scale": cfg["scale"],
                             "penalty": cfg.get("penalty", 0.0), "seed": seed, "rank": rank,
                             "target_rank": 8, "updates": args.updates, "out": str(out)})
    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    free, active, i = list(gpu_ids), {}, 0
    while i < len(jobs) or active:
        while i < len(jobs) and free:
            j = jobs[i]; i += 1; gpu = free.pop(0)
            cmd = [args.python, "scripts/run_one_v2.py"]
            for key in ("method", "init", "scale", "lr", "penalty", "seed", "updates", "out", "rank", "target_rank"):
                cmd += [f"--{key.replace('_', '-')}", str(j[key])]
            cmd += ["--microbatch", str(args.microbatch)]
            env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "src",
                "HF_HOME": os.environ.get("HF_HOME", "/var/tmp/dora-adaptive-rank/hf"),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            out = Path(j["out"]); log = (out / "stdout.log").open("w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[proc.pid] = (proc, log, j, gpu)
            (out / "launcher_status.json").write_text(json.dumps({"status": "running", "pid": proc.pid, "gpu": gpu, "stage": "v2_tuning"}, indent=2) + "\n")
            print(json.dumps({"event": "launch", "pid": proc.pid, "gpu": gpu, "method": j["method"], "seed": j["seed"]}), flush=True)
        time.sleep(1)
        for pid, (proc, log, j, gpu) in list(active.items()):
            rc = proc.poll()
            if rc is None: continue
            log.close(); del active[pid]; free.append(gpu); free.sort(key=gpu_ids.index)
            (Path(j["out"]) / "launcher_status.json").write_text(json.dumps({"status": "complete" if rc == 0 else "failed", "returncode": rc, "gpu": gpu, "stage": "v2_tuning"}, indent=2) + "\n")
            print(json.dumps({"event": "finish", "pid": pid, "gpu": gpu, "status": "complete" if rc == 0 else "failed", "method": j["method"], "seed": j["seed"]}), flush=True)
    print(json.dumps({"jobs": len(jobs), "methods": len(select(args.hpo_root)), "out_root": str(args.out_root)}))


if __name__ == "__main__": main()
