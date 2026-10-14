#!/usr/bin/env python3
"""Fresh screening seeds for pipeline selections chosen by mean tuning NLL."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from collections import defaultdict
from pathlib import Path


def select(tune_root):
    grouped = defaultdict(lambda: defaultdict(list))
    for d in sorted(Path(tune_root).glob("*_c*_seed*")):
        c, m = d / "config.json", d / "metadata.json"
        if not c.exists() or not m.exists(): continue
        cf, md = json.loads(c.read_text()), json.loads(m.read_text()); score = md.get("group_balanced_nll")
        if md.get("status") == "complete" and score is not None:
            grouped[cf["method"]][(float(cf["lr"]), float(cf["scale"]))].append((int(cf["seed"]), float(score), cf))
    out = {}
    for method, configs in grouped.items():
        valid = [(sum(r[1] for r in rows) / len(rows), key, rows[0][2]) for key, rows in configs.items() if {r[0] for r in rows} >= {1002, 1003}]
        if valid: out[method] = sorted(valid, key=lambda x: (x[0], x[1]))[0][2]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tune-root", type=Path, required=True); ap.add_argument("--out-root", type=Path, required=True); ap.add_argument("--gpus", default="0,1,2,3"); ap.add_argument("--updates", type=int, default=512); ap.add_argument("--seeds", default="2001,2002,2003"); ap.add_argument("--python", default=".venv-study/bin/python"); ap.add_argument("--microbatch", type=int, default=2); ap.add_argument("--model", default="/var/tmp/dora-adaptive-rank/models/Qwen2.5-Math-1.5B"); ap.add_argument("--resume", action="store_true"); args = ap.parse_args()
    jobs = []
    for method, cf in sorted(select(args.tune_root).items()):
        for seed in map(int, args.seeds.split(",")):
            out = args.out_root / f"{method}_seed{seed}"
            if args.resume and (out / "metadata.json").exists(): continue
            out.mkdir(parents=True, exist_ok=True); (out / "launcher_status.json").write_text(json.dumps({"status": "pending", "stage": "v2_pipeline_screen"}, indent=2) + "\n")
            jobs.append({"method": method, "lr": cf["lr"], "scale": cf["scale"], "seed": seed, "updates": args.updates, "out": str(out)})
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]; free, active, i = list(gpus), {}, 0
    while i < len(jobs) or active:
        while i < len(jobs) and free:
            j = jobs[i]; i += 1; gpu = free.pop(0); out = Path(j["out"]); cmd = [args.python, "scripts/run_pipeline_v2.py", "--model", args.model]
            for key in ("method", "scale", "lr", "seed", "updates", "out"): cmd += [f"--{key.replace('_', '-')}", str(j[key])]
            cmd += ["--microbatch", str(args.microbatch)]; env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "src", "HF_HOME": os.environ.get("HF_HOME", "/var/tmp/dora-adaptive-rank/hf"), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}); log = (out / "stdout.log").open("w"); proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT); active[proc.pid] = (proc, log, j, gpu); (out / "launcher_status.json").write_text(json.dumps({"status": "running", "pid": proc.pid, "gpu": gpu, "stage": "v2_pipeline_screen"}, indent=2) + "\n"); print(json.dumps({"event": "launch", "pid": proc.pid, "gpu": gpu, "method": j["method"], "seed": j["seed"]}), flush=True)
        time.sleep(1)
        for pid, (proc, log, j, gpu) in list(active.items()):
            rc = proc.poll()
            if rc is None: continue
            log.close(); del active[pid]; free.append(gpu); free.sort(key=gpus.index); (Path(j["out"]) / "launcher_status.json").write_text(json.dumps({"status": "complete" if rc == 0 else "failed", "returncode": rc, "gpu": gpu, "stage": "v2_pipeline_screen"}, indent=2) + "\n"); print(json.dumps({"event": "finish", "pid": pid, "gpu": gpu, "status": "complete" if rc == 0 else "failed", "method": j["method"], "seed": j["seed"]}), flush=True)
    print(json.dumps({"jobs": len(jobs), "methods": len(select(args.tune_root)), "out_root": str(args.out_root)}))


if __name__ == "__main__": main()
